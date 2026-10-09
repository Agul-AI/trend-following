"""Independent selection/accounting checks using fabricated inputs only.

The helpers can audit newly released validation arrays without invoking any
production simulator, feature builder, accounting helper or score function.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from trend_following import research_intersection_validation as validation


def independent_shared_pairs(metrics, benchmark):
    """Strict, finite development intersection; timing is not selected per pair."""
    if not np.isfinite(benchmark):
        raise ValueError("Finite cost-matched BH Sharpe required")
    sets = []
    for hours in (2.0, 3.0, 4.0):
        frame = metrics.loc[metrics.entry_confirmation_hours.eq(hours)]
        if not frame.exit_confirmation_hours.eq(hours).all():
            raise ValueError("A single shared wait applies to both sides")
        sets.append(
            {
                (row.entry_rule, row.exit_rule)
                for row in frame.itertuples()
                if np.isfinite(row.cash_excess_sharpe)
                and row.cash_excess_sharpe > benchmark + 1e-10
            }
        )
    return sorted(sets[0] & sets[1] & sets[2])


def independent_daily_statistics(equity, cash_returns, *, initial_cash, elapsed_years):
    """No production math helpers; 252 sessions and sample SD (ddof=1)."""
    equity = np.asarray(equity, dtype=float)
    cash_returns = np.asarray(cash_returns, dtype=float)
    if equity.ndim == 1:
        equity = equity[:, None]
    if (
        equity.ndim != 2
        or equity.shape[0] < 2
        or cash_returns.shape != (equity.shape[0],)
        or not np.isfinite(equity).all()
        or not np.isfinite(cash_returns).all()
        or (equity <= 0).any()
        or initial_cash <= 0
        or not np.isfinite(initial_cash)
        or elapsed_years <= 0
        or not np.isfinite(elapsed_years)
    ):
        raise ValueError("Finite positive wealth and matching session cash returns required")
    previous = np.vstack((np.full(equity.shape[1], initial_cash), equity[:-1]))
    returns = equity / previous - 1
    excess = returns - cash_returns[:, None]
    sd = np.std(excess, axis=0, ddof=1)
    sharpe = np.full(equity.shape[1], np.nan)
    np.divide(np.sqrt(252) * excess.mean(axis=0), sd, out=sharpe, where=sd > 0)
    full_wealth = np.vstack((np.full(equity.shape[1], initial_cash), equity))
    return {
        "returns": returns,
        "cash_excess_sharpe": sharpe,
        "cagr": (equity[-1] / initial_cash) ** (1 / elapsed_years) - 1,
        "max_drawdown": (full_wealth / np.maximum.accumulate(full_wealth, axis=0) - 1).min(axis=0),
        "annualized_volatility": returns.std(axis=0, ddof=1) * np.sqrt(252),
        "terminal_equity": equity[-1],
    }


def independent_nominal_cash_ledger(clock, initial_cash=1000.0):
    clock = pd.DatetimeIndex(clock)
    if (
        clock.tz is None
        or len(clock) == 0
        or clock.hasnans
        or clock.has_duplicates
        or not clock.is_monotonic_increasing
        or initial_cash <= 0
        or not np.isfinite(initial_cash)
    ):
        raise ValueError("Ordered unique timezone-aware cash clock required")
    elapsed_seconds = np.r_[0, np.diff(clock.asi8).astype(float) / 1e9]
    return initial_cash * np.cumprod(1 + 0.03 * elapsed_seconds / (365 * 86400))


def independent_counts(scores, benchmark):
    """Keep all frozen members, including undefined scores, in the denominator."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(benchmark):
        raise ValueError("Complete selected cohort and finite benchmark required")
    finite = np.isfinite(scores)
    return {
        "configuration_count": len(scores),
        "finite_Sharpe_count": int(finite.sum()),
        "undefined_Sharpe_count": int((~finite).sum()),
        "beat_BH_Sharpe_count": sum(
            bool(valid and score > benchmark + 1e-10)
            for valid, score in zip(finite, scores, strict=True)
        ),
        "tie_BH_Sharpe_count": sum(
            bool(valid and abs(score - benchmark) <= 1e-10)
            for valid, score in zip(finite, scores, strict=True)
        ),
        "below_BH_Sharpe_count": sum(
            bool(valid and score < benchmark - 1e-10)
            for valid, score in zip(finite, scores, strict=True)
        ),
    }


def independent_passive_ledgers(
    daily, bars, calendar, coverage, blackouts, *, start, end, initial_cash=1000.0
):
    """Independent raw-share BH and cash, with 1/3-bp one-way assumptions.

    All inputs must already be authorized stage-bounded packets. This helper
    neither loads files nor calls any production event/accounting function.
    """
    scored = calendar.loc[calendar.session.between(start, end)].copy()
    if scored.empty:
        raise ValueError("Nonempty calendar segment required")
    if not (scored.session.is_unique and scored.session.is_monotonic_increasing):
        raise ValueError("Unique chronological session calendar required")
    observed = bars.loc[bars.session.isin(scored.session)]
    quote = daily.set_index("session")
    eligibility = coverage.set_index("bar_start").fill_eligible
    first = scored.market_open.iloc[0]
    final_close = scored.market_close.iloc[-1]
    marks = {
        pd.Timestamp(f"{s} 17:00", tz="America/New_York").tz_convert("UTC"): s
        for s in scored.session
    }
    last = max(marks)
    events = []
    for row in scored.itertuples():
        events.append((row.market_open, 1, "actions", row.session))
    for row in observed.itertuples():
        events.append((row.bar_start, 2, "open", row))
        events.append((row.bar_end, 0, "close", row))
    for time, session in marks.items():
        events.append((time, 3, "mark", session))
    for row in blackouts.itertuples():
        if row.end > first and row.start <= last:
            events.append((max(first, row.start), 0, "clock_only", None))
            events.append((min(last, row.end), 0, "clock_only", None))
    events.sort(key=lambda event: (event[0], event[1]))
    cash, shares, pending_dividend, bought = initial_cash, 0.0, 0.0, False
    gross_cash, gross_shares, gross_dividend = initial_cash, 0.0, 0.0
    cash_reference, previous = initial_cash, first
    fee, slip = 0.0001, 0.0003
    values, references, gross_values, times = [], [], [], []
    dividend_actions, split_actions = 0, 0
    for instant, _, kind, item in events:
        if instant != previous:
            factor = 1 + 0.03 * (instant - previous).total_seconds() / (365 * 86400)
            cash *= factor
            pending_dividend *= factor
            gross_cash *= factor
            gross_dividend *= factor
            cash_reference *= factor
            previous = instant
        if kind == "actions":
            action = quote.loc[item]
            shares *= action.split_coefficient
            gross_shares *= action.split_coefficient
            dividend = shares * action.dividend_amount
            gross_distribution = gross_shares * action.dividend_amount
            cash += dividend
            pending_dividend += dividend
            gross_cash += gross_distribution
            gross_dividend += gross_distribution
            dividend_actions += int(action.dividend_amount > 0)
            split_actions += int(action.split_coefficient != 1)
        elif kind == "open" and bool(eligibility.loc[item.bar_start]):
            if shares > 0 and pending_dividend > 0:
                shares += pending_dividend / item.open
                cash -= pending_dividend
                pending_dividend = 0.0
            if gross_shares > 0 and gross_dividend > 0:
                gross_shares += gross_dividend / item.open
                gross_cash -= gross_dividend
                gross_dividend = 0.0
            if not bought:
                shares = cash / (item.open * (1 + slip) * (1 + fee))
                gross_shares = gross_cash / item.open
                cash, gross_cash, bought = 0.0, 0.0, True
        elif kind == "close" and instant == final_close:
            if not bool(eligibility.loc[item.bar_start]):
                raise ValueError("Terminal bar must have safe observed execution prices")
            cash += shares * item.close * (1 - slip) * (1 - fee)
            gross_cash += gross_shares * item.close
            shares, gross_shares = 0.0, 0.0
        elif kind == "mark":
            values.append(cash + shares * quote.loc[item, "close"])
            gross_values.append(gross_cash + gross_shares * quote.loc[item, "close"])
            references.append(cash_reference)
            times.append(instant)
    times = pd.DatetimeIndex(times)
    if len(times) != len(scored):
        raise ValueError("Exactly one mark per scored session required")
    clock = pd.DatetimeIndex(sorted({event[0] for event in events}))
    reference = independent_nominal_cash_ledger(clock, initial_cash)
    np.testing.assert_allclose(np.asarray(references), reference[clock.get_indexer(times)])
    return {
        "clock": clock,
        "times": times,
        "buy_and_hold_equity": np.asarray(values),
        "buy_and_hold_gross_equity": np.asarray(gross_values),
        "cash_equity": np.asarray(references),
        "cash_returns": np.asarray(references) / np.r_[initial_cash, references[:-1]] - 1,
        "dividend_actions": dividend_actions,
        "split_actions": split_actions,
        "elapsed_years": (last - first).total_seconds() / (365 * 86400),
    }


def test_selection_is_pair_intersection_not_union_or_per_pair_wait_optimization():
    frame = pd.DataFrame(
        {
            "entry_rule": entry,
            "exit_rule": exit_,
            "entry_confirmation_hours": hours,
            "exit_confirmation_hours": hours,
            "cash_excess_sharpe": score,
        }
        for entry, exit_, scores in (
            ("ema_200", "ema_100", (0.8, 0.7, 0.6)),
            ("sma_10", "sma_20", (0.8, 0.7, 0.4)),
            ("ema_50", "sma_50", (0.4, 0.8, 0.8)),
            ("sma_200", "ema_100", (0.8, np.nan, 0.8)),
        )
        for hours, score in zip((2.0, 3.0, 4.0), scores, strict=True)
    )
    assert independent_shared_pairs(frame, 0.5) == [("ema_200", "ema_100")]


def test_selection_is_strict_at_tolerance_and_rejects_undefined_benchmark():
    frame = pd.DataFrame(
        {
            "entry_rule": "ema_200",
            "exit_rule": "ema_100",
            "entry_confirmation_hours": hours,
            "exit_confirmation_hours": hours,
            "cash_excess_sharpe": 0.5 + 1e-10,
        }
        for hours in (2.0, 3.0, 4.0)
    )
    assert independent_shared_pairs(frame, 0.5) == []
    with pytest.raises(ValueError, match="Finite"):
        independent_shared_pairs(frame, np.nan)
    frame.loc[0, "exit_confirmation_hours"] = 3
    with pytest.raises(ValueError, match="shared wait"):
        independent_shared_pairs(frame, 0.5)


def test_frozen_denominator_survives_nan_inf_ties_and_losses():
    scores = np.full(159, 0.2)
    scores[:6] = [0.8, np.nan, np.inf, 0.5, 0.5 + 5e-11, 0.5 - 5e-11]
    actual = independent_counts(scores, 0.5)
    assert actual == {
        "configuration_count": 159,
        "finite_Sharpe_count": 157,
        "undefined_Sharpe_count": 2,
        "beat_BH_Sharpe_count": 1,
        "tie_BH_Sharpe_count": 3,
        "below_BH_Sharpe_count": 153,
    }


def test_independent_statistics_retain_159_columns_and_initial_cash_drawdown():
    wealth = np.tile(np.array([900.0, 810.0, 850.0])[:, None], (1, 159))
    stats = independent_daily_statistics(wealth, np.zeros(3), initial_cash=1000, elapsed_years=1)
    assert stats["cash_excess_sharpe"].shape == (159,)
    np.testing.assert_allclose(stats["max_drawdown"], -0.19)
    np.testing.assert_allclose(stats["cagr"], -0.15)
    returns = np.array([-0.1, -0.1, 850 / 810 - 1])
    expected_sharpe = np.sqrt(252) * returns.mean() / returns.std(ddof=1)
    np.testing.assert_allclose(stats["cash_excess_sharpe"], expected_sharpe)


def test_cash_reference_subtracts_actual_calendar_accrual_not_three_pct_over_252():
    clock = pd.DatetimeIndex(["2012-01-06 22:00Z", "2012-01-09 14:30Z", "2012-01-09 15:00Z"])
    cash_equity = independent_nominal_cash_ledger(clock)
    cash_returns = cash_equity / np.r_[1000, cash_equity[:-1]] - 1
    assert cash_returns[1] > 2 * 0.03 / 365
    stats = independent_daily_statistics(
        cash_equity, cash_returns, initial_cash=1000, elapsed_years=1
    )
    assert np.isnan(stats["cash_excess_sharpe"]).all()
    assert stats["max_drawdown"][0] == 0
    assert cash_equity[-1] == pytest.approx(
        1000 * (1 + 0.03 * 64.5 / (365 * 24)) * (1 + 0.03 * 0.5 / (365 * 24))
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0, -1.0])
def test_independent_math_stops_on_invalid_selected_wealth_instead_of_dropping_it(bad):
    with pytest.raises(ValueError, match="Finite positive"):
        independent_daily_statistics([1000, bad], [0, 0], initial_cash=1000, elapsed_years=1)


def test_cash_nominal_is_not_exact_three_percent_apy():
    clock = pd.date_range("2012-01-01 00:00Z", "2013-01-01 00:00Z", freq="D")
    wealth = independent_nominal_cash_ledger(clock)
    assert wealth[-1] == pytest.approx(1000 * (1 + 0.03 / 365) ** 366)
    assert 1.03 < wealth[-1] / wealth[0] < 1.031


def test_independent_passive_raw_share_ledger_split_dividend_cost_and_liquidation():
    sessions = ["2012-01-03", "2012-01-04", "2012-01-05"]
    daily = pd.DataFrame(
        {
            "session": sessions,
            "close": [110.0, 60.0, 66.0],
            "dividend_amount": [0.0, 1.0, 0.0],
            "split_coefficient": [1.0, 2.0, 1.0],
        }
    )
    calendar, bars = [], []
    for session, opening_price, closing_price in zip(
        sessions, [100.0, 54.0, 60.0], [110.0, 60.0, 66.0], strict=True
    ):
        opening = pd.Timestamp(f"{session} 09:30", tz="America/New_York").tz_convert("UTC")
        close = opening + pd.Timedelta(hours=6.5)
        calendar.append({"session": session, "market_open": opening, "market_close": close})
        bars.extend(
            [
                {
                    "session": session,
                    "bar_start": opening,
                    "bar_end": opening + pd.Timedelta(minutes=30),
                    "open": opening_price,
                    "close": opening_price,
                },
                {
                    "session": session,
                    "bar_start": close - pd.Timedelta(minutes=30),
                    "bar_end": close,
                    "open": closing_price,
                    "close": closing_price,
                },
            ]
        )
    bars = pd.DataFrame(bars)
    coverage = bars[["bar_start"]].assign(fill_eligible=True)
    ledgers = independent_passive_ledgers(
        daily,
        bars,
        pd.DataFrame(calendar),
        coverage,
        pd.DataFrame(columns=["start", "end"]),
        start=sessions[0],
        end=sessions[-1],
    )
    expected_shares = 20 * (1 + 1 / 54)
    expected_gross = expected_shares * 66 * (1 + 0.03 / (365 * 24))
    expected_net = expected_gross * (1 - 0.0003) * (1 - 0.0001) / ((1 + 0.0003) * (1 + 0.0001))
    assert ledgers["buy_and_hold_gross_equity"][0] == pytest.approx(1100)
    assert ledgers["buy_and_hold_gross_equity"][1] == pytest.approx(expected_shares * 60)
    assert ledgers["buy_and_hold_gross_equity"][-1] == pytest.approx(expected_gross)
    assert ledgers["buy_and_hold_equity"][-1] == pytest.approx(expected_net)
    assert ledgers["dividend_actions"] == 1
    assert ledgers["split_actions"] == 1
    assert len(ledgers["times"]) == 3
    assert (ledgers["times"].tz_convert("America/New_York").hour == 17).all()


def synthetic_full_development_metrics():
    return pd.DataFrame(
        {
            "candidate_id": candidate.candidate_id,
            "entry_rule": candidate.entry_rule.rule_id,
            "exit_rule": candidate.exit_rule.rule_id,
            "entry_confirmation_hours": hours,
            "exit_confirmation_hours": hours,
            "cash_excess_sharpe": np.sin(index / 10 + hours / 5),
        }
        for hours in validation.fixed.WAIT_HOURS
        for index, candidate in enumerate(validation.fixed.candidates_for_wait(hours))
    )


def test_production_intersection_agrees_with_independent_development_math():
    metrics = synthetic_full_development_metrics()
    pairs, summary = validation.select_shared_pairs(metrics, 0.5)
    actual = [(pair["entry_rule"]["family"], pair["pair_id"]) for pair in pairs]
    expected = independent_shared_pairs(metrics, 0.5)
    assert [pair_id for _, pair_id in actual] == sorted(f"{en}|{ex}" for en, ex in expected)
    assert summary["shared_pair_count"] == len(expected)
    assert summary["candidate_count"] == 3 * len(expected)
    assert summary["uses_validation_or_OOS_outcomes"] is False
    assert summary["filters_gap_flags"] is False
    for hours in (2.0, 3.0, 4.0):
        candidates = validation.candidates_from_pairs(pairs, hours)
        assert len(candidates) == len(expected)
        assert all(
            c.entry_confirmation_hours == c.exit_confirmation_hours == hours for c in candidates
        )
        assert all(
            c.entry_required_checks == c.exit_required_checks == 1 + 2 * hours for c in candidates
        )


def test_production_cohort_count_keeps_every_frozen_member_and_matches_independent_counts():
    pairs, _ = validation.select_shared_pairs(synthetic_full_development_metrics(), 0.5)
    rows = []
    for hours in (2.0, 3.0, 4.0):
        candidates = validation.candidates_from_pairs(pairs, hours)
        scores = np.full(len(candidates), 0.2)
        scores[:6] = [0.8, np.nan, np.inf, 0.5, 0.5 + 5e-11, 0.5 - 5e-11]
        for candidate, score in zip(candidates, scores, strict=True):
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "entry_rule": candidate.entry_rule.rule_id,
                    "exit_rule": candidate.exit_rule.rule_id,
                    "entry_confirmation_hours": hours,
                    "exit_confirmation_hours": hours,
                    "cash_excess_sharpe": score,
                    "trade_count": 10,
                }
            )
    frame = pd.DataFrame(rows)
    counts = validation.counts_against_bh(frame, 0.5, pairs)
    for hours in (2.0, 3.0, 4.0):
        expected = independent_counts(
            frame.loc[frame.entry_confirmation_hours.eq(hours)].cash_excess_sharpe, 0.5
        )
        actual = counts.loc[counts.wait_hours.eq(hours)].iloc[0]
        for key, number in expected.items():
            assert actual[key] == number


@pytest.mark.parametrize("bad", ["missing", "duplicate", "unequal_wait", "unrecognized_rule"])
def test_production_intersection_rejects_partial_or_identity_changed_grid(bad):
    metrics = synthetic_full_development_metrics()
    if bad == "missing":
        metrics = metrics.iloc[1:]
    elif bad == "duplicate":
        metrics.loc[1, "candidate_id"] = metrics.loc[0, "candidate_id"]
    elif bad == "unequal_wait":
        metrics.loc[0, "exit_confirmation_hours"] = 4
    else:
        metrics.loc[0, "entry_rule"] = "sma_300"
    with pytest.raises(validation.Error):
        validation.select_shared_pairs(metrics, 0.5)


def test_no_numeric_validation_loader_is_called_before_candidate_seal_check(tmp_path, monkeypatch):
    monkeypatch.setattr(
        validation,
        "verify_study",
        lambda *_: (_ for _ in ()).throw(validation.Error("Missing candidate freeze")),
    )
    monkeypatch.setattr(
        validation,
        "load_validation_packet",
        lambda *_: pytest.fail(
            "A numeric validation packet was opened before its candidate freeze"
        ),
    )
    with pytest.raises(validation.Error, match="candidate freeze"):
        validation.evaluate_study(tmp_path)
    assert (tmp_path / "attempts.jsonl").is_file()
    assert json.loads((tmp_path / "attempts.jsonl").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "key,value",
    [
        ("stage", "historical_oos"),
        ("historical_oos_authorized", True),
        ("period", {"start": "2016-01-01", "end": "2026-05-28"}),
    ],
)
def test_validation_loader_rejects_oos_stage_before_any_numeric_read(monkeypatch, key, value):
    payload = {
        "stage": "validation",
        "historical_oos_authorized": False,
        "period": {"start": "2012-01-01", "end": "2015-12-31"},
    }
    payload[key] = value
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: pytest.fail("OOS numeric read"))
    with pytest.raises(validation.Error):
        validation.load_validation_packet(payload)


def test_validation_metadata_is_only_json_and_own_loader_is_stage_bounded():
    metadata = inspect.getsource(validation._read_validation_metadata)
    assert "pd.read_csv" not in metadata
    assert "prepared/validation" in metadata
    assert "2015-12-31" in metadata
    loader = inspect.getsource(validation.load_validation_packet)
    assert "prepared/validation" in loader
    assert "2015-12-31" in loader
    for source in (metadata, loader):
        for forbidden in (
            "_verify_prepared(",
            "_load_stage(",
            "read_reconciled_packet(",
            "DTB3_available.csv",
        ):
            assert forbidden not in source
    assert validation.SOURCE_FILES == (
        *validation.fixed.SOURCE_FILES,
        "src/trend_following/research_intersection_validation.py",
    )
    assert validation.ANNUAL_CASH_RATE == 0.03
    assert validation.COMMISSION_BPS == 1
    assert validation.SLIPPAGE_BPS == 3
