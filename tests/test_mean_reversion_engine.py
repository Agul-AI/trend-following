"""Independent synthetic mean-reversion event-ledger tests; no market files."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trend_following import research_mean_reversion as mr
from trend_following import research_mean_reversion_engine as engine
from trend_following import research_quality_engine_v5 as quality


def fixture(specs, history=(100.0,) * 60):
    historical = pd.bdate_range("1999-11-01", periods=len(history))
    daily = [
        dict(session=x.strftime("%Y-%m-%d"), close=p, dividend_amount=0.0, split_coefficient=1.0)
        for x, p in zip(historical, history, strict=True)
    ]
    bars, coverage, calendar = [], [], []
    for i, spec in enumerate(specs):
        session = spec.get(
            "session", (pd.Timestamp("2001-01-02") + pd.offsets.BDay(i)).strftime("%Y-%m-%d")
        )
        opening = pd.Timestamp(f"{session} 09:30", tz="America/New_York").tz_convert("UTC")
        closes = spec.get("closes", [90.0] * 13)
        opens = spec.get("opens", [100.0] * len(closes))
        missing = set(spec.get("missing", []))
        for k, (op, cl) in enumerate(zip(opens, closes, strict=True)):
            start = opening + pd.Timedelta(minutes=30 * k)
            end = start + pd.Timedelta(minutes=30)
            coverage.append(
                dict(
                    session=session,
                    bar_start=start,
                    bar_end=end,
                    status="unknown_missing" if k in missing else "observed",
                    signal_eligible=k not in missing,
                    fill_eligible=k not in missing,
                )
            )
            if k not in missing:
                bars.append(
                    dict(
                        session=session,
                        bar_start=start,
                        bar_end=end,
                        open=op,
                        high=max(op, cl),
                        low=min(op, cl),
                        close=cl,
                        volume=100.0,
                        duration_minutes=30.0,
                        is_completed=True,
                        is_full_hour=False,
                    )
                )
        calendar.append(
            dict(
                session=session,
                market_open=opening,
                market_close=opening + pd.Timedelta(minutes=30 * len(closes)),
            )
        )
        daily.append(
            dict(
                session=session,
                close=spec.get("daily_close", closes[-1]),
                dividend_amount=spec.get("dividend", 0.0),
                split_coefficient=spec.get("split", 1.0),
            )
        )
    return pd.DataFrame(daily), pd.DataFrame(bars), pd.DataFrame(calendar), pd.DataFrame(coverage)


def candidate():
    return mr.Candidate(mr.Rule("SMA_DISTANCE", 10), -0.02, 0.0)


def cache_for(specs, *, values=None, post_gap=False):
    daily, bars, calendar, coverage = fixture(specs)
    cache = engine.build_cache(
        daily,
        bars,
        calendar,
        coverage=coverage,
        quality_policy=quality.QualityPolicy(
            "observed_only_blackout_v1", post_gap_diagnostics=post_gap
        ),
    )
    if values is not None:
        # A deliberately synthetic signed cache isolates threshold/streak mechanics
        # from feature arithmetic, which has separate independent reference tests.
        margins = cache.margins.copy()
        margins[:, cache.rules.index(candidate().rule)] = np.asarray(values, dtype=float)
        cache = replace(cache, margins=margins, feature_fingerprint="")
        cache = replace(cache, feature_fingerprint=quality._feature_fingerprint(cache))
    return cache


def observations(cache, annual=0.0):
    dates = pd.date_range(
        cache.calendar.market_open.iloc[0].normalize() - pd.Timedelta(days=1),
        cache.calendar.market_close.iloc[-1].normalize() + pd.Timedelta(days=2),
        freq="D",
        tz="UTC",
    )
    return pd.DataFrame({"available_at": dates, "annual_rate": annual})


def simulate(cache, *, annual=0.0, fee=0.0, slip=0.0, mode="candidate"):
    return engine.result(
        engine.run(
            cache,
            [candidate()],
            start=cache.calendar.session.iloc[0],
            end=cache.calendar.session.iloc[-1],
            cash_observations=observations(cache, annual),
            commission_bps=fee,
            slippage_bps=slip,
            record=True,
            mode=mode,
        )
    )


def ny(value):
    return pd.Timestamp(value, tz="America/New_York")


def test_seven_checks_use_next_open_not_signal_close_and_terminal_closes_position():
    cache = cache_for([{"opens": [100] * 7 + [130] * 6}], values=[-0.02] * 7 + [-0.01] * 6)
    result = simulate(cache)
    buy, sell = result.trades.iloc[0], result.trades.iloc[1]
    assert buy.decision_at.tz_convert("America/New_York") == ny("2001-01-02 13:00")
    assert buy.timestamp == buy.decision_at  # Subsequent bar at the same boundary, not same quote.
    assert buy.reference_price == 130
    assert buy.reference_price != cache.bars.close.iloc[6]
    assert sell.reason == "terminal_liquidation"
    assert sell.reference_price == 90
    assert sell.timestamp.tz_convert("America/New_York") == ny("2001-01-02 16:00")
    assert result.metrics["terminal_equity"] == pytest.approx(1000 / 130 * 90)
    assert result.metrics["order_count"] == 2
    assert result.metrics["trade_count"] == 1
    assert result.metrics["entry_required_checks"] == result.metrics["exit_required_checks"] == 7
    assert result.metrics["indicator_units"] == "daily_sessions"
    assert result.daily_equity.index.tz_convert("America/New_York").hour.tolist() == [17]


def test_real_feature_entry_does_not_require_recovery_support_simultaneously():
    cache = cache_for([{}])
    result = simulate(cache)
    assert result.trades.side.tolist() == ["buy", "sell"]
    assert cache.margins[:7, cache.rules.index(candidate().rule)].max() < -0.02


def test_recovery_exit_includes_equality_and_uses_seventh_check_next_open():
    cache = cache_for(
        [{}, {"opens": [100] * 7 + [140] * 6}],
        values=[-0.02] * 7 + [-0.01] * 6 + [0.0] * 7 + [-0.01] * 6,
    )
    result = simulate(cache)
    assert result.trades.reason.tolist() == ["confirmed_entry", "confirmed_exit"]
    assert result.trades.iloc[1].decision_at.tz_convert("America/New_York") == ny(
        "2001-01-03 13:00"
    )
    assert result.trades.iloc[1].reference_price == 140
    assert result.metrics["terminal_equity"] == pytest.approx(1400)


@pytest.mark.parametrize("interruption", [-0.0199, np.nan, np.inf, -np.inf])
def test_failed_or_undefined_entry_resets_streak(interruption):
    cache = cache_for(
        [{}, {}],
        values=[-0.02] * 6 + [interruption] + [-0.02] * 5 + [-0.01] + [-0.02] * 7 + [-0.01] * 6,
    )
    result = simulate(cache)
    assert result.trades.iloc[0].decision_at.tz_convert("America/New_York") == ny(
        "2001-01-03 13:00"
    )


def test_overnight_carry_counts_closing_halfhour_and_not_market_closure_hours():
    cache = cache_for([{}, {}], values=[-0.01] * 9 + [-0.02] * 4 + [-0.02] * 3 + [-0.01] * 10)
    result = simulate(cache)
    assert result.trades.iloc[0].decision_at.tz_convert("America/New_York") == ny(
        "2001-01-03 11:00"
    )


def test_early_close_carries_pending_fill_into_next_session_without_extra_observation():
    cache = cache_for([{"closes": [90] * 7}, {}], values=[-0.02] * 7 + [-0.01] * 13)
    result = simulate(cache)
    buy = result.trades.iloc[0]
    assert buy.decision_at.tz_convert("America/New_York") == ny("2001-01-02 13:00")
    assert buy.timestamp.tz_convert("America/New_York") == ny("2001-01-03 09:30")


def test_gap_cancels_pending_requires_fresh_seven_observations_and_flags_only_decision():
    cache = cache_for(
        [{}, {"missing": [0]}],
        post_gap=True,
        values=[-0.01] * 6 + [-0.02] * 7 + [-0.02] * 7 + [-0.01] * 5,
    )
    result = simulate(cache)
    buy = result.trades.iloc[0]
    assert buy.decision_at.tz_convert("America/New_York") == ny("2001-01-03 13:30")
    assert result.metrics["cancelled_blackout_order_count"] == 1
    assert result.metrics["post_gap_confirmed_buy_count"] == 1
    assert result.metrics["post_gap_confirmed_sell_count"] == 0
    flagged = result.diagnostics.attrs["post_gap_signal_events"]
    assert len(flagged) == 1
    assert flagged.side.tolist() == ["buy"]
    assert flagged.window_end.iloc[0].tz_convert("America/New_York") == ny("2001-01-04 16:00")


def test_held_position_split_dividend_and_missing_day_drip_reconcile_independently():
    cache = cache_for(
        [
            {"closes": [90] * 13, "daily_close": 100},
            {
                "missing": list(range(13)),
                "closes": [49.5] * 13,
                "daily_close": 49.5,
                "split": 2,
                "dividend": 0.5,
            },
            {"closes": [55] * 13, "opens": [50] * 13, "daily_close": 60},
        ],
        values=[-0.02] * 7 + [-0.01] * 6 + [-0.01] * 13,
    )
    result = simulate(cache)
    assert result.trades.side.tolist() == ["buy", "sell"]
    assert result.daily_equity.iloc[:2].tolist() == pytest.approx([1000, 1000])
    # Buy10 shares. At the missing next open, split to20 and credit$10; no fake
    # fill. At the next observed50 open, reinvest into0.2 shares then sell at55.
    assert result.trades.iloc[-1].quantity == pytest.approx(20.2)
    assert result.metrics["terminal_equity"] == pytest.approx(20.2 * 55)
    assert result.metrics["unknown_missing_slots"] == 13


def test_costs_once_adverse_prices_gross_shadow_and_no_tax():
    cache = cache_for([{}], values=[-0.02] * 7 + [-0.01] * 6)
    result = simulate(cache, fee=1, slip=3)
    qty = 1000 / (100 * 1.0003 * 1.0001)
    expected = qty * 90 * 0.9997 * 0.9999
    assert result.trades.iloc[0].quantity == pytest.approx(qty)
    assert result.metrics["terminal_equity"] == pytest.approx(expected)
    assert result.metrics["gross_terminal_equity"] == pytest.approx(900)
    assert result.metrics["total_commission"] == pytest.approx(
        qty * (100 * 1.0003 + 90 * 0.9997) * 0.0001
    )
    assert result.metrics["total_slippage"] == pytest.approx(qty * (100 + 90) * 0.0003)
    assert result.metrics["cost_drag_return"] == pytest.approx((900 - expected) / 1000)
    assert not any("tax" in key for key in result.metrics)


def independent_clock_cash(cache, annual=0.03):
    _, _, clock, _ = quality._segment_events(
        cache, cache.calendar.session.iloc[0], cache.calendar.session.iloc[-1]
    )
    seconds = np.diff(clock.asi8 / 1e9)
    return 1000 * np.prod(1 + annual * seconds / (365 * 86400))


def test_cash_only_compounds_nominal_ACT365_on_shared_event_clock_and_has_undefined_excess():
    cache = cache_for([{}, {}])
    result = simulate(cache, annual=0.03, fee=1, slip=3, mode="cash")
    assert result.metrics["terminal_equity"] == pytest.approx(independent_clock_cash(cache))
    assert np.isnan(result.metrics["cash_excess_sharpe"])
    assert result.metrics["order_count"] == 0
    assert result.metrics["exposure_fraction"] == 0
    assert result.trades.empty


def test_passive_BH_first_open_and_terminal_costs_match_independent_ledger():
    cache = cache_for([{}, {}])
    result = simulate(cache, annual=0.03, fee=1, slip=3, mode="buy_and_hold")
    qty = 1000 / (100 * 1.0003 * 1.0001)
    terminal_at_16 = qty * 90 * 0.9997 * 0.9999
    expected_at_17 = terminal_at_16 * (1 + 0.03 / (365 * 24))
    assert result.metrics["terminal_equity"] == pytest.approx(expected_at_17)
    assert result.trades.reason.tolist() == ["benchmark_entry", "terminal_liquidation"]
    assert result.metrics["candidate_id"] == "benchmark_buy_and_hold"


def test_segment_boundary_starts_cash_and_clears_old_streaks():
    cache = cache_for([{}, {}], values=[-0.02] * 13 + [-0.02] * 6 + [-0.01] * 7)
    parts = engine.run(
        cache,
        [candidate()],
        start="2001-01-03",
        end="2001-01-03",
        cash_observations=observations(cache),
        commission_bps=0,
        slippage_bps=0,
    )
    assert parts[0][0]["order_count"] == 0
    assert parts[0][0]["terminal_equity"] == pytest.approx(1000)


@pytest.mark.parametrize("field", ["source", "features", "execution", "eligibility"])
def test_cache_mutation_is_rejected(field):
    cache = cache_for([{}])
    if field == "source":
        cache.daily.loc[0, "close"] = 99
    elif field == "features":
        cache.margins[0, 0] = 123
    elif field == "execution":
        cache = replace(cache, bar_rows=cache.bar_rows[:-1])
    else:
        cache.signal_eligible[0] = False
    with pytest.raises(ValueError, match="identity changed"):
        simulate(cache)


def test_non_MR_cache_duplicate_or_missing_rule_and_invalid_mode_fail_closed():
    cache = cache_for([{}])
    kwargs = dict(start="2001-01-02", end="2001-01-02", cash_observations=observations(cache))
    with pytest.raises(ValueError, match="cache required"):
        engine.run(object(), [candidate()], **kwargs)
    with pytest.raises(ValueError, match="unique batch"):
        engine.run(cache, [candidate(), candidate()], **kwargs)
    with pytest.raises(ValueError, match="Mode"):
        engine.run(cache, [candidate()], mode="validation", **kwargs)
    with pytest.raises(ValueError, match="one placeholder"):
        engine.run(cache, mr.candidate_grid()[:2], mode="cash", **kwargs)


def test_undefined_recovery_interrupts_exit_streak_and_does_not_force_sell():
    cache = cache_for([{}, {}], values=[-0.02] * 7 + [-0.01] * 6 + [0.0] * 6 + [np.nan] + [0.0] * 6)
    result = simulate(cache)
    assert result.trades.reason.tolist() == ["confirmed_entry", "terminal_liquidation"]
    assert result.metrics["cancelled_terminal_order_count"] == 0


def test_terminal_confirmation_cancels_order_without_inventing_next_fill():
    cache = cache_for([{}], values=[-0.01] * 6 + [-0.02] * 7)
    result = simulate(cache)
    assert result.trades.empty
    assert result.metrics["cancelled_terminal_order_count"] == 1
    assert result.metrics["terminal_equity"] == 1000


def test_future_quotes_do_not_change_earlier_features_decisions_or_fills():
    original = cache_for([{}, {"closes": [110] * 13, "opens": [115] * 13}])
    changed = cache_for([{}, {"closes": [40] * 13, "opens": [45] * 13}])
    np.testing.assert_array_equal(original.margins[:13], changed.margins[:13])
    original_result, changed_result = simulate(original), simulate(changed)
    pd.testing.assert_series_equal(original_result.trades.iloc[0], changed_result.trades.iloc[0])
    assert original_result.daily_equity.iloc[0] == changed_result.daily_equity.iloc[0]


def test_all90_batch_is_numerically_identical_to_scalar_across_families():
    cache = cache_for([{}, {"closes": [110] * 13, "opens": [115] * 13}])
    candidates = mr.candidate_grid()
    kwargs = dict(
        start=cache.calendar.session.iloc[0],
        end=cache.calendar.session.iloc[-1],
        cash_observations=observations(cache, 0.03),
        commission_bps=1,
        slippage_bps=3,
    )
    batch = engine.run(cache, candidates, **kwargs)
    assert len(batch[0]) == 90
    assert batch[2].shape == (2, 90)
    for index in (0, 18, 36, 63, 89):
        scalar = engine.run(cache, [candidates[index]], **kwargs)
        for position in (2, 3, 5):
            np.testing.assert_array_equal(batch[position][:, index], scalar[position][:, 0])
        np.testing.assert_array_equal(batch[4], scalar[4])
        for name, value in scalar[0][0].items():
            actual = batch[0][index][name]
            if isinstance(value, float) and np.isnan(value):
                assert np.isnan(actual)
            else:
                assert actual == value


def test_fixed3pct_cash_before_entry_and_after_terminal_reconciles_event_ledger():
    cache = cache_for([{}], values=[-0.02] * 7 + [-0.01] * 6)
    result = simulate(cache, annual=0.03, fee=1, slip=3)
    # Seven observed half-hour intervals elapse before the13:00 next-bar fill;
    # cash then earns no interest while invested, and one17:00 tail interval
    # accrues only after16:00 terminal liquidation.
    before_buy = 1000 * (1 + 0.03 * 0.5 / (365 * 24)) ** 7
    qty = before_buy / (100 * 1.0003 * 1.0001)
    expected = qty * 90 * 0.9997 * 0.9999 * (1 + 0.03 / (365 * 24))
    assert result.trades.iloc[0].equity_before == pytest.approx(before_buy)
    assert result.metrics["terminal_equity"] == pytest.approx(expected)
