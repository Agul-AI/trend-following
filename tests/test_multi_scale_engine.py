"""Synthetic multi-scale gates, execution and fixed-sleeve arithmetic tests."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trend_following import research_mean_reversion as mr
from trend_following import research_mean_reversion_engine as mr_engine
from trend_following import research_multi_scale_engine as engine
from trend_following import research_portfolio_engine as portfolio
from trend_following import research_quality_engine_v5 as quality


def fixture(specs, history=(100.0,) * 210):
    dates = pd.bdate_range("1999-11-01", periods=len(history))
    daily = [
        dict(session=d.strftime("%Y-%m-%d"), close=p, dividend_amount=0.0, split_coefficient=1.0)
        for d, p in zip(dates, history, strict=True)
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
        for k, (op, close) in enumerate(zip(opens, closes, strict=True)):
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
                        high=max(op, close),
                        low=min(op, close),
                        close=close,
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


def base_candidate():
    return mr.Candidate(mr.Rule("SMA_DISTANCE", 10), -0.02, 0.0)


def candidate(policy="entry_only", period=100):
    return engine.GatedCandidate(base_candidate(), period, policy)


def cache_for(specs, *, values=None, gates=None, post_gap=False, history=(100.0,) * 210):
    daily, bars, calendar, coverage = fixture(specs, history)
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
        margins = cache.base.margins.copy()
        margins[:, cache.rules.index(base_candidate().rule)] = np.asarray(values, dtype=float)
        base = replace(cache.base, margins=margins, feature_fingerprint="")
        base = replace(base, feature_fingerprint=quality._feature_fingerprint(base))
        cache = replace(cache, base=base, gate_fingerprint="")
    if gates is not None:
        gate_values = cache.gates.copy()
        gate_values.iloc[len(history) - 1 :, :] = np.repeat(
            np.asarray(gates, dtype=np.int8)[:, None], 2, axis=1
        )
        cache = replace(cache, gates=gate_values, gate_fingerprint="")
    return replace(cache, gate_fingerprint=engine._gate_fingerprint(cache))


def observations(cache, annual=0.03):
    dates = pd.date_range(
        cache.calendar.market_open.iloc[0].normalize() - pd.Timedelta(days=1),
        cache.calendar.market_close.iloc[-1].normalize() + pd.Timedelta(days=2),
        freq="D",
        tz="UTC",
    )
    return pd.DataFrame(dict(available_at=dates, annual_rate=annual))


def run(
    cache,
    candidates=None,
    *,
    annual=0.03,
    fee=1.0,
    slip=3.0,
    start=None,
    end=None,
    mode="candidate",
):
    return engine.run(
        cache,
        [candidate()] if candidates is None else candidates,
        start=cache.calendar.session.iloc[0] if start is None else start,
        end=cache.calendar.session.iloc[-1] if end is None else end,
        cash_observations=observations(cache, annual),
        commission_bps=fee,
        slippage_bps=slip,
        record=True,
        mode=mode,
    )


def ny(value):
    return pd.Timestamp(value, tz="America/New_York").tz_convert("UTC")


def test_grid_has_128_exact_unmodified_members_and_four_baskets():
    members = tuple(mr.candidate_grid()[:32])
    grid = engine.candidate_grid(reversed(members))
    assert len(grid) == len({c.candidate_id for c in grid}) == 128
    assert len({c.basket_id for c in grid}) == 4
    for basket in {c.basket_id for c in grid}:
        assert {c.base for c in grid if c.basket_id == basket} == set(members)
    assert all(c.confirmation_hours == 3 and c.required_checks == 7 for c in grid)
    assert all(engine.GatedCandidate.from_dict(c.to_dict()) == c for c in grid)
    assert len(engine.gate_only_candidates()) == 2


@pytest.mark.parametrize("period,policy", [(100, "entry_only"), (200, "recovery_or_trend_break")])
def test_always_on_reproduces_original_mr_paths_fills_and_all_ledger_charges(period, policy):
    members = tuple(mr.candidate_grid()[:32])
    cache = cache_for([{}, {"closes": [130] * 13, "opens": [105] * 13}, {"closes": [85] * 13}])
    cache = engine.with_constant_gate(cache, engine.ON)
    kwargs = dict(
        start=cache.calendar.session.iloc[0],
        end=cache.calendar.session.iloc[-1],
        cash_observations=observations(cache),
        commission_bps=1,
        slippage_bps=25,
        record=True,
    )
    original = mr_engine.run(cache.base, members, **kwargs)
    modified = engine.run(
        cache, [engine.GatedCandidate(c, period, policy) for c in members], **kwargs
    )
    for i in (2, 3, 4, 5):
        np.testing.assert_array_equal(original[i], modified[i])
    assert original[1].equals(modified[1])
    for left, right in zip(original[0], modified[0], strict=True):
        for key in (
            "cash_excess_sharpe",
            "cagr",
            "max_drawdown",
            "terminal_equity",
            "gross_terminal_equity",
            "total_commission",
            "total_slippage",
            "trade_count",
            "average_holding_hours",
            "total_turnover",
        ):
            assert left[key] == pytest.approx(right[key], nan_ok=True)
    for left, right in zip(original[6], modified[6], strict=True):
        for key in left:
            if key != "candidate_id":
                assert left[key] == right[key]


@pytest.mark.parametrize("state", [engine.OFF, engine.UNKNOWN])
def test_constant_off_or_unknown_is_exact_cash(state):
    cache = engine.with_constant_gate(cache_for([{}, {}, {}]), state)
    parts = run(cache)
    cash = run(cache, mode="cash")
    np.testing.assert_array_equal(parts[2], cash[2])
    np.testing.assert_array_equal(parts[4], cash[4])
    assert parts[6] == []
    assert np.isnan(parts[0][0]["cash_excess_sharpe"])
    assert parts[0][0]["blocked_oversold_checks"] == 39


@pytest.mark.parametrize("period", [100, 200])
def test_gate_includes_current_final_value_and_strict_equality(period):
    values = np.full(period + 2, 100.2)
    gates = engine.finalized_gates(values)
    assert (gates[period].iloc[: period - 1] == engine.UNKNOWN).all()
    assert (gates[period].iloc[period - 1 :] == engine.OFF).all()
    values[period - 1] = 101
    assert engine.finalized_gates(values)[period].iloc[period - 1] == engine.ON
    values[period - 1] = 99
    assert engine.finalized_gates(values)[period].iloc[period - 1] == engine.OFF


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf, 0, -1])
def test_incomplete_or_nonfinite_gate_window_unknown(missing):
    values = np.arange(1, 202, dtype=float)
    values[102] = missing
    result = engine.finalized_gates(values)
    assert result[100].iloc[-1] == engine.UNKNOWN
    assert result[200].iloc[-1] == engine.UNKNOWN


def test_final_gate_only_changes_at_seventeen_and_not_intraday():
    history = tuple(np.linspace(80, 100, 210))
    cache = cache_for(
        [{"daily_close": 60, "opens": [80] * 13}, {}], values=[-0.03] * 26, history=history
    )
    parts = run(cache)
    buy = parts[6][0]
    assert buy["decision_at"] == ny("2001-01-02 13:00")
    assert buy["timestamp"] == ny("2001-01-02 13:00")
    assert buy["reference_price"] == 80
    updates = [d for d in parts[7] if d["event"] == "gate_update"]
    assert all(d["timestamp"].tz_convert("America/New_York").hour == 17 for d in updates)
    assert updates[0]["gate_state"] == engine.OFF
    assert updates[0]["entry_count"] == 0


def test_off_update_cancels_overnight_buy_and_resets_entry_not_counting_daily_update():
    cache = cache_for(
        [{}, {}, {}], values=[-0.01] * 6 + [-0.03] * 7 + [-0.03] * 26, gates=[1, 0, 1, 1]
    )
    parts = run(cache)
    assert parts[0][0]["entry_cancellation_count"] == 1
    assert parts[6][0]["decision_at"] == ny("2001-01-04 13:00")
    assert parts[6][0]["timestamp"] == ny("2001-01-04 13:00")
    assert parts[0][0]["blocked_oversold_checks"] == 13


def test_entry_only_keeps_position_under_off_and_exits_on_original_recovery():
    cache = cache_for(
        [{}, {}, {}],
        values=[-0.03] * 13 + [-0.01] * 13 + [0.0] * 7 + [-0.01] * 6,
        gates=[1, 0, 0, 0],
    )
    parts = run(cache)
    assert parts[6][1]["decision_at"] == ny("2001-01-04 13:00")
    assert parts[6][1]["exit_causes"] == "recovery"
    assert parts[0][0]["trend_break_exit_count"] == 0


def test_trend_break_has_independent_seven_checks_and_both_causes_are_retained():
    cache = cache_for([{}, {}], values=[-0.03] * 13 + [0.0] * 13, gates=[1, 0, 0])
    parts = run(cache, [candidate("recovery_or_trend_break")])
    sell = parts[6][1]
    assert sell["decision_at"] == ny("2001-01-03 13:00")
    assert sell["exit_causes"] == "recovery+trend_break"
    assert parts[0][0]["trend_break_exit_count"] == 1


def test_on_update_cancels_break_only_overnight_exit_but_keeps_recovery_exit():
    # Enter day1. Day2's six-slot early close leaves six OFF checks; day3's
    # first check is number7 then filling at the next open would be immediate.
    # Instead a seven-slot earlyclose queues the exit at its final bar close.
    cache = cache_for(
        [{}, {"closes": [90] * 7}, {}],
        values=[-0.03] * 13 + [-0.01] * 7 + [-0.01] * 13,
        gates=[1, 0, 1, 1],
    )
    parts = run(cache, [candidate("recovery_or_trend_break")])
    assert parts[0][0]["exit_cancellation_count"] == 1
    assert parts[6][1]["reason"] == "terminal_liquidation"
    values = [-0.03] * 13 + [0.0] * 7 + [-0.01] * 13
    both = run(
        cache_for([{}, {"closes": [90] * 7}, {}], values=values, gates=[1, 0, 1, 1]),
        [candidate("recovery_or_trend_break")],
    )
    assert both[0][0]["exit_cancellation_count"] == 0
    assert both[6][1]["timestamp"] == ny("2001-01-04 09:30")
    assert both[6][1]["exit_causes"] == "recovery+trend_break"


def test_unknown_update_holds_cancels_orders_resets_then_requires_new_streak():
    cache = cache_for(
        [{}, {"closes": [90] * 7}, {}, {}],
        values=[-0.03] * 13 + [0.0] * 7 + [0.0] * 26,
        gates=[1, 0, -1, 1, 1],
    )
    parts = run(cache, [candidate("recovery_or_trend_break")])
    assert parts[0][0]["exit_cancellation_count"] == 1
    assert parts[6][1]["decision_at"] == ny("2001-01-05 13:00")
    assert parts[6][1]["exit_causes"] == "recovery"


def test_gap_resets_counters_cancels_and_retains_descriptive_flags():
    cache = cache_for(
        [{}, {"missing": [0]}],
        values=[-0.01] * 6 + [-0.03] * 7 + [-0.03] * 7 + [-0.01] * 5,
        gates=[1, 1, 1],
        post_gap=True,
    )
    parts = run(cache)
    assert parts[0][0]["cancelled_blackout_order_count"] == 1
    assert parts[6][0]["decision_at"] == ny("2001-01-03 13:30")
    assert parts[0][0]["post_gap_confirmed_signal_count"] == 1
    assert parts[8]["rows"][0][-1] == ny("2001-01-04 16:00")


def test_closures_carry_entry_streak_without_counting_weekend_or_overnight():
    cache = cache_for(
        [{"session": "2001-01-05"}, {"session": "2001-01-08"}],
        values=[-0.01] * 9 + [-0.03] * 4 + [-0.03] * 3 + [-0.01] * 10,
        gates=[1, 1, 1],
    )
    parts = run(cache)
    assert parts[6][0]["decision_at"] == ny("2001-01-08 11:00")


def test_gate_only_controls_do_not_require_oversold_or_recovery():
    cache = cache_for([{}, {}], values=[np.nan] * 26, gates=[1, 0, 0])
    parts = run(cache, engine.gate_only_candidates())
    assert len(parts[0]) == 2
    for i in range(2):
        records = [r for r in parts[6] if r["candidate_id"] == parts[0][i]["candidate_id"]]
        assert records[0]["decision_at"] == ny("2001-01-02 13:00")
        assert records[1]["decision_at"] == ny("2001-01-03 13:00")
        assert records[1]["exit_causes"] == "trend_break"


def test_future_daily_or_intraday_quotes_do_not_alter_earlier_decisions():
    history = tuple(np.linspace(80, 100, 210))
    first = cache_for([{}, {}, {"daily_close": 40}], history=history)
    future = cache_for(
        [{}, {}, {"daily_close": 2000, "closes": [2000] * 13, "opens": [1990] * 13}],
        history=history,
    )
    a, b = run(first), run(future)
    cutoff = ny("2001-01-04 09:30")
    assert [r for r in a[6] if r["timestamp"] < cutoff] == [
        r for r in b[6] if r["timestamp"] < cutoff
    ]
    np.testing.assert_array_equal(first.gates.iloc[:-1], future.gates.iloc[:-1])
    np.testing.assert_array_equal(a[2][:-1], b[2][:-1])


def test_segment_boundaries_reset_position_counters_and_orders():
    cache = cache_for([{}, {}], values=[-0.03] * 13 + [-0.03] * 6 + [-0.01] * 7, gates=[1, 1, 1])
    second = run(cache, start="2001-01-03")
    assert second[6] == []
    assert second[0][0]["trade_count"] == 0


def test_split_dividend_costs_cash_and_terminal_reconcile_independent_replay():
    cache = cache_for(
        [
            {"daily_close": 100},
            {
                "missing": list(range(13)),
                "closes": [49.5] * 13,
                "daily_close": 49.5,
                "split": 2,
                "dividend": 0.5,
            },
            {"closes": [55] * 13, "opens": [50] * 13, "daily_close": 60},
        ],
        values=[-0.03] * 7 + [-0.01] * 19,
        gates=[1, 1, 1, 1],
    )
    parts = run(cache)
    replay = portfolio.replay_recorded(
        cache, parts, start="2001-01-02", end="2001-01-04", cash_observations=observations(cache)
    )
    basket = engine.aggregate_basket(replay, "one")
    np.testing.assert_allclose(basket.daily_equity, parts[2][:, 0], rtol=0, atol=1e-10)
    assert basket.metrics["terminal_equity"] == parts[0][0]["terminal_equity"]
    assert basket.metrics["total_commission"] == parts[0][0]["total_commission"]
    assert basket.metrics["gross_sleeve_order_count"] == 2
    assert basket.metrics["gross_sleeve_completed_round_trips"] == 1


def test_generic_batch_mask_has_no_implicit_rebalancing_or_zero_sleeve_charges():
    cache = cache_for(
        [{}, {"closes": [130] * 13}, {}],
        values=[-0.03] * 13 + [0] * 13 + [-0.03] * 13,
        gates=[1, 0, 1, 1],
    )
    candidates = [candidate("entry_only"), candidate("recovery_or_trend_break")]
    parts = run(cache, candidates)
    paths = portfolio.replay_recorded(
        cache, parts, start="2001-01-02", end="2001-01-04", cash_observations=observations(cache)
    )
    result = engine.aggregate_basket(paths, "mixed", [0.25, 0.75])
    np.testing.assert_allclose(result.daily_equity, paths.equity @ np.array([0.25, 0.75]))
    single = engine.aggregate_basket(paths, "single", [0, 1])
    np.testing.assert_array_equal(single.daily_equity, paths.equity[:, 1])
    assert single.metrics["gross_sleeve_order_count"] == paths.metrics[1]["order_count"]
    assert single.metrics["total_commission"] == paths.metrics[1]["total_commission"]


@pytest.mark.parametrize("weights", [[-0.1, 1.1], [0.4, 0.4], [np.nan, 1], [1]])
def test_invalid_basket_weights_rejected(weights):
    cache = cache_for([{}, {}], values=[-0.03] * 26, gates=[1, 1, 1])
    parts = run(cache, [candidate(), candidate(period=200)])
    paths = portfolio.replay_recorded(
        cache, parts, start="2001-01-02", end="2001-01-03", cash_observations=observations(cache)
    )
    with pytest.raises(ValueError, match="weights"):
        engine.aggregate_basket(paths, "bad", weights)


def test_cache_source_and_gate_mutation_are_rejected():
    cache = cache_for([{}])
    cache.gates.iloc[-1, 0] = engine.ON
    with pytest.raises(ValueError, match="gate identity"):
        run(cache)
    cache = cache_for([{}])
    cache.base.daily.loc[0, "close"] = 123
    with pytest.raises(ValueError, match="source/coverage"):
        run(cache)


@pytest.mark.parametrize("bad", [[], [base_candidate()], [candidate()] * 2])
def test_invalid_candidate_batches_rejected(bad):
    with pytest.raises(ValueError, match="unique batch"):
        run(cache_for([{}]), bad)


def test_recovery_and_off_streaks_never_pool_into_one_exit():
    # Six recovery observations while ON cannot be pooled with one OFF
    # observation whose MR value fails recovery the following morning.
    cache = cache_for(
        [{}, {}],
        values=[-0.03] * 7 + [0.0] * 6 + [-0.01] * 13,
        gates=[1, 0, 0],
    )
    parts = run(cache, [candidate("recovery_or_trend_break")])
    sell = parts[6][1]
    assert sell["decision_at"] == ny("2001-01-03 13:00")
    assert sell["exit_causes"] == "trend_break"
    first_off = next(
        d
        for d in parts[7]
        if d["event"] == "bar_close" and d["timestamp"] == ny("2001-01-03 10:00")
    )
    assert first_off["exit_count"] == 0
    assert first_off["trend_break_count"] == 1
    assert first_off["pending"] == 0


def test_gate_update_does_not_increment_recovery_counter_overnight():
    cache = cache_for([{}, {}], values=[-0.03] * 7 + [0.0] * 6 + [0.0] * 13, gates=[1, 1, 1])
    parts = run(cache)
    first_update = next(d for d in parts[7] if d["event"] == "gate_update")
    assert first_update["exit_count"] == 6
    assert first_update["pending"] == 0
    assert parts[6][1]["decision_at"] == ny("2001-01-03 10:00")


@pytest.mark.parametrize("period", [True, 100.0, 101, "100"])
def test_gate_period_requires_exact_integer_frozen_lookback(period):
    with pytest.raises(ValueError, match="period"):
        engine.GatedCandidate(base_candidate(), period, "entry_only")
    with pytest.raises(ValueError, match="period"):
        engine.GateOnlyCandidate(period)


def test_full_result_container_preserves_both_exit_causes():
    cache = cache_for([{}, {}], values=[-0.03] * 13 + [0] * 13, gates=[1, 0, 0])
    result = engine.result(run(cache, [candidate("recovery_or_trend_break")]))
    assert result.trades.exit_causes.tolist() == ["", "recovery+trend_break"]


def test_per_candidate_cancellation_trace_in_batch():
    cache = cache_for(
        [{}, {}, {}], values=[-0.01] * 6 + [-0.03] * 7 + [-0.03] * 26, gates=[1, 0, 1, 1]
    )
    candidates = [candidate(), candidate(period=200)]
    parts = run(cache, candidates)
    cancellations = [d for d in parts[7] if d["event"] == "gate_order_cancelled"]
    assert {d["candidate_id"] for d in cancellations} == {c.candidate_id for c in candidates}
    assert all(d["reason"] == "off_entry_gate" and d["side"] == "buy" for d in cancellations)


def test_basket_retains_blackout_and_terminal_cancellation_categories():
    cache = cache_for(
        [{}, {"missing": [0]}],
        values=[-0.01] * 6 + [-0.03] * 7 + [-0.01] * 5 + [-0.03] * 7,
        gates=[1, 1, 1],
    )
    parts = run(cache, [candidate(), candidate(period=200)])
    assert all(m["cancelled_blackout_order_count"] == 1 for m in parts[0])
    assert all(m["cancelled_terminal_order_count"] == 1 for m in parts[0])
    paths = portfolio.replay_recorded(
        cache,
        parts,
        start="2001-01-02",
        end="2001-01-03",
        cash_observations=observations(cache),
    )
    basket = engine.aggregate_basket(paths, "equal")
    assert basket.metrics["cancelled_blackout_order_count"] == 2
    assert basket.metrics["cancelled_terminal_order_count"] == 2
    selected = engine.aggregate_basket(paths, "single", [1, 0])
    assert selected.metrics["cancelled_blackout_order_count"] == 1
    assert selected.metrics["cancelled_terminal_order_count"] == 1
