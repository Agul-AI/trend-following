"""Hand-checkable causal execution examples, independent of legacy backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from trend_following.research_execution import (
    ExecutionPolicyEngine,
    build_execution_path,
    causal_features,
    normalize_spec,
)


def features(**changes):
    return {
        "entry": True,
        "exit": False,
        "long_ma": 90.0,
        "short_ma": 95.0,
        "medium_ma": 93.0,
        "distance": 0.1,
        "long_slope": 1.0,
        "medium_slope": 1.0,
        "vol_percentile": 0.5,
        **changes,
    }


def tick(engine, day, risk=100.0, qqq=100.0, f=None):
    return engine.step(pd.Timestamp(day), qqq, risk, features() if f is None else f)


def test_next_close_fill_then_only_following_bar_returns():
    idx = pd.date_range("2024-01-02 10:00", periods=4, freq="h")
    risk = pd.Series([100.0, 200.0, 400.0, 800.0], index=idx)
    path = build_execution_path(risk, risk, {"kind": "buyhold"})
    assert path.weights_at_close.tolist() == [0.0, 1.0, 1.0, 1.0]
    assert path.target_weights.tolist() == [0.0, 0.0, 1.0, 1.0]
    fill = path.events.query("event == 'filled'").iloc[0]
    assert fill.observation_timestamp == idx[0]
    assert fill.observation_timestamp < fill.decision_timestamp < fill.order_timestamp
    assert fill.order_timestamp < fill.fill_timestamp == idx[1]
    assert fill.return_timestamp == idx[2]
    assert path.decisions.entry_price.iloc[1] == 200.0


def test_extra_delay_is_in_bars_not_days_and_fill_cost_reference():
    idx = pd.date_range("2024-01-02 10:00", periods=5, freq="h")
    p = pd.Series(np.arange(100.0, 105.0), index=idx)
    path = build_execution_path(
        p, p, {"kind": "buyhold", "fill_delay_bars": 2}, slippage_bps=10, transaction_cost_bps=5
    )
    fill = path.events.query("event == 'filled'").iloc[0]
    assert fill.fill_timestamp == idx[2]
    assert fill.fill_quote == pytest.approx(102 * 1.001)
    assert path.decisions.entry_price.iloc[2] == pytest.approx(102 * (1 + 0.001 + 0.0005))


def test_fill_session_not_return_label_enforces_one_change_including_exits():
    engine = ExecutionPolicyEngine({"kind": "buyhold"})
    tick(engine, "2024-01-02 15:00")  # submit entry
    tick(engine, "2024-01-03 10:00")  # fill entry in Jan 3 session
    engine.switch_candidate({"kind": "cash"}, pd.Timestamp("2024-01-03 10:30"))
    tick(engine, "2024-01-03 11:00")  # submit exit
    held = tick(engine, "2024-01-03 12:00")  # cannot fill another change today
    assert held["weights_at_close"] == 1
    exited = tick(engine, "2024-01-04 10:00")
    assert exited["weights_at_close"] == 0
    fills = pd.DataFrame(engine.events).query("event == 'filled'")
    assert fills.groupby("fill_session").size().max() == 1
    assert fills.fill_timestamp.iloc[-1] == pd.Timestamp("2024-01-04 10:00")


def test_bear_blocked_raw_entry_cannot_start_trade_or_learn_overlay():
    engine = ExecutionPolicyEngine({"kind": "b5"})
    blocked = features(long_slope=-1.0, medium_slope=-1.0)
    first = tick(engine, "2024-01-02", risk=100.0, f=blocked)
    second = tick(engine, "2024-01-03", risk=500.0, f=blocked)
    assert first["blocked_entry"] and second["blocked_entry"]
    assert np.isnan(second["entry_price"])
    assert np.isnan(second["qtrim_threshold"])
    assert engine.events == []
    tick(engine, "2024-01-04", risk=600.0)  # now valid signal, submit
    filled = tick(engine, "2024-01-05", risk=700.0)
    assert filled["entry_price"] == 700.0
    assert filled["profit_weight"] == 1.0
    assert np.isnan(filled["qtrim_threshold"])
    assert filled["peak_price"] == 700.0


def test_profit_lock_uses_accepted_entry_and_peak_stop_latches_until_base_exit():
    engine = ExecutionPolicyEngine({"kind": "b5", "use_qtrim": False})
    tick(engine, "2024-01-02", risk=10.0)
    tick(engine, "2024-01-03", risk=100.0)
    row = tick(engine, "2024-01-04", risk=400.0)
    assert row["profit_weight"] == 0.75
    assert row["desired_weight"] == 0.75
    row = tick(engine, "2024-01-05", risk=500.0)
    assert row["profit_weight"] == 0.5
    # Peak 500 -> 299 exceeds 40% stop, despite any pending partial reduction.
    row = tick(engine, "2024-01-06", risk=299.0)
    assert row["desired_weight"] == 0
    row = tick(engine, "2024-01-07", risk=500.0)
    assert row["weights_at_close"] == 0
    assert row["desired_weight"] == 0
    tick(engine, "2024-01-08", risk=500.0, f=features(entry=False, exit=True))
    row = tick(engine, "2024-01-09", risk=500.0)
    assert row["desired_weight"] == 1


def test_q110_learns_running_max_after_filled_entry_and_readds_on_ma20():
    engine = ExecutionPolicyEngine({"kind": "b5", "use_profit_lock": False})
    tick(engine, "2024-01-02", risk=10.0, f=features(distance=0.9))
    tick(engine, "2024-01-03", risk=100.0, f=features(distance=0.10))
    tick(engine, "2024-01-04", risk=150.0, f=features(distance=0.20))
    row = tick(engine, "2024-01-05", risk=210.0, f=features(distance=0.15))
    assert row["qtrim_threshold"] == pytest.approx(0.20)
    assert not row["qtrim_defensive"]
    row = tick(engine, "2024-01-06", risk=220.0, f=features(distance=0.20))
    assert row["qtrim_defensive"] and row["desired_weight"] == 0.5
    row = tick(engine, "2024-01-07", risk=215.0, qqq=94.0, f=features(distance=0.08))
    assert not row["qtrim_defensive"] and row["desired_weight"] == 1


def test_b5_five_day_cooldown_preserves_frozen_bar_count_convention():
    engine = ExecutionPolicyEngine({"kind": "b5"}, bars_per_day=1)
    tick(engine, "2024-01-01")
    tick(engine, "2024-01-02")
    row = tick(engine, "2024-01-03", f=features(vol_percentile=0.96))
    assert row["desired_weight"] == pytest.approx(1 / 3)
    for day, remaining in [(4, 4), (5, 3), (6, 2), (7, 1)]:
        row = tick(engine, f"2024-01-{day:02}", f=features(vol_percentile=0.1))
        assert row["cooldown_remaining"] == remaining
        assert row["desired_weight"] == pytest.approx(1 / 3)
    row = tick(engine, "2024-01-08", f=features(vol_percentile=0.1))
    assert row["desired_weight"] == 1.0
    anchor = ExecutionPolicyEngine({"kind": "anchor"}, bars_per_day=1)
    tick(anchor, "2024-01-01")
    tick(anchor, "2024-01-02")
    tick(anchor, "2024-01-03", f=features(vol_percentile=0.96))
    assert tick(anchor, "2024-01-04", f=features(vol_percentile=0.1))["desired_weight"] == 1.0


def test_switch_preserves_actual_trade_reference_and_cancels_pending_rule():
    engine = ExecutionPolicyEngine({"kind": "buyhold"})
    tick(engine, "2024-01-02", risk=100.0)
    tick(engine, "2024-01-03", risk=110.0)
    engine.switch_candidate({"kind": "b5"}, pd.Timestamp("2024-01-04"))
    row = tick(engine, "2024-01-04", risk=120.0)
    assert row["entry_price"] == 110.0
    assert row["trade_id"] == 1
    assert row["peak_price"] == 120.0
    engine.switch_candidate({"kind": "cash"}, pd.Timestamp("2024-01-05"))
    tick(engine, "2024-01-05", risk=120.0)  # cash order pending
    engine.switch_candidate({"kind": "b5"}, pd.Timestamp("2024-01-06"))
    row = tick(engine, "2024-01-06", risk=130.0)
    assert row["entry_price"] == 110.0
    assert row["weights_at_close"] == 1.0
    assert any(e["event"] == "cancelled_candidate_switch" for e in engine.events)


def test_transition_only_allocation_drifts_not_maintained_cap():
    engine = ExecutionPolicyEngine({"kind": "buyhold", "effective_leverage": 1.0})
    tick(engine, "2024-01-02")
    tick(engine, "2024-01-03")
    row = tick(engine, "2024-01-04", risk=130.0)
    assert row["weights_at_close"] == pytest.approx(1 / 3)
    assert row["effective_target_exposure"] == 1.0
    assert row["actual_effective_exposure"] == pytest.approx(1.3 / 1.1)
    assert sum(e["event"] == "filled" for e in engine.events) == 1


def test_evaluation_starts_flat_even_with_warmup_and_schedule_switch_is_continuous():
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    price = pd.Series(np.arange(100.0, 120.0), index=idx)
    path = build_execution_path(
        price,
        price,
        {"kind": "buyhold"},
        evaluate_start=idx[10],
        candidate_schedule={idx[14]: {"kind": "buyhold", "effective_leverage": 1.0}},
    )
    assert path.weights_at_close.iloc[0] == 0.0
    assert path.decisions.entry_price.iloc[1] == 111.0
    assert path.decisions.entry_price.iloc[-1] == 111.0
    assert path.decisions.trade_id.iloc[-1] == 1
    assert path.weights_at_close.iloc[-1] == pytest.approx(1 / 3)


def test_future_price_perturbations_do_not_change_past_features_decisions_or_fills():
    idx = pd.date_range("2020-01-01", periods=500, freq="D")
    price = pd.Series(
        100 * np.exp(np.arange(500) * 0.002 + 0.05 * np.sin(np.arange(500) / 13)), index=idx
    )
    risk = pd.Series(100 * np.exp(np.arange(500) * 0.004), index=idx)
    spec = {
        "kind": "b5",
        "long_ma_days": 20,
        "short_ma_days": 3,
        "medium_ma_days": 8,
        "macd_fast_days": 3,
        "macd_slow_days": 6,
        "macd_signal_days": 2,
        "vol_window_days": 5,
        "vol_lookback_days": 8,
        "bear_filter_slope_days": 4,
    }
    cutoff = idx[350]
    altered, altered_risk = price.copy(), risk.copy()
    altered.loc[cutoff:] *= np.exp(np.linspace(1.0, 3.0, 150))
    altered_risk.loc[cutoff:] *= 3.0
    a = build_execution_path(price, risk, spec, bars_per_day=1, evaluate_start=idx[30])
    b = build_execution_path(altered, altered_risk, spec, bars_per_day=1, evaluate_start=idx[30])
    assert_frame_equal(a.decisions.loc[: idx[349]], b.decisions.loc[: idx[349]])
    assert_frame_equal(
        a.events[a.events.event_timestamp < cutoff].reset_index(drop=True),
        b.events[b.events.event_timestamp < cutoff].reset_index(drop=True),
    )
    fa, fb = (
        causal_features(price, spec, bars_per_day=1),
        causal_features(altered, spec, bars_per_day=1),
    )
    assert_frame_equal(fa.loc[: idx[349]], fb.loc[: idx[349]])
    assert_series_equal(a.weights_at_close.loc[: idx[349]], b.weights_at_close.loc[: idx[349]])


def test_cash_has_schema_and_rejects_invalid_or_misaligned_input():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    p = pd.Series(100.0, index=idx)
    path = build_execution_path(p, p, {"kind": "cash"})
    assert path.events.empty and "event" in path.events.columns
    assert path.weights_at_close.eq(0).all()
    with pytest.raises(ValueError):
        build_execution_path(p, p.iloc[1:])
    with pytest.raises(ValueError):
        normalize_spec({"kind": "unknown"})
    with pytest.raises(ValueError):
        normalize_spec({"kind": "buyhold", "delay_bars": 0})


def test_training_chosen_allocation_multiplier_scales_targets_not_price_signals():
    a = ExecutionPolicyEngine({"kind": "b5"})
    b = ExecutionPolicyEngine({"kind": "b5", "allocation_multiplier": 0.5})
    for day, risk in enumerate([100.0, 100.0, 130.0, 200.0, 400.0], start=1):
        ar = tick(a, f"2024-01-{day:02}", risk=risk)
        br = tick(b, f"2024-01-{day:02}", risk=risk)
        assert br["desired_weight"] == pytest.approx(ar["desired_weight"] * 0.5)
        assert br["weights_at_close"] == pytest.approx(ar["weights_at_close"] * 0.5)
    with pytest.raises(ValueError):
        normalize_spec({"allocation_multiplier": 1.1})


def test_actual_open_trade_peak_tracks_simple_policy_before_b5_switch():
    engine = ExecutionPolicyEngine({"kind": "buyhold"})
    tick(engine, "2024-01-01", risk=100.0)
    tick(engine, "2024-01-02", risk=100.0)
    tick(engine, "2024-01-03", risk=200.0)
    engine.switch_candidate({"kind": "b5"}, pd.Timestamp("2024-01-04"))
    row = tick(engine, "2024-01-04", risk=110.0)
    assert row["peak_price"] == 200.0
    assert row["desired_weight"] == 0.0


def test_entry_fill_preserves_causal_vol_trigger_from_order_decision():
    engine = ExecutionPolicyEngine({"kind": "b5"})
    submitted = tick(engine, "2024-01-01", f=features(vol_percentile=0.96))
    assert submitted["desired_weight"] == pytest.approx(1 / 3)
    filled = tick(engine, "2024-01-02", f=features(vol_percentile=0.5))
    assert filled["weights_at_close"] == pytest.approx(1 / 3)
    assert filled["vol_defensive"]
    assert filled["desired_weight"] == pytest.approx(1 / 3)


def test_anchor_switch_changes_recovery_rule_without_resetting_accepted_trade():
    engine = ExecutionPolicyEngine({"kind": "b5"}, bars_per_day=1)
    tick(engine, "2024-01-01")
    tick(engine, "2024-01-02")
    tick(engine, "2024-01-03", f=features(vol_percentile=0.96))
    tick(engine, "2024-01-04", f=features(vol_percentile=0.1))
    assert engine.cooldown_remaining == 4
    engine.switch_candidate({"kind": "anchor"}, pd.Timestamp("2024-01-05"))
    row = tick(engine, "2024-01-05", f=features(vol_percentile=0.3))
    assert row["entry_price"] == 100.0
    assert row["desired_weight"] == 1.0
    assert not row["vol_defensive"]
