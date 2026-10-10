"""Fabricated hybrid shadows/ledgers; no market files or prior outcome access."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trend_following import research_daily_hourly_v5 as core
from trend_following import research_hybrid_allocation_engine as engine
from trend_following import research_mean_reversion as mr
from trend_following import research_mean_reversion_engine as mr_engine
from trend_following import research_quality_engine_v5 as quality
from trend_following.research_constant_cash import constant_cash_observations


def tf_candidate():
    return core.Candidate(core.IndicatorRule("SMA", 10), core.IndicatorRule("SMA", 20), 3, 3)


def mr_candidate():
    return mr.Candidate(mr.Rule("SMA_DISTANCE", 10), -0.02, 0)


def fixture(specs):
    history = pd.bdate_range("1999-11-01", periods=210)
    daily = [
        dict(
            session=d.strftime("%Y-%m-%d"), close=100.0, dividend_amount=0.0, split_coefficient=1.0
        )
        for d in history
    ]
    bars, calendar, coverage = [], [], []
    for i, spec in enumerate(specs):
        session = spec.get(
            "session", (pd.Timestamp("2001-01-02") + pd.offsets.BDay(i)).strftime("%Y-%m-%d")
        )
        opening = pd.Timestamp(f"{session} 09:30", tz="America/New_York").tz_convert("UTC")
        closes = spec.get("closes", [100.0] * 13)
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


def make_cache(specs, *, tf_values=None, mr_values=None, er_values=None, post_gap=False):
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
    tf, reversal = cache.tf, cache.mr
    if tf_values is not None:
        margins = tf.margins.copy()
        values = np.asarray(tf_values)
        for rule in (tf_candidate().entry_rule, tf_candidate().exit_rule):
            margins[:, tf.rules.index(rule)] = values
        tf = replace(tf, margins=margins, feature_fingerprint="")
        tf = replace(tf, feature_fingerprint=quality._feature_fingerprint(tf))
    if mr_values is not None:
        margins = reversal.margins.copy()
        margins[:, reversal.rules.index(mr_candidate().rule)] = np.asarray(mr_values)
        reversal = replace(reversal, margins=margins, feature_fingerprint="")
        reversal = replace(reversal, feature_fingerprint=quality._feature_fingerprint(reversal))
    er = cache.finalized_er.copy()
    if er_values is not None:
        er.iloc[209:] = np.asarray(er_values)
    cache = replace(cache, tf=tf, mr=reversal, finalized_er=er, er_fingerprint="")
    return replace(cache, er_fingerprint=engine._er_fingerprint(cache))


def shadows(cache, trends=None, reversions=None):
    return engine.build_shadow_paths(
        cache,
        [tf_candidate()] if trends is None else trends,
        [mr_candidate()] if reversions is None else reversions,
        start=cache.calendar.session.iloc[0],
        end=cache.calendar.session.iloc[-1],
        require_full=False,
    )


def execute(cache, states=None, *, fee=1, slip=3):
    states = shadows(cache) if states is None else states
    cash = constant_cash_observations(states.clock[0], states.clock[-1])
    return engine.run(cache, states, cash_observations=cash, commission_bps=fee, slippage_bps=slip)


def ny(value):
    return pd.Timestamp(value, tz="America/New_York").tz_convert("UTC")


def test_exact_grid_membership_and_stable_roundtrip_metadata():
    trends = [
        core.Candidate(rule, core.IndicatorRule("SMA", 200), 3, 3)
        for rule in core.indicator_rules()[:12]
    ]
    reversions = mr.candidate_grid()[:32]
    grid = engine.pair_grid(reversed(trends), reversed(reversions))
    assert len(grid) == len({p.pair_id for p in grid}) == 384
    assert all(engine.Pair.from_dict(p.to_dict()) == p for p in grid)
    assert {p.tf for p in grid} == set(trends) and {p.mr for p in grid} == set(reversions)
    assert len(engine.PRINCIPAL_IDS) == 8 and len(engine.PORTFOLIO_IDS) == 9
    with pytest.raises(ValueError, match="12 TF"):
        engine.pair_grid([tf_candidate()], [mr_candidate()])


@pytest.mark.parametrize("kind", ["flat", "linear", "zigzag"])
def test_er20_independent_reference_and_twenty_changes_require_twenty_one_values(kind):
    values = (
        np.full(45, 100.2)
        if kind == "flat"
        else np.arange(100, 145)
        if kind == "linear"
        else 100 + np.tile([0, 1], 23)[:45]
    )
    result = engine.efficiency_ratio(values)
    assert result.iloc[:20].isna().all()
    for i in range(20, 45):
        changes = [abs(float(values[j] - values[j - 1])) for j in range(i - 19, i + 1)]
        reference = (
            0 if sum(changes) == 0 else abs(float(values[i] - values[i - 20])) / sum(changes)
        )
        assert result.iloc[i] == reference


def test_er_missing_is_unknown_and_future_suffix_does_not_change_prefix():
    values = np.arange(100, 145, dtype=float)
    values[22] = np.nan
    result = engine.efficiency_ratio(values)
    assert result.iloc[22:43].isna().all()
    altered = values.copy()
    altered[35:] = 2000
    np.testing.assert_array_equal(result.iloc[:35], engine.efficiency_ratio(altered).iloc[:35])


def test_fresh_tactical_entry_starts_after_tf_fill_with_no_extra_fill_delay():
    cache = make_cache(
        [{}, {}], tf_values=[1] * 26, mr_values=[-0.03] * 26, er_values=[0.5, 0.5, 0.5]
    )
    state = shadows(cache)
    tfbuy = state.shadow_trades.query("strategy == 'TF' and side == 'buy'").iloc[0]
    mrbuy = state.shadow_trades.query("strategy == 'GMR' and side == 'buy'").iloc[0]
    assert tfbuy.timestamp == tfbuy.decision_at == ny("2001-01-02 13:00")
    assert tfbuy.execution_bar == tfbuy.decision_bar + 1
    assert mrbuy.timestamp == mrbuy.decision_at == ny("2001-01-03 10:00")
    first_after_tf = np.where(state.trace["timestamp_ns"] == ny("2001-01-02 13:30").value)[0][0]
    assert state.trace["mr_entry_count"][first_after_tf, 0] == 1
    assert not state.trace["mr_long"][first_after_tf, 0]


def test_tf_core_records_and_wealth_reproduce_frozen_tf_engine():
    cache = make_cache(
        [{}, {}, {}],
        tf_values=[1] * 19 + [-1] * 7 + [1] * 13,
        mr_values=[-0.03] * 39,
        er_values=[0.5] * 4,
    )
    state = shadows(cache)
    result = execute(cache, state, slip=25)
    cash = constant_cash_observations(state.clock[0], state.clock[-1])
    original = quality._run(
        cache.tf,
        [tf_candidate()],
        start="2001-01-02",
        end="2001-01-04",
        cash_observations=cash,
        commission_bps=1,
        slippage_bps=25,
        initial_cash=1000,
        record=True,
    )
    np.testing.assert_allclose(result.account_equity[:, 0], original[2][:, 0], atol=2e-10, rtol=0)
    records = state.shadow_trades.query("strategy == 'TF'")
    for row, reference in zip(records.itertuples(), original[6], strict=True):
        assert row.timestamp == reference["timestamp"] and row.side == reference["side"]
        assert (
            row.decision_at == reference["decision_at"]
            or pd.isna(row.decision_at)
            and reference["decision_at"] is None
        )
    assert result.metrics[7]["gross_sleeve_order_count"] == original[0][0]["order_count"]
    assert result.metrics[7]["total_commission"] == pytest.approx(
        original[0][0]["total_commission"]
    )


def test_simultaneous_tf_exit_and_recovery_force_same_next_open_and_one_blend_order():
    cache = make_cache(
        [{}, {}, {}],
        tf_values=[1] * 19 + [-1] * 7 + [-1] * 13,
        mr_values=[-0.03] * 19 + [0] * 7 + [-0.03] * 13,
        er_values=[0.5] * 4,
    )
    state = shadows(cache)
    actual = execute(cache, state)
    tfexit = state.shadow_trades.query(
        "strategy == 'TF' and phase == 'open' and side == 'sell'"
    ).iloc[0]
    mrex = state.shadow_trades.query(
        "strategy == 'GMR' and phase == 'open' and side == 'sell'"
    ).iloc[0]
    assert tfexit.timestamp == mrex.timestamp == ny("2001-01-04 09:30")
    assert mrex.cause == "recovery+tf_exit"
    simultaneous = actual.trade_rows.query("account_group == 'B050'")
    at_exit = simultaneous.loc[simultaneous.timestamp.eq(tfexit.timestamp)]
    assert len(at_exit) == 1 and at_exit.iloc[0].target_fraction == 0
    assert at_exit.iloc[0].reason == "tf_exit_priority"


def test_regime_mode_has_hysteresis_seven_checks_and_daily_pending_invalidation():
    cache = make_cache(
        [{"closes": [100] * 7}, {}],
        tf_values=[1] * 20,
        mr_values=[-0.03] * 20,
        er_values=[0.25, 0.3, 0.3],
    )
    state = shadows(cache)
    assert np.all(state.mode_open_state == engine.TF_MODE)
    assert state.trace["mode_count"][6] == 7
    assert state.trace["pending_mode"][6] == engine.MR_MODE
    assert (state.diagnostics.event == "regime_daily_cancelled").sum() == 1
    normal = shadows(
        make_cache(
            [{}, {}], tf_values=[1] * 26, mr_values=[-0.03] * 26, er_values=[0.25, 0.25, 0.25]
        )
    )
    switch = normal.diagnostics.query("event == 'regime_switch_filled'").iloc[0]
    assert switch.timestamp == ny("2001-01-02 13:00")
    assert switch.execution_bar == switch.decision_bar + 1


def test_warm_handoff_does_not_sell_rebuy_identical_incoming_long_state():
    cache = make_cache(
        [{}, {}, {}], tf_values=[1] * 39, mr_values=[-0.03] * 39, er_values=[0.5, 0.5, 0.25, 0.25]
    )
    state = shadows(cache)
    actual = execute(cache, state)
    switches = state.diagnostics.query("event == 'regime_switch_filled'")
    assert switches.iloc[0].timestamp == ny("2001-01-04 13:00")
    c = actual.trade_rows.query("account_group == 'C'")
    assert c.side.tolist() == ["buy", "sell"]
    assert c.iloc[0].timestamp == ny("2001-01-02 13:00")
    assert c.iloc[1].phase == "terminal"
    assert not c.timestamp.isin(switches.timestamp).any()


def test_unknown_er_holds_mode_blocks_adds_then_reconciles_confirmed_target():
    cache = make_cache(
        [{}, {}, {}],
        tf_values=[1] * 39,
        mr_values=[-0.01] * 39,
        er_values=[np.nan, np.nan, 0.5, 0.5],
    )
    state = shadows(cache)
    actual = execute(cache, state)
    c = actual.trade_rows.query("account_group == 'C'")
    assert c.iloc[0].timestamp == ny("2001-01-04 09:30")
    assert c.iloc[0].reason == "known_er_reconciliation"
    assert c.iloc[0].decision_at == ny("2001-01-03 17:00")
    assert np.all(state.mode_open_state == engine.TF_MODE)
    assert actual.metrics[6]["er_unknown_blocked_risk_add_count"] > 0


def test_unknown_er_does_not_block_otherwise_valid_tf_risk_reduction():
    cache = make_cache(
        [{}, {}],
        tf_values=[1] * 13 + [-1] * 13,
        mr_values=[-0.01] * 26,
        er_values=[0.5, np.nan, np.nan],
    )
    state = shadows(cache)
    actual = execute(cache, state)
    c = actual.trade_rows.query("account_group == 'C'")
    assert c.side.tolist() == ["buy", "sell"]
    assert c.iloc[1].timestamp == ny("2001-01-03 13:00")
    assert c.iloc[1].reason == "tf_exit_priority"


def test_partial_target_drifts_without_maintenance_trades_and_core_weights_drift():
    cache = make_cache(
        [
            {},
            {"opens": [200] * 13, "closes": [200] * 13},
            {"opens": [150] * 13, "closes": [150] * 13},
        ],
        tf_values=[1] * 39,
        mr_values=[-0.01] * 39,
        er_values=[0.5] * 4,
    )
    result = execute(cache, fee=0, slip=0)
    b = result.trade_rows.query("account_group == 'B050'")
    assert b.side.tolist() == ["buy", "sell"]
    assert b.iloc[-1].phase == "terminal"
    assert result.account_exposure[1, 2] > 0.6
    assert result.metrics[0]["ending_trend_weight"] > 0.5
    assert np.isnan(result.metrics[3]["ending_trend_weight"])
    assert (
        result.metrics[3]["style_weight_definition"]
        == "signal_target_attribution_not_funded_sleeve_capital"
    )


def test_always_on_gate_reproduces_original_mr_and_off_gate_cash():
    cache = make_cache(
        [{}, {}, {}], mr_values=[-0.03] * 13 + [0] * 13 + [-0.03] * 13, er_values=[0.5] * 4
    )
    on = engine.with_test_tf_gate(cache, True)
    state = shadows(on)
    result = execute(on, state, slip=10)
    cash = constant_cash_observations(state.clock[0], state.clock[-1])
    original = mr_engine.run(
        cache.mr,
        [mr_candidate()],
        start="2001-01-02",
        end="2001-01-04",
        cash_observations=cash,
        commission_bps=1,
        slippage_bps=10,
        record=True,
    )
    np.testing.assert_allclose(result.account_equity[:, 1], original[2][:, 0], atol=2e-10, rtol=0)
    mr_records = state.shadow_trades.query("strategy == 'GMR'")
    assert mr_records.timestamp.tolist() == [r["timestamp"] for r in original[6]]
    off = engine.with_test_tf_gate(cache, False)
    no_risk = execute(off, shadows(off))
    assert no_risk.trade_rows.empty
    assert all(np.isnan(row["cash_excess_sharpe"]) for row in no_risk.metrics)


def test_gap_cancels_pending_resets_streaks_and_never_invents_confirmations():
    cache = make_cache(
        [{}, {"missing": [0]}, {}],
        tf_values=[-1] * 6 + [1] * 7 + [1] * 25,
        mr_values=[-0.03] * 38,
        er_values=[0.5] * 4,
        post_gap=True,
    )
    state = shadows(cache)
    first = state.shadow_trades.query("strategy == 'TF' and side == 'buy'").iloc[0]
    assert first.timestamp == ny("2001-01-03 13:30")
    assert state.diagnostics.query("event == 'blackout_cancel_reset'").iloc[0].pending_tf == 1
    result = execute(cache, state)
    assert result.metrics[7]["actual_post_gap_order_count"] == 1


def test_split_dividend_cash_terminal_and_zero_cost_replay_reconcile():
    cache = make_cache(
        [
            {"daily_close": 100},
            {
                "closes": [49.5] * 13,
                "opens": [49.5] * 13,
                "daily_close": 49.5,
                "split": 2,
                "dividend": 0.5,
            },
            {"closes": [55] * 13, "opens": [50] * 13},
        ],
        tf_values=[1] * 39,
        mr_values=[-0.03] * 39,
        er_values=[0.5] * 4,
    )
    state = shadows(cache)
    result = execute(cache, state, fee=0, slip=0)
    # The core buys10shares; ex-open split doubles them, then$10DRIP buys10/49.5.
    initial_q = 1000 * (1 + 0.03 * 1800 / (365 * 86400)) ** 7 / 100
    expected_core = (2 * initial_q + initial_q / 49.5) * 55
    final_tf = result.trade_rows.query("account_group == 'TF' and phase == 'terminal'").iloc[0]
    assert final_tf.equity_before == pytest.approx(expected_core)
    np.testing.assert_array_equal(result.account_gross_equity, result.account_equity)
    assert all(row["cost_drag_return"] == 0 for row in result.metrics)


def test_future_suffix_changes_neither_earlier_shadow_states_nor_account_paths():
    a = make_cache([{}, {}, {}], tf_values=[1] * 39, mr_values=[-0.03] * 39, er_values=[0.5] * 4)
    b = make_cache(
        [{}, {}, {"opens": [2500] * 13, "closes": [3000] * 13}],
        tf_values=[1] * 26 + [-1] * 13,
        mr_values=[-0.03] * 26 + [0.5] * 13,
        er_values=[0.5] * 3 + [0.01],
    )
    sa, sb = shadows(a), shadows(b)
    early = sa.open_times < ny("2001-01-04 09:30")
    for field in ("tf_open_state", "mr_open_state", "mode_open_state"):
        np.testing.assert_array_equal(getattr(sa, field)[early], getattr(sb, field)[early])
    ra, rb = execute(a, sa), execute(b, sb)
    np.testing.assert_array_equal(ra.account_equity[:2], rb.account_equity[:2])


def test_shadow_derived_mutation_is_detected_before_financial_execution():
    cache = make_cache([{}, {}], tf_values=[1] * 26, mr_values=[-0.03] * 26)
    state = shadows(cache)
    state.mr_open_state[-1, 0] = ~state.mr_open_state[-1, 0]
    with pytest.raises(ValueError, match="Intact matching"):
        execute(cache, state)


def test_pair_path_aggregation_matches_funded_core_with_no_duplicate_core_orders():
    cache = make_cache([{}, {}], tf_values=[1] * 26, mr_values=[-0.03] * 26, er_values=[0.5] * 3)
    state = shadows(cache)
    result = execute(cache, state)
    assert len(result.pair_metrics) == 7
    for i, pid in enumerate(engine.PRINCIPAL_IDS[:7]):
        paths = engine.pair_daily_equity(result, state, pid)
        np.testing.assert_allclose(
            paths.mean(axis=1), result.daily_equity[:, i], atol=2e-10, rtol=0
        )
    tf_orders = result.account_metrics[0]["order_count"]
    mr_orders = result.account_metrics[1]["order_count"]
    assert result.metrics[0]["gross_sleeve_order_count"] == tf_orders + mr_orders
    assert result.metrics[7]["gross_sleeve_order_count"] == tf_orders


def full_universe():
    trends = [
        core.Candidate(rule, core.IndicatorRule("SMA", 20), 3, 3)
        for rule in core.indicator_rules()[:12]
    ]
    all_mr = mr.candidate_grid()
    reversions = [
        c for family in mr.FAMILIES for c in [x for x in all_mr if x.family == family][:8]
    ]
    return trends, reversions


def test_full_384_pair_account_layout_and_chunk_invariance():
    trends, reversions = full_universe()
    cache = make_cache([{}, {}], tf_values=[1] * 26, mr_values=[-0.03] * 26, er_values=[0.5] * 3)
    full = engine.build_shadow_paths(
        cache, trends, reversions, start="2001-01-02", end="2001-01-03"
    )
    result = execute(cache, full)
    assert not full.test_only and len(full.pairs) == 384
    assert result.account_equity.shape == (2, 1932)
    assert result.daily_equity.shape == (2, 9)
    assert len(result.pair_metrics) == 2688
    assert result.account_groups.count("TF") == 12
    assert result.metrics[0]["positive_weight_sleeve_count"] == 396
    assert result.metrics[7]["positive_weight_sleeve_count"] == 12
    assert result.metrics[7]["gross_sleeve_order_count"] == sum(
        row["order_count"] for row in result.account_metrics[:12]
    )
    for pid in engine.PRINCIPAL_IDS[:7]:
        paired = engine.pair_daily_equity(result, full, pid)
        principal = result.daily_equity[:, result.portfolio_ids.index(pid)]
        np.testing.assert_allclose(paired.mean(axis=1), principal, atol=2e-9, rtol=0)
    # One TF / all32MR chunk must reproduce its rows from the full Cartesian run.
    chosen = full.pairs[0].tf
    chunk = shadows(cache, [chosen], reversions)
    part = execute(cache, chunk)
    column = {cid: j for j, cid in enumerate(result.account_ids)}
    for j, cid in enumerate(part.account_ids):
        np.testing.assert_array_equal(
            part.account_equity[:, j], result.account_equity[:, column[cid]]
        )
    np.testing.assert_array_equal(chunk.mr_open_state, full.mr_open_state[:, :32])
    assert (
        chunk.shadow_trades.query("strategy == 'TF'").timestamp.tolist()
        == full.shadow_trades.query(
            "strategy == 'TF' and candidate_id == @chosen.candidate_id"
        ).timestamp.tolist()
    )


@pytest.mark.parametrize("slip", [1, 3, 10, 25])
def test_all_cost_scenarios_and_all32_mr_members_pass_always_on_parity(slip):
    _, reversions = full_universe()
    cache = make_cache(
        [{}, {"opens": [130] * 13, "closes": [130] * 13}, {}],
        mr_values=[-0.03] * 13 + [0] * 13 + [-0.03] * 13,
        er_values=[0.5] * 4,
    )
    on = engine.with_test_tf_gate(cache, True)
    state = shadows(on, [tf_candidate()], reversions)
    result = execute(on, state, slip=slip)
    cash = constant_cash_observations(state.clock[0], state.clock[-1])
    canonical = tuple(sorted(reversions, key=lambda c: c.candidate_id))
    original = mr_engine.run(
        cache.mr,
        canonical,
        start="2001-01-02",
        end="2001-01-04",
        cash_observations=cash,
        commission_bps=1,
        slippage_bps=slip,
        record=True,
    )
    np.testing.assert_allclose(result.account_equity[:, 1:33], original[2], atol=2e-10, rtol=0)
    for i in range(32):
        assert result.account_metrics[1 + i]["order_count"] == original[0][i]["order_count"]
        assert result.account_metrics[1 + i]["total_commission"] == pytest.approx(
            original[0][i]["total_commission"]
        )


def test_same_open_regime_handoff_preserves_tf_exit_priority_cause():
    cache = make_cache(
        [{}, {}, {}],
        tf_values=[1] * 26 + [-1] * 13,
        mr_values=[-0.03] * 39,
        er_values=[0.5, 0.5, 0.25, 0.25],
    )
    state = shadows(cache)
    result = execute(cache, state)
    c = result.trade_rows.query("account_group == 'C' and side == 'sell'").iloc[0]
    assert c.timestamp == ny("2001-01-04 13:00")
    assert c.target_fraction == 0
    assert c.reason == "regime_handoff+tf_exit_priority"


def test_b_c_funded_weight_fields_are_missing_and_signal_components_are_explicit():
    cache = make_cache([{}, {}], tf_values=[1] * 26, mr_values=[-0.03] * 26, er_values=[0.5] * 3)
    result = execute(cache)
    for row in result.metrics[3:7]:
        for field in (
            "initial_trend_weight",
            "ending_trend_weight",
            "initial_mean_reversion_weight",
            "ending_mean_reversion_weight",
        ):
            assert np.isnan(row[field])
    b = result.metrics[3]
    assert b["blend_coefficient"] == 0.5
    assert b["ending_tf_signal_contribution"] == b["ending_mr_signal_contribution"] == 0.5
    assert "not_capital" in b["signal_contribution_definition"]
    assert result.metrics[6]["ending_regime_mode"] == "TF"
    assert result.metrics[0]["tactical_average_holding_hours"] > 0


def test_actual_and_shadow_post_gap_flags_retain_window_without_selection_filter():
    cache = make_cache(
        [{}, {"missing": [0]}, {}],
        tf_values=[-1] * 6 + [1] * 7 + [1] * 25,
        mr_values=[-0.03] * 38,
        er_values=[0.5] * 4,
        post_gap=True,
    )
    state = shadows(cache)
    result = execute(cache, state)
    tf_buy = result.trade_rows.query("account_group == 'TF' and side == 'buy'").iloc[0]
    assert tf_buy.post_gap_flag
    assert "2001-01-04T21:00:00+00:00" in tf_buy.post_gap_windows
    shadow_buy = state.shadow_trades.query("strategy == 'TF' and side == 'buy'").iloc[0]
    assert shadow_buy.post_gap_flag and shadow_buy.post_gap_windows == tf_buy.post_gap_windows
    assert result.metrics[7]["actual_post_gap_order_count"] == 1


def test_mr_undefined_cannot_block_valid_tf_forced_exit():
    cache = make_cache(
        [{}, {}, {}],
        tf_values=[1] * 26 + [-1] * 13,
        mr_values=[-0.03] * 26 + [np.nan] * 13,
        er_values=[0.5] * 4,
    )
    state = shadows(cache)
    exits = state.shadow_trades.query("strategy == 'GMR' and side == 'sell' and phase == 'open'")
    assert exits.iloc[0].timestamp == ny("2001-01-04 13:00")
    assert exits.iloc[0].cause == "tf_forced_exit"


def test_tf_core_pending_cancellations_count_once_not_once_per_pair():
    cache = make_cache(
        [{}, {"missing": [0]}, {}],
        tf_values=[-1] * 6 + [1] * 7 + [1] * 25,
        mr_values=[-0.03] * 38,
        er_values=[0.5] * 4,
    )
    state = shadows(cache)
    result = execute(cache, state)
    assert state.tf_diagnostics[0]["cancelled_blackout_order_count"] == 1
    assert result.metrics[7]["cancelled_blackout_order_count"] == 1
    assert result.metrics[0]["cancelled_blackout_order_count"] == 1
    assert result.metrics[0]["cancellation_count_definition"] == "funded_sleeve_pending_intents"
    assert "not_cancelled_broker_orders" in result.metrics[3]["cancellation_count_definition"]
