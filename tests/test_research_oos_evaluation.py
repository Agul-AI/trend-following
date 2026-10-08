"""Deterministic artificial fixtures: no market downloads or performance claims."""

import copy

import numpy as np
import pandas as pd
import pytest

from trend_following.research_ledger import LedgerConfig
from trend_following.research_oos_accounting import (
    causal_split_adjusted_prices,
    simulate_oos_ledger,
    terminal_snapshot,
)
from trend_following.research_oos_evaluation import (
    IncompleteHoldoutError,
    _run_track,
    bootstrap_boundary_returns,
    checkpoint_end_for_months,
    completed_bar_schedule,
    evaluate_oos_checkpoint,
    fit_pretrain_multiplier,
    paired_compound_bootstrap,
    required_warmup_observations,
    synthetic_financing_companion,
)

# Short windows make unit tests fast; production normalizes the fully frozen B5
# settings, whose independent minimum warmup assertion is tested below.
SMALL_SPEC = {
    "kind": "b5",
    "long_ma_days": 8,
    "short_ma_days": 2,
    "medium_ma_days": 4,
    "macd_fast_days": 1,
    "macd_slow_days": 2,
    "macd_signal_days": 1,
    "bear_filter_slope_days": 2,
    "vol_window_days": 3,
    "vol_lookback_days": 4,
    "use_bear_filter": False,
    "use_profit_lock": False,
    "use_qtrim": False,
    "use_peak_stop": False,
}


def synthetic_calendar(start="2024-01-02", end="2027-01-04"):
    """Fictional weekday sessions, NOT a claim about actual NYSE holidays."""
    days = pd.bdate_range(start, end)
    opens = (days + pd.Timedelta(hours=9, minutes=30)).tz_localize("America/New_York")
    closes = (days + pd.Timedelta(hours=16)).tz_localize("America/New_York")
    return pd.DataFrame(
        {"market_open": opens.tz_convert("UTC"), "market_close": closes.tz_convert("UTC")}
    )


@pytest.fixture(scope="module")
def artificial_history():
    calendar = synthetic_calendar()
    schedule = completed_bar_schedule(calendar)
    t = np.arange(len(schedule), dtype=float)
    # Mixed trends and varying volatility create nonzero, imperfectly correlated
    # strategy volatility without requiring any real ticker observations.
    log_returns = 0.00015 + 0.001 * np.sin(t / 31) + 0.0004 * np.cos(t / 7)
    prices = pd.DataFrame(
        {
            "QQQ": 100 * np.exp(np.cumsum(log_returns)),
            "TQQQ": 40 * np.exp(np.cumsum(3 * log_returns)),
        },
        index=schedule.index,
    )
    return calendar, prices


def score(calendar, prices, **kwargs):
    defaults = dict(
        policy_spec=SMALL_SPEC,
        frozen_multiplier=0.5,
        holdout_start="2025-01-02T14:30Z",
        checkpoint_months=12,
        as_of="2027-01-04T22:00Z",
        slippage_bps=(5.0,),
        include_tax_scenarios=False,
        include_gross_companions=False,
        include_attribution=False,
        n_bootstrap=30,
    )
    defaults.update(kwargs)
    return evaluate_oos_checkpoint(prices, calendar, **defaults)


def test_scheduled_regular_and_half_day_bars_have_observed_end_semantics():
    calendar = synthetic_calendar("2026-11-25", "2026-11-27")
    calendar.loc[2, "market_close"] = pd.Timestamp("2026-11-27T18:00Z")
    schedule = completed_bar_schedule(calendar)
    assert schedule.groupby("session").size().tolist() == [7, 7, 4]
    last = schedule.iloc[-1]
    assert last.bar_start == pd.Timestamp("2026-11-27T17:30Z")
    assert schedule.index[-1] == pd.Timestamp("2026-11-27T18:00Z")
    assert schedule.index[0] == pd.Timestamp("2026-11-25T15:30Z")


def test_calendar_not_inferred_from_observations_and_invalid_sessions_fail():
    calendar = synthetic_calendar("2026-01-05", "2026-01-06")
    with pytest.raises(ValueError, match="chronological"):
        completed_bar_schedule(calendar.iloc[::-1])
    calendar.loc[0, "market_open"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="Unsupported"):
        completed_bar_schedule(calendar)


def test_default_warmup_includes_slope_not_only_ma_or_volatility_windows():
    assert required_warmup_observations() == 1610
    assert required_warmup_observations(SMALL_SPEC) == 70


def test_checkpoint_uses_last_session_strictly_before_calendar_anniversary():
    calendar = synthetic_calendar("2025-01-04", "2027-01-08")
    start = calendar.market_open.iloc[0]  # Monday January 6, 2025.
    assert checkpoint_end_for_months(calendar, start, 12) == pd.Timestamp("2026-01-05T21:00Z")
    assert checkpoint_end_for_months(calendar, start, 24) == pd.Timestamp("2027-01-05T21:00Z")
    with pytest.raises(ValueError, match="12-month"):
        checkpoint_end_for_months(calendar, start, 6)
    with pytest.raises(ValueError, match="session open"):
        checkpoint_end_for_months(calendar, start + pd.Timedelta(minutes=1), 12)
    with pytest.raises(IncompleteHoldoutError, match="does not cover"):
        checkpoint_end_for_months(calendar.iloc[:10], start, 12)


def test_incomplete_checkpoint_refuses_before_even_reading_prices(artificial_history):
    calendar, _ = artificial_history
    with pytest.raises(IncompleteHoldoutError, match="no results emitted"):
        score(calendar, None, as_of="2026-01-01T20:59:59Z")
    with pytest.raises(ValueError, match="cutoff differs"):
        score(calendar, None, checkpoint_end="2026-01-02T20:00Z")


def test_missing_bar_is_never_interpolated(artificial_history):
    calendar, prices = artificial_history
    with pytest.raises(IncompleteHoldoutError, match="1 missing"):
        score(calendar, prices.drop(prices.index[3000]))


def test_calibration_requires_exactly_252_complete_sessions(artificial_history):
    calendar, prices = artificial_history
    with pytest.raises(ValueError, match="exactly 252"):
        fit_pretrain_multiplier(
            prices, calendar, training_end="2025-12-31T22:00Z", training_sessions=251
        )
    with pytest.raises(ValueError, match="Fewer than 252"):
        fit_pretrain_multiplier(prices, calendar, training_end="2024-02-01T22:00Z")


def test_train_only_multiplier_ignores_future_values_and_future_actions(artificial_history):
    calendar, prices = artificial_history
    cutoff = pd.Timestamp("2025-12-31T22:00Z")
    params = dict(policy_spec=SMALL_SPEC, training_end=cutoff)
    base = fit_pretrain_multiplier(prices, calendar, **params)
    poisoned = prices.copy()
    poisoned.loc[poisoned.index > cutoff] = -np.inf
    future_action = [{"effective_at": "2026-01-02T14:30Z", "kind": "invalid_unseen_future_action"}]
    changed = fit_pretrain_multiplier(poisoned, calendar, corporate_actions=future_action, **params)
    assert changed == base
    assert 0 < base["multiplier"] <= 1
    assert base["multiplier"] == min(
        1, base["qqq_daily_net_volatility"] / base["b5_daily_net_volatility"]
    )
    assert base["training_sessions"] == 252
    assert base["training_end"] == "2025-12-31T21:00:00+00:00"
    assert base["preceding_warmup_observations"] >= 69
    assert base["terminal_liquidation_in_calibration"] is False
    assert base["calibration_label"] == "retrospective_training_only_not_oos"


def test_calibration_rejects_zero_strategy_volatility_no_favorable_fallback(artificial_history):
    calendar, prices = artificial_history
    constant = prices.copy()
    constant.loc[:, :] = 100.0
    with pytest.raises(ValueError, match="no fallback"):
        fit_pretrain_multiplier(
            constant, calendar, policy_spec=SMALL_SPEC, training_end="2025-12-31T22:00Z"
        )


def test_observed_boundary_factors_reconstruct_exact_clone_wealth():
    dates = pd.date_range("2027-01-04T15:30Z", periods=5, freq="D")
    close = pd.DataFrame({"TQQQ": [100, 102, 106, 103, 105]}, index=dates)
    weights = pd.DataFrame({"TQQQ": [0, 0.5, 0.5, 0, 0.5]}, index=dates)
    result = simulate_oos_ledger(
        close,
        weights,
        pd.Series(0.0, index=dates),
        LedgerConfig(fee_bps=1, slippage_bps=5),
        initial_cash=1000,
    )
    original = copy.deepcopy(vars(result.ledger))
    terminal = terminal_snapshot(result.ledger, dates[-1], settle_tax=False)
    daily, factor, info = bootstrap_boundary_returns(result, terminal)
    assert (1 + daily).prod() * factor == pytest.approx(terminal["liquidated_equity"] / 1000)
    first = result.events.query("event == 'buy'").iloc[0]
    assert info["entry_factor"] == pytest.approx(1 - (first.fee + first.slippage) / 1000)
    assert info["terminal_factor"] == pytest.approx(
        terminal["liquidated_equity"] / terminal["marked_equity"]
    )
    assert len(result.events.query("event == 'buy'")) == 2
    assert vars(result.ledger) == original
    assert result.ledger.quantity("TQQQ") > 0
    assert terminal["ledger"].quantity("TQQQ") == 0


def test_never_invested_boundary_charges_are_exactly_one():
    dates = pd.date_range("2027-01-04T15:30Z", periods=3, freq="D")
    close = pd.DataFrame({"TQQQ": [100, 200, 50]}, index=dates)
    result = simulate_oos_ledger(
        close,
        close * 0,
        pd.Series(0.0, index=dates),
        LedgerConfig(fee_bps=1, slippage_bps=5),
        initial_cash=1000,
    )
    terminal = terminal_snapshot(result.ledger, dates[-1], settle_tax=False)
    daily, factor, info = bootstrap_boundary_returns(result, terminal)
    assert daily.eq(0).all()
    assert factor == 1
    assert info == {"entry_factor": 1.0, "terminal_factor": 1.0}


def test_paired_bootstrap_is_deterministic_and_identical_paths_have_zero_delta():
    returns = np.sin(np.arange(100)) / 100
    daily = pd.DataFrame({"scaled": returns, "benchmark": returns})
    args = dict(boundary_factors=(0.999, 0.999), n_simulations=101, seed=42)
    a, sa = paired_compound_bootstrap(daily, **args)
    b, sb = paired_compound_bootstrap(daily, **args)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(sa, sb)
    assert sa.delta_pct_points.eq(0).all()
    reverse, sr = paired_compound_bootstrap(daily, block_lengths=(60, 20), **args)
    pd.testing.assert_frame_equal(
        sa.sort_values(["block_sessions", "simulation"]).reset_index(drop=True),
        sr.sort_values(["block_sessions", "simulation"]).reset_index(drop=True),
    )
    assert a.block_sessions.tolist() == [20, 60]
    assert reverse.block_sessions.tolist() == [60, 20]


def test_compound_bootstrap_endpoint_is_percentage_point_total_return_delta():
    daily = pd.DataFrame({"scaled": [0.01, -0.02, 0.03], "benchmark": [0.0, 0.01, -0.01]})
    summary, _ = paired_compound_bootstrap(daily, boundary_factors=(0.99, 0.98), n_simulations=20)
    expected = ((1 + daily.scaled).prod() * 0.99 - (1 + daily.benchmark).prod() * 0.98) * 100
    assert summary.observed_delta_pct_points.tolist() == pytest.approx([expected, expected])
    assert summary.interpretation.str.contains("not_future_risk").all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_simulations": 0},
        {"n_simulations": True},
        {"n_simulations": 1.5},
        {"seed": -1},
        {"block_lengths": (20, 20)},
        {"block_lengths": (0,)},
        {"boundary_factors": (1, 0)},
    ],
)
def test_bootstrap_invalid_inputs_fail(kwargs):
    with pytest.raises(ValueError):
        paired_compound_bootstrap(pd.DataFrame({"a": [0.01, 0.02], "b": [0.02, 0.01]}), **kwargs)


def test_cost_scenarios_primary_endpoint_and_independent_benchmark(artificial_history):
    calendar, prices = artificial_history
    result = score(
        calendar,
        prices,
        frozen_multiplier=0.0,
        slippage_bps=(5, 10, 20, 30),
        include_tax_scenarios=True,
        include_gross_companions=True,
        include_attribution=True,
    )
    assert len(result.metrics) == 36
    for track in ["b5_scaled", "qqq_buy_hold", "b5_unscaled", "cash", "tqqq_buy_hold", "qqq_200ma"]:
        subset = result.metrics.loc[
            result.metrics.track.eq(track)
            & ~result.metrics.hypothetical_taxes
            & ~result.metrics.before_investor_trading_costs
        ]
        assert subset.slippage_bps.tolist() == [5, 10, 20, 30]
        assert subset.fee_bps.eq(1).all()
    gross = result.metrics.loc[result.metrics.before_investor_trading_costs]
    assert len(gross) == 6
    assert gross.fee_bps.eq(0).all() and gross.slippage_bps.eq(0).all()
    assert gross.actual_fund_costs_embedded.all()
    assert gross.role.eq("secondary").all()
    assert result.paths["cash__slippage_5"].equity.eq(1000).all()
    tqqq_buys = result.paths["tqqq_buy_hold__slippage_5"].events.query("event == 'buy'")
    assert tqqq_buys.timestamp.iloc[0] == result.paths["tqqq_buy_hold__slippage_5"].returns.index[1]
    ma_orders = result.paths["qqq_200ma__slippage_5"].events.query(
        "event == 'buy' or event == 'sell'"
    )
    assert not ma_orders.empty
    assert ma_orders.asset.eq("QQQ").all()
    assert result.paths["qqq_200ma__slippage_5"].ledger.config.exposure_multipliers == {"QQQ": 1.0}
    scaled, qqq = "b5_scaled__slippage_5", "qqq_buy_hold__slippage_5"
    assert result.paths[scaled].returns.eq(0).all()
    assert result.paths[scaled].events.query("event == 'buy'").empty
    first_benchmark_fill = result.paths[qqq].events.query("event == 'buy'").timestamp.iloc[0]
    assert first_benchmark_fill == result.paths[qqq].returns.index[1]
    assert result.paths[qqq].ledger.quantity("QQQ") > 0
    assert result.terminal_snapshots[qqq]["ledger"].quantity("QQQ") == 0
    expected = (
        (
            result.terminal_snapshots[scaled]["liquidated_equity"]
            - result.terminal_snapshots[qqq]["liquidated_equity"]
        )
        / 1000
        * 100
    )
    assert result.metadata["primary_delta_pct_points"] == pytest.approx(expected)
    assert result.bootstrap_summary.observed_delta_pct_points.tolist() == pytest.approx(
        [expected] * 2
    )
    assert result.metadata["checkpoint_role"] == "preliminary_no_success_decision"
    assert result.metadata["cash_yield"] == 0
    assert "unavailable" in result.metadata["financing_companion"]
    assert result.metrics.query("role == 'primary_comparison'").track.tolist() == [
        "b5_scaled",
        "qqq_buy_hold",
    ]
    assert len(result.attribution_metrics) == 18
    assert result.attribution_metrics.path_mode.eq(
        "baseline_fixed_target_allocations_not_fixed_share_orders"
    ).all()
    net = result.attribution_metrics.query("attribution_layer == 'net_trading'")
    for row in net.itertuples():
        expected = result.metrics.loc[result.metrics.scenario.eq(row.baseline_target_source)].iloc[
            0
        ]
        assert row.total_return_pct == pytest.approx(expected.total_return_pct)
        assert row.target_position_changes == expected.target_position_changes
        assert row.end_to_end_minus_matched_target_pct_points == pytest.approx(0)
    for track, rows in result.attribution_metrics.groupby("track"):
        assert rows.target_position_changes.nunique() == 1, track
    assert result.metrics.annualized_volatility.notna().all()
    assert result.metrics.worst_calendar_month_return.notna().all()
    assert result.metrics.query("track == 'cash'").cash_weight_bar_mean.eq(1).all()
    for key, daily in result.daily_returns.items():
        assert (1 + daily).prod() == pytest.approx(
            result.terminal_snapshots[key]["liquidated_equity"] / 1000
        )


def test_later_prices_cannot_change_checkpoint_and_terminal_clone_does_not_reset(
    artificial_history,
):
    calendar, prices = artificial_history
    first = score(calendar, prices)
    cutoff = pd.Timestamp(first.metadata["checkpoint_end"])
    poisoned = prices.copy()
    poisoned.loc[poisoned.index > cutoff] = np.nan
    second = score(calendar, poisoned)
    pd.testing.assert_frame_equal(first.metrics, second.metrics)
    pd.testing.assert_frame_equal(first.bootstrap_samples, second.bootstrap_samples)
    final = score(calendar, prices, checkpoint_months=24)
    key = "qqq_buy_hold__slippage_5"
    pd.testing.assert_series_equal(first.paths[key].equity, final.paths[key].equity.loc[:cutoff])
    assert first.paths[key].ledger.quantity("QQQ") > 0
    assert final.metadata["checkpoint_role"] == "primary"
    # Final event history has one initial QQQ purchase, never a checkpoint sale.
    assert final.paths[key].events.event.eq("buy").sum() == 1
    assert final.paths[key].events.event.eq("sell").sum() == 0


def test_actual_qqq_200ma_receives_qqq_splits_and_distributions_not_tqqq(artificial_history):
    calendar, prices = artificial_history
    panel = prices.loc[:"2025-01-06"].copy()
    panel.QQQ = 100 * np.exp(np.arange(len(panel)) * 0.0002)
    ex = pd.Timestamp("2025-01-03T14:30Z")
    panel.loc[panel.index >= ex, "QQQ"] = panel.loc[panel.index >= ex, "QQQ"] / 2 - 0.5
    common = dict(
        asset="QQQ",
        effective_at=ex,
        available_at=ex,
        availability_basis="assumed_public_by_ex_date",
        source_observed_at="2025-01-07T21:00Z",
    )
    actions = pd.DataFrame(
        [
            dict(common, action_id="qqq-split", kind="split", split_ratio=2),
            dict(
                common,
                action_id="qqq-dividend",
                kind="distribution",
                amount_per_share=0.5,
                pay_at="2025-01-06T14:30Z",
                character="qualified",
            ),
        ]
    )
    normalized = causal_split_adjusted_prices(panel, actions)
    result = _run_track(
        panel,
        normalized,
        actions,
        completed_bar_schedule(calendar),
        dict(SMALL_SPEC, allocation_multiplier=0.0),
        pd.Timestamp("2025-01-02T14:30Z"),
        track="qqq_200ma",
        fee_bps=1,
        slippage_bps=5,
        initial_cash=1000,
    )
    buy = result.events.query("event == 'buy'").iloc[0]
    assert buy.asset == "QQQ"
    assert buy.timestamp == result.returns.index[1]
    assert result.ledger.quantity("QQQ") == pytest.approx(2 * buy.quantity)
    paid = result.events.query("event == 'distribution_payment'").cash_flow.sum()
    assert paid == pytest.approx(buy.quantity)
    assert result.cash.iloc[-1] == pytest.approx(paid)
    assert not result.events.asset.eq("TQQQ").any()


def financing_fixture():
    calendar = synthetic_calendar("2027-01-08", "2027-01-11")  # Friday and Monday.
    schedule = completed_bar_schedule(calendar)
    underlying = pd.Series(100.0, index=schedule.index, name="causal_QQQ_total_return_index")
    rates = pd.DataFrame(
        {
            "series_id": ["DFF"],
            "available_at": [pd.Timestamp("2027-01-07T21:00Z")],
            "annual_rate": [0.04],
        }
    )
    return calendar, underlying, rates


def test_synthetic_companion_calendar_financing_and_daily_reset_hand_calculation():
    calendar, underlying, rates = financing_fixture()
    result = synthetic_financing_companion(
        underlying, calendar, rates, spread_bps=50, expense_ratio=0
    )
    friday, monday = result.iloc[6], result.iloc[7]
    elapsed_days = (result.index[7] - result.index[6]).total_seconds() / 86400
    assert elapsed_days > 2
    assert monday.financing_cost == pytest.approx(friday.debt * 0.045 * elapsed_days / 360)
    assert friday.debt == pytest.approx(2 * friday.synthetic_nav)
    assert result.reset_at_close.sum() == 2
    assert result.synthetic_nav.iloc[-1] < 1
    assert result.attrs["model"].endswith("not_actual_TQQQ")
    zero_rates = rates.assign(annual_rate=0)
    rising = underlying.copy()
    rising.iloc[:7] = np.linspace(100, 110, 7)
    rising.iloc[7:] = np.linspace(111, 121, 7)
    gross = synthetic_financing_companion(
        rising, calendar, zero_rates, spread_bps=0, expense_ratio=0
    )
    assert gross.synthetic_nav.iloc[6] == pytest.approx(1 + 3 * (110 / 100 - 1))
    assert gross.synthetic_nav.iloc[-1] == pytest.approx(1.3 * (1 + 3 * (121 / 110 - 1)))


def test_synthetic_companion_never_zero_fills_or_double_counts_actual_fund_financing():
    calendar, underlying, rates = financing_fixture()
    with pytest.raises(ValueError, match="double count"):
        synthetic_financing_companion(underlying.rename("TQQQ"), calendar, rates)
    with pytest.raises(ValueError, match="DFF"):
        synthetic_financing_companion(underlying, calendar, rates.assign(series_id="DTB3"))
    with pytest.raises(ValueError, match="Stale"):
        synthetic_financing_companion(
            underlying, calendar, rates.assign(available_at=pd.Timestamp("2026-01-01T21:00Z"))
        )
    with pytest.raises(ValueError, match="No published"):
        synthetic_financing_companion(
            underlying, calendar, rates.assign(available_at=pd.Timestamp("2027-01-09T21:00Z"))
        )
    with pytest.raises(IncompleteHoldoutError, match="complete scheduled"):
        synthetic_financing_companion(underlying.drop(underlying.index[3]), calendar, rates)


def test_synthetic_financing_future_rate_perturbation_cannot_change_history():
    calendar, underlying, rates = financing_fixture()
    base = synthetic_financing_companion(underlying, calendar, rates)
    future = pd.DataFrame(
        {
            "series_id": ["NOT_DFF_FUTURE"],
            "available_at": [pd.Timestamp("2027-01-12T21:00Z")],
            "annual_rate": [np.inf],
        }
    )
    changed = synthetic_financing_companion(underlying, calendar, pd.concat([rates, future]))
    pd.testing.assert_frame_equal(base, changed)
