"""Synthetic portfolio/replay tests; no market files or prior outcome reads."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trend_following import research_mean_reversion as mr
from trend_following import research_mean_reversion_engine as mr_engine
from trend_following import research_portfolio_engine as engine
from trend_following import research_quality_engine_v5 as quality


def synthetic_paths():
    ids = tuple([f"tf_{i}" for i in range(12)] + [f"mr_{i}" for i in range(32)])
    times = pd.date_range("2001-01-02 22:00", periods=3, freq="D", tz="UTC")
    start = pd.Timestamp("2001-01-02 14:30", tz="UTC")
    events = pd.DatetimeIndex([start, *times])
    equity = np.column_stack(
        [np.tile([1100.0, 990.0, 1188.0], (12, 1)).T, np.tile([1000.0, 1500.0, 1200.0], (32, 1)).T]
    )
    event_equity = np.vstack([np.full(44, 1000), equity])
    invested = event_equity * np.r_[np.full(12, 0.8), np.full(32, 0.2)]
    metrics = tuple(
        dict(
            candidate_id=candidate_id,
            exposure_fraction=1.0,
            commission_bps=1.0,
            slippage_bps=3.0,
        )
        for candidate_id in ids
    )
    return engine.SleevePaths(
        ids,
        times,
        equity,
        equity / np.vstack([np.full(44, 1000), equity[:-1]]) - 1,
        np.zeros(3),
        equity[-1] + 10,
        events,
        event_equity,
        invested,
        np.ones_like(event_equity, dtype=bool),
        (),
        start,
        times[-1],
        1000.0,
        metrics,
    )


def fixture(*, split=False, missing=False):
    history = pd.bdate_range("1999-11-01", periods=60)
    daily = [
        dict(
            session=x.strftime("%Y-%m-%d"), close=100.0, dividend_amount=0.0, split_coefficient=1.0
        )
        for x in history
    ]
    bars, calendar, coverage = [], [], []
    for day, session in enumerate(("2001-01-02", "2001-01-03", "2001-01-04")):
        opening = pd.Timestamp(f"{session} 09:30", tz="America/New_York").tz_convert("UTC")
        if split and day >= 1:
            prices = [45.0] * 13 if day == 1 else [52.0] * 13
        else:
            prices = [90.0] * 13 if day < 2 else [104.0] * 13
        for bar, price in enumerate(prices):
            start = opening + pd.Timedelta(minutes=30 * bar)
            absent = missing and day == 1 and bar == 2
            coverage.append(
                dict(
                    session=session,
                    bar_start=start,
                    bar_end=start + pd.Timedelta(minutes=30),
                    status="unknown_missing" if absent else "observed",
                    signal_eligible=not absent,
                    fill_eligible=not absent,
                )
            )
            if not absent:
                bars.append(
                    dict(
                        session=session,
                        bar_start=start,
                        bar_end=start + pd.Timedelta(minutes=30),
                        open=price + 0.1,
                        high=price + 0.1,
                        low=price,
                        close=price,
                        volume=100.0,
                        duration_minutes=30.0,
                    )
                )
        calendar.append(
            dict(
                session=session, market_open=opening, market_close=opening + pd.Timedelta(hours=6.5)
            )
        )
        daily.append(
            dict(
                session=session,
                close=prices[-1],
                dividend_amount=0.5 if day == 1 else 0.0,
                split_coefficient=2.0 if split and day == 1 else 1.0,
            )
        )
    cache = mr_engine.build_cache(
        pd.DataFrame(daily),
        pd.DataFrame(bars),
        pd.DataFrame(calendar),
        coverage=pd.DataFrame(coverage),
        quality_policy=quality.QualityPolicy("observed_only_blackout_v1"),
    )
    obs = pd.DataFrame(
        dict(available_at=pd.date_range("2001-01-01", "2001-01-05", tz="UTC"), annual_rate=0.03)
    )
    return cache, obs


def recorded(cache, obs, *, candidates=None, mode="candidate", fee=1, slip=3):
    candidates = [mr.candidate_grid()[0]] if candidates is None else candidates
    kwargs = dict(start="2001-01-02", end="2001-01-04", cash_observations=obs)
    parts = mr_engine.run(
        cache, candidates, **kwargs, commission_bps=fee, slippage_bps=slip, mode=mode, record=True
    )
    return parts, engine.replay_recorded(cache, parts, **kwargs)


def test_grid_exact_unique_and_equal_capital_44():
    allocations = engine.allocation_grid()
    assert len(allocations) == len({x.portfolio_id for x in allocations}) == 12
    assert [x.trend_weight for x in allocations[:11]] == [i / 10 for i in range(11)]
    assert allocations[-1].trend_weight == 12 / 44
    for allocation in allocations:
        assert allocation.weights().sum() == pytest.approx(1)
        assert (allocation.weights() >= 0).all()
    assert allocations[-1].weights() == pytest.approx(np.full(44, 1 / 44))
    assert allocations[0].weights()[:12].sum() == 0
    assert allocations[10].weights()[12:].sum() == 0


@pytest.mark.parametrize("weight", [True, False, -0.1, 1.1, np.inf, np.nan, "0.5"])
def test_invalid_allocation_weights_rejected(weight):
    with pytest.raises(ValueError):
        engine.Allocation("x", weight)


def test_aggregate_wealth_does_not_rebalance_daily_returns():
    paths = synthetic_paths()
    allocation = engine.Allocation("half", 0.5)
    result = engine.evaluate_portfolio(paths, allocation)
    expected = np.array([1050, 1245, 1194])
    assert result.daily_equity.to_numpy() == pytest.approx(expected)
    expected_returns = expected / np.r_[1000, expected[:-1]] - 1
    assert result.daily_returns.to_numpy() == pytest.approx(expected_returns)
    assert result.daily_returns.iloc[1] != pytest.approx(0.5 * (-0.1 + 0.5))
    excess = expected_returns
    expected_sharpe = np.sqrt(252) * excess.mean() / excess.std(ddof=1)
    assert result.metrics["cash_excess_sharpe"] == pytest.approx(expected_sharpe)
    assert result.metrics["cash_excess_sharpe"] != pytest.approx(
        0.5
        * (
            engine.cash_excess_sharpe(paths.returns[:, 0], paths.cash_returns)
            + engine.cash_excess_sharpe(paths.returns[:, 12], paths.cash_returns)
        )
    )
    assert result.metrics["max_drawdown"] == pytest.approx(1194 / 1245 - 1)
    assert result.metrics["ending_trend_weight"] == pytest.approx(594 / 1194)
    assert result.metrics["cost_drag_return"] == pytest.approx(0.01)
    assert result.annual_returns.annual_return.iloc[0] == pytest.approx(0.194)


def test_exposure_uses_drifting_actual_long_notional_not_fixed_style_average():
    paths = synthetic_paths()
    result = engine.evaluate_portfolio(paths, engine.Allocation("half", 0.5))
    exposure = (
        paths.event_invested_notional @ np.r_[np.full(12, 0.5 / 12), np.full(32, 0.5 / 32)]
    ) / (paths.event_equity @ np.r_[np.full(12, 0.5 / 12), np.full(32, 0.5 / 32)])
    expected = (
        np.sum(exposure[:-1] * np.diff(paths.event_times.asi8) / 1e9)
        / (paths.segment_end - paths.segment_start).total_seconds()
    )
    assert result.event_exposure.to_numpy() == pytest.approx(exposure)
    assert result.metrics["exposure_fraction"] == pytest.approx(expected)
    assert expected != pytest.approx(0.5)
    assert result.metrics["initial_weighted_binary_long_time_fraction"] == pytest.approx(1)


def test_simultaneous_gross_orders_use_full_portfolio_pre_fill_wealth():
    paths = synthetic_paths()
    before = np.r_[np.full(12, 1250.0), np.full(32, 800.0)]
    trades = pd.DataFrame(
        [
            dict(candidate_id="tf_0", reference_notional=1250, commission=0.125, slippage=0.375),
            dict(candidate_id="mr_0", reference_notional=800, commission=0.08, slippage=0.24),
        ]
    )
    group = engine.FillGroup(paths.times[0], "open", 100.0, before, trades)
    paths = replace(paths, fill_groups=(group,))
    result = engine.evaluate_portfolio(paths, engine.Allocation("half", 0.5))
    denominator = 0.5 * 1250 + 0.5 * 800
    assert result.metrics["total_turnover"] == pytest.approx(
        (0.5 / 12 * 1250 + 0.5 / 32 * 800) / denominator
    )
    assert result.metrics["gross_sleeve_order_count"] == 2
    assert result.metrics["distinct_execution_timestamp_count"] == 1
    assert result.metrics["total_commission"] == pytest.approx(0.5 / 12 * 0.125 + 0.5 / 32 * 0.08)
    pure_trend = engine.evaluate_portfolio(paths, engine.Allocation("pure", 1))
    assert pure_trend.metrics["gross_sleeve_order_count"] == 1
    assert pure_trend.metrics["total_turnover"] == pytest.approx(1 / 12)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("mode", ["candidate", "cash", "buy_and_hold"])
def test_recorded_ledger_reconciles_splits_dividends_drip_gaps_and_cash(split, missing, mode):
    cache, obs = fixture(split=split, missing=missing)
    parts, paths = recorded(cache, obs, mode=mode)
    assert paths.equity == pytest.approx(parts[2])
    assert paths.returns == pytest.approx(parts[3])
    assert paths.event_long[-1].tolist() == [False]
    assert paths.event_equity[-1] == pytest.approx(parts[2][-1])
    assert sum(len(group.trades) for group in paths.fill_groups) == parts[0][0]["order_count"]
    assert all(group.phase in {"open", "terminal"} for group in paths.fill_groups)
    if mode == "cash":
        assert len(paths.fill_groups) == 0
        assert paths.event_invested_notional.max() == 0
    if mode == "buy_and_hold":
        assert parts[0][0]["order_count"] == 2  # DRIP is cost-free, not a gross order.


def test_unrecorded_trade_result_is_rejected():
    cache, obs = fixture()
    parts, _ = recorded(cache, obs, mode="buy_and_hold")
    broken = (*parts[:6], [], *parts[7:])
    with pytest.raises(ValueError, match="record=True"):
        engine.replay_recorded(
            cache, broken, start="2001-01-02", end="2001-01-04", cash_observations=obs
        )


def test_post_run_cache_mutation_rejected_even_without_a_trade_at_changed_quote():
    cache, obs = fixture()
    parts, _ = recorded(cache, obs, mode="cash")
    cache.bars.loc[0, "open"] += 1
    with pytest.raises(ValueError, match="source/cache identity changed"):
        engine.replay_recorded(
            cache, parts, start="2001-01-02", end="2001-01-04", cash_observations=obs
        )


@pytest.mark.parametrize(
    "field", ["commission", "execution_price", "quantity", "equity_before", "holding_hours"]
)
def test_tampered_trade_data_is_rejected(field):
    cache, obs = fixture()
    parts, _ = recorded(cache, obs, mode="buy_and_hold")
    trades = [dict(row) for row in parts[6]]
    trades[0][field] += 1
    broken = (*parts[:6], trades, *parts[7:])
    with pytest.raises(ValueError, match="reconcile"):
        engine.replay_recorded(
            cache, broken, start="2001-01-02", end="2001-01-04", cash_observations=obs
        )


def test_merge_exact_44_and_roundtrip_aggregate():
    cache, obs = fixture()
    _, trend = recorded(cache, obs, candidates=mr.candidate_grid()[:12])
    _, mean = recorded(cache, obs, candidates=mr.candidate_grid()[12:44])
    paths = engine.merge_sleeves(trend, mean)
    result = engine.evaluate_portfolio(paths, engine.allocation_grid()[-1])
    assert len(paths.candidate_ids) == 44
    assert result.daily_equity.to_numpy() == pytest.approx(paths.equity.mean(axis=1))
    assert result.metrics["gross_sleeve_order_count"] == sum(
        x["order_count"] for x in paths.metrics
    )
    assert result.metrics["terminal_equity"] == pytest.approx(paths.equity[-1].mean())


def test_merge_rejects_mismatched_cash_references_no_fill():
    cache, obs = fixture()
    _, trend = recorded(cache, obs, candidates=mr.candidate_grid()[:12])
    _, mean = recorded(cache, obs, candidates=mr.candidate_grid()[12:44])
    with pytest.raises(ValueError, match="match exactly"):
        engine.merge_sleeves(trend, replace(mean, cash_returns=mean.cash_returns + 1e-12))


def test_pure_style_identical_sleeves_reproduces_single_wealth():
    paths = synthetic_paths()
    result = engine.evaluate_portfolio(paths, engine.Allocation("pure_trend", 1))
    assert result.daily_equity.to_numpy() == pytest.approx(paths.equity[:, 0])
    assert result.daily_returns.to_numpy() == pytest.approx(paths.returns[:, 0])


def test_all_cash_reconciles_and_sharpe_undefined():
    paths = synthetic_paths()
    cash_returns = np.array([0.001, 0.002, 0.003])
    equity = np.tile(1000 * np.cumprod(1 + cash_returns), (44, 1)).T
    event_equity = np.vstack([np.full(44, 1000), equity])
    paths = replace(
        paths,
        equity=equity,
        returns=np.tile(cash_returns, (44, 1)).T,
        cash_returns=equity[:, 0] / np.r_[1000, equity[:-1, 0]] - 1,
        gross_terminal_equity=equity[-1],
        event_equity=event_equity,
        event_invested_notional=np.zeros_like(event_equity),
        event_long=np.zeros_like(event_equity, dtype=bool),
    )
    result = engine.evaluate_portfolio(paths, engine.Allocation("cash", 0))
    # Weighted summation can introduce one-ulp account arithmetic. This cash
    # test intentionally retains identical generated cash-wealth references.
    assert result.daily_equity.to_numpy() == pytest.approx(equity[:, 0])
    assert result.metrics["exposure_fraction"] == 0
    assert result.metrics["gross_sleeve_order_count"] == 0
    assert result.metrics["cost_drag_return"] == pytest.approx(0)
    assert np.isnan(result.metrics["cash_excess_sharpe"])


def test_selection_finite_ties_closest_half_and_stable_id():
    paths = synthetic_paths()
    results = [
        engine.evaluate_portfolio(paths, allocation) for allocation in engine.allocation_grid()
    ]
    results = [replace(x, metrics={**x.metrics, "cash_excess_sharpe": 1}) for x in results]
    assert engine.select_allocation(results).trend_weight == 0.5
    tied = [
        replace(results[4], metrics={"cash_excess_sharpe": 1}),
        replace(results[6], metrics={"cash_excess_sharpe": 1 + 5e-11}),
    ]
    assert engine.select_allocation(tied).portfolio_id == "portfolio_tf_040"
    symmetric = [
        replace(results[3], metrics={"cash_excess_sharpe": 1}),
        replace(results[7], metrics={"cash_excess_sharpe": 1}),
    ]
    assert engine.select_allocation(symmetric).portfolio_id == "portfolio_tf_030"
    with pytest.raises(ValueError, match="No finite"):
        engine.select_allocation([replace(results[0], metrics={"cash_excess_sharpe": np.nan})])


def test_constant_excess_correlation_remains_nan():
    paths = synthetic_paths()
    equity, returns = paths.equity.copy(), paths.returns.copy()
    equity[:, 0] = 1000
    returns[:, 0] = 0
    event_equity = paths.event_equity.copy()
    event_equity[:, 0] = 1000
    invested = paths.event_invested_notional.copy()
    invested[:, 0] = 0
    paths = replace(
        paths,
        equity=equity,
        returns=returns,
        event_equity=event_equity,
        event_invested_notional=invested,
    )
    result = engine.correlations(paths)
    assert len(result["constituent"]) == 44
    assert result["constituent"].loc["tf_0"].isna().all()
    assert result["style"].index.tolist() == ["trend", "mean_reversion"]


def test_annual_returns_use_prior_year_wealth_not_reset_capital():
    times = pd.DatetimeIndex(["2001-12-31 22:00Z", "2002-01-02 22:00Z", "2002-12-31 22:00Z"])
    result = engine.annual_return_table(pd.Series([1100, 990, 1210], index=times))
    assert result.annual_return.tolist() == pytest.approx([0.1, 0.1])
    assert result.start_wealth.tolist() == [1000, 1100]


def test_bootstrap_paired_reproducible_and_identity_comparator_zero():
    rng = np.random.default_rng(42)
    chosen = rng.normal(0.001, 0.01, 100)
    bh = rng.normal(0.0005, 0.013, 100)
    cash = np.full(100, 0.0001)
    one = engine.bootstrap_sharpe_differences(chosen, bh, chosen, cash, replicates=100)
    two = engine.bootstrap_sharpe_differences(chosen, bh, chosen, cash, replicates=100)
    pd.testing.assert_frame_equal(one, two)
    assert len(one) == 6
    exact = one.loc[one.comparator.eq("50_50")]
    assert exact.lower_95_percentile.eq(0).all()
    assert exact.upper_95_percentile.eq(0).all()
    assert exact.observed_sharpe_difference.eq(0).all()
    assert not one.corrects_candidate_selection_bias.any()


def test_invalid_bootstrap_or_path_shapes_rejected():
    with pytest.raises(ValueError):
        engine.bootstrap_sharpe_differences(
            [0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0, 0], block_lengths=(3,)
        )
    paths = synthetic_paths()
    with pytest.raises(ValueError, match="reconcile"):
        engine.evaluate_portfolio(
            replace(paths, returns=paths.returns + 1e-5), engine.Allocation("x", 0.5)
        )
    with pytest.raises(ValueError, match="excess invested"):
        engine.evaluate_portfolio(
            replace(paths, event_invested_notional=paths.event_equity + 1),
            engine.Allocation("x", 0.5),
        )
