"""Artificial daily fixtures only; never read sealed historical/test outcomes."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import trend_following.research_split_v3 as v3


def raw_frame(close, dates=None, *, dividends=None, splits=None):
    dates = (
        pd.bdate_range("2020-01-02", periods=len(close)) if dates is None else pd.to_datetime(dates)
    )
    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "adj_close": close,
            "dividend_amount": dividends if dividends is not None else 0.0,
            "split_coefficient": splits if splits is not None else 1.0,
        }
    )


def rates(panel, rate=0.0, series_id="DFF"):
    stamps = pd.date_range(
        panel.data.index[0] - pd.Timedelta(days=3),
        panel.data.index[-1] + pd.Timedelta(days=3),
        freq="D",
    )
    return pd.DataFrame({"available_at": stamps, "annual_rate": rate, "series_id": series_id})


def run(
    panel,
    spec=None,
    *,
    cash_rate=0.0,
    fin_rate=0.0,
    costs=None,
    layers=v3.LAYERS,
    start=None,
    end=None,
):
    return v3.evaluate_daily_segment(
        panel,
        rates(panel, fin_rate),
        rates(panel, cash_rate, "DTB3"),
        spec or v3.benchmark_specs()["qqq_buy_hold"],
        start=panel.data.index[0] if start is None else start,
        end=panel.data.index[-1] if end is None else end,
        costs=costs or v3.SplitCosts(),
        layers=layers,
    )


@pytest.fixture(scope="module")
def history():
    t = np.arange(420)
    return v3.prepare_daily_panel(
        raw_frame(100 * np.exp(np.cumsum(0.0005 + 0.003 * np.sin(t / 23))))
    )


def test_grid_is_fresh_nine_ma_session_allocations_without_B5():
    candidates = v3.default_candidates()
    assert len(candidates) == 9
    assert {x["ma_sessions"] for x in candidates.values()} == {50, 100, 200}
    assert {x["allocation"] for x in candidates.values()} == {1 / 3, 2 / 3, 1}
    assert all(x["kind"] == "ma" and x["asset"] == "synthetic_3x" for x in candidates.values())


def test_causal_raw_split_dividend_total_return_and_declared_daily_clock():
    raw = raw_frame([100, 100, 49, 49], dividends=[0, 0, 1, 0], splits=[1, 1, 2, 1])
    panel = v3.prepare_daily_panel(raw)
    assert panel.data.total_return_index.tolist() == pytest.approx([100] * 4)
    assert panel.data.split_price_index.tolist() == [100, 100, 98, 98]
    assert panel.data.split_neutral_price_return.tolist() == pytest.approx([0, 0, -0.02, 0])
    assert (panel.data.index.hour == 16).all()
    assert str(panel.data.index.tz) == "America/New_York"
    raw.adj_close *= np.linspace(1, 2, len(raw))
    changed = v3.prepare_daily_panel(raw)
    pd.testing.assert_series_equal(panel.data.total_return_index, changed.data.total_return_index)


@pytest.mark.parametrize(
    "change", ["duplicate", "nonpositive", "missing_dividend", "negative_dividend", "weekend"]
)
def test_invalid_raw_daily_source_fails(change):
    frame = raw_frame([100, 101, 102])
    if change == "duplicate":
        frame.loc[1, "date"] = frame.loc[0, "date"]
    elif change == "nonpositive":
        frame.loc[1, "close"] = 0
    elif change == "missing_dividend":
        frame = frame.drop(columns="dividend_amount")
    elif change == "negative_dividend":
        frame.loc[1, "dividend_amount"] = -0.1
    else:
        frame.loc[2, "date"] = pd.Timestamp("2020-01-04")
    with pytest.raises(ValueError):
        v3.prepare_daily_panel(frame)


def test_ma_counts_completed_sessions_and_delays_one_session():
    panel = v3.prepare_daily_panel(raw_frame(np.arange(70, dtype=float) + 100))
    spec = v3.default_candidates()["ma50_alloc_1_3"]
    start = panel.data.index[49]
    path = v3.build_daily_ma_targets(panel, spec, start=start, end=panel.data.index[-1])
    assert path.moving_average.iloc[0] == pytest.approx(
        panel.data.total_return_index.iloc[:50].mean()
    )
    assert path.decision_target.iloc[0] == 1 / 3
    assert path.weights_at_close.iloc[0] == 0
    assert path.weights_at_close.iloc[1] == 1 / 3
    assert path.decision_timestamp_for_fill.iloc[1] == start
    assert path.fill_timestamp.iloc[1] == panel.data.index[50]
    assert path.target_change.sum() == 1
    with pytest.raises(ValueError, match="warmup"):
        v3.build_daily_ma_targets(panel, spec, start=panel.data.index[48], end=panel.data.index[-1])


def test_equal_to_ma_means_cash_and_prior_orders_never_carry():
    panel = v3.prepare_daily_panel(raw_frame([100.0] * 250))
    spec = v3.default_candidates()["ma200_alloc_3_3"]
    path = v3.build_daily_ma_targets(
        panel, spec, start=panel.data.index[220], end=panel.data.index[-1]
    )
    assert path.decision_target.eq(0).all()
    assert path.weights_at_close.eq(0).all()


def test_daily_reset_not_hourly_reset_no_fifty_percent_clip():
    panel = v3.prepare_daily_panel(raw_frame([100, 110, 121]))
    series = v3.synthetic_daily_returns(
        panel, rates(panel), end=panel.data.index[-1], layer="gross"
    )
    assert series.synthetic_nav.tolist() == pytest.approx([100, 130, 169])
    loss = v3.prepare_daily_panel(raw_frame([100, 80]))
    uncut = v3.synthetic_daily_returns(loss, rates(loss), end=loss.data.index[-1], layer="gross")
    assert uncut.modeled_return.iloc[-1] == pytest.approx(-0.6)
    bankrupt = v3.prepare_daily_panel(raw_frame([100, 66]))
    with pytest.raises(ValueError, match="bankruptcy"):
        v3.synthetic_daily_returns(
            bankrupt, rates(bankrupt), end=bankrupt.data.index[-1], layer="gross"
        )


def test_financing_and_expense_accrue_elapsed_calendar_time_once():
    panel = v3.prepare_daily_panel(raw_frame([100, 100], ["2020-01-03", "2020-01-06"]))
    costs = v3.SplitCosts()
    result = v3.synthetic_daily_returns(
        panel, rates(panel, 0.04), end=panel.data.index[-1], costs=costs
    )
    assert result.financing_drag.iloc[-1] == pytest.approx(2 * (0.04 + 0.005) * 3 / 360)
    assert result.expense_tracking_drag.iloc[-1] == pytest.approx((0.0095 + 0.005) * 3 / 365)
    assert result.modeled_return.iloc[-1] == pytest.approx(-2 * 0.045 * 3 / 360 - 0.0145 * 3 / 365)
    assert result.iloc[0][["gross_return", "financing_drag", "expense_tracking_drag"]].eq(0).all()


def test_same_close_buy_cannot_earn_return_and_portfolio_allocations_drift():
    panel = v3.prepare_daily_panel(raw_frame([100, 110, 121]))
    spec = dict(name="artificial_fraction", kind="buy_hold", asset="synthetic_3x", allocation=1 / 3)
    result = run(panel, spec, layers=("gross",))
    ledger = result.paths["gross"]
    assert ledger.equity.tolist() == pytest.approx([1000, 1000, 1100])
    assert ledger.orders.event.tolist() == ["buy"]
    assert ledger.holdings.actual_weight.iloc[-1] == pytest.approx((1000 / 3 * 1.3) / 1100)
    assert ledger.holdings.effective_exposure.iloc[-1] > 1
    assert result.metrics.strategy_target_changes.iloc[0] == 1
    assert result.metrics.filled_orders.iloc[0] == 2  # includes cloned exit


def test_physical_qqq_split_distribution_only_old_holders_receive_cash():
    panel = v3.prepare_daily_panel(
        raw_frame([100, 100, 49, 49], dividends=[0, 0, 1, 0], splits=[1, 1, 2, 1])
    )
    result = run(panel, costs=v3.SplitCosts(fee_bps=0, slippage_bps=0), layers=("gross", "tax"))
    gross = result.paths["gross"]
    assert gross.equity.tolist() == pytest.approx([1000] * 4)
    assert gross.holdings.quantity.tolist() == pytest.approx([0, 10, 20 + 20 / 49, 20 + 20 / 49])
    assert gross.cash.iloc[-1] == pytest.approx(0)
    assert gross.ledger.quantity("QQQ") == pytest.approx(20 + 20 / 49)
    assert result.metrics.dividend_reinvestment_orders.tolist() == [1, 1]
    assert result.metrics.strategy_target_changes.tolist() == [1, 1]
    assert result.terminal_snapshots["gross"]["ledger"].quantity("QQQ") == pytest.approx(0)
    assert result.metrics.set_index("layer").loc["tax", "total_taxes_paid"] == pytest.approx(4.8)
    assert result.metrics.set_index("layer").loc["tax", "terminal_equity"] == pytest.approx(995.2)
    # Moving distribution to the buy date does not grant a new buyer entitlement.
    ex_entry = v3.prepare_daily_panel(raw_frame([100, 99, 99], dividends=[0, 1, 0]))
    new_buyer = run(ex_entry, layers=("gross",)).paths["gross"]
    assert new_buyer.cash.iloc[-1] == pytest.approx(0)


def test_cash_benchmark_retains_negative_rates():
    panel = v3.prepare_daily_panel(raw_frame([100, 101, 102]))
    result = run(panel, v3.benchmark_specs()["cash"], cash_rate=-0.01, layers=("expense",))
    assert result.paths["expense"].cash_interest.iloc[1] < 0
    assert result.metrics.terminal_equity.iloc[0] < 1000
    assert result.paths["expense"].orders.empty
    assert result.metrics.mean_cash_weight.iloc[0] == pytest.approx(1)


def test_same_close_dividend_cash_never_earns_preceding_weekend_interest():
    panel = v3.prepare_daily_panel(raw_frame([100, 100, 99], dividends=[0, 0, 1]))
    result = run(panel, cash_rate=0.365, layers=("gross",))
    ledger = result.paths["gross"]
    assert ledger.cash_interest.iloc[1] == pytest.approx(1.0)
    assert ledger.cash_interest.iloc[2] == pytest.approx(0.0)
    assert ledger.equity.tolist() == pytest.approx([1000, 1001, 1001])
    assert ledger.cash.iloc[-1] == pytest.approx(0.0)
    assert result.metrics.dividend_reinvestment_orders.iloc[0] == 1


def test_all_five_layers_share_targets_and_return_products_match_cloned_wealth(history):
    spec = v3.default_candidates()["ma50_alloc_2_3"]
    result = run(
        history,
        spec,
        cash_rate=0.02,
        fin_rate=0.04,
        start=history.data.index[220],
        end=history.data.index[300],
    )
    assert result.metrics.layer.tolist() == list(v3.LAYERS)
    for layer in v3.LAYERS:
        pd.testing.assert_series_equal(
            result.paths[layer].holdings.set_index("timestamp").target_weight,
            result.target_weights,
            check_names=False,
        )
        snapshot = result.terminal_snapshots[layer]
        assert (1 + result.daily_returns[layer]).prod() == pytest.approx(
            snapshot["liquidated_equity"] / 1000
        )
        assert result.paths[layer].ledger is not snapshot["ledger"]
        assert snapshot["ledger"].quantity("synthetic_3x") == 0


def test_qqq_and_cash_never_receive_synthetic_financing_expense_double_charge(history):
    for name in ["qqq_buy_hold", "cash", "qqq200ma"]:
        result = run(
            history,
            v3.benchmark_specs()[name],
            cash_rate=0.02,
            fin_rate=0.2,
            start=history.data.index[220],
            end=history.data.index[250],
        )
        pd.testing.assert_series_equal(
            result.daily_returns.trading, result.daily_returns.financing, check_names=False
        )
        pd.testing.assert_series_equal(
            result.daily_returns.trading, result.daily_returns.expense, check_names=False
        )


def test_first_entry_boundary_uses_actual_wealth_after_cash_carry():
    panel = v3.prepare_daily_panel(raw_frame([100, 100, 100]))
    result = run(panel, cash_rate=0.04, layers=("expense",))
    cleaned, factor, info = v3.bootstrap_components(result)
    assert cleaned.iloc[1] == pytest.approx(0.04 / 365)
    assert info["first_entry_factor"] == pytest.approx(1 / 1.0006)
    assert info["terminal_factor"] == pytest.approx(0.9994)
    assert factor == pytest.approx(0.9994 / 1.0006)
    assert (1 + cleaned).prod() * factor == pytest.approx((1 + result.daily_returns.expense).prod())


def test_flat_reset_does_not_carry_warmup_profits_orders_or_tax_lots(history):
    spec = v3.benchmark_specs()["qqq_buy_hold"]
    first = run(
        history,
        spec,
        start=history.data.index[210],
        end=history.data.index[300],
        layers=("expense",),
    )
    saved = copy.deepcopy(vars(first.paths["expense"].ledger))
    later = run(
        history,
        spec,
        start=history.data.index[301],
        end=history.data.index[400],
        layers=("expense",),
    )
    assert later.paths["expense"].equity.iloc[0] == 1000
    assert later.target_weights.iloc[0] == 0
    assert later.paths["expense"].orders.timestamp.iloc[0] == history.data.index[302]
    assert vars(first.paths["expense"].ledger) == saved
    assert all(
        lot.acquired >= history.data.index[302] for lot in later.paths["expense"].ledger.lots["QQQ"]
    )


def test_validation_selection_ignores_all_later_test_prices_and_rates(history):
    cutoff = history.data.index[320]
    kwargs = dict(start=history.data.index[220], end=cutoff)
    financing, cash = rates(history, 0.04), rates(history, 0.02, "DTB3")
    winner, table = v3.select_validation_candidate(history, financing, cash, **kwargs)
    future_changed = copy.deepcopy(history)
    future_changed.data.loc[future_changed.data.index > cutoff] = np.inf
    fin2, cash2 = financing.copy(), cash.copy()
    fin2.loc[fin2.available_at > cutoff, "annual_rate"] = np.inf
    cash2.loc[cash2.available_at > cutoff, "annual_rate"] = -np.inf
    winner2, table2 = v3.select_validation_candidate(future_changed, fin2, cash2, **kwargs)
    assert winner2 == winner
    pd.testing.assert_frame_equal(table, table2)
    assert len(table) == 9 and table.selected.sum() == 1
    assert table.layer.eq("expense").all()
    assert table.selection_cutoff.eq(cutoff.isoformat()).all()


def test_ties_favor_lower_allocation_then_longer_ma_and_no_drawdown_hurdle(monkeypatch):
    def fake(panel, fin, cash, spec, **kwargs):
        score = -5.0 if spec["allocation"] == 1 / 3 else -5.0 + 5e-11
        return SimpleNamespace(
            metrics=pd.DataFrame(
                [
                    dict(
                        excess_return_Sharpe=score,
                        allocation=spec["allocation"],
                        ma_sessions=spec["ma_sessions"],
                        max_drawdown=-0.99,
                    )
                ]
            )
        )

    monkeypatch.setattr(v3, "evaluate_daily_segment", fake)
    winner, _ = v3.select_validation_candidate(
        None, None, None, start="2012-01-01", end="2015-12-31"
    )
    assert winner == "ma200_alloc_1_3"
    with pytest.raises(ValueError, match="nine"):
        v3.select_validation_candidate(None, None, None, {}, start="2012-01-01", end="2015-12-31")
    with pytest.raises(ValueError, match="tolerance"):
        v3.select_validation_candidate(
            None, None, None, start="2012-01-01", end="2015-12-31", tie_tolerance=0.001
        )


def test_all_undefined_validation_scores_fail_instead_of_arbitrary_cash(monkeypatch):
    def fake(panel, fin, cash, spec, **kwargs):
        return SimpleNamespace(
            metrics=pd.DataFrame(
                [
                    dict(
                        excess_return_Sharpe=np.nan,
                        allocation=spec["allocation"],
                        ma_sessions=spec["ma_sessions"],
                    )
                ]
            )
        )

    monkeypatch.setattr(v3, "evaluate_daily_segment", fake)
    with pytest.raises(ValueError, match="No finite"):
        v3.select_validation_candidate(None, None, None, start="2012-01-01", end="2015-12-31")


def test_rate_missing_or_stale_is_error_not_zero_fill():
    panel = v3.prepare_daily_panel(raw_frame([100, 101, 102]))
    missing = pd.DataFrame({"available_at": [panel.data.index[-1]], "annual_rate": [0.04]})
    with pytest.raises(ValueError, match="No published"):
        v3.synthetic_daily_returns(panel, missing, end=panel.data.index[-1])
    stale = pd.DataFrame(
        {"available_at": [panel.data.index[0] - pd.Timedelta(days=30)], "annual_rate": [0.04]}
    )
    with pytest.raises(ValueError, match="Stale"):
        v3.synthetic_daily_returns(panel, stale, end=panel.data.index[-1])


def test_cost_config_and_different_source_rate_roles_are_validated():
    with pytest.raises(ValueError):
        replace(v3.SplitCosts(), initial_cash=-1)
    with pytest.raises(ValueError):
        replace(v3.SplitCosts(), expense_ratio=-0.001)
    panel = v3.prepare_daily_panel(raw_frame([100, 101, 102]))
    with pytest.raises(ValueError, match="DFF"):
        v3.synthetic_daily_returns(panel, rates(panel, 0.04, "DTB3"), end=panel.data.index[-1])
