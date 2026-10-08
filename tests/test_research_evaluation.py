from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_following.research_data import stationary_day_bootstrap, synthetic_session_reset_price
from trend_following.research_evaluation import (
    CostScenario,
    chronological_folds,
    inner_validation_windows,
    scenario_asset_returns,
    select_candidate,
    waterfall_scenarios,
)


def test_daily_not_hourly_leverage_reset():
    index = pd.to_datetime(
        ["2020-01-02 10:00", "2020-01-02 11:00", "2020-01-03 10:00"]
    ).tz_localize("America/New_York")
    price = synthetic_session_reset_price(pd.Series([100, 110, 121], index=index))
    assert price.tolist() == pytest.approx([100, 130, 169])
    with pytest.raises(ValueError, match="nonpositive"):
        synthetic_session_reset_price(pd.Series([100, 60, 60], index=index))


def test_gross_does_not_include_financing():
    index = pd.date_range("2020-01-01", periods=3, tz="UTC")
    raw = pd.Series([100, 100, 100], index=index)
    financing = pd.Series([0, 0.001, 0.001], index=index)
    scenarios = waterfall_scenarios(CostScenario(name="base"))
    assert scenario_asset_returns(raw, financing, scenarios[0]).eq(0).all()
    assert scenario_asset_returns(raw, financing, scenarios[1]).eq(0).all()
    assert scenario_asset_returns(raw, financing, scenarios[2]).iloc[1] < -0.002
    assert (
        scenario_asset_returns(raw, financing, scenarios[2], fund_costs_embedded=True).eq(0).all()
    )


def test_folds_do_not_overlap_and_inner_dates_precede_outer():
    index = pd.bdate_range("2002-01-10", "2026-06-01", tz="America/New_York")
    folds = chronological_folds(index)
    assert len(folds) == 6
    assert all(a.test_end < b.test_start for a, b in zip(folds, folds[1:], strict=False))
    windows = inner_validation_windows(index, folds[0].train_end)
    assert len(windows) == 4
    assert all(end <= folds[0].train_end for _, end in windows)


def test_selector_isolated_from_future_and_tie_break(monkeypatch):
    index = pd.bdate_range("2002-01-10", "2011-12-30", tz="America/New_York")
    prices = pd.Series(np.arange(len(index)) + 100.0, index=index)
    fold = chronological_folds(index)[0]

    def fake(qqq, risk, financing, cash, spec, scenario, **kwargs):
        assert qqq.index[-1] <= fold.train_end
        assert risk.index[-1] <= fold.train_end
        return None, None, {"excess_return_Sharpe": 1.0, "max_drawdown": -0.2}

    monkeypatch.setattr("trend_following.research_evaluation.evaluate_path", fake)
    candidates = {"b5": {"complexity_rank": 4}, "simple": {"complexity_rank": 1}}
    chosen, table = select_candidate(
        prices, prices, prices, prices, candidates, fold, CostScenario("x")
    )
    assert chosen == "simple"
    assert (table.validation_end <= fold.train_end).all()


def test_whole_day_bootstrap_does_not_pad_partial_sessions():
    index = pd.to_datetime(
        [
            "2020-01-02 10:00",
            "2020-01-02 11:00",
            "2020-01-03 10:00",
            "2020-01-06 10:00",
            "2020-01-06 11:00",
        ]
    ).tz_localize("America/New_York")
    prices = pd.Series([100, 101, 99, 102, 103], index=index)
    a, paired, sampled = stationary_day_bootstrap(
        prices, block_days=2, seed=4, paired_series={"rate": prices * 0 + 0.01}
    )
    b, _, _ = stationary_day_bootstrap(prices, block_days=2, seed=4)
    pd.testing.assert_series_equal(a, b)
    assert len(sampled) == 3
    assert paired["rate"].eq(0.01).all()
    assert a.iloc[0] == 100


def test_physical_distribution_ledger_is_not_double_counted():
    from trend_following.research_evaluation import evaluate_path

    index = pd.date_range("2020-01-01", periods=5, tz="America/New_York")
    total_price = pd.Series(100.0, index=index)
    raw_price = pd.Series([100.0, 100.0, 100.0, 99.0, 99.0], index=index)
    physical_ret = raw_price.pct_change().fillna(0)
    zero = pd.Series(0.0, index=index)
    actions = pd.DataFrame(
        [
            dict(
                timestamp=index[3],
                asset="risk",
                split_ratio=1.0,
                distribution_per_share=1.0,
                tax_kind="ordinary",
            )
        ]
    )
    scenario = CostScenario(
        "zero", fee_bps=0, slippage_bps=0, include_financing=False, expense_ratio=0, extra_drag=0
    )
    _, ledger, metrics = evaluate_path(
        total_price,
        total_price,
        zero,
        zero,
        {"kind": "buyhold", "product_leverage": 1, "effective_leverage": 1},
        scenario,
        start=index[0],
        fund_costs_embedded=True,
        physical_returns=physical_ret,
        physical_prices=raw_price,
        corporate_actions=actions,
    )
    assert ledger.equity.iloc[-1] == pytest.approx(1.0)
    distribution = ledger.events.loc[ledger.events.event.eq("distribution")]
    assert distribution.cash_flow.sum() == pytest.approx(0.01)
    assert metrics["mean_effective_exposure"] <= 1


def test_evaluation_wires_costs_and_overrides_frozen_delay_aliases():
    from trend_following.research_evaluation import evaluate_path

    index = pd.date_range("2020-01-01", periods=12, tz="America/New_York")
    price = pd.Series(np.linspace(100, 120, len(index)), index=index)
    zero = price * 0
    scenario = CostScenario("delayed", fee_bps=10, slippage_bps=20, fill_delay_bars=7)
    spec = {"kind": "buyhold", "delay_bars": 1}
    path, _, _ = evaluate_path(
        price,
        price,
        zero,
        zero,
        spec,
        scenario,
        start=index[0],
        schedule={index[0]: spec},
        fund_costs_embedded=True,
    )
    assert path.metadata["costs_affect_trade_reference"] == {
        "slippage_bps": 20,
        "transaction_cost_bps": 10,
    }
    fills = path.events.loc[path.events.event.eq("filled")]
    assert fills.iloc[0].fill_timestamp == index[7]


def test_taxable_cash_benchmark_is_exact_and_zero_yield_sensitivity_uses_market_rf():
    from trend_following.research_evaluation import evaluate_path

    index = pd.date_range("2020-12-20", periods=40, tz="America/New_York")
    price = pd.Series(100.0, index=index)
    cash = pd.Series(0.001, index=index)
    zero = cash * 0
    scenario = CostScenario("tax", interest_tax_rate=0.24)
    _, _, metrics = evaluate_path(
        price, price, zero, cash, {"kind": "cash"}, scenario, start=index[0]
    )
    assert np.isnan(metrics["excess_return_Sharpe"])
    from dataclasses import replace

    _, _, stressed = evaluate_path(
        price,
        price,
        zero,
        cash,
        {"kind": "cash"},
        replace(scenario, cash_multiplier=0),
        start=index[0],
    )
    assert stressed["excess_return_Sharpe"] < 0


def test_first_evaluation_split_quote_initialization():
    from trend_following.research_evaluation import evaluate_path

    index = pd.date_range("2020-01-01", periods=5, tz="America/New_York")
    total = pd.Series(100.0, index=index)
    raw = pd.Series(50.0, index=index)
    zero = total * 0
    actions = pd.DataFrame(
        [
            dict(
                timestamp=index[0],
                asset="risk",
                split_ratio=2.0,
                distribution_per_share=0.0,
                tax_kind="ordinary",
            )
        ]
    )
    _, ledger, _ = evaluate_path(
        total,
        total,
        zero,
        zero,
        {"kind": "buyhold", "product_leverage": 1, "effective_leverage": 1},
        CostScenario("zero", fee_bps=0, slippage_bps=0),
        start=index[0],
        fund_costs_embedded=True,
        physical_returns=zero,
        physical_prices=raw,
        corporate_actions=actions,
    )
    assert ledger.holdings.price.eq(50).all()
    assert ledger.equity.iloc[-1] == pytest.approx(1)


def test_distribution_panel_rejects_missing_interior_actions(tmp_path):
    from trend_following.research_data import distribution_aware_panel

    index = pd.date_range("2020-01-01 10:00", periods=3)
    hourly = tmp_path / "hourly.parquet"
    daily = tmp_path / "daily.parquet"
    pd.DataFrame({"date": index, "adj_close": 100.0}).to_parquet(hourly)
    pd.DataFrame(
        {
            "date": index[[0, 2]].normalize(),
            "adj_close": 100.0,
            "close": 100.0,
            "dividend_amount": 0.0,
            "split_coefficient": 1.0,
        }
    ).to_parquet(daily)
    with pytest.raises(ValueError, match="interior"):
        distribution_aware_panel(hourly, daily)


def test_real_selector_future_mutation_cannot_change_training_result():
    index = pd.bdate_range("2002-01-10", "2010-12-31", tz="America/New_York")
    price = pd.Series(100 * np.exp(np.arange(len(index)) * 0.0005), index=index)
    zero = price * 0
    fold = chronological_folds(index)[0]
    candidates = {
        "cash": {"kind": "cash", "complexity_rank": 0},
        "buyhold": {
            "kind": "buyhold",
            "product_leverage": 1,
            "effective_leverage": 1,
            "complexity_rank": 1,
        },
    }
    first, table = select_candidate(
        price, price, zero, zero, candidates, fold, CostScenario("test"), bars_per_day=1
    )
    mutated = price.copy()
    mutated.loc[mutated.index > fold.train_end] *= 0.01
    second, other = select_candidate(
        mutated, mutated, zero, zero, candidates, fold, CostScenario("test"), bars_per_day=1
    )
    assert first == second
    pd.testing.assert_frame_equal(table, other)
