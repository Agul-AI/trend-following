"""Hand-calculated scenarios for the executable ledger (not tax-compliance tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_following.research_ledger import LedgerConfig, PortfolioLedger, simulate_ledger


def run_path(returns, weights, *, dates=None, config=None, **kwargs):
    index = pd.to_datetime(dates or ["2020-01-02", "2020-01-03"])
    return simulate_ledger(
        pd.DataFrame({"X": returns}, index=index),
        pd.DataFrame({"X": weights}, index=index),
        pd.Series(0.0, index=index),
        config or LedgerConfig(),
        **kwargs,
    )


def test_flat_price_entry_and_terminal_exit_charge_explicit_cash_inclusive_costs():
    result = run_path([0.0, 0.0], [1.0, 1.0], config=LedgerConfig(fee_bps=60, slippage_bps=40))
    # Buying N requires N + 1%*N cash; selling returns N - 1%*N.
    assert result.equity.iloc[-1] == pytest.approx(0.99 / 1.01)
    assert result.orders.event.tolist() == ["buy", "sell"]
    assert result.orders.reason.tolist() == ["rebalance", "terminal_liquidation"]
    assert result.orders.fee.sum() == pytest.approx(2 * (1 / 1.01) * 0.006)
    assert result.orders.slippage.sum() == pytest.approx(2 * (1 / 1.01) * 0.004)
    assert result.turnover.iloc[-1] == pytest.approx(1.0)
    assert result.ledger.quantity("X") == 0


def test_same_close_fill_cannot_earn_completed_price_jump():
    close = run_path([0.0, 1.0], [0.0, 1.0], targets_at="close")
    interval = run_path([0.0, 1.0], [0.0, 1.0], targets_at="interval_start")
    assert close.equity.iloc[-1] == pytest.approx(1.0)
    assert interval.equity.iloc[-1] == pytest.approx(2.0)
    assert close.orders.iloc[0].price == pytest.approx(2.0)


def test_cash_first_tax_payment_does_not_change_shares_or_basis():
    ledger = PortfolioLedger(LedgerConfig(interest_tax_rate=0.25), 2.0, {"X": 1.0})
    ledger.buy("X", 1.0, pd.Timestamp("2020-01-02"))
    ledger.credit_distribution("X", 0.5, pd.Timestamp("2020-06-01"))
    paid = ledger.settle_taxes(pd.Timestamp("2020-12-31"), close_year=2020)
    assert paid == pytest.approx(0.125)
    assert ledger.cash == pytest.approx(1.375)
    assert ledger.quantity("X") == pytest.approx(1.0)
    assert ledger.lots["X"][0].basis == pytest.approx(1.0)
    assert not any(e["reason"] == "tax_funding" for e in ledger.events)


def funded_gain_ledger(fee_bps=0):
    ledger = PortfolioLedger(LedgerConfig(short_tax_rate=0.25, fee_bps=fee_bps), 100.0, {"X": 10.0})
    ledger.buy("X", 100, pd.Timestamp("2020-01-02"))
    ledger.mark_to_market({"X": 20}, pd.Timestamp("2020-06-01"))
    ledger.sell("X", 5, pd.Timestamp("2020-06-01"))
    ledger.buy("X", ledger.cash, pd.Timestamp("2020-07-01"))
    return ledger


def test_tax_funding_sales_realize_gains_and_reconcile_same_year_liability():
    ledger = funded_gain_ledger()
    paid = ledger.settle_taxes(pd.Timestamp("2020-12-31"), close_year=2020)
    # Initial realized gain 50; tax sale S realizes 0.5*S: S=.25*(50+.5*S).
    assert paid == pytest.approx(100 / 7)
    assert ledger.tax_years[2020].realized_short == pytest.approx(400 / 7)
    assert ledger.quantity("X") == pytest.approx(10 - 5 / 7)
    assert ledger.cash == pytest.approx(0, abs=1e-10)
    assert sum(
        e["notional"]
        for e in ledger.events
        if e["reason"] == "tax_funding" and e["event"] == "sell"
    ) == pytest.approx(paid)


def test_next_year_funding_sales_belong_to_new_tax_year():
    ledger = funded_gain_ledger()
    paid = ledger.settle_taxes(pd.Timestamp("2021-01-04"), close_year=2020)
    assert paid == pytest.approx(12.5)
    assert ledger.tax_years[2020].realized_short == pytest.approx(50)
    assert ledger.tax_years[2021].realized_long == pytest.approx(6.25)


def test_close_mode_tax_funding_uses_current_executable_close_not_previous_close():
    result = run_path(
        [0, 1, 0, 1],
        [1, 0.5, 1, 1],
        dates=["2020-01-02", "2020-06-01", "2020-12-01", "2021-01-04"],
        config=LedgerConfig(short_tax_rate=0.25),
        targets_at="close",
        liquidate_at_end=False,
        finalize_tax_at_end=False,
    )
    sales = result.orders.loc[result.orders.reason.eq("tax_funding")]
    assert len(sales) == 1
    assert sales.iloc[0].price == pytest.approx(4)
    assert sales.iloc[0].notional == pytest.approx(0.125)
    assert result.taxes_paid.iloc[-1] == pytest.approx(0.125)


def test_tax_funding_sales_charge_costs():
    ledger = funded_gain_ledger(fee_bps=100)
    ledger.settle_taxes(pd.Timestamp("2021-01-04"), close_year=2020)
    sales = [e for e in ledger.events if e["event"] == "sell" and e["reason"] == "tax_funding"]
    assert sales and sum(e["fee"] for e in sales) > 0
    assert all(e["cash_flow"] == pytest.approx(e["notional"] - e["fee"]) for e in sales)


def test_fifo_distinguishes_short_and_long_holding_periods():
    ledger = PortfolioLedger(LedgerConfig(short_tax_rate=0.35, long_tax_rate=0.15), 100, {"X": 10})
    ledger.buy("X", 50, pd.Timestamp("2020-01-02"))
    ledger.buy("X", 50, pd.Timestamp("2021-01-03"))
    ledger.mark_to_market({"X": 20}, pd.Timestamp("2021-01-04"))
    ledger.sell("X", 8, pd.Timestamp("2021-01-04"))
    assert ledger.tax_years[2021].realized_long == pytest.approx(50)
    assert ledger.tax_years[2021].realized_short == pytest.approx(30)
    assert ledger.settle_taxes(pd.Timestamp("2021-12-31"), close_year=2021) == pytest.approx(18)
    assert ledger.lots["X"][0].quantity == pytest.approx(2)


def test_exact_holding_threshold_is_short_and_next_day_is_long():
    ledger = PortfolioLedger(LedgerConfig(), 100, {"X": 10})
    ledger.buy("X", 100, pd.Timestamp("2021-01-01"))
    ledger.mark_to_market({"X": 20}, pd.Timestamp("2022-01-01"))
    ledger.sell("X", 5, pd.Timestamp("2022-01-01"))
    ledger.sell("X", 5, pd.Timestamp("2022-01-02"))
    assert ledger.tax_years[2022].realized_short == pytest.approx(50)
    assert ledger.tax_years[2022].realized_long == pytest.approx(50)


def test_capital_cross_netting_and_character_carryforward():
    ledger = PortfolioLedger(LedgerConfig(short_tax_rate=0.25, long_tax_rate=0.15), 100)
    ledger._book(2020).realized_short = 10
    ledger._book(2020).realized_long = -30
    assert ledger.settle_taxes(pd.Timestamp("2021-01-04"), close_year=2020) == 0
    assert ledger.long_loss_carryforward == pytest.approx(-20)
    ledger._book(2021).realized_short = 50
    assert ledger.settle_taxes(pd.Timestamp("2022-01-03"), close_year=2021) == pytest.approx(7.5)
    assert ledger.long_loss_carryforward == 0


def test_wash_sale_subsequent_replacement_adds_basis_and_carries_holding_date():
    ledger = PortfolioLedger(LedgerConfig(wash_sales=True), 100, {"X": 10})
    ledger.buy("X", 100, pd.Timestamp("2020-01-01"))
    ledger.mark_to_market({"X": 8}, pd.Timestamp("2020-02-10"))
    ledger.sell("X", 10, pd.Timestamp("2020-02-10"))
    assert sum(p.quantity * p.loss_per_share for p in ledger.pending_wash_losses) == pytest.approx(
        20
    )
    ledger.buy("X", 80, pd.Timestamp("2020-02-20"))
    assert ledger.pending_wash_losses == []
    lot = ledger.lots["X"][0]
    assert lot.basis == pytest.approx(100)
    assert lot.acquired == pd.Timestamp("2020-02-20")
    # Tack 40 actually-held days; exclude the ten days out of the market.
    assert lot.holding_start == pd.Timestamp("2020-01-11")
    assert ledger.tax_years.get(2020) is None or ledger.tax_years[2020].realized_short == 0


def test_wash_sale_prior_partial_replacement_and_unmatched_loss_expiry():
    ledger = PortfolioLedger(LedgerConfig(wash_sales=True), 140, {"X": 10})
    ledger.buy("X", 100, pd.Timestamp("2020-01-01"))
    ledger.mark_to_market({"X": 8}, pd.Timestamp("2020-02-01"))
    ledger.buy("X", 40, pd.Timestamp("2020-02-01"))
    ledger.sell("X", 10, pd.Timestamp("2020-02-10"))
    assert ledger.quantity("X") == pytest.approx(5)
    assert ledger.lots["X"][0].basis == pytest.approx(50)
    assert ledger.pending_wash_losses[0].quantity == pytest.approx(5)
    ledger.expire_wash_losses(pd.Timestamp("2020-03-12"))
    assert ledger.tax_years[2020].realized_short == pytest.approx(-10)


def test_wash_holding_period_does_not_count_days_between_sale_and_repurchase():
    ledger = PortfolioLedger(LedgerConfig(wash_sales=True), 100, {"X": 10})
    ledger.buy("X", 100, pd.Timestamp("2021-01-01"))
    ledger.mark_to_market({"X": 8}, pd.Timestamp("2021-12-20"))
    ledger.sell("X", 10, pd.Timestamp("2021-12-20"))
    ledger.buy("X", 80, pd.Timestamp("2022-01-10"))
    ledger.mark_to_market({"X": 12}, pd.Timestamp("2022-01-15"))
    ledger.sell("X", 10, pd.Timestamp("2022-01-15"))
    # 353 original holding days + 5 replacement holding days is still short.
    assert ledger.tax_years[2022].realized_short == pytest.approx(20)
    assert ledger.tax_years[2022].realized_long == 0


def test_wash_sale_full_liquidation_does_not_match_shares_in_same_order():
    ledger = PortfolioLedger(LedgerConfig(wash_sales=True), 100, {"X": 10})
    ledger.buy("X", 50, pd.Timestamp("2020-01-01"))
    ledger.buy("X", 50, pd.Timestamp("2020-01-02"))
    ledger.mark_to_market({"X": 8}, pd.Timestamp("2020-01-10"))
    ledger.liquidate(pd.Timestamp("2020-01-10"))
    ledger.expire_wash_losses(pd.Timestamp("2020-01-10"), no_future_replacements=True)
    assert ledger.tax_years[2020].realized_short == pytest.approx(-20)
    assert not any(e["event"] == "wash_basis_adjustment" for e in ledger.events)


def test_cross_year_wash_expiry_explicitly_refunds_provisional_assessment():
    ledger = PortfolioLedger(LedgerConfig(wash_sales=True, short_tax_rate=0.25), 200, {"X": 10})
    ledger._book(2020).realized_short = 100
    ledger.buy("X", 100, pd.Timestamp("2020-11-01"))
    ledger.mark_to_market({"X": 8}, pd.Timestamp("2020-12-20"))
    ledger.sell("X", 10, pd.Timestamp("2020-12-20"))
    assert ledger.settle_taxes(pd.Timestamp("2021-01-04"), close_year=2020) == pytest.approx(25)
    ledger.expire_wash_losses(pd.Timestamp("2021-01-20"))
    assert ledger.settle_taxes(pd.Timestamp("2021-01-20")) == pytest.approx(-5)
    assert ledger.tax_years[2020].paid == pytest.approx(20)
    assert any(e["event"] == "tax_refund" and e["tax_year"] == 2020 for e in ledger.events)


def test_cross_year_replacement_does_not_release_previous_year_loss():
    ledger = PortfolioLedger(LedgerConfig(wash_sales=True, short_tax_rate=0.25), 200, {"X": 10})
    ledger._book(2020).realized_short = 100
    ledger.buy("X", 100, pd.Timestamp("2020-11-01"))
    ledger.mark_to_market({"X": 8}, pd.Timestamp("2020-12-20"))
    ledger.sell("X", 10, pd.Timestamp("2020-12-20"))
    ledger.settle_taxes(pd.Timestamp("2021-01-04"), close_year=2020)
    ledger.buy("X", 80, pd.Timestamp("2021-01-05"))
    ledger.expire_wash_losses(pd.Timestamp("2021-02-01"))
    assert ledger.settle_taxes(pd.Timestamp("2021-02-01")) == 0
    assert ledger.tax_years[2020].paid == pytest.approx(25)
    assert ledger.lots["X"][0].basis == pytest.approx(100)


def test_split_preserves_wealth_and_basis_distribution_is_explicit_cash():
    ledger = PortfolioLedger(LedgerConfig(), 100, {"X": 10})
    ledger.buy("X", 100, pd.Timestamp("2020-01-01"))
    ledger.apply_split("X", 2, pd.Timestamp("2020-01-10"))
    assert ledger.quantity("X") == 20
    assert ledger.prices["X"] == 5
    assert ledger.equity == pytest.approx(100)
    assert sum(lot.basis for lot in ledger.lots["X"]) == pytest.approx(100)
    ledger.mark_to_market({"X": 4.5}, pd.Timestamp("2020-02-01"))
    ledger.credit_distribution("X", 0.5, pd.Timestamp("2020-02-01"))
    assert ledger.cash == pytest.approx(10)
    assert ledger.equity == pytest.approx(100)
    assert ledger.tax_years[2020].ordinary_income == pytest.approx(10)


def test_return_of_capital_cannot_make_basis_negative():
    ledger = PortfolioLedger(LedgerConfig(), 100, {"X": 10})
    ledger.buy("X", 100, pd.Timestamp("2020-01-01"))
    ledger.credit_distribution("X", 12, pd.Timestamp("2020-02-01"), tax_kind="return_of_capital")
    assert ledger.lots["X"][0].basis == 0
    assert ledger.tax_years[2020].realized_short == pytest.approx(20)


def test_partial_target_drift_is_visible_not_presented_as_a_hard_cap():
    result = run_path(
        [0, 1],
        [1 / 3, 1 / 3],
        liquidate_at_end=False,
        config=LedgerConfig(exposure_multipliers={"X": 3}),
    )
    row = result.holdings.iloc[-1]
    assert row.actual_weight == pytest.approx(0.5)
    assert row.effective_exposure == pytest.approx(1.5)
    assert row.target_weight == pytest.approx(1 / 3)


def test_every_bar_close_rebalance_restores_target_but_pays_turnover():
    result = run_path(
        [0, 1],
        [1 / 3, 1 / 3],
        liquidate_at_end=False,
        targets_at="close",
        config=LedgerConfig(rebalance_policy="every_bar"),
    )
    assert result.holdings.iloc[-1].actual_weight == pytest.approx(1 / 3)
    assert result.turnover.iloc[-1] > 0


def test_persistent_fold_state_matches_single_run_across_year_boundary():
    idx = pd.to_datetime(["2020-01-02", "2020-06-01", "2020-12-31", "2021-01-04", "2021-06-01"])
    returns = pd.DataFrame({"X": [0.1, 0.1, -0.05, 0.02, 0.04]}, index=idx)
    targets = pd.DataFrame({"X": [1, 0.5, 1, 1, 0]}, index=idx)
    cash = pd.Series(0.001, index=idx)
    cfg = LedgerConfig(
        fee_bps=1,
        slippage_bps=5,
        short_tax_rate=0.3,
        long_tax_rate=0.15,
        interest_tax_rate=0.3,
        wash_sales=True,
    )
    full = simulate_ledger(returns, targets, cash, cfg)
    first = simulate_ledger(
        returns.iloc[:3],
        targets.iloc[:3],
        cash.iloc[:3],
        cfg,
        liquidate_at_end=False,
        finalize_tax_at_end=False,
    )
    assert first.taxes_paid.sum() == 0
    second = simulate_ledger(
        returns.iloc[3:], targets.iloc[3:], cash.iloc[3:], cfg, ledger=first.ledger
    )
    pd.testing.assert_series_equal(pd.concat([first.returns, second.returns]), full.returns)
    pd.testing.assert_series_equal(
        pd.concat([first.taxes_paid, second.taxes_paid]), full.taxes_paid
    )
    assert second.ledger.cash == pytest.approx(full.ledger.cash)


def test_initial_price_units_and_explicit_interval_start_prices():
    result = run_path([0, 0], [1, 1], initial_prices={"X": 100}, liquidate_at_end=False)
    assert result.ledger.quantity("X") == pytest.approx(0.01)
    starts = pd.DataFrame({"X": [100, 110]}, index=result.returns.index)
    marked = run_path([0, 0], [1, 1], initial_prices={"X": 100}, start_prices=starts)
    assert marked.equity.iloc[-1] == pytest.approx(1.1)


def test_random_multiasset_cash_and_order_reconciliation():
    rng = np.random.default_rng(4)
    idx = pd.date_range("2020-01-02", periods=90, freq="D")
    returns = pd.DataFrame(rng.normal(0, 0.02, (90, 2)), index=idx, columns=["X", "Y"])
    weights = pd.DataFrame(rng.dirichlet([1, 1, 1], 90)[:, :2], index=idx, columns=["X", "Y"])
    result = simulate_ledger(
        returns, weights, pd.Series(0, index=idx), LedgerConfig(fee_bps=5, slippage_bps=10)
    )
    assert result.cash.ge(-1e-10).all()
    assert result.holdings.quantity.ge(0).all()
    assert result.ledger.quantity("X") == result.ledger.quantity("Y") == 0
    assert 1 + result.events.cash_flow.sum() == pytest.approx(result.cash.iloc[-1])
    assert (1 + result.returns).prod() == pytest.approx(result.equity.iloc[-1])


@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.1])
def test_invalid_targets_rejected_not_clipped(bad):
    with pytest.raises(ValueError, match="targets"):
        run_path([0, 0], [0, bad])


def test_missing_data_and_replayed_fold_rejected():
    with pytest.raises(ValueError, match="asset returns"):
        run_path([0, np.nan], [0, 1])
    first = run_path([0, 0], [1, 1], liquidate_at_end=False, finalize_tax_at_end=False)
    with pytest.raises(ValueError, match="after the previous"):
        run_path([0, 0], [1, 1], ledger=first.ledger)


def test_dynamic_cash_wrapper_delegates_and_partial_fold_does_not_finalize_tax():
    from trend_following.accounting import simulate_after_tax_portfolio_with_cash_returns

    idx = pd.to_datetime(["2020-01-02", "2020-06-01"])
    r = pd.DataFrame({"X": [0.1, 0.1]}, index=idx)
    w = pd.DataFrame({"X": [1, 0]}, index=idx)
    ledger = PortfolioLedger(
        LedgerConfig(short_tax_rate=0.25, long_tax_rate=0.25, interest_tax_rate=0.25)
    )
    _, taxes, _, _, _ = simulate_after_tax_portfolio_with_cash_returns(
        r,
        w,
        cash_returns=pd.Series(0, index=idx),
        transaction_cost_bps=0,
        slippage_bps=0,
        tax_rate=0.25,
        liquidate_at_end=False,
        ledger=ledger,
    )
    assert taxes.sum() == 0
    assert ledger.tax_years[2020].realized_short == pytest.approx(0.1)


def test_simulated_split_preserves_wealth_without_double_applying_raw_price_drop():
    # Raw prices 100 -> 50 and split2: root must supply (50/100)*2 - 1 = 0.
    actions = pd.DataFrame([dict(timestamp="2020-01-03", asset="X", split_ratio=2)])
    result = run_path(
        [0, 0],
        [1, 1],
        initial_cash=100,
        initial_prices={"X": 100},
        corporate_actions=actions,
        price_basis="unadjusted",
        targets_at="close",
        liquidate_at_end=False,
    )
    assert result.ledger.equity == pytest.approx(100)
    assert result.ledger.prices["X"] == pytest.approx(50)
    assert result.ledger.quantity("X") == pytest.approx(2)
    assert sum(lot.basis for lot in result.ledger.lots["X"]) == pytest.approx(100)


def test_simulated_cash_distribution_offsets_exdate_price_drop_and_uses_prior_holdings():
    actions = pd.DataFrame(
        [dict(timestamp="2020-01-03", asset="X", distribution_per_share=1, tax_kind="ordinary")]
    )
    result = run_path(
        [0, -0.01],
        [1, 1],
        initial_cash=100,
        initial_prices={"X": 100},
        corporate_actions=actions,
        price_basis="unadjusted",
        targets_at="close",
        liquidate_at_end=False,
    )
    assert result.ledger.prices["X"] == pytest.approx(99)
    assert result.ledger.quantity("X") == pytest.approx(1)
    assert result.ledger.cash == pytest.approx(1)
    assert result.ledger.equity == pytest.approx(100)
    assert result.ledger.tax_years[2020].ordinary_income == pytest.approx(1)
    # New ex-date buyers get no distribution on shares they did not yet hold.
    late = run_path(
        [0, -0.01],
        [0, 1],
        initial_cash=100,
        initial_prices={"X": 100},
        corporate_actions=actions,
        price_basis="unadjusted",
        targets_at="close",
    )
    assert late.events.loc[late.events.event.eq("distribution"), "cash_flow"].sum() == 0


def test_corporate_actions_reject_total_return_double_counting_and_missing_alignment():
    action = pd.DataFrame([dict(timestamp="2020-01-03", asset="X", split_ratio=2)])
    with pytest.raises(ValueError, match="avoid double counting"):
        run_path([0, 0], [1, 1], corporate_actions=action)
    action.loc[0, "timestamp"] = "2020-01-04"
    with pytest.raises(ValueError, match="timestamps must belong"):
        run_path([0, 0], [1, 1], corporate_actions=action, price_basis="unadjusted")
