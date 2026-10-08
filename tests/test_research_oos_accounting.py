"""Hand-calculated artificial examples, not market/performance evidence."""

import copy

import numpy as np
import pandas as pd
import pytest

from trend_following.research_execution import ExecutionPolicyEngine
from trend_following.research_ledger import LedgerConfig
from trend_following.research_oos_accounting import (
    apply_execution_split,
    causal_split_adjusted_prices,
    normalize_oos_actions,
    simulate_oos_ledger,
    terminal_snapshot,
)


def fixture(prices, weights=None, dates=None):
    dates = pd.DatetimeIndex(
        dates or [f"2027-01-{d:02d}T15:30Z" for d in range(4, 4 + len(prices))]
    )
    close = pd.DataFrame({"TQQQ": prices}, index=dates)
    targets = pd.DataFrame({"TQQQ": weights or [1.0] * len(prices)}, index=dates)
    return close, targets, pd.Series(0.0, index=dates)


def action(
    kind="distribution",
    *,
    day="2027-01-05",
    amount=1.0,
    pay="2027-01-06T14:30Z",
    ratio=2.0,
    character="ordinary",
    action_id="a",
):
    effective = f"{day}T14:30Z"
    return dict(
        action_id=action_id,
        asset="TQQQ",
        kind=kind,
        effective_at=effective,
        available_at=effective,
        availability_basis="assumed_public_by_ex_date",
        source_observed_at="2027-02-01T21:00Z",
        split_ratio=ratio,
        amount_per_share=amount,
        pay_at=pay,
        character=character,
    )


def run(data, actions=None, config=None, **kwargs):
    return simulate_oos_ledger(
        *data, config or LedgerConfig(), corporate_actions=actions, initial_cash=100.0, **kwargs
    )


def test_split_preserves_wealth_total_basis_and_raw_quantities():
    data = fixture([100, 50, 55])
    result = run(data, [action("split")])
    assert result.equity.tolist() == pytest.approx([100, 100, 110])
    assert result.holdings.quantity.tolist() == pytest.approx([1, 2, 2])
    assert result.holdings.basis.tolist() == pytest.approx([100, 100, 100])
    assert result.ledger.lots["TQQQ"][0].basis_per_share == 50
    assert result.returns.iloc[1] == pytest.approx(0)
    normalized = causal_split_adjusted_prices(data[0], [action("split")])
    assert normalized.TQQQ.tolist() == [100, 100, 110]


def test_distribution_is_receivable_until_pay_date_and_earns_no_cash_interest():
    data = fixture([100, 99, 99, 99])
    data[2].iloc[:] = 0.1
    result = run(data, [action()])
    assert result.equity.tolist() == pytest.approx([100, 100, 100, 100.1])
    assert result.cash.tolist() == pytest.approx([0, 0, 1, 1.1])
    assert result.holdings.receivables.tolist() == pytest.approx([0, 1, 0, 0])
    assert result.cash_interest.tolist() == pytest.approx([0, 0, 0, 0.1])
    assert result.ledger.tax_years[2027].ordinary_income == pytest.approx(1.1)
    assert result.events.query("event == 'distribution_entitlement'").cash_flow.iloc[0] == 0
    assert result.events.query("event == 'distribution_payment'").cash_flow.iloc[0] == 1


def test_same_close_new_buyer_gets_no_distribution():
    result = run(fixture([100, 99, 99], [0, 1, 1]), [action()])
    assert result.holdings.receivables.eq(0).all()
    assert result.ledger.quantity("TQQQ") == pytest.approx(100 / 99)
    assert result.equity.tolist() == pytest.approx([100, 100, 100])


def test_exdate_sale_keeps_old_holder_payment_entitlement():
    result = run(fixture([100, 99, 99], [1, 0, 0]), [action()])
    assert result.cash.tolist() == pytest.approx([0, 99, 100])
    assert result.holdings.receivables.tolist() == pytest.approx([0, 1, 0])
    assert result.ledger.quantity("TQQQ") == 0
    assert result.ledger.tax_years[2027].realized_short == pytest.approx(-1)


def test_receivable_cannot_fund_restore_and_payment_is_not_automatic_reinvestment():
    result = run(
        fixture([100, 90, 90, 90], [1, 0.5, 1, 1]), [action(amount=10, pay="2027-01-07T14:30Z")]
    )
    assert result.equity.tolist() == pytest.approx([100] * 4)
    assert result.cash.tolist() == pytest.approx([0, 40, 0, 10])
    assert result.holdings.quantity.iloc[2] == pytest.approx(1)
    assert result.holdings.quantity.iloc[3] == pytest.approx(1)


def test_same_day_split_then_post_split_distribution_units():
    actions = [action(amount=0.5), action("split", action_id="s")]
    result = run(fixture([100, 49.5, 49.5]), actions)
    assert result.equity.tolist() == pytest.approx([100, 100, 100])
    assert result.holdings.quantity.tolist() == pytest.approx([1, 2, 2])
    assert result.cash.iloc[-1] == pytest.approx(1)
    assert result.holdings.basis.eq(100).all()


def test_later_split_does_not_multiply_existing_receivable():
    actions = [action(pay="2027-01-07T14:30Z"), action("split", day="2027-01-06", action_id="s")]
    result = run(fixture([100, 99, 49.5, 49.5]), actions)
    assert result.equity.tolist() == pytest.approx([100] * 4)
    assert result.holdings.receivables.tolist() == pytest.approx([0, 1, 1, 0])
    assert result.cash.iloc[-1] == 1


def test_entry_and_cloned_exit_one_percent_cost_hand_calculation():
    result = run(fixture([100, 100]), config=LedgerConfig(fee_bps=100))
    original = copy.deepcopy(vars(result.ledger))
    terminal = terminal_snapshot(result.ledger, result.equity.index[-1])
    assert terminal["marked_equity"] == pytest.approx(100 / 1.01)
    assert terminal["liquidated_equity"] == pytest.approx(100 / 1.01 * 0.99)
    assert terminal["exit_fees"] == pytest.approx(100 / 1.01 * 0.01)
    assert terminal["ledger"].quantity("TQQQ") == 0
    assert vars(result.ledger) == original
    assert result.ledger.quantity("TQQQ") > 0
    with pytest.raises(ValueError, match="second time"):
        terminal_snapshot(terminal["ledger"], result.equity.index[-1])


def test_unknown_character_keeps_pretax_valid_and_terminal_tax_explicitly_provisional():
    data = fixture([100, 99], dates=["2026-12-30T15:30Z", "2026-12-31T15:30Z"])
    a = action(day="2026-12-31", pay="2027-01-06T14:30Z", character="unknown")
    config = LedgerConfig(short_tax_rate=0.24, long_tax_rate=0.15, interest_tax_rate=0.24)
    result = run(data, [a], config)
    original = copy.deepcopy(vars(result.ledger))
    terminal = terminal_snapshot(result.ledger, data[0].index[-1])
    assert result.equity.iloc[-1] == 100
    assert terminal["tax_provisional"] is True
    assert terminal["receivables"] == 1
    assert terminal["unpaid_distribution_tax_reserve"] == pytest.approx(0.24)
    assert terminal["liquidated_equity"] == pytest.approx(99.76)
    assert terminal["ledger"].cash == 99  # future receipt was NOT forced into cash
    assert terminal["terminal_taxes_paid"] == 0
    assert vars(result.ledger) == original
    pretax = terminal_snapshot(result.ledger, data[0].index[-1], settle_tax=False)
    assert pretax["liquidated_equity"] == 100


@pytest.mark.parametrize(
    "character,book_field,rate",
    [
        ("ordinary", "ordinary_income", 0.24),
        ("unknown", "ordinary_income", 0.24),
        ("qualified", "qualified_income", 0.15),
        ("long_capital_gain", "realized_long", 0.15),
    ],
)
def test_payment_tax_character_scenarios(character, book_field, rate):
    result = run(
        fixture([100, 99, 99]),
        [action(character=character)],
        LedgerConfig(short_tax_rate=0.24, long_tax_rate=0.15, interest_tax_rate=0.24),
    )
    assert getattr(result.ledger.tax_years[2027], book_field) == 1
    # Before liquidation the receipt is ordinary/qualified/long capital income;
    # reserve evaluation after sale may net long distributions with short losses.
    if character != "long_capital_gain":
        terminal = terminal_snapshot(result.ledger, result.equity.index[-1])
        assert terminal["terminal_taxes_paid"] == pytest.approx(rate)
        assert terminal["liquidated_equity"] == pytest.approx(100 - rate)
    assert result.ledger.tax_provisional is (character == "unknown")


def test_continuous_chunks_preserve_entitlement_and_do_not_duplicate_same_date_action():
    data = fixture(
        [100, 99, 99, 99],
        dates=["2027-01-04T15:30Z", "2027-01-05T15:30Z", "2027-01-05T16:30Z", "2027-01-06T15:30Z"],
    )
    actions = [action()]
    full = run(data, actions)
    first = run(tuple(x.iloc[:2] for x in data), actions)
    rest = run(tuple(x.iloc[2:] for x in data), actions, ledger=first.ledger)
    assert pd.concat([first.equity, rest.equity]).tolist() == full.equity.tolist()
    assert rest.ledger.cash == full.ledger.cash == 1
    assert len([e for e in rest.ledger.events if e["event"] == "distribution_entitlement"]) == 1


def test_terminal_clone_cannot_replace_continuous_account():
    data = fixture([100, 100, 100])
    first = run(tuple(x.iloc[:2] for x in data))
    terminal = terminal_snapshot(first.ledger, first.equity.index[-1])
    with pytest.raises(ValueError, match="terminal-clone"):
        run(tuple(x.iloc[2:] for x in data), ledger=terminal["ledger"])
    with pytest.raises(ValueError, match="actual last"):
        terminal_snapshot(first.ledger, data[0].index[-1])


def test_raw_policy_reference_split_adjustment_preserves_gains_and_peak_stop():
    engine = ExecutionPolicyEngine(
        {"kind": "b5", "use_qtrim": False, "use_profit_lock": False, "use_bear_filter": False}
    )
    engine.weight = engine.gross_holding = 1.0
    engine.gross_cash = 0
    engine.base_long = True
    engine.entry_price, engine.peak_price, engine.previous_risk_price = 100, 120, 100
    apply_execution_split(engine, 2)
    features = dict(
        entry=False,
        exit=False,
        vol_percentile=0,
        distance=0,
        long_ma=100,
        long_slope=1,
        medium_slope=1,
        short_ma=100,
        medium_ma=100,
    )
    decision = engine.step(pd.Timestamp("2027-01-05T15:30Z"), 100, 50, features)
    assert engine.entry_price == 50 and engine.peak_price == 60
    assert engine.gross_holding == 1
    assert not engine.stopped
    assert decision["weights_at_close"] == 1


def test_appending_future_split_cannot_rewrite_earlier_policy_prices():
    data = fixture([100, 110, 55])
    actions = [action("split", day="2027-01-06")]
    prefix = causal_split_adjusted_prices(data[0].iloc[:2], actions)
    extended = causal_split_adjusted_prices(data[0], actions)
    pd.testing.assert_frame_equal(prefix, extended.iloc[:2])
    assert extended.TQQQ.tolist() == [100, 110, 110]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"available_at": "2027-01-05T16:30Z"}, "unavailable"),
        ({"pay_at": None}, "pay_at"),
        ({"pay_at": "2027-01-04T14:30Z"}, "precede"),
        ({"character": "return_of_capital"}, "unsupported"),
        ({"amount_per_share": np.nan}, "positive"),
        ({"amount_per_share": True}, "positive"),
        ({"availability_basis": "invented"}, "availability_basis"),
    ],
)
def test_action_errors_fail_explicitly(change, match):
    a = action()
    a.update(change)
    with pytest.raises(ValueError, match=match):
        run(fixture([100, 99, 99]), [a])


def test_duplicate_actions_and_missing_exdate_observation_fail():
    with pytest.raises(ValueError, match="duplicate"):
        normalize_oos_actions([action(), action()])
    with pytest.raises(ValueError, match="missing ex-date"):
        run(fixture([100, 99], dates=["2027-01-04T15:30Z", "2027-01-06T15:30Z"]), [action()])


def test_zero_target_account_has_well_formed_empty_orders():
    result = run(fixture([100, 100], [0, 0]))
    assert result.orders.empty
    assert result.equity.eq(100).all()


def test_unpaid_receivable_cannot_fund_annual_tax_payment():
    data = fixture(
        [100, 100, 50],
        [0.5, 1, 1],
        dates=["2026-12-30T15:30Z", "2026-12-31T15:30Z", "2027-01-04T15:30Z"],
    )
    data[2].iloc[1] = 0.1  # $50 cash earns $5 before the year-end allocation.
    a = action(day="2027-01-04", amount=50, pay="2027-01-11T14:30Z")
    result = run(data, [a], LedgerConfig(short_tax_rate=0.24, interest_tax_rate=0.24))
    assert result.taxes_paid.iloc[-1] == pytest.approx(1.2)
    assert result.holdings.receivables.iloc[-1] == pytest.approx(52.5)
    assert result.ledger.quantity("TQQQ") == pytest.approx(1.05 - 1.2 / 50)
    assert result.equity.iloc[-1] == pytest.approx(103.8)
    funding = result.events.query("event == 'sell' and reason == 'tax_funding'")
    assert len(funding) == 1 and funding.price.iloc[0] == 50


def test_split_rescales_pending_wash_replacement_units_without_losing_basis():
    result = run(
        fixture([100, 80, 40, 40], [1, 0, 0, 1]),
        [action("split", day="2027-01-06")],
        LedgerConfig(wash_sales=True),
    )
    assert result.ledger.quantity("TQQQ") == pytest.approx(2)
    assert sum(lot.basis for lot in result.ledger.lots["TQQQ"]) == pytest.approx(100)
    assert not result.ledger.pending_wash_losses
    assert result.equity.iloc[-1] == 80


def test_multiple_distribution_components_share_exdate_without_being_duplicated():
    actions = [
        action(amount=1, action_id="ordinary"),
        action(amount=2, character="long_capital_gain", action_id="capital"),
    ]
    result = run(fixture([100, 97, 97]), actions)
    assert result.holdings.receivables.tolist() == [0, 3, 0]
    assert result.cash.iloc[-1] == 3
    assert result.equity.tolist() == pytest.approx([100, 100, 100])
    assert result.ledger.tax_years[2027].ordinary_income == 1
    assert result.ledger.tax_years[2027].realized_long == 2
