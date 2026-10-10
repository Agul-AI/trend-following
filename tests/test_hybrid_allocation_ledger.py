"""Independent arithmetic checks for actual partial-target order accounting."""

import numpy as np
import pandas as pd
import pytest

from trend_following.research_hybrid_allocation_ledger import TargetBook


def instant(i):
    return pd.Timestamp("2001-01-02 15:00", tz="UTC") + pd.Timedelta(minutes=30 * i)


def order(book, target, price, i):
    return book.execute([target], [True], price, instant(i))


def test_partial_buy_and_sell_hit_post_cost_target_with_independent_cash_equations():
    book = TargetBook(["one"], ["B070"], commission_bps=1, slippage_bps=3)
    expected_q = 700 / (100 * (1 + 0.7 * ((1.0003 * 1.0001) - 1)))
    order(book, 0.7, 100, 0)
    assert book.shares[0] == pytest.approx(expected_q)
    assert book.cash[0] == pytest.approx(1000 - expected_q * 100.03 * 1.0001)
    assert book.shares[0] * 100 / book.wealth(100)[0] == pytest.approx(0.7)
    pre_q, pre_cash = book.shares[0], book.cash[0]
    pre_e = pre_cash + pre_q * 120
    sold = (pre_q * 120 - 0.5 * pre_e) / (120 * (1 - 0.5 * (1 - 0.9997 * 0.9999)))
    order(book, 0.5, 120, 1)
    assert book.shares[0] == pytest.approx(pre_q - sold)
    assert book.cash[0] == pytest.approx(pre_cash + sold * 119.964 * 0.9999)
    assert book.shares[0] * 120 / book.wealth(120)[0] == pytest.approx(0.5)
    assert book.orders[0] == 2 and book.trips[0] == 0
    order(book, 0, 110, 2)
    assert book.trips[0] == 1 and book.shares[0] == 0
    assert book.holds[0] == 1


def test_zero_cost_replay_uses_same_target_commands_but_own_quantities():
    net = TargetBook(["one"], ["B050"], commission_bps=1, slippage_bps=25)
    gross = TargetBook(["one"], ["B050"], commission_bps=0, slippage_bps=0)
    for i, (target, price) in enumerate([(0.5, 100), (1, 120), (0.5, 80), (0, 90)]):
        order(net, target, price, i)
        order(gross, target, price, i)
    assert [r["target_fraction"] for r in net.command_rows] == [
        r["target_fraction"] for r in gross.command_rows
    ]
    assert net.trade_rows[0]["quantity"] != gross.trade_rows[0]["quantity"]
    assert gross.cash[0] > net.cash[0]
    assert gross.cash[0] - net.cash[0] != pytest.approx(net.commission[0] + net.slippage[0])


def test_unchanged_target_without_command_does_not_rebalance_price_drift():
    book = TargetBook(["one"], ["B050"])
    order(book, 0.5, 100, 0)
    q, cash = book.shares.copy(), book.cash.copy()
    book.execute([0.5], [False], 200, instant(1))
    np.testing.assert_array_equal(book.shares, q)
    np.testing.assert_array_equal(book.cash, cash)
    assert book.orders[0] == 1
    assert book.shares[0] * 200 / book.wealth(200)[0] > 0.6


def test_split_dividend_and_cost_free_drip_preserve_partial_account_equity():
    book = TargetBook(["one"], ["B050"], commission_bps=0, slippage_bps=0)
    order(book, 0.5, 100, 0)
    book.actions(2, 0.5)
    before = book.wealth(49.5)[0]
    assert before == 1000
    book.drip(49.5)
    assert book.wealth(49.5)[0] == before
    assert book.shares[0] == pytest.approx(10 + 5 / 49.5)
    assert book.orders[0] == 1 and book.commission[0] == 0


@pytest.mark.parametrize("target", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_target_cannot_borrow_or_short(target):
    book = TargetBook(["one"], ["B050"])
    with pytest.raises(ValueError, match="unlevered"):
        order(book, target, 100, 0)
