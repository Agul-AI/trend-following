"""Prospective recorder allocation/corporate-action fixtures, never market records."""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from test_research_forward import append, event, sha
from test_research_forward import setup as recorder_setup
from trend_following import research_forward as forward

setup = recorder_setup


def buy_five(env):
    env["clock"][0] = pd.Timestamp("2026-10-02T14:00:00Z")
    append(env, event(env))
    env["clock"][0] += pd.Timedelta(seconds=1)
    order = event(env, "hypothetical_order", "o1", "2026-10-02T14:00:01Z")
    order.update(
        signal_id="s1",
        requested_quantities={"QQQ": 5.0},
        observation_timestamp="2026-10-02T14:00:00Z",
        decision_timestamp=order["event_timestamp"],
        order_timestamp=order["event_timestamp"],
    )
    append(env, order)
    env["clock"][0] = pd.Timestamp("2026-10-02T15:00:00Z")
    fill = event(env, "fill", "f1", "2026-10-02T15:00:00Z")
    fill.update(
        order_id="o1",
        observation_timestamp=order["observation_timestamp"],
        decision_timestamp=order["decision_timestamp"],
        order_timestamp=order["order_timestamp"],
        fill_timestamp=fill["event_timestamp"],
        fills=[dict(symbol="QQQ", quantity=5.0, price=100.0)],
    )
    fill["state"] = dict(cash=500.0, positions={"QQQ": 5.0}, prices={"QQQ": 100.0}, equity=1000.0)
    append(env, fill)


def corporate(
    env, *, ratio=2.0, dividend=0.0, quote=50.0, time="2026-10-05T14:00:00Z", identifier="split1"
):
    env["clock"][0] = pd.Timestamp(time)
    head = forward.verify_forward_ledger(env["root"], env["manifest"], env["ledger"])
    old = head["state"]
    qty = old["positions"].get("QQQ", 0.0)
    action = dict(
        action_id=identifier,
        symbol="QQQ",
        effective_timestamp=time,
        split_ratio=ratio,
        distribution_per_share=dividend,
        distribution_unit="post_split_share",
        cash_timing="first_observed_ex_date_bar_hypothetical",
    )
    snapshot = env["root"] / f"action_{identifier}.json"
    snapshot.write_text(
        json.dumps(dict(price_timestamp=time, prices={"QQQ": quote}, corporate_actions=[action]))
    )
    output = event(env, "corporate_action", f"ca_{identifier}", time)
    output["source"] = dict(
        provider="fixture_not_market_data",
        path=str(snapshot),
        sha256=sha(snapshot),
        retrieved_at=time,
    )
    credit = qty * ratio * dividend
    output["corporate_action"] = dict(
        action,
        entitled_pre_split_quantity=qty,
        post_split_reference_price=old["prices"].get("QQQ", 0.0) / ratio
        if "QQQ" in old["prices"]
        else None,
        cash_credit=credit,
    )
    cash = old["cash"] + credit
    output["state"] = dict(
        cash=cash,
        positions={"QQQ": qty * ratio},
        prices={"QQQ": quote},
        equity=cash + qty * ratio * quote,
    )
    return output


def test_corporate_split_preserves_units_and_value_without_a_trade(setup):
    env = setup
    buy_five(env)
    action = corporate(env)
    append(env, action)
    head = forward.verify_forward_ledger(env["root"], env["manifest"], env["ledger"])
    assert head["state"] == dict(
        cash=500.0, positions={"QQQ": 10.0}, prices={"QQQ": 50.0}, equity=1000.0
    )
    assert not action.get("fills") and action["corporate_action"]["cash_credit"] == 0


def test_distribution_uses_prior_holder_shares_and_is_not_interest(setup):
    env = setup
    buy_five(env)
    action = corporate(env, ratio=1, dividend=1, quote=99)
    append(env, action)
    assert action["state"]["cash"] == 505
    assert action["state"]["equity"] == 1000
    assert not action.get("cash_interest")


def test_combined_split_dividend_uses_post_split_units(setup):
    env = setup
    buy_five(env)
    action = corporate(env, ratio=2, dividend=0.5, quote=49.5)
    append(env, action)
    assert action["corporate_action"]["cash_credit"] == 5
    assert action["state"]["equity"] == 1000


def test_corporate_declared_amount_must_match_hash_bound_source(setup):
    env = setup
    buy_five(env)
    action = corporate(env)
    action["corporate_action"]["split_ratio"] = 3
    with pytest.raises(ValueError, match="differs from its source"):
        append(env, action)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("entitled_pre_split_quantity", 6, "prior-holder"),
        ("post_split_reference_price", 100, "Mechanical split"),
        ("cash_credit", 10, "cash credit"),
    ],
)
def test_corporate_inventory_and_cash_entitlement_cannot_be_invented(setup, field, value, match):
    env = setup
    buy_five(env)
    action = corporate(env)
    action["corporate_action"][field] = value
    with pytest.raises(ValueError, match=match):
        append(env, action)


def test_corporate_quote_must_match_source_not_just_equity_arithmetic(setup):
    env = setup
    buy_five(env)
    action = corporate(env)
    action["state"]["prices"]["QQQ"] = 51
    action["state"]["equity"] = 1010
    with pytest.raises(ValueError, match="source-bound post-action quote"):
        append(env, action)


def test_duplicate_corporate_action_and_late_same_session_entitlement_rejected(setup):
    env = setup
    buy_five(env)
    late = corporate(env, time="2026-10-02T16:00:00Z")
    with pytest.raises(ValueError, match="before same-session fills"):
        append(env, late)
    action = corporate(env)
    append(env, action)
    duplicate = copy.deepcopy(action)
    duplicate["event_id"] = "different_event_id"
    duplicate["expected_previous_hash"] = forward.verify_forward_ledger(
        env["root"], env["manifest"], env["ledger"]
    )["head_hash"]
    with pytest.raises(ValueError, match="Duplicate corporate action"):
        append(env, duplicate)


def test_corporate_action_cannot_be_labeled_as_interest_or_a_fill(setup):
    env = setup
    buy_five(env)
    action = corporate(env)
    action["cash_interest"] = 1
    with pytest.raises(ValueError, match="disguise"):
        append(env, action)
    action.pop("cash_interest")
    action["fills"] = [dict(symbol="QQQ", quantity=5, price=50)]
    with pytest.raises(ValueError, match="Non-fill"):
        append(env, action)


def allocation_order(env, *, weight=1.0, fee=10.0, slip=20.0):
    env["clock"][0] = pd.Timestamp("2026-10-02T14:00:00Z")
    append(env, event(env))
    env["clock"][0] += pd.Timedelta(seconds=1)
    order = event(env, "hypothetical_order", "o1", "2026-10-02T14:00:01Z")
    order.update(
        signal_id="s1",
        observation_timestamp="2026-10-02T14:00:00Z",
        decision_timestamp=order["event_timestamp"],
        order_timestamp=order["event_timestamp"],
        allocation_order=dict(
            asset="QQQ",
            target_weight=weight,
            fee_bps=fee,
            slippage_bps=slip,
            sizing_convention="mark_notional_then_cash_budget",
        ),
    )
    append(env, order)
    return order


def allocation_fill(env, order, *, quote=200):
    env["clock"][0] = pd.Timestamp("2026-10-02T15:00:00Z")
    allocation = order["allocation_order"]
    n = min(
        1000 * allocation["target_weight"],
        1000 / (1 + (allocation["fee_bps"] + allocation["slippage_bps"]) / 10_000),
    )
    fee, slip = n * allocation["fee_bps"] / 10_000, n * allocation["slippage_bps"] / 10_000
    fill = event(env, "fill", "f1", "2026-10-02T15:00:00Z")
    fill.update(
        order_id="o1",
        observation_timestamp=order["observation_timestamp"],
        decision_timestamp=order["decision_timestamp"],
        order_timestamp=order["order_timestamp"],
        fill_timestamp=fill["event_timestamp"],
        fills=[dict(symbol="QQQ", quantity=n / quote, price=quote)],
    )
    fill["costs"].update(transaction=fee, slippage=slip)
    cash = max(1000 - n - fee - slip, 0.0)
    fill["state"] = dict(
        cash=cash, positions={"QQQ": n / quote}, prices={"QQQ": quote}, equity=cash + n
    )
    return fill


def test_allocation_order_sizes_at_later_quote_with_cash_inclusive_costs(setup):
    env = setup
    order = allocation_order(env)
    fill = allocation_fill(env, order, quote=200)
    append(env, fill)
    assert fill["fills"][0]["quantity"] == pytest.approx((1000 / 1.003) / 200)
    assert fill["state"]["cash"] == pytest.approx(0, abs=1e-10)


def test_allocation_partial_target_only_spends_target_notional_plus_costs(setup):
    env = setup
    order = allocation_order(env, weight=0.5)
    fill = allocation_fill(env, order, quote=200)
    append(env, fill)
    assert fill["state"]["cash"] == pytest.approx(498.5)
    assert fill["fills"][0]["quantity"] == pytest.approx(2.5)


def test_allocation_order_rejects_future_quote_sized_stale_quantity(setup):
    env = setup
    order = allocation_order(env)
    fill = allocation_fill(env, order)
    fill["fills"][0]["quantity"] /= 2
    with pytest.raises(ValueError, match="quantities differ"):
        append(env, fill)


def test_allocation_order_rejects_undeclared_costs_or_mismatched_mark(setup):
    env = setup
    order = allocation_order(env)
    fill = allocation_fill(env, order)
    fill["costs"]["transaction"] += 1
    with pytest.raises(ValueError, match="costs must match"):
        append(env, fill)
    fill = allocation_fill(env, order)
    fill["fills"][0]["price"] = 201
    with pytest.raises(ValueError, match="current raw mark"):
        append(env, fill)


def test_allocation_sell_sizes_from_current_mark_and_preserves_cash_continuity(setup):
    env = setup
    buy_five(env)
    env["clock"][0] = pd.Timestamp("2026-10-05T14:00:00Z")
    signal = event(env, event_id="s2", time="2026-10-05T14:00:00Z")
    signal["state"] = dict(cash=500.0, positions={"QQQ": 5.0}, prices={"QQQ": 100.0}, equity=1000.0)
    append(env, signal)
    env["clock"][0] += pd.Timedelta(seconds=1)
    order = event(env, "hypothetical_order", "o2", "2026-10-05T14:00:01Z")
    order.update(
        signal_id="s2",
        observation_timestamp=signal["event_timestamp"],
        decision_timestamp=order["event_timestamp"],
        order_timestamp=order["event_timestamp"],
        allocation_order=dict(
            asset="QQQ",
            target_weight=0.25,
            fee_bps=10.0,
            slippage_bps=20.0,
            sizing_convention="mark_notional_then_cash_budget",
        ),
    )
    order["state"] = signal["state"]
    append(env, order)
    env["clock"][0] = pd.Timestamp("2026-10-05T15:00:00Z")
    fill = event(env, "fill", "f2", "2026-10-05T15:00:00Z")
    qty = -325 / 120  # current equity1100 -> target275; currently600 invested
    fill.update(
        order_id="o2",
        observation_timestamp=order["observation_timestamp"],
        decision_timestamp=order["decision_timestamp"],
        order_timestamp=order["order_timestamp"],
        fill_timestamp=fill["event_timestamp"],
        fills=[dict(symbol="QQQ", quantity=qty, price=120.0)],
    )
    fill["costs"].update(transaction=0.325, slippage=0.65)
    fill["state"] = dict(
        cash=824.025, positions={"QQQ": 5 + qty}, prices={"QQQ": 120.0}, equity=1099.025
    )
    append(env, fill)
    assert fill["state"]["positions"]["QQQ"] * 120 == pytest.approx(275)


def test_corporate_action_requires_prior_pending_order_resolution(setup):
    env = setup
    allocation_order(env)
    action = corporate(env)
    with pytest.raises(ValueError, match="Resolve pending"):
        append(env, action)


def test_allocation_schema_rejects_ambiguous_or_levered_cash_orders():
    valid = dict(
        asset="QQQ",
        target_weight=0.5,
        fee_bps=1.0,
        slippage_bps=5.0,
        sizing_convention="mark_notional_then_cash_budget",
    )
    for update in [
        dict(target_weight=1.1),
        dict(fee_bps=-1),
        dict(sizing_convention="future_quote"),
    ]:
        with pytest.raises(ValueError):
            forward._allocation_order(dict(valid, **update))
