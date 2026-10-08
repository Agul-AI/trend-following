"""Opt-in physical-vehicle OOS accounting; historical ledger semantics are untouched.

Raw close prices are mandatory. Splits alter units and basis; distributions create
an ex-date receivable, not spendable cash, until the first observed pay-date bar.
Only cash held over an interval earns its cash return. Policy prices are a separate
causal, cumulative-split-normalized PRICE index (no dividend reinvestment).

Taxes are isolated hypothetical scenarios, not investor tax compliance. Unknown
distribution character is modeled as ordinary income and prominently provisional.
The terminal clone reserves marginal scenario tax on unpaid receipts without
inventing their early payment or assuming future capital-loss offsets.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trend_following.research_ledger import (
    EPS,
    EVENT_COLUMNS,
    LedgerConfig,
    LedgerResult,
    PortfolioLedger,
)

ACTION_COLUMNS = [
    "action_id",
    "asset",
    "kind",
    "effective_at",
    "available_at",
    "split_ratio",
    "amount_per_share",
    "pay_at",
    "character",
    "availability_basis",
    "source_observed_at",
]
CHARACTERS = {"ordinary", "qualified", "long_capital_gain", "unknown"}


def _stamp(value: Any, name: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must be a nonmissing timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def normalize_oos_actions(actions: pd.DataFrame | list[dict] | None) -> pd.DataFrame:
    """Validate root's public action schema without imputing availability or tax facts.

    ``effective_at`` is the ex-date/session opening timestamp; ``pay_at`` is the
    first permitted cash-availability timestamp, not a retroactive credit. If a
    split and distribution coincide, distribution units are POST-split shares.
    """
    frame = pd.DataFrame(actions).copy() if actions is not None else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(columns=ACTION_COLUMNS)
    required = {
        "action_id",
        "asset",
        "kind",
        "effective_at",
        "available_at",
        "availability_basis",
        "source_observed_at",
    }
    if not required.issubset(frame):
        raise ValueError(f"corporate actions require {sorted(required)}")
    rows = []
    for row in frame.to_dict("records"):
        if any(not isinstance(row[k], str) or not row[k] for k in ["action_id", "asset"]):
            raise ValueError("nonempty action_id and asset required")
        kind = row["kind"]
        if kind not in {"split", "distribution"}:
            raise ValueError("action kind must be split or distribution")
        effective = _stamp(row["effective_at"], "effective_at")
        available = _stamp(row["available_at"], "available_at")
        basis = row["availability_basis"]
        if basis not in {"assumed_public_by_ex_date", "official_announcement"}:
            raise ValueError("explicit supported action availability_basis required")
        observed = _stamp(row["source_observed_at"], "source_observed_at")
        result = dict(
            action_id=row["action_id"],
            asset=row["asset"],
            kind=kind,
            effective_at=effective,
            available_at=available,
            split_ratio=1.0,
            amount_per_share=0.0,
            pay_at=pd.NaT,
            character="unknown",
            availability_basis=basis,
            source_observed_at=observed,
        )
        field = "split_ratio" if kind == "split" else "amount_per_share"
        value = row.get(field)
        if isinstance(value, (bool, np.bool_)) or value is None:
            raise ValueError(f"finite positive {field} required")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"finite positive {field} required")
        result[field] = value
        if kind == "distribution":
            pay = _stamp(row.get("pay_at"), "pay_at")
            if pay < effective:
                raise ValueError("payment cannot precede entitlement")
            character = row.get("character", "unknown")
            if character not in CHARACTERS:
                raise ValueError("unsupported distribution character; use unknown provisionally")
            result.update(pay_at=pay, character=character)
        rows.append(result)
    result = pd.DataFrame(rows, columns=ACTION_COLUMNS)
    # Multiple distribution components (ordinary and capital-gain, for example)
    # can legitimately share an ex-date; their source action IDs must differ.
    if (
        result.action_id.duplicated().any()
        or result.loc[result.kind.eq("split")].duplicated(["asset", "effective_at"]).any()
    ):
        raise ValueError("duplicate corporate action id or split asset/effective timestamp")
    return result.sort_values(["effective_at", "asset", "kind"], kind="stable").reset_index(
        drop=True
    )


def _prices(prices: pd.DataFrame) -> pd.DataFrame:
    if (
        not isinstance(prices.index, pd.DatetimeIndex)
        or prices.index.tz is None
        or prices.empty
        or not prices.index.is_unique
        or not prices.index.is_monotonic_increasing
        or not len(prices.columns)
        or prices.columns.has_duplicates
    ):
        raise ValueError("prices require unique chronological aware bar ends and unique assets")
    values = prices.astype(float)
    if not np.isfinite(values.to_numpy()).all() or values.le(0).any().any():
        raise ValueError("raw close prices must be finite and positive")
    return values


def _action_bars(actions: pd.DataFrame, prices: pd.DataFrame) -> dict[pd.Timestamp, list[dict]]:
    """Map session-open actions to their first observed ex-date completed bar."""
    mapped: dict[pd.Timestamp, list[dict]] = {}
    index = prices.index
    local_dates = index.tz_convert("America/New_York").date
    for action in actions.to_dict("records"):
        if action["asset"] not in prices:
            raise ValueError("action contains an asset outside this ledger/policy price panel")
        effective = action["effective_at"]
        day = effective.tz_convert("America/New_York").date()
        if day < local_dates[0] or day > local_dates[-1]:
            continue
        candidates = index[(local_dates == day) & (index >= effective)]
        if not len(candidates):
            raise ValueError("missing ex-date price observation for corporate action")
        stamp = candidates[0]
        if action["available_at"] > stamp:
            raise ValueError("corporate action was unavailable at its first ex-date observation")
        mapped.setdefault(stamp, []).append(action)
    for records in mapped.values():
        records.sort(key=lambda a: (a["kind"] != "split", a["asset"], a["action_id"]))
    return mapped


def causal_split_adjusted_prices(
    close_prices: pd.DataFrame,
    corporate_actions: pd.DataFrame | list[dict] | None = None,
) -> pd.DataFrame:
    """Return forward-unit price indices; appending a future split never revises past rows.

    These are policy references, NOT raw execution quotes or total-return prices.
    Use with unchanged execution engine to keep entry/peak references continuous.
    Do not additionally call apply_execution_split on a normalized price stream.
    """
    prices = _prices(close_prices)
    actions = normalize_oos_actions(corporate_actions)
    mapped = _action_bars(actions, prices)
    factors = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    for stamp, records in mapped.items():
        for action in records:
            if action["kind"] == "split":
                factors.loc[stamp, action["asset"]] *= action["split_ratio"]
    return prices * factors.cumprod()


def apply_execution_split(engine: Any, ratio: float) -> None:
    """Optional raw-price engine hook, before step; never combine with normalized prices.

    Target-weight pending orders contain no fixed share quantity/limit price;
    they need no adjustment. Historical event quotes retain their original units.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("split ratio must be positive and finite")
    for field in ("entry_price", "peak_price", "previous_risk_price"):
        value = getattr(engine, field)
        if value is not None and np.isfinite(value):
            setattr(engine, field, value / ratio)


@dataclass
class DistributionReceivable:
    action_id: str
    asset: str
    ex_at: pd.Timestamp
    pay_at: pd.Timestamp
    entitled_quantity: float
    amount_per_share: float
    amount: float
    character: str


class AccrualPortfolioLedger(PortfolioLedger):
    """Opt-in subclass: account equity includes unpaid, non-interest-bearing receipts."""

    def __init__(self, *args, **kwargs):
        self.receivables: dict[str, DistributionReceivable] = {}
        self.processed_actions: dict[str, dict] = {}
        self.tax_provisional = False
        self.provisional_action_ids: set[str] = set()
        self.unpaid_distribution_tax_reserve = 0.0
        self.terminal_clone = False
        super().__init__(*args, **kwargs)

    @property
    def receivable_value(self) -> float:
        return sum(receipt.amount for receipt in self.receivables.values())

    @property
    def equity(self) -> float:
        return super().equity + self.receivable_value - self.unpaid_distribution_tax_reserve

    def apply_action(self, action: dict, timestamp: pd.Timestamp) -> None:
        key = action["action_id"]
        if key in self.processed_actions:
            if self.processed_actions[key] != action:
                raise ValueError("previous corporate action changed; never silently revise state")
            return
        if action["kind"] == "split":
            self.apply_split(action["asset"], action["split_ratio"], timestamp)
            self.events[-1].update(action_id=key, effective_at=action["effective_at"])
        else:
            quantity = self.quantity(action["asset"])
            amount = quantity * action["amount_per_share"]
            if amount > EPS:
                self.receivables[key] = DistributionReceivable(
                    key,
                    action["asset"],
                    action["effective_at"],
                    action["pay_at"],
                    quantity,
                    action["amount_per_share"],
                    amount,
                    action["character"],
                )
                if action["character"] == "unknown":
                    self.tax_provisional = True
                    self.provisional_action_ids.add(key)
            self._event(
                timestamp,
                "distribution_entitlement",
                asset=action["asset"],
                reason=action["character"],
                quantity=quantity,
                action_id=key,
                effective_at=action["effective_at"],
                pay_at=action["pay_at"],
                receivable_change=amount,
            )
        self.processed_actions[key] = copy.deepcopy(action)

    def pay_distributions(self, timestamp: pd.Timestamp) -> float:
        paid = 0.0
        for key, receipt in list(self.receivables.items()):
            if receipt.pay_at > timestamp:
                continue
            self.cash += receipt.amount
            # Timing and marginal rates are scenario conventions, not a filing implementation.
            book = self._book(receipt.pay_at.year)
            if receipt.character == "qualified":
                book.qualified_income += receipt.amount
            elif receipt.character == "long_capital_gain":
                book.realized_long += receipt.amount
            else:
                book.ordinary_income += receipt.amount
            self._event(
                timestamp,
                "distribution_payment",
                asset=receipt.asset,
                reason=receipt.character,
                action_id=key,
                cash_flow=receipt.amount,
                receivable_change=-receipt.amount,
                tax_year=receipt.pay_at.year,
                pay_at=receipt.pay_at,
            )
            paid += receipt.amount
            del self.receivables[key]
        return paid

    def snapshot(self, timestamp: pd.Timestamp) -> list[dict]:
        return [
            dict(
                row,
                receivables=self.receivable_value,
                unpaid_distribution_tax_reserve=self.unpaid_distribution_tax_reserve,
                tax_provisional=self.tax_provisional,
            )
            for row in super().snapshot(timestamp)
        ]


def simulate_oos_ledger(
    close_prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cash_returns: pd.Series,
    config: LedgerConfig | None = None,
    *,
    corporate_actions: pd.DataFrame | list[dict] | None = None,
    ledger: AccrualPortfolioLedger | None = None,
    initial_cash: float = 1000.0,
    initial_prices: dict[str, float] | None = None,
) -> LedgerResult:
    """Continuous raw-price CLOSE-target account; never liquidates at a chunk boundary.

    Per completed bar: accrue cash held over the ending interval; process splits
    and prior-holder distribution entitlement; mark raw prices; release pay-date
    cash; settle prior-year modeled taxes; execute already-delayed target. Only
    cash, not receivables, can fund orders/taxes. Same-close buys get no ex-date
    entitlement. A pay date without a price bar releases cash on the next bar,
    without retroactively earning interest for the unobserved interval.
    """
    prices = _prices(close_prices)
    if not target_weights.index.equals(prices.index) or not cash_returns.index.equals(prices.index):
        raise ValueError("targets and cash returns must exactly align with close prices")
    if set(target_weights.columns) != set(prices.columns):
        raise ValueError("target and raw-price assets must match")
    targets = target_weights.reindex(columns=prices.columns).astype(float)
    cash_returns = cash_returns.astype(float)
    if (
        not np.isfinite(targets.to_numpy()).all()
        or targets.lt(0).any().any()
        or targets.gt(1).any().any()
        or targets.sum(axis=1).gt(1 + EPS).any()
        or not np.isfinite(cash_returns.to_numpy()).all()
        or cash_returns.lt(-1).any()
    ):
        raise ValueError("invalid target weights or cash returns")
    actions = normalize_oos_actions(corporate_actions)
    mapped = _action_bars(actions, prices)
    if ledger is None:
        ledger = AccrualPortfolioLedger(
            config, initial_cash, initial_prices or prices.iloc[0].to_dict()
        )
    elif (
        not isinstance(ledger, AccrualPortfolioLedger)
        or ledger.terminal_clone
        or (config is not None and config != ledger.config)
        or initial_prices is not None
    ):
        raise ValueError("invalid persistent OOS ledger/config or attempted terminal-clone reuse")
    if set(ledger.prices) != set(prices.columns):
        raise ValueError("persistent account assets must match prices")
    if ledger.last_timestamp is not None and prices.index[0] <= ledger.last_timestamp:
        raise ValueError("continued account timestamps must strictly increase")
    for action in actions.to_dict("records"):
        old = ledger.processed_actions.get(action["action_id"])
        if old is not None and old != action:
            raise ValueError("previous corporate action changed; never silently revise state")
    event_start = len(ledger.events)
    output, holdings = [], []
    for stamp, quotes in prices.iterrows():
        start = ledger.equity
        if start <= EPS:
            raise ValueError("portfolio equity exhausted")
        first_event = len(ledger.events)
        cash_weight = ledger.cash / start
        # Fresh account has earned no pre-start cash interval.
        interest = ledger.accrue_cash(
            float(cash_returns.loc[stamp]) if ledger.last_timestamp is not None else 0.0, stamp
        )
        for action in mapped.get(stamp, []):
            ledger.apply_action(action, stamp)
        ledger.expire_wash_losses(stamp)
        ledger.mark_to_market(quotes.to_dict(), stamp)
        ledger.pay_distributions(stamp)
        if ledger.last_timestamp is not None and stamp.year != ledger.last_timestamp.year:
            ledger.settle_taxes(stamp, close_year=ledger.last_timestamp.year)
        elif ledger.closed_tax_years:
            ledger.settle_taxes(stamp)
        ledger.rebalance(targets.loc[stamp].to_dict(), stamp)
        events = ledger.events[first_event:]
        output.append(
            dict(
                returns=ledger.equity / start - 1,
                turnover=sum(e["turnover_notional"] for e in events) / start,
                taxes_paid=-sum(
                    e["cash_flow"] for e in events if e["event"] in {"tax_payment", "tax_refund"}
                ),
                cash_interest=interest,
                cash_weight=cash_weight,
                cash=ledger.cash,
                equity=ledger.equity,
            )
        )
        holdings.extend(ledger.snapshot(stamp))
        ledger.last_timestamp = stamp
        ledger._check_invariants()
    frame = pd.DataFrame(output, index=prices.index)
    events_frame = pd.DataFrame(ledger.events[event_start:])
    if events_frame.empty:
        events_frame = pd.DataFrame(columns=EVENT_COLUMNS)
    return LedgerResult(
        **{c: frame[c].rename(c) for c in frame},
        events=events_frame,
        holdings=pd.DataFrame(holdings),
        ledger=ledger,
    )


def terminal_snapshot(
    ledger: AccrualPortfolioLedger,
    timestamp: pd.Timestamp,
    *,
    settle_tax: bool = True,
) -> dict[str, Any]:
    """Liquidate a deep CLONE; retain original holdings, basis, cash and receivables.

    Unpaid receipts remain assets. A separate marginal scenario-tax reserve on
    them reduces terminal NAV; it is not an early cash payment. No future capital
    loss offsets are assumed for that reserve. The resulting clone is terminal
    and cannot be fed back into the continuous simulator.
    """
    stamp = _stamp(timestamp, "terminal timestamp")
    if ledger.last_timestamp is None or stamp != ledger.last_timestamp:
        raise ValueError("checkpoint must use the account's actual last observed mark")
    if ledger.terminal_clone:
        raise ValueError("terminal clone cannot be liquidated a second time")
    marked = ledger.equity
    clone = copy.deepcopy(ledger)
    first_event = len(clone.events)
    clone.liquidate(stamp)
    taxes = 0.0
    reserve = 0.0
    if settle_tax:
        clone.expire_wash_losses(stamp, no_future_replacements=True)
        taxes = clone.settle_taxes(stamp, close_year=stamp.year)
        for receipt in clone.receivables.values():
            rate = (
                clone.config.long_tax_rate
                if receipt.character in {"qualified", "long_capital_gain"}
                else clone.config.interest_tax_rate
            )
            reserve += receipt.amount * rate
        clone.unpaid_distribution_tax_reserve = reserve
        if reserve:
            clone._event(
                stamp,
                "terminal_distribution_tax_reserve",
                reason="marginal_scenario",
                reserve_change=reserve,
            )
    clone.terminal_clone = True
    events = clone.events[first_event:]
    return dict(
        marked_equity=marked,
        liquidated_equity=clone.equity,
        exit_fees=sum(e["fee"] for e in events),
        exit_slippage=sum(e["slippage"] for e in events),
        terminal_taxes_paid=taxes,
        receivables=clone.receivable_value,
        unpaid_distribution_tax_reserve=reserve,
        tax_provisional=clone.tax_provisional,
        ledger=clone,
    )
