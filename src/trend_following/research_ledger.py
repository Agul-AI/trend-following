"""Self-financing, long-only research ledger with explicit lots and cash flows.

Taxes are *hypothetical scenarios*, not a tax-return/compliance implementation.
The model uses FIFO, separate short/long capital-loss carryforwards, cross-netting,
and a configurable same-security wash-sale window. It excludes other accounts,
substantially-identical instruments, ordinary-income loss deductions, estimated
payments, state taxes, and tax-rate changes. A long-term lot must be held MORE
than ``long_term_days`` calendar days. Every payment is cash funded; sales needed
to fund taxes incur costs and realize gains. There are no implicit share burns.

``simulate_ledger(..., targets_at='interval_start')`` buys before the supplied
interval return (legacy return-aligned targets). ``targets_at='close'`` marks the
completed interval FIRST, then executes the close target, so a same-close fill
cannot earn that interval's return. Initial prices only set units, not wealth.
Pass the same ledger through folds, with BOTH end-finalization flags false on
intermediate folds. Targets must already obey the research execution policy.

Total-return/adjusted-price streams must NOT also receive split or distribution
events. The explicit corporate-action hooks are for an unadjusted-price ledger.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

EPS = 1e-11


@dataclass(frozen=True)
class LedgerConfig:
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    short_tax_rate: float = 0.0
    long_tax_rate: float = 0.0
    interest_tax_rate: float = 0.0
    wash_sales: bool = False
    rebalance_policy: Literal["on_target_change", "every_bar"] = "on_target_change"
    long_term_days: int = 365
    wash_window_days: int = 30
    exposure_multipliers: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("fee_bps", "slippage_bps"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.cost_rate >= 1:
            raise ValueError("combined one-way costs must be below 100%")
        for name in ("short_tax_rate", "long_tax_rate", "interest_tax_rate"):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.rebalance_policy not in {"on_target_change", "every_bar"}:
            raise ValueError("unknown rebalance_policy")
        if self.long_term_days < 0 or self.wash_window_days < 0:
            raise ValueError("holding/window days must be nonnegative")
        if any(not np.isfinite(v) or v < 0 for v in self.exposure_multipliers.values()):
            raise ValueError("exposure multipliers must be finite and nonnegative")

    @property
    def cost_rate(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000.0


@dataclass
class TaxLot:
    lot_id: int
    asset: str
    quantity: float
    basis_per_share: float
    acquired: pd.Timestamp
    # Effective holding-period origin after tacking actually-held calendar days;
    # acquired remains the real replacement purchase date for wash eligibility.
    holding_start: pd.Timestamp
    wash_replacement_used: bool = False

    @property
    def basis(self) -> float:
        return self.quantity * self.basis_per_share


@dataclass
class PendingWashLoss:
    asset: str
    quantity: float
    loss_per_share: float
    sold: pd.Timestamp
    holding_start: pd.Timestamp
    character: Literal["short", "long"]


@dataclass
class TaxYear:
    realized_short: float = 0.0
    realized_long: float = 0.0
    ordinary_income: float = 0.0
    qualified_income: float = 0.0
    assessment: float = 0.0
    paid: float = 0.0
    short_carry_out: float = 0.0
    long_carry_out: float = 0.0


@dataclass
class LedgerResult:
    returns: pd.Series
    turnover: pd.Series
    taxes_paid: pd.Series
    cash_interest: pd.Series
    cash_weight: pd.Series
    events: pd.DataFrame
    holdings: pd.DataFrame
    cash: pd.Series
    equity: pd.Series
    ledger: PortfolioLedger

    @property
    def orders(self) -> pd.DataFrame:
        return self.events.loc[self.events["event"].isin(["buy", "sell"])].copy()

    @property
    def positions(self) -> pd.DataFrame:
        return self.holdings


EVENT_COLUMNS = [
    "timestamp",
    "event",
    "asset",
    "reason",
    "quantity",
    "price",
    "notional",
    "fee",
    "slippage",
    "cash_flow",
    "realized_short",
    "realized_long",
    "deferred_loss",
    "tax_year",
    "lot_id",
    "turnover_notional",
]


class PortfolioLedger:
    """Persistent FIFO ledger. Public hooks accept actual dated execution events."""

    def __init__(
        self,
        config: LedgerConfig | None = None,
        initial_cash: float = 1.0,
        initial_prices: Mapping[str, float] | None = None,
    ) -> None:
        if not np.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        self.config = config or LedgerConfig()
        self.cash = float(initial_cash)
        self.prices: dict[str, float] = {}
        self.lots: dict[str, list[TaxLot]] = {}
        self.pending_wash_losses: list[PendingWashLoss] = []
        self.tax_years: dict[int, TaxYear] = {}
        self.closed_tax_years: set[int] = set()
        self.events: list[dict] = []
        self.previous_targets: dict[str, float] | None = None
        self.last_timestamp: pd.Timestamp | None = None
        self._next_lot_id = 1
        for asset, price in (initial_prices or {}).items():
            self._set_price(str(asset), float(price), require_positive=True)

    def _set_price(self, asset: str, price: float, *, require_positive: bool = False) -> None:
        if not np.isfinite(price) or price < 0 or (require_positive and price == 0):
            raise ValueError("prices must be finite and nonnegative (positive at initialization)")
        self.prices[asset] = float(price)
        self.lots.setdefault(asset, [])

    def _event(self, timestamp: pd.Timestamp, event: str, **kwargs) -> None:
        row = dict.fromkeys(EVENT_COLUMNS, 0.0)
        row.update(timestamp=pd.Timestamp(timestamp), event=event, asset="", reason="", lot_id=None)
        row.update(kwargs)
        self.events.append(row)

    def _book(self, year: int) -> TaxYear:
        return self.tax_years.setdefault(int(year), TaxYear())

    def quantity(self, asset: str) -> float:
        return sum(lot.quantity for lot in self.lots.get(asset, []))

    def value(self, asset: str) -> float:
        return self.quantity(asset) * self.prices[asset]

    @property
    def equity(self) -> float:
        return self.cash + sum(self.value(asset) for asset in self.prices)

    @property
    def short_loss_carryforward(self) -> float:
        return self._last_closed_book().short_carry_out if self.closed_tax_years else 0.0

    @property
    def long_loss_carryforward(self) -> float:
        return self._last_closed_book().long_carry_out if self.closed_tax_years else 0.0

    def _last_closed_book(self) -> TaxYear:
        return self.tax_years[max(self.closed_tax_years)]

    def mark_to_market(self, prices: Mapping[str, float], timestamp: pd.Timestamp) -> None:
        """Mark absolute prices; no trading, cash transfer, or basis adjustment."""
        for asset, price in prices.items():
            self._set_price(str(asset), float(price))

    def _new_lot(
        self,
        asset: str,
        quantity: float,
        basis_per_share: float,
        acquired: pd.Timestamp,
        holding_start: pd.Timestamp | None = None,
        replacement_used: bool = False,
    ) -> TaxLot:
        lot = TaxLot(
            self._next_lot_id,
            asset,
            quantity,
            basis_per_share,
            acquired,
            holding_start if holding_start is not None else acquired,
            replacement_used,
        )
        self._next_lot_id += 1
        return lot

    def _character(self, holding_start: pd.Timestamp, sold: pd.Timestamp) -> str:
        days = (sold.normalize() - holding_start.normalize()).days
        return "long" if days > self.config.long_term_days else "short"

    def _recognize(
        self,
        gain: float,
        character: str,
        year: int,
        timestamp: pd.Timestamp,
        *,
        asset: str,
        reason: str,
        lot_id: int | None = None,
    ) -> None:
        book = self._book(year)
        field_name = f"realized_{character}"
        setattr(book, field_name, getattr(book, field_name) + gain)
        self._event(
            timestamp,
            "realization",
            asset=asset,
            reason=reason,
            tax_year=year,
            lot_id=lot_id,
            **{field_name: gain},
        )

    def _match_prior_replacements(self, loss: PendingWashLoss, timestamp: pd.Timestamp) -> None:
        earliest = timestamp.normalize() - pd.Timedelta(days=self.config.wash_window_days)
        rebuilt: list[TaxLot] = []
        for lot in self.lots[loss.asset]:
            eligible = (
                not lot.wash_replacement_used
                and earliest <= lot.acquired.normalize() <= timestamp.normalize()
                and loss.quantity > EPS
            )
            if not eligible:
                rebuilt.append(lot)
                continue
            matched = min(lot.quantity, loss.quantity)
            adjusted = self._new_lot(
                lot.asset,
                matched,
                lot.basis_per_share + loss.loss_per_share,
                lot.acquired,
                lot.holding_start - (loss.sold.normalize() - loss.holding_start.normalize()),
                True,
            )
            rebuilt.append(adjusted)
            lot.quantity -= matched
            if lot.quantity > EPS:
                rebuilt.append(lot)
            loss.quantity -= matched
            self._event(
                timestamp,
                "wash_basis_adjustment",
                asset=loss.asset,
                reason="prior_replacement",
                quantity=matched,
                deferred_loss=matched * loss.loss_per_share,
                lot_id=adjusted.lot_id,
                tax_year=loss.sold.year,
            )
        self.lots[loss.asset] = rebuilt

    def buy(
        self, asset: str, notional: float, timestamp: pd.Timestamp, *, reason: str = "rebalance"
    ) -> float:
        """Buy up to requested gross notional, cash-inclusive of fees and slippage."""
        timestamp = pd.Timestamp(timestamp)
        if not np.isfinite(notional) or notional < 0:
            raise ValueError("buy notional must be finite and nonnegative")
        price = self.prices[asset]
        if price <= 0 and notional > EPS:
            raise ValueError("cannot buy an asset at nonpositive price")
        notional = min(float(notional), max(self.cash, 0.0) / (1 + self.config.cost_rate))
        if notional <= EPS:
            return 0.0
        fee = notional * self.config.fee_bps / 10_000
        slip = notional * self.config.slippage_bps / 10_000
        quantity = notional / price
        unit_basis = (notional + fee + slip) / quantity
        self.cash -= notional + fee + slip
        remaining = quantity
        if self.config.wash_sales:
            for loss in self.pending_wash_losses:
                age = (timestamp.normalize() - loss.sold.normalize()).days
                if loss.asset != asset or not 0 <= age <= self.config.wash_window_days:
                    continue
                matched = min(remaining, loss.quantity)
                if matched <= EPS:
                    continue
                lot = self._new_lot(
                    asset,
                    matched,
                    unit_basis + loss.loss_per_share,
                    timestamp,
                    timestamp - (loss.sold.normalize() - loss.holding_start.normalize()),
                    True,
                )
                self.lots[asset].append(lot)
                loss.quantity -= matched
                remaining -= matched
                self._event(
                    timestamp,
                    "wash_basis_adjustment",
                    asset=asset,
                    reason="subsequent_replacement",
                    quantity=matched,
                    deferred_loss=matched * loss.loss_per_share,
                    tax_year=loss.sold.year,
                    lot_id=lot.lot_id,
                )
        if remaining > EPS:
            self.lots[asset].append(self._new_lot(asset, remaining, unit_basis, timestamp))
        self.pending_wash_losses = [p for p in self.pending_wash_losses if p.quantity > EPS]
        self._event(
            timestamp,
            "buy",
            asset=asset,
            reason=reason,
            quantity=quantity,
            price=price,
            notional=notional,
            fee=fee,
            slippage=slip,
            cash_flow=-(notional + fee + slip),
            turnover_notional=notional,
        )
        self._check_invariants()
        return notional

    def sell(
        self, asset: str, quantity: float, timestamp: pd.Timestamp, *, reason: str = "rebalance"
    ) -> float:
        """Sell FIFO lots, recognizing net proceeds after fees and slippage."""
        timestamp = pd.Timestamp(timestamp)
        if not np.isfinite(quantity) or quantity < 0:
            raise ValueError("sell quantity must be finite and nonnegative")
        quantity = min(float(quantity), self.quantity(asset))
        if quantity <= 0:
            return 0.0
        price = self.prices[asset]
        notional = quantity * price
        fee = notional * self.config.fee_bps / 10_000
        slip = notional * self.config.slippage_bps / 10_000
        net_unit = price * (1 - self.config.cost_rate)
        remaining = quantity
        losses: list[PendingWashLoss] = []
        # Remove ALL sold shares first: another portion of this same order must
        # never be incorrectly classified as an unsold prior replacement.
        for lot in self.lots[asset]:
            sold = min(remaining, lot.quantity)
            if sold <= 0:
                continue
            gain = sold * (net_unit - lot.basis_per_share)
            character = self._character(lot.holding_start, timestamp)
            if gain < -EPS and self.config.wash_sales:
                losses.append(
                    PendingWashLoss(
                        asset, sold, -gain / sold, timestamp, lot.holding_start, character
                    )
                )
                self._event(
                    timestamp,
                    "wash_loss_pending",
                    asset=asset,
                    reason=reason,
                    quantity=sold,
                    deferred_loss=-gain,
                    tax_year=timestamp.year,
                    lot_id=lot.lot_id,
                )
            else:
                self._recognize(
                    gain,
                    character,
                    timestamp.year,
                    timestamp,
                    asset=asset,
                    reason=reason,
                    lot_id=lot.lot_id,
                )
            lot.quantity -= sold
            remaining -= sold
        self.lots[asset] = [lot for lot in self.lots[asset] if lot.quantity > 0]
        for loss in losses:
            self._match_prior_replacements(loss, timestamp)
            if loss.quantity > EPS:
                self.pending_wash_losses.append(loss)
        self.cash += notional - fee - slip
        self._event(
            timestamp,
            "sell",
            asset=asset,
            reason=reason,
            quantity=quantity,
            price=price,
            notional=notional,
            fee=fee,
            slippage=slip,
            cash_flow=notional - fee - slip,
            turnover_notional=notional,
        )
        self._check_invariants()
        return notional - fee - slip

    def expire_wash_losses(
        self, timestamp: pd.Timestamp, *, no_future_replacements: bool = False
    ) -> None:
        """Recognize unmatched losses after the window; preserve original sale year.

        ``no_future_replacements`` is only valid for a terminal closed portfolio.
        Expiry can change a prior-year assessment/carryforward; settle_taxes will
        explicitly refund overpayments rather than silently rewriting wealth.
        """
        timestamp = pd.Timestamp(timestamp)
        if no_future_replacements and any(self.quantity(a) > EPS for a in self.prices):
            raise ValueError("cannot assume no replacements for a nonliquidated portfolio")
        retained = []
        for loss in self.pending_wash_losses:
            age = (timestamp.normalize() - loss.sold.normalize()).days
            if no_future_replacements or age > self.config.wash_window_days:
                self._recognize(
                    -loss.quantity * loss.loss_per_share,
                    loss.character,
                    loss.sold.year,
                    timestamp,
                    asset=loss.asset,
                    reason="wash_window_expired"
                    if not no_future_replacements
                    else "terminal_no_future_replacements",
                )
            else:
                retained.append(loss)
        self.pending_wash_losses = retained

    def accrue_cash(self, period_return: float, timestamp: pd.Timestamp) -> float:
        if not np.isfinite(period_return) or period_return < -1:
            raise ValueError("cash return must be finite and at least -1")
        income = self.cash * period_return
        self.cash += income
        self._book(pd.Timestamp(timestamp).year).ordinary_income += income
        # Per-bar income is returned by simulate_ledger; only nonzero cash flows
        # need an event, keeping zero-yield primary simulations compact.
        if abs(income) > EPS:
            self._event(
                timestamp, "cash_interest", cash_flow=income, tax_year=pd.Timestamp(timestamp).year
            )
        return income

    def _reassess_closed_years(self) -> None:
        short_carry = long_carry = 0.0
        for year in sorted(self.closed_tax_years):
            book = self._book(year)
            short = book.realized_short + short_carry
            long = book.realized_long + long_carry
            if short * long < 0:
                offset = min(abs(short), abs(long))
                if short > 0:
                    short -= offset
                    long += offset
                else:
                    long -= offset
                    short += offset
            short_carry, long_carry = min(short, 0.0), min(long, 0.0)
            book.short_carry_out, book.long_carry_out = short_carry, long_carry
            book.assessment = (
                max(short, 0.0) * self.config.short_tax_rate
                + max(long, 0.0) * self.config.long_tax_rate
                + max(book.ordinary_income, 0.0) * self.config.interest_tax_rate
                + max(book.qualified_income, 0.0) * self.config.long_tax_rate
            )

    def settle_taxes(self, timestamp: pd.Timestamp, *, close_year: int | None = None) -> float:
        """Cash-first settlement, with costed FIFO funding sales and reassessment.

        Funding sales are realized in their actual execution year. When the
        payment is made in that same closed year, the resulting extra liability
        is funded iteratively. Cross-year wash adjustments produce explicit
        refunds/additional payments; no basis or shares disappear to pay tax.
        """
        timestamp = pd.Timestamp(timestamp)
        if close_year is not None:
            if close_year > timestamp.year:
                raise ValueError("cannot close a future tax year")
            self.closed_tax_years.add(int(close_year))
            self._book(int(close_year))
        net_paid = 0.0
        for _ in range(200):
            self._reassess_closed_years()
            # Refund first, since it is available cash for other year balances.
            for year in sorted(self.closed_tax_years):
                book = self._book(year)
                refund = max(book.paid - book.assessment, 0.0)
                if refund > EPS:
                    self.cash += refund
                    book.paid -= refund
                    net_paid -= refund
                    self._event(
                        timestamp,
                        "tax_refund",
                        cash_flow=refund,
                        reason="assessment_adjustment",
                        tax_year=year,
                    )
            due = sum(
                max(self._book(y).assessment - self._book(y).paid, 0.0)
                for y in self.closed_tax_years
            )
            if due <= self.cash + EPS:
                for year in sorted(self.closed_tax_years):
                    book = self._book(year)
                    payment = max(book.assessment - book.paid, 0.0)
                    if payment > EPS:
                        self.cash -= payment
                        book.paid += payment
                        net_paid += payment
                        self._event(
                            timestamp,
                            "tax_payment",
                            cash_flow=-payment,
                            reason="annual_net_scenario",
                            tax_year=year,
                        )
                self._check_invariants()
                return net_paid
            shortfall = due - self.cash
            assets = sorted(self.prices, key=lambda a: (-self.value(a), a))
            asset = next((a for a in assets if self.value(a) > EPS), None)
            if asset is None:
                raise ValueError("insufficient portfolio equity to fund hypothetical tax")
            quantity = min(
                self.quantity(asset), shortfall / (self.prices[asset] * (1 - self.config.cost_rate))
            )
            self.sell(asset, quantity, timestamp, reason="tax_funding")
        raise RuntimeError("tax-funding iteration failed to converge")

    def rebalance(self, targets: Mapping[str, float], timestamp: pd.Timestamp) -> None:
        targets = {asset: float(targets.get(asset, 0.0)) for asset in self.prices}
        if any(not np.isfinite(w) or w < 0 or w > 1 for w in targets.values()):
            raise ValueError("target weights must be finite and in [0, 1]")
        if sum(targets.values()) > 1 + EPS:
            raise ValueError("target weights must sum to at most one")
        if (
            self.config.rebalance_policy == "on_target_change"
            and self.previous_targets is not None
            and all(abs(targets[a] - self.previous_targets.get(a, 0.0)) <= EPS for a in targets)
        ):
            return
        wealth = self.equity
        desired = {asset: weight * wealth for asset, weight in targets.items()}
        # Sell first, so multi-asset rotation never requires temporary borrowing.
        for asset in sorted(targets):
            excess = self.value(asset) - desired[asset]
            if excess > EPS and self.prices[asset] > 0:
                self.sell(asset, excess / self.prices[asset], timestamp)
            elif desired[asset] <= EPS and self.quantity(asset) > EPS and self.prices[asset] == 0:
                self.sell(asset, self.quantity(asset), timestamp)
        deficits = {asset: max(desired[asset] - self.value(asset), 0.0) for asset in targets}
        total = sum(deficits.values())
        scale = min(1.0, self.cash / (total * (1 + self.config.cost_rate))) if total > EPS else 0.0
        for asset in sorted(targets):
            self.buy(asset, deficits[asset] * scale, timestamp)
        self.previous_targets = dict(targets)

    def liquidate(self, timestamp: pd.Timestamp) -> None:
        for asset in sorted(self.prices):
            self.sell(asset, self.quantity(asset), timestamp, reason="terminal_liquidation")
        self.previous_targets = None

    def apply_split(self, asset: str, ratio: float, timestamp: pd.Timestamp) -> None:
        """Unadjusted-price hook: change units/marks, preserving value and basis."""
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("split ratio must be finite and positive")
        for lot in self.lots[asset]:
            lot.quantity *= ratio
            lot.basis_per_share /= ratio
        for loss in self.pending_wash_losses:
            if loss.asset == asset:
                loss.quantity *= ratio
                loss.loss_per_share /= ratio
        self.prices[asset] /= ratio
        self._event(timestamp, "split", asset=asset, reason="unadjusted_price_hook", quantity=ratio)

    def credit_distribution(
        self,
        asset: str,
        amount_per_share: float,
        timestamp: pd.Timestamp,
        *,
        tax_kind: Literal["ordinary", "qualified", "return_of_capital"] = "ordinary",
    ) -> float:
        """Unadjusted-price hook; caller marks ex-distribution prices separately.

        A return of capital reduces each lot's basis, with any excess recorded
        as a gain in that lot's holding-period category under this scenario.
        """
        if not np.isfinite(amount_per_share) or amount_per_share < 0:
            raise ValueError("distribution per share must be finite and nonnegative")
        if tax_kind not in {"ordinary", "qualified", "return_of_capital"}:
            raise ValueError("unknown distribution tax kind")
        timestamp = pd.Timestamp(timestamp)
        amount = self.quantity(asset) * amount_per_share
        self.cash += amount
        book = self._book(timestamp.year)
        if tax_kind == "ordinary":
            book.ordinary_income += amount
        elif tax_kind == "qualified":
            book.qualified_income += amount
        else:
            for lot in self.lots[asset]:
                excess = max(amount_per_share - lot.basis_per_share, 0.0) * lot.quantity
                lot.basis_per_share = max(lot.basis_per_share - amount_per_share, 0.0)
                if excess > EPS:
                    self._recognize(
                        excess,
                        self._character(lot.holding_start, timestamp),
                        timestamp.year,
                        timestamp,
                        asset=asset,
                        reason="return_of_capital_exceeds_basis",
                        lot_id=lot.lot_id,
                    )
        self._event(
            timestamp,
            "distribution",
            asset=asset,
            reason=tax_kind,
            cash_flow=amount,
            tax_year=timestamp.year,
        )
        return amount

    def _check_invariants(self) -> None:
        if self.cash < -EPS or not np.isfinite(self.cash):
            raise AssertionError("ledger has invalid/negative cash")
        if -EPS < self.cash < 0:
            self.cash = 0.0
        for lots in self.lots.values():
            for lot in lots:
                if lot.quantity < -EPS or lot.basis_per_share < -EPS:
                    raise AssertionError("invalid tax lot")

    def snapshot(self, timestamp: pd.Timestamp) -> list[dict]:
        wealth = self.equity
        rows = []
        for asset in sorted(self.prices):
            value = self.value(asset)
            weight = value / wealth if wealth > 0 else 0.0
            rows.append(
                dict(
                    timestamp=timestamp,
                    asset=asset,
                    quantity=self.quantity(asset),
                    price=self.prices[asset],
                    value=value,
                    basis=sum(lot.basis for lot in self.lots[asset]),
                    actual_weight=weight,
                    effective_exposure=weight * self.config.exposure_multipliers.get(asset, 1.0),
                    target_weight=(self.previous_targets or {}).get(asset, 0.0),
                    cash=self.cash,
                    equity=wealth,
                )
            )
        return rows


def simulate_ledger(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    cash_returns: pd.Series,
    config: LedgerConfig | None = None,
    *,
    liquidate_at_end: bool = True,
    finalize_tax_at_end: bool = True,
    ledger: PortfolioLedger | None = None,
    initial_cash: float = 1.0,
    initial_prices: Mapping[str, float] | None = None,
    start_prices: pd.DataFrame | None = None,
    targets_at: Literal["interval_start", "close"] = "interval_start",
    corporate_actions: pd.DataFrame | None = None,
    price_basis: Literal["total_return", "unadjusted"] = "total_return",
) -> LedgerResult:
    """Execute an aligned return/target stream, retaining state across fold calls.

    ``returns[t]`` is the return of the interval ending at t. For interval_start
    targets the supplied target is return-aligned; for close targets it is a
    decision/fill at t that only participates in subsequent returns. Optional
    ``start_prices[t]`` are absolute prices BEFORE that interval, and replace the
    previous ending marks (any gap is included in portfolio returns). Do not
    provide close prices as start_prices. Supplying no marks compounds the
    return stream from initial_prices (default 1 per asset).

    Annual settlement occurs at the first event of the following year, making
    full-run and chunked results identical. In close mode funding sales execute
    at the newly marked close; in interval_start mode they execute at the
    beginning mark. Final scenario-end settlement is an explicit exception,
    not an actual tax due date.

    Intermediate folds: ``ledger=prior.ledger, liquidate_at_end=False,
    finalize_tax_at_end=False``. A new calendar year settles the previous year if
    the prior chunk did not already do so. Missing data is an error, never a
    silent zero-fill. Holdings report ACTUAL drifting exposure, not only targets.

    Corporate actions require ``price_basis='unadjusted'``. Rows use timestamp,
    asset, split_ratio (default 1), distribution_per_share (default 0), and
    tax_kind (default ordinary). Split then distribution are processed BEFORE
    same-bar executions. Distribution units are POST-split shares if both occur
    together. Entitlement is based on prior holdings; cash is credited at the
    supplied first ex-date bar, NOT the actual pay date. Split-bearing interval
    returns must be neutralized by the caller: (1 + raw_return)*split_ratio - 1.
    Marks/basis already adjust in apply_split; using the raw split drop again
    would double count it. Optional start_prices must likewise be POST-action
    beginning marks. Never feed adjusted total returns AND corporate actions.
    """
    if targets_at not in {"interval_start", "close"}:
        raise ValueError("targets_at must be interval_start or close")
    if price_basis not in {"total_return", "unadjusted"}:
        raise ValueError("price_basis must be total_return or unadjusted")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")
    if returns.empty or returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("returns must be nonempty, chronological, and uniquely indexed")
    if returns.columns.has_duplicates or not len(returns.columns):
        raise ValueError("asset columns must be nonempty and unique")
    if not target_weights.index.equals(returns.index) or not cash_returns.index.equals(
        returns.index
    ):
        raise ValueError("returns, targets, and cash returns must have identical indexes")
    if set(returns.columns) != set(target_weights.columns):
        raise ValueError("returns and targets must have the same assets")
    ret = returns.astype(float)
    target = target_weights.reindex(columns=ret.columns).astype(float)
    cash_ret = cash_returns.astype(float)
    if not np.isfinite(ret.to_numpy()).all() or ret.lt(-1).any().any():
        raise ValueError("asset returns must be finite and at least -1")
    if (
        not np.isfinite(target.to_numpy()).all()
        or target.lt(0).any().any()
        or target.gt(1).any().any()
    ):
        raise ValueError("targets must be finite and in [0, 1]")
    if target.sum(axis=1).gt(1 + EPS).any():
        raise ValueError("target weights must sum to at most one")
    if not np.isfinite(cash_ret.to_numpy()).all() or cash_ret.lt(-1).any():
        raise ValueError("cash returns must be finite and at least -1")
    if start_prices is not None:
        if not start_prices.index.equals(ret.index) or set(start_prices.columns) != set(
            ret.columns
        ):
            raise ValueError("start_prices must align exactly with returns")
        if not np.isfinite(start_prices.to_numpy()).all() or start_prices.le(0).any().any():
            raise ValueError("start_prices must be finite and positive")
    actions_by_timestamp: dict[pd.Timestamp, list[dict]] = {}
    if corporate_actions is not None:
        if price_basis != "unadjusted":
            raise ValueError(
                "corporate_actions require price_basis='unadjusted'; avoid double counting"
            )
        actions = corporate_actions.copy()
        if not {"timestamp", "asset"}.issubset(actions.columns):
            raise ValueError("corporate_actions require timestamp and asset columns")
        actions["timestamp"] = pd.to_datetime(actions["timestamp"])
        for column, default in (
            ("split_ratio", 1.0),
            ("distribution_per_share", 0.0),
            ("tax_kind", "ordinary"),
        ):
            if column not in actions:
                actions[column] = default
        if actions[["timestamp", "asset"]].isna().any().any():
            raise ValueError("corporate_actions identifiers must be nonmissing")
        if actions.duplicated(["timestamp", "asset"]).any():
            raise ValueError("duplicate corporate action for timestamp/asset")
        if not actions["timestamp"].isin(ret.index).all():
            raise ValueError("corporate_actions timestamps must belong to the simulated index")
        if not actions["asset"].isin(ret.columns).all():
            raise ValueError("corporate_actions contain unknown assets")
        numeric = actions[["split_ratio", "distribution_per_share"]].astype(float)
        if (
            not np.isfinite(numeric.to_numpy()).all()
            or numeric["split_ratio"].le(0).any()
            or numeric["distribution_per_share"].lt(0).any()
        ):
            raise ValueError("corporate action split/distribution values are invalid")
        if not actions["tax_kind"].isin(["ordinary", "qualified", "return_of_capital"]).all():
            raise ValueError("unknown corporate action tax kind")
        for action in actions.to_dict("records"):
            actions_by_timestamp.setdefault(action["timestamp"], []).append(action)
    if ledger is None:
        ledger = PortfolioLedger(config, initial_cash, initial_prices)
    elif config is not None and config != ledger.config:
        raise ValueError("supplied config differs from persistent ledger config")
    elif initial_prices is not None:
        raise ValueError("initial_prices cannot reset an existing ledger")
    if ledger.last_timestamp is not None and ret.index[0] <= ledger.last_timestamp:
        raise ValueError("continued simulation must start after the previous timestamp")
    for asset in ret.columns:
        if asset not in ledger.prices:
            ledger._set_price(asset, 1.0, require_positive=True)
    if set(ledger.prices) != set(ret.columns):
        raise ValueError("persistent ledger assets must match the simulation assets")
    event_start = len(ledger.events)
    output: list[dict] = []
    holdings: list[dict] = []
    values, targets, cash_values = ret.to_numpy(), target.to_numpy(), cash_ret.to_numpy()
    timestamps = list(ret.index)
    assets = list(ret.columns)
    for i, timestamp in enumerate(timestamps):
        equity_start = ledger.equity
        bar_event_start = len(ledger.events)
        if equity_start <= EPS:
            raise ValueError("portfolio equity exhausted")
        for action in actions_by_timestamp.get(timestamp, []):
            if float(action["split_ratio"]) != 1.0:
                ledger.apply_split(action["asset"], float(action["split_ratio"]), timestamp)
            if float(action["distribution_per_share"]) != 0.0:
                ledger.credit_distribution(
                    action["asset"],
                    float(action["distribution_per_share"]),
                    timestamp,
                    tax_kind=action["tax_kind"],
                )
        ledger.expire_wash_losses(timestamp)
        if start_prices is not None:
            ledger.mark_to_market(start_prices.loc[timestamp].to_dict(), timestamp)
        new_year = (
            ledger.last_timestamp is not None and timestamp.year != ledger.last_timestamp.year
        )
        if targets_at == "interval_start":
            if new_year:
                ledger.settle_taxes(timestamp, close_year=ledger.last_timestamp.year)
            elif ledger.closed_tax_years:
                ledger.settle_taxes(timestamp)
        weights = dict(zip(assets, targets[i], strict=True))
        if targets_at == "interval_start":
            ledger.rebalance(weights, timestamp)
        weight_before_interest = ledger.cash / ledger.equity if ledger.equity > 0 else 0.0
        ledger.mark_to_market(
            {asset: ledger.prices[asset] * (1 + values[i, j]) for j, asset in enumerate(assets)},
            timestamp,
        )
        interest = ledger.accrue_cash(float(cash_values[i]), timestamp)
        if targets_at == "close":
            if new_year:
                ledger.settle_taxes(timestamp, close_year=ledger.last_timestamp.year)
            elif ledger.closed_tax_years:
                ledger.settle_taxes(timestamp)
            ledger.rebalance(weights, timestamp)
        last = i == len(timestamps) - 1
        if last and liquidate_at_end:
            ledger.liquidate(timestamp)
            if finalize_tax_at_end:
                ledger.expire_wash_losses(timestamp, no_future_replacements=True)
        if last and finalize_tax_at_end:
            ledger.settle_taxes(timestamp, close_year=timestamp.year)
        bar_events = ledger.events[bar_event_start:]
        taxes = -sum(
            e["cash_flow"] for e in bar_events if e["event"] in {"tax_payment", "tax_refund"}
        )
        turnover = sum(e["turnover_notional"] for e in bar_events) / equity_start
        output.append(
            dict(
                returns=ledger.equity / equity_start - 1,
                turnover=turnover,
                taxes_paid=taxes,
                cash_interest=interest,
                cash_weight=weight_before_interest,
                cash=ledger.cash,
                equity=ledger.equity,
            )
        )
        holdings.extend(ledger.snapshot(timestamp))
        ledger.last_timestamp = timestamp
        ledger._check_invariants()
    frame = pd.DataFrame(output, index=ret.index)
    events = pd.DataFrame(ledger.events[event_start:], columns=EVENT_COLUMNS)
    return LedgerResult(
        **{c: frame[c].rename(c) for c in frame.columns},
        events=events,
        holdings=pd.DataFrame(holdings),
        ledger=ledger,
    )
