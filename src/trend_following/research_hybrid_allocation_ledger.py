"""Independent fractional-share target ledger for the hybrid allocation study.

No files, strategy features or regime rules are read here. Targets must already
be causally confirmed. Every target is measured against post-cost equity at the
raw execution quote; zero-cost ledgers replay the same commands with own shares.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_following import research_daily_hourly_v5 as core


class TargetBook:
    """Vectorized separately funded normalized accounts, with no implicit rebalance."""

    COMMAND_COLUMNS = (
        "account_id",
        "account_group",
        "timestamp",
        "phase",
        "reason",
        "target_fraction",
        "decision_at",
        "decision_bar",
        "execution_bar",
        "reference_price",
        "post_gap_flag",
        "post_gap_windows",
    )
    TRADE_COLUMNS = (
        *COMMAND_COLUMNS,
        "side",
        "quantity",
        "execution_price",
        "reference_notional",
        "commission",
        "slippage",
        "equity_before",
        "equity_after",
        "shares_before",
        "shares_after",
        "cash_before",
        "cash_after",
        "holding_hours",
        "closed_episode",
    )

    def __init__(self, ids, groups, *, commission_bps=1, slippage_bps=3, initial_cash=1000):
        self.ids, self.groups = tuple(ids), tuple(groups)
        if not self.ids or len(set(self.ids)) != len(self.ids) or len(self.groups) != len(self.ids):
            raise ValueError("Unique aligned account identities required")
        self.fee, self.slip = core._costs(commission_bps, slippage_bps, initial_cash)
        self.initial_cash = float(initial_cash)
        n = len(ids)
        self.cash = np.full(n, initial_cash, dtype=float)
        self.shares = np.zeros(n)
        self.dividends = np.zeros(n)
        self.targets = np.zeros(n)
        self.orders = np.zeros(n, dtype=int)
        self.trips = np.zeros(n, dtype=int)
        self.holds = np.zeros(n)
        self.entered_at = np.full(n, np.nan)
        self.commission = np.zeros(n)
        self.slippage = np.zeros(n)
        self.turnover = np.zeros(n)
        self.trade_rows = []
        self.command_rows = []

    def wealth(self, price):
        values = self.cash + self.shares * price
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Nonfinite or exhausted account wealth")
        return values

    def accrue(self, rate_return):
        if not np.isfinite(rate_return) or rate_return <= -1:
            raise ValueError("Invalid cash accrual")
        self.cash *= 1 + rate_return
        self.dividends *= 1 + rate_return

    def actions(self, split, dividend):
        if not np.isfinite([split, dividend]).all() or split <= 0 or dividend < 0:
            raise ValueError("Invalid corporate action")
        self.shares *= split
        distribution = self.shares * dividend
        self.cash += distribution
        self.dividends += distribution

    def drip(self, price):
        mask = (self.shares > 0) & (self.dividends > 0)
        self.shares[mask] += self.dividends[mask] / price
        self.cash[mask] -= self.dividends[mask]
        self.dividends[mask] = 0

    def execute(
        self,
        targets,
        mask,
        price,
        timestamp,
        *,
        phase="open",
        reasons=None,
        decision_at=None,
        decision_bar=None,
        execution_bar=None,
        record=True,
    ):
        """Apply one atomic command group; return per-account notional/charges.

        The mask is determined by confirmed target changes, not exposure drift.
        Returns all pre-group account wealth for common-denominator turnover.
        """
        targets, mask = np.asarray(targets, dtype=float), np.asarray(mask, dtype=bool)
        n = len(self.ids)
        if (
            targets.shape != (n,)
            or mask.shape != (n,)
            or not np.isfinite(targets).all()
            or ((targets < 0) | (targets > 1)).any()
            or not np.isfinite(price)
            or price <= 0
        ):
            raise ValueError("Finite unlevered targets and a positive raw quote required")
        timestamp = pd.Timestamp(timestamp)
        if timestamp.tzinfo is None or phase not in {"open", "terminal"}:
            raise ValueError("Timezone-aware safe execution phase required")
        before = self.wealth(price).copy()
        q_before, cash_before = self.shares.copy(), self.cash.copy()
        current = q_before * price
        delta_value = targets * before - current
        buys = mask & (delta_value > 1e-11)
        sells = mask & (delta_value < -1e-11)
        actual = buys | sells
        quantity = np.zeros(n)
        execution = np.full(n, price, dtype=float)
        execution[buys] *= 1 + self.slip
        execution[sells] *= 1 - self.slip
        kb = (1 + self.slip) * (1 + self.fee)
        ks = (1 - self.slip) * (1 - self.fee)
        quantity[buys] = delta_value[buys] / (price * (1 + targets[buys] * (kb - 1)))
        quantity[sells] = -delta_value[sells] / (price * (1 - targets[sells] * (1 - ks)))
        # Binary endpoints retain the immutable engine's operation ordering.
        full_buys = buys & (targets == 1)
        quantity[full_buys] = self.cash[full_buys] / (execution[full_buys] * (1 + self.fee))
        full_sells = sells & (targets == 0)
        quantity[full_sells] = self.shares[full_sells]
        charged = quantity * execution * self.fee
        adverse = quantity * abs(execution - price)
        self.cash[buys] -= quantity[buys] * execution[buys] + charged[buys]
        self.shares[buys] += quantity[buys]
        self.cash[sells] += quantity[sells] * execution[sells] - charged[sells]
        self.shares[sells] -= quantity[sells]
        self.cash[full_buys] = 0
        self.shares[full_sells] = 0
        self.dividends[full_buys | full_sells] = 0
        if (self.cash < -2e-8).any() or (self.shares < -2e-10).any():
            raise ValueError("Target order would borrow cash or short shares")
        self.cash[abs(self.cash) < 1e-11] = 0
        self.shares[abs(self.shares) < 1e-13] = 0
        after = self.wealth(price)
        if not np.allclose(
            self.shares[mask] * price / after[mask], targets[mask], atol=2e-11, rtol=0
        ):
            raise ValueError("Post-cost target fraction failed reconciliation")
        self.targets[mask] = targets[mask]
        new_longs = actual & (q_before == 0) & (self.shares > 0)
        new_flats = actual & (q_before > 0) & (self.shares == 0)
        self.entered_at[new_longs] = timestamp.value / 1e9
        holding = np.zeros(n)
        holding[new_flats] = (timestamp.value / 1e9 - self.entered_at[new_flats]) / 3600
        self.holds += holding
        self.trips += new_flats
        notional = quantity * price
        self.orders += actual
        self.commission += charged
        self.slippage += adverse
        self.turnover += notional / before
        if record:
            for j in np.flatnonzero(mask):
                decision = (
                    None
                    if decision_at is None or decision_at[j] == np.iinfo(np.int64).min
                    else pd.Timestamp(int(decision_at[j]), tz="UTC")
                )
                reason = "confirmed_target" if reasons is None else str(reasons[j])
                command = dict(
                    account_id=self.ids[j],
                    account_group=self.groups[j],
                    timestamp=timestamp,
                    phase=phase,
                    reason=reason,
                    target_fraction=float(targets[j]),
                    decision_at=decision,
                    decision_bar=None if decision_bar is None else int(decision_bar[j]),
                    execution_bar=execution_bar,
                    reference_price=float(price),
                    post_gap_flag=False,
                    post_gap_windows="[]",
                )
                self.command_rows.append(command)
                if actual[j]:
                    self.trade_rows.append(
                        {
                            **command,
                            "side": "buy" if buys[j] else "sell",
                            "quantity": float(quantity[j]),
                            "execution_price": float(execution[j]),
                            "reference_notional": float(notional[j]),
                            "commission": float(charged[j]),
                            "slippage": float(adverse[j]),
                            "equity_before": float(before[j]),
                            "equity_after": float(after[j]),
                            "shares_before": float(q_before[j]),
                            "shares_after": float(self.shares[j]),
                            "cash_before": float(cash_before[j]),
                            "cash_after": float(self.cash[j]),
                            "holding_hours": float(holding[j]),
                            "closed_episode": bool(new_flats[j]),
                        }
                    )
        return before, notional, charged, adverse, actual
