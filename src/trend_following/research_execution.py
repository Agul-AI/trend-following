"""Causal, fill-referenced policy execution for the frozen research rule family.

This is a *new execution version*, not a byte-for-byte reproduction of the old
raw-signal overlay backtests. Indicators retain the frozen B5 parameters, but a
trade starts only at an accepted fill. Observation close t -> decision/order
just after t -> fill at close t+1 -> first earned return ending t+2. Nanosecond
sequence markers distinguish logical event phases, not measured market latency.

``weights_at_close`` is the canonical input to a close-target portfolio ledger.
Rebalance only on target changes: a 1/3 allocation to a 3x fund is NOT a maintained
1x exposure cap. ``actual_effective_exposure`` is a separate zero-friction,
zero-cash-yield drift diagnostic; the accounting ledger is authoritative after
costs/taxes. No synthetic counterfactual financial state is imported at switches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SPEC: dict[str, Any] = {
    "allocation_multiplier": 1.0,
    "name": "b5",
    "kind": "b5",
    "product_leverage": 3.0,
    "long_ma_days": 200,
    "short_ma_days": 20,
    "medium_ma_days": 50,
    "macd_fast_days": 12,
    "macd_slow_days": 24,
    "macd_signal_days": 9,
    "entry_confirm_bars": 2,
    "exit_confirm_bars": 3,
    "exit_buffer": 0.01,
    "bear_filter_slope_days": 30,
    "bear_filter_buffer": 0.01,
    "profit_lock_scheme": ((3.0, 0.75), (4.0, 0.50)),
    "q100_activation_gain": 1.10,
    "q100_trim_weight": 0.50,
    "peak_stop_drawdown": 0.40,
    "vol_window_days": 84,
    "vol_lookback_days": 126,
    "vol_trigger": 0.95,
    "vol_recover": 0.20,
    "vol_cap_effective": 1.0,
    "cooldown_trading_days": 5,
    "max_changes_per_session": 1,
    "delay_bars": 1,
    "use_bear_filter": True,
    "use_profit_lock": True,
    "use_qtrim": True,
    "use_peak_stop": True,
}


def normalize_spec(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize documented candidate aliases, rejecting unknown policy kinds."""
    supplied = dict(spec or {})
    if "fill_delay_bars" in supplied:
        if "delay_bars" in supplied and supplied["delay_bars"] != supplied["fill_delay_bars"]:
            raise ValueError("Conflicting delay_bars and fill_delay_bars")
        supplied["delay_bars"] = supplied.pop("fill_delay_bars")
    kind = str(supplied.get("kind", supplied.get("name", "b5"))).lower()
    aliases = {
        "b5_recover20_then_5_trading_day_cooldown": "b5",
        "vol_state_w84_lb126_trig95_rec20_cap1x": "anchor",
        "simpleqqq200ma1x": "trend_1x",
        "simple_qqq_200ma_1x": "trend_1x",
        "qqq_200ma_1x": "trend_1x",
        "simpleqqq200ma2x": "trend_2x",
        "simple_qqq_200ma_2x": "trend_2x",
        "qqq_200ma_2x": "trend_2x",
        "buy_and_hold": "buyhold",
        "buy_hold": "buyhold",
    }
    kind = aliases.get(kind, kind)
    if kind not in {"b5", "anchor", "trend_1x", "trend_2x", "cash", "buyhold"}:
        raise ValueError(f"Unknown research execution policy: {kind}")
    out = {**DEFAULT_SPEC, **supplied, "kind": kind}
    out["name"] = str(supplied.get("name", kind))
    if kind == "anchor" and "cooldown_trading_days" not in supplied:
        out["cooldown_trading_days"] = 0
    if kind.startswith("trend_"):
        out["effective_leverage"] = float(supplied.get("effective_leverage", kind[-2]))
    if kind == "buyhold":
        out["effective_leverage"] = float(supplied.get("effective_leverage", 3.0))
    if not 0 <= float(out["allocation_multiplier"]) <= 1:
        raise ValueError("allocation_multiplier must be between zero and one")
    if float(out["product_leverage"]) <= 0:
        raise ValueError("product_leverage must be positive")
    if int(out["delay_bars"]) < 1 or int(out["max_changes_per_session"]) < 1:
        raise ValueError("delay_bars and max_changes_per_session must be >= 1")
    if int(out["cooldown_trading_days"]) < 0:
        raise ValueError("cooldown_trading_days must be non-negative")
    if not 0 <= out["vol_recover"] <= out["vol_trigger"] <= 1:
        raise ValueError("Require 0 <= recover <= trigger <= 1")
    if not 0 < out["peak_stop_drawdown"] < 1:
        raise ValueError("peak_stop_drawdown must be between zero and one")
    leverage = float(out.get("effective_leverage", out["product_leverage"]))
    if not 0 <= leverage <= float(out["product_leverage"]):
        raise ValueError("Unfunded leverage is not supported")
    return out


def causal_features(
    qqq_prices: pd.Series, spec: Mapping[str, Any] | None = None, *, bars_per_day: int = 6
) -> pd.DataFrame:
    """Trailing SMA-MACD and B5 features; full windows, no future data fills."""
    p = normalize_spec(spec)
    if bars_per_day < 1:
        raise ValueError("bars_per_day must be positive")
    price = qqq_prices.astype(float)

    def window(key: str) -> int:
        value = int(round(float(p[key]) * bars_per_day))
        if value < 1:
            raise ValueError(f"{key} gives an empty window")
        return value

    def ma(key: str) -> pd.Series:
        n = window(key)
        return price.rolling(n, min_periods=n).mean()

    if window("macd_fast_days") >= window("macd_slow_days"):
        raise ValueError("MACD fast window must be shorter than slow")
    long, short, medium = ma("long_ma_days"), ma("short_ma_days"), ma("medium_ma_days")
    macd = ma("macd_fast_days") - ma("macd_slow_days")
    signal_n = window("macd_signal_days")
    hist = macd - macd.rolling(signal_n, min_periods=signal_n).mean()
    returns = price.pct_change(fill_method=None).fillna(0.0)
    vol_n, pct_n = window("vol_window_days"), window("vol_lookback_days")
    vol = returns.rolling(vol_n, min_periods=vol_n).std(ddof=0)
    pct = vol.rolling(pct_n, min_periods=pct_n).rank(method="max", pct=True)
    entry = (hist > 0) & (price > long)
    exit_condition = price < long * (1 - float(p["exit_buffer"]))
    entry_n, exit_n = int(p["entry_confirm_bars"]), int(p["exit_confirm_bars"])
    if min(entry_n, exit_n) < 1:
        raise ValueError("Confirmation windows must be positive")
    slope_n = window("bear_filter_slope_days")
    return pd.DataFrame(
        {
            "long_ma": long,
            "short_ma": short,
            "medium_ma": medium,
            "macd_hist": hist,
            "distance": price / long - 1,
            "long_slope": long.diff(slope_n),
            "medium_slope": medium.diff(slope_n),
            "vol_percentile": pct,
            "entry": entry.rolling(entry_n, min_periods=entry_n).sum().eq(entry_n),
            "exit": exit_condition.rolling(exit_n, min_periods=exit_n).sum().eq(exit_n),
        },
        index=price.index,
    )


@dataclass
class ExecutionPath:
    target_weights: pd.Series
    weights_at_close: pd.Series
    events: pd.DataFrame
    decisions: pd.DataFrame
    effective_target_exposure: pd.Series
    actual_effective_exposure: pd.Series
    metadata: dict[str, Any]


class ExecutionPolicyEngine:
    """One chronological rule engine; call ``step`` exactly once per close.

    ``switch_candidate`` cancels unfilled old-rule orders but retains accepted
    holdings and all open-trade entry/peak/profit-lock/qtrim reference state.
    Base-regime and volatility/cooldown state also persist; a switch does not
    simulate an imaginary earlier history for the new candidate. Trade-reference
    states reset on a genuinely filled exit. All changes, including exits, are
    subject to the declared fill-session limit; no hidden urgent-exit override.
    """

    def __init__(
        self,
        spec: Mapping[str, Any] | None = None,
        *,
        bars_per_day: int = 6,
        slippage_bps: float = 0.0,
        transaction_cost_bps: float = 0.0,
    ):
        self.spec = normalize_spec(spec)
        self.bars_per_day = bars_per_day
        self.slippage_bps = float(slippage_bps)
        self.transaction_cost_bps = float(transaction_cost_bps)
        if min(slippage_bps, transaction_cost_bps) < 0 or bars_per_day < 1:
            raise ValueError("Nonnegative costs and positive bars_per_day required")
        self.bar = -1
        self.last_timestamp: pd.Timestamp | None = None
        self.last_session: Any = None
        self.session_fills = 0
        self.weight = 0.0
        self.gross_holding = 0.0
        self.gross_cash = 1.0
        self.previous_risk_price: float | None = None
        self.pending: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.order_id = 0
        self.trade_id = 0
        self.base_long = False
        self.stopped = False
        self._reset_trade()

    def _reset_trade(self) -> None:
        self.entry_price = np.nan
        self.entry_timestamp = pd.NaT
        self.peak_price = np.nan
        self.profit_weight = 1.0
        self.distance_max = -np.inf
        self.qtrim_threshold = np.nan
        self.qtrim_defensive = False
        self.vol_defensive = False
        self.cooldown_remaining = 0
        self.recovery_seen = False

    def _event(
        self, event: str, timestamp: pd.Timestamp, order: Mapping[str, Any], **extra: Any
    ) -> None:
        self.events.append(
            {
                "event": event,
                "event_timestamp": timestamp,
                "order_id": order["order_id"],
                "candidate": order["candidate"],
                "observation_timestamp": order["observation_timestamp"],
                "decision_timestamp": order["decision_timestamp"],
                "order_timestamp": order["order_timestamp"],
                "fill_timestamp": pd.NaT,
                "return_timestamp": pd.NaT,
                "target_weight": order["target_weight"],
                "reason": order["reason"],
                **extra,
            }
        )

    def switch_candidate(
        self, spec: Mapping[str, Any], timestamp: pd.Timestamp | None = None
    ) -> None:
        """Activate a selected policy without importing counterfactual trades."""
        new = normalize_spec(spec)
        if new["product_leverage"] != self.spec["product_leverage"]:
            raise ValueError("Cannot change traded product within one policy account")
        if self.pending is not None:
            if timestamp is None:
                raise ValueError("timestamp required when cancelling a pending order")
            self._event("cancelled_candidate_switch", pd.Timestamp(timestamp), self.pending)
            self.pending = None
        if self.vol_defensive and self.recovery_seen:
            elapsed = (
                int(self.spec["cooldown_trading_days"] * self.bars_per_day)
                - self.cooldown_remaining
            )
            self.cooldown_remaining = max(
                int(new["cooldown_trading_days"] * self.bars_per_day) - elapsed, 0
            )
            if self.cooldown_remaining == 0:
                self.vol_defensive = False
        self.spec = new

    def _vol_cap(self, percentile: float, active: bool) -> float:
        p = self.spec
        if not active:
            self.vol_defensive = False
            self.cooldown_remaining = 0
            self.recovery_seen = False
        elif not self.vol_defensive and np.isfinite(percentile) and percentile >= p["vol_trigger"]:
            self.vol_defensive = True
            self.recovery_seen = False
            self.cooldown_remaining = 0
        elif self.vol_defensive:
            if (
                np.isfinite(percentile)
                and percentile <= p["vol_recover"]
                and not self.recovery_seen
            ):
                self.recovery_seen = True
                self.cooldown_remaining = int(p["cooldown_trading_days"] * self.bars_per_day)
                if not self.cooldown_remaining:
                    self.vol_defensive = False
            # Match the frozen B5 convention: recovery bar is cooldown bar one.
            if self.vol_defensive and self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
                if self.cooldown_remaining == 0:
                    self.vol_defensive = False
        return float(p["vol_cap_effective"] / p["product_leverage"] if self.vol_defensive else 1)

    def _desired(self, qqq: float, risk: float, f: Mapping[str, Any]) -> tuple[float, str, bool]:
        p = self.spec
        kind = p["kind"]
        if not self.base_long and bool(f["entry"]):
            self.base_long = True
        elif self.base_long and bool(f["exit"]):
            self.base_long = False
        if not self.base_long:
            self.stopped = False
        if kind == "cash":
            return 0.0, "cash_policy", False
        if kind == "buyhold":
            return p["effective_leverage"] / p["product_leverage"], "buyhold", False
        if kind.startswith("trend_"):
            desired = p["effective_leverage"] / p["product_leverage"] if qqq > f["long_ma"] else 0
            return float(desired), "simple_200ma", False
        blocked = False
        if self.weight == 0 and self.base_long and p["use_bear_filter"]:
            bear = np.isfinite(f["long_slope"]) and f["long_slope"] < 0
            confirmation = (
                f["distance"] >= p["bear_filter_buffer"]
                and f["medium_slope"] > 0
                and f["short_ma"] > f["medium_ma"]
            )
            blocked = bool(bear and not confirmation)
        if self.weight > 0 and np.isfinite(self.entry_price):
            self.peak_price = max(self.peak_price, risk)
            gain = risk / self.entry_price - 1
            if p["use_peak_stop"] and risk / self.peak_price - 1 <= -p["peak_stop_drawdown"]:
                self.stopped = True
            if p["use_profit_lock"]:
                for threshold, weight in p["profit_lock_scheme"]:
                    if gain >= threshold:
                        self.profit_weight = min(self.profit_weight, float(weight))
            if p["use_qtrim"]:
                if not np.isfinite(self.qtrim_threshold) and np.isfinite(f["distance"]):
                    self.distance_max = max(self.distance_max, float(f["distance"]))
                    if gain >= p["q100_activation_gain"]:
                        self.qtrim_threshold = self.distance_max
                if self.qtrim_defensive and qqq <= f["short_ma"]:
                    self.qtrim_defensive = False
                if (
                    np.isfinite(self.qtrim_threshold)
                    and gain >= p["q100_activation_gain"]
                    and f["distance"] >= self.qtrim_threshold
                ):
                    self.qtrim_defensive = True
        active = self.base_long and not self.stopped and not blocked
        vol_cap = self._vol_cap(float(f["vol_percentile"]), active)
        if not active:
            reason = (
                "bear_entry_blocked" if blocked else ("peak_stop" if self.stopped else "base_exit")
            )
            return 0.0, reason, blocked
        cap = min(
            self.profit_weight if p["use_profit_lock"] else 1.0,
            p["q100_trim_weight"] if self.qtrim_defensive and p["use_qtrim"] else 1.0,
            vol_cap,
        )
        return float(cap), "active_overlay_allocation", False

    def step(
        self,
        timestamp: pd.Timestamp,
        qqq_price: float,
        risk_price: float,
        features: Mapping[str, Any],
        *,
        session: Any = None,
    ) -> dict[str, Any]:
        """Mark old holdings, fill eligible prior order, then observe/decide."""
        ts = pd.Timestamp(timestamp)
        if self.last_timestamp is not None and ts <= self.last_timestamp:
            raise ValueError("step timestamps must increase strictly")
        if not np.isfinite([qqq_price, risk_price]).all() or min(qqq_price, risk_price) <= 0:
            raise ValueError("Prices must be positive and finite")
        self.bar += 1
        session = _session(ts) if session is None else session
        if session != self.last_session:
            self.session_fills = 0
            self.last_session = session
        if self.previous_risk_price is not None:
            self.gross_holding *= risk_price / self.previous_risk_price
        interval_weight = self.weight
        if self.pending is not None and self.bar >= self.pending["due_bar"]:
            if self.session_fills < int(self.spec["max_changes_per_session"]):
                order = self.pending
                before, self.weight = self.weight, float(order["target_weight"])
                equity = self.gross_holding + self.gross_cash
                self.gross_holding, self.gross_cash = (
                    equity * self.weight,
                    equity * (1 - self.weight),
                )
                side = 1 if self.weight > before else -1
                fill_quote = risk_price * (1 + side * self.slippage_bps / 10000)
                if before == 0 and self.weight > 0:
                    # Vol regime is causal signal state, not a fictional trade.
                    # Keep a trigger established by the accepted entry decision.
                    vol_state = (self.vol_defensive, self.cooldown_remaining, self.recovery_seen)
                    self._reset_trade()
                    self.vol_defensive, self.cooldown_remaining, self.recovery_seen = vol_state
                    self.trade_id += 1
                    self.entry_price = risk_price * (
                        1 + (self.slippage_bps + self.transaction_cost_bps) / 10000
                    )
                    self.entry_timestamp = ts
                    self.peak_price = risk_price
                elif self.weight == 0:
                    self._reset_trade()
                self.session_fills += 1
                self._event(
                    "filled",
                    ts,
                    order,
                    fill_timestamp=ts,
                    fill_session=session,
                    fill_quote=fill_quote,
                    trade_id=self.trade_id,
                    previous_weight=before,
                )
                self.pending = None
            else:
                self._event("deferred_session_limit", ts, self.pending, fill_session=session)
        if self.weight > 0 and np.isfinite(self.peak_price):
            self.peak_price = max(self.peak_price, risk_price)
        desired, reason, blocked = self._desired(float(qqq_price), float(risk_price), features)
        desired *= float(self.spec["allocation_multiplier"])
        if self.pending is not None and not np.isclose(desired, self.pending["target_weight"]):
            self._event("cancelled_replaced", ts + pd.Timedelta(nanoseconds=1), self.pending)
            self.pending = None
        if self.pending is None and not np.isclose(desired, self.weight):
            self.order_id += 1
            self.pending = {
                "order_id": self.order_id,
                "candidate": self.spec["name"],
                "observation_timestamp": ts,
                "decision_timestamp": ts + pd.Timedelta(nanoseconds=1),
                "order_timestamp": ts + pd.Timedelta(nanoseconds=2),
                "target_weight": desired,
                "reason": reason,
                "due_bar": self.bar + int(self.spec["delay_bars"]),
            }
            self._event("submitted", self.pending["order_timestamp"], self.pending)
        self.previous_risk_price = risk_price
        self.last_timestamp = ts
        equity = self.gross_holding + self.gross_cash
        return {
            "timestamp": ts,
            "candidate": self.spec["name"],
            "desired_weight": desired,
            "target_weights": interval_weight,
            "weights_at_close": self.weight,
            "effective_target_exposure": self.weight * self.spec["product_leverage"],
            "actual_effective_exposure": self.gross_holding
            / equity
            * self.spec["product_leverage"],
            "reason": reason,
            "blocked_entry": blocked,
            "base_long": self.base_long,
            "entry_price": self.entry_price,
            "entry_timestamp": self.entry_timestamp,
            "trade_id": self.trade_id if self.weight > 0 else 0,
            "peak_price": self.peak_price,
            "profit_weight": self.profit_weight,
            "qtrim_threshold": self.qtrim_threshold,
            "qtrim_defensive": self.qtrim_defensive,
            "vol_defensive": self.vol_defensive,
            "cooldown_remaining": self.cooldown_remaining,
            "observation_timestamp": ts,
            "decision_timestamp": ts + pd.Timedelta(nanoseconds=1),
        }


def _session(timestamp: pd.Timestamp) -> Any:
    local = timestamp.tz_convert("America/New_York") if timestamp.tzinfo is not None else timestamp
    return local.date()


def build_execution_path(
    qqq_prices: pd.Series,
    risk_prices: pd.Series,
    spec: Mapping[str, Any] | None = None,
    *,
    bars_per_day: int = 6,
    evaluate_start: str | pd.Timestamp | None = None,
    candidate_schedule: Mapping[pd.Timestamp, Mapping[str, Any]] | None = None,
    session_labels: pd.Series | None = None,
    slippage_bps: float = 0.0,
    transaction_cost_bps: float = 0.0,
) -> ExecutionPath:
    """Build one flat-start, continuously selected executed policy path.

    Both price series must have the same monotone, unique observation timestamps;
    do not silently drop missing bars. Six bars/day is the canonical historical
    *indicator window convention*, not a promise every session has six bars.
    ``session_labels`` permits an audited exchange-calendar mapping. Existing
    history warms indicators only, never fictitious positions or trade gains.
    Schedule entries take effect at their first observation timestamp or later.
    """
    index = qqq_prices.index
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError("Nonempty DatetimeIndex required")
    if (
        not index.is_monotonic_increasing
        or not index.is_unique
        or not index.equals(risk_prices.index)
    ):
        raise ValueError("Price indices must be identical, sorted and unique")
    if not np.isfinite(qqq_prices).all() or not np.isfinite(risk_prices).all():
        raise ValueError("No missing/nonfinite price bars permitted")
    if (qqq_prices <= 0).any() or (risk_prices <= 0).any():
        raise ValueError("Prices must be positive")
    if session_labels is not None and (
        not session_labels.index.equals(index) or session_labels.isna().any()
    ):
        raise ValueError("session_labels must cover the complete price index")
    start = pd.Timestamp(evaluate_start) if evaluate_start is not None else index[0]
    if index.tz is not None and start.tzinfo is None:
        start = start.tz_localize(index.tz)
    schedule = sorted(
        (pd.Timestamp(k), normalize_spec(v)) for k, v in (candidate_schedule or {}).items()
    )
    schedule = [
        (t.tz_localize(index.tz) if index.tz is not None and t.tzinfo is None else t, p)
        for t, p in schedule
    ]
    initial = normalize_spec(spec)
    all_specs = [initial, *(p for _, p in schedule)]
    frames: dict[str, pd.DataFrame] = {}
    for p in all_specs:
        key = repr(sorted(p.items()))
        if key not in frames:
            frames[key] = causal_features(qqq_prices, p, bars_per_day=bars_per_day)
    engine = ExecutionPolicyEngine(
        initial,
        bars_per_day=bars_per_day,
        slippage_bps=slippage_bps,
        transaction_cost_bps=transaction_cost_bps,
    )
    rows: list[dict[str, Any]] = []
    schedule_i = 0
    for ts in index[index >= start]:
        while schedule_i < len(schedule) and schedule[schedule_i][0] <= ts:
            engine.switch_candidate(schedule[schedule_i][1], ts)
            schedule_i += 1
        feature = frames[repr(sorted(engine.spec.items()))].loc[ts]
        rows.append(
            engine.step(
                ts,
                float(qqq_prices.loc[ts]),
                float(risk_prices.loc[ts]),
                feature,
                session=None if session_labels is None else session_labels.loc[ts],
            )
        )
    if not rows:
        raise ValueError("evaluate_start is beyond the supplied observations")
    decisions = pd.DataFrame(rows).set_index("timestamp")
    events = pd.DataFrame(
        engine.events,
        columns=[
            "event",
            "event_timestamp",
            "order_id",
            "candidate",
            "observation_timestamp",
            "decision_timestamp",
            "order_timestamp",
            "fill_timestamp",
            "return_timestamp",
            "target_weight",
            "reason",
            "fill_session",
            "fill_quote",
            "trade_id",
            "previous_weight",
        ],
    )
    events["return_timestamp"] = pd.Series(pd.NaT, index=events.index, dtype=index.dtype)
    if not events.empty:
        next_timestamp = dict(zip(index[:-1], index[1:], strict=True))
        fills = events["event"].eq("filled")
        events.loc[fills, "return_timestamp"] = events.loc[fills, "fill_timestamp"].map(
            next_timestamp
        )
    return ExecutionPath(
        target_weights=decisions["target_weights"].rename("risk_weight"),
        weights_at_close=decisions["weights_at_close"].rename("risk_weight"),
        events=events,
        decisions=decisions,
        effective_target_exposure=decisions["effective_target_exposure"],
        actual_effective_exposure=decisions["actual_effective_exposure"],
        metadata={
            "execution_version": "fill_referenced_v1",
            "targets_at": "close",
            "rebalancing": "transition_only",
            "bars_per_day": bars_per_day,
            "trade_reference": "first_accepted_fill_quote_plus_entry_fee",
            "switch_state": "preserve_open_trade_and_policy_state_cancel_pending_orders",
            "drift_diagnostic": "gross_no_costs_no_cash_yield_not_accounting_ledger",
            "daily_limit": "all_discretionary_changes_including_exits",
            "costs_affect_trade_reference": {
                "slippage_bps": slippage_bps,
                "transaction_cost_bps": transaction_cost_bps,
            },
        },
    )
