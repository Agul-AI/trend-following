"""Independent finalized-daily trend gates around immutable MR signal features.

The new kernel adapts the frozen MR event ledger without modifying that source.
Daily gates advance at the provider quote's assumed 17:00 New York availability;
only completed half-hour observations advance confirmation counters. This module
opens no files, consumes only supplied bounded packets, and has no OOS route.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from trend_following import research_daily_hourly_v5 as core
from trend_following import research_mean_reversion as features
from trend_following import research_mean_reversion_engine as mr_engine
from trend_following import research_portfolio_engine as portfolio
from trend_following import research_quality_engine_v5 as quality
from trend_following.research_rates import rate_accrual

# Read-only reuse of coverage/calendar/post-gap helpers; never monkeypatch them.
_post_gap_context = quality._post_gap_context


GATE_PERIODS = (100, 200)
EXIT_POLICIES = ("entry_only", "recovery_or_trend_break")
UNKNOWN, OFF, ON = -1, 0, 1


@dataclass(frozen=True)
class GatedCandidate:
    base: features.Candidate
    gate_period: int
    exit_policy: str

    def __post_init__(self):
        if not isinstance(self.base, features.Candidate):
            raise ValueError("An unchanged mean-reversion candidate is required")
        if not features._positive_integer(self.gate_period) or self.gate_period not in GATE_PERIODS:
            raise ValueError("Gate period must be 100 or 200 daily sessions")
        if self.exit_policy not in EXIT_POLICIES:
            raise ValueError("Unknown frozen exit policy")
        object.__setattr__(self, "gate_period", int(self.gate_period))

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def basket_id(self):
        return f"multi_scale_sma{self.gate_period}_{self.exit_policy}"

    @property
    def candidate_id(self):
        return f"{self.basket_id}__{self.base.candidate_id}"

    def to_dict(self):
        return dict(
            base=self.base.to_dict(), gate_period=self.gate_period, exit_policy=self.exit_policy
        )

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"base", "gate_period", "exit_policy"}:
            raise ValueError("Gated candidate requires the explicit unchanged base and policy")
        return cls(
            features.Candidate.from_dict(value["base"]), value["gate_period"], value["exit_policy"]
        )


@dataclass(frozen=True)
class GateOnlyCandidate:
    gate_period: int

    def __post_init__(self):
        if not features._positive_integer(self.gate_period) or self.gate_period not in GATE_PERIODS:
            raise ValueError("Gate-only period must be 100 or 200 daily sessions")
        object.__setattr__(self, "gate_period", int(self.gate_period))

    @property
    def candidate_id(self):
        return f"multi_scale_sma{self.gate_period}_only"

    @property
    def basket_id(self):
        return self.candidate_id

    @property
    def rule(self):
        # Read-only feature placeholder: gate-only predicates never inspect it.
        return features.Rule("SMA_DISTANCE", 10)

    family = "GATE_ONLY"
    entry_threshold = np.nan
    exit_threshold = np.nan
    confirmation_hours = 3.0
    required_checks = 7
    exit_policy = "gate_only"

    def to_dict(self):
        return dict(gate_period=self.gate_period)

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"gate_period"}:
            raise ValueError("Gate-only candidate requires one daily period")
        return cls(value["gate_period"])


def candidate_grid(mean_reversion_candidates):
    members = tuple(mean_reversion_candidates)
    if (
        len(members) != 32
        or not all(isinstance(c, features.Candidate) for c in members)
        or len({c.candidate_id for c in members}) != 32
    ):
        raise ValueError("Exactly 32 unchanged unique mean-reversion members are required")
    members = tuple(sorted(members, key=lambda c: c.candidate_id))
    return tuple(
        GatedCandidate(base, period, policy)
        for policy in EXIT_POLICIES
        for period in GATE_PERIODS
        for base in members
    )


def gate_only_candidates():
    return tuple(GateOnlyCandidate(period) for period in GATE_PERIODS)


def finalized_gates(values, periods=GATE_PERIODS):
    """Strict SMA gates; nonfinite/incomplete windows remain UNKNOWN.

    Centering at today's value prevents flat non-binary prices from manufacturing
    an ON gate through roundoff. Each row includes its own finalized observation.
    """
    series = pd.Series(values, copy=True, dtype=float)
    if (
        not periods
        or len(set(periods)) != len(periods)
        or any(not features._positive_integer(x) for x in periods)
    ):
        raise ValueError("Unique positive daily gate periods required")
    result = pd.DataFrame(UNKNOWN, index=series.index, columns=periods, dtype=np.int8)
    array = series.to_numpy()
    for period in periods:
        for i in range(period - 1, len(array)):
            window = array[i - period + 1 : i + 1]
            if np.isfinite(window).all() and (window > 0).all():
                result.iloc[i, result.columns.get_loc(period)] = (
                    ON if np.mean(window - window[-1]) < 0 else OFF
                )
    return result


@dataclass(frozen=True)
class GateCache:
    base: features.MeanReversionCache
    gates: pd.DataFrame
    available_at: pd.DatetimeIndex
    gate_fingerprint: str = ""
    gate_override: str | None = None

    def __getattr__(self, name):
        return getattr(self.base, name)


def _gate_fingerprint(cache):
    digest = hashlib.sha256()
    digest.update(cache.base.feature_fingerprint.encode())
    digest.update(
        json.dumps(
            [list(cache.gates.index), list(cache.gates.columns), cache.gate_override]
        ).encode()
    )
    digest.update(np.ascontiguousarray(cache.gates.to_numpy(dtype=np.int8)).tobytes())
    digest.update(np.ascontiguousarray(cache.available_at.asi8).tobytes())
    return digest.hexdigest()


def build_cache(
    daily, bars, calendar, *, coverage=None, blackouts=None, quality_policy=quality.DEFAULT_POLICY
):
    base = mr_engine.build_cache(
        daily, bars, calendar, coverage=coverage, blackouts=blackouts, quality_policy=quality_policy
    )
    gates = finalized_gates(base.finalized_total_return_index)
    available = pd.DatetimeIndex(
        [quality._valuation_clock(session, base.policy) for session in base.daily.session]
    )
    cache = GateCache(base, gates, available)
    return replace(cache, gate_fingerprint=_gate_fingerprint(cache))


def with_constant_gate(cache, state):
    """Synthetic/preflight identity override; production protocols must reject it."""
    _validate_cache(cache)
    if isinstance(state, (bool, np.bool_)) or state not in {UNKNOWN, OFF, ON}:
        raise ValueError("Explicit UNKNOWN=-1, OFF=0 or ON=1 gate state required")
    result = replace(
        cache, gates=cache.gates * 0 + state, gate_override=f"constant_{state}", gate_fingerprint=""
    )
    return replace(result, gate_fingerprint=_gate_fingerprint(result))


def _validate_cache(cache):
    if not isinstance(cache, GateCache):
        raise ValueError("Explicit multi-scale GateCache required")
    mr_engine._validate_cache(cache.base)
    expected = pd.DatetimeIndex(
        [quality._valuation_clock(x, cache.policy) for x in cache.daily.session]
    )
    if (
        list(cache.gates.columns) != list(GATE_PERIODS)
        or not cache.gates.index.equals(cache.finalized_total_return_index.index)
        or not cache.available_at.equals(expected)
        or not np.isin(cache.gates.to_numpy(), [UNKNOWN, OFF, ON]).all()
    ):
        raise ValueError("Gate history, states or finalized daily availability changed")
    if not cache.gate_fingerprint or cache.gate_fingerprint != _gate_fingerprint(cache):
        raise ValueError("Cached finalized gate identity changed")


def _batch(
    cache,
    calendar,
    events,
    clock,
    terminal_index,
    candidates,
    cash_observations,
    commission_bps,
    slippage_bps,
    initial_cash,
    *,
    mode="candidate",
    record=False,
    clock_returns=None,
    post_gap_context=None,
    post_gap_event_budget=None,
):
    fee, slip = core._costs(commission_bps, slippage_bps, initial_cash)
    if "series_id" in cash_observations and not cash_observations.series_id.eq("DTB3").all():
        raise ValueError("Cash observations must be causal DTB3")
    accrued = (
        rate_accrual(clock, cash_observations, interval_start=clock[0])
        if clock_returns is None
        else clock_returns
    )
    if (accrued <= -1).any():
        raise ValueError("Cash wealth exhausted")
    factor = dict(zip(clock, accrued.to_numpy(), strict=True))
    n = len(candidates)
    ids = [c.candidate_id for c in candidates] if mode == "candidate" else [f"benchmark_{mode}"]
    lookup = {r: i for i, r in enumerate(cache.rules)}
    indices = np.asarray([lookup[c.rule] for c in candidates])
    entry_thresholds = np.asarray([c.entry_threshold for c in candidates])
    exit_thresholds = np.asarray([c.exit_threshold for c in candidates])
    required = np.asarray([c.required_checks for c in candidates])
    gate_columns = np.asarray([cache.gates.columns.get_loc(c.gate_period) for c in candidates])
    gate_only = np.asarray([isinstance(c, GateOnlyCandidate) for c in candidates])
    break_enabled = np.asarray(
        [c.exit_policy in {"recovery_or_trend_break", "gate_only"} for c in candidates]
    )
    gate = np.full(n, UNKNOWN, dtype=np.int8)
    initial_gate = cache.available_at.searchsorted(clock[0], side="right") - 1
    if initial_gate >= 0:
        gate[:] = cache.gates.iloc[initial_gate].to_numpy()[gate_columns]
    last_gate_session = cache.daily.session.iloc[initial_gate] if initial_gate >= 0 else None
    previous_sessions = dict(
        zip(cache.calendar.session.iloc[1:], cache.calendar.session.iloc[:-1], strict=True)
    )
    break_count = np.zeros(n, dtype=int)
    pending_causes = np.zeros(n, dtype=np.int8)
    blocked_oversold = np.zeros(n, dtype=int)
    trend_break_exits = np.zeros(n, dtype=int)
    entry_cancellations = np.zeros(n, dtype=int)
    exit_cancellations = np.zeros(n, dtype=int)
    unknown_resets = np.zeros(n, dtype=int)
    cash = np.full(n, initial_cash, dtype=float)
    gross_cash = cash.copy()
    shares = np.zeros(n)
    gross_shares = np.zeros(n)
    dividend_cash = np.zeros(n)
    gross_dividend_cash = np.zeros(n)
    long = np.zeros(n, dtype=bool)
    pending = np.zeros(n, dtype=np.int8)
    entry_count = np.zeros(n, dtype=int)
    exit_count = np.zeros(n, dtype=int)
    pending_at = np.full(n, np.nan)
    entered_at = np.full(n, np.nan)
    commission = np.zeros(n)
    slippage = np.zeros(n)
    turnover = np.zeros(n)
    trades = np.zeros(n, dtype=int)
    orders = np.zeros(n, dtype=int)
    holds = np.zeros(n)
    held_seconds = np.zeros(n)
    cancelled = np.zeros(n, dtype=int)
    terminal_cancelled = np.zeros(n, dtype=int)
    cash_reference = initial_cash
    previous = clock[0]
    daily = {row.session: row for row in cache.daily.itertuples(index=False)}
    daily_positions = dict(zip(cache.daily.session, range(len(cache.daily)), strict=True))
    day_equity = []
    day_gross = []
    day_cash = []
    day_exposure = []
    day_times = []
    trade_rows = []
    diagnostics = []
    benchmark_bought = False
    post_enabled = mode == "candidate" and cache.policy.post_gap_diagnostics
    if post_enabled and post_gap_context is None:
        post_gap_context = _post_gap_context(cache, calendar)
    post_buy = np.zeros(n, dtype=np.int64)
    post_sell = np.zeros(n, dtype=np.int64)
    post_rows = []
    post_total_rows = 0
    post_limit = (
        cache.policy.post_gap_event_limit
        if post_gap_event_budget is None
        else max(0, post_gap_event_budget)
    )

    def fill(mask, side, price, timestamp, session, reason):
        if not mask.any():
            return
        equity = cash + shares * price
        execution = price * (1 + slip if side == "buy" else 1 - slip)
        if side == "buy":
            qty = cash[mask] / (execution * (1 + fee))
            charged = qty * execution * fee
            cash[mask] = 0
            dividend_cash[mask] = 0
            shares[mask] = qty
            gross_shares[mask] = gross_cash[mask] / price
            gross_cash[mask] = 0
            gross_dividend_cash[mask] = 0
            long[mask] = True
            entered_at[mask] = timestamp.value / 1e9
            holding = np.zeros(mask.sum())
        else:
            qty = shares[mask].copy()
            charged = qty * execution * fee
            cash[mask] += qty * execution - charged
            shares[mask] = 0
            dividend_cash[mask] = 0
            gross_cash[mask] += gross_shares[mask] * price
            gross_shares[mask] = 0
            gross_dividend_cash[mask] = 0
            long[mask] = False
            holding = (timestamp.value / 1e9 - entered_at[mask]) / 3600
            holds[mask] += holding
            trades[mask] += 1
        if side == "sell" and reason != "terminal_liquidation":
            trend_break_exits[mask] += (pending_causes[mask] & 2) != 0
        notional = qty * price
        adverse = qty * abs(execution - price)
        commission[mask] += charged
        slippage[mask] += adverse
        turnover[mask] += notional / equity[mask]
        orders[mask] += 1
        if record:
            for k, j in enumerate(np.flatnonzero(mask)):
                decision = (
                    None
                    if np.isnan(pending_at[j])
                    else pd.Timestamp(pending_at[j], unit="s", tz="UTC")
                )
                trade_rows.append(
                    dict(
                        candidate_id=ids[j],
                        side=side,
                        decision_at=decision,
                        timestamp=timestamp,
                        session=session,
                        reason=reason,
                        reference_price=price,
                        execution_price=execution,
                        quantity=qty[k],
                        reference_notional=notional[k],
                        commission=charged[k],
                        slippage=adverse[k],
                        equity_before=equity[j],
                        holding_hours=holding[k],
                        exit_causes=(
                            ""
                            if reason == "terminal_liquidation"
                            else "recovery+trend_break"
                            if pending_causes[j] == 3
                            else "recovery"
                            if pending_causes[j] == 1
                            else "trend_break"
                            if pending_causes[j] == 2
                            else ""
                        ),
                    )
                )
        pending[mask] = 0
        pending_at[mask] = np.nan
        entry_count[mask] = 0
        exit_count[mask] = 0
        break_count[mask] = 0
        pending_causes[mask] = 0

    def update_gate(values, *, stale=False):
        previous_gate = gate.copy()
        previous_causes = pending_causes.copy()
        previous_pending = pending.copy()
        gate[:] = values
        unknown = gate == UNKNOWN
        entry_cancel = (pending == 1) & (gate != ON)
        exit_cancel = (pending == -1) & (unknown | ((gate == ON) & (pending_causes == 2)))
        entry_cancellations[:] += entry_cancel
        exit_cancellations[:] += exit_cancel
        cancel = entry_cancel | exit_cancel
        pending[cancel] = 0
        pending_at[cancel] = np.nan
        pending_causes[cancel] = 0
        # Interruptions reset streaks without counting daily availability as a check.
        entry_count[gate != ON] = 0
        break_count[gate != OFF] = 0
        exit_count[unknown] = 0
        unknown_resets[:] += unknown
        if record:
            for j in np.flatnonzero(cancel):
                diagnostics.append(
                    dict(
                        timestamp=timestamp,
                        event="gate_order_cancelled",
                        candidate_id=ids[j],
                        side="buy" if previous_pending[j] == 1 else "sell",
                        previous_gate_state=int(previous_gate[j]),
                        gate_state=int(gate[j]),
                        exit_cause_bits=int(previous_causes[j]),
                        reason="stale"
                        if stale
                        else "unknown"
                        if unknown[j]
                        else "off_entry_gate"
                        if entry_cancel[j]
                        else "on_cancelled_trend_break",
                    )
                )
            diagnostics.append(
                dict(
                    timestamp=timestamp,
                    event="gate_stale" if stale else "gate_update",
                    gate_state=int(gate[0]),
                    entry_count=int(entry_count[0]),
                    exit_count=int(exit_count[0]),
                    trend_break_count=int(break_count[0]),
                    position=int(long[0]),
                    pending=int(pending[0]),
                    candidate_id=ids[0],
                )
            )

    for timestamp, _, kind, item in events:
        if timestamp != previous:
            elapsed = (timestamp - previous).total_seconds()
            held_seconds += long * elapsed
            cash *= 1 + factor[timestamp]
            gross_cash *= 1 + factor[timestamp]
            dividend_cash *= 1 + factor[timestamp]
            gross_dividend_cash *= 1 + factor[timestamp]
            cash_reference *= 1 + factor[timestamp]
            previous = timestamp
        if kind == "blackout_start":
            cancelled += pending != 0
            pending[:] = 0
            pending_at[:] = np.nan
            entry_count[:] = 0
            exit_count[:] = 0
            break_count[:] = 0
            pending_causes[:] = 0
            if record:
                diagnostics.append(
                    dict(
                        timestamp=timestamp,
                        event="blackout_start",
                        reason=cache.blackouts.iloc[item].reason,
                        entry_count=0,
                        exit_count=0,
                        position=int(long[0]),
                        pending=0,
                    )
                )
        elif kind == "actions":
            action = daily[item]
            shares *= action.split_coefficient
            gross_shares *= action.split_coefficient
            distribution = shares * action.dividend_amount
            gross_distribution = gross_shares * action.dividend_amount
            cash += distribution
            dividend_cash += distribution
            gross_cash += gross_distribution
            gross_dividend_cash += gross_distribution
        elif kind == "bar_open":
            row = cache.bar_rows[item]
            expected_prior = previous_sessions.get(row.session)
            if (
                mode == "candidate"
                and expected_prior is not None
                and last_gate_session != expected_prior
            ):
                update_gate(np.full(n, UNKNOWN, dtype=np.int8), stale=True)
            if cache.fill_eligible[item]:
                mask = long & (dividend_cash > 0)
                shares[mask] += dividend_cash[mask] / row.open
                cash[mask] -= dividend_cash[mask]
                dividend_cash[mask] = 0
                gross_mask = long & (gross_dividend_cash > 0)
                gross_shares[gross_mask] += gross_dividend_cash[gross_mask] / row.open
                gross_cash[gross_mask] -= gross_dividend_cash[gross_mask]
                gross_dividend_cash[gross_mask] = 0
                if mode == "buy_and_hold" and not benchmark_bought:
                    fill(
                        np.ones(n, dtype=bool),
                        "buy",
                        row.open,
                        timestamp,
                        row.session,
                        "benchmark_entry",
                    )
                    benchmark_bought = True
                else:
                    fill(pending == -1, "sell", row.open, timestamp, row.session, "confirmed_exit")
                    fill(pending == 1, "buy", row.open, timestamp, row.session, "confirmed_entry")
        elif kind == "bar_close":
            row = cache.bar_rows[item]
            if mode == "candidate" and cache.signal_eligible[item]:
                values = cache.margins[item, indices]
                valid = np.isfinite(values)
                oversold = valid & (values <= entry_thresholds)
                entry_ok = (gate == ON) & (gate_only | oversold)
                exit_ok = (gate != UNKNOWN) & ~gate_only & valid & (values >= exit_thresholds)
                blocked_oversold[:] += ~long & ~gate_only & oversold & (gate != ON)
                entry_count[:] = np.where(~long & entry_ok, entry_count + 1, 0)
                exit_count[:] = np.where(long & exit_ok, exit_count + 1, 0)
                break_count[:] = np.where(long & break_enabled & (gate == OFF), break_count + 1, 0)
                buys = ~long & (entry_count >= required)
                recovery_sells = long & (exit_count >= required)
                break_sells = long & (break_count >= required)
                sells = recovery_sells | break_sells
                # A confirmed recovery is sticky until fill even if its level later
                # changes; ON only cancels a trend-break-only pending exit.
                pending_causes[sells] |= recovery_sells[sells].astype(np.int8) | (
                    break_sells[sells].astype(np.int8) << 1
                )
                new_buys = buys & (pending != 1)
                new_sells = sells & (pending != -1)
                if post_enabled and item in post_gap_context["by_bar"]:
                    # Count distinct newly CONFIRMED decisions, never mere support,
                    # fills, pre-gap cancellation, or terminal liquidation.
                    post_buy += new_buys
                    post_sell += new_sells
                    decisions = np.flatnonzero(new_buys | new_sells)
                    for gap_index in post_gap_context["by_bar"][item]:
                        gap_start, gap_end, window_end = post_gap_context["windows"][gap_index]
                        post_total_rows += len(decisions)
                        available = max(0, post_limit - len(post_rows))
                        for j in decisions[:available]:
                            post_rows.append(
                                (
                                    ids[j],
                                    "buy" if new_buys[j] else "sell",
                                    timestamp,
                                    gap_start,
                                    gap_end,
                                    window_end,
                                )
                            )
                pending[buys] = 1
                pending[sells] = -1
                pending_at[buys | sells] = timestamp.value / 1e9
            if record:
                diagnostics.append(
                    dict(
                        timestamp=timestamp,
                        event="bar_close",
                        session=row.session,
                        signal_eligible=bool(cache.signal_eligible[item]),
                        fill_eligible=bool(cache.fill_eligible[item]),
                        entry_count=int(entry_count[0]),
                        exit_count=int(exit_count[0]),
                        trend_break_count=int(break_count[0]),
                        gate_state=int(gate[0]),
                        position=int(long[0]),
                        pending=int(pending[0]),
                    )
                )
            if item == terminal_index:
                terminal_cancelled[:] = pending != 0
                pending[:] = 0
                pending_at[:] = np.nan
                fill(long.copy(), "sell", row.close, timestamp, row.session, "terminal_liquidation")
        elif kind == "valuation":
            if mode == "candidate":
                daily_index = daily_positions[item]
                update_gate(cache.gates.iloc[daily_index].to_numpy()[gate_columns])
                last_gate_session = item
            price = float(daily[item].close)
            equity = cash + shares * price
            gross = gross_cash + gross_shares * price
            if not np.isfinite(equity).all() or (equity <= 0).any() or not np.isfinite(gross).all():
                raise ValueError("Account wealth is nonfinite or exhausted")
            day_equity.append(equity.copy())
            day_gross.append(gross.copy())
            day_cash.append(cash_reference)
            day_exposure.append(long.astype(float))
            day_times.append(timestamp)
    times = pd.DatetimeIndex(day_times, name="timestamp")
    equity = np.asarray(day_equity)
    gross = np.asarray(day_gross)
    returns = equity / np.vstack([np.full(n, initial_cash), equity[:-1]]) - 1
    ref = np.asarray(day_cash)
    cash_returns = ref / np.r_[initial_cash, ref[:-1]] - 1
    seconds = (clock[-1] - clock[0]).total_seconds()
    years = seconds / (365 * 86400)
    metrics = []
    cov = cache.coverage.loc[cache.coverage.session.isin(calendar.session)]
    unknown_slots = int(cov.status.eq("unknown_missing").sum())
    halt_slots = int(cov.status.eq("known_halt_overlap").sum())
    source_presence_complete = bool(cov.bar_start.isin(cache.bars.bar_start).all())
    all_signals_eligible = bool(cov.signal_eligible.all())
    raw_price_complete = bool(cov.status.eq("observed").all()) and source_presence_complete
    for j, c in enumerate(candidates):
        ret = returns[:, j]
        excess = ret - cash_returns
        sd = np.std(excess, ddof=1) if len(excess) > 1 else np.nan
        sharpe = float(np.sqrt(252) * np.mean(excess) / sd) if sd > 0 else np.nan
        wealth = np.r_[initial_cash, equity[:, j]]
        metrics.append(
            dict(
                candidate_id=ids[j],
                base_candidate_id=c.base.candidate_id if isinstance(c, GatedCandidate) else None,
                gate_period=c.gate_period,
                gate_indicator_units="daily_sessions",
                gate_update_hour_ny=17,
                exit_policy=c.exit_policy,
                basket_id=c.basket_id,
                blocked_oversold_checks=int(blocked_oversold[j]),
                trend_break_exit_count=int(trend_break_exits[j]),
                entry_cancellation_count=int(entry_cancellations[j]),
                exit_cancellation_count=int(exit_cancellations[j]),
                unknown_gate_reset_count=int(unknown_resets[j]),
                family=c.family,
                entry_family=c.family,
                exit_family=c.family,
                family_cell=c.family,
                rule=f"daily_sma_gate_{c.gate_period}"
                if isinstance(c, GateOnlyCandidate)
                else c.rule.rule_id,
                entry_rule=f"daily_sma_gate_{c.gate_period}"
                if isinstance(c, GateOnlyCandidate)
                else c.rule.rule_id,
                exit_rule=f"daily_sma_gate_{c.gate_period}"
                if isinstance(c, GateOnlyCandidate)
                else c.rule.rule_id,
                period=c.gate_period if isinstance(c, GateOnlyCandidate) else c.rule.period,
                indicator_units="daily_sessions",
                entry_threshold=float(c.entry_threshold),
                exit_threshold=float(c.exit_threshold),
                confirmation_hours=c.confirmation_hours,
                entry_confirmation_hours=c.confirmation_hours,
                exit_confirmation_hours=c.confirmation_hours,
                entry_required_checks=c.required_checks,
                exit_required_checks=c.required_checks,
                cash_excess_sharpe=sharpe,
                cash_excess_Sharpe=sharpe,
                cagr=float((equity[-1, j] / initial_cash) ** (1 / years) - 1) if years else np.nan,
                max_drawdown=float((wealth / np.maximum.accumulate(wealth) - 1).min()),
                annualized_volatility=float(np.std(ret, ddof=1) * np.sqrt(252))
                if len(ret) > 1
                else np.nan,
                exposure_fraction=float(held_seconds[j] / seconds) if seconds else 0,
                total_turnover=float(turnover[j]),
                annualized_turnover=float(turnover[j] / years) if years else np.nan,
                trade_count=int(trades[j]),
                completed_round_trips_per_year=float(trades[j] / years) if years else np.nan,
                trades_per_year=float(trades[j] / years) if years else np.nan,
                order_count=int(orders[j]),
                average_holding_hours=float(holds[j] / trades[j]) if trades[j] else 0,
                total_commission=float(commission[j]),
                total_slippage=float(slippage[j]),
                cost_drag_return=float((gross[-1, j] - equity[-1, j]) / initial_cash),
                terminal_equity=float(equity[-1, j]),
                gross_terminal_equity=float(gross[-1, j]),
                cash_terminal_equity=float(ref[-1]),
                total_return=float(equity[-1, j] / initial_cash - 1),
                scored_sessions=len(times),
                start_session=calendar.session.iloc[0],
                end_session=calendar.session.iloc[-1],
                sampling_minutes=30,
                commission_bps=float(commission_bps),
                slippage_bps=float(slippage_bps),
                quality_profile=cache.policy.profile,
                daily_valuation_hour_ny=17,
                daily_quote_kind="provider_daily_raw_close_not_NOCP",
                unknown_missing_slots=unknown_slots,
                halt_affected_slots=halt_slots,
                quality_classification_complete=True,
                tradable_price_coverage_complete=unknown_slots == 0,
                source_row_presence_complete=source_presence_complete,
                raw_intraday_price_coverage_complete=raw_price_complete,
                all_slots_signal_eligible=all_signals_eligible,
                cancelled_blackout_order_count=int(cancelled[j]),
                cancelled_terminal_order_count=int(terminal_cancelled[j]),
            )
        )
    post_payload = None
    if post_enabled:
        for j, stats in enumerate(metrics):
            stats.update(
                post_gap_confirmed_buy_count=int(post_buy[j]),
                post_gap_confirmed_sell_count=int(post_sell[j]),
                post_gap_confirmed_signal_count=int(post_buy[j] + post_sell[j]),
                has_post_gap_confirmed_signal=bool(post_buy[j] + post_sell[j]),
            )
        if record:
            for candidate_id, side, decision_at, gap_start, gap_end, window_end in post_rows:
                diagnostics.append(
                    dict(
                        event="post_gap_confirmed_signal",
                        timestamp=decision_at,
                        candidate_id=candidate_id,
                        side=side,
                        decision_at=decision_at,
                        gap_start=gap_start,
                        gap_end=gap_end,
                        window_end=window_end,
                    )
                )
        post_payload = {
            "rows": post_rows,
            "total_rows": post_total_rows,
            "context": post_gap_context,
        }
    return (
        metrics,
        times,
        equity,
        returns,
        cash_returns,
        np.asarray(day_exposure),
        trade_rows,
        diagnostics,
        post_payload,
    )


def run(
    cache,
    candidates,
    *,
    start,
    end,
    cash_observations,
    commission_bps=1.0,
    slippage_bps=3.0,
    initial_cash=1000.0,
    record=False,
    mode="candidate",
):
    """Evaluate an in-memory segment with identical passive accounting.

    Returns metrics, times, daily equity, returns, cash returns, close exposure,
    recorded trades, diagnostics and post-gap payload (the quality-kernel shape).
    Candidate mode accepts a unique batch of at most 256 candidates; passive
    modes require one gated placeholder, whose signal and gate are never used.
    """
    _validate_cache(cache)
    candidates = tuple(candidates)
    if mode not in {"candidate", "cash", "buy_and_hold"}:
        raise ValueError("Mode must be candidate, cash or buy_and_hold")
    if not isinstance(record, bool):
        raise ValueError("Record must be an explicit boolean")
    if (
        not candidates
        or len(candidates) > 256
        or not all(isinstance(c, (GatedCandidate, GateOnlyCandidate)) for c in candidates)
        or len({c.candidate_id for c in candidates}) != len(candidates)
    ):
        raise ValueError(
            "A nonempty unique batch of at most 256 gated/gate-only candidates is required"
        )
    if mode != "candidate" and len(candidates) != 1:
        raise ValueError("Passive benchmark mode requires exactly one placeholder")
    if any(c.rule not in cache.rules for c in candidates):
        raise ValueError("Candidate rule is absent from the mean-reversion feature cache")
    calendar, events, clock, terminal = quality._segment_events(cache, start, end)
    return _batch(
        cache,
        calendar,
        events,
        clock,
        terminal,
        candidates,
        cash_observations,
        commission_bps,
        slippage_bps,
        initial_cash,
        mode=mode,
        record=record,
    )


def result(parts):
    """Convert a single-candidate/benchmark tuple to the common result container."""
    if len(parts[0]) != 1:
        raise ValueError("One simulation is required for a result container")
    converted = quality._result(parts)
    return replace(
        converted, trades=pd.DataFrame(parts[6], columns=[*core.TRADE_COLUMNS, "exit_causes"])
    )


@dataclass(frozen=True)
class BasketResult:
    metrics: dict
    daily_equity: pd.Series
    daily_returns: pd.Series
    annual_returns: pd.DataFrame
    event_exposure: pd.Series
    daily_exposure: pd.Series


def aggregate_basket(paths: portfolio.SleevePaths, basket_id, weights=None):
    """Sum fixed initial sleeves; true notional exposure and common-fill turnover.

    The caller freezes basket membership. Optional zero weights permit selecting
    one unchanged 32-sleeve basket out of a batch of all 128 recorded paths. No
    daily rebalancing, cash transfer, order netting or fresh signals occur here.
    """
    portfolio._validate_paths(paths)
    if not isinstance(basket_id, str) or not basket_id.strip():
        raise ValueError("A nonempty stable basket ID is required")
    n = len(paths.candidate_ids)
    weights = np.full(n, 1 / n) if weights is None else np.asarray(weights, dtype=float)
    if (
        weights.shape != (n,)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Nonnegative finite initial weights must sum to one")
    weights = weights / weights.sum()
    equity = portfolio._weighted_wealth(paths.equity, weights)
    returns = equity / np.r_[paths.initial_cash, equity[:-1]] - 1
    gross_terminal = float(portfolio._weighted_wealth(paths.gross_terminal_equity, weights))
    seconds = (paths.segment_end - paths.segment_start).total_seconds()
    years = seconds / portfolio.SECONDS_YEAR
    event_wealth = portfolio._weighted_wealth(paths.event_equity, weights)
    exposure_path = paths.event_invested_notional @ weights / event_wealth
    elapsed = np.diff(paths.event_times.asi8) / 1e9
    exposure = float(exposure_path[:-1] @ elapsed / seconds)
    daily_exposure = exposure_path[paths.event_times.get_indexer(paths.times)]
    id_weights = dict(zip(paths.candidate_ids, weights, strict=True))
    orders, turnover, timestamps, fees, slips = 0, 0.0, set(), 0.0, 0.0
    for group in paths.fill_groups:
        group_weights = group.trades.candidate_id.map(id_weights).to_numpy(float)
        active = group_weights > 0
        if not active.any():
            continue
        orders += int(active.sum())
        timestamps.add(group.timestamp)
        common_before = float(portfolio._weighted_wealth(group.equity_before, weights))
        turnover += float(
            group.trades.reference_notional.to_numpy(float) @ group_weights / common_before
        )
        fees += float(group.trades.commission.to_numpy(float) @ group_weights)
        slips += float(group.trades.slippage.to_numpy(float) @ group_weights)
    wealth = np.r_[paths.initial_cash, equity]
    active_metrics = [m for m, weight in zip(paths.metrics, weights, strict=True) if weight > 0]
    trips = sum(m["trade_count"] for m in active_metrics)
    hold_hours = sum(m["average_holding_hours"] * m["trade_count"] for m in active_metrics)
    metrics = dict(
        portfolio_id=basket_id,
        basket_id=basket_id,
        cash_excess_sharpe=portfolio.cash_excess_sharpe(returns, paths.cash_returns),
        cagr=float((equity[-1] / paths.initial_cash) ** (1 / years) - 1),
        max_drawdown=float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        annualized_volatility=float(np.std(returns, ddof=1) * np.sqrt(252))
        if len(returns) > 1
        else np.nan,
        exposure_fraction=exposure,
        exposure_definition="calendar_time_weighted_event_left_QQQ_notional_over_total_wealth",
        initial_weighted_binary_long_time_fraction=float(
            np.asarray([x["exposure_fraction"] for x in paths.metrics]) @ weights
        ),
        gross_sleeve_order_count=orders,
        gross_sleeve_orders_per_year=orders / years,
        distinct_execution_timestamp_count=len(timestamps),
        distinct_execution_timestamps_per_year=len(timestamps) / years,
        total_turnover=turnover,
        annualized_turnover=turnover / years,
        total_commission=fees,
        total_slippage=slips,
        cost_drag_return=(gross_terminal - equity[-1]) / paths.initial_cash,
        terminal_equity=float(equity[-1]),
        gross_terminal_equity=gross_terminal,
        total_return=float(equity[-1] / paths.initial_cash - 1),
        positive_weight_sleeve_count=int((weights > 0).sum()),
        gross_sleeve_completed_round_trips=trips,
        average_gross_sleeve_holding_hours=hold_hours / trips if trips else 0.0,
        holding_duration_definition="mean_elapsed_calendar_hours_per_completed_gross_sleeve_round_trip",
        scored_sessions=len(paths.times),
        elapsed_calendar_years=years,
        start_session=paths.times[0].tz_convert("America/New_York").strftime("%Y-%m-%d"),
        end_session=paths.times[-1].tz_convert("America/New_York").strftime("%Y-%m-%d"),
        commission_bps=float(paths.metrics[0]["commission_bps"]),
        slippage_bps=float(paths.metrics[0]["slippage_bps"]),
    )
    for key in (
        "blocked_oversold_checks",
        "trend_break_exit_count",
        "entry_cancellation_count",
        "exit_cancellation_count",
        "unknown_gate_reset_count",
        "cancelled_blackout_order_count",
        "cancelled_terminal_order_count",
    ):
        metrics[key] = sum(m.get(key, 0) for m in active_metrics)
    return BasketResult(
        metrics,
        pd.Series(equity, paths.times, name=basket_id),
        pd.Series(returns, paths.times, name=basket_id),
        portfolio.annual_return_table(pd.Series(equity, paths.times)),
        pd.Series(exposure_path, paths.event_times, name="QQQ_exposure_fraction"),
        pd.Series(daily_exposure, paths.times, name="QQQ_exposure_fraction"),
    )
