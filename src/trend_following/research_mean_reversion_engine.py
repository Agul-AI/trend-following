"""Independent observed-only QQQ/cash mean-reversion simulation kernel.

The event ledger is adapted verbatim from the immutable quality engine except
for candidate metadata and the independent oversold/recovery signal predicates.
No trend-following candidate, signal state or frozen source is modified. Shared
read-only helpers validate price/coverage identities and build the event clock;
commission, slippage, dividend DRIP, splits, accrual, terminal liquidation and
post-gap accounting retain the frozen conventions. This module opens no files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_following import research_daily_hourly_v5 as core
from trend_following import research_mean_reversion as features
from trend_following import research_quality_engine_v5 as quality
from trend_following.research_rates import rate_accrual

# Read-only reuse of coverage/calendar/post-gap helpers; never monkeypatch them.
_post_gap_context = quality._post_gap_context


def build_cache(
    daily, bars, calendar, *, coverage=None, blackouts=None, quality_policy=quality.DEFAULT_POLICY
):
    """Normalize observed inputs and construct only mean-reversion daily features."""
    return features.build_feature_cache(
        daily,
        bars,
        calendar,
        coverage=coverage,
        blackouts=blackouts,
        quality_policy=quality_policy,
    )


def _validate_cache(cache):
    """Reject post-build source/derived mutation, without opening any stage file."""
    if not isinstance(cache, features.MeanReversionCache):
        raise ValueError("Mean-reversion cache required; trend support caches are incompatible")
    if not isinstance(cache.policy, quality.QualityPolicy):
        raise ValueError("Explicit validated quality policy required")
    if (
        cache.policy.profile == "strict_tradable_v1"
        and cache.coverage.status.eq("unknown_missing").any()
    ):
        raise ValueError("Strict-tradable cache rejects unknown missing-source intervals")
    actual = quality._input_fingerprint(
        cache.daily,
        cache.bars,
        cache.calendar,
        cache.coverage,
        cache.blackouts,
        cache.policy,
    )
    if not cache.input_fingerprint or actual != cache.input_fingerprint:
        raise ValueError("Cached mean-reversion source/coverage identity changed")
    if (
        not cache.feature_fingerprint
        or quality._feature_fingerprint(cache) != cache.feature_fingerprint
    ):
        raise ValueError("Cached mean-reversion feature/eligibility/execution identity changed")
    if not all(isinstance(rule, features.Rule) for rule in cache.rules):
        raise ValueError("Cache must contain only mean-reversion rules")


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
                    )
                )
        pending[mask] = 0
        pending_at[mask] = np.nan
        entry_count[mask] = 0
        exit_count[mask] = 0

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
                entry_ok = valid & (values <= entry_thresholds)
                exit_ok = valid & (values >= exit_thresholds)
                entry_count[:] = np.where(~long & entry_ok, entry_count + 1, 0)
                exit_count[:] = np.where(long & exit_ok, exit_count + 1, 0)
                buys = ~long & (entry_count >= required)
                sells = long & (exit_count >= required)
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
                family=c.family,
                entry_family=c.family,
                exit_family=c.family,
                family_cell=c.family,
                rule=c.rule.rule_id,
                entry_rule=c.rule.rule_id,
                exit_rule=c.rule.rule_id,
                period=c.rule.period,
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
    modes require one mean-reversion placeholder, whose signal is never used.
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
        or not all(isinstance(c, features.Candidate) for c in candidates)
        or len({c.candidate_id for c in candidates}) != len(candidates)
    ):
        raise ValueError(
            "A nonempty unique batch of at most 256 mean-reversion candidates is required"
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
    return quality._result(parts)
