"""Read-only fixed-initial-capital portfolio arithmetic and recorded-ledger replay.

This module opens no files and makes no trading decisions. Its inputs are bounded
in-memory packets and recorded results from the unchanged constituent engines.
Portfolio wealth is a sum of initially scaled sleeve wealth, not a constantly
rebalanced sum of returns. Carried last-labelled prices are used only for event
valuation, never for a signal or execution across missing intervals.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trend_following import research_quality_engine_v5 as quality
from trend_following.research_rates import rate_accrual

TREND_COUNT = 12
MEAN_REVERSION_COUNT = 32
INITIAL_CASH = 1000.0
TIE_TOLERANCE = 1e-10
SECONDS_YEAR = 365 * 86400


@dataclass(frozen=True)
class Allocation:
    portfolio_id: str
    trend_weight: float

    def __post_init__(self):
        if (
            not isinstance(self.portfolio_id, str)
            or not self.portfolio_id.strip()
            or isinstance(self.trend_weight, (bool, np.bool_))
            or not isinstance(self.trend_weight, (float, int, np.floating, np.integer))
            or not np.isfinite(self.trend_weight)
            or not 0 <= self.trend_weight <= 1
        ):
            raise ValueError("Allocation requires a stable ID and a finite weight in [0, 1]")
        object.__setattr__(self, "trend_weight", float(self.trend_weight))

    def weights(self):
        return np.r_[
            np.full(TREND_COUNT, self.trend_weight / TREND_COUNT),
            np.full(MEAN_REVERSION_COUNT, (1 - self.trend_weight) / MEAN_REVERSION_COUNT),
        ]

    def to_dict(self):
        return dict(
            portfolio_id=self.portfolio_id,
            trend_weight=self.trend_weight,
            mean_reversion_weight=1 - self.trend_weight,
        )


def allocation_grid():
    return tuple(
        [Allocation(f"portfolio_tf_{tick * 10:03d}", tick / 10) for tick in range(11)]
        + [Allocation("portfolio_equal_44", TREND_COUNT / (TREND_COUNT + MEAN_REVERSION_COUNT))]
    )


@dataclass(frozen=True)
class FillGroup:
    timestamp: pd.Timestamp
    phase: str
    reference_price: float
    equity_before: np.ndarray
    trades: pd.DataFrame


@dataclass(frozen=True)
class SleevePaths:
    candidate_ids: tuple[str, ...]
    times: pd.DatetimeIndex
    equity: np.ndarray
    returns: np.ndarray
    cash_returns: np.ndarray
    gross_terminal_equity: np.ndarray
    event_times: pd.DatetimeIndex
    event_equity: np.ndarray
    event_invested_notional: np.ndarray
    event_long: np.ndarray
    fill_groups: tuple[FillGroup, ...]
    segment_start: pd.Timestamp
    segment_end: pd.Timestamp
    initial_cash: float
    metrics: tuple[dict, ...]
    input_fingerprint: str = ""


@dataclass(frozen=True)
class PortfolioResult:
    allocation: Allocation
    metrics: dict
    daily_equity: pd.Series
    daily_returns: pd.Series
    annual_returns: pd.DataFrame
    event_exposure: pd.Series
    daily_exposure: pd.Series


def _same(actual, expected, label, *, atol=2e-8, rtol=2e-11):
    if np.shape(actual) != np.shape(expected) or not np.allclose(
        actual, expected, atol=atol, rtol=rtol, equal_nan=True
    ):
        raise ValueError(f"Recorded replay does not reconcile: {label}")


def _utc_index(values, label):
    index = pd.DatetimeIndex(values)
    if (
        index.tz is None
        or index.hasnans
        or index.has_duplicates
        or not index.is_monotonic_increasing
        or len(index) == 0
    ):
        raise ValueError(f"{label} must be a nonempty sorted unique timezone-aware clock")
    return index.tz_convert("UTC")


def _weighted_wealth(values, weights):
    """Affine sum avoids turning identical cash sleeves into floating-point alpha."""
    base = np.asarray(values)[..., np.flatnonzero(weights > 0)[0]]
    return base + (np.asarray(values) - np.expand_dims(base, axis=-1)) @ weights


def _validate_paths(paths):
    n, days, events = len(paths.candidate_ids), len(paths.times), len(paths.event_times)
    if n == 0 or len(set(paths.candidate_ids)) != n or len(paths.metrics) != n:
        raise ValueError("Sleeve IDs and metrics must be nonempty, unique and aligned")
    if tuple(x["candidate_id"] for x in paths.metrics) != paths.candidate_ids:
        raise ValueError("Sleeve metrics and column identity disagree")
    for key in ("commission_bps", "slippage_bps"):
        values = [x[key] for x in paths.metrics]
        if not np.isfinite(values).all() or min(values) < 0 or len(set(values)) != 1:
            raise ValueError("All sleeves must use one finite nonnegative cost scenario")
    _utc_index(paths.times, "daily clock")
    _utc_index(paths.event_times, "event clock")
    if paths.initial_cash != INITIAL_CASH or paths.segment_end <= paths.segment_start:
        raise ValueError("Use the fixed $1000 sleeve normalization and a positive segment")
    if (
        paths.event_times[0] != paths.segment_start
        or paths.event_times[-1] != paths.segment_end
        or not paths.times.isin(paths.event_times).all()
    ):
        raise ValueError("Segment and daily marks must match the exact event clock")
    for value, shape, positive in (
        (paths.equity, (days, n), True),
        (paths.returns, (days, n), False),
        (paths.cash_returns, (days,), False),
        (paths.gross_terminal_equity, (n,), True),
        (paths.event_equity, (events, n), True),
        (paths.event_invested_notional, (events, n), False),
        (paths.event_long, (events, n), False),
    ):
        if np.shape(value) != shape or not np.isfinite(value).all():
            raise ValueError("Malformed or nonfinite sleeve/event paths")
        if positive and (value <= 0).any():
            raise ValueError("Account wealth must remain positive")
    if (paths.event_invested_notional < 0).any() or (
        paths.event_invested_notional > paths.event_equity + 2e-8
    ).any():
        raise ValueError("Unlevered sleeves cannot have negative/excess invested notional")
    _same(
        paths.returns,
        paths.equity / np.vstack([np.full(n, paths.initial_cash), paths.equity[:-1]]) - 1,
        "wealth-to-return identity",
    )
    _same(
        paths.event_equity[paths.event_times.get_indexer(paths.times)],
        paths.equity,
        "daily event wealth",
    )
    for group in paths.fill_groups:
        if (
            group.timestamp not in paths.event_times
            or group.phase not in {"open", "terminal"}
            or not np.isfinite(group.reference_price)
            or group.reference_price <= 0
            or np.shape(group.equity_before) != (n,)
            or not np.isfinite(group.equity_before).all()
            or (group.equity_before <= 0).any()
            or not group.trades.candidate_id.isin(paths.candidate_ids).all()
            or group.trades.candidate_id.duplicated().any()
            or not np.isfinite(group.trades[["reference_notional", "commission", "slippage"]])
            .all()
            .all()
            or (group.trades[["reference_notional", "commission", "slippage"]] < 0).any().any()
        ):
            raise ValueError("Malformed private fill group")


def replay_recorded(cache, parts, *, start, end, cash_observations, initial_cash=INITIAL_CASH):
    """Replay recorded fills on the unchanged engine event clock; never re-signal.

    Every recorded fill is verified against the engine's raw reference quote,
    fee/slippage, position quantity and pre-fill wealth. Split credits, dividend
    cash, cost-free DRIP and event-time cash accrual are independently replayed.
    Daily paths, order counts, total charges and binary holding time reconcile.
    The result includes private prices in fill groups; do not publish them.
    """
    if len(parts) != 9 or initial_cash != INITIAL_CASH:
        raise ValueError(
            "The recorded nine-part engine result and $1000 normalization are required"
        )
    actual_fingerprint = quality._input_fingerprint(
        cache.daily, cache.bars, cache.calendar, cache.coverage, cache.blackouts, cache.policy
    )
    if not cache.input_fingerprint or cache.input_fingerprint != actual_fingerprint:
        raise ValueError("Recorded replay source/cache identity changed")
    stats, day_times, equity, returns, cash_returns, close_exposure, records, _, _ = parts
    ids = tuple(x["candidate_id"] for x in stats)
    n = len(ids)
    if not ids or len(set(ids)) != n:
        raise ValueError("Recorded constituent IDs must be unique and aligned")
    calendar, events, clock, terminal = quality._segment_events(cache, start, end)
    clock = _utc_index(clock, "replay clock")
    times = _utc_index(day_times, "daily clock")
    daily = {row.session: row for row in cache.daily.itertuples(index=False)}
    accrual = rate_accrual(clock, cash_observations, interval_start=clock[0]).to_numpy()
    if (accrual <= -1).any():
        raise ValueError("Cash account exhausted")
    cash, shares = np.full(n, initial_cash), np.zeros(n)
    dividends, long = np.zeros(n), np.zeros(n, dtype=bool)
    held, fees, slips, trips = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)
    orders, turnover, hold_hours = np.zeros(n, dtype=int), np.zeros(n), np.zeros(n)
    entered_at = np.full(n, np.nan)
    trades = pd.DataFrame(records)
    if not trades.empty:
        required = {
            "candidate_id",
            "timestamp",
            "side",
            "reason",
            "reference_price",
            "execution_price",
            "quantity",
            "reference_notional",
            "commission",
            "slippage",
            "equity_before",
            "holding_hours",
        }
        if not required.issubset(trades) or not trades.candidate_id.isin(ids).all():
            raise ValueError("Complete recorded trade rows are required")
        trades = trades.copy()
        trades["timestamp"] = pd.to_datetime(trades.timestamp, utc=True)
        if not trades.timestamp.isin(clock).all() or not trades.side.isin(["buy", "sell"]).all():
            raise ValueError("Recorded fills must lie on the frozen event clock")
        trades["phase"] = np.where(trades.reason.eq("terminal_liquidation"), "terminal", "open")
        if trades.duplicated(["candidate_id", "timestamp", "phase"]).any():
            raise ValueError("Duplicate sleeve fills at one execution phase")
    elif any(int(x.get("order_count", 0)) != 0 for x in stats):
        raise ValueError("record=True is required for constituents that trade")
    lookup = {candidate_id: i for i, candidate_id in enumerate(ids)}
    grouped_trades = (
        {(t, phase): frame for (t, phase), frame in trades.groupby(["timestamp", "phase"])}
        if not trades.empty
        else {}
    )
    consumed = set()
    fills = []
    observed_day_equity, observed_day_long = [], []
    event_equity, event_invested, event_long = [], [], []
    cash_reference, observed_cash = initial_cash, []
    events_by_time = {}
    for event in events:
        events_by_time.setdefault(event[0], []).append(event)
    mark = None

    def fill(timestamp, phase, price):
        frame = grouped_trades.get((timestamp, phase))
        if frame is None:
            return
        consumed.add((timestamp, phase))
        before = cash + shares * price
        fills.append(FillGroup(timestamp, phase, float(price), before.copy(), frame.copy()))
        # Different sleeves' gross orders are simultaneous for portfolio metrics;
        # the original frame order cannot change the common pre-group denominator.
        for row in frame.itertuples(index=False):
            j = lookup[row.candidate_id]
            fee = float(stats[j]["commission_bps"]) / 10000
            slip = float(stats[j]["slippage_bps"]) / 10000
            execution = price * (1 + slip if row.side == "buy" else 1 - slip)
            _same(row.reference_price, price, "fill raw quote")
            _same(row.execution_price, execution, "adverse slippage quote")
            _same(row.equity_before, before[j], "sleeve pre-fill wealth")
            if row.side == "buy":
                if long[j] or shares[j] != 0:
                    raise ValueError("Recorded entry requires a cash sleeve")
                quantity = cash[j] / (execution * (1 + fee))
                cash[j], dividends[j], shares[j], long[j] = 0, 0, quantity, True
                entered_at[j] = timestamp.value / 1e9
                holding = 0
            else:
                if not long[j]:
                    raise ValueError("Recorded exit requires a long sleeve")
                quantity = shares[j]
                cash[j] += quantity * execution * (1 - fee)
                shares[j], dividends[j], long[j] = 0, 0, False
                holding = (timestamp.value / 1e9 - entered_at[j]) / 3600
                hold_hours[j] += holding
                trips[j] += 1
            commission = quantity * execution * fee
            slippage = quantity * abs(execution - price)
            notional = quantity * price
            for actual, expected, label in (
                (row.quantity, quantity, "quantity"),
                (row.commission, commission, "commission"),
                (row.slippage, slippage, "slippage"),
                (row.reference_notional, notional, "reference notional"),
                (row.holding_hours, holding, "holding duration"),
            ):
                _same(actual, expected, label)
            fees[j] += commission
            slips[j] += slippage
            turnover[j] += notional / before[j]
            orders[j] += 1

    for k, timestamp in enumerate(clock):
        if k:
            held += long * (timestamp - clock[k - 1]).total_seconds()
        cash *= 1 + accrual[k]
        dividends *= 1 + accrual[k]
        cash_reference *= 1 + accrual[k]
        for _, _, kind, item in events_by_time[timestamp]:
            if kind == "actions":
                action = daily[item]
                if mark is not None:
                    mark /= action.split_coefficient
                shares *= action.split_coefficient
                distribution = shares * action.dividend_amount
                cash += distribution
                dividends += distribution
            elif kind == "bar_open":
                row = cache.bar_rows[item]
                mark = float(row.open)
                if cache.fill_eligible[item]:
                    mask = long & (dividends > 0)
                    shares[mask] += dividends[mask] / mark
                    cash[mask] -= dividends[mask]
                    dividends[mask] = 0
                    fill(timestamp, "open", mark)
                elif (timestamp, "open") in grouped_trades:
                    raise ValueError("Recorded execution at an unsafe open")
            elif kind == "bar_close":
                row = cache.bar_rows[item]
                mark = float(row.close)
                if item == terminal:
                    fill(timestamp, "terminal", mark)
            elif kind == "valuation":
                mark = float(daily[item].close)
                observed_day_equity.append((cash + shares * mark).copy())
                observed_day_long.append(long.copy())
                observed_cash.append(cash_reference)
        if mark is None:
            if shares.any():
                raise ValueError("Invested sleeve has no causal observed valuation quote")
            invested = np.zeros(n)
        else:
            invested = shares * mark
        event_equity.append((cash + invested).copy())
        event_invested.append(invested.copy())
        event_long.append(long.copy())
    if len(consumed) != len(grouped_trades) or long.any() or shares.any():
        raise ValueError("Unconsumed recorded fills or missing terminal liquidation")
    day_cash = np.asarray(observed_cash)
    _same(np.asarray(observed_day_equity), equity, "daily wealth")
    _same(np.asarray(observed_day_long), close_exposure, "daily long state")
    _same(day_cash / np.r_[initial_cash, day_cash[:-1]] - 1, cash_returns, "cash reference")
    seconds = (clock[-1] - clock[0]).total_seconds()
    for j, stats_row in enumerate(stats):
        for key, value in (
            ("order_count", orders[j]),
            ("trade_count", trips[j]),
            ("total_commission", fees[j]),
            ("total_slippage", slips[j]),
            ("total_turnover", turnover[j]),
            ("exposure_fraction", held[j] / seconds),
            ("average_holding_hours", hold_hours[j] / trips[j] if trips[j] else 0),
        ):
            _same(stats_row[key], value, key)
    result = SleevePaths(
        ids,
        times,
        np.asarray(equity),
        np.asarray(returns),
        np.asarray(cash_returns),
        np.array([x["gross_terminal_equity"] for x in stats]),
        clock,
        np.asarray(event_equity),
        np.asarray(event_invested),
        np.asarray(event_long),
        tuple(fills),
        clock[0],
        clock[-1],
        float(initial_cash),
        tuple(stats),
        cache.input_fingerprint,
    )
    _validate_paths(result)
    return result


def merge_sleeves(trend: SleevePaths, mean_reversion: SleevePaths):
    """Merge the frozen 12 + 32 groups only, without changing any sleeve state."""
    for paths in (trend, mean_reversion):
        _validate_paths(paths)
    if (
        len(trend.candidate_ids) != TREND_COUNT
        or len(mean_reversion.candidate_ids) != MEAN_REVERSION_COUNT
    ):
        raise ValueError(
            "Portfolio membership must contain exactly 12 trend and 32 mean-reversion sleeves"
        )
    if (
        not trend.times.equals(mean_reversion.times)
        or not trend.event_times.equals(mean_reversion.event_times)
        or trend.segment_start != mean_reversion.segment_start
        or trend.segment_end != mean_reversion.segment_end
        or trend.initial_cash != mean_reversion.initial_cash
        or not np.array_equal(trend.cash_returns, mean_reversion.cash_returns)
        or not trend.input_fingerprint
        or trend.input_fingerprint != mean_reversion.input_fingerprint
    ):
        raise ValueError("Constituent daily/event clocks and cash references must match exactly")
    n = TREND_COUNT + MEAN_REVERSION_COUNT
    groups = {}
    indexed_groups = [
        {(group.timestamp, group.phase): group for group in paths.fill_groups}
        for paths in (trend, mean_reversion)
    ]
    for offset, paths in ((0, trend), (TREND_COUNT, mean_reversion)):
        for group in paths.fill_groups:
            key = (group.timestamp, group.phase)
            if key not in groups:
                # Value every sleeve at this group's raw reference quote. Since
                # the two caches share the exact quote packet, a group absent on
                # the other style can be reconstructed from its cash/holdings.
                other = mean_reversion if offset == 0 else trend
                event_index = paths.event_times.get_loc(group.timestamp)
                # Snapshots are post-event; undo other-style orders if present
                # below when their verified pre-group values overwrite these.
                same_group = indexed_groups[1 if offset == 0 else 0].get(key)
                if same_group is not None:
                    other_before = same_group.equity_before
                    _same(same_group.reference_price, group.reference_price, "common fill quote")
                else:
                    other_before = other.event_equity[event_index].copy()
                before = np.empty(n)
                before[offset : offset + len(paths.candidate_ids)] = group.equity_before
                other_offset = TREND_COUNT if offset == 0 else 0
                before[other_offset : other_offset + len(other.candidate_ids)] = other_before
                groups[key] = [group.reference_price, before, []]
            _same(groups[key][0], group.reference_price, "common group quote")
            groups[key][1][offset : offset + len(paths.candidate_ids)] = group.equity_before
            groups[key][2].append(group.trades)
    fills = tuple(
        FillGroup(timestamp, phase, value[0], value[1], pd.concat(value[2], ignore_index=True))
        for (timestamp, phase), value in sorted(groups.items())
    )
    result = SleevePaths(
        trend.candidate_ids + mean_reversion.candidate_ids,
        trend.times,
        np.column_stack([trend.equity, mean_reversion.equity]),
        np.column_stack([trend.returns, mean_reversion.returns]),
        trend.cash_returns,
        np.r_[trend.gross_terminal_equity, mean_reversion.gross_terminal_equity],
        trend.event_times,
        np.column_stack([trend.event_equity, mean_reversion.event_equity]),
        np.column_stack([trend.event_invested_notional, mean_reversion.event_invested_notional]),
        np.column_stack([trend.event_long, mean_reversion.event_long]),
        fills,
        trend.segment_start,
        trend.segment_end,
        trend.initial_cash,
        trend.metrics + mean_reversion.metrics,
        trend.input_fingerprint,
    )
    _validate_paths(result)
    return result


def cash_excess_sharpe(returns, cash_returns):
    values = np.asarray(returns, dtype=float) - np.asarray(cash_returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Sharpe requires aligned finite daily returns")
    sd = np.std(values, ddof=1) if len(values) > 1 else np.nan
    return float(np.sqrt(252) * np.mean(values) / sd) if sd > 0 else np.nan


def annual_return_table(equity: pd.Series, initial_cash=INITIAL_CASH):
    years = equity.index.tz_convert("America/New_York").year
    rows, previous = [], initial_cash
    for year in sorted(set(years)):
        selected = equity.loc[years == year]
        end = float(selected.iloc[-1])
        rows.append(
            dict(
                year=int(year),
                start_wealth=previous,
                end_wealth=end,
                annual_return=end / previous - 1,
                scored_sessions=len(selected),
            )
        )
        previous = end
    return pd.DataFrame(rows)


def evaluate_portfolio(paths: SleevePaths, allocation: Allocation, *, gross_paths=None):
    """Aggregate initially weighted wealth and gross orders without rebalancing."""
    _validate_paths(paths)
    if len(paths.candidate_ids) != TREND_COUNT + MEAN_REVERSION_COUNT:
        raise ValueError("Portfolio requires the frozen 44-sleeve order")
    weights = allocation.weights()
    equity = _weighted_wealth(paths.equity, weights)
    returns = equity / np.r_[paths.initial_cash, equity[:-1]] - 1
    gross_terminal = float(_weighted_wealth(paths.gross_terminal_equity, weights))
    if gross_paths is not None:
        _validate_paths(gross_paths)
        if (
            gross_paths.candidate_ids != paths.candidate_ids
            or not gross_paths.times.equals(paths.times)
            or not gross_paths.event_times.equals(paths.event_times)
            or not np.array_equal(gross_paths.cash_returns, paths.cash_returns)
            or any(x["commission_bps"] != 0 or x["slippage_bps"] != 0 for x in gross_paths.metrics)
        ):
            raise ValueError("Cost drag requires matching zero-cost constituent paths")
        _same(gross_paths.equity[-1], paths.gross_terminal_equity, "zero-cost terminal references")
        gross_terminal = float(_weighted_wealth(gross_paths.equity[-1], weights))
    seconds = (paths.segment_end - paths.segment_start).total_seconds()
    years = seconds / SECONDS_YEAR
    event_wealth = _weighted_wealth(paths.event_equity, weights)
    event_invested = paths.event_invested_notional @ weights
    event_exposure = event_invested / event_wealth
    elapsed = np.diff(paths.event_times.asi8) / 1e9
    exposure = float(event_exposure[:-1] @ elapsed / seconds)
    daily_positions = paths.event_times.get_indexer(paths.times)
    daily_exposure = event_exposure[daily_positions]
    id_weights = dict(zip(paths.candidate_ids, weights, strict=True))
    order_count, turnover, timestamps, total_commission, total_slippage = 0, 0.0, set(), 0.0, 0.0
    for group in paths.fill_groups:
        group_weights = group.trades.candidate_id.map(id_weights).to_numpy(float)
        active = group_weights > 0
        if not active.any():
            continue
        order_count += int(active.sum())
        timestamps.add(group.timestamp)
        common_before = float(_weighted_wealth(group.equity_before, weights))
        turnover += float(
            group.trades.reference_notional.to_numpy(float) @ group_weights / common_before
        )
        total_commission += float(group.trades.commission.to_numpy(float) @ group_weights)
        total_slippage += float(group.trades.slippage.to_numpy(float) @ group_weights)
    wealth = np.r_[paths.initial_cash, equity]
    end_trend = (
        allocation.trend_weight
        if allocation.trend_weight in {0.0, 1.0}
        else float(paths.equity[-1, :TREND_COUNT] @ weights[:TREND_COUNT] / equity[-1])
    )
    metrics = {
        **allocation.to_dict(),
        "cash_excess_sharpe": cash_excess_sharpe(returns, paths.cash_returns),
        "cagr": float((equity[-1] / paths.initial_cash) ** (1 / years) - 1),
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        "annualized_volatility": float(np.std(returns, ddof=1) * np.sqrt(252))
        if len(returns) > 1
        else np.nan,
        "exposure_fraction": exposure,
        "exposure_definition": "calendar_time_weighted_event_left_QQQ_notional_over_total_wealth",
        "initial_weighted_binary_long_time_fraction": float(
            np.asarray([x["exposure_fraction"] for x in paths.metrics]) @ weights
        ),
        "gross_sleeve_order_count": order_count,
        "gross_sleeve_orders_per_year": order_count / years,
        "distinct_execution_timestamp_count": len(timestamps),
        "distinct_execution_timestamps_per_year": len(timestamps) / years,
        "total_turnover": turnover,
        "annualized_turnover": turnover / years,
        "total_commission": total_commission,
        "total_slippage": total_slippage,
        "cost_drag_return": (gross_terminal - equity[-1]) / paths.initial_cash,
        "terminal_equity": float(equity[-1]),
        "gross_terminal_equity": gross_terminal,
        "total_return": float(equity[-1] / paths.initial_cash - 1),
        "initial_trend_weight": allocation.trend_weight,
        "ending_trend_weight": end_trend,
        "initial_mean_reversion_weight": 1 - allocation.trend_weight,
        "ending_mean_reversion_weight": 1 - end_trend,
        "positive_weight_sleeve_count": int((weights > 0).sum()),
        "scored_sessions": len(paths.times),
        "elapsed_calendar_years": years,
        "start_session": paths.times[0].tz_convert("America/New_York").strftime("%Y-%m-%d"),
        "end_session": paths.times[-1].tz_convert("America/New_York").strftime("%Y-%m-%d"),
        "commission_bps": float(paths.metrics[0]["commission_bps"]),
        "slippage_bps": float(paths.metrics[0]["slippage_bps"]),
    }
    return PortfolioResult(
        allocation,
        metrics,
        pd.Series(equity, paths.times, name=allocation.portfolio_id),
        pd.Series(returns, paths.times, name=allocation.portfolio_id),
        annual_return_table(pd.Series(equity, paths.times)),
        pd.Series(event_exposure, paths.event_times, name="QQQ_exposure_fraction"),
        pd.Series(daily_exposure, paths.times, name="QQQ_exposure_fraction"),
    )


def select_allocation(results: Iterable[PortfolioResult]):
    results = tuple(results)
    if len({r.allocation.portfolio_id for r in results}) != len(results):
        raise ValueError("Allocation selection requires unique portfolio IDs")
    eligible = [r for r in results if np.isfinite(r.metrics["cash_excess_sharpe"])]
    if not eligible:
        raise ValueError("No finite-scoring development portfolio remains")
    best = max(r.metrics["cash_excess_sharpe"] for r in eligible)
    tied = [r for r in eligible if best - r.metrics["cash_excess_sharpe"] <= TIE_TOLERANCE]
    closest = min(abs(r.allocation.trend_weight - 0.5) for r in tied)
    nearest = [r for r in tied if abs(r.allocation.trend_weight - 0.5) - closest <= 1e-15]
    return min(nearest, key=lambda r: r.allocation.portfolio_id).allocation


def correlations(paths: SleevePaths):
    _validate_paths(paths)
    excess = paths.returns - paths.cash_returns[:, None]
    frame = pd.DataFrame(excess, index=paths.times, columns=paths.candidate_ids)
    # Exact constant excess series have undefined correlation, retained as NaN.
    frame.loc[:, np.ptp(excess, axis=0) == 0] = np.nan
    style_data = {}
    for label, allocation in (
        ("trend", Allocation("trend", 1)),
        ("mean_reversion", Allocation("mean_reversion", 0)),
    ):
        result = evaluate_portfolio(paths, allocation)
        style_data[label] = result.daily_returns.to_numpy() - paths.cash_returns
    styles = pd.DataFrame(style_data, index=paths.times)
    styles.loc[:, np.ptp(styles.to_numpy(), axis=0) == 0] = np.nan
    return {"constituent": frame.corr(), "style": styles.corr()}


def bootstrap_sharpe_differences(
    selected_returns,
    bh_returns,
    half_returns,
    cash_returns,
    *,
    replicates=2000,
    block_lengths=(5, 20, 60),
    seed=20261009,
):
    """Paired circular moving-block bootstrap; no correction for selection bias.

    Common sampled indices are applied to all three portfolios and cash, and
    each difference is recomputed from the sampled daily cash-excess returns.
    Percentile 95% intervals are conditional on the fixed historical membership.
    """
    matrix = np.column_stack([selected_returns, bh_returns, half_returns, cash_returns])
    if matrix.ndim != 2 or len(matrix) < 2 or not np.isfinite(matrix).all():
        raise ValueError("Bootstrap requires aligned finite daily portfolio/cash returns")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not block_lengths
        or any(
            isinstance(b, bool) or not isinstance(b, int) or not 1 <= b <= len(matrix)
            for b in block_lengths
        )
        or len(set(block_lengths)) != len(block_lengths)
    ):
        raise ValueError("Invalid reproducible bootstrap settings")
    rows = []
    for block in block_lengths:
        rng = np.random.default_rng(np.random.SeedSequence([seed, block]))
        differences = np.full((replicates, 2), np.nan)
        block_count = int(np.ceil(len(matrix) / block))
        for rep in range(replicates):
            starts = rng.integers(0, len(matrix), block_count)
            positions = ((starts[:, None] + np.arange(block)) % len(matrix)).ravel()[: len(matrix)]
            sampled = matrix[positions]
            chosen = cash_excess_sharpe(sampled[:, 0], sampled[:, 3])
            for k in range(2):
                differences[rep, k] = chosen - cash_excess_sharpe(sampled[:, k + 1], sampled[:, 3])
        for k, comparator in enumerate(("buy_and_hold", "50_50")):
            finite = differences[:, k][np.isfinite(differences[:, k])]
            lower, upper = np.quantile(finite, [0.025, 0.975]) if len(finite) else [np.nan, np.nan]
            rows.append(
                dict(
                    comparator=comparator,
                    block_length_sessions=block,
                    replicates=replicates,
                    finite_replicates=len(finite),
                    seed=seed,
                    observed_sharpe_difference=cash_excess_sharpe(matrix[:, 0], matrix[:, 3])
                    - cash_excess_sharpe(matrix[:, k + 1], matrix[:, 3]),
                    lower_95_percentile=float(lower),
                    upper_95_percentile=float(upper),
                    method="paired_circular_moving_block_percentile",
                    corrects_candidate_selection_bias=False,
                )
            )
    return pd.DataFrame(rows)
