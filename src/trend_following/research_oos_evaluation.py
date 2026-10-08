"""Frozen-policy prospective OOS scoring, never a live execution claim.

Only completed, independently scheduled sessions are scored. All policy paths
start flat; warmup supplies causal split-normalized features, not fictional P&L.
The primary endpoint is 24-month compounded net-return DIFFERENCE in percentage
points, scaled B5 minus independently delayed QQQ. Twelve months is preliminary.
The paired bootstrap estimates conditional sampling uncertainty, not future risk
or a re-optimized/re-executed trading strategy. Taxes are secondary scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trend_following.research_execution import build_execution_path, causal_features, normalize_spec
from trend_following.research_ledger import LedgerConfig
from trend_following.research_oos_accounting import (
    causal_split_adjusted_prices,
    simulate_oos_ledger,
    terminal_snapshot,
)
from trend_following.research_rates import rate_accrual


class IncompleteHoldoutError(ValueError):
    """A requested scoring horizon is not yet both elapsed and fully observed."""


@dataclass
class OOSCheckpointResult:
    metrics: pd.DataFrame
    daily_returns: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    bootstrap_samples: pd.DataFrame
    paths: dict[str, Any]
    terminal_snapshots: dict[str, dict]
    metadata: dict[str, Any]
    attribution_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)


def _aware(value: Any, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware and nonmissing")
    return timestamp.tz_convert("UTC")


def completed_bar_schedule(calendar: pd.DataFrame) -> pd.DataFrame:
    """Use supplied verified exchange sessions, never infer holidays from prices."""
    if not {"market_open", "market_close"}.issubset(calendar.columns) or calendar.empty:
        raise ValueError("Verified exchange calendar with opens/closes is required")
    opens = pd.DatetimeIndex([_aware(x, "market_open") for x in calendar.market_open])
    closes = pd.DatetimeIndex([_aware(x, "market_close") for x in calendar.market_close])
    if not opens.is_unique or not opens.is_monotonic_increasing or (closes <= opens).any():
        raise ValueError("Calendar sessions must be chronological, unique and positive length")
    rows = []
    previous_close = None
    for opening, close in zip(opens, closes, strict=True):
        local_open, local_close = (
            opening.tz_convert("America/New_York"),
            close.tz_convert("America/New_York"),
        )
        if (
            local_open.strftime("%H:%M:%S") != "09:30:00"
            or local_close.strftime("%H:%M:%S") not in {"13:00:00", "16:00:00"}
            or local_open.date() != local_close.date()
            or (previous_close is not None and opening <= previous_close)
        ):
            raise ValueError("Unsupported or overlapping scheduled session")
        previous_close = close
        for start in pd.date_range(opening, close, freq="60min", inclusive="left"):
            rows.append(
                {
                    "bar_start": start,
                    "bar_end": min(start + pd.Timedelta(hours=1), close),
                    "session": str(local_open.date()),
                    "session_open": opening,
                    "session_close": close,
                }
            )
    return pd.DataFrame(rows).set_index("bar_end")


def required_warmup_observations(policy_spec: dict | None = None) -> int:
    """Observations INCLUDING first decision; defaults need 1610, not just MA200."""
    spec = normalize_spec(policy_spec)

    def bars(key):
        return int(round(float(spec[key]) * 7))

    return max(
        bars("long_ma_days") + bars("bear_filter_slope_days"),
        bars("medium_ma_days") + bars("bear_filter_slope_days"),
        bars("vol_window_days") + bars("vol_lookback_days") - 1,
        bars("macd_slow_days") + bars("macd_signal_days") - 1,
        bars("short_ma_days"),
        int(spec["entry_confirm_bars"]),
        int(spec["exit_confirm_bars"]),
    )


def _fixed_spec(policy_spec: dict | None, multiplier: float = 1.0) -> dict:
    spec = normalize_spec(policy_spec)
    if (
        spec["kind"] != "b5"
        or spec["product_leverage"] != 3
        or spec["delay_bars"] != 1
        or spec["max_changes_per_session"] != 1
    ):
        raise ValueError(
            "Only fixed B5, actual 3x fund, one-bar delay and one change/session supported"
        )
    if not np.isfinite(multiplier) or not 0 <= multiplier <= 1:
        raise ValueError("Frozen multiplier must be finite and in [0,1]")
    return dict(spec, allocation_multiplier=float(multiplier))


def _prices_through(
    close_prices: pd.DataFrame, cutoff: pd.Timestamp, schedule: pd.DataFrame
) -> pd.DataFrame:
    if not isinstance(close_prices.index, pd.DatetimeIndex) or close_prices.index.tz is None:
        raise ValueError("Raw closes require timezone-aware completed-bar-end index")
    if not {"QQQ", "TQQQ"}.issubset(close_prices.columns):
        raise ValueError("Both actual QQQ and TQQQ raw closes are required")
    # Slice BEFORE inspecting numeric values: later observations cannot influence
    # a calibration or an earlier checkpoint, including by poisoning validation.
    panel = close_prices.loc[close_prices.index <= cutoff, ["QQQ", "TQQQ"]].copy()
    panel.index = panel.index.tz_convert("UTC")
    if panel.empty or not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError("Prices must be nonempty, chronological and unique through cutoff")
    if not np.isfinite(panel.to_numpy(dtype=float)).all() or panel.le(0).any().any():
        raise ValueError("Missing/nonpositive raw closes cannot be filled")
    expected = schedule.index[(schedule.index >= panel.index[0]) & (schedule.index <= cutoff)]
    if not panel.index.equals(expected):
        missing, extra = expected.difference(panel.index), panel.index.difference(expected)
        raise IncompleteHoldoutError(
            f"Incomplete official bar grid: {len(missing)} missing, {len(extra)} unexpected; no interpolation"
        )
    return panel.astype(float)


def _actions_through(actions: pd.DataFrame | list[dict] | None, cutoff: pd.Timestamp):
    if actions is None:
        return None
    frame = pd.DataFrame(actions).copy()
    if frame.empty:
        return None
    if "effective_at" not in frame:
        raise ValueError("Corporate actions require effective_at")
    effective = pd.DatetimeIndex([_aware(x, "action effective_at") for x in frame.effective_at])
    return frame.loc[effective <= cutoff].copy()


def checkpoint_end_for_months(
    calendar: pd.DataFrame, holdout_start: Any, months: int
) -> pd.Timestamp:
    """Last scheduled close strictly before the local calendar anniversary.

    Calendar coverage must extend through the anniversary; a truncated calendar
    must not silently make the last supplied session the checkpoint.
    """
    if months not in {12, 24}:
        raise ValueError(
            "Only the frozen 12-month preliminary and 24-month primary checkpoints exist"
        )
    start = _aware(holdout_start, "holdout_start")
    schedule = completed_bar_schedule(calendar)
    sessions = schedule.drop_duplicates("session")
    if not sessions.session_open.eq(start).any():
        raise ValueError("Holdout must begin at a supplied complete session open")
    anniversary = (start.tz_convert("America/New_York") + pd.DateOffset(months=months)).date()
    if sessions.session.max() < str(anniversary):
        raise IncompleteHoldoutError("Verified calendar does not cover checkpoint anniversary")
    eligible = sessions.loc[
        (sessions.session >= str(start.date())) & (sessions.session < str(anniversary))
    ]
    if eligible.empty:
        raise IncompleteHoldoutError("No scheduled session precedes checkpoint anniversary")
    return eligible.session_close.iloc[-1]


def _features_ready(normalized: pd.DataFrame, spec: dict, start: pd.Timestamp) -> None:
    at_start = normalized.index[normalized.index >= start]
    if not len(at_start):
        raise ValueError("No evaluation bar follows start")
    if len(normalized.loc[: at_start[0]]) < required_warmup_observations(spec):
        raise ValueError("Insufficient preceding feature-only warmup observations")
    row = causal_features(normalized.QQQ, spec, bars_per_day=7).loc[at_start[0]]
    if not np.isfinite(row.drop(labels=["entry", "exit"]).astype(float)).all():
        raise ValueError("Initial policy features are incomplete; warmup cannot be manufactured")


def _daily(returns: pd.Series) -> pd.Series:
    labels = returns.index.tz_convert("America/New_York").normalize()
    return returns.add(1).groupby(labels).prod().sub(1)


def _run_track(
    panel: pd.DataFrame,
    normalized: pd.DataFrame,
    actions,
    schedule: pd.DataFrame,
    spec: dict,
    start: pd.Timestamp,
    *,
    track: str,
    fee_bps: float,
    slippage_bps: float,
    initial_cash: float,
    taxable: bool = False,
    fixed_targets: pd.Series | None = None,
):
    dates = panel.index[panel.index >= start]
    if len(dates) < 2:
        raise ValueError("At least two completed bars required for independently delayed benchmark")
    symbol = "QQQ" if track in {"qqq_buy_hold", "qqq_200ma", "cash"} else "TQQQ"
    if fixed_targets is not None:
        targets = fixed_targets.reindex(dates)
        if (
            targets.isna().any()
            or not np.isfinite(targets).all()
            or not targets.between(0, 1).all()
        ):
            raise ValueError(
                "Matched target path must cover every bar with finite weights in [0,1]"
            )
    elif track == "cash":
        targets = pd.Series(0.0, index=dates)
    elif track in {"qqq_buy_hold", "tqqq_buy_hold"}:
        # Predeclared order after first observed close; fill at second close.
        # Never initialize benchmark from the B5 entry time or its target series.
        targets = pd.Series(1.0, index=dates)
        targets.iloc[0] = 0.0
    else:
        if track == "qqq_200ma":
            # Actual QQQ, NOT a 1/3 allocation to actual TQQQ. Its fixed 200-day
            # window is independent of B5 settings and the calibrated multiplier.
            spec = dict(
                spec,
                kind="trend_1x",
                long_ma_days=200,
                product_leverage=1.0,
                effective_leverage=1.0,
                allocation_multiplier=1.0,
            )
            if len(normalized.loc[: dates[0]]) < 200 * 7:
                raise ValueError("Actual QQQ 200MA comparator requires 1400 causal observations")
        else:
            _features_ready(normalized, spec, start)
        path = build_execution_path(
            normalized.QQQ,
            normalized[symbol],
            spec,
            evaluate_start=start,
            bars_per_day=7,
            session_labels=schedule.session.reindex(panel.index),
            slippage_bps=slippage_bps,
            transaction_cost_bps=fee_bps,
        )
        targets = path.weights_at_close.reindex(dates)
    filtered = None
    if actions is not None and len(actions):
        effective = pd.DatetimeIndex(
            [_aware(x, "action effective_at") for x in actions.effective_at]
        )
        # No account holds warmup shares. Only actions from the actual account
        # start onward may create entitlements or change its physical units.
        filtered = actions.loc[actions.asset.eq(symbol) & (effective >= start)].copy()
    config = LedgerConfig(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        short_tax_rate=0.24 if taxable else 0.0,
        long_tax_rate=0.15 if taxable else 0.0,
        interest_tax_rate=0.24 if taxable else 0.0,
        wash_sales=taxable,
        rebalance_policy="on_target_change",
        exposure_multipliers={symbol: 1.0 if symbol == "QQQ" else 3.0},
    )
    return simulate_oos_ledger(
        panel.loc[dates, [symbol]],
        targets.to_frame(symbol),
        pd.Series(0.0, index=dates),
        config,
        corporate_actions=filtered,
        initial_cash=initial_cash,
        initial_prices={symbol: float(panel.loc[dates[0], symbol])},
    )


def fit_pretrain_multiplier(
    close_prices: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    policy_spec: dict | None = None,
    training_end: Any,
    corporate_actions: pd.DataFrame | list[dict] | None = None,
    training_sessions: int = 252,
    initial_cash: float = 1000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 5.0,
) -> dict:
    """Freeze min(1, sigma_QQQ/sigma_unscaled_B5) on LAST252 completed sessions.

    Calibration is retrospective and flat-start with preceding features-only
    warmup; standard deviations are sample daily net-return deviations (ddof=1).
    No terminal liquidation is fabricated at the calibration cutoff. Undefined
    B5 volatility fails, rather than selecting a favorable fallback multiplier.
    """
    if training_sessions != 252:
        raise ValueError("Frozen calibration requires exactly 252 completed sessions")
    cutoff = _aware(training_end, "training_end")
    schedule = completed_bar_schedule(calendar)
    sessions = schedule.drop_duplicates("session")
    eligible = sessions.loc[sessions.session_close <= cutoff]
    if len(eligible) < 252:
        raise ValueError("Fewer than 252 complete calibration sessions")
    chosen = eligible.tail(252)
    end, start = chosen.session_close.iloc[-1], chosen.session_open.iloc[0]
    panel = _prices_through(close_prices, end, schedule)
    dates = schedule.index[(schedule.session_open >= start) & (schedule.index <= end)]
    if not panel.index[panel.index >= start].equals(dates):
        raise IncompleteHoldoutError("Calibration is not exactly the last 252 complete sessions")
    actions = _actions_through(corporate_actions, end)
    normalized = causal_split_adjusted_prices(panel, actions)
    spec = _fixed_spec(policy_spec, 1.0)
    _features_ready(normalized, spec, start)
    returns = {}
    for track in ["b5_unscaled", "qqq_buy_hold"]:
        ledger = _run_track(
            panel,
            normalized,
            actions,
            schedule,
            spec,
            start,
            track=track,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            initial_cash=initial_cash,
        )
        returns[track] = _daily(ledger.returns)
        if len(returns[track]) != 252:
            raise ValueError("Calibration daily denominator is not 252")
    bvol, qvol = (float(returns[t].std(ddof=1)) for t in ["b5_unscaled", "qqq_buy_hold"])
    if not np.isfinite([bvol, qvol]).all() or bvol <= 1e-12 or qvol <= 1e-12:
        raise ValueError("Nonpositive/undefined calibration volatility; no fallback fitting")
    return {
        "multiplier": min(1.0, qvol / bvol),
        "training_sessions": 252,
        "training_start": start.isoformat(),
        "training_end": end.isoformat(),
        "requested_training_cutoff": cutoff.isoformat(),
        "b5_daily_net_volatility": bvol,
        "qqq_daily_net_volatility": qvol,
        "volatility_ddof": 1,
        "volatility_ratio": qvol / bvol,
        "required_observations_at_first_decision": required_warmup_observations(spec),
        "preceding_warmup_observations": int((panel.index < start).sum()),
        "bars_per_day_convention": 7,
        "initial_cash": initial_cash,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "cash_yield": 0.0,
        "calibration_label": "retrospective_training_only_not_oos",
        "terminal_liquidation_in_calibration": False,
    }


def _metric_row(result, terminal: dict, daily: pd.Series, *, initial_cash: float) -> dict:
    wealth = float(terminal["liquidated_equity"]) / initial_cash
    equity = (1 + daily).cumprod()
    dd = equity / equity.cummax().clip(lower=1.0) - 1
    vol = float(daily.std(ddof=1))
    orders = result.events.loc[result.events.event.isin(["buy", "sell"])]
    months = pd.PeriodIndex(daily.index.tz_localize(None), freq="M")
    monthly = daily.add(1).groupby(months).prod().sub(1)
    targets = result.holdings.set_index("timestamp").target_weight
    target_changes = targets.diff().fillna(targets.iloc[0]).abs().gt(1e-12)
    return {
        "total_return": wealth - 1,
        "total_return_pct": (wealth - 1) * 100,
        "terminal_wealth": wealth,
        "terminal_equity": terminal["liquidated_equity"],
        "marked_equity": terminal["marked_equity"],
        "CAGR": wealth ** (252 / len(daily)) - 1 if wealth > 0 else -1.0,
        "zero_RF_Sharpe": float(daily.mean() / vol * np.sqrt(252)) if vol > 1e-12 else np.nan,
        "annualized_volatility": vol * np.sqrt(252),
        "max_drawdown": float(dd.min()),
        "worst_calendar_month_return": float(monthly.min()),
        "sessions": len(daily),
        "turnover": float(result.turnover.sum()),
        "trade_count": len(orders),
        "target_position_changes": int(target_changes.sum()),
        "bars_with_trades": int(result.turnover.gt(1e-12).sum()),
        "cash_weight_bar_mean": float((result.cash / result.equity).mean()),
        "invested_weight_bar_mean": float(result.holdings.actual_weight.mean()),
        "effective_exposure_bar_mean": float(result.holdings.effective_exposure.mean()),
        "receivable_weight_bar_mean": float(
            (result.holdings.receivables / result.holdings.equity).mean()
        ),
        "trading_fees_before_terminal": float(orders.fee.sum()),
        "slippage_before_terminal": float(orders.slippage.sum()),
        "terminal_exit_fees": terminal["exit_fees"],
        "terminal_exit_slippage": terminal["exit_slippage"],
        "terminal_taxes_paid": terminal["terminal_taxes_paid"],
        "terminal_receivables": terminal["receivables"],
        "terminal_unpaid_distribution_tax_reserve": terminal["unpaid_distribution_tax_reserve"],
        "tax_provisional": bool(terminal["tax_provisional"]),
    }


def bootstrap_boundary_returns(result, terminal: dict, *, initial_cash: float = 1000.0):
    """Remove observed first-entry loss; terminal and entry charged once per draw.

    Later turnover charges remain inside the paired return observations. A
    never-invested track has entry factor1. This conditional observed-boundary
    convention is not a simulation of newly optimized/re-executed strategies.
    """
    returns = result.returns.copy()
    buys = result.events.loc[result.events.event.eq("buy")]
    entry_factor = 1.0
    if len(buys):
        first = buys.iloc[0]
        cost = float(first.fee + first.slippage)
        entry_factor = 1.0 - cost / initial_cash
        if not 0 < entry_factor <= 1:
            raise ValueError("Invalid first entry charge multiplier")
        timestamp = pd.Timestamp(first.timestamp)
        returns.loc[timestamp] = (1 + returns.loc[timestamp]) / entry_factor - 1
    marked = float(terminal["marked_equity"])
    exit_factor = float(terminal["liquidated_equity"]) / marked
    if not np.isfinite(exit_factor) or not 0 < exit_factor <= 1 + 1e-10:
        raise ValueError("Primary pretax terminal multiplier must be positive and at most one")
    daily = _daily(returns)
    reconstructed = float((1 + daily).prod()) * entry_factor * exit_factor
    if not np.isclose(
        reconstructed, float(terminal["liquidated_equity"]) / initial_cash, rtol=1e-9, atol=1e-11
    ):
        raise ValueError("Observed entry/terminal boundary factors fail wealth reconciliation")
    return (
        daily,
        entry_factor * exit_factor,
        {"entry_factor": entry_factor, "terminal_factor": exit_factor},
    )


def paired_compound_bootstrap(
    daily_returns: pd.DataFrame,
    *,
    boundary_factors=(1.0, 1.0),
    n_simulations: int = 5000,
    seed: int = 42,
    block_lengths: tuple[int, ...] = (20, 60),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired circular stationary bootstrap of adjusted daily net returns.

    The first column is scaled B5 and second QQQ. Same sampled session indices
    preserve pairing. Output deltas are percentage points, not decimal returns.
    Block20 is primary; block60 is a prespecified dependence sensitivity, never
    selected according to a more favorable confidence interval.
    """
    values = daily_returns.to_numpy(dtype=float)
    factors = np.asarray(boundary_factors, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
        raise ValueError("Exactly two aligned return columns and at least two sessions required")
    if (
        not np.isfinite(values).all()
        or (values <= -1).any()
        or not daily_returns.index.is_unique
        or not daily_returns.index.is_monotonic_increasing
    ):
        raise ValueError("Finite unique paired returns greater than -100% required")
    if factors.shape != (2,) or not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("Two positive boundary factors required")
    if (
        isinstance(n_simulations, bool)
        or not isinstance(n_simulations, (int, np.integer))
        or n_simulations < 1
        or isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or seed < 0
        or not block_lengths
        or len(set(block_lengths)) != len(block_lengths)
    ):
        raise ValueError(
            "Positive integer simulation count, nonnegative seed and unique blocks required"
        )
    observed = (np.prod(1 + values, axis=0) * factors - 1) * 100
    summaries, all_samples = [], []
    logs, n = np.log1p(values), len(values)
    for block in block_lengths:
        if isinstance(block, bool) or int(block) != block or block < 1:
            raise ValueError("Block means must be positive integer sessions")
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(block)]))
        position = rng.integers(0, n, size=n_simulations)
        accumulated = logs[position].copy()
        for _ in range(1, n):
            restart = rng.random(n_simulations) < 1.0 / block
            position = np.where(restart, rng.integers(0, n, size=n_simulations), (position + 1) % n)
            accumulated += logs[position]
        wealth = np.exp(accumulated) * factors
        deltas = (wealth[:, 0] - wealth[:, 1]) * 100
        low, median, high = np.quantile(deltas, [0.025, 0.5, 0.975])
        summaries.append(
            {
                "block_sessions": block,
                "simulations": n_simulations,
                "seed": seed,
                "observed_delta_pct_points": float(observed[0] - observed[1]),
                "delta_percentile_025": float(low),
                "delta_median": float(median),
                "delta_percentile_975": float(high),
                "role": "primary_block" if block == 20 else "dependence_sensitivity",
                "interpretation": "conditional_paired_return_uncertainty_not_future_risk",
            }
        )
        all_samples.append(
            pd.DataFrame(
                {
                    "block_sessions": block,
                    "simulation": np.arange(n_simulations),
                    "delta_pct_points": deltas,
                }
            )
        )
    return pd.DataFrame(summaries), pd.concat(all_samples, ignore_index=True)


def evaluate_oos_checkpoint(
    close_prices: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    policy_spec: dict | None = None,
    frozen_multiplier: float,
    holdout_start: Any,
    checkpoint_months: int,
    as_of: Any,
    corporate_actions: pd.DataFrame | list[dict] | None = None,
    checkpoint_end: Any | None = None,
    initial_cash: float = 1000.0,
    fee_bps: float = 1.0,
    slippage_bps: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0),
    include_tax_scenarios: bool = True,
    include_gross_companions: bool = True,
    include_attribution: bool = True,
    n_bootstrap: int = 5000,
    seed: int = 42,
    block_lengths: tuple[int, ...] = (20, 60),
) -> OOSCheckpointResult:
    """Score a FULL elapsed horizon; never manufacture early or placeholder P&L.

    Replaying from inception through each checkpoint preserves one continuous
    account. Liquidation/tax snapshots are deep copies, so a 12-month preliminary
    report cannot sell holdings or reset the 24-month primary experiment.
    """
    start, now = _aware(holdout_start, "holdout_start"), _aware(as_of, "as_of")
    cutoff = checkpoint_end_for_months(calendar, start, checkpoint_months)
    if checkpoint_end is not None and _aware(checkpoint_end, "checkpoint_end") != cutoff:
        raise ValueError("Checkpoint cutoff differs from the frozen calendar anniversary rule")
    if now < cutoff:
        raise IncompleteHoldoutError(
            f"Checkpoint not complete until {cutoff.isoformat()}; no results emitted"
        )
    if (
        not np.isfinite(initial_cash)
        or initial_cash <= 0
        or not np.isfinite(fee_bps)
        or fee_bps < 0
    ):
        raise ValueError("Positive finite initial capital and nonnegative fee required")
    if (
        5.0 not in slippage_bps
        or len(set(slippage_bps)) != len(slippage_bps)
        or any(not np.isfinite(s) or s < 0 for s in slippage_bps)
    ):
        raise ValueError("Include baseline5bps and unique nonnegative sensitivity slippages")
    schedule = completed_bar_schedule(calendar)
    panel = _prices_through(close_prices, cutoff, schedule)
    expected = schedule.index[(schedule.session_open >= start) & (schedule.index <= cutoff)]
    if not panel.index[panel.index >= start].equals(expected):
        raise IncompleteHoldoutError("Holdout is missing a scheduled completed bar")
    actions = _actions_through(corporate_actions, cutoff)
    normalized = causal_split_adjusted_prices(panel, actions)
    scaled = _fixed_spec(policy_spec, frozen_multiplier)
    unscaled = _fixed_spec(policy_spec, 1.0)
    _features_ready(normalized, scaled, start)
    paths, snapshots, daily, rows = {}, {}, {}, []
    scenarios = (
        [(s, False, False) for s in slippage_bps]
        + ([(5.0, True, False)] if include_tax_scenarios else [])
        + ([(0.0, False, True)] if include_gross_companions else [])
    )
    for slip, taxable, gross in scenarios:
        scenario_fee = 0.0 if gross else fee_bps
        for track, spec in [
            ("b5_scaled", scaled),
            ("qqq_buy_hold", scaled),
            ("b5_unscaled", unscaled),
            ("cash", unscaled),
            ("tqqq_buy_hold", unscaled),
            ("qqq_200ma", unscaled),
        ]:
            key = (
                f"{track}__before_investor_trading_costs"
                if gross
                else f"{track}__slippage_{slip:g}" + ("__hypothetical_tax" if taxable else "")
            )
            result = _run_track(
                panel,
                normalized,
                actions,
                schedule,
                spec,
                start,
                track=track,
                fee_bps=scenario_fee,
                slippage_bps=slip,
                initial_cash=initial_cash,
                taxable=taxable,
            )
            snapshot = terminal_snapshot(result.ledger, cutoff, settle_tax=taxable)
            dr = _daily(result.returns)
            # Clone-only exit charge is attached once to the last observed day.
            dr.iloc[-1] = (1 + dr.iloc[-1]) * snapshot["liquidated_equity"] / snapshot[
                "marked_equity"
            ] - 1
            row = _metric_row(result, snapshot, dr, initial_cash=initial_cash)
            row.update(
                track=track,
                scenario=key,
                slippage_bps=slip,
                fee_bps=scenario_fee,
                hypothetical_taxes=taxable,
                before_investor_trading_costs=gross,
                actual_fund_costs_embedded=True,
                traded_asset="QQQ"
                if track in {"qqq_buy_hold", "qqq_200ma"}
                else "TQQQ"
                if track != "cash"
                else "none",
                role="primary_comparison"
                if not taxable
                and not gross
                and slip == 5
                and track in {"b5_scaled", "qqq_buy_hold"}
                else "secondary",
            )
            paths[key], snapshots[key], daily[key] = result, snapshot, dr
            rows.append(row)
    keys = ["b5_scaled__slippage_5", "qqq_buy_hold__slippage_5"]
    paired, factors, boundary = {}, [], {}
    for key in keys:
        series, factor, info = bootstrap_boundary_returns(
            paths[key], snapshots[key], initial_cash=initial_cash
        )
        paired[key], boundary[key] = series, info
        factors.append(factor)
    bootstrap, samples = paired_compound_bootstrap(
        pd.DataFrame(paired),
        boundary_factors=factors,
        n_simulations=n_bootstrap,
        seed=seed,
        block_lengths=block_lengths,
    )
    comparison_delta = (
        (snapshots[keys[0]]["liquidated_equity"] - snapshots[keys[1]]["liquidated_equity"])
        / initial_cash
        * 100
    )
    attribution_rows = []
    if include_attribution:
        for track in [
            "b5_scaled",
            "qqq_buy_hold",
            "b5_unscaled",
            "cash",
            "tqqq_buy_hold",
            "qqq_200ma",
        ]:
            baseline_key = f"{track}__slippage_5"
            baseline = paths[baseline_key]
            fixed_targets = baseline.holdings.set_index("timestamp").target_weight
            layers = [
                ("before_investor_trading_costs", 0.0, 0.0, False),
                ("net_trading", fee_bps, 5.0, False),
            ]
            if include_tax_scenarios:
                layers.append(("hypothetical_tax", fee_bps, 5.0, True))
            for label, layer_fee, layer_slip, taxable in layers:
                if label == "net_trading":
                    result, snapshot = baseline, snapshots[baseline_key]
                else:
                    result = _run_track(
                        panel,
                        normalized,
                        actions,
                        schedule,
                        scaled if track == "b5_scaled" else unscaled,
                        start,
                        track=track,
                        fee_bps=layer_fee,
                        slippage_bps=layer_slip,
                        initial_cash=initial_cash,
                        taxable=taxable,
                        fixed_targets=fixed_targets,
                    )
                    snapshot = terminal_snapshot(result.ledger, cutoff, settle_tax=taxable)
                dr = _daily(result.returns)
                dr.iloc[-1] = (1 + dr.iloc[-1]) * snapshot["liquidated_equity"] / snapshot[
                    "marked_equity"
                ] - 1
                row = _metric_row(result, snapshot, dr, initial_cash=initial_cash)
                end_to_end_key = (
                    f"{track}__before_investor_trading_costs"
                    if label == "before_investor_trading_costs"
                    else baseline_key + ("__hypothetical_tax" if taxable else "")
                )
                e2e = snapshots.get(end_to_end_key)
                e2e_return = (e2e["liquidated_equity"] / initial_cash - 1) * 100 if e2e else np.nan
                row.update(
                    track=track,
                    attribution_layer=label,
                    path_mode="baseline_fixed_target_allocations_not_fixed_share_orders",
                    baseline_target_source=baseline_key,
                    fee_bps=layer_fee,
                    slippage_bps=layer_slip,
                    hypothetical_taxes=taxable,
                    role="secondary_attribution",
                    end_to_end_total_return_pct=e2e_return,
                    end_to_end_minus_matched_target_pct_points=e2e_return - row["total_return_pct"],
                )
                attribution_rows.append(row)
    attribution = pd.DataFrame(attribution_rows)
    if len(attribution):
        gross_by_track = (
            attribution.loc[attribution.attribution_layer.eq("before_investor_trading_costs")]
            .set_index("track")
            .total_return_pct
        )
        attribution["delta_vs_matched_gross_pct_points"] = (
            attribution.total_return_pct - attribution.track.map(gross_by_track)
        )
    return OOSCheckpointResult(
        pd.DataFrame(rows),
        pd.DataFrame(daily),
        bootstrap,
        samples,
        paths,
        snapshots,
        {
            "study_kind": "prospective_frozen_policy_backtest_not_real_execution",
            "checkpoint_months": checkpoint_months,
            "checkpoint_role": "primary"
            if checkpoint_months == 24
            else "preliminary_no_success_decision",
            "holdout_start": start.isoformat(),
            "checkpoint_end": cutoff.isoformat(),
            "scoring_as_of": now.isoformat(),
            "primary_statistic": "net_compounded_total_return_delta_percentage_points_scaled_B5_minus_QQQ",
            "primary_delta_pct_points": comparison_delta,
            "frozen_multiplier": float(frozen_multiplier),
            "cash_yield": 0.0,
            "benchmark_entry": "second_completed_holdout_bar_close_independent_of_B5",
            "secondary_comparators": "cash_zero_yield_actual_TQQQ_buyhold_actual_QQQ_200MA_unscaled_B5",
            "gross_companion": "zero_investor_fees_and_slippage_but_actual_fund_expenses_financing_remain_embedded",
            "cost_attribution": "separate_baseline_fixed_target_weights_not_fixed_shares_quantities_and_tax_funding_sales_may_differ",
            "annualization": "252_sessions_per_year_sample_daily_volatility_ddof1_CAGR_252_over_session_count",
            "drawdown_convention": "daily_close_wealth_including_one_cloned_exit_charge_not_intrabar_drawdown",
            "monthly_convention": "calendar_month_compounding_including_partial_first_last_months",
            "exposure_convention": "arithmetic_bar_mean_of_marked_close_weights_including_drift_receivables_separate",
            "terminal_convention": "clone_only_liquidation_costs_checkpoint_never_resets_continuous_account",
            "bootstrap_boundary_factors": boundary,
            "bootstrap_primary_block_sessions": 20,
            "bootstrap_limitation": "paired_observed_return_uncertainty_not_strategy_reexecution_or_future_loss_probability",
            "rates": "no_risk_free_or_financing_series_invented_zero_RF_Sharpe_secondary_only",
            "financing_companion": "unavailable_without_separately_supplied_causal_DFF_no_actual_fund_double_count",
            "taxes": "secondary_ST24_LT15_cash24_unknown_character_provisional_not_personal_tax_liability",
            "source_provenance_requirement": "caller_must_verify_frozen_code_parameters_data_hashes_and_release_identity",
            "warmup_minimum_observations_at_first_decision": required_warmup_observations(scaled),
            "success_decision": "not_automatic_12month_is_preliminary_24month_primary_must_report_uncertainty",
        },
        attribution,
    )


def synthetic_financing_companion(
    underlying_total_return_index: pd.Series,
    calendar: pd.DataFrame,
    financing_observations: pd.DataFrame,
    *,
    leverage: float = 3.0,
    spread_bps: float = 50.0,
    expense_ratio: float = 0.0095,
    max_rate_age_days: float = 10.0,
) -> pd.DataFrame:
    """Secondary daily-close-reset leveraged QQQ model, NEVER actual TQQQ.

    Requires an independently sourced, causal QQQ total-return index and DFF
    observations with modeled/verified available_at timestamps. Do not pass
    actual TQQQ returns: deducting financing from them would double count it.
    Shares are fixed intraday, debt accrues at DFF+spread ACT/360 over calendar
    intervals, expense accrues ACT/365, then leverage resets at each session
    close. This is a simplified synthetic self-financing companion, not a claim
    to reproduce an ETF's swaps, tracking, distributions, expenses or tax basis.
    No missing/stale rate is zero-filled. Policy/transaction/tax costs are NOT
    included here and must be modeled separately if this NAV is traded.
    """
    if not isinstance(underlying_total_return_index.index, pd.DatetimeIndex):
        raise ValueError("Causal underlying total-return index requires timestamped observations")
    if underlying_total_return_index.index.tz is None:
        raise ValueError("Underlying total-return timestamps must be timezone-aware")
    if "TQQQ" in str(underlying_total_return_index.name).upper():
        raise ValueError("Actual TQQQ input would double count embedded financing")
    if (
        not np.isfinite([leverage, spread_bps, expense_ratio]).all()
        or leverage < 1
        or expense_ratio < 0
    ):
        raise ValueError("Finite leverage>=1 and nonnegative expense ratio required")
    underlying = underlying_total_return_index.astype(float).copy()
    underlying.index = underlying.index.tz_convert("UTC")
    if (
        underlying.empty
        or not underlying.index.is_unique
        or not underlying.index.is_monotonic_increasing
        or not np.isfinite(underlying).all()
        or underlying.le(0).any()
    ):
        raise ValueError("Positive, finite, chronological causal underlying index required")
    schedule = completed_bar_schedule(calendar)
    expected = schedule.index[
        (schedule.index >= underlying.index[0]) & (schedule.index <= underlying.index[-1])
    ]
    if not underlying.index.equals(expected):
        raise IncompleteHoldoutError(
            "Synthetic financing companion requires complete scheduled bars"
        )
    if not {"series_id", "available_at", "annual_rate"}.issubset(financing_observations):
        raise ValueError("Timestamp-available DFF observations are required, not ETF cash returns")
    available = pd.DatetimeIndex(
        [_aware(x, "rate available_at") for x in financing_observations.available_at]
    )
    known = financing_observations.loc[available <= underlying.index[-1]].copy()
    if not known.series_id.eq("DFF").all():
        raise ValueError("The frozen financing companion requires DFF, not a cash benchmark")
    financing = rate_accrual(
        underlying.index,
        known,
        spread_bps=spread_bps,
        day_count=360.0,
        max_age_days=max_rate_age_days,
    )
    years = underlying.index.to_series().diff().dt.total_seconds().fillna(0) / (365 * 86400)
    shares, cash = leverage / underlying.iloc[0], 1.0 - leverage
    rows = []
    for i, (timestamp, price) in enumerate(underlying.items()):
        financing_cost = max(-cash, 0.0) * float(financing.iloc[i])
        marked = shares * price + cash
        expense_cost = max(marked, 0.0) * expense_ratio * float(years.iloc[i])
        cash -= financing_cost + expense_cost
        nav = shares * price + cash
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError("Synthetic NAV exhausted; bankruptcy cannot be clipped away")
        reset = timestamp == schedule.loc[timestamp, "session_close"]
        if reset:
            shares, cash = leverage * nav / price, (1.0 - leverage) * nav
        rows.append(
            dict(
                timestamp=timestamp,
                synthetic_nav=nav,
                financing_cost=financing_cost,
                expense_cost=expense_cost,
                debt=max(-cash, 0.0),
                reset_at_close=reset,
            )
        )
    result = pd.DataFrame(rows).set_index("timestamp")
    result.attrs.update(
        model="synthetic_QQQ_total_return_daily_close_reset_not_actual_TQQQ",
        leverage=leverage,
        spread_bps=spread_bps,
        expense_ratio=expense_ratio,
        financing_day_count="ACT/360",
        expense_day_count="ACT/365",
        cash_yield=0.0,
        investor_trading_costs_included=False,
        tax_model_included=False,
        underlying_requirement="independently_causal_QQQ_total_return_index_with_provenance",
    )
    return result
