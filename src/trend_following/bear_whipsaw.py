"""Bear-market whipsaw overlays for the QQQ / synthetic-TQQQ case study.

These helpers operate on raw close-bar desired weights.  They intentionally do
not perform the project's no-lookahead execution shift; callers should pass the
result through the existing executable-weight conversion before applying
returns.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WhipsawOverlayResult:
    """Overlay output and diagnostics."""

    weights: pd.Series
    diagnostics: pd.DataFrame


def bear_market_features(
    qqq_price: pd.Series,
    *,
    bars_per_day: int = 6,
    long_ma_days: int = 200,
    medium_ma_days: int = 50,
    short_ma_days: int = 20,
    slope_days: tuple[int, ...] = (5, 10, 20),
    vol_window_days: int = 20,
    vol_percentile_days: int = 252,
) -> pd.DataFrame:
    """Return QQQ features used by bear-whipsaw overlays."""
    price = qqq_price.astype(float).sort_index()
    long_window = long_ma_days * bars_per_day
    medium_window = medium_ma_days * bars_per_day
    short_window = short_ma_days * bars_per_day
    ma_long = price.rolling(long_window, min_periods=long_window).mean()
    ma_medium = price.rolling(medium_window, min_periods=medium_window).mean()
    ma_short = price.rolling(short_window, min_periods=short_window).mean()

    returns = price.pct_change(fill_method=None)
    vol_window = vol_window_days * bars_per_day
    vol_lookback = vol_percentile_days * bars_per_day
    realized_vol = returns.rolling(vol_window, min_periods=vol_window).std(ddof=0) * np.sqrt(
        252 * bars_per_day
    )
    vol_percentile = realized_vol.rolling(vol_lookback, min_periods=vol_lookback).apply(
        _last_value_percentile,
        raw=True,
    )

    features = pd.DataFrame(
        {
            "qqq_price": price,
            "ma_long": ma_long,
            "ma_medium": ma_medium,
            "ma_short": ma_short,
            "distance_to_long_ma": price / ma_long - 1.0,
            "short_gt_medium": ma_short.gt(ma_medium),
            "realized_vol_20d": realized_vol,
            # Conservative: use the percentile known before this close-bar.
            "realized_vol_percentile_known": vol_percentile.shift(1),
        },
        index=price.index,
    )
    for days in slope_days:
        bars = max(int(days * bars_per_day), 1)
        features[f"long_ma_slope_{days}d"] = ma_long.diff(bars)
        features[f"medium_ma_slope_{days}d"] = ma_medium.diff(bars)
    return features


def bear_reentry_filter_raw(
    base_raw: pd.Series,
    features: pd.DataFrame,
    *,
    distance_buffer: float,
    slope_days: int,
    require_short_gt_medium: bool = False,
) -> WhipsawOverlayResult:
    """Delay new entries in negative-long-MA-slope regimes until confirmation."""
    base = base_raw.fillna(0.0).astype(float)
    feat = features.reindex(base.index)
    long_col = f"long_ma_slope_{slope_days}d"
    medium_col = f"medium_ma_slope_{slope_days}d"
    _require_columns(feat, ["distance_to_long_ma", long_col, medium_col, "short_gt_medium"])

    in_trade = False
    values: list[float] = []
    blocked: list[bool] = []
    release_ok: list[bool] = []

    for timestamp, desired in base.items():
        if desired <= 0.0:
            in_trade = False
            values.append(0.0)
            blocked.append(False)
            release_ok.append(False)
            continue

        row = feat.loc[timestamp]
        long_slope = float(row[long_col]) if pd.notna(row[long_col]) else np.nan
        negative_long_slope = np.isfinite(long_slope) and long_slope < 0.0
        confirmation = (
            float(row["distance_to_long_ma"]) >= distance_buffer
            and float(row[medium_col]) > 0.0
            and (not require_short_gt_medium or bool(row["short_gt_medium"]))
        )
        should_block = (not in_trade) and negative_long_slope and not confirmation
        if should_block:
            values.append(0.0)
            blocked.append(True)
            release_ok.append(False)
        else:
            in_trade = True
            values.append(1.0)
            blocked.append(False)
            release_ok.append(bool(confirmation))

    weights = pd.Series(values, index=base.index, name=base_raw.name, dtype=float)
    diagnostics = pd.DataFrame(
        {
            "base_raw": base,
            "filtered_raw": weights,
            "blocked_entry": blocked,
            "release_confirmation": release_ok,
        },
        index=base.index,
    )
    return WhipsawOverlayResult(weights=weights, diagnostics=diagnostics)


def failed_breakout_cooldown_raw(
    base_raw: pd.Series,
    traded_price: pd.Series,
    features: pd.DataFrame,
    *,
    weak_trade_return: float = 0.03,
    weak_trade_count: int = 2,
    lookback_days: int = 90,
    slope_days: int = 10,
    distance_buffer: float = 0.02,
) -> WhipsawOverlayResult:
    """Block entries after repeated weak/loss trades until trend confirmation."""
    if weak_trade_count <= 0:
        raise ValueError("weak_trade_count must be positive")
    base = base_raw.fillna(0.0).astype(float)
    price = traded_price.reindex(base.index).astype(float)
    feat = features.reindex(base.index)
    medium_col = f"medium_ma_slope_{slope_days}d"
    _require_columns(feat, ["distance_to_long_ma", medium_col])

    in_trade = False
    cooldown = False
    entry_price = np.nan
    weak_exits: deque[pd.Timestamp] = deque()
    values: list[float] = []
    blocked: list[bool] = []
    releases: list[bool] = []
    cooldown_flags: list[bool] = []
    weak_exit_flags: list[bool] = []

    for timestamp, desired in base.items():
        current_price = float(price.loc[timestamp]) if np.isfinite(price.loc[timestamp]) else np.nan
        weak_exit = False
        release = False
        if desired <= 0.0 or not np.isfinite(current_price):
            if in_trade and np.isfinite(entry_price) and np.isfinite(current_price):
                trade_return = current_price / entry_price - 1.0
                if trade_return <= weak_trade_return:
                    weak_exit = True
                    weak_exits.append(pd.Timestamp(timestamp))
            in_trade = False
            entry_price = np.nan
            _drop_old_weak_exits(weak_exits, pd.Timestamp(timestamp), lookback_days)
            if len(weak_exits) >= weak_trade_count:
                cooldown = True
            values.append(0.0)
            blocked.append(False)
            releases.append(False)
            cooldown_flags.append(cooldown)
            weak_exit_flags.append(weak_exit)
            continue

        _drop_old_weak_exits(weak_exits, pd.Timestamp(timestamp), lookback_days)
        if cooldown and _trend_release(feat.loc[timestamp], medium_col, distance_buffer):
            cooldown = False
            release = True
            weak_exits.clear()

        if cooldown and not in_trade:
            values.append(0.0)
            blocked.append(True)
        else:
            if not in_trade:
                entry_price = current_price
            in_trade = True
            values.append(1.0)
            blocked.append(False)
        releases.append(release)
        cooldown_flags.append(cooldown)
        weak_exit_flags.append(weak_exit)

    weights = pd.Series(values, index=base.index, name=base_raw.name, dtype=float)
    diagnostics = pd.DataFrame(
        {
            "base_raw": base,
            "cooldown_raw": weights,
            "blocked_by_cooldown": blocked,
            "cooldown_release": releases,
            "cooldown_active": cooldown_flags,
            "weak_exit": weak_exit_flags,
        },
        index=base.index,
    )
    return WhipsawOverlayResult(weights=weights, diagnostics=diagnostics)


def volatility_cap(
    features: pd.DataFrame,
    *,
    percentile_threshold: float,
    defensive_cap: float,
) -> WhipsawOverlayResult:
    """Cap exposure when known realized-volatility percentile is high."""
    if not 0.0 <= defensive_cap <= 1.0:
        raise ValueError("defensive_cap must be between 0 and 1")
    _require_columns(features, ["realized_vol_percentile_known"])
    high_vol = features["realized_vol_percentile_known"].ge(percentile_threshold).fillna(False)
    cap = pd.Series(1.0, index=features.index, name="volatility_cap", dtype=float)
    cap.loc[high_vol] = defensive_cap
    diagnostics = pd.DataFrame(
        {
            "volatility_cap": cap,
            "volatility_cap_active": high_vol,
            "realized_vol_percentile_known": features["realized_vol_percentile_known"],
        },
        index=features.index,
    )
    return WhipsawOverlayResult(weights=cap, diagnostics=diagnostics)


def portfolio_drawdown_circuit_breaker(
    raw_weight: pd.Series,
    asset_returns: pd.Series,
    features: pd.DataFrame,
    *,
    trigger_drawdown: float,
    recover_drawdown: float,
    defensive_cap: float,
    slope_days: int = 10,
) -> WhipsawOverlayResult:
    """Cap exposure using prior realized strategy drawdown."""
    if not 0.0 <= defensive_cap <= 1.0:
        raise ValueError("defensive_cap must be between 0 and 1")
    raw = raw_weight.fillna(0.0).astype(float)
    returns = asset_returns.reindex(raw.index).fillna(0.0).astype(float)
    feat = features.reindex(raw.index)
    medium_col = f"medium_ma_slope_{slope_days}d"
    _require_columns(feat, [medium_col])

    equity = 1.0
    peak = 1.0
    defensive = False
    caps: list[float] = []
    prior_drawdowns: list[float] = []
    triggers: list[bool] = []
    releases: list[bool] = []
    equity_path: list[float] = []

    for timestamp, desired in raw.items():
        prior_dd = equity / peak - 1.0
        trigger = False
        release = False
        if not defensive and prior_dd <= -abs(trigger_drawdown):
            defensive = True
            trigger = True
        if defensive and (
            prior_dd >= -abs(recover_drawdown) or float(feat.at[timestamp, medium_col]) > 0.0
        ):
            defensive = False
            release = True
        cap = defensive_cap if defensive else 1.0
        weight = min(float(desired), cap)
        equity *= 1.0 + weight * float(returns.loc[timestamp])
        peak = max(peak, equity)
        caps.append(cap)
        prior_drawdowns.append(prior_dd)
        triggers.append(trigger)
        releases.append(release)
        equity_path.append(equity)

    cap_series = pd.Series(caps, index=raw.index, name="circuit_breaker_cap", dtype=float)
    diagnostics = pd.DataFrame(
        {
            "circuit_breaker_cap": cap_series,
            "circuit_breaker_trigger": triggers,
            "circuit_breaker_release": releases,
            "prior_strategy_drawdown": prior_drawdowns,
            "approx_overlay_equity": equity_path,
        },
        index=raw.index,
    )
    return WhipsawOverlayResult(weights=cap_series, diagnostics=diagnostics)


def two_stage_bear_reentry_cap(
    raw_weight: pd.Series,
    features: pd.DataFrame,
    *,
    initial_weight: float,
    slope_days: int,
    release_rule: str,
) -> WhipsawOverlayResult:
    """Enter at reduced size in negative-long-MA-slope regimes, then release."""
    if not 0.0 <= initial_weight <= 1.0:
        raise ValueError("initial_weight must be between 0 and 1")
    raw = raw_weight.fillna(0.0).astype(float)
    feat = features.reindex(raw.index)
    long_col = f"long_ma_slope_{slope_days}d"
    medium_col = f"medium_ma_slope_{slope_days}d"
    _require_columns(feat, [long_col, medium_col, "short_gt_medium"])

    in_trade = False
    staged = False
    caps: list[float] = []
    staged_flags: list[bool] = []
    releases: list[bool] = []

    for timestamp, desired in raw.items():
        release = False
        if desired <= 0.0:
            in_trade = False
            staged = False
            caps.append(0.0)
            staged_flags.append(False)
            releases.append(False)
            continue
        row = feat.loc[timestamp]
        if not in_trade:
            in_trade = True
            long_slope = float(row[long_col]) if pd.notna(row[long_col]) else np.nan
            staged = np.isfinite(long_slope) and long_slope < 0.0
        if staged and _stage_release(row, medium_col, release_rule):
            staged = False
            release = True
        caps.append(initial_weight if staged else 1.0)
        staged_flags.append(staged)
        releases.append(release)

    cap_series = pd.Series(caps, index=raw.index, name="two_stage_cap", dtype=float)
    diagnostics = pd.DataFrame(
        {
            "two_stage_cap": cap_series,
            "two_stage_staged": staged_flags,
            "two_stage_release": releases,
        },
        index=raw.index,
    )
    return WhipsawOverlayResult(weights=cap_series, diagnostics=diagnostics)


def _last_value_percentile(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return np.nan
    return float((clean <= clean[-1]).sum() / clean.size)


def _drop_old_weak_exits(
    weak_exits: deque[pd.Timestamp],
    timestamp: pd.Timestamp,
    lookback_days: int,
) -> None:
    cutoff = timestamp - pd.Timedelta(days=lookback_days)
    while weak_exits and weak_exits[0] < cutoff:
        weak_exits.popleft()


def _trend_release(row: pd.Series, medium_col: str, distance_buffer: float) -> bool:
    medium_slope = float(row[medium_col]) if pd.notna(row[medium_col]) else np.nan
    distance = float(row["distance_to_long_ma"]) if pd.notna(row["distance_to_long_ma"]) else np.nan
    return (np.isfinite(medium_slope) and medium_slope > 0.0) or (
        np.isfinite(distance) and distance >= distance_buffer
    )


def _stage_release(row: pd.Series, medium_col: str, release_rule: str) -> bool:
    if release_rule == "medium_slope":
        value = row[medium_col]
        return bool(pd.notna(value) and float(value) > 0.0)
    if release_rule == "short_gt_medium":
        return bool(row["short_gt_medium"])
    raise ValueError("release_rule must be one of: medium_slope, short_gt_medium")


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required feature columns: {missing}")
