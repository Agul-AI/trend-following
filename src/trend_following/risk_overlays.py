"""Risk overlays for the QQQ / synthetic-TQQQ research case study.

The functions in this module operate on **raw** close-bar strategy states.  They
do not shift signals into executable positions.  Callers should pass the output
through :func:`trend_following.signals.make_executable_positions` (or the
project's ``executable_weights`` helper) before applying returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OverlayDiagnostics:
    """Container for overlay output and diagnostics."""

    weights: pd.Series
    diagnostics: pd.DataFrame


def qqq_mean_reversion_features(
    qqq_price: pd.Series,
    *,
    bars_per_day: int = 6,
    long_ma_days: int = 200,
    short_ma_days: int = 20,
    medium_ma_days: int = 50,
    z_window_days: int = 20,
) -> pd.DataFrame:
    """Return QQQ features used by mean-reversion overlays.

    Distances are expressed as decimal fractions, e.g. ``0.20`` means QQQ is
    20% above the reference moving average.
    """
    price = qqq_price.astype(float).sort_index()
    ma_long = price.rolling(
        long_ma_days * bars_per_day, min_periods=long_ma_days * bars_per_day
    ).mean()
    ma_short = price.rolling(
        short_ma_days * bars_per_day, min_periods=short_ma_days * bars_per_day
    ).mean()
    ma_medium = price.rolling(
        medium_ma_days * bars_per_day, min_periods=medium_ma_days * bars_per_day
    ).mean()
    z_window = z_window_days * bars_per_day
    rolling_mean = price.rolling(z_window, min_periods=z_window).mean()
    rolling_std = price.rolling(z_window, min_periods=z_window).std(ddof=0)
    zscore = (price - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "qqq_price": price,
            "ma_long": ma_long,
            "ma_short": ma_short,
            "ma_medium": ma_medium,
            "distance_to_long_ma": price / ma_long - 1.0,
            "zscore": zscore,
            "above_long_ma": price.gt(ma_long),
            "touch_short_ma": price.le(ma_short),
            "touch_medium_ma": price.le(ma_medium),
            "zscore_le_0": zscore.le(0.0),
        },
        index=price.index,
    )


def trade_profit_lock_tiers(
    base_raw: pd.Series,
    price: pd.Series,
    *,
    thresholds_to_weights: list[tuple[float, float]],
) -> pd.Series:
    """Reduce size within a trade after unrealized-gain thresholds are crossed."""
    thresholds_to_weights = sorted(thresholds_to_weights)
    in_trade = False
    entry_price = np.nan
    current_weight = 0.0
    values: list[float] = []

    for signal, current_price in zip(
        base_raw.fillna(0.0), price.reindex(base_raw.index), strict=True
    ):
        if signal <= 0.0 or not np.isfinite(current_price):
            in_trade = False
            entry_price = np.nan
            current_weight = 0.0
            values.append(0.0)
            continue

        if not in_trade:
            in_trade = True
            entry_price = float(current_price)
            current_weight = 1.0

        gain = float(current_price) / entry_price - 1.0 if entry_price > 0 else 0.0
        for threshold, weight in thresholds_to_weights:
            if gain >= threshold:
                current_weight = min(current_weight, weight)
        values.append(current_weight)

    return pd.Series(values, index=base_raw.index, name=base_raw.name, dtype=float)


def raw_with_peak_drawdown_stop(
    base_raw: pd.Series,
    traded_price: pd.Series,
    *,
    stop_drawdown: float | pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    """Force raw signal to cash if trade-level peak drawdown breaches threshold.

    ``stop_drawdown`` can be a constant positive fraction, e.g. ``0.40``, or a
    timestamp-aligned Series for dynamic regimes such as Fed hiking cycles.
    """
    base = base_raw.fillna(0.0).astype(float)
    price = traded_price.reindex(base.index).astype(float)
    if isinstance(stop_drawdown, pd.Series):
        thresholds = -stop_drawdown.reindex(base.index).astype(float).abs()
    else:
        thresholds = pd.Series(-abs(float(stop_drawdown)), index=base.index, dtype=float)

    in_trade = False
    stopped_until_base_exit = False
    peak = np.nan
    values: list[float] = []
    peaks: list[float] = []
    drawdowns: list[float] = []
    triggers: list[bool] = []

    for base_signal, current_price, threshold in zip(base, price, thresholds, strict=True):
        current_price = float(current_price) if np.isfinite(current_price) else np.nan
        threshold = float(threshold) if np.isfinite(threshold) else -np.inf
        trigger = False
        if base_signal <= 0.0 or not np.isfinite(current_price):
            in_trade = False
            stopped_until_base_exit = False
            peak = np.nan
            value = 0.0
            drawdown = np.nan
        else:
            if not in_trade:
                in_trade = True
                stopped_until_base_exit = False
                peak = current_price
            else:
                peak = max(float(peak), current_price)
            drawdown = current_price / peak - 1.0 if peak > 0 else np.nan
            if stopped_until_base_exit:
                value = 0.0
            elif np.isfinite(drawdown) and drawdown <= threshold:
                trigger = True
                stopped_until_base_exit = True
                value = 0.0
            else:
                value = 1.0
        values.append(value)
        peaks.append(peak)
        drawdowns.append(drawdown)
        triggers.append(trigger)

    stopped = pd.Series(values, index=base.index, name=base_raw.name, dtype=float)
    diagnostics = pd.DataFrame(
        {
            "base_raw": base,
            "stopped_raw": stopped,
            "trade_peak_price": peaks,
            "trade_peak_drawdown": drawdowns,
            "stop_threshold": thresholds,
            "stop_trigger": triggers,
        },
        index=base.index,
    )
    return stopped, diagnostics


def extension_trim_rebuy_cap(
    base_raw: pd.Series,
    traded_price: pd.Series,
    qqq_features: pd.DataFrame,
    *,
    activation_gain: float,
    distance_threshold: float,
    trim_weight: float,
    reentry_rule: str,
) -> OverlayDiagnostics:
    """Cap weight after a large gain and QQQ overextension, then re-add on pullback."""
    return _stateful_trim_rebuy_cap(
        base_raw=base_raw,
        traded_price=traded_price,
        qqq_features=qqq_features,
        activation_gain=activation_gain,
        trim_weight=trim_weight,
        reentry_rule=reentry_rule,
        trigger_kind="extension",
        trigger_threshold=distance_threshold,
    )


def dynamic_pre100_distance_trim_rebuy_cap(
    base_raw: pd.Series,
    traded_price: pd.Series,
    qqq_features: pd.DataFrame,
    *,
    activation_gain: float = 1.0,
    threshold_quantile: float = 1.0,
    trim_weight: float = 0.50,
    reentry_rule: str = "ma20",
) -> OverlayDiagnostics:
    """Trim after +100%-style gains using a trade-specific learned threshold.

    For each raw long trade, the threshold is learned only once the trade first
    reaches ``activation_gain``.  The threshold is the requested quantile of
    QQQ's distance to its long moving average observed from trade entry through
    that first activation bar.  After that point, the overlay caps exposure to
    ``trim_weight`` whenever QQQ's distance revisits/exceeds the learned
    threshold, and restores full exposure on the chosen mean-reversion re-entry
    rule.

    This is intentionally stateful and uses only information available up to the
    current close-bar signal timestamp.  Callers should still shift the returned
    raw weights into executable positions before applying returns.
    """
    if activation_gain <= 0:
        raise ValueError("activation_gain must be positive")
    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1")
    if not 0.0 <= trim_weight <= 1.0:
        raise ValueError("trim_weight must be between 0 and 1")

    base = base_raw.fillna(0.0).astype(float)
    price = traded_price.reindex(base.index).astype(float)
    features = qqq_features.reindex(base.index)

    in_trade = False
    defensive = False
    entry_price = np.nan
    threshold = np.nan
    learned = False
    pre_activation_distances: list[float] = []
    trade_id = 0

    caps: list[float] = []
    triggers: list[bool] = []
    reentries: list[bool] = []
    gains: list[float] = []
    thresholds: list[float] = []
    learned_flags: list[bool] = []
    pre_activation_counts: list[int] = []
    trade_ids: list[int] = []

    for timestamp, signal in base.items():
        current_price = float(price.loc[timestamp]) if np.isfinite(price.loc[timestamp]) else np.nan
        trigger = False
        reentry = False
        gain = np.nan
        current_trade_id = 0

        if signal <= 0.0 or not np.isfinite(current_price):
            in_trade = False
            defensive = False
            entry_price = np.nan
            threshold = np.nan
            learned = False
            pre_activation_distances = []
            cap = 0.0
        else:
            if not in_trade:
                in_trade = True
                defensive = False
                entry_price = current_price
                threshold = np.nan
                learned = False
                pre_activation_distances = []
                trade_id += 1
            current_trade_id = trade_id
            gain = current_price / entry_price - 1.0 if entry_price > 0 else np.nan

            distance = float(features.at[timestamp, "distance_to_long_ma"])
            if not learned and np.isfinite(distance):
                pre_activation_distances.append(distance)
            if (
                not learned
                and np.isfinite(gain)
                and gain >= activation_gain
                and pre_activation_distances
            ):
                threshold = float(
                    np.nanquantile(
                        np.asarray(pre_activation_distances, dtype=float),
                        threshold_quantile,
                    )
                )
                learned = True

            if defensive and _reentry_condition(features.loc[timestamp], reentry_rule):
                defensive = False
                reentry = True

            if (
                learned
                and np.isfinite(gain)
                and gain >= activation_gain
                and not defensive
                and np.isfinite(distance)
                and np.isfinite(threshold)
                and distance >= threshold
            ):
                defensive = True
                trigger = True

            cap = trim_weight if defensive else 1.0

        caps.append(cap)
        triggers.append(trigger)
        reentries.append(reentry)
        gains.append(gain)
        thresholds.append(threshold)
        learned_flags.append(learned)
        pre_activation_counts.append(len(pre_activation_distances))
        trade_ids.append(current_trade_id)

    cap_series = pd.Series(caps, index=base.index, name="overlay_cap", dtype=float)
    diagnostics = pd.DataFrame(
        {
            "overlay_cap": cap_series,
            "overlay_trigger": triggers,
            "overlay_reentry": reentries,
            "trade_gain": gains,
            "dynamic_distance_threshold": thresholds,
            "threshold_learned": learned_flags,
            "pre_activation_observations": pre_activation_counts,
            "dynamic_trade_id": trade_ids,
        },
        index=base.index,
    )
    return OverlayDiagnostics(weights=cap_series, diagnostics=diagnostics)


def peak_drawdown_trim_rebuy_cap(
    base_raw: pd.Series,
    traded_price: pd.Series,
    qqq_features: pd.DataFrame,
    *,
    activation_gain: float,
    peak_drawdown: float,
    trim_weight: float,
    reentry_rule: str,
) -> OverlayDiagnostics:
    """Cap weight after a post-gain pullback from the synthetic-3x trade peak."""
    return _stateful_trim_rebuy_cap(
        base_raw=base_raw,
        traded_price=traded_price,
        qqq_features=qqq_features,
        activation_gain=activation_gain,
        trim_weight=trim_weight,
        reentry_rule=reentry_rule,
        trigger_kind="peak_drawdown",
        trigger_threshold=peak_drawdown,
    )


def _stateful_trim_rebuy_cap(
    *,
    base_raw: pd.Series,
    traded_price: pd.Series,
    qqq_features: pd.DataFrame,
    activation_gain: float,
    trim_weight: float,
    reentry_rule: str,
    trigger_kind: str,
    trigger_threshold: float,
) -> OverlayDiagnostics:
    """Shared state machine for temporary defensive caps."""
    if not 0.0 <= trim_weight <= 1.0:
        raise ValueError("trim_weight must be between 0 and 1")
    base = base_raw.fillna(0.0).astype(float)
    price = traded_price.reindex(base.index).astype(float)
    features = qqq_features.reindex(base.index)

    in_trade = False
    defensive = False
    entry_price = np.nan
    peak_price = np.nan
    max_gain = 0.0
    caps: list[float] = []
    triggers: list[bool] = []
    reentries: list[bool] = []
    gains: list[float] = []
    peak_drawdowns: list[float] = []

    for timestamp, signal in base.items():
        current_price = float(price.loc[timestamp]) if np.isfinite(price.loc[timestamp]) else np.nan
        trigger = False
        reentry = False
        if signal <= 0.0 or not np.isfinite(current_price):
            in_trade = False
            defensive = False
            entry_price = np.nan
            peak_price = np.nan
            max_gain = 0.0
            cap = 0.0
            gain = np.nan
            peak_dd = np.nan
        else:
            if not in_trade:
                in_trade = True
                defensive = False
                entry_price = current_price
                peak_price = current_price
                max_gain = 0.0
            else:
                peak_price = max(float(peak_price), current_price)
            gain = current_price / entry_price - 1.0 if entry_price > 0 else np.nan
            max_gain = max(max_gain, float(gain) if np.isfinite(gain) else 0.0)
            peak_dd = current_price / peak_price - 1.0 if peak_price > 0 else np.nan

            if defensive and _reentry_condition(features.loc[timestamp], reentry_rule):
                defensive = False
                reentry = True

            if not defensive and max_gain >= activation_gain:
                if trigger_kind == "extension":
                    distance = float(features.at[timestamp, "distance_to_long_ma"])
                    trigger = np.isfinite(distance) and distance >= trigger_threshold
                elif trigger_kind == "peak_drawdown":
                    trigger = np.isfinite(peak_dd) and peak_dd <= -abs(trigger_threshold)
                else:
                    raise ValueError(f"unknown trigger_kind: {trigger_kind}")
                if trigger:
                    defensive = True

            cap = trim_weight if defensive else 1.0

        caps.append(cap)
        triggers.append(trigger)
        reentries.append(reentry)
        gains.append(gain)
        peak_drawdowns.append(peak_dd)

    cap_series = pd.Series(caps, index=base.index, name="overlay_cap", dtype=float)
    diagnostics = pd.DataFrame(
        {
            "overlay_cap": cap_series,
            "overlay_trigger": triggers,
            "overlay_reentry": reentries,
            "trade_gain": gains,
            "trade_peak_drawdown": peak_drawdowns,
        },
        index=base.index,
    )
    return OverlayDiagnostics(weights=cap_series, diagnostics=diagnostics)


def _reentry_condition(row: pd.Series, reentry_rule: str) -> bool:
    if reentry_rule == "ma20":
        return bool(row.get("above_long_ma", False) and row.get("touch_short_ma", False))
    if reentry_rule == "ma50":
        return bool(row.get("above_long_ma", False) and row.get("touch_medium_ma", False))
    if reentry_rule == "z20_le_0":
        return bool(row.get("above_long_ma", False) and row.get("zscore_le_0", False))
    raise ValueError("reentry_rule must be one of: ma20, ma50, z20_le_0")


def apply_cap(raw_weight: pd.Series, cap: pd.Series) -> pd.Series:
    """Apply a long-only cap to a raw weight series."""
    aligned_cap = cap.reindex(raw_weight.index).fillna(1.0).clip(0.0, 1.0)
    return (
        pd.concat([raw_weight.astype(float), aligned_cap], axis=1)
        .min(axis=1)
        .rename(raw_weight.name)
    )
