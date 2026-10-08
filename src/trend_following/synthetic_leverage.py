"""Synthetic daily-reset leveraged ETF construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLC_COLUMNS = ["open", "high", "low", "close"]


def adjusted_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Return adjusted OHLC columns using the adjusted-close factor when present."""
    missing = [column for column in OHLC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame missing required OHLC columns: {missing}")

    if "adj_close" not in frame.columns:
        adjusted = frame[OHLC_COLUMNS].astype(float).copy()
        adjusted["adj_close"] = adjusted["close"]
        return adjusted

    adjustment_factor = frame["adj_close"].astype(float) / frame["close"].astype(float)
    adjustment_factor = adjustment_factor.replace([np.inf, -np.inf], np.nan)
    adjusted = pd.DataFrame(index=frame.index)
    for column in OHLC_COLUMNS:
        adjusted[column] = frame[column].astype(float) * adjustment_factor
    adjusted["adj_close"] = frame["adj_close"].astype(float)
    return adjusted


def synthetic_daily_leveraged_ohlcv(
    underlying: pd.DataFrame,
    leverage: float = 3.0,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """Construct a synthetic daily-reset leveraged OHLCV series.

    The synthetic close return is exactly ``leverage`` times the underlying
    adjusted close-to-close return each day, before fees/financing/tracking
    error. Intraday daily OHLC values are mapped relative to the prior
    underlying adjusted close, preserving the daily reset convention.
    """
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    if initial_price <= 0:
        raise ValueError("initial_price must be positive")
    if not isinstance(underlying.index, pd.DatetimeIndex):
        raise TypeError("underlying must have a DatetimeIndex")

    frame = underlying.sort_index().copy()
    adjusted = adjusted_ohlc(frame)
    under_close = adjusted["adj_close"]
    if under_close.isna().any() or under_close.le(0).any():
        raise ValueError("underlying adjusted close must be positive and non-missing")

    synthetic_close = pd.Series(index=frame.index, dtype=float, name="adj_close")
    synthetic_close.iloc[0] = float(initial_price)
    daily_returns = under_close.pct_change(fill_method=None)
    for i in range(1, len(frame)):
        leveraged_return = leverage * float(daily_returns.iloc[i])
        next_close = synthetic_close.iloc[i - 1] * (1.0 + leveraged_return)
        if next_close <= 0:
            raise ValueError(
                f"Synthetic close became non-positive at {frame.index[i]}: {next_close}"
            )
        synthetic_close.iloc[i] = next_close

    output = pd.DataFrame(index=frame.index)
    output.iloc[0:0]
    output["close"] = synthetic_close
    output["adj_close"] = synthetic_close

    prior_under_close = under_close.shift(1)
    prior_synthetic_close = synthetic_close.shift(1)
    for column in ["open", "high", "low"]:
        mapped = prior_synthetic_close * (
            1.0 + leverage * (adjusted[column] / prior_under_close - 1.0)
        )
        mapped.iloc[0] = initial_price * adjusted[column].iloc[0] / adjusted["adj_close"].iloc[0]
        output[column] = mapped

    # Reorder and repair any tiny ordering issues caused by vendor rounded OHLC.
    output = output[["open", "high", "low", "close", "adj_close"]]
    output["high"] = output[["open", "high", "low", "close"]].max(axis=1)
    output["low"] = output[["open", "high", "low", "close"]].min(axis=1)
    if output[["open", "high", "low", "close", "adj_close"]].le(0).any().any():
        first_bad = output[output[["open", "high", "low", "close", "adj_close"]].le(0).any(axis=1)]
        raise ValueError(f"Synthetic daily OHLC became non-positive at {first_bad.index[0]}")

    output["volume"] = 0.0
    output.index.name = "date"
    return output


def synthetic_intraday_leveraged_ohlcv(
    intraday_underlying: pd.DataFrame,
    daily_underlying: pd.DataFrame,
    daily_synthetic: pd.DataFrame,
    leverage: float = 3.0,
) -> pd.DataFrame:
    """Construct synthetic intraday OHLCV from adjusted underlying bars.

    For each intraday bar on date ``D``, price is mapped relative to the
    previous daily close:

    ``synthetic_bar = synthetic_close[D-1] * (1 + L * (underlying_bar / underlying_close[D-1] - 1))``.

    This keeps the synthetic product's daily reset rule while exposing an
    intraday path derived from the underlying QQQ bars.
    """
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    for name, frame in {
        "intraday_underlying": intraday_underlying,
        "daily_underlying": daily_underlying,
        "daily_synthetic": daily_synthetic,
    }.items():
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{name} must have a DatetimeIndex")

    intraday = intraday_underlying.sort_index().copy()
    daily = daily_underlying.sort_index().copy()
    daily_synth = daily_synthetic.sort_index().copy()
    intraday_adjusted = adjusted_ohlc(intraday)
    daily_adjusted = adjusted_ohlc(daily)

    daily_dates = daily_adjusted.index.normalize()
    prior_under_close_by_date = pd.Series(
        daily_adjusted["adj_close"].shift(1).to_numpy(),
        index=daily_dates,
    )
    prior_synthetic_close_by_date = pd.Series(
        daily_synth["adj_close"].shift(1).to_numpy(),
        index=daily_synth.index.normalize(),
    )

    intraday_dates = intraday_adjusted.index.normalize()
    prior_under_close = prior_under_close_by_date.reindex(intraday_dates).to_numpy(dtype=float)
    prior_synthetic_close = prior_synthetic_close_by_date.reindex(intraday_dates).to_numpy(
        dtype=float
    )
    valid = np.isfinite(prior_under_close) & np.isfinite(prior_synthetic_close)

    output = pd.DataFrame(index=intraday_adjusted.index)
    for column in ["open", "high", "low", "close", "adj_close"]:
        mapped = prior_synthetic_close * (
            1.0 + leverage * (intraday_adjusted[column].to_numpy(dtype=float) / prior_under_close - 1.0)
        )
        output[column] = mapped

    output = output.loc[valid].copy()
    output["high"] = output[["open", "high", "low", "close"]].max(axis=1)
    output["low"] = output[["open", "high", "low", "close"]].min(axis=1)
    if output[["open", "high", "low", "close", "adj_close"]].le(0).any().any():
        first_bad = output[output[["open", "high", "low", "close", "adj_close"]].le(0).any(axis=1)]
        raise ValueError(f"Synthetic intraday OHLC became non-positive at {first_bad.index[0]}")

    output["volume"] = 0.0
    output.index.name = "date"
    return output[["open", "high", "low", "close", "adj_close", "volume"]]
