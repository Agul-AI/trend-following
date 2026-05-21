"""Trend-following signal construction.

Raw signal functions use information available at the close of the signal date.
Use ``make_executable_positions`` before backtesting to convert raw close-date
signals into positions aligned to daily returns under the chosen execution lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PandasLike = pd.Series | pd.DataFrame


def _as_float_signal(signal: PandasLike) -> PandasLike:
    return signal.fillna(False).astype(float).clip(lower=0.0, upper=1.0)


def sma_trend_signal(prices: PandasLike, window: int = 200) -> PandasLike:
    """Long when price is above its simple moving average."""
    if window <= 0:
        raise ValueError("window must be positive")
    moving_average = prices.rolling(window=window, min_periods=window).mean()
    return _as_float_signal(prices > moving_average)


def sma_crossover_signal(
    prices: PandasLike,
    short_window: int = 50,
    long_window: int = 200,
) -> PandasLike:
    """Long when short-window SMA is above long-window SMA."""
    if short_window <= 0 or long_window <= 0:
        raise ValueError("SMA windows must be positive")
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")
    short_sma = prices.rolling(window=short_window, min_periods=short_window).mean()
    long_sma = prices.rolling(window=long_window, min_periods=long_window).mean()
    return _as_float_signal(short_sma > long_sma)


def time_series_momentum_signal(prices: PandasLike, lookback: int = 252) -> PandasLike:
    """Long when the past lookback-day adjusted-price return is positive."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    momentum = prices.pct_change(periods=lookback, fill_method=None)
    return _as_float_signal(momentum > 0)


def apply_volatility_targeting(
    raw_signal: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 63,
    max_leverage: float = 1.0,
    annualization: int = 252,
) -> pd.DataFrame:
    """Scale raw long-only signals by trailing realized volatility.

    The trailing volatility estimate at date ``t`` uses returns through date
    ``t``. The scaled raw signal must still be passed through
    ``make_executable_positions`` before backtesting, so the estimate is known
    before the position earns returns.
    """
    if target_vol <= 0:
        raise ValueError("target_vol must be positive")
    if lookback <= 1:
        raise ValueError("volatility lookback must be greater than one")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be positive")

    aligned_returns = returns.reindex_like(raw_signal)
    realized_vol = aligned_returns.rolling(window=lookback, min_periods=lookback).std() * np.sqrt(
        annualization
    )
    scale = (target_vol / realized_vol).clip(lower=0.0, upper=max_leverage)
    return raw_signal * scale.fillna(0.0)


def signals_to_equal_weight_positions(
    signals: pd.Series | pd.DataFrame,
    mode: str = "equal_sleeves",
) -> pd.Series | pd.DataFrame:
    """Convert asset-level long/cash signals into portfolio raw weights.

    ``equal_sleeves`` assigns each asset a fixed capital sleeve of ``1/N`` and
    leaves that sleeve in cash when the asset signal is zero. ``active_equal``
    normalizes across currently active assets so the portfolio is fully invested
    whenever at least one asset is active.
    """
    if isinstance(signals, pd.Series):
        return signals.clip(lower=0.0, upper=1.0)
    if signals.empty:
        return signals.copy()
    clipped = signals.clip(lower=0.0, upper=1.0)
    if mode == "equal_sleeves":
        return clipped / clipped.shape[1]
    if mode == "active_equal":
        active_counts = clipped.gt(0).sum(axis=1).replace(0, np.nan)
        return clipped.div(active_counts, axis=0).fillna(0.0)
    raise ValueError("portfolio mode must be 'equal_sleeves' or 'active_equal'")


def make_executable_positions(
    raw_weights: pd.Series | pd.DataFrame,
    execution_delay_days: int = 1,
    return_convention: str = "close_to_close",
) -> pd.Series | pd.DataFrame:
    """Shift raw close-date weights into positions aligned with return labels.

    For close-to-close returns, return ``r[t]`` is the return from close ``t-1``
    to close ``t``. A raw signal observed at close ``t`` cannot earn ``r[t]``.

    With the default next-close execution assumption, a close-``t`` signal is
    executed at close ``t+1`` and first earns the return ending on ``t+2``. This
    requires a shift of ``execution_delay_days + 1`` return labels.
    """
    if execution_delay_days < 0:
        raise ValueError("execution_delay_days must be non-negative")
    if return_convention != "close_to_close":
        raise NotImplementedError("Only close_to_close return timing is implemented in v1")
    periods = execution_delay_days + 1
    return raw_weights.shift(periods).fillna(0.0)
