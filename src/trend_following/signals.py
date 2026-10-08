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


def donchian_breakout_signal(
    prices: PandasLike,
    entry_lookback: int = 252,
    exit_lookback: int = 126,
) -> PandasLike:
    """Long after an N-day high breakout until an exit-window low is hit.

    This is the canonical breakout trend-following family. The signal is a
    close-date state variable: if today's close exceeds the prior entry-window
    high, the raw signal becomes long at today's close; if it breaches the
    prior exit-window low, the raw signal goes to cash at today's close. The
    backtester later shifts this raw signal into executable positions.
    """
    if entry_lookback <= 0 or exit_lookback <= 0:
        raise ValueError("entry_lookback and exit_lookback must be positive")

    if isinstance(prices, pd.DataFrame):
        return prices.apply(
            lambda column: donchian_breakout_signal(
                column,
                entry_lookback=entry_lookback,
                exit_lookback=exit_lookback,
            )
        )

    rolling_high = prices.rolling(window=entry_lookback, min_periods=entry_lookback).max().shift(1)
    rolling_low = prices.rolling(window=exit_lookback, min_periods=exit_lookback).min().shift(1)
    entries = prices > rolling_high
    exits = prices <= rolling_low

    state = 0.0
    values: list[float] = []
    for entry, exit_ in zip(entries.fillna(False), exits.fillna(False), strict=False):
        if exit_:
            state = 0.0
        if entry:
            state = 1.0
        values.append(state)
    return pd.Series(values, index=prices.index, name=prices.name, dtype=float)


def regression_slope_signal(
    prices: PandasLike,
    window: int = 126,
    min_r_squared: float = 0.0,
) -> PandasLike:
    """Long when a rolling log-price regression slope is positive.

    ``min_r_squared`` can require the fitted trend to explain some fraction of
    recent log-price variation. The default keeps the estimator simple and only
    checks the sign of the slope.
    """
    if window <= 1:
        raise ValueError("window must be greater than one")
    if min_r_squared < 0 or min_r_squared > 1:
        raise ValueError("min_r_squared must be in [0, 1]")

    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    x_denom = float(np.dot(x_centered, x_centered))

    def fit_flag(values: np.ndarray) -> float:
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            return 0.0
        y = np.log(values)
        y_centered = y - y.mean()
        slope = float(np.dot(x_centered, y_centered) / x_denom)
        if slope <= 0:
            return 0.0
        if min_r_squared == 0:
            return 1.0
        fitted = y.mean() + slope * x_centered
        total_ss = float(np.dot(y_centered, y_centered))
        if total_ss == 0:
            return 0.0
        residual = y - fitted
        residual_ss = float(np.dot(residual, residual))
        r_squared = 1.0 - residual_ss / total_ss
        return float(r_squared >= min_r_squared)

    signal = prices.rolling(window=window, min_periods=window).apply(fit_flag, raw=True)
    return signal.fillna(0.0).astype(float)


def kalman_trend_signal(
    prices: PandasLike,
    process_level_var: float = 1e-5,
    process_trend_var: float = 1e-7,
    observation_var: float = 1e-3,
    min_periods: int = 20,
) -> PandasLike:
    """Long when a simple local-linear Kalman filter estimates positive trend.

    The state is ``[log_price_level, log_price_slope]``. This gives a transparent
    state-estimated trend family without introducing an external dependency.
    """
    if process_level_var <= 0 or process_trend_var <= 0 or observation_var <= 0:
        raise ValueError("Kalman variances must be positive")
    if min_periods < 1:
        raise ValueError("min_periods must be positive")

    if isinstance(prices, pd.DataFrame):
        return prices.apply(
            lambda column: kalman_trend_signal(
                column,
                process_level_var=process_level_var,
                process_trend_var=process_trend_var,
                observation_var=observation_var,
                min_periods=min_periods,
            )
        )

    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    observation = np.array([1.0, 0.0])
    process_cov = np.diag([process_level_var, process_trend_var])
    state = np.array([np.nan, 0.0])
    covariance = np.eye(2)
    initialized = False
    observations_seen = 0
    values: list[float] = []

    for price in prices.astype(float):
        if not np.isfinite(price) or price <= 0:
            values.append(0.0)
            continue

        observed = float(np.log(price))
        if not initialized:
            state = np.array([observed, 0.0])
            covariance = np.eye(2)
            initialized = True
            observations_seen = 1
            values.append(0.0)
            continue

        predicted_state = transition @ state
        predicted_covariance = transition @ covariance @ transition.T + process_cov
        innovation = observed - float(observation @ predicted_state)
        innovation_var = float(observation @ predicted_covariance @ observation.T + observation_var)
        kalman_gain = predicted_covariance @ observation.T / innovation_var
        state = predicted_state + kalman_gain * innovation
        covariance = (np.eye(2) - np.outer(kalman_gain, observation)) @ predicted_covariance
        observations_seen += 1
        values.append(float(observations_seen >= min_periods and state[1] > 0))

    return pd.Series(values, index=prices.index, name=prices.name, dtype=float)


def cross_sectional_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 126,
    top_n: int = 3,
    require_positive: bool = True,
) -> pd.DataFrame:
    """Long the strongest assets by trailing return across the universe.

    This is the irreducible relative-trend family. It ranks assets against each
    other on each date, optionally requiring positive absolute momentum too.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("cross_sectional_momentum_signal requires a DataFrame")
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    momentum = prices.pct_change(periods=lookback, fill_method=None)
    ranks = momentum.rank(axis=1, ascending=False, method="first")
    selected = ranks.le(min(top_n, prices.shape[1]))
    if require_positive:
        selected &= momentum.gt(0)
    return selected.fillna(False).astype(float)


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


def limit_trades_per_day(
    desired_weights: pd.Series | pd.DataFrame,
    max_trades_per_day: int | None = 1,
) -> pd.Series | pd.DataFrame:
    """Limit executable position changes to at most N per calendar day.

    This is intended for intraday backtests after raw signals have already been
    shifted into executable positions. It scans forward in timestamp order and
    accepts the first position change of each calendar day. Later same-day
    changes are ignored, so a long/cash strategy cannot buy and sell on the
    same day.

    For multi-column weights, any vector change counts as one portfolio trade.
    """
    if max_trades_per_day is None:
        return desired_weights.copy()
    if max_trades_per_day < 0:
        raise ValueError("max_trades_per_day must be non-negative or None")

    was_series = isinstance(desired_weights, pd.Series)
    if was_series:
        frame = desired_weights.to_frame(name=desired_weights.name or "asset")
    else:
        frame = desired_weights.copy()
    if frame.empty:
        return desired_weights.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("limit_trades_per_day requires a DatetimeIndex")

    frame = frame.sort_index().fillna(0.0)
    current = pd.Series(0.0, index=frame.columns, dtype=float)
    trades_by_day: dict[pd.Timestamp, int] = {}
    limited_rows: list[pd.Series] = []

    for timestamp, desired in frame.iterrows():
        day = pd.Timestamp(timestamp).normalize()
        trades_today = trades_by_day.get(day, 0)
        changed = not np.allclose(
            desired.to_numpy(dtype=float),
            current.to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
        )
        if changed and trades_today < max_trades_per_day:
            current = desired.astype(float).copy()
            trades_by_day[day] = trades_today + 1
        limited_rows.append(current.copy())

    limited = pd.DataFrame(limited_rows, index=frame.index, columns=frame.columns)
    if was_series:
        return limited.iloc[:, 0].rename(desired_weights.name)
    return limited
