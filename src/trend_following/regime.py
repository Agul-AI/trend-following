"""Market-regime features and QQQ trend/mean-reversion switcher."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trend_following.signals import sma_crossover_signal

DEFAULT_REGIME_SWITCH_PARAMS: dict[str, Any] = {
    "target_ticker": "QQQ",
    "regime_ticker": "QQQ",
    "sma_window": 200,
    "sma_slope_window": 20,
    "variance_window": 63,
    "variance_horizon": 5,
    "use_variance_ratio_for_trend": False,
    "trend_variance_ratio_threshold": 1.05,
    "mean_reversion_variance_ratio_threshold": 0.98,
    "volatility_window": 20,
    "volatility_percentile_window": 252,
    "volatility_percentile_threshold": 0.80,
    "zscore_window": 20,
    "entry_zscore": -1.5,
    "exit_zscore": 0.0,
    "trend_short_window": 50,
    "trend_long_window": 200,
    "state_machine_entry_ma_days": 20,
    "state_machine_exit_ma_days": 50,
    "state_machine_entry_slope_days": 5,
    "state_machine_entry_confirm_bars": 1,
    "state_machine_exit_confirm_bars": 2,
    "state_machine_entry_buffer": 0.0,
    "state_machine_exit_buffer": 0.0,
    "state_machine_min_hold_days": 0,
}


def _params(params: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_REGIME_SWITCH_PARAMS)
    if params:
        merged.update(params)
    return merged


def _as_series(data: pd.Series | pd.DataFrame, ticker: str, name: str) -> pd.Series:
    """Extract a ticker Series from a Series/DataFrame input."""
    if isinstance(data, pd.Series):
        return data.astype(float).rename(ticker)
    if ticker not in data.columns:
        raise ValueError(f"{name} must include ticker {ticker!r}")
    return data[ticker].astype(float)


def _rolling_percentile_of_last(values: np.ndarray) -> float:
    """Percentile rank of the last value inside a rolling window."""
    last = values[-1]
    finite = values[np.isfinite(values)]
    if not np.isfinite(last) or finite.size == 0:
        return np.nan
    return float((finite <= last).mean())


def compute_regime_features(
    prices: pd.DataFrame | pd.Series,
    returns: pd.DataFrame | pd.Series,
    regime_ticker: str = "QQQ",
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute close/return-based regime features for one regime ticker.

    All features are timestamped at the close of the feature date. They must be
    shifted into executable positions later before being used in a backtest.
    """
    cfg = _params(params)
    ticker = str(cfg.get("regime_ticker", regime_ticker))
    price = _as_series(prices, ticker, "prices")
    returns_series = _as_series(returns, ticker, "returns").reindex(price.index)

    sma_window = int(cfg["sma_window"])
    sma_slope_window = int(cfg["sma_slope_window"])
    variance_window = int(cfg["variance_window"])
    variance_horizon = int(cfg["variance_horizon"])
    volatility_window = int(cfg["volatility_window"])
    volatility_percentile_window = int(cfg["volatility_percentile_window"])
    zscore_window = int(cfg["zscore_window"])

    if min(
        sma_window,
        sma_slope_window,
        variance_window,
        variance_horizon,
        volatility_window,
        volatility_percentile_window,
        zscore_window,
    ) <= 0:
        raise ValueError("Regime windows/horizons must be positive")

    sma = price.rolling(window=sma_window, min_periods=sma_window).mean()
    sma_slope = sma.pct_change(periods=sma_slope_window, fill_method=None)

    log_price = np.log(price.where(price > 0))
    one_period_log_return = log_price.diff()
    horizon_log_return = log_price.diff(periods=variance_horizon)
    one_period_var = one_period_log_return.rolling(
        window=variance_window, min_periods=variance_window
    ).var(ddof=0)
    horizon_var = horizon_log_return.rolling(
        window=variance_window, min_periods=variance_window
    ).var(ddof=0)
    variance_ratio = horizon_var / (variance_horizon * one_period_var.replace(0.0, np.nan))

    realized_vol = returns_series.rolling(
        window=volatility_window, min_periods=volatility_window
    ).std(ddof=0)
    volatility_percentile = realized_vol.rolling(
        window=volatility_percentile_window,
        min_periods=volatility_percentile_window,
    ).apply(_rolling_percentile_of_last, raw=True)

    rolling_mean = price.rolling(window=zscore_window, min_periods=zscore_window).mean()
    rolling_std = price.rolling(window=zscore_window, min_periods=zscore_window).std(ddof=0)
    zscore = (price - rolling_mean) / rolling_std.replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "price": price,
            "sma": sma,
            "sma_slope": sma_slope,
            "variance_ratio": variance_ratio,
            "realized_volatility": realized_vol,
            "volatility_percentile": volatility_percentile,
            "zscore": zscore,
        },
        index=price.index,
    )


def classify_regimes(
    features: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.Series:
    """Classify each row into trend, mean_reversion, risk_off, or neutral."""
    cfg = _params(params)
    required = ["price", "sma", "sma_slope", "variance_ratio", "volatility_percentile"]
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ValueError(f"Regime features missing required columns: {missing}")

    regimes = pd.Series("neutral", index=features.index, name="regime", dtype="object")
    price = features["price"]
    sma = features["sma"]
    sma_slope = features["sma_slope"]
    variance_ratio = features["variance_ratio"]
    volatility_percentile = features["volatility_percentile"]

    risk_off = price.lt(sma) & sma_slope.lt(0)
    use_variance_ratio_for_trend = bool(cfg.get("use_variance_ratio_for_trend", True))
    if use_variance_ratio_for_trend:
        trend_persistence = variance_ratio.gt(float(cfg["trend_variance_ratio_threshold"]))
    else:
        # Optional simpler trend definition: price above a rising long-term SMA.
        # Variance ratio can still be used for the mean-reversion rule below.
        trend_persistence = pd.Series(True, index=features.index)

    trend = price.gt(sma) & sma_slope.gt(0) & trend_persistence & ~risk_off
    mean_reversion = (
        variance_ratio.lt(float(cfg["mean_reversion_variance_ratio_threshold"]))
        & volatility_percentile.lt(float(cfg["volatility_percentile_threshold"]))
        & ~risk_off
        & ~trend
    )

    regimes.loc[risk_off.fillna(False)] = "risk_off"
    regimes.loc[trend.fillna(False)] = "trend"
    regimes.loc[mean_reversion.fillna(False)] = "mean_reversion"
    return regimes


def lagged_regime_estimate(regimes: pd.Series, lag_days: int = 1) -> pd.Series:
    """Estimate today's regime using only information through prior closes.

    ``classify_regimes`` labels the regime known at the close of each date. To
    ask "what would I have estimated for today's regime before seeing today's
    close?", shift yesterday's close-based regime estimate forward by one
    business row. This is a regime-confirmation diagnostic; it is separate from
    the strategy's executable-position shift.
    """
    if lag_days <= 0:
        raise ValueError("lag_days must be positive")
    return regimes.shift(lag_days).rename("yesterday_estimate_for_today")


def regime_confirmation_table(
    features: pd.DataFrame,
    regimes: pd.Series,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Compare lagged no-lookahead regime estimates with current estimates."""
    if lag_days <= 0:
        raise ValueError("lag_days must be positive")

    current = regimes.reindex(features.index).rename("current_regime")
    lagged = lagged_regime_estimate(current, lag_days=lag_days)
    valid_estimate = current.notna() & lagged.notna()

    table = features.copy()
    table["current_regime"] = current
    table["yesterday_estimate_for_today"] = lagged
    table["regime_match"] = current.eq(lagged).where(valid_estimate).astype("boolean")

    feature_columns = [
        column
        for column in [
            "sma",
            "sma_slope",
            "variance_ratio",
            "volatility_percentile",
            "zscore",
        ]
        if column in table.columns
    ]
    if feature_columns:
        table["feature_ready"] = (
            table[feature_columns].notna().all(axis=1)
            & table[feature_columns].shift(lag_days).notna().all(axis=1)
        )
    else:
        table["feature_ready"] = valid_estimate
    return table


def regime_confirmation_accuracy(
    confirmation: pd.DataFrame,
    feature_ready_only: bool = True,
) -> float:
    """Return the share of valid lagged regime estimates matching current labels."""
    required = {"regime_match", "yesterday_estimate_for_today", "current_regime"}
    missing = required.difference(confirmation.columns)
    if missing:
        raise ValueError(f"Confirmation table missing required columns: {sorted(missing)}")

    valid = (
        confirmation["regime_match"].notna()
        & confirmation["yesterday_estimate_for_today"].notna()
        & confirmation["current_regime"].notna()
    )
    if feature_ready_only and "feature_ready" in confirmation.columns:
        valid &= confirmation["feature_ready"]
    if not valid.any():
        return float("nan")
    return float(confirmation.loc[valid, "regime_match"].mean())


def align_daily_regimes_to_intraday(
    daily_regimes: pd.Series,
    intraday_index: pd.DatetimeIndex,
    lag_days: int = 1,
    fill_method: str | None = "ffill",
) -> pd.Series:
    """Map daily close-based regimes to intraday bars without same-day leakage.

    The daily regime for date ``D`` is only known after date ``D`` closes. For
    intraday bars on date ``D``, the no-lookahead regime estimate is therefore
    the prior daily estimate, normally from date ``D-1``. The default
    ``lag_days=1`` implements that convention before mapping by intraday bar
    date.

    ``fill_method='ffill'`` lets the most recent available lagged daily regime
    carry across missing daily rows while still never using a future close.
    """
    if lag_days <= 0:
        raise ValueError("lag_days must be positive")
    if fill_method not in {None, "ffill"}:
        raise ValueError("fill_method must be None or 'ffill'")
    if not isinstance(intraday_index, pd.DatetimeIndex):
        intraday_index = pd.DatetimeIndex(intraday_index)

    daily = daily_regimes.copy()
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    lagged = lagged_regime_estimate(daily, lag_days=lag_days)

    intraday_dates = intraday_index.tz_localize(None).normalize()
    unique_dates = pd.DatetimeIndex(intraday_dates.unique()).sort_values()
    aligned_by_date = lagged.reindex(unique_dates, method=fill_method)
    aligned_values = aligned_by_date.reindex(intraday_dates).to_numpy()
    return pd.Series(
        aligned_values,
        index=intraday_index,
        name=f"daily_regime_lag_{lag_days}",
        dtype="object",
    )


def _intraday_trend_windows(cfg: dict[str, Any]) -> tuple[int, int]:
    """Return hourly trend windows, optionally converted from trading-day windows."""
    bars_per_day = float(cfg.get("intraday_bars_per_day", 6))
    if bars_per_day <= 0:
        raise ValueError("intraday_bars_per_day must be positive")

    if "intraday_trend_short_days" in cfg:
        short_window = int(round(float(cfg["intraday_trend_short_days"]) * bars_per_day))
    else:
        short_window = int(cfg.get("intraday_trend_short_window", cfg["trend_short_window"]))

    if "intraday_trend_long_days" in cfg:
        long_window = int(round(float(cfg["intraday_trend_long_days"]) * bars_per_day))
    else:
        long_window = int(cfg.get("intraday_trend_long_window", cfg["trend_long_window"]))

    if short_window <= 0 or long_window <= 0:
        raise ValueError("Intraday trend windows must be positive")
    if short_window >= long_window:
        raise ValueError("Intraday short window must be less than long window")
    return short_window, long_window


def _days_to_intraday_bars(days: float, cfg: dict[str, Any], *, minimum: int = 1) -> int:
    """Convert trading-day windows into intraday bar windows."""
    bars_per_day = float(cfg.get("intraday_bars_per_day", 6))
    if bars_per_day <= 0:
        raise ValueError("intraday_bars_per_day must be positive")
    bars = int(round(float(days) * bars_per_day))
    return max(minimum, bars)


def _confirmed(condition: pd.Series, confirm_bars: int) -> pd.Series:
    """Require a boolean condition to hold for N consecutive bars."""
    if confirm_bars <= 0:
        raise ValueError("confirmation bars must be positive")
    clean = condition.fillna(False).astype(bool)
    if confirm_bars == 1:
        return clean
    return clean.rolling(window=confirm_bars, min_periods=confirm_bars).sum().ge(confirm_bars)


def hourly_fast_entry_slow_exit_state_machine(
    price: pd.Series,
    allowed_regime: pd.Series,
    params: dict[str, Any] | None = None,
) -> pd.Series:
    """Return a raw long/cash signal from a fast-entry/slow-exit state machine.

    The state machine is intentionally simple and interview-readable:

    * Entry is fast: when the daily gate allows trading, price is above a
      shorter hourly moving average, and that moving average is rising.
    * Exit is slower: remain invested through shallow pullbacks and exit only
      when the daily gate turns off or price stays below a longer hourly moving
      average for the configured confirmation bars.

    The returned signal is timestamped at the close of each intraday bar. It is
    still raw information and must be shifted with ``make_executable_positions``
    before backtesting.
    """
    cfg = _params(params)
    clean_price = price.astype(float).sort_index()
    gate = allowed_regime.reindex(clean_price.index).fillna(False).astype(bool)

    entry_window = _days_to_intraday_bars(cfg["state_machine_entry_ma_days"], cfg)
    exit_window = _days_to_intraday_bars(cfg["state_machine_exit_ma_days"], cfg)
    slope_window = _days_to_intraday_bars(
        cfg["state_machine_entry_slope_days"], cfg, minimum=1
    )
    min_hold_bars = _days_to_intraday_bars(
        cfg.get("state_machine_min_hold_days", 0), cfg, minimum=0
    )

    entry_confirm_bars = int(cfg["state_machine_entry_confirm_bars"])
    exit_confirm_bars = int(cfg["state_machine_exit_confirm_bars"])
    entry_buffer = float(cfg["state_machine_entry_buffer"])
    exit_buffer = float(cfg["state_machine_exit_buffer"])

    if entry_window <= 0 or exit_window <= 0:
        raise ValueError("State-machine moving-average windows must be positive")
    if entry_buffer < 0 or exit_buffer < 0:
        raise ValueError("State-machine buffers must be non-negative")

    entry_ma = clean_price.rolling(window=entry_window, min_periods=entry_window).mean()
    exit_ma = clean_price.rolling(window=exit_window, min_periods=exit_window).mean()
    entry_ma_slope = entry_ma.pct_change(periods=slope_window, fill_method=None)

    raw_entry = (
        gate
        & clean_price.gt(entry_ma * (1.0 + entry_buffer))
        & entry_ma_slope.gt(0.0)
    )
    entry = _confirmed(raw_entry, entry_confirm_bars)

    raw_price_exit = clean_price.lt(exit_ma * (1.0 - exit_buffer))
    price_exit = _confirmed(raw_price_exit, exit_confirm_bars)
    regime_exit = ~gate

    state = 0.0
    bars_held = 0
    values: list[float] = []
    for entry_now, regime_exit_now, price_exit_now in zip(
        entry, regime_exit, price_exit, strict=False
    ):
        if state == 0.0:
            if bool(entry_now):
                state = 1.0
                bars_held = 0
        else:
            bars_held += 1
            forced_exit = bool(regime_exit_now)
            slow_exit = bool(price_exit_now) and bars_held >= min_hold_bars
            if forced_exit or slow_exit:
                state = 0.0
                bars_held = 0
        values.append(state)

    return pd.Series(values, index=clean_price.index, name=price.name, dtype=float)


def daily_regime_hourly_fast_slow_signal(
    intraday_prices: pd.DataFrame,
    daily_prices: pd.DataFrame | pd.Series,
    daily_returns: pd.DataFrame | pd.Series,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Use a daily regime gate and hourly fast-entry/slow-exit state machine.

    The daily regime is computed from daily closes and shifted before it is
    aligned to intraday bars, so bars on date ``D`` only see a regime estimate
    based on closes through date ``D-1``. The hourly state-machine output is
    also a raw close-bar signal and must be shifted before returns are earned.
    """
    cfg = _params(params)
    target_ticker = str(cfg["target_ticker"])
    regime_ticker = str(cfg["regime_ticker"])
    target_price = _as_series(intraday_prices, target_ticker, "intraday_prices")

    features = compute_regime_features(
        daily_prices,
        daily_returns,
        regime_ticker=regime_ticker,
        params=cfg,
    )
    daily_regimes = classify_regimes(features, params=cfg)
    fill_method = cfg.get("daily_regime_fill_method", "ffill")
    if fill_method is not None:
        fill_method = str(fill_method)
    intraday_regimes = align_daily_regimes_to_intraday(
        daily_regimes,
        target_price.index,
        lag_days=int(cfg.get("daily_regime_lag_days", 1)),
        fill_method=fill_method,
    ).fillna("neutral")

    raw_signal = hourly_fast_entry_slow_exit_state_machine(
        target_price,
        allowed_regime=intraday_regimes.eq("trend"),
        params=cfg,
    )
    return raw_signal.fillna(0.0).clip(lower=0.0, upper=1.0).to_frame(target_ticker)


def daily_regime_hourly_trend_signal(
    intraday_prices: pd.DataFrame,
    daily_prices: pd.DataFrame | pd.Series,
    daily_returns: pd.DataFrame | pd.Series,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Use hourly trend-following signals gated by yesterday's daily regime.

    V1 is intentionally conservative: the intraday trend leg trades the target
    ticker long/cash only when the daily regime estimate available before the
    session is ``trend``. ``mean_reversion``, ``neutral``, and ``risk_off`` map
    to cash. The returned signal is still a raw close-bar signal and must be
    shifted with ``make_executable_positions`` before backtesting.
    """
    cfg = _params(params)
    target_ticker = str(cfg["target_ticker"])
    regime_ticker = str(cfg["regime_ticker"])
    target_price = _as_series(intraday_prices, target_ticker, "intraday_prices")

    features = compute_regime_features(
        daily_prices,
        daily_returns,
        regime_ticker=regime_ticker,
        params=cfg,
    )
    daily_regimes = classify_regimes(features, params=cfg)
    fill_method = cfg.get("daily_regime_fill_method", "ffill")
    if fill_method is not None:
        fill_method = str(fill_method)
    intraday_regimes = align_daily_regimes_to_intraday(
        daily_regimes,
        target_price.index,
        lag_days=int(cfg.get("daily_regime_lag_days", 1)),
        fill_method=fill_method,
    ).fillna("neutral")

    short_window, long_window = _intraday_trend_windows(cfg)
    trend_signal = sma_crossover_signal(
        target_price,
        short_window=short_window,
        long_window=long_window,
    ).reindex(target_price.index)

    raw_signal = trend_signal.where(intraday_regimes.eq("trend"), 0.0)
    return raw_signal.fillna(0.0).clip(lower=0.0, upper=1.0).to_frame(target_ticker)


def mean_reversion_pullback_signal(
    price: pd.Series,
    regime: pd.Series,
    params: dict[str, Any] | None = None,
) -> pd.Series:
    """Long during mean-reversion pullbacks until price z-score recovers."""
    cfg = _params(params)
    zscore_window = int(cfg["zscore_window"])
    if zscore_window <= 0:
        raise ValueError("zscore_window must be positive")

    aligned_regime = regime.reindex(price.index).fillna("neutral")
    rolling_mean = price.rolling(window=zscore_window, min_periods=zscore_window).mean()
    rolling_std = price.rolling(window=zscore_window, min_periods=zscore_window).std(ddof=0)
    zscore = (price - rolling_mean) / rolling_std.replace(0.0, np.nan)

    entry_zscore = float(cfg["entry_zscore"])
    exit_zscore = float(cfg["exit_zscore"])
    state = 0.0
    values: list[float] = []
    for current_regime, current_zscore in zip(aligned_regime, zscore, strict=False):
        if current_regime != "mean_reversion" or not np.isfinite(current_zscore):
            state = 0.0
        elif state == 0.0 and current_zscore <= entry_zscore:
            state = 1.0
        elif state == 1.0 and current_zscore >= exit_zscore:
            state = 0.0
        values.append(state)

    return pd.Series(values, index=price.index, name=price.name, dtype=float)


def regime_switch_signal(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build raw close-date QQQ trend/mean-reversion switch signals."""
    cfg = _params(params)
    target_ticker = str(cfg["target_ticker"])
    regime_ticker = str(cfg["regime_ticker"])
    target_price = _as_series(prices, target_ticker, "prices")

    features = compute_regime_features(
        prices,
        returns,
        regime_ticker=regime_ticker,
        params=cfg,
    )
    regimes = classify_regimes(features, params=cfg)

    trend_signal = sma_crossover_signal(
        target_price,
        short_window=int(cfg["trend_short_window"]),
        long_window=int(cfg["trend_long_window"]),
    ).reindex(target_price.index)
    pullback_signal = mean_reversion_pullback_signal(target_price, regimes, params=cfg)

    raw_signal = pd.Series(0.0, index=target_price.index, name=target_ticker)
    raw_signal.loc[regimes.eq("trend")] = trend_signal.loc[regimes.eq("trend")]
    raw_signal.loc[regimes.eq("mean_reversion")] = pullback_signal.loc[
        regimes.eq("mean_reversion")
    ]
    return raw_signal.fillna(0.0).clip(lower=0.0, upper=1.0).to_frame(target_ticker)
