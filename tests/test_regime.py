from __future__ import annotations

import numpy as np
import pandas as pd

from trend_following.regime import (
    align_daily_regimes_to_intraday,
    classify_regimes,
    compute_regime_features,
    daily_regime_hourly_fast_slow_signal,
    daily_regime_hourly_trend_signal,
    hourly_fast_entry_slow_exit_state_machine,
    lagged_regime_estimate,
    mean_reversion_pullback_signal,
    regime_confirmation_accuracy,
    regime_confirmation_table,
    regime_switch_signal,
)
from trend_following.signals import make_executable_positions


def _price_return_panels(log_returns: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2020-01-01", periods=len(log_returns))
    price = pd.Series(100 * np.exp(np.cumsum(log_returns)), index=index, name="QQQ")
    returns = price.pct_change(fill_method=None).fillna(0.0)
    return pd.DataFrame({"QQQ": price}), pd.DataFrame({"QQQ": returns})


def _small_regime_params() -> dict[str, float | int | str]:
    return {
        "target_ticker": "QQQ",
        "regime_ticker": "QQQ",
        "sma_window": 50,
        "sma_slope_window": 5,
        "variance_window": 40,
        "variance_horizon": 5,
        "use_variance_ratio_for_trend": True,
        "trend_variance_ratio_threshold": 1.05,
        "mean_reversion_variance_ratio_threshold": 0.98,
        "volatility_window": 10,
        "volatility_percentile_window": 40,
        "volatility_percentile_threshold": 1.01,
        "zscore_window": 10,
        "entry_zscore": -1.5,
        "exit_zscore": 0.0,
        "trend_short_window": 5,
        "trend_long_window": 20,
    }


def test_trending_series_classifies_mostly_trend() -> None:
    log_returns = 0.001 + 0.002 * np.sin(np.arange(420) / 12)
    prices, returns = _price_return_panels(log_returns)
    params = _small_regime_params()

    features = compute_regime_features(prices, returns, params=params)
    regimes = classify_regimes(features, params=params)

    assert regimes.iloc[-100:].eq("trend").mean() > 0.70


def test_choppy_series_classifies_mostly_mean_reversion() -> None:
    log_returns = 0.0015 + 0.003 * ((-1) ** np.arange(420))
    prices, returns = _price_return_panels(log_returns)
    params = _small_regime_params()

    features = compute_regime_features(prices, returns, params=params)
    regimes = classify_regimes(features, params=params)

    assert regimes.iloc[-100:].eq("mean_reversion").mean() > 0.70


def test_falling_series_classifies_risk_off() -> None:
    log_returns = np.r_[np.full(150, 0.002), np.full(150, -0.004)]
    prices, returns = _price_return_panels(log_returns)
    params = _small_regime_params()

    features = compute_regime_features(prices, returns, params=params)
    regimes = classify_regimes(features, params=params)

    assert regimes.iloc[-50:].eq("risk_off").mean() > 0.90


def test_can_classify_trend_without_variance_ratio_filter() -> None:
    log_returns = 0.001 + 0.003 * np.sin(np.arange(420) / 3)
    prices, returns = _price_return_panels(log_returns)
    params = {
        **_small_regime_params(),
        "use_variance_ratio_for_trend": False,
        "trend_variance_ratio_threshold": 999.0,
    }

    features = compute_regime_features(prices, returns, params=params)
    regimes = classify_regimes(features, params=params)

    recent = features.iloc[-100:]
    eligible = recent["price"].gt(recent["sma"]) & recent["sma_slope"].gt(0)
    assert regimes.iloc[-100:].loc[eligible].eq("trend").all()


def test_mean_reversion_pullback_enters_and_exits() -> None:
    index = pd.bdate_range("2020-01-01", periods=8)
    price = pd.Series([100, 101, 102, 98, 97, 99, 101, 102], index=index, name="QQQ")
    regime = pd.Series("mean_reversion", index=index)

    signal = mean_reversion_pullback_signal(
        price,
        regime,
        params={"zscore_window": 3, "entry_zscore": -1.0, "exit_zscore": 0.0},
    )

    expected = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=index, name="QQQ")
    pd.testing.assert_series_equal(signal, expected)


def test_regime_switch_signal_is_shifted_before_backtest() -> None:
    index = pd.bdate_range("2020-01-01", periods=120)
    log_returns = 0.002 + 0.003 * np.sin(np.arange(120) / 6)
    price = pd.Series(100 * np.exp(np.cumsum(log_returns)), index=index, name="QQQ")
    returns = price.pct_change(fill_method=None).fillna(0.0)
    prices = pd.DataFrame({"QQQ": price})
    returns_df = pd.DataFrame({"QQQ": returns})
    params = {
        **_small_regime_params(),
        "sma_window": 20,
        "sma_slope_window": 3,
        "variance_window": 20,
        "variance_horizon": 3,
        "trend_variance_ratio_threshold": 0.0,
        "mean_reversion_variance_ratio_threshold": 0.0,
        "volatility_window": 5,
        "volatility_percentile_window": 20,
        "zscore_window": 5,
        "trend_short_window": 3,
        "trend_long_window": 8,
    }

    raw_signal = regime_switch_signal(prices, returns_df, params=params)
    first_signal_index = raw_signal.index[raw_signal["QQQ"].gt(0)][0]
    first_position = raw_signal.index.get_loc(first_signal_index)
    positions = make_executable_positions(raw_signal, execution_delay_days=1)

    assert positions.iloc[first_position]["QQQ"] == 0.0
    assert positions.iloc[first_position + 1]["QQQ"] == 0.0
    assert positions.iloc[first_position + 2]["QQQ"] == raw_signal.iloc[first_position]["QQQ"]


def test_lagged_regime_confirmation_uses_yesterday_estimate() -> None:
    index = pd.bdate_range("2020-01-01", periods=4)
    features = pd.DataFrame(
        {
            "sma": [1.0, 1.0, 1.0, 1.0],
            "sma_slope": [0.1, 0.1, 0.1, 0.1],
            "variance_ratio": [1.1, 1.1, 1.1, 1.1],
            "volatility_percentile": [0.5, 0.5, 0.5, 0.5],
            "zscore": [0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    regimes = pd.Series(["trend", "trend", "neutral", "neutral"], index=index, name="regime")

    estimate = lagged_regime_estimate(regimes)
    confirmation = regime_confirmation_table(features, regimes)

    assert pd.isna(estimate.iloc[0])
    assert estimate.iloc[1] == "trend"
    assert pd.isna(confirmation["regime_match"].iloc[0])
    assert confirmation["regime_match"].iloc[1:].tolist() == [True, False, True]
    assert regime_confirmation_accuracy(confirmation) == 2 / 3


def test_align_daily_regimes_to_intraday_uses_previous_daily_close() -> None:
    daily_index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    regimes = pd.Series(["risk_off", "trend", "neutral"], index=daily_index)
    intraday_index = pd.to_datetime(
        [
            "2020-01-02 10:00",
            "2020-01-02 11:00",
            "2020-01-03 10:00",
            "2020-01-03 11:00",
            "2020-01-06 10:00",
            "2020-01-06 11:00",
        ]
    )

    aligned = align_daily_regimes_to_intraday(regimes, pd.DatetimeIndex(intraday_index))

    assert aligned.iloc[:2].isna().all()
    assert aligned.iloc[2:4].tolist() == ["risk_off", "risk_off"]
    assert aligned.iloc[4:6].tolist() == ["trend", "trend"]


def test_daily_regime_hourly_trend_signal_gates_with_lagged_daily_regime() -> None:
    daily_index = pd.bdate_range("2020-01-01", periods=5)
    daily_price = pd.Series([100, 101, 102, 103, 104], index=daily_index, name="QQQ")
    daily_prices = pd.DataFrame({"QQQ": daily_price})
    daily_returns = daily_prices.pct_change(fill_method=None).fillna(0.0)

    intraday_index = pd.DatetimeIndex(
        [
            timestamp
            for day in daily_index
            for timestamp in pd.date_range(day + pd.Timedelta(hours=10), periods=3, freq="h")
        ]
    )
    intraday_prices = pd.DataFrame(
        {"QQQ": np.linspace(100, 120, len(intraday_index))},
        index=intraday_index,
    )
    params = {
        **_small_regime_params(),
        "sma_window": 2,
        "sma_slope_window": 1,
        "variance_window": 2,
        "variance_horizon": 1,
        "trend_variance_ratio_threshold": 0.0,
        "volatility_window": 2,
        "volatility_percentile_window": 2,
        "intraday_trend_short_window": 1,
        "intraday_trend_long_window": 2,
    }

    signal = daily_regime_hourly_trend_signal(
        intraday_prices,
        daily_prices,
        daily_returns,
        params=params,
    )

    # The daily close on day 3 is trend, but day-3 intraday bars can only use
    # day-2 information, so they must remain cash.
    day3_mask = signal.index.normalize() == daily_index[2]
    day4_mask = signal.index.normalize() == daily_index[3]
    assert signal.loc[day3_mask, "QQQ"].eq(0.0).all()
    assert signal.loc[day4_mask, "QQQ"].eq(1.0).any()


def test_daily_regime_hourly_trend_signal_converts_days_to_hourly_bars() -> None:
    daily_index = pd.bdate_range("2020-01-01", periods=8)
    daily_price = pd.Series(np.arange(100, 108), index=daily_index, name="QQQ")
    daily_prices = pd.DataFrame({"QQQ": daily_price})
    daily_returns = daily_prices.pct_change(fill_method=None).fillna(0.0)

    intraday_index = pd.DatetimeIndex(
        [
            timestamp
            for day in daily_index
            for timestamp in pd.date_range(day + pd.Timedelta(hours=10), periods=2, freq="h")
        ]
    )
    intraday_prices = pd.DataFrame(
        {"QQQ": np.linspace(100, 130, len(intraday_index))},
        index=intraday_index,
    )
    params = {
        **_small_regime_params(),
        "sma_window": 2,
        "sma_slope_window": 1,
        "variance_window": 2,
        "variance_horizon": 1,
        "trend_variance_ratio_threshold": 0.0,
        "volatility_window": 2,
        "volatility_percentile_window": 2,
        "intraday_bars_per_day": 2,
        "intraday_trend_short_days": 1,
        "intraday_trend_long_days": 3,
    }

    signal = daily_regime_hourly_trend_signal(
        intraday_prices,
        daily_prices,
        daily_returns,
        params=params,
    )

    # Long window is 3 days * 2 bars/day = 6 hourly bars, so the strategy
    # cannot emit a trend signal before six hourly observations are available.
    assert signal.iloc[:5]["QQQ"].eq(0.0).all()
    assert signal["QQQ"].sum() > 0


def test_hourly_fast_entry_slow_exit_state_machine_uses_hysteresis() -> None:
    index = pd.date_range("2020-01-01 10:00", periods=10, freq="h")
    price = pd.Series([10, 11, 12, 13, 12, 11, 10, 9, 10, 11], index=index, name="QQQ")
    allowed = pd.Series(True, index=index)
    params = {
        "target_ticker": "QQQ",
        "regime_ticker": "QQQ",
        "intraday_bars_per_day": 1,
        "state_machine_entry_ma_days": 2,
        "state_machine_exit_ma_days": 4,
        "state_machine_entry_slope_days": 1,
        "state_machine_entry_confirm_bars": 1,
        "state_machine_exit_confirm_bars": 2,
    }

    signal = hourly_fast_entry_slow_exit_state_machine(price, allowed, params=params)

    assert signal.iloc[2] == 1.0
    assert signal.iloc[5] == 1.0
    assert signal.iloc[6] == 0.0


def test_daily_regime_hourly_fast_slow_signal_gates_with_lagged_daily_regime() -> None:
    daily_index = pd.bdate_range("2020-01-01", periods=5)
    daily_price = pd.Series([100, 101, 102, 103, 104], index=daily_index, name="QQQ")
    daily_prices = pd.DataFrame({"QQQ": daily_price})
    daily_returns = daily_prices.pct_change(fill_method=None).fillna(0.0)

    intraday_index = pd.DatetimeIndex(
        [
            timestamp
            for day in daily_index
            for timestamp in pd.date_range(day + pd.Timedelta(hours=10), periods=3, freq="h")
        ]
    )
    intraday_prices = pd.DataFrame(
        {"QQQ": np.linspace(100, 120, len(intraday_index))},
        index=intraday_index,
    )
    params = {
        **_small_regime_params(),
        "sma_window": 2,
        "sma_slope_window": 1,
        "variance_window": 2,
        "variance_horizon": 1,
        "use_variance_ratio_for_trend": False,
        "volatility_window": 2,
        "volatility_percentile_window": 2,
        "intraday_bars_per_day": 3,
        "state_machine_entry_ma_days": 1,
        "state_machine_exit_ma_days": 2,
        "state_machine_entry_slope_days": 1,
        "state_machine_entry_confirm_bars": 1,
        "state_machine_exit_confirm_bars": 1,
    }

    signal = daily_regime_hourly_fast_slow_signal(
        intraday_prices,
        daily_prices,
        daily_returns,
        params=params,
    )

    day3_mask = signal.index.normalize() == daily_index[2]
    day4_mask = signal.index.normalize() == daily_index[3]
    assert signal.loc[day3_mask, "QQQ"].eq(0.0).all()
    assert signal.loc[day4_mask, "QQQ"].eq(1.0).any()
