from __future__ import annotations

import pandas as pd

from trend_following.signals import (
    cross_sectional_momentum_signal,
    donchian_breakout_signal,
    kalman_trend_signal,
    limit_trades_per_day,
    make_executable_positions,
    regression_slope_signal,
    sma_crossover_signal,
    sma_trend_signal,
    time_series_momentum_signal,
)


def test_sma_trend_signal_small_series() -> None:
    prices = pd.Series([1.0, 2.0, 3.0, 2.0, 4.0], index=pd.bdate_range("2020-01-01", periods=5))
    signal = sma_trend_signal(prices, window=3)
    expected = pd.Series([0.0, 0.0, 1.0, 0.0, 1.0], index=prices.index)
    pd.testing.assert_series_equal(signal, expected)


def test_sma_crossover_signal_small_series() -> None:
    prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.bdate_range("2020-01-01", periods=5))
    signal = sma_crossover_signal(prices, short_window=2, long_window=3)
    expected = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0], index=prices.index)
    pd.testing.assert_series_equal(signal, expected)


def test_time_series_momentum_signal_small_series() -> None:
    prices = pd.Series([1.0, 2.0, 3.0, 2.0, 4.0], index=pd.bdate_range("2020-01-01", periods=5))
    signal = time_series_momentum_signal(prices, lookback=2)
    expected = pd.Series([0.0, 0.0, 1.0, 0.0, 1.0], index=prices.index)
    pd.testing.assert_series_equal(signal, expected)


def test_donchian_breakout_signal_persists_until_exit() -> None:
    prices = pd.Series(
        [10.0, 11.0, 12.0, 11.0, 10.0, 13.0],
        index=pd.bdate_range("2020-01-01", periods=6),
    )
    signal = donchian_breakout_signal(prices, entry_lookback=2, exit_lookback=3)
    expected = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0, 1.0], index=prices.index)
    pd.testing.assert_series_equal(signal, expected)


def test_regression_slope_signal_detects_positive_slope() -> None:
    prices = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.bdate_range("2020-01-01", periods=4))
    signal = regression_slope_signal(prices, window=3)
    expected = pd.Series([0.0, 0.0, 1.0, 1.0], index=prices.index)
    pd.testing.assert_series_equal(signal, expected)


def test_kalman_trend_signal_detects_positive_trend_after_warmup() -> None:
    prices = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4], index=pd.bdate_range("2020-01-01", periods=5))
    signal = kalman_trend_signal(prices, min_periods=3)
    assert signal.iloc[:2].eq(0.0).all()
    assert signal.iloc[-1] == 1.0


def test_cross_sectional_momentum_selects_top_positive_assets() -> None:
    index = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame(
        {
            "A": [1.0, 1.1, 1.2, 1.3],
            "B": [1.0, 1.0, 1.0, 1.0],
            "C": [1.0, 0.9, 0.8, 0.7],
        },
        index=index,
    )
    signal = cross_sectional_momentum_signal(prices, lookback=2, top_n=2, require_positive=True)
    expected = pd.DataFrame(
        {
            "A": [0.0, 0.0, 1.0, 1.0],
            "B": [0.0, 0.0, 0.0, 0.0],
            "C": [0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    pd.testing.assert_frame_equal(signal, expected)


def test_make_executable_positions_next_close_shift() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    raw = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0], index=index)
    positions = make_executable_positions(raw, execution_delay_days=1)
    expected = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0], index=index)
    pd.testing.assert_series_equal(positions, expected)


def test_limit_trades_per_day_prevents_intraday_buy_and_sell() -> None:
    index = pd.to_datetime(
        [
            "2020-01-02 10:00",
            "2020-01-02 11:00",
            "2020-01-02 12:00",
            "2020-01-02 13:00",
            "2020-01-03 10:00",
            "2020-01-03 11:00",
        ]
    )
    desired = pd.Series([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], index=index, name="QQQ")

    limited = limit_trades_per_day(desired, max_trades_per_day=1)

    expected = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 0.0], index=index, name="QQQ")
    pd.testing.assert_series_equal(limited, expected)
