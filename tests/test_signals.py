from __future__ import annotations

import pandas as pd

from trend_following.signals import (
    make_executable_positions,
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


def test_make_executable_positions_next_close_shift() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    raw = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0], index=index)
    positions = make_executable_positions(raw, execution_delay_days=1)
    expected = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0], index=index)
    pd.testing.assert_series_equal(positions, expected)
