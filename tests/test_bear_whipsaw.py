from __future__ import annotations

import pandas as pd

from trend_following.bear_whipsaw import (
    bear_reentry_filter_raw,
    failed_breakout_cooldown_raw,
    portfolio_drawdown_circuit_breaker,
    two_stage_bear_reentry_cap,
    volatility_cap,
)


def test_bear_reentry_filter_delays_entry_until_confirmation() -> None:
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    base = pd.Series([0, 1, 1, 1, 0], index=index, dtype=float)
    features = pd.DataFrame(
        {
            "distance_to_long_ma": [0.0, 0.005, 0.015, 0.025, 0.0],
            "long_ma_slope_5d": [0.0, -1.0, -1.0, -1.0, 0.0],
            "medium_ma_slope_5d": [0.0, -1.0, 1.0, 1.0, 0.0],
            "short_gt_medium": [False, False, False, False, False],
        },
        index=index,
    )

    result = bear_reentry_filter_raw(
        base,
        features,
        distance_buffer=0.02,
        slope_days=5,
    )

    assert result.weights.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert result.diagnostics["blocked_entry"].tolist() == [False, True, True, False, False]


def test_failed_breakout_cooldown_blocks_after_repeated_weak_trades() -> None:
    index = pd.date_range("2020-01-01", periods=9, freq="D")
    base = pd.Series([1, 1, 0, 1, 1, 0, 1, 1, 1], index=index, dtype=float)
    price = pd.Series([100, 98, 98, 100, 99, 99, 101, 102, 103], index=index, dtype=float)
    features = pd.DataFrame(
        {
            "distance_to_long_ma": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.03],
            "medium_ma_slope_10d": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
        },
        index=index,
    )

    result = failed_breakout_cooldown_raw(
        base,
        price,
        features,
        weak_trade_return=0.03,
        weak_trade_count=2,
        lookback_days=90,
        slope_days=10,
        distance_buffer=0.02,
    )

    assert result.diagnostics.loc[index[5], "cooldown_active"]
    assert result.weights.loc[index[6]] == 0.0
    assert result.diagnostics.loc[index[6], "blocked_by_cooldown"]
    assert result.weights.loc[index[7]] == 1.0
    assert result.diagnostics.loc[index[7], "cooldown_release"]


def test_volatility_cap_uses_known_percentile() -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    features = pd.DataFrame(
        {"realized_vol_percentile_known": [0.2, 0.79, 0.80, 0.95]},
        index=index,
    )

    result = volatility_cap(features, percentile_threshold=0.80, defensive_cap=0.50)

    assert result.weights.tolist() == [1.0, 1.0, 0.5, 0.5]


def test_portfolio_drawdown_circuit_breaker_uses_prior_drawdown() -> None:
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    raw = pd.Series(1.0, index=index)
    returns = pd.Series([0.0, -0.30, 0.10, 0.10, 0.10], index=index)
    features = pd.DataFrame({"medium_ma_slope_10d": [-1.0, -1.0, -1.0, 1.0, 1.0]}, index=index)

    result = portfolio_drawdown_circuit_breaker(
        raw,
        returns,
        features,
        trigger_drawdown=0.25,
        recover_drawdown=0.10,
        defensive_cap=0.25,
        slope_days=10,
    )

    assert not result.diagnostics.loc[index[1], "circuit_breaker_trigger"]
    assert result.diagnostics.loc[index[2], "circuit_breaker_trigger"]
    assert result.weights.loc[index[2]] == 0.25


def test_two_stage_bear_reentry_releases_after_confirmation() -> None:
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    raw = pd.Series([0, 1, 1, 1, 0], index=index, dtype=float)
    features = pd.DataFrame(
        {
            "long_ma_slope_10d": [0.0, -1.0, -1.0, -1.0, 0.0],
            "medium_ma_slope_10d": [-1.0, -1.0, -1.0, 1.0, 0.0],
            "short_gt_medium": [False, False, False, False, False],
        },
        index=index,
    )

    result = two_stage_bear_reentry_cap(
        raw,
        features,
        initial_weight=0.50,
        slope_days=10,
        release_rule="medium_slope",
    )

    assert result.weights.tolist() == [0.0, 0.5, 0.5, 1.0, 0.0]
    assert result.diagnostics.loc[index[3], "two_stage_release"]
