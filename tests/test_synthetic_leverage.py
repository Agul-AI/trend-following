from __future__ import annotations

import pandas as pd
import pytest

from trend_following.synthetic_leverage import (
    synthetic_daily_leveraged_ohlcv,
    synthetic_intraday_leveraged_ohlcv,
)


def _daily_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    return pd.DataFrame(
        {
            "open": [100.0, 110.0, 104.5],
            "high": [101.0, 112.0, 105.0],
            "low": [99.0, 108.0, 103.0],
            "close": [100.0, 110.0, 104.5],
            "adj_close": [100.0, 110.0, 104.5],
            "volume": [1_000, 1_100, 1_200],
        },
        index=index,
    )


def test_synthetic_daily_leveraged_close_uses_daily_reset() -> None:
    synthetic = synthetic_daily_leveraged_ohlcv(_daily_frame(), leverage=3.0, initial_price=100.0)

    assert synthetic["adj_close"].iloc[0] == 100.0
    assert synthetic["adj_close"].iloc[1] == pytest.approx(130.0)
    assert synthetic["adj_close"].iloc[2] == pytest.approx(110.5)


def test_synthetic_intraday_leveraged_uses_previous_daily_close_anchor() -> None:
    daily = _daily_frame()
    daily_synthetic = synthetic_daily_leveraged_ohlcv(daily, leverage=3.0, initial_price=100.0)
    intraday_index = pd.to_datetime(["2020-01-03 10:00", "2020-01-03 11:00"])
    intraday = pd.DataFrame(
        {
            "open": [100.0, 105.0],
            "high": [106.0, 111.0],
            "low": [99.0, 104.0],
            "close": [105.0, 110.0],
            "adj_close": [105.0, 110.0],
            "volume": [100, 200],
        },
        index=intraday_index,
    )

    synthetic = synthetic_intraday_leveraged_ohlcv(
        intraday_underlying=intraday,
        daily_underlying=daily,
        daily_synthetic=daily_synthetic,
        leverage=3.0,
    )

    assert synthetic["adj_close"].iloc[0] == pytest.approx(115.0)
    assert synthetic["adj_close"].iloc[1] == pytest.approx(130.0)
