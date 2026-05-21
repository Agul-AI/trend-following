from __future__ import annotations

import pandas as pd

from trend_following.data_validation import validate_price_frame


def _valid_frame() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=3)
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "adj_close": [10.5, 11.5, 12.5],
            "volume": [1000, 1100, 1200],
        },
        index=index,
    )


def test_validation_detects_duplicate_dates() -> None:
    frame = _valid_frame()
    frame = pd.concat([frame, frame.iloc[[1]]]).sort_index()
    report = validate_price_frame(frame, ticker="TEST")
    assert report["status"] == "fail"
    assert report["duplicate_dates"] == 2


def test_validation_detects_missing_adjusted_close() -> None:
    frame = _valid_frame().drop(columns=["adj_close"])
    report = validate_price_frame(frame, ticker="TEST")
    assert report["status"] == "fail"
    assert report["missing_adj_close"] is True


def test_validation_detects_invalid_prices() -> None:
    frame = _valid_frame()
    frame.iloc[1, frame.columns.get_loc("close")] = 0.0
    report = validate_price_frame(frame, ticker="TEST")
    assert report["status"] == "fail"
    assert report["invalid_prices"] == 1
