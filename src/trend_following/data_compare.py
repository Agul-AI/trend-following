"""Utilities for comparing overlapping vendor price data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trend_following.data_validation import read_price_file
from trend_following.utils import ensure_directory

PRICE_COMPARE_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize_index_timezone(frame: pd.DataFrame, source_timezone: str | None) -> pd.DataFrame:
    """Return a copy whose DatetimeIndex is naive UTC when a source timezone is provided."""
    normalized = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(normalized.index))
    if source_timezone:
        index = (
            index.tz_localize(source_timezone, ambiguous="NaT", nonexistent="shift_forward")
            .tz_convert("UTC")
            .tz_localize(None)
        )
    else:
        index = index.tz_localize(None) if index.tz is not None else index
    normalized.index = index
    normalized = normalized[~normalized.index.isna()]
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    return normalized


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return np.nan
    return float(left.corr(right))


def compare_price_files(
    left_path: str | Path,
    right_path: str | Path,
    ticker: str,
    left_label: str,
    right_label: str,
    left_timezone: str | None = None,
    right_timezone: str | None = None,
) -> dict[str, object]:
    """Compare two raw parquet OHLCV files over their shared timestamps."""
    left_path = Path(left_path)
    right_path = Path(right_path)
    if not left_path.exists() or not right_path.exists():
        return {
            "ticker": ticker,
            "left_label": left_label,
            "right_label": right_label,
            "status": "missing_file",
            "left_rows": int(left_path.exists()),
            "right_rows": int(right_path.exists()),
            "overlap_rows": 0,
            "message": f"missing: {left_path if not left_path.exists() else right_path}",
        }

    left = _normalize_index_timezone(read_price_file(left_path), left_timezone)
    right = _normalize_index_timezone(read_price_file(right_path), right_timezone)

    shared_index = left.index.intersection(right.index).sort_values()
    row: dict[str, object] = {
        "ticker": ticker,
        "left_label": left_label,
        "right_label": right_label,
        "status": "pass" if len(shared_index) else "no_overlap",
        "left_rows": len(left),
        "right_rows": len(right),
        "left_start": left.index.min().isoformat() if len(left) else "",
        "left_end": left.index.max().isoformat() if len(left) else "",
        "right_start": right.index.min().isoformat() if len(right) else "",
        "right_end": right.index.max().isoformat() if len(right) else "",
        "overlap_rows": len(shared_index),
        "overlap_start": shared_index.min().isoformat() if len(shared_index) else "",
        "overlap_end": shared_index.max().isoformat() if len(shared_index) else "",
        "message": "",
    }
    if not len(shared_index):
        return row

    left_overlap = left.loc[shared_index]
    right_overlap = right.loc[shared_index]
    for column in PRICE_COMPARE_COLUMNS:
        if column not in left_overlap.columns or column not in right_overlap.columns:
            continue
        left_values = pd.to_numeric(left_overlap[column], errors="coerce")
        right_values = pd.to_numeric(right_overlap[column], errors="coerce")
        valid = left_values.notna() & right_values.notna()
        if not valid.any():
            continue
        diff = left_values[valid] - right_values[valid]
        abs_diff = diff.abs()
        denominator = right_values[valid].abs().replace(0, np.nan)
        abs_pct_diff = (abs_diff / denominator).replace([np.inf, -np.inf], np.nan)
        row[f"{column}_mean_abs_diff"] = float(abs_diff.mean())
        row[f"{column}_median_abs_diff"] = float(abs_diff.median())
        row[f"{column}_max_abs_diff"] = float(abs_diff.max())
        row[f"{column}_mean_abs_pct_diff"] = float(abs_pct_diff.mean())
        row[f"{column}_corr"] = _safe_corr(left_values[valid], right_values[valid])

    return row


def compare_price_directories(
    left_dir: str | Path,
    right_dir: str | Path,
    tickers: list[str],
    output_path: str | Path,
    left_label: str = "left",
    right_label: str = "right",
    left_timezone: str | None = None,
    right_timezone: str | None = None,
) -> pd.DataFrame:
    """Compare all ticker parquet files in two raw-data directories and save a CSV report."""
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    rows = [
        compare_price_files(
            left_dir / f"{ticker}.parquet",
            right_dir / f"{ticker}.parquet",
            ticker=ticker,
            left_label=left_label,
            right_label=right_label,
            left_timezone=left_timezone,
            right_timezone=right_timezone,
        )
        for ticker in tickers
    ]
    report = pd.DataFrame(rows)
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    report.to_csv(output_path, index=False)
    return report
