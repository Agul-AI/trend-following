"""Validation checks for downloaded OHLCV data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trend_following.utils import ensure_directory

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]
REQUIRED_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with snake_case lowercase column names."""
    renamed = {}
    for column in df.columns:
        name = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        renamed[column] = name
    return df.rename(columns=renamed)


def read_price_file(path: str | Path) -> pd.DataFrame:
    """Read a cached parquet price file with a DatetimeIndex named ``date``."""
    frame = pd.read_parquet(path)
    frame = standardize_columns(frame)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frame = frame.set_index("date", drop=True)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.index.name = "date"
    return frame


def validate_price_frame(
    df: pd.DataFrame,
    ticker: str = "UNKNOWN",
    suspicious_gap_days: int = 7,
) -> dict[str, Any]:
    """Validate one ticker's OHLCV frame and return a report row.

    The function reports both fatal issues and warnings. Downstream processing
    treats missing adjusted close, duplicated dates, non-monotonic dates, and
    invalid prices as fatal because they can directly corrupt returns.
    """
    frame = standardize_columns(df.copy())
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        index = pd.DatetimeIndex(frame["date"])
    else:
        index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None)
    frame.index = index
    frame.index.name = "date"

    messages: list[str] = []
    fatal = False

    row_count = int(len(frame))
    duplicate_dates = int(frame.index.duplicated(keep=False).sum())
    is_monotonic = bool(frame.index.is_monotonic_increasing)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    missing_adj_close = "adj_close" not in frame.columns

    if row_count == 0:
        fatal = True
        messages.append("empty file")
    if duplicate_dates:
        fatal = True
        messages.append(f"{duplicate_dates} duplicated date rows")
    if not is_monotonic:
        fatal = True
        messages.append("dates are not monotonic increasing")
    if missing_columns:
        if missing_adj_close:
            fatal = True
        messages.append(f"missing columns: {missing_columns}")

    existing_price_cols = [column for column in PRICE_COLUMNS if column in frame.columns]
    numeric = (
        frame[existing_price_cols].apply(pd.to_numeric, errors="coerce")
        if existing_price_cols
        else pd.DataFrame(index=frame.index)
    )
    missing_values = int(
        frame[[c for c in REQUIRED_COLUMNS if c in frame.columns]].isna().sum().sum()
    )
    invalid_prices = int((numeric <= 0).sum().sum()) if not numeric.empty else 0
    if missing_values:
        messages.append(f"{missing_values} missing values")
    if invalid_prices:
        fatal = True
        messages.append(f"{invalid_prices} zero/negative prices")

    sorted_unique_dates = frame.index.sort_values().unique()
    if len(sorted_unique_dates) > 1:
        gaps = pd.Series(sorted_unique_dates).diff().dt.days.iloc[1:]
        suspicious_gaps = int((gaps > suspicious_gap_days).sum())
        max_gap_days = int(gaps.max())
    else:
        suspicious_gaps = 0
        max_gap_days = 0
    if suspicious_gaps:
        messages.append(f"{suspicious_gaps} suspicious calendar gaps > {suspicious_gap_days} days")

    weekdays_only = bool((frame.index.weekday < 5).all()) if row_count else True
    if not weekdays_only:
        messages.append("contains weekend dates")

    status = "fail" if fatal else ("warn" if messages else "pass")
    return {
        "ticker": ticker,
        "status": status,
        "rows": row_count,
        "start_date": frame.index.min().date().isoformat() if row_count else "",
        "end_date": frame.index.max().date().isoformat() if row_count else "",
        "duplicate_dates": duplicate_dates,
        "is_monotonic": is_monotonic,
        "missing_adj_close": bool(missing_adj_close),
        "missing_values": missing_values,
        "invalid_prices": invalid_prices,
        "suspicious_gaps": suspicious_gaps,
        "max_gap_days": max_gap_days,
        "weekdays_only": weekdays_only,
        "messages": "; ".join(messages),
    }


def validation_has_fatal_issue(report_row: dict[str, Any]) -> bool:
    """Return True if a validation report row should block processing."""
    return str(report_row.get("status", "fail")) == "fail"


def validate_files(
    raw_dir: str | Path,
    tickers: list[str],
    report_path: str | Path,
    suspicious_gap_days: int = 7,
) -> pd.DataFrame:
    """Validate cached raw parquet files and write a CSV report."""
    rows: list[dict[str, Any]] = []
    raw_path = Path(raw_dir)
    for ticker in tickers:
        file_path = raw_path / f"{ticker}.parquet"
        if not file_path.exists():
            rows.append(
                {
                    "ticker": ticker,
                    "status": "fail",
                    "rows": 0,
                    "start_date": "",
                    "end_date": "",
                    "duplicate_dates": np.nan,
                    "is_monotonic": False,
                    "missing_adj_close": True,
                    "missing_values": np.nan,
                    "invalid_prices": np.nan,
                    "suspicious_gaps": np.nan,
                    "max_gap_days": np.nan,
                    "weekdays_only": np.nan,
                    "messages": f"missing raw file: {file_path}",
                }
            )
            continue
        frame = read_price_file(file_path)
        rows.append(
            validate_price_frame(frame, ticker=ticker, suspicious_gap_days=suspicious_gap_days)
        )

    report = pd.DataFrame(rows)
    ensure_directory(Path(report_path).parent)
    report.to_csv(report_path, index=False)
    return report
