#!/usr/bin/env python
"""Download daily VIX data, trying Alpha Vantage first and Yahoo Finance second."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.utils import ensure_directory

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
ALPHA_CANDIDATES = [
    ("TIME_SERIES_DAILY_ADJUSTED", "VIX"),
    ("TIME_SERIES_DAILY", "VIX"),
    ("TIME_SERIES_DAILY", "^VIX"),
    ("TIME_SERIES_DAILY", "CBOE:VIX"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1990-01-01", help="Earliest date to keep")
    parser.add_argument("--end", default=None, help="Optional exclusive end date")
    parser.add_argument(
        "--output",
        default="data/raw/market_indicators/VIX.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--report",
        default="reports/tables/vix_download_report.csv",
        help="CSV report path",
    )
    parser.add_argument("--force-yahoo", action="store_true", help="Skip Alpha Vantage")
    return parser.parse_args()


def _standard_output(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Return the repo's standard OHLCV parquet format."""
    output = frame.copy()
    output.index = pd.to_datetime(output.index).tz_localize(None)
    output = output.sort_index()
    columns = {column: str(column).strip().lower().replace(" ", "_") for column in output.columns}
    output = output.rename(columns=columns)
    if "adj_close" not in output.columns:
        output["adj_close"] = output["close"]
    if "volume" not in output.columns:
        output["volume"] = 0.0
    output["source"] = source
    output.index.name = "date"
    return output[["open", "high", "low", "close", "adj_close", "volume", "source"]]


def _parse_alpha_time_series(payload: dict[str, Any]) -> tuple[str | None, pd.DataFrame | None]:
    series_key = next((key for key in payload if key.startswith("Time Series")), None)
    if series_key is None:
        message = (
            payload.get("Information")
            or payload.get("Note")
            or payload.get("Error Message")
            or str(payload)[:300]
        )
        return str(message), None

    rows: list[dict[str, Any]] = []
    for date_text, values in payload[series_key].items():
        row = {"date": pd.Timestamp(date_text)}
        normalized = {
            key.split(". ", 1)[-1].replace(" ", "_").lower(): value
            for key, value in values.items()
        }
        row["open"] = normalized.get("open")
        row["high"] = normalized.get("high")
        row["low"] = normalized.get("low")
        row["close"] = normalized.get("close")
        row["adj_close"] = normalized.get("adjusted_close", normalized.get("close"))
        row["volume"] = normalized.get("volume", 0.0)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return "Alpha Vantage returned an empty time series", None
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "adj_close"])
    if frame.empty:
        return "Alpha Vantage time series had no valid OHLC rows", None
    frame = frame.set_index("date").sort_index()
    return None, _standard_output(frame, source="alpha_vantage")


def try_alpha_vantage(start: str, end: str | None) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return None, {"source": "alpha_vantage", "status": "skipped", "message": "missing API key"}

    attempts: list[str] = []
    for function, symbol in ALPHA_CANDIDATES:
        response = requests.get(
            ALPHA_VANTAGE_URL,
            params={
                "function": function,
                "symbol": symbol,
                "outputsize": "full",
                "apikey": api_key,
            },
            timeout=30,
        )
        try:
            payload = response.json()
        except ValueError:
            attempts.append(f"{function}/{symbol}: non-JSON HTTP {response.status_code}")
            continue
        message, frame = _parse_alpha_time_series(payload)
        if frame is None:
            attempts.append(f"{function}/{symbol}: {message}")
            continue

        frame = _date_filter(frame, start=start, end=end)
        return frame, {
            "source": "alpha_vantage",
            "status": "success",
            "function": function,
            "symbol": symbol,
            "message": "downloaded from Alpha Vantage",
        }

    return None, {
        "source": "alpha_vantage",
        "status": "unavailable",
        "message": " | ".join(attempts),
    }


def download_yahoo(start: str, end: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    import yfinance as yf

    frame = yf.download(
        "^VIX",
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise RuntimeError("Yahoo Finance returned no rows for ^VIX")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] for column in frame.columns]
    frame = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    output = _standard_output(frame, source="yahoo_finance")
    return _date_filter(output, start=start, end=end), {
        "source": "yahoo_finance",
        "status": "success",
        "symbol": "^VIX",
        "message": "downloaded from Yahoo Finance",
    }


def _date_filter(frame: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    output = frame.loc[frame.index >= pd.Timestamp(start)].copy()
    if end:
        output = output.loc[output.index < pd.Timestamp(end)]
    if output.empty:
        raise ValueError("No VIX rows remain after date filtering")
    return output


def _write_output(frame: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    out = frame.reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out.to_parquet(output_path, index=False)


def main() -> None:
    load_dotenv()
    args = parse_args()
    output_path = Path(args.output)
    report_path = Path(args.report)

    reports: list[dict[str, Any]] = []
    frame: pd.DataFrame | None = None
    if not args.force_yahoo:
        frame, alpha_report = try_alpha_vantage(start=args.start, end=args.end)
        reports.append(alpha_report)

    if frame is None:
        yahoo_frame, yahoo_report = download_yahoo(start=args.start, end=args.end)
        frame = yahoo_frame
        reports.append(yahoo_report)

    _write_output(frame, output_path)
    final_source = str(frame["source"].iloc[0])
    reports.append(
        {
            "source": final_source,
            "status": "saved",
            "path": str(output_path),
            "rows": len(frame),
            "start": frame.index.min().date().isoformat(),
            "end": frame.index.max().date().isoformat(),
            "message": "standard OHLCV parquet written",
        }
    )
    ensure_directory(report_path.parent)
    pd.DataFrame(reports).to_csv(report_path, index=False)
    print(f"VIX saved to {output_path}")
    print(f"Download report saved to {report_path}")
    print(
        pd.DataFrame(
            [
                {
                    "source": final_source,
                    "rows": len(frame),
                    "start": frame.index.min().date().isoformat(),
                    "end": frame.index.max().date().isoformat(),
                }
            ]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
