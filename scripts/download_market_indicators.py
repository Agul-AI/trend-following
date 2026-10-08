#!/usr/bin/env python
"""Download daily market-stress indicators from Yahoo Finance.

Alpha Vantage usually does not expose market indexes such as VXN, VIX3M, MOVE,
or Treasury-yield indexes through the equity time-series endpoints. This helper
uses Yahoo tickers and stores them in the same OHLCV parquet format used by the
project.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.utils import ensure_directory

DEFAULT_TICKERS = {
    "VXN": "^VXN",
    "VIX3M": "^VIX3M",
    "MOVE": "^MOVE",
    "TNX": "^TNX",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default="data/raw/market_indicators")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=list(DEFAULT_TICKERS),
        choices=list(DEFAULT_TICKERS),
        help="Indicator aliases to download.",
    )
    parser.add_argument(
        "--report",
        default="reports/tables/market_indicators_download_report.csv",
    )
    return parser.parse_args()


def _standardize_yahoo(frame: pd.DataFrame, source_symbol: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] for column in frame.columns]
    output = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    ).copy()
    if "adj_close" not in output.columns:
        output["adj_close"] = output["close"]
    if "volume" not in output.columns:
        output["volume"] = 0.0
    output["source_symbol"] = source_symbol
    output.index = pd.to_datetime(output.index).tz_localize(None)
    output.index.name = "date"
    return output[["open", "high", "low", "close", "adj_close", "volume", "source_symbol"]]


def main() -> None:
    import yfinance as yf

    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)
    rows: list[dict[str, object]] = []

    for alias in args.tickers:
        symbol = DEFAULT_TICKERS[alias]
        try:
            frame = yf.download(
                symbol,
                start=args.start,
                end=args.end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if frame.empty:
                raise RuntimeError("empty download")
            standardized = _standardize_yahoo(frame, source_symbol=symbol)
            output_path = output_dir / f"{alias}.parquet"
            out = standardized.reset_index()
            out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
            out.to_parquet(output_path, index=False)
            rows.append(
                {
                    "alias": alias,
                    "symbol": symbol,
                    "status": "success",
                    "rows": len(standardized),
                    "start": standardized.index.min().date().isoformat(),
                    "end": standardized.index.max().date().isoformat(),
                    "path": str(output_path),
                    "message": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep report for unavailable symbols
            rows.append(
                {
                    "alias": alias,
                    "symbol": symbol,
                    "status": "fail",
                    "rows": 0,
                    "start": "",
                    "end": "",
                    "path": "",
                    "message": str(exc),
                }
            )

    report = pd.DataFrame(rows)
    report_path = Path(args.report)
    ensure_directory(report_path.parent)
    report.to_csv(report_path, index=False)
    print(f"Report saved to {report_path}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
