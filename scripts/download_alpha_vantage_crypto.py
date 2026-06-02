#!/usr/bin/env python
"""Download Alpha Vantage crypto intraday OHLCV data to parquet caches.

This script is intentionally separate from the equity/ETF downloader because
crypto trades 24/7 and uses Alpha Vantage's CRYPTO_INTRADAY endpoint rather
than the equity monthly TIME_SERIES_INTRADAY endpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.data_download import _alpha_vantage_get, _normalize_alpha_vantage_csv
from trend_following.utils import ensure_directory

SUPPORTED_INTERVALS = {
    "1min": "1min",
    "1m": "1min",
    "5min": "5min",
    "5m": "5min",
    "15min": "15min",
    "15m": "15min",
    "30min": "30min",
    "30m": "30min",
    "60min": "60min",
    "60m": "60min",
    "1h": "60min",
}
DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "LINK", "AVAX", "LTC", "BCH", "DOT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Crypto symbols to download, without market suffix.",
    )
    parser.add_argument(
        "--market",
        default="USD",
        help="Quote market/currency for Alpha Vantage crypto pairs.",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=["15min", "30min", "60min"],
        help="Intraday intervals: 1min, 5min, 15min, 30min, 60min.",
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Root raw data directory. Interval folders are created below it.",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports/tables",
        help="Directory for crypto validation summary CSVs.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload cached parquet files.")
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=None,
        help="Override ALPHA_VANTAGE_PAUSE_SECONDS / request-start interval.",
    )
    return parser.parse_args()


def normalize_interval(interval: str) -> str:
    normalized = interval.strip().lower()
    if normalized not in SUPPORTED_INTERVALS:
        supported = ", ".join(sorted(SUPPORTED_INTERVALS))
        raise ValueError(f"Unsupported crypto interval {interval!r}; supported: {supported}")
    return SUPPORTED_INTERVALS[normalized]


def crypto_raw_dir(raw_root: Path, interval: str) -> Path:
    return raw_root / f"alpha_vantage_crypto_{interval}"


def summarize_frame(frame: pd.DataFrame, symbol: str, source: str) -> str:
    if frame.empty:
        return f"{symbol}: {source}, empty"
    dates = pd.to_datetime(frame["date"])
    return f"{symbol}: {source}, {len(frame):,} rows, {dates.min()} -> {dates.max()}"


def download_crypto_symbol(
    symbol: str,
    market: str,
    interval: str,
    output_dir: Path,
    force: bool = False,
) -> Path:
    """Download one crypto symbol/market/interval to a parquet file."""
    symbol = symbol.upper()
    market = market.upper()
    output_path = output_dir / f"{symbol}_{market}.parquet"
    if output_path.exists() and not force:
        cached = pd.read_parquet(output_path)
        print(summarize_frame(cached, f"{symbol}/{market}", "cached"))
        return output_path

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY is not set. Add it to .env or the shell.")

    params = {
        "function": "CRYPTO_INTRADAY",
        "symbol": symbol,
        "market": market,
        "interval": interval,
        "outputsize": "full",
        "datatype": "csv",
        "apikey": api_key,
    }
    response = _alpha_vantage_get(params, ticker=f"{symbol}/{market}", label=f"crypto_{interval}")
    frame = _normalize_alpha_vantage_csv(response.text, ticker=f"{symbol}/{market}")
    if frame.empty:
        raise ValueError(f"No Alpha Vantage crypto rows returned for {symbol}/{market} {interval}")
    frame = frame.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    frame.to_parquet(output_path, index=False)
    print(summarize_frame(frame, f"{symbol}/{market}", "downloaded"))
    return output_path


def validate_crypto_files(
    symbols: list[str],
    market: str,
    interval: str,
    raw_dir: Path,
    report_path: Path,
) -> pd.DataFrame:
    """Write a lightweight crypto validation summary.

    Crypto trades 24/7, so weekend timestamps are expected and are not warnings.
    """
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        pair = f"{symbol.upper()}_{market.upper()}"
        path = raw_dir / f"{pair}.parquet"
        if not path.exists():
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "market": market.upper(),
                    "status": "fail",
                    "rows": 0,
                    "start": "",
                    "end": "",
                    "messages": f"missing raw file: {path}",
                }
            )
            continue
        try:
            frame = pd.read_parquet(path)
            dates = pd.to_datetime(frame["date"], errors="coerce")
            missing_values = int(frame.isna().sum().sum())
            invalid_prices = int((frame[["open", "high", "low", "close", "adj_close"]] <= 0).sum().sum())
            duplicate_dates = int(dates.duplicated().sum())
            monotonic = bool(dates.is_monotonic_increasing)
            messages = []
            if dates.isna().any():
                messages.append(f"{int(dates.isna().sum())} invalid timestamps")
            if duplicate_dates:
                messages.append(f"{duplicate_dates} duplicate timestamps")
            if not monotonic:
                messages.append("timestamps are not monotonic increasing")
            if missing_values:
                messages.append(f"{missing_values} missing values")
            if invalid_prices:
                messages.append(f"{invalid_prices} zero/negative prices")
            status = "fail" if invalid_prices or duplicate_dates or not monotonic else ("warn" if messages else "pass")
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "market": market.upper(),
                    "status": status,
                    "rows": len(frame),
                    "start": dates.min(),
                    "end": dates.max(),
                    "messages": "; ".join(messages),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "market": market.upper(),
                    "status": "fail",
                    "rows": 0,
                    "start": "",
                    "end": "",
                    "messages": f"read/validation error: {type(exc).__name__}: {exc}",
                }
            )
    report = pd.DataFrame(rows)
    ensure_directory(report_path.parent)
    report.to_csv(report_path, index=False)
    return report


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.pause_seconds is not None:
        os.environ["ALPHA_VANTAGE_PAUSE_SECONDS"] = str(args.pause_seconds)
        os.environ["ALPHA_VANTAGE_REQUEST_START_INTERVAL_SECONDS"] = str(args.pause_seconds)
    os.environ.setdefault("ALPHA_VANTAGE_RATE_LIMIT_MODE", "start_time")

    raw_root = Path(args.raw_root)
    reports_dir = Path(args.reports_dir)
    symbols = [symbol.upper() for symbol in args.symbols]
    intervals = [normalize_interval(interval) for interval in args.intervals]

    print(f"Crypto universe: {', '.join(symbols)}")
    print(f"Market: {args.market.upper()}")
    print(f"Intervals: {', '.join(intervals)}")

    failures: list[dict[str, str]] = []
    for interval in intervals:
        output_dir = crypto_raw_dir(raw_root, interval)
        ensure_directory(output_dir)
        print(f"\n=== Downloading Alpha Vantage crypto {interval} ===")
        for index, symbol in enumerate(symbols, start=1):
            print(f"[{index}/{len(symbols)}] {symbol}/{args.market.upper()}")
            try:
                download_crypto_symbol(
                    symbol=symbol,
                    market=args.market,
                    interval=interval,
                    output_dir=output_dir,
                    force=args.force,
                )
            except Exception as exc:  # pragma: no cover - network/vendor failures vary
                message = str(exc)
                print(f"WARNING: failed {symbol}/{args.market.upper()} {interval}: {message}")
                failures.append(
                    {
                        "symbol": symbol,
                        "market": args.market.upper(),
                        "interval": interval,
                        "error": message,
                    }
                )

        report_path = reports_dir / f"data_validation_alpha_vantage_crypto_{interval}.csv"
        report = validate_crypto_files(symbols, args.market, interval, output_dir, report_path)
        print(f"Validation report saved to {report_path}")
        print(report[["symbol", "market", "status", "rows", "start", "end", "messages"]].to_string(index=False))

    if failures:
        failure_path = reports_dir / "alpha_vantage_crypto_download_failures.csv"
        existing = pd.read_csv(failure_path) if failure_path.exists() else pd.DataFrame()
        combined = pd.concat([existing, pd.DataFrame(failures)], ignore_index=True)
        ensure_directory(failure_path.parent)
        combined.to_csv(failure_path, index=False)
        print(f"\nLogged {len(failures)} crypto download failure(s) to {failure_path}")


if __name__ == "__main__":
    main()
