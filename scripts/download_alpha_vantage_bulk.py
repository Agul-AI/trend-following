#!/usr/bin/env python
"""Bulk bootstrap Alpha Vantage max-history ETF datasets.

This script is designed for a short Alpha Vantage Premium subscription window.
It downloads daily adjusted data first, uses each ticker's actual first daily
observation to avoid pre-inception monthly intraday calls, then downloads the
requested intraday intervals into separate parquet caches.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.config import ProjectConfig, load_config
from trend_following.data_download import download_ticker
from trend_following.data_validation import validate_files
from trend_following.utils import as_list, ensure_directory

INTERVAL_DIR_NAMES = {
    "15min": "15min",
    "15m": "15min",
    "30min": "30min",
    "30m": "30min",
    "60min": "60min",
    "60m": "60min",
    "1h": "60min",
    "1d": "daily_adjusted",
    "d": "daily_adjusted",
    "daily": "daily_adjusted",
}
INTRADAY_INTERVALS = {"15min", "15m", "30min", "30m", "60min", "60m", "1h"}
DAILY_INTERVALS = {"1d", "d", "daily"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/alpha_vantage_max_history.yaml",
        help="Config containing the Alpha Vantage ticker universe and start date.",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=["15min", "30min", "60min", "1d"],
        help="Intervals to download. Supported: 15min 30min 60min 1d.",
    )
    parser.add_argument("--tickers", nargs="*", help="Optional ticker subset override.")
    parser.add_argument("--force", action="store_true", help="Redownload existing parquet files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print request estimates without downloading.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip writing validation CSVs after each interval.",
    )
    parser.add_argument(
        "--no-infer-starts",
        action="store_true",
        help="Do not use daily adjusted first dates to avoid pre-inception intraday months.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=None,
        help="Override ALPHA_VANTAGE_PAUSE_SECONDS for this run, e.g. 0.85 for 75 req/min.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on a ticker failure instead of logging failures and continuing.",
    )
    return parser.parse_args()


def normalize_interval(interval: str) -> str:
    normalized = interval.lower()
    if normalized not in INTERVAL_DIR_NAMES:
        supported = ", ".join(sorted(INTERVAL_DIR_NAMES))
        raise ValueError(f"Unsupported interval {interval!r}; supported: {supported}")
    if normalized == "15m":
        return "15min"
    if normalized == "30m":
        return "30min"
    if normalized in {"60m", "1h"}:
        return "60min"
    if normalized in {"d", "daily"}:
        return "1d"
    return normalized


def interval_config(
    config: ProjectConfig, interval: str, start_date: str | None = None
) -> ProjectConfig:
    normalized = normalize_interval(interval)
    dirname = INTERVAL_DIR_NAMES[normalized]
    data = replace(
        config.data,
        source="alpha_vantage",
        interval=normalized,
        start_date=start_date or config.data.start_date,
        raw_dir=config.root / "data" / "raw" / f"alpha_vantage_{dirname}",
        processed_dir=config.root / "data" / "processed" / f"alpha_vantage_{dirname}",
        alignment="outer",
    )
    ensure_directory(data.raw_dir)
    ensure_directory(data.processed_dir)
    return replace(config, data=data)


def month_count(start_date: str, end_date: str | None) -> int:
    start = pd.Timestamp(start_date).normalize().replace(day=1)
    end = pd.Timestamp(end_date).normalize() if end_date else pd.Timestamp.today().normalize()
    end = end.replace(day=1)
    return len(pd.date_range(start=start, end=end, freq="MS"))


def daily_first_dates(daily_raw_dir: Path, tickers: list[str]) -> dict[str, str]:
    first_dates: dict[str, str] = {}
    for ticker in tickers:
        path = daily_raw_dir / f"{ticker}.parquet"
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path, columns=["date"])
        except Exception:
            frame = pd.read_parquet(path)
        if "date" not in frame or frame.empty:
            continue
        first = pd.to_datetime(frame["date"]).min()
        if pd.notna(first):
            first_dates[ticker] = first.date().isoformat()
    return first_dates


def estimate_requests(
    base_config: ProjectConfig,
    intervals: list[str],
    tickers: list[str],
    force: bool,
    first_dates: dict[str, str] | None = None,
) -> dict[str, int]:
    first_dates = first_dates or {}
    estimates: dict[str, int] = {}
    for interval in intervals:
        cfg = interval_config(base_config, interval)
        normalized = normalize_interval(interval)
        count = 0
        for ticker in tickers:
            if (cfg.data.raw_dir / f"{ticker}.parquet").exists() and not force:
                continue
            if normalized in DAILY_INTERVALS:
                count += 1
            else:
                ticker_start = max(
                    base_config.data.start_date,
                    first_dates.get(ticker, base_config.data.start_date),
                )
                count += month_count(ticker_start, base_config.data.end_date)
        estimates[normalized] = count
    return estimates


def maybe_validate(config: ProjectConfig, tickers: list[str], interval: str) -> None:
    report_path = config.reports.tables_dir / f"data_validation_alpha_vantage_{interval}.csv"
    report = validate_files(
        config.data.raw_dir,
        tickers,
        report_path=report_path,
        suspicious_gap_days=config.data.suspicious_gap_days,
    )
    print(f"Validation report saved to {report_path}")
    print(
        report[["ticker", "status", "rows", "start_date", "end_date", "messages"]].to_string(
            index=False
        )
    )


def download_interval(
    base_config: ProjectConfig,
    interval: str,
    tickers: list[str],
    force: bool,
    first_dates: dict[str, str],
    skip_validation: bool,
    stop_on_error: bool,
) -> list[dict[str, str]]:
    normalized = normalize_interval(interval)
    print(f"\n=== Downloading Alpha Vantage {normalized} for {len(tickers)} ticker(s) ===")
    base_interval_config = interval_config(base_config, normalized)
    pause_seconds = float(os.getenv("ALPHA_VANTAGE_PAUSE_SECONDS", "12"))
    start_time_rate_limiter = os.getenv(
        "ALPHA_VANTAGE_RATE_LIMIT_MODE", "start_time"
    ).strip().lower() in {"start_time", "start", "start-time"}
    failures: list[dict[str, str]] = []

    for index, ticker in enumerate(tickers, start=1):
        ticker_start = base_config.data.start_date
        if normalized in INTRADAY_INTERVALS and ticker in first_dates:
            ticker_start = max(base_config.data.start_date, first_dates[ticker])
        ticker_config = interval_config(base_config, normalized, start_date=ticker_start)
        print(f"[{index}/{len(tickers)}] {ticker} start={ticker_start}")
        output_path = ticker_config.data.raw_dir / f"{ticker}.parquet"
        was_cached = output_path.exists() and not force
        try:
            download_ticker(ticker_config, ticker, force=force)
        except Exception as exc:
            if stop_on_error:
                raise
            message = str(exc)
            print(f"WARNING: failed {ticker} {normalized}: {message}")
            failures.append(
                {
                    "interval": normalized,
                    "ticker": ticker,
                    "start_date": ticker_start,
                    "error": message,
                }
            )
        if (
            not start_time_rate_limiter
            and not was_cached
            and pause_seconds > 0
            and index < len(tickers)
        ):
            # The monthly intraday downloader pauses between monthly requests. This
            # adds the missing pause between the last request of one ticker and the
            # first request of the next ticker, and rate-limits daily downloads too.
            # In start_time mode, all Alpha Vantage requests are already paced by
            # request start time inside trend_following.data_download.
            time.sleep(pause_seconds)

    if not skip_validation:
        maybe_validate(base_interval_config, tickers, normalized)

    return failures


def main() -> None:
    args = parse_args()
    if args.pause_seconds is not None:
        os.environ["ALPHA_VANTAGE_PAUSE_SECONDS"] = str(args.pause_seconds)
    os.environ.setdefault("ALPHA_VANTAGE_RATE_LIMIT_MODE", "start_time")

    config = load_config(args.config)
    if config.data.source != "alpha_vantage":
        raise ValueError("Bulk script requires data.source: alpha_vantage")
    os.environ.setdefault(
        "ALPHA_VANTAGE_SKIPPED_MONTHS_PATH",
        str(config.reports.tables_dir / "alpha_vantage_skipped_months.csv"),
    )
    tickers = as_list(args.tickers) or config.data.tickers
    intervals = [normalize_interval(interval) for interval in args.intervals]

    print(f"Universe: {len(tickers)} ticker(s)")
    print(f"Intervals: {', '.join(intervals)}")
    print(
        f"Configured start date: {config.data.start_date}; end date: {config.data.end_date or 'today'}"
    )
    print(f"Alpha Vantage pause: {os.getenv('ALPHA_VANTAGE_PAUSE_SECONDS', '12')} seconds/request")

    first_dates: dict[str, str] = {}
    all_failures: list[dict[str, str]] = []
    if "1d" in intervals and not args.dry_run:
        all_failures.extend(
            download_interval(
                config,
                "1d",
                tickers=tickers,
                force=args.force,
                first_dates={},
                skip_validation=args.skip_validation,
                stop_on_error=args.stop_on_error,
            )
        )
        first_dates = daily_first_dates(interval_config(config, "1d").data.raw_dir, tickers)
    elif not args.no_infer_starts:
        first_dates = daily_first_dates(interval_config(config, "1d").data.raw_dir, tickers)

    if first_dates:
        print(f"Using daily-adjusted first dates for {len(first_dates)} ticker(s).")

    estimates = estimate_requests(config, intervals, tickers, args.force, first_dates)
    total = sum(estimates.values())
    print("\nEstimated remaining/new requests by interval:")
    for interval, count in estimates.items():
        print(f"  {interval}: {count:,}")
    print(f"  total: {total:,}")
    if args.dry_run:
        return

    for interval in intervals:
        if normalize_interval(interval) in DAILY_INTERVALS:
            # Already handled first so first-date inference can reduce intraday calls.
            continue
        all_failures.extend(
            download_interval(
                config,
                interval,
                tickers=tickers,
                force=args.force,
                first_dates={} if args.no_infer_starts else first_dates,
                skip_validation=args.skip_validation,
                stop_on_error=args.stop_on_error,
            )
        )

    if all_failures:
        failure_path = config.reports.tables_dir / "alpha_vantage_download_failures.csv"
        existing = pd.read_csv(failure_path) if failure_path.exists() else pd.DataFrame()
        failures = pd.DataFrame(all_failures)
        combined = pd.concat([existing, failures], ignore_index=True)
        combined.to_csv(failure_path, index=False)
        print(f"\nLogged {len(all_failures)} download failure(s) to {failure_path}")


if __name__ == "__main__":
    main()
