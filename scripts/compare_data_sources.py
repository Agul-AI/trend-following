#!/usr/bin/env python
"""Compare overlapping raw OHLCV data from two vendor cache directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.config import load_config
from trend_following.data_compare import compare_price_directories
from trend_following.utils import as_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Base config for tickers/report path"
    )
    parser.add_argument("--left-dir", required=True, help="First raw parquet directory")
    parser.add_argument("--right-dir", required=True, help="Second raw parquet directory")
    parser.add_argument("--left-label", default="left", help="Label for first source")
    parser.add_argument("--right-label", default="right", help="Label for second source")
    parser.add_argument(
        "--left-timezone",
        default=None,
        help="Timezone for naive timestamps in left source, e.g. America/New_York",
    )
    parser.add_argument(
        "--right-timezone",
        default=None,
        help="Timezone for naive timestamps in right source, e.g. America/New_York",
    )
    parser.add_argument("--tickers", nargs="*", help="Optional ticker override")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path; defaults to reports/tables/data_source_comparison_<labels>.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tickers = as_list(args.tickers) or config.data.tickers
    output = (
        Path(args.output)
        if args.output
        else config.reports.tables_dir
        / f"data_source_comparison_{args.left_label}_vs_{args.right_label}.csv"
    )
    report = compare_price_directories(
        left_dir=args.left_dir,
        right_dir=args.right_dir,
        tickers=tickers,
        output_path=output,
        left_label=args.left_label,
        right_label=args.right_label,
        left_timezone=args.left_timezone,
        right_timezone=args.right_timezone,
    )
    print(f"Comparison report saved to {output}")
    columns = [
        "ticker",
        "status",
        "left_rows",
        "right_rows",
        "overlap_rows",
        "overlap_start",
        "overlap_end",
        "close_mean_abs_diff",
        "close_mean_abs_pct_diff",
        "close_corr",
    ]
    existing = [column for column in columns if column in report.columns]
    print(report[existing].to_string(index=False))


if __name__ == "__main__":
    main()
