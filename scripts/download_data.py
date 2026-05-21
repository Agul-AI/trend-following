#!/usr/bin/env python
"""Download, validate, and process market data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.config import load_config
from trend_following.data_download import download_universe
from trend_following.data_processing import build_adjusted_panels
from trend_following.data_validation import validate_files
from trend_following.utils import as_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--force", action="store_true", help="Redownload even if local cache exists"
    )
    parser.add_argument("--tickers", nargs="*", help="Optional ticker override")
    parser.add_argument("--skip-processing", action="store_true", help="Only download raw files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tickers = as_list(args.tickers) or config.data.tickers
    print(f"Downloading {len(tickers)} ticker(s): {', '.join(tickers)}")
    download_universe(config, tickers=tickers, force=args.force)

    report_path = config.reports.tables_dir / "data_validation.csv"
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

    if not args.skip_processing:
        panels = build_adjusted_panels(config, tickers=tickers)
        print(
            "Processed panels saved to "
            f"{config.data.processed_dir} with {len(panels['returns']):,} aligned rows."
        )


if __name__ == "__main__":
    main()
