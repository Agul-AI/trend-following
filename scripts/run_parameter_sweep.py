#!/usr/bin/env python
"""Run parameter sweep experiments with in-sample/out-of-sample reporting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.config import load_config
from trend_following.data_processing import build_adjusted_panels, load_processed_panels
from trend_following.experiments import run_parameter_sweep
from trend_following.utils import as_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--tickers", nargs="*", help="Optional ticker override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tickers = as_list(args.tickers) or config.data.tickers
    try:
        panels = load_processed_panels(config.data.processed_dir)
        prices = panels["adjusted_close"][tickers]
        returns = panels["returns"][tickers]
    except Exception:
        panels = build_adjusted_panels(config, tickers=tickers)
        prices = panels["adjusted_close"]
        returns = panels["returns"]
    table = run_parameter_sweep(prices, returns, config)
    print(f"Parameter sweep saved to {config.reports.tables_dir / 'parameter_sweep.csv'}")
    print(table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
