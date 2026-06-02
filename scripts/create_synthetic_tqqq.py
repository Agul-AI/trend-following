#!/usr/bin/env python
"""Create synthetic TQQQ from QQQ using a perfect daily-reset 3x rule."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.data_validation import read_price_file
from trend_following.synthetic_leverage import (
    synthetic_daily_leveraged_ohlcv,
    synthetic_intraday_leveraged_ohlcv,
)
from trend_following.utils import ensure_directory

DEFAULT_INTRADAY_DIRS = {
    "15min": Path("data/raw/alpha_vantage_15min"),
    "30min": Path("data/raw/alpha_vantage_30min"),
    "60min": Path("data/raw/alpha_vantage_60min"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", default="QQQ", help="Underlying ticker")
    parser.add_argument("--output-ticker", default="TQQQ_CALC", help="Synthetic output ticker")
    parser.add_argument("--leverage", type=float, default=3.0, help="Daily leverage multiple")
    parser.add_argument(
        "--initial-price",
        type=float,
        default=100.0,
        help="Synthetic adjusted close at first daily observation",
    )
    parser.add_argument(
        "--daily-input",
        default="data/raw/alpha_vantage_daily_adjusted/QQQ.parquet",
        help="Daily adjusted QQQ parquet input",
    )
    parser.add_argument(
        "--output-root",
        default="data/raw",
        help="Root directory for synthetic output folders",
    )
    parser.add_argument(
        "--intervals",
        nargs="*",
        default=["1d", "15min", "30min", "60min"],
        choices=["1d", "15min", "30min", "60min"],
        help="Synthetic intervals to create",
    )
    return parser.parse_args()


def _write_price_frame(frame: pd.DataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    out = frame.reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    daily_input = Path(args.daily_input)
    daily_underlying = read_price_file(daily_input).sort_index()
    daily_synthetic = synthetic_daily_leveraged_ohlcv(
        daily_underlying,
        leverage=args.leverage,
        initial_price=args.initial_price,
    )

    rows: list[dict[str, object]] = []
    if "1d" in args.intervals:
        daily_output = output_root / "synthetic_tqqq_1d" / f"{args.output_ticker}.parquet"
        _write_price_frame(daily_synthetic, daily_output)
        rows.append(
            {
                "interval": "1d",
                "path": str(daily_output),
                "rows": len(daily_synthetic),
                "start": daily_synthetic.index.min(),
                "end": daily_synthetic.index.max(),
            }
        )

    for interval, input_dir in DEFAULT_INTRADAY_DIRS.items():
        if interval not in args.intervals:
            continue
        input_path = input_dir / f"{args.underlying}.parquet"
        if not input_path.exists():
            rows.append(
                {
                    "interval": interval,
                    "path": "",
                    "rows": 0,
                    "start": "",
                    "end": "",
                    "message": f"missing input: {input_path}",
                }
            )
            continue
        intraday_underlying = read_price_file(input_path).sort_index()
        intraday_synthetic = synthetic_intraday_leveraged_ohlcv(
            intraday_underlying=intraday_underlying,
            daily_underlying=daily_underlying,
            daily_synthetic=daily_synthetic,
            leverage=args.leverage,
        )
        output_path = output_root / f"synthetic_tqqq_{interval}" / f"{args.output_ticker}.parquet"
        _write_price_frame(intraday_synthetic, output_path)
        rows.append(
            {
                "interval": interval,
                "path": str(output_path),
                "rows": len(intraday_synthetic),
                "start": intraday_synthetic.index.min(),
                "end": intraday_synthetic.index.max(),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = Path("reports/tables/synthetic_tqqq_creation_summary.csv")
    ensure_directory(summary_path.parent)
    summary.to_csv(summary_path, index=False)
    print(f"Synthetic {args.output_ticker} summary saved to {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
