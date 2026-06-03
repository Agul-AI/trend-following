#!/usr/bin/env python
"""Compare Top-8 QQQ/200MA-distance stats before first +100% 3x gain.

The main question is whether the QQQ distance-from-hourly-200-day-MA
distribution changes materially if each Top-8 winner is truncated at the first
time synthetic QQQ_3X_CALC reaches +100% unrealized gain from trade entry.

For trades that never reach +100% (currently T55), the default comparison keeps
the available trade window in the "pre100_or_full_if_no_hit" sample and also
reports a separate "hit_trades_only" sample.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qqq-path", default="data/raw/alpha_vantage_60min/QQQ.parquet")
    parser.add_argument("--target-path", default="data/raw/synthetic_3x_60min/QQQ_3X_CALC.parquet")
    parser.add_argument(
        "--top-trades-path",
        default="reports/tables/preferred_plus_40pct_peak_stop_best_12_winning_trades.csv",
    )
    parser.add_argument(
        "--top8-label-path",
        default="reports/tables/preferred_top8_winning_trades_time_order.csv",
    )
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--current-trade-id", type=int, default=55)
    parser.add_argument("--bars-per-day", type=int, default=6)
    parser.add_argument("--ma-days", type=int, default=200)
    parser.add_argument("--gain-threshold", type=float, default=1.0)
    parser.add_argument("--output-prefix", default="preferred_top8_pre100_qqq_200ma_distance")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--tables-dir", default="reports/tables")
    return parser.parse_args()


def load_close(path: Path, name: str) -> pd.Series:
    """Load close/adjusted-close from parquet."""
    frame = pd.read_parquet(path)
    price_col = "adj_close" if "adj_close" in frame.columns else "close"
    if "date" not in frame.columns or price_col not in frame.columns:
        raise ValueError(f"{path} must contain date and {price_col}")
    frame = frame[["date", price_col]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna().drop_duplicates("date").sort_values("date").set_index("date")
    return frame[price_col].astype(float).rename(name)


def load_top_trades(top_trades_path: Path, top8_label_path: Path, top_n: int) -> pd.DataFrame:
    """Load Top-N winners with stable labels."""
    trades = pd.read_csv(top_trades_path)
    top = trades.sort_values("final_return_pct", ascending=False).head(top_n).copy()
    top["top_rank"] = np.arange(1, len(top) + 1)
    top["Trade"] = "T" + top["trade_id"].astype(str)
    top["Top8"] = "Top8 #" + top["top_rank"].astype(str)
    if top8_label_path.exists():
        labels = pd.read_csv(top8_label_path)
        if {"Top8", "Trade"}.issubset(labels.columns):
            label_map = dict(zip(labels["Trade"], labels["Top8"], strict=True))
            top["Top8"] = top["Trade"].map(label_map).fillna(top["Top8"])
    top["entry_ts"] = pd.to_datetime(top["entry_date"])
    top["exit_ts"] = pd.to_datetime(top["exit_date"])
    return top.sort_values("entry_ts").reset_index(drop=True)


def distribution_summary(values: pd.Series, sample: str, trades: str) -> dict[str, Any]:
    """Return distribution summary in percentage units."""
    pct = values.dropna().astype(float) * 100.0
    return {
        "sample": sample,
        "trades": trades,
        "observations": int(pct.size),
        "mean_distance_pct": float(pct.mean()),
        "median_distance_pct": float(pct.median()),
        "std_distance_pct": float(pct.std(ddof=1)),
        "min_distance_pct": float(pct.min()),
        "p05_distance_pct": float(pct.quantile(0.05)),
        "p10_distance_pct": float(pct.quantile(0.10)),
        "p25_distance_pct": float(pct.quantile(0.25)),
        "p75_distance_pct": float(pct.quantile(0.75)),
        "p90_distance_pct": float(pct.quantile(0.90)),
        "p95_distance_pct": float(pct.quantile(0.95)),
        "max_distance_pct": float(pct.max()),
        "pct_below_200ma": float((pct < 0).mean() * 100.0),
        "pct_0_to_5pct_above": float(((pct >= 0) & (pct < 5)).mean() * 100.0),
        "pct_5_to_10pct_above": float(((pct >= 5) & (pct < 10)).mean() * 100.0),
        "pct_10_to_20pct_above": float(((pct >= 10) & (pct < 20)).mean() * 100.0),
        "pct_20_to_30pct_above": float(((pct >= 20) & (pct < 30)).mean() * 100.0),
        "pct_above_30pct": float((pct >= 30).mean() * 100.0),
    }


def build_samples(
    trades: pd.DataFrame,
    qqq_distance: pd.Series,
    target: pd.Series,
    *,
    current_trade_id: int,
    gain_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build full and pre-first-gain samples plus per-trade summary."""
    latest = target.index.max()
    full_rows: list[pd.DataFrame] = []
    pre_rows: list[pd.DataFrame] = []
    per_trade_rows: list[dict[str, Any]] = []

    for trade in trades.itertuples(index=False):
        data_end = min(trade.exit_ts, latest)
        if int(trade.trade_id) == current_trade_id:
            data_end = latest

        target_window = target.loc[(target.index >= trade.entry_ts) & (target.index <= data_end)].dropna()
        if target_window.empty:
            continue
        entry_price = float(target_window.iloc[0])
        trade_return = target_window / entry_price - 1.0
        hits = trade_return.loc[trade_return >= gain_threshold]
        hit = not hits.empty
        first_hit_ts = hits.index[0] if hit else pd.NaT
        pre_end = first_hit_ts if hit else data_end

        full_window = qqq_distance.loc[
            (qqq_distance.index >= trade.entry_ts) & (qqq_distance.index <= data_end)
        ].dropna()
        pre_window = qqq_distance.loc[
            (qqq_distance.index >= trade.entry_ts) & (qqq_distance.index <= pre_end)
        ].dropna()

        if not full_window.empty:
            full_rows.append(
                pd.DataFrame(
                    {
                        "timestamp": full_window.index,
                        "Top8": trade.Top8,
                        "Trade": trade.Trade,
                        "trade_id": int(trade.trade_id),
                        "qqq_distance_to_200ma": full_window.values,
                    }
                )
            )
        if not pre_window.empty:
            pre_rows.append(
                pd.DataFrame(
                    {
                        "timestamp": pre_window.index,
                        "Top8": trade.Top8,
                        "Trade": trade.Trade,
                        "trade_id": int(trade.trade_id),
                        "hit_gain_threshold": hit,
                        "qqq_distance_to_200ma": pre_window.values,
                    }
                )
            )

        hit_dist = qqq_distance.reindex([first_hit_ts]).iloc[0] if hit else np.nan
        denominator = (data_end - trade.entry_ts).total_seconds()
        normalized_hit_time = (
            (first_hit_ts - trade.entry_ts).total_seconds() / denominator
            if hit and denominator > 0
            else np.nan
        )
        per_trade_rows.append(
            {
                "Top8": trade.Top8,
                "Trade": trade.Trade,
                "hit_gain_threshold": hit,
                "first_threshold_hit": first_hit_ts,
                "normalized_time_to_threshold": normalized_hit_time,
                "qqq_distance_at_threshold_hit_pct": float(hit_dist) * 100.0 if pd.notna(hit_dist) else np.nan,
                "pre_observations": int(pre_window.size),
                "full_observations": int(full_window.size),
                "pre_mean_distance_pct": float(pre_window.mean() * 100.0),
                "full_mean_distance_pct": float(full_window.mean() * 100.0),
                "pre_median_distance_pct": float(pre_window.median() * 100.0),
                "full_median_distance_pct": float(full_window.median() * 100.0),
            }
        )

    return (
        pd.concat(full_rows, ignore_index=True),
        pd.concat(pre_rows, ignore_index=True),
        pd.DataFrame(per_trade_rows),
    )


def make_plot(summary: pd.DataFrame, full: pd.DataFrame, pre: pd.DataFrame, output_path: Path) -> None:
    """Make a compact comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    full_pct = full["qqq_distance_to_200ma"] * 100.0
    pre_pct = pre["qqq_distance_to_200ma"] * 100.0

    axes[0].hist(full_pct, bins=45, alpha=0.55, label="Full Top8", color="#4c78a8")
    axes[0].hist(pre_pct, bins=45, alpha=0.55, label="Before first +100% / full if no hit", color="#f58518")
    axes[0].axvline(full_pct.median(), color="#4c78a8", linestyle="--", linewidth=1.2)
    axes[0].axvline(pre_pct.median(), color="#f58518", linestyle="--", linewidth=1.2)
    axes[0].set_title("Distance distribution")
    axes[0].set_xlabel("QQQ distance from hourly 200-day MA (%)")
    axes[0].set_ylabel("Hourly bars")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    metric_cols = [
        "mean_distance_pct",
        "median_distance_pct",
        "p10_distance_pct",
        "p90_distance_pct",
        "pct_20_to_30pct_above",
    ]
    display_names = ["Mean", "Median", "P10", "P90", "20-30% share"]
    compact = summary.set_index("sample").loc[
        ["full_top8", "pre_first_100_or_full_if_no_hit_top8"], metric_cols
    ]
    x = np.arange(len(metric_cols))
    width = 0.35
    axes[1].bar(x - width / 2, compact.iloc[0].values, width, label="Full", color="#4c78a8")
    axes[1].bar(x + width / 2, compact.iloc[1].values, width, label="Pre-100", color="#f58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_names, rotation=20, ha="right")
    axes[1].set_title("Key statistics")
    axes[1].set_ylabel("%")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("Top-8 QQQ/200MA distance: full trade vs before first +100% synthetic-3x gain")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    qqq = load_close(Path(args.qqq_path), "QQQ")
    target = load_close(Path(args.target_path), "QQQ_3X_CALC")
    common = qqq.index.intersection(target.index)
    qqq = qqq.loc[common]
    target = target.loc[common]
    qqq_ma = qqq.rolling(args.ma_days * args.bars_per_day, min_periods=args.ma_days * args.bars_per_day).mean()
    qqq_distance = (qqq / qqq_ma - 1.0).rename("qqq_distance_to_200ma")

    trades = load_top_trades(Path(args.top_trades_path), Path(args.top8_label_path), args.top_n)
    full, pre, per_trade = build_samples(
        trades,
        qqq_distance,
        target,
        current_trade_id=args.current_trade_id,
        gain_threshold=args.gain_threshold,
    )
    pre_hit_only = pre.loc[pre["hit_gain_threshold"]].copy()

    summary = pd.DataFrame(
        [
            distribution_summary(
                full["qqq_distance_to_200ma"],
                "full_top8",
                ",".join(full["Trade"].drop_duplicates()),
            ),
            distribution_summary(
                pre["qqq_distance_to_200ma"],
                "pre_first_100_or_full_if_no_hit_top8",
                ",".join(pre["Trade"].drop_duplicates()),
            ),
            distribution_summary(
                pre_hit_only["qqq_distance_to_200ma"],
                "pre_first_100_hit_trades_only",
                ",".join(pre_hit_only["Trade"].drop_duplicates()),
            ),
        ]
    )
    summary.to_csv(tables_dir / f"{args.output_prefix}_comparison.csv", index=False)
    per_trade.to_csv(tables_dir / f"{args.output_prefix}_by_trade_summary.csv", index=False)
    pre.to_csv(tables_dir / f"{args.output_prefix}_observations.csv", index=False)
    make_plot(summary, full, pre, figures_dir / f"{args.output_prefix}_comparison.png")

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved table: {tables_dir / f'{args.output_prefix}_comparison.csv'}")
    print(f"Saved per-trade table: {tables_dir / f'{args.output_prefix}_by_trade_summary.csv'}")
    print(f"Saved figure: {figures_dir / f'{args.output_prefix}_comparison.png'}")


if __name__ == "__main__":
    main()
