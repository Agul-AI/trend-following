#!/usr/bin/env python
"""Summarize QQQ distance from its hourly 200-day MA during Top-8 winners.

This analysis is tied to the current preferred strategy research thread:
- QQQ hourly MACD histogram entry.
- QQQ hourly 200-day moving-average entry/exit gate.
- Synthetic QQQ_3X_CALC exposure.
- Optional -40% synthetic-3x peak stop in the trade table used here.

The statistic reported here is:

    QQQ hourly close / QQQ hourly 200-day MA - 1

where the hourly 200-day MA is implemented as 200 trading days * 6 hourly bars
per day = 1,200 bars, matching the Alpha Vantage 60min cache.
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
    parser.add_argument(
        "--top-trades-path",
        default="reports/tables/preferred_plus_40pct_peak_stop_best_12_winning_trades.csv",
    )
    parser.add_argument(
        "--top8-label-path",
        default="reports/tables/preferred_top8_winning_trades_time_order.csv",
        help="Optional prior table used only for stable Top8 labels and time order.",
    )
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--bars-per-day", type=int, default=6)
    parser.add_argument("--ma-days", type=int, default=200)
    parser.add_argument(
        "--output-prefix",
        default="preferred_top8_including_current_qqq_200ma_distance",
    )
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--tables-dir", default="reports/tables")
    return parser.parse_args()


def load_hourly_close(path: Path) -> pd.Series:
    """Load QQQ hourly close from a parquet file."""
    frame = pd.read_parquet(path)
    if "date" not in frame.columns:
        raise ValueError(f"{path} must contain a 'date' column")
    price_column = "adj_close" if "adj_close" in frame.columns else "close"
    if price_column not in frame.columns:
        raise ValueError(f"{path} must contain 'close' or 'adj_close'")
    data = frame[["date", price_column]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.dropna(subset=["date", price_column]).drop_duplicates("date")
    data = data.sort_values("date").set_index("date")
    close = data[price_column].astype(float)
    if close.le(0).any():
        raise ValueError(f"{path} contains non-positive prices")
    return close.rename("QQQ")


def load_top_trades(top_trades_path: Path, top8_label_path: Path | None, top_n: int) -> pd.DataFrame:
    """Load Top-N winning trades and attach stable Top8 labels."""
    trades = pd.read_csv(top_trades_path)
    required = {"trade_id", "entry_date", "exit_date", "final_return_pct"}
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"{top_trades_path} is missing columns: {sorted(missing)}")

    top = trades.sort_values("final_return_pct", ascending=False).head(top_n).copy()
    top["top_rank"] = np.arange(1, len(top) + 1)
    top["Top8"] = "Top8 #" + top["top_rank"].astype(str)
    top["Trade"] = "T" + top["trade_id"].astype(str)

    if top8_label_path is not None and top8_label_path.exists():
        labels = pd.read_csv(top8_label_path)
        if {"Top8", "Trade"}.issubset(labels.columns):
            labels = labels[["Top8", "Trade"]].copy()
            label_map = dict(zip(labels["Trade"], labels["Top8"], strict=True))
            top["Top8"] = top["Trade"].map(label_map).fillna(top["Top8"])

    top["entry_ts"] = pd.to_datetime(top["entry_date"])
    top["exit_ts"] = pd.to_datetime(top["exit_date"])
    top = top.sort_values("entry_ts").reset_index(drop=True)
    return top


def summary_stats(values: pd.Series, sample: str, trades_included: str) -> dict[str, Any]:
    """Return percent-based distribution summary for a distance series."""
    clean = values.dropna().astype(float)
    if clean.empty:
        raise ValueError(f"No observations available for {sample}")
    pct = clean * 100.0
    return {
        "sample": sample,
        "trades_included": trades_included,
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


def trade_observations(trades: pd.DataFrame, distance: pd.Series) -> pd.DataFrame:
    """Collect hourly distance observations for each trade window."""
    rows: list[pd.DataFrame] = []
    for trade in trades.itertuples(index=False):
        window = distance.loc[
            (distance.index >= trade.entry_ts) & (distance.index <= trade.exit_ts)
        ].dropna()
        if window.empty:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "Top8": trade.Top8,
                    "trade_id": int(trade.trade_id),
                    "timestamp": window.index,
                    "qqq_distance_to_200ma": window.values,
                }
            )
        )
    if not rows:
        raise ValueError("No Top-8 trade observations overlap the QQQ distance series")
    return pd.concat(rows, ignore_index=True)


def by_trade_summary(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize distance distribution separately for each trade."""
    rows: list[dict[str, Any]] = []
    for (top8, trade_id), group in observations.groupby(["Top8", "trade_id"], sort=False):
        stats = summary_stats(
            group["qqq_distance_to_200ma"],
            sample=str(top8),
            trades_included=f"T{int(trade_id)}",
        )
        rows.append(
            {
                "Top8": top8,
                "trade_id": int(trade_id),
                "observations": stats["observations"],
                "mean_distance_pct": stats["mean_distance_pct"],
                "median_distance_pct": stats["median_distance_pct"],
                "p10_distance_pct": stats["p10_distance_pct"],
                "p25_distance_pct": stats["p25_distance_pct"],
                "p75_distance_pct": stats["p75_distance_pct"],
                "p90_distance_pct": stats["p90_distance_pct"],
                "min_distance_pct": stats["min_distance_pct"],
                "max_distance_pct": stats["max_distance_pct"],
                "pct_below_200ma": stats["pct_below_200ma"],
                "pct_within_0_5pct_above": stats["pct_0_to_5pct_above"],
                "pct_above_10pct": stats["pct_10_to_20pct_above"]
                + stats["pct_20_to_30pct_above"]
                + stats["pct_above_30pct"],
                "pct_above_20pct": stats["pct_20_to_30pct_above"] + stats["pct_above_30pct"],
            }
        )
    return pd.DataFrame(rows)


def make_plot(
    observations: pd.DataFrame,
    by_trade: pd.DataFrame,
    output_path: Path,
    *,
    ma_days: int,
) -> None:
    """Create histogram/ECDF/box/bucket distribution plot."""
    distance_pct = observations["qqq_distance_to_200ma"].astype(float) * 100.0
    labels = [f"T{int(row.trade_id)}\n{row.Top8.replace('Top8 ', '')}" for row in by_trade.itertuples()]
    trade_values = [
        observations.loc[observations["trade_id"] == trade_id, "qqq_distance_to_200ma"].values
        * 100.0
        for trade_id in by_trade["trade_id"]
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0))
    ax_hist, ax_ecdf, ax_box, ax_bucket = axes.ravel()

    ax_hist.hist(distance_pct, bins=45, color="#4c78a8", edgecolor="white", alpha=0.85)
    ax_hist.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--", label="200MA")
    ax_hist.axvline(distance_pct.mean(), color="#d62728", linewidth=1.5, label="mean")
    ax_hist.axvline(distance_pct.median(), color="#ff7f0e", linewidth=1.5, label="median")
    ax_hist.set_title("Pooled hourly-bar distribution")
    ax_hist.set_xlabel(f"QQQ distance from {ma_days}-day MA (%)")
    ax_hist.set_ylabel("Hourly bars")
    ax_hist.grid(True, alpha=0.25)
    ax_hist.legend(fontsize=9)

    sorted_values = np.sort(distance_pct.to_numpy())
    ecdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax_ecdf.plot(sorted_values, ecdf, color="#4c78a8", linewidth=1.6)
    ax_ecdf.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    for pctile in [10, 50, 90]:
        value = float(np.percentile(sorted_values, pctile))
        ax_ecdf.axvline(value, color="#999999", linewidth=0.8, linestyle=":")
        ax_ecdf.text(value, 0.05 + pctile / 120.0, f"p{pctile}={value:.1f}%", rotation=90)
    ax_ecdf.set_title("Empirical CDF")
    ax_ecdf.set_xlabel(f"QQQ distance from {ma_days}-day MA (%)")
    ax_ecdf.set_ylabel("Cumulative share")
    ax_ecdf.grid(True, alpha=0.25)

    box = ax_box.boxplot(trade_values, patch_artist=True, showfliers=False)
    ax_box.set_xticks(range(1, len(labels) + 1))
    ax_box.set_xticklabels(labels)
    for patch, trade_id in zip(box["boxes"], by_trade["trade_id"], strict=True):
        patch.set_facecolor("#f58518" if int(trade_id) == 55 else "#72b7b2")
        patch.set_alpha(0.70)
    ax_box.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    ax_box.set_title("Distribution by Top-8 trade; T55 highlighted as current/latest")
    ax_box.set_ylabel(f"QQQ distance from {ma_days}-day MA (%)")
    ax_box.grid(True, axis="y", alpha=0.25)

    bins = [-np.inf, 0, 5, 10, 20, 30, np.inf]
    bucket_labels = ["<0%", "0-5%", "5-10%", "10-20%", "20-30%", ">=30%"]
    bucket = pd.cut(distance_pct, bins=bins, labels=bucket_labels, right=False)
    shares = bucket.value_counts(sort=False) / len(bucket) * 100.0
    colors = ["#d62728", "#ffbb78", "#f2cf5b", "#72b7b2", "#4c78a8", "#1f4e79"]
    ax_bucket.bar(bucket_labels, shares.values, color=colors)
    for i, value in enumerate(shares.values):
        ax_bucket.text(i, value + 0.7, f"{value:.1f}%", ha="center", fontsize=9)
    ax_bucket.set_ylim(0, max(50, float(shares.max()) + 8))
    ax_bucket.set_title("Pooled distance buckets")
    ax_bucket.set_ylabel("Share of hourly bars")
    ax_bucket.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "QQQ distance from hourly 200-day MA during Top-8 winning trades "
        "(including current/latest T55)",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    qqq_path = Path(args.qqq_path)
    top_trades_path = Path(args.top_trades_path)
    top8_label_path = Path(args.top8_label_path) if args.top8_label_path else None
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    qqq = load_hourly_close(qqq_path)
    ma_bars = args.ma_days * args.bars_per_day
    qqq_ma = qqq.rolling(ma_bars, min_periods=ma_bars).mean()
    qqq_distance = (qqq / qqq_ma - 1.0).rename("qqq_distance_to_200ma")

    top_trades = load_top_trades(top_trades_path, top8_label_path, args.top_n)
    observations = trade_observations(top_trades, qqq_distance)
    trades_included = ",".join(f"T{int(x)}" for x in observations["trade_id"].drop_duplicates())
    overall_summary = pd.DataFrame(
        [
            summary_stats(
                observations["qqq_distance_to_200ma"],
                sample=f"Top{args.top_n}_including_current_all_hourly_bars",
                trades_included=trades_included,
            )
        ]
    )
    trade_summary = by_trade_summary(observations)

    observations.to_csv(tables_dir / f"{args.output_prefix}_bar_observations.csv", index=False)
    overall_summary.to_csv(tables_dir / f"{args.output_prefix}_distribution_summary.csv", index=False)
    trade_summary.to_csv(tables_dir / f"{args.output_prefix}_by_trade_summary.csv", index=False)
    make_plot(
        observations,
        trade_summary,
        figures_dir / f"{args.output_prefix}_distribution.png",
        ma_days=args.ma_days,
    )

    row = overall_summary.iloc[0]
    print("Top-8 QQQ distance-to-200MA distribution, including current/latest trade")
    print(f"Trades: {row['trades_included']}")
    print(f"Observations: {int(row['observations']):,}")
    print(
        "Mean / median / p10 / p90: "
        f"{row['mean_distance_pct']:.2f}% / {row['median_distance_pct']:.2f}% / "
        f"{row['p10_distance_pct']:.2f}% / {row['p90_distance_pct']:.2f}%"
    )
    print(
        "Below / 0-5 / 5-10 / 10-20 / 20-30 / >=30: "
        f"{row['pct_below_200ma']:.2f}% / {row['pct_0_to_5pct_above']:.2f}% / "
        f"{row['pct_5_to_10pct_above']:.2f}% / {row['pct_10_to_20pct_above']:.2f}% / "
        f"{row['pct_20_to_30pct_above']:.2f}% / {row['pct_above_30pct']:.2f}%"
    )
    print(f"Saved figure: {figures_dir / f'{args.output_prefix}_distribution.png'}")


if __name__ == "__main__":
    main()
