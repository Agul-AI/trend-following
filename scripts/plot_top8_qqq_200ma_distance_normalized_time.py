#!/usr/bin/env python
"""Plot QQQ/200MA distance over normalized time for completed Top-8 winners.

The x-axis is normalized trade progress: 0% is the first hourly bar in that
round trip and 100% is the final hourly bar in that round trip. This uses
trading-time bar order rather than calendar-time elapsed days, so all seven
completed Top-8 trades can be overlaid on the same 0-100% axis.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations-path",
        default="reports/tables/preferred_top8_ex_current_qqq_200ma_distance_bar_observations.csv",
        help="Hourly QQQ distance observations for Top8 trades excluding current/latest.",
    )
    parser.add_argument(
        "--top8-path",
        default="reports/tables/preferred_top8_winning_trades_time_order.csv",
        help="Top8 label table; T55/current is excluded.",
    )
    parser.add_argument("--grid-points", type=int, default=501)
    parser.add_argument(
        "--output-prefix",
        default="preferred_top8_ex_current_qqq_200ma_distance_normalized_time",
    )
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--tables-dir", default="reports/tables")
    return parser.parse_args()


def load_trade_label_order(path: Path) -> list[tuple[str, int]]:
    """Return completed Top8 trades in chronological order, excluding T55."""
    frame = pd.read_csv(path)
    if {"Top8", "Trade"}.difference(frame.columns):
        raise ValueError(f"{path} must contain Top8 and Trade columns")
    frame = frame[frame["Trade"] != "T55"].copy()
    frame["trade_id"] = frame["Trade"].str.replace("T", "", regex=False).astype(int)
    return [(row.Top8, int(row.trade_id)) for row in frame.itertuples(index=False)]


def normalize_observations(
    observations: pd.DataFrame,
    order: list[tuple[str, int]],
    *,
    grid_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize each trade to a 0-100% x-axis and interpolate to a common grid."""
    observations = observations.copy()
    observations["timestamp"] = pd.to_datetime(observations["timestamp"])
    observations["distance_pct"] = observations["qqq_distance_to_200ma"].astype(float) * 100.0

    grid = np.linspace(0.0, 100.0, grid_points)
    normalized_rows: list[pd.DataFrame] = []
    interpolated: dict[str, np.ndarray] = {}

    for top8, trade_id in order:
        group = observations.loc[observations["trade_id"] == trade_id].sort_values("timestamp").copy()
        if group.empty:
            continue
        n = len(group)
        if n == 1:
            progress = np.array([0.0])
        else:
            progress = np.linspace(0.0, 100.0, n)
        group["normalized_trade_progress_pct"] = progress
        group["Top8"] = top8
        group["Trade"] = f"T{trade_id}"
        normalized_rows.append(
            group[
                [
                    "Top8",
                    "Trade",
                    "trade_id",
                    "timestamp",
                    "normalized_trade_progress_pct",
                    "distance_pct",
                ]
            ]
        )
        interpolated[f"T{trade_id} {top8.replace('Top8 ', '')}"] = np.interp(
            grid, progress, group["distance_pct"].to_numpy()
        )

    if not normalized_rows:
        raise ValueError("No observations available after applying Top8 order")

    normalized = pd.concat(normalized_rows, ignore_index=True)
    grid_frame = pd.DataFrame({"normalized_trade_progress_pct": grid, **interpolated})
    values = grid_frame.drop(columns=["normalized_trade_progress_pct"])
    grid_frame["median_distance_pct"] = values.median(axis=1)
    grid_frame["p25_distance_pct"] = values.quantile(0.25, axis=1)
    grid_frame["p75_distance_pct"] = values.quantile(0.75, axis=1)
    grid_frame["p10_distance_pct"] = values.quantile(0.10, axis=1)
    grid_frame["p90_distance_pct"] = values.quantile(0.90, axis=1)
    return normalized, grid_frame


def make_plot(grid_frame: pd.DataFrame, output_path: Path) -> None:
    """Create one combined normalized-time plot."""
    x = grid_frame["normalized_trade_progress_pct"].to_numpy()
    trade_columns = [
        column
        for column in grid_frame.columns
        if column.startswith("T") and "distance" not in column
    ]

    fig, ax = plt.subplots(figsize=(15, 7.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trade_columns)))
    for color, column in zip(colors, trade_columns, strict=True):
        ax.plot(
            x,
            grid_frame[column],
            label=column,
            linewidth=1.25,
            alpha=0.80,
            color=color,
        )

    ax.fill_between(
        x,
        grid_frame["p25_distance_pct"],
        grid_frame["p75_distance_pct"],
        color="#808080",
        alpha=0.16,
        label="25th-75th percentile band",
    )
    ax.plot(
        x,
        grid_frame["median_distance_pct"],
        color="black",
        linewidth=2.4,
        label="Cross-trade median",
    )
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, label="QQQ 200MA")
    ax.axhline(5.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.axhline(10.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.axhline(20.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.text(100.8, 5.0, "+5%", va="center", fontsize=9, color="#666666")
    ax.text(100.8, 10.0, "+10%", va="center", fontsize=9, color="#666666")
    ax.text(100.8, 20.0, "+20%", va="center", fontsize=9, color="#666666")
    ax.set_xlim(0, 104)
    ax.set_xlabel("Normalized trade progress: entry = 0%, exit = 100%")
    ax.set_ylabel("QQQ distance from hourly 200-day MA (%)")
    ax.set_title("QQQ distance from hourly 200-day MA across completed Top-8 winning trades")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9, loc="upper left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_summary(normalized: pd.DataFrame) -> pd.DataFrame:
    """Create a small per-trade normalized-time summary."""
    rows = []
    for (top8, trade, trade_id), group in normalized.groupby(["Top8", "Trade", "trade_id"], sort=False):
        distance = group["distance_pct"].astype(float)
        rows.append(
            {
                "Top8": top8,
                "Trade": trade,
                "trade_id": trade_id,
                "observations": int(distance.size),
                "start_distance_pct": float(distance.iloc[0]),
                "end_distance_pct": float(distance.iloc[-1]),
                "mean_distance_pct": float(distance.mean()),
                "median_distance_pct": float(distance.median()),
                "max_distance_pct": float(distance.max()),
                "min_distance_pct": float(distance.min()),
                "pct_time_above_10pct": float((distance >= 10.0).mean() * 100.0),
                "pct_time_below_200ma": float((distance < 0.0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    observations_path = Path(args.observations_path)
    top8_path = Path(args.top8_path)
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    observations = pd.read_csv(observations_path)
    order = load_trade_label_order(top8_path)
    normalized, grid_frame = normalize_observations(
        observations,
        order,
        grid_points=args.grid_points,
    )
    summary = make_summary(normalized)

    figure_path = figures_dir / f"{args.output_prefix}.png"
    normalized_path = tables_dir / f"{args.output_prefix}_observations.csv"
    grid_path = tables_dir / f"{args.output_prefix}_grid.csv"
    summary_path = tables_dir / f"{args.output_prefix}_summary.csv"

    make_plot(grid_frame, figure_path)
    normalized.to_csv(normalized_path, index=False)
    grid_frame.to_csv(grid_path, index=False)
    summary.to_csv(summary_path, index=False)

    median_start = float(grid_frame["median_distance_pct"].iloc[0])
    median_mid = float(grid_frame.loc[grid_frame["normalized_trade_progress_pct"].sub(50).abs().idxmin(), "median_distance_pct"])
    median_end = float(grid_frame["median_distance_pct"].iloc[-1])

    print("Saved normalized-time Top8 distance plot:")
    print(f"  {figure_path}")
    print("Saved tables:")
    print(f"  {summary_path}")
    print(f"  {grid_path}")
    print(f"  {normalized_path}")
    print(
        "Cross-trade median distance at start / midpoint / end: "
        f"{median_start:.2f}% / {median_mid:.2f}% / {median_end:.2f}%"
    )


if __name__ == "__main__":
    main()
