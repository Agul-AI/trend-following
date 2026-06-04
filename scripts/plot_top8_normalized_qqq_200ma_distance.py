#!/usr/bin/env python
"""Plot Top-8 winning trades on a normalized QQQ/200MA-distance timeline.

Each Top-8 trade is rescaled to the same x-axis:

    x = (timestamp - entry_time) / (exit_time - entry_time)

For the current/latest trade, the normalization endpoint is "today" by
default, rather than the last saved trade-table timestamp. The line itself only
uses locally available price bars.

Star markers are placed at the synthetic-Q_3X trade-price peak for the 7
completed Top-8 trades. Diamond markers label the first bar where a trade's
synthetic-Q_3X unrealized gain reaches +100%. The current/latest trade is
intentionally not peak-marked because its final peak is not known yet.
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
    parser.add_argument(
        "--current-end",
        default="today",
        help="Normalization endpoint for the current trade. Use 'today' or a timestamp.",
    )
    parser.add_argument("--bars-per-day", type=int, default=6)
    parser.add_argument("--ma-days", type=int, default=200)
    parser.add_argument(
        "--output-prefix",
        default="preferred_top8_normalized_qqq_200ma_distance",
    )
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--tables-dir", default="reports/tables")
    return parser.parse_args()


def load_close(path: Path, name: str) -> pd.Series:
    """Load a close/adjusted-close series from a local parquet file."""
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
    return close.rename(name)


def today_market_close_like(reference_index: pd.DatetimeIndex) -> pd.Timestamp:
    """Return today's timestamp using the latest bar time-of-day as convention."""
    now = pd.Timestamp.now()
    latest_time = reference_index.max().time()
    return pd.Timestamp.combine(now.date(), latest_time)


def parse_current_end(value: str, reference_index: pd.DatetimeIndex) -> pd.Timestamp:
    """Parse current-trade normalization endpoint."""
    if value.lower() == "today":
        return today_market_close_like(reference_index)
    return pd.Timestamp(value)


def load_top_trades(
    top_trades_path: Path,
    top8_label_path: Path,
    *,
    top_n: int,
    current_trade_id: int,
    current_end: pd.Timestamp,
) -> pd.DataFrame:
    """Load Top-N winners, attach labels, and set current-trade end to today."""
    trades = pd.read_csv(top_trades_path)
    required = {"trade_id", "entry_date", "exit_date", "final_return_pct", "peak_return_pct"}
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"{top_trades_path} is missing columns: {sorted(missing)}")

    top = trades.sort_values("final_return_pct", ascending=False).head(top_n).copy()
    top["top_rank"] = np.arange(1, len(top) + 1)
    top["Top8"] = "Top8 #" + top["top_rank"].astype(str)
    top["Trade"] = "T" + top["trade_id"].astype(str)

    if top8_label_path.exists():
        labels = pd.read_csv(top8_label_path)
        if {"Top8", "Trade"}.issubset(labels.columns):
            label_map = dict(zip(labels["Trade"], labels["Top8"], strict=True))
            top["Top8"] = top["Trade"].map(label_map).fillna(top["Top8"])

    top["entry_ts"] = pd.to_datetime(top["entry_date"])
    top["original_exit_ts"] = pd.to_datetime(top["exit_date"])
    top["normalization_exit_ts"] = top["original_exit_ts"]
    top.loc[top["trade_id"].eq(current_trade_id), "normalization_exit_ts"] = current_end
    top["is_current"] = top["trade_id"].eq(current_trade_id)
    top = top.sort_values("entry_ts").reset_index(drop=True)
    return top


def normalized_x(index: pd.DatetimeIndex, entry: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Convert timestamps to normalized trade-progress coordinates."""
    denominator = (end - entry).total_seconds()
    if denominator <= 0:
        raise ValueError(f"Normalization end must be after entry: {entry=} {end=}")
    seconds = (index - entry).total_seconds()
    return np.asarray(seconds / denominator, dtype=float)


def build_series_and_markers(
    trades: pd.DataFrame,
    qqq_distance: pd.Series,
    target_price: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build normalized distance rows, peak markers, and first +100% markers."""
    series_rows: list[pd.DataFrame] = []
    marker_rows: list[dict[str, Any]] = []
    first_100_rows: list[dict[str, Any]] = []

    for trade in trades.itertuples(index=False):
        trade_end_for_data = min(trade.normalization_exit_ts, qqq_distance.index.max())
        distance_window = qqq_distance.loc[
            (qqq_distance.index >= trade.entry_ts) & (qqq_distance.index <= trade_end_for_data)
        ].dropna()
        if distance_window.empty:
            continue

        x_values = normalized_x(distance_window.index, trade.entry_ts, trade.normalization_exit_ts)
        series_rows.append(
            pd.DataFrame(
                {
                    "Top8": trade.Top8,
                    "trade_id": int(trade.trade_id),
                    "timestamp": distance_window.index,
                    "normalized_time": x_values,
                    "qqq_distance_to_200ma": distance_window.values,
                    "is_current": bool(trade.is_current),
                }
            )
        )

        target_window_for_gain = target_price.loc[
            (target_price.index >= trade.entry_ts)
            & (target_price.index <= trade_end_for_data)
        ].dropna()
        if not target_window_for_gain.empty:
            entry_price_for_gain = float(target_window_for_gain.iloc[0])
            gain = target_window_for_gain / entry_price_for_gain - 1.0
            first_100 = gain.loc[gain >= 1.0]
            if not first_100.empty:
                first_100_ts = first_100.index[0]
                first_100_distance = qqq_distance.reindex([first_100_ts]).iloc[0]
                if pd.notna(first_100_distance):
                    first_100_rows.append(
                        {
                            "Top8": trade.Top8,
                            "trade_id": int(trade.trade_id),
                            "first_100_gain_timestamp": first_100_ts,
                            "normalized_time": float(
                                normalized_x(
                                    pd.DatetimeIndex([first_100_ts]),
                                    trade.entry_ts,
                                    trade.normalization_exit_ts,
                                )[0]
                            ),
                            "qqq_distance_to_200ma_at_first_100_gain": float(first_100_distance),
                            "synthetic_3x_gain_pct": float(gain.loc[first_100_ts]) * 100.0,
                            "entry_timestamp": trade.entry_ts,
                            "normalization_exit_timestamp": trade.normalization_exit_ts,
                            "is_current": bool(trade.is_current),
                        }
                    )

        if not bool(trade.is_current):
            target_window = target_price.loc[
                (target_price.index >= trade.entry_ts)
                & (target_price.index <= trade.original_exit_ts)
            ].dropna()
            if target_window.empty:
                continue
            peak_ts = target_window.idxmax()
            entry_price = float(target_window.iloc[0])
            peak_price = float(target_window.loc[peak_ts])
            marker_distance = qqq_distance.reindex([peak_ts]).iloc[0]
            marker_rows.append(
                {
                    "Top8": trade.Top8,
                    "trade_id": int(trade.trade_id),
                    "peak_timestamp": peak_ts,
                    "normalized_time": float(
                        normalized_x(pd.DatetimeIndex([peak_ts]), trade.entry_ts, trade.normalization_exit_ts)[
                            0
                        ]
                    ),
                    "qqq_distance_to_200ma_at_peak": float(marker_distance),
                    "synthetic_3x_peak_from_entry_pct": (peak_price / entry_price - 1.0) * 100.0,
                    "entry_timestamp": trade.entry_ts,
                    "exit_timestamp": trade.original_exit_ts,
                }
            )

    if not series_rows:
        raise ValueError("No normalized series observations were generated")
    return (
        pd.concat(series_rows, ignore_index=True),
        pd.DataFrame(marker_rows),
        pd.DataFrame(first_100_rows),
    )


def make_plot(
    series: pd.DataFrame,
    markers: pd.DataFrame,
    first_100_markers: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create the combined normalized timeline plot."""
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    completed = series.loc[~series["is_current"]]
    current = series.loc[series["is_current"]]

    completed_trade_ids = list(dict.fromkeys(completed["trade_id"].tolist()))
    cmap = plt.get_cmap("tab10")
    color_by_trade = {
        trade_id: cmap(i % 10) for i, trade_id in enumerate(completed_trade_ids)
    }

    for trade_id, group in completed.groupby("trade_id", sort=False):
        label = f"T{int(trade_id)}"
        top8 = str(group["Top8"].iloc[0]).replace("Top8 ", "")
        ax.plot(
            group["normalized_time"],
            group["qqq_distance_to_200ma"] * 100.0,
            color=color_by_trade[trade_id],
            linewidth=1.55,
            alpha=0.85,
            label=f"{label} ({top8})",
        )

    if not current.empty:
        trade_id = int(current["trade_id"].iloc[0])
        top8 = str(current["Top8"].iloc[0]).replace("Top8 ", "")
        ax.plot(
            current["normalized_time"],
            current["qqq_distance_to_200ma"] * 100.0,
            color="black",
            linewidth=2.4,
            alpha=0.95,
            label=f"T{trade_id} ({top8}, current)",
            linestyle="-",
        )

    for row in markers.itertuples(index=False):
        marker_color = color_by_trade.get(int(row.trade_id), "black")
        y_value = float(row.qqq_distance_to_200ma_at_peak) * 100.0
        ax.scatter(
            row.normalized_time,
            y_value,
            marker="*",
            s=180,
            color=marker_color,
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"T{int(row.trade_id)} peak",
            xy=(row.normalized_time, y_value),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )

    for row in first_100_markers.itertuples(index=False):
        marker_color = color_by_trade.get(int(row.trade_id), "black")
        y_value = float(row.qqq_distance_to_200ma_at_first_100_gain) * 100.0
        ax.scatter(
            row.normalized_time,
            y_value,
            marker="D",
            s=70,
            color="white",
            edgecolor=marker_color,
            linewidth=2.0,
            zorder=6,
        )
        ax.annotate(
            f"T{int(row.trade_id)} +100%",
            xy=(row.normalized_time, y_value),
            xytext=(5, -14),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )

    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, label="QQQ 200MA")
    ax.axhline(10.0, color="#999999", linestyle=":", linewidth=0.8)
    ax.axhline(20.0, color="#999999", linestyle=":", linewidth=0.8)
    ax.set_xlim(-0.01, 1.01)
    ax.set_xlabel("Normalized trade time: entry = 0, exit/today = 1")
    ax.set_ylabel("QQQ distance from hourly 200-day MA (%)")
    ax.set_title(
        "Top-8 winning trades: QQQ distance from hourly 200-day MA over normalized time\n"
        "Stars mark completed-trade peaks; diamonds mark first +100% synthetic-Q_3X gain"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    qqq = load_close(Path(args.qqq_path), "QQQ")
    target = load_close(Path(args.target_path), "QQQ_3X_CALC")
    common = qqq.index.intersection(target.index)
    qqq = qqq.loc[common]
    target = target.loc[common]

    ma_bars = args.ma_days * args.bars_per_day
    qqq_ma = qqq.rolling(ma_bars, min_periods=ma_bars).mean()
    qqq_distance = (qqq / qqq_ma - 1.0).rename("qqq_distance_to_200ma")

    current_end = parse_current_end(args.current_end, qqq.index)
    top_trades = load_top_trades(
        Path(args.top_trades_path),
        Path(args.top8_label_path),
        top_n=args.top_n,
        current_trade_id=args.current_trade_id,
        current_end=current_end,
    )
    series, markers, first_100_markers = build_series_and_markers(top_trades, qqq_distance, target)

    series_path = tables_dir / f"{args.output_prefix}_series.csv"
    markers_path = tables_dir / f"{args.output_prefix}_peak_markers.csv"
    first_100_markers_path = tables_dir / f"{args.output_prefix}_first_100_gain_markers.csv"
    plot_path = figures_dir / f"{args.output_prefix}.png"
    series.to_csv(series_path, index=False)
    markers.to_csv(markers_path, index=False)
    first_100_markers.to_csv(first_100_markers_path, index=False)
    make_plot(series, markers, first_100_markers, plot_path)

    latest_current_bar = series.loc[series["is_current"], "timestamp"].max()
    print("Saved normalized Top-8 QQQ/200MA distance plot")
    print(f"Figure: {plot_path}")
    print(f"Series table: {series_path}")
    print(f"Peak-marker table: {markers_path}")
    print(f"First +100% gain-marker table: {first_100_markers_path}")
    print(f"Current-trade normalization end: {current_end}")
    print(f"Latest locally available current-trade bar used: {latest_current_bar}")
    print("Completed-trade peak markers:")
    if markers.empty:
        print("  none")
    else:
        preview = markers[
            [
                "Top8",
                "trade_id",
                "peak_timestamp",
                "normalized_time",
                "qqq_distance_to_200ma_at_peak",
                "synthetic_3x_peak_from_entry_pct",
            ]
        ].copy()
        preview["qqq_distance_to_200ma_at_peak"] *= 100.0
        print(preview.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("First +100% synthetic-3x gain markers:")
    if first_100_markers.empty:
        print("  none")
    else:
        preview_100 = first_100_markers[
            [
                "Top8",
                "trade_id",
                "first_100_gain_timestamp",
                "normalized_time",
                "qqq_distance_to_200ma_at_first_100_gain",
                "synthetic_3x_gain_pct",
            ]
        ].copy()
        preview_100["qqq_distance_to_200ma_at_first_100_gain"] *= 100.0
        print(preview_100.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
