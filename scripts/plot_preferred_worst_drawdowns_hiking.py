#!/usr/bin/env python
"""Plot worst drawdowns for the current preferred hourly-200MA-gate strategy.

The preferred strategy is the new candidate promoted on 2026-06-02:
- QQQ signal source.
- Synthetic TQQQ exposure.
- No daily regime gate.
- Entry requires QQQ MACD histogram > 0 and QQQ hourly close > QQQ hourly 200-day MA.
- Exit when QQQ hourly close < QQQ hourly 200-day MA.
- Out-of-market cash return is included in the input return stream.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.offsets import BDay

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.utils import ensure_directory  # noqa: E402

PREFERRED_COLUMN = "new_candidate_no_daily_gate__qqq_hourly_200ma_entry_exit"
PREFERRED_LABEL = "Preferred: QQQ hourly 200MA gate"

# Effective-date hiking cycles used in prior studies. 2022-2023 latest hike
# ended with the July 26/27, 2023 increase; the Fed then cut rates beginning in
# September 2024, so this analysis has no post-2023 hiking cycle.
EFFECTIVE_HIKING_CYCLES: tuple[tuple[str, str, str], ...] = (
    ("1999-2000 hike cycle partial", "1999-06-30", "2000-05-16"),
    ("2004-2006 hike cycle", "2004-06-30", "2006-06-29"),
    ("2015-2018 normalization/hike cycle", "2015-12-17", "2018-12-20"),
    ("2022-2023 hike cycle", "2022-03-17", "2023-07-27"),
)

ANNOUNCED_HIKING_CYCLES: tuple[tuple[str, str, str], ...] = (
    ("2004-2006 announced hike window", "2004-05-04", "2006-06-29"),
    ("2015-2018 announced hike window", "2015-10-28", "2018-12-20"),
    ("2022-2023 announced hike window", "2022-01-26", "2023-07-27"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--returns-file",
        default="reports/tables/tqqq_cash_yield_candidate_comparison_after_tax_returns_with_cash_yield.csv",
    )
    parser.add_argument(
        "--weights-file",
        default="reports/tables/tqqq_cash_yield_candidate_comparison_weights.csv",
    )
    parser.add_argument("--target-price-file", default="data/raw/synthetic_3x_60min/QQQ_3X_CALC.parquet")
    parser.add_argument("--qqq-price-file", default="data/raw/alpha_vantage_60min/QQQ.parquet")
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--context-days", type=int, default=42)
    parser.add_argument("--output-prefix", default="preferred_hourly_200ma_worst6_drawdowns")
    return parser.parse_args()


def _load_series_from_csv(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if column not in frame.columns:
        raise ValueError(f"Column {column!r} not found in {path}")
    return frame[column].astype(float).rename(column)


def _equity(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def find_drawdown_episodes(returns: pd.Series) -> pd.DataFrame:
    equity = _equity(returns)
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    peak_times = []
    last_peak_time = equity.index[0]
    max_so_far = float(equity.iloc[0])
    for timestamp, value in equity.items():
        if value >= max_so_far - 1e-12:
            max_so_far = float(value)
            last_peak_time = timestamp
        peak_times.append(last_peak_time)
    peak_time_series = pd.Series(peak_times, index=equity.index)

    episodes: list[dict[str, Any]] = []
    in_episode = False
    peak_time = None
    start = None
    trough = None
    trough_dd = 0.0

    for timestamp, dd_value in drawdown.items():
        if not in_episode and dd_value < -1e-12:
            in_episode = True
            peak_time = pd.Timestamp(peak_time_series.loc[timestamp])
            start = timestamp
            trough = timestamp
            trough_dd = float(dd_value)
        elif in_episode:
            if dd_value < trough_dd:
                trough = timestamp
                trough_dd = float(dd_value)
            if dd_value >= -1e-12:
                episodes.append(
                    {
                        "peak": peak_time,
                        "start": start,
                        "trough": trough,
                        "recovery": timestamp,
                        "max_drawdown": trough_dd,
                        "duration_calendar_days_peak_to_trough": (
                            pd.Timestamp(trough) - pd.Timestamp(peak_time)
                        ).days,
                        "duration_calendar_days_peak_to_recovery": (
                            pd.Timestamp(timestamp) - pd.Timestamp(peak_time)
                        ).days,
                    }
                )
                in_episode = False
                peak_time = None
                start = None
                trough = None
                trough_dd = 0.0
    if in_episode:
        episodes.append(
            {
                "peak": peak_time,
                "start": start,
                "trough": trough,
                "recovery": pd.NaT,
                "max_drawdown": trough_dd,
                "duration_calendar_days_peak_to_trough": (
                    pd.Timestamp(trough) - pd.Timestamp(peak_time)
                ).days,
                "duration_calendar_days_peak_to_recovery": pd.NA,
            }
        )
    return pd.DataFrame(episodes).sort_values("max_drawdown").reset_index(drop=True)


def _overlap_detail(
    peak: pd.Timestamp,
    trough: pd.Timestamp,
    cycles: tuple[tuple[str, str, str], ...],
) -> tuple[bool, bool, str, int]:
    names: list[str] = []
    bottom_in_cycle = False
    overlap_days = 0
    for name, start, end in cycles:
        cycle_start = pd.Timestamp(start)
        cycle_end = pd.Timestamp(end)
        overlap_start = max(pd.Timestamp(peak).normalize(), cycle_start)
        overlap_end = min(pd.Timestamp(trough).normalize(), cycle_end)
        if overlap_start <= overlap_end:
            names.append(name)
            overlap_days += int((overlap_end - overlap_start).days) + 1
        bottom_in_cycle = bottom_in_cycle or (cycle_start <= pd.Timestamp(trough).normalize() <= cycle_end)
    return bool(names), bottom_in_cycle, "; ".join(names), overlap_days


def classify_hiking_relationship(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank, row in episodes.iterrows():
        peak = pd.Timestamp(row["peak"])
        trough = pd.Timestamp(row["trough"])
        eff_overlap, eff_bottom, eff_names, eff_days = _overlap_detail(
            peak,
            trough,
            EFFECTIVE_HIKING_CYCLES,
        )
        ann_overlap, ann_bottom, ann_names, ann_days = _overlap_detail(
            peak,
            trough,
            ANNOUNCED_HIKING_CYCLES,
        )
        rows.append(
            {
                "rank": int(rank) + 1,
                **row.to_dict(),
                "overlaps_effective_hiking_cycle": eff_overlap,
                "bottom_in_effective_hiking_cycle": eff_bottom,
                "matching_effective_hiking_cycle": eff_names,
                "effective_hiking_overlap_days_peak_to_trough": eff_days,
                "overlaps_announced_hiking_window": ann_overlap,
                "bottom_in_announced_hiking_window": ann_bottom,
                "matching_announced_hiking_window": ann_names,
                "announced_hiking_overlap_days_peak_to_trough": ann_days,
            }
        )
    return pd.DataFrame(rows)


def _slice_window(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, context_days: int) -> pd.DatetimeIndex:
    left = pd.Timestamp(start) - BDay(context_days)
    right_base = pd.Timestamp(end) if pd.notna(end) else index[-1]
    right = right_base + BDay(context_days)
    return index[(index >= left) & (index <= right)]


def _shade_cycles(ax: plt.Axes, start: pd.Timestamp, end: pd.Timestamp) -> None:
    for name, cycle_start, cycle_end in EFFECTIVE_HIKING_CYCLES:
        cs = pd.Timestamp(cycle_start)
        ce = pd.Timestamp(cycle_end)
        if ce < start or cs > end:
            continue
        ax.axvspan(max(cs, start), min(ce, end), color="#f4a261", alpha=0.18, label="Fed hiking cycle")
        mid = max(cs, start) + (min(ce, end) - max(cs, start)) / 2
        ax.text(mid, 0.98, name.split(" hike")[0], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color="#9a4f00")


def make_summary_plot(
    *,
    returns: pd.Series,
    weights: pd.Series,
    target_price: pd.Series,
    qqq_price: pd.Series,
    analysis: pd.DataFrame,
    output_path: Path,
    context_days: int,
) -> None:
    equity = _equity(returns)
    drawdown = _drawdown(equity)
    fig, axes = plt.subplots(len(analysis), 1, figsize=(15, 3.0 * len(analysis)), sharex=False)
    if len(analysis) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, analysis.iterrows(), strict=False):
        peak = pd.Timestamp(row["peak"])
        trough = pd.Timestamp(row["trough"])
        recovery = pd.Timestamp(row["recovery"]) if pd.notna(row["recovery"]) else returns.index[-1]
        window_index = _slice_window(returns.index, peak, recovery, context_days)
        if window_index.empty:
            continue
        start = window_index[0]
        end = window_index[-1]
        _shade_cycles(ax, start, end)

        eq_window = equity.reindex(window_index)
        eq_norm = eq_window / eq_window.iloc[0]
        qqq_norm = qqq_price.reindex(window_index).ffill() / qqq_price.reindex(window_index).ffill().iloc[0]
        tqqq_norm = target_price.reindex(window_index).ffill() / target_price.reindex(window_index).ffill().iloc[0]
        eq_norm.plot(ax=ax, color="#005f73", linewidth=1.35, label="strategy equity")
        qqq_norm.plot(ax=ax, color="#6c757d", linewidth=0.95, alpha=0.8, label="QQQ price norm")
        tqqq_norm.plot(ax=ax, color="#94d2bd", linewidth=0.8, alpha=0.8, label="synthetic TQQQ price norm")
        ax2 = ax.twinx()
        drawdown.reindex(window_index).plot(ax=ax2, color="#c1121f", linewidth=1.0, alpha=0.75, label="strategy DD")
        ax2.set_ylabel("Drawdown")
        ax2.set_ylim(min(-0.65, float(drawdown.reindex(window_index).min()) * 1.1), 0.05)

        for when, label, color in [
            (peak, "peak", "#005f73"),
            (trough, "trough", "#c1121f"),
            (recovery, "recovery", "#2a9d8f"),
        ]:
            if start <= when <= end:
                ax.axvline(when, color=color, linestyle="--", linewidth=1.0, alpha=0.9)
                ax.text(when, 0.02, label, transform=ax.get_xaxis_transform(), rotation=90, va="bottom", ha="right", fontsize=7, color=color)

        # Buy/sell markers from executable weight changes.
        weight_window = weights.reindex(window_index).fillna(0.0)
        changes = weight_window.diff().fillna(weight_window).abs().gt(1e-12)
        buy_times = weight_window.index[changes & weight_window.gt(weight_window.shift(1).fillna(0.0))]
        sell_times = weight_window.index[changes & weight_window.lt(weight_window.shift(1).fillna(0.0))]
        y_min, y_max = ax.get_ylim()
        ax.scatter(buy_times, [y_min + 0.06 * (y_max - y_min)] * len(buy_times), marker="^", s=18, color="#2a9d8f", label="buy")
        ax.scatter(sell_times, [y_min + 0.12 * (y_max - y_min)] * len(sell_times), marker="v", s=18, color="#e76f51", label="sell")

        relation = (
            row["matching_effective_hiking_cycle"]
            if bool(row["overlaps_effective_hiking_cycle"])
            else "no effective hike-cycle overlap"
        )
        ax.set_title(
            f"Rank {int(row['rank'])}: {peak:%Y-%m-%d} to {trough:%Y-%m-%d}, "
            f"DD {row['max_drawdown']:.1%}; {relation}",
            fontsize=10,
        )
        ax.set_ylabel("Normalized value")
        ax.grid(True, alpha=0.25)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Deduplicate labels, especially the repeated hiking-cycle shade.
        items = list(dict(zip(labels + labels2, lines + lines2, strict=False)).items())
        ax.legend([line for _, line in items], [label for label, _ in items], fontsize=7, loc="upper left")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.suptitle("Worst drawdowns for preferred QQQ hourly-200MA-gate strategy", fontsize=14)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_individual_plots(
    *,
    returns: pd.Series,
    weights: pd.Series,
    target_price: pd.Series,
    qqq_price: pd.Series,
    analysis: pd.DataFrame,
    output_dir: Path,
    output_prefix: str,
    context_days: int,
) -> None:
    ensure_directory(output_dir)
    for _, row in analysis.iterrows():
        rank = int(row["rank"])
        output_path = output_dir / f"{output_prefix}_rank{rank}.png"
        make_summary_plot(
            returns=returns,
            weights=weights,
            target_price=target_price,
            qqq_price=qqq_price,
            analysis=pd.DataFrame([row]),
            output_path=output_path,
            context_days=context_days,
        )


def main() -> None:
    args = parse_args()
    returns = _load_series_from_csv(Path(args.returns_file), PREFERRED_COLUMN)
    weights = _load_series_from_csv(Path(args.weights_file), PREFERRED_COLUMN)
    target_price = read_price_file(Path(args.target_price_file)).sort_index()["adj_close"].astype(float)
    qqq_price = read_price_file(Path(args.qqq_price_file)).sort_index()["adj_close"].astype(float)
    common = returns.index.intersection(weights.index)
    returns = returns.loc[common]
    weights = weights.loc[common]
    target_price = target_price.reindex(common).ffill()
    qqq_price = qqq_price.reindex(common).ffill()

    episodes = find_drawdown_episodes(returns).head(args.top_n)
    analysis = classify_hiking_relationship(episodes)

    tables_dir = Path("reports/tables")
    figures_dir = Path("reports/figures")
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)
    table_path = tables_dir / f"{args.output_prefix}_hiking_analysis.csv"
    summary_plot_path = figures_dir / f"{args.output_prefix}_hiking_analysis.png"
    individual_dir = figures_dir / args.output_prefix
    analysis.to_csv(table_path, index=False)
    make_summary_plot(
        returns=returns,
        weights=weights,
        target_price=target_price,
        qqq_price=qqq_price,
        analysis=analysis,
        output_path=summary_plot_path,
        context_days=args.context_days,
    )
    make_individual_plots(
        returns=returns,
        weights=weights,
        target_price=target_price,
        qqq_price=qqq_price,
        analysis=analysis,
        output_dir=individual_dir,
        output_prefix=args.output_prefix,
        context_days=args.context_days,
    )

    print(f"Worst drawdown hiking analysis saved to {table_path}")
    print(f"Summary plot saved to {summary_plot_path}")
    print(f"Individual plots saved to {individual_dir}")
    cols = [
        "rank",
        "peak",
        "trough",
        "recovery",
        "max_drawdown",
        "overlaps_effective_hiking_cycle",
        "bottom_in_effective_hiking_cycle",
        "matching_effective_hiking_cycle",
        "overlaps_announced_hiking_window",
        "matching_announced_hiking_window",
    ]
    print(analysis[cols].to_string(index=False))


if __name__ == "__main__":
    main()
