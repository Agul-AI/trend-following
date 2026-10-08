#!/usr/bin/env python
"""Plot sell timing for the four hiking-cycle shared TQQQ drawdowns."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

STRATEGIES: dict[str, dict[str, str]] = {
    "qqq_entry_qqq_exit_no_lock": {
        "column": "entry_qqq__exit_qqq__full_no_lock",
        "label": "QQQ entry + QQQ exit, no lock",
        "short_label": "qqq_entry_qqq_exit_no_lock",
    },
}

HIKING_CYCLES: dict[str, tuple[str, str]] = {
    "1999-2000 hike cycle partial": ("1999-06-30", "2000-05-16"),
    "2004-2006 hike cycle": ("2004-06-30", "2006-06-29"),
    "2015-2018 normalization/hike cycle": ("2015-12-17", "2018-12-20"),
    "2022-2023 hike cycle": ("2022-03-17", "2023-07-27"),
}

FED_HIKE_SIGNAL_EVENTS: dict[str, tuple[str, str, str]] = {
    # Official FOMC statement language:
    # - 2004-05-04: "policy accommodation can be removed..."
    # - 2015-10-28: "whether it will be appropriate to raise ... at its next meeting"
    # - 2022-01-26: "expects it will soon be appropriate to raise..."
    "2004-2006 hike cycle": (
        "2004-05-04",
        "Fed hike signal: measured-pace guidance",
        "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20040504/default.htm",
    ),
    "2015-2018 normalization/hike cycle": (
        "2015-10-28",
        "Fed hike signal: next-meeting language",
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20151028a.htm",
    ),
    "2022-2023 hike cycle": (
        "2022-01-26",
        "Fed hike signal: soon appropriate to raise",
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220126a.htm",
    ),
}

# Monthly NASDAQ-100 trailing P/E values used as a QQQ valuation proxy because
# QQQ tracks the NASDAQ-100. These are the months needed for the synthetic
# TQQQ price-peak/price-bottom dates and strategy peak/bottom dates in this
# analysis.
# Source: Trendonify "Nasdaq 100 PE Ratio" historical-data table.
NASDAQ100_PE_PROXY_BY_MONTH: dict[str, float] = {
    "2003-04": 50.38,
    "2004-01": 46.05,
    "2004-05": 37.64,
    "2004-06": 37.64,
    "2004-08": 33.74,
    "2005-04": 30.03,
    "2005-06": 28.97,
    "2006-06": 25.67,
    "2006-11": 26.36,
    "2013-01": 15.75,
    "2014-10": 15.20,
    "2015-07": 20.67,
    "2015-10": 21.73,
    "2015-12": 22.69,
    "2016-02": 20.76,
    "2016-06": 21.13,
    "2018-10": 23.43,
    "2018-12": 18.92,
    "2019-08": 22.98,
    "2020-04": 26.18,
    "2020-05": 27.79,
    "2021-11": 32.07,
    "2022-01": 28.42,
    "2022-03": 28.65,
    "2022-12": 23.97,
    "2023-03": 30.81,
    "2023-07": 33.91,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default="QQQ_3X_CALC")
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument(
        "--returns-file",
        default="reports/tables/tqqq_mixed_entry_exit_source_comparison_after_tax_returns.csv",
    )
    parser.add_argument(
        "--weights-file",
        default="reports/tables/tqqq_mixed_entry_exit_source_comparison_weights.csv",
    )
    parser.add_argument(
        "--drawdowns-file",
        default="reports/tables/tqqq_three_candidates_shared_worst_10_drawdowns.csv",
    )
    parser.add_argument(
        "--classification-file",
        default="reports/tables/tqqq_shared_worst_10_rate_hike_cycle_classification.csv",
    )
    parser.add_argument("--months-before", type=int, default=2)
    parser.add_argument("--months-after", type=int, default=2)
    parser.add_argument("--output-dir", default="reports/figures/tqqq_hiking_dd_sell_timing")
    parser.add_argument("--output-prefix", default="tqqq_hiking_dd_sell_timing")
    return parser.parse_args()


def _load_price(path: Path) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename("synthetic_tqqq_price")


def _strategy_peak_bottom(
    returns: pd.Series,
    *,
    episode_start: pd.Timestamp,
    episode_recovery: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    end = episode_recovery if episode_recovery is not None and pd.notna(episode_recovery) else returns.index[-1]
    episode_drawdown = drawdown.loc[episode_start:end]
    bottom_time = pd.Timestamp(episode_drawdown.idxmin())
    peak_mask = equity.loc[:bottom_time].eq(equity.loc[:bottom_time].cummax())
    peak_time = pd.Timestamp(peak_mask[peak_mask].index[-1])
    return peak_time, bottom_time, float(drawdown.loc[bottom_time])


def _trade_events(weights: pd.Series, price: pd.Series) -> pd.DataFrame:
    clean = weights.fillna(0.0).astype(float)
    before = clean.shift(1).fillna(0.0)
    changes = (clean - before).abs().gt(1e-12)
    events = pd.DataFrame(
        {
            "timestamp": clean.index[changes],
            "weight_before": before[changes].to_numpy(dtype=float),
            "weight_after": clean[changes].to_numpy(dtype=float),
        }
    )
    if events.empty:
        events["event_type"] = []
        events["price"] = []
        return events
    events["event_type"] = events.apply(
        lambda row: (
            "full_entry"
            if row["weight_before"] <= 1e-12 and row["weight_after"] > 1e-12
            else (
                "increase"
                if row["weight_after"] > row["weight_before"]
                else ("full_exit" if row["weight_after"] <= 1e-12 else "reduce")
            )
        ),
        axis=1,
    )
    events["price"] = price.reindex(pd.DatetimeIndex(events["timestamp"])).to_numpy(dtype=float)
    return events


def _cycle_for_rank(classification: pd.DataFrame, rank: int) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    row = classification[classification["rank"].astype(int).eq(rank)].iloc[0]
    cycle_name = str(row["matching_hike_cycle_overlap"]).split(";")[0].strip()
    if cycle_name not in HIKING_CYCLES:
        raise ValueError(f"No known hiking-cycle dates for rank {rank}: {cycle_name!r}")
    start, end = HIKING_CYCLES[cycle_name]
    return cycle_name, pd.Timestamp(start), pd.Timestamp(end)


def _pe_proxy(timestamp: pd.Timestamp) -> float | None:
    """Return monthly NASDAQ-100 P/E proxy value for a timestamp, if available."""
    return NASDAQ100_PE_PROXY_BY_MONTH.get(pd.Timestamp(timestamp).strftime("%Y-%m"))


def _format_pe(pe_value: float | None) -> str:
    return "N/A" if pe_value is None else f"{pe_value:.1f}x"


def _annotate_vertical_qqq_pe(
    ax: plt.Axes,
    *,
    timestamp: pd.Timestamp,
    color: str,
    y_fraction: float,
) -> None:
    """Annotate the QQQ PE at a vertical reference line."""
    pe_value = _pe_proxy(timestamp)
    ax.annotate(
        f"QQQ PE: {_format_pe(pe_value)}",
        xy=(timestamp, y_fraction),
        xycoords=("data", "axes fraction"),
        xytext=(3, 0),
        textcoords="offset points",
        rotation=90,
        va="bottom",
        ha="left",
        color=color,
        fontsize=7.0,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": color, "alpha": 0.68},
    )


def _tqqq_price_peak_bottom(
    price: pd.Series,
    *,
    shared_peak: pd.Timestamp,
    shared_bottom: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """Return synthetic-TQQQ price peak-before-trough and trough inside the shared DD.

    The shared drawdown period is defined from the strategy drawdown analysis,
    but the PE labels requested here should refer to the underlying synthetic
    TQQQ price high/low inside that period, not the strategy equity trough.
    """
    drawdown_price_window = price.loc[shared_peak:shared_bottom].dropna()
    if drawdown_price_window.empty:
        raise ValueError("No synthetic TQQQ prices inside shared drawdown window")
    tqqq_bottom = pd.Timestamp(drawdown_price_window.idxmin())
    peak_before_bottom = price.loc[shared_peak:tqqq_bottom].dropna()
    tqqq_peak = pd.Timestamp(peak_before_bottom.idxmax())
    tqqq_price_drawdown = float(price.loc[tqqq_bottom] / price.loc[tqqq_peak] - 1.0)
    return tqqq_peak, tqqq_bottom, tqqq_price_drawdown


def _plot_one(
    *,
    price: pd.Series,
    returns: pd.Series,
    weights: pd.Series,
    rank: int,
    shared_peak: pd.Timestamp,
    shared_bottom: pd.Timestamp,
    hike_cycle_name: str,
    hike_cycle_start: pd.Timestamp,
    hike_cycle_end: pd.Timestamp,
    strategy_peak: pd.Timestamp,
    strategy_bottom: pd.Timestamp,
    strategy_drawdown: float,
    average_drawdown: float,
    strategy_label: str,
    output_path: Path,
    months_before: int,
    months_after: int,
) -> pd.DataFrame:
    events = _trade_events(weights, price)
    entry_events_all = events[events["event_type"].isin(["full_entry", "increase"])]
    prior_entries = entry_events_all[entry_events_all["timestamp"].le(shared_peak)]
    last_buy_before_peak = (
        pd.Timestamp(prior_entries.iloc[-1]["timestamp"]) if not prior_entries.empty else None
    )

    # Make the window large enough to show the whole hiking cycle plus two
    # months on both sides, while also preserving the shared peak/bottom
    # markers if the drawdown starts before the rate-hike cycle. Also extend
    # the left side to include the last buy before the drawdown peak.
    base_start = min(shared_peak, hike_cycle_start) - pd.DateOffset(months=months_before)
    if last_buy_before_peak is not None:
        base_start = min(base_start, last_buy_before_peak - pd.DateOffset(days=5))
    window_start = base_start
    window_end = max(shared_bottom, hike_cycle_end) + pd.DateOffset(months=months_after)
    price_window = price.loc[window_start:window_end]
    returns_window = returns.loc[window_start:window_end]
    weights_window = weights.loc[window_start:window_end].fillna(0.0)
    equity_window = (1.0 + returns_window.fillna(0.0)).cumprod()
    equity_window = equity_window / equity_window.iloc[0] if not equity_window.empty else equity_window

    events_window = events[
        (events["timestamp"] >= window_start) & (events["timestamp"] <= window_end)
    ].copy()
    tqqq_peak, tqqq_bottom, tqqq_price_drawdown = _tqqq_price_peak_bottom(
        price,
        shared_peak=shared_peak,
        shared_bottom=shared_bottom,
    )
    tqqq_peak_pe = _pe_proxy(tqqq_peak)
    tqqq_bottom_pe = _pe_proxy(tqqq_bottom)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.5, 7.5),
        gridspec_kw={"height_ratios": [3.0, 1.15]},
        sharex=True,
    )
    ax = axes[0]
    ax2 = ax.twinx()
    price_window.plot(ax=ax, color="#1f77b4", linewidth=1.25, label="Synthetic TQQQ price")
    equity_window.plot(ax=ax2, color="#2ca02c", linewidth=1.0, alpha=0.75, label="Strategy equity")
    vertical_pe_times: list[tuple[pd.Timestamp, str, float]] = []
    ax.axvline(
        window_start,
        color="#555555",
        linestyle=":",
        linewidth=1.0,
        alpha=0.75,
        label="plot start",
    )
    vertical_pe_times.append((pd.Timestamp(window_start), "#555555", 0.03))
    ax.axvspan(hike_cycle_start, hike_cycle_end, color="#f5b041", alpha=0.13, label=hike_cycle_name)
    ax.axvspan(shared_peak, shared_bottom, color="#d62728", alpha=0.10, label="shared peak→bottom")
    ax.axvline(hike_cycle_start, color="#f39c12", linestyle=":", linewidth=1.2, alpha=0.9, label="hike start")
    ax.axvline(hike_cycle_end, color="#b9770e", linestyle=":", linewidth=1.2, alpha=0.9, label="hike end")
    vertical_pe_times.append((hike_cycle_start, "#f39c12", 0.11))
    vertical_pe_times.append((hike_cycle_end, "#b9770e", 0.11))
    signal_event = FED_HIKE_SIGNAL_EVENTS.get(hike_cycle_name)
    if signal_event is not None:
        signal_date, signal_label, _ = signal_event
        signal_ts = pd.Timestamp(signal_date)
        if window_start <= signal_ts <= window_end:
            ax.axvline(
                signal_ts,
                color="#7d3c98",
                linestyle="-.",
                linewidth=1.35,
                alpha=0.95,
                label="Fed signals hikes",
            )
            ax.annotate(
                signal_label,
                xy=(signal_ts, 0.97),
                xycoords=("data", "axes fraction"),
                xytext=(5, -22),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="left",
                color="#7d3c98",
                fontsize=7.5,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#7d3c98", "alpha": 0.72},
            )
            vertical_pe_times.append((signal_ts, "#7d3c98", 0.23))
    ax.axvline(shared_peak, color="#9467bd", linestyle="--", linewidth=1.0, alpha=0.8, label="shared peak")
    ax.axvline(shared_bottom, color="#8c564b", linestyle="--", linewidth=1.0, alpha=0.8, label="shared bottom")
    vertical_pe_times.append((shared_peak, "#9467bd", 0.35))
    vertical_pe_times.append((shared_bottom, "#8c564b", 0.35))

    for timestamp, color, y_fraction in vertical_pe_times:
        if window_start <= timestamp <= window_end:
            _annotate_vertical_qqq_pe(
                ax,
                timestamp=pd.Timestamp(timestamp),
                color=color,
                y_fraction=y_fraction,
            )

    for timestamp, color, marker, label in (
        (shared_peak, "#17becf", "o", "shared peak"),
        (shared_bottom, "#111111", "X", "shared bottom"),
    ):
        if timestamp in price_window.index:
            ax.scatter(
                [timestamp],
                [price.loc[timestamp]],
                color=color,
                marker=marker,
                s=80,
                zorder=5,
                label=label,
            )

    for timestamp, color, marker, label in (
        (tqqq_peak, "#005f73", "P", "TQQQ price peak"),
        (tqqq_bottom, "#c1121f", "D", "TQQQ price bottom"),
    ):
        if timestamp in price_window.index:
            ax.scatter(
                [timestamp],
                [price.loc[timestamp]],
                color=color,
                marker=marker,
                s=90,
                zorder=6,
                label=label,
                edgecolor="white",
                linewidth=0.5,
            )

    for timestamp, pe_value, label, xytext, color in (
        (tqqq_peak, tqqq_peak_pe, "QQQ PE", (10, -32), "#005f73"),
        (tqqq_bottom, tqqq_bottom_pe, "QQQ PE", (10, 24), "#c1121f"),
    ):
        if timestamp in price_window.index:
            ax.annotate(
                f"{label}: {_format_pe(pe_value)}",
                xy=(timestamp, price.loc[timestamp]),
                xytext=xytext,
                textcoords="offset points",
                color=color,
                fontsize=8,
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8},
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.75},
            )

    reduce_events = events_window[events_window["event_type"].eq("reduce")]
    exit_events = events_window[events_window["event_type"].eq("full_exit")]
    entry_events = events_window[events_window["event_type"].isin(["full_entry", "increase"])]
    if not entry_events.empty:
        ax.scatter(
            entry_events["timestamp"],
            entry_events["price"],
            marker="^",
            color="#2ca02c",
            edgecolor="#145a32",
            linewidth=0.35,
            s=42,
            zorder=4,
            label="entry/buy",
        )
    if not reduce_events.empty:
        ax.scatter(
            reduce_events["timestamp"],
            reduce_events["price"],
            marker="v",
            color="#ff7f0e",
            s=45,
            zorder=4,
            label="reduce/lock sell",
        )
    if not exit_events.empty:
        ax.scatter(
            exit_events["timestamp"],
            exit_events["price"],
            marker="v",
            color="#d62728",
            s=55,
            zorder=4,
            label="full exit sell",
        )

    ax.set_ylabel("Synthetic TQQQ price")
    ax2.set_ylabel("Strategy equity in window")
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"Rank {rank}: {strategy_label}\n"
        f"{hike_cycle_name} {hike_cycle_start:%Y-%m-%d}→{hike_cycle_end:%Y-%m-%d}; "
        f"shared DD {shared_peak:%Y-%m-%d}→{shared_bottom:%Y-%m-%d}, "
        f"avg DD {average_drawdown:.1%}; strategy episode DD {strategy_drawdown:.1%}\n"
        f"QQQ PE: "
        f"{_format_pe(tqqq_peak_pe)} / {_format_pe(tqqq_bottom_pe)}; "
        f"TQQQ price DD {tqqq_price_drawdown:.1%}"
    )
    ax.set_xlim(window_start, window_end)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="best", fontsize=8)

    axes[1].step(
        weights_window.index,
        weights_window.to_numpy(dtype=float),
        where="post",
        color="#222222",
        linewidth=1.15,
    )
    axes[1].fill_between(
        weights_window.index,
        weights_window.to_numpy(dtype=float),
        step="post",
        alpha=0.18,
        color="#222222",
    )
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("Position")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xlim(window_start, window_end)
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[1].xaxis.get_major_locator()))

    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return events_window


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    price_path = resolve_path(config.root, args.target_raw_dir) / f"{args.target_ticker}.parquet"
    price = _load_price(price_path)

    returns = pd.read_csv(resolve_path(config.root, args.returns_file), parse_dates=["date"]).set_index("date")
    weights = pd.read_csv(resolve_path(config.root, args.weights_file), parse_dates=["date"]).set_index("date")
    drawdowns = pd.read_csv(resolve_path(config.root, args.drawdowns_file), parse_dates=[
        "shared_peak_start",
        "shared_bottom",
        "shared_recovery",
    ])
    classification = pd.read_csv(resolve_path(config.root, args.classification_file))

    overlap_ranks = classification.loc[
        classification["overlaps_rate_hike_cycle"].astype(bool),
        "rank",
    ].astype(int)
    selected_drawdowns = drawdowns[drawdowns["rank"].isin(overlap_ranks)].sort_values("rank")

    output_dir = resolve_path(config.root, args.output_dir)
    ensure_directory(output_dir)
    event_rows: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for _, row in selected_drawdowns.iterrows():
        rank = int(row["rank"])
        shared_peak = pd.Timestamp(row["shared_peak_start"])
        shared_bottom = pd.Timestamp(row["shared_bottom"])
        shared_recovery = pd.Timestamp(row["shared_recovery"]) if pd.notna(row["shared_recovery"]) else None
        average_drawdown = float(row["average_drawdown"])
        hike_cycle_name, hike_cycle_start, hike_cycle_end = _cycle_for_rank(classification, rank)

        for spec in STRATEGIES.values():
            column = spec["column"]
            strategy_returns = returns[column].reindex(price.index).dropna()
            strategy_weights = weights[column].reindex(price.index).fillna(0.0)
            strategy_peak, strategy_bottom, strategy_drawdown = _strategy_peak_bottom(
                strategy_returns,
                episode_start=shared_peak,
                episode_recovery=shared_recovery,
            )
            output_path = output_dir / f"{args.output_prefix}_rank{rank}_{spec['short_label']}.png"
            events_window = _plot_one(
                price=price,
                returns=strategy_returns,
                weights=strategy_weights,
                rank=rank,
                shared_peak=shared_peak,
                shared_bottom=shared_bottom,
                hike_cycle_name=hike_cycle_name,
                hike_cycle_start=hike_cycle_start,
                hike_cycle_end=hike_cycle_end,
                strategy_peak=strategy_peak,
                strategy_bottom=strategy_bottom,
                strategy_drawdown=strategy_drawdown,
                average_drawdown=average_drawdown,
                strategy_label=spec["label"],
                output_path=output_path,
                months_before=args.months_before,
                months_after=args.months_after,
            )
            if not events_window.empty:
                events_window.insert(0, "strategy", spec["label"])
                events_window.insert(0, "rank", rank)
                event_rows.append(events_window)
            tqqq_price_peak, tqqq_price_bottom, tqqq_price_drawdown = _tqqq_price_peak_bottom(
                price,
                shared_peak=shared_peak,
                shared_bottom=shared_bottom,
            )
            strategy_peak_pe = _pe_proxy(strategy_peak)
            strategy_bottom_pe = _pe_proxy(strategy_bottom)
            shared_peak_pe = _pe_proxy(shared_peak)
            shared_bottom_pe = _pe_proxy(shared_bottom)
            tqqq_price_peak_pe = _pe_proxy(tqqq_price_peak)
            tqqq_price_bottom_pe = _pe_proxy(tqqq_price_bottom)
            prior_buys = events_window[
                events_window["event_type"].isin(["full_entry", "increase"])
                & events_window["timestamp"].le(shared_peak)
            ]
            last_buy_before_shared_peak = (
                pd.Timestamp(prior_buys.iloc[-1]["timestamp"]) if not prior_buys.empty else pd.NaT
            )
            signal_event = FED_HIKE_SIGNAL_EVENTS.get(hike_cycle_name)
            signal_date = signal_event[0] if signal_event is not None else ""
            signal_label = signal_event[1] if signal_event is not None else ""
            signal_source = signal_event[2] if signal_event is not None else ""
            manifest_rows.append(
                {
                    "rank": rank,
                    "strategy": spec["label"],
                    "hike_cycle": hike_cycle_name,
                    "hike_cycle_start": hike_cycle_start,
                    "hike_cycle_end": hike_cycle_end,
                    "fed_hike_signal_date": signal_date,
                    "fed_hike_signal_label": signal_label,
                    "fed_hike_signal_source": signal_source,
                    "shared_peak": shared_peak,
                    "shared_bottom": shared_bottom,
                    "shared_peak_pe_proxy": shared_peak_pe,
                    "shared_bottom_pe_proxy": shared_bottom_pe,
                    "tqqq_price_peak": tqqq_price_peak,
                    "tqqq_price_bottom": tqqq_price_bottom,
                    "tqqq_price_peak_pe_proxy": tqqq_price_peak_pe,
                    "tqqq_price_bottom_pe_proxy": tqqq_price_bottom_pe,
                    "tqqq_price_peak_to_bottom_drawdown": tqqq_price_drawdown,
                    "strategy_peak": strategy_peak,
                    "strategy_bottom": strategy_bottom,
                    "strategy_peak_pe_proxy": strategy_peak_pe,
                    "strategy_bottom_pe_proxy": strategy_bottom_pe,
                    "last_buy_before_shared_peak": last_buy_before_shared_peak,
                    "strategy_drawdown": strategy_drawdown,
                    "average_shared_drawdown": average_drawdown,
                    "plot_path": output_path.as_posix(),
                    "trade_events_in_window": int(len(events_window)),
                }
            )

    tables_dir = config.reports.tables_dir
    ensure_directory(tables_dir)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = tables_dir / f"{args.output_prefix}_manifest.csv"
    events_path = tables_dir / f"{args.output_prefix}_events.csv"
    pe_path = tables_dir / f"{args.output_prefix}_pe_proxy_values.csv"
    fed_signal_path = tables_dir / f"{args.output_prefix}_fed_hike_signal_events.csv"
    manifest.to_csv(manifest_path, index=False)
    if event_rows:
        pd.concat(event_rows, ignore_index=True).to_csv(events_path, index=False)
    else:
        pd.DataFrame().to_csv(events_path, index=False)
    pd.DataFrame(
        [
            {
                "month": month,
                "nasdaq100_pe_proxy": value,
                "source": "Trendonify Nasdaq 100 PE Ratio historical-data table",
                "note": "QQQ PE proxy from Nasdaq-100 because QQQ tracks the NASDAQ-100.",
            }
            for month, value in sorted(NASDAQ100_PE_PROXY_BY_MONTH.items())
        ]
    ).to_csv(pe_path, index=False)
    pd.DataFrame(
        [
            {
                "hike_cycle": cycle,
                "fed_hike_signal_date": values[0],
                "fed_hike_signal_label": values[1],
                "source": values[2],
            }
            for cycle, values in FED_HIKE_SIGNAL_EVENTS.items()
        ]
    ).to_csv(fed_signal_path, index=False)

    print(f"Saved {len(manifest)} plots to {output_dir}")
    print(f"Manifest saved to {manifest_path}")
    print(f"Trade event table saved to {events_path}")
    print(f"PE proxy table saved to {pe_path}")
    print(f"Fed hike signal table saved to {fed_signal_path}")
    print(
        manifest[
            [
                "rank",
                "strategy",
                "hike_cycle",
                "fed_hike_signal_date",
                "tqqq_price_peak",
                "tqqq_price_bottom",
                "tqqq_price_peak_pe_proxy",
                "tqqq_price_bottom_pe_proxy",
                "last_buy_before_shared_peak",
                "strategy_peak",
                "strategy_bottom",
                "trade_events_in_window",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
