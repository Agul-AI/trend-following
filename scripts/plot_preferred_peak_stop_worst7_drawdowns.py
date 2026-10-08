#!/usr/bin/env python
"""Plot the seven worst drawdowns for the current preferred +40% peak-stop strategy.

Each plot shows:
- QQQ hourly close and the QQQ hourly 200-day MA used for entry/exit gating.
- Synthetic TQQQ price with round-trip entry/exit labels and round-trip return.
- Profit-lock hits from executable weight reductions (+300% -> 75%, +400% -> 50%).
- Strategy equity and drawdown, with peak/trough/recovery marked.

The current preferred strategy column is ``profit_lock_300_400_stop_40pct`` from
``reports/tables/preferred_profit_lock_stop_exit_comparison_*``.
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
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.utils import ensure_directory  # noqa: E402

STRATEGY_COLUMN = "profit_lock_300_400_stop_40pct"
QQQ_MA_DAYS = 200
BARS_PER_DAY = 6
QQQ_MA_BARS = QQQ_MA_DAYS * BARS_PER_DAY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--returns-file",
        default="reports/tables/preferred_profit_lock_stop_exit_comparison_returns.csv",
    )
    parser.add_argument(
        "--weights-file",
        default="reports/tables/preferred_profit_lock_stop_exit_comparison_weights.csv",
    )
    parser.add_argument(
        "--trade-stats-file",
        default="reports/tables/preferred_plus_40pct_peak_stop_trade_stats.csv",
    )
    parser.add_argument("--qqq-price-file", default="data/raw/alpha_vantage_60min/QQQ.parquet")
    parser.add_argument("--target-price-file", default="data/raw/synthetic_3x_60min/QQQ_3X_CALC.parquet")
    parser.add_argument("--top-n", type=int, default=7)
    parser.add_argument("--months-before", type=int, default=2)
    parser.add_argument("--months-after", type=int, default=2)
    parser.add_argument("--output-dir", default="reports/figures/preferred_peak_stop_worst7_drawdowns")
    parser.add_argument("--output-prefix", default="preferred_peak_stop_worst_dd")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def _equity(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def find_drawdown_episodes(returns: pd.Series) -> pd.DataFrame:
    """Return all complete/incomplete strategy drawdown episodes sorted worst first."""
    equity = _equity(returns)
    drawdown = _drawdown(equity)
    peak_times: list[pd.Timestamp] = []
    running_peak_value = -np.inf
    running_peak_time = pd.Timestamp(equity.index[0])
    for timestamp, value in equity.items():
        if float(value) >= running_peak_value - 1e-12:
            running_peak_value = float(value)
            running_peak_time = pd.Timestamp(timestamp)
        peak_times.append(running_peak_time)
    peak_time_series = pd.Series(peak_times, index=equity.index)

    episodes: list[dict[str, Any]] = []
    in_episode = False
    peak = start = trough = None
    trough_dd = 0.0
    for timestamp, dd_value in drawdown.items():
        if not in_episode and dd_value < -1e-12:
            in_episode = True
            peak = pd.Timestamp(peak_time_series.loc[timestamp])
            start = pd.Timestamp(timestamp)
            trough = pd.Timestamp(timestamp)
            trough_dd = float(dd_value)
        elif in_episode:
            if dd_value < trough_dd:
                trough = pd.Timestamp(timestamp)
                trough_dd = float(dd_value)
            if dd_value >= -1e-12:
                episodes.append(
                    {
                        "peak": peak,
                        "start": start,
                        "trough": trough,
                        "recovery": pd.Timestamp(timestamp),
                        "max_drawdown": trough_dd,
                    }
                )
                in_episode = False
                peak = start = trough = None
                trough_dd = 0.0
    if in_episode:
        episodes.append(
            {
                "peak": peak,
                "start": start,
                "trough": trough,
                "recovery": pd.NaT,
                "max_drawdown": trough_dd,
            }
        )
    frame = pd.DataFrame(episodes)
    frame["calendar_days_peak_to_trough"] = (
        pd.to_datetime(frame["trough"]).dt.normalize() - pd.to_datetime(frame["peak"]).dt.normalize()
    ).dt.days
    frame["calendar_days_peak_to_recovery"] = (
        pd.to_datetime(frame["recovery"]).dt.normalize() - pd.to_datetime(frame["peak"]).dt.normalize()
    ).dt.days
    return frame.sort_values("max_drawdown").reset_index(drop=True)


def _strategy_window_index(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return index[(index >= start) & (index <= end)]


def _profit_lock_events(weights: pd.Series) -> pd.DataFrame:
    before = weights.shift(1).fillna(0.0)
    after = weights.fillna(0.0)
    reduced_inside_trade = before.gt(0.0) & after.gt(0.0) & after.lt(before - 1e-12)
    events = pd.DataFrame(
        {
            "timestamp": after.index[reduced_inside_trade],
            "weight_before": before[reduced_inside_trade].to_numpy(dtype=float),
            "weight_after": after[reduced_inside_trade].to_numpy(dtype=float),
        }
    )
    if events.empty:
        events["label"] = []
        return events
    events["label"] = events.apply(
        lambda row: "+300% -> 75%" if abs(float(row["weight_after"]) - 0.75) < 1e-8 else "+400% -> 50%",
        axis=1,
    )
    return events


def _overlapping_trades(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (trades["exit_return_label"] >= start) & (trades["entry_execution_close"] <= end)
    return trades.loc[mask].copy()


def _nearest_value(series: pd.Series, timestamp: pd.Timestamp) -> float:
    if timestamp in series.index:
        return float(series.loc[timestamp])
    loc = series.index.get_indexer([timestamp], method="nearest")
    if loc[0] < 0:
        return np.nan
    return float(series.iloc[loc[0]])


def _vline_all(axes: list[plt.Axes], timestamp: pd.Timestamp, *, color: str, linestyle: str, label: str) -> None:
    for ax in axes:
        ax.axvline(timestamp, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.85)
    axes[0].text(
        timestamp,
        0.98,
        label,
        transform=axes[0].get_xaxis_transform(),
        rotation=90,
        va="top",
        ha="right",
        fontsize=7,
        color=color,
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": color, "alpha": 0.65},
    )


def _annotate_trade(
    ax: plt.Axes,
    *,
    timestamp: pd.Timestamp,
    y: float,
    text: str,
    color: str,
    offset: tuple[int, int],
) -> None:
    ax.annotate(
        text,
        xy=(timestamp, y),
        xytext=offset,
        textcoords="offset points",
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        fontsize=7.0,
        color=color,
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7, "alpha": 0.8},
        bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": color, "alpha": 0.78},
    )


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100.0:+.1f}%"


def _trade_summary_table(trade_window: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Build compact trade DD/RT table and compounded rows for the plot footer.

    Trade DD is the trade-level max drawdown from that trade's own peak.
    RT is the sized pre-tax round-trip return.
    The compounded DD is a proxy: product(1 + each trade DD) - 1.
    The compounded RT is product(1 + each RT) - 1.
    """
    if trade_window.empty:
        empty = pd.DataFrame(columns=["Trade", "Entry", "Exit", "Trade DD", "RT"])
        return empty, np.nan, np.nan
    rows: list[dict[str, str]] = []
    dds = trade_window["max_peak_drawdown"].astype(float)
    rts = trade_window["sized_pre_tax_return"].astype(float)
    comp_dd = float((1.0 + dds).prod() - 1.0)
    comp_rt = float((1.0 + rts).prod() - 1.0)
    for row in trade_window.itertuples(index=False):
        rows.append(
            {
                "Trade": f"T{int(row.trade_id)}",
                "Entry": pd.Timestamp(row.entry_execution_close).strftime("%Y-%m-%d"),
                "Exit": pd.Timestamp(row.exit_return_label).strftime("%Y-%m-%d") if pd.notna(row.exit_return_label) else "open",
                "Trade DD": _fmt_pct(float(row.max_peak_drawdown)),
                "RT": _fmt_pct(float(row.sized_pre_tax_return)),
            }
        )
    rows.append(
        {
            "Trade": "Comp",
            "Entry": "",
            "Exit": "",
            "Trade DD": _fmt_pct(comp_dd),
            "RT": _fmt_pct(comp_rt),
        }
    )
    return pd.DataFrame(rows), comp_dd, comp_rt


def plot_episode(
    *,
    rank: int,
    episode: pd.Series,
    returns: pd.Series,
    weights: pd.Series,
    qqq: pd.Series,
    qqq_ma: pd.Series,
    target: pd.Series,
    trades: pd.DataFrame,
    pl_events: pd.DataFrame,
    output_path: Path,
    months_before: int,
    months_after: int,
) -> dict[str, Any]:
    peak = pd.Timestamp(episode["peak"])
    trough = pd.Timestamp(episode["trough"])
    recovery = pd.Timestamp(episode["recovery"]) if pd.notna(episode["recovery"]) else returns.index[-1]
    left = peak - pd.DateOffset(months=months_before)
    right = recovery + pd.DateOffset(months=months_after)
    window_index = _strategy_window_index(returns.index, left, right)
    if window_index.empty:
        raise ValueError(f"Empty plot window for episode rank {rank}")
    left = pd.Timestamp(window_index[0])
    right = pd.Timestamp(window_index[-1])

    qqq_window = qqq.loc[left:right]
    qqq_ma_window = qqq_ma.loc[left:right]
    target_window = target.loc[left:right]
    equity = _equity(returns)
    drawdown = _drawdown(equity)
    equity_window = equity.loc[left:right]
    dd_window = drawdown.loc[left:right]
    weights_window = weights.loc[left:right]
    trade_window = _overlapping_trades(trades, left, right)
    pl_window = pl_events[(pl_events["timestamp"] >= left) & (pl_events["timestamp"] <= right)].copy()

    table_frame, comp_trade_dd, comp_trade_rt = _trade_summary_table(trade_window)
    table_height = max(0.80, min(1.55, 0.42 + 0.075 * max(len(table_frame), 1)))
    fig_height = 12.4 + 0.16 * max(len(table_frame), 1)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(18, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.35, 1.05, table_height]},
    )
    ax_q, ax_t, ax_e, ax_tbl = axes
    plot_axes = [ax_q, ax_t, ax_e]

    # DD span and reference lines.
    for ax in plot_axes:
        ax.axvspan(peak, recovery, color="#d62728", alpha=0.055, label="drawdown episode")
        ax.axvspan(peak, trough, color="#d62728", alpha=0.10, label="peak to trough")
    _vline_all(plot_axes, peak, color="#1f77b4", linestyle="--", label="DD peak")
    _vline_all(plot_axes, trough, color="#7f0000", linestyle="--", label="DD trough")
    if pd.notna(episode["recovery"]):
        _vline_all(plot_axes, recovery, color="#2ca02c", linestyle="--", label="DD recovery")

    # Panel 1: signal source and 200-day MA.
    ax_q.plot(qqq_window.index, qqq_window, color="#264653", linewidth=1.05, label="QQQ hourly close")
    ax_q.plot(
        qqq_ma_window.index,
        qqq_ma_window,
        color="#f4a261",
        linewidth=1.20,
        label=f"QQQ hourly {QQQ_MA_DAYS}d MA ({QQQ_MA_BARS} bars)",
    )
    ax_q.set_ylabel("QQQ price")
    ax_q.grid(True, alpha=0.25)
    ax_q.legend(loc="upper left", fontsize=8)

    # Panel 2: traded synthetic TQQQ price and weight.
    ax_t.plot(target_window.index, target_window, color="#005f73", linewidth=1.05, label="synthetic TQQQ price")
    ax_t.set_ylabel("Synthetic TQQQ price")
    ax_t.grid(True, alpha=0.25)
    ax_w = ax_t.twinx()
    ax_w.step(weights_window.index, weights_window, where="post", color="#6a4c93", alpha=0.45, linewidth=1.1, label="executable weight")
    ax_w.set_ylabel("Weight", color="#6a4c93")
    ax_w.set_ylim(-0.05, 1.08)
    ax_w.tick_params(axis="y", labelcolor="#6a4c93")

    # Panel 3: strategy equity/drawdown.
    equity_norm = equity_window / equity_window.iloc[0]
    ax_e.plot(equity_norm.index, equity_norm, color="#005f73", linewidth=1.10, label="strategy equity, window-normalized")
    ax_e.set_ylabel("Equity norm")
    ax_e.grid(True, alpha=0.25)
    ax_dd = ax_e.twinx()
    ax_dd.fill_between(dd_window.index, dd_window * 100.0, 0, color="#d62728", alpha=0.22, label="strategy drawdown")
    ax_dd.plot(dd_window.index, dd_window * 100.0, color="#d62728", linewidth=0.85)
    ax_dd.set_ylabel("Drawdown (%)", color="#d62728")
    ax_dd.tick_params(axis="y", labelcolor="#d62728")

    # Entry/exit labels for overlapping round trips.
    for j, row in trade_window.reset_index(drop=True).iterrows():
        trade_id = int(row["trade_id"])
        rt = float(row["sized_pre_tax_return"]) * 100.0
        trade_dd = float(row["max_peak_drawdown"]) * 100.0
        label_rt = f"{rt:+.1f}%"
        label_dd = f"{trade_dd:+.1f}%"
        entry_ts = pd.Timestamp(row["entry_execution_close"])
        exit_ts = pd.Timestamp(row["exit_return_label"])
        color_entry = "#2ca02c"
        color_exit = "#d62728"
        if left <= entry_ts <= right:
            yq = _nearest_value(qqq, entry_ts)
            yt = _nearest_value(target, entry_ts)
            ax_q.scatter(entry_ts, yq, marker="^", s=55, color=color_entry, edgecolor="black", zorder=5)
            ax_t.scatter(entry_ts, yt, marker="^", s=58, color=color_entry, edgecolor="black", zorder=6)
            _annotate_trade(
                ax_t,
                timestamp=entry_ts,
                y=yt,
                text=f"T{trade_id} entry",
                color=color_entry,
                offset=(7, 12 + (j % 3) * 7),
            )
        if left <= exit_ts <= right:
            yq = _nearest_value(qqq, exit_ts)
            yt = _nearest_value(target, exit_ts)
            ax_q.scatter(exit_ts, yq, marker="v", s=55, color=color_exit, edgecolor="black", zorder=5)
            ax_t.scatter(exit_ts, yt, marker="v", s=58, color=color_exit, edgecolor="black", zorder=6)
            _annotate_trade(
                ax_t,
                timestamp=exit_ts,
                y=yt,
                text=f"T{trade_id} exit\nRT {label_rt}\nDD {label_dd}",
                color=color_exit,
                offset=(-7, -18 - (j % 3) * 7),
            )

    # Profit lock hits.
    if pl_window.empty:
        ax_t.text(
            0.985,
            0.965,
            "Profit-lock hits in shown window: none",
            transform=ax_t.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#7a3b00",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#ffb703", "alpha": 0.80},
        )
    for _, row in pl_window.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        yt = _nearest_value(target, ts)
        label = str(row["label"])
        marker_color = "#ffb703" if "300" in label else "#fb8500"
        ax_t.scatter(ts, yt, marker="*", s=145, color=marker_color, edgecolor="black", zorder=8)
        ax_t.annotate(
            f"PL {label}",
            xy=(ts, yt),
            xytext=(5, 25 if "300" in label else 40),
            textcoords="offset points",
            fontsize=7.5,
            color="#7a3b00",
            arrowprops={"arrowstyle": "-", "color": "#7a3b00", "lw": 0.75},
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": marker_color, "alpha": 0.82},
        )

    # Legends for panels with twin axes.
    lines_t, labels_t = ax_t.get_legend_handles_labels()
    lines_w, labels_w = ax_w.get_legend_handles_labels()
    ax_t.legend(lines_t + lines_w, labels_t + labels_w, loc="upper left", fontsize=8)
    lines_e, labels_e = ax_e.get_legend_handles_labels()
    lines_dd, labels_dd = ax_dd.get_legend_handles_labels()
    ax_e.legend(lines_e + lines_dd, labels_e + labels_dd, loc="upper left", fontsize=8)

    # Footer table: all visible overlapping round trips plus compounded DD/RT.
    ax_tbl.axis("off")
    if table_frame.empty:
        ax_tbl.text(0.5, 0.5, "No overlapping round-trip trades in this window", ha="center", va="center")
    else:
        table = ax_tbl.table(
            cellText=table_frame.to_numpy().tolist(),
            colLabels=table_frame.columns.tolist(),
            loc="center",
            cellLoc="center",
            colLoc="center",
            colWidths=[0.11, 0.20, 0.20, 0.16, 0.16],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.7 if len(table_frame) > 10 else 7.3)
        table.scale(1.0, 1.08)
        # Bold header and compounded row.
        last_row = len(table_frame)
        for (r, _c), cell in table.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f2f2f2")
            if r == last_row:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#fff3cd")
        ax_tbl.set_title(
            "Visible round-trip trade summary: Trade DD = max drawdown from that trade's own peak; "
            "Comp DD/RT = product across visible rows",
            fontsize=8.5,
            pad=2,
        )

    ax_e.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    ax_e.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_e.xaxis.get_major_locator()))

    title = (
        f"Worst drawdown #{rank}: {float(episode['max_drawdown']) * 100:.1f}% "
        f"| peak {peak:%Y-%m-%d %H:%M} -> trough {trough:%Y-%m-%d %H:%M}"
    )
    if pd.notna(episode["recovery"]):
        title += f" -> recovery {recovery:%Y-%m-%d %H:%M}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    table_output_path = output_path.with_name(output_path.stem + "_trade_table.csv")
    table_frame.to_csv(table_output_path, index=False)

    return {
        "rank": rank,
        "output_path": str(output_path),
        "trade_table_path": str(table_output_path),
        "compounded_visible_trade_dd": comp_trade_dd,
        "compounded_visible_trade_rt": comp_trade_rt,
        "peak": peak,
        "trough": trough,
        "recovery": recovery,
        "max_drawdown": float(episode["max_drawdown"]),
        "plot_left": left,
        "plot_right": right,
        "overlapping_trade_ids": ",".join(trade_window["trade_id"].astype(int).astype(str).tolist()),
        "profit_lock_events_in_window": int(len(pl_window)),
    }


def main() -> None:
    args = parse_args()
    returns_frame = pd.read_csv(args.returns_file, parse_dates=["date"]).set_index("date").sort_index()
    weights_frame = pd.read_csv(args.weights_file, parse_dates=["date"]).set_index("date").sort_index()
    if STRATEGY_COLUMN not in returns_frame.columns:
        raise ValueError(f"Missing strategy column {STRATEGY_COLUMN!r} in returns file")
    if STRATEGY_COLUMN not in weights_frame.columns:
        raise ValueError(f"Missing strategy column {STRATEGY_COLUMN!r} in weights file")
    returns = returns_frame[STRATEGY_COLUMN].astype(float)
    weights = weights_frame[STRATEGY_COLUMN].reindex(returns.index).fillna(0.0).astype(float)

    qqq = _load_price(Path(args.qqq_price_file), "QQQ").reindex(returns.index).ffill()
    target = _load_price(Path(args.target_price_file), "QQQ_3X_CALC").reindex(returns.index).ffill()
    qqq_ma = qqq.rolling(window=QQQ_MA_BARS, min_periods=QQQ_MA_BARS).mean().rename("QQQ_200d_hourly_MA")
    trades = pd.read_csv(
        args.trade_stats_file,
        parse_dates=["entry_return_label", "entry_execution_close", "exit_return_label"],
    ).sort_values("trade_id")
    pl_events = _profit_lock_events(weights)

    episodes = find_drawdown_episodes(returns).head(args.top_n).copy()
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)
    plot_records: list[dict[str, Any]] = []
    for rank, (_, episode) in enumerate(episodes.iterrows(), start=1):
        output_path = output_dir / f"{args.output_prefix}_{rank:02d}.png"
        record = plot_episode(
            rank=rank,
            episode=episode,
            returns=returns,
            weights=weights,
            qqq=qqq,
            qqq_ma=qqq_ma,
            target=target,
            trades=trades,
            pl_events=pl_events,
            output_path=output_path,
            months_before=args.months_before,
            months_after=args.months_after,
        )
        plot_records.append(record)

    summary = pd.DataFrame(plot_records)
    summary_path = Path("reports/tables/preferred_peak_stop_worst7_drawdown_plot_summary.csv")
    ensure_directory(summary_path.parent)
    summary.to_csv(summary_path, index=False)
    episodes_out = episodes.copy()
    episodes_out.insert(0, "rank", range(1, len(episodes_out) + 1))
    episodes_path = Path("reports/tables/preferred_peak_stop_worst7_drawdown_episodes.csv")
    episodes_out.to_csv(episodes_path, index=False)

    print(f"Saved {len(summary)} plots to {output_dir}")
    print(f"Plot summary saved to {summary_path}")
    print(summary[["rank", "max_drawdown", "peak", "trough", "recovery", "overlapping_trade_ids", "profit_lock_events_in_window", "output_path"]].to_string(index=False))


if __name__ == "__main__":
    main()
