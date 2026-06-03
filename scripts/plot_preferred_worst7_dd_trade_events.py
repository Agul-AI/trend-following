#!/usr/bin/env python
"""Plot the 7 worst current-preferred drawdowns with trade/profit-lock labels.

Current preferred variant plotted here:
- QQQ hourly MACD hist > 0 entry + QQQ hourly 200-day MA gate.
- QQQ hourly 200-day MA exit.
- Synthetic QQQ_3X_CALC exposure.
- Profit lock: +300% -> 75%, +400% -> 50%.
- Synthetic-Q_3X trade-peak stop: -40%.

Each plot covers the drawdown peak-to-recovery window with at least two calendar
months on both sides. If needed, the window is extended further to include full
entry/exit markers for trades overlapping that drawdown episode.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_preferred_profit_lock_comparison import _load_price  # noqa: E402
from run_preferred_profit_lock_stop_exit_comparison import (  # noqa: E402
    PROFIT_LOCK_SCHEME,
    raw_with_peak_drawdown_stop,
)
from run_tqqq_daily_gate_ablation import no_daily_gate_hourly_ma_gate_signal  # noqa: E402
from run_tqqq_entry_signal_comparison import _equity  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

TARGET_TICKER = "QQQ_3X_CALC"
BENCHMARK_TICKER = "QQQ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default=TARGET_TICKER)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--returns-path", default="reports/tables/preferred_profit_lock_stop_exit_comparison_returns.csv")
    parser.add_argument("--trades-path", default="reports/tables/preferred_plus_40pct_peak_stop_trade_stats.csv")
    parser.add_argument("--variant-name", default="profit_lock_300_400_stop_40pct")
    parser.add_argument("--threshold", type=float, default=-0.40)
    parser.add_argument("--top-n", type=int, default=7)
    parser.add_argument("--extend-months", type=int, default=2)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--output-dir", default="reports/figures/preferred_stop40_worst7_dd_trade_events")
    parser.add_argument("--output-prefix", default="preferred_stop40_worst_dd")
    return parser.parse_args()


def drawdown_episodes(returns: pd.Series, *, threshold: float) -> pd.DataFrame:
    """Return strategy drawdown episodes whose trough breaches threshold."""
    equity = _equity(returns)
    drawdown = equity / equity.cummax() - 1.0
    rows: list[dict[str, Any]] = []
    in_episode = False
    start = None
    peak = None
    trough = None
    trough_dd = 0.0
    for timestamp, dd_value in drawdown.items():
        if not in_episode and dd_value < -1e-12:
            in_episode = True
            start = timestamp
            peak = equity.loc[:timestamp].idxmax()
            trough = timestamp
            trough_dd = float(dd_value)
        elif in_episode:
            if dd_value < trough_dd:
                trough = timestamp
                trough_dd = float(dd_value)
            if dd_value >= -1e-12:
                if trough_dd <= threshold:
                    rows.append(
                        {
                            "peak": peak,
                            "start": start,
                            "trough": trough,
                            "recovery": timestamp,
                            "max_drawdown": trough_dd,
                        }
                    )
                in_episode = False
                start = peak = trough = None
                trough_dd = 0.0
    if in_episode and trough_dd <= threshold:
        rows.append(
            {
                "peak": peak,
                "start": start,
                "trough": trough,
                "recovery": pd.NaT,
                "max_drawdown": trough_dd,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ["peak", "start", "trough", "recovery"]:
        frame[column] = pd.to_datetime(frame[column])
    frame = frame.sort_values("peak").reset_index(drop=True)
    frame["chronological_dd_id"] = np.arange(1, len(frame) + 1)
    frame["severity_rank"] = frame["max_drawdown"].rank(method="first", ascending=True).astype(int)
    frame["peak_to_trough_calendar_days"] = (frame["trough"].dt.normalize() - frame["peak"].dt.normalize()).dt.days
    frame["peak_to_recovery_calendar_days"] = (frame["recovery"].dt.normalize() - frame["peak"].dt.normalize()).dt.days
    return frame


def profit_lock_events(raw_signal: pd.Series, price: pd.Series, trades: pd.DataFrame) -> pd.DataFrame:
    """Find raw-bar profit-lock threshold hit events and map them to trade IDs."""
    base = raw_signal.fillna(0.0).astype(float)
    clean_price = price.reindex(base.index).astype(float)
    thresholds = sorted(PROFIT_LOCK_SCHEME)
    in_trade = False
    entry_price = np.nan
    hit_thresholds: set[float] = set()
    events: list[dict[str, Any]] = []

    def map_trade_id(timestamp: pd.Timestamp) -> int | None:
        mask = (trades["entry_execution_close"] <= timestamp) & (trades["exit_return_label"] >= timestamp)
        if mask.any():
            return int(trades.loc[mask, "trade_id"].iloc[0])
        return None

    for timestamp, signal in base.items():
        current_price = clean_price.loc[timestamp]
        if signal <= 0.0 or not np.isfinite(current_price):
            in_trade = False
            entry_price = np.nan
            hit_thresholds.clear()
            continue
        if not in_trade:
            in_trade = True
            entry_price = float(current_price)
            hit_thresholds.clear()
        gain = float(current_price) / entry_price - 1.0 if entry_price > 0 else np.nan
        for threshold, new_weight in thresholds:
            if np.isfinite(gain) and gain >= threshold and threshold not in hit_thresholds:
                hit_thresholds.add(threshold)
                events.append(
                    {
                        "timestamp": timestamp,
                        "threshold_gain": threshold,
                        "new_weight": new_weight,
                        "price": float(current_price),
                        "gain_pct": gain * 100.0,
                        "trade_id": map_trade_id(timestamp),
                    }
                )
    frame = pd.DataFrame(events)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def format_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def overlapping_trades(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return trades[(trades["exit_return_label"] >= start) & (trades["entry_execution_close"] <= end)].copy()


def plot_episode(
    *,
    episode: pd.Series,
    returns: pd.Series,
    target_price: pd.Series,
    trades: pd.DataFrame,
    pl_events: pd.DataFrame,
    output_path: Path,
    extend_months: int,
) -> dict[str, Any]:
    equity = _equity(returns)
    drawdown = equity / equity.cummax() - 1.0

    base_start = episode["peak"] - pd.DateOffset(months=extend_months)
    recovery_or_trough = episode["recovery"] if pd.notna(episode["recovery"]) else episode["trough"]
    base_end = recovery_or_trough + pd.DateOffset(months=extend_months)
    episode_trades = overlapping_trades(trades, episode["peak"], recovery_or_trough)
    if not episode_trades.empty:
        plot_start = min(base_start, episode_trades["entry_execution_close"].min())
        plot_end = max(base_end, episode_trades["exit_return_label"].max())
    else:
        plot_start = base_start
        plot_end = base_end

    price_window = target_price.loc[(target_price.index >= plot_start) & (target_price.index <= plot_end)]
    dd_window = drawdown.loc[(drawdown.index >= plot_start) & (drawdown.index <= plot_end)]
    trades_window = overlapping_trades(trades, plot_start, plot_end)
    pl_window = pl_events[(pl_events["timestamp"] >= plot_start) & (pl_events["timestamp"] <= plot_end)].copy()

    fig, axes = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0]})
    ax_price, ax_dd = axes

    ax_price.plot(price_window.index, price_window.values, color="#1f77b4", linewidth=1.15, label="Synthetic TQQQ price")
    ax_price.axvspan(episode["peak"], episode["trough"], color="#d62728", alpha=0.12, label="Peak → trough")
    if pd.notna(episode["recovery"]):
        ax_price.axvspan(episode["trough"], episode["recovery"], color="#ff9896", alpha=0.08, label="Trough → recovery")
    for when, _label, color in [
        (episode["peak"], "equity peak", "black"),
        (episode["trough"], "DD trough", "#d62728"),
        (episode["recovery"], "recovery", "#2ca02c"),
    ]:
        if pd.notna(when):
            ax_price.axvline(when, color=color, linestyle="--", linewidth=0.9, alpha=0.8)

    y_min, y_max = float(price_window.min()), float(price_window.max())
    y_range = y_max - y_min if y_max > y_min else y_max if y_max else 1.0

    for j, row in trades_window.sort_values("trade_id").reset_index(drop=True).iterrows():
        entry_ts = row["entry_execution_close"]
        exit_ts = row["exit_return_label"]
        ret_pct = row["sized_pre_tax_return"] * 100.0
        trade_id = int(row["trade_id"])
        color = "#2ca02c" if ret_pct >= 0 else "#d62728"
        if entry_ts in target_price.index and plot_start <= entry_ts <= plot_end:
            entry_price = float(target_price.loc[entry_ts])
            ax_price.scatter(entry_ts, entry_price, marker="^", color="#2ca02c", edgecolor="black", s=55, zorder=5)
            ax_price.annotate(
                f"T{trade_id} buy",
                xy=(entry_ts, entry_price),
                xytext=(4, 10 + (j % 3) * 8),
                textcoords="offset points",
                fontsize=6.7,
                color="#1b5e20",
                arrowprops=dict(arrowstyle="-", color="#1b5e20", lw=0.6),
            )
        if exit_ts in target_price.index and plot_start <= exit_ts <= plot_end:
            exit_price = float(target_price.loc[exit_ts])
            ax_price.scatter(exit_ts, exit_price, marker="v", color=color, edgecolor="black", s=55, zorder=5)
            ax_price.annotate(
                f"T{trade_id} sell\n{ret_pct:+.1f}%",
                xy=(exit_ts, exit_price),
                xytext=(4, -22 - (j % 3) * 9),
                textcoords="offset points",
                fontsize=6.7,
                color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
            )

    for _, row in pl_window.iterrows():
        ts = row["timestamp"]
        if ts in target_price.index:
            event_price = float(target_price.loc[ts])
        else:
            nearest = target_price.index[target_price.index.get_indexer([ts], method="nearest")[0]]
            event_price = float(target_price.loc[nearest])
        threshold_pct = int(round(float(row["threshold_gain"]) * 100))
        new_weight_pct = int(round(float(row["new_weight"]) * 100))
        trade_label = "" if pd.isna(row.get("trade_id")) else f"T{int(row['trade_id'])} "
        ax_price.scatter(ts, event_price, marker="*", color="#ffbf00", edgecolor="black", s=145, zorder=7)
        ax_price.annotate(
            f"{trade_label}+{threshold_pct}% → {new_weight_pct}%",
            xy=(ts, event_price),
            xytext=(6, 20),
            textcoords="offset points",
            fontsize=7.0,
            color="#8a5a00",
            bbox=dict(facecolor="white", edgecolor="#ffbf00", alpha=0.85, pad=1.5),
            arrowprops=dict(arrowstyle="-", color="#8a5a00", lw=0.7),
        )

    ax_price.set_ylabel("Synthetic TQQQ price")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.set_ylim(y_min - 0.08 * y_range, y_max + 0.18 * y_range)

    ax_dd.plot(dd_window.index, dd_window.values * 100.0, color="#d62728", linewidth=1.1, label="Strategy drawdown")
    ax_dd.fill_between(dd_window.index, dd_window.values * 100.0, 0.0, color="#d62728", alpha=0.15)
    ax_dd.axhline(-40.0, color="gray", linestyle="--", linewidth=0.8, label="-40%")
    ax_dd.axhline(-50.0, color="black", linestyle=":", linewidth=0.8, label="-50%")
    ax_dd.axvspan(episode["peak"], episode["trough"], color="#d62728", alpha=0.10)
    if pd.notna(episode["recovery"]):
        ax_dd.axvspan(episode["trough"], episode["recovery"], color="#ff9896", alpha=0.07)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Time")
    ax_dd.grid(True, alpha=0.25)
    ax_dd.legend(loc="lower left", fontsize=8)

    dd_id = int(episode["chronological_dd_id"])
    severity_rank = int(episode["severity_rank"])
    fig.suptitle(
        f"Current preferred + -40% peak stop: DD{dd_id} / severity rank {severity_rank} | "
        f"max DD {episode['max_drawdown'] * 100:.1f}%\n"
        f"Peak {episode['peak']} → trough {episode['trough']} → recovery {episode['recovery']}",
        fontsize=12,
    )

    summary = (
        f"Overlapping trades: {', '.join('T' + str(int(x)) for x in trades_window['trade_id'])}\n"
        f"Trade returns shown at sell markers. Stars = profit-lock hits.\n"
        f"Plot window: {plot_start:%Y-%m-%d} to {plot_end:%Y-%m-%d}"
    )
    ax_price.text(
        0.99,
        0.03,
        summary,
        transform=ax_price.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.3,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.86),
    )

    ensure_directory(output_path.parent)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return {
        "chronological_dd_id": dd_id,
        "severity_rank": severity_rank,
        "plot_path": str(output_path),
        "plot_start": plot_start,
        "plot_end": plot_end,
        "overlap_trade_ids": ",".join(str(int(x)) for x in trades_window["trade_id"]),
        "profit_lock_events_in_plot": len(pl_window),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    target_dir = resolve_path(config.root, args.target_raw_dir)
    benchmark_dir = resolve_path(config.root, args.benchmark_raw_dir)
    returns_path = resolve_path(config.root, args.returns_path)
    trades_path = resolve_path(config.root, args.trades_path)
    output_dir = resolve_path(config.root, args.output_dir)

    target = _load_price(target_dir / f"{args.target_ticker}.parquet", args.target_ticker)
    qqq = _load_price(benchmark_dir / f"{args.benchmark_ticker}.parquet", args.benchmark_ticker)
    common = target.index.intersection(qqq.index)
    target = target.loc[common]
    qqq = qqq.loc[common]

    returns_frame = pd.read_csv(returns_path, parse_dates=["date"]).set_index("date")
    if args.variant_name not in returns_frame.columns:
        raise ValueError(f"{args.variant_name} not found in {returns_path}")
    returns = returns_frame[args.variant_name].astype(float)

    trades = pd.read_csv(
        trades_path,
        parse_dates=["entry_return_label", "entry_execution_close", "exit_return_label"],
    ).sort_values("trade_id")

    params = dict(config.strategies.regime_switch)
    bars_per_day = int(params.get("intraday_bars_per_day", 6))
    raw_base, _ = no_daily_gate_hourly_ma_gate_signal(
        entry_price=qqq,
        exit_price=qqq,
        output_index=common,
        bars_per_day=bars_per_day,
        average_type=args.average_type,
        macd_unit=args.macd_unit,
        entry_confirm_bars=2,
        exit_confirm_bars=3,
        exit_ma_days=200,
    )
    raw_stop, _ = raw_with_peak_drawdown_stop(raw_base.rename(args.target_ticker), target, stop_drawdown=0.40)
    pl_events = profit_lock_events(raw_stop.rename(args.target_ticker), target, trades)

    episodes = drawdown_episodes(returns, threshold=args.threshold)
    worst = episodes.sort_values("max_drawdown", ascending=True).head(args.top_n).copy()
    worst = worst.sort_values("peak").reset_index(drop=True)

    ensure_directory(output_dir)
    plot_rows: list[dict[str, Any]] = []
    for _, episode in worst.iterrows():
        file_name = (
            f"{args.output_prefix}_chronDD{int(episode['chronological_dd_id']):02d}_"
            f"rank{int(episode['severity_rank']):02d}_{episode['peak']:%Y%m%d}_{episode['trough']:%Y%m%d}.png"
        )
        plot_rows.append(
            plot_episode(
                episode=episode,
                returns=returns,
                target_price=target,
                trades=trades,
                pl_events=pl_events,
                output_path=output_dir / file_name,
                extend_months=args.extend_months,
            )
        )

    tables_dir = config.reports.tables_dir
    ensure_directory(tables_dir)
    episodes_path = tables_dir / "preferred_stop40_worst7_dd_episodes_for_plots.csv"
    pl_events_path = tables_dir / "preferred_stop40_profit_lock_events.csv"
    plot_index_path = tables_dir / "preferred_stop40_worst7_dd_plot_index.csv"
    worst.to_csv(episodes_path, index=False)
    pl_events.to_csv(pl_events_path, index=False)
    pd.DataFrame(plot_rows).to_csv(plot_index_path, index=False)

    print(f"Saved {len(plot_rows)} plots to {output_dir}")
    print(f"Episode table: {episodes_path}")
    print(f"Profit-lock events: {pl_events_path}")
    print(f"Plot index: {plot_index_path}")
    print(pd.DataFrame(plot_rows).to_string(index=False))


if __name__ == "__main__":
    main()
