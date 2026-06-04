#!/usr/bin/env python
"""Focused trade-timing check for MACD variants around worst DD periods."""

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

from plot_preferred_peak_stop_worst7_drawdowns import find_drawdown_episodes  # noqa: E402
from preferred_cv_utils import (  # noqa: E402
    CURRENT_PREFERRED_NAME,
    OFFICIAL_EVALUATION_START,
    TARGET_TICKER,
    load_cv_data,
    make_eval_args,
    simulate_weight_path,
)
from trend_following.config import load_config  # noqa: E402
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

NAMES = [CURRENT_PREFERRED_NAME, "macd_signal_8d", "macd_slow_24d"]
DISPLAY = {
    CURRENT_PREFERRED_NAME: "current_preferred",
    "macd_signal_8d": "macd_signal_8d",
    "macd_slow_24d": "macd_slow_24d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default=TARGET_TICKER)
    parser.add_argument("--benchmark-ticker", default="QQQ")
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--months-before", type=int, default=2)
    parser.add_argument("--months-after", type=int, default=2)
    parser.add_argument("--output-prefix", default="macd_variant_worst_dd")
    return parser.parse_args()


def equity(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def drawdown_from_returns(returns: pd.Series) -> pd.Series:
    eq = equity(returns)
    return eq / eq.cummax() - 1.0


def load_weights(config: Any, args: argparse.Namespace, names: list[str]) -> dict[str, pd.Series]:
    path = resolve_path(config.root, "reports/tables/preferred_start_date_cv_weights.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing cached CV weights: {path}. Run scripts/run_preferred_start_date_cv.py first.")
    stored = pd.read_parquet(path)
    out: dict[str, pd.Series] = {}
    for name in names:
        if name not in stored.columns.get_level_values(0):
            raise KeyError(f"Strategy {name!r} not found in {path}")
        series = stored[name][args.target_ticker].astype(float).rename(name)
        out[name] = series
    return out


def trade_events(weights: pd.Series, returns: pd.Series) -> pd.DataFrame:
    w = weights.reindex(returns.index).fillna(0.0).astype(float)
    before = w.shift(1).fillna(0.0)
    delta = w - before
    mask = delta.abs() > 1e-10
    rows = []
    for timestamp in w.index[mask]:
        b = float(before.loc[timestamp])
        a = float(w.loc[timestamp])
        d = float(delta.loc[timestamp])
        if b <= 1e-10 and a > 1e-10:
            event_type = "entry"
        elif b > 1e-10 and a <= 1e-10:
            event_type = "exit"
        elif d > 0:
            event_type = "add"
        else:
            event_type = "reduce"
        rows.append(
            {
                "timestamp": pd.Timestamp(timestamp),
                "event_type": event_type,
                "weight_before": b,
                "weight_after": a,
                "weight_delta": d,
            }
        )
    return pd.DataFrame(rows)


def summarize_window(
    *,
    name: str,
    episode_rank: int,
    focus_episode: pd.Series,
    simulation: dict[str, pd.Series],
    weights: pd.Series,
    events: pd.DataFrame,
    qqq: pd.Series,
    target: pd.Series,
    qqq_ma: pd.Series,
    months_before: int,
    months_after: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    peak = pd.Timestamp(focus_episode["peak"])
    trough = pd.Timestamp(focus_episode["trough"])
    recovery = pd.Timestamp(focus_episode["recovery"]) if pd.notna(focus_episode["recovery"]) else pd.Timestamp(simulation["after_tax"].index.max())
    window_start = peak - pd.DateOffset(months=months_before)
    window_end = recovery + pd.DateOffset(months=months_after)

    returns = simulation["after_tax"]
    dd = drawdown_from_returns(returns)
    strategy_eq = equity(returns)
    w = weights.reindex(returns.index).fillna(0.0)

    # Variant-specific worst DD inside the current-preferred focus window.
    win = dd.loc[(dd.index >= peak) & (dd.index <= recovery)]
    if not win.empty:
        variant_trough = pd.Timestamp(win.idxmin())
        variant_window_dd = float(win.min())
    else:
        variant_trough = pd.NaT
        variant_window_dd = np.nan

    peak_loc = returns.index.get_indexer([peak], method="nearest")[0]
    trough_loc = returns.index.get_indexer([trough], method="nearest")[0]
    peak_ts = returns.index[peak_loc]
    trough_ts = returns.index[trough_loc]

    ev = events.loc[(events["timestamp"] >= window_start) & (events["timestamp"] <= window_end)].copy()
    ev_after_peak = ev.loc[ev["timestamp"] >= peak]
    first_reduce = ev_after_peak.loc[ev_after_peak["event_type"].isin(["reduce", "exit"])].head(1)
    first_exit = ev_after_peak.loc[ev_after_peak["event_type"].eq("exit")].head(1)
    first_reentry_after_trough = ev.loc[(ev["timestamp"] > trough) & ev["event_type"].isin(["entry", "add"])].head(1)

    def event_info(row: pd.DataFrame, prefix: str) -> dict[str, Any]:
        if row.empty:
            return {
                f"{prefix}_time": pd.NaT,
                f"{prefix}_type": "",
                f"{prefix}_lag_calendar_days_from_peak": np.nan,
                f"{prefix}_strategy_dd": np.nan,
                f"{prefix}_weight_after": np.nan,
                f"{prefix}_qqq_vs_200ma": np.nan,
                f"{prefix}_target_return_from_peak": np.nan,
            }
        r = row.iloc[0]
        ts = pd.Timestamp(r["timestamp"])
        qval = float(qqq.reindex(qqq.index.union([ts])).sort_index().ffill().loc[ts]) if ts not in qqq.index else float(qqq.loc[ts])
        mval = float(qqq_ma.reindex(qqq_ma.index.union([ts])).sort_index().ffill().loc[ts]) if ts not in qqq_ma.index else float(qqq_ma.loc[ts])
        target_peak = float(target.reindex(target.index.union([peak_ts])).sort_index().ffill().loc[peak_ts])
        target_event = float(target.reindex(target.index.union([ts])).sort_index().ffill().loc[ts])
        return {
            f"{prefix}_time": ts,
            f"{prefix}_type": str(r["event_type"]),
            f"{prefix}_lag_calendar_days_from_peak": (ts.normalize() - peak_ts.normalize()).days,
            f"{prefix}_strategy_dd": float(dd.reindex(dd.index.union([ts])).sort_index().ffill().loc[ts]),
            f"{prefix}_weight_after": float(r["weight_after"]),
            f"{prefix}_qqq_vs_200ma": qval / mval - 1.0 if mval > 0 else np.nan,
            f"{prefix}_target_return_from_peak": target_event / target_peak - 1.0 if target_peak > 0 else np.nan,
        }

    row = {
        "focus_rank": episode_rank,
        "focus_peak": peak,
        "focus_trough": trough,
        "focus_recovery": focus_episode["recovery"],
        "focus_current_preferred_dd": float(focus_episode["max_drawdown"]),
        "strategy": DISPLAY[name],
        "weight_at_focus_peak": float(w.iloc[peak_loc]),
        "weight_at_focus_trough": float(w.iloc[trough_loc]),
        "strategy_dd_at_focus_trough": float(dd.iloc[trough_loc]),
        "strategy_equity_at_focus_peak": float(strategy_eq.iloc[peak_loc]),
        "strategy_equity_at_focus_trough": float(strategy_eq.iloc[trough_loc]),
        "strategy_worst_dd_inside_focus_episode": variant_window_dd,
        "strategy_worst_dd_time_inside_focus_episode": variant_trough,
        "events_in_extended_window": int(len(ev)),
        "entries_or_adds_in_peak_to_trough": int(
            len(ev.loc[(ev["timestamp"] >= peak) & (ev["timestamp"] <= trough) & ev["event_type"].isin(["entry", "add"])])
        ),
        "reduces_or_exits_in_peak_to_trough": int(
            len(ev.loc[(ev["timestamp"] >= peak) & (ev["timestamp"] <= trough) & ev["event_type"].isin(["reduce", "exit"])])
        ),
    }
    row.update(event_info(first_reduce, "first_reduce_or_exit_after_peak"))
    row.update(event_info(first_exit, "first_full_exit_after_peak"))
    if first_reentry_after_trough.empty:
        row.update({"first_reentry_after_trough_time": pd.NaT, "first_reentry_after_trough_lag_calendar_days": np.nan})
    else:
        ts = pd.Timestamp(first_reentry_after_trough.iloc[0]["timestamp"])
        row.update(
            {
                "first_reentry_after_trough_time": ts,
                "first_reentry_after_trough_lag_calendar_days": (ts.normalize() - trough_ts.normalize()).days,
            }
        )

    if not ev.empty:
        ev = ev.assign(
            focus_rank=episode_rank,
            strategy=DISPLAY[name],
            focus_peak=peak,
            focus_trough=trough,
            strategy_dd_at_event=ev["timestamp"].map(lambda ts: float(dd.reindex(dd.index.union([ts])).sort_index().ffill().loc[ts])),
            qqq_vs_200ma_at_event=ev["timestamp"].map(
                lambda ts: float(qqq.reindex(qqq.index.union([ts])).sort_index().ffill().loc[ts])
                / float(qqq_ma.reindex(qqq_ma.index.union([ts])).sort_index().ffill().loc[ts])
                - 1.0
            ),
        )
    return row, ev


def plot_focus_episodes(
    *,
    focus: pd.DataFrame,
    simulations: dict[str, dict[str, pd.Series]],
    weights: dict[str, pd.Series],
    output_path: Path,
) -> None:
    top = focus.drop_duplicates("focus_rank").sort_values("focus_rank").head(6)
    fig, axes = plt.subplots(len(top), 2, figsize=(15, max(3.2 * len(top), 8)), sharex=False)
    if len(top) == 1:
        axes = np.array([axes])
    for i, ep in enumerate(top.itertuples(index=False)):
        start = pd.Timestamp(ep.focus_peak) - pd.DateOffset(months=2)
        end_base = pd.Timestamp(ep.focus_recovery) if pd.notna(ep.focus_recovery) else pd.Timestamp(ep.focus_trough)
        end = end_base + pd.DateOffset(months=2)
        ax_eq, ax_w = axes[i]
        for name in NAMES:
            label = DISPLAY[name]
            ret = simulations[name]["after_tax"]
            eq = equity(ret)
            # normalize at window start for timing comparison readability.
            ewin = eq.loc[(eq.index >= start) & (eq.index <= end)]
            if ewin.empty:
                continue
            (ewin / float(ewin.iloc[0])).plot(ax=ax_eq, linewidth=1.05, label=label)
            weights[name].reindex(ewin.index).fillna(0.0).plot(ax=ax_w, linewidth=1.0, label=label)
        for ax in (ax_eq, ax_w):
            ax.axvline(pd.Timestamp(ep.focus_peak), color="black", linestyle="--", linewidth=0.9)
            ax.axvline(pd.Timestamp(ep.focus_trough), color="red", linestyle=":", linewidth=0.9)
            ax.grid(True, alpha=0.22)
        ax_eq.set_title(f"Focus DD #{int(ep.focus_rank)}: {pd.Timestamp(ep.focus_peak).date()} → {pd.Timestamp(ep.focus_trough).date()}")
        ax_eq.set_ylabel("Equity normalized")
        ax_w.set_ylabel("Target weight")
        ax_w.set_ylim(-0.05, 1.05)
        if i == 0:
            ax_eq.legend(fontsize=7)
            ax_w.legend(fontsize=7)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tables_dir = resolve_path(config.root, "reports/tables")
    figures_dir = resolve_path(config.root, "reports/figures")
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

    target, qqq, returns, bars_per_day = load_cv_data(config, args)
    eval_args = make_eval_args(args)
    weights = load_weights(config, args, NAMES)
    simulations = {
        name: simulate_weight_path(weights=weights[name].to_frame(args.target_ticker), returns=returns, config=config, args=eval_args)
        for name in NAMES
    }
    qqq_ma = qqq.rolling(200 * bars_per_day, min_periods=200 * bars_per_day).mean()

    episode_rows = []
    for name in NAMES:
        ret = simulations[name]["after_tax"].loc[lambda s: s.index >= OFFICIAL_EVALUATION_START]
        eps = find_drawdown_episodes(ret).head(max(args.top_n, 8)).copy()
        eps.insert(0, "strategy", DISPLAY[name])
        eps.insert(1, "rank", range(1, len(eps) + 1))
        episode_rows.append(eps)
    episodes = pd.concat(episode_rows, ignore_index=True)

    current_eps = episodes.loc[episodes["strategy"].eq(DISPLAY[CURRENT_PREFERRED_NAME])].head(args.top_n).copy()
    timing_rows = []
    event_rows = []
    event_by_name = {name: trade_events(weights[name], simulations[name]["after_tax"]) for name in NAMES}
    for ep in current_eps.itertuples(index=False):
        ep_series = pd.Series(ep._asdict())
        for name in NAMES:
            row, ev = summarize_window(
                name=name,
                episode_rank=int(ep.rank),
                focus_episode=ep_series,
                simulation=simulations[name],
                weights=weights[name],
                events=event_by_name[name],
                qqq=qqq,
                target=target,
                qqq_ma=qqq_ma,
                months_before=args.months_before,
                months_after=args.months_after,
            )
            timing_rows.append(row)
            if not ev.empty:
                event_rows.append(ev)
    timing = pd.DataFrame(timing_rows)
    events = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()

    episodes_path = tables_dir / f"{args.output_prefix}_episodes_by_strategy.csv"
    timing_path = tables_dir / f"{args.output_prefix}_timing_focus.csv"
    events_path = tables_dir / f"{args.output_prefix}_trade_events_focus.csv"
    episodes.to_csv(episodes_path, index=False)
    timing.to_csv(timing_path, index=False)
    events.to_csv(events_path, index=False)

    fig_path = figures_dir / f"{args.output_prefix}_timing_focus.png"
    plot_focus_episodes(focus=timing, simulations=simulations, weights=weights, output_path=fig_path)

    print(f"Saved {episodes_path}")
    print(f"Saved {timing_path}")
    print(f"Saved {events_path}")
    print(f"Saved {fig_path}")
    cols = [
        "focus_rank",
        "focus_peak",
        "focus_trough",
        "focus_current_preferred_dd",
        "strategy",
        "weight_at_focus_peak",
        "weight_at_focus_trough",
        "strategy_dd_at_focus_trough",
        "strategy_worst_dd_inside_focus_episode",
        "first_reduce_or_exit_after_peak_time",
        "first_reduce_or_exit_after_peak_type",
        "first_reduce_or_exit_after_peak_lag_calendar_days_from_peak",
        "first_reduce_or_exit_after_peak_strategy_dd",
        "first_full_exit_after_peak_time",
        "first_full_exit_after_peak_lag_calendar_days_from_peak",
        "first_full_exit_after_peak_strategy_dd",
        "first_reentry_after_trough_time",
        "first_reentry_after_trough_lag_calendar_days",
        "entries_or_adds_in_peak_to_trough",
        "reduces_or_exits_in_peak_to_trough",
    ]
    print(timing[cols].to_string(index=False))


if __name__ == "__main__":
    main()
