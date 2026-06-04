#!/usr/bin/env python
"""Inspect trade-level behavior for the best bear-whipsaw candidate."""

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

from trend_following.bear_whipsaw import bear_market_features  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

BASELINE = "preferred_q100_baseline"
BEST = "bear_reentry_buf1_slope20_20gt50"
TARGET = "QQQ_3X_CALC"
QQQ = "QQQ"
PERIODS = {
    "2004_2005": ("2004-01-01", "2005-12-31 23:59:59"),
    "2007_2009": ("2007-01-01", "2009-12-31 23:59:59"),
    "2010": ("2010-01-01", "2010-12-31 23:59:59"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument(
        "--weights-path",
        default="reports/tables/preferred_bear_whipsaw_experiments_weights.parquet",
    )
    parser.add_argument(
        "--returns-path",
        default="reports/tables/preferred_bear_whipsaw_experiments_returns.csv",
    )
    parser.add_argument(
        "--diagnostics-path",
        default="reports/tables/preferred_bear_whipsaw_experiments_diagnostics.parquet",
    )
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--candidate", default=BEST)
    parser.add_argument("--output-prefix", default="preferred_bear_whipsaw_trade_behavior")
    return parser.parse_args()


def _load_close(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def _strategy_weight(weights: pd.DataFrame, strategy: str, ticker: str) -> pd.Series:
    if isinstance(weights.columns, pd.MultiIndex):
        return weights[(strategy, ticker)].astype(float).rename(strategy)
    return weights[strategy].astype(float).rename(strategy)


def _trade_intervals(weight: pd.Series) -> pd.DataFrame:
    active = weight.fillna(0.0).gt(0.0)
    starts = active & ~active.shift(1, fill_value=False)
    ends = ~active & active.shift(1, fill_value=False)
    start_times = list(weight.index[starts])
    end_times = list(weight.index[ends])
    if active.iloc[-1]:
        end_times.append(weight.index[-1])
    rows: list[dict[str, Any]] = []
    for i, start in enumerate(start_times, start=1):
        end = end_times[i - 1] if i - 1 < len(end_times) else weight.index[-1]
        rows.append({"trade_id": i, "entry": pd.Timestamp(start), "exit": pd.Timestamp(end)})
    return pd.DataFrame(rows)


def _trade_summary(
    *,
    strategy: str,
    period_name: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    weight: pd.Series,
    strategy_returns: pd.Series,
    target_price: pd.Series,
) -> pd.DataFrame:
    intervals = _trade_intervals(weight)
    rows: list[dict[str, Any]] = []
    for trade in intervals.itertuples(index=False):
        if trade.exit < period_start or trade.entry > period_end:
            continue
        window_start = max(trade.entry, period_start)
        window_end = min(trade.exit, period_end)
        trade_weight = weight.loc[(weight.index >= trade.entry) & (weight.index <= trade.exit)]
        price_window = target_price.loc[
            (target_price.index >= trade.entry) & (target_price.index <= trade.exit)
        ].dropna()
        period_returns = strategy_returns.loc[
            (strategy_returns.index >= window_start) & (strategy_returns.index <= window_end)
        ].fillna(0.0)
        if price_window.empty or period_returns.empty:
            continue
        entry_price = float(price_window.iloc[0])
        exit_price = float(price_window.iloc[-1])
        asset_return = exit_price / entry_price - 1.0
        asset_equity = price_window / entry_price
        asset_trade_dd = float((asset_equity / asset_equity.cummax() - 1.0).min())
        rows.append(
            {
                "period": period_name,
                "strategy": strategy,
                "trade_id": int(trade.trade_id),
                "entry": trade.entry,
                "exit": trade.exit,
                "calendar_days": int((trade.exit - trade.entry).days),
                "overlap_start": window_start,
                "overlap_end": window_end,
                "weight_min": float(trade_weight.min()),
                "weight_max": float(trade_weight.max()),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "synthetic_asset_return": asset_return,
                "synthetic_trade_peak_dd": asset_trade_dd,
                "after_tax_strategy_return_in_period_overlap": float(
                    (1.0 + period_returns).prod() - 1.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _period_summary(
    *,
    strategy: str,
    period_name: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    weight: pd.Series,
    strategy_returns: pd.Series,
) -> dict[str, Any]:
    returns = strategy_returns.loc[
        (strategy_returns.index >= period_start) & (strategy_returns.index <= period_end)
    ].fillna(0.0)
    sample_weight = weight.loc[(weight.index >= period_start) & (weight.index <= period_end)]
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    transitions = sample_weight.diff().fillna(sample_weight).ne(0.0) & sample_weight.ne(
        sample_weight.shift(1).fillna(0.0)
    )
    return {
        "period": period_name,
        "strategy": strategy,
        "start": period_start,
        "end": period_end,
        "final_return": float(equity.iloc[-1] - 1.0) if not equity.empty else np.nan,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else np.nan,
        "exposure": float(sample_weight.gt(0).mean()) if not sample_weight.empty else np.nan,
        "avg_weight": float(sample_weight.mean()) if not sample_weight.empty else np.nan,
        "position_changes": int(transitions.sum()),
    }


def _blocked_summary(
    diagnostics: pd.DataFrame,
    *,
    strategy: str,
    period_name: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> dict[str, Any]:
    diag = diagnostics[strategy] if isinstance(diagnostics.columns, pd.MultiIndex) else diagnostics
    window = diag.loc[(diag.index >= period_start) & (diag.index <= period_end)]
    blocked = window.loc[
        window.get("blocked_entry", pd.Series(False, index=window.index)).astype(bool)
    ]
    releases = window.loc[
        window.get("release_confirmation", pd.Series(False, index=window.index)).astype(bool)
    ]
    return {
        "period": period_name,
        "strategy": strategy,
        "blocked_entry_bars": int(len(blocked)),
        "first_blocked_entry": blocked.index[0] if not blocked.empty else pd.NaT,
        "last_blocked_entry": blocked.index[-1] if not blocked.empty else pd.NaT,
        "release_confirmation_bars": int(len(releases)),
        "first_release_confirmation": releases.index[0] if not releases.empty else pd.NaT,
        "last_release_confirmation": releases.index[-1] if not releases.empty else pd.NaT,
    }


def _blocked_intervals(
    diagnostics: pd.DataFrame,
    *,
    strategy: str,
    period_name: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    """Return contiguous blocked-entry spans for one period."""
    diag = diagnostics[strategy] if isinstance(diagnostics.columns, pd.MultiIndex) else diagnostics
    window = diag.loc[(diag.index >= period_start) & (diag.index <= period_end)]
    blocked = window.get("blocked_entry", pd.Series(False, index=window.index)).astype(bool)
    rows: list[dict[str, Any]] = []
    in_span = False
    span_start = pd.NaT
    last = pd.NaT
    count = 0
    for timestamp, is_blocked in blocked.items():
        if is_blocked and not in_span:
            in_span = True
            span_start = pd.Timestamp(timestamp)
            count = 1
        elif is_blocked:
            count += 1
        elif in_span:
            rows.append(
                {
                    "period": period_name,
                    "strategy": strategy,
                    "blocked_start": span_start,
                    "blocked_end": last,
                    "blocked_bars": count,
                }
            )
            in_span = False
            count = 0
        last = pd.Timestamp(timestamp)
    if in_span:
        rows.append(
            {
                "period": period_name,
                "strategy": strategy,
                "blocked_start": span_start,
                "blocked_end": last,
                "blocked_bars": count,
            }
        )
    return pd.DataFrame(rows)


def _plot_period(
    *,
    period_name: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    qqq: pd.Series,
    features: pd.DataFrame,
    returns: pd.DataFrame,
    weights: dict[str, pd.Series],
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    idx = (qqq.index >= period_start) & (qqq.index <= period_end)
    qqq_window = qqq.loc[idx]
    features_window = features.loc[idx]
    returns_window = returns.loc[
        (returns.index >= period_start) & (returns.index <= period_end),
        list(weights),
    ].fillna(0.0)
    diag = diagnostics[BEST].loc[
        (diagnostics.index >= period_start) & (diagnostics.index <= period_end)
    ]
    blocked = diag.loc[diag["blocked_entry"].astype(bool)]

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    axes[0].plot(qqq_window.index, qqq_window, label="QQQ", color="#1f77b4", linewidth=1.1)
    axes[0].plot(
        features_window.index, features_window["ma_long"], label="QQQ 200MA", color="#222222"
    )
    axes[0].plot(
        features_window.index, features_window["ma_medium"], label="QQQ 50MA", color="#ff7f0e"
    )
    axes[0].plot(
        features_window.index, features_window["ma_short"], label="QQQ 20MA", color="#2ca02c"
    )
    if not blocked.empty:
        axes[0].scatter(
            blocked.index,
            qqq.reindex(blocked.index),
            s=18,
            color="red",
            alpha=0.8,
            label="candidate blocked raw entry",
            zorder=5,
        )
    axes[0].set_title(f"{period_name}: QQQ and moving averages")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    for name, weight in weights.items():
        sample = weight.loc[(weight.index >= period_start) & (weight.index <= period_end)]
        axes[1].step(sample.index, sample, where="post", label=name, linewidth=1.2)
    axes[1].set_title("Executable weights")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    for name in weights:
        equity = (1.0 + returns_window[name]).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        axes[2].plot(drawdown.index, drawdown * 100.0, label=f"{name} DD", linewidth=1.1)
    axes[2].set_title("Period-local drawdown")
    axes[2].set_ylabel("Drawdown (%)")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)
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

    qqq = _load_close(resolve_path(config.root, args.benchmark_raw_dir) / f"{QQQ}.parquet", QQQ)
    target = _load_close(
        resolve_path(config.root, args.target_raw_dir) / f"{TARGET}.parquet", TARGET
    )
    weights_frame = pd.read_parquet(resolve_path(config.root, args.weights_path))
    diagnostics = pd.read_parquet(resolve_path(config.root, args.diagnostics_path))
    returns = pd.read_csv(
        resolve_path(config.root, args.returns_path), parse_dates=["date"]
    ).set_index("date")

    common = (
        qqq.index.intersection(target.index)
        .intersection(weights_frame.index)
        .intersection(returns.index)
    )
    qqq = qqq.loc[common]
    target = target.loc[common]
    weights = {
        args.baseline: _strategy_weight(weights_frame.loc[common], args.baseline, TARGET),
        args.candidate: _strategy_weight(weights_frame.loc[common], args.candidate, TARGET),
    }
    returns = returns.loc[common, [args.baseline, args.candidate]]
    features = bear_market_features(
        qqq, bars_per_day=int(config.strategies.regime_switch.get("intraday_bars_per_day", 6))
    )

    trade_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    block_interval_tables: list[pd.DataFrame] = []
    figure_paths: list[str] = []
    for period_name, (start, end) in PERIODS.items():
        period_start = pd.Timestamp(start)
        period_end = pd.Timestamp(end)
        for strategy, weight in weights.items():
            trade_tables.append(
                _trade_summary(
                    strategy=strategy,
                    period_name=period_name,
                    period_start=period_start,
                    period_end=period_end,
                    weight=weight,
                    strategy_returns=returns[strategy],
                    target_price=target,
                )
            )
            summary_rows.append(
                _period_summary(
                    strategy=strategy,
                    period_name=period_name,
                    period_start=period_start,
                    period_end=period_end,
                    weight=weight,
                    strategy_returns=returns[strategy],
                )
            )
        block_rows.append(
            _blocked_summary(
                diagnostics,
                strategy=args.candidate,
                period_name=period_name,
                period_start=period_start,
                period_end=period_end,
            )
        )
        block_interval_tables.append(
            _blocked_intervals(
                diagnostics,
                strategy=args.candidate,
                period_name=period_name,
                period_start=period_start,
                period_end=period_end,
            )
        )
        figure_path = figures_dir / f"{args.output_prefix}_{period_name}.png"
        _plot_period(
            period_name=period_name,
            period_start=period_start,
            period_end=period_end,
            qqq=qqq,
            features=features,
            returns=returns,
            weights=weights,
            diagnostics=diagnostics,
            output_path=figure_path,
        )
        figure_paths.append(str(figure_path))

    trades = pd.concat([table for table in trade_tables if not table.empty], ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    blocks = pd.DataFrame(block_rows)
    block_intervals = pd.concat(
        [table for table in block_interval_tables if not table.empty],
        ignore_index=True,
    )
    trades.to_csv(tables_dir / f"{args.output_prefix}_trades.csv", index=False)
    summary.to_csv(tables_dir / f"{args.output_prefix}_period_summary.csv", index=False)
    blocks.to_csv(tables_dir / f"{args.output_prefix}_blocked_entries.csv", index=False)
    block_intervals.to_csv(
        tables_dir / f"{args.output_prefix}_blocked_entry_intervals.csv",
        index=False,
    )

    print("Period summary:")
    print(summary.to_string(index=False))
    print("\nBlocked entries:")
    print(blocks.to_string(index=False))
    print("\nBlocked entry intervals:")
    print(block_intervals.to_string(index=False))
    print("\nTrade summary:")
    print(
        trades[
            [
                "period",
                "strategy",
                "trade_id",
                "entry",
                "exit",
                "calendar_days",
                "weight_min",
                "weight_max",
                "synthetic_asset_return",
                "synthetic_trade_peak_dd",
                "after_tax_strategy_return_in_period_overlap",
            ]
        ].to_string(index=False)
    )
    print("\nFigures:")
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
