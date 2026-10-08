#!/usr/bin/env python
"""Test a +100% split-to-MACD sleeve overlay on the preferred strategy.

Question tested:
- Start from the current preferred strategy plus a synthetic-3x 40% trade-peak
  stop.
- When a base trade first reaches +100% unrealized synthetic-3x return, split
  exposure into two halves:
    1. one half keeps following the preferred strategy, including the
       +300% -> 75% and +400% -> 50% profit-lock sizing inside that half;
    2. the other half switches to a faster MACD sleeve, exiting when QQQ MACD
       histogram is no longer positive and re-entering when it is positive
       again, until the QQQ 200MA/base-trade regime ends.

No-lookahead convention:
- The +100% split trigger and MACD sleeve signals are observed at an hourly
  close.
- The resulting raw target weights are passed through executable_weights, so
  they are shifted before they can earn returns.
"""

from __future__ import annotations

import argparse
import json
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

from run_preferred_profit_lock_stop_exit_comparison import (  # noqa: E402
    PROFIT_LOCK_SCHEME,
    raw_with_peak_drawdown_stop,
)
from run_tqqq_cash_yield_candidate_comparison import (  # noqa: E402
    simulate_after_tax_portfolio_with_cash_yield,
)
from run_tqqq_daily_gate_ablation import no_daily_gate_hourly_ma_gate_signal  # noqa: E402
from run_tqqq_entry_signal_comparison import (  # noqa: E402
    _confirmed,
    _days_to_bars,
    _drawdown,
    _equity,
    _returns_from_prices,
    executable_weights,
    macd_components,
)
from run_tqqq_macd_entry_experiments import count_profit_lock_hits  # noqa: E402
from run_tqqq_position_risk_sizing_experiments import drawdown_episode_count  # noqa: E402
from run_tqqq_tiered_sizing_experiments import trade_profit_lock_tiers  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.metrics import calculate_metrics, metrics_to_frame  # noqa: E402
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default="QQQ_3X_CALC")
    parser.add_argument("--benchmark-ticker", default="QQQ")
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--split-trigger-return", type=float, default=1.0)
    parser.add_argument("--stop-drawdown", type=float, default=0.40)
    parser.add_argument("--macd-exit-confirm-bars", type=int, default=2)
    parser.add_argument("--macd-entry-confirm-bars", type=int, default=2)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--output-prefix", default="preferred_split100_macd_sleeve")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def _add_dd_counts(metrics: dict[str, Any], returns: pd.Series) -> None:
    metrics["drawdown_episodes_gt_20pct"] = drawdown_episode_count(returns, threshold=-0.20)
    metrics["drawdown_episodes_gt_30pct"] = drawdown_episode_count(returns, threshold=-0.30)
    metrics["drawdown_episodes_gt_40pct"] = drawdown_episode_count(returns, threshold=-0.40)
    metrics["drawdown_episodes_gt_50pct"] = drawdown_episode_count(returns, threshold=-0.50)
    metrics["dd_episodes_gt_20_30_40_50pct"] = (
        f"{metrics['drawdown_episodes_gt_20pct']}/"
        f"{metrics['drawdown_episodes_gt_30pct']}/"
        f"{metrics['drawdown_episodes_gt_40pct']}/"
        f"{metrics['drawdown_episodes_gt_50pct']}"
    )


def macd_entry_exit_flags(
    qqq_price: pd.Series,
    *,
    bars_per_day: int,
    average_type: str,
    macd_unit: str,
    entry_confirm_bars: int,
    exit_confirm_bars: int,
) -> pd.DataFrame:
    """Return confirmed MACD entry/exit flags for the fast sleeve."""
    if macd_unit == "days":
        fast_window = _days_to_bars(12, bars_per_day)
        slow_window = _days_to_bars(26, bars_per_day)
        signal_window = _days_to_bars(9, bars_per_day)
    elif macd_unit == "bars":
        fast_window = 12
        slow_window = 26
        signal_window = 9
    else:
        raise ValueError("macd_unit must be 'days' or 'bars'")

    components = macd_components(
        qqq_price,
        fast_window=fast_window,
        slow_window=slow_window,
        signal_window=signal_window,
        average_type=average_type,
    )
    hist = components["macd_hist"]
    components["macd_entry_flag"] = _confirmed(hist.gt(0.0), entry_confirm_bars)
    components["macd_exit_flag"] = _confirmed(hist.le(0.0), exit_confirm_bars)
    return components


def split100_macd_sleeve_weight(
    *,
    base_raw: pd.Series,
    stopped_raw: pd.Series,
    preferred_weight: pd.Series,
    target_price: pd.Series,
    macd_entry_flag: pd.Series,
    macd_exit_flag: pd.Series,
    split_trigger_return: float,
    apply_stop_to_fast_sleeve: bool,
    defer_macd_exit_until_after_trigger_bar: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build raw total weights for the split-to-MACD-sleeve overlay.

    ``preferred_weight`` is the full-size current preferred raw weight after
    profit-lock sizing and after the 40% stop has been applied. Once the split
    trigger occurs, only half of that preferred weight is kept.
    """
    index = base_raw.index
    base = base_raw.reindex(index).fillna(0.0).astype(float)
    stopped = stopped_raw.reindex(index).fillna(0.0).astype(float)
    preferred = preferred_weight.reindex(index).fillna(0.0).astype(float)
    price = target_price.reindex(index).astype(float)
    macd_entry = macd_entry_flag.reindex(index).fillna(False).astype(bool)
    macd_exit = macd_exit_flag.reindex(index).fillna(False).astype(bool)

    values: list[float] = []
    preferred_half_values: list[float] = []
    macd_half_values: list[float] = []
    activated_values: list[bool] = []
    fast_state_values: list[float] = []
    split_triggers: list[bool] = []
    entry_prices: list[float] = []
    trade_returns: list[float] = []

    in_base_trade = False
    activated = False
    fast_state = 0.0
    entry_price = np.nan

    for timestamp in index:
        base_active = float(base.loc[timestamp]) > 0.0
        stopped_active = float(stopped.loc[timestamp]) > 0.0
        current_price = float(price.loc[timestamp]) if np.isfinite(price.loc[timestamp]) else np.nan
        trigger = False

        if not base_active or not np.isfinite(current_price):
            in_base_trade = False
            activated = False
            fast_state = 0.0
            entry_price = np.nan
            trade_return = np.nan
        else:
            if not in_base_trade:
                in_base_trade = True
                activated = False
                fast_state = 0.0
                entry_price = current_price

            trade_return = current_price / entry_price - 1.0 if entry_price > 0 else np.nan
            if (not activated) and stopped_active and np.isfinite(trade_return) and trade_return >= split_trigger_return:
                activated = True
                fast_state = 1.0
                trigger = True

            if activated and not (trigger and defer_macd_exit_until_after_trigger_bar):
                if bool(macd_exit.loc[timestamp]):
                    fast_state = 0.0
                elif bool(macd_entry.loc[timestamp]):
                    fast_state = 1.0

            if apply_stop_to_fast_sleeve and not stopped_active:
                fast_state = 0.0

        if not activated:
            preferred_half = float(preferred.loc[timestamp])
            macd_half = 0.0
        else:
            preferred_half = 0.5 * float(preferred.loc[timestamp])
            fast_allowed = stopped_active if apply_stop_to_fast_sleeve else base_active
            macd_half = 0.5 * fast_state if fast_allowed else 0.0

        values.append(preferred_half + macd_half)
        preferred_half_values.append(preferred_half)
        macd_half_values.append(macd_half)
        activated_values.append(bool(activated))
        fast_state_values.append(float(fast_state))
        split_triggers.append(bool(trigger))
        entry_prices.append(entry_price)
        trade_returns.append(trade_return)

    raw_weight = pd.Series(values, index=index, name=preferred_weight.name, dtype=float)
    diagnostics = pd.DataFrame(
        {
            "base_raw": base,
            "stopped_raw": stopped,
            "preferred_full_raw_weight": preferred,
            "split_overlay_raw_weight": raw_weight,
            "preferred_sleeve_raw_weight": preferred_half_values,
            "macd_sleeve_raw_weight": macd_half_values,
            "split_activated": activated_values,
            "split_trigger": split_triggers,
            "fast_macd_state": fast_state_values,
            "base_trade_entry_price": entry_prices,
            "base_trade_return": trade_returns,
            "macd_entry_flag": macd_entry,
            "macd_exit_flag": macd_exit,
        },
        index=index,
    )
    return raw_weight, diagnostics


def _plot(returns_by_name: dict[str, pd.Series], output_path: Path) -> None:
    selected = [
        "preferred_profit_lock_stop40",
        "split100_macd_sleeve_global_stop40",
        "split100_macd_sleeve_future_exit_global_stop40",
        "split100_macd_sleeve_branch_stop40",
        "base_no_lock_stop40",
        "QQQ_BH",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.7))
    for name in selected:
        if name in returns_by_name:
            _equity(returns_by_name[name]).plot(ax=axes[0], label=name, linewidth=1.15)
            _drawdown(returns_by_name[name]).plot(ax=axes[1], label=name, linewidth=1.15)
    axes[0].set_title("After-tax equity / growth of $1")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("After-tax drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("+100% split-to-MACD-sleeve overlay test")
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    target_dir = resolve_path(config.root, args.target_raw_dir)
    benchmark_dir = resolve_path(config.root, args.benchmark_raw_dir)

    target = _load_price(target_dir / f"{args.target_ticker}.parquet", args.target_ticker)
    qqq = _load_price(benchmark_dir / f"{args.benchmark_ticker}.parquet", args.benchmark_ticker)
    common = target.index.intersection(qqq.index)
    target_prices = target.loc[common].to_frame()
    qqq_prices = qqq.loc[common].to_frame()
    target_returns = _returns_from_prices(target_prices)
    qqq_returns = _returns_from_prices(qqq_prices)[args.benchmark_ticker]

    params = dict(config.strategies.regime_switch)
    bars_per_day = int(params.get("intraday_bars_per_day", 6))
    raw_base, base_diag = no_daily_gate_hourly_ma_gate_signal(
        entry_price=qqq_prices[args.benchmark_ticker],
        exit_price=qqq_prices[args.benchmark_ticker],
        output_index=common,
        bars_per_day=bars_per_day,
        average_type=args.average_type,
        macd_unit=args.macd_unit,
        entry_confirm_bars=2,
        exit_confirm_bars=3,
        exit_ma_days=200,
    )
    raw_base = raw_base.rename(args.target_ticker)
    stopped_raw, stop_diag = raw_with_peak_drawdown_stop(
        raw_base,
        target_prices[args.target_ticker],
        stop_drawdown=args.stop_drawdown,
    )
    stopped_raw = stopped_raw.rename(args.target_ticker)
    preferred_weight = trade_profit_lock_tiers(
        stopped_raw,
        target_prices[args.target_ticker],
        thresholds_to_weights=PROFIT_LOCK_SCHEME,
    ).rename(args.target_ticker)

    macd_diag = macd_entry_exit_flags(
        qqq_prices[args.benchmark_ticker],
        bars_per_day=bars_per_day,
        average_type=args.average_type,
        macd_unit=args.macd_unit,
        entry_confirm_bars=args.macd_entry_confirm_bars,
        exit_confirm_bars=args.macd_exit_confirm_bars,
    )

    split_global_stop, split_global_diag = split100_macd_sleeve_weight(
        base_raw=raw_base,
        stopped_raw=stopped_raw,
        preferred_weight=preferred_weight,
        target_price=target_prices[args.target_ticker],
        macd_entry_flag=macd_diag["macd_entry_flag"],
        macd_exit_flag=macd_diag["macd_exit_flag"],
        split_trigger_return=args.split_trigger_return,
        apply_stop_to_fast_sleeve=True,
    )
    split_branch_stop, split_branch_diag = split100_macd_sleeve_weight(
        base_raw=raw_base,
        stopped_raw=stopped_raw,
        preferred_weight=preferred_weight,
        target_price=target_prices[args.target_ticker],
        macd_entry_flag=macd_diag["macd_entry_flag"],
        macd_exit_flag=macd_diag["macd_exit_flag"],
        split_trigger_return=args.split_trigger_return,
        apply_stop_to_fast_sleeve=False,
    )
    split_global_future_exit, split_global_future_exit_diag = split100_macd_sleeve_weight(
        base_raw=raw_base,
        stopped_raw=stopped_raw,
        preferred_weight=preferred_weight,
        target_price=target_prices[args.target_ticker],
        macd_entry_flag=macd_diag["macd_entry_flag"],
        macd_exit_flag=macd_diag["macd_exit_flag"],
        split_trigger_return=args.split_trigger_return,
        apply_stop_to_fast_sleeve=True,
        defer_macd_exit_until_after_trigger_bar=True,
    )

    raw_variants: dict[str, pd.Series] = {
        "base_no_lock_stop40": stopped_raw,
        "preferred_profit_lock_stop40": preferred_weight,
        "split100_macd_sleeve_global_stop40": split_global_stop,
        "split100_macd_sleeve_branch_stop40": split_branch_stop,
        "split100_macd_sleeve_future_exit_global_stop40": split_global_future_exit,
    }

    metric_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.Series] = {}
    cash_weight_by_name: dict[str, pd.Series] = {}
    diagnostics_by_name: dict[str, pd.DataFrame] = {
        "base": base_diag,
        "stop": stop_diag,
        "macd": macd_diag,
        "split_global_stop": split_global_diag,
        "split_branch_stop": split_branch_diag,
        "split_global_future_exit": split_global_future_exit_diag,
    }

    for name, raw_weight in raw_variants.items():
        weights = executable_weights(raw_weight.to_frame(args.target_ticker), config=config).reindex(common).fillna(0.0)
        returns, taxes, turnover, cash_interest, cash_weight = simulate_after_tax_portfolio_with_cash_yield(
            target_returns[[args.target_ticker]],
            weights[[args.target_ticker]],
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            tax_rate=args.short_term_tax_rate,
            cash_annual_yield=args.cash_annual_yield,
            annualization=config.backtest.annualization,
            cash_interest_tax_rate=args.cash_interest_tax_rate,
        )
        metrics = calculate_metrics(
            returns,
            turnover=turnover,
            weights=weights.sum(axis=1),
            annualization=config.backtest.annualization,
        )
        _add_dd_counts(metrics, returns)
        split_diag = (
            split_global_diag
            if name == "split100_macd_sleeve_global_stop40"
            else split_branch_diag
            if name == "split100_macd_sleeve_branch_stop40"
            else split_global_future_exit_diag
            if name == "split100_macd_sleeve_future_exit_global_stop40"
            else None
        )
        metrics.update(
            {
                "name": name,
                "strategy": "preferred_split100_macd_sleeve",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "profit_lock_scheme": PROFIT_LOCK_SCHEME if name != "base_no_lock_stop40" else [],
                        "stop_drawdown": args.stop_drawdown,
                        "split_trigger_return": args.split_trigger_return
                        if name.startswith("split100")
                        else None,
                        "fast_sleeve_macd_exit_confirm_bars": args.macd_exit_confirm_bars,
                        "fast_sleeve_macd_entry_confirm_bars": args.macd_entry_confirm_bars,
                        "cash_annual_yield": args.cash_annual_yield,
                        "transaction_cost_bps": args.transaction_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "short_term_tax_rate": args.short_term_tax_rate,
                    },
                    sort_keys=True,
                ),
                "final_return": metrics["cumulative_return"],
                "average_cash_weight": float(cash_weight.mean()),
                "profit_lock_300_hit_count": (
                    float(count_profit_lock_hits(stopped_raw, target_prices[args.target_ticker], threshold=3.0))
                    if name != "base_no_lock_stop40"
                    else np.nan
                ),
                "profit_lock_400_hit_count": (
                    float(count_profit_lock_hits(stopped_raw, target_prices[args.target_ticker], threshold=4.0))
                    if name != "base_no_lock_stop40"
                    else np.nan
                ),
                "stop_trigger_count": int(stop_diag["stop_trigger"].sum()),
                "split_trigger_count": int(split_diag["split_trigger"].sum()) if split_diag is not None else np.nan,
                "fast_sleeve_trade_count": (
                    int((split_diag["macd_sleeve_raw_weight"].diff().abs().fillna(0.0) > 1e-12).sum())
                    if split_diag is not None
                    else np.nan
                ),
                "tax_paid_pct_initial_capital": float(taxes.sum()),
                "cash_interest_pct_initial_capital": float(cash_interest.sum()),
            }
        )
        metric_rows.append(metrics)
        returns_by_name[name] = returns
        weights_by_name[name] = weights.sum(axis=1)
        cash_weight_by_name[name] = cash_weight

    qqq_metrics = calculate_metrics(
        qqq_returns,
        weights=pd.Series(1.0, index=qqq_returns.index),
        annualization=config.backtest.annualization,
    )
    _add_dd_counts(qqq_metrics, qqq_returns)
    qqq_metrics.update(
        {
            "name": "QQQ_BH",
            "strategy": "benchmark",
            "segment": "full_sample",
            "parameters": "{}",
            "final_return": qqq_metrics["cumulative_return"],
            "average_cash_weight": 0.0,
            "profit_lock_300_hit_count": np.nan,
            "profit_lock_400_hit_count": np.nan,
            "stop_trigger_count": 0,
            "split_trigger_count": np.nan,
            "fast_sleeve_trade_count": np.nan,
            "tax_paid_pct_initial_capital": 0.0,
            "cash_interest_pct_initial_capital": 0.0,
        }
    )
    metric_rows.append(qqq_metrics)
    returns_by_name["QQQ_BH"] = qqq_returns

    metrics = metrics_to_frame(metric_rows)
    compact_cols = [
        "name",
        "final_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "number_of_trades",
        "exposure_percentage",
        "average_cash_weight",
        "dd_episodes_gt_20_30_40_50pct",
        "split_trigger_count",
        "fast_sleeve_trade_count",
        "profit_lock_300_hit_count",
        "profit_lock_400_hit_count",
        "stop_trigger_count",
    ]
    compact = metrics[compact_cols].copy().sort_values("annualized_return", ascending=False)

    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)
    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    compact_path = tables_dir / f"{args.output_prefix}_compact.csv"
    returns_path = tables_dir / f"{args.output_prefix}_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    cash_path = tables_dir / f"{args.output_prefix}_cash_weights.csv"
    diagnostics_path = tables_dir / f"{args.output_prefix}_diagnostics.parquet"
    plot_path = figures_dir / f"{args.output_prefix}_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    compact.to_csv(compact_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(weights_by_name).to_csv(weights_path)
    pd.DataFrame(cash_weight_by_name).to_csv(cash_path)
    pd.concat(diagnostics_by_name, axis=1).to_parquet(diagnostics_path)
    _plot(returns_by_name, plot_path)

    print(f"Compact comparison saved to {compact_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
