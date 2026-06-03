#!/usr/bin/env python
"""Test entry-loss stops with the preferred +300/+400 profit-lock overlay.

Stop definition here is different from peak-drawdown stop:
- It exits if synthetic TQQQ unrealized return from the current trade entry falls
  below -30/-35/-40/-45/-50%.
- It is not measured from the trade peak.
- Raw stop signal is shifted by executable_weights before earning returns.
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

from run_preferred_profit_lock_comparison import _add_dd_counts, _load_price  # noqa: E402
from run_tqqq_cash_yield_candidate_comparison import (
    simulate_after_tax_portfolio_with_cash_yield,  # noqa: E402
)
from run_tqqq_daily_gate_ablation import no_daily_gate_hourly_ma_gate_signal  # noqa: E402
from run_tqqq_entry_signal_comparison import (  # noqa: E402
    _drawdown,
    _equity,
    _returns_from_prices,
    executable_weights,
)
from run_tqqq_macd_entry_experiments import count_profit_lock_hits  # noqa: E402
from run_tqqq_tiered_sizing_experiments import trade_profit_lock_tiers  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.metrics import calculate_metrics, metrics_to_frame  # noqa: E402
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

TARGET_TICKER = "QQQ_3X_CALC"
BENCHMARK_TICKER = "QQQ"
PROFIT_LOCK_SCHEME = [(3.00, 0.75), (4.00, 0.50)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default=TARGET_TICKER)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--output-prefix", default="preferred_profit_lock_loss_stop_comparison")
    return parser.parse_args()


def raw_with_entry_loss_stop(
    base_raw: pd.Series,
    traded_price: pd.Series,
    *,
    loss_stop: float | None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Force raw signal to cash if trade return from entry breaches loss_stop."""
    base = base_raw.fillna(0.0).astype(float)
    price = traded_price.reindex(base.index).astype(float)
    if loss_stop is None:
        diagnostics = pd.DataFrame(
            {
                "base_raw": base,
                "stopped_raw": base,
                "entry_price": np.nan,
                "trade_return": np.nan,
                "loss_stop_trigger": False,
            },
            index=base.index,
        )
        return base.rename(base_raw.name), diagnostics
    threshold = -abs(float(loss_stop))
    in_trade = False
    stopped_until_base_exit = False
    entry_price = np.nan
    values: list[float] = []
    entry_prices: list[float] = []
    trade_returns: list[float] = []
    triggers: list[bool] = []

    for base_signal, current_price in zip(base, price, strict=True):
        current_price = float(current_price) if np.isfinite(current_price) else np.nan
        trigger = False
        if base_signal <= 0.0 or not np.isfinite(current_price):
            in_trade = False
            stopped_until_base_exit = False
            entry_price = np.nan
            value = 0.0
            trade_return = np.nan
        else:
            if not in_trade:
                in_trade = True
                stopped_until_base_exit = False
                entry_price = current_price
            trade_return = current_price / entry_price - 1.0 if entry_price > 0 else np.nan
            if stopped_until_base_exit:
                value = 0.0
            elif np.isfinite(trade_return) and trade_return <= threshold:
                trigger = True
                stopped_until_base_exit = True
                value = 0.0
            else:
                value = 1.0
        values.append(value)
        entry_prices.append(entry_price)
        trade_returns.append(trade_return)
        triggers.append(trigger)

    stopped = pd.Series(values, index=base.index, name=base_raw.name, dtype=float)
    diagnostics = pd.DataFrame(
        {
            "base_raw": base,
            "stopped_raw": stopped,
            "entry_price": entry_prices,
            "trade_return": trade_returns,
            "loss_stop_trigger": triggers,
        },
        index=base.index,
    )
    return stopped, diagnostics


def _plot_selected(returns_by_name: dict[str, pd.Series], output_path: Path) -> None:
    selected = [
        "profit_lock_300_400_no_loss_stop",
        "profit_lock_300_400_loss_stop_30pct",
        "profit_lock_300_400_loss_stop_35pct",
        "profit_lock_300_400_loss_stop_40pct",
        "profit_lock_300_400_loss_stop_45pct",
        "profit_lock_300_400_loss_stop_50pct",
        "base_no_lock_no_loss_stop",
        "QQQ_BH",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.7))
    for name in selected:
        if name not in returns_by_name:
            continue
        _equity(returns_by_name[name]).plot(ax=axes[0], label=name, linewidth=1.05)
        _drawdown(returns_by_name[name]).plot(ax=axes[1], label=name, linewidth=1.05)
    axes[0].set_title("Equity / growth of $1")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("Preferred +300/+400 profit lock with entry-loss stops")
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
    raw_base, base_diagnostics = no_daily_gate_hourly_ma_gate_signal(
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

    schemes: dict[str, tuple[pd.Series, str, float | None, pd.DataFrame]] = {
        "base_no_lock_no_loss_stop": (raw_base, "none", None, base_diagnostics),
        "profit_lock_300_400_no_loss_stop": (raw_base, "300_400", None, base_diagnostics),
    }
    for stop in (0.30, 0.35, 0.40, 0.45, 0.50):
        stopped_raw, stop_diag = raw_with_entry_loss_stop(raw_base, target_prices[args.target_ticker], loss_stop=stop)
        schemes[f"profit_lock_300_400_loss_stop_{int(stop * 100)}pct"] = (stopped_raw, "300_400", stop, stop_diag)

    metric_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.Series] = {}
    diagnostics_by_name: dict[str, pd.DataFrame] = {}

    for name, (raw_signal, lock_label, stop, diag) in schemes.items():
        if lock_label == "300_400":
            raw_weight = trade_profit_lock_tiers(
                raw_signal.rename(args.target_ticker),
                target_prices[args.target_ticker],
                thresholds_to_weights=PROFIT_LOCK_SCHEME,
            ).rename(args.target_ticker)
        else:
            raw_weight = raw_signal.rename(args.target_ticker).astype(float)
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
        stop_triggers = int(diag.get("loss_stop_trigger", pd.Series(False, index=diag.index)).sum()) if stop is not None else 0
        metrics.update(
            {
                "name": name,
                "strategy": "preferred_profit_lock_loss_stop",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "profit_lock_scheme": PROFIT_LOCK_SCHEME if lock_label == "300_400" else [],
                        "loss_stop": stop,
                        "cash_annual_yield": args.cash_annual_yield,
                        "transaction_cost_bps": args.transaction_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "short_term_tax_rate": args.short_term_tax_rate,
                    },
                    sort_keys=True,
                ),
                "final_return": metrics["cumulative_return"],
                "average_cash_weight": float(cash_weight.mean()),
                "profit_lock_first_threshold_hit_count": (
                    float(count_profit_lock_hits(raw_signal, target_prices[args.target_ticker], threshold=3.0))
                    if lock_label == "300_400"
                    else np.nan
                ),
                "profit_lock_second_threshold_hit_count": (
                    float(count_profit_lock_hits(raw_signal, target_prices[args.target_ticker], threshold=4.0))
                    if lock_label == "300_400"
                    else np.nan
                ),
                "loss_stop_threshold": stop,
                "loss_stop_trigger_count": stop_triggers,
                "tax_paid_pct_initial_capital": float(taxes.sum()),
                "cash_interest_pct_initial_capital": float(cash_interest.sum()),
            }
        )
        metric_rows.append(metrics)
        returns_by_name[name] = returns
        weights_by_name[name] = weights.sum(axis=1)
        diagnostics_by_name[name] = diag

    benchmark_metrics = calculate_metrics(
        qqq_returns,
        weights=pd.Series(1.0, index=qqq_returns.index),
        annualization=config.backtest.annualization,
    )
    _add_dd_counts(benchmark_metrics, qqq_returns)
    benchmark_metrics.update(
        {
            "name": "QQQ_BH",
            "strategy": "benchmark",
            "segment": "full_sample",
            "parameters": "{}",
            "final_return": benchmark_metrics["cumulative_return"],
            "average_cash_weight": 0.0,
            "profit_lock_first_threshold_hit_count": np.nan,
            "profit_lock_second_threshold_hit_count": np.nan,
            "loss_stop_threshold": np.nan,
            "loss_stop_trigger_count": 0,
            "tax_paid_pct_initial_capital": 0.0,
            "cash_interest_pct_initial_capital": 0.0,
        }
    )
    metric_rows.append(benchmark_metrics)
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
        "profit_lock_first_threshold_hit_count",
        "profit_lock_second_threshold_hit_count",
        "loss_stop_threshold",
        "loss_stop_trigger_count",
    ]
    compact = metrics[compact_cols].sort_values("annualized_return", ascending=False)

    # Save stop-trigger events.
    event_rows: list[dict[str, Any]] = []
    for name, diag in diagnostics_by_name.items():
        if "loss_stop_trigger" not in diag:
            continue
        for timestamp, row in diag[diag["loss_stop_trigger"]].iterrows():
            event_rows.append(
                {
                    "variant": name,
                    "stop_trigger_raw": timestamp,
                    "entry_price": row["entry_price"],
                    "trigger_price": target_prices.at[timestamp, args.target_ticker],
                    "trade_return_at_trigger": row["trade_return"],
                }
            )
    stop_events = pd.DataFrame(event_rows)

    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)
    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    compact_path = tables_dir / f"{args.output_prefix}_compact.csv"
    returns_path = tables_dir / f"{args.output_prefix}_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    diagnostics_path = tables_dir / f"{args.output_prefix}_diagnostics.parquet"
    stop_events_path = tables_dir / f"{args.output_prefix}_stop_events.csv"
    plot_path = figures_dir / f"{args.output_prefix}_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    compact.to_csv(compact_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(weights_by_name).to_csv(weights_path)
    pd.concat(diagnostics_by_name, axis=1).to_parquet(diagnostics_path)
    stop_events.to_csv(stop_events_path, index=False)
    _plot_selected(returns_by_name, plot_path)

    print(f"Compact comparison saved to {compact_path}")
    print(f"Stop events saved to {stop_events_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))
    if not stop_events.empty:
        print("\nLoss-stop events:")
        print(stop_events.to_string(index=False))


if __name__ == "__main__":
    main()
