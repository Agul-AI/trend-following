#!/usr/bin/env python
"""Apply profit-lock sizing variants to the current preferred QQQ/synthetic-TQQQ strategy.

Current preferred base:
- QQQ MACD histogram > 0 for entry.
- QQQ hourly close > QQQ hourly 200-day MA as entry gate.
- QQQ hourly close < QQQ hourly 200-day MA as exit.
- No daily regime gate.
- Synthetic TQQQ exposure (QQQ_3X_CALC).
- Max one trade per day via executable weight conversion.
- Out-of-market cash earns 3% annualized in evaluation.

Profit lock variants reduce exposure inside an open base trade after unrealized
synthetic-TQQQ trade gain crosses thresholds. Signals are still shifted by the
project executable-weight convention before returns are earned.
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
from run_tqqq_position_risk_sizing_experiments import drawdown_episode_count  # noqa: E402
from run_tqqq_tiered_sizing_experiments import trade_profit_lock_tiers  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.metrics import calculate_metrics, metrics_to_frame  # noqa: E402
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
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--output-prefix", default="preferred_profit_lock_comparison")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def lock_schemes() -> dict[str, list[tuple[float, float]]]:
    """Reasonable profit-lock grid: +X% gain -> 75%, +Y% gain -> 50%."""
    schemes: dict[str, list[tuple[float, float]]] = {"base_no_lock": []}
    for first in (1.00, 1.50, 2.00, 2.50, 3.00, 4.00):
        for second in (2.00, 2.50, 3.00, 4.00, 5.00, 6.00):
            if second <= first:
                continue
            label = f"lock_{int(first * 100)}_{int(second * 100)}_to_75_50"
            schemes[label] = [(first, 0.75), (second, 0.50)]

    # Single-step locks requested by the user.
    schemes["lock_300_to_75_only"] = [(3.00, 0.75)]
    schemes["lock_400_to_75_only"] = [(4.00, 0.75)]
    return schemes


def _apply_lock(raw: pd.Series, traded_price: pd.Series, scheme: list[tuple[float, float]]) -> pd.Series:
    if not scheme:
        return raw.astype(float).copy()
    return trade_profit_lock_tiers(raw, traded_price, thresholds_to_weights=scheme)


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


def _plot(returns_by_name: dict[str, pd.Series], metrics: pd.DataFrame, output_path: Path) -> None:
    selected = ["base_no_lock"]
    selected.extend(
        metrics[metrics["strategy"].eq("preferred_profit_lock")]
        .sort_values(["sharpe_ratio", "annualized_return"], ascending=[False, False])
        .head(5)["name"]
        .tolist()
    )
    selected.append("QQQ_BH")
    selected = list(dict.fromkeys([name for name in selected if name in returns_by_name]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for name in selected:
        returns = returns_by_name[name]
        _equity(returns).plot(ax=axes[0], label=name, linewidth=1.15)
        _drawdown(returns).plot(ax=axes[1], label=name, linewidth=1.15)
    axes[0].set_title("Equity / growth of $1")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("Current preferred strategy: profit-lock variants vs QQQ BH")
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
    params.update(
        {
            "target_ticker": args.target_ticker,
            "regime_ticker": args.benchmark_ticker,
            "sma_window": 200,
            "use_variance_ratio_for_trend": False,
        }
    )
    bars_per_day = int(params.get("intraday_bars_per_day", 6))

    raw_base, diagnostics = no_daily_gate_hourly_ma_gate_signal(
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

    metric_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.Series] = {}
    cash_weights_by_name: dict[str, pd.Series] = {}

    for name, scheme in lock_schemes().items():
        raw_locked = _apply_lock(raw_base, target_prices[args.target_ticker], scheme).rename(args.target_ticker)
        raw_weights = raw_locked.to_frame(args.target_ticker)
        weights = executable_weights(raw_weights, config=config).reindex(common).fillna(0.0)
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
        metrics.update(
            {
                "name": name,
                "strategy": "preferred_profit_lock",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "profit_lock_scheme": scheme,
                        "cash_annual_yield": args.cash_annual_yield,
                        "transaction_cost_bps": args.transaction_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "short_term_tax_rate": args.short_term_tax_rate,
                        "base_params": params,
                    },
                    sort_keys=True,
                ),
                "final_return": metrics["cumulative_return"],
                "average_cash_weight": float(cash_weight.mean()),
                "profit_lock_first_threshold": scheme[0][0] if scheme else np.nan,
                "profit_lock_second_threshold": scheme[1][0] if len(scheme) > 1 else np.nan,
                "profit_lock_first_threshold_hit_count": (
                    float(count_profit_lock_hits(raw_base, target_prices[args.target_ticker], threshold=scheme[0][0]))
                    if scheme
                    else np.nan
                ),
                "profit_lock_second_threshold_hit_count": (
                    float(count_profit_lock_hits(raw_base, target_prices[args.target_ticker], threshold=scheme[1][0]))
                    if len(scheme) > 1
                    else np.nan
                ),
                "tax_paid_pct_initial_capital": float(taxes.sum()),
                "cash_interest_pct_initial_capital": float(cash_interest.sum()),
            }
        )
        metric_rows.append(metrics)
        returns_by_name[name] = returns
        weights_by_name[name] = weights.sum(axis=1)
        cash_weights_by_name[name] = cash_weight

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
            "profit_lock_first_threshold": np.nan,
            "profit_lock_second_threshold": np.nan,
            "profit_lock_first_threshold_hit_count": np.nan,
            "profit_lock_second_threshold_hit_count": np.nan,
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
    ]
    compact = metrics[compact_cols].sort_values("annualized_return", ascending=False)

    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)
    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    compact_path = tables_dir / f"{args.output_prefix}_compact.csv"
    returns_path = tables_dir / f"{args.output_prefix}_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    cash_weights_path = tables_dir / f"{args.output_prefix}_cash_weights.csv"
    diagnostics_path = tables_dir / f"{args.output_prefix}_diagnostics.parquet"
    plot_path = figures_dir / f"{args.output_prefix}_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    compact.to_csv(compact_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(weights_by_name).to_csv(weights_path)
    pd.DataFrame(cash_weights_by_name).to_csv(cash_weights_path)
    diagnostics.to_parquet(diagnostics_path)
    _plot(returns_by_name, metrics, plot_path)

    print(f"Metrics saved to {metrics_path}")
    print(f"Compact comparison saved to {compact_path}")
    print(f"Returns saved to {returns_path}")
    print(f"Weights saved to {weights_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
