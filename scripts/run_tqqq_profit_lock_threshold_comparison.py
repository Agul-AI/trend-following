#!/usr/bin/env python
"""Compare profit-lock thresholds for MACD signals trading synthetic TQQQ."""

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

from run_tqqq_entry_signal_comparison import (  # noqa: E402
    _drawdown,
    _equity,
    _returns_from_prices,
    executable_weights,
    macd_entry_slow_exit_signal,
)
from run_tqqq_macd_entry_experiments import count_profit_lock_hits  # noqa: E402
from run_tqqq_position_risk_sizing_experiments import (  # noqa: E402
    drawdown_episode_count,
    simulate_after_tax_portfolio,
)
from run_tqqq_tiered_sizing_experiments import trade_profit_lock_tiers  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.metrics import calculate_metrics, metrics_to_frame  # noqa: E402
from trend_following.regime import (  # noqa: E402
    align_daily_regimes_to_intraday,
    classify_regimes,
    compute_regime_features,
)
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

LOCK_SCHEMES: dict[str, list[tuple[float, float]]] = {
    "full_no_lock": [],
    "lock_150_250_to_75_50": [(1.50, 0.75), (2.50, 0.50)],
    "lock_200_300_to_75_50": [(2.00, 0.75), (3.00, 0.50)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default="QQQ_3X_CALC")
    parser.add_argument("--benchmark-ticker", default="QQQ")
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--daily-regime-raw-dir", default="data/raw/alpha_vantage_daily_adjusted")
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--min-original-ann-return", type=float, default=0.21)
    parser.add_argument("--output-prefix", default="tqqq_profit_lock_threshold_comparison")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def _plot_top(
    returns_by_name: dict[str, pd.Series],
    metrics: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 10,
) -> None:
    selected = (
        metrics[metrics["strategy"].ne("benchmark")]
        .sort_values(["sharpe_ratio", "max_drawdown"], ascending=[False, False])
        .head(top_n)["name"]
        .tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    for name in selected:
        returns = returns_by_name[name]
        _equity(returns).plot(ax=axes[0], label=name, linewidth=1.1)
        _drawdown(returns).plot(ax=axes[1], label=name, linewidth=1.1)
    axes[0].set_title("After-tax equity")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=6)
    axes[1].set_title("After-tax drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _parse_case_name(name: str) -> tuple[str, str, str]:
    base = name.removesuffix("_profit_lock_150_250").removesuffix("_full")
    parts = base.split("_")
    average_type = parts[1]
    unit = parts[2]
    if base.endswith("hist_rising"):
        entry_mode = "hist_gt_0_and_rising"
    elif base.endswith("hist_gt_0"):
        entry_mode = "hist_gt_0"
    elif base.endswith("hist_macd_pos"):
        entry_mode = "hist_gt_0_macd_gt_0"
    else:
        raise ValueError(f"Cannot parse MACD entry mode from {name}")
    return average_type, unit, entry_mode


def _load_high_return_case_specs(min_ann_return: float) -> list[tuple[str, str, str, str]]:
    metrics_path = Path("reports/tables/tqqq_entry_signal_comparison_metrics.csv")
    if not metrics_path.exists():
        raise FileNotFoundError("Run scripts/run_tqqq_entry_signal_comparison.py first.")
    metrics = pd.read_csv(metrics_path)
    filtered = metrics[
        metrics["name"].str.startswith("macd_")
        & metrics["annualized_return"].ge(min_ann_return)
    ].copy()
    # Deduplicate full/profit-lock variants down to unique entry-signal specs.
    specs: dict[tuple[str, str, str], str] = {}
    for name in filtered.sort_values("sharpe_ratio", ascending=False)["name"]:
        avg, unit, mode = _parse_case_name(name)
        specs.setdefault((avg, unit, mode), name)
    return [(label, *spec) for spec, label in specs.items()]


def _apply_lock(raw: pd.Series, price: pd.Series, scheme: list[tuple[float, float]]) -> pd.Series:
    if not scheme:
        return raw.astype(float).copy()
    return trade_profit_lock_tiers(raw, price, thresholds_to_weights=scheme)


def _hit_count(raw: pd.Series, price: pd.Series, threshold: float, scheme: list[tuple[float, float]]) -> float:
    if not scheme:
        return np.nan
    return float(count_profit_lock_hits(raw, price, threshold=threshold))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    target_dir = resolve_path(config.root, args.target_raw_dir)
    benchmark_dir = resolve_path(config.root, args.benchmark_raw_dir)
    daily_dir = resolve_path(config.root, args.daily_regime_raw_dir)

    target = _load_price(target_dir / f"{args.target_ticker}.parquet", args.target_ticker)
    qqq = _load_price(benchmark_dir / f"{args.benchmark_ticker}.parquet", args.benchmark_ticker)
    daily_qqq = _load_price(daily_dir / f"{args.benchmark_ticker}.parquet", args.benchmark_ticker)

    common = target.index.intersection(qqq.index)
    target_prices = target.loc[common].to_frame()
    qqq_prices = qqq.loc[common].to_frame()
    target_returns = _returns_from_prices(target_prices)
    qqq_returns = _returns_from_prices(qqq_prices)
    daily_prices = daily_qqq.to_frame()
    daily_returns = _returns_from_prices(daily_prices)

    params = dict(config.strategies.regime_switch)
    params.update(
        {
            "target_ticker": args.target_ticker,
            "regime_ticker": args.benchmark_ticker,
            "sma_window": 200,
            "use_variance_ratio_for_trend": False,
            "state_machine_exit_ma_days": 200.0,
            "state_machine_entry_confirm_bars": 2,
            "state_machine_exit_confirm_bars": 3,
        }
    )
    bars_per_day = int(params.get("intraday_bars_per_day", 6))

    daily_features = compute_regime_features(
        daily_prices,
        daily_returns,
        regime_ticker=args.benchmark_ticker,
        params=params,
    )
    daily_regimes = classify_regimes(daily_features, params=params)
    intraday_regimes = align_daily_regimes_to_intraday(
        daily_regimes,
        target_prices.index,
        lag_days=int(params.get("daily_regime_lag_days", 1)),
        fill_method=params.get("daily_regime_fill_method", "ffill"),
    ).fillna("neutral")
    allowed_regime = intraday_regimes.eq("trend")

    signal_sources = {
        "tqqq_signal": target_prices[args.target_ticker],
        "qqq_signal": qqq_prices[args.benchmark_ticker],
    }
    case_specs = _load_high_return_case_specs(args.min_original_ann_return)

    raw_variants: dict[str, pd.DataFrame] = {}
    meta_by_name: dict[str, dict[str, Any]] = {}
    raw_cache: dict[tuple[str, str], pd.Series] = {}
    for case_label, average_type, unit, entry_mode in case_specs:
        for source_name, signal_price in signal_sources.items():
            cache_key = (case_label, source_name)
            raw, _ = macd_entry_slow_exit_signal(
                signal_price,
                allowed_regime=allowed_regime,
                bars_per_day=bars_per_day,
                average_type=average_type,
                macd_unit=unit,
                entry_mode=entry_mode,
            )
            raw = raw.rename(args.target_ticker)
            raw_cache[cache_key] = raw
            for scheme_name, scheme in LOCK_SCHEMES.items():
                variant_name = f"{source_name}__{case_label}__{scheme_name}"
                weights = _apply_lock(raw, target_prices[args.target_ticker], scheme)
                raw_variants[variant_name] = weights.to_frame(args.target_ticker)
                meta_by_name[variant_name] = {
                    "entry_case": case_label,
                    "signal_source": source_name.removesuffix("_signal"),
                    "lock_scheme": scheme_name,
                    "average_type": average_type,
                    "macd_unit": unit,
                    "entry_mode": entry_mode,
                    "profit_lock_first_threshold_hit_count": _hit_count(
                        raw,
                        target_prices[args.target_ticker],
                        scheme[0][0] if scheme else np.nan,
                        scheme,
                    ),
                    "profit_lock_second_threshold_hit_count": _hit_count(
                        raw,
                        target_prices[args.target_ticker],
                        scheme[1][0] if len(scheme) > 1 else np.nan,
                        scheme,
                    ),
                }

    metric_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_out: dict[str, pd.Series] = {}
    all_returns = pd.concat([target_returns, qqq_returns], axis=1).loc[common]

    for name, raw_weights in raw_variants.items():
        weights = executable_weights(raw_weights, config=config).reindex(common).fillna(0.0)
        after_tax, pretax, taxes_paid, turnover = simulate_after_tax_portfolio(
            all_returns[[args.target_ticker]],
            weights[[args.target_ticker]],
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            tax_rate=args.short_term_tax_rate,
        )
        metrics = calculate_metrics(
            after_tax,
            turnover=turnover,
            weights=weights.sum(axis=1),
            annualization=config.backtest.annualization,
        )
        metrics.update(
            {
                "name": name,
                "strategy": "profit_lock_threshold_comparison",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "transaction_cost_bps": args.transaction_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "short_term_tax_rate": args.short_term_tax_rate,
                        "base_params": params,
                    },
                    sort_keys=True,
                ),
                "pretax_cumulative_return": float((1.0 + pretax).prod() - 1.0),
                "tax_paid_pct_initial_capital": float(taxes_paid.sum()),
                "drawdown_episodes_gt_30pct": drawdown_episode_count(after_tax, threshold=-0.30),
                "drawdown_episodes_gt_40pct": drawdown_episode_count(after_tax, threshold=-0.40),
                "drawdown_episodes_gt_50pct": drawdown_episode_count(after_tax, threshold=-0.50),
                **meta_by_name[name],
            }
        )
        metric_rows.append(metrics)
        returns_by_name[name] = after_tax
        weights_out[name] = weights.sum(axis=1)

    benchmark_metrics = calculate_metrics(
        qqq_returns[args.benchmark_ticker],
        annualization=config.backtest.annualization,
    )
    benchmark_metrics.update(
        {
            "name": "buy_hold_qqq",
            "strategy": "benchmark",
            "segment": "full_sample",
            "parameters": "{}",
            "pretax_cumulative_return": float((1.0 + qqq_returns[args.benchmark_ticker]).prod() - 1.0),
            "tax_paid_pct_initial_capital": 0.0,
            "drawdown_episodes_gt_30pct": drawdown_episode_count(qqq_returns[args.benchmark_ticker], threshold=-0.30),
            "drawdown_episodes_gt_40pct": drawdown_episode_count(qqq_returns[args.benchmark_ticker], threshold=-0.40),
            "drawdown_episodes_gt_50pct": drawdown_episode_count(qqq_returns[args.benchmark_ticker], threshold=-0.50),
            "entry_case": "buy_hold_qqq",
            "signal_source": "none",
            "lock_scheme": "none",
            "average_type": "",
            "macd_unit": "",
            "entry_mode": "",
            "profit_lock_first_threshold_hit_count": np.nan,
            "profit_lock_second_threshold_hit_count": np.nan,
        }
    )
    metric_rows.append(benchmark_metrics)
    returns_by_name["buy_hold_qqq"] = qqq_returns[args.benchmark_ticker]

    metrics = metrics_to_frame(metric_rows)
    metrics["dd_episodes_gt_30_40_50pct"] = metrics.apply(
        lambda row: (
            f"{int(row['drawdown_episodes_gt_30pct'])}/"
            f"{int(row['drawdown_episodes_gt_40pct'])}/"
            f"{int(row['drawdown_episodes_gt_50pct'])}"
        ),
        axis=1,
    )

    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    compact_path = tables_dir / f"{args.output_prefix}_compact.csv"
    deltas_path = tables_dir / f"{args.output_prefix}_lock_deltas.csv"
    returns_path = tables_dir / f"{args.output_prefix}_after_tax_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    plot_path = figures_dir / f"{args.output_prefix}_top_equity_drawdown.png"

    compact_cols = [
        "name",
        "entry_case",
        "signal_source",
        "lock_scheme",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "number_of_trades",
        "dd_episodes_gt_30_40_50pct",
        "profit_lock_first_threshold_hit_count",
        "profit_lock_second_threshold_hit_count",
    ]
    compact = metrics[compact_cols].sort_values("sharpe_ratio", ascending=False)

    strategy_metrics = metrics[metrics["strategy"].eq("profit_lock_threshold_comparison")]
    pivot = strategy_metrics.pivot_table(
        index=["entry_case", "signal_source"],
        columns="lock_scheme",
        values=[
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "number_of_trades",
            "drawdown_episodes_gt_30pct",
            "drawdown_episodes_gt_40pct",
            "drawdown_episodes_gt_50pct",
        ],
        aggfunc="first",
    )
    deltas = []
    for index_value in pivot.index:
        row: dict[str, Any] = {"entry_case": index_value[0], "signal_source": index_value[1]}
        for column in [
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "number_of_trades",
            "drawdown_episodes_gt_30pct",
            "drawdown_episodes_gt_40pct",
            "drawdown_episodes_gt_50pct",
        ]:
            old = pivot.loc[index_value, (column, "lock_150_250_to_75_50")]
            new = pivot.loc[index_value, (column, "lock_200_300_to_75_50")]
            full = pivot.loc[index_value, (column, "full_no_lock")]
            row[f"{column}_full"] = full
            row[f"{column}_old_lock"] = old
            row[f"{column}_new_lock"] = new
            row[f"{column}_new_minus_old"] = new - old
        deltas.append(row)
    deltas_df = pd.DataFrame(deltas).sort_values("sharpe_ratio_new_lock", ascending=False)

    metrics.to_csv(metrics_path, index=False)
    compact.to_csv(compact_path, index=False)
    deltas_df.to_csv(deltas_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(weights_out).to_csv(weights_path)
    _plot_top(
        returns_by_name,
        metrics,
        plot_path,
        title="Synthetic TQQQ profit-lock threshold comparison",
    )

    print(f"Metrics saved to {metrics_path}")
    print(f"Compact table saved to {compact_path}")
    print(f"Lock deltas saved to {deltas_path}")
    print(f"After-tax returns saved to {returns_path}")
    print(f"Weights saved to {weights_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))
    print("\nNew lock vs old lock deltas:")
    print(deltas_df.to_string(index=False))


if __name__ == "__main__":
    main()
