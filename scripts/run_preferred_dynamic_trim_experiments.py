#!/usr/bin/env python
"""Test dynamic mean-reversion trims for the preferred QQQ/synthetic-TQQQ rule.

The overlay replaces a fixed "QQQ is X% above the 200-day MA" threshold with a
trade-specific threshold learned from the trade itself:

1. Run the current preferred baseline: QQQ hourly MACD/200MA entry-exit,
   synthetic QQQ_3X_CALC exposure, +300%/+400% profit locks, and a 40% synthetic
   trade-peak stop.
2. For each long trade, wait until the synthetic trade first reaches +100%.
3. Learn a threshold from QQQ's distance to its hourly 200-day MA observed from
   entry through that first +100% bar.  The grid tests threshold quantiles.
4. After +100%, trim to 50% if QQQ's distance revisits/exceeds the learned
   threshold.  Re-add full size when QQQ pulls back to its 20-day MA.

Outputs:
- reports/tables/preferred_dynamic_trim_experiments_compact.csv
- reports/tables/preferred_dynamic_trim_experiments_metrics.csv
- reports/tables/preferred_dynamic_trim_trade_triggers.csv
- reports/tables/preferred_dynamic_trim_hiking_cycle_performance.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_preferred_drawdown_reduction_experiments import (  # noqa: E402
    _evaluate,
    _preferred_weight,
    _single_asset_weights,
)
from run_tqqq_daily_gate_ablation import no_daily_gate_hourly_ma_gate_signal  # noqa: E402
from run_tqqq_entry_signal_comparison import _returns_from_prices  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.fed_cycles import cycles_to_frame, load_cycle_config  # noqa: E402
from trend_following.metrics import calculate_metrics, metrics_to_frame  # noqa: E402
from trend_following.risk_overlays import (  # noqa: E402
    apply_cap,
    dynamic_pre100_distance_trim_rebuy_cap,
    qqq_mean_reversion_features,
)
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

TARGET_TICKER = "QQQ_3X_CALC"
BENCHMARK_TICKER = "QQQ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--fed-cycle-config", default="configs/fed_hiking_cycles.yaml")
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
    parser.add_argument(
        "--threshold-quantiles",
        default="0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95,1.00",
        help="Comma-separated quantiles of pre-+100% QQQ/200MA distance.",
    )
    parser.add_argument("--output-prefix", default="preferred_dynamic_trim")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def _parse_quantiles(value: str) -> list[float]:
    quantiles = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not quantiles:
        raise ValueError("At least one threshold quantile is required")
    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Invalid quantile {quantile}; expected 0..1")
    return quantiles


def _add_trigger_counts(metrics: dict[str, Any], diagnostics: pd.DataFrame) -> None:
    active = diagnostics.loc[diagnostics["dynamic_trade_id"].gt(0)]
    if active.empty:
        metrics["total_baseline_trades"] = 0
        metrics["triggered_trades"] = 0
        metrics["total_trim_triggers"] = 0
        metrics["total_reentries"] = 0
        return
    trigger_by_trade = active.groupby("dynamic_trade_id")["overlay_trigger"].sum()
    metrics["total_baseline_trades"] = int(active["dynamic_trade_id"].max())
    metrics["triggered_trades"] = int(trigger_by_trade.gt(0).sum())
    metrics["total_trim_triggers"] = int(trigger_by_trade.sum())
    metrics["total_reentries"] = int(active["overlay_reentry"].sum())


def _trigger_table_for_quantile(
    *,
    quantile: float,
    diagnostics: pd.DataFrame,
    trade_stats: pd.DataFrame,
) -> pd.DataFrame:
    active = diagnostics.loc[diagnostics["dynamic_trade_id"].gt(0)]
    rows: list[dict[str, Any]] = []
    for trade in trade_stats.itertuples(index=False):
        trade_id = int(trade.trade_id)
        window = active.loc[active["dynamic_trade_id"].eq(trade_id)]
        trigger_rows = window.loc[window["overlay_trigger"].astype(bool)]
        reentry_rows = window.loc[window["overlay_reentry"].astype(bool)]
        learned_rows = window.loc[window["threshold_learned"].astype(bool)]
        rows.append(
            {
                "pre100_distance_quantile": quantile,
                "trade_id": trade_id,
                "entry_date": trade.entry_return_label,
                "exit_date": trade.exit_return_label,
                "final_return_pct": getattr(trade, "after_tax_cash_return", np.nan) * 100.0,
                "trigger_count": int(len(trigger_rows)),
                "first_trigger": trigger_rows.index[0] if not trigger_rows.empty else pd.NaT,
                "reentry_count": int(len(reentry_rows)),
                "learned_threshold": (
                    float(learned_rows["dynamic_distance_threshold"].dropna().iloc[0])
                    if not learned_rows["dynamic_distance_threshold"].dropna().empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _cycle_metrics(
    returns_by_name: dict[str, pd.Series],
    cycles: pd.DataFrame,
    *,
    annualization: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cycle in cycles.itertuples(index=False):
        start = pd.Timestamp(cycle.start)
        end = pd.Timestamp(cycle.end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        for name, returns in returns_by_name.items():
            sample = returns.loc[(returns.index >= start) & (returns.index <= end)].dropna()
            if sample.empty:
                continue
            metrics = calculate_metrics(sample, annualization=annualization)
            rows.append(
                {
                    "cycle_group": cycle.cycle_group,
                    "cycle_name": cycle.name,
                    "start": cycle.start,
                    "end": cycle.end,
                    "strategy": name,
                    "bars": int(len(sample)),
                    "final_return": metrics["cumulative_return"],
                    "annualized_return": metrics["annualized_return"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "hit_rate": metrics["hit_rate"],
                }
            )
    return pd.DataFrame(rows)


def _worst_drawdown_summary(returns_by_name: dict[str, pd.Series]) -> pd.DataFrame:
    """Return peak/trough/recovery rows for each return series."""
    rows: list[dict[str, Any]] = []
    for name, returns in returns_by_name.items():
        clean = returns.fillna(0.0)
        if clean.empty:
            continue
        equity = (1.0 + clean).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        trough = drawdown.idxmin()
        peak = equity.loc[:trough].idxmax()
        recovery = equity.loc[trough:][equity.loc[trough:].ge(equity.loc[peak])]
        rows.append(
            {
                "strategy": name,
                "peak": peak,
                "trough": trough,
                "recovery": recovery.index[0] if not recovery.empty else pd.NaT,
                "max_drawdown": float(drawdown.loc[trough]),
                "peak_equity": float(equity.loc[peak]),
                "trough_equity": float(equity.loc[trough]),
                "calendar_days_peak_to_trough": int((trough - peak).days),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tables_dir = resolve_path(config.root, "reports/tables")
    ensure_directory(tables_dir)

    target = _load_price(
        resolve_path(config.root, args.target_raw_dir) / f"{args.target_ticker}.parquet",
        args.target_ticker,
    )
    qqq = _load_price(
        resolve_path(config.root, args.benchmark_raw_dir) / f"{args.benchmark_ticker}.parquet",
        args.benchmark_ticker,
    )
    common = target.index.intersection(qqq.index)
    target = target.loc[common]
    qqq = qqq.loc[common]
    returns = pd.concat(
        [_returns_from_prices(target.to_frame()), _returns_from_prices(qqq.to_frame())],
        axis=1,
    ).loc[common]

    params = dict(config.strategies.regime_switch)
    bars_per_day = int(params.get("intraday_bars_per_day", 6))
    raw_base, _ = no_daily_gate_hourly_ma_gate_signal(
        entry_price=qqq,
        exit_price=qqq,
        output_index=common,
        bars_per_day=bars_per_day,
        average_type=args.average_type,
        macd_unit=args.macd_unit,
    )
    baseline_weight, _ = _preferred_weight(
        raw_base=raw_base, target_price=target, stop_drawdown=0.40
    )
    features = qqq_mean_reversion_features(qqq, bars_per_day=bars_per_day)

    eval_args = argparse.Namespace(
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        short_term_tax_rate=args.short_term_tax_rate,
        cash_annual_yield=args.cash_annual_yield,
        cash_interest_tax_rate=args.cash_interest_tax_rate,
        target_ticker=args.target_ticker,
    )

    metrics_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    trigger_tables: list[pd.DataFrame] = []

    baseline_metrics, baseline_returns, _ = _evaluate(
        name="baseline_preferred_lock300_400_stop40",
        family="baseline",
        raw_weights=_single_asset_weights(baseline_weight, args.target_ticker),
        returns=returns,
        config=config,
        args=eval_args,
        parameters={"profit_lock": [(3.0, 0.75), (4.0, 0.50)], "peak_stop": 0.40},
    )
    baseline_metrics["pre100_distance_quantile"] = np.nan
    baseline_metrics["total_baseline_trades"] = int(
        baseline_weight.gt(0).astype(int).diff().eq(1).sum()
    )
    baseline_metrics["triggered_trades"] = 0
    baseline_metrics["total_trim_triggers"] = 0
    baseline_metrics["total_reentries"] = 0
    metrics_rows.append(baseline_metrics)
    returns_by_name[baseline_metrics["name"]] = baseline_returns

    trade_stats_path = tables_dir / "preferred_plus_40pct_peak_stop_trade_stats.csv"
    trade_stats = pd.read_csv(trade_stats_path) if trade_stats_path.exists() else pd.DataFrame()

    for quantile in _parse_quantiles(args.threshold_quantiles):
        overlay = dynamic_pre100_distance_trim_rebuy_cap(
            baseline_weight.gt(0).astype(float),
            target,
            features,
            activation_gain=1.0,
            threshold_quantile=quantile,
            trim_weight=0.50,
            reentry_rule="ma20",
        )
        candidate_weight = apply_cap(baseline_weight, overlay.weights)
        name = f"dynamic_pre100_q{int(round(quantile * 100)):03d}_to50_rema20"
        metrics, candidate_returns, _ = _evaluate(
            name=name,
            family="dynamic_pre100_distance",
            raw_weights=_single_asset_weights(candidate_weight, args.target_ticker),
            returns=returns,
            config=config,
            args=eval_args,
            parameters={
                "activation_gain": 1.0,
                "threshold_quantile": quantile,
                "trim_weight": 0.50,
                "reentry_rule": "ma20",
            },
        )
        metrics["pre100_distance_quantile"] = quantile
        _add_trigger_counts(metrics, overlay.diagnostics)
        metrics_rows.append(metrics)
        returns_by_name[name] = candidate_returns
        if not trade_stats.empty:
            trigger_tables.append(
                _trigger_table_for_quantile(
                    quantile=quantile,
                    diagnostics=overlay.diagnostics,
                    trade_stats=trade_stats,
                )
            )

    qqq_bh_metrics, qqq_bh_returns, _ = _evaluate(
        name="QQQ_BH",
        family="benchmark",
        raw_weights=pd.DataFrame({args.benchmark_ticker: pd.Series(1.0, index=common)}),
        returns=returns,
        config=config,
        args=eval_args,
        parameters={},
    )
    qqq_bh_metrics["pre100_distance_quantile"] = np.nan
    qqq_bh_metrics["total_baseline_trades"] = 0
    qqq_bh_metrics["triggered_trades"] = 0
    qqq_bh_metrics["total_trim_triggers"] = 0
    qqq_bh_metrics["total_reentries"] = 0
    metrics_rows.append(qqq_bh_metrics)
    returns_by_name["QQQ_BH"] = qqq_bh_returns

    metrics = metrics_to_frame(metrics_rows)
    baseline = metrics.loc[metrics["name"].eq("baseline_preferred_lock300_400_stop40")].iloc[0]
    metrics["delta_ann_return_vs_baseline"] = metrics["annualized_return"] - float(
        baseline["annualized_return"]
    )
    metrics["delta_max_dd_vs_baseline"] = metrics["max_drawdown"] - float(baseline["max_drawdown"])
    metrics["objective_score"] = (
        metrics["annualized_return"]
        + 0.35 * metrics["delta_max_dd_vs_baseline"]
        - 0.0025 * metrics["number_of_trades"].clip(lower=0) / 10.0
    )
    metrics = metrics.sort_values(
        ["objective_score", "annualized_return", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    compact_cols = [
        "name",
        "family",
        "pre100_distance_quantile",
        "final_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "delta_ann_return_vs_baseline",
        "delta_max_dd_vs_baseline",
        "number_of_trades",
        "trades_per_year",
        "exposure_percentage",
        "dd_episodes_gt_20_30_40_50pct",
        "total_baseline_trades",
        "triggered_trades",
        "total_trim_triggers",
        "total_reentries",
        "objective_score",
    ]
    metrics.to_csv(tables_dir / f"{args.output_prefix}_experiments_metrics.csv", index=False)
    metrics[[column for column in compact_cols if column in metrics.columns]].to_csv(
        tables_dir / f"{args.output_prefix}_experiments_compact.csv",
        index=False,
    )
    if trigger_tables:
        pd.concat(trigger_tables, ignore_index=True).to_csv(
            tables_dir / f"{args.output_prefix}_trade_triggers.csv",
            index=False,
        )
    pd.concat(returns_by_name.values(), axis=1).to_csv(
        tables_dir / f"{args.output_prefix}_experiments_returns.csv"
    )

    best = metrics.loc[metrics["family"].eq("dynamic_pre100_distance")].iloc[0]
    best_name = str(best["name"])
    cycle_groups = load_cycle_config(resolve_path(config.root, args.fed_cycle_config))
    cycles_frame = cycles_to_frame(cycle_groups)
    cycle_group = "pre_announcement_to_pre_cut"
    cycles_frame.to_csv(tables_dir / f"{args.output_prefix}_fed_cycles.csv", index=False)
    selected_cycles = cycles_frame.loc[cycles_frame["cycle_group"].eq(cycle_group)]
    cycle_perf = _cycle_metrics(
        {
            best_name: returns_by_name[best_name],
            "baseline_preferred_lock300_400_stop40": returns_by_name[
                "baseline_preferred_lock300_400_stop40"
            ],
            "QQQ_BH": returns_by_name["QQQ_BH"],
        },
        selected_cycles,
        annualization=config.backtest.annualization,
    )
    cycle_perf.to_csv(
        tables_dir / f"{args.output_prefix}_hiking_cycle_performance.csv", index=False
    )
    _worst_drawdown_summary(
        {
            best_name: returns_by_name[best_name],
            "baseline_preferred_lock300_400_stop40": returns_by_name[
                "baseline_preferred_lock300_400_stop40"
            ],
            "QQQ_BH": returns_by_name["QQQ_BH"],
        }
    ).to_csv(tables_dir / f"{args.output_prefix}_worst_drawdown_summary.csv", index=False)

    print(f"Saved compact table: {tables_dir / f'{args.output_prefix}_experiments_compact.csv'}")
    print(f"Saved trigger table: {tables_dir / f'{args.output_prefix}_trade_triggers.csv'}")
    print(f"Saved cycle table: {tables_dir / f'{args.output_prefix}_hiking_cycle_performance.csv'}")
    print("Best dynamic candidate:")
    print(
        best[
            [
                "name",
                "pre100_distance_quantile",
                "final_return",
                "annualized_return",
                "sharpe_ratio",
                "max_drawdown",
                "number_of_trades",
                "triggered_trades",
                "total_trim_triggers",
                "dd_episodes_gt_20_30_40_50pct",
            ]
        ].to_string()
    )


if __name__ == "__main__":
    main()
