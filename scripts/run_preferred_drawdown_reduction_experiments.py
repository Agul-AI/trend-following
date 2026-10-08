#!/usr/bin/env python
"""Run drawdown-reduction overlays for the preferred QQQ/synthetic-TQQQ strategy.

The baseline is the current preferred rule:
- QQQ hourly MACD histogram entry.
- QQQ hourly 200-day MA entry/exit gate.
- Synthetic QQQ_3X_CALC exposure.
- +300% -> 75%, +400% -> 50% profit lock.
- 40% synthetic-3x trade-peak stop.
- Max one trade per day after no-lookahead signal shifting.
- 3% annualized out-of-market cash yield in the evaluation.

This script tests two research directions:
1. Mean-reversion overlays motivated by Top-8 winner diagnostics.
2. Fed hiking-cycle overlays using known/announced/oracle cycle windows.

Outputs include CSV tables, figures, and a static local website at
``reports/site/index.html``.
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

from run_tqqq_cash_yield_candidate_comparison import (  # noqa: E402
    simulate_after_tax_portfolio_with_cash_yield,
)
from run_tqqq_daily_gate_ablation import no_daily_gate_hourly_ma_gate_signal  # noqa: E402
from run_tqqq_entry_signal_comparison import (  # noqa: E402
    _drawdown,
    _equity,
    _returns_from_prices,
    executable_weights,
)
from run_tqqq_position_risk_sizing_experiments import drawdown_episode_count  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.fed_cycles import (  # noqa: E402
    cycle_flag,
    cycles_to_frame,
    load_cycle_config,
    monthly_pe_known,
)
from trend_following.metrics import calculate_metrics, metrics_to_frame  # noqa: E402
from trend_following.risk_overlays import (  # noqa: E402
    apply_cap,
    extension_trim_rebuy_cap,
    peak_drawdown_trim_rebuy_cap,
    qqq_mean_reversion_features,
    raw_with_peak_drawdown_stop,
    trade_profit_lock_tiers,
)
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

TARGET_TICKER = "QQQ_3X_CALC"
BENCHMARK_TICKER = "QQQ"
PROFIT_LOCK_SCHEME = [(3.0, 0.75), (4.0, 0.50)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--fed-cycle-config", default="configs/fed_hiking_cycles.yaml")
    parser.add_argument("--target-ticker", default=TARGET_TICKER)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--qqq-pe-history", default="reports/tables/qqq_pe_worldperatio_monthly_history.csv")
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--output-prefix", default="preferred_dd_reduction_experiments")
    parser.add_argument("--site-dir", default="reports/site")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def _add_dd_counts(metrics: dict[str, Any], returns: pd.Series) -> None:
    for threshold in (20, 30, 40, 50):
        metrics[f"drawdown_episodes_gt_{threshold}pct"] = drawdown_episode_count(
            returns, threshold=-threshold / 100.0
        )
    metrics["dd_episodes_gt_20_30_40_50pct"] = (
        f"{metrics['drawdown_episodes_gt_20pct']}/"
        f"{metrics['drawdown_episodes_gt_30pct']}/"
        f"{metrics['drawdown_episodes_gt_40pct']}/"
        f"{metrics['drawdown_episodes_gt_50pct']}"
    )


def _evaluate(
    *,
    name: str,
    family: str,
    raw_weights: pd.DataFrame,
    returns: pd.DataFrame,
    config: Any,
    args: argparse.Namespace,
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], pd.Series, pd.DataFrame]:
    weights = executable_weights(raw_weights, config=config).reindex(returns.index).fillna(0.0)
    after_tax, taxes, turnover, cash_interest, cash_weight = simulate_after_tax_portfolio_with_cash_yield(
        returns[[column for column in weights.columns if column in returns.columns]],
        weights[[column for column in weights.columns if column in returns.columns]],
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        tax_rate=args.short_term_tax_rate,
        cash_annual_yield=args.cash_annual_yield,
        annualization=config.backtest.annualization,
        cash_interest_tax_rate=args.cash_interest_tax_rate,
    )
    metrics = calculate_metrics(
        after_tax,
        turnover=turnover,
        weights=weights.sum(axis=1),
        annualization=config.backtest.annualization,
    )
    _add_dd_counts(metrics, after_tax)
    years = len(after_tax) / config.backtest.annualization
    metrics.update(
        {
            "name": name,
            "family": family,
            "strategy": "preferred_drawdown_reduction",
            "segment": "full_sample",
            "parameters": json.dumps(parameters, sort_keys=True),
            "final_return": metrics["cumulative_return"],
            "average_cash_weight": float(cash_weight.mean()),
            "tax_paid_pct_initial_capital": float(taxes.sum()),
            "cash_interest_pct_initial_capital": float(cash_interest.sum()),
            "trades_per_year": metrics["number_of_trades"] / years if years > 0 else np.nan,
        }
    )
    return metrics, after_tax.rename(name), weights


def _preferred_weight(
    *,
    raw_base: pd.Series,
    target_price: pd.Series,
    stop_drawdown: float | pd.Series = 0.40,
) -> tuple[pd.Series, pd.DataFrame]:
    stopped_raw, stop_diag = raw_with_peak_drawdown_stop(raw_base, target_price, stop_drawdown=stop_drawdown)
    weight = trade_profit_lock_tiers(
        stopped_raw.rename(TARGET_TICKER),
        target_price,
        thresholds_to_weights=PROFIT_LOCK_SCHEME,
    ).rename(TARGET_TICKER)
    return weight, stop_diag


def _single_asset_weights(weight: pd.Series, ticker: str = TARGET_TICKER) -> pd.DataFrame:
    return weight.rename(ticker).to_frame(ticker)


def _switch_to_qqq_weights(
    weight: pd.Series,
    switch_flag: pd.Series,
    *,
    target_ticker: str,
    qqq_ticker: str,
) -> pd.DataFrame:
    switch = switch_flag.reindex(weight.index).fillna(False).astype(bool)
    target_weight = weight.where(~switch, 0.0)
    qqq_weight = weight.where(switch, 0.0)
    return pd.DataFrame({target_ticker: target_weight, qqq_ticker: qqq_weight}, index=weight.index)


def _plot_return_vs_drawdown(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    families = metrics["family"].fillna("other").unique().tolist()
    cmap = plt.get_cmap("tab10")
    for i, family in enumerate(families):
        group = metrics[metrics["family"].eq(family)]
        ax.scatter(
            group["max_drawdown"] * 100.0,
            group["annualized_return"] * 100.0,
            s=46 if family != "baseline" else 95,
            alpha=0.75,
            label=family,
            color=cmap(i % 10),
            edgecolor="black" if family == "baseline" else "none",
        )
    base = metrics.loc[metrics["name"].eq("baseline_preferred_lock300_400_stop40")]
    if not base.empty:
        row = base.iloc[0]
        ax.annotate(
            "baseline",
            xy=(row["max_drawdown"] * 100.0, row["annualized_return"] * 100.0),
            xytext=(8, 8),
            textcoords="offset points",
            weight="bold",
        )
    ax.axhline(23.5, color="#999999", linestyle=":", linewidth=1.0, label="23.5% ann. return")
    ax.set_xlabel("Max drawdown (%)")
    ax.set_ylabel("Annualized return (%)")
    ax.set_title("Drawdown-reduction overlays: return vs max drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_equity_drawdown(
    returns_by_name: dict[str, pd.Series],
    selected_names: list[str],
    output_path: Path,
    *,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.8))
    for name in selected_names:
        if name not in returns_by_name:
            continue
        _equity(returns_by_name[name]).plot(ax=axes[0], linewidth=1.15, label=name)
        _drawdown(returns_by_name[name]).plot(ax=axes[1], linewidth=1.15, label=name)
    axes[0].set_title("Equity / growth of $1")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _write_site(
    *,
    site_dir: Path,
    metrics: pd.DataFrame,
    compact: pd.DataFrame,
    figures: dict[str, Path],
    output_prefix: str,
) -> None:
    ensure_directory(site_dir)
    css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #222; }
h1, h2 { color: #0f3557; }
.note { background: #fff8df; border-left: 4px solid #f0ad4e; padding: 10px 14px; margin: 14px 0; }
table { border-collapse: collapse; font-size: 13px; width: 100%; margin: 12px 0 24px; }
th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }
th { background: #eef3f8; position: sticky; top: 0; }
td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) { text-align: left; }
img { max-width: 100%; border: 1px solid #ddd; margin: 10px 0 30px; }
.small { color: #666; font-size: 12px; }
code { background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }
"""
    (site_dir / "style.css").write_text(css)
    display = compact.copy()
    pct_cols = [
        "final_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "exposure_percentage",
        "average_cash_weight",
        "delta_ann_return_vs_baseline",
        "delta_max_dd_vs_baseline",
    ]
    for col in pct_cols:
        if col in display.columns and col != "sharpe_ratio":
            display[col] = display[col].map(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "")
    if "sharpe_ratio" in display.columns:
        display["sharpe_ratio"] = display["sharpe_ratio"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    top = display.head(40)
    all_rows = display
    fig_html = "\n".join(
        f"<h2>{label}</h2><img src='../{path.relative_to(site_dir.parent).as_posix()}' alt='{label}'>"
        for label, path in figures.items()
    )
    index = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Preferred Strategy Drawdown-Reduction Experiments</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <h1>Preferred Strategy Drawdown-Reduction Experiments</h1>
  <p class="small">Generated by <code>scripts/run_preferred_drawdown_reduction_experiments.py</code>.</p>
  <div class="note">
    Baseline is the current preferred QQQ/synthetic-TQQQ strategy with +300/+400 profit lock and 40% trade-peak stop.
    The goal is to reduce large drawdowns without materially hurting annualized return. This is research-only, not financial advice.
  </div>
  <h2>Output files</h2>
  <ul>
    <li><a href="../tables/{output_prefix}_compact.csv">Compact CSV</a></li>
    <li><a href="../tables/{output_prefix}_metrics.csv">Full metrics CSV</a></li>
    <li><a href="../tables/{output_prefix}_returns.csv">Returns CSV</a></li>
    <li><a href="../tables/{output_prefix}_weights.parquet">Weights parquet</a></li>
  </ul>
  {fig_html}
  <h2>Top 40 rows by objective</h2>
  {top.to_html(index=False, escape=False)}
  <h2>All compact results</h2>
  {all_rows.to_html(index=False, escape=False)}
</body>
</html>
"""
    (site_dir / "index.html").write_text(index)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = config.root
    tables_dir = resolve_path(root, config.reports.tables_dir)
    figures_dir = resolve_path(root, config.reports.figures_dir)
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

    target = _load_price(resolve_path(root, args.target_raw_dir) / f"{args.target_ticker}.parquet", args.target_ticker)
    qqq = _load_price(resolve_path(root, args.benchmark_raw_dir) / f"{args.benchmark_ticker}.parquet", args.benchmark_ticker)
    common = target.index.intersection(qqq.index)
    target_price = target.loc[common]
    qqq_price = qqq.loc[common]
    returns = pd.concat(
        [
            _returns_from_prices(target_price.rename(args.target_ticker).to_frame()),
            _returns_from_prices(qqq_price.rename(args.benchmark_ticker).to_frame()),
        ],
        axis=1,
    ).loc[common]

    params = dict(config.strategies.regime_switch)
    bars_per_day = int(params.get("intraday_bars_per_day", 6))
    raw_base, base_diag = no_daily_gate_hourly_ma_gate_signal(
        entry_price=qqq_price,
        exit_price=qqq_price,
        output_index=common,
        bars_per_day=bars_per_day,
        average_type=args.average_type,
        macd_unit=args.macd_unit,
        entry_confirm_bars=2,
        exit_confirm_bars=3,
        exit_ma_days=200,
    )
    raw_base = raw_base.rename(args.target_ticker)
    features = qqq_mean_reversion_features(qqq_price, bars_per_day=bars_per_day)

    cycle_groups = load_cycle_config(resolve_path(root, args.fed_cycle_config))
    cycles_to_frame(cycle_groups).to_csv(tables_dir / f"{args.output_prefix}_fed_cycles.csv", index=False)
    cycle_flags = {
        group: cycle_flag(common, cycles, lag_days=1, name=f"{group}_known")
        for group, cycles in cycle_groups.items()
    }
    monthly_pe_path = resolve_path(root, args.qqq_pe_history)
    monthly_pe = pd.read_csv(monthly_pe_path) if monthly_pe_path.exists() else pd.DataFrame()
    pe_known = (
        monthly_pe_known(common, monthly_pe, lag_months=1)
        if not monthly_pe.empty
        else pd.Series(np.nan, index=common, name="qqq_pe_known_lag_1m")
    )
    pe_gt_30 = pe_known.gt(30.0).fillna(False)

    metrics_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.DataFrame] = {}

    def add_candidate(
        name: str,
        family: str,
        raw_weights: pd.DataFrame,
        parameters: dict[str, Any],
    ) -> None:
        if len(metrics_rows) % 10 == 0:
            print(f"Evaluating candidate {len(metrics_rows) + 1}: {name}", flush=True)
        metrics, ret, weights = _evaluate(
            name=name,
            family=family,
            raw_weights=raw_weights,
            returns=returns,
            config=config,
            args=args,
            parameters=parameters,
        )
        metrics_rows.append(metrics)
        returns_by_name[name] = ret
        weights_by_name[name] = weights

    base_weight, _ = _preferred_weight(raw_base=raw_base, target_price=target_price, stop_drawdown=0.40)
    add_candidate(
        "baseline_preferred_lock300_400_stop40",
        "baseline",
        _single_asset_weights(base_weight, args.target_ticker),
        {"profit_lock": PROFIT_LOCK_SCHEME, "peak_stop": 0.40},
    )

    # 1) Mean-reversion extension trim/rebuy overlays.
    # Keep the first pass intentionally compact so it is fast enough to rerun.
    # Broader grids can be added after a promising family appears.
    for activation in (1.00, 2.00):
        for distance in (0.18, 0.22):
            for trim_weight in (0.75, 0.50):
                for reentry_rule in ("ma20", "ma50", "z20_le_0"):
                    cap = extension_trim_rebuy_cap(
                        base_weight.gt(0).astype(float),
                        target_price,
                        features,
                        activation_gain=activation,
                        distance_threshold=distance,
                        trim_weight=trim_weight,
                        reentry_rule=reentry_rule,
                    ).weights
                    candidate_weight = apply_cap(base_weight, cap)
                    name = (
                        f"extension_trim_g{int(activation * 100)}_dist{int(distance * 100)}"
                        f"_to{int(trim_weight * 100)}_re{reentry_rule}"
                    )
                    add_candidate(
                        name,
                        "extension_mean_reversion",
                        _single_asset_weights(candidate_weight, args.target_ticker),
                        {
                            "activation_gain": activation,
                            "distance_threshold": distance,
                            "trim_weight": trim_weight,
                            "reentry_rule": reentry_rule,
                        },
                    )

    # 2) Soft peak-drawdown trim/rebuy overlays.
    for activation in (1.00, 2.00):
        for peak_dd in (0.25, 0.30):
            for trim_weight in (0.75, 0.50):
                for reentry_rule in ("ma50",):
                    cap = peak_drawdown_trim_rebuy_cap(
                        base_weight.gt(0).astype(float),
                        target_price,
                        features,
                        activation_gain=activation,
                        peak_drawdown=peak_dd,
                        trim_weight=trim_weight,
                        reentry_rule=reentry_rule,
                    ).weights
                    candidate_weight = apply_cap(base_weight, cap)
                    name = (
                        f"soft_peakdd_g{int(activation * 100)}_dd{int(peak_dd * 100)}"
                        f"_to{int(trim_weight * 100)}_re{reentry_rule}"
                    )
                    add_candidate(
                        name,
                        "soft_peak_drawdown_mean_reversion",
                        _single_asset_weights(candidate_weight, args.target_ticker),
                        {
                            "activation_gain": activation,
                            "peak_drawdown": peak_dd,
                            "trim_weight": trim_weight,
                            "reentry_rule": reentry_rule,
                        },
                    )

    # 3) Fed hiking-cycle caps and tighter stops.
    for cycle_group, flag in cycle_flags.items():
        for max_weight in (0.75, 0.50):
            cap = pd.Series(1.0, index=common).mask(flag, max_weight)
            candidate_weight = apply_cap(base_weight, cap)
            add_candidate(
                f"hiking_{cycle_group}_cap{int(max_weight * 100)}",
                "hiking_cycle_cap",
                _single_asset_weights(candidate_weight, args.target_ticker),
                {"cycle_group": cycle_group, "max_weight": max_weight, "lag_days": 1},
            )

        for hiking_stop in (0.30, 0.35):
            stop_series = pd.Series(0.40, index=common).mask(flag, hiking_stop)
            dynamic_weight, _ = _preferred_weight(
                raw_base=raw_base,
                target_price=target_price,
                stop_drawdown=stop_series,
            )
            add_candidate(
                f"hiking_{cycle_group}_stop{int(hiking_stop * 100)}_else40",
                "hiking_cycle_tighter_stop",
                _single_asset_weights(dynamic_weight, args.target_ticker),
                {"cycle_group": cycle_group, "hiking_stop": hiking_stop, "normal_stop": 0.40, "lag_days": 1},
            )

        switch_flag = flag & pe_gt_30
        add_candidate(
            f"hiking_{cycle_group}_pegt30_switch_to_qqq",
            "hiking_cycle_pe_switch",
            _switch_to_qqq_weights(
                base_weight,
                switch_flag,
                target_ticker=args.target_ticker,
                qqq_ticker=args.benchmark_ticker,
            ),
            {"cycle_group": cycle_group, "pe_threshold": 30.0, "pe_lag_months": 1, "switch": "same_weight_to_qqq"},
        )

    # 4) Small combined set from interpretable rules.
    announced_flag = cycle_flags.get("announced", pd.Series(False, index=common))
    extension_cap = extension_trim_rebuy_cap(
        base_weight.gt(0).astype(float),
        target_price,
        features,
        activation_gain=1.0,
        distance_threshold=0.20,
        trim_weight=0.75,
        reentry_rule="ma50",
    ).weights
    for hiking_stop in (0.35, 0.30):
        stop_series = pd.Series(0.40, index=common).mask(announced_flag, hiking_stop)
        dynamic_weight, _ = _preferred_weight(raw_base=raw_base, target_price=target_price, stop_drawdown=stop_series)
        candidate_weight = apply_cap(dynamic_weight, extension_cap)
        add_candidate(
            f"combined_ext100_dist20_75_ma50_announced_stop{int(hiking_stop * 100)}",
            "combined",
            _single_asset_weights(candidate_weight, args.target_ticker),
            {
                "extension_activation_gain": 1.0,
                "distance_threshold": 0.20,
                "trim_weight": 0.75,
                "reentry_rule": "ma50",
                "cycle_group": "announced",
                "hiking_stop": hiking_stop,
            },
        )

    qqq_bh_weight = pd.DataFrame({args.benchmark_ticker: pd.Series(1.0, index=common)})
    add_candidate("QQQ_BH", "benchmark", qqq_bh_weight, {})

    metrics = metrics_to_frame(metrics_rows)
    baseline = metrics.loc[metrics["name"].eq("baseline_preferred_lock300_400_stop40")].iloc[0]
    metrics["delta_ann_return_vs_baseline"] = metrics["annualized_return"] - float(baseline["annualized_return"])
    metrics["delta_max_dd_vs_baseline"] = metrics["max_drawdown"] - float(baseline["max_drawdown"])
    metrics["objective_score"] = (
        metrics["annualized_return"]
        + 0.35 * metrics["delta_max_dd_vs_baseline"]
        - 0.0025 * metrics["number_of_trades"].clip(lower=0) / 10.0
    )
    metrics = metrics.sort_values(
        ["objective_score", "annualized_return", "max_drawdown"], ascending=[False, False, False]
    ).reset_index(drop=True)

    compact_cols = [
        "name",
        "family",
        "final_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "delta_ann_return_vs_baseline",
        "delta_max_dd_vs_baseline",
        "number_of_trades",
        "trades_per_year",
        "exposure_percentage",
        "average_cash_weight",
        "dd_episodes_gt_20_30_40_50pct",
        "objective_score",
    ]
    compact = metrics[[column for column in compact_cols if column in metrics.columns]].copy()

    metrics.to_csv(tables_dir / f"{args.output_prefix}_metrics.csv", index=False)
    compact.to_csv(tables_dir / f"{args.output_prefix}_compact.csv", index=False)
    pd.DataFrame(returns_by_name).to_csv(tables_dir / f"{args.output_prefix}_returns.csv", index_label="date")
    # Store wide weights with a column MultiIndex flattened for parquet compatibility.
    weight_frames = []
    for name, frame in weights_by_name.items():
        renamed = frame.copy()
        renamed.columns = [f"{name}::{column}" for column in renamed.columns]
        weight_frames.append(renamed)
    pd.concat(weight_frames, axis=1).to_parquet(tables_dir / f"{args.output_prefix}_weights.parquet")
    pd.DataFrame({"date": common, **{key: value.values for key, value in cycle_flags.items()}, "qqq_pe_known": pe_known.values}).to_csv(
        tables_dir / f"{args.output_prefix}_fed_and_pe_flags.csv",
        index=False,
    )

    return_vs_dd_path = figures_dir / f"{args.output_prefix}_return_vs_drawdown.png"
    top_equity_path = figures_dir / f"{args.output_prefix}_top_candidates_equity_drawdown.png"
    worst_dd_path = figures_dir / f"{args.output_prefix}_worst_dd_comparison.png"
    _plot_return_vs_drawdown(metrics, return_vs_dd_path)

    eligible = metrics[
        metrics["name"].ne("QQQ_BH")
        & metrics["annualized_return"].ge(float(baseline["annualized_return"]) - 0.015)
    ].copy()
    selected = ["baseline_preferred_lock300_400_stop40"]
    selected += eligible.sort_values(["max_drawdown", "annualized_return"], ascending=[False, False])["name"].head(7).tolist()
    selected += eligible.sort_values("annualized_return", ascending=False)["name"].head(4).tolist()
    selected = list(dict.fromkeys(selected))[:10]
    _plot_equity_drawdown(
        returns_by_name,
        selected,
        top_equity_path,
        title="Drawdown-reduction experiments: selected candidates",
    )

    dd_selected = ["baseline_preferred_lock300_400_stop40"]
    dd_selected += metrics[metrics["name"].ne("QQQ_BH")].sort_values("max_drawdown", ascending=False)["name"].head(7).tolist()
    dd_selected = list(dict.fromkeys(dd_selected))[:8]
    _plot_equity_drawdown(
        returns_by_name,
        dd_selected,
        worst_dd_path,
        title="Candidates with the least severe max drawdown",
    )

    site_dir = resolve_path(root, args.site_dir)
    _write_site(
        site_dir=site_dir,
        metrics=metrics,
        compact=compact,
        figures={
            "Return vs drawdown": return_vs_dd_path,
            "Selected top candidates": top_equity_path,
            "Least severe max drawdown candidates": worst_dd_path,
        },
        output_prefix=args.output_prefix,
    )

    print("Drawdown-reduction experiments complete.")
    print(f"Compact table: {tables_dir / f'{args.output_prefix}_compact.csv'}")
    print(f"Full metrics:  {tables_dir / f'{args.output_prefix}_metrics.csv'}")
    print(f"Figures:       {return_vs_dd_path}, {top_equity_path}, {worst_dd_path}")
    print(f"Local site:    {site_dir / 'index.html'}")
    print("\nTop 12 compact rows:")
    print(compact.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
