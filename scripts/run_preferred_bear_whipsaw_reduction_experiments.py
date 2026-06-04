#!/usr/bin/env python
"""Run bear-market whipsaw-reduction overlays on the preferred QQQ/TQQQ rule."""

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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_preferred_drawdown_reduction_experiments import (  # noqa: E402
    _evaluate,
    _preferred_weight,
    _single_asset_weights,
)
from run_tqqq_daily_gate_ablation import no_daily_gate_hourly_ma_gate_signal  # noqa: E402
from run_tqqq_entry_signal_comparison import _drawdown, _equity, _returns_from_prices  # noqa: E402
from trend_following.bear_whipsaw import (  # noqa: E402
    bear_market_features,
    bear_reentry_filter_raw,
    failed_breakout_cooldown_raw,
    portfolio_drawdown_circuit_breaker,
    two_stage_bear_reentry_cap,
    volatility_cap,
)
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
BASELINE_NAME = "preferred_q100_baseline"
BEAR_WINDOW_START = pd.Timestamp("2007-10-01")
BEAR_WINDOW_END = pd.Timestamp("2009-12-31 23:59:59")


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
    parser.add_argument("--output-prefix", default="preferred_bear_whipsaw")
    parser.add_argument("--site-dir", default="reports/site")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def _add_dd_counts(metrics: dict[str, Any], returns: pd.Series) -> None:
    from run_tqqq_position_risk_sizing_experiments import drawdown_episode_count

    for threshold in (20, 30, 40, 50):
        metrics[f"drawdown_episodes_gt_{threshold}pct"] = drawdown_episode_count(
            returns,
            threshold=-threshold / 100.0,
        )
    metrics["dd_episodes_gt_20_30_40_50pct"] = (
        f"{metrics['drawdown_episodes_gt_20pct']}/"
        f"{metrics['drawdown_episodes_gt_30pct']}/"
        f"{metrics['drawdown_episodes_gt_40pct']}/"
        f"{metrics['drawdown_episodes_gt_50pct']}"
    )


def _worst_drawdown(returns: pd.Series) -> dict[str, Any]:
    equity = _equity(returns.fillna(0.0))
    drawdown = equity / equity.cummax() - 1.0
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax()
    recovery = equity.loc[trough:][equity.loc[trough:].ge(equity.loc[peak])]
    return {
        "peak": peak,
        "trough": trough,
        "recovery": recovery.index[0] if not recovery.empty else pd.NaT,
        "max_drawdown": float(drawdown.loc[trough]),
        "peak_equity": float(equity.loc[peak]),
        "trough_equity": float(equity.loc[trough]),
        "calendar_days_peak_to_trough": int((trough - peak).days),
    }


def _window_drawdown(returns: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    sample = returns.loc[(returns.index >= start) & (returns.index <= end)].fillna(0.0)
    if sample.empty:
        return float("nan")
    return float(_drawdown(sample).min())


def _eval_candidate(
    *,
    name: str,
    family: str,
    raw_weight: pd.Series | pd.DataFrame,
    returns: pd.DataFrame,
    config: Any,
    args: argparse.Namespace,
    parameters: dict[str, Any],
    overlay_diagnostics: pd.DataFrame | None,
) -> tuple[dict[str, Any], pd.Series, pd.DataFrame]:
    raw_weights = (
        raw_weight
        if isinstance(raw_weight, pd.DataFrame)
        else _single_asset_weights(raw_weight, args.target_ticker)
    )
    metrics, after_tax, weights = _evaluate(
        name=name,
        family=family,
        raw_weights=raw_weights,
        returns=returns,
        config=config,
        args=args,
        parameters=parameters,
    )
    _add_dd_counts(metrics, after_tax)
    metrics["bear_2007_2009_max_drawdown"] = _window_drawdown(
        after_tax,
        BEAR_WINDOW_START,
        BEAR_WINDOW_END,
    )
    metrics["overlay_trigger_count"] = _count_diag_flags(overlay_diagnostics, "trigger")
    metrics["overlay_active_bars"] = _count_active_bars(overlay_diagnostics)
    return metrics, after_tax, weights


def _count_diag_flags(diagnostics: pd.DataFrame | None, substring: str) -> int:
    if diagnostics is None or diagnostics.empty:
        return 0
    total = 0
    for column in diagnostics.columns:
        if substring in column and diagnostics[column].dtype == bool:
            total += int(diagnostics[column].sum())
    return total


def _count_active_bars(diagnostics: pd.DataFrame | None) -> int:
    if diagnostics is None or diagnostics.empty:
        return 0
    candidates = [
        column
        for column in diagnostics.columns
        if column.endswith("_active")
        or column.endswith("_staged")
        or column.endswith("_by_cooldown")
        or column.endswith("_entry")
    ]
    if not candidates:
        cap_cols = [column for column in diagnostics.columns if column.endswith("_cap")]
        if cap_cols:
            return int(diagnostics[cap_cols[0]].lt(1.0).sum())
        return 0
    active = pd.Series(False, index=diagnostics.index)
    for column in candidates:
        if diagnostics[column].dtype == bool:
            active |= diagnostics[column]
    return int(active.sum())


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


def _plot_return_vs_drawdown(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for family, group in metrics.groupby("family"):
        ax.scatter(
            group["bear_2007_2009_max_drawdown"] * 100.0,
            group["annualized_return"] * 100.0,
            s=70 if family == "baseline" else 42,
            alpha=0.75,
            label=family,
            edgecolor="black" if family == "baseline" else "none",
        )
    ax.axhline(24.0, color="#777777", linestyle=":", linewidth=1.0, label="24% ann. return")
    ax.set_xlabel("2007-2009 max drawdown (%)")
    ax.set_ylabel("Annualized return (%)")
    ax.set_title("Bear-whipsaw candidates: return vs 2007-2009 drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_top_equity_drawdown(
    returns_by_name: dict[str, pd.Series],
    metrics: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int = 6,
) -> None:
    selected = ["preferred_q100_baseline"]
    selected += (
        metrics.loc[
            metrics["family"].ne("benchmark") & metrics["name"].ne("preferred_q100_baseline")
        ]
        .head(top_n)["name"]
        .tolist()
    )
    selected = list(dict.fromkeys([name for name in selected if name in returns_by_name]))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    for name in selected:
        returns = returns_by_name[name]
        _equity(returns).plot(ax=axes[0], label=name, linewidth=1.15)
        _drawdown(returns).plot(ax=axes[1], label=name, linewidth=1.15)
    axes[0].set_title("After-tax equity")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("After-tax drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("Top bear-whipsaw candidates")
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _make_site(
    *,
    site_dir: Path,
    compact: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    ensure_directory(site_dir)
    html_path = site_dir / "bear_whipsaw.html"
    display = compact.head(20).copy()
    for column in [
        "final_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "bear_2007_2009_max_drawdown",
        "trades_per_year",
    ]:
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:.3f}")
    figures_html = "\n".join(
        f'<h2>{path.name}</h2><img src="../figures/{path.name}" alt="{path.name}">'
        for path in figure_paths
    )
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'>",
                "<title>Bear Whipsaw Reduction</title>",
                "<link rel='stylesheet' href='style.css'>",
                "</head><body>",
                "<h1>Bear Whipsaw Reduction Experiments</h1>",
                "<p>Updated preferred q100 strategy plus targeted bear-market whipsaw overlays.</p>",
                display.to_html(index=False),
                figures_html,
                "</body></html>",
            ]
        )
    )
    index_path = site_dir / "index.html"
    if index_path.exists():
        index = index_path.read_text()
        link = '<li><a href="bear_whipsaw.html">Bear Whipsaw Reduction</a></li>'
        if "bear_whipsaw.html" not in index:
            if "</body>" in index:
                index = index.replace(
                    "</body>", f"<h2>Additional experiment pages</h2><ul>{link}</ul></body>"
                )
            else:
                index += f"\n<ul>{link}</ul>\n"
            index_path.write_text(index)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tables_dir = resolve_path(config.root, "reports/tables")
    figures_dir = resolve_path(config.root, "reports/figures")
    site_dir = resolve_path(config.root, args.site_dir)
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

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
    profit_locked_weight, _ = _preferred_weight(
        raw_base=raw_base,
        target_price=target,
        stop_drawdown=0.40,
    )
    mr_features = qqq_mean_reversion_features(qqq, bars_per_day=bars_per_day)
    q100 = dynamic_pre100_distance_trim_rebuy_cap(
        profit_locked_weight.gt(0).astype(float),
        target,
        mr_features,
        activation_gain=1.0,
        threshold_quantile=1.0,
        trim_weight=0.50,
        reentry_rule="ma20",
    )
    preferred_weight = apply_cap(profit_locked_weight, q100.weights).rename(args.target_ticker)
    preferred_binary = preferred_weight.gt(0.0).astype(float)
    features = bear_market_features(qqq, bars_per_day=bars_per_day)

    eval_args = argparse.Namespace(
        target_ticker=args.target_ticker,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        short_term_tax_rate=args.short_term_tax_rate,
        cash_annual_yield=args.cash_annual_yield,
        cash_interest_tax_rate=args.cash_interest_tax_rate,
    )

    metrics_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.DataFrame] = {}
    diagnostics_by_name: dict[str, pd.DataFrame] = {}

    def add_candidate(
        name: str,
        family: str,
        raw_weight: pd.Series | pd.DataFrame,
        parameters: dict[str, Any],
        diagnostics: pd.DataFrame | None = None,
    ) -> None:
        metrics, after_tax, weights = _eval_candidate(
            name=name,
            family=family,
            raw_weight=raw_weight,
            returns=returns,
            config=config,
            args=eval_args,
            parameters=parameters,
            overlay_diagnostics=diagnostics,
        )
        metrics_rows.append(metrics)
        returns_by_name[name] = after_tax
        weights_by_name[name] = weights
        if diagnostics is not None:
            diagnostics_by_name[name] = diagnostics

    add_candidate(
        BASELINE_NAME,
        "baseline",
        preferred_weight,
        {"preferred": "q100_profit_lock_stop40"},
        q100.diagnostics,
    )

    for buffer in (0.01, 0.02, 0.03):
        for slope_days in (5, 10, 20):
            for require_short in (False, True):
                overlay = bear_reentry_filter_raw(
                    preferred_binary,
                    features,
                    distance_buffer=buffer,
                    slope_days=slope_days,
                    require_short_gt_medium=require_short,
                )
                name = (
                    f"bear_reentry_buf{int(buffer * 100)}_slope{slope_days}"
                    f"{'_20gt50' if require_short else ''}"
                )
                add_candidate(
                    name,
                    "bear_reentry_filter",
                    apply_cap(preferred_weight, overlay.weights),
                    {
                        "distance_buffer": buffer,
                        "slope_days": slope_days,
                        "require_short_gt_medium": require_short,
                    },
                    overlay.diagnostics,
                )

    for weak_count in (2, 3):
        for lookback_days in (90, 180):
            overlay = failed_breakout_cooldown_raw(
                preferred_binary,
                target,
                features,
                weak_trade_return=0.03,
                weak_trade_count=weak_count,
                lookback_days=lookback_days,
                slope_days=10,
                distance_buffer=0.02,
            )
            name = f"cooldown_weak{weak_count}_lookback{lookback_days}"
            add_candidate(
                name,
                "failed_breakout_cooldown",
                apply_cap(preferred_weight, overlay.weights),
                {
                    "weak_trade_return": 0.03,
                    "weak_trade_count": weak_count,
                    "lookback_days": lookback_days,
                    "release": "50MA_slope_positive_or_QQQ_2pct_above_200MA",
                },
                overlay.diagnostics,
            )

    for percentile in (0.80, 0.90, 0.95):
        for cap in (0.50, 0.25, 0.0):
            overlay = volatility_cap(
                features,
                percentile_threshold=percentile,
                defensive_cap=cap,
            )
            name = f"vol_cap_p{int(percentile * 100)}_to{int(cap * 100)}"
            add_candidate(
                name,
                "crisis_volatility_sizing",
                apply_cap(preferred_weight, overlay.weights),
                {"percentile_threshold": percentile, "defensive_cap": cap},
                overlay.diagnostics,
            )

    target_returns = returns[args.target_ticker]
    for trigger_dd in (0.25, 0.30, 0.35):
        for cap in (0.50, 0.25, 0.0):
            for recover_dd in (0.10, 0.15):
                overlay = portfolio_drawdown_circuit_breaker(
                    preferred_weight,
                    target_returns,
                    features,
                    trigger_drawdown=trigger_dd,
                    recover_drawdown=recover_dd,
                    defensive_cap=cap,
                    slope_days=10,
                )
                name = f"circuit_dd{int(trigger_dd * 100)}_to{int(cap * 100)}_rec{int(recover_dd * 100)}"
                add_candidate(
                    name,
                    "portfolio_dd_circuit_breaker",
                    apply_cap(preferred_weight, overlay.weights),
                    {
                        "trigger_drawdown": trigger_dd,
                        "defensive_cap": cap,
                        "recover_drawdown": recover_dd,
                    },
                    overlay.diagnostics,
                )

    for initial_weight in (0.25, 0.50):
        for release_rule in ("medium_slope", "short_gt_medium"):
            overlay = two_stage_bear_reentry_cap(
                preferred_weight,
                features,
                initial_weight=initial_weight,
                slope_days=10,
                release_rule=release_rule,
            )
            name = f"two_stage_to{int(initial_weight * 100)}_rel_{release_rule}"
            add_candidate(
                name,
                "two_stage_bear_reentry",
                apply_cap(preferred_weight, overlay.weights),
                {
                    "initial_weight": initial_weight,
                    "slope_days": 10,
                    "release_rule": release_rule,
                },
                overlay.diagnostics,
            )

    add_candidate(
        "QQQ_BH",
        "benchmark",
        pd.DataFrame({args.benchmark_ticker: pd.Series(1.0, index=common)}),
        {},
        None,
    )

    metrics = metrics_to_frame(metrics_rows)
    baseline = metrics.loc[metrics["name"].eq(BASELINE_NAME)].iloc[0]
    metrics["delta_ann_return_vs_baseline"] = metrics["annualized_return"] - float(
        baseline["annualized_return"]
    )
    metrics["delta_max_dd_vs_baseline"] = metrics["max_drawdown"] - float(baseline["max_drawdown"])
    metrics["delta_bear_2007_2009_dd_vs_baseline"] = metrics["bear_2007_2009_max_drawdown"] - float(
        baseline["bear_2007_2009_max_drawdown"]
    )
    metrics["passes_candidate_filter"] = metrics["annualized_return"].ge(0.24) | metrics[
        "max_drawdown"
    ].gt(-0.50)
    metrics["promising_candidate"] = metrics["delta_max_dd_vs_baseline"].ge(0.03) | metrics[
        "delta_bear_2007_2009_dd_vs_baseline"
    ].ge(0.05)
    metrics["objective_score"] = (
        metrics["annualized_return"]
        + 0.45 * metrics["delta_max_dd_vs_baseline"]
        + 0.55 * metrics["delta_bear_2007_2009_dd_vs_baseline"]
        - 0.0025 * metrics["number_of_trades"].clip(lower=0) / 10.0
    )
    metrics = metrics.sort_values(
        ["objective_score", "annualized_return", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    compact_cols = [
        "name",
        "family",
        "final_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "bear_2007_2009_max_drawdown",
        "delta_max_dd_vs_baseline",
        "delta_bear_2007_2009_dd_vs_baseline",
        "number_of_trades",
        "trades_per_year",
        "exposure_percentage",
        "dd_episodes_gt_20_30_40_50pct",
        "overlay_trigger_count",
        "overlay_active_bars",
        "passes_candidate_filter",
        "promising_candidate",
        "objective_score",
    ]
    compact = metrics[[column for column in compact_cols if column in metrics.columns]]
    metrics.to_csv(tables_dir / f"{args.output_prefix}_experiments_metrics.csv", index=False)
    compact.to_csv(tables_dir / f"{args.output_prefix}_experiments_compact.csv", index=False)
    pd.concat(returns_by_name.values(), axis=1).to_csv(
        tables_dir / f"{args.output_prefix}_experiments_returns.csv"
    )
    pd.concat(weights_by_name, axis=1).to_parquet(
        tables_dir / f"{args.output_prefix}_experiments_weights.parquet"
    )
    if diagnostics_by_name:
        pd.concat(diagnostics_by_name, axis=1).to_parquet(
            tables_dir / f"{args.output_prefix}_experiments_diagnostics.parquet"
        )

    worst_rows = []
    for name, after_tax in returns_by_name.items():
        row = {"name": name, **_worst_drawdown(after_tax)}
        worst_rows.append(row)
    pd.DataFrame(worst_rows).to_csv(
        tables_dir / f"{args.output_prefix}_worst_drawdown_summary.csv",
        index=False,
    )

    focused_rows = []
    for name, diagnostics in diagnostics_by_name.items():
        window = diagnostics.loc[
            (diagnostics.index >= BEAR_WINDOW_START) & (diagnostics.index <= BEAR_WINDOW_END)
        ]
        focused_rows.append(
            {
                "name": name,
                "family": metrics.set_index("name").at[name, "family"],
                "overlay_trigger_count_2007_2009": _count_diag_flags(window, "trigger"),
                "overlay_active_bars_2007_2009": _count_active_bars(window),
            }
        )
    pd.DataFrame(focused_rows).to_csv(
        tables_dir / f"{args.output_prefix}_2007_2009_trigger_summary.csv",
        index=False,
    )

    cycles = cycles_to_frame(load_cycle_config(resolve_path(config.root, args.fed_cycle_config)))
    cycles.to_csv(tables_dir / f"{args.output_prefix}_fed_cycles.csv", index=False)
    selected_names = [BASELINE_NAME, "QQQ_BH"] + metrics.loc[
        metrics["family"].ne("benchmark") & metrics["name"].ne(BASELINE_NAME)
    ].head(5)["name"].tolist()
    cycle_perf = _cycle_metrics(
        {name: returns_by_name[name] for name in selected_names if name in returns_by_name},
        cycles.loc[cycles["cycle_group"].eq("pre_announcement_to_pre_cut")],
        annualization=config.backtest.annualization,
    )
    cycle_perf.to_csv(
        tables_dir / f"{args.output_prefix}_hiking_cycle_performance.csv", index=False
    )

    scatter_path = figures_dir / f"{args.output_prefix}_return_vs_2007_2009_dd.png"
    top_path = figures_dir / f"{args.output_prefix}_top_equity_drawdown.png"
    _plot_return_vs_drawdown(metrics, scatter_path)
    _plot_top_equity_drawdown(returns_by_name, metrics, top_path)
    _make_site(site_dir=site_dir, compact=compact, figure_paths=[scatter_path, top_path])

    print(f"Saved compact metrics: {tables_dir / f'{args.output_prefix}_experiments_compact.csv'}")
    print(
        f"Saved worst drawdown summary: {tables_dir / f'{args.output_prefix}_worst_drawdown_summary.csv'}"
    )
    print(
        f"Saved focused trigger summary: {tables_dir / f'{args.output_prefix}_2007_2009_trigger_summary.csv'}"
    )
    print(f"Saved website page: {site_dir / 'bear_whipsaw.html'}")
    print("Top rows:")
    print(compact.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
