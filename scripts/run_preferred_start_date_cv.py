#!/usr/bin/env python
"""Cross-validate the preferred QQQ/synthetic-TQQQ strategy across start dates.

This is a robustness test around the current preferred rule, not a full-sample
optimizer.  It evaluates controlled one-factor parameter variants from multiple
official evaluation start dates, always comparing QQQ buy-and-hold from the same
start date.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from preferred_cv_utils import (  # noqa: E402
    BENCHMARK_TICKER,
    CURRENT_PREFERRED_NAME,
    START_DATES,
    TARGET_TICKER,
    benchmark_weights,
    build_candidate_raw_weight,
    executable_candidate_weights,
    load_cv_data,
    make_eval_args,
    metrics_from_simulated_slice,
    preferred_candidate_specs,
    robustness_summary,
    simulate_weight_path,
)
from trend_following.config import load_config  # noqa: E402
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402


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
    parser.add_argument("--start-dates", default=",".join(START_DATES))
    parser.add_argument("--output-prefix", default="preferred_start_date_cv")
    parser.add_argument("--site-dir", default="reports/site")
    parser.add_argument("--top-heatmap-n", type=int, default=30)
    return parser.parse_args()


def _parse_start_dates(value: str) -> list[pd.Timestamp]:
    starts = [pd.Timestamp(item.strip()) for item in value.split(",") if item.strip()]
    if not starts:
        raise ValueError("At least one start date is required")
    return starts


def _plot_heatmap(metrics: pd.DataFrame, summary: pd.DataFrame, output_path: Path, *, top_n: int) -> None:
    selected = summary.head(top_n)["name"].tolist()
    selected = list(dict.fromkeys([CURRENT_PREFERRED_NAME] + selected))
    sample = metrics.loc[metrics["name"].isin(selected) & metrics["family"].ne("benchmark")].copy()
    sample["start_label"] = pd.to_datetime(sample["start_date"]).dt.strftime("%Y-%m-%d")
    pivot = sample.pivot_table(
        index="name",
        columns="start_label",
        values="annualized_return",
        aggfunc="first",
    )
    pivot = pivot.reindex(selected).dropna(how="all")
    fig_height = max(7.0, 0.28 * len(pivot) + 2.0)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    data = pivot.to_numpy(dtype=float) * 100.0
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=np.nanpercentile(data, 5), vmax=np.nanpercentile(data, 95))
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title("Start-date CV: annualized return by candidate and evaluation start")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Annualized return (%)")
    ax.grid(False)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_rank_stability(metrics: pd.DataFrame, output_path: Path, *, top_names: list[str]) -> None:
    rows: list[pd.DataFrame] = []
    for start, group in metrics.loc[metrics["family"].ne("benchmark")].groupby("start_date"):
        ranked = group.sort_values(
            ["annualized_return", "max_drawdown"], ascending=[False, False]
        ).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked["start_date"] = start
        rows.append(ranked[["name", "start_date", "rank"]])
    ranks = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if ranks.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 6.5))
    for name in top_names:
        series = ranks.loc[ranks["name"].eq(name)].copy()
        if series.empty:
            continue
        series["start_date"] = pd.to_datetime(series["start_date"])
        ax.plot(series["start_date"], series["rank"], marker="o", linewidth=1.2, label=name)
    ax.invert_yaxis()
    ax.set_title("Rank stability across evaluation start dates")
    ax.set_ylabel("Annualized-return rank; lower is better")
    ax.set_xlabel("Evaluation start")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _make_site(site_dir: Path, summary: pd.DataFrame, metrics: pd.DataFrame, figures: list[Path]) -> None:
    ensure_directory(site_dir)
    html_path = site_dir / "start_date_cv.html"
    show = summary.head(40).copy()
    for column in [
        "median_annualized_return",
        "p10_annualized_return",
        "min_annualized_return",
        "median_sharpe",
        "worst_max_drawdown",
        "median_trades_per_year",
        "robustness_score",
    ]:
        if column in show.columns:
            show[column] = show[column].map(lambda value: f"{value:.3f}")
    benchmark = metrics.loc[metrics["family"].eq("benchmark")].copy()
    for column in ["annualized_return", "max_drawdown", "sharpe_ratio", "final_return"]:
        if column in benchmark.columns:
            benchmark[column] = benchmark[column].map(lambda value: f"{value:.3f}")
    figures_html = "\n".join(
        f'<h2>{path.name}</h2><img src="../figures/{path.name}" alt="{path.name}">'
        for path in figures
    )
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'>",
                "<title>Preferred Strategy Start-Date CV</title>",
                "<link rel='stylesheet' href='style.css'>",
                "</head><body>",
                "<h1>Preferred Strategy Start-Date Cross-Validation</h1>",
                "<p>Controlled parameter variants evaluated from multiple official start dates. QQQ buy-and-hold is aligned to each same start date.</p>",
                "<h2>Robustness summary</h2>",
                show.to_html(index=False),
                "<h2>QQQ benchmark rows by start date</h2>",
                benchmark[[c for c in ["name", "start_date", "annualized_return", "max_drawdown", "final_return"] if c in benchmark.columns]].to_html(index=False),
                figures_html,
                "</body></html>",
            ]
        )
    )
    index_path = site_dir / "index.html"
    if index_path.exists():
        index = index_path.read_text()
        link = '<li><a href="start_date_cv.html">Start-Date CV</a></li>'
        if "start_date_cv.html" not in index:
            index = index.replace("</body>", f"<h2>Additional experiment pages</h2><ul>{link}</ul></body>")
            index_path.write_text(index)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tables_dir = resolve_path(config.root, "reports/tables")
    figures_dir = resolve_path(config.root, "reports/figures")
    site_dir = resolve_path(config.root, args.site_dir)
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

    target, qqq, returns, bars_per_day = load_cv_data(config, args)
    eval_args = make_eval_args(args)
    starts = _parse_start_dates(args.start_dates)
    specs = preferred_candidate_specs()

    metrics_rows: list[dict] = []
    full_weights: dict[str, pd.DataFrame] = {}
    diagnostics_rows: list[dict] = []

    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] building {spec.name}")
        raw_weight, diag = build_candidate_raw_weight(
            spec,
            target=target,
            qqq=qqq,
            bars_per_day=bars_per_day,
        )
        weights = executable_candidate_weights(
            raw_weight,
            config=config,
            target_ticker=args.target_ticker,
        )
        full_weights[spec.name] = weights
        simulation = simulate_weight_path(
            weights=weights,
            returns=returns,
            config=config,
            args=eval_args,
        )
        diagnostics_rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "parameters": spec.params,
                "q100_trigger_count": int(diag["q100_trigger"].sum()),
                "bear_blocked_entry_count": int(diag["bear_blocked_entry"].sum()),
                "stop_trigger_count": int(diag["stop_trigger"].sum()),
            }
        )
        for start in starts:
            row, _ = metrics_from_simulated_slice(
                name=spec.name,
                family=spec.family,
                simulation=simulation,
                config=config,
                parameters=spec.params,
                start=start,
                segment="start_date_cv",
            )
            metrics_rows.append(row)

    qqq_bh = benchmark_weights(returns.index, args.benchmark_ticker)
    qqq_simulation = simulate_weight_path(
        weights=qqq_bh,
        returns=returns,
        config=config,
        args=eval_args,
    )
    for start in starts:
        row, _ = metrics_from_simulated_slice(
            name="QQQ_BH",
            family="benchmark",
            simulation=qqq_simulation,
            config=config,
            parameters={},
            start=start,
            segment="start_date_cv",
        )
        metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows)
    summary = robustness_summary(metrics)
    diagnostics = pd.DataFrame(diagnostics_rows)

    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    summary_path = tables_dir / f"{args.output_prefix}_summary.csv"
    rank_path = tables_dir / "preferred_parameter_robustness_rank.csv"
    diagnostics_path = tables_dir / f"{args.output_prefix}_diagnostics.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.parquet"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(rank_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    pd.concat(full_weights, axis=1).to_parquet(weights_path)

    heatmap_path = figures_dir / f"{args.output_prefix}_heatmap.png"
    rank_path_fig = figures_dir / "preferred_parameter_rank_stability.png"
    _plot_heatmap(metrics, summary, heatmap_path, top_n=args.top_heatmap_n)
    top_rank_names = list(dict.fromkeys([CURRENT_PREFERRED_NAME] + summary.head(8)["name"].tolist()))
    _plot_rank_stability(metrics, rank_path_fig, top_names=top_rank_names)
    _make_site(site_dir, summary, metrics, [heatmap_path, rank_path_fig])

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved site: {site_dir / 'start_date_cv.html'}")
    print("Top robustness rows:")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
