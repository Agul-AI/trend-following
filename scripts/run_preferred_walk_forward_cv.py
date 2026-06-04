#!/usr/bin/env python
"""Walk-forward validation for preferred QQQ/synthetic-TQQQ parameter variants.

For each fold, candidate parameters are scored on an expanding training window and
then evaluated on the next out-of-sample test window.  The frozen current
preferred rule and QQQ buy-and-hold are also reported for every test window.
"""

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

from preferred_cv_utils import (  # noqa: E402
    BENCHMARK_TICKER,
    CURRENT_PREFERRED_NAME,
    OFFICIAL_EVALUATION_START,
    TARGET_TICKER,
    benchmark_weights,
    build_candidate_raw_weight,
    executable_candidate_weights,
    load_cv_data,
    make_eval_args,
    metrics_from_simulated_slice,
    preferred_candidate_specs,
    simulate_weight_path,
)
from run_tqqq_entry_signal_comparison import _drawdown, _equity  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

FOLDS = [
    ("2007_2010", "2006-12-31 23:59:59", "2007-01-01", "2010-12-31 23:59:59"),
    ("2011_2014", "2010-12-31 23:59:59", "2011-01-01", "2014-12-31 23:59:59"),
    ("2015_2018", "2014-12-31 23:59:59", "2015-01-01", "2018-12-31 23:59:59"),
    ("2019_2022", "2018-12-31 23:59:59", "2019-01-01", "2022-12-31 23:59:59"),
    ("2023_2026", "2022-12-31 23:59:59", "2023-01-01", None),
]


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
    parser.add_argument("--output-prefix", default="preferred_walk_forward_cv")
    parser.add_argument("--site-dir", default="reports/site")
    parser.add_argument("--reuse-start-date-weights", action="store_true", default=True)
    return parser.parse_args()


def _train_objective(row: pd.Series) -> float:
    """Score training candidates while penalizing deep DD and excessive trading."""
    trades_penalty = max(float(row.get("trades_per_year", 0.0)) - 8.0, 0.0) * 0.01
    return (
        float(row["annualized_return"])
        + 0.35 * float(row["sharpe_ratio"])
        + 0.55 * float(row["max_drawdown"])
        - trades_penalty
    )


def _load_or_build_weights(
    *,
    config: Any,
    args: argparse.Namespace,
    target: pd.Series,
    qqq: pd.Series,
    bars_per_day: int,
) -> dict[str, pd.DataFrame]:
    weights_path = resolve_path(config.root, "reports/tables/preferred_start_date_cv_weights.parquet")
    specs = preferred_candidate_specs()
    if args.reuse_start_date_weights and weights_path.exists():
        stored = pd.read_parquet(weights_path)
        weights: dict[str, pd.DataFrame] = {}
        for spec in specs:
            if isinstance(stored.columns, pd.MultiIndex) and spec.name in stored.columns.get_level_values(0):
                part = stored[spec.name]
                if args.target_ticker in part.columns:
                    weights[spec.name] = part[[args.target_ticker]].copy()
        missing = [spec.name for spec in specs if spec.name not in weights]
        if not missing:
            return weights
        print(f"Start-date weights missing {len(missing)} candidate(s); rebuilding all weights.")

    weights = {}
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] building {spec.name}")
        raw_weight, _ = build_candidate_raw_weight(
            spec,
            target=target,
            qqq=qqq,
            bars_per_day=bars_per_day,
        )
        weights[spec.name] = executable_candidate_weights(
            raw_weight,
            config=config,
            target_ticker=args.target_ticker,
        )
    return weights


def _plot_walk_forward(
    returns_by_name: dict[str, pd.Series],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    for name, returns in returns_by_name.items():
        clean = returns.sort_index().fillna(0.0)
        _equity(clean).plot(ax=axes[0], label=name, linewidth=1.15)
        _drawdown(clean).plot(ax=axes[1], label=name, linewidth=1.15)
    axes[0].set_title("Walk-forward OOS equity")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Walk-forward OOS drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _make_site(site_dir: Path, chosen: pd.DataFrame, all_rows: pd.DataFrame, figure: Path) -> None:
    ensure_directory(site_dir)
    show = chosen.copy()
    for column in [
        "train_annualized_return",
        "train_max_drawdown",
        "test_annualized_return",
        "test_max_drawdown",
        "test_sharpe_ratio",
        "test_final_return",
    ]:
        if column in show.columns:
            show[column] = show[column].map(lambda value: f"{value:.3f}")
    top = all_rows.sort_values("train_objective", ascending=False).groupby("fold").head(10)
    html_path = site_dir / "walk_forward_cv.html"
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'>",
                "<title>Preferred Strategy Walk-Forward CV</title>",
                "<link rel='stylesheet' href='style.css'>",
                "</head><body>",
                "<h1>Preferred Strategy Walk-Forward Validation</h1>",
                "<p>Parameters are selected on expanding training windows and evaluated out-of-sample on the next period.</p>",
                "<h2>Chosen candidates by fold</h2>",
                show.to_html(index=False),
                "<h2>Top training candidates by fold</h2>",
                top[[c for c in ["fold", "name", "family", "train_objective", "annualized_return", "max_drawdown", "trades_per_year"] if c in top.columns]].to_html(index=False),
                f'<h2>{figure.name}</h2><img src="../figures/{figure.name}" alt="{figure.name}">',
                "</body></html>",
            ]
        )
    )
    index_path = site_dir / "index.html"
    if index_path.exists():
        index = index_path.read_text()
        link = '<li><a href="walk_forward_cv.html">Walk-Forward CV</a></li>'
        if "walk_forward_cv.html" not in index:
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
    specs = preferred_candidate_specs()
    weights_by_name = _load_or_build_weights(
        config=config,
        args=args,
        target=target,
        qqq=qqq,
        bars_per_day=bars_per_day,
    )
    qqq_bh = benchmark_weights(returns.index, args.benchmark_ticker)
    simulations = {
        name: simulate_weight_path(
            weights=weights,
            returns=returns,
            config=config,
            args=eval_args,
        )
        for name, weights in weights_by_name.items()
    }
    qqq_simulation = simulate_weight_path(
        weights=qqq_bh,
        returns=returns,
        config=config,
        args=eval_args,
    )

    all_train_rows: list[dict[str, Any]] = []
    chosen_rows: list[dict[str, Any]] = []
    selected_oos_returns: list[pd.Series] = []
    preferred_oos_returns: list[pd.Series] = []
    qqq_oos_returns: list[pd.Series] = []

    for fold_name, train_end, test_start, test_end in FOLDS:
        train_start = OFFICIAL_EVALUATION_START
        train_end_ts = pd.Timestamp(train_end)
        test_start_ts = pd.Timestamp(test_start)
        test_end_ts = pd.Timestamp(test_end) if test_end is not None else None
        print(f"Fold {fold_name}: train <= {train_end_ts}, test {test_start_ts} to {test_end_ts or 'end'}")

        train_metrics: list[dict[str, Any]] = []
        for spec in specs:
            row, _ = metrics_from_simulated_slice(
                name=spec.name,
                family=spec.family,
                simulation=simulations[spec.name],
                config=config,
                parameters=spec.params,
                start=train_start,
                end=train_end_ts,
                segment="walk_forward_train",
            )
            row["fold"] = fold_name
            row["train_objective"] = _train_objective(pd.Series(row))
            train_metrics.append(row)
            all_train_rows.append(row)

        train_frame = pd.DataFrame(train_metrics).sort_values(
            ["train_objective", "annualized_return", "max_drawdown"],
            ascending=[False, False, False],
        )
        selected = train_frame.iloc[0]
        selected_name = str(selected["name"])
        selected_spec = next(spec for spec in specs if spec.name == selected_name)

        test_row, test_returns = metrics_from_simulated_slice(
            name=selected_name,
            family=str(selected["family"]),
            simulation=simulations[selected_name],
            config=config,
            parameters=selected_spec.params,
            start=test_start_ts,
            end=test_end_ts,
            segment="walk_forward_test_selected",
        )
        preferred_row, preferred_returns = metrics_from_simulated_slice(
            name=CURRENT_PREFERRED_NAME,
            family="current_preferred",
            simulation=simulations[CURRENT_PREFERRED_NAME],
            config=config,
            parameters=next(spec.params for spec in specs if spec.name == CURRENT_PREFERRED_NAME),
            start=test_start_ts,
            end=test_end_ts,
            segment="walk_forward_test_preferred",
        )
        qqq_row, qqq_returns = metrics_from_simulated_slice(
            name="QQQ_BH",
            family="benchmark",
            simulation=qqq_simulation,
            config=config,
            parameters={},
            start=test_start_ts,
            end=test_end_ts,
            segment="walk_forward_test_benchmark",
        )
        selected_oos_returns.append(test_returns.rename(f"selected_{fold_name}"))
        preferred_oos_returns.append(preferred_returns.rename(f"preferred_{fold_name}"))
        qqq_oos_returns.append(qqq_returns.rename(f"qqq_{fold_name}"))

        chosen_rows.append(
            {
                "fold": fold_name,
                "train_start": train_start,
                "train_end": train_end_ts,
                "test_start": test_start_ts,
                "test_end": test_row["end_date"],
                "selected_name": selected_name,
                "selected_family": selected["family"],
                "train_objective": selected["train_objective"],
                "train_annualized_return": selected["annualized_return"],
                "train_max_drawdown": selected["max_drawdown"],
                "test_final_return": test_row["final_return"],
                "test_annualized_return": test_row["annualized_return"],
                "test_sharpe_ratio": test_row["sharpe_ratio"],
                "test_max_drawdown": test_row["max_drawdown"],
                "test_trades_per_year": test_row["trades_per_year"],
                "preferred_test_annualized_return": preferred_row["annualized_return"],
                "preferred_test_max_drawdown": preferred_row["max_drawdown"],
                "qqq_bh_test_annualized_return": qqq_row["annualized_return"],
                "qqq_bh_test_max_drawdown": qqq_row["max_drawdown"],
            }
        )

    all_rows = pd.DataFrame(all_train_rows)
    chosen = pd.DataFrame(chosen_rows)
    all_rows.to_csv(tables_dir / f"{args.output_prefix}_all_candidates.csv", index=False)
    chosen.to_csv(tables_dir / f"{args.output_prefix}.csv", index=False)

    def concat_fold_returns(series_list: list[pd.Series], name: str) -> pd.Series:
        pieces = [series.rename(name) for series in series_list]
        return pd.concat(pieces).sort_index().rename(name)

    returns_by_name = {
        "walk_forward_selected": concat_fold_returns(selected_oos_returns, "walk_forward_selected"),
        "current_preferred": concat_fold_returns(preferred_oos_returns, "current_preferred"),
        "QQQ_BH": concat_fold_returns(qqq_oos_returns, "QQQ_BH"),
    }
    pd.concat(returns_by_name.values(), axis=1).to_csv(
        tables_dir / f"{args.output_prefix}_returns.csv"
    )
    figure = figures_dir / f"{args.output_prefix}_equity_drawdown.png"
    _plot_walk_forward(returns_by_name, figure)
    _make_site(site_dir, chosen, all_rows, figure)

    print(f"Saved chosen folds: {tables_dir / f'{args.output_prefix}.csv'}")
    print(f"Saved all-candidate train table: {tables_dir / f'{args.output_prefix}_all_candidates.csv'}")
    print(f"Saved site: {site_dir / 'walk_forward_cv.html'}")
    print(chosen.to_string(index=False))


if __name__ == "__main__":
    main()
