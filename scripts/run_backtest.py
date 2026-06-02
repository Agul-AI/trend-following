#!/usr/bin/env python
"""Run a configured trend-following backtest and save metrics/plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.config import load_config
from trend_following.data_processing import build_adjusted_panels, load_processed_panels
from trend_following.experiments import (
    STRATEGY_NAMES,
    benchmark_backtests,
    run_strategy_backtest,
    strategy_default_params,
)
from trend_following.metrics import calculate_metrics, metrics_to_frame
from trend_following.plots import (
    plot_drawdowns,
    plot_equity_curves,
    plot_positions,
    plot_regime_diagnostic,
    plot_rolling_sharpe,
    plot_rolling_volatility,
)
from trend_following.regime import (
    classify_regimes,
    compute_regime_features,
    regime_confirmation_accuracy,
    regime_confirmation_table,
)
from trend_following.utils import as_list, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=STRATEGY_NAMES,
        help="Strategy to run",
    )
    parser.add_argument("--tickers", nargs="*", help="Optional ticker override")
    return parser.parse_args()


def _load_or_build_panels(config, tickers: list[str]) -> dict[str, pd.DataFrame]:
    try:
        panels = load_processed_panels(config.data.processed_dir)
        missing = [ticker for ticker in tickers if ticker not in panels["returns"].columns]
        if missing:
            raise FileNotFoundError(f"Processed panels missing requested tickers: {missing}")
        start = pd.Timestamp(config.data.start_date)
        if panels["returns"].index.min() > start:
            raise FileNotFoundError(
                "Processed panels start after configured start date: "
                f"{panels['returns'].index.min().date()} > {start.date()}"
            )
        return {
            name: frame[tickers]
            for name, frame in panels.items()
            if name in {"adjusted_close", "returns"}
        }
    except FileNotFoundError:
        panels = build_adjusted_panels(config, tickers=tickers)
        return {"adjusted_close": panels["adjusted_close"], "returns": panels["returns"]}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tickers = as_list(args.tickers) or config.data.tickers
    panels = _load_or_build_panels(config, tickers)
    prices = panels["adjusted_close"]
    returns = panels["returns"]

    params = strategy_default_params(config, args.strategy)
    result = run_strategy_backtest(prices, returns, args.strategy, params, config)
    benchmarks = benchmark_backtests(returns, config)

    metric_rows = []
    strategy_metrics = calculate_metrics(
        result.daily_returns,
        turnover=result.turnover,
        weights=result.weights,
        annualization=config.backtest.annualization,
    )
    strategy_metrics.update(
        {
            "name": args.strategy,
            "strategy": args.strategy,
            "segment": "full_sample",
            "parameters": json.dumps(params, sort_keys=True),
        }
    )
    metric_rows.append(strategy_metrics)

    for name, benchmark in benchmarks.items():
        row = calculate_metrics(
            benchmark.daily_returns,
            turnover=benchmark.turnover,
            weights=benchmark.weights,
            annualization=config.backtest.annualization,
        )
        row.update(
            {"name": name, "strategy": "benchmark", "segment": "full_sample", "parameters": "{}"}
        )
        metric_rows.append(row)

    metrics = metrics_to_frame(metric_rows)
    ensure_directory(config.reports.tables_dir)
    metrics_path = config.reports.tables_dir / f"{args.strategy}_metrics.csv"
    daily_path = config.reports.tables_dir / f"{args.strategy}_daily_results.csv"
    weights_path = config.reports.tables_dir / f"{args.strategy}_weights.csv"
    metrics.to_csv(metrics_path, index=False)
    result.to_frame().to_csv(daily_path)
    result.weights.to_csv(weights_path)

    equity_curves = {args.strategy: result.equity_curve}
    equity_curves.update({name: benchmark.equity_curve for name, benchmark in benchmarks.items()})
    return_streams = {args.strategy: result.daily_returns}
    return_streams.update({name: benchmark.daily_returns for name, benchmark in benchmarks.items()})

    ensure_directory(config.reports.figures_dir)
    plot_equity_curves(
        equity_curves,
        config.reports.figures_dir / f"{args.strategy}_equity_curve.png",
        title=f"{args.strategy} vs Benchmarks",
    )
    plot_drawdowns(
        return_streams,
        config.reports.figures_dir / f"{args.strategy}_drawdown.png",
        title=f"{args.strategy} Drawdown vs Benchmarks",
    )
    plot_rolling_sharpe(
        result.daily_returns,
        config.reports.figures_dir / f"{args.strategy}_rolling_sharpe.png",
        annualization=config.backtest.annualization,
    )
    plot_rolling_volatility(
        result.daily_returns,
        config.reports.figures_dir / f"{args.strategy}_rolling_volatility.png",
        annualization=config.backtest.annualization,
    )
    plot_positions(
        result.weights,
        config.reports.figures_dir / f"{args.strategy}_positions.png",
        ticker=tickers[0],
    )

    if args.strategy == "regime_switch":
        regime_ticker = str(params.get("regime_ticker", "QQQ"))
        features = compute_regime_features(
            prices,
            returns,
            regime_ticker=regime_ticker,
            params=params,
        )
        regimes = classify_regimes(features, params=params)
        regime_table = features.copy()
        regime_table["regime"] = regimes
        regimes_path = config.reports.tables_dir / "regime_switch_regimes.csv"
        regime_table.to_csv(regimes_path)
        confirmation = regime_confirmation_table(features, regimes, lag_days=1)
        confirmation_path = config.reports.tables_dir / "regime_switch_regime_confirmation.csv"
        confirmation.to_csv(confirmation_path)
        confirmation_accuracy = regime_confirmation_accuracy(
            confirmation,
            feature_ready_only=True,
        )
        all_valid_accuracy = regime_confirmation_accuracy(
            confirmation,
            feature_ready_only=False,
        )
        confirmation_summary = pd.DataFrame(
            [
                {
                    "lag_days": 1,
                    "feature_ready_only": True,
                    "accuracy": confirmation_accuracy,
                    "observations": int(
                        (
                            confirmation["regime_match"].notna()
                            & confirmation["feature_ready"]
                        ).sum()
                    ),
                },
                {
                    "lag_days": 1,
                    "feature_ready_only": False,
                    "accuracy": all_valid_accuracy,
                    "observations": int(confirmation["regime_match"].notna().sum()),
                },
            ]
        )
        confirmation_summary_path = (
            config.reports.tables_dir / "regime_switch_regime_confirmation_summary.csv"
        )
        confirmation_summary.to_csv(confirmation_summary_path, index=False)
        plot_regime_diagnostic(
            features["price"].rename(regime_ticker),
            regimes,
            config.reports.figures_dir / "regime_switch_regime_diagnostic.png",
            title=f"{regime_ticker} Regime Diagnostic",
        )
        print(f"Regime table saved to {regimes_path}")
        print(f"Regime confirmation saved to {confirmation_path}")
        print(f"Regime confirmation summary saved to {confirmation_summary_path}")
        print(f"Regime one-day lag confirmation accuracy: {confirmation_accuracy:.2%}")

    print(f"Metrics saved to {metrics_path}")
    print(f"Daily results saved to {daily_path}")
    print(f"Weights saved to {weights_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
