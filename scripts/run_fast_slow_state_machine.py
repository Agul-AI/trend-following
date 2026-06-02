#!/usr/bin/env python
"""Run daily-regime-gated hourly fast-entry/slow-exit state-machine tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.backtest import backtest
from trend_following.config import load_config
from trend_following.data_validation import read_price_file
from trend_following.metrics import calculate_metrics, metrics_to_frame
from trend_following.plots import plot_drawdowns, plot_equity_curves, plot_positions
from trend_following.regime import (
    classify_regimes,
    compute_regime_features,
    daily_regime_hourly_fast_slow_signal,
)
from trend_following.signals import (
    limit_trades_per_day,
    make_executable_positions,
    signals_to_equal_weight_positions,
)
from trend_following.utils import ensure_directory, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/regime_hourly_qqq.yaml",
        help="Mixed-frequency config path.",
    )
    parser.add_argument(
        "--target-ticker",
        default=None,
        help="Ticker to trade. Defaults to strategies.regime_switch.target_ticker.",
    )
    parser.add_argument(
        "--regime-ticker",
        default=None,
        help="Daily regime ticker. Defaults to strategies.regime_switch.regime_ticker.",
    )
    parser.add_argument(
        "--intraday-raw-dir",
        default=None,
        help="Raw intraday parquet directory for the target ticker.",
    )
    parser.add_argument(
        "--daily-regime-raw-dir",
        default=None,
        help="Raw daily parquet directory for the regime ticker.",
    )
    parser.add_argument(
        "--benchmark-ticker",
        default="QQQ",
        help="Buy-and-hold benchmark ticker. Use '' to skip.",
    )
    parser.add_argument(
        "--benchmark-intraday-raw-dir",
        default="data/raw/alpha_vantage_60min",
        help="Raw intraday parquet directory for the benchmark ticker.",
    )
    parser.add_argument(
        "--daily-sma-windows",
        nargs="+",
        type=int,
        default=[200, 50, 20],
        help="Daily regime moving-average windows to test.",
    )
    parser.add_argument(
        "--entry-ma-days",
        type=float,
        default=20.0,
        help="Fast hourly entry MA window, measured in trading days.",
    )
    parser.add_argument(
        "--exit-ma-days",
        type=float,
        default=50.0,
        help="Slow hourly exit MA window, measured in trading days.",
    )
    parser.add_argument(
        "--entry-slope-days",
        type=float,
        default=5.0,
        help="Entry MA slope lookback, measured in trading days.",
    )
    parser.add_argument(
        "--entry-confirm-bars",
        type=int,
        default=1,
        help="Bars the fast-entry condition must persist.",
    )
    parser.add_argument(
        "--exit-confirm-bars",
        type=int,
        default=2,
        help="Bars the slow-exit condition must persist.",
    )
    parser.add_argument(
        "--entry-buffer",
        type=float,
        default=0.0,
        help="Require price above entry MA by this decimal buffer, e.g. 0.002.",
    )
    parser.add_argument(
        "--exit-buffer",
        type=float,
        default=0.0,
        help="Exit when price is below exit MA by this decimal buffer.",
    )
    parser.add_argument(
        "--output-prefix",
        default="fast_slow_state_machine",
        help="Prefix for report tables/figures.",
    )
    return parser.parse_args()


def _load_price_series(raw_dir: Path, ticker: str) -> pd.Series:
    path = raw_dir / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data for {ticker}: {path}")
    frame = read_price_file(path).sort_index()
    if "adj_close" not in frame.columns:
        raise ValueError(f"{path} is missing adj_close")
    return frame["adj_close"].astype(float).rename(ticker)


def _returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if not returns.empty:
        returns.iloc[0] = 0.0
    return returns


def _filter_range(
    series: pd.Series,
    start_date: str | None,
    end_date: str | None,
) -> pd.Series:
    filtered = series.copy()
    if start_date:
        filtered = filtered.loc[filtered.index >= pd.Timestamp(start_date)]
    if end_date:
        filtered = filtered.loc[filtered.index < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    if filtered.empty:
        raise ValueError(f"No rows remain for {series.name} after date filtering")
    return filtered


def _max_trades_per_day(weights: pd.DataFrame) -> tuple[int, int]:
    changes = weights.diff().abs().sum(axis=1).gt(1e-12)
    if changes.empty:
        return 0, 0
    trades_by_day = changes.groupby(changes.index.normalize()).sum()
    return int(trades_by_day.max()), int(trades_by_day.gt(1).sum())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_params = dict(config.strategies.regime_switch)

    target_ticker = args.target_ticker or str(
        base_params.get("target_ticker", config.data.tickers[0])
    )
    regime_ticker = args.regime_ticker or str(
        base_params.get("regime_ticker", target_ticker)
    )

    intraday_raw_dir = resolve_path(
        config.root,
        args.intraday_raw_dir or str(config.data.raw_dir),
    )
    daily_regime_raw_dir = resolve_path(
        config.root,
        args.daily_regime_raw_dir
        or str(base_params.get("daily_regime_raw_dir", "data/raw/alpha_vantage_daily_adjusted")),
    )
    benchmark_raw_dir = resolve_path(config.root, args.benchmark_intraday_raw_dir)

    target_price = _filter_range(
        _load_price_series(intraday_raw_dir, target_ticker),
        config.data.start_date,
        config.data.end_date,
    )
    target_prices = target_price.to_frame()
    target_returns = _returns_from_prices(target_prices)

    benchmark_ticker = args.benchmark_ticker.strip()
    benchmark_prices: pd.DataFrame | None = None
    benchmark_returns: pd.DataFrame | None = None
    if benchmark_ticker:
        benchmark_price = _filter_range(
            _load_price_series(benchmark_raw_dir, benchmark_ticker),
            config.data.start_date,
            config.data.end_date,
        )
        common_index = target_prices.index.intersection(benchmark_price.index)
        target_prices = target_prices.loc[common_index]
        target_returns = target_returns.loc[common_index]
        benchmark_prices = benchmark_price.loc[common_index].to_frame()
        benchmark_returns = _returns_from_prices(benchmark_prices)

    daily_price = _load_price_series(daily_regime_raw_dir, regime_ticker)
    daily_prices = daily_price.to_frame()
    daily_returns = _returns_from_prices(daily_prices)

    ensure_directory(config.reports.tables_dir)
    ensure_directory(config.reports.figures_dir)

    metric_rows: list[dict[str, object]] = []
    equity_curves: dict[str, pd.Series] = {}
    return_streams: dict[str, pd.Series] = {}
    positions_table = pd.DataFrame(index=target_returns.index)
    regime_count_rows: list[dict[str, object]] = []

    for daily_sma_window in args.daily_sma_windows:
        params = dict(base_params)
        params.update(
            {
                "target_ticker": target_ticker,
                "regime_ticker": regime_ticker,
                "daily_regime_raw_dir": str(daily_regime_raw_dir),
                "sma_window": int(daily_sma_window),
                "use_variance_ratio_for_trend": False,
                "state_machine_entry_ma_days": float(args.entry_ma_days),
                "state_machine_exit_ma_days": float(args.exit_ma_days),
                "state_machine_entry_slope_days": float(args.entry_slope_days),
                "state_machine_entry_confirm_bars": int(args.entry_confirm_bars),
                "state_machine_exit_confirm_bars": int(args.exit_confirm_bars),
                "state_machine_entry_buffer": float(args.entry_buffer),
                "state_machine_exit_buffer": float(args.exit_buffer),
            }
        )

        raw_signal = daily_regime_hourly_fast_slow_signal(
            intraday_prices=target_prices,
            daily_prices=daily_prices,
            daily_returns=daily_returns,
            params=params,
        )
        raw_weights = signals_to_equal_weight_positions(
            raw_signal,
            mode=str(params.get("portfolio_mode", config.backtest.portfolio_mode)),
        )
        positions = make_executable_positions(
            raw_weights,
            execution_delay_days=config.backtest.execution_delay_days,
            return_convention=config.backtest.return_convention,
        )
        positions = limit_trades_per_day(
            positions,
            max_trades_per_day=config.backtest.max_trades_per_day,
        ).reindex_like(target_returns)

        result = backtest(
            returns=target_returns,
            weights=positions,
            transaction_cost_bps=config.backtest.transaction_cost_bps,
            slippage_bps=config.backtest.slippage_bps,
            initial_capital=config.backtest.initial_capital,
        )
        label = f"Fast/slow state machine, daily {daily_sma_window}-day gate"
        max_trades, multi_trade_days = _max_trades_per_day(result.weights)
        metrics = calculate_metrics(
            result.daily_returns,
            turnover=result.turnover,
            weights=result.weights,
            annualization=config.backtest.annualization,
        )
        metrics.update(
            {
                "name": label,
                "strategy": "fast_slow_state_machine",
                "segment": "full_sample",
                "target": target_ticker,
                "regime_ticker": regime_ticker,
                "daily_regime_sma_window": daily_sma_window,
                "parameters": json.dumps(params, sort_keys=True),
                "max_trades_per_day_observed": max_trades,
                "days_with_more_than_one_trade": multi_trade_days,
            }
        )
        metric_rows.append(metrics)
        equity_curves[label] = result.equity_curve
        return_streams[label] = result.daily_returns
        positions_table[f"daily_{daily_sma_window}_gate_weight"] = result.weights[target_ticker]

        features = compute_regime_features(
            daily_prices,
            daily_returns,
            regime_ticker=regime_ticker,
            params=params,
        )
        regimes = classify_regimes(features, params=params)
        counts = regimes.value_counts(dropna=False)
        regime_count_rows.append(
            {
                "daily_regime_sma_window": daily_sma_window,
                "trend_days": int(counts.get("trend", 0)),
                "mean_reversion_days": int(counts.get("mean_reversion", 0)),
                "risk_off_days": int(counts.get("risk_off", 0)),
                "neutral_days": int(counts.get("neutral", 0)),
                "mean_reversion_pct": float(regimes.eq("mean_reversion").mean()),
                "trend_pct": float(regimes.eq("trend").mean()),
            }
        )

    if benchmark_ticker and benchmark_returns is not None:
        benchmark_weights = pd.DataFrame({benchmark_ticker: 1.0}, index=benchmark_returns.index)
        benchmark = backtest(
            returns=benchmark_returns,
            weights=benchmark_weights,
            initial_capital=config.backtest.initial_capital,
        )
        benchmark_metrics = calculate_metrics(
            benchmark.daily_returns,
            turnover=benchmark.turnover,
            weights=benchmark.weights,
            annualization=config.backtest.annualization,
        )
        benchmark_metrics.update(
            {
                "name": f"Buy & Hold {benchmark_ticker}",
                "strategy": "benchmark",
                "segment": "full_sample",
                "target": benchmark_ticker,
                "regime_ticker": "",
                "daily_regime_sma_window": np.nan,
                "parameters": "{}",
                "max_trades_per_day_observed": 1,
                "days_with_more_than_one_trade": 0,
            }
        )
        metric_rows.append(benchmark_metrics)
        equity_curves[f"Buy & Hold {benchmark_ticker}"] = benchmark.equity_curve
        return_streams[f"Buy & Hold {benchmark_ticker}"] = benchmark.daily_returns

    metrics_frame = metrics_to_frame(metric_rows)
    metrics_path = config.reports.tables_dir / f"{args.output_prefix}_metrics.csv"
    positions_path = config.reports.tables_dir / f"{args.output_prefix}_weights.csv"
    counts_path = config.reports.tables_dir / f"{args.output_prefix}_regime_counts.csv"
    equity_path = config.reports.figures_dir / f"{args.output_prefix}_equity_curve.png"
    drawdown_path = config.reports.figures_dir / f"{args.output_prefix}_drawdown.png"
    positions_plot_path = config.reports.figures_dir / f"{args.output_prefix}_positions.png"

    metrics_frame.to_csv(metrics_path, index=False)
    positions_table.to_csv(positions_path)
    pd.DataFrame(regime_count_rows).to_csv(counts_path, index=False)
    plot_equity_curves(
        equity_curves,
        equity_path,
        title=f"{target_ticker} fast-entry/slow-exit state machine",
    )
    plot_drawdowns(
        return_streams,
        drawdown_path,
        title=f"{target_ticker} fast-entry/slow-exit drawdowns",
    )
    plot_positions(positions_table, positions_plot_path)

    print(f"Metrics saved to {metrics_path}")
    print(f"Weights saved to {positions_path}")
    print(f"Regime counts saved to {counts_path}")
    print(f"Equity plot saved to {equity_path}")
    print(f"Drawdown plot saved to {drawdown_path}")
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
