#!/usr/bin/env python
"""Run hourly trend following gated by a daily no-lookahead regime estimate."""

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
from trend_following.data_processing import build_adjusted_panels, load_processed_panels
from trend_following.data_validation import (
    read_price_file,
    validate_price_frame,
    validation_has_fatal_issue,
)
from trend_following.metrics import calculate_metrics, metrics_to_frame
from trend_following.plots import (
    plot_drawdowns,
    plot_equity_curves,
    plot_positions,
    plot_rolling_sharpe,
    plot_rolling_volatility,
)
from trend_following.regime import (
    align_daily_regimes_to_intraday,
    classify_regimes,
    compute_regime_features,
    daily_regime_hourly_trend_signal,
)
from trend_following.signals import (
    limit_trades_per_day,
    make_executable_positions,
    signals_to_equal_weight_positions,
)
from trend_following.utils import ensure_directory, resolve_path

STRATEGY_NAME = "regime_switch_hourly_daily_regime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/regime_hourly_qqq.yaml",
        help="Path to mixed-frequency YAML config",
    )
    parser.add_argument("--ticker", help="Target ticker override; default uses config")
    parser.add_argument(
        "--daily-regime-raw-dir",
        help="Daily raw parquet directory override; default uses strategies.regime_switch",
    )
    return parser.parse_args()


def _load_or_build_intraday_panels(config, tickers: list[str]) -> dict[str, pd.DataFrame]:
    try:
        panels = load_processed_panels(config.data.processed_dir)
        missing = [ticker for ticker in tickers if ticker not in panels["returns"].columns]
        if missing:
            raise FileNotFoundError(f"Processed panels missing requested tickers: {missing}")
        return {
            name: frame[tickers]
            for name, frame in panels.items()
            if name in {"adjusted_close", "returns"}
        }
    except FileNotFoundError:
        panels = build_adjusted_panels(config, tickers=tickers)
        return {"adjusted_close": panels["adjusted_close"], "returns": panels["returns"]}


def _load_daily_panels(raw_dir: Path, tickers: list[str]) -> dict[str, pd.DataFrame]:
    adjusted_close: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = raw_dir / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing daily regime raw data for {ticker}: {path}")
        frame = read_price_file(path).sort_index()
        report = validate_price_frame(frame, ticker=ticker, suspicious_gap_days=7)
        if validation_has_fatal_issue(report):
            raise ValueError(f"Fatal daily data issue for {ticker}: {report['messages']}")
        adjusted_close[ticker] = frame["adj_close"].astype(float)

    close = pd.DataFrame(adjusted_close).sort_index()
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if not returns.empty:
        returns.iloc[0] = 0.0
    return {"adjusted_close": close, "returns": returns}


def _filter_intraday_range(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = pd.Series(True, index=prices.index)
    if start_date:
        mask &= prices.index >= pd.Timestamp(start_date)
    if end_date:
        # Include all intraday bars on the configured end date.
        mask &= prices.index < pd.Timestamp(end_date) + pd.Timedelta(days=1)
    filtered_prices = prices.loc[mask]
    filtered_returns = returns.loc[filtered_prices.index]
    if filtered_prices.empty:
        raise ValueError("No intraday rows remain after applying the configured date range")
    return filtered_prices, filtered_returns


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    params = dict(config.strategies.regime_switch)

    ticker = args.ticker or str(params.get("target_ticker", config.data.tickers[0]))
    params["target_ticker"] = ticker
    params.setdefault("regime_ticker", ticker)
    regime_ticker = str(params["regime_ticker"])

    intraday_panels = _load_or_build_intraday_panels(config, [ticker])
    intraday_prices, intraday_returns = _filter_intraday_range(
        intraday_panels["adjusted_close"],
        intraday_panels["returns"],
        config.data.start_date,
        config.data.end_date,
    )

    daily_raw_dir_value = (
        args.daily_regime_raw_dir
        or params.get("daily_regime_raw_dir")
        or "data/raw/alpha_vantage_daily_adjusted"
    )
    daily_raw_dir = resolve_path(config.root, daily_raw_dir_value)
    daily_panels = _load_daily_panels(daily_raw_dir, [regime_ticker])
    daily_prices = daily_panels["adjusted_close"]
    daily_returns = daily_panels["returns"]

    raw_signal = daily_regime_hourly_trend_signal(
        intraday_prices=intraday_prices,
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
    ).reindex_like(intraday_returns)

    result = backtest(
        returns=intraday_returns,
        weights=positions,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
        slippage_bps=config.backtest.slippage_bps,
        initial_capital=config.backtest.initial_capital,
    )
    benchmark_weights = pd.DataFrame({ticker: 1.0}, index=intraday_returns.index)
    benchmark = backtest(
        returns=intraday_returns,
        weights=benchmark_weights,
        initial_capital=config.backtest.initial_capital,
    )

    metric_rows = []
    strategy_metrics = calculate_metrics(
        result.daily_returns,
        turnover=result.turnover,
        weights=result.weights,
        annualization=config.backtest.annualization,
    )
    strategy_metrics.update(
        {
            "name": STRATEGY_NAME,
            "strategy": STRATEGY_NAME,
            "segment": "full_sample",
            "parameters": json.dumps(params, sort_keys=True),
        }
    )
    metric_rows.append(strategy_metrics)

    benchmark_metrics = calculate_metrics(
        benchmark.daily_returns,
        turnover=benchmark.turnover,
        weights=benchmark.weights,
        annualization=config.backtest.annualization,
    )
    benchmark_metrics.update(
        {
            "name": f"Buy & Hold {ticker}",
            "strategy": "benchmark",
            "segment": "full_sample",
            "parameters": "{}",
        }
    )
    metric_rows.append(benchmark_metrics)
    metrics = metrics_to_frame(metric_rows)

    ensure_directory(config.reports.tables_dir)
    ensure_directory(config.reports.figures_dir)
    metrics_path = config.reports.tables_dir / f"{STRATEGY_NAME}_metrics.csv"
    results_path = config.reports.tables_dir / f"{STRATEGY_NAME}_intraday_results.csv"
    weights_path = config.reports.tables_dir / f"{STRATEGY_NAME}_weights.csv"
    daily_regimes_path = config.reports.tables_dir / f"{STRATEGY_NAME}_daily_regimes.csv"
    alignment_path = config.reports.tables_dir / f"{STRATEGY_NAME}_intraday_alignment.csv"

    metrics.to_csv(metrics_path, index=False)
    result.to_frame().to_csv(results_path)
    result.weights.to_csv(weights_path)

    daily_features = compute_regime_features(
        daily_prices,
        daily_returns,
        regime_ticker=regime_ticker,
        params=params,
    )
    daily_regimes = classify_regimes(daily_features, params=params)
    daily_table = daily_features.copy()
    daily_table["regime"] = daily_regimes
    daily_table.to_csv(daily_regimes_path)

    intraday_regimes = align_daily_regimes_to_intraday(
        daily_regimes,
        intraday_prices.index,
        lag_days=int(params.get("daily_regime_lag_days", 1)),
        fill_method=params.get("daily_regime_fill_method", "ffill"),
    ).fillna("neutral")
    pd.DataFrame(
        {
            "price": intraday_prices[ticker],
            "daily_regime_available_at_bar": intraday_regimes,
            "raw_signal": raw_signal[ticker],
            "executable_weight": result.weights[ticker],
        }
    ).to_csv(alignment_path)

    equity_curves = {
        STRATEGY_NAME: result.equity_curve,
        f"Buy & Hold {ticker}": benchmark.equity_curve,
    }
    return_streams = {
        STRATEGY_NAME: result.daily_returns,
        f"Buy & Hold {ticker}": benchmark.daily_returns,
    }
    plot_equity_curves(
        equity_curves,
        config.reports.figures_dir / f"{STRATEGY_NAME}_equity_curve.png",
        title=f"{STRATEGY_NAME} vs Buy & Hold {ticker}",
    )
    plot_drawdowns(
        return_streams,
        config.reports.figures_dir / f"{STRATEGY_NAME}_drawdown.png",
        title=f"{STRATEGY_NAME} Drawdown",
    )
    plot_rolling_sharpe(
        result.daily_returns,
        config.reports.figures_dir / f"{STRATEGY_NAME}_rolling_sharpe.png",
        annualization=config.backtest.annualization,
    )
    plot_rolling_volatility(
        result.daily_returns,
        config.reports.figures_dir / f"{STRATEGY_NAME}_rolling_volatility.png",
        annualization=config.backtest.annualization,
    )
    plot_positions(
        result.weights,
        config.reports.figures_dir / f"{STRATEGY_NAME}_positions.png",
        ticker=ticker,
    )

    print(f"Metrics saved to {metrics_path}")
    print(f"Intraday results saved to {results_path}")
    print(f"Weights saved to {weights_path}")
    print(f"Daily regimes saved to {daily_regimes_path}")
    print(f"Intraday regime alignment saved to {alignment_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
