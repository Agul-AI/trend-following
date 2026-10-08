#!/usr/bin/env python
"""Create synthetic +3x ETFs from underlyings and run the best fast/slow strategy."""

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
from trend_following.plots import plot_drawdowns, plot_equity_curves
from trend_following.regime import daily_regime_hourly_fast_slow_signal
from trend_following.signals import (
    limit_trades_per_day,
    make_executable_positions,
    signals_to_equal_weight_positions,
)
from trend_following.synthetic_leverage import (
    synthetic_daily_leveraged_ohlcv,
    synthetic_intraday_leveraged_ohlcv,
)
from trend_following.utils import ensure_directory, resolve_path

DEFAULT_UNDERLYINGS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "TLT", "EEM", "GLD"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/regime_hourly_qqq.yaml",
        help="Config used for backtest costs, date range, and report paths.",
    )
    parser.add_argument(
        "--underlyings",
        nargs="+",
        default=DEFAULT_UNDERLYINGS,
        help="Underlying ETF tickers to turn into synthetic +3x targets.",
    )
    parser.add_argument(
        "--daily-raw-dir",
        default="data/raw/alpha_vantage_daily_adjusted",
        help="Daily adjusted parquet directory for underlying ETFs.",
    )
    parser.add_argument(
        "--intraday-raw-dir",
        default="data/raw/alpha_vantage_60min",
        help="Intraday parquet directory for underlying ETFs.",
    )
    parser.add_argument(
        "--synthetic-daily-dir",
        default="data/raw/synthetic_3x_1d",
        help="Output directory for synthetic daily +3x parquet files.",
    )
    parser.add_argument(
        "--synthetic-intraday-dir",
        default="data/raw/synthetic_3x_60min",
        help="Output directory for synthetic intraday +3x parquet files.",
    )
    parser.add_argument("--leverage", type=float, default=3.0)
    parser.add_argument("--initial-price", type=float, default=100.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate synthetic parquet files even if they already exist.",
    )
    parser.add_argument(
        "--output-prefix",
        default="synthetic_3x_fast_slow_200_gate",
        help="Prefix for report tables/figures.",
    )
    return parser.parse_args()


def _write_price_frame(frame: pd.DataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    out = frame.reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out.to_parquet(path, index=False)


def _returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if not returns.empty:
        returns.iloc[0] = 0.0
    return returns


def _filter_range(
    frame: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    filtered = frame.copy()
    if start_date:
        filtered = filtered.loc[filtered.index >= pd.Timestamp(start_date)]
    if end_date:
        filtered = filtered.loc[filtered.index < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    if filtered.empty:
        raise ValueError("No rows remain after date filtering")
    return filtered


def _synthetic_ticker(underlying: str, leverage: float) -> str:
    leverage_label = int(leverage) if float(leverage).is_integer() else leverage
    return f"{underlying}_{leverage_label}X_CALC"


def _max_trades_per_day(weights: pd.DataFrame) -> tuple[int, int]:
    changes = weights.diff().abs().sum(axis=1).gt(1e-12)
    if changes.empty:
        return 0, 0
    trades_by_day = changes.groupby(changes.index.normalize()).sum()
    return int(trades_by_day.max()), int(trades_by_day.gt(1).sum())


def _create_or_load_synthetic(
    *,
    underlying: str,
    synthetic_ticker: str,
    daily_raw_dir: Path,
    intraday_raw_dir: Path,
    synthetic_daily_dir: Path,
    synthetic_intraday_dir: Path,
    leverage: float,
    initial_price: float,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    daily_input = daily_raw_dir / f"{underlying}.parquet"
    intraday_input = intraday_raw_dir / f"{underlying}.parquet"
    daily_output = synthetic_daily_dir / f"{synthetic_ticker}.parquet"
    intraday_output = synthetic_intraday_dir / f"{synthetic_ticker}.parquet"
    if not daily_input.exists():
        raise FileNotFoundError(f"Missing daily underlying data: {daily_input}")
    if not intraday_input.exists():
        raise FileNotFoundError(f"Missing intraday underlying data: {intraday_input}")

    daily_underlying = read_price_file(daily_input).sort_index()
    intraday_underlying = read_price_file(intraday_input).sort_index()

    if force or not daily_output.exists():
        daily_synthetic = synthetic_daily_leveraged_ohlcv(
            daily_underlying,
            leverage=leverage,
            initial_price=initial_price,
        )
        _write_price_frame(daily_synthetic, daily_output)
    else:
        daily_synthetic = read_price_file(daily_output).sort_index()

    if force or not intraday_output.exists():
        intraday_synthetic = synthetic_intraday_leveraged_ohlcv(
            intraday_underlying=intraday_underlying,
            daily_underlying=daily_underlying,
            daily_synthetic=daily_synthetic,
            leverage=leverage,
        )
        _write_price_frame(intraday_synthetic, intraday_output)
    else:
        intraday_synthetic = read_price_file(intraday_output).sort_index()

    creation_row = {
        "underlying": underlying,
        "synthetic_ticker": synthetic_ticker,
        "daily_rows": len(daily_synthetic),
        "daily_start": daily_synthetic.index.min(),
        "daily_end": daily_synthetic.index.max(),
        "intraday_rows": len(intraday_synthetic),
        "intraday_start": intraday_synthetic.index.min(),
        "intraday_end": intraday_synthetic.index.max(),
        "daily_path": str(daily_output),
        "intraday_path": str(intraday_output),
    }
    return daily_underlying, intraday_underlying, intraday_synthetic, creation_row


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_params = dict(config.strategies.regime_switch)

    daily_raw_dir = resolve_path(config.root, args.daily_raw_dir)
    intraday_raw_dir = resolve_path(config.root, args.intraday_raw_dir)
    synthetic_daily_dir = resolve_path(config.root, args.synthetic_daily_dir)
    synthetic_intraday_dir = resolve_path(config.root, args.synthetic_intraday_dir)

    ensure_directory(config.reports.tables_dir)
    ensure_directory(config.reports.figures_dir)

    metric_rows: list[dict[str, object]] = []
    creation_rows: list[dict[str, object]] = []
    strategy_equity: dict[str, pd.Series] = {}
    benchmark_equity: dict[str, pd.Series] = {}
    strategy_returns: dict[str, pd.Series] = {}
    benchmark_returns_map: dict[str, pd.Series] = {}

    for underlying in args.underlyings:
        synthetic_ticker = _synthetic_ticker(underlying, args.leverage)
        daily_underlying, intraday_underlying, intraday_synthetic, creation_row = (
            _create_or_load_synthetic(
                underlying=underlying,
                synthetic_ticker=synthetic_ticker,
                daily_raw_dir=daily_raw_dir,
                intraday_raw_dir=intraday_raw_dir,
                synthetic_daily_dir=synthetic_daily_dir,
                synthetic_intraday_dir=synthetic_intraday_dir,
                leverage=args.leverage,
                initial_price=args.initial_price,
                force=args.force,
            )
        )
        creation_rows.append(creation_row)

        target_prices = _filter_range(
            intraday_synthetic[["adj_close"]].rename(columns={"adj_close": synthetic_ticker}),
            config.data.start_date,
            config.data.end_date,
        )
        benchmark_prices = _filter_range(
            intraday_underlying[["adj_close"]].rename(columns={"adj_close": underlying}),
            config.data.start_date,
            config.data.end_date,
        )
        common_index = target_prices.index.intersection(benchmark_prices.index)
        target_prices = target_prices.loc[common_index]
        benchmark_prices = benchmark_prices.loc[common_index]
        target_returns = _returns_from_prices(target_prices)
        underlying_returns = _returns_from_prices(benchmark_prices)

        daily_prices = daily_underlying[["adj_close"]].rename(columns={"adj_close": underlying})
        daily_returns = _returns_from_prices(daily_prices)

        params = dict(base_params)
        params.update(
            {
                "target_ticker": synthetic_ticker,
                "regime_ticker": underlying,
                "daily_regime_raw_dir": str(daily_raw_dir),
                "sma_window": 200,
                "use_variance_ratio_for_trend": False,
                "state_machine_entry_ma_days": 20.0,
                "state_machine_exit_ma_days": 200.0,
                "state_machine_entry_slope_days": 5.0,
                "state_machine_entry_confirm_bars": 2,
                "state_machine_exit_confirm_bars": 3,
                "state_machine_entry_buffer": 0.0,
                "state_machine_exit_buffer": 0.0,
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
        benchmark_weights = pd.DataFrame({underlying: 1.0}, index=underlying_returns.index)
        benchmark = backtest(
            returns=underlying_returns,
            weights=benchmark_weights,
            initial_capital=config.backtest.initial_capital,
        )

        max_trades, multi_trade_days = _max_trades_per_day(result.weights)
        strategy_metrics = calculate_metrics(
            result.daily_returns,
            turnover=result.turnover,
            weights=result.weights,
            annualization=config.backtest.annualization,
        )
        strategy_metrics.update(
            {
                "name": f"{synthetic_ticker}: 200-day {underlying} gate + hourly fast/slow",
                "strategy": "synthetic_3x_fast_slow_200_gate",
                "segment": "full_sample",
                "underlying": underlying,
                "target": synthetic_ticker,
                "benchmark": underlying,
                "parameters": json.dumps(params, sort_keys=True),
                "max_trades_per_day_observed": max_trades,
                "days_with_more_than_one_trade": multi_trade_days,
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
                "name": f"Buy & Hold {underlying}",
                "strategy": "benchmark",
                "segment": "full_sample",
                "underlying": underlying,
                "target": underlying,
                "benchmark": underlying,
                "parameters": "{}",
                "max_trades_per_day_observed": 1,
                "days_with_more_than_one_trade": 0,
            }
        )
        metric_rows.append(benchmark_metrics)

        strategy_equity[synthetic_ticker] = result.equity_curve
        benchmark_equity[f"BH {underlying}"] = benchmark.equity_curve
        strategy_returns[synthetic_ticker] = result.daily_returns
        benchmark_returns_map[f"BH {underlying}"] = benchmark.daily_returns

    creation = pd.DataFrame(creation_rows)
    metrics = metrics_to_frame(metric_rows)
    creation_path = config.reports.tables_dir / f"{args.output_prefix}_creation_summary.csv"
    metrics_path = config.reports.tables_dir / f"{args.output_prefix}_metrics.csv"
    strategy_metrics_path = (
        config.reports.tables_dir / f"{args.output_prefix}_strategy_only_metrics.csv"
    )
    equity_path = config.reports.figures_dir / f"{args.output_prefix}_strategy_equity_curve.png"
    drawdown_path = config.reports.figures_dir / f"{args.output_prefix}_strategy_drawdown.png"

    creation.to_csv(creation_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    metrics[metrics["strategy"].ne("benchmark")].to_csv(strategy_metrics_path, index=False)
    plot_equity_curves(
        strategy_equity,
        equity_path,
        title="Synthetic +3x fast/slow strategies, daily 200 gate",
    )
    plot_drawdowns(
        strategy_returns,
        drawdown_path,
        title="Synthetic +3x fast/slow strategy drawdowns",
    )

    print(f"Creation summary saved to {creation_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Strategy-only metrics saved to {strategy_metrics_path}")
    print(f"Strategy equity plot saved to {equity_path}")
    print(f"Strategy drawdown plot saved to {drawdown_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
