#!/usr/bin/env python
"""Test no-lookahead VIX exit filters on synthetic TQQQ fast/slow strategy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.backtest import backtest
from trend_following.config import load_config
from trend_following.data_validation import read_price_file
from trend_following.metrics import calculate_metrics, metrics_to_frame
from trend_following.plots import plot_drawdowns, plot_equity_curves
from trend_following.regime import (
    align_daily_regimes_to_intraday,
    classify_regimes,
    compute_regime_features,
    hourly_fast_entry_slow_exit_state_machine,
)
from trend_following.signals import (
    limit_trades_per_day,
    make_executable_positions,
    signals_to_equal_weight_positions,
)
from trend_following.utils import ensure_directory, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default="QQQ_3X_CALC")
    parser.add_argument("--regime-ticker", default="QQQ")
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--regime-daily-dir", default="data/raw/alpha_vantage_daily_adjusted")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--vix-path", default="data/raw/market_indicators/VIX.parquet")
    parser.add_argument("--vix-level", type=float, default=25.0)
    parser.add_argument("--vix-percentile-window", type=int, default=252)
    parser.add_argument("--vix-percentile-threshold", type=float, default=0.90)
    parser.add_argument("--vix-jump-lookback", type=int, default=5)
    parser.add_argument("--vix-jump-threshold", type=float, default=0.20)
    parser.add_argument(
        "--output-prefix",
        default="tqqq_vix_exit_experiments",
        help="Prefix for report tables and figures.",
    )
    return parser.parse_args()


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


def _rolling_percentile_of_last(values: np.ndarray) -> float:
    last = values[-1]
    finite = values[np.isfinite(values)]
    if not np.isfinite(last) or finite.size == 0:
        return np.nan
    return float((finite <= last).mean())


def _align_daily_bool_to_intraday(
    daily_flag: pd.Series,
    intraday_index: pd.DatetimeIndex,
    fill_method: str | None = "ffill",
) -> pd.Series:
    daily = daily_flag.copy()
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    intraday_dates = intraday_index.tz_localize(None).normalize()
    unique_dates = pd.DatetimeIndex(intraday_dates.unique()).sort_values()
    aligned_by_date = daily.reindex(unique_dates, method=fill_method)
    aligned_values = aligned_by_date.reindex(intraday_dates).fillna(False).to_numpy(dtype=bool)
    return pd.Series(aligned_values, index=intraday_index, name=daily_flag.name, dtype=bool)


def build_yesterday_vix_flags(
    vix_close: pd.Series,
    *,
    level: float,
    percentile_window: int,
    percentile_threshold: float,
    jump_lookback: int,
    jump_threshold: float,
) -> pd.DataFrame:
    """Compute VIX risk flags for date D using only VIX closes through D-1."""
    if percentile_window <= 0 or jump_lookback <= 0:
        raise ValueError("VIX windows must be positive")

    vix = vix_close.astype(float).sort_index()
    yesterday_vix = vix.shift(1)
    yesterday_sma20 = vix.rolling(20, min_periods=20).mean().shift(1)
    yesterday_percentile = (
        vix.rolling(percentile_window, min_periods=percentile_window)
        .apply(_rolling_percentile_of_last, raw=True)
        .shift(1)
    )
    yesterday_jump = vix.pct_change(jump_lookback, fill_method=None).shift(1)
    yesterday_rising = vix.diff().gt(0).shift(1).astype("boolean").fillna(False)

    flags = pd.DataFrame(
        {
            "vix_yesterday": yesterday_vix,
            "vix_sma20_yesterday": yesterday_sma20,
            "vix_percentile_yesterday": yesterday_percentile,
            "vix_5d_return_yesterday": yesterday_jump,
            "vix_rising_yesterday": yesterday_rising,
        },
        index=vix.index,
    )
    flags["level_gt_25"] = flags["vix_yesterday"].gt(level)
    flags["percentile_gt_90"] = flags["vix_percentile_yesterday"].gt(percentile_threshold)
    flags["jump_gt_20pct_5d"] = flags["vix_5d_return_yesterday"].gt(jump_threshold)
    flags["above_sma20_and_rising"] = (
        flags["vix_yesterday"].gt(flags["vix_sma20_yesterday"])
        & flags["vix_rising_yesterday"].astype(bool)
    )
    return flags


def _max_trades_per_day(weights: pd.DataFrame) -> tuple[int, int]:
    changes = weights.diff().abs().sum(axis=1).gt(1e-12)
    if changes.empty:
        return 0, 0
    trades_by_day = changes.groupby(changes.index.normalize()).sum()
    return int(trades_by_day.max()), int(trades_by_day.gt(1).sum())


def _run_one(
    *,
    label: str,
    vix_risk_intraday: pd.Series | None,
    daily_trend_intraday: pd.Series,
    target_prices: pd.DataFrame,
    target_returns: pd.DataFrame,
    params: dict[str, Any],
    config,
) -> tuple[dict[str, Any], pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    target_ticker = str(params["target_ticker"])
    allowed = daily_trend_intraday.astype(bool)
    risk = pd.Series(False, index=target_prices.index) if vix_risk_intraday is None else vix_risk_intraday
    allowed = allowed & ~risk.reindex(target_prices.index).fillna(False).astype(bool)
    raw_signal = hourly_fast_entry_slow_exit_state_machine(
        target_prices[target_ticker],
        allowed_regime=allowed,
        params=params,
    ).to_frame(target_ticker)
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
            "strategy": "tqqq_fast_slow_vix_exit",
            "segment": "full_sample",
            "parameters": json.dumps(params, sort_keys=True),
            "vix_rule": label,
            "vix_risk_bar_percentage": float(risk.reindex(target_prices.index).fillna(False).mean()),
            "max_trades_per_day_observed": max_trades,
            "days_with_more_than_one_trade": multi_trade_days,
        }
    )
    return metrics, result.equity_curve, result.daily_returns, result.weights, risk


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    target_dir = resolve_path(config.root, args.target_raw_dir)
    regime_daily_dir = resolve_path(config.root, args.regime_daily_dir)
    benchmark_dir = resolve_path(config.root, args.benchmark_raw_dir)
    vix_path = resolve_path(config.root, args.vix_path)

    target_ticker = args.target_ticker
    regime_ticker = args.regime_ticker
    target_frame = read_price_file(target_dir / f"{target_ticker}.parquet").sort_index()
    target_prices = target_frame[["adj_close"]].rename(columns={"adj_close": target_ticker})
    benchmark_frame = read_price_file(benchmark_dir / f"{regime_ticker}.parquet").sort_index()
    benchmark_prices = benchmark_frame[["adj_close"]].rename(columns={"adj_close": regime_ticker})
    daily_frame = read_price_file(regime_daily_dir / f"{regime_ticker}.parquet").sort_index()
    daily_prices = daily_frame[["adj_close"]].rename(columns={"adj_close": regime_ticker})
    daily_returns = _returns_from_prices(daily_prices)
    vix_frame = read_price_file(vix_path).sort_index()
    vix_close = vix_frame["adj_close"].astype(float).rename("VIX")

    target_prices = _filter_range(target_prices, config.data.start_date, config.data.end_date)
    benchmark_prices = _filter_range(benchmark_prices, config.data.start_date, config.data.end_date)
    common_index = target_prices.index.intersection(benchmark_prices.index)
    target_prices = target_prices.loc[common_index]
    benchmark_prices = benchmark_prices.loc[common_index]
    target_returns = _returns_from_prices(target_prices)
    benchmark_returns = _returns_from_prices(benchmark_prices)

    params = dict(config.strategies.regime_switch)
    params.update(
        {
            "target_ticker": target_ticker,
            "regime_ticker": regime_ticker,
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

    daily_features = compute_regime_features(
        daily_prices,
        daily_returns,
        regime_ticker=regime_ticker,
        params=params,
    )
    daily_regimes = classify_regimes(daily_features, params=params)
    daily_regime_intraday = align_daily_regimes_to_intraday(
        daily_regimes,
        target_prices.index,
        lag_days=int(params.get("daily_regime_lag_days", 1)),
        fill_method=params.get("daily_regime_fill_method", "ffill"),
    ).fillna("neutral")
    daily_trend_intraday = daily_regime_intraday.eq("trend")

    vix_flags = build_yesterday_vix_flags(
        vix_close,
        level=args.vix_level,
        percentile_window=args.vix_percentile_window,
        percentile_threshold=args.vix_percentile_threshold,
        jump_lookback=args.vix_jump_lookback,
        jump_threshold=args.vix_jump_threshold,
    )
    vix_rules = {
        "base_no_vix_exit": None,
        f"yesterday_vix_gt_{args.vix_level:g}": vix_flags["level_gt_25"],
        f"yesterday_vix_pctile_gt_{args.vix_percentile_threshold:.0%}": vix_flags[
            "percentile_gt_90"
        ],
        f"yesterday_vix_{args.vix_jump_lookback}d_jump_gt_{args.vix_jump_threshold:.0%}": vix_flags[
            "jump_gt_20pct_5d"
        ],
        "yesterday_vix_above_sma20_and_rising": vix_flags["above_sma20_and_rising"],
    }

    metric_rows: list[dict[str, Any]] = []
    equity_curves: dict[str, pd.Series] = {}
    return_streams: dict[str, pd.Series] = {}
    weights_table = pd.DataFrame(index=target_prices.index)
    risk_table = pd.DataFrame(index=target_prices.index)

    for label, daily_flag in vix_rules.items():
        risk_intraday = (
            None
            if daily_flag is None
            else _align_daily_bool_to_intraday(daily_flag, target_prices.index)
        )
        metrics, equity, returns, weights, risk = _run_one(
            label=label,
            vix_risk_intraday=risk_intraday,
            daily_trend_intraday=daily_trend_intraday,
            target_prices=target_prices,
            target_returns=target_returns,
            params=params,
            config=config,
        )
        metric_rows.append(metrics)
        equity_curves[label] = equity
        return_streams[label] = returns
        weights_table[label] = weights[target_ticker]
        risk_table[label] = risk.reindex(target_prices.index).fillna(False).astype(int)

    benchmark_weights = pd.DataFrame({regime_ticker: 1.0}, index=benchmark_returns.index)
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
            "name": f"Buy & Hold {regime_ticker}",
            "strategy": "benchmark",
            "segment": "full_sample",
            "parameters": "{}",
            "vix_rule": "benchmark",
            "vix_risk_bar_percentage": np.nan,
            "max_trades_per_day_observed": 1,
            "days_with_more_than_one_trade": 0,
        }
    )
    metric_rows.append(benchmark_metrics)
    equity_curves[f"Buy & Hold {regime_ticker}"] = benchmark.equity_curve
    return_streams[f"Buy & Hold {regime_ticker}"] = benchmark.daily_returns

    ensure_directory(config.reports.tables_dir)
    ensure_directory(config.reports.figures_dir)
    metrics = metrics_to_frame(metric_rows)
    metrics_path = config.reports.tables_dir / f"{args.output_prefix}_metrics.csv"
    vix_flags_path = config.reports.tables_dir / f"{args.output_prefix}_daily_vix_flags.csv"
    weights_path = config.reports.tables_dir / f"{args.output_prefix}_weights.csv"
    risk_path = config.reports.tables_dir / f"{args.output_prefix}_intraday_vix_risk_flags.csv"
    equity_path = config.reports.figures_dir / f"{args.output_prefix}_equity_curve.png"
    drawdown_path = config.reports.figures_dir / f"{args.output_prefix}_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    vix_flags.to_csv(vix_flags_path)
    weights_table.to_csv(weights_path)
    risk_table.to_csv(risk_path)
    plot_equity_curves(equity_curves, equity_path, title="Synthetic TQQQ VIX exit filters")
    plot_drawdowns(return_streams, drawdown_path, title="Synthetic TQQQ VIX exit drawdowns")

    print(f"Metrics saved to {metrics_path}")
    print(f"Daily VIX flags saved to {vix_flags_path}")
    print(f"Weights saved to {weights_path}")
    print(f"Intraday VIX risk flags saved to {risk_path}")
    print(f"Equity plot saved to {equity_path}")
    print(f"Drawdown plot saved to {drawdown_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
