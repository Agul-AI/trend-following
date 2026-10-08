"""Strategy construction and parameter experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from trend_following.backtest import BacktestResult, backtest, buy_and_hold_weights
from trend_following.config import ProjectConfig
from trend_following.metrics import calculate_metrics, metrics_to_frame
from trend_following.plots import plot_heatmap
from trend_following.regime import regime_switch_signal
from trend_following.signals import (
    apply_volatility_targeting,
    cross_sectional_momentum_signal,
    donchian_breakout_signal,
    kalman_trend_signal,
    limit_trades_per_day,
    make_executable_positions,
    regression_slope_signal,
    signals_to_equal_weight_positions,
    sma_crossover_signal,
    sma_trend_signal,
    time_series_momentum_signal,
)
from trend_following.utils import ensure_directory

STRATEGY_NAMES = [
    "sma_trend",
    "sma_crossover",
    "tsmom",
    "donchian_breakout",
    "regression_slope",
    "kalman_trend",
    "cross_sectional_momentum",
    "regime_switch",
]


def raw_signal_for_strategy(
    prices: pd.DataFrame,
    strategy: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Build unshifted close-date raw signals for a named strategy."""
    if strategy == "sma_trend":
        return sma_trend_signal(prices, window=int(params.get("window", 200)))
    if strategy == "sma_crossover":
        return sma_crossover_signal(
            prices,
            short_window=int(params.get("short_window", 50)),
            long_window=int(params.get("long_window", 200)),
        )
    if strategy == "tsmom":
        return time_series_momentum_signal(prices, lookback=int(params.get("lookback", 252)))
    if strategy == "donchian_breakout":
        return donchian_breakout_signal(
            prices,
            entry_lookback=int(params.get("entry_lookback", 252)),
            exit_lookback=int(params.get("exit_lookback", 126)),
        )
    if strategy == "regression_slope":
        return regression_slope_signal(
            prices,
            window=int(params.get("window", 126)),
            min_r_squared=float(params.get("min_r_squared", 0.0)),
        )
    if strategy == "kalman_trend":
        return kalman_trend_signal(
            prices,
            process_level_var=float(params.get("process_level_var", 1e-5)),
            process_trend_var=float(params.get("process_trend_var", 1e-7)),
            observation_var=float(params.get("observation_var", 1e-3)),
            min_periods=int(params.get("min_periods", 20)),
        )
    if strategy == "cross_sectional_momentum":
        return cross_sectional_momentum_signal(
            prices,
            lookback=int(params.get("lookback", 126)),
            top_n=int(params.get("top_n", 3)),
            require_positive=bool(params.get("require_positive", True)),
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def build_strategy_positions(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    strategy: str,
    params: dict[str, Any],
    config: ProjectConfig,
) -> pd.DataFrame:
    """Create executable portfolio weights for a strategy."""
    if strategy == "regime_switch":
        raw_signal = regime_switch_signal(prices, returns, params=params)
    else:
        raw_signal = raw_signal_for_strategy(prices, strategy=strategy, params=params)

    if config.strategies.volatility_targeting.enabled:
        vol = config.strategies.volatility_targeting
        raw_signal = apply_volatility_targeting(
            raw_signal=raw_signal,
            returns=returns,
            target_vol=vol.target_vol,
            lookback=vol.lookback,
            max_leverage=vol.max_leverage,
            annualization=config.backtest.annualization,
        )

    portfolio_mode = str(params.get("portfolio_mode", config.backtest.portfolio_mode))
    raw_weights = signals_to_equal_weight_positions(raw_signal, mode=portfolio_mode)
    positions = make_executable_positions(
        raw_weights,
        execution_delay_days=config.backtest.execution_delay_days,
        return_convention=config.backtest.return_convention,
    )
    positions = limit_trades_per_day(
        positions,
        max_trades_per_day=config.backtest.max_trades_per_day,
    )
    return positions.reindex_like(returns).fillna(0.0)


def run_strategy_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    strategy: str,
    params: dict[str, Any],
    config: ProjectConfig,
) -> BacktestResult:
    """Build positions and run the vectorized backtester."""
    positions = build_strategy_positions(prices, returns, strategy, params, config)
    return backtest(
        returns=returns,
        weights=positions,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
        slippage_bps=config.backtest.slippage_bps,
        initial_capital=config.backtest.initial_capital,
    )


def benchmark_backtests(returns: pd.DataFrame, config: ProjectConfig) -> dict[str, BacktestResult]:
    """Run default buy-and-hold benchmark backtests."""
    benchmarks: dict[str, BacktestResult] = {}
    if "SPY" in returns.columns:
        spy_weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        spy_weights["SPY"] = 1.0
        benchmarks["Buy & Hold SPY"] = backtest(
            returns, spy_weights, initial_capital=config.backtest.initial_capital
        )
    equal_weight = buy_and_hold_weights(returns)
    benchmarks["Equal-Weight Buy & Hold"] = backtest(
        returns[equal_weight.columns],
        equal_weight,
        initial_capital=config.backtest.initial_capital,
    )
    return benchmarks


def strategy_default_params(config: ProjectConfig, strategy: str) -> dict[str, Any]:
    """Return default strategy parameters from config."""
    if strategy == "sma_trend":
        return dict(config.strategies.sma_trend)
    if strategy == "sma_crossover":
        return dict(config.strategies.sma_crossover)
    if strategy == "tsmom":
        return dict(config.strategies.tsmom)
    if strategy == "donchian_breakout":
        return dict(config.strategies.donchian_breakout)
    if strategy == "regression_slope":
        return dict(config.strategies.regression_slope)
    if strategy == "kalman_trend":
        return dict(config.strategies.kalman_trend)
    if strategy == "cross_sectional_momentum":
        return dict(config.strategies.cross_sectional_momentum)
    if strategy == "regime_switch":
        return dict(config.strategies.regime_switch)
    raise ValueError(f"Unknown strategy: {strategy}")


def _segment_mask(index: pd.DatetimeIndex, segment: str, split_date: str) -> pd.Series:
    split = pd.Timestamp(split_date)
    if segment == "in_sample":
        return pd.Series(index <= split, index=index)
    if segment == "out_of_sample":
        return pd.Series(index > split, index=index)
    if segment == "full_sample":
        return pd.Series(True, index=index)
    raise ValueError(f"Unknown segment: {segment}")


def _metrics_for_segments(
    result: BacktestResult,
    strategy: str,
    params: dict[str, Any],
    config: ProjectConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in ["in_sample", "out_of_sample", "full_sample"]:
        mask = _segment_mask(result.daily_returns.index, segment, config.backtest.train_end_date)
        row = calculate_metrics(
            result.daily_returns.loc[mask],
            turnover=result.turnover.loc[mask],
            weights=result.weights.loc[mask],
            annualization=config.backtest.annualization,
        )
        row.update(
            {
                "strategy": strategy,
                "segment": segment,
                "parameters": json.dumps(params, sort_keys=True),
            }
        )
        rows.append(row)
    return rows


def run_parameter_sweep(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Evaluate configured strategy parameter grids in and out of sample."""
    rows: list[dict[str, Any]] = []

    for window in config.experiments.sma_windows:
        params = {"window": int(window)}
        result = run_strategy_backtest(prices, returns, "sma_trend", params, config)
        rows.extend(_metrics_for_segments(result, "sma_trend", params, config))

    for lookback in config.experiments.tsmom_lookbacks:
        params = {"lookback": int(lookback)}
        result = run_strategy_backtest(prices, returns, "tsmom", params, config)
        rows.extend(_metrics_for_segments(result, "tsmom", params, config))

    for short_window in config.experiments.crossover_short_windows:
        for long_window in config.experiments.crossover_long_windows:
            if short_window >= long_window:
                continue
            params = {"short_window": int(short_window), "long_window": int(long_window)}
            result = run_strategy_backtest(prices, returns, "sma_crossover", params, config)
            rows.extend(_metrics_for_segments(result, "sma_crossover", params, config))

    for entry_lookback in config.experiments.breakout_entry_lookbacks:
        for exit_lookback in config.experiments.breakout_exit_lookbacks:
            params = {
                "entry_lookback": int(entry_lookback),
                "exit_lookback": int(exit_lookback),
            }
            result = run_strategy_backtest(prices, returns, "donchian_breakout", params, config)
            rows.extend(_metrics_for_segments(result, "donchian_breakout", params, config))

    for window in config.experiments.regression_windows:
        params = {"window": int(window), "min_r_squared": 0.0}
        result = run_strategy_backtest(prices, returns, "regression_slope", params, config)
        rows.extend(_metrics_for_segments(result, "regression_slope", params, config))

    kalman_params = dict(config.strategies.kalman_trend)
    kalman_result = run_strategy_backtest(prices, returns, "kalman_trend", kalman_params, config)
    rows.extend(_metrics_for_segments(kalman_result, "kalman_trend", kalman_params, config))

    for lookback in config.experiments.cross_sectional_lookbacks:
        for top_n in config.experiments.cross_sectional_top_ns:
            params = {
                "lookback": int(lookback),
                "top_n": int(top_n),
                "require_positive": True,
                "portfolio_mode": "active_equal",
            }
            result = run_strategy_backtest(
                prices, returns, "cross_sectional_momentum", params, config
            )
            rows.extend(_metrics_for_segments(result, "cross_sectional_momentum", params, config))

    table = metrics_to_frame(rows)
    ensure_directory(config.reports.tables_dir)
    output_path = config.reports.tables_dir / "parameter_sweep.csv"
    table.to_csv(output_path, index=False)

    _save_parameter_plots(table, config.reports.figures_dir)
    return table


def _json_param_value(parameters: str, key: str) -> int | None:
    try:
        return int(json.loads(parameters)[key])
    except Exception:
        return None


def _save_parameter_plots(table: pd.DataFrame, figures_dir: Path) -> None:
    """Save simple heatmaps/tables for parameter sensitivity."""
    ensure_directory(figures_dir)
    crossover = table[
        (table["strategy"] == "sma_crossover") & (table["segment"] == "out_of_sample")
    ].copy()
    if not crossover.empty:
        crossover["short_window"] = crossover["parameters"].map(
            lambda p: _json_param_value(p, "short_window")
        )
        crossover["long_window"] = crossover["parameters"].map(
            lambda p: _json_param_value(p, "long_window")
        )
        matrix = crossover.pivot_table(
            index="short_window",
            columns="long_window",
            values="sharpe_ratio",
            aggfunc="mean",
        )
        if not matrix.empty:
            plot_heatmap(
                matrix,
                figures_dir / "parameter_sweep_crossover_oos_sharpe.png",
                title="MA Crossover OOS Sharpe",
            )
