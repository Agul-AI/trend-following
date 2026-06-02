#!/usr/bin/env python
"""Compare 100%-entry MA and MACD variants for synthetic TQQQ."""

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

from run_tqqq_macd_entry_experiments import count_profit_lock_hits  # noqa: E402
from run_tqqq_position_risk_sizing_experiments import (  # noqa: E402
    drawdown_episode_count,
    simulate_after_tax_portfolio,
)
from run_tqqq_tiered_sizing_experiments import trade_profit_lock_tiers  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.metrics import calculate_metrics, metrics_to_frame  # noqa: E402
from trend_following.regime import (  # noqa: E402
    align_daily_regimes_to_intraday,
    classify_regimes,
    compute_regime_features,
)
from trend_following.signals import limit_trades_per_day, make_executable_positions  # noqa: E402
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default="QQQ_3X_CALC")
    parser.add_argument("--benchmark-ticker", default="QQQ")
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--daily-regime-raw-dir", default="data/raw/alpha_vantage_daily_adjusted")
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--output-prefix", default="tqqq_entry_signal_comparison")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def _returns_from_prices(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if not returns.empty:
        if isinstance(returns, pd.DataFrame):
            returns.iloc[0] = 0.0
        else:
            returns.iloc[0] = 0.0
    return returns


def _equity(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _drawdown(returns: pd.Series) -> pd.Series:
    equity = _equity(returns)
    return equity / equity.cummax() - 1.0


def _days_to_bars(days: float, bars_per_day: int) -> int:
    return max(int(round(float(days) * bars_per_day)), 1)


def _confirmed(flag: pd.Series, bars: int) -> pd.Series:
    if bars <= 1:
        return flag.fillna(False).astype(bool)
    return (
        flag.fillna(False)
        .astype(float)
        .rolling(window=bars, min_periods=bars)
        .sum()
        .eq(float(bars))
        .fillna(False)
        .astype(bool)
    )


def macd_components(
    price: pd.Series,
    *,
    fast_window: int,
    slow_window: int,
    signal_window: int,
    average_type: str,
) -> pd.DataFrame:
    """Return MACD line/signal/histogram using EMA or SMA averaging."""
    if not 0 < fast_window < slow_window:
        raise ValueError("fast_window must be positive and less than slow_window")
    if signal_window <= 0:
        raise ValueError("signal_window must be positive")

    clean = price.astype(float).sort_index()
    if average_type == "ema":
        fast = clean.ewm(span=fast_window, adjust=False, min_periods=fast_window).mean()
        slow = clean.ewm(span=slow_window, adjust=False, min_periods=slow_window).mean()
        macd = fast - slow
        signal = macd.ewm(span=signal_window, adjust=False, min_periods=signal_window).mean()
    elif average_type == "sma":
        fast = clean.rolling(window=fast_window, min_periods=fast_window).mean()
        slow = clean.rolling(window=slow_window, min_periods=slow_window).mean()
        macd = fast - slow
        signal = macd.rolling(window=signal_window, min_periods=signal_window).mean()
    else:
        raise ValueError("average_type must be 'ema' or 'sma'")

    return pd.DataFrame(
        {
            "price": clean,
            "macd": macd,
            "macd_signal": signal,
            "macd_hist": macd - signal,
        },
        index=clean.index,
    )


def macd_entry_slow_exit_signal(
    price: pd.Series,
    *,
    allowed_regime: pd.Series,
    bars_per_day: int,
    average_type: str,
    macd_unit: str,
    entry_mode: str,
    entry_confirm_bars: int = 2,
    exit_ma_days: float = 200.0,
    exit_confirm_bars: int = 3,
) -> tuple[pd.Series, pd.DataFrame]:
    """Raw 100%-entry signal from MACD entry and slow 200-day-MA exit."""
    clean = price.astype(float).sort_index()
    gate = allowed_regime.reindex(clean.index).fillna(False).astype(bool)
    if macd_unit == "days":
        fast_window = _days_to_bars(12, bars_per_day)
        slow_window = _days_to_bars(26, bars_per_day)
        signal_window = _days_to_bars(9, bars_per_day)
    elif macd_unit == "bars":
        fast_window = 12
        slow_window = 26
        signal_window = 9
    else:
        raise ValueError("macd_unit must be 'bars' or 'days'")

    components = macd_components(
        clean,
        fast_window=fast_window,
        slow_window=slow_window,
        signal_window=signal_window,
        average_type=average_type,
    )
    hist = components["macd_hist"]
    macd_line = components["macd"]
    hist_slope = hist.diff()

    if entry_mode == "hist_gt_0":
        raw_entry = hist.gt(0.0)
    elif entry_mode == "hist_gt_0_and_rising":
        raw_entry = hist.gt(0.0) & hist_slope.gt(0.0)
    elif entry_mode == "hist_gt_0_macd_gt_0":
        raw_entry = hist.gt(0.0) & macd_line.gt(0.0)
    else:
        raise ValueError(f"unsupported entry_mode: {entry_mode}")
    entry = _confirmed(gate & raw_entry, entry_confirm_bars)

    exit_window = _days_to_bars(exit_ma_days, bars_per_day)
    exit_ma = clean.rolling(window=exit_window, min_periods=exit_window).mean()
    price_exit = _confirmed(clean.lt(exit_ma), exit_confirm_bars)
    regime_exit = ~gate

    state = 0.0
    values: list[float] = []
    for entry_now, regime_exit_now, price_exit_now in zip(
        entry, regime_exit, price_exit, strict=False
    ):
        if state == 0.0:
            if bool(entry_now):
                state = 1.0
        elif bool(regime_exit_now) or bool(price_exit_now):
            state = 0.0
        values.append(state)

    signal = pd.Series(values, index=clean.index, name=price.name, dtype=float)
    components["entry_flag"] = entry.astype(float)
    components["exit_ma"] = exit_ma
    components["price_exit"] = price_exit.astype(float)
    components["allowed_regime"] = gate.astype(float)
    return signal, components


def executable_weights(raw_weights: pd.DataFrame, *, config) -> pd.DataFrame:
    shifted = make_executable_positions(
        raw_weights,
        execution_delay_days=config.backtest.execution_delay_days,
        return_convention=config.backtest.return_convention,
    )
    limited = limit_trades_per_day(
        shifted,
        max_trades_per_day=config.backtest.max_trades_per_day,
    )
    return limited.fillna(0.0)


def _plot_top(
    returns_by_name: dict[str, pd.Series],
    metrics: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 8,
) -> None:
    ranked = metrics[metrics["strategy"].ne("benchmark")].copy()
    selected = (
        ranked.sort_values(["sharpe_ratio", "max_drawdown"], ascending=[False, False])
        .head(top_n)["name"]
        .tolist()
    )
    selected = list(dict.fromkeys(selected))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for name in selected:
        returns = returns_by_name[name]
        _equity(returns).plot(ax=axes[0], label=name, linewidth=1.2)
        _drawdown(returns).plot(ax=axes[1], label=name, linewidth=1.2)
    axes[0].set_title("After-tax equity")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("After-tax drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _profit_lock_variant(raw: pd.Series, price: pd.Series, ticker: str) -> pd.DataFrame:
    return trade_profit_lock_tiers(
        raw,
        price,
        thresholds_to_weights=[(1.50, 0.75), (2.50, 0.50)],
    ).to_frame(ticker)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    target_dir = resolve_path(config.root, args.target_raw_dir)
    benchmark_dir = resolve_path(config.root, args.benchmark_raw_dir)
    daily_dir = resolve_path(config.root, args.daily_regime_raw_dir)

    target = _load_price(target_dir / f"{args.target_ticker}.parquet", args.target_ticker)
    qqq = _load_price(benchmark_dir / f"{args.benchmark_ticker}.parquet", args.benchmark_ticker)
    daily_qqq = _load_price(daily_dir / f"{args.benchmark_ticker}.parquet", args.benchmark_ticker)

    common = target.index.intersection(qqq.index)
    target_prices = target.loc[common].to_frame()
    qqq_prices = qqq.loc[common].to_frame()
    target_returns = _returns_from_prices(target_prices)
    qqq_returns = _returns_from_prices(qqq_prices)
    daily_prices = daily_qqq.to_frame()
    daily_returns = _returns_from_prices(daily_prices)

    base_params = dict(config.strategies.regime_switch)
    base_params.update(
        {
            "target_ticker": args.target_ticker,
            "regime_ticker": args.benchmark_ticker,
            "sma_window": 200,
            "use_variance_ratio_for_trend": False,
            "state_machine_exit_ma_days": 200.0,
            "state_machine_entry_slope_days": 5.0,
            "state_machine_entry_confirm_bars": 2,
            "state_machine_exit_confirm_bars": 3,
            "state_machine_entry_buffer": 0.0,
            "state_machine_exit_buffer": 0.0,
        }
    )
    bars_per_day = int(base_params.get("intraday_bars_per_day", 6))

    raw_signals: dict[str, pd.Series] = {}
    diagnostics: dict[str, pd.DataFrame] = {}

    daily_features = compute_regime_features(
        daily_prices,
        daily_returns,
        regime_ticker=args.benchmark_ticker,
        params=base_params,
    )
    daily_regimes = classify_regimes(daily_features, params=base_params)
    intraday_regimes = align_daily_regimes_to_intraday(
        daily_regimes,
        target_prices.index,
        lag_days=int(base_params.get("daily_regime_lag_days", 1)),
        fill_method=base_params.get("daily_regime_fill_method", "ffill"),
    ).fillna("neutral")
    allowed_regime = intraday_regimes.eq("trend")

    macd_specs = [
        ("macd_ema_bars_hist_rising", "ema", "bars", "hist_gt_0_and_rising"),
        ("macd_sma_bars_hist_rising", "sma", "bars", "hist_gt_0_and_rising"),
        ("macd_ema_days_hist_gt_0", "ema", "days", "hist_gt_0"),
        ("macd_sma_days_hist_gt_0", "sma", "days", "hist_gt_0"),
        ("macd_ema_days_hist_macd_pos", "ema", "days", "hist_gt_0_macd_gt_0"),
        ("macd_sma_days_hist_macd_pos", "sma", "days", "hist_gt_0_macd_gt_0"),
    ]
    for label, avg_type, unit, entry_mode in macd_specs:
        raw, diag = macd_entry_slow_exit_signal(
            target_prices[args.target_ticker],
            allowed_regime=allowed_regime,
            bars_per_day=bars_per_day,
            average_type=avg_type,
            macd_unit=unit,
            entry_mode=entry_mode,
        )
        raw_signals[label] = raw
        diagnostics[label] = diag

    raw_variants: dict[str, pd.DataFrame] = {}
    profit_lock_hit_counts: dict[str, dict[str, float]] = {}
    for name, raw in raw_signals.items():
        raw_variants[f"{name}_full"] = raw.to_frame(args.target_ticker)
        profit_lock_hit_counts[f"{name}_full"] = {
            "profit_lock_150pct_hit_count": np.nan,
            "profit_lock_250pct_hit_count": np.nan,
        }
        lock_name = f"{name}_profit_lock_150_250"
        raw_variants[lock_name] = _profit_lock_variant(raw, target, args.target_ticker)
        profit_lock_hit_counts[lock_name] = {
            "profit_lock_150pct_hit_count": float(
                count_profit_lock_hits(raw, target, threshold=1.50)
            ),
            "profit_lock_250pct_hit_count": float(
                count_profit_lock_hits(raw, target, threshold=2.50)
            ),
        }

    all_returns = pd.concat([target_returns, qqq_returns], axis=1).loc[common]
    metric_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    pretax_returns_by_name: dict[str, pd.Series] = {}
    weights_out: dict[str, pd.Series] = {}

    for name, raw_weights in raw_variants.items():
        weights = executable_weights(raw_weights, config=config).reindex(common).fillna(0.0)
        returns_subset = all_returns[[column for column in weights.columns if column in all_returns]]
        after_tax, pretax, taxes_paid, turnover = simulate_after_tax_portfolio(
            returns_subset,
            weights,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            tax_rate=args.short_term_tax_rate,
        )
        metrics = calculate_metrics(
            after_tax,
            turnover=turnover,
            weights=weights.sum(axis=1),
            annualization=config.backtest.annualization,
        )
        metrics.update(
            {
                "name": name,
                "strategy": "entry_signal_comparison",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "transaction_cost_bps": args.transaction_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "short_term_tax_rate": args.short_term_tax_rate,
                        "base_params": base_params,
                    },
                    sort_keys=True,
                ),
                "pretax_cumulative_return": float((1.0 + pretax).prod() - 1.0),
                "tax_paid_pct_initial_capital": float(taxes_paid.sum()),
                "drawdown_episodes_gt_30pct": drawdown_episode_count(after_tax, threshold=-0.30),
                "drawdown_episodes_gt_40pct": drawdown_episode_count(after_tax, threshold=-0.40),
                "drawdown_episodes_gt_50pct": drawdown_episode_count(after_tax, threshold=-0.50),
                **profit_lock_hit_counts[name],
            }
        )
        metric_rows.append(metrics)
        returns_by_name[name] = after_tax
        pretax_returns_by_name[name] = pretax
        weights_out[name] = weights.sum(axis=1)

    benchmark_metrics = calculate_metrics(
        qqq_returns[args.benchmark_ticker],
        annualization=config.backtest.annualization,
    )
    benchmark_metrics.update(
        {
            "name": "buy_hold_qqq",
            "strategy": "benchmark",
            "segment": "full_sample",
            "parameters": "{}",
            "pretax_cumulative_return": float(
                (1.0 + qqq_returns[args.benchmark_ticker]).prod() - 1.0
            ),
            "tax_paid_pct_initial_capital": 0.0,
            "drawdown_episodes_gt_30pct": drawdown_episode_count(
                qqq_returns[args.benchmark_ticker],
                threshold=-0.30,
            ),
            "drawdown_episodes_gt_40pct": drawdown_episode_count(
                qqq_returns[args.benchmark_ticker],
                threshold=-0.40,
            ),
            "drawdown_episodes_gt_50pct": drawdown_episode_count(
                qqq_returns[args.benchmark_ticker],
                threshold=-0.50,
            ),
            "profit_lock_150pct_hit_count": np.nan,
            "profit_lock_250pct_hit_count": np.nan,
        }
    )
    metric_rows.append(benchmark_metrics)
    returns_by_name["buy_hold_qqq"] = qqq_returns[args.benchmark_ticker]

    metrics = metrics_to_frame(metric_rows)
    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    compact_path = tables_dir / f"{args.output_prefix}_macd_only_compact.csv"
    returns_path = tables_dir / f"{args.output_prefix}_after_tax_returns.csv"
    pretax_returns_path = tables_dir / f"{args.output_prefix}_pretax_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    diagnostics_path = tables_dir / f"{args.output_prefix}_macd_diagnostics.parquet"
    plot_path = figures_dir / f"{args.output_prefix}_top_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    compact = metrics.copy()
    compact["dd_episodes_gt_30_40_50pct"] = compact.apply(
        lambda row: (
            f"{int(row['drawdown_episodes_gt_30pct'])}/"
            f"{int(row['drawdown_episodes_gt_40pct'])}/"
            f"{int(row['drawdown_episodes_gt_50pct'])}"
        ),
        axis=1,
    )
    compact = compact[
        [
            "name",
            "cumulative_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "number_of_trades",
            "exposure_percentage",
            "dd_episodes_gt_30_40_50pct",
            "profit_lock_150pct_hit_count",
            "profit_lock_250pct_hit_count",
        ]
    ].sort_values("sharpe_ratio", ascending=False)
    compact.to_csv(compact_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(pretax_returns_by_name).to_csv(pretax_returns_path)
    pd.DataFrame(weights_out).to_csv(weights_path)
    if diagnostics:
        pd.concat(diagnostics, axis=1).to_parquet(diagnostics_path)
    _plot_top(
        returns_by_name,
        metrics,
        plot_path,
        title="Synthetic TQQQ 100%-entry MACD comparison",
    )

    print(f"Metrics saved to {metrics_path}")
    print(f"Compact MACD-only table saved to {compact_path}")
    print(f"After-tax returns saved to {returns_path}")
    print(f"Pretax returns saved to {pretax_returns_path}")
    print(f"Weights saved to {weights_path}")
    print(f"MACD diagnostics saved to {diagnostics_path}")
    print(f"Plot saved to {plot_path}")
    full_columns = metrics[
        [
            "name",
            "cumulative_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "number_of_trades",
            "exposure_percentage",
            "drawdown_episodes_gt_30pct",
            "drawdown_episodes_gt_40pct",
            "drawdown_episodes_gt_50pct",
            "profit_lock_150pct_hit_count",
            "profit_lock_250pct_hit_count",
        ]
    ].sort_values("sharpe_ratio", ascending=False)
    print(compact.to_string(index=False))
    print(full_columns.to_string(index=False))


if __name__ == "__main__":
    main()
