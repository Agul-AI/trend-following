#!/usr/bin/env python
"""Compare 20MA entry with 12/26 MACD entry for synthetic TQQQ."""

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
    daily_regime_hourly_fast_slow_signal,
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
    parser.add_argument("--drawdown-threshold-pct", type=float, default=30.0)
    parser.add_argument("--output-prefix", default="tqqq_macd_entry_experiments")
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
    fast_span: int,
    slow_span: int,
    signal_span: int,
) -> pd.DataFrame:
    """Return standard EMA MACD line/signal/histogram from close-bar prices."""
    if not 0 < fast_span < slow_span:
        raise ValueError("MACD fast_span must be positive and less than slow_span")
    if signal_span <= 0:
        raise ValueError("MACD signal_span must be positive")
    clean = price.astype(float).sort_index()
    fast = clean.ewm(span=fast_span, adjust=False, min_periods=fast_span).mean()
    slow = clean.ewm(span=slow_span, adjust=False, min_periods=slow_span).mean()
    macd = fast - slow
    signal = macd.ewm(span=signal_span, adjust=False, min_periods=signal_span).mean()
    hist = macd - signal
    return pd.DataFrame(
        {
            "price": clean,
            "macd": macd,
            "macd_signal": signal,
            "macd_hist": hist,
        },
        index=clean.index,
    )


def gradual_entry_raw_weights(
    raw_signal: pd.Series,
    *,
    tiers: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0),
) -> pd.Series:
    """Scale into a raw long signal over successive bars.

    The first raw long bar is 25%, then 50%, 75%, and 100% if the base signal
    remains on. The result is still a raw close-bar signal and is shifted before
    it can earn returns, so this does not introduce lookahead.
    """
    if not tiers:
        raise ValueError("tiers must not be empty")
    if any(tier < 0.0 or tier > 1.0 for tier in tiers):
        raise ValueError("tiers must be between 0 and 1")

    bars_in_trade = 0
    values: list[float] = []
    for value in raw_signal.fillna(0.0):
        if value <= 0.0:
            bars_in_trade = 0
            values.append(0.0)
            continue
        tier_index = min(bars_in_trade, len(tiers) - 1)
        values.append(float(tiers[tier_index]))
        bars_in_trade += 1
    return pd.Series(values, index=raw_signal.index, name=raw_signal.name, dtype=float)


def count_profit_lock_hits(
    raw_signal: pd.Series,
    price: pd.Series,
    *,
    threshold: float = 1.50,
) -> int:
    """Count trades where unrealized gain first reaches the profit-lock threshold."""
    in_trade = False
    entry_price = np.nan
    hit_this_trade = False
    count = 0
    for signal, current_price in zip(raw_signal.fillna(0.0), price.reindex(raw_signal.index), strict=True):
        if signal <= 0.0 or not np.isfinite(current_price):
            in_trade = False
            entry_price = np.nan
            hit_this_trade = False
            continue
        if not in_trade:
            in_trade = True
            entry_price = float(current_price)
            hit_this_trade = False
        if not hit_this_trade and entry_price > 0.0:
            gain = float(current_price) / entry_price - 1.0
            if gain >= threshold:
                count += 1
                hit_this_trade = True
    return count


def macd_entry_slow_exit_state_machine(
    price: pd.Series,
    *,
    allowed_regime: pd.Series,
    bars_per_day: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    macd_unit: str,
    entry_mode: str,
    entry_confirm_bars: int,
    exit_ma_days: float,
    exit_confirm_bars: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Raw long/cash signal: MACD entry plus slow 200-day-MA exit.

    ``macd_unit`` can be ``"bars"`` for standard 12/26 hourly bars, or
    ``"days"`` for 12/26 trading-day-equivalent EMAs on hourly data.
    """
    clean = price.astype(float).sort_index()
    gate = allowed_regime.reindex(clean.index).fillna(False).astype(bool)
    if macd_unit == "days":
        fast_span = _days_to_bars(macd_fast, bars_per_day)
        slow_span = _days_to_bars(macd_slow, bars_per_day)
        signal_span = _days_to_bars(macd_signal, bars_per_day)
    elif macd_unit == "bars":
        fast_span = int(macd_fast)
        slow_span = int(macd_slow)
        signal_span = int(macd_signal)
    else:
        raise ValueError("macd_unit must be 'bars' or 'days'")

    components = macd_components(
        clean,
        fast_span=fast_span,
        slow_span=slow_span,
        signal_span=signal_span,
    )
    hist = components["macd_hist"]
    macd_line = components["macd"]
    hist_slope = hist.diff()

    if entry_mode == "hist_gt_0":
        raw_entry = hist.gt(0.0)
    elif entry_mode == "hist_cross_above_0":
        raw_entry = hist.gt(0.0) & hist.shift(1).le(0.0)
    elif entry_mode == "hist_gt_0_macd_gt_0":
        raw_entry = hist.gt(0.0) & macd_line.gt(0.0)
    elif entry_mode == "hist_gt_0_and_rising":
        raw_entry = hist.gt(0.0) & hist_slope.gt(0.0)
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

    raw_signal = pd.Series(values, index=clean.index, name=price.name, dtype=float)
    components["entry_flag"] = entry.astype(float)
    components["exit_ma"] = exit_ma
    components["price_exit"] = price_exit.astype(float)
    components["allowed_regime"] = gate.astype(float)
    return raw_signal, components


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
    top_n: int = 9,
) -> None:
    ranked = metrics[metrics["strategy"].ne("benchmark")].copy()
    selected = (
        ranked.sort_values(["sharpe_ratio", "max_drawdown"], ascending=[False, False])
        .head(top_n)["name"]
        .tolist()
    )
    for required in ["base_20ma_entry_full", "base_20ma_entry_profit_lock_150_250"]:
        if required not in selected:
            selected = [required, *selected[: top_n - 1]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for name in dict.fromkeys(selected):
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

    params = dict(config.strategies.regime_switch)
    params.update(
        {
            "target_ticker": args.target_ticker,
            "regime_ticker": args.benchmark_ticker,
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
    bars_per_day = int(params.get("intraday_bars_per_day", 6))

    base_raw = daily_regime_hourly_fast_slow_signal(
        intraday_prices=target_prices,
        daily_prices=daily_prices,
        daily_returns=daily_returns,
        params=params,
    )[args.target_ticker]

    daily_features = compute_regime_features(
        daily_prices,
        daily_returns,
        regime_ticker=args.benchmark_ticker,
        params=params,
    )
    daily_regimes = classify_regimes(daily_features, params=params)
    intraday_regimes = align_daily_regimes_to_intraday(
        daily_regimes,
        target_prices.index,
        lag_days=int(params.get("daily_regime_lag_days", 1)),
        fill_method=params.get("daily_regime_fill_method", "ffill"),
    ).fillna("neutral")
    allowed_regime = intraday_regimes.eq("trend")

    raw_variants: dict[str, pd.DataFrame] = {
        "base_20ma_entry_full": base_raw.to_frame(args.target_ticker),
    }
    profit_lock_150pct_hit_counts: dict[str, float] = {"base_20ma_entry_full": np.nan}
    base_profit_lock = trade_profit_lock_tiers(
        base_raw,
        target,
        thresholds_to_weights=[(1.50, 0.75), (2.50, 0.50)],
    )
    raw_variants["base_20ma_entry_profit_lock_150_250"] = base_profit_lock.to_frame(
        args.target_ticker
    )
    profit_lock_150pct_hit_counts["base_20ma_entry_profit_lock_150_250"] = (
        count_profit_lock_hits(base_raw, target, threshold=1.50)
    )

    macd_diagnostics: dict[str, pd.DataFrame] = {}
    for unit in ["bars", "days"]:
        for entry_mode in [
            "hist_gt_0",
            "hist_cross_above_0",
            "hist_gt_0_macd_gt_0",
            "hist_gt_0_and_rising",
        ]:
            label = f"macd_12_26_{unit}_{entry_mode}"
            raw, diagnostics = macd_entry_slow_exit_state_machine(
                target_prices[args.target_ticker],
                allowed_regime=allowed_regime,
                bars_per_day=bars_per_day,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
                macd_unit=unit,
                entry_mode=entry_mode,
                entry_confirm_bars=1 if "cross" in entry_mode else 2,
                exit_ma_days=200.0,
                exit_confirm_bars=3,
            )
            raw_variants[f"{label}_full"] = raw.to_frame(args.target_ticker)
            profit_lock_150pct_hit_counts[f"{label}_full"] = np.nan

            gradual = gradual_entry_raw_weights(raw)
            raw_variants[f"{label}_gradual_entry"] = gradual.to_frame(args.target_ticker)
            profit_lock_150pct_hit_counts[f"{label}_gradual_entry"] = np.nan

            profit_lock = trade_profit_lock_tiers(
                raw,
                target,
                thresholds_to_weights=[(1.50, 0.75), (2.50, 0.50)],
            )
            profit_lock_count = count_profit_lock_hits(raw, target, threshold=1.50)
            raw_variants[f"{label}_profit_lock_150_250"] = profit_lock.to_frame(
                args.target_ticker
            )
            profit_lock_150pct_hit_counts[f"{label}_profit_lock_150_250"] = profit_lock_count

            gradual_plus_profit_lock = (
                pd.concat([gradual, profit_lock], axis=1).min(axis=1).rename(args.target_ticker)
            )
            raw_variants[f"{label}_gradual_entry_profit_lock_150_250"] = (
                gradual_plus_profit_lock.to_frame(args.target_ticker)
            )
            profit_lock_150pct_hit_counts[
                f"{label}_gradual_entry_profit_lock_150_250"
            ] = profit_lock_count
            macd_diagnostics[label] = diagnostics

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
                "strategy": "macd_entry_experiment",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "transaction_cost_bps": args.transaction_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "short_term_tax_rate": args.short_term_tax_rate,
                        "base_params": params,
                    },
                    sort_keys=True,
                ),
                "pretax_cumulative_return": float((1.0 + pretax).prod() - 1.0),
                "tax_paid_pct_initial_capital": float(taxes_paid.sum()),
                "drawdown_episodes_gt_30pct": drawdown_episode_count(
                    after_tax,
                    threshold=-abs(args.drawdown_threshold_pct / 100.0),
                ),
                "profit_lock_150pct_hit_count": profit_lock_150pct_hit_counts.get(
                    name,
                    np.nan,
                ),
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
                threshold=-abs(args.drawdown_threshold_pct / 100.0),
            ),
            "profit_lock_150pct_hit_count": np.nan,
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
    returns_path = tables_dir / f"{args.output_prefix}_after_tax_returns.csv"
    pretax_returns_path = tables_dir / f"{args.output_prefix}_pretax_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    diagnostics_path = tables_dir / f"{args.output_prefix}_macd_diagnostics.parquet"
    plot_path = figures_dir / f"{args.output_prefix}_top_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(pretax_returns_by_name).to_csv(pretax_returns_path)
    pd.DataFrame(weights_out).to_csv(weights_path)
    pd.concat(macd_diagnostics, axis=1).to_parquet(diagnostics_path)
    _plot_top(
        returns_by_name,
        metrics,
        plot_path,
        title="Synthetic TQQQ 20MA vs 12/26 MACD entry after tax and slippage",
    )

    print(f"Metrics saved to {metrics_path}")
    print(f"After-tax returns saved to {returns_path}")
    print(f"Pretax returns saved to {pretax_returns_path}")
    print(f"Weights saved to {weights_path}")
    print(f"MACD diagnostics saved to {diagnostics_path}")
    print(f"Plot saved to {plot_path}")
    compact = metrics[
        [
            "name",
            "cumulative_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "number_of_trades",
            "exposure_percentage",
            "drawdown_episodes_gt_30pct",
            "profit_lock_150pct_hit_count",
        ]
    ].sort_values("sharpe_ratio", ascending=False)
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
