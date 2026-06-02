#!/usr/bin/env python
"""Try 0/25/50/75/100% no-lookahead sizing overlays for synthetic TQQQ."""

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

from run_tqqq_ma_derivative_filter_experiments import build_ma_features  # noqa: E402
from run_tqqq_position_risk_sizing_experiments import (  # noqa: E402
    drawdown_episode_count,
    simulate_after_tax_portfolio,
)
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.metrics import calculate_metrics, metrics_to_frame  # noqa: E402
from trend_following.regime import daily_regime_hourly_fast_slow_signal  # noqa: E402
from trend_following.signals import limit_trades_per_day, make_executable_positions  # noqa: E402
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

TIER_VALUES = (0.0, 0.25, 0.50, 0.75, 1.0)


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
    parser.add_argument("--output-prefix", default="tqqq_tiered_sizing_experiments")
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


def _round_to_tier(value: pd.Series) -> pd.Series:
    """Round continuous values in [0, 1] to the nearest allowed tier."""
    rounded = (value.clip(0.0, 1.0) * 4.0).round() / 4.0
    return rounded.clip(0.0, 1.0)


def _known_today(size_from_close: pd.Series) -> pd.Series:
    """Shift a close-D sizing estimate so row D uses only data through D-1."""
    return size_from_close.astype(float).shift(1).fillna(0.0).clip(0.0, 1.0)


def _daily_size_to_intraday(size_known_today: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    daily = size_known_today.astype(float).clip(0.0, 1.0).copy()
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    intraday_dates = index.tz_localize(None).normalize()
    unique_dates = pd.DatetimeIndex(intraday_dates.unique()).sort_values()
    aligned_by_date = daily.reindex(unique_dates, method="ffill")
    return pd.Series(
        aligned_by_date.reindex(intraday_dates).fillna(0.0).to_numpy(dtype=float),
        index=index,
        dtype=float,
    )


def _map_trend_score_linear(score: pd.Series) -> pd.Series:
    mapping = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.25, 4: 0.50, 5: 0.75, 6: 1.0}
    return score.round().clip(0, 6).map(mapping).astype(float)


def _map_trend_score_lenient(score: pd.Series) -> pd.Series:
    mapping = {0: 0.0, 1: 0.0, 2: 0.25, 3: 0.50, 4: 0.75, 5: 0.75, 6: 1.0}
    return score.round().clip(0, 6).map(mapping).astype(float)


def _map_trend_score_conservative(score: pd.Series) -> pd.Series:
    mapping = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.25, 5: 0.50, 6: 1.0}
    return score.round().clip(0, 6).map(mapping).astype(float)


def _map_fast_score_linear(score: pd.Series) -> pd.Series:
    mapping = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}
    return score.round().clip(0, 4).map(mapping).astype(float)


def _map_fast_score_lenient(score: pd.Series) -> pd.Series:
    mapping = {0: 0.25, 1: 0.50, 2: 0.75, 3: 1.0, 4: 1.0}
    return score.round().clip(0, 4).map(mapping).astype(float)


def _map_fast_health_thresholds(fast_health: pd.Series) -> pd.Series:
    """Map the composite short/mid-term health factor to five sizing tiers."""
    return pd.Series(
        np.select(
            [
                fast_health.ge(0.02),
                fast_health.ge(0.00),
                fast_health.ge(-0.01),
                fast_health.ge(-0.02),
            ],
            [1.0, 0.75, 0.50, 0.25],
            default=0.0,
        ),
        index=fast_health.index,
        dtype=float,
    )


def _map_slope200_percentile(percentile: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                percentile.ge(0.60),
                percentile.ge(0.40),
                percentile.ge(0.20),
                percentile.ge(0.10),
            ],
            [1.0, 0.75, 0.50, 0.25],
            default=0.0,
        ),
        index=percentile.index,
        dtype=float,
    )


def build_daily_sizing_tiers(features: pd.DataFrame) -> pd.DataFrame:
    """Build close-date sizing estimates from daily QQQ MA-health features."""
    fast_score = pd.concat(
        [
            features["price"].gt(features["sma20"]),
            features["sma20"].gt(features["sma50"]),
            features["slope20_5"].gt(0.0),
            features["slope50_10"].gt(0.0),
        ],
        axis=1,
    ).sum(axis=1)
    trend_score = features["trend_score"]

    trend_linear = _map_trend_score_linear(trend_score)
    fast_linear = _map_fast_score_linear(fast_score)

    short_damage = features["price"].lt(features["sma20"]) & features["slope20_5"].lt(0.0)
    mid_damage = features["price"].lt(features["sma50"]) & features["sma20"].lt(features["sma50"])
    crash_defense = pd.Series(1.0, index=features.index, dtype=float)
    crash_defense = crash_defense.mask(short_damage, 0.50)
    crash_defense = crash_defense.mask(mid_damage, 0.25)
    crash_defense = crash_defense.mask(trend_score.le(3), 0.0)
    crash_defense = crash_defense.mask(trend_score.eq(4), np.minimum(crash_defense, 0.50))
    crash_defense = crash_defense.mask(trend_score.eq(5), np.minimum(crash_defense, 0.75))

    tiers = pd.DataFrame(
        {
            "tier_trend_score_linear": trend_linear,
            "tier_trend_score_lenient": _map_trend_score_lenient(trend_score),
            "tier_trend_score_conservative": _map_trend_score_conservative(trend_score),
            "tier_fast_score_linear": fast_linear,
            "tier_fast_score_lenient": _map_fast_score_lenient(fast_score),
            "tier_fast_health_thresholds": _map_fast_health_thresholds(features["fast_health"]),
            "tier_slope200_percentile": _map_slope200_percentile(features["slope200_pctile"]),
            "tier_hybrid_min_trend_fast": pd.concat([trend_linear, fast_linear], axis=1).min(axis=1),
            "tier_hybrid_avg_trend_fast": _round_to_tier(
                pd.concat([trend_linear, fast_linear], axis=1).mean(axis=1)
            ),
            "tier_crash_defense": crash_defense,
        },
        index=features.index,
    )
    return tiers.apply(_known_today).clip(0.0, 1.0)


def trade_profit_lock_tiers(
    base_raw: pd.Series,
    price: pd.Series,
    *,
    thresholds_to_weights: list[tuple[float, float]],
) -> pd.Series:
    """Reduce size within a trade after unrealized-gain thresholds are crossed.

    The sizing level can only move down while a base trade remains open; it
    resets to 100% on the next new base entry. This reduces churn versus daily
    indicator rebalancing and keeps every raw weight in {0, .25, .50, .75, 1}.
    """
    thresholds_to_weights = sorted(thresholds_to_weights)
    if any(weight not in TIER_VALUES for _, weight in thresholds_to_weights):
        raise ValueError("profit-lock weights must be one of the allowed tiers")

    in_trade = False
    entry_price = np.nan
    current_weight = 0.0
    values: list[float] = []

    for signal, current_price in zip(base_raw.fillna(0.0), price.reindex(base_raw.index), strict=True):
        if signal <= 0 or not np.isfinite(current_price):
            in_trade = False
            entry_price = np.nan
            current_weight = 0.0
            values.append(0.0)
            continue

        if not in_trade:
            in_trade = True
            entry_price = float(current_price)
            current_weight = 1.0

        gain = float(current_price) / entry_price - 1.0 if entry_price > 0 else 0.0
        for threshold, weight in thresholds_to_weights:
            if gain >= threshold:
                current_weight = min(current_weight, weight)
        values.append(current_weight)

    return pd.Series(values, index=base_raw.index, name=base_raw.name, dtype=float)


def trade_peak_drawdown_tiers(
    base_raw: pd.Series,
    price: pd.Series,
    *,
    drawdowns_to_weights: list[tuple[float, float]],
) -> pd.Series:
    """Reduce size within a trade as price falls from its trade-level peak."""
    drawdowns_to_weights = sorted(drawdowns_to_weights)
    if any(weight not in TIER_VALUES for _, weight in drawdowns_to_weights):
        raise ValueError("drawdown-tier weights must be one of the allowed tiers")

    in_trade = False
    peak = np.nan
    current_weight = 0.0
    values: list[float] = []

    for signal, current_price in zip(base_raw.fillna(0.0), price.reindex(base_raw.index), strict=True):
        if signal <= 0 or not np.isfinite(current_price):
            in_trade = False
            peak = np.nan
            current_weight = 0.0
            values.append(0.0)
            continue

        if not in_trade:
            in_trade = True
            peak = float(current_price)
            current_weight = 1.0
        else:
            peak = max(float(peak), float(current_price))

        drawdown = float(current_price) / peak - 1.0 if peak > 0 else 0.0
        for threshold, weight in drawdowns_to_weights:
            if drawdown <= -abs(threshold):
                current_weight = min(current_weight, weight)
        values.append(current_weight)

    return pd.Series(values, index=base_raw.index, name=base_raw.name, dtype=float)


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
    if "base_full_tqqq" not in selected:
        selected = ["base_full_tqqq", *selected[: top_n - 1]]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for name in selected:
        returns = returns_by_name[name]
        _equity(returns).plot(ax=axes[0], label=name, linewidth=1.25)
        _drawdown(returns).plot(ax=axes[1], label=name, linewidth=1.25)
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
    base_raw = daily_regime_hourly_fast_slow_signal(
        intraday_prices=target_prices,
        daily_prices=daily_prices,
        daily_returns=daily_returns,
        params=params,
    )[args.target_ticker]

    daily_features = build_ma_features(daily_qqq)
    daily_tiers = build_daily_sizing_tiers(daily_features)
    intraday_tiers = pd.DataFrame(
        {
            column: _daily_size_to_intraday(daily_tiers[column], target_prices.index)
            for column in daily_tiers.columns
        },
        index=target_prices.index,
    )

    raw_variants: dict[str, pd.DataFrame] = {
        "base_full_tqqq": base_raw.to_frame(args.target_ticker),
    }
    raw_variants.update(
        {
            name: (base_raw * intraday_tiers[name]).rename(args.target_ticker).to_frame()
            for name in intraday_tiers.columns
        }
    )
    raw_variants.update(
        {
            "trade_gain_50_100_150_to_75_50_25": trade_profit_lock_tiers(
                base_raw,
                target,
                thresholds_to_weights=[(0.50, 0.75), (1.00, 0.50), (1.50, 0.25)],
            ).to_frame(args.target_ticker),
            "trade_gain_100_150_200_to_75_50_25": trade_profit_lock_tiers(
                base_raw,
                target,
                thresholds_to_weights=[(1.00, 0.75), (1.50, 0.50), (2.00, 0.25)],
            ).to_frame(args.target_ticker),
            "trade_gain_150_to_50": trade_profit_lock_tiers(
                base_raw,
                target,
                thresholds_to_weights=[(1.50, 0.50)],
            ).to_frame(args.target_ticker),
            "trade_gain_150_250_to_75_50": trade_profit_lock_tiers(
                base_raw,
                target,
                thresholds_to_weights=[(1.50, 0.75), (2.50, 0.50)],
            ).to_frame(args.target_ticker),
            "trade_peakdd_15_25_35_to_75_50_25": trade_peak_drawdown_tiers(
                base_raw,
                target,
                drawdowns_to_weights=[(0.15, 0.75), (0.25, 0.50), (0.35, 0.25)],
            ).to_frame(args.target_ticker),
            "trade_peakdd_20_30_40_to_75_50_0": trade_peak_drawdown_tiers(
                base_raw,
                target,
                drawdowns_to_weights=[(0.20, 0.75), (0.30, 0.50), (0.40, 0.0)],
            ).to_frame(args.target_ticker),
            "trade_peakdd_25_to_50": trade_peak_drawdown_tiers(
                base_raw,
                target,
                drawdowns_to_weights=[(0.25, 0.50)],
            ).to_frame(args.target_ticker),
        }
    )

    # Conservative hybrid: trade-level profit lock plus daily slope-percentile
    # tier. Taking the min of two tiered signals preserves the allowed set.
    profit_lock_150_to_50 = raw_variants["trade_gain_150_to_50"][args.target_ticker]
    slope_tier = intraday_tiers["tier_slope200_percentile"].reindex(base_raw.index).fillna(0.0)
    crash_tier = intraday_tiers["tier_crash_defense"].reindex(base_raw.index).fillna(0.0)
    raw_variants["trade_gain_150_to_50_min_slope200_tier"] = (
        pd.concat([profit_lock_150_to_50, base_raw * slope_tier], axis=1)
        .min(axis=1)
        .rename(args.target_ticker)
        .to_frame()
    )
    raw_variants["trade_gain_150_to_50_min_crash_defense"] = (
        pd.concat([profit_lock_150_to_50, base_raw * crash_tier], axis=1)
        .min(axis=1)
        .rename(args.target_ticker)
        .to_frame()
    )

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
                "strategy": "tiered_sizing_overlay",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "allowed_tiers": TIER_VALUES,
                        "transaction_cost_bps": args.transaction_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "short_term_tax_rate": args.short_term_tax_rate,
                        "base_params": params,
                    },
                    sort_keys=True,
                ),
                "pretax_cumulative_return": float((1.0 + pretax).prod() - 1.0),
                "tax_paid_pct_initial_capital": float(taxes_paid.sum()),
                "drawdown_episodes_gt_20pct": drawdown_episode_count(after_tax),
                "average_position_when_base_long": float(
                    weights.sum(axis=1)[base_raw.reindex(weights.index).fillna(0.0).gt(0.0)].mean()
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
            "drawdown_episodes_gt_20pct": drawdown_episode_count(
                qqq_returns[args.benchmark_ticker]
            ),
            "average_position_when_base_long": np.nan,
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
    daily_features_path = tables_dir / f"{args.output_prefix}_daily_ma_features.csv"
    daily_tiers_path = tables_dir / f"{args.output_prefix}_daily_tiers_known_today.csv"
    intraday_tiers_path = tables_dir / f"{args.output_prefix}_intraday_tiers.csv"
    plot_path = figures_dir / f"{args.output_prefix}_top_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(pretax_returns_by_name).to_csv(pretax_returns_path)
    pd.DataFrame(weights_out).to_csv(weights_path)
    daily_features.to_csv(daily_features_path)
    daily_tiers.to_csv(daily_tiers_path)
    intraday_tiers.to_csv(intraday_tiers_path)
    _plot_top(
        returns_by_name,
        metrics,
        plot_path,
        title="Synthetic TQQQ five-tier sizing overlays after tax and slippage",
    )

    print(f"Metrics saved to {metrics_path}")
    print(f"After-tax returns saved to {returns_path}")
    print(f"Pretax returns saved to {pretax_returns_path}")
    print(f"Weights saved to {weights_path}")
    print(f"Daily MA features saved to {daily_features_path}")
    print(f"Daily no-lookahead tiers saved to {daily_tiers_path}")
    print(f"Intraday tiers saved to {intraday_tiers_path}")
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
            "average_position_when_base_long",
            "drawdown_episodes_gt_20pct",
        ]
    ].sort_values("sharpe_ratio", ascending=False)
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
