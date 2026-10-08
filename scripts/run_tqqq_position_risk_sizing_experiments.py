#!/usr/bin/env python
"""Try position/risk-sizing overlays for the synthetic TQQQ strategy."""

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

from trend_following.config import load_config
from trend_following.data_validation import read_price_file
from trend_following.metrics import calculate_metrics, metrics_to_frame
from trend_following.regime import daily_regime_hourly_fast_slow_signal
from trend_following.signals import limit_trades_per_day, make_executable_positions
from trend_following.utils import ensure_directory, resolve_path


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
    parser.add_argument("--output-prefix", default="tqqq_position_risk_sizing_experiments")
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


def drawdown_episode_count(returns: pd.Series, threshold: float = -0.20) -> int:
    dd = _drawdown(returns)
    in_episode = False
    crossed = False
    count = 0
    for value in dd:
        if not in_episode and value < -1e-12:
            in_episode = True
            crossed = value <= threshold
        elif in_episode:
            crossed = crossed or value <= threshold
            if value >= -1e-12:
                if crossed:
                    count += 1
                in_episode = False
                crossed = False
    if in_episode and crossed:
        count += 1
    return count


def simulate_after_tax_portfolio(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    transaction_cost_bps: float,
    slippage_bps: float,
    tax_rate: float,
    liquidate_at_end: bool = True,
    rebalance_on_weight_change_only: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Approximate after-tax portfolio returns with proportional basis tracking.

    This is a research approximation, not tax advice. It treats all realized
    gains as short-term, nets realized gains/losses by calendar year, carries
    losses forward, and pays tax from the portfolio at year-end. Partial sells
    realize gains/losses pro rata against tracked cost basis.
    """
    if not 0.0 <= tax_rate <= 1.0:
        raise ValueError("tax_rate must be between 0 and 1")

    common_index = returns.index.intersection(target_weights.index)
    common_columns = [column for column in returns.columns if column in target_weights.columns]
    returns = returns.loc[common_index, common_columns].fillna(0.0).astype(float)
    weights = target_weights.loc[common_index, common_columns].fillna(0.0).clip(0.0, 1.0)
    weight_sum = weights.sum(axis=1)
    if weight_sum.gt(1.0 + 1e-9).any():
        raise ValueError("Target weights must sum to at most 1.0")

    holdings = pd.Series(0.0, index=common_columns, dtype=float)
    basis = pd.Series(0.0, index=common_columns, dtype=float)
    cash = 1.0
    loss_carryforward = 0.0
    realized_this_year = 0.0
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0

    after_tax_returns: list[float] = []
    pretax_returns: list[float] = []
    taxes_paid: list[float] = []
    turnover_values: list[float] = []
    timestamps = list(common_index)
    previous_target_weights = pd.Series(np.nan, index=common_columns, dtype=float)

    def equity_value() -> float:
        return float(cash + holdings.sum())

    def withdraw(amount: float) -> None:
        nonlocal cash, holdings, basis
        if amount <= 0:
            return
        equity_before = equity_value()
        if equity_before <= 0:
            return
        scale = max((equity_before - amount) / equity_before, 0.0)
        cash *= scale
        holdings *= scale
        basis *= scale

    for i, timestamp in enumerate(timestamps):
        equity_start = equity_value()
        if equity_start <= 0:
            after_tax_returns.append(np.nan)
            pretax_returns.append(np.nan)
            taxes_paid.append(0.0)
            turnover_values.append(0.0)
            continue

        current_target_weights = weights.loc[timestamp].astype(float)
        target_changed = not np.allclose(
            current_target_weights.to_numpy(),
            previous_target_weights.fillna(0.0).to_numpy(),
            atol=1e-12,
            rtol=0.0,
        )
        turnover_dollars = 0.0
        if target_changed or not rebalance_on_weight_change_only:
            desired_values = current_target_weights * equity_start
            turnover_dollars = float((desired_values - holdings).abs().sum())

            for asset in common_columns:
                current_value = float(holdings[asset])
                desired_value = float(desired_values[asset])
                delta = desired_value - current_value
                if delta < -1e-12 and current_value > 0:
                    sell_amount = min(-delta, current_value)
                    fraction_sold = sell_amount / current_value
                    realized_this_year += sell_amount - float(basis[asset]) * fraction_sold
                    basis[asset] *= 1.0 - fraction_sold
                    holdings[asset] -= sell_amount
                    cash += sell_amount
                elif delta > 1e-12:
                    buy_amount = min(delta, cash + delta)  # guard against tiny floating errors
                    holdings[asset] += buy_amount
                    basis[asset] += buy_amount
                    cash -= buy_amount

            trading_cost = turnover_dollars * cost_rate
            withdraw(trading_cost)
            previous_target_weights = current_target_weights.copy()

        equity_after_trading = equity_value()
        for asset in common_columns:
            holdings[asset] *= 1.0 + float(returns.at[timestamp, asset])

        if liquidate_at_end and i == len(timestamps) - 1:
            for asset in common_columns:
                if holdings[asset] > 1e-12:
                    realized_this_year += float(holdings[asset] - basis[asset])
                    cash += float(holdings[asset])
                    holdings[asset] = 0.0
                    basis[asset] = 0.0

        next_year_differs = i == len(timestamps) - 1 or timestamps[i + 1].year != timestamp.year
        tax_paid = 0.0
        if next_year_differs:
            taxable_after_carry = realized_this_year + loss_carryforward
            if taxable_after_carry > 0:
                tax_paid = tax_rate * taxable_after_carry
                withdraw(tax_paid)
                loss_carryforward = 0.0
            else:
                loss_carryforward = taxable_after_carry
            realized_this_year = 0.0

        equity_end = equity_value()
        after_tax_returns.append(equity_end / equity_start - 1.0)
        pretax_returns.append(equity_after_trading / equity_start - 1.0)
        taxes_paid.append(tax_paid)
        turnover_values.append(turnover_dollars / equity_start)

    return (
        pd.Series(after_tax_returns, index=common_index, name="after_tax_return"),
        pd.Series(pretax_returns, index=common_index, name="pretax_return"),
        pd.Series(taxes_paid, index=common_index, name="tax_paid"),
        pd.Series(turnover_values, index=common_index, name="turnover"),
    )


def profit_lock_raw_weights(
    base_raw: pd.Series,
    price: pd.Series,
    *,
    gain_threshold: float,
    reduced_weight: float,
) -> pd.Series:
    """Scale down a trade after its unrealized gain crosses a threshold."""
    in_trade = False
    entry_price = np.nan
    locked = False
    values: list[float] = []
    for signal, current_price in zip(base_raw.fillna(0.0), price.reindex(base_raw.index), strict=True):
        if signal <= 0 or not np.isfinite(current_price):
            in_trade = False
            entry_price = np.nan
            locked = False
            values.append(0.0)
            continue
        if not in_trade:
            in_trade = True
            entry_price = float(current_price)
            locked = False
        if not locked and entry_price > 0 and current_price / entry_price - 1.0 >= gain_threshold:
            locked = True
        values.append(reduced_weight if locked else 1.0)
    return pd.Series(values, index=base_raw.index, name=base_raw.name, dtype=float)


def trailing_stop_raw_weights(
    base_raw: pd.Series,
    price: pd.Series,
    *,
    trail_pct: float,
) -> pd.Series:
    """Exit after a trade-level trailing drawdown and wait for base reset."""
    peak = np.nan
    stopped = False
    base_was_on = False
    values: list[float] = []
    for signal, current_price in zip(base_raw.fillna(0.0), price.reindex(base_raw.index), strict=True):
        if signal <= 0 or not np.isfinite(current_price):
            peak = np.nan
            stopped = False
            base_was_on = False
            values.append(0.0)
            continue
        if not base_was_on:
            peak = float(current_price)
            stopped = False
            base_was_on = True
        peak = max(float(peak), float(current_price))
        if not stopped and current_price / peak - 1.0 <= -trail_pct:
            stopped = True
        values.append(0.0 if stopped else 1.0)
    return pd.Series(values, index=base_raw.index, name=base_raw.name, dtype=float)


def volatility_target_raw_weights(
    base_raw: pd.Series,
    returns: pd.Series,
    *,
    target_vol: float,
    lookback_bars: int,
    annualization: int,
    max_weight: float = 1.0,
) -> pd.Series:
    """Scale exposure by realized volatility, capped at max_weight."""
    realized_vol = returns.rolling(lookback_bars, min_periods=lookback_bars).std(ddof=0)
    realized_vol = realized_vol * np.sqrt(annualization)
    scale = (target_vol / realized_vol.replace(0.0, np.nan)).clip(upper=max_weight)
    return (base_raw * scale).fillna(0.0).clip(0.0, max_weight).rename(base_raw.name)


def executable_weights(
    raw_weights: pd.DataFrame,
    *,
    config,
) -> pd.DataFrame:
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
        selected = ["base_full_tqqq"] + selected[: top_n - 1]
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

    raw_variants: dict[str, pd.DataFrame] = {
        "base_full_tqqq": base_raw.to_frame(args.target_ticker),
        "profit_lock_50pct_gain_to_50pct": profit_lock_raw_weights(
            base_raw, target, gain_threshold=0.50, reduced_weight=0.50
        ).to_frame(args.target_ticker),
        "profit_lock_100pct_gain_to_50pct": profit_lock_raw_weights(
            base_raw, target, gain_threshold=1.00, reduced_weight=0.50
        ).to_frame(args.target_ticker),
        "profit_lock_150pct_gain_to_50pct": profit_lock_raw_weights(
            base_raw, target, gain_threshold=1.50, reduced_weight=0.50
        ).to_frame(args.target_ticker),
        "trailing_stop_20pct": trailing_stop_raw_weights(
            base_raw, target, trail_pct=0.20
        ).to_frame(args.target_ticker),
        "trailing_stop_25pct": trailing_stop_raw_weights(
            base_raw, target, trail_pct=0.25
        ).to_frame(args.target_ticker),
        "trailing_stop_30pct": trailing_stop_raw_weights(
            base_raw, target, trail_pct=0.30
        ).to_frame(args.target_ticker),
    }

    for target_vol in [0.25, 0.30, 0.35, 0.40]:
        raw_variants[f"vol_target_{int(target_vol * 100)}pct_cap1"] = (
            volatility_target_raw_weights(
                base_raw,
                target_returns[args.target_ticker],
                target_vol=target_vol,
                lookback_bars=20 * int(params.get("intraday_bars_per_day", 6)),
                annualization=config.backtest.annualization,
                max_weight=1.0,
            ).to_frame(args.target_ticker)
        )

    for tqqq_weight in [0.75, 0.50, 0.33]:
        qqq_weight = 1.0 - tqqq_weight
        raw_variants[f"split_{int(tqqq_weight * 100)}pct_tqqq_{int(qqq_weight * 100)}pct_qqq"] = (
            pd.DataFrame(
                {
                    args.target_ticker: base_raw * tqqq_weight,
                    args.benchmark_ticker: base_raw * qqq_weight,
                },
                index=base_raw.index,
            )
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
                "strategy": "position_risk_sizing_overlay",
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
                "drawdown_episodes_gt_20pct": drawdown_episode_count(after_tax),
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
        }
    )
    metric_rows.append(benchmark_metrics)
    returns_by_name["buy_hold_qqq"] = qqq_returns[args.benchmark_ticker]

    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

    metrics = metrics_to_frame(metric_rows)
    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    returns_path = tables_dir / f"{args.output_prefix}_after_tax_returns.csv"
    pretax_returns_path = tables_dir / f"{args.output_prefix}_pretax_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    plot_path = figures_dir / f"{args.output_prefix}_top_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(pretax_returns_by_name).to_csv(pretax_returns_path)
    pd.DataFrame(weights_out).to_csv(weights_path)
    _plot_top(
        returns_by_name,
        metrics,
        plot_path,
        title="Synthetic TQQQ position/risk-sizing overlays after tax and slippage",
    )

    print(f"Metrics saved to {metrics_path}")
    print(f"After-tax returns saved to {returns_path}")
    print(f"Pretax returns saved to {pretax_returns_path}")
    print(f"Weights saved to {weights_path}")
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
            "drawdown_episodes_gt_20pct",
        ]
    ].sort_values("sharpe_ratio", ascending=False)
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
