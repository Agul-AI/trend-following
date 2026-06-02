#!/usr/bin/env python
"""Recalculate kept synthetic TQQQ candidates with cash earning a risk-free rate.

Candidates:
1. QQQ entry + TQQQ/synthetic-TQQQ exit + +200/+300 profit lock.
2. TQQQ/synthetic-TQQQ entry + TQQQ/synthetic-TQQQ exit + +200/+300 profit lock.
3. Current preferred: QQQ entry + QQQ exit + no profit lock.
4. New serious candidate: no daily regime gate; QQQ hourly 200MA entry/exit gate.

Cash-yield assumption:
- Uninvested cash earns ``cash_annual_yield`` while out of the market.
- The default is 3% annual, approximated per hourly bar using the configured
  annualization count.
- In the taxable approximation, cash interest is taxed at the same short-term
  tax rate at year-end. Set ``--cash-interest-tax-rate 0`` for pre-tax cash yield.
"""

from __future__ import annotations

import argparse
import json
import math
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

from run_tqqq_daily_gate_ablation import no_daily_gate_hourly_ma_gate_signal  # noqa: E402
from run_tqqq_entry_signal_comparison import (  # noqa: E402
    _drawdown,
    _equity,
    _returns_from_prices,
    executable_weights,
)
from run_tqqq_macd_entry_experiments import count_profit_lock_hits  # noqa: E402
from run_tqqq_mixed_entry_exit_source_comparison import (  # noqa: E402
    LOCK_SCHEMES,
    _apply_lock,
    mixed_source_signal,
)
from run_tqqq_position_risk_sizing_experiments import drawdown_episode_count  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.metrics import calculate_metrics, metrics_to_frame  # noqa: E402
from trend_following.regime import (  # noqa: E402
    align_daily_regimes_to_intraday,
    classify_regimes,
    compute_regime_features,
)
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
    parser.add_argument("--cash-interest-tax-rate", type=float, default=None)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--output-prefix", default="tqqq_cash_yield_candidate_comparison")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def simulate_after_tax_portfolio_with_cash_yield(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    transaction_cost_bps: float,
    slippage_bps: float,
    tax_rate: float,
    cash_annual_yield: float,
    annualization: int,
    cash_interest_tax_rate: float | None = None,
    liquidate_at_end: bool = True,
    rebalance_on_weight_change_only: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Approximate after-tax returns with cash yield on uninvested capital.

    This extends the project's simple after-tax simulator by crediting a per-bar
    risk-free return to the cash balance. Cash interest is separately taxed at
    year-end by ``cash_interest_tax_rate``; if omitted, it uses ``tax_rate``.
    """
    if not 0.0 <= tax_rate <= 1.0:
        raise ValueError("tax_rate must be between 0 and 1")
    if cash_interest_tax_rate is None:
        cash_interest_tax_rate = tax_rate
    if not 0.0 <= cash_interest_tax_rate <= 1.0:
        raise ValueError("cash_interest_tax_rate must be between 0 and 1")
    if annualization <= 0:
        raise ValueError("annualization must be positive")

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
    cash_interest_this_year = 0.0
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0
    cash_period_return = (1.0 + cash_annual_yield) ** (1.0 / annualization) - 1.0

    after_tax_returns: list[float] = []
    turnover_values: list[float] = []
    taxes_paid: list[float] = []
    cash_interest_values: list[float] = []
    cash_weight_values: list[float] = []
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
            taxes_paid.append(0.0)
            turnover_values.append(0.0)
            cash_interest_values.append(0.0)
            cash_weight_values.append(0.0)
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
                    buy_amount = min(delta, cash + delta)  # guard against tiny float errors
                    holdings[asset] += buy_amount
                    basis[asset] += buy_amount
                    cash -= buy_amount

            trading_cost = turnover_dollars * cost_rate
            withdraw(trading_cost)
            previous_target_weights = current_target_weights.copy()

        cash_weight_before_interest = cash / equity_value() if equity_value() > 0 else 0.0

        for asset in common_columns:
            holdings[asset] *= 1.0 + float(returns.at[timestamp, asset])

        cash_interest = max(cash, 0.0) * cash_period_return
        cash += cash_interest
        cash_interest_this_year += cash_interest

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
                tax_paid += tax_rate * taxable_after_carry
                loss_carryforward = 0.0
            else:
                loss_carryforward = taxable_after_carry
            if cash_interest_this_year > 0:
                tax_paid += cash_interest_tax_rate * cash_interest_this_year
            withdraw(tax_paid)
            realized_this_year = 0.0
            cash_interest_this_year = 0.0

        equity_end = equity_value()
        after_tax_returns.append(equity_end / equity_start - 1.0)
        taxes_paid.append(tax_paid)
        turnover_values.append(turnover_dollars / equity_start)
        cash_interest_values.append(cash_interest)
        cash_weight_values.append(cash_weight_before_interest)

    return (
        pd.Series(after_tax_returns, index=common_index, name="after_tax_return"),
        pd.Series(taxes_paid, index=common_index, name="tax_paid"),
        pd.Series(turnover_values, index=common_index, name="turnover"),
        pd.Series(cash_interest_values, index=common_index, name="cash_interest"),
        pd.Series(cash_weight_values, index=common_index, name="cash_weight"),
    )


def _add_dd_counts(metrics: dict[str, Any], returns: pd.Series) -> None:
    metrics["drawdown_episodes_gt_20pct"] = drawdown_episode_count(returns, threshold=-0.20)
    metrics["drawdown_episodes_gt_30pct"] = drawdown_episode_count(returns, threshold=-0.30)
    metrics["drawdown_episodes_gt_40pct"] = drawdown_episode_count(returns, threshold=-0.40)
    metrics["drawdown_episodes_gt_50pct"] = drawdown_episode_count(returns, threshold=-0.50)
    metrics["dd_episodes_gt_20_30_40_50pct"] = (
        f"{metrics['drawdown_episodes_gt_20pct']}/"
        f"{metrics['drawdown_episodes_gt_30pct']}/"
        f"{metrics['drawdown_episodes_gt_40pct']}/"
        f"{metrics['drawdown_episodes_gt_50pct']}"
    )


def _plot(returns_by_name: dict[str, pd.Series], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for name, returns in returns_by_name.items():
        _equity(returns).plot(ax=axes[0], label=name, linewidth=1.15)
        _drawdown(returns).plot(ax=axes[1], label=name, linewidth=1.15)
    axes[0].set_title("After-tax equity with 3% cash yield")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("After-tax drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("Synthetic TQQQ candidates with risk-free return on out-of-market cash")
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cash_interest_tax_rate = (
        args.short_term_tax_rate if args.cash_interest_tax_rate is None else args.cash_interest_tax_rate
    )
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
    all_returns = pd.concat([target_returns, qqq_returns], axis=1).loc[common]

    params = dict(config.strategies.regime_switch)
    params.update(
        {
            "target_ticker": args.target_ticker,
            "regime_ticker": args.benchmark_ticker,
            "sma_window": 200,
            "use_variance_ratio_for_trend": False,
        }
    )
    bars_per_day = int(params.get("intraday_bars_per_day", 6))

    daily_prices = daily_qqq.to_frame()
    daily_returns = _returns_from_prices(daily_prices)
    daily_features = compute_regime_features(
        daily_prices,
        daily_returns,
        regime_ticker=args.benchmark_ticker,
        params=params,
    )
    daily_regimes = classify_regimes(daily_features, params=params)
    intraday_regimes = align_daily_regimes_to_intraday(
        daily_regimes,
        common,
        lag_days=int(params.get("daily_regime_lag_days", 1)),
        fill_method=params.get("daily_regime_fill_method", "ffill"),
    ).fillna("neutral")
    allowed_regime = intraday_regimes.eq("trend")

    sources = {
        "tqqq": target_prices[args.target_ticker],
        "qqq": qqq_prices[args.benchmark_ticker],
    }

    candidate_specs = (
        ("qqq_entry__tqqq_exit__new_lock_200_300", "qqq", "tqqq", "new_lock_200_300"),
        ("tqqq_entry__tqqq_exit__new_lock_200_300", "tqqq", "tqqq", "new_lock_200_300"),
        ("preferred_qqq_entry__qqq_exit__no_lock", "qqq", "qqq", "full_no_lock"),
    )

    raw_variants: dict[str, pd.DataFrame] = {}
    meta_by_name: dict[str, dict[str, Any]] = {}
    for name, entry_source, exit_source, lock_label in candidate_specs:
        raw, _ = mixed_source_signal(
            entry_price=sources[entry_source],
            exit_price=sources[exit_source],
            output_index=common,
            allowed_regime=allowed_regime,
            bars_per_day=bars_per_day,
            average_type=args.average_type,
            macd_unit=args.macd_unit,
            entry_confirm_bars=2,
            exit_confirm_bars=3,
            exit_ma_days=200,
        )
        raw = raw.rename(args.target_ticker)
        scheme = LOCK_SCHEMES[lock_label]
        weights = _apply_lock(raw, target_prices[args.target_ticker], scheme)
        raw_variants[name] = weights.to_frame(args.target_ticker)
        meta_by_name[name] = {
            "entry_source": entry_source,
            "exit_source": exit_source,
            "lock_scheme": lock_label,
            "daily_regime_gate": True,
            "hourly_200ma_entry_gate": False,
            "lock_hit_200_count": (
                float(count_profit_lock_hits(raw, target_prices[args.target_ticker], threshold=2.00))
                if scheme
                else math.nan
            ),
            "lock_hit_300_count": (
                float(count_profit_lock_hits(raw, target_prices[args.target_ticker], threshold=3.00))
                if scheme
                else math.nan
            ),
        }

    candidate_raw, _ = no_daily_gate_hourly_ma_gate_signal(
        entry_price=qqq_prices[args.benchmark_ticker],
        exit_price=qqq_prices[args.benchmark_ticker],
        output_index=common,
        bars_per_day=bars_per_day,
        average_type=args.average_type,
        macd_unit=args.macd_unit,
        entry_confirm_bars=2,
        exit_confirm_bars=3,
        exit_ma_days=200,
    )
    candidate_name = "new_candidate_no_daily_gate__qqq_hourly_200ma_entry_exit"
    raw_variants[candidate_name] = candidate_raw.rename(args.target_ticker).to_frame(args.target_ticker)
    meta_by_name[candidate_name] = {
        "entry_source": "qqq",
        "exit_source": "qqq",
        "lock_scheme": "full_no_lock",
        "daily_regime_gate": False,
        "hourly_200ma_entry_gate": True,
        "lock_hit_200_count": math.nan,
        "lock_hit_300_count": math.nan,
    }

    metric_rows: list[dict[str, Any]] = []
    returns_with_cash_by_name: dict[str, pd.Series] = {}
    returns_zero_cash_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.Series] = {}
    cash_interest_by_name: dict[str, pd.Series] = {}
    cash_weight_by_name: dict[str, pd.Series] = {}

    for name, raw_weights in raw_variants.items():
        weights = executable_weights(raw_weights, config=config).reindex(common).fillna(0.0)
        after_tax_cash, taxes_cash, turnover_cash, cash_interest, cash_weight = (
            simulate_after_tax_portfolio_with_cash_yield(
                all_returns[[args.target_ticker]],
                weights[[args.target_ticker]],
                transaction_cost_bps=args.transaction_cost_bps,
                slippage_bps=args.slippage_bps,
                tax_rate=args.short_term_tax_rate,
                cash_annual_yield=args.cash_annual_yield,
                annualization=config.backtest.annualization,
                cash_interest_tax_rate=cash_interest_tax_rate,
            )
        )
        after_tax_zero, taxes_zero, turnover_zero, _, _ = simulate_after_tax_portfolio_with_cash_yield(
            all_returns[[args.target_ticker]],
            weights[[args.target_ticker]],
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            tax_rate=args.short_term_tax_rate,
            cash_annual_yield=0.0,
            annualization=config.backtest.annualization,
            cash_interest_tax_rate=cash_interest_tax_rate,
        )

        metrics = calculate_metrics(
            after_tax_cash,
            turnover=turnover_cash,
            weights=weights.sum(axis=1),
            annualization=config.backtest.annualization,
        )
        baseline = calculate_metrics(
            after_tax_zero,
            turnover=turnover_zero,
            weights=weights.sum(axis=1),
            annualization=config.backtest.annualization,
        )
        metrics.update(
            {
                "name": name,
                "strategy": "cash_yield_candidate_comparison",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "transaction_cost_bps": args.transaction_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "short_term_tax_rate": args.short_term_tax_rate,
                        "cash_interest_tax_rate": cash_interest_tax_rate,
                        "cash_annual_yield": args.cash_annual_yield,
                        "base_params": params,
                    },
                    sort_keys=True,
                ),
                "zero_cash_annualized_return": baseline["annualized_return"],
                "zero_cash_sharpe_ratio": baseline["sharpe_ratio"],
                "zero_cash_max_drawdown": baseline["max_drawdown"],
                "annualized_return_delta_vs_zero_cash": metrics["annualized_return"]
                - baseline["annualized_return"],
                "sharpe_delta_vs_zero_cash": metrics["sharpe_ratio"] - baseline["sharpe_ratio"],
                "max_drawdown_delta_vs_zero_cash": metrics["max_drawdown"] - baseline["max_drawdown"],
                "average_cash_weight": float(cash_weight.mean()),
                "gross_cash_interest_pct_initial_capital": float(cash_interest.sum()),
                "tax_paid_pct_initial_capital_with_cash": float(taxes_cash.sum()),
                "tax_paid_pct_initial_capital_zero_cash": float(taxes_zero.sum()),
                **meta_by_name[name],
            }
        )
        _add_dd_counts(metrics, after_tax_cash)
        metric_rows.append(metrics)
        returns_with_cash_by_name[name] = after_tax_cash
        returns_zero_cash_by_name[name] = after_tax_zero
        weights_by_name[name] = weights.sum(axis=1)
        cash_interest_by_name[name] = cash_interest
        cash_weight_by_name[name] = cash_weight

    metrics = metrics_to_frame(metric_rows)
    compact_cols = [
        "name",
        "annualized_return",
        "zero_cash_annualized_return",
        "annualized_return_delta_vs_zero_cash",
        "sharpe_ratio",
        "zero_cash_sharpe_ratio",
        "max_drawdown",
        "zero_cash_max_drawdown",
        "number_of_trades",
        "exposure_percentage",
        "average_cash_weight",
        "dd_episodes_gt_20_30_40_50pct",
        "gross_cash_interest_pct_initial_capital",
        "tax_paid_pct_initial_capital_with_cash",
    ]
    compact = metrics[compact_cols].sort_values("annualized_return", ascending=False)

    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)
    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    compact_path = tables_dir / f"{args.output_prefix}_compact.csv"
    returns_cash_path = tables_dir / f"{args.output_prefix}_after_tax_returns_with_cash_yield.csv"
    returns_zero_path = tables_dir / f"{args.output_prefix}_after_tax_returns_zero_cash.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    cash_interest_path = tables_dir / f"{args.output_prefix}_cash_interest.csv"
    cash_weight_path = tables_dir / f"{args.output_prefix}_cash_weights.csv"
    plot_path = figures_dir / f"{args.output_prefix}_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    compact.to_csv(compact_path, index=False)
    pd.DataFrame(returns_with_cash_by_name).to_csv(returns_cash_path)
    pd.DataFrame(returns_zero_cash_by_name).to_csv(returns_zero_path)
    pd.DataFrame(weights_by_name).to_csv(weights_path)
    pd.DataFrame(cash_interest_by_name).to_csv(cash_interest_path)
    pd.DataFrame(cash_weight_by_name).to_csv(cash_weight_path)
    _plot(returns_with_cash_by_name, plot_path)

    print(f"Metrics saved to {metrics_path}")
    print(f"Compact table saved to {compact_path}")
    print(f"After-tax returns with cash yield saved to {returns_cash_path}")
    print(f"Zero-cash returns saved to {returns_zero_path}")
    print(f"Weights saved to {weights_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
