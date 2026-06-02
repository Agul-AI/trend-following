#!/usr/bin/env python
"""Recalculate synthetic TQQQ no-VIX strategy with slippage and short-term taxes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.metrics import calculate_metrics, metrics_to_frame
from trend_following.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="reports/tables/synthetic_tqqq_fast_slow_qqq_regime_e20_x200_c2c3_weights.csv",
        help="CSV of executable strategy weights.",
    )
    parser.add_argument("--weight-column", default="daily_200_gate_weight")
    parser.add_argument(
        "--target-price",
        default="data/raw/synthetic_3x_60min/QQQ_3X_CALC.parquet",
        help="Synthetic TQQQ hourly parquet.",
    )
    parser.add_argument(
        "--benchmark-price",
        default="data/raw/alpha_vantage_60min/QQQ.parquet",
        help="QQQ hourly parquet for benchmark.",
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="One-way slippage assumption. Default 5 bps for conservative liquid-ETF fills.",
    )
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--annualization", type=int, default=1512)
    parser.add_argument(
        "--output-prefix",
        default="tqqq_no_vix_tax_slippage",
        help="Prefix for output tables and figures.",
    )
    return parser.parse_args()


def _load_price(path: Path, column_name: str) -> pd.Series:
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frame = frame.set_index("date", drop=True)
    else:
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame["adj_close"].astype(float).sort_index().rename(column_name)


def _equity_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _drawdown_from_returns(returns: pd.Series) -> pd.Series:
    equity = _equity_from_returns(returns)
    return equity / equity.cummax() - 1.0


def after_tax_returns_annual_net(
    asset_returns: pd.Series,
    weights: pd.Series,
    *,
    transaction_cost_bps: float,
    slippage_bps: float,
    tax_rate: float,
    liquidate_at_end: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute after-tax returns for a long/cash strategy.

    Assumptions are intentionally explicit and simple:

    * One-way transaction cost + slippage is charged on absolute position changes.
    * Realized trade gains/losses are netted by calendar year.
    * Positive annual net realized gains are taxed at ``tax_rate``.
    * Realized losses carry forward to offset future realized gains.
    * All gains are treated as short-term; no state tax or wash-sale mechanics.
    * If ``liquidate_at_end`` is true, any final open gain/loss is realized on
      the last timestamp for full-sample reporting.
    """
    if not 0.0 <= tax_rate <= 1.0:
        raise ValueError("tax_rate must be between 0 and 1")
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("cost and slippage bps must be non-negative")

    index = asset_returns.index.intersection(weights.index)
    returns = asset_returns.loc[index].fillna(0.0).astype(float)
    desired = weights.loc[index].fillna(0.0).astype(float).clip(lower=0.0, upper=1.0)
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0

    equity = 1.0
    previous_weight = 0.0
    basis: float | None = None
    loss_carryforward = 0.0
    realized_this_year = 0.0

    after_tax_return_values: list[float] = []
    pretax_return_values: list[float] = []
    tax_values: list[float] = []
    turnover_values: list[float] = []

    timestamps = list(index)
    for i, timestamp in enumerate(timestamps):
        weight = float(desired.loc[timestamp])
        asset_return = float(returns.loc[timestamp])
        turnover = abs(weight - previous_weight)
        cost = turnover * cost_rate

        equity_before = equity
        pretax_return = weight * asset_return - cost
        equity *= 1.0 + pretax_return

        entered = previous_weight <= 1e-12 and weight > 1e-12
        exited = previous_weight > 1e-12 and weight <= 1e-12
        if entered:
            basis = equity_before * (1.0 - cost)
        if exited and basis is not None:
            realized_this_year += equity - basis
            basis = None

        next_year_differs = (
            i == len(timestamps) - 1 or timestamps[i + 1].year != timestamp.year
        )
        final_liquidation = liquidate_at_end and i == len(timestamps) - 1 and weight > 1e-12
        if final_liquidation and basis is not None:
            realized_this_year += equity - basis
            basis = None

        tax_paid = 0.0
        if next_year_differs:
            taxable_after_carry = realized_this_year + loss_carryforward
            if taxable_after_carry > 0:
                tax_paid = tax_rate * taxable_after_carry
                equity -= tax_paid
                loss_carryforward = 0.0
            else:
                loss_carryforward = taxable_after_carry
            realized_this_year = 0.0

        after_tax_return_values.append(equity / equity_before - 1.0)
        pretax_return_values.append(pretax_return)
        tax_values.append(tax_paid / equity_before if equity_before > 0 else np.nan)
        turnover_values.append(turnover)
        previous_weight = weight

    return (
        pd.Series(after_tax_return_values, index=index, name="after_tax_return"),
        pd.Series(pretax_return_values, index=index, name="pretax_return"),
        pd.Series(tax_values, index=index, name="tax_drag"),
        pd.Series(turnover_values, index=index, name="turnover"),
    )


def _window_metrics(
    windows: list[tuple[str, str, str]],
    return_streams: dict[str, pd.Series],
    annualization: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, start, end in windows:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        for name, returns in return_streams.items():
            sub = returns.loc[(returns.index >= start_ts) & (returns.index < end_ts)]
            if sub.empty:
                continue
            metrics = calculate_metrics(sub, annualization=annualization)
            metrics.update({"window": label, "series": name})
            rows.append(metrics)
    return pd.DataFrame(rows)


def _plot_side_by_side(
    return_streams: dict[str, pd.Series],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for name, returns in return_streams.items():
        _equity_from_returns(returns).plot(ax=axes[0], label=name, linewidth=1.4)
        _drawdown_from_returns(returns).plot(ax=axes[1], label=name, linewidth=1.4)
    axes[0].set_title("Performance / Equity Curve")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_five_panels(
    windows: list[tuple[str, str, str]],
    return_streams: dict[str, pd.Series],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(len(windows), 2, figsize=(16, 18), sharex=False)
    for row, (label, start, end) in enumerate(windows):
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        ax_eq = axes[row, 0]
        ax_dd = axes[row, 1]
        for name, returns in return_streams.items():
            sub = returns.loc[(returns.index >= start_ts) & (returns.index < end_ts)]
            if sub.empty:
                continue
            _equity_from_returns(sub).plot(ax=ax_eq, label=name, linewidth=1.25)
            _drawdown_from_returns(sub).plot(ax=ax_dd, label=name, linewidth=1.25)
        ax_eq.set_title(f"{label}: growth of $1 within window")
        ax_eq.set_ylabel("Growth")
        ax_eq.grid(True, alpha=0.25)
        ax_dd.set_title(f"{label}: drawdown within window")
        ax_dd.set_ylabel("Drawdown")
        ax_dd.grid(True, alpha=0.25)
        if row == 0:
            ax_eq.legend(loc="upper left", fontsize=8)
            ax_dd.legend(loc="lower left", fontsize=8)
    fig.suptitle(title, y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(".").resolve()
    weights = pd.read_csv(root / args.weights, index_col=0, parse_dates=True)
    if args.weight_column not in weights.columns:
        raise ValueError(f"Missing weight column {args.weight_column!r}")

    target_price = _load_price(root / args.target_price, "QQQ_3X_CALC")
    benchmark_price = _load_price(root / args.benchmark_price, "QQQ")
    index = weights.index.intersection(target_price.index).intersection(benchmark_price.index)
    weight = weights.loc[index, args.weight_column]
    target_returns = target_price.loc[index].pct_change(fill_method=None).fillna(0.0)
    benchmark_returns = benchmark_price.loc[index].pct_change(fill_method=None).fillna(0.0)

    after_tax, pretax, tax_drag, turnover = after_tax_returns_annual_net(
        target_returns,
        weight,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        tax_rate=args.short_term_tax_rate,
    )

    return_streams = {
        "Strategy after 5bp slippage + 24% tax": after_tax,
        "Strategy pretax, 5bp slippage": pretax,
        "Buy & Hold QQQ": benchmark_returns.loc[after_tax.index],
    }

    metric_rows: list[dict[str, object]] = []
    for name, returns in return_streams.items():
        metrics = calculate_metrics(
            returns,
            turnover=turnover if name.startswith("Strategy") else None,
            weights=weight if name.startswith("Strategy") else None,
            annualization=args.annualization,
        )
        metrics.update(
            {
                "name": name,
                "transaction_cost_bps": args.transaction_cost_bps,
                "slippage_bps": args.slippage_bps if name.startswith("Strategy") else 0.0,
                "short_term_tax_rate": args.short_term_tax_rate
                if name.startswith("Strategy after")
                else 0.0,
                "total_tax_paid_pct_initial_capital": float(tax_drag.mul(_equity_from_returns(after_tax).shift(1).fillna(1.0)).sum())
                if name.startswith("Strategy after")
                else 0.0,
            }
        )
        metric_rows.append(metrics)

    tables_dir = root / "reports/tables"
    figures_dir = root / "reports/figures"
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    returns_path = tables_dir / f"{args.output_prefix}_returns.csv"
    window_metrics_path = tables_dir / f"{args.output_prefix}_5_time_panels_metrics.csv"
    side_by_side_path = figures_dir / f"{args.output_prefix}_equity_drawdown_side_by_side.png"
    five_panel_path = figures_dir / f"{args.output_prefix}_5_time_panels_equity_drawdown.png"

    metrics_to_frame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(
        {
            "after_tax_return": after_tax,
            "pretax_return": pretax,
            "tax_drag": tax_drag,
            "turnover": turnover,
            "weight": weight.loc[after_tax.index],
            "benchmark_qqq_return": benchmark_returns.loc[after_tax.index],
        }
    ).to_csv(returns_path)

    windows = [
        ("2000-2004", "2000-01-01", "2005-01-01"),
        ("2005-2009", "2005-01-01", "2010-01-01"),
        ("2010-2014", "2010-01-01", "2015-01-01"),
        ("2015-2019", "2015-01-01", "2020-01-01"),
        ("2020-2026", "2020-01-01", "2027-01-01"),
    ]
    _window_metrics(windows, return_streams, args.annualization).to_csv(
        window_metrics_path,
        index=False,
    )
    title = (
        "Synthetic TQQQ no-VIX strategy with 5bp slippage and 24% short-term tax"
    )
    _plot_side_by_side(return_streams, side_by_side_path, title)
    _plot_five_panels(windows, return_streams, five_panel_path, title)

    print(f"Metrics saved to {metrics_path}")
    print(f"Returns saved to {returns_path}")
    print(f"5-panel metrics saved to {window_metrics_path}")
    print(f"Side-by-side plot saved to {side_by_side_path}")
    print(f"5-panel plot saved to {five_panel_path}")
    print(metrics_to_frame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
