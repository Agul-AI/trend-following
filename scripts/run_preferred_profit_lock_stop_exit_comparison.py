#!/usr/bin/env python
"""Test drawdown stop exits with the preferred +300/+400 profit-lock overlay.

Base raw signal:
- QQQ hourly MACD histogram > 0 entry.
- QQQ hourly close > QQQ hourly 200-day MA entry gate.
- QQQ hourly close < QQQ hourly 200-day MA exit.
- No daily regime gate.

Overlay:
- Profit lock: +300% synthetic-TQQQ trade gain -> 75%; +400% -> 50%.
- Additional stop exits: if synthetic TQQQ falls by a configured percentage from
  the current trade's peak, force raw state to cash. The raw exit is generated
  at the bar where the stop is observed, then shifted by executable_weights, so
  it cannot earn same-bar returns.
"""

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

from run_preferred_profit_lock_comparison import (  # noqa: E402
    _add_dd_counts,
    _load_price,
)
from run_tqqq_cash_yield_candidate_comparison import (
    simulate_after_tax_portfolio_with_cash_yield,  # noqa: E402
)
from run_tqqq_daily_gate_ablation import no_daily_gate_hourly_ma_gate_signal  # noqa: E402
from run_tqqq_entry_signal_comparison import (  # noqa: E402
    _drawdown,
    _equity,
    _returns_from_prices,
    executable_weights,
)
from run_tqqq_macd_entry_experiments import count_profit_lock_hits  # noqa: E402
from run_tqqq_tiered_sizing_experiments import trade_profit_lock_tiers  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.metrics import calculate_metrics, metrics_to_frame  # noqa: E402
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

TARGET_TICKER = "QQQ_3X_CALC"
BENCHMARK_TICKER = "QQQ"
PROFIT_LOCK_SCHEME = [(3.00, 0.75), (4.00, 0.50)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default=TARGET_TICKER)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--output-prefix", default="preferred_profit_lock_stop_exit_comparison")
    return parser.parse_args()


def raw_with_peak_drawdown_stop(
    base_raw: pd.Series,
    traded_price: pd.Series,
    *,
    stop_drawdown: float | None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Force raw signal to cash if trade-level peak drawdown breaches threshold."""
    base = base_raw.fillna(0.0).astype(float)
    price = traded_price.reindex(base.index).astype(float)
    if stop_drawdown is None:
        diagnostics = pd.DataFrame(
            {
                "base_raw": base,
                "stopped_raw": base,
                "trade_peak_price": np.nan,
                "trade_peak_drawdown": np.nan,
                "stop_trigger": False,
            },
            index=base.index,
        )
        return base.rename(base_raw.name), diagnostics
    threshold = -abs(float(stop_drawdown))
    in_trade = False
    stopped_until_base_exit = False
    peak = np.nan
    values: list[float] = []
    peaks: list[float] = []
    drawdowns: list[float] = []
    triggers: list[bool] = []

    for base_signal, current_price in zip(base, price, strict=True):
        current_price = float(current_price) if np.isfinite(current_price) else np.nan
        trigger = False
        if base_signal <= 0.0 or not np.isfinite(current_price):
            in_trade = False
            stopped_until_base_exit = False
            peak = np.nan
            value = 0.0
            drawdown = np.nan
        else:
            if not in_trade:
                in_trade = True
                stopped_until_base_exit = False
                peak = current_price
            else:
                peak = max(float(peak), current_price)
            drawdown = current_price / peak - 1.0 if peak > 0 else np.nan
            if stopped_until_base_exit:
                value = 0.0
            elif np.isfinite(drawdown) and drawdown <= threshold:
                trigger = True
                stopped_until_base_exit = True
                value = 0.0
            else:
                value = 1.0
        values.append(value)
        peaks.append(peak)
        drawdowns.append(drawdown)
        triggers.append(trigger)

    stopped = pd.Series(values, index=base.index, name=base_raw.name, dtype=float)
    diagnostics = pd.DataFrame(
        {
            "base_raw": base,
            "stopped_raw": stopped,
            "trade_peak_price": peaks,
            "trade_peak_drawdown": drawdowns,
            "stop_trigger": triggers,
        },
        index=base.index,
    )
    return stopped, diagnostics


def profit_lock_hit_periods(base_raw: pd.Series, target_price: pd.Series, common: pd.DatetimeIndex) -> pd.DataFrame:
    """Return raw trade periods where +300% or +400% profit-lock thresholds hit."""
    records: list[dict[str, Any]] = []
    in_trade = False
    current: dict[str, Any] | None = None
    entry_price = np.nan
    for timestamp, signal in base_raw.fillna(0.0).items():
        price = float(target_price.loc[timestamp]) if timestamp in target_price.index else np.nan
        if not in_trade and signal > 0.0 and np.isfinite(price):
            in_trade = True
            entry_price = price
            current = {
                "entry_raw": timestamp,
                "entry_price": price,
                "hit_300_raw": pd.NaT,
                "hit_300_price": np.nan,
                "hit_400_raw": pd.NaT,
                "hit_400_price": np.nan,
                "max_gain_ts": timestamp,
                "max_gain_pct": 0.0,
                "exit_raw": pd.NaT,
                "exit_price": np.nan,
            }
        if in_trade and current is not None and np.isfinite(price):
            gain = price / entry_price - 1.0 if entry_price > 0 else np.nan
            if np.isfinite(gain) and gain * 100.0 > float(current["max_gain_pct"]):
                current["max_gain_pct"] = gain * 100.0
                current["max_gain_ts"] = timestamp
            if np.isfinite(gain) and pd.isna(current["hit_300_raw"]) and gain >= 3.0:
                current["hit_300_raw"] = timestamp
                current["hit_300_price"] = price
            if np.isfinite(gain) and pd.isna(current["hit_400_raw"]) and gain >= 4.0:
                current["hit_400_raw"] = timestamp
                current["hit_400_price"] = price
        if in_trade and signal <= 0.0 and current is not None:
            current["exit_raw"] = timestamp
            current["exit_price"] = price
            records.append(current)
            in_trade = False
            current = None
            entry_price = np.nan
    if in_trade and current is not None:
        records.append(current)

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    hits = frame[frame["hit_300_raw"].notna()].copy()
    for column in ["entry_raw", "hit_300_raw", "hit_400_raw", "max_gain_ts", "exit_raw"]:
        hits[column] = pd.to_datetime(hits[column])
    hits["entry_to_hit_300_calendar_days"] = (
        hits["hit_300_raw"].dt.normalize() - hits["entry_raw"].dt.normalize()
    ).dt.days
    hits["entry_to_hit_300_trading_days"] = [
        len(pd.Index(common[(common >= row.entry_raw) & (common <= row.hit_300_raw)].normalize().unique()))
        for row in hits.itertuples()
    ]
    hits["hit_300_to_exit_trading_days"] = [
        None
        if pd.isna(row.exit_raw)
        else len(pd.Index(common[(common >= row.hit_300_raw) & (common <= row.exit_raw)].normalize().unique()))
        for row in hits.itertuples()
    ]
    hits["gain_at_300_hit_pct"] = (hits["hit_300_price"] / hits["entry_price"] - 1.0) * 100.0
    hits["gain_at_400_hit_pct"] = (hits["hit_400_price"] / hits["entry_price"] - 1.0) * 100.0
    hits["hit_400"] = hits["hit_400_raw"].notna()
    return hits


def _plot_selected(returns_by_name: dict[str, pd.Series], output_path: Path) -> None:
    selected = [
        "profit_lock_300_400_no_stop",
        "profit_lock_300_400_stop_30pct",
        "profit_lock_300_400_stop_35pct",
        "profit_lock_300_400_stop_40pct",
        "profit_lock_300_400_stop_45pct",
        "profit_lock_300_400_stop_50pct",
        "base_no_lock_no_stop",
        "QQQ_BH",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.7))
    for name in selected:
        if name not in returns_by_name:
            continue
        _equity(returns_by_name[name]).plot(ax=axes[0], label=name, linewidth=1.05)
        _drawdown(returns_by_name[name]).plot(ax=axes[1], label=name, linewidth=1.05)
    axes[0].set_title("Equity / growth of $1")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("Preferred +300/+400 profit lock with additional peak-drawdown exits")
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    target_dir = resolve_path(config.root, args.target_raw_dir)
    benchmark_dir = resolve_path(config.root, args.benchmark_raw_dir)

    target = _load_price(target_dir / f"{args.target_ticker}.parquet", args.target_ticker)
    qqq = _load_price(benchmark_dir / f"{args.benchmark_ticker}.parquet", args.benchmark_ticker)
    common = target.index.intersection(qqq.index)
    target_prices = target.loc[common].to_frame()
    qqq_prices = qqq.loc[common].to_frame()
    target_returns = _returns_from_prices(target_prices)
    qqq_returns = _returns_from_prices(qqq_prices)[args.benchmark_ticker]

    params = dict(config.strategies.regime_switch)
    bars_per_day = int(params.get("intraday_bars_per_day", 6))
    raw_base, base_diagnostics = no_daily_gate_hourly_ma_gate_signal(
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
    raw_base = raw_base.rename(args.target_ticker)

    schemes: dict[str, tuple[pd.Series, str, float | None]] = {
        "base_no_lock_no_stop": (raw_base, "none", None),
        "profit_lock_300_400_no_stop": (raw_base, "300_400", None),
    }
    for stop in (0.30, 0.35, 0.40, 0.45, 0.50):
        stopped_raw, _ = raw_with_peak_drawdown_stop(raw_base, target_prices[args.target_ticker], stop_drawdown=stop)
        schemes[f"profit_lock_300_400_stop_{int(stop * 100)}pct"] = (stopped_raw, "300_400", stop)

    metric_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.Series] = {}
    diagnostics_by_name: dict[str, pd.DataFrame] = {"base": base_diagnostics}

    for name, (raw_signal, lock_label, stop) in schemes.items():
        if lock_label == "300_400":
            raw_weight = trade_profit_lock_tiers(
                raw_signal.rename(args.target_ticker),
                target_prices[args.target_ticker],
                thresholds_to_weights=PROFIT_LOCK_SCHEME,
            ).rename(args.target_ticker)
        else:
            raw_weight = raw_signal.rename(args.target_ticker).astype(float)
        weights = executable_weights(raw_weight.to_frame(args.target_ticker), config=config).reindex(common).fillna(0.0)
        returns, taxes, turnover, cash_interest, cash_weight = simulate_after_tax_portfolio_with_cash_yield(
            target_returns[[args.target_ticker]],
            weights[[args.target_ticker]],
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            tax_rate=args.short_term_tax_rate,
            cash_annual_yield=args.cash_annual_yield,
            annualization=config.backtest.annualization,
            cash_interest_tax_rate=args.cash_interest_tax_rate,
        )
        metrics = calculate_metrics(
            returns,
            turnover=turnover,
            weights=weights.sum(axis=1),
            annualization=config.backtest.annualization,
        )
        _add_dd_counts(metrics, returns)
        if stop is not None:
            _, stop_diag = raw_with_peak_drawdown_stop(raw_base, target_prices[args.target_ticker], stop_drawdown=stop)
            diagnostics_by_name[name] = stop_diag
            stop_triggers = int(stop_diag["stop_trigger"].sum())
        else:
            stop_triggers = 0
        metrics.update(
            {
                "name": name,
                "strategy": "preferred_profit_lock_stop_exit",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "profit_lock_scheme": PROFIT_LOCK_SCHEME if lock_label == "300_400" else [],
                        "stop_drawdown": stop,
                        "cash_annual_yield": args.cash_annual_yield,
                        "transaction_cost_bps": args.transaction_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "short_term_tax_rate": args.short_term_tax_rate,
                    },
                    sort_keys=True,
                ),
                "final_return": metrics["cumulative_return"],
                "average_cash_weight": float(cash_weight.mean()),
                "profit_lock_first_threshold_hit_count": (
                    float(count_profit_lock_hits(raw_signal, target_prices[args.target_ticker], threshold=3.0))
                    if lock_label == "300_400"
                    else np.nan
                ),
                "profit_lock_second_threshold_hit_count": (
                    float(count_profit_lock_hits(raw_signal, target_prices[args.target_ticker], threshold=4.0))
                    if lock_label == "300_400"
                    else np.nan
                ),
                "stop_drawdown_threshold": stop,
                "stop_trigger_count": stop_triggers,
                "tax_paid_pct_initial_capital": float(taxes.sum()),
                "cash_interest_pct_initial_capital": float(cash_interest.sum()),
            }
        )
        metric_rows.append(metrics)
        returns_by_name[name] = returns
        weights_by_name[name] = weights.sum(axis=1)

    benchmark_metrics = calculate_metrics(
        qqq_returns,
        weights=pd.Series(1.0, index=qqq_returns.index),
        annualization=config.backtest.annualization,
    )
    _add_dd_counts(benchmark_metrics, qqq_returns)
    benchmark_metrics.update(
        {
            "name": "QQQ_BH",
            "strategy": "benchmark",
            "segment": "full_sample",
            "parameters": "{}",
            "final_return": benchmark_metrics["cumulative_return"],
            "average_cash_weight": 0.0,
            "profit_lock_first_threshold_hit_count": np.nan,
            "profit_lock_second_threshold_hit_count": np.nan,
            "stop_drawdown_threshold": np.nan,
            "stop_trigger_count": 0,
            "tax_paid_pct_initial_capital": 0.0,
            "cash_interest_pct_initial_capital": 0.0,
        }
    )
    metric_rows.append(benchmark_metrics)
    returns_by_name["QQQ_BH"] = qqq_returns

    metrics = metrics_to_frame(metric_rows)
    compact_cols = [
        "name",
        "final_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "number_of_trades",
        "exposure_percentage",
        "average_cash_weight",
        "dd_episodes_gt_20_30_40_50pct",
        "profit_lock_first_threshold_hit_count",
        "profit_lock_second_threshold_hit_count",
        "stop_drawdown_threshold",
        "stop_trigger_count",
    ]
    compact = metrics[compact_cols].sort_values("annualized_return", ascending=False)

    hit_periods = profit_lock_hit_periods(raw_base, target_prices[args.target_ticker], common)

    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)
    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    compact_path = tables_dir / f"{args.output_prefix}_compact.csv"
    returns_path = tables_dir / f"{args.output_prefix}_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    diagnostics_path = tables_dir / f"{args.output_prefix}_diagnostics.parquet"
    hit_periods_path = tables_dir / "preferred_profit_lock_300_hit_periods.csv"
    plot_path = figures_dir / f"{args.output_prefix}_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    compact.to_csv(compact_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(weights_by_name).to_csv(weights_path)
    pd.concat(diagnostics_by_name, axis=1).to_parquet(diagnostics_path)
    hit_periods.to_csv(hit_periods_path, index=False)
    _plot_selected(returns_by_name, plot_path)

    print(f"Compact comparison saved to {compact_path}")
    print(f"+300 hit periods saved to {hit_periods_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))
    print("\n+300 profit-lock hit periods:")
    if hit_periods.empty:
        print("No +300% hits")
    else:
        columns = [
            "entry_raw",
            "hit_300_raw",
            "hit_400_raw",
            "exit_raw",
            "entry_to_hit_300_trading_days",
            "hit_300_to_exit_trading_days",
            "max_gain_ts",
            "max_gain_pct",
        ]
        print(hit_periods[columns].to_string(index=False))


if __name__ == "__main__":
    main()
