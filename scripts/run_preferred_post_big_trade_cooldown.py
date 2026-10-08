#!/usr/bin/env python
"""Test a post-big-winner stay-out rule for the preferred strategy.

Rule tested:
- Use the usual preferred raw strategy continuously in the background.
- After an *actual* completed round-trip trade whose synthetic-TQQQ peak gain
  exceeded a threshold (default +100%), enter a stay-out/cooldown state.
- During cooldown, do not take new entries, but keep paper-running the usual
  raw strategy.
- End cooldown once either:
    1. a blocked/paper trade has a synthetic-TQQQ loss of at least 20% from its
       paper entry; or
    2. three calendar months have passed from the actual exit.
- After cooldown ends, entries again follow the usual rule.

The filtered raw signal is still converted through executable_weights, so this
keeps the existing no-lookahead timing convention.
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

from run_preferred_profit_lock_comparison import _add_dd_counts, _load_price  # noqa: E402
from run_preferred_profit_lock_stop_exit_comparison import (  # noqa: E402
    PROFIT_LOCK_SCHEME,
    raw_with_peak_drawdown_stop,
)
from run_tqqq_cash_yield_candidate_comparison import (  # noqa: E402
    simulate_after_tax_portfolio_with_cash_yield,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default=TARGET_TICKER)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--big-peak-gain", type=float, default=1.00)
    parser.add_argument("--paper-loss", type=float, default=0.20)
    parser.add_argument("--cooldown-months", type=int, default=3)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--output-prefix", default="preferred_post_big_trade_cooldown")
    return parser.parse_args()


def raw_with_post_big_trade_cooldown(
    raw_desired: pd.Series,
    traded_price: pd.Series,
    *,
    big_peak_gain: float,
    paper_loss: float,
    cooldown_months: int,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Suppress entries after big winning trades until paper loss or time reset."""
    desired = raw_desired.fillna(0.0).astype(float)
    price = traded_price.reindex(desired.index).astype(float)
    if big_peak_gain <= 0:
        raise ValueError("big_peak_gain must be positive")
    if paper_loss <= 0:
        raise ValueError("paper_loss must be positive")
    if cooldown_months <= 0:
        raise ValueError("cooldown_months must be positive")

    in_actual_trade = False
    actual_entry_time: pd.Timestamp | None = None
    actual_entry_price = np.nan
    actual_peak_price = np.nan
    actual_peak_time: pd.Timestamp | None = None

    in_cooldown = False
    cooldown_start: pd.Timestamp | None = None
    cooldown_deadline: pd.Timestamp | None = None
    current_event: dict[str, Any] | None = None

    in_paper_trade = False
    paper_entry_time: pd.Timestamp | None = None
    paper_entry_price = np.nan
    paper_min_return = np.nan

    values: list[float] = []
    actual_peak_gains: list[float] = []
    cooldown_flags: list[bool] = []
    paper_returns: list[float] = []
    release_triggers: list[str] = []
    events: list[dict[str, Any]] = []

    def start_actual_trade(timestamp: pd.Timestamp, current_price: float) -> None:
        nonlocal in_actual_trade, actual_entry_time, actual_entry_price, actual_peak_price, actual_peak_time
        in_actual_trade = True
        actual_entry_time = timestamp
        actual_entry_price = current_price
        actual_peak_price = current_price
        actual_peak_time = timestamp

    def close_actual_trade(timestamp: pd.Timestamp, current_price: float) -> None:
        nonlocal in_actual_trade, actual_entry_time, actual_entry_price, actual_peak_price, actual_peak_time
        nonlocal in_cooldown, cooldown_start, cooldown_deadline, current_event
        peak_gain = actual_peak_price / actual_entry_price - 1.0 if actual_entry_price > 0 else np.nan
        final_gain = current_price / actual_entry_price - 1.0 if actual_entry_price > 0 else np.nan
        should_cooldown = bool(np.isfinite(peak_gain) and peak_gain >= big_peak_gain)
        if should_cooldown:
            in_cooldown = True
            cooldown_start = timestamp
            cooldown_deadline = timestamp + pd.DateOffset(months=cooldown_months)
            current_event = {
                "actual_entry_time": actual_entry_time,
                "actual_exit_time": timestamp,
                "actual_entry_price": actual_entry_price,
                "actual_exit_price": current_price,
                "actual_peak_time": actual_peak_time,
                "actual_peak_price": actual_peak_price,
                "actual_peak_gain": peak_gain,
                "actual_final_gain": final_gain,
                "cooldown_start": cooldown_start,
                "cooldown_deadline": cooldown_deadline,
                "cooldown_end": pd.NaT,
                "release_reason": "",
                "paper_entry_time": pd.NaT,
                "paper_entry_price": np.nan,
                "paper_min_return": np.nan,
                "paper_loss_threshold": -abs(paper_loss),
                "cooldown_months": cooldown_months,
            }
        in_actual_trade = False
        actual_entry_time = None
        actual_entry_price = np.nan
        actual_peak_price = np.nan
        actual_peak_time = None

    def reset_paper() -> None:
        nonlocal in_paper_trade, paper_entry_time, paper_entry_price, paper_min_return
        in_paper_trade = False
        paper_entry_time = None
        paper_entry_price = np.nan
        paper_min_return = np.nan

    def release_cooldown(timestamp: pd.Timestamp, reason: str) -> None:
        nonlocal in_cooldown, cooldown_start, cooldown_deadline, current_event
        if current_event is not None:
            current_event["cooldown_end"] = timestamp
            current_event["release_reason"] = reason
            current_event["paper_entry_time"] = paper_entry_time if paper_entry_time is not None else pd.NaT
            current_event["paper_entry_price"] = paper_entry_price
            current_event["paper_min_return"] = paper_min_return
            events.append(current_event)
        in_cooldown = False
        cooldown_start = None
        cooldown_deadline = None
        current_event = None
        reset_paper()

    for timestamp, desired_signal, current_price in zip(desired.index, desired, price, strict=True):
        timestamp = pd.Timestamp(timestamp)
        current_price = float(current_price) if np.isfinite(current_price) else np.nan
        release_reason = ""
        paper_return = np.nan

        if not np.isfinite(current_price):
            value = 0.0
            values.append(value)
            actual_peak_gains.append(np.nan)
            cooldown_flags.append(in_cooldown)
            paper_returns.append(np.nan)
            release_triggers.append("")
            continue

        if in_cooldown:
            # The background desired strategy keeps running. A desired long state
            # during cooldown becomes a paper trade only, not an actual position.
            if desired_signal > 0.0:
                if not in_paper_trade:
                    in_paper_trade = True
                    paper_entry_time = timestamp
                    paper_entry_price = current_price
                    paper_min_return = 0.0
                paper_return = current_price / paper_entry_price - 1.0 if paper_entry_price > 0 else np.nan
                if np.isfinite(paper_return):
                    paper_min_return = min(float(paper_min_return), float(paper_return))
            else:
                reset_paper()

            time_reset = cooldown_deadline is not None and timestamp >= cooldown_deadline
            paper_loss_reset = bool(np.isfinite(paper_return) and paper_return <= -abs(paper_loss))
            if paper_loss_reset:
                release_reason = "paper_loss"
                release_cooldown(timestamp, release_reason)
            elif time_reset:
                release_reason = "time"
                release_cooldown(timestamp, release_reason)

        if in_cooldown:
            value = 0.0
        else:
            if desired_signal > 0.0:
                if not in_actual_trade:
                    start_actual_trade(timestamp, current_price)
                else:
                    if current_price > actual_peak_price:
                        actual_peak_price = current_price
                        actual_peak_time = timestamp
                value = 1.0
            else:
                if in_actual_trade:
                    close_actual_trade(timestamp, current_price)
                value = 0.0

        peak_gain_now = actual_peak_price / actual_entry_price - 1.0 if in_actual_trade and actual_entry_price > 0 else np.nan
        values.append(value)
        actual_peak_gains.append(peak_gain_now)
        cooldown_flags.append(in_cooldown)
        paper_returns.append(paper_return)
        release_triggers.append(release_reason)

    if current_event is not None:
        events.append(current_event)

    filtered = pd.Series(values, index=desired.index, name=raw_desired.name, dtype=float)
    diagnostics = pd.DataFrame(
        {
            "raw_desired": desired,
            "raw_filtered": filtered,
            "actual_trade_peak_gain_running": actual_peak_gains,
            "in_cooldown": cooldown_flags,
            "paper_trade_return": paper_returns,
            "cooldown_release_trigger": release_triggers,
        },
        index=desired.index,
    )
    events_frame = pd.DataFrame(events)
    return filtered, diagnostics, events_frame


def _plot(returns_by_name: dict[str, pd.Series], output_path: Path) -> None:
    selected = [name for name in returns_by_name if name != "QQQ_BH"]
    if "QQQ_BH" in returns_by_name:
        selected.append("QQQ_BH")
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.8))
    for name in selected:
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
    fig.suptitle("Post-big-winner stay-out rule: paper -20% loss or 3-month reset")
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
    raw_stop40, stop40_diagnostics = raw_with_peak_drawdown_stop(
        raw_base,
        target_prices[args.target_ticker],
        stop_drawdown=0.40,
    )

    cooldown_base, cooldown_base_diag, cooldown_base_events = raw_with_post_big_trade_cooldown(
        raw_base,
        target_prices[args.target_ticker],
        big_peak_gain=args.big_peak_gain,
        paper_loss=args.paper_loss,
        cooldown_months=args.cooldown_months,
    )
    cooldown_stop40, cooldown_stop40_diag, cooldown_stop40_events = raw_with_post_big_trade_cooldown(
        raw_stop40,
        target_prices[args.target_ticker],
        big_peak_gain=args.big_peak_gain,
        paper_loss=args.paper_loss,
        cooldown_months=args.cooldown_months,
    )

    raw_variants: dict[str, tuple[pd.Series, pd.DataFrame, pd.DataFrame | None]] = {
        "profit_lock_300_400_no_cooldown": (raw_base, base_diagnostics, None),
        "profit_lock_300_400_stop_40pct": (raw_stop40, stop40_diagnostics, None),
        "profit_lock_300_400_post_peak100_cooldown": (
            cooldown_base,
            cooldown_base_diag,
            cooldown_base_events,
        ),
        "profit_lock_300_400_stop40_post_peak100_cooldown": (
            cooldown_stop40,
            cooldown_stop40_diag,
            cooldown_stop40_events,
        ),
    }

    metric_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.Series] = {}
    diagnostics_by_name: dict[str, pd.DataFrame] = {}
    events_by_name: dict[str, pd.DataFrame] = {}

    for name, (raw_signal, diag, events) in raw_variants.items():
        raw_weight = trade_profit_lock_tiers(
            raw_signal.rename(args.target_ticker),
            target_prices[args.target_ticker],
            thresholds_to_weights=PROFIT_LOCK_SCHEME,
        ).rename(args.target_ticker)
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
        cooldown_event_count = 0 if events is None or events.empty else int(len(events))
        paper_loss_release_count = (
            0 if events is None or events.empty else int((events["release_reason"] == "paper_loss").sum())
        )
        time_release_count = 0 if events is None or events.empty else int((events["release_reason"] == "time").sum())
        if "stop40" in name or name == "profit_lock_300_400_stop_40pct":
            peak_stop_count = int(stop40_diagnostics.get("stop_trigger", pd.Series(False, index=stop40_diagnostics.index)).sum())
        else:
            peak_stop_count = int(diag.get("stop_trigger", pd.Series(False, index=diag.index)).sum())
        metrics.update(
            {
                "name": name,
                "strategy": "preferred_post_big_trade_cooldown",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "profit_lock_scheme": PROFIT_LOCK_SCHEME,
                        "big_peak_gain": args.big_peak_gain,
                        "paper_loss": args.paper_loss,
                        "cooldown_months": args.cooldown_months,
                        "cash_annual_yield": args.cash_annual_yield,
                        "transaction_cost_bps": args.transaction_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "short_term_tax_rate": args.short_term_tax_rate,
                    },
                    sort_keys=True,
                ),
                "final_return": metrics["cumulative_return"],
                "average_cash_weight": float(cash_weight.mean()),
                "profit_lock_first_threshold_hit_count": float(
                    count_profit_lock_hits(raw_signal, target_prices[args.target_ticker], threshold=3.0)
                ),
                "profit_lock_second_threshold_hit_count": float(
                    count_profit_lock_hits(raw_signal, target_prices[args.target_ticker], threshold=4.0)
                ),
                "cooldown_event_count": cooldown_event_count,
                "paper_loss_release_count": paper_loss_release_count,
                "time_release_count": time_release_count,
                "peak_stop_40_trigger_count": peak_stop_count,
                "tax_paid_pct_initial_capital": float(taxes.sum()),
                "cash_interest_pct_initial_capital": float(cash_interest.sum()),
            }
        )
        metric_rows.append(metrics)
        returns_by_name[name] = returns
        weights_by_name[name] = weights.sum(axis=1)
        diagnostics_by_name[name] = diag
        if events is not None:
            events_by_name[name] = events

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
            "cooldown_event_count": np.nan,
            "paper_loss_release_count": np.nan,
            "time_release_count": np.nan,
            "peak_stop_40_trigger_count": np.nan,
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
        "cooldown_event_count",
        "paper_loss_release_count",
        "time_release_count",
        "peak_stop_40_trigger_count",
    ]
    compact = metrics[compact_cols].sort_values("annualized_return", ascending=False)

    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)
    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    compact_path = tables_dir / f"{args.output_prefix}_compact.csv"
    returns_path = tables_dir / f"{args.output_prefix}_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    diagnostics_path = tables_dir / f"{args.output_prefix}_diagnostics.parquet"
    events_path = tables_dir / f"{args.output_prefix}_events.csv"
    plot_path = figures_dir / f"{args.output_prefix}_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    compact.to_csv(compact_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(weights_by_name).to_csv(weights_path)
    pd.concat(diagnostics_by_name, axis=1).to_parquet(diagnostics_path)
    if events_by_name:
        events_frame = pd.concat(events_by_name, names=["variant", "event_index"]).reset_index(level=0)
        events_frame.to_csv(events_path, index=False)
    _plot(returns_by_name, plot_path)

    print(f"Compact comparison saved to {compact_path}")
    print(f"Events saved to {events_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))
    if events_by_name:
        print("\nCooldown events:")
        print(events_frame.to_string(index=False))


if __name__ == "__main__":
    main()
