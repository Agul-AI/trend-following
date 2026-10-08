#!/usr/bin/env python
"""Test stay-out-after-big-winner rule for the preferred synthetic-TQQQ strategy.

Experimental rule:
- After a completed raw round-trip trade whose synthetic QQQ_3X_CALC peak gain
  exceeded a threshold (default +100%), force the raw signal to cash.
- While staying out, keep running the usual/base strategy as an imaginary paper
  strategy.
- Allow re-entry after either the imaginary strategy suffers a 20% drawdown from
  its post-exit high-water mark, or six calendar months pass.
- Once released, follow the usual rule again.

The stay-out decision is applied to raw close-bar signals and then passed through
profit-lock sizing and the existing executable-weight shift. This keeps the
usual no-lookahead execution convention.
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
    parser.add_argument("--big-winner-peak-gain", type=float, default=1.0)
    parser.add_argument("--imaginary-dd-threshold", type=float, default=0.20)
    parser.add_argument("--max-stay-out-months", type=int, default=6)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--output-prefix", default="preferred_stay_out_after_big_winner")
    return parser.parse_args()


def apply_stay_out_after_big_winner(
    base_raw: pd.Series,
    target_price: pd.Series,
    target_returns: pd.Series,
    *,
    big_winner_peak_gain: float = 1.0,
    imaginary_dd_threshold: float = 0.20,
    max_stay_out_months: int = 6,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Apply the big-winner stay-out rule to a raw 0/1 signal."""
    base = base_raw.fillna(0.0).astype(float).sort_index()
    price = target_price.reindex(base.index).astype(float)
    returns = target_returns.reindex(base.index).fillna(0.0).astype(float)
    threshold = -abs(float(imaginary_dd_threshold))

    in_trade = False
    entry_price = np.nan
    trade_peak_gain = np.nan

    stay_out = False
    stay_out_start: pd.Timestamp | None = None
    stay_out_deadline: pd.Timestamp | None = None
    imaginary_equity = 1.0
    imaginary_peak = 1.0

    raw_values: list[float] = []
    state_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for timestamp, base_signal in base.items():
        current_price = float(price.loc[timestamp]) if np.isfinite(price.loc[timestamp]) else np.nan
        base_signal = float(base_signal)
        release_reason = ""
        start_reason = ""

        # If a prior actual raw trade exits at this timestamp, decide whether to
        # start a stay-out period. The raw exit itself remains cash at this row.
        would_be_in_market = bool(base_signal > 0.0 and not stay_out and np.isfinite(current_price))
        if in_trade and not would_be_in_market:
            event_rows.append(
                {
                    "event": "trade_exit",
                    "timestamp": timestamp,
                    "trade_entry_price": entry_price,
                    "trade_peak_gain": trade_peak_gain,
                    "big_winner": bool(np.isfinite(trade_peak_gain) and trade_peak_gain >= big_winner_peak_gain),
                }
            )
            if np.isfinite(trade_peak_gain) and trade_peak_gain >= big_winner_peak_gain:
                stay_out = True
                stay_out_start = timestamp
                stay_out_deadline = timestamp + pd.DateOffset(months=max_stay_out_months)
                imaginary_equity = 1.0
                imaginary_peak = 1.0
                start_reason = "big_winner_exit"
                event_rows.append(
                    {
                        "event": "stay_out_start",
                        "timestamp": timestamp,
                        "trade_entry_price": entry_price,
                        "trade_peak_gain": trade_peak_gain,
                        "big_winner": True,
                        "deadline": stay_out_deadline,
                    }
                )
            in_trade = False
            entry_price = np.nan
            trade_peak_gain = np.nan

        actual_raw = 0.0 if stay_out else (1.0 if base_signal > 0.0 and np.isfinite(current_price) else 0.0)

        # Start/update actual raw trade state.
        if actual_raw > 0.0:
            if not in_trade:
                in_trade = True
                entry_price = current_price
                trade_peak_gain = 0.0
            if entry_price > 0 and np.isfinite(current_price):
                trade_peak_gain = max(float(trade_peak_gain), current_price / entry_price - 1.0)

        # During stay-out, update imaginary strategy after this bar and release
        # only for subsequent rows. The imaginary strategy follows base_raw.
        imaginary_dd = np.nan
        if stay_out:
            imaginary_return = base_signal * float(returns.loc[timestamp])
            imaginary_equity *= 1.0 + imaginary_return
            imaginary_peak = max(imaginary_peak, imaginary_equity)
            imaginary_dd = imaginary_equity / imaginary_peak - 1.0 if imaginary_peak > 0 else np.nan
            time_release = stay_out_deadline is not None and timestamp >= stay_out_deadline
            dd_release = np.isfinite(imaginary_dd) and imaginary_dd <= threshold
            if dd_release or time_release:
                release_reason = "imaginary_dd" if dd_release else "time_6m"
                event_rows.append(
                    {
                        "event": "stay_out_release",
                        "timestamp": timestamp,
                        "release_reason": release_reason,
                        "stay_out_start": stay_out_start,
                        "deadline": stay_out_deadline,
                        "imaginary_equity": imaginary_equity,
                        "imaginary_peak": imaginary_peak,
                        "imaginary_drawdown": imaginary_dd,
                    }
                )
                stay_out = False
                stay_out_start = None
                stay_out_deadline = None
                imaginary_equity = 1.0
                imaginary_peak = 1.0

        raw_values.append(actual_raw)
        state_rows.append(
            {
                "base_raw": base_signal,
                "actual_raw": actual_raw,
                "stay_out_active": bool(stay_out),
                "start_reason": start_reason,
                "release_reason": release_reason,
                "actual_trade_entry_price": entry_price,
                "actual_trade_peak_gain": trade_peak_gain,
                "imaginary_equity": imaginary_equity if stay_out else np.nan,
                "imaginary_peak": imaginary_peak if stay_out else np.nan,
                "imaginary_drawdown": imaginary_dd,
                "stay_out_deadline": stay_out_deadline,
            }
        )

    raw = pd.Series(raw_values, index=base.index, name=base_raw.name, dtype=float)
    diagnostics = pd.DataFrame(state_rows, index=base.index)
    events = pd.DataFrame(event_rows)
    return raw, diagnostics, events


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
    fig.suptitle("Stay out after >100% peak winner until imaginary -20% DD or 6 months")
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

    bars_per_day = int(dict(config.strategies.regime_switch).get("intraday_bars_per_day", 6))
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

    raw_stop40, diag_stop40 = raw_with_peak_drawdown_stop(
        raw_base,
        target_prices[args.target_ticker],
        stop_drawdown=0.40,
    )
    raw_stayout_no_stop, diag_stayout_no_stop, events_no_stop = apply_stay_out_after_big_winner(
        raw_base,
        target_prices[args.target_ticker],
        target_returns[args.target_ticker],
        big_winner_peak_gain=args.big_winner_peak_gain,
        imaginary_dd_threshold=args.imaginary_dd_threshold,
        max_stay_out_months=args.max_stay_out_months,
    )
    raw_stayout_stop40, diag_stayout_stop40, events_stop40 = apply_stay_out_after_big_winner(
        raw_stop40,
        target_prices[args.target_ticker],
        target_returns[args.target_ticker],
        big_winner_peak_gain=args.big_winner_peak_gain,
        imaginary_dd_threshold=args.imaginary_dd_threshold,
        max_stay_out_months=args.max_stay_out_months,
    )

    raw_variants: dict[str, tuple[pd.Series, pd.DataFrame, pd.DataFrame | None]] = {
        "profit_lock_300_400_no_overlay": (raw_base, base_diagnostics, None),
        "profit_lock_300_400_stop_40pct": (raw_stop40, diag_stop40, None),
        "profit_lock_300_400_stay_out_after_100pct_peak_no_stop": (
            raw_stayout_no_stop,
            diag_stayout_no_stop,
            events_no_stop,
        ),
        "profit_lock_300_400_stay_out_after_100pct_peak_stop40_base": (
            raw_stayout_stop40,
            diag_stayout_stop40,
            events_stop40,
        ),
    }

    metric_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.Series] = {}
    diagnostics_by_name: dict[str, pd.DataFrame] = {}
    event_frames: list[pd.DataFrame] = []

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
        stay_out_starts = 0
        stay_out_releases = 0
        imaginary_releases = 0
        time_releases = 0
        if events is not None and not events.empty:
            ev = events.copy()
            ev.insert(0, "variant", name)
            event_frames.append(ev)
            stay_out_starts = int(ev["event"].eq("stay_out_start").sum())
            releases = ev[ev["event"].eq("stay_out_release")]
            stay_out_releases = int(len(releases))
            imaginary_releases = int(releases.get("release_reason", pd.Series(dtype=object)).eq("imaginary_dd").sum())
            time_releases = int(releases.get("release_reason", pd.Series(dtype=object)).eq("time_6m").sum())
        metrics.update(
            {
                "name": name,
                "strategy": "preferred_stay_out_after_big_winner",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "profit_lock_scheme": PROFIT_LOCK_SCHEME,
                        "big_winner_peak_gain": args.big_winner_peak_gain,
                        "imaginary_dd_threshold": args.imaginary_dd_threshold,
                        "max_stay_out_months": args.max_stay_out_months,
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
                "stay_out_start_count": stay_out_starts,
                "stay_out_release_count": stay_out_releases,
                "imaginary_dd_release_count": imaginary_releases,
                "time_6m_release_count": time_releases,
                "tax_paid_pct_initial_capital": float(taxes.sum()),
                "cash_interest_pct_initial_capital": float(cash_interest.sum()),
            }
        )
        metric_rows.append(metrics)
        returns_by_name[name] = returns
        weights_by_name[name] = weights.sum(axis=1)
        diagnostics_by_name[name] = diag

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
            "stay_out_start_count": np.nan,
            "stay_out_release_count": np.nan,
            "imaginary_dd_release_count": np.nan,
            "time_6m_release_count": np.nan,
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
        "stay_out_start_count",
        "imaginary_dd_release_count",
        "time_6m_release_count",
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
    if event_frames:
        pd.concat(event_frames, ignore_index=True, sort=False).to_csv(events_path, index=False)
    _plot(returns_by_name, plot_path)

    print(f"Compact comparison saved to {compact_path}")
    print(f"Events saved to {events_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
