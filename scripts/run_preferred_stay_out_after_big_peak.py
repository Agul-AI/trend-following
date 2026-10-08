#!/usr/bin/env python
"""Test a stay-out overlay after large winning trades.

Base preferred rule:
- QQQ hourly MACD histogram > 0 entry.
- QQQ hourly close > QQQ hourly 200-day MA entry gate.
- QQQ hourly close < QQQ hourly 200-day MA base exit.
- Synthetic QQQ_3X_CALC exposure.
- Profit lock: +300% unrealized trade gain -> 75%; +400% -> 50%.
- Peak stop: synthetic QQQ_3X_CALC -40% from current trade peak.

Experimental overlay:
- After an actual raw round-trip exits, if its maximum unrealized synthetic
  QQQ_3X_CALC gain exceeded +100%, activate a stay-out mode.
- While stay-out mode is active, continue running the base preferred raw signal
  as an "imaginary" strategy, but keep actual exposure at cash.
- When the imaginary base trade suffers a -10% drawdown from its imaginary
  trade peak, allow actual entry again. From that point onward, behavior returns
  to normal until the next large-winning round trip exits.

No-lookahead convention:
- The stay-out release is observed at a bar close. The resulting raw exposure is
  still passed through executable_weights, so it cannot earn same-bar returns.
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
    parser.add_argument("--imaginary-dd-entry", type=float, default=0.10)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--output-prefix", default="preferred_stay_out_after_big_peak")
    return parser.parse_args()


def raw_with_stay_out_after_big_peak(
    base_raw: pd.Series,
    traded_price: pd.Series,
    *,
    big_peak_gain: float,
    imaginary_dd_entry: float,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Apply a post-big-winner stay-out overlay to a 0/1 raw signal.

    The overlay is intentionally raw-signal-level. The returned raw signal must
    still be passed through the project's executable-weight shift.
    """
    base = base_raw.fillna(0.0).astype(float)
    price = traded_price.reindex(base.index).astype(float)
    dd_threshold = -abs(float(imaginary_dd_entry))

    stay_out = False
    actual_in_trade = False
    actual_entry_price = np.nan
    actual_peak = np.nan
    actual_max_gain = np.nan

    imaginary_in_trade = False
    imaginary_entry_price = np.nan
    imaginary_peak = np.nan

    values: list[float] = []
    stay_out_values: list[bool] = []
    imaginary_dd_values: list[float] = []
    actual_gain_values: list[float] = []
    activation_events: list[bool] = []
    release_events: list[bool] = []
    skipped_base_trade_values: list[bool] = []
    trade_records: list[dict[str, Any]] = []
    pending_actual_trade: dict[str, Any] | None = None
    pending_imaginary: dict[str, Any] | None = None

    for timestamp, base_signal in base.items():
        current_price = float(price.loc[timestamp]) if timestamp in price.index and np.isfinite(price.loc[timestamp]) else np.nan
        activation_event = False
        release_event = False
        imaginary_dd = np.nan
        actual_gain = np.nan
        skipped_base_trade = False
        output = 0.0

        if not np.isfinite(current_price):
            # Missing price: stay in cash and do not update state.
            values.append(0.0)
            stay_out_values.append(stay_out)
            imaginary_dd_values.append(np.nan)
            actual_gain_values.append(np.nan)
            activation_events.append(False)
            release_events.append(False)
            skipped_base_trade_values.append(False)
            continue

        if stay_out:
            # Actual exposure is cash. Run the base signal as an imaginary trade.
            if base_signal > 0.0:
                skipped_base_trade = True
                if not imaginary_in_trade:
                    imaginary_in_trade = True
                    imaginary_entry_price = current_price
                    imaginary_peak = current_price
                    pending_imaginary = {
                        "imaginary_start": timestamp,
                        "imaginary_entry_price": current_price,
                        "release_timestamp": pd.NaT,
                        "release_price": np.nan,
                        "max_imaginary_gain_pct": 0.0,
                        "min_imaginary_dd_pct": 0.0,
                    }
                else:
                    imaginary_peak = max(float(imaginary_peak), current_price)
                imaginary_gain = current_price / imaginary_entry_price - 1.0 if imaginary_entry_price > 0 else np.nan
                imaginary_dd = current_price / imaginary_peak - 1.0 if imaginary_peak > 0 else np.nan
                if pending_imaginary is not None and np.isfinite(imaginary_gain):
                    pending_imaginary["max_imaginary_gain_pct"] = max(
                        float(pending_imaginary["max_imaginary_gain_pct"]), imaginary_gain * 100.0
                    )
                if pending_imaginary is not None and np.isfinite(imaginary_dd):
                    pending_imaginary["min_imaginary_dd_pct"] = min(
                        float(pending_imaginary["min_imaginary_dd_pct"]), imaginary_dd * 100.0
                    )
                if np.isfinite(imaginary_dd) and imaginary_dd <= dd_threshold:
                    # Release stay-out and start an actual trade at this raw bar.
                    stay_out = False
                    imaginary_in_trade = False
                    release_event = True
                    output = 1.0
                    if pending_imaginary is not None:
                        pending_imaginary["release_timestamp"] = timestamp
                        pending_imaginary["release_price"] = current_price
                        trade_records.append({"record_type": "stay_out_release", **pending_imaginary})
                        pending_imaginary = None
                    actual_in_trade = True
                    actual_entry_price = current_price
                    actual_peak = current_price
                    actual_max_gain = 0.0
                    pending_actual_trade = {
                        "record_type": "actual_trade",
                        "entry_timestamp": timestamp,
                        "entry_price": current_price,
                        "exit_timestamp": pd.NaT,
                        "exit_price": np.nan,
                        "max_gain_pct": 0.0,
                        "triggered_stay_out_after_exit": False,
                    }
                else:
                    output = 0.0
            else:
                # No imaginary base trade right now. Keep waiting.
                if imaginary_in_trade and pending_imaginary is not None:
                    pending_imaginary["base_trade_ended_before_release"] = timestamp
                    trade_records.append({"record_type": "stay_out_unreleased", **pending_imaginary})
                imaginary_in_trade = False
                imaginary_entry_price = np.nan
                imaginary_peak = np.nan
                pending_imaginary = None
                output = 0.0
        else:
            if base_signal > 0.0:
                output = 1.0
                if not actual_in_trade:
                    actual_in_trade = True
                    actual_entry_price = current_price
                    actual_peak = current_price
                    actual_max_gain = 0.0
                    pending_actual_trade = {
                        "record_type": "actual_trade",
                        "entry_timestamp": timestamp,
                        "entry_price": current_price,
                        "exit_timestamp": pd.NaT,
                        "exit_price": np.nan,
                        "max_gain_pct": 0.0,
                        "triggered_stay_out_after_exit": False,
                    }
                else:
                    actual_peak = max(float(actual_peak), current_price)
                actual_gain = current_price / actual_entry_price - 1.0 if actual_entry_price > 0 else np.nan
                actual_max_gain = max(float(actual_max_gain), float(actual_gain)) if np.isfinite(actual_gain) else actual_max_gain
                if pending_actual_trade is not None and np.isfinite(actual_gain):
                    pending_actual_trade["max_gain_pct"] = max(
                        float(pending_actual_trade["max_gain_pct"]), actual_gain * 100.0
                    )
            else:
                output = 0.0
                if actual_in_trade:
                    if np.isfinite(actual_max_gain) and actual_max_gain >= big_peak_gain:
                        stay_out = True
                        activation_event = True
                    if pending_actual_trade is not None:
                        pending_actual_trade["exit_timestamp"] = timestamp
                        pending_actual_trade["exit_price"] = current_price
                        pending_actual_trade["triggered_stay_out_after_exit"] = bool(activation_event)
                        trade_records.append(pending_actual_trade)
                    actual_in_trade = False
                    actual_entry_price = np.nan
                    actual_peak = np.nan
                    actual_max_gain = np.nan
                    pending_actual_trade = None

        values.append(output)
        stay_out_values.append(stay_out)
        imaginary_dd_values.append(imaginary_dd)
        actual_gain_values.append(actual_gain)
        activation_events.append(activation_event)
        release_events.append(release_event)
        skipped_base_trade_values.append(skipped_base_trade and output == 0.0)

    if actual_in_trade and pending_actual_trade is not None:
        trade_records.append(pending_actual_trade)
    if imaginary_in_trade and pending_imaginary is not None:
        trade_records.append({"record_type": "stay_out_unreleased", **pending_imaginary})

    raw = pd.Series(values, index=base.index, name=base_raw.name, dtype=float)
    diagnostics = pd.DataFrame(
        {
            "base_raw": base,
            "stayed_out_raw": raw,
            "stay_out_active_after_bar": stay_out_values,
            "imaginary_trade_drawdown": imaginary_dd_values,
            "actual_trade_gain": actual_gain_values,
            "stay_out_activation_event": activation_events,
            "stay_out_release_event": release_events,
            "skipped_base_trade_bar": skipped_base_trade_values,
        },
        index=base.index,
    )
    records = pd.DataFrame(trade_records)
    return raw, diagnostics, records


def _plot(returns_by_name: dict[str, pd.Series], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.8))
    for name, returns in returns_by_name.items():
        _equity(returns).plot(ax=axes[0], label=name, linewidth=1.1)
        _drawdown(returns).plot(ax=axes[1], label=name, linewidth=1.1)
    axes[0].set_title("Equity / growth of $1")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("Stay-out overlay after >100% peak winning trades")
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
    raw_preferred, preferred_stop_diag = raw_with_peak_drawdown_stop(
        raw_base,
        target_prices[args.target_ticker],
        stop_drawdown=0.40,
    )
    raw_stay_out, stay_diag, stay_records = raw_with_stay_out_after_big_peak(
        raw_preferred.rename(args.target_ticker),
        target_prices[args.target_ticker],
        big_peak_gain=args.big_peak_gain,
        imaginary_dd_entry=args.imaginary_dd_entry,
    )

    raw_variants: dict[str, tuple[pd.Series, pd.DataFrame]] = {
        "preferred_profit_lock_300_400_stop_40pct": (raw_preferred, preferred_stop_diag),
        "stay_out_after_100pct_peak_wait_10pct_imaginary_dd": (raw_stay_out, stay_diag),
    }

    metric_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.Series] = {}
    diagnostics_by_name: dict[str, pd.DataFrame] = {}

    for name, (raw_signal, diag) in raw_variants.items():
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
        metrics.update(
            {
                "name": name,
                "strategy": "preferred_stay_out_after_big_peak",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "profit_lock_scheme": PROFIT_LOCK_SCHEME,
                        "base_peak_stop": 0.40,
                        "big_peak_gain": args.big_peak_gain,
                        "imaginary_dd_entry": args.imaginary_dd_entry,
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
                "stay_out_activation_count": int(diag.get("stay_out_activation_event", pd.Series(False, index=diag.index)).sum()),
                "stay_out_release_count": int(diag.get("stay_out_release_event", pd.Series(False, index=diag.index)).sum()),
                "skipped_base_trade_bars": int(diag.get("skipped_base_trade_bar", pd.Series(False, index=diag.index)).sum()),
                "peak_stop_40_trigger_count": int(diag.get("stop_trigger", pd.Series(False, index=diag.index)).sum()),
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
            "stay_out_activation_count": np.nan,
            "stay_out_release_count": np.nan,
            "skipped_base_trade_bars": np.nan,
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
        "stay_out_activation_count",
        "stay_out_release_count",
        "skipped_base_trade_bars",
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
    records_path = tables_dir / f"{args.output_prefix}_stay_out_records.csv"
    plot_path = figures_dir / f"{args.output_prefix}_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    compact.to_csv(compact_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(weights_by_name).to_csv(weights_path)
    pd.concat(diagnostics_by_name, axis=1).to_parquet(diagnostics_path)
    stay_records.to_csv(records_path, index=False)
    _plot(returns_by_name, plot_path)

    print(f"Compact comparison saved to {compact_path}")
    print(f"Stay-out records saved to {records_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))
    if not stay_records.empty:
        print("\nStay-out records:")
        print(stay_records.to_string(index=False))


if __name__ == "__main__":
    main()
