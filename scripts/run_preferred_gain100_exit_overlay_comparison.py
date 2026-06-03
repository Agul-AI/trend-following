#!/usr/bin/env python
"""Test gain-activated exits for the preferred synthetic-TQQQ strategy.

Base rule:
- QQQ hourly MACD histogram > 0 entry.
- QQQ hourly close > QQQ hourly 200-day MA entry gate.
- QQQ hourly close < QQQ hourly 200-day MA base exit.
- No daily regime gate.
- Synthetic QQQ_3X_CALC exposure.
- Profit lock: +300% unrealized trade gain -> 75%; +400% -> 50%.

Experimental overlay:
After synthetic QQQ_3X_CALC has reached at least +100% unrealized gain in the
current base trade, force raw state to cash using one of:
- QQQ hourly close below its 100-day hourly MA.
- QQQ hourly close below its 50-day hourly MA.
- Synthetic QQQ_3X_CALC falling 30% from the current trade peak.

The overlay exit is generated at the close of the bar where it is observed and
then shifted by executable_weights, preserving the no-lookahead convention.
After an overlay exit, the strategy stays in cash until the base 200MA/MACD
state resets to zero; this avoids immediate re-entry inside the same base trade.
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
    _confirmed,
    _days_to_bars,
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


def activation_label(activation_gain: float) -> str:
    """Return a compact label like gain100 or gain300."""
    return f"gain{int(round(float(activation_gain) * 100))}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default=TARGET_TICKER)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--activation-gain", type=float, default=1.00)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--output-prefix", default="preferred_gain100_exit_overlay_comparison")
    return parser.parse_args()


def raw_with_gain_activated_exit(
    base_raw: pd.Series,
    traded_price: pd.Series,
    *,
    activation_gain: float,
    exit_flag: pd.Series,
    overlay_name: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Force raw state to cash after activation_gain and an overlay exit flag.

    ``base_raw`` is the 0/1 base strategy state. Once the overlay exits, the raw
    state remains cash until ``base_raw`` itself goes to zero, so the next base
    trade starts cleanly.
    """
    base = base_raw.fillna(0.0).astype(float)
    price = traded_price.reindex(base.index).astype(float)
    flag = exit_flag.reindex(base.index).fillna(False).astype(bool)

    in_trade = False
    activated = False
    stopped_until_base_exit = False
    entry_price = np.nan
    peak = np.nan

    values: list[float] = []
    activations: list[bool] = []
    triggers: list[bool] = []
    gains: list[float] = []
    peaks: list[float] = []

    for base_signal, current_price, exit_now in zip(base, price, flag, strict=True):
        current_price = float(current_price) if np.isfinite(current_price) else np.nan
        trigger = False
        new_activation = False
        gain = np.nan

        if base_signal <= 0.0 or not np.isfinite(current_price):
            in_trade = False
            activated = False
            stopped_until_base_exit = False
            entry_price = np.nan
            peak = np.nan
            value = 0.0
        else:
            if not in_trade:
                in_trade = True
                activated = False
                stopped_until_base_exit = False
                entry_price = current_price
                peak = current_price
            else:
                peak = max(float(peak), current_price)
            gain = current_price / entry_price - 1.0 if entry_price > 0 else np.nan
            if not activated and np.isfinite(gain) and gain >= activation_gain:
                activated = True
                new_activation = True
            if stopped_until_base_exit:
                value = 0.0
            elif activated and bool(exit_now):
                trigger = True
                stopped_until_base_exit = True
                value = 0.0
            else:
                value = 1.0

        values.append(value)
        activations.append(new_activation)
        triggers.append(trigger)
        gains.append(gain)
        peaks.append(peak)

    stopped = pd.Series(values, index=base.index, name=base_raw.name, dtype=float)
    diagnostics = pd.DataFrame(
        {
            "base_raw": base,
            "stopped_raw": stopped,
            "trade_entry_gain": gains,
            "trade_peak_price": peaks,
            "activation_trigger": activations,
            "overlay_exit_flag": flag,
            "overlay_exit_trigger": triggers,
            "overlay_name": overlay_name,
        },
        index=base.index,
    )
    return stopped, diagnostics


def make_gain_exit_variants(
    raw_base: pd.Series,
    target_price: pd.Series,
    qqq_price: pd.Series,
    *,
    activation_gain: float,
    bars_per_day: int,
    exit_confirm_bars: int = 3,
) -> dict[str, tuple[pd.Series, pd.DataFrame]]:
    """Build the requested gain-activated overlay exit variants."""
    variants: dict[str, tuple[pd.Series, pd.DataFrame]] = {}
    gain_label = f"gain{int(round(activation_gain * 100))}"

    for ma_days in (100, 50):
        ma_window = _days_to_bars(ma_days, bars_per_day)
        ma = qqq_price.rolling(window=ma_window, min_periods=ma_window).mean()
        exit_flag = _confirmed(qqq_price.lt(ma), exit_confirm_bars)
        raw, diag = raw_with_gain_activated_exit(
            raw_base,
            target_price,
            activation_gain=activation_gain,
            exit_flag=exit_flag,
            overlay_name=f"{gain_label}_qqq_{ma_days}ma_exit",
        )
        diag[f"qqq_{ma_days}ma"] = ma
        variants[f"{gain_label}_exit_qqq_{ma_days}ma"] = (raw, diag)

    # Synthetic TQQQ 30% peak stop, but active only after +100% gain.
    # Build the flag directly from trade-level peak drawdown inside the active base trade.
    base = raw_base.fillna(0.0).astype(float)
    price = target_price.reindex(base.index).astype(float)
    in_trade = False
    peak = np.nan
    peak_dd_values: list[float] = []
    peak_stop_flags: list[bool] = []
    for base_signal, current_price in zip(base, price, strict=True):
        current_price = float(current_price) if np.isfinite(current_price) else np.nan
        if base_signal <= 0.0 or not np.isfinite(current_price):
            in_trade = False
            peak = np.nan
            peak_dd = np.nan
            flag = False
        else:
            if not in_trade:
                in_trade = True
                peak = current_price
            else:
                peak = max(float(peak), current_price)
            peak_dd = current_price / peak - 1.0 if peak > 0 else np.nan
            flag = bool(np.isfinite(peak_dd) and peak_dd <= -0.30)
        peak_dd_values.append(peak_dd)
        peak_stop_flags.append(flag)
    stop_flag = pd.Series(peak_stop_flags, index=base.index, dtype=bool)
    raw, diag = raw_with_gain_activated_exit(
        raw_base,
        target_price,
        activation_gain=activation_gain,
        exit_flag=stop_flag,
        overlay_name=f"{gain_label}_peak_stop_30pct",
    )
    diag["trade_peak_drawdown"] = peak_dd_values
    variants[f"{gain_label}_peak_stop_30pct"] = (raw, diag)
    return variants


def _plot(returns_by_name: dict[str, pd.Series], output_path: Path, *, activation_gain: float) -> None:
    gain_label = f"gain{int(round(activation_gain * 100))}"
    selected = [
        "profit_lock_300_400_no_overlay",
        "profit_lock_300_400_stop_40pct",
        f"profit_lock_300_400_{gain_label}_exit_qqq_100ma",
        f"profit_lock_300_400_{gain_label}_exit_qqq_50ma",
        f"profit_lock_300_400_{gain_label}_peak_stop_30pct",
        "QQQ_BH",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.8))
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
    fig.suptitle(
        f"Preferred profit lock: +{activation_gain * 100:.0f}%-activated "
        "100MA/50MA/30%-peak-stop exits"
    )
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

    peak40_raw, peak40_diag = raw_with_peak_drawdown_stop(
        raw_base,
        target_prices[args.target_ticker],
        stop_drawdown=0.40,
    )
    gain_exit_variants = make_gain_exit_variants(
        raw_base,
        target_prices[args.target_ticker],
        qqq_prices[args.benchmark_ticker],
        activation_gain=args.activation_gain,
        bars_per_day=bars_per_day,
        exit_confirm_bars=3,
    )

    raw_variants: dict[str, tuple[pd.Series, pd.DataFrame]] = {
        "profit_lock_300_400_no_overlay": (raw_base, base_diagnostics),
        "profit_lock_300_400_stop_40pct": (peak40_raw, peak40_diag),
    }
    for name, item in gain_exit_variants.items():
        raw_variants[f"profit_lock_300_400_{name}"] = item

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
        overlay_trigger_count = int(diag.get("overlay_exit_trigger", pd.Series(False, index=diag.index)).sum())
        activation_count = int(diag.get("activation_trigger", pd.Series(False, index=diag.index)).sum())
        peak_stop_count = int(diag.get("stop_trigger", pd.Series(False, index=diag.index)).sum())
        metrics.update(
            {
                "name": name,
                "strategy": "preferred_gain100_exit_overlay",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "profit_lock_scheme": PROFIT_LOCK_SCHEME,
                        "activation_gain": args.activation_gain,
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
                "activation_gain": args.activation_gain,
                "activation_count": activation_count,
                "overlay_exit_trigger_count": overlay_trigger_count,
                "peak_stop_40_trigger_count": peak_stop_count,
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
            "activation_gain": np.nan,
            "activation_count": np.nan,
            "overlay_exit_trigger_count": np.nan,
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
        "activation_count",
        "overlay_exit_trigger_count",
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
    plot_path = figures_dir / f"{args.output_prefix}_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    compact.to_csv(compact_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(weights_by_name).to_csv(weights_path)
    pd.concat(diagnostics_by_name, axis=1).to_parquet(diagnostics_path)
    _plot(returns_by_name, plot_path, activation_gain=args.activation_gain)

    print(f"Compact comparison saved to {compact_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
