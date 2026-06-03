#!/usr/bin/env python
"""Test stay-out rule after a big winning round trip.

Current research base:
- QQQ hourly MACD histogram > 0 entry.
- QQQ hourly close > QQQ hourly 200-day MA entry gate.
- QQQ hourly close < QQQ hourly 200-day MA base exit.
- Synthetic QQQ_3X_CALC exposure.
- Profit lock: +300% unrealized trade gain -> 75%; +400% -> 50%.
- Candidate 40% synthetic-TQQQ trade-peak stop included in the base variant.

New overlay:
After an actual raw round-trip trade exits with max synthetic-TQQQ unrealized
price gain >= 100%, stay out of the market. Continue running the usual strategy
as an imaginary account. Re-enable entries after either:
- the imaginary strategy suffers a 20% drawdown from its post-exit high-water mark, or
- 12/18 calendar months pass.

The cooldown gates raw entries, then the project's executable-position shift is
applied, preserving the no-lookahead convention at the signal level.
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
    parser.add_argument("--peak-gain-threshold", type=float, default=1.00)
    parser.add_argument("--imaginary-dd-threshold", type=float, default=0.20)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--output-prefix", default="preferred_post_bigwin_cooldown_comparison")
    return parser.parse_args()


def pre_tax_strategy_returns(
    asset_returns: pd.Series,
    weights: pd.Series,
    *,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> pd.Series:
    """Simple pre-tax one-asset strategy returns with trading costs."""
    aligned_returns = asset_returns.reindex(weights.index).fillna(0.0).astype(float)
    w = weights.reindex(aligned_returns.index).fillna(0.0).astype(float)
    turnover = w.diff().abs().fillna(w.abs())
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0
    return (w * aligned_returns - cost_rate * turnover).rename("pre_tax_strategy_return")


def raw_with_post_bigwin_cooldown(
    base_raw: pd.Series,
    traded_price: pd.Series,
    imaginary_returns: pd.Series,
    *,
    peak_gain_threshold: float,
    imaginary_dd_threshold: float,
    timeout_months: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Gate raw entries after a big winning trade exits.

    A trade's peak is measured using the synthetic traded price from actual raw
    entry to actual raw exit. While in cooldown, actual raw exposure is zero;
    the imaginary strategy return stream is still tracked for a drawdown reset.
    """
    base = base_raw.fillna(0.0).astype(float)
    price = traded_price.reindex(base.index).astype(float)
    imag_ret = imaginary_returns.reindex(base.index).fillna(0.0).astype(float)

    in_trade = False
    entry_price = np.nan
    peak_price = np.nan
    max_gain = np.nan

    cooldown = False
    cooldown_until: pd.Timestamp | pd.NaT = pd.NaT
    imag_equity = 1.0
    imag_hwm = 1.0
    unblock_next = False

    actual_values: list[float] = []
    cooldown_values: list[bool] = []
    start_triggers: list[bool] = []
    dd_unblocks: list[bool] = []
    timeout_unblocks: list[bool] = []
    trade_peak_gains: list[float] = []
    imag_equities: list[float] = []
    imag_dds: list[float] = []

    for timestamp, base_signal in base.items():
        current_price = float(price.loc[timestamp]) if timestamp in price.index and np.isfinite(price.loc[timestamp]) else np.nan
        start_trigger = False
        dd_unblock = False
        timeout_unblock = False

        if cooldown and not pd.isna(cooldown_until) and timestamp >= cooldown_until:
            cooldown = False
            timeout_unblock = True
            cooldown_until = pd.NaT
            imag_equity = 1.0
            imag_hwm = 1.0
            unblock_next = False
        elif cooldown and unblock_next:
            cooldown = False
            cooldown_until = pd.NaT
            imag_equity = 1.0
            imag_hwm = 1.0
            unblock_next = False

        if cooldown:
            actual = 0.0
            # Update imaginary account using information through this bar; any DD
            # trigger can only release entries on a later bar.
            imag_equity *= 1.0 + float(imag_ret.loc[timestamp])
            imag_hwm = max(imag_hwm, imag_equity)
            imag_dd = imag_equity / imag_hwm - 1.0 if imag_hwm > 0 else np.nan
            if np.isfinite(imag_dd) and imag_dd <= -abs(imaginary_dd_threshold):
                dd_unblock = True
                unblock_next = True
            # If base is zero while cooling down, do not count it as an actual trade.
            in_trade = False
            entry_price = np.nan
            peak_price = np.nan
            max_gain = np.nan
        else:
            actual = float(base_signal > 0.0 and np.isfinite(current_price))
            imag_dd = 0.0
            if actual > 0.0:
                if not in_trade:
                    in_trade = True
                    entry_price = current_price
                    peak_price = current_price
                    max_gain = 0.0
                else:
                    peak_price = max(float(peak_price), current_price)
                    max_gain = peak_price / entry_price - 1.0 if entry_price > 0 else np.nan
            elif in_trade:
                # Trade exits on this raw bar. If it was a big winner at any
                # point, start the cooldown after this bar.
                start_trigger = bool(np.isfinite(max_gain) and max_gain >= peak_gain_threshold)
                if start_trigger:
                    cooldown = True
                    cooldown_until = timestamp + pd.DateOffset(months=timeout_months)
                    imag_equity = 1.0
                    imag_hwm = 1.0
                    unblock_next = False
                in_trade = False
                entry_price = np.nan
                peak_price = np.nan
                max_gain = np.nan

        actual_values.append(actual)
        cooldown_values.append(cooldown)
        start_triggers.append(start_trigger)
        dd_unblocks.append(dd_unblock)
        timeout_unblocks.append(timeout_unblock)
        trade_peak_gains.append(max_gain)
        imag_equities.append(imag_equity if cooldown else np.nan)
        imag_dds.append(imag_dd if cooldown else np.nan)

    actual_raw = pd.Series(actual_values, index=base.index, name=base_raw.name, dtype=float)
    diagnostics = pd.DataFrame(
        {
            "base_raw": base,
            "actual_raw": actual_raw,
            "cooldown_active": cooldown_values,
            "cooldown_start_trigger": start_triggers,
            "dd_unblock_trigger": dd_unblocks,
            "timeout_unblock_trigger": timeout_unblocks,
            "trade_peak_gain": trade_peak_gains,
            "imaginary_equity": imag_equities,
            "imaginary_drawdown": imag_dds,
        },
        index=base.index,
    )
    return actual_raw, diagnostics


def _plot(returns_by_name: dict[str, pd.Series], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.8))
    order = [
        "profit_lock_300_400_stop_40pct",
        "cooldown_after_peak100_dd20_or_12m",
        "cooldown_after_peak100_dd20_or_18m",
        "profit_lock_300_400_no_overlay",
        "QQQ_BH",
    ]
    for name in order:
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
    fig.suptitle("Post-big-win stay-out rule: wait for imaginary -20% DD or 12/18 months")
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
    raw_stop40, stop40_diag = raw_with_peak_drawdown_stop(
        raw_base,
        target_prices[args.target_ticker],
        stop_drawdown=0.40,
    )

    # Usual executable strategy return stream used only for the imaginary DD trigger.
    usual_raw_weight = trade_profit_lock_tiers(
        raw_stop40.rename(args.target_ticker),
        target_prices[args.target_ticker],
        thresholds_to_weights=PROFIT_LOCK_SCHEME,
    ).rename(args.target_ticker)
    usual_weights = executable_weights(usual_raw_weight.to_frame(args.target_ticker), config=config).reindex(common).fillna(0.0)
    imaginary_returns = pre_tax_strategy_returns(
        target_returns[args.target_ticker],
        usual_weights[args.target_ticker],
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )

    raw_variants: dict[str, tuple[pd.Series, pd.DataFrame]] = {
        "profit_lock_300_400_no_overlay": (raw_base, base_diagnostics),
        "profit_lock_300_400_stop_40pct": (raw_stop40, stop40_diag),
    }
    for months in (12, 18):
        raw_cooldown, cooldown_diag = raw_with_post_bigwin_cooldown(
            raw_stop40.rename(args.target_ticker),
            target_prices[args.target_ticker],
            imaginary_returns,
            peak_gain_threshold=args.peak_gain_threshold,
            imaginary_dd_threshold=args.imaginary_dd_threshold,
            timeout_months=months,
        )
        raw_variants[f"cooldown_after_peak100_dd20_or_{months}m"] = (raw_cooldown, cooldown_diag)

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
                "strategy": "preferred_post_bigwin_cooldown",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "profit_lock_scheme": PROFIT_LOCK_SCHEME,
                        "peak_gain_threshold": args.peak_gain_threshold,
                        "imaginary_dd_threshold": args.imaginary_dd_threshold,
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
                "cooldown_start_count": int(diag.get("cooldown_start_trigger", pd.Series(False, index=diag.index)).sum()),
                "cooldown_dd_unblock_count": int(diag.get("dd_unblock_trigger", pd.Series(False, index=diag.index)).sum()),
                "cooldown_timeout_unblock_count": int(diag.get("timeout_unblock_trigger", pd.Series(False, index=diag.index)).sum()),
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
            "cooldown_start_count": np.nan,
            "cooldown_dd_unblock_count": np.nan,
            "cooldown_timeout_unblock_count": np.nan,
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
        "cooldown_start_count",
        "cooldown_dd_unblock_count",
        "cooldown_timeout_unblock_count",
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
    _plot(returns_by_name, plot_path)

    print(f"Compact comparison saved to {compact_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
