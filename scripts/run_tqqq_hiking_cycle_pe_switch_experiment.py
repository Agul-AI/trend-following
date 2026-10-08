#!/usr/bin/env python
"""Test switching from synthetic TQQQ to QQQ during announced Fed hike cycles when QQQ P/E > 30.

Primary no-lookahead convention for the valuation flag:
- The QQQ P/E proxy is monthly Nasdaq-100 trailing P/E.
- ``prior_month`` mode uses only the last completed month's P/E for current-month
  intraday decisions.
- Fed hike-cycle announcement flags are shifted by one observed trading day.
- Raw desired weights are still passed through ``executable_weights`` so a signal
  at bar t does not earn the return ending at bar t.
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

from run_tqqq_entry_signal_comparison import (  # noqa: E402
    _drawdown,
    _equity,
    _returns_from_prices,
    executable_weights,
)
from run_tqqq_hiking_cycle_exit_ma_experiments import hiking_cycle_known_flag  # noqa: E402
from run_tqqq_mixed_entry_exit_source_comparison import mixed_source_signal  # noqa: E402
from run_tqqq_position_risk_sizing_experiments import (  # noqa: E402
    drawdown_episode_count,
    simulate_after_tax_portfolio,
)
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.metrics import calculate_metrics, metrics_to_frame  # noqa: E402
from trend_following.regime import (  # noqa: E402
    align_daily_regimes_to_intraday,
    classify_regimes,
    compute_regime_features,
)
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

# Official FOMC statement-date windows used in the prior sell-timing study.
# The end date is the final effective hike date in that hiking cycle.
ANNOUNCED_HIKING_CYCLES: tuple[tuple[str, str, str], ...] = (
    ("2004-2006 hike cycle", "2004-05-04", "2006-06-29"),
    ("2015-2018 normalization/hike cycle", "2015-10-28", "2018-12-20"),
    ("2022-2023 hike cycle", "2022-01-26", "2023-07-27"),
)

# Monthly Nasdaq-100 trailing P/E proxy for QQQ, enough to cover the announced
# hike-cycle windows above plus one prior month for no-lookahead alignment.
# Source: Trendonify Nasdaq 100 P/E historical-data table.
NASDAQ100_PE_PROXY_BY_MONTH: dict[str, float] = {
    # 2004-2006 cycle and one prior month.
    "2004-04": 37.64,
    "2004-05": 37.64,
    "2004-06": 37.64,
    "2004-07": 36.87,
    "2004-08": 33.74,
    "2004-09": 33.74,
    "2004-10": 33.74,
    "2004-11": 33.69,
    "2004-12": 33.69,
    "2005-01": 32.73,
    "2005-02": 32.73,
    "2005-03": 30.03,
    "2005-04": 30.03,
    "2005-05": 30.03,
    "2005-06": 28.97,
    "2005-07": 28.97,
    "2005-08": 28.97,
    "2005-09": 28.97,
    "2005-10": 28.97,
    "2005-11": 28.96,
    "2005-12": 27.22,
    "2006-01": 25.85,
    "2006-02": 25.23,
    "2006-03": 25.23,
    "2006-04": 25.19,
    "2006-05": 23.40,
    "2006-06": 25.67,
    # 2015-2018 cycle and one prior month.
    "2015-09": 19.53,
    "2015-10": 21.73,
    "2015-11": 21.80,
    "2015-12": 22.69,
    "2016-01": 21.14,
    "2016-02": 20.76,
    "2016-03": 22.11,
    "2016-04": 21.41,
    "2016-05": 22.30,
    "2016-06": 21.13,
    "2016-07": 22.62,
    "2016-08": 22.82,
    "2016-09": 24.30,
    "2016-10": 23.93,
    "2016-11": 23.98,
    "2016-12": 24.24,
    "2017-01": 23.68,
    "2017-02": 24.66,
    "2017-03": 23.81,
    "2017-04": 24.45,
    "2017-05": 25.35,
    "2017-06": 24.06,
    "2017-07": 25.06,
    "2017-08": 25.53,
    "2017-09": 25.49,
    "2017-10": 24.09,
    "2017-11": 24.54,
    "2017-12": 24.66,
    "2018-01": 28.72,
    "2018-02": 28.32,
    "2018-03": 27.20,
    "2018-04": 25.70,
    "2018-05": 27.11,
    "2018-06": 27.39,
    "2018-07": 25.81,
    "2018-08": 27.31,
    "2018-09": 27.22,
    "2018-10": 23.43,
    "2018-11": 23.37,
    "2018-12": 18.92,
    # 2022-2023 cycle and one prior month.
    "2021-12": 31.08,
    "2022-01": 28.42,
    "2022-02": 27.11,
    "2022-03": 28.65,
    "2022-04": 24.82,
    "2022-05": 24.41,
    "2022-06": 23.85,
    "2022-07": 26.85,
    "2022-08": 25.46,
    "2022-09": 24.04,
    "2022-10": 24.99,
    "2022-11": 26.36,
    "2022-12": 23.97,
    "2023-01": 28.40,
    "2023-02": 28.26,
    "2023-03": 30.81,
    "2023-04": 30.96,
    "2023-05": 33.33,
    "2023-06": 32.66,
    "2023-07": 33.91,
}

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
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--hike-cycle-lag-days", type=int, default=1)
    parser.add_argument("--pe-threshold", type=float, default=30.0)
    parser.add_argument("--output-prefix", default="tqqq_hiking_cycle_pe_switch_experiment")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def announced_hiking_cycle_known_flag(index: pd.DatetimeIndex, *, lag_days: int = 1) -> pd.Series:
    """Return announced-hike-cycle status known before the decision day."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    normalized = index.tz_localize(None).normalize()
    unique_dates = pd.DatetimeIndex(pd.unique(normalized)).sort_values()
    daily = pd.Series(False, index=unique_dates)
    for _, start, end in ANNOUNCED_HIKING_CYCLES:
        daily |= (daily.index >= pd.Timestamp(start)) & (daily.index <= pd.Timestamp(end))
    known = daily.shift(lag_days, fill_value=False) if lag_days > 0 else daily
    return pd.Series(known.reindex(normalized).to_numpy(dtype=bool), index=index, name="announced_hiking_cycle_known")


def qqq_pe_gt_threshold_flag(
    index: pd.DatetimeIndex,
    *,
    threshold: float = 30.0,
    mode: str = "prior_month",
) -> pd.Series:
    """Monthly QQQ P/E proxy flag aligned to intraday bars.

    mode='prior_month' is the conservative no-lookahead version: a date in
    2023-04 uses the 2023-03 P/E proxy. mode='same_month' is a sensitivity
    check that assumes a timely current-month P/E proxy is already available.
    """
    if mode not in {"prior_month", "same_month"}:
        raise ValueError("mode must be 'prior_month' or 'same_month'")
    months = pd.PeriodIndex(index.tz_localize(None).to_period("M"), freq="M")
    lookup_months = months - 1 if mode == "prior_month" else months
    pe_values = [NASDAQ100_PE_PROXY_BY_MONTH.get(str(month), np.nan) for month in lookup_months]
    flag = pd.Series(np.asarray(pe_values, dtype=float) > threshold, index=index, name=f"qqq_pe_gt_{threshold:g}_{mode}")
    return flag.fillna(False).astype(bool)




def announced_hiking_cycle_entry_pe_known_flag(
    index: pd.DatetimeIndex,
    *,
    threshold: float = 30.0,
    mode: str = "same_month",
    lag_days: int = 1,
) -> tuple[pd.Series, pd.DataFrame]:
    """Flag entire announced hike cycles only if QQQ P/E at announcement exceeds threshold.

    This is a sensitivity for the interpretation: "when the Fed announces a
    hiking cycle, if QQQ P/E is already > threshold, switch from synthetic TQQQ
    to QQQ until the cycle ends."
    """
    if mode not in {"prior_month", "same_month"}:
        raise ValueError("mode must be 'prior_month' or 'same_month'")
    normalized = index.tz_localize(None).normalize()
    unique_dates = pd.DatetimeIndex(pd.unique(normalized)).sort_values()
    daily = pd.Series(False, index=unique_dates)
    decisions: list[dict[str, Any]] = []
    for cycle_name, start, end in ANNOUNCED_HIKING_CYCLES:
        announcement_month = pd.Timestamp(start).to_period("M")
        lookup_month = announcement_month - 1 if mode == "prior_month" else announcement_month
        pe_value = NASDAQ100_PE_PROXY_BY_MONTH.get(str(lookup_month), np.nan)
        include_cycle = bool(np.isfinite(pe_value) and pe_value > threshold)
        decisions.append(
            {
                "cycle_name": cycle_name,
                "announcement_date": start,
                "cycle_end_date": end,
                "pe_alignment": mode,
                "pe_lookup_month": str(lookup_month),
                "qqq_pe_proxy": pe_value,
                "pe_threshold": threshold,
                "include_cycle": include_cycle,
            }
        )
        if include_cycle:
            daily |= (daily.index >= pd.Timestamp(start)) & (daily.index <= pd.Timestamp(end))
    known = daily.shift(lag_days, fill_value=False) if lag_days > 0 else daily
    flag = pd.Series(
        known.reindex(normalized).to_numpy(dtype=bool),
        index=index,
        name=f"announced_hiking_cycle_entry_pe_gt_{threshold:g}_{mode}",
    )
    return flag, pd.DataFrame(decisions)


def _plot_comparison(returns_by_name: dict[str, pd.Series], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for name, returns in returns_by_name.items():
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
    fig.suptitle("Synthetic TQQQ: switch to QQQ during announced hike cycle when QQQ P/E > 30")
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _add_dd_counts(metrics: dict[str, Any], returns: pd.Series) -> dict[str, Any]:
    metrics["drawdown_episodes_gt_30pct"] = drawdown_episode_count(returns, threshold=-0.30)
    metrics["drawdown_episodes_gt_40pct"] = drawdown_episode_count(returns, threshold=-0.40)
    metrics["drawdown_episodes_gt_50pct"] = drawdown_episode_count(returns, threshold=-0.50)
    metrics["dd_episodes_gt_30_40_50pct"] = (
        f"{metrics['drawdown_episodes_gt_30pct']}/"
        f"{metrics['drawdown_episodes_gt_40pct']}/"
        f"{metrics['drawdown_episodes_gt_50pct']}"
    )
    return metrics


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
    all_returns = pd.concat(
        [
            _returns_from_prices(target_prices),
            _returns_from_prices(qqq_prices),
        ],
        axis=1,
    ).loc[common]

    daily_prices = daily_qqq.to_frame()
    daily_returns = _returns_from_prices(daily_prices)
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

    # Preferred candidate: QQQ MACD entry + QQQ 200-day exit, no lock.
    raw_tqqq, diagnostics = mixed_source_signal(
        entry_price=qqq_prices[args.benchmark_ticker],
        exit_price=qqq_prices[args.benchmark_ticker],
        output_index=common,
        allowed_regime=allowed_regime,
        bars_per_day=bars_per_day,
        average_type=args.average_type,
        macd_unit=args.macd_unit,
        entry_confirm_bars=2,
        exit_confirm_bars=3,
        exit_ma_days=200,
    )
    raw_tqqq = raw_tqqq.rename(args.target_ticker)
    raw_base_weights = raw_tqqq.to_frame(args.target_ticker)

    hike_announcement_known = announced_hiking_cycle_known_flag(common, lag_days=args.hike_cycle_lag_days)
    # Sensitivity: effective hiking cycles from the previous script, not just pre-hike announcements.
    hike_effective_known = hiking_cycle_known_flag(common, lag_days=args.hike_cycle_lag_days)

    raw_variants: dict[str, pd.DataFrame] = {"base_tqqq_only": raw_base_weights}
    flag_outputs: dict[str, pd.Series] = {
        "announced_hiking_cycle_known": hike_announcement_known,
        "effective_hiking_cycle_known": hike_effective_known,
    }
    overlay_meta: dict[str, dict[str, Any]] = {
        "base_tqqq_only": {
            "overlay_rule": "none",
            "hiking_window": "none",
            "pe_alignment": "none",
            "pe_threshold": np.nan,
            "switch_to_qqq_when_overlay_active": False,
        }
    }

    for window_name, hike_flag in {
        "announced": hike_announcement_known,
        "effective": hike_effective_known,
    }.items():
        for pe_mode in ("prior_month", "same_month"):
            pe_flag = qqq_pe_gt_threshold_flag(common, threshold=args.pe_threshold, mode=pe_mode)
            overlay_flag = (hike_flag & pe_flag).rename(f"overlay_{window_name}_{pe_mode}_pe_gt_{args.pe_threshold:g}")
            name = f"switch_to_qqq_{window_name}_hike_pe_gt_{args.pe_threshold:g}_{pe_mode}"
            flag_outputs[f"pe_gt_{args.pe_threshold:g}_{pe_mode}"] = pe_flag
            flag_outputs[name] = overlay_flag

            raw = pd.DataFrame(0.0, index=common, columns=[args.target_ticker, args.benchmark_ticker])
            base_on = raw_tqqq.reindex(common).fillna(0.0).astype(float)
            raw.loc[:, args.target_ticker] = base_on.where(~overlay_flag, 0.0).to_numpy(dtype=float)
            raw.loc[:, args.benchmark_ticker] = base_on.where(overlay_flag, 0.0).to_numpy(dtype=float)
            raw_variants[name] = raw
            overlay_meta[name] = {
                "overlay_rule": "dynamic_hike_and_pe_gt_threshold",
                "hiking_window": window_name,
                "pe_alignment": pe_mode,
                "pe_threshold": args.pe_threshold,
                "switch_to_qqq_when_overlay_active": True,
                "overlay_active_bars_raw": int(overlay_flag.sum()),
                "overlay_active_days_raw": int(pd.DatetimeIndex(overlay_flag[overlay_flag].index.normalize()).nunique()),
                "overlay_active_when_base_long_bars_raw": int((overlay_flag & base_on.gt(0)).sum()),
            }


    cycle_entry_decisions: list[pd.DataFrame] = []
    for pe_mode in ("prior_month", "same_month"):
        cycle_entry_flag, decisions = announced_hiking_cycle_entry_pe_known_flag(
            common,
            threshold=args.pe_threshold,
            mode=pe_mode,
            lag_days=args.hike_cycle_lag_days,
        )
        cycle_entry_decisions.append(decisions)
        name = f"switch_to_qqq_announced_hike_entry_pe_gt_{args.pe_threshold:g}_{pe_mode}"
        flag_outputs[name] = cycle_entry_flag
        raw = pd.DataFrame(0.0, index=common, columns=[args.target_ticker, args.benchmark_ticker])
        base_on = raw_tqqq.reindex(common).fillna(0.0).astype(float)
        raw.loc[:, args.target_ticker] = base_on.where(~cycle_entry_flag, 0.0).to_numpy(dtype=float)
        raw.loc[:, args.benchmark_ticker] = base_on.where(cycle_entry_flag, 0.0).to_numpy(dtype=float)
        raw_variants[name] = raw
        overlay_meta[name] = {
            "overlay_rule": "whole_announced_cycle_if_entry_pe_gt_threshold",
            "hiking_window": "announced",
            "pe_alignment": pe_mode,
            "pe_threshold": args.pe_threshold,
            "switch_to_qqq_when_overlay_active": True,
            "overlay_active_bars_raw": int(cycle_entry_flag.sum()),
            "overlay_active_days_raw": int(
                pd.DatetimeIndex(cycle_entry_flag[cycle_entry_flag].index.normalize()).nunique()
            ),
            "overlay_active_when_base_long_bars_raw": int((cycle_entry_flag & base_on.gt(0)).sum()),
        }

    metric_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.Series] = {}
    raw_weights_by_name: dict[str, pd.Series] = {}

    for name, raw_weights in raw_variants.items():
        weights = executable_weights(raw_weights, config=config).reindex(common).fillna(0.0)
        columns = [column for column in weights.columns if column in all_returns.columns]
        after_tax, pretax, taxes_paid, turnover = simulate_after_tax_portfolio(
            all_returns[columns],
            weights[columns],
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            tax_rate=args.short_term_tax_rate,
        )
        metrics = calculate_metrics(
            after_tax,
            turnover=turnover,
            weights=weights[columns],
            annualization=config.backtest.annualization,
        )
        metrics.update(
            {
                "name": name,
                "strategy": "hiking_cycle_pe_switch_experiment",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "transaction_cost_bps": args.transaction_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "short_term_tax_rate": args.short_term_tax_rate,
                        "announcement_cycles": ANNOUNCED_HIKING_CYCLES,
                        "base_candidate": "QQQ entry + QQQ exit + no lock",
                        "base_params": params,
                    },
                    sort_keys=True,
                ),
                "pretax_cumulative_return": float((1.0 + pretax).prod() - 1.0),
                "tax_paid_pct_initial_capital": float(taxes_paid.sum()),
                **overlay_meta[name],
            }
        )
        _add_dd_counts(metrics, after_tax)
        metric_rows.append(metrics)
        returns_by_name[name] = after_tax
        weights_by_name[f"{name}__tqqq_weight"] = weights.get(args.target_ticker, pd.Series(0.0, index=common))
        weights_by_name[f"{name}__qqq_weight"] = weights.get(args.benchmark_ticker, pd.Series(0.0, index=common))
        raw_weights_by_name[f"{name}__tqqq_raw"] = raw_weights.get(args.target_ticker, pd.Series(0.0, index=common))
        raw_weights_by_name[f"{name}__qqq_raw"] = raw_weights.get(args.benchmark_ticker, pd.Series(0.0, index=common))

    benchmark_returns = all_returns[args.benchmark_ticker]
    benchmark_metrics = calculate_metrics(benchmark_returns, annualization=config.backtest.annualization)
    benchmark_metrics.update(
        {
            "name": "buy_hold_qqq",
            "strategy": "benchmark",
            "segment": "full_sample",
            "parameters": "{}",
            "pretax_cumulative_return": float((1.0 + benchmark_returns).prod() - 1.0),
            "tax_paid_pct_initial_capital": 0.0,
            "overlay_rule": "none",
            "hiking_window": "none",
            "pe_alignment": "none",
            "pe_threshold": np.nan,
            "switch_to_qqq_when_overlay_active": False,
        }
    )
    _add_dd_counts(benchmark_metrics, benchmark_returns)
    metric_rows.append(benchmark_metrics)
    returns_by_name["buy_hold_qqq"] = benchmark_returns

    metrics = metrics_to_frame(metric_rows)
    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    compact_path = tables_dir / f"{args.output_prefix}_compact.csv"
    returns_path = tables_dir / f"{args.output_prefix}_after_tax_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    raw_weights_path = tables_dir / f"{args.output_prefix}_raw_weights.csv"
    flags_path = tables_dir / f"{args.output_prefix}_flags.csv"
    diagnostics_path = tables_dir / f"{args.output_prefix}_diagnostics.parquet"
    pe_path = tables_dir / f"{args.output_prefix}_pe_proxy_source.csv"
    cycle_entry_decisions_path = tables_dir / f"{args.output_prefix}_cycle_entry_pe_decisions.csv"
    plot_path = figures_dir / f"{args.output_prefix}_equity_drawdown.png"

    compact_cols = [
        "name",
        "overlay_rule",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "number_of_trades",
        "exposure_percentage",
        "dd_episodes_gt_30_40_50pct",
        "hiking_window",
        "pe_alignment",
        "overlay_active_days_raw",
        "overlay_active_when_base_long_bars_raw",
    ]
    compact = metrics[compact_cols].copy()
    compact.to_csv(compact_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(weights_by_name).to_csv(weights_path)
    pd.DataFrame(raw_weights_by_name).to_csv(raw_weights_path)
    pd.DataFrame(flag_outputs).to_csv(flags_path)
    diagnostics.to_parquet(diagnostics_path)
    pd.DataFrame(
        [
            {
                "month": month,
                "nasdaq100_pe_proxy": value,
                "source": "Trendonify Nasdaq 100 P/E historical-data table",
            }
            for month, value in sorted(NASDAQ100_PE_PROXY_BY_MONTH.items())
        ]
    ).to_csv(pe_path, index=False)
    pd.concat(cycle_entry_decisions, ignore_index=True).to_csv(cycle_entry_decisions_path, index=False)
    _plot_comparison(
        {
            "base_tqqq_only": returns_by_name["base_tqqq_only"],
            "dynamic_announced_prior_month": returns_by_name[
                f"switch_to_qqq_announced_hike_pe_gt_{args.pe_threshold:g}_prior_month"
            ],
            "entry_pe_whole_cycle_same_month": returns_by_name[
                f"switch_to_qqq_announced_hike_entry_pe_gt_{args.pe_threshold:g}_same_month"
            ],
            "entry_pe_whole_cycle_prior_month": returns_by_name[
                f"switch_to_qqq_announced_hike_entry_pe_gt_{args.pe_threshold:g}_prior_month"
            ],
            "buy_hold_qqq": returns_by_name["buy_hold_qqq"],
        },
        plot_path,
    )

    print(f"Metrics saved to {metrics_path}")
    print(f"Compact table saved to {compact_path}")
    print(f"Returns saved to {returns_path}")
    print(f"Weights saved to {weights_path}")
    print(f"Flags saved to {flags_path}")
    print(f"P/E proxy source table saved to {pe_path}")
    print(f"Cycle-entry P/E decisions saved to {cycle_entry_decisions_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
