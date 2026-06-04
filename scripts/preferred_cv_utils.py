"""Utilities for preferred QQQ/synthetic-TQQQ robustness cross-validation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tqqq_cash_yield_candidate_comparison import (  # noqa: E402
    simulate_after_tax_portfolio_with_cash_yield,
)
from run_tqqq_daily_gate_ablation import _load_price  # noqa: E402
from run_tqqq_entry_signal_comparison import (  # noqa: E402
    _confirmed,
    _days_to_bars,
    _returns_from_prices,
    executable_weights,
    macd_components,
)
from run_tqqq_position_risk_sizing_experiments import drawdown_episode_count  # noqa: E402
from trend_following.bear_whipsaw import bear_market_features, bear_reentry_filter_raw  # noqa: E402
from trend_following.metrics import calculate_metrics  # noqa: E402
from trend_following.risk_overlays import (  # noqa: E402
    apply_cap,
    dynamic_pre100_distance_trim_rebuy_cap,
    qqq_mean_reversion_features,
    raw_with_peak_drawdown_stop,
    trade_profit_lock_tiers,
)
from trend_following.utils import resolve_path  # noqa: E402

TARGET_TICKER = "QQQ_3X_CALC"
BENCHMARK_TICKER = "QQQ"
CURRENT_PREFERRED_NAME = "preferred_q110_best_robustness_bear_filter_macd_slow24"
OFFICIAL_EVALUATION_START = pd.Timestamp("2002-01-10 15:00:00")
TAX_TIMING = "annual_net_eoy"

DEFAULT_PARAMS: dict[str, Any] = {
    "long_ma_days": 200.0,
    "entry_confirm_bars": 2,
    "exit_confirm_bars": 3,
    "macd_fast_days": 12.0,
    "macd_slow_days": 24.0,
    "macd_signal_days": 9.0,
    "average_type": "sma",
    "peak_stop_drawdown": 0.40,
    "profit_lock_scheme": ((3.0, 0.75), (4.0, 0.50)),
    "q100_activation_gain": 1.10,
    "q100_trim_weight": 0.50,
    "q100_distance_quantile": 1.0,
    "q100_reentry_rule": "ma20",
    "bear_filter_buffer": 0.01,
    "bear_filter_slope_days": 30,
    "bear_filter_require_20gt50": True,
}

START_DATES = [
    "2002-01-10 15:00:00",
    "2003-01-01",
    "2004-01-01",
    "2005-01-01",
    "2007-01-01",
    "2009-01-01",
    "2010-01-01",
    "2011-01-01",
    "2013-01-01",
    "2016-01-01",
    "2018-01-01",
    "2020-01-01",
    "2022-01-01",
]


@dataclass(frozen=True)
class CandidateSpec:
    """Parameter variant to evaluate."""

    name: str
    family: str
    overrides: dict[str, Any]

    @property
    def params(self) -> dict[str, Any]:
        params = dict(DEFAULT_PARAMS)
        params.update(self.overrides)
        return params


def make_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    """Return the subset of args needed by the evaluator/simulator."""
    return argparse.Namespace(
        target_ticker=args.target_ticker,
        benchmark_ticker=args.benchmark_ticker,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        short_term_tax_rate=args.short_term_tax_rate,
        cash_annual_yield=args.cash_annual_yield,
        cash_interest_tax_rate=args.cash_interest_tax_rate,
    )


def load_cv_data(config: Any, args: argparse.Namespace) -> tuple[pd.Series, pd.Series, pd.DataFrame, int]:
    """Load synthetic-3x target, QQQ signal/benchmark, returns, and bars/day."""
    target = _load_price(
        resolve_path(config.root, args.target_raw_dir) / f"{args.target_ticker}.parquet",
        args.target_ticker,
    )
    qqq = _load_price(
        resolve_path(config.root, args.benchmark_raw_dir) / f"{args.benchmark_ticker}.parquet",
        args.benchmark_ticker,
    )
    common = target.index.intersection(qqq.index)
    target = target.loc[common]
    qqq = qqq.loc[common]
    returns = pd.concat(
        [_returns_from_prices(target.to_frame()), _returns_from_prices(qqq.to_frame())],
        axis=1,
    ).loc[common]
    params = dict(config.strategies.regime_switch)
    bars_per_day = int(params.get("intraday_bars_per_day", 6))
    return target, qqq, returns, bars_per_day


def preferred_candidate_specs() -> list[CandidateSpec]:
    """Return controlled one-factor variants around the current preferred rule."""
    specs: list[CandidateSpec] = [
        CandidateSpec(CURRENT_PREFERRED_NAME, "current_preferred", {}),
        CandidateSpec(
            "macd_standard_12_26_9",
            "retained_macd_option",
            {"macd_fast_days": 12.0, "macd_slow_days": 26.0, "macd_signal_days": 9.0},
        ),
        CandidateSpec(
            "macd_signal_8d",
            "retained_macd_option",
            {"macd_fast_days": 12.0, "macd_slow_days": 26.0, "macd_signal_days": 8.0},
        ),
        CandidateSpec(
            "macd_slow_24d",
            "retained_macd_option",
            {"macd_fast_days": 12.0, "macd_slow_days": 24.0, "macd_signal_days": 9.0},
        ),
    ]

    for value in (150.0, 200.0, 250.0):
        specs.append(CandidateSpec(f"long_ma_{int(value)}d", "trend_gate_long_ma", {"long_ma_days": value}))
    for value in (1, 2, 3):
        specs.append(
            CandidateSpec(f"entry_confirm_{value}bar", "entry_confirmation", {"entry_confirm_bars": value})
        )
    for value in (1, 2, 3, 4):
        specs.append(CandidateSpec(f"exit_confirm_{value}bar", "exit_confirmation", {"exit_confirm_bars": value}))

    for value in (10.0, 12.0, 14.0):
        specs.append(CandidateSpec(f"macd_fast_{int(value)}d", "macd_entry", {"macd_fast_days": value}))
    for value in (24.0, 26.0, 30.0):
        specs.append(CandidateSpec(f"macd_slow_{int(value)}d", "macd_entry", {"macd_slow_days": value}))
    for value in (8.0, 9.0, 12.0):
        specs.append(CandidateSpec(f"macd_signal_{int(value)}d", "macd_entry", {"macd_signal_days": value}))

    for value in (0.90, 1.00, 1.10, 1.20, 1.30):
        specs.append(
            CandidateSpec(f"q100_activation_{int(value * 100)}", "q100_activation", {"q100_activation_gain": value})
        )
    for value in (0.50, 0.60, 0.75):
        specs.append(CandidateSpec(f"q100_trim_to_{int(value * 100)}", "q100_trim_weight", {"q100_trim_weight": value}))
    for rule in ("ma20", "ma50"):
        specs.append(CandidateSpec(f"q100_reentry_{rule}", "q100_reentry", {"q100_reentry_rule": rule}))

    for buffer in (0.005, 0.01, 0.015, 0.02):
        for slope_days in (20, 30, 40):
            for require in (False, True):
                suffix = "_20gt50" if require else ""
                specs.append(
                    CandidateSpec(
                        f"bear_buf{int(buffer * 1000):03d}bp_slope{slope_days}{suffix}",
                        "bear_filter",
                        {
                            "bear_filter_buffer": buffer,
                            "bear_filter_slope_days": slope_days,
                            "bear_filter_require_20gt50": require,
                        },
                    )
                )

    for value in (0.35, 0.40, 0.45, 0.50):
        specs.append(CandidateSpec(f"peak_stop_{int(value * 100)}", "peak_stop", {"peak_stop_drawdown": value}))

    lock_schemes = {
        "profit_lock_none": tuple(),
        "profit_lock_300_400": ((3.0, 0.75), (4.0, 0.50)),
        "profit_lock_300_500": ((3.0, 0.75), (5.0, 0.50)),
        "profit_lock_400_600": ((4.0, 0.75), (6.0, 0.50)),
    }
    for name, scheme in lock_schemes.items():
        specs.append(CandidateSpec(name, "profit_lock", {"profit_lock_scheme": scheme}))

    # Drop duplicate names that can occur where a variant equals the baseline.
    seen: set[str] = set()
    unique: list[CandidateSpec] = []
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        unique.append(spec)
    return unique


def build_base_signal(
    qqq: pd.Series,
    *,
    output_index: pd.DatetimeIndex,
    bars_per_day: int,
    params: dict[str, Any],
) -> tuple[pd.Series, pd.DataFrame]:
    """Build raw QQQ MACD + long-MA gate/exit signal with configurable windows."""
    price = qqq.reindex(output_index).astype(float).sort_index()
    fast = _days_to_bars(float(params["macd_fast_days"]), bars_per_day)
    slow = _days_to_bars(float(params["macd_slow_days"]), bars_per_day)
    signal = _days_to_bars(float(params["macd_signal_days"]), bars_per_day)
    if fast >= slow:
        raise ValueError(f"MACD fast window must be < slow window, got {fast} >= {slow}")
    macd = macd_components(
        price,
        fast_window=fast,
        slow_window=slow,
        signal_window=signal,
        average_type=str(params["average_type"]),
    )
    long_window = _days_to_bars(float(params["long_ma_days"]), bars_per_day)
    long_ma = price.rolling(window=long_window, min_periods=long_window).mean()
    above_long = price.gt(long_ma)
    entry = _confirmed(
        macd["macd_hist"].gt(0.0) & above_long,
        int(params["entry_confirm_bars"]),
    )
    exit_flag = _confirmed(price.lt(long_ma), int(params["exit_confirm_bars"]))

    state = 0.0
    values: list[float] = []
    for entry_now, exit_now in zip(entry, exit_flag, strict=False):
        if state == 0.0 and bool(entry_now):
            state = 1.0
        elif state > 0.0 and bool(exit_now):
            state = 0.0
        values.append(state)

    raw = pd.Series(values, index=output_index, name="raw_signal", dtype=float)
    diagnostics = pd.DataFrame(
        {
            "qqq_price": price,
            "macd_hist": macd["macd_hist"],
            "entry_flag": entry.astype(float),
            "long_ma": long_ma,
            "above_long_ma": above_long.astype(float),
            "price_exit": exit_flag.astype(float),
        },
        index=output_index,
    )
    return raw, diagnostics


def build_candidate_raw_weight(
    spec: CandidateSpec,
    *,
    target: pd.Series,
    qqq: pd.Series,
    bars_per_day: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build final raw target weight before executable no-lookahead shifting."""
    params = spec.params
    output_index = target.index.intersection(qqq.index)
    target = target.loc[output_index]
    qqq = qqq.loc[output_index]
    base_raw, base_diag = build_base_signal(
        qqq,
        output_index=output_index,
        bars_per_day=bars_per_day,
        params=params,
    )

    stopped_raw, stop_diag = raw_with_peak_drawdown_stop(
        base_raw.rename(TARGET_TICKER),
        target,
        stop_drawdown=float(params["peak_stop_drawdown"]),
    )
    profit_lock_scheme = list(params["profit_lock_scheme"])
    if profit_lock_scheme:
        profit_weight = trade_profit_lock_tiers(
            stopped_raw.rename(TARGET_TICKER),
            target,
            thresholds_to_weights=profit_lock_scheme,
        ).rename(TARGET_TICKER)
    else:
        profit_weight = stopped_raw.rename(TARGET_TICKER)

    mr_features = qqq_mean_reversion_features(
        qqq,
        bars_per_day=bars_per_day,
        long_ma_days=int(float(params["long_ma_days"])),
    )
    dynamic_trim = dynamic_pre100_distance_trim_rebuy_cap(
        profit_weight.gt(0.0).astype(float),
        target,
        mr_features,
        activation_gain=float(params["q100_activation_gain"]),
        threshold_quantile=float(params["q100_distance_quantile"]),
        trim_weight=float(params["q100_trim_weight"]),
        reentry_rule=str(params["q100_reentry_rule"]),
    )
    q100_weight = apply_cap(profit_weight, dynamic_trim.weights).rename(TARGET_TICKER)

    slope_days = int(params["bear_filter_slope_days"])
    bear_features = bear_market_features(
        qqq,
        bars_per_day=bars_per_day,
        long_ma_days=int(float(params["long_ma_days"])),
        slope_days=(slope_days,),
    )
    bear_filter = bear_reentry_filter_raw(
        q100_weight.gt(0.0).astype(float),
        bear_features,
        distance_buffer=float(params["bear_filter_buffer"]),
        slope_days=slope_days,
        require_short_gt_medium=bool(params["bear_filter_require_20gt50"]),
    )
    final_weight = apply_cap(q100_weight, bear_filter.weights).rename(TARGET_TICKER)
    diagnostics = pd.DataFrame(
        {
            "base_raw": base_raw,
            "stopped_raw": stopped_raw,
            "profit_weight": profit_weight,
            "q100_weight": q100_weight,
            "final_raw_weight": final_weight,
            "stop_trigger": stop_diag["stop_trigger"],
            "q100_trigger": dynamic_trim.diagnostics["overlay_trigger"],
            "bear_blocked_entry": bear_filter.diagnostics["blocked_entry"],
            "macd_hist": base_diag["macd_hist"],
            "long_ma": base_diag["long_ma"],
        },
        index=output_index,
    )
    return final_weight, diagnostics


def executable_candidate_weights(
    raw_weight: pd.Series,
    *,
    config: Any,
    target_ticker: str = TARGET_TICKER,
) -> pd.DataFrame:
    """Convert raw target weight into executable no-lookahead weights."""
    raw_weights = raw_weight.to_frame(target_ticker)
    return executable_weights(raw_weights, config=config).fillna(0.0)


def evaluate_weight_window(
    *,
    name: str,
    family: str,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    config: Any,
    args: argparse.Namespace,
    parameters: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    segment: str = "window",
) -> tuple[dict[str, Any], pd.Series]:
    """Evaluate one candidate/benchmark over a specific start/end window."""
    mask = returns.index >= start
    if end is not None:
        mask &= returns.index <= end
    sample_returns = returns.loc[mask]
    sample_weights = weights.reindex(sample_returns.index).fillna(0.0)
    cols = [column for column in sample_weights.columns if column in sample_returns.columns]
    sample_weights = sample_weights[cols]
    sample_returns = sample_returns[cols]

    after_tax, taxes, turnover, cash_interest, cash_weight = simulate_after_tax_portfolio_with_cash_yield(
        sample_returns,
        sample_weights,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        tax_rate=args.short_term_tax_rate,
        cash_annual_yield=args.cash_annual_yield,
        annualization=config.backtest.annualization,
        cash_interest_tax_rate=args.cash_interest_tax_rate,
    )
    metrics = calculate_metrics(
        after_tax,
        turnover=turnover,
        weights=sample_weights.sum(axis=1),
        annualization=config.backtest.annualization,
    )
    for threshold in (20, 30, 40, 50):
        metrics[f"drawdown_episodes_gt_{threshold}pct"] = drawdown_episode_count(
            after_tax,
            threshold=-threshold / 100.0,
        )
    metrics["dd_episodes_gt_20_30_40_50pct"] = (
        f"{metrics['drawdown_episodes_gt_20pct']}/"
        f"{metrics['drawdown_episodes_gt_30pct']}/"
        f"{metrics['drawdown_episodes_gt_40pct']}/"
        f"{metrics['drawdown_episodes_gt_50pct']}"
    )
    years = len(after_tax) / config.backtest.annualization
    metrics.update(
        {
            "name": name,
            "family": family,
            "segment": segment,
            "start_date": start,
            "end_date": sample_returns.index.max() if not sample_returns.empty else pd.NaT,
            "bars": int(len(after_tax)),
            "parameters": json.dumps(parameters, sort_keys=True, default=str),
            "tax_timing": TAX_TIMING,
            "final_return": metrics["cumulative_return"],
            "average_cash_weight": float(cash_weight.mean()) if not cash_weight.empty else np.nan,
            "tax_paid_pct_initial_capital": float(taxes.sum()) if not taxes.empty else 0.0,
            "cash_interest_pct_initial_capital": float(cash_interest.sum()) if not cash_interest.empty else 0.0,
            "trades_per_year": metrics["number_of_trades"] / years if years > 0 else np.nan,
        }
    )
    return metrics, after_tax.rename(name)


def simulate_weight_path(
    *,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    config: Any,
    args: argparse.Namespace,
) -> dict[str, pd.Series]:
    """Simulate one full after-tax path once for fast window slicing."""
    sample_weights = weights.reindex(returns.index).fillna(0.0)
    cols = [column for column in sample_weights.columns if column in returns.columns]
    sample_weights = sample_weights[cols]
    sample_returns = returns[cols]
    after_tax, taxes, turnover, cash_interest, cash_weight = simulate_after_tax_portfolio_with_cash_yield(
        sample_returns,
        sample_weights,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        tax_rate=args.short_term_tax_rate,
        cash_annual_yield=args.cash_annual_yield,
        annualization=config.backtest.annualization,
        cash_interest_tax_rate=args.cash_interest_tax_rate,
    )
    return {
        "after_tax": after_tax,
        "taxes": taxes,
        "turnover": turnover,
        "cash_interest": cash_interest,
        "cash_weight": cash_weight,
        "weights_sum": sample_weights.sum(axis=1),
    }


def metrics_from_simulated_slice(
    *,
    name: str,
    family: str,
    simulation: dict[str, pd.Series],
    config: Any,
    parameters: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    segment: str = "window",
) -> tuple[dict[str, Any], pd.Series]:
    """Calculate metrics from an already-simulated path slice."""
    returns = simulation["after_tax"]
    mask = returns.index >= start
    if end is not None:
        mask &= returns.index <= end
    sliced = returns.loc[mask].fillna(0.0)
    turnover = simulation["turnover"].reindex(sliced.index).fillna(0.0)
    weights = simulation["weights_sum"].reindex(sliced.index).fillna(0.0)
    cash_weight = simulation["cash_weight"].reindex(sliced.index).fillna(0.0)
    taxes = simulation["taxes"].reindex(sliced.index).fillna(0.0)
    cash_interest = simulation["cash_interest"].reindex(sliced.index).fillna(0.0)

    metrics = calculate_metrics(
        sliced,
        turnover=turnover,
        weights=weights,
        annualization=config.backtest.annualization,
    )
    for threshold in (20, 30, 40, 50):
        metrics[f"drawdown_episodes_gt_{threshold}pct"] = drawdown_episode_count(
            sliced,
            threshold=-threshold / 100.0,
        )
    metrics["dd_episodes_gt_20_30_40_50pct"] = (
        f"{metrics['drawdown_episodes_gt_20pct']}/"
        f"{metrics['drawdown_episodes_gt_30pct']}/"
        f"{metrics['drawdown_episodes_gt_40pct']}/"
        f"{metrics['drawdown_episodes_gt_50pct']}"
    )
    years = len(sliced) / config.backtest.annualization
    metrics.update(
        {
            "name": name,
            "family": family,
            "segment": segment,
            "start_date": start,
            "end_date": sliced.index.max() if not sliced.empty else pd.NaT,
            "bars": int(len(sliced)),
            "parameters": json.dumps(parameters, sort_keys=True, default=str),
            "tax_timing": TAX_TIMING,
            "final_return": metrics["cumulative_return"],
            "average_cash_weight": float(cash_weight.mean()) if not cash_weight.empty else np.nan,
            "tax_paid_pct_initial_capital": float(taxes.sum()) if not taxes.empty else 0.0,
            "cash_interest_pct_initial_capital": (
                float(cash_interest.sum()) if not cash_interest.empty else 0.0
            ),
            "trades_per_year": metrics["number_of_trades"] / years if years > 0 else np.nan,
        }
    )
    return metrics, sliced.rename(name)


def benchmark_weights(index: pd.DatetimeIndex, ticker: str) -> pd.DataFrame:
    """Return buy-and-hold benchmark weights."""
    return pd.DataFrame({ticker: pd.Series(1.0, index=index, dtype=float)})


def annual_tax_payment_summary(
    *,
    name: str,
    family: str,
    simulation: dict[str, pd.Series],
    source_segment: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize annual tax-payment dates and audit year-end-only tax timing."""
    taxes = simulation["taxes"].fillna(0.0).sort_index()
    cash_interest = simulation["cash_interest"].reindex(taxes.index).fillna(0.0).sort_index()
    if taxes.empty:
        empty = pd.DataFrame(
            columns=[
                "name",
                "family",
                "source_segment",
                "tax_timing",
                "year",
                "tax_payment_timestamp",
                "tax_paid_pct_initial_capital",
                "cash_interest_earned_pct_initial_capital",
                "positive_tax_payment",
            ]
        )
        return empty, {
            "name": name,
            "family": family,
            "source_segment": source_segment,
            "tax_timing": TAX_TIMING,
            "audit_pass": True,
            "tax_payment_year_count": 0,
            "positive_tax_payment_count": 0,
            "non_year_end_tax_payment_count": 0,
            "total_tax_paid_pct_initial_capital": 0.0,
            "total_cash_interest_earned_pct_initial_capital": 0.0,
        }

    years = pd.Series(taxes.index.year, index=taxes.index, name="year")
    year_end_timestamps = taxes.groupby(years).apply(lambda series: series.index.max())
    tax_by_year = taxes.groupby(years).sum()
    cash_interest_by_year = cash_interest.groupby(years).sum()
    rows: list[dict[str, Any]] = []
    for year, timestamp in year_end_timestamps.items():
        tax_paid = float(tax_by_year.loc[year])
        rows.append(
            {
                "name": name,
                "family": family,
                "source_segment": source_segment,
                "tax_timing": TAX_TIMING,
                "year": int(year),
                "tax_payment_timestamp": pd.Timestamp(timestamp),
                "tax_paid_pct_initial_capital": tax_paid,
                "cash_interest_earned_pct_initial_capital": float(cash_interest_by_year.get(year, 0.0)),
                "positive_tax_payment": bool(abs(tax_paid) > 1e-12),
            }
        )
    annual = pd.DataFrame(rows)

    allowed_tax_dates = set(pd.Timestamp(timestamp) for timestamp in year_end_timestamps.to_list())
    positive_tax_dates = taxes.loc[taxes.abs().gt(1e-12)].index
    non_year_end_dates = [timestamp for timestamp in positive_tax_dates if pd.Timestamp(timestamp) not in allowed_tax_dates]
    audit = {
        "name": name,
        "family": family,
        "source_segment": source_segment,
        "tax_timing": TAX_TIMING,
        "audit_pass": len(non_year_end_dates) == 0,
        "tax_payment_year_count": int(annual["positive_tax_payment"].sum()) if not annual.empty else 0,
        "positive_tax_payment_count": int(len(positive_tax_dates)),
        "non_year_end_tax_payment_count": int(len(non_year_end_dates)),
        "first_tax_payment_date": positive_tax_dates.min() if len(positive_tax_dates) else pd.NaT,
        "last_tax_payment_date": positive_tax_dates.max() if len(positive_tax_dates) else pd.NaT,
        "total_tax_paid_pct_initial_capital": float(taxes.sum()),
        "total_cash_interest_earned_pct_initial_capital": float(cash_interest.sum()),
    }
    return annual, audit


def robustness_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate start-date CV rows into one robustness table per candidate."""
    candidate_rows = metrics.loc[metrics["family"].ne("benchmark")].copy()
    grouped = candidate_rows.groupby(["name", "family"], dropna=False)
    summary = grouped.agg(
        median_annualized_return=("annualized_return", "median"),
        p10_annualized_return=("annualized_return", lambda s: float(s.quantile(0.10))),
        min_annualized_return=("annualized_return", "min"),
        median_sharpe=("sharpe_ratio", "median"),
        worst_max_drawdown=("max_drawdown", "min"),
        median_max_drawdown=("max_drawdown", "median"),
        median_trades_per_year=("trades_per_year", "median"),
        max_trades_per_year=("trades_per_year", "max"),
        median_exposure=("exposure_percentage", "median"),
        max_dd50_episodes=("drawdown_episodes_gt_50pct", "max"),
        starts_tested=("start_date", "nunique"),
    ).reset_index()
    summary["tax_timing"] = TAX_TIMING
    summary["robustness_score"] = (
        summary["p10_annualized_return"]
        + 0.35 * summary["median_annualized_return"]
        + 0.50 * summary["worst_max_drawdown"]
        - 0.01 * (summary["median_trades_per_year"] - 8.0).clip(lower=0.0)
    )
    baseline = summary.loc[summary["name"].eq(CURRENT_PREFERRED_NAME)]
    if not baseline.empty:
        base = baseline.iloc[0]
        summary["delta_median_ann_vs_preferred"] = (
            summary["median_annualized_return"] - float(base["median_annualized_return"])
        )
        summary["delta_p10_ann_vs_preferred"] = (
            summary["p10_annualized_return"] - float(base["p10_annualized_return"])
        )
        summary["delta_worst_dd_vs_preferred"] = summary["worst_max_drawdown"] - float(
            base["worst_max_drawdown"]
        )
    return summary.sort_values(
        ["robustness_score", "p10_annualized_return", "worst_max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
