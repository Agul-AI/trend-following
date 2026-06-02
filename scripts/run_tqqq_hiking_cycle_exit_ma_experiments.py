#!/usr/bin/env python
"""Test shorter MA exits only during Fed hiking cycles for synthetic TQQQ."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tqqq_entry_signal_comparison import (  # noqa: E402
    _confirmed,
    _days_to_bars,
    _returns_from_prices,
    executable_weights,
    macd_components,
)
from run_tqqq_macd_entry_experiments import count_profit_lock_hits  # noqa: E402
from run_tqqq_position_risk_sizing_experiments import (  # noqa: E402
    drawdown_episode_count,
    simulate_after_tax_portfolio,
)
from run_tqqq_tiered_sizing_experiments import trade_profit_lock_tiers  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.metrics import calculate_metrics, metrics_to_frame  # noqa: E402
from trend_following.regime import (  # noqa: E402
    align_daily_regimes_to_intraday,
    classify_regimes,
    compute_regime_features,
)
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

LOCK_SCHEMES: dict[str, list[tuple[float, float]]] = {
    "full_no_lock": [],
    "new_lock_200_300": [(2.00, 0.75), (3.00, 0.50)],
}

# Effective dates for FOMC hiking cycles. The 1999-2000 cycle is only partially
# covered by this backtest because the intraday test starts in January 2000.
HIKING_CYCLES: tuple[tuple[str, str, str], ...] = (
    ("1999-2000 hike cycle partial", "1999-06-30", "2000-05-16"),
    ("2004-2006 hike cycle", "2004-06-30", "2006-06-29"),
    ("2015-2018 normalization/hike cycle", "2015-12-17", "2018-12-20"),
    ("2022-2023 hike cycle", "2022-03-17", "2023-07-27"),
)


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
    parser.add_argument("--output-prefix", default="tqqq_hiking_cycle_exit_ma_experiments")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def hiking_cycle_known_flag(index: pd.DatetimeIndex, *, lag_days: int = 1) -> pd.Series:
    """Return an intraday flag for hiking-cycle status known before today.

    We first label daily dates by whether they fall inside a predefined Fed
    hiking cycle, then shift that daily label by one observed trading day before
    mapping it to intraday bars. This preserves the project's convention that a
    decision on date D should only use information known through D-1.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    normalized = index.tz_localize(None).normalize()
    unique_dates = pd.DatetimeIndex(pd.unique(normalized)).sort_values()
    daily = pd.Series(False, index=unique_dates)
    for _, start, end in HIKING_CYCLES:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        daily |= (daily.index >= start_ts) & (daily.index <= end_ts)
    known = daily.shift(lag_days, fill_value=False) if lag_days > 0 else daily
    return pd.Series(known.reindex(normalized).to_numpy(dtype=bool), index=index, name="hiking_cycle_known")


def dynamic_hiking_exit_signal(
    *,
    entry_price: pd.Series,
    exit_price: pd.Series,
    output_index: pd.DatetimeIndex,
    allowed_regime: pd.Series,
    hiking_cycle_known: pd.Series,
    bars_per_day: int,
    average_type: str,
    macd_unit: str,
    hiking_exit_ma_days: int,
    entry_confirm_bars: int = 2,
    exit_confirm_bars: int = 3,
) -> tuple[pd.Series, pd.DataFrame]:
    """MACD entry, 200-day exit normally, shorter exit MA during hike cycles."""
    entry_clean = entry_price.reindex(output_index).astype(float).sort_index()
    exit_clean = exit_price.reindex(output_index).astype(float).sort_index()
    gate = allowed_regime.reindex(output_index).fillna(False).astype(bool)
    hiking_flag = hiking_cycle_known.reindex(output_index).fillna(False).astype(bool)

    if macd_unit == "days":
        fast_window = _days_to_bars(12, bars_per_day)
        slow_window = _days_to_bars(26, bars_per_day)
        signal_window = _days_to_bars(9, bars_per_day)
    elif macd_unit == "bars":
        fast_window = 12
        slow_window = 26
        signal_window = 9
    else:
        raise ValueError("macd_unit must be days or bars")

    macd = macd_components(
        entry_clean,
        fast_window=fast_window,
        slow_window=slow_window,
        signal_window=signal_window,
        average_type=average_type,
    )
    entry = _confirmed(gate & macd["macd_hist"].gt(0.0), entry_confirm_bars)

    ma_200_window = _days_to_bars(200, bars_per_day)
    hike_ma_window = _days_to_bars(hiking_exit_ma_days, bars_per_day)
    ma_200 = exit_clean.rolling(window=ma_200_window, min_periods=ma_200_window).mean()
    hike_ma = exit_clean.rolling(window=hike_ma_window, min_periods=hike_ma_window).mean()
    active_exit_ma = hike_ma.where(hiking_flag, ma_200)
    active_exit_days = pd.Series(200, index=output_index, dtype=float).where(
        ~hiking_flag,
        float(hiking_exit_ma_days),
    )

    price_exit = _confirmed(exit_clean.lt(active_exit_ma), exit_confirm_bars)
    regime_exit = ~gate

    state = 0.0
    values: list[float] = []
    for entry_now, regime_exit_now, price_exit_now in zip(
        entry,
        regime_exit,
        price_exit,
        strict=False,
    ):
        if state == 0.0:
            if bool(entry_now):
                state = 1.0
        elif bool(regime_exit_now) or bool(price_exit_now):
            state = 0.0
        values.append(state)

    raw = pd.Series(values, index=output_index, name="raw_signal", dtype=float)
    diagnostics = pd.DataFrame(
        {
            "entry_price": entry_clean,
            "exit_price": exit_clean,
            "macd_hist": macd["macd_hist"],
            "entry_flag": entry.astype(float),
            "exit_ma": active_exit_ma,
            "active_exit_ma_days": active_exit_days,
            "price_exit": price_exit.astype(float),
            "allowed_regime": gate.astype(float),
            "hiking_cycle_known": hiking_flag.astype(float),
        },
        index=output_index,
    )
    return raw, diagnostics


def _apply_lock(raw: pd.Series, traded_price: pd.Series, scheme: list[tuple[float, float]]) -> pd.Series:
    if not scheme:
        return raw.astype(float).copy()
    return trade_profit_lock_tiers(raw, traded_price, thresholds_to_weights=scheme)


def _drawdown(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


def _max_drawdown_when(returns: pd.Series, flag: pd.Series) -> float:
    dd = _drawdown(returns)
    subset = dd[flag.reindex(dd.index).fillna(False).astype(bool)]
    return float(subset.min()) if not subset.empty else np.nan


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
    target_returns = _returns_from_prices(target_prices)
    qqq_returns = _returns_from_prices(qqq_prices)
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
    hike_flag = hiking_cycle_known_flag(common, lag_days=args.hike_cycle_lag_days)

    sources = {
        "tqqq": target_prices[args.target_ticker],
        "qqq": qqq_prices[args.benchmark_ticker],
    }
    candidate_specs = (
        ("qqq", "tqqq", "new_lock_200_300"),
        ("tqqq", "tqqq", "new_lock_200_300"),
        ("qqq", "qqq", "full_no_lock"),
    )
    exit_ma_days_grid = (200, 150, 100, 50)

    raw_variants: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, pd.DataFrame] = {}
    meta_by_name: dict[str, dict[str, Any]] = {}
    for entry_source, exit_source, lock_label in candidate_specs:
        for exit_ma_days in exit_ma_days_grid:
            raw, diag = dynamic_hiking_exit_signal(
                entry_price=sources[entry_source],
                exit_price=sources[exit_source],
                output_index=common,
                allowed_regime=allowed_regime,
                hiking_cycle_known=hike_flag,
                bars_per_day=bars_per_day,
                average_type=args.average_type,
                macd_unit=args.macd_unit,
                hiking_exit_ma_days=exit_ma_days,
            )
            raw = raw.rename(args.target_ticker)
            scheme = LOCK_SCHEMES[lock_label]
            weights = _apply_lock(raw, target_prices[args.target_ticker], scheme)
            name = (
                f"entry_{entry_source}__exit_{exit_source}"
                f"__{lock_label}__hike_exit_ma_{exit_ma_days}"
            )
            raw_variants[name] = weights.to_frame(args.target_ticker)
            diagnostics[name] = diag
            meta_by_name[name] = {
                "entry_source": entry_source,
                "exit_source": exit_source,
                "lock_scheme": lock_label,
                "hiking_exit_ma_days": exit_ma_days,
                "non_hiking_exit_ma_days": 200,
                "average_type": args.average_type,
                "macd_unit": args.macd_unit,
                "entry_rule": "hist_gt_0",
                "hike_cycle_lag_days": args.hike_cycle_lag_days,
                "lock_hit_200_count": (
                    float(count_profit_lock_hits(raw, target_prices[args.target_ticker], threshold=2.00))
                    if scheme
                    else np.nan
                ),
                "lock_hit_300_count": (
                    float(count_profit_lock_hits(raw, target_prices[args.target_ticker], threshold=3.00))
                    if scheme
                    else np.nan
                ),
            }

    metric_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_out: dict[str, pd.Series] = {}
    all_returns = pd.concat([target_returns, qqq_returns], axis=1).loc[common]

    for name, raw_weights in raw_variants.items():
        weights = executable_weights(raw_weights, config=config).reindex(common).fillna(0.0)
        after_tax, pretax, taxes_paid, turnover = simulate_after_tax_portfolio(
            all_returns[[args.target_ticker]],
            weights[[args.target_ticker]],
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            tax_rate=args.short_term_tax_rate,
        )
        metrics = calculate_metrics(
            after_tax,
            turnover=turnover,
            weights=weights.sum(axis=1),
            annualization=config.backtest.annualization,
        )
        metrics.update(
            {
                "name": name,
                "strategy": "hiking_cycle_exit_ma_experiment",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "transaction_cost_bps": args.transaction_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "short_term_tax_rate": args.short_term_tax_rate,
                        "base_params": params,
                        "hiking_cycles": HIKING_CYCLES,
                    },
                    sort_keys=True,
                ),
                "pretax_cumulative_return": float((1.0 + pretax).prod() - 1.0),
                "tax_paid_pct_initial_capital": float(taxes_paid.sum()),
                "drawdown_episodes_gt_30pct": drawdown_episode_count(after_tax, threshold=-0.30),
                "drawdown_episodes_gt_40pct": drawdown_episode_count(after_tax, threshold=-0.40),
                "drawdown_episodes_gt_50pct": drawdown_episode_count(after_tax, threshold=-0.50),
                "max_drawdown_while_hike_flag_known": _max_drawdown_when(after_tax, hike_flag),
                **meta_by_name[name],
            }
        )
        metric_rows.append(metrics)
        returns_by_name[name] = after_tax
        weights_out[name] = weights.sum(axis=1)

    metrics = metrics_to_frame(metric_rows)
    metrics["dd_episodes_gt_30_40_50pct"] = metrics.apply(
        lambda row: (
            f"{int(row['drawdown_episodes_gt_30pct'])}/"
            f"{int(row['drawdown_episodes_gt_40pct'])}/"
            f"{int(row['drawdown_episodes_gt_50pct'])}"
        ),
        axis=1,
    )
    metrics["candidate"] = (
        "entry_"
        + metrics["entry_source"]
        + "__exit_"
        + metrics["exit_source"]
        + "__"
        + metrics["lock_scheme"]
    )

    tables_dir = config.reports.tables_dir
    ensure_directory(tables_dir)
    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    compact_path = tables_dir / f"{args.output_prefix}_compact.csv"
    returns_path = tables_dir / f"{args.output_prefix}_after_tax_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    diagnostics_path = tables_dir / f"{args.output_prefix}_diagnostics.parquet"

    compact_cols = [
        "candidate",
        "hiking_exit_ma_days",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "max_drawdown_while_hike_flag_known",
        "number_of_trades",
        "exposure_percentage",
        "dd_episodes_gt_30_40_50pct",
        "lock_hit_200_count",
        "lock_hit_300_count",
    ]
    compact = metrics[compact_cols].sort_values(
        ["candidate", "hiking_exit_ma_days"],
        ascending=[True, False],
    )

    metrics.to_csv(metrics_path, index=False)
    compact.to_csv(compact_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(weights_out).to_csv(weights_path)
    pd.concat(diagnostics, axis=1).to_parquet(diagnostics_path)

    print(f"Metrics saved to {metrics_path}")
    print(f"Compact table saved to {compact_path}")
    print(f"After-tax returns saved to {returns_path}")
    print(f"Weights saved to {weights_path}")
    print(f"Diagnostics saved to {diagnostics_path}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
