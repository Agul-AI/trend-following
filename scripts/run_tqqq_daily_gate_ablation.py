#!/usr/bin/env python
"""Ablate the daily QQQ trend-regime gate for the preferred synthetic TQQQ strategy."""

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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tqqq_entry_signal_comparison import (  # noqa: E402
    _confirmed,
    _days_to_bars,
    _drawdown,
    _equity,
    _returns_from_prices,
    executable_weights,
    macd_components,
)
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
    parser.add_argument("--output-prefix", default="tqqq_daily_gate_ablation")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def no_daily_gate_hourly_ma_gate_signal(
    *,
    entry_price: pd.Series,
    exit_price: pd.Series,
    output_index: pd.DatetimeIndex,
    bars_per_day: int,
    average_type: str,
    macd_unit: str,
    entry_confirm_bars: int = 2,
    exit_confirm_bars: int = 3,
    exit_ma_days: float = 200.0,
    macd_fast_days: float = 12.0,
    macd_slow_days: float = 26.0,
    macd_signal_days: float = 9.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """MACD entry gated only by the same hourly 200-day MA used for exit."""
    entry_clean = entry_price.reindex(output_index).astype(float).sort_index()
    exit_clean = exit_price.reindex(output_index).astype(float).sort_index()

    if macd_unit == "days":
        fast_window = _days_to_bars(macd_fast_days, bars_per_day)
        slow_window = _days_to_bars(macd_slow_days, bars_per_day)
        signal_window = _days_to_bars(macd_signal_days, bars_per_day)
    elif macd_unit == "bars":
        fast_window = int(macd_fast_days)
        slow_window = int(macd_slow_days)
        signal_window = int(macd_signal_days)
    else:
        raise ValueError("macd_unit must be days or bars")
    if fast_window >= slow_window:
        raise ValueError(f"MACD fast window must be < slow window, got {fast_window} >= {slow_window}")

    macd = macd_components(
        entry_clean,
        fast_window=fast_window,
        slow_window=slow_window,
        signal_window=signal_window,
        average_type=average_type,
    )
    exit_window = _days_to_bars(exit_ma_days, bars_per_day)
    exit_ma = exit_clean.rolling(window=exit_window, min_periods=exit_window).mean()
    above_exit_ma = exit_clean.gt(exit_ma)

    entry = _confirmed(macd["macd_hist"].gt(0.0) & above_exit_ma, entry_confirm_bars)
    price_exit = _confirmed(exit_clean.lt(exit_ma), exit_confirm_bars)

    state = 0.0
    values: list[float] = []
    for entry_now, price_exit_now in zip(entry, price_exit, strict=False):
        if state == 0.0:
            if bool(entry_now):
                state = 1.0
        elif bool(price_exit_now):
            state = 0.0
        values.append(state)

    raw = pd.Series(values, index=output_index, name="raw_signal", dtype=float)
    diagnostics = pd.DataFrame(
        {
            "entry_price": entry_clean,
            "exit_price": exit_clean,
            "macd_hist": macd["macd_hist"],
            "entry_flag": entry.astype(float),
            "exit_ma": exit_ma,
            "above_exit_ma": above_exit_ma.astype(float),
            "price_exit": price_exit.astype(float),
        },
        index=output_index,
    )
    return raw, diagnostics


def _add_dd_counts(metrics: dict[str, Any], returns: pd.Series) -> None:
    metrics["drawdown_episodes_gt_30pct"] = drawdown_episode_count(returns, threshold=-0.30)
    metrics["drawdown_episodes_gt_40pct"] = drawdown_episode_count(returns, threshold=-0.40)
    metrics["drawdown_episodes_gt_50pct"] = drawdown_episode_count(returns, threshold=-0.50)
    metrics["dd_episodes_gt_30_40_50pct"] = (
        f"{metrics['drawdown_episodes_gt_30pct']}/"
        f"{metrics['drawdown_episodes_gt_40pct']}/"
        f"{metrics['drawdown_episodes_gt_50pct']}"
    )


def _plot(returns_by_name: dict[str, pd.Series], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for name, returns in returns_by_name.items():
        _equity(returns).plot(ax=axes[0], label=name, linewidth=1.25)
        _drawdown(returns).plot(ax=axes[1], label=name, linewidth=1.25)
    axes[0].set_title("After-tax equity")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_title("After-tax drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Daily regime gate ablation: QQQ signal / synthetic TQQQ exposure")
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


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
        [_returns_from_prices(target_prices), _returns_from_prices(qqq_prices)],
        axis=1,
    ).loc[common]

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

    daily_prices = daily_qqq.to_frame()
    daily_returns = _returns_from_prices(daily_prices)
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
    daily_gate = intraday_regimes.eq("trend")
    no_gate = pd.Series(True, index=common, name="no_gate")

    raw_current, diag_current = mixed_source_signal(
        entry_price=qqq_prices[args.benchmark_ticker],
        exit_price=qqq_prices[args.benchmark_ticker],
        output_index=common,
        allowed_regime=daily_gate,
        bars_per_day=bars_per_day,
        average_type=args.average_type,
        macd_unit=args.macd_unit,
        entry_confirm_bars=2,
        exit_confirm_bars=3,
        exit_ma_days=200,
    )
    raw_no_gate, diag_no_gate = mixed_source_signal(
        entry_price=qqq_prices[args.benchmark_ticker],
        exit_price=qqq_prices[args.benchmark_ticker],
        output_index=common,
        allowed_regime=no_gate,
        bars_per_day=bars_per_day,
        average_type=args.average_type,
        macd_unit=args.macd_unit,
        entry_confirm_bars=2,
        exit_confirm_bars=3,
        exit_ma_days=200,
    )
    raw_hourly_gate, diag_hourly_gate = no_daily_gate_hourly_ma_gate_signal(
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

    raw_variants = {
        "current_daily_200d_regime_gate": raw_current.rename(args.target_ticker).to_frame(args.target_ticker),
        "no_daily_gate_exit_only": raw_no_gate.rename(args.target_ticker).to_frame(args.target_ticker),
        "no_daily_gate_hourly_200d_entry_gate": raw_hourly_gate.rename(args.target_ticker).to_frame(args.target_ticker),
    }
    diagnostics = {
        "current_daily_200d_regime_gate": diag_current,
        "no_daily_gate_exit_only": diag_no_gate,
        "no_daily_gate_hourly_200d_entry_gate": diag_hourly_gate,
    }

    metric_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.Series] = {}

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
            weights=weights[args.target_ticker],
            annualization=config.backtest.annualization,
        )
        metrics.update(
            {
                "name": name,
                "strategy": "daily_gate_ablation",
                "segment": "full_sample",
                "parameters": json.dumps(
                    {
                        "transaction_cost_bps": args.transaction_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "short_term_tax_rate": args.short_term_tax_rate,
                        "base_params": params,
                    },
                    sort_keys=True,
                ),
                "pretax_cumulative_return": float((1.0 + pretax).prod() - 1.0),
                "tax_paid_pct_initial_capital": float(taxes_paid.sum()),
            }
        )
        _add_dd_counts(metrics, after_tax)
        metric_rows.append(metrics)
        returns_by_name[name] = after_tax
        weights_by_name[name] = weights[args.target_ticker]

    benchmark_returns = all_returns[args.benchmark_ticker]
    benchmark_metrics = calculate_metrics(benchmark_returns, annualization=config.backtest.annualization)
    benchmark_metrics.update(
        {
            "name": "buy_hold_qqq",
            "strategy": "benchmark",
            "segment": "full_sample",
            "parameters": "{}",
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
    diagnostics_path = tables_dir / f"{args.output_prefix}_diagnostics.parquet"
    plot_path = figures_dir / f"{args.output_prefix}_equity_drawdown.png"

    compact_cols = [
        "name",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "number_of_trades",
        "exposure_percentage",
        "dd_episodes_gt_30_40_50pct",
    ]
    compact = metrics[compact_cols]

    metrics.to_csv(metrics_path, index=False)
    compact.to_csv(compact_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(weights_by_name).to_csv(weights_path)
    pd.concat(diagnostics, axis=1).to_parquet(diagnostics_path)
    _plot(returns_by_name, plot_path)

    print(f"Metrics saved to {metrics_path}")
    print(f"Compact table saved to {compact_path}")
    print(f"After-tax returns saved to {returns_path}")
    print(f"Weights saved to {weights_path}")
    print(f"Diagnostics saved to {diagnostics_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
