#!/usr/bin/env python
"""Compare mixed TQQQ/QQQ sources for MACD entry and 200MA exit."""

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
    _confirmed,
    _days_to_bars,
    _drawdown,
    _equity,
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
    parser.add_argument("--output-prefix", default="tqqq_mixed_entry_exit_source_comparison")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def mixed_source_signal(
    *,
    entry_price: pd.Series,
    exit_price: pd.Series,
    output_index: pd.DatetimeIndex,
    allowed_regime: pd.Series,
    bars_per_day: int,
    average_type: str,
    macd_unit: str,
    entry_confirm_bars: int = 2,
    exit_confirm_bars: int = 3,
    exit_ma_days: float = 200.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Raw signal with MACD hist>0 entry and 200MA exit from independent sources."""
    entry_clean = entry_price.reindex(output_index).astype(float).sort_index()
    exit_clean = exit_price.reindex(output_index).astype(float).sort_index()
    gate = allowed_regime.reindex(output_index).fillna(False).astype(bool)

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
    raw_entry = macd["macd_hist"].gt(0.0)
    entry = _confirmed(gate & raw_entry, entry_confirm_bars)

    exit_window = _days_to_bars(exit_ma_days, bars_per_day)
    exit_ma = exit_clean.rolling(window=exit_window, min_periods=exit_window).mean()
    price_exit = _confirmed(exit_clean.lt(exit_ma), exit_confirm_bars)
    regime_exit = ~gate

    state = 0.0
    values: list[float] = []
    for entry_now, regime_exit_now, price_exit_now in zip(
        entry, regime_exit, price_exit, strict=False
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
            "exit_ma": exit_ma,
            "price_exit": price_exit.astype(float),
            "allowed_regime": gate.astype(float),
        },
        index=output_index,
    )
    return raw, diagnostics


def _apply_lock(raw: pd.Series, traded_price: pd.Series, scheme: list[tuple[float, float]]) -> pd.Series:
    if not scheme:
        return raw.astype(float).copy()
    return trade_profit_lock_tiers(raw, traded_price, thresholds_to_weights=scheme)


def _plot_top(
    returns_by_name: dict[str, pd.Series],
    metrics: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 8,
) -> None:
    selected = (
        metrics[metrics["strategy"].ne("benchmark")]
        .sort_values(["sharpe_ratio", "max_drawdown"], ascending=[False, False])
        .head(top_n)["name"]
        .tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for name in selected:
        returns = returns_by_name[name]
        _equity(returns).plot(ax=axes[0], label=name, linewidth=1.15)
        _drawdown(returns).plot(ax=axes[1], label=name, linewidth=1.15)
    axes[0].set_title("After-tax equity")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("After-tax drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle(title)
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

    sources = {
        "tqqq": target_prices[args.target_ticker],
        "qqq": qqq_prices[args.benchmark_ticker],
    }

    raw_signals: dict[str, pd.Series] = {}
    diagnostics: dict[str, pd.DataFrame] = {}
    for entry_source, entry_price in sources.items():
        for exit_source, exit_price in sources.items():
            label = f"entry_{entry_source}__exit_{exit_source}"
            raw, diag = mixed_source_signal(
                entry_price=entry_price,
                exit_price=exit_price,
                output_index=common,
                allowed_regime=allowed_regime,
                bars_per_day=bars_per_day,
                average_type=args.average_type,
                macd_unit=args.macd_unit,
            )
            raw = raw.rename(args.target_ticker)
            raw_signals[label] = raw
            diagnostics[label] = diag

    raw_variants: dict[str, pd.DataFrame] = {}
    meta_by_name: dict[str, dict[str, Any]] = {}
    for source_label, raw in raw_signals.items():
        entry_source = source_label.split("__")[0].removeprefix("entry_")
        exit_source = source_label.split("__")[1].removeprefix("exit_")
        for lock_label, scheme in LOCK_SCHEMES.items():
            name = f"{source_label}__{lock_label}"
            weights = _apply_lock(raw, target_prices[args.target_ticker], scheme)
            raw_variants[name] = weights.to_frame(args.target_ticker)
            meta_by_name[name] = {
                "entry_source": entry_source,
                "exit_source": exit_source,
                "lock_scheme": lock_label,
                "average_type": args.average_type,
                "macd_unit": args.macd_unit,
                "entry_rule": "hist_gt_0",
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
                "strategy": "mixed_entry_exit_source_comparison",
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
                "drawdown_episodes_gt_30pct": drawdown_episode_count(after_tax, threshold=-0.30),
                "drawdown_episodes_gt_40pct": drawdown_episode_count(after_tax, threshold=-0.40),
                "drawdown_episodes_gt_50pct": drawdown_episode_count(after_tax, threshold=-0.50),
                **meta_by_name[name],
            }
        )
        metric_rows.append(metrics)
        returns_by_name[name] = after_tax
        weights_out[name] = weights.sum(axis=1)

    benchmark_metrics = calculate_metrics(
        qqq_returns[args.benchmark_ticker],
        annualization=config.backtest.annualization,
    )
    benchmark_metrics.update(
        {
            "name": "buy_hold_qqq",
            "strategy": "benchmark",
            "segment": "full_sample",
            "parameters": "{}",
            "pretax_cumulative_return": float((1.0 + qqq_returns[args.benchmark_ticker]).prod() - 1.0),
            "tax_paid_pct_initial_capital": 0.0,
            "drawdown_episodes_gt_30pct": drawdown_episode_count(qqq_returns[args.benchmark_ticker], threshold=-0.30),
            "drawdown_episodes_gt_40pct": drawdown_episode_count(qqq_returns[args.benchmark_ticker], threshold=-0.40),
            "drawdown_episodes_gt_50pct": drawdown_episode_count(qqq_returns[args.benchmark_ticker], threshold=-0.50),
            "entry_source": "none",
            "exit_source": "none",
            "lock_scheme": "none",
            "average_type": "",
            "macd_unit": "",
            "entry_rule": "",
            "lock_hit_200_count": np.nan,
            "lock_hit_300_count": np.nan,
        }
    )
    metric_rows.append(benchmark_metrics)
    returns_by_name["buy_hold_qqq"] = qqq_returns[args.benchmark_ticker]

    metrics = metrics_to_frame(metric_rows)
    metrics["dd_episodes_gt_30_40_50pct"] = metrics.apply(
        lambda row: (
            f"{int(row['drawdown_episodes_gt_30pct'])}/"
            f"{int(row['drawdown_episodes_gt_40pct'])}/"
            f"{int(row['drawdown_episodes_gt_50pct'])}"
        ),
        axis=1,
    )

    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    compact_path = tables_dir / f"{args.output_prefix}_compact.csv"
    returns_path = tables_dir / f"{args.output_prefix}_after_tax_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    diagnostics_path = tables_dir / f"{args.output_prefix}_diagnostics.parquet"
    plot_path = figures_dir / f"{args.output_prefix}_top_equity_drawdown.png"

    compact_cols = [
        "entry_source",
        "exit_source",
        "lock_scheme",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "number_of_trades",
        "exposure_percentage",
        "dd_episodes_gt_30_40_50pct",
        "lock_hit_200_count",
        "lock_hit_300_count",
    ]
    compact = metrics[metrics["strategy"].eq("mixed_entry_exit_source_comparison")][compact_cols]
    compact = compact.sort_values("sharpe_ratio", ascending=False)

    metrics.to_csv(metrics_path, index=False)
    compact.to_csv(compact_path, index=False)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(weights_out).to_csv(weights_path)
    pd.concat(diagnostics, axis=1).to_parquet(diagnostics_path)
    _plot_top(
        returns_by_name,
        metrics,
        plot_path,
        title="Synthetic TQQQ mixed MACD-entry / 200MA-exit sources",
    )

    print(f"Metrics saved to {metrics_path}")
    print(f"Compact table saved to {compact_path}")
    print(f"After-tax returns saved to {returns_path}")
    print(f"Weights saved to {weights_path}")
    print(f"Diagnostics saved to {diagnostics_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
