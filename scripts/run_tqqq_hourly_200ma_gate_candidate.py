#!/usr/bin/env python
"""Serious candidate test: QQQ hourly 200MA entry/exit gate, no daily regime gate.

Candidate rule:
- Signal source: QQQ hourly price.
- Target exposure: synthetic TQQQ (`QQQ_3X_CALC`).
- No daily QQQ regime gate.
- Entry: QQQ MACD histogram > 0 AND QQQ hourly close > QQQ 200-day hourly MA.
- Exit: QQQ hourly close < QQQ 200-day hourly MA.
- Same no-lookahead executable-weight shift and max-one-trade-per-day convention.
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tqqq_daily_gate_ablation import no_daily_gate_hourly_ma_gate_signal  # noqa: E402
from run_tqqq_entry_signal_comparison import (  # noqa: E402
    _drawdown,
    _equity,
    _returns_from_prices,
    executable_weights,
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
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--output-prefix", default="tqqq_hourly_200ma_gate_candidate")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def _add_dd_counts(metrics: dict[str, Any], returns: pd.Series) -> None:
    metrics["drawdown_episodes_gt_20pct"] = drawdown_episode_count(returns, threshold=-0.20)
    metrics["drawdown_episodes_gt_30pct"] = drawdown_episode_count(returns, threshold=-0.30)
    metrics["drawdown_episodes_gt_40pct"] = drawdown_episode_count(returns, threshold=-0.40)
    metrics["drawdown_episodes_gt_50pct"] = drawdown_episode_count(returns, threshold=-0.50)
    metrics["dd_episodes_gt_20_30_40_50pct"] = (
        f"{metrics['drawdown_episodes_gt_20pct']}/"
        f"{metrics['drawdown_episodes_gt_30pct']}/"
        f"{metrics['drawdown_episodes_gt_40pct']}/"
        f"{metrics['drawdown_episodes_gt_50pct']}"
    )


def _drawdown_episodes(returns: pd.Series, *, min_threshold: float = -0.30) -> pd.DataFrame:
    """Return drawdown episodes whose trough breaches min_threshold."""
    equity = _equity(returns)
    drawdown = equity / equity.cummax() - 1.0
    episodes: list[dict[str, Any]] = []
    in_episode = False
    start = None
    trough = None
    trough_dd = 0.0
    peak_equity = None
    for timestamp, dd_value in drawdown.items():
        if not in_episode and dd_value < -1e-12:
            in_episode = True
            start = timestamp
            trough = timestamp
            trough_dd = float(dd_value)
            peak_equity = float(equity.loc[:timestamp].cummax().iloc[-1])
        elif in_episode:
            if dd_value < trough_dd:
                trough = timestamp
                trough_dd = float(dd_value)
            if dd_value >= -1e-12:
                if trough_dd <= min_threshold:
                    episodes.append(
                        {
                            "start": start,
                            "trough": trough,
                            "recovery": timestamp,
                            "max_drawdown": trough_dd,
                            "peak_equity_before_episode": peak_equity,
                        }
                    )
                in_episode = False
                start = None
                trough = None
                trough_dd = 0.0
                peak_equity = None
    if in_episode and trough_dd <= min_threshold:
        episodes.append(
            {
                "start": start,
                "trough": trough,
                "recovery": pd.NaT,
                "max_drawdown": trough_dd,
                "peak_equity_before_episode": peak_equity,
            }
        )
    return pd.DataFrame(episodes).sort_values("max_drawdown").reset_index(drop=True)


def _segment_masks(index: pd.DatetimeIndex, train_end_date: str) -> dict[str, pd.Series]:
    train_end = pd.Timestamp(train_end_date)
    return {
        "full_sample": pd.Series(True, index=index),
        "in_sample": pd.Series(index <= train_end, index=index),
        "out_of_sample": pd.Series(index > train_end, index=index),
    }


def _plot(returns_by_name: dict[str, pd.Series], output_path: Path) -> None:
    selected = {
        key: value
        for key, value in returns_by_name.items()
        if key in {"current_preferred_daily_gate", "candidate_hourly_200ma_gate", "buy_hold_qqq"}
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for name, returns in selected.items():
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
    fig.suptitle("Candidate test: no daily gate, QQQ hourly 200MA entry/exit gate")
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _annual_returns(returns_by_name: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, returns in returns_by_name.items():
        grouped = returns.fillna(0.0).groupby(returns.index.year)
        for year, values in grouped:
            rows.append(
                {
                    "name": name,
                    "year": int(year),
                    "annual_return": float((1.0 + values).prod() - 1.0),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    target_dir = resolve_path(config.root, args.target_raw_dir)
    benchmark_dir = resolve_path(config.root, args.benchmark_raw_dir)
    daily_dir = resolve_path(config.root, args.daily_regime_raw_dir)
    train_end_date = args.train_end_date or str(config.backtest.train_end_date)

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
    raw_candidate, diag_candidate = no_daily_gate_hourly_ma_gate_signal(
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
        "current_preferred_daily_gate": raw_current.rename(args.target_ticker).to_frame(args.target_ticker),
        "candidate_hourly_200ma_gate": raw_candidate.rename(args.target_ticker).to_frame(args.target_ticker),
    }
    diagnostics = {
        "current_preferred_daily_gate": diag_current,
        "candidate_hourly_200ma_gate": diag_candidate,
    }

    full_returns_by_name: dict[str, pd.Series] = {}
    pretax_returns_by_name: dict[str, pd.Series] = {}
    taxes_by_name: dict[str, pd.Series] = {}
    turnover_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.Series] = {}
    metric_rows: list[dict[str, Any]] = []

    for name, raw_weights in raw_variants.items():
        weights = executable_weights(raw_weights, config=config).reindex(common).fillna(0.0)
        after_tax, pretax, taxes_paid, turnover = simulate_after_tax_portfolio(
            all_returns[[args.target_ticker]],
            weights[[args.target_ticker]],
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            tax_rate=args.short_term_tax_rate,
        )
        full_returns_by_name[name] = after_tax
        pretax_returns_by_name[name] = pretax
        taxes_by_name[name] = taxes_paid
        turnover_by_name[name] = turnover
        weights_by_name[name] = weights[args.target_ticker]

    benchmark_returns = all_returns[args.benchmark_ticker]
    full_returns_by_name["buy_hold_qqq"] = benchmark_returns

    masks = _segment_masks(common, train_end_date=train_end_date)
    for name, returns in full_returns_by_name.items():
        for segment, mask in masks.items():
            segment_returns = returns.loc[mask]
            if segment_returns.empty:
                continue
            turnover = turnover_by_name.get(name, pd.Series(0.0, index=common)).loc[mask]
            weights = weights_by_name.get(name, pd.Series(0.0, index=common)).loc[mask]
            metrics = calculate_metrics(
                segment_returns,
                turnover=turnover if name != "buy_hold_qqq" else None,
                weights=weights if name != "buy_hold_qqq" else None,
                annualization=config.backtest.annualization,
            )
            metrics.update(
                {
                    "name": name,
                    "strategy": "hourly_200ma_gate_candidate" if name != "buy_hold_qqq" else "benchmark",
                    "segment": segment,
                    "parameters": json.dumps(
                        {
                            "train_end_date": train_end_date,
                            "transaction_cost_bps": args.transaction_cost_bps,
                            "slippage_bps": args.slippage_bps,
                            "short_term_tax_rate": args.short_term_tax_rate,
                            "base_params": params,
                        },
                        sort_keys=True,
                    ),
                }
            )
            _add_dd_counts(metrics, segment_returns)
            metric_rows.append(metrics)

    metrics = metrics_to_frame(metric_rows)
    metrics = metrics.sort_values(["segment", "name"])

    dd_tables = []
    for name in ["current_preferred_daily_gate", "candidate_hourly_200ma_gate"]:
        episodes = _drawdown_episodes(full_returns_by_name[name], min_threshold=-0.30).head(20)
        episodes.insert(0, "name", name)
        dd_tables.append(episodes)
    drawdown_episodes = pd.concat(dd_tables, ignore_index=True) if dd_tables else pd.DataFrame()

    annual = _annual_returns(full_returns_by_name)

    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    compact_path = tables_dir / f"{args.output_prefix}_compact.csv"
    returns_path = tables_dir / f"{args.output_prefix}_after_tax_returns.csv"
    pretax_path = tables_dir / f"{args.output_prefix}_pretax_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    diagnostics_path = tables_dir / f"{args.output_prefix}_diagnostics.parquet"
    drawdown_path = tables_dir / f"{args.output_prefix}_drawdown_episodes_gt30.csv"
    annual_path = tables_dir / f"{args.output_prefix}_annual_returns.csv"
    plot_path = figures_dir / f"{args.output_prefix}_equity_drawdown.png"

    compact_cols = [
        "segment",
        "name",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "number_of_trades",
        "exposure_percentage",
        "dd_episodes_gt_20_30_40_50pct",
    ]
    compact = metrics[compact_cols].sort_values(["segment", "name"])

    metrics.to_csv(metrics_path, index=False)
    compact.to_csv(compact_path, index=False)
    pd.DataFrame(full_returns_by_name).to_csv(returns_path)
    pd.DataFrame(pretax_returns_by_name).to_csv(pretax_path)
    pd.DataFrame(weights_by_name).to_csv(weights_path)
    pd.concat(diagnostics, axis=1).to_parquet(diagnostics_path)
    drawdown_episodes.to_csv(drawdown_path, index=False)
    annual.to_csv(annual_path, index=False)
    _plot(full_returns_by_name, plot_path)

    print(f"Metrics saved to {metrics_path}")
    print(f"Compact table saved to {compact_path}")
    print(f"After-tax returns saved to {returns_path}")
    print(f"Weights saved to {weights_path}")
    print(f"Drawdown episodes saved to {drawdown_path}")
    print(f"Annual returns saved to {annual_path}")
    print(f"Plot saved to {plot_path}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
