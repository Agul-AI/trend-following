#!/usr/bin/env python
"""Test QQQ moving-average derivative/combo filters for synthetic TQQQ drawdowns."""

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
    hourly_fast_entry_slow_exit_state_machine,
)
from trend_following.signals import limit_trades_per_day, make_executable_positions  # noqa: E402
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default="QQQ_3X_CALC")
    parser.add_argument("--regime-ticker", default="QQQ")
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--regime-daily-dir", default="data/raw/alpha_vantage_daily_adjusted")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--output-prefix", default="tqqq_ma_derivative_filter_experiments")
    return parser.parse_args()


def _load_close(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def _returns_from_prices(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if not returns.empty:
        if isinstance(returns, pd.DataFrame):
            returns.iloc[0] = 0.0
        else:
            returns.iloc[0] = 0.0
    return returns


def _equity(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _drawdown(returns: pd.Series) -> pd.Series:
    equity = _equity(returns)
    return equity / equity.cummax() - 1.0


def _rolling_percentile_last(values: np.ndarray) -> float:
    last = values[-1]
    finite = values[np.isfinite(values)]
    if not np.isfinite(last) or finite.size == 0:
        return np.nan
    return float((finite <= last).mean())


def _known_today(raw_close_based_flag: pd.Series) -> pd.Series:
    """Shift a close-D flag so row D uses only data through D-1."""
    return raw_close_based_flag.astype("boolean").shift(1).fillna(False).astype(bool)


def _daily_flag_to_intraday(flag_known_today: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    daily = flag_known_today.astype("boolean").fillna(False).astype(bool).copy()
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    intraday_dates = index.tz_localize(None).normalize()
    unique_dates = pd.DatetimeIndex(intraday_dates.unique()).sort_values()
    aligned_by_date = daily.reindex(unique_dates, method="ffill")
    return pd.Series(
        aligned_by_date.reindex(intraday_dates).fillna(False).to_numpy(dtype=bool),
        index=index,
        dtype=bool,
    )


def build_ma_features(price: pd.Series) -> pd.DataFrame:
    """Build daily QQQ MA derivative and combo features."""
    sma20 = price.rolling(20, min_periods=20).mean()
    sma50 = price.rolling(50, min_periods=50).mean()
    sma200 = price.rolling(200, min_periods=200).mean()
    slope20_5 = sma20.pct_change(5, fill_method=None)
    slope50_10 = sma50.pct_change(10, fill_method=None)
    slope200_20 = sma200.pct_change(20, fill_method=None)
    slope200_63 = sma200.pct_change(63, fill_method=None)
    slope200_pctile = slope200_20.rolling(252, min_periods=252).apply(
        _rolling_percentile_last,
        raw=True,
    )

    fast_health = pd.concat(
        [
            price / sma20 - 1.0,
            sma20 / sma50 - 1.0,
            slope20_5,
            slope50_10,
        ],
        axis=1,
    ).mean(axis=1)
    trend_score = pd.concat(
        [
            price.gt(sma20),
            sma20.gt(sma50),
            sma50.gt(sma200),
            slope20_5.gt(0.0),
            slope50_10.gt(0.0),
            slope200_20.gt(0.0),
        ],
        axis=1,
    ).sum(axis=1)

    return pd.DataFrame(
        {
            "price": price,
            "sma20": sma20,
            "sma50": sma50,
            "sma200": sma200,
            "price_vs_sma20": price / sma20 - 1.0,
            "sma20_vs_sma50": sma20 / sma50 - 1.0,
            "sma50_vs_sma200": sma50 / sma200 - 1.0,
            "slope20_5": slope20_5,
            "slope50_10": slope50_10,
            "slope200_20": slope200_20,
            "slope200_63": slope200_63,
            "slope200_pctile": slope200_pctile,
            "slope200_accel_20": slope200_20 - slope200_20.shift(20),
            "fast_health": fast_health,
            "trend_score": trend_score,
        }
    )


def build_ma_filters(features: pd.DataFrame) -> pd.DataFrame:
    """Return candidate risk filters shifted so today's row uses yesterday's close."""
    raw: dict[str, pd.Series] = {}
    raw["sma200_slope20_le_0"] = features["slope200_20"].le(0.0)
    raw["sma200_slope20_lt_0p5pct"] = features["slope200_20"].lt(0.005)
    raw["sma200_slope_pctile_lt_20"] = features["slope200_pctile"].lt(0.20)
    raw["sma200_decelerating_and_slope_lt_1pct"] = features["slope200_accel_20"].lt(
        0.0
    ) & features["slope200_20"].lt(0.01)
    raw["sma20_slope5_negative"] = features["slope20_5"].lt(0.0)
    raw["sma50_slope10_negative"] = features["slope50_10"].lt(0.0)
    raw["sma20_below_sma50"] = features["sma20"].lt(features["sma50"])
    raw["price_below_sma50"] = features["price"].lt(features["sma50"])
    raw["fast_health_lt_0"] = features["fast_health"].lt(0.0)
    raw["fast_health_lt_minus_1pct"] = features["fast_health"].lt(-0.01)
    raw["trend_score_le_4"] = features["trend_score"].le(4)
    raw["trend_score_le_5"] = features["trend_score"].le(5)
    raw["short_damage_price_below_20_and_slope20_neg"] = features["price"].lt(
        features["sma20"]
    ) & features["slope20_5"].lt(0.0)
    raw["mid_damage_price_below_50_and_20_below_50"] = features["price"].lt(
        features["sma50"]
    ) & features["sma20"].lt(features["sma50"])
    return pd.DataFrame({name: _known_today(flag) for name, flag in raw.items()})


def drawdown_episodes(returns: pd.Series, threshold: float = -0.20) -> pd.DataFrame:
    dd = _drawdown(returns)
    rows: list[dict[str, Any]] = []
    in_episode = False
    crossed = False
    start = None
    cross_date = None
    trough_date = None
    trough = 0.0
    for timestamp, value in dd.items():
        if not in_episode and value < -1e-12:
            in_episode = True
            crossed = value <= threshold
            start = timestamp
            cross_date = timestamp if crossed else None
            trough_date = timestamp
            trough = float(value)
        elif in_episode:
            if value < trough:
                trough = float(value)
                trough_date = timestamp
            if not crossed and value <= threshold:
                crossed = True
                cross_date = timestamp
            if value >= -1e-12:
                if crossed:
                    rows.append(
                        {
                            "start": start,
                            "first_below_20": cross_date,
                            "trough_date": trough_date,
                            "recovery_date": timestamp,
                            "max_drawdown": trough,
                        }
                    )
                in_episode = False
                crossed = False
    if in_episode and crossed:
        rows.append(
            {
                "start": start,
                "first_below_20": cross_date,
                "trough_date": trough_date,
                "recovery_date": pd.NaT,
                "max_drawdown": trough,
            }
        )
    return pd.DataFrame(rows)


def pre_drawdown_warning_stats(
    base_returns: pd.Series,
    risk_intraday: pd.Series,
    lookback_days: int = 30,
) -> dict[str, float]:
    episodes = drawdown_episodes(base_returns, threshold=-0.20)
    if episodes.empty:
        return {"pre_dd_hit_rate": np.nan, "pre_dd_hit_count": 0}
    hits = 0
    for first_below in episodes["first_below_20"]:
        start = pd.Timestamp(first_below) - pd.Timedelta(days=lookback_days)
        window = risk_intraday.loc[(risk_intraday.index >= start) & (risk_intraday.index < first_below)]
        if bool(window.any()):
            hits += 1
    return {
        "pre_dd_hit_rate": hits / len(episodes),
        "pre_dd_hit_count": hits,
    }


def run_variant(
    *,
    name: str,
    risk_intraday: pd.Series | None,
    target_prices: pd.DataFrame,
    returns: pd.DataFrame,
    daily_trend_intraday: pd.Series,
    params: dict[str, Any],
    config,
    transaction_cost_bps: float,
    slippage_bps: float,
    tax_rate: float,
) -> tuple[dict[str, Any], pd.Series, pd.Series]:
    target = str(params["target_ticker"])
    risk = pd.Series(False, index=target_prices.index) if risk_intraday is None else risk_intraday
    allowed = daily_trend_intraday.reindex(target_prices.index).fillna(False).astype(bool)
    allowed = allowed & ~risk.reindex(target_prices.index).fillna(False).astype(bool)

    raw = hourly_fast_entry_slow_exit_state_machine(
        target_prices[target],
        allowed_regime=allowed,
        params=params,
    ).to_frame(target)
    weights = make_executable_positions(
        raw,
        execution_delay_days=config.backtest.execution_delay_days,
        return_convention=config.backtest.return_convention,
    )
    weights = limit_trades_per_day(
        weights,
        max_trades_per_day=config.backtest.max_trades_per_day,
    ).reindex(returns.index).fillna(0.0)
    after_tax, pretax, taxes_paid, turnover = simulate_after_tax_portfolio(
        returns[[target]],
        weights[[target]],
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        tax_rate=tax_rate,
    )
    metrics = calculate_metrics(
        after_tax,
        turnover=turnover,
        weights=weights[target],
        annualization=config.backtest.annualization,
    )
    metrics.update(
        {
            "name": name,
            "strategy": "ma_derivative_combo_filter",
            "segment": "full_sample",
            "parameters": json.dumps(params, sort_keys=True),
            "risk_bar_percentage": float(risk.reindex(returns.index).fillna(False).mean()),
            "pretax_cumulative_return": float((1.0 + pretax).prod() - 1.0),
            "tax_paid_pct_initial_capital": float(taxes_paid.sum()),
            "drawdown_episodes_gt_20pct": drawdown_episode_count(after_tax),
        }
    )
    return metrics, after_tax, risk


def plot_top(
    returns_by_name: dict[str, pd.Series],
    metrics: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    selected = (
        metrics[metrics["strategy"].ne("benchmark")]
        .sort_values(["sharpe_ratio", "max_drawdown"], ascending=[False, False])
        .head(8)["name"]
        .tolist()
    )
    if "base_no_ma_filter" not in selected:
        selected = ["base_no_ma_filter"] + selected[:7]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for name in selected:
        returns = returns_by_name[name]
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
    fig.suptitle(title)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    target_dir = resolve_path(config.root, args.target_raw_dir)
    daily_dir = resolve_path(config.root, args.regime_daily_dir)
    benchmark_dir = resolve_path(config.root, args.benchmark_raw_dir)

    target = _load_close(target_dir / f"{args.target_ticker}.parquet", args.target_ticker)
    qqq_intraday = _load_close(benchmark_dir / f"{args.regime_ticker}.parquet", args.regime_ticker)
    qqq_daily = _load_close(daily_dir / f"{args.regime_ticker}.parquet", args.regime_ticker)

    common = target.index.intersection(qqq_intraday.index)
    target_prices = target.loc[common].to_frame()
    target_returns = _returns_from_prices(target_prices)
    qqq_returns = _returns_from_prices(qqq_intraday.loc[common])
    daily_prices = qqq_daily.to_frame()
    daily_returns = _returns_from_prices(daily_prices)

    params = dict(config.strategies.regime_switch)
    params.update(
        {
            "target_ticker": args.target_ticker,
            "regime_ticker": args.regime_ticker,
            "sma_window": 200,
            "use_variance_ratio_for_trend": False,
            "state_machine_entry_ma_days": 20.0,
            "state_machine_exit_ma_days": 200.0,
            "state_machine_entry_slope_days": 5.0,
            "state_machine_entry_confirm_bars": 2,
            "state_machine_exit_confirm_bars": 3,
            "state_machine_entry_buffer": 0.0,
            "state_machine_exit_buffer": 0.0,
        }
    )
    daily_features = compute_regime_features(
        daily_prices,
        daily_returns,
        regime_ticker=args.regime_ticker,
        params=params,
    )
    daily_regimes = classify_regimes(daily_features, params=params)
    daily_trend_intraday = align_daily_regimes_to_intraday(
        daily_regimes,
        target_prices.index,
        lag_days=int(params.get("daily_regime_lag_days", 1)),
        fill_method=params.get("daily_regime_fill_method", "ffill"),
    ).fillna("neutral").eq("trend")

    ma_features = build_ma_features(qqq_daily)
    ma_filters = build_ma_filters(ma_features)

    metrics_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    risks_by_name: dict[str, pd.Series] = {}

    variants: dict[str, pd.Series | None] = {"base_no_ma_filter": None}
    variants.update({column: ma_filters[column] for column in ma_filters.columns})

    base_returns: pd.Series | None = None
    for name, daily_flag in variants.items():
        risk_intraday = (
            None
            if daily_flag is None
            else _daily_flag_to_intraday(daily_flag, target_prices.index)
        )
        metrics, returns, risk = run_variant(
            name=name,
            risk_intraday=risk_intraday,
            target_prices=target_prices,
            returns=target_returns,
            daily_trend_intraday=daily_trend_intraday,
            params=params,
            config=config,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            tax_rate=args.short_term_tax_rate,
        )
        if name == "base_no_ma_filter":
            base_returns = returns
        returns_by_name[name] = returns
        risks_by_name[name] = risk.reindex(target_returns.index).fillna(False)
        metrics_rows.append(metrics)

    if base_returns is None:
        raise RuntimeError("base variant did not run")
    for row in metrics_rows:
        risk = risks_by_name[row["name"]]
        row.update(pre_drawdown_warning_stats(base_returns, risk))

    benchmark_metrics = calculate_metrics(
        qqq_returns.loc[common],
        annualization=config.backtest.annualization,
    )
    benchmark_metrics.update(
        {
            "name": "buy_hold_qqq",
            "strategy": "benchmark",
            "segment": "full_sample",
            "parameters": "{}",
            "risk_bar_percentage": np.nan,
            "pretax_cumulative_return": float((1.0 + qqq_returns.loc[common]).prod() - 1.0),
            "tax_paid_pct_initial_capital": 0.0,
            "drawdown_episodes_gt_20pct": drawdown_episode_count(qqq_returns.loc[common]),
            "pre_dd_hit_rate": np.nan,
            "pre_dd_hit_count": np.nan,
        }
    )
    metrics_rows.append(benchmark_metrics)
    returns_by_name["buy_hold_qqq"] = qqq_returns.loc[common]

    metrics = metrics_to_frame(metrics_rows)
    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    features_path = tables_dir / f"{args.output_prefix}_daily_ma_features.csv"
    filters_path = tables_dir / f"{args.output_prefix}_daily_ma_filters.csv"
    returns_path = tables_dir / f"{args.output_prefix}_after_tax_returns.csv"
    risks_path = tables_dir / f"{args.output_prefix}_intraday_risk_flags.csv"
    plot_path = figures_dir / f"{args.output_prefix}_top_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    ma_features.to_csv(features_path)
    ma_filters.to_csv(filters_path)
    pd.DataFrame(returns_by_name).to_csv(returns_path)
    pd.DataFrame(risks_by_name).to_csv(risks_path)
    plot_top(
        returns_by_name,
        metrics,
        plot_path,
        title="QQQ MA derivative/combo filters for synthetic TQQQ",
    )

    print(f"Metrics saved to {metrics_path}")
    print(f"Daily MA features saved to {features_path}")
    print(f"Daily MA filters saved to {filters_path}")
    print(f"After-tax returns saved to {returns_path}")
    print(f"Intraday risk flags saved to {risks_path}")
    print(f"Plot saved to {plot_path}")
    compact = metrics[
        [
            "name",
            "cumulative_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "number_of_trades",
            "exposure_percentage",
            "risk_bar_percentage",
            "drawdown_episodes_gt_20pct",
            "pre_dd_hit_rate",
        ]
    ].sort_values("sharpe_ratio", ascending=False)
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
