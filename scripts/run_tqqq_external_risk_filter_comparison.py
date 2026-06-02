#!/usr/bin/env python
"""Compare external-data risk filters for synthetic TQQQ fast/slow strategy."""

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

from run_tqqq_tax_slippage_analysis import after_tax_returns_annual_net
from trend_following.config import load_config
from trend_following.data_validation import read_price_file
from trend_following.metrics import calculate_metrics, metrics_to_frame
from trend_following.regime import (
    align_daily_regimes_to_intraday,
    classify_regimes,
    compute_regime_features,
    hourly_fast_entry_slow_exit_state_machine,
)
from trend_following.signals import limit_trades_per_day, make_executable_positions
from trend_following.utils import ensure_directory, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default="QQQ_3X_CALC")
    parser.add_argument("--regime-ticker", default="QQQ")
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--regime-daily-dir", default="data/raw/alpha_vantage_daily_adjusted")
    parser.add_argument("--daily-etf-dir", default="data/raw/alpha_vantage_daily_adjusted")
    parser.add_argument("--indicator-dir", default="data/raw/market_indicators")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--output-prefix", default="tqqq_external_risk_filter_comparison")
    return parser.parse_args()


def _returns_from_prices(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if not returns.empty:
        if isinstance(returns, pd.DataFrame):
            returns.iloc[0] = 0.0
        else:
            returns.iloc[0] = 0.0
    return returns


def _load_close(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def _rolling_percentile_of_last(values: np.ndarray) -> float:
    last = values[-1]
    finite = values[np.isfinite(values)]
    if not np.isfinite(last) or finite.size == 0:
        return np.nan
    return float((finite <= last).mean())


def _daily_flag_to_intraday(
    flag_known_today: pd.Series,
    intraday_index: pd.DatetimeIndex,
) -> pd.Series:
    """Align a daily boolean flag already shifted to be known before today."""
    daily = flag_known_today.astype("boolean").fillna(False).astype(bool).copy()
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    intraday_dates = intraday_index.tz_localize(None).normalize()
    unique_dates = pd.DatetimeIndex(intraday_dates.unique()).sort_values()
    aligned_by_date = daily.reindex(unique_dates, method="ffill")
    return pd.Series(
        aligned_by_date.reindex(intraday_dates).fillna(False).to_numpy(dtype=bool),
        index=intraday_index,
        dtype=bool,
    )


def _known_today(raw_close_based_flag: pd.Series) -> pd.Series:
    """Use yesterday's close-based flag as today's executable risk estimate."""
    return raw_close_based_flag.astype("boolean").shift(1).fillna(False).astype(bool)


def _sma_break(series: pd.Series, window: int) -> pd.Series:
    sma = series.rolling(window, min_periods=window).mean()
    return series.lt(sma)


def build_daily_risk_filters(
    *,
    daily_etf_dir: Path,
    indicator_dir: Path,
) -> pd.DataFrame:
    """Build external daily risk filters, shifted so each row uses D-1 data."""
    filters: dict[str, pd.Series] = {}

    # 1) ETF breadth proxy for Nasdaq/tech risk: broad risk-on universe loses participation.
    breadth_tickers = ["SPY", "QQQ", "IWM", "XLK", "SMH", "SOXX", "FDN", "IGV", "XLY", "XLF", "XLV", "XLE"]
    breadth_prices: dict[str, pd.Series] = {}
    for ticker in breadth_tickers:
        path = daily_etf_dir / f"{ticker}.parquet"
        if path.exists():
            breadth_prices[ticker] = _load_close(path, ticker)
    breadth = pd.DataFrame(breadth_prices).sort_index()
    above_50 = breadth.gt(breadth.rolling(50, min_periods=50).mean())
    available = breadth.notna() & breadth.rolling(50, min_periods=50).mean().notna()
    breadth_pct = above_50.sum(axis=1) / available.sum(axis=1).replace(0, np.nan)
    raw_breadth_risk = breadth_pct.lt(0.50) & available.sum(axis=1).ge(5)
    filters["breadth_proxy_pct_above_50d_lt_50"] = _known_today(raw_breadth_risk)

    # 2) Credit stress: high-yield underperforms investment-grade credit.
    hyg = _load_close(daily_etf_dir / "HYG.parquet", "HYG")
    lqd = _load_close(daily_etf_dir / "LQD.parquet", "LQD")
    credit_ratio = (hyg / lqd).rename("HYG_LQD")
    raw_credit_risk = _sma_break(credit_ratio, 100) & credit_ratio.pct_change(20).lt(0)
    filters["credit_hyg_lqd_ratio_below_100d_and_falling"] = _known_today(raw_credit_risk)

    # 3) Semiconductor leadership: SMH weak versus QQQ.
    smh = _load_close(daily_etf_dir / "SMH.parquet", "SMH")
    qqq = _load_close(daily_etf_dir / "QQQ.parquet", "QQQ")
    smh_qqq = (smh / qqq).rename("SMH_QQQ")
    raw_semis_risk = _sma_break(smh_qqq, 100) & smh_qqq.pct_change(20).lt(0)
    filters["semis_smh_qqq_ratio_below_100d_and_falling"] = _known_today(raw_semis_risk)

    # 4) Defensive rotation: QQQ weak versus utilities/consumer staples.
    xlu = _load_close(daily_etf_dir / "XLU.parquet", "XLU")
    xlp = _load_close(daily_etf_dir / "XLP.parquet", "XLP")
    qqq_xlu = (qqq / xlu).rename("QQQ_XLU")
    qqq_xlp = (qqq / xlp).rename("QQQ_XLP")
    raw_defensive_risk = (
        (_sma_break(qqq_xlu, 100) & qqq_xlu.pct_change(20).lt(0))
        | (_sma_break(qqq_xlp, 100) & qqq_xlp.pct_change(20).lt(0))
    )
    filters["defensive_rotation_qqq_vs_xlu_xlp_breakdown"] = _known_today(raw_defensive_risk)

    # 5) Rates shock proxy: TLT sells off quickly.
    tlt = _load_close(daily_etf_dir / "TLT.parquet", "TLT")
    raw_rates_risk = tlt.pct_change(20, fill_method=None).lt(-0.08)
    filters["rates_tlt_20d_return_lt_minus_8pct"] = _known_today(raw_rates_risk)

    # 6) Nasdaq implied vol: VXN high percentile.
    vxn_path = indicator_dir / "VXN.parquet"
    if vxn_path.exists():
        vxn = _load_close(vxn_path, "VXN")
        vxn_pct = vxn.rolling(252, min_periods=252).apply(_rolling_percentile_of_last, raw=True)
        filters["vxn_percentile_gt_90"] = _known_today(vxn_pct.gt(0.90))

    # 7) Volatility term structure proxy: VIX above VIX3M.
    vix_path = indicator_dir / "VIX.parquet"
    vix3m_path = indicator_dir / "VIX3M.parquet"
    if vix_path.exists() and vix3m_path.exists():
        vix = _load_close(vix_path, "VIX")
        vix3m = _load_close(vix3m_path, "VIX3M")
        raw_backwardation = (vix / vix3m).gt(1.0)
        filters["vix_term_structure_backwardation_vix_gt_vix3m"] = _known_today(
            raw_backwardation
        )

    # 8) Bond volatility: MOVE high percentile.
    move_path = indicator_dir / "MOVE.parquet"
    if move_path.exists():
        move = _load_close(move_path, "MOVE")
        move_pct = move.rolling(252, min_periods=252).apply(_rolling_percentile_of_last, raw=True)
        filters["move_percentile_gt_90"] = _known_today(move_pct.gt(0.90))

    # 9) Treasury-yield shock: 10Y yield rising fast.
    tnx_path = indicator_dir / "TNX.parquet"
    if tnx_path.exists():
        tnx = _load_close(tnx_path, "TNX")
        filters["tnx_20d_change_gt_75bp"] = _known_today(tnx.diff(20).gt(0.75))

    return pd.DataFrame(filters).sort_index()


def _run_strategy(
    *,
    label: str,
    risk_intraday: pd.Series | None,
    target_prices: pd.DataFrame,
    target_returns: pd.Series,
    daily_trend_intraday: pd.Series,
    params: dict[str, Any],
    config,
    transaction_cost_bps: float,
    slippage_bps: float,
    tax_rate: float,
) -> tuple[dict[str, Any], pd.Series, pd.Series, pd.Series]:
    target_ticker = str(params["target_ticker"])
    risk = pd.Series(False, index=target_prices.index) if risk_intraday is None else risk_intraday
    allowed = daily_trend_intraday.reindex(target_prices.index).fillna(False).astype(bool)
    allowed = allowed & ~risk.reindex(target_prices.index).fillna(False).astype(bool)

    raw_signal = hourly_fast_entry_slow_exit_state_machine(
        target_prices[target_ticker],
        allowed_regime=allowed,
        params=params,
    )
    raw_weights = raw_signal.to_frame(target_ticker)
    positions = make_executable_positions(
        raw_weights,
        execution_delay_days=config.backtest.execution_delay_days,
        return_convention=config.backtest.return_convention,
    )
    positions = limit_trades_per_day(
        positions,
        max_trades_per_day=config.backtest.max_trades_per_day,
    )[target_ticker].reindex(target_returns.index).fillna(0.0)

    after_tax, pretax, tax_drag, turnover = after_tax_returns_annual_net(
        target_returns,
        positions,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        tax_rate=tax_rate,
    )
    metrics = calculate_metrics(
        after_tax,
        turnover=turnover,
        weights=positions,
        annualization=config.backtest.annualization,
    )
    metrics.update(
        {
            "name": label,
            "strategy": "tqqq_external_risk_filter_after_tax",
            "segment": "full_sample",
            "parameters": json.dumps(params, sort_keys=True),
            "risk_filter": label,
            "risk_bar_percentage": float(risk.reindex(target_returns.index).fillna(False).mean()),
            "pretax_cumulative_return": float((1.0 + pretax).prod() - 1.0),
            "tax_paid_pct_initial_capital": float(
                tax_drag.mul((1.0 + after_tax).cumprod().shift(1).fillna(1.0)).sum()
            ),
        }
    )
    return metrics, after_tax, positions, risk


def _drawdown_episode_count(returns: pd.Series, threshold: float = -0.20) -> int:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    in_episode = False
    crossed = False
    count = 0
    for value in drawdown:
        if not in_episode and value < -1e-12:
            in_episode = True
            crossed = value <= threshold
        elif in_episode:
            crossed = crossed or value <= threshold
            if value >= -1e-12:
                if crossed:
                    count += 1
                in_episode = False
                crossed = False
    if in_episode and crossed:
        count += 1
    return count


def _plot_results(
    return_streams: dict[str, pd.Series],
    metrics: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 6,
) -> None:
    strategy_metrics = metrics[metrics["strategy"].ne("benchmark")].copy()
    selected = (
        strategy_metrics.sort_values(["sharpe_ratio", "max_drawdown"], ascending=[False, False])
        .head(top_n)["name"]
        .tolist()
    )
    if "base_no_extra_filter" not in selected:
        selected = ["base_no_extra_filter"] + selected[: top_n - 1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for name in selected:
        returns = return_streams[name]
        equity = (1.0 + returns.fillna(0.0)).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        equity.plot(ax=axes[0], label=name, linewidth=1.35)
        drawdown.plot(ax=axes[1], label=name, linewidth=1.35)
    axes[0].set_title("After-tax equity curves")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("After-tax drawdowns")
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
    regime_daily_dir = resolve_path(config.root, args.regime_daily_dir)
    daily_etf_dir = resolve_path(config.root, args.daily_etf_dir)
    indicator_dir = resolve_path(config.root, args.indicator_dir)
    benchmark_dir = resolve_path(config.root, args.benchmark_raw_dir)

    target_ticker = args.target_ticker
    regime_ticker = args.regime_ticker
    target_price = _load_close(target_dir / f"{target_ticker}.parquet", target_ticker)
    qqq_benchmark_price = _load_close(benchmark_dir / f"{regime_ticker}.parquet", regime_ticker)
    target_prices = target_price.to_frame()
    common_index = target_prices.index.intersection(qqq_benchmark_price.index)
    target_prices = target_prices.loc[common_index]
    target_returns = _returns_from_prices(target_prices[target_ticker]).loc[common_index]

    daily_price = _load_close(regime_daily_dir / f"{regime_ticker}.parquet", regime_ticker)
    daily_prices = daily_price.to_frame()
    daily_returns = _returns_from_prices(daily_prices)
    params = dict(config.strategies.regime_switch)
    params.update(
        {
            "target_ticker": target_ticker,
            "regime_ticker": regime_ticker,
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
    features = compute_regime_features(
        daily_prices,
        daily_returns,
        regime_ticker=regime_ticker,
        params=params,
    )
    daily_regimes = classify_regimes(features, params=params)
    daily_trend_intraday = align_daily_regimes_to_intraday(
        daily_regimes,
        target_prices.index,
        lag_days=int(params.get("daily_regime_lag_days", 1)),
        fill_method=params.get("daily_regime_fill_method", "ffill"),
    ).fillna("neutral").eq("trend")

    daily_filters = build_daily_risk_filters(
        daily_etf_dir=daily_etf_dir,
        indicator_dir=indicator_dir,
    )
    metric_rows: list[dict[str, Any]] = []
    return_streams: dict[str, pd.Series] = {}
    weights_table = pd.DataFrame(index=target_returns.index)
    risk_table = pd.DataFrame(index=target_returns.index)

    filters: dict[str, pd.Series | None] = {"base_no_extra_filter": None}
    filters.update({column: daily_filters[column] for column in daily_filters.columns})

    for label, daily_flag in filters.items():
        risk_intraday = (
            None if daily_flag is None else _daily_flag_to_intraday(daily_flag, target_prices.index)
        )
        metrics, returns, positions, risk = _run_strategy(
            label=label,
            risk_intraday=risk_intraday,
            target_prices=target_prices,
            target_returns=target_returns,
            daily_trend_intraday=daily_trend_intraday,
            params=params,
            config=config,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            tax_rate=args.short_term_tax_rate,
        )
        metrics["drawdown_episodes_gt_20pct"] = _drawdown_episode_count(returns)
        metric_rows.append(metrics)
        return_streams[label] = returns
        weights_table[label] = positions
        risk_table[label] = risk.reindex(target_returns.index).fillna(False).astype(int)

    # Add buy-and-hold QQQ for context.
    benchmark_returns = _returns_from_prices(qqq_benchmark_price.loc[common_index])
    benchmark_metrics = calculate_metrics(
        benchmark_returns,
        annualization=config.backtest.annualization,
    )
    benchmark_metrics.update(
        {
            "name": f"Buy & Hold {regime_ticker}",
            "strategy": "benchmark",
            "segment": "full_sample",
            "parameters": "{}",
            "risk_filter": "benchmark",
            "risk_bar_percentage": np.nan,
            "pretax_cumulative_return": float((1.0 + benchmark_returns).prod() - 1.0),
            "tax_paid_pct_initial_capital": 0.0,
            "drawdown_episodes_gt_20pct": _drawdown_episode_count(benchmark_returns),
        }
    )
    metric_rows.append(benchmark_metrics)
    return_streams[f"Buy & Hold {regime_ticker}"] = benchmark_returns

    metrics = metrics_to_frame(metric_rows)
    tables_dir = config.reports.tables_dir
    figures_dir = config.reports.figures_dir
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    filters_path = tables_dir / f"{args.output_prefix}_daily_filters.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    risks_path = tables_dir / f"{args.output_prefix}_intraday_risk_flags.csv"
    plot_path = figures_dir / f"{args.output_prefix}_top_filters_equity_drawdown.png"

    metrics.to_csv(metrics_path, index=False)
    daily_filters.to_csv(filters_path)
    weights_table.to_csv(weights_path)
    risk_table.to_csv(risks_path)
    _plot_results(
        return_streams,
        metrics,
        plot_path,
        title="External-data risk filters for synthetic TQQQ strategy",
    )

    print(f"Metrics saved to {metrics_path}")
    print(f"Daily filters saved to {filters_path}")
    print(f"Weights saved to {weights_path}")
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
        ]
    ]
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
