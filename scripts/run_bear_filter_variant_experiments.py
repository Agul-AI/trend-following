#!/usr/bin/env python
"""Run variants around the serious bear re-entry filter candidate.

This script starts from the q100 preferred strategy and the serious bear-filter
candidate, then tests:
- q100 activation levels from +50% through +150%.
- Robust bear-filter parameter variants.
- Partial-entry variants.
- Post-big-winner protection overlays.
- Bear-regime volatility caps.
- Bear-regime portfolio drawdown circuit breakers.
"""

from __future__ import annotations

import argparse
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

from run_preferred_drawdown_reduction_experiments import (  # noqa: E402
    _evaluate,
    _preferred_weight,
    _single_asset_weights,
)
from run_tqqq_daily_gate_ablation import no_daily_gate_hourly_ma_gate_signal  # noqa: E402
from run_tqqq_entry_signal_comparison import (  # noqa: E402
    _drawdown,
    _equity,
    _returns_from_prices,
    macd_components,
)
from run_tqqq_position_risk_sizing_experiments import drawdown_episode_count  # noqa: E402
from trend_following.bear_whipsaw import (  # noqa: E402
    bear_market_features,
    bear_reentry_filter_raw,
)
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.metrics import metrics_to_frame  # noqa: E402
from trend_following.risk_overlays import (  # noqa: E402
    apply_cap,
    dynamic_pre100_distance_trim_rebuy_cap,
    qqq_mean_reversion_features,
)
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

TARGET_TICKER = "QQQ_3X_CALC"
BENCHMARK_TICKER = "QQQ"
BASELINE_NAME = "q100_activation100_preferred"
SERIOUS_NAME = "serious_bear_filter_q100"
CURRENT_PREFERRED_NAME = "preferred_q110_best_robustness_bear_filter"
BEAR_WINDOW_START = pd.Timestamp("2007-10-01")
BEAR_WINDOW_END = pd.Timestamp("2009-12-31 23:59:59")
DD_PERIODS = {
    "2010": (pd.Timestamp("2010-01-01"), pd.Timestamp("2010-12-31 23:59:59")),
    "2004_2006": (pd.Timestamp("2004-01-01"), pd.Timestamp("2006-12-31 23:59:59")),
    "2007_2009": (pd.Timestamp("2007-01-01"), pd.Timestamp("2009-12-31 23:59:59")),
    "2011": (pd.Timestamp("2011-01-01"), pd.Timestamp("2011-12-31 23:59:59")),
    "2018_2019": (pd.Timestamp("2018-01-01"), pd.Timestamp("2019-12-31 23:59:59")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--target-ticker", default=TARGET_TICKER)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--target-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--benchmark-raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--output-prefix", default="bear_filter_variant")
    parser.add_argument("--site-dir", default="reports/site")
    return parser.parse_args()


def _load_price(path: Path, name: str) -> pd.Series:
    frame = read_price_file(path).sort_index()
    return frame["adj_close"].astype(float).rename(name)


def _days_to_bars(days: float, bars_per_day: int) -> int:
    return max(int(round(days * bars_per_day)), 1)


def _window_drawdown(returns: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    sample = returns.loc[(returns.index >= start) & (returns.index <= end)].fillna(0.0)
    if sample.empty:
        return float("nan")
    return float(_drawdown(sample).min())


def _worst_drawdown_rows(returns_by_name: dict[str, pd.Series], *, top_n: int = 5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, returns in returns_by_name.items():
        equity = _equity(returns.fillna(0.0))
        drawdown = equity / equity.cummax() - 1.0
        episodes: list[dict[str, Any]] = []
        in_dd = False
        peak = pd.NaT
        trough = pd.NaT
        trough_dd = 0.0
        for timestamp, value in drawdown.items():
            if not in_dd and value < 0:
                in_dd = True
                peak = equity.loc[:timestamp].idxmax()
                trough = timestamp
                trough_dd = float(value)
            elif in_dd:
                if value < trough_dd:
                    trough = timestamp
                    trough_dd = float(value)
                if value >= -1e-12:
                    episodes.append(
                        {
                            "strategy": name,
                            "peak": peak,
                            "trough": trough,
                            "recovery": timestamp,
                            "max_drawdown": trough_dd,
                            "calendar_days_peak_to_trough": int((trough - peak).days),
                        }
                    )
                    in_dd = False
        if in_dd:
            episodes.append(
                {
                    "strategy": name,
                    "peak": peak,
                    "trough": trough,
                    "recovery": pd.NaT,
                    "max_drawdown": trough_dd,
                    "calendar_days_peak_to_trough": int((trough - peak).days),
                }
            )
        ranked = sorted(episodes, key=lambda row: row["max_drawdown"])[:top_n]
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
            rows.append(row)
    return pd.DataFrame(rows)


def _add_dd_counts(metrics: dict[str, Any], returns: pd.Series) -> None:
    for threshold in (20, 30, 40, 50):
        metrics[f"drawdown_episodes_gt_{threshold}pct"] = drawdown_episode_count(
            returns,
            threshold=-threshold / 100.0,
        )
    metrics["dd_episodes_gt_20_30_40_50pct"] = (
        f"{metrics['drawdown_episodes_gt_20pct']}/"
        f"{metrics['drawdown_episodes_gt_30pct']}/"
        f"{metrics['drawdown_episodes_gt_40pct']}/"
        f"{metrics['drawdown_episodes_gt_50pct']}"
    )


def _eval_candidate(
    *,
    name: str,
    family: str,
    raw_weight: pd.Series | pd.DataFrame,
    returns: pd.DataFrame,
    config: Any,
    args: argparse.Namespace,
    parameters: dict[str, Any],
    diagnostic_count: int = 0,
) -> tuple[dict[str, Any], pd.Series, pd.DataFrame]:
    raw_weights = (
        raw_weight
        if isinstance(raw_weight, pd.DataFrame)
        else _single_asset_weights(raw_weight, args.target_ticker)
    )
    metrics, after_tax, weights = _evaluate(
        name=name,
        family=family,
        raw_weights=raw_weights,
        returns=returns,
        config=config,
        args=args,
        parameters=parameters,
    )
    _add_dd_counts(metrics, after_tax)
    metrics["bear_2007_2009_max_drawdown"] = _window_drawdown(
        after_tax,
        BEAR_WINDOW_START,
        BEAR_WINDOW_END,
    )
    for period_name, (start, end) in DD_PERIODS.items():
        metrics[f"{period_name}_max_drawdown"] = _window_drawdown(after_tax, start, end)
    metrics["overlay_trigger_count"] = diagnostic_count
    return metrics, after_tax, weights


def _q100_weight(
    *,
    profit_locked_weight: pd.Series,
    target_price: pd.Series,
    mr_features: pd.DataFrame,
    activation_gain: float,
) -> tuple[pd.Series, pd.DataFrame]:
    overlay = dynamic_pre100_distance_trim_rebuy_cap(
        profit_locked_weight.gt(0.0).astype(float),
        target_price,
        mr_features,
        activation_gain=activation_gain,
        threshold_quantile=1.0,
        trim_weight=0.50,
        reentry_rule="ma20",
    )
    return apply_cap(profit_locked_weight, overlay.weights).rename(
        profit_locked_weight.name
    ), overlay.diagnostics


def _serious_bear_filtered_weight(
    base_weight: pd.Series,
    features: pd.DataFrame,
    *,
    buffer: float = 0.01,
    slope_days: int = 20,
    require_short_gt_medium: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    overlay = bear_reentry_filter_raw(
        base_weight.gt(0.0).astype(float),
        features,
        distance_buffer=buffer,
        slope_days=slope_days,
        require_short_gt_medium=require_short_gt_medium,
    )
    return apply_cap(base_weight, overlay.weights).rename(base_weight.name), overlay.diagnostics


def _partial_bear_entry_cap(
    base_weight: pd.Series,
    features: pd.DataFrame,
    *,
    partial_weight: float,
    buffer: float,
    slope_days: int,
    require_short_gt_medium: bool,
) -> tuple[pd.Series, pd.DataFrame]:
    feat = features.reindex(base_weight.index)
    long_col = f"long_ma_slope_{slope_days}d"
    medium_col = f"medium_ma_slope_{slope_days}d"
    in_trade = False
    staged = False
    caps: list[float] = []
    staged_flags: list[bool] = []
    releases: list[bool] = []
    for timestamp, desired in base_weight.fillna(0.0).items():
        release = False
        if desired <= 0.0:
            in_trade = False
            staged = False
            caps.append(0.0)
            staged_flags.append(False)
            releases.append(False)
            continue
        row = feat.loc[timestamp]
        confirmation = (
            float(row["distance_to_long_ma"]) >= buffer
            and float(row[medium_col]) > 0.0
            and (not require_short_gt_medium or bool(row["short_gt_medium"]))
        )
        if not in_trade:
            long_slope = float(row[long_col]) if pd.notna(row[long_col]) else np.nan
            staged = np.isfinite(long_slope) and long_slope < 0.0 and not confirmation
            in_trade = True
        if staged and confirmation:
            staged = False
            release = True
        caps.append(partial_weight if staged else 1.0)
        staged_flags.append(staged)
        releases.append(release)
    cap = pd.Series(caps, index=base_weight.index, name="partial_bear_entry_cap")
    diagnostics = pd.DataFrame(
        {
            "partial_bear_entry_cap": cap,
            "partial_bear_entry_staged": staged_flags,
            "partial_bear_entry_release": releases,
        },
        index=base_weight.index,
    )
    return apply_cap(base_weight, cap).rename(base_weight.name), diagnostics


def _post_big_winner_cap(
    base_weight: pd.Series,
    target_price: pd.Series,
    features: pd.DataFrame,
    *,
    activation_gain: float,
    trigger_kind: str,
    defensive_cap: float,
    macd_hist: pd.Series | None = None,
    peak_dd_threshold: float = 0.25,
) -> tuple[pd.Series, pd.DataFrame]:
    price = target_price.reindex(base_weight.index).astype(float)
    feat = features.reindex(base_weight.index)
    hist = macd_hist.reindex(base_weight.index) if macd_hist is not None else None
    in_trade = False
    entry_price = np.nan
    peak_price = np.nan
    caps: list[float] = []
    triggers: list[bool] = []
    gains: list[float] = []
    peak_dds: list[float] = []
    for timestamp, desired in base_weight.fillna(0.0).items():
        current_price = float(price.loc[timestamp]) if np.isfinite(price.loc[timestamp]) else np.nan
        trigger = False
        gain = np.nan
        peak_dd = np.nan
        if desired <= 0.0 or not np.isfinite(current_price):
            in_trade = False
            entry_price = np.nan
            peak_price = np.nan
            cap = 0.0
        else:
            if not in_trade:
                in_trade = True
                entry_price = current_price
                peak_price = current_price
            else:
                peak_price = max(float(peak_price), current_price)
            gain = current_price / entry_price - 1.0 if entry_price > 0 else np.nan
            peak_dd = current_price / peak_price - 1.0 if peak_price > 0 else np.nan
            if np.isfinite(gain) and gain >= activation_gain:
                if trigger_kind == "qqq_below_50ma":
                    trigger = bool(
                        feat.at[timestamp, "qqq_price"] < feat.at[timestamp, "ma_medium"]
                    )
                elif trigger_kind == "macd_hist_lt_0":
                    trigger = bool(hist is not None and hist.loc[timestamp] < 0.0)
                elif trigger_kind == "synthetic_peak_dd":
                    trigger = bool(np.isfinite(peak_dd) and peak_dd <= -abs(peak_dd_threshold))
                else:
                    raise ValueError(f"unknown trigger_kind: {trigger_kind}")
            cap = defensive_cap if trigger else 1.0
        caps.append(cap)
        triggers.append(trigger)
        gains.append(gain)
        peak_dds.append(peak_dd)
    cap_series = pd.Series(caps, index=base_weight.index, name=f"{trigger_kind}_cap")
    diagnostics = pd.DataFrame(
        {
            f"{trigger_kind}_cap": cap_series,
            f"{trigger_kind}_trigger": triggers,
            "trade_gain": gains,
            "trade_peak_drawdown": peak_dds,
        },
        index=base_weight.index,
    )
    return apply_cap(base_weight, cap_series).rename(base_weight.name), diagnostics


def _bear_volatility_cap(
    base_weight: pd.Series,
    features: pd.DataFrame,
    *,
    percentile_threshold: float,
    defensive_cap: float,
    slope_days: int = 20,
) -> tuple[pd.Series, pd.DataFrame]:
    feat = features.reindex(base_weight.index)
    bear = feat[f"long_ma_slope_{slope_days}d"].lt(0.0)
    high_vol = feat["realized_vol_percentile_known"].ge(percentile_threshold).fillna(False)
    active = bear & high_vol
    cap = pd.Series(1.0, index=base_weight.index, name="bear_vol_cap")
    cap.loc[active] = defensive_cap
    diagnostics = pd.DataFrame(
        {"bear_vol_cap": cap, "bear_vol_cap_active": active},
        index=base_weight.index,
    )
    return apply_cap(base_weight, cap).rename(base_weight.name), diagnostics


def _bear_circuit_breaker_cap(
    base_weight: pd.Series,
    target_returns: pd.Series,
    features: pd.DataFrame,
    *,
    trigger_drawdown: float,
    recover_drawdown: float,
    defensive_cap: float,
    slope_days: int = 20,
) -> tuple[pd.Series, pd.DataFrame]:
    feat = features.reindex(base_weight.index)
    returns = target_returns.reindex(base_weight.index).fillna(0.0)
    equity = 1.0
    peak = 1.0
    defensive = False
    caps: list[float] = []
    triggers: list[bool] = []
    releases: list[bool] = []
    prior_dds: list[float] = []
    for timestamp, desired in base_weight.fillna(0.0).items():
        prior_dd = equity / peak - 1.0
        bear = bool(feat.at[timestamp, f"long_ma_slope_{slope_days}d"] < 0.0)
        trigger = False
        release = False
        if bear and not defensive and prior_dd <= -abs(trigger_drawdown):
            defensive = True
            trigger = True
        if defensive and (
            prior_dd >= -abs(recover_drawdown)
            or float(feat.at[timestamp, f"medium_ma_slope_{slope_days}d"]) > 0.0
            or not bear
        ):
            defensive = False
            release = True
        cap = defensive_cap if defensive else 1.0
        actual_weight = min(float(desired), cap)
        equity *= 1.0 + actual_weight * float(returns.loc[timestamp])
        peak = max(peak, equity)
        caps.append(cap)
        triggers.append(trigger)
        releases.append(release)
        prior_dds.append(prior_dd)
    cap_series = pd.Series(caps, index=base_weight.index, name="bear_circuit_cap")
    diagnostics = pd.DataFrame(
        {
            "bear_circuit_cap": cap_series,
            "bear_circuit_trigger": triggers,
            "bear_circuit_release": releases,
            "prior_strategy_drawdown": prior_dds,
        },
        index=base_weight.index,
    )
    return apply_cap(base_weight, cap_series).rename(base_weight.name), diagnostics


def _diag_trigger_count(diagnostics: pd.DataFrame) -> int:
    total = 0
    for column in diagnostics.columns:
        if ("trigger" in column or "active" in column or "staged" in column) and diagnostics[
            column
        ].dtype == bool:
            total += int(diagnostics[column].sum())
    return total


def _plot_return_vs_dd(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.2))
    for family, group in metrics.groupby("family"):
        ax.scatter(
            group["max_drawdown"] * 100.0,
            group["annualized_return"] * 100.0,
            s=72 if family in {"baseline", "serious_candidate"} else 36,
            alpha=0.72,
            label=family,
        )
    ax.axhline(26.0, color="#777777", linestyle=":", linewidth=1.0, label="26% ann. return")
    ax.axvline(-50.0, color="#aa0000", linestyle=":", linewidth=1.0, label="-50% DD")
    ax.set_xlabel("Full-sample max drawdown (%)")
    ax.set_ylabel("Annualized return (%)")
    ax.set_title("Bear-filter variants: annualized return vs max drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_top_equity_drawdown(
    returns_by_name: dict[str, pd.Series],
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    selected = [BASELINE_NAME, SERIOUS_NAME, CURRENT_PREFERRED_NAME]
    selected += (
        metrics.loc[
            ~metrics["name"].isin(selected + ["QQQ_BH"]) & metrics["passes_candidate_filter"]
        ]
        .head(6)["name"]
        .tolist()
    )
    selected = list(dict.fromkeys([name for name in selected if name in returns_by_name]))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    for name in selected:
        returns = returns_by_name[name]
        _equity(returns).plot(ax=axes[0], label=name, linewidth=1.1)
        _drawdown(returns).plot(ax=axes[1], label=name, linewidth=1.1)
    axes[0].set_title("After-tax equity")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("After-tax drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("Top bear-filter variants")
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _make_site(site_dir: Path, compact: pd.DataFrame, figures: list[Path]) -> None:
    ensure_directory(site_dir)
    html_path = site_dir / "bear_filter_variants.html"
    show = compact.head(30).copy()
    for column in [
        "final_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "2010_max_drawdown",
        "bear_2007_2009_max_drawdown",
        "trades_per_year",
    ]:
        if column in show.columns:
            show[column] = show[column].map(lambda value: f"{value:.3f}")
    figures_html = "\n".join(
        f'<h2>{path.name}</h2><img src="../figures/{path.name}" alt="{path.name}">'
        for path in figures
    )
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'>",
                "<title>Bear Filter Variants</title>",
                "<link rel='stylesheet' href='style.css'>",
                "</head><body>",
                "<h1>Bear Filter Variant Backtests</h1>",
                "<p>Variants around the serious bear re-entry filter, including q100 activation thresholds from +50% through +150%.</p>",
                show.to_html(index=False),
                figures_html,
                "</body></html>",
            ]
        )
    )
    index_path = site_dir / "index.html"
    if index_path.exists():
        index = index_path.read_text()
        link = '<li><a href="bear_filter_variants.html">Bear Filter Variants</a></li>'
        if "bear_filter_variants.html" not in index:
            index = index.replace(
                "</body>", f"<h2>Additional experiment pages</h2><ul>{link}</ul></body>"
            )
            index_path.write_text(index)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tables_dir = resolve_path(config.root, "reports/tables")
    figures_dir = resolve_path(config.root, "reports/figures")
    site_dir = resolve_path(config.root, args.site_dir)
    ensure_directory(tables_dir)
    ensure_directory(figures_dir)

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
    raw_base, base_diag = no_daily_gate_hourly_ma_gate_signal(
        entry_price=qqq,
        exit_price=qqq,
        output_index=common,
        bars_per_day=bars_per_day,
        average_type=args.average_type,
        macd_unit=args.macd_unit,
    )
    profit_locked_weight, _ = _preferred_weight(
        raw_base=raw_base,
        target_price=target,
        stop_drawdown=0.40,
    )
    mr_features = qqq_mean_reversion_features(qqq, bars_per_day=bars_per_day)
    features = bear_market_features(
        qqq,
        bars_per_day=bars_per_day,
        slope_days=(5, 10, 15, 20, 30),
    )
    macd = macd_components(
        qqq,
        fast_window=_days_to_bars(12, bars_per_day),
        slow_window=_days_to_bars(26, bars_per_day),
        signal_window=_days_to_bars(9, bars_per_day),
        average_type=args.average_type,
    )

    eval_args = argparse.Namespace(
        target_ticker=args.target_ticker,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        short_term_tax_rate=args.short_term_tax_rate,
        cash_annual_yield=args.cash_annual_yield,
        cash_interest_tax_rate=args.cash_interest_tax_rate,
    )

    metrics_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, pd.Series] = {}
    weights_by_name: dict[str, pd.DataFrame] = {}
    diagnostics_summary_rows: list[dict[str, Any]] = []

    def add_candidate(
        name: str,
        family: str,
        raw_weight: pd.Series | pd.DataFrame,
        parameters: dict[str, Any],
        diagnostics: pd.DataFrame | None = None,
    ) -> None:
        diagnostic_count = _diag_trigger_count(diagnostics) if diagnostics is not None else 0
        metrics, after_tax, weights = _eval_candidate(
            name=name,
            family=family,
            raw_weight=raw_weight,
            returns=returns,
            config=config,
            args=eval_args,
            parameters=parameters,
            diagnostic_count=diagnostic_count,
        )
        metrics_rows.append(metrics)
        returns_by_name[name] = after_tax
        weights_by_name[name] = weights
        diagnostics_summary_rows.append(
            {
                "name": name,
                "family": family,
                "diagnostic_event_count": diagnostic_count,
                "parameters": parameters,
            }
        )

    q100_weights: dict[float, pd.Series] = {}
    for activation in (1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        weight, diag = _q100_weight(
            profit_locked_weight=profit_locked_weight.rename(args.target_ticker),
            target_price=target,
            mr_features=mr_features,
            activation_gain=activation,
        )
        q100_weights[activation] = weight
        name = f"q100_activation{int(activation * 100)}"
        if activation == 1.0:
            name = BASELINE_NAME
        add_candidate(
            name,
            "dynamic_q100_activation",
            weight,
            {"q100_activation_gain": activation},
            diag,
        )

    serious_weight, serious_diag = _serious_bear_filtered_weight(
        q100_weights[1.0],
        features,
        buffer=0.01,
        slope_days=20,
        require_short_gt_medium=True,
    )
    add_candidate(
        SERIOUS_NAME,
        "serious_candidate",
        serious_weight,
        {"buffer": 0.01, "slope_days": 20, "require_20gt50": True},
        serious_diag,
    )

    current_preferred_weight, current_preferred_diag = _serious_bear_filtered_weight(
        q100_weights[1.1],
        features,
        buffer=0.01,
        slope_days=30,
        require_short_gt_medium=True,
    )
    add_candidate(
        CURRENT_PREFERRED_NAME,
        "current_preferred",
        current_preferred_weight,
        {
            "q100_activation_gain": 1.1,
            "buffer": 0.01,
            "slope_days": 30,
            "require_20gt50": True,
        },
        current_preferred_diag,
    )

    # Family A: robustness variants around the bear filter.
    for buffer in (0.005, 0.01, 0.015, 0.02):
        for slope_days in (10, 15, 20, 30):
            for require_short in (False, True):
                weight, diag = _serious_bear_filtered_weight(
                    q100_weights[1.0],
                    features,
                    buffer=buffer,
                    slope_days=slope_days,
                    require_short_gt_medium=require_short,
                )
                name = (
                    f"robust_buf{int(buffer * 1000):03d}bp_slope{slope_days}"
                    f"{'_20gt50' if require_short else ''}"
                )
                add_candidate(
                    name,
                    "robust_bear_filter",
                    weight,
                    {
                        "buffer": buffer,
                        "slope_days": slope_days,
                        "require_20gt50": require_short,
                    },
                    diag,
                )

    # Family B: partial entry while the serious filter is not confirmed.
    for partial in (0.25, 0.50):
        weight, diag = _partial_bear_entry_cap(
            q100_weights[1.0],
            features,
            partial_weight=partial,
            buffer=0.01,
            slope_days=20,
            require_short_gt_medium=True,
        )
        add_candidate(
            f"partial_bear_entry_{int(partial * 100)}",
            "partial_bear_entry",
            weight,
            {"partial_weight": partial, "buffer": 0.01, "slope_days": 20, "require_20gt50": True},
            diag,
        )

    # Family C: post-big-winner protection on top of the serious candidate.
    for trigger_kind in ("qqq_below_50ma", "macd_hist_lt_0", "synthetic_peak_dd"):
        for cap in (0.50, 0.25, 0.0):
            weight, diag = _post_big_winner_cap(
                serious_weight,
                target,
                features,
                activation_gain=1.0,
                trigger_kind=trigger_kind,
                defensive_cap=cap,
                macd_hist=macd["macd_hist"],
                peak_dd_threshold=0.25,
            )
            add_candidate(
                f"post100_{trigger_kind}_to{int(cap * 100)}",
                "post_big_winner_protection",
                weight,
                {"trigger_kind": trigger_kind, "defensive_cap": cap, "activation_gain": 1.0},
                diag,
            )

    # Family D: bear-regime volatility caps on top of the serious candidate.
    for percentile in (0.85, 0.90, 0.95):
        for cap in (0.50, 0.25, 0.0):
            weight, diag = _bear_volatility_cap(
                serious_weight,
                features,
                percentile_threshold=percentile,
                defensive_cap=cap,
                slope_days=20,
            )
            add_candidate(
                f"bear_vol_p{int(percentile * 100)}_to{int(cap * 100)}",
                "bear_volatility_cap",
                weight,
                {"percentile_threshold": percentile, "defensive_cap": cap},
                diag,
            )

    # Family E: bear-regime circuit breakers on top of the serious candidate.
    for trigger_dd in (0.30, 0.35, 0.40):
        for cap in (0.50, 0.0):
            for recover_dd in (0.10, 0.15):
                weight, diag = _bear_circuit_breaker_cap(
                    serious_weight,
                    returns[args.target_ticker],
                    features,
                    trigger_drawdown=trigger_dd,
                    recover_drawdown=recover_dd,
                    defensive_cap=cap,
                    slope_days=20,
                )
                add_candidate(
                    f"bear_circuit_dd{int(trigger_dd * 100)}_to{int(cap * 100)}_rec{int(recover_dd * 100)}",
                    "bear_circuit_breaker",
                    weight,
                    {
                        "trigger_drawdown": trigger_dd,
                        "defensive_cap": cap,
                        "recover_drawdown": recover_dd,
                    },
                    diag,
                )

    add_candidate(
        "QQQ_BH",
        "benchmark",
        pd.DataFrame({args.benchmark_ticker: pd.Series(1.0, index=common)}),
        {},
        None,
    )

    metrics = metrics_to_frame(metrics_rows)
    serious = metrics.loc[metrics["name"].eq(SERIOUS_NAME)].iloc[0]
    baseline = metrics.loc[metrics["name"].eq(BASELINE_NAME)].iloc[0]
    metrics["delta_ann_return_vs_serious"] = metrics["annualized_return"] - float(
        serious["annualized_return"]
    )
    metrics["delta_max_dd_vs_serious"] = metrics["max_drawdown"] - float(serious["max_drawdown"])
    metrics["delta_2010_dd_vs_serious"] = metrics["2010_max_drawdown"] - float(
        serious["2010_max_drawdown"]
    )
    metrics["delta_max_dd_vs_q100_baseline"] = metrics["max_drawdown"] - float(
        baseline["max_drawdown"]
    )
    metrics["passes_candidate_filter"] = (
        metrics["annualized_return"].ge(0.26)
        & metrics["max_drawdown"].gt(-0.50)
        & metrics["trades_per_year"].le(8.0)
    )
    metrics["beats_serious_two_of_four"] = (
        metrics["delta_ann_return_vs_serious"].gt(0).astype(int)
        + metrics["delta_max_dd_vs_serious"].gt(0).astype(int)
        + metrics["delta_2010_dd_vs_serious"].gt(0).astype(int)
        + metrics["drawdown_episodes_gt_40pct"]
        .lt(serious["drawdown_episodes_gt_40pct"])
        .astype(int)
    ).ge(2)
    metrics["objective_score"] = (
        metrics["annualized_return"]
        + 0.50 * metrics["delta_max_dd_vs_q100_baseline"]
        + 0.35 * metrics["delta_2010_dd_vs_serious"]
        - 0.0025 * metrics["number_of_trades"].clip(lower=0) / 10.0
    )
    metrics = metrics.sort_values(
        ["objective_score", "annualized_return", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    compact_cols = [
        "name",
        "family",
        "final_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "2010_max_drawdown",
        "2004_2006_max_drawdown",
        "bear_2007_2009_max_drawdown",
        "2011_max_drawdown",
        "2018_2019_max_drawdown",
        "number_of_trades",
        "trades_per_year",
        "exposure_percentage",
        "dd_episodes_gt_20_30_40_50pct",
        "overlay_trigger_count",
        "passes_candidate_filter",
        "beats_serious_two_of_four",
        "objective_score",
    ]
    compact = metrics[[column for column in compact_cols if column in metrics.columns]]
    metrics.to_csv(tables_dir / f"{args.output_prefix}_experiments_metrics.csv", index=False)
    compact.to_csv(tables_dir / f"{args.output_prefix}_experiments_compact.csv", index=False)
    pd.DataFrame(diagnostics_summary_rows).to_csv(
        tables_dir / f"{args.output_prefix}_diagnostics_summary.csv",
        index=False,
    )
    pd.concat(returns_by_name.values(), axis=1).to_csv(
        tables_dir / f"{args.output_prefix}_experiments_returns.csv"
    )
    pd.concat(weights_by_name, axis=1).to_parquet(
        tables_dir / f"{args.output_prefix}_experiments_weights.parquet"
    )

    worst = _worst_drawdown_rows(
        {
            name: returns_by_name[name]
            for name in list(
                dict.fromkeys(
                    [BASELINE_NAME, SERIOUS_NAME, CURRENT_PREFERRED_NAME]
                    + compact.head(12)["name"].tolist()
                )
            )
            if name in returns_by_name
        },
        top_n=5,
    )
    worst.to_csv(tables_dir / f"{args.output_prefix}_worst5_drawdowns.csv", index=False)

    period_rows: list[dict[str, Any]] = []
    for name in [BASELINE_NAME, SERIOUS_NAME, CURRENT_PREFERRED_NAME] + compact.head(10)[
        "name"
    ].tolist():
        if name not in returns_by_name:
            continue
        for period_name, (start, end) in DD_PERIODS.items():
            sample = returns_by_name[name].loc[
                (returns_by_name[name].index >= start) & (returns_by_name[name].index <= end)
            ]
            if sample.empty:
                continue
            period_rows.append(
                {
                    "name": name,
                    "period": period_name,
                    "final_return": float((1.0 + sample.fillna(0.0)).prod() - 1.0),
                    "max_drawdown": _window_drawdown(
                        sample, sample.index.min(), sample.index.max()
                    ),
                }
            )
    pd.DataFrame(period_rows).to_csv(
        tables_dir / f"{args.output_prefix}_period_summary.csv",
        index=False,
    )

    return_dd_path = figures_dir / f"{args.output_prefix}_return_vs_dd.png"
    top_path = figures_dir / f"{args.output_prefix}_top_equity_drawdown.png"
    _plot_return_vs_dd(metrics, return_dd_path)
    _plot_top_equity_drawdown(returns_by_name, metrics, top_path)
    _make_site(site_dir, compact, [return_dd_path, top_path])

    print(f"Saved compact table: {tables_dir / f'{args.output_prefix}_experiments_compact.csv'}")
    print(f"Saved website: {site_dir / 'bear_filter_variants.html'}")
    print("Top rows:")
    print(compact.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
