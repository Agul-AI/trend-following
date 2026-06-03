#!/usr/bin/env python
"""Analyze QQQ P/E percentiles around the preferred strategy's worst drawdowns.

The project's live Option-B QQQ P/E history is point-in-time only, so this
script uses a complete monthly QQQ/Nasdaq-100 P/E history from WorldPERatio for
historical drawdown analysis. For no-lookahead alignment, intraday event dates
use the previous completed calendar month's P/E value.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tqqq_cash_yield_candidate_comparison import (
    simulate_after_tax_portfolio_with_cash_yield,  # noqa: E402
)
from run_tqqq_entry_signal_comparison import _returns_from_prices  # noqa: E402
from run_tqqq_position_risk_sizing_experiments import drawdown_episode_count  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.metrics import calculate_metrics  # noqa: E402

WORLD_PE_RATIO_URL = "https://worldperatio.com/index/nasdaq-100/"
TARGET_TICKER = "QQQ_3X_CALC"
BENCHMARK_TICKER = "QQQ"
PREFERRED_NAME = "new_candidate_no_daily_gate__qqq_hourly_200ma_entry_exit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--pe-url", default=WORLD_PE_RATIO_URL)
    parser.add_argument(
        "--worst6-file",
        default="reports/tables/preferred_hourly_200ma_worst6_drawdowns_hiking_analysis.csv",
    )
    parser.add_argument(
        "--weights-file",
        default="reports/tables/tqqq_cash_yield_candidate_comparison_weights.csv",
    )
    parser.add_argument("--target-price-file", default="data/raw/synthetic_3x_60min/QQQ_3X_CALC.parquet")
    parser.add_argument("--benchmark-price-file", default="data/raw/alpha_vantage_60min/QQQ.parquet")
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    return parser.parse_args()


def fetch_worldperatio_monthly_pe(url: str) -> pd.DataFrame:
    """Fetch monthly QQQ/Nasdaq-100 P/E values embedded in WorldPERatio HTML."""
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - fixed public read-only URL by default.
        html = response.read().decode("utf-8", errors="ignore")
    match = re.search(r"detailPE_data\s*=\s*(\[\[Date\.UTC\(.*?\],\s*\]);", html, flags=re.S)
    if not match:
        raise ValueError("Could not find detailPE_data in WorldPERatio HTML")
    triples = re.findall(
        r"Date\.UTC\((\d+),\s*(\d+),\s*(\d+)\),\s*(-?\d+(?:\.\d+)?|null)",
        match.group(1),
    )
    if not triples:
        raise ValueError("Could not parse monthly P/E points from detailPE_data")

    rows: list[dict[str, Any]] = []
    for year, zero_based_month, day, value in triples:
        if value == "null":
            continue
        timestamp = pd.Timestamp(int(year), int(zero_based_month) + 1, int(day))
        rows.append({"month": timestamp.to_period("M"), "qqq_pe": float(value)})
    frame = pd.DataFrame(rows).drop_duplicates("month").sort_values("month")
    if frame.empty:
        raise ValueError("Parsed P/E history is empty")
    return frame.set_index("month")


def add_rolling_percentiles(pe: pd.DataFrame) -> pd.DataFrame:
    """Add trailing 3-year and 5-year rolling percentile columns."""
    result = pe.copy()
    for window, label in ((36, "3y"), (60, "5y")):
        values: list[float] = []
        series = result["qqq_pe"].astype(float)
        for i, value in enumerate(series):
            window_values = series.iloc[max(0, i - window + 1) : i + 1]
            values.append(float(window_values.le(value).mean() * 100.0))
        result[f"pe_pctile_{label}"] = values
    return result


def previous_completed_month(timestamp: str | pd.Timestamp) -> pd.Period:
    """Return the conservative P/E lookup month for an intraday event."""
    return pd.Timestamp(timestamp).to_period("M") - 1


def lookup_pe(pe: pd.DataFrame, timestamp: str | pd.Timestamp) -> dict[str, Any]:
    month = previous_completed_month(timestamp)
    if month not in pe.index:
        return {
            "known_month": str(month),
            "qqq_pe": np.nan,
            "pe_pctile_3y": np.nan,
            "pe_pctile_5y": np.nan,
        }
    row = pe.loc[month]
    return {
        "known_month": str(month),
        "qqq_pe": float(row["qqq_pe"]),
        "pe_pctile_3y": float(row["pe_pctile_3y"]),
        "pe_pctile_5y": float(row["pe_pctile_5y"]),
    }


def build_event_table(pe: pd.DataFrame, worst6_file: Path) -> pd.DataFrame:
    worst6 = pd.read_csv(worst6_file)
    rows: list[dict[str, Any]] = []
    for _, episode in worst6.iterrows():
        peak = lookup_pe(pe, episode["peak"])
        bottom = lookup_pe(pe, episode["trough"])
        rows.append(
            {
                "rank": int(episode["rank"]),
                "peak": episode["peak"],
                "trough": episode["trough"],
                "max_drawdown": float(episode["max_drawdown"]),
                "peak_known_month": peak["known_month"],
                "peak_qqq_pe": peak["qqq_pe"],
                "peak_pe_pctile_3y": peak["pe_pctile_3y"],
                "peak_pe_pctile_5y": peak["pe_pctile_5y"],
                "bottom_known_month": bottom["known_month"],
                "bottom_qqq_pe": bottom["qqq_pe"],
                "bottom_pe_pctile_3y": bottom["pe_pctile_3y"],
                "bottom_pe_pctile_5y": bottom["pe_pctile_5y"],
                "pe_pctile_3y_peak_minus_bottom": peak["pe_pctile_3y"] - bottom["pe_pctile_3y"],
                "pe_pctile_5y_peak_minus_bottom": peak["pe_pctile_5y"] - bottom["pe_pctile_5y"],
                "peak_pctile_3y_ge_80": peak["pe_pctile_3y"] >= 80.0,
                "peak_pctile_5y_ge_80": peak["pe_pctile_5y"] >= 80.0,
                "peak_pctile_3y_ge_90": peak["pe_pctile_3y"] >= 90.0,
                "peak_pctile_5y_ge_90": peak["pe_pctile_5y"] >= 90.0,
            }
        )
    return pd.DataFrame(rows)


def pe_percentiles_for_index(pe: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    months = pd.PeriodIndex(index.to_period("M"), freq="M") - 1
    return pd.DataFrame(
        {
            "pe_pctile_3y": [pe.at[month, "pe_pctile_3y"] if month in pe.index else np.nan for month in months],
            "pe_pctile_5y": [pe.at[month, "pe_pctile_5y"] if month in pe.index else np.nan for month in months],
        },
        index=index,
    )


def run_filter_sensitivity(args: argparse.Namespace, pe: pd.DataFrame) -> pd.DataFrame:
    """Test simple high-P/E cash/half-size filters on the preferred weights."""
    config = load_config(args.config)
    price = read_price_file(Path(args.target_price_file)).sort_index()["adj_close"].astype(float).rename(TARGET_TICKER)
    benchmark_price = (
        read_price_file(Path(args.benchmark_price_file)).sort_index()["adj_close"].astype(float).rename(BENCHMARK_TICKER)
    )
    weights = pd.read_csv(args.weights_file, parse_dates=["date"]).set_index("date")
    common = price.index.intersection(weights.index).intersection(benchmark_price.index)
    target_returns = _returns_from_prices(price.loc[common].to_frame())
    benchmark_returns = _returns_from_prices(benchmark_price.loc[common].to_frame())[BENCHMARK_TICKER]
    base_weight = weights.loc[common, PREFERRED_NAME].astype(float)
    pe_flags = pe_percentiles_for_index(pe, common)

    variants: dict[str, pd.Series] = {"base": base_weight}
    for horizon in ("3y", "5y"):
        series = pe_flags[f"pe_pctile_{horizon}"]
        for threshold in (70, 80, 90):
            high = series.ge(threshold).fillna(False)
            variants[f"cash_if_pe_{horizon}_pct_ge_{threshold}"] = base_weight.where(~high, 0.0)
            variants[f"half_if_pe_{horizon}_pct_ge_{threshold}"] = base_weight.where(~high, base_weight * 0.5)
    for threshold in (80, 90):
        high = (pe_flags["pe_pctile_3y"].ge(threshold) | pe_flags["pe_pctile_5y"].ge(threshold)).fillna(False)
        variants[f"cash_if_pe_3y_or_5y_pct_ge_{threshold}"] = base_weight.where(~high, 0.0)
        variants[f"half_if_pe_3y_or_5y_pct_ge_{threshold}"] = base_weight.where(~high, base_weight * 0.5)

    rows: list[dict[str, Any]] = []
    for name, variant_weight in variants.items():
        after_tax, _, turnover, _, cash_weight = simulate_after_tax_portfolio_with_cash_yield(
            target_returns,
            variant_weight.to_frame(TARGET_TICKER),
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
            weights=variant_weight,
            annualization=config.backtest.annualization,
        )
        dd_counts = {
            f"dd_gt_{threshold}": drawdown_episode_count(after_tax, threshold=-(threshold / 100.0))
            for threshold in (20, 30, 40, 50)
        }
        rows.append(
            {
                "name": name,
                "final_return": metrics["cumulative_return"],
                "annualized_return": metrics["annualized_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "number_of_trades": metrics["number_of_trades"],
                "exposure_percentage": metrics["exposure_percentage"],
                "average_cash_weight": float(cash_weight.mean()),
                "dd_20_30_40_50": (
                    f"{dd_counts['dd_gt_20']}/{dd_counts['dd_gt_30']}/"
                    f"{dd_counts['dd_gt_40']}/{dd_counts['dd_gt_50']}"
                ),
            }
        )
    benchmark_metrics = calculate_metrics(
        benchmark_returns,
        turnover=None,
        weights=pd.Series(1.0, index=benchmark_returns.index),
        annualization=config.backtest.annualization,
    )
    benchmark_dd_counts = {
        f"dd_gt_{threshold}": drawdown_episode_count(benchmark_returns, threshold=-(threshold / 100.0))
        for threshold in (20, 30, 40, 50)
    }
    rows.append(
        {
            "name": "QQQ_BH",
            "final_return": benchmark_metrics["cumulative_return"],
            "annualized_return": benchmark_metrics["annualized_return"],
            "sharpe_ratio": benchmark_metrics["sharpe_ratio"],
            "max_drawdown": benchmark_metrics["max_drawdown"],
            "number_of_trades": 0,
            "exposure_percentage": 1.0,
            "average_cash_weight": 0.0,
            "dd_20_30_40_50": (
                f"{benchmark_dd_counts['dd_gt_20']}/{benchmark_dd_counts['dd_gt_30']}/"
                f"{benchmark_dd_counts['dd_gt_40']}/{benchmark_dd_counts['dd_gt_50']}"
            ),
        }
    )
    return pd.DataFrame(rows).sort_values("annualized_return", ascending=False)


def main() -> None:
    args = parse_args()
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    pe = add_rolling_percentiles(fetch_worldperatio_monthly_pe(args.pe_url))
    pe_history = pe.reset_index().assign(month=lambda frame: frame["month"].astype(str))
    pe_history.to_csv(tables_dir / "qqq_pe_worldperatio_monthly_history.csv", index=False)

    event_table = build_event_table(pe, Path(args.worst6_file))
    event_table.to_csv(tables_dir / "preferred_hourly_200ma_worst6_qqq_pe_percentiles.csv", index=False)

    sensitivity = run_filter_sensitivity(args, pe)
    sensitivity.to_csv(tables_dir / "preferred_hourly_200ma_pe_filter_sensitivity.csv", index=False)

    print("Worst-6 event P/E percentiles:")
    print(event_table.to_string(index=False))
    print("\nSimple P/E filter sensitivity:")
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
