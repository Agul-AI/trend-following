#!/usr/bin/env python
"""Compare two ways to estimate QQQ daily P/E up to the latest available day.

Option A: official/fact-sheet anchored daily proxy.
- Use an official Invesco QQQ P/E anchor as of 2026-03-31.
- Convert that to an implied QQQ EPS proxy using QQQ close on the anchor date.
- Revalue daily PE as QQQ close / implied EPS.

Option B: Alpha Vantage holdings look-through snapshot.
- Pull QQQ holdings from ETF_PROFILE.
- Pull each holding's current PERatio from OVERVIEW.
- Compute a weighted harmonic average PE across included positive-PE holdings.

This script never prints or stores the Alpha Vantage API key.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.utils import ensure_directory  # noqa: E402

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# Official Invesco QQQ fact sheet anchor. Search result/opened text gives:
# Q1 2026, as of March 31, 2026: P/E ratio 36.52.
DEFAULT_ANCHOR_DATE = "2026-03-31"
DEFAULT_ANCHOR_PE = 36.52


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--anchor-date", default=DEFAULT_ANCHOR_DATE)
    parser.add_argument("--anchor-pe", type=float, default=DEFAULT_ANCHOR_PE)
    parser.add_argument("--asof-date", default="2026-06-01", help="Latest decision date to report.")
    parser.add_argument("--snapshot-date", default="2026-06-02", help="Folder label for AV snapshot cache.")
    parser.add_argument("--pause-seconds", type=float, default=0.85)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-prefix", default="qqq_pe_method_comparison")
    return parser.parse_args()


def _api_key() -> str:
    load_dotenv(".env")
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY is missing from .env")
    return key


def _get_json(params: dict[str, Any], *, api_key: str, timeout: int = 30) -> dict[str, Any]:
    params = dict(params)
    params["apikey"] = api_key
    response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error for {params.get('function')}: {data['Error Message']}")
    if "Information" in data and len(data) == 1:
        raise RuntimeError(f"Alpha Vantage information/rate-limit message: {data['Information'][:200]}")
    return data


def _cache_json(path: Path, params: dict[str, Any], *, api_key: str, force: bool) -> dict[str, Any]:
    if path.exists() and not force:
        return json.loads(path.read_text())
    data = _get_json(params, api_key=api_key)
    ensure_directory(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data


def fetch_daily_prices(symbol: str, *, api_key: str, raw_dir: Path, force: bool) -> pd.DataFrame:
    path = raw_dir / f"alpha_vantage_daily_adjusted_{symbol}.json"
    data = _cache_json(
        path,
        {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
        },
        api_key=api_key,
        force=force,
    )
    series = data.get("Time Series (Daily)")
    if not isinstance(series, dict):
        raise RuntimeError(f"Daily adjusted response for {symbol} did not contain Time Series (Daily)")
    rows = []
    for day, values in series.items():
        rows.append(
            {
                "date": pd.Timestamp(day),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "adjusted_close": float(values["5. adjusted close"]),
                "volume": float(values["6. volume"]),
                "dividend_amount": float(values.get("7. dividend amount", 0.0)),
                "split_coefficient": float(values.get("8. split coefficient", 1.0)),
            }
        )
    frame = pd.DataFrame(rows).sort_values("date").set_index("date")
    return frame


def option_a_daily_proxy(
    qqq_daily: pd.DataFrame,
    *,
    anchor_date: str,
    anchor_pe: float,
    asof_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchor_ts = pd.Timestamp(anchor_date)
    asof_ts = pd.Timestamp(asof_date)
    if anchor_ts not in qqq_daily.index:
        raise RuntimeError(f"Anchor date {anchor_date} not found in QQQ daily price data")
    available = qqq_daily.loc[:asof_ts].copy()
    if available.empty:
        raise RuntimeError(f"No QQQ daily prices available through {asof_date}")
    actual_asof = pd.Timestamp(available.index[-1])
    anchor_close = float(qqq_daily.at[anchor_ts, "close"])
    implied_eps = anchor_close / anchor_pe
    daily_pe = available["close"] / implied_eps
    out = pd.DataFrame(
        {
            "qqq_close": available["close"],
            "qqq_pe_option_a_proxy": daily_pe,
            # This is the value that can be used for a same-day trading decision.
            "qqq_pe_option_a_known_today_no_lookahead": daily_pe.shift(1),
        }
    )
    summary = {
        "method": "option_a_fact_sheet_anchor_daily_proxy",
        "anchor_date": anchor_date,
        "anchor_pe": anchor_pe,
        "anchor_qqq_close": anchor_close,
        "implied_eps_proxy": implied_eps,
        "requested_asof_date": asof_date,
        "actual_latest_price_date": actual_asof.date().isoformat(),
        "latest_qqq_close": float(out.at[actual_asof, "qqq_close"]),
        "latest_proxy_pe": float(out.at[actual_asof, "qqq_pe_option_a_proxy"]),
        "latest_known_today_no_lookahead_pe": float(
            out.at[actual_asof, "qqq_pe_option_a_known_today_no_lookahead"]
        )
        if pd.notna(out.at[actual_asof, "qqq_pe_option_a_known_today_no_lookahead"])
        else math.nan,
    }
    return out, summary


def fetch_av_holdings(symbol: str, *, api_key: str, raw_dir: Path, force: bool) -> list[dict[str, Any]]:
    path = raw_dir / f"alpha_vantage_etf_profile_{symbol}.json"
    data = _cache_json(
        path,
        {"function": "ETF_PROFILE", "symbol": symbol},
        api_key=api_key,
        force=force,
    )
    holdings = data.get("holdings", [])
    if not holdings:
        raise RuntimeError(f"No holdings returned by Alpha Vantage ETF_PROFILE for {symbol}")
    return holdings


def _parse_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text.lower() in {"", "none", "null", "nan", "n/a", "-"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def fetch_overviews(
    symbols: list[str],
    *,
    api_key: str,
    raw_dir: Path,
    pause_seconds: float,
    force: bool,
) -> dict[str, dict[str, Any]]:
    overviews: dict[str, dict[str, Any]] = {}
    ensure_directory(raw_dir)
    next_start = time.monotonic()
    for i, symbol in enumerate(symbols, start=1):
        safe = symbol.replace("/", "_").replace(".", "_")
        path = raw_dir / f"{safe}.json"
        if path.exists() and not force:
            overviews[symbol] = json.loads(path.read_text())
            continue
        wait = next_start - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        next_start = time.monotonic() + pause_seconds
        try:
            data = _get_json({"function": "OVERVIEW", "symbol": symbol}, api_key=api_key)
        except Exception as exc:  # noqa: BLE001 - record failures in output table
            data = {"symbol": symbol, "download_error": str(exc)}
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
        overviews[symbol] = data
        if i % 25 == 0 or i == len(symbols):
            print(f"Fetched/cached {i}/{len(symbols)} Alpha Vantage OVERVIEW records")
    return overviews


def option_b_av_harmonic_pe(
    holdings: list[dict[str, Any]],
    overviews: dict[str, dict[str, Any]],
    *,
    snapshot_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for holding in holdings:
        symbol = str(holding.get("symbol", "")).strip()
        weight = _parse_float(holding.get("weight"))
        overview = overviews.get(symbol, {})
        pe = _parse_float(overview.get("PERatio"))
        eps = _parse_float(overview.get("EPS"))
        market_cap = _parse_float(overview.get("MarketCapitalization"))
        included = bool(np.isfinite(weight) and weight > 0 and np.isfinite(pe) and pe > 0)
        rows.append(
            {
                "symbol": symbol,
                "description": holding.get("description"),
                "weight": weight,
                "alpha_vantage_pe_ratio": pe,
                "alpha_vantage_eps": eps,
                "alpha_vantage_market_cap": market_cap,
                "included_in_harmonic_pe": included,
                "download_error": overview.get("download_error", ""),
            }
        )
    table = pd.DataFrame(rows).sort_values("weight", ascending=False)
    included = table[table["included_in_harmonic_pe"]].copy()
    denominator = float((included["weight"] / included["alpha_vantage_pe_ratio"]).sum())
    included_weight = float(included["weight"].sum())
    total_weight = float(table["weight"].sum())
    harmonic_pe_normalized = included_weight / denominator if denominator > 0 else math.nan
    harmonic_pe_raw_total_weight_assumption = 1.0 / denominator if denominator > 0 else math.nan
    summary = {
        "method": "option_b_alpha_vantage_holdings_harmonic_pe_snapshot",
        "snapshot_date": snapshot_date,
        "holdings_count": int(len(table)),
        "holdings_total_weight": total_weight,
        "included_holdings_count": int(len(included)),
        "included_weight": included_weight,
        "excluded_weight": total_weight - included_weight,
        "weighted_harmonic_pe_normalized_to_included_weight": harmonic_pe_normalized,
        "weighted_harmonic_pe_raw_total_weight_assumption": harmonic_pe_raw_total_weight_assumption,
    }
    return table, summary


def main() -> None:
    args = parse_args()
    api_key = _api_key()
    raw_dir = Path("data/raw/valuation")
    snapshot_dir = raw_dir / f"alpha_vantage_overview_{args.snapshot_date}"
    tables_dir = Path("reports/tables")
    ensure_directory(raw_dir)
    ensure_directory(tables_dir)

    qqq_daily = fetch_daily_prices(args.symbol, api_key=api_key, raw_dir=raw_dir, force=args.force)
    option_a_table, option_a_summary = option_a_daily_proxy(
        qqq_daily,
        anchor_date=args.anchor_date,
        anchor_pe=args.anchor_pe,
        asof_date=args.asof_date,
    )

    holdings = fetch_av_holdings(args.symbol, api_key=api_key, raw_dir=raw_dir, force=args.force)
    symbols = [str(row["symbol"]).strip() for row in holdings if str(row.get("symbol", "")).strip()]
    overviews = fetch_overviews(
        symbols,
        api_key=api_key,
        raw_dir=snapshot_dir,
        pause_seconds=args.pause_seconds,
        force=args.force,
    )
    option_b_table, option_b_summary = option_b_av_harmonic_pe(
        holdings,
        overviews,
        snapshot_date=args.snapshot_date,
    )

    latest_a = float(option_a_summary["latest_proxy_pe"])
    latest_a_known = float(option_a_summary["latest_known_today_no_lookahead_pe"])
    latest_b = float(option_b_summary["weighted_harmonic_pe_normalized_to_included_weight"])
    comparison = pd.DataFrame(
        [
            option_a_summary,
            option_b_summary,
            {
                "method": "difference_b_minus_a_latest_proxy",
                "option_a_latest_proxy_pe": latest_a,
                "option_a_latest_known_today_no_lookahead_pe": latest_a_known,
                "option_b_alpha_vantage_snapshot_pe": latest_b,
                "difference_b_minus_a_proxy": latest_b - latest_a,
                "pct_difference_vs_option_a_proxy": (latest_b / latest_a - 1.0) if latest_a else math.nan,
                "difference_b_minus_a_known_today": latest_b - latest_a_known,
                "pct_difference_vs_option_a_known_today": (latest_b / latest_a_known - 1.0)
                if latest_a_known
                else math.nan,
            },
        ]
    )

    option_a_path = tables_dir / f"{args.output_prefix}_option_a_daily_proxy.csv"
    option_b_path = tables_dir / f"{args.output_prefix}_option_b_av_holdings.csv"
    comparison_path = tables_dir / f"{args.output_prefix}_summary.csv"
    option_a_table.to_csv(option_a_path)
    option_b_table.to_csv(option_b_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    print(f"Option A daily proxy saved to {option_a_path}")
    print(f"Option B holdings snapshot saved to {option_b_path}")
    print(f"Comparison summary saved to {comparison_path}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
