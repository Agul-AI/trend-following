#!/usr/bin/env python
"""Save a point-in-time Option B QQQ P/E snapshot and append it to daily history.

Option B is a look-through weighted harmonic P/E estimate:
1. Pull QQQ holdings from Alpha Vantage ETF_PROFILE.
2. Pull each holding's current PERatio from Alpha Vantage OVERVIEW.
3. Compute weighted harmonic P/E across holdings with positive usable P/E.

Important limitation:
Alpha Vantage ETF_PROFILE and OVERVIEW are current snapshots, not historical
point-in-time series. This script therefore creates an append-only history from
snapshots saved on each run. It intentionally does not backfill past dates with
current holdings/fundamentals, because that would create lookahead bias.
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
from pandas.tseries.offsets import BDay

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.utils import ensure_directory  # noqa: E402

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="Snapshot date label, default today in local system date.",
    )
    parser.add_argument("--pause-seconds", type=float, default=0.85)
    parser.add_argument("--force", action="store_true", help="Refetch API files even if cached.")
    parser.add_argument("--raw-dir", default="data/raw/valuation")
    parser.add_argument("--processed-dir", default="data/processed/valuation")
    parser.add_argument("--tables-dir", default="reports/tables")
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


def fetch_holdings(symbol: str, *, api_key: str, snapshot_raw_dir: Path, force: bool) -> list[dict[str, Any]]:
    path = snapshot_raw_dir / f"alpha_vantage_etf_profile_{symbol}.json"
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


def fetch_overviews(
    symbols: list[str],
    *,
    api_key: str,
    overview_dir: Path,
    pause_seconds: float,
    force: bool,
) -> dict[str, dict[str, Any]]:
    overviews: dict[str, dict[str, Any]] = {}
    ensure_directory(overview_dir)
    next_start = time.monotonic()
    for i, symbol in enumerate(symbols, start=1):
        safe = symbol.replace("/", "_").replace(".", "_")
        path = overview_dir / f"{safe}.json"
        if path.exists() and not force:
            overviews[symbol] = json.loads(path.read_text())
            continue
        wait = next_start - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        next_start = time.monotonic() + pause_seconds
        try:
            data = _get_json({"function": "OVERVIEW", "symbol": symbol}, api_key=api_key)
        except Exception as exc:  # noqa: BLE001 - preserve failures in output
            data = {"Symbol": symbol, "download_error": str(exc)}
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
        overviews[symbol] = data
        if i % 25 == 0 or i == len(symbols):
            print(f"Fetched/cached {i}/{len(symbols)} Alpha Vantage OVERVIEW records")
    return overviews


def compute_option_b_pe(
    holdings: list[dict[str, Any]],
    overviews: dict[str, dict[str, Any]],
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
    normalized_pe = included_weight / denominator if denominator > 0 else math.nan
    raw_total_weight_pe = 1.0 / denominator if denominator > 0 else math.nan
    summary = {
        "holdings_count": int(len(table)),
        "holdings_total_weight": total_weight,
        "included_holdings_count": int(len(included)),
        "included_weight": included_weight,
        "excluded_weight": total_weight - included_weight,
        "qqq_pe_option_b_weighted_harmonic": normalized_pe,
        "qqq_pe_option_b_raw_total_weight_assumption": raw_total_weight_pe,
    }
    return table, summary


def append_history(history_path: Path, row: dict[str, Any]) -> pd.DataFrame:
    new = pd.DataFrame([row])
    if history_path.exists():
        existing = pd.read_parquet(history_path)
        history = pd.concat([existing, new], ignore_index=True)
    else:
        history = new
    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"]).dt.date.astype(str)
    history = history.sort_values("snapshot_date").drop_duplicates("snapshot_date", keep="last")
    ensure_directory(history_path.parent)
    history.to_parquet(history_path, index=False)
    return history


def main() -> None:
    args = parse_args()
    api_key = _api_key()
    snapshot_date = args.snapshot_date or pd.Timestamp.today().date().isoformat()
    usable_from = (pd.Timestamp(snapshot_date) + BDay(1)).date().isoformat()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    tables_dir = Path(args.tables_dir)
    snapshot_raw_dir = raw_dir / "alpha_vantage_option_b_snapshots" / snapshot_date
    overview_dir = snapshot_raw_dir / "overviews"
    ensure_directory(snapshot_raw_dir)
    ensure_directory(processed_dir)
    ensure_directory(tables_dir)

    holdings = fetch_holdings(args.symbol, api_key=api_key, snapshot_raw_dir=snapshot_raw_dir, force=args.force)
    symbols = [str(row.get("symbol", "")).strip() for row in holdings if str(row.get("symbol", "")).strip()]
    overviews = fetch_overviews(
        symbols,
        api_key=api_key,
        overview_dir=overview_dir,
        pause_seconds=args.pause_seconds,
        force=args.force,
    )
    holdings_table, summary = compute_option_b_pe(holdings, overviews)

    row = {
        "snapshot_date": snapshot_date,
        "usable_from_date_no_lookahead": usable_from,
        "symbol": args.symbol,
        "source": "Alpha Vantage ETF_PROFILE + OVERVIEW",
        "method": "weighted_harmonic_pe_normalized_to_positive_pe_holdings",
        **summary,
        "notes": (
            "Point-in-time snapshot saved on run date. Do not backfill older dates with this row; "
            "for no-lookahead trading, use from usable_from_date_no_lookahead onward."
        ),
    }

    history_path = processed_dir / "qqq_pe_option_b_daily_history.parquet"
    history = append_history(history_path, row)

    latest_holdings_path = tables_dir / "qqq_pe_option_b_latest_holdings.csv"
    history_csv_path = tables_dir / "qqq_pe_option_b_daily_history.csv"
    latest_summary_path = tables_dir / "qqq_pe_option_b_latest_summary.csv"
    holdings_table.to_csv(latest_holdings_path, index=False)
    history.to_csv(history_csv_path, index=False)
    pd.DataFrame([row]).to_csv(latest_summary_path, index=False)

    print(f"Saved Option B QQQ PE history parquet to {history_path}")
    print(f"Saved Option B QQQ PE history csv to {history_csv_path}")
    print(f"Saved latest holdings detail to {latest_holdings_path}")
    print(f"Saved latest summary to {latest_summary_path}")
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
