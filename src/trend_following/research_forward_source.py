"""Manual, immutable source snapshots for the frozen PAPER bridge.

Fetches no orders, runs no strategy and creates no schedule. Quotes and provider
bar labels are source attestations, not independently authenticated market truth.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from trend_following.research_protocol import sha256_file, verify_frozen_inputs


def latest_closed_bar(calendar: pd.DataFrame, now: pd.Timestamp, holdout: pd.Timestamp):
    if now.tzinfo is None or holdout.tzinfo is None:
        raise ValueError("timezone-aware clock and holdout required")
    opens = pd.DatetimeIndex([pd.Timestamp(v).tz_convert("UTC") for v in calendar.market_open])
    closes = pd.DatetimeIndex([pd.Timestamp(v).tz_convert("UTC") for v in calendar.market_close])
    if (
        opens.hasnans
        or closes.hasnans
        or not opens.is_unique
        or not opens.is_monotonic_increasing
        or (closes <= opens).any()
        or ((closes - opens) > pd.Timedelta(hours=6, minutes=30)).any()
    ):
        raise ValueError("invalid, duplicate or unordered exchange calendar")
    latest = None
    for row in calendar.itertuples(index=False):
        opening = pd.Timestamp(row.market_open).tz_convert("UTC")
        close = pd.Timestamp(row.market_close).tz_convert("UTC")
        if opening < holdout or opening >= now:
            continue
        starts = pd.date_range(opening, close, freq="h", inclusive="left")
        for start in starts:
            end = min(start + pd.Timedelta(hours=1), close)
            if end <= now:
                latest = (start, end)
    if latest is None:
        raise ValueError("no completed holdout bar is available yet")
    if (now - latest[1]).total_seconds() > 900:
        raise ValueError("latest completed bar is stale; no backfilled paper evidence")
    return latest


def source_row(frame: pd.DataFrame, start: pd.Timestamp) -> dict:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("provider rows require timezone-aware bar-start labels")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("provider timestamps must be unique and chronological")
    frame = frame.copy()
    frame.index = frame.index.tz_convert("UTC")
    frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
    required = {"open", "high", "low", "close", "volume", "dividends", "stock_splits"}
    if not required.issubset(frame.columns) or start not in frame.index:
        raise ValueError("provider is missing the completed bar or required action fields")
    row = frame.loc[start]
    if any(isinstance(row[key], (bool, np.bool_)) for key in required):
        raise ValueError("boolean provider prices/actions/volume are invalid")
    values = {key: float(row[key]) for key in required}
    if "capital_gains" in row.index:
        raw_gain = row["capital_gains"]
        if (
            isinstance(raw_gain, (bool, np.bool_))
            or not np.isfinite(float(raw_gain))
            or float(raw_gain) != 0
        ):
            raise ValueError("capital-gain distribution requires review; bridge V1 cannot advance")
        values["capital_gains"] = float(raw_gain)
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("nonfinite provider values")
    if min(values[k] for k in ["open", "high", "low", "close"]) <= 0 or values["volume"] < 0:
        raise ValueError("invalid provider prices/volume")
    if (
        not values["low"]
        <= min(values["open"], values["close"])
        <= max(values["open"], values["close"])
        <= values["high"]
    ):
        raise ValueError("inconsistent provider OHLC")
    if values["dividends"] or values["stock_splits"]:
        raise ValueError("corporate action requires review; bridge V1 cannot advance")
    return {"bar_start": start.isoformat(), **values}


def collect_snapshot(
    root: Path,
    manifest_path: Path,
    output_file: Path,
    *,
    loader: Callable[[str], pd.DataFrame] | None = None,
    clock: Callable[[], pd.Timestamp] | None = None,
) -> dict:
    clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("stage") != "final_code"
        or manifest.get("protocol_id") != "qqq_forward_paper_v1"
        or manifest.get("prospective_eligible") is not True
    ):
        raise ValueError("the final fixed-B5 forward manifest is required")
    verify_frozen_inputs(root, manifest_path)
    contract_file = manifest_path.parent / "protocol.yaml"
    if sha256_file(contract_file) != manifest["protocol_sha256"]:
        raise ValueError("frozen forward contract changed")
    contract = yaml.safe_load(contract_file.read_text())
    if (
        contract.get("bars_per_day") != 7
        or contract.get("provider") != "Yahoo Finance via yfinance"
    ):
        raise ValueError("unsupported source contract")
    holdout = pd.Timestamp(manifest["genuine_holdout_start_utc"])
    records = manifest["data_files"] + manifest["code_files"]
    record = next(r for r in records if r["path"] == contract["exchange_calendar"]["path"])
    if record["sha256"] != contract["exchange_calendar"]["sha256"]:
        raise ValueError("exchange calendar is not bound to the frozen contract")
    calendar_file = manifest_path.parent / record["snapshot_path"]
    calendar = pd.read_csv(calendar_file)
    start, end = latest_closed_bar(calendar, clock().tz_convert("UTC"), holdout)
    if output_file.exists():
        raise FileExistsError("refusing to overwrite source evidence")
    try:
        output_file.resolve().relative_to((manifest_path.parent / "snapshot").resolve())
    except ValueError:
        pass
    else:
        raise ValueError("new observations must not be written into a frozen archive")
    if loader is None:
        import yfinance as yf

        def loader(symbol):
            return yf.Ticker(symbol).history(
                period="5d",
                interval="60m",
                auto_adjust=False,
                back_adjust=False,
                actions=True,
                prepost=False,
            )

    bars = {symbol: source_row(loader(symbol), start) for symbol in ["QQQ", "TQQQ"]}
    retrieved = clock().tz_convert("UTC")
    if not end <= retrieved <= end + pd.Timedelta(seconds=900):
        raise ValueError("retrieval is not fresh and completed; no snapshot written")
    payload = {
        "schema_version": 1,
        "provider": contract["provider"],
        "interval": "60m",
        "retrieved_at": retrieved.isoformat(),
        "bars": bars,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x") as stream:
        stream.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "path": str(output_file.resolve()),
        "sha256": sha256_file(output_file),
        "bar_start": start.isoformat(),
        "bar_end": end.isoformat(),
        "retrieved_at": retrieved.isoformat(),
        "paper_only": True,
    }
