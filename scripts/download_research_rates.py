#!/usr/bin/env python
"""Download public FRED DFF/DTB3 into an immutable, documented research cache."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.research_rates import (  # noqa: E402
    PUBLICATION_SOURCES,
    RATE_SOURCES,
    prepare_rate_observations,
)


def download_rates(output_dir: Path, *, start: str = "1999-01-01") -> dict:
    """Create a fresh cache; a repeated output path refuses to overwrite history."""
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "start_requested": start,
        "publication_sources": PUBLICATION_SOURCES,
        "availability_policy": "two US federal business days after observation, 18:00 New York",
        "availability_limitation": "assumed conservative lag, not historical release-time reconstruction",
        "vintage": "latest_vintage_not_point_in_time; no ALFRED vintages retrieved",
        "cash_proxy": "DTB3 is bank-discount quote; convert assuming 91-day bill to ACT/365 simple investment yield; not traded cash total returns",
        "financing_proxy": "DFF plus explicit spread; unsecured rate proxy, not actual investor borrowing terms; ACT/360",
        "missing_policy": "drop unavailable quotes; accrual fails on uncovered or stale intervals; no zero fill",
        "series": {},
    }
    for series in RATE_SOURCES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        raw_bytes = response.content
        raw = pd.read_csv(io.BytesIO(raw_bytes))
        observations = prepare_rate_observations(raw, series)
        (output_dir / f"{series}_raw.csv").write_bytes(raw_bytes)
        normalized = output_dir / f"{series}_available.csv"
        observations.to_csv(normalized, index=False)
        metadata["series"][series] = {
            "url": url,
            "documentation": RATE_SOURCES[series],
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "normalized_sha256": hashlib.sha256(normalized.read_bytes()).hexdigest(),
            "rows_raw": len(raw),
            "rows_available": len(observations),
            "dropped_missing_quotes": len(raw) - len(observations),
            "observation_start": str(observations.observation_date.min()),
            "observation_end": str(observations.observation_date.max()),
        }
    # Written last: absent metadata identifies an incomplete download.
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", default="1999-01-01")
    args = parser.parse_args()
    output = args.output_dir or Path("data/raw/research_rates") / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    metadata = download_rates(output, start=args.start)
    print(json.dumps({"output_dir": str(output.resolve()), "series": metadata["series"]}, indent=2))


if __name__ == "__main__":
    main()
