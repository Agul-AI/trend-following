#!/usr/bin/env python
"""Freeze QQQ protocol and provenance before evaluation, never backdate holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.research_protocol import freeze_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, default=Path("configs/qqq_research_v1.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["protocol_only", "final_code"], default="protocol_only")
    parser.add_argument("--attest-final-code-ready", action="store_true")
    parser.add_argument("--session-calendar", type=Path, help="CSV with verified timezone-aware XNYS market_open timestamps")
    args = parser.parse_args()
    protocol = args.protocol if args.protocol.is_absolute() else args.root / args.protocol
    manifest = freeze_protocol(
        args.root, protocol, args.output_dir, stage=args.stage,
        final_code_ready=args.attest_final_code_ready, session_calendar_path=args.session_calendar,
    )
    print(json.dumps({key: manifest[key] for key in ["protocol_id", "stage", "frozen_at_utc", "parameters_sha256", "genuine_holdout_start_utc"]}, indent=2))
    print(f"Manifest: {(args.output_dir / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
