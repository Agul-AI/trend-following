#!/usr/bin/env python3
"""Explicit staged CLI for the isolated QQQ daily-indicator/hourly v5 study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trend_following import research_hourly_protocol_v5 as protocol  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated Alpha-Vantage-only v5 research: completed 30-minute checks, "
            "confirmation waits 0 to 6.5 hours after first hit (1 + 2*h checks), "
            "56,644 candidates; historical cutoff 2026-05-28. "
            "Prepare never downloads or backtests. Historical performance requires verified, "
            "frozen raw packets. No Yahoo price data or v4 output is read or modified."
        ),
    )
    parser.add_argument(
        "stage", choices=["prepare", "development", "validation", "historical-oos", "monitor"]
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qqq_daily_hourly_v5.yaml")
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=None,
        help="Private immutable artifacts; default .cache/qqq_daily_halfhour_v5_alpha_only",
    )
    parser.add_argument(
        "--input-packet",
        type=Path,
        default=None,
        help="Verified Alpha Vantage packet root (prepare or read-only monitor only)",
    )
    parser.add_argument(
        "--cash-file", type=Path, default=None, help="Causal DTB3 available-at CSV (prepare only)"
    )
    parser.add_argument(
        "--as-of", default=None, help="Timezone-aware monitor timestamp (monitor only)"
    )
    parser.add_argument(
        "--live", action="store_true", help="Use verified intraday monitor packet (monitor only)"
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Read-only half-hour v2 paper-state JSON; monitor without state is condition-only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stage not in {"prepare", "monitor"} and args.input_packet is not None:
        parser.error(
            "performance stages cannot accept arbitrary input packets; use sealed stage prefixes"
        )
    if args.stage != "prepare" and args.cash_file is not None:
        parser.error("--cash-file is prepare-only")
    if args.stage != "monitor" and (args.as_of is not None or args.live or args.state is not None):
        parser.error("--as-of, --live and --state are monitor-only")
    common = {"config_path": args.config, "study_dir": args.study_dir}
    try:
        if args.stage == "prepare":
            result = protocol.prepare_study(
                **common, input_packet=args.input_packet, cash_file=args.cash_file
            )
        elif args.stage == "development":
            result = protocol.run_development(**common)
        elif args.stage == "validation":
            result = protocol.run_validation(**common)
        elif args.stage == "historical-oos":
            result = protocol.run_historical_oos(**common)
        else:
            state = None if args.state is None else json.loads(args.state.read_text())
            result = protocol.monitor_winner(
                **common,
                input_packet=args.input_packet,
                as_of=args.as_of,
                state=state,
                live=args.live,
            )
    except (protocol.ProtocolError, ValueError, OSError, KeyError) as error:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "status": "rejected",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "historical_cutoff": protocol.HISTORICAL_CUTOFF,
                    "price_provider": protocol.PRICE_PROVIDER,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    result.setdefault("historical_cutoff", protocol.HISTORICAL_CUTOFF)
    result.setdefault("price_provider", protocol.PRICE_PROVIDER)
    print(json.dumps(result, indent=2, sort_keys=True, default=str, allow_nan=False))
    return 3 if result.get("status") == "data_not_ready" else 0


if __name__ == "__main__":
    raise SystemExit(main())
