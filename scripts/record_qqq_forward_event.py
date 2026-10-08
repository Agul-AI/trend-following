#!/usr/bin/env python
"""Record genuine prospective PAPER events; no network, broker, or scheduling.

Examples (only AFTER a final_code manifest exists):
  python scripts/record_qqq_forward_event.py init --manifest /abs/manifest.json --ledger /abs/paper.jsonl
  python scripts/record_qqq_forward_event.py verify --manifest /abs/manifest.json --ledger /abs/paper.jsonl
  python scripts/record_qqq_forward_event.py append --manifest /abs/manifest.json --ledger /abs/paper.jsonl --event /abs/observed_event.json

Append requires the current head_hash as event.expected_previous_hash. Events
must be <=15 minutes old, at/after the real holdout start, never in the future.
Do not backfill historical events or fabricate timestamps to satisfy the gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.research_forward import (  # noqa: E402
    append_forward_event,
    initialize_forward_ledger,
    verify_forward_ledger,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["init", "append", "verify"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--event", type=Path, help="Genuinely observed event JSON, required for append"
    )
    parser.add_argument("--initial-cash", type=float, default=1.0)
    args = parser.parse_args()
    if args.operation == "init":
        result = initialize_forward_ledger(
            args.root, args.manifest, args.ledger, initial_cash=args.initial_cash
        )
    elif args.operation == "verify":
        result = verify_forward_ledger(args.root, args.manifest, args.ledger)
    else:
        if args.event is None:
            parser.error("append requires --event")
        result = append_forward_event(
            args.root, args.manifest, args.ledger, json.loads(args.event.read_text())
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
