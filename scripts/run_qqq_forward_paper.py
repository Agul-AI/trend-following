#!/usr/bin/env python3
"""Manual, offline source-to-policy PAPER bridge. Never places real orders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.research_forward_bridge import (  # noqa: E402
    initialize_paper_bridge,
    record_completed_bar,
    verify_paper_bridge,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "verify", "record"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Immutable combined single-bar source JSON")
    parser.add_argument(
        "--acknowledge-fill",
        action="store_true",
        help=(
            "Accept the prior target order's paper next-close fill assumption; no broker execution. "
            "Without this, an eligible fill is recorded as missed and the pilot stops."
        ),
    )
    args = parser.parse_args()
    common = (args.root, args.contract, args.manifest, args.ledger)
    if args.command == "init":
        result = initialize_paper_bridge(*common)
    elif args.command == "verify":
        result = verify_paper_bridge(*common)
    else:
        if args.source is None:
            parser.error("record requires --source")
        result = record_completed_bar(*common, args.source, acknowledge_fill=args.acknowledge_fill)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
