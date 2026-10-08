#!/usr/bin/env python
"""Fresh chronological daily experiment, separate from all previous results.

Each stage writes new artifacts and refuses to overwrite an earlier attempt.
No broker, trade placement, scheduler, notification or resume modification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True  # Keep archived numerical snapshots byte-inventory clean.
sys.path.insert(0, str(ROOT / "src"))

from trend_following.research_split_protocol_v3 import (  # noqa: E402
    evaluate_split_study,
    freeze_split_study,
    select_split_candidate,
    verify_split_study,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze", help="seal code, split, candidates and raw inputs BEFORE selection"
    )
    freeze.add_argument("--config", type=Path, default=ROOT / "configs/qqq_split_v3.yaml")
    freeze.add_argument("--qqq-daily", type=Path, required=True)
    freeze.add_argument("--reference-daily", type=Path, required=True)
    freeze.add_argument("--rate-dir", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    for name in ("verify", "select", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_split_study(
            ROOT,
            args.config.resolve(),
            args.qqq_daily.resolve(),
            args.reference_daily.resolve(),
            args.rate_dir.resolve(),
            args.output.resolve(),
        )
    elif args.command == "verify":
        result = verify_split_study(ROOT, args.study.resolve())
    elif args.command == "select":
        result = select_split_candidate(ROOT, args.study.resolve())
    else:
        result = evaluate_split_study(ROOT, args.study.resolve())
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
