#!/usr/bin/env python
"""Coordinator-only phase locks/releases; never expose held data to the researcher."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from trend_following.research_blind_protocol_v4 import (  # noqa: E402
    lock_proposal,
    run_test,
    run_validation,
    verify_protocol,
    verify_selection,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    proposal = subs.add_parser("lock-proposal")
    proposal.add_argument("--study", type=Path, required=True)
    proposal.add_argument("--policy", type=Path, required=True)
    proposal.add_argument("--proposal", type=Path, required=True)
    proposal.add_argument("--development-log", type=Path, required=True)
    for name in ["validate", "test", "verify", "verify-selection"]:
        command = subs.add_parser(name)
        command.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lock-proposal":
        value = lock_proposal(args.study, args.policy, args.proposal, args.development_log)
    elif args.command == "validate":
        value = run_validation(args.study)
    elif args.command == "test":
        value = run_test(args.study)
    elif args.command == "verify":
        value = verify_protocol(args.study)
    else:
        value = verify_selection(args.study)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
