#!/usr/bin/env python3
"""Trend Following Study: frozen research and separately sealed shortlist follow-up."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_qqq_gap_flags_v5 import main as frozen_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "shortlist":
        sys.path.insert(0, str(ROOT / "src"))
        from trend_following.research_shortlist import main as shortlist_main

        return shortlist_main(args[1:])
    if "--help" in args or "-h" in args:
        print("Trend Following Study\n")
    result = frozen_main(args)
    if "--help" in args or "-h" in args:
        print(
            "\nExploratory set comparison: scripts/run_qqq_research.py shortlist --help\n"
            "The shortlist follow-up never replaces the original sealed winner or monitor."
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
