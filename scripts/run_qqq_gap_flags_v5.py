#!/usr/bin/env python3
"""New, separately sealed user-authorized gap-flags entry point; no automatic market runs."""

from __future__ import annotations

import sys
from pathlib import Path

from run_qqq_reconciled_v5 import build_parser
from run_qqq_reconciled_v5 import main as quality_main

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args:
        parser = build_parser()
        parser.description = (
            "Separately sealed, user-authorized observed-only gap study: hold the last "
            "filled QQQ/cash position through missing quotes; flag confirmed buy/sell "
            "signals after gaps through the NEXT trading session close. Flags are "
            "descriptive only and never filter/rank candidates. Preserve all events; "
            "reject sealing if truncated. No automatic market evaluation."
        )
        for action in parser._actions:
            if action.dest == "study_dir":
                action.help = "Private immutable artifacts; default .cache/qqq_daily_halfhour_v5_alpha_gap_flags_v2"
            elif action.dest == "config":
                action.help = "Default configs/qqq_daily_halfhour_v5_gap_flags_v2.yaml"
        parser.print_help()
        return 0
    if not any(a == "--config" or a.startswith("--config=") for a in args):
        args += ["--config", str(ROOT / "configs/qqq_daily_halfhour_v5_gap_flags_v2.yaml")]
    if not any(a == "--study-dir" or a.startswith("--study-dir=") for a in args):
        args += ["--study-dir", str(ROOT / ".cache/qqq_daily_halfhour_v5_alpha_gap_flags_v2")]
    return quality_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
