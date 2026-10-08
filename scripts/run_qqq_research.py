#!/usr/bin/env python3
"""Trend Following Study: public entry point for the frozen QQQ/cash research."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_qqq_gap_flags_v5 import main  # noqa: E402

if __name__ == "__main__":
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        print("Trend Following Study\n")
    raise SystemExit(main())
