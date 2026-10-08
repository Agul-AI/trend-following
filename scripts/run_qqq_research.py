#!/usr/bin/env python3
"""Unversioned public entry point for the frozen QQQ gap-held research study."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_qqq_gap_flags_v5 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
