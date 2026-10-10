#!/usr/bin/env python3
"""QQQ hybrid allocation comparison; development and retrospective validation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trend_following.research_hybrid_allocation_protocol import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
