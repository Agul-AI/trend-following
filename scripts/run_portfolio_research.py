#!/usr/bin/env python3
"""QQQ Strategy Portfolio Study; retrospective development/validation only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trend_following.research_portfolio_protocol import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
