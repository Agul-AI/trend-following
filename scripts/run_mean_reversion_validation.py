#!/usr/bin/env python3
"""Mean-Reversion Study: frozen 32-case validation only; no OOS route."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.research_mean_reversion_validation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
