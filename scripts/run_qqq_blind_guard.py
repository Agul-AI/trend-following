#!/usr/bin/env python3
"""Standalone custody CLI; never imports prior repository strategy modules."""

import importlib.util
import sys
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "src/trend_following/research_blind_guard_v4.py"
spec = importlib.util.spec_from_file_location("research_blind_guard_v4", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
if __name__ == "__main__":
    raise SystemExit(module.main())
