#!/usr/bin/env python
"""Save one fresh completed source bar for manual PAPER validation only.

Run only after the final holdout starts, within 15 minutes of a scheduled bar end.
Never submits orders, appends performance, refreshes frozen warmup or schedules jobs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trend_following.research_forward_source import collect_snapshot  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect_snapshot(args.root, args.manifest, args.output_file), indent=2))


if __name__ == "__main__":
    main()
