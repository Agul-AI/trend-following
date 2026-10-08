"""Presentation alias must not rename or mutate any frozen research interface."""

import subprocess
import sys
from pathlib import Path


def test_unversioned_public_help_delegates_without_running_market_stages():
    script = Path(__file__).resolve().parents[1] / "scripts/run_qqq_research.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.startswith("Trend Following Study\n")
    assert "usage: run_qqq_research.py" in result.stdout
    assert "user-authorized observed-only gap study" in result.stdout
    assert "No automatic market evaluation" in result.stdout
    assert "{prepare,development,validation,historical-oos,monitor}" in result.stdout
