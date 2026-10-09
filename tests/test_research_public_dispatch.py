"""Public routing tests are synthetic; never authorize or run market stages."""

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "public_research", ROOT / "scripts/run_qqq_research.py"
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_existing_arguments_forwarded_without_change(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "frozen_main", lambda args: seen.append(args) or 0)
    assert cli.main(["validation", "--config", "example.yaml"]) == 0
    assert seen == [["validation", "--config", "example.yaml"]]


def test_shortlist_routes_separately_without_original_stage_call(monkeypatch):
    import sys

    module = ModuleType("trend_following.research_shortlist")
    seen = []
    module.main = lambda args: seen.append(args) or 0
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cli, "frozen_main", lambda args: (_ for _ in ()).throw(AssertionError()))
    assert cli.main(["shortlist", "seal", "--per-family", "5"]) == 0
    assert seen == [["seal", "--per-family", "5"]]


def test_public_help_explains_original_and_exploratory_routes(monkeypatch, capsys):
    monkeypatch.setattr(cli, "frozen_main", lambda args: 0)
    assert cli.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "Trend Following Study" in output
    assert "shortlist --help" in output
    assert "never replaces the original sealed winner" in output
