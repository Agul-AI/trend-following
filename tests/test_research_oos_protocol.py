"""Artificial trust-boundary tests; production CLI has no time-override flag."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from tests.test_research_oos_data import production_package

import trend_following.research_oos_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artificial_study(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    cfg = yaml.safe_load((ROOT / "configs/qqq_oos_v2.yaml").read_text())
    for relative in {
        *cfg["frozen_code_paths"],
        cfg["calendar"],
        cfg["distribution_calendar"],
        "configs/qqq_oos_v2.yaml",
    }:
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, p)
    package, _ = production_package(tmp_path)

    def calibration(*args, training_end, **kwargs):
        return {
            "multiplier": 0.3,
            "training_end": pd.Timestamp(training_end).isoformat(),
            "requested_training_cutoff": pd.Timestamp(training_end).isoformat(),
            "calibration_label": "artificial_unit_test_not_empirical_performance",
        }

    monkeypatch.setattr(protocol, "fit_pretrain_multiplier", calibration)
    return root, root / "configs/qqq_oos_v2.yaml", package, tmp_path / "freeze"


def test_required_code_closure_cannot_be_removed(tmp_path):
    d = yaml.safe_load((ROOT / "configs/qqq_oos_v2.yaml").read_text())
    d["frozen_code_paths"].remove("src/trend_following/research_rates.py")
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="required runtime code closure"):
        protocol.load_oos_config(p)


def test_module_origins_bind_executing_checkout(tmp_path):
    protocol.verify_runtime_origins(ROOT)
    with pytest.raises(ValueError, match="executing modules"):
        protocol.verify_runtime_origins(tmp_path)


def test_synthetic_freeze_and_reconstruction(artificial_study):
    root, cfg, warm, out = artificial_study
    m = protocol.freeze_oos_study(
        root, cfg, warm, out, attest_ready=True, _test_now=pd.Timestamp("2024-10-03 22:00Z")
    )
    assert m["returns_observed"] == 0
    assert m["genuine_holdout_start_utc"] == "2024-10-04T13:30:00+00:00"
    assert protocol.verify_oos_freeze(root, out / "manifest.json", verify_runtime=False) == m
    with pytest.raises(FileExistsError):
        protocol.freeze_oos_study(
            root, cfg, warm, out, attest_ready=True, _test_now=pd.Timestamp("2024-10-03 22:00Z")
        )


def test_no_freeze_without_attestation(artificial_study):
    root, cfg, warm, out = artificial_study
    with pytest.raises(ValueError, match="attestation"):
        protocol.freeze_oos_study(root, cfg, warm, out, _test_now=pd.Timestamp("2024-10-03 22:00Z"))
    assert not out.exists()


def test_missing_warmup_refuses_no_completion(artificial_study, tmp_path):
    root, cfg, _, out = artificial_study
    warm, _ = production_package(tmp_path, "missing", missing=True)
    r = protocol.assess_readiness(root, cfg, warm, as_of=pd.Timestamp("2024-10-03 22:00Z"))
    assert not r["ready"]
    with pytest.raises(ValueError, match="not ready"):
        protocol.freeze_oos_study(
            root, cfg, warm, out, attest_ready=True, _test_now=pd.Timestamp("2024-10-03 22:00Z")
        )
    assert not out.exists()


def test_open_session_refuses_freeze(artificial_study):
    root, cfg, warm, out = artificial_study
    with pytest.raises(ValueError, match="not ready"):
        protocol.freeze_oos_study(
            root, cfg, warm, out, attest_ready=True, _test_now=pd.Timestamp("2024-10-04 14:00Z")
        )
    assert not out.exists()


def _resign(out, m):
    path = out / "manifest.json"
    path.write_text(json.dumps(m, indent=2) + "\n")
    receipt = json.loads((out / "receipt.json").read_text())
    receipt["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "receipt.json").write_text(json.dumps(receipt))


@pytest.mark.parametrize("change", ["start", "multiplier", "training_cutoff", "future_time"])
def test_rederives_timings_even_if_manifest_and_receipt_changed(artificial_study, change):
    root, cfg, warm, out = artificial_study
    m = protocol.freeze_oos_study(
        root, cfg, warm, out, attest_ready=True, _test_now=pd.Timestamp("2024-10-03 22:00Z")
    )
    if change == "start":
        m["genuine_holdout_start_utc"] = "2024-10-07T13:30:00+00:00"
    if change == "multiplier":
        m["calibration"]["multiplier"] = 0.9
    if change == "training_cutoff":
        m["calibration"]["training_end"] = "2024-10-02T20:00:00+00:00"
    if change == "future_time":
        m["frozen_at_utc"] = "2099-01-01T00:00:00+00:00"
    _resign(out, m)
    with pytest.raises(ValueError):
        protocol.verify_oos_freeze(root, out / "manifest.json", verify_runtime=False)


def test_manifest_symlink_rejected_before_resolve(artificial_study):
    root, cfg, warm, out = artificial_study
    protocol.freeze_oos_study(
        root, cfg, warm, out, attest_ready=True, _test_now=pd.Timestamp("2024-10-03 22:00Z")
    )
    link = out / "link.json"
    link.symlink_to(out / "manifest.json")
    with pytest.raises(ValueError, match="symlink"):
        protocol.verify_oos_freeze(root, link, verify_runtime=False)


def test_changed_runtime_source_rejected(artificial_study):
    root, cfg, warm, out = artificial_study
    protocol.freeze_oos_study(
        root, cfg, warm, out, attest_ready=True, _test_now=pd.Timestamp("2024-10-03 22:00Z")
    )
    p = root / "src/trend_following/research_rates.py"
    p.write_text(p.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="frozen code"):
        protocol.verify_oos_freeze(root, out / "manifest.json", verify_runtime=False)


def test_final_clock_after_expensive_work_rejects_crossed_open(artificial_study, monkeypatch):
    root, cfg, warm, out = artificial_study
    stamps = iter(
        [
            pd.Timestamp("2024-10-03 22:00Z"),
            pd.Timestamp("2024-10-04 13:29Z"),
            pd.Timestamp("2024-10-04 13:30Z"),
        ]
    )
    monkeypatch.setattr(protocol, "now_utc", lambda: next(stamps))
    monkeypatch.setattr(protocol, "verify_runtime_origins", lambda root: None)
    with pytest.raises(ValueError, match="crossed a market"):
        protocol.freeze_oos_study(root, cfg, warm, out, attest_ready=True)
    assert not (out / "manifest.json").exists()


def test_marker_write_crossing_open_is_not_completed_freeze(artificial_study, monkeypatch):
    root, cfg, warm, out = artificial_study
    stamps = iter(
        [
            pd.Timestamp("2024-10-03 22:00Z"),
            pd.Timestamp("2024-10-04 13:29Z"),
            pd.Timestamp("2024-10-04 13:29:59Z"),
            pd.Timestamp("2024-10-04 13:30Z"),
        ]
    )
    monkeypatch.setattr(protocol, "now_utc", lambda: next(stamps))
    monkeypatch.setattr(protocol, "verify_runtime_origins", lambda root: None)
    with pytest.raises(ValueError, match="completion marker crossed"):
        protocol.freeze_oos_study(root, cfg, warm, out, attest_ready=True)
    assert not (out / "manifest.json").exists()


def test_cli_does_not_accept_future_clock_or_foreign_checkout(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/run_qqq_oos_study.py")]
    r = subprocess.run(
        [*command, "--root", str(tmp_path), "verify", "--manifest", "unused"],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "executing checkout" in r.stderr
    r = subprocess.run(
        [
            *command,
            "score",
            "--as-of",
            "2099-01-01",
            "--manifest",
            "x",
            "--months",
            "12",
            "--output",
            "x",
            "--future-package",
            "x",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "unrecognized arguments" in r.stderr


def test_premature_score_writes_nothing(artificial_study, monkeypatch, tmp_path):
    root, cfg, warm, out = artificial_study
    m = protocol.freeze_oos_study(
        root, cfg, warm, out, attest_ready=True, _test_now=pd.Timestamp("2024-10-03 22:00Z")
    )
    monkeypatch.setattr(protocol, "verify_oos_freeze", lambda *args, **kwargs: m)
    target = tmp_path / "no_early_results"
    with pytest.raises(ValueError, match="not yet elapsed"):
        protocol.score_oos_study(
            root, out / "manifest.json", [], 12, target, _test_now=pd.Timestamp("2025-01-01 22:00Z")
        )
    assert not target.exists()


def test_output_cannot_recursively_copy_warmup(artificial_study):
    root, cfg, warm, _ = artificial_study
    out = warm / "recursive_output"
    with pytest.raises(ValueError, match="inside immutable"):
        protocol.freeze_oos_study(
            root, cfg, warm, out, attest_ready=True, _test_now=pd.Timestamp("2024-10-03 22:00Z")
        )
    assert not out.exists()


def test_score_output_cannot_mutate_inputs(artificial_study):
    root, _, warm, out = artificial_study
    with pytest.raises(ValueError, match="inside immutable"):
        protocol.score_oos_study(root, out / "manifest.json", [warm], 12, warm / "results")
