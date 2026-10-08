"""Synthetic-only mechanical custody tests; no historical strategy inspection."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

MODULE = Path(__file__).resolve().parents[1] / "src/trend_following/research_blind_guard_v4.py"
spec = importlib.util.spec_from_file_location("research_blind_guard_v4_tests", MODULE)
guard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guard
spec.loader.exec_module(guard)


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    sessions = [
        "1999-11-01",
        "2002-01-02",
        "2011-12-30",
        "2012-01-03",
        "2015-12-31",
        "2016-01-04",
        "2026-06-01",
        "2026-06-02",
    ]
    pd.DataFrame(
        {
            "date": pd.to_datetime(sessions),
            "close": 100.0,
            "dividend_amount": 0.0,
            "split_coefficient": 1.0,
            "adj_close": 123456.0,
        }
    ).to_parquet(source / "av.parquet", index=False)
    pd.DataFrame({"date": pd.to_datetime(sessions), "close": -99999.0}).to_parquet(
        source / "reference.parquet", index=False
    )
    observations = pd.to_datetime(
        [
            "1999-01-04",
            "2011-12-27",
            "2011-12-30",
            "2015-12-28",
            "2015-12-31",
            "2026-05-27",
            "2026-06-01",
        ]
    )
    offset = CustomBusinessDay(2, calendar=USFederalHolidayCalendar())
    timestamps = (
        pd.DatetimeIndex([d + offset for d in observations]).tz_localize("America/New_York")
        + pd.Timedelta(hours=18)
    ).tz_convert("UTC")
    for series in ("DFF", "DTB3"):
        pd.DataFrame({"observation_date": observations.strftime("%Y-%m-%d"), series: 4.0}).to_csv(
            source / f"{series}_raw.csv", index=False
        )
        annual = 0.04 if series == "DFF" else 0.04 * (365 / 360) / (1 - 0.04 * (91 / 360))
        pd.DataFrame(
            {
                "observation_date": observations.strftime("%Y-%m-%d"),
                "quoted_percent": 4.0,
                "available_at": timestamps,
                "annual_rate": annual,
                "series_id": series,
                "source_url": "synthetic-source",
                "vintage_type": "synthetic",
                "availability_assumption": "synthetic",
            }
        ).to_csv(source / f"{series}_available.csv", index=False)
    (source / "metadata.json").write_text(
        json.dumps({"vintage": "synthetic", "availability_limitation": "synthetic lag"})
    )
    engine = tmp_path / "new_engine"
    engine.mkdir()
    (engine / "engine.py").write_text('def marker():\n    return "fresh-accounting-engine"\n')
    sources = {name: source / name for name in guard.SOURCES if name not in ("qqq", "sessions")}
    sources.update(qqq=source / "av.parquet", sessions=source / "reference.parquet")
    return tmp_path, engine, sources


def prepared(inputs):
    tmp, engine, sources = inputs
    study = tmp / "private_study"
    return study, guard.prepare_study(study, engine, sources=sources)


def test_physical_partition_and_rate_availability(inputs):
    study, manifest = prepared(inputs)
    dev = pd.read_csv(study / "developer/input/QQQ.csv")
    assert list(dev.columns) == ["session", "close", "dividend_amount", "split_coefficient"]
    assert dev.session.tolist() == ["1999-11-01", "2002-01-02", "2011-12-30"]
    assert (dev.close == 100).all()  # reference price is deliberately unrelated
    assert pd.read_csv(study / "held/validation/QQQ.csv").session.tolist() == [
        "2012-01-03",
        "2015-12-31",
    ]
    assert pd.read_csv(study / "held/test/QQQ.csv").session.tolist() == ["2016-01-04", "2026-06-01"]
    for stage in guard.STAGES:
        directory = study / ("developer/input" if stage == "development" else f"held/{stage}")
        for series in ("DFF", "DTB3"):
            rate = pd.read_csv(directory / f"{series}_available.csv")
            assert (
                pd.to_datetime(rate.available_at, utc=True)
                <= pd.Timestamp(manifest["stages"][stage]["rate_cutoff_utc"])
            ).all()
    assert manifest["stages"]["development"]["scoring_start"] == "2002-01-01"
    assert "not Unix uid 0" in manifest["custodian_model"]
    assert not os.stat(study / "developer/input/QQQ.csv").st_mode & 0o222
    assert set(manifest["rate_provenance_hashes"]) == {
        "metadata.json",
        "DFF_raw.csv",
        "DTB3_raw.csv",
        "DFF_available.csv",
        "DTB3_available.csv",
    }


def test_coverage_mismatch_fails_before_creation(inputs):
    tmp, engine, sources = inputs
    pd.read_parquet(sources["sessions"], columns=["date"]).iloc[:-2].to_parquet(
        sources["sessions"], index=False
    )
    with pytest.raises(guard.GuardError, match="session dates"):
        guard.prepare_study(tmp / "failed", engine, sources=sources)
    assert not (tmp / "failed").exists()


def test_rate_normalization_mismatch_fails_closed(inputs):
    tmp, engine, sources = inputs
    frame = pd.read_csv(sources["DFF_available.csv"])
    frame.loc[0, "annual_rate"] = 99
    frame.to_csv(sources["DFF_available.csv"], index=False)
    with pytest.raises(guard.GuardError, match="annual-rate"):
        guard.prepare_study(tmp / "failed", engine, sources=sources)


def test_prepare_does_not_overwrite(inputs):
    mapping = inputs[0] / "explicit_sources.json"
    mapping.write_text(json.dumps({name: str(path) for name, path in inputs[2].items()}))
    assert guard.source_map_from_json(mapping) == inputs[2]
    mapping.write_text(json.dumps({"qqq": str(inputs[2]["qqq"])}))
    with pytest.raises(guard.GuardError, match="all required source roles"):
        guard.source_map_from_json(mapping)
    roles = {name: str(path) for name, path in inputs[2].items()}
    roles["qqq"] = "relative.parquet"
    mapping.write_text(json.dumps(roles))
    with pytest.raises(guard.GuardError, match="absolute filesystem path"):
        guard.source_map_from_json(mapping)
    study, _ = prepared(inputs)
    with pytest.raises(guard.GuardError, match="already exists"):
        guard.prepare_study(study, inputs[1], sources=inputs[2])


def test_input_modification_detected(inputs):
    study, _ = prepared(inputs)
    file = study / "developer/input/QQQ.csv"
    file.chmod(0o644)
    file.write_text("changed")
    with pytest.raises(guard.GuardError, match="Protected content changed"):
        guard.seal_study(study)


def test_symlink_detected(inputs):
    study, _ = prepared(inputs)
    (study / "developer/workspace/bypass").symlink_to(inputs[2]["qqq"])
    with pytest.raises(guard.GuardError, match="Symlinks"):
        guard._load(study)


def test_audit_chain_detects_modification(inputs):
    study, _ = prepared(inputs)
    log = study / "custodian/audit.jsonl"
    log.write_text(log.read_text().replace("prepare", "tampered"))
    with pytest.raises(guard.GuardError, match="hash chain"):
        guard._load(study)


def test_probe_required(inputs):
    study, _ = prepared(inputs)
    script = study / "developer/workspace/example.py"
    script.write_text('print("hello")\n')
    with pytest.raises(guard.GuardError, match="successful bound kernel probe"):
        guard.run_guarded(study, script)


@pytest.mark.skipif(
    sys.platform != "darwin" or not guard.SANDBOX.exists(), reason="requires macOS sandbox-exec"
)
def test_actual_kernel_probes_run_and_seal(inputs):
    study, manifest = prepared(inputs)
    prior = inputs[0] / "actual_prior_report.txt"
    prior.write_text("fixture never read in sandbox")
    probe = guard.probe_guard(study, prior)
    assert probe["passed"] and all(x is True for x in probe["checks"].values())
    script = study / "developer/workspace/research.py"
    script.write_text(
        'import os,pathlib,engine\nprint(engine.marker())\npathlib.Path(os.environ["BLIND_OUTPUT_DIR"],"new.txt").write_text("new artifact")\n'
    )
    result = guard.run_guarded(study, script)
    assert result["returncode"] == 0, result["stderr"]
    assert "fresh-accounting-engine" in result["stdout"]
    assert result["input_access_audit"]
    result2 = guard.run_guarded(study, script)
    assert result2["returncode"] != 0
    assert (study / "developer/output/new.txt").read_text() == "new artifact"
    assert guard.seal_study(study)["held_hashes"] == manifest["held_hashes"]
    with pytest.raises(guard.GuardError, match="Development sealed"):
        guard.run_guarded(study, script)
    with pytest.raises(FileExistsError):
        guard.seal_study(study)


def test_sandbox_absence_never_falls_back(inputs, monkeypatch):
    study, _ = prepared(inputs)
    prior = inputs[0] / "real_prior"
    prior.write_text("fixture")
    monkeypatch.setattr(guard, "SANDBOX", Path("/nonexistent/sandbox-exec"))
    with pytest.raises(guard.GuardError, match="refusing unsandboxed fallback"):
        guard.probe_guard(study, prior)


@pytest.mark.skipif(
    sys.platform != "darwin" or not guard.SANDBOX.exists(), reason="requires macOS sandbox-exec"
)
def test_existing_directory_rename_blocked(inputs):
    study, _ = prepared(inputs)
    prior = inputs[0] / "prior"
    prior.write_text("fixture")
    guard.probe_guard(study, prior)
    old = study / "developer/output/old_dir"
    old.mkdir()
    (old / "old.txt").write_text("immutable")
    script = study / "developer/workspace/rename.py"
    script.write_text(
        'import os,pathlib\nbase=pathlib.Path(os.environ["BLIND_OUTPUT_DIR"])\ntry:\n    (base/"old_dir").rename(base/"moved_dir")\nexcept PermissionError:\n    print("directory rename denied")\nelse:\n    raise SystemExit(91)\n'
    )
    result = guard.run_guarded(study, script)
    assert result["returncode"] == 0, result["stderr"]
    assert "directory rename denied" in result["stdout"]
    assert (old / "old.txt").read_text() == "immutable"


def test_concurrent_custody_operation_fails_closed(inputs):
    import fcntl

    study, _ = prepared(inputs)
    with open(study / "custodian/operation.lock", "a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(guard.GuardError, match="already active"):
            guard.seal_study(study)


def test_manifest_change_is_detected(inputs):
    study, _ = prepared(inputs)
    manifest = study / "custodian/manifest.json"
    manifest.chmod(0o644)
    row = json.loads(manifest.read_text())
    row["custodian_model"] = "changed"
    manifest.write_text(json.dumps(row))
    with pytest.raises(guard.GuardError, match="Manifest no longer matches"):
        guard._load(study)


def test_hardlink_detected(inputs):
    study, _ = prepared(inputs)
    os.link(inputs[2]["qqq"], study / "developer/workspace/alias.parquet")
    with pytest.raises(guard.GuardError, match="Hard links"):
        guard._load(study)


def test_unexpected_rate_column_rejected(inputs):
    tmp, engine, sources = inputs
    frame = pd.read_csv(sources["DFF_available.csv"])
    frame["future_quote"] = 999
    frame.to_csv(sources["DFF_available.csv"], index=False)
    with pytest.raises(guard.GuardError, match="unexpected columns forbidden"):
        guard.prepare_study(tmp / "failed", engine, sources=sources)


def test_special_filesystem_node_rejected(inputs):
    study, _ = prepared(inputs)
    os.mkfifo(study / "developer/workspace/unsealed_pipe")
    with pytest.raises(guard.GuardError, match="Special filesystem nodes"):
        guard._load(study)


@pytest.mark.skipif(
    sys.platform != "darwin" or not guard.SANDBOX.exists(), reason="requires macOS sandbox-exec"
)
def test_coordinator_release_cumulative_packets_and_one_attempt(inputs):
    tmp, engine, sources = inputs
    with open(engine / "engine.py", "a") as stream:
        stream.write("""
if __name__ == '__main__':
    import argparse,json,pathlib,os,socket
    parser=argparse.ArgumentParser(); parser.add_argument('--request',required=True)
    request=json.loads(pathlib.Path(parser.parse_args().request).read_text())
    pathlib.Path(request['output']).mkdir(parents=True,exist_ok=False)
    assert pathlib.Path(request['policy_path']).read_text()=='frozen synthetic policy'
    for target in request['blocked_read_paths']:
        try:
            fd=os.open(target,os.O_RDONLY); os.close(fd)
        except PermissionError:
            pass
        else:
            raise SystemExit(91)
    for target in request['blocked_write_paths']:
        try:
            fd=os.open(target,os.O_WRONLY); os.close(fd)
        except PermissionError:
            pass
        else:
            raise SystemExit(92)
    rows=pathlib.Path(request['input_dir'],'QQQ.csv').read_text().splitlines()
    pathlib.Path(request['output'],'mechanical_count.json').write_text(json.dumps({'session_count':len(rows)-1}))
""")
    study, manifest = prepared(inputs)
    prior = tmp / "prior"
    prior.write_text("fixture")
    guard.probe_guard(study, prior)
    guard.seal_study(study)
    proposal_dir = study / "custodian/proposal"
    proposal_dir.mkdir()
    proposal = proposal_dir / "proposal.json"
    proposal.write_text('{"synthetic":true}')
    policy = proposal_dir / "policy.py"
    policy.write_text("frozen synthetic policy")
    validation_lock = None
    for stage, count in [("validation", 5), ("test", 7)]:
        request_path = proposal_dir / f"{stage}_request.json"
        request = {
            "input_dir": str(study / f"released/{stage}/input"),
            "output": str(study / f"operator_output/{stage}/results"),
            "start": guard.STAGES[stage][2],
            "end": guard.STAGES[stage][1],
            "policy_path": str(policy),
            "blocked_read_paths": [
                str(study / "held/test/QQQ.csv"),
                str(study / "held/validation/QQQ.csv"),
                str(study / "custodian/audit.jsonl"),
                str(prior),
            ],
            "blocked_write_paths": [
                str(study / "developer/input/QQQ.csv"),
                str(study / "custodian/audit.jsonl"),
            ],
        }
        request_path.write_text(json.dumps(request))
        lock = {
            "stage": stage,
            "development_seal_sha256": guard.sha256(study / "custodian/DEVELOPMENT_SEALED.json"),
            "proposal_path": str(proposal),
            "proposal_sha256": guard.sha256(proposal),
            "request_path": str(request_path),
            "files": {str(p): guard.sha256(p) for p in (proposal, policy, request_path)},
        }
        if stage == "test":
            receipt = study / "custodian/seals/selection.json"
            receipt.write_text('{"synthetic_selected":true}')
            lock.update(
                validation_lock_sha256=guard.sha256(validation_lock),
                selection_receipt_path=str(receipt),
                selection_receipt_sha256=guard.sha256(receipt),
            )
            lock["files"][str(receipt)] = guard.sha256(receipt)
        lock_path = study / f"custodian/seals/{stage}_lock.json"
        lock_path.write_text(json.dumps(lock))
        result = guard.gate_release(study, stage, lock_path)
        assert result["returncode"] == 0, result["stderr"]
        assert (
            json.loads(
                (study / f"operator_output/{stage}/results/mechanical_count.json").read_text()
            )["session_count"]
            == count
        )
        assert (study / f"custodian/STAGE_{stage.upper()}_COMPLETED.json").is_file()
        with pytest.raises((guard.GuardError, FileExistsError)):
            guard.gate_release(study, stage, lock_path)
        if stage == "validation":
            validation_lock = lock_path


@pytest.mark.skipif(
    sys.platform != "darwin" or not guard.SANDBOX.exists(), reason="requires macOS sandbox-exec"
)
def test_actual_seeded_engine_held_entrypoint(inputs):
    """Exercise the actual newly supplied engine, using fictional prices/rates only."""
    tmp, engine, sources = inputs
    # Self-contained public-code fixture: no private prepared study or data.
    source_dir = guard.REPO / "src/trend_following"
    mechanical_sources = {
        "research_blind_engine_v4.py": "engine.py",
        "research_ledger.py": "ledger_kernel.py",
        "research_rates.py": "rate_kernel.py",
    }
    for source_name, fixture_name in mechanical_sources.items():
        source_path = source_dir / source_name
        assert source_path.is_file(), f"Missing published mechanical source: {source_name}"
        (engine / fixture_name).write_bytes(source_path.read_bytes())
    observations = pd.date_range("1999-01-01", "2026-06-01", freq="D")
    offset = CustomBusinessDay(2, calendar=USFederalHolidayCalendar())
    timestamps = (
        pd.DatetimeIndex([d + offset for d in observations]).tz_localize("America/New_York")
        + pd.Timedelta(hours=18)
    ).tz_convert("UTC")
    for series in ("DFF", "DTB3"):
        pd.DataFrame({"observation_date": observations.strftime("%Y-%m-%d"), series: 4.0}).to_csv(
            sources[f"{series}_raw.csv"], index=False
        )
        annual = 0.04 if series == "DFF" else 0.04 * (365 / 360) / (1 - 0.04 * (91 / 360))
        pd.DataFrame(
            {
                "observation_date": observations.strftime("%Y-%m-%d"),
                "quoted_percent": 4.0,
                "available_at": timestamps,
                "annual_rate": annual,
                "series_id": series,
                "source_url": "fictional-fixture",
                "vintage_type": "synthetic",
                "availability_assumption": "synthetic",
            }
        ).to_csv(sources[f"{series}_available.csv"], index=False)
    study, _ = prepared(inputs)
    prior = tmp / "prior"
    prior.write_text("synthetic fixture")
    guard.probe_guard(study, prior)
    guard.seal_study(study)
    proposal_dir = study / "custodian/proposal"
    proposal_dir.mkdir()
    proposal = proposal_dir / "proposal.json"
    proposal.write_text('{"fixture":"mechanical zero-allocation callback"}')
    policy = proposal_dir / "policy.py"
    policy.write_text("def target(history, parameters):\n    return 0.0\n")
    validation_lock = None
    for stage in ("validation", "test"):
        request_path = proposal_dir / f"{stage}_request.json"
        request = {
            "input_dir": str(study / f"released/{stage}/input"),
            "output": str(study / f"operator_output/{stage}/results"),
            "start": guard.STAGES[stage][2],
            "end": guard.STAGES[stage][1],
            "policy_path": str(policy),
            "candidates": [
                {"candidate_id": "mechanical_fixture", "parameters": {}, "history_sessions": 1}
            ],
            "layers": ["gross"],
            "benchmarks": False,
        }
        request_path.write_text(json.dumps(request))
        lock = {
            "stage": stage,
            "development_seal_sha256": guard.sha256(study / "custodian/DEVELOPMENT_SEALED.json"),
            "proposal_path": str(proposal),
            "proposal_sha256": guard.sha256(proposal),
            "request_path": str(request_path),
            "files": {str(p): guard.sha256(p) for p in (proposal, policy, request_path)},
        }
        if stage == "test":
            receipt = study / "custodian/seals/selection.json"
            receipt.write_text('{"fixture_only":true}')
            lock.update(
                validation_lock_sha256=guard.sha256(validation_lock),
                selection_receipt_path=str(receipt),
                selection_receipt_sha256=guard.sha256(receipt),
            )
            lock["files"][str(receipt)] = guard.sha256(receipt)
        lock_path = study / f"custodian/seals/{stage}_lock.json"
        lock_path.write_text(json.dumps(lock))
        result = guard.gate_release(study, stage, lock_path, timeout_seconds=120)
        assert result["returncode"] == 0, result["stderr"]
        assert json.loads(result["stdout"])["status"] == "completed"
        assert (study / f"operator_output/{stage}/results/all_metrics.csv").is_file()
        assert (study / f"custodian/STAGE_{stage.upper()}_COMPLETED.json").is_file()
        if stage == "validation":
            validation_lock = lock_path
