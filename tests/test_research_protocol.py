from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from trend_following import research_protocol
from trend_following.research_protocol import (
    freeze_protocol,
    load_protocol,
    next_complete_session,
    verify_frozen_inputs,
)


def fixture_project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "data").mkdir()
    (root / "src" / "model.py").write_text("value = 1\n")
    (root / ".env").write_text("TEST_SECRET=never_capture_me\n")
    (root / "src" / "credentials.json").write_text('{"fixture":"never_capture_me"}')
    frame = pd.DataFrame({"price": [1.0, 1.1]}, index=pd.date_range("2024-01-01", periods=2))
    frame.to_parquet(root / "data" / "qqq.parquet")
    protocol = {
        "protocol_id": "fixture", "schema_version": 1,
        "candidates": {"cash": {}, "b5": {}},
        "selection": {"minimum_training_years": 3},
        "evaluation": {"outer_folds": [{"train_end": "2008-12-31", "test_start": "2009-01-01", "test_end": "2011-12-31"}]},
        "costs": {},
        "provenance": {"code_roots": ["src", "configs"], "data_globs": ["data/*.parquet"], "research_inventory_roots": ["src"]},
    }
    config = root / "configs" / "protocol.yaml"
    config.write_text(yaml.safe_dump(protocol))
    return root, config


def test_freeze_is_immutable_and_excludes_secret_files(tmp_path: Path) -> None:
    root, config = fixture_project(tmp_path)
    output = tmp_path / "frozen"
    manifest = freeze_protocol(root, config, output)
    assert manifest["stage"] == "protocol_only"
    assert manifest["genuine_holdout_start_utc"] is None
    assert not manifest["prospective_eligible"]
    text = json.dumps(manifest)
    assert "never_capture_me" not in text
    assert "credentials.json" not in text
    assert ".env" not in text
    assert manifest["known_local_bar_endpoints"][0]["index_end"].startswith("2024-01-02")
    verify_frozen_inputs(root, output / "manifest.json")
    with pytest.raises(FileExistsError):
        freeze_protocol(root, config, output)


def test_verifier_detects_source_and_data_changes(tmp_path: Path) -> None:
    root, config = fixture_project(tmp_path)
    output = tmp_path / "frozen"
    freeze_protocol(root, config, output)
    (root / "src" / "model.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="Frozen input changed"):
        verify_frozen_inputs(root, output / "manifest.json")


def test_no_backdated_prospective_claim_without_final_code_and_calendar(tmp_path: Path) -> None:
    root, config = fixture_project(tmp_path)
    with pytest.raises(ValueError, match="readiness attestation"):
        freeze_protocol(root, config, tmp_path / "bad", stage="final_code")
    calendar = tmp_path / "sessions.csv"
    pd.DataFrame({"market_open": ["2026-10-02T13:30Z", "2026-10-05T13:30Z"]}).to_csv(calendar, index=False)
    manifest = freeze_protocol(
        root, config, tmp_path / "final", stage="final_code", final_code_ready=True,
        session_calendar_path=calendar, frozen_at=pd.Timestamp("2026-10-02T14:00Z"),
    )
    assert manifest["genuine_holdout_start_utc"].startswith("2026-10-05T13:30")
    assert manifest["prospective_eligible"]


def test_session_calendar_requires_future_and_aware_times() -> None:
    with pytest.raises(ValueError, match="No future exchange session"):
        next_complete_session(pd.Timestamp("2026-10-06T13:00Z"), pd.to_datetime(["2026-10-05T13:30Z"]))
    with pytest.raises(ValueError, match="timezone-aware"):
        next_complete_session(pd.Timestamp("2026-10-02"), pd.to_datetime(["2026-10-05T13:30Z"]))


def test_final_freeze_does_not_guess_naive_calendar_timezone(tmp_path: Path) -> None:
    root, config = fixture_project(tmp_path)
    calendar = tmp_path / "naive.csv"
    pd.DataFrame({"market_open": ["2026-10-05 09:30"]}).to_csv(calendar, index=False)
    with pytest.raises(ValueError, match="timezone-aware"):
        freeze_protocol(
            root, config, tmp_path / "bad-final", stage="final_code", final_code_ready=True,
            session_calendar_path=calendar, frozen_at=pd.Timestamp("2026-10-02T14:00Z"),
        )


def test_actual_protocol_has_frozen_small_pool_and_six_folds() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "qqq_research_v1.yaml"
    protocol = load_protocol(path)
    assert set(protocol["candidates"]) == {"b5", "anchor", "qqq200ma1x", "qqq200ma2x", "cash"}
    assert len(protocol["evaluation"]["outer_folds"]) == 6
    assert protocol["costs"]["financing_series"] == "DFF"
    assert protocol["costs"]["cash_series"] == "DTB3"
    assert protocol["costs"]["financing_spread_bps"] == 50
    assert protocol["evaluation"]["genuine_holdout_start"] is None


def final_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    root, config = fixture_project(tmp_path)
    calendar = tmp_path / "sessions.csv"
    pd.DataFrame({"market_open": ["2026-10-02T13:30Z", "2026-10-05T13:30Z"]}).to_csv(calendar, index=False)
    output = tmp_path / "final"
    manifest = freeze_protocol(
        root, config, output, stage="final_code", final_code_ready=True,
        session_calendar_path=calendar, frozen_at=pd.Timestamp("2026-10-02T14:00Z"),
    )
    return root, output, manifest


def test_final_freeze_market_cache_can_update_but_archive_cannot(tmp_path: Path) -> None:
    root, output, manifest = final_fixture(tmp_path)
    (root / "data" / "qqq.parquet").write_bytes(b"new incoming market cache")
    verify_frozen_inputs(root, output / "manifest.json")
    snapshot = output / manifest["data_files"][0]["snapshot_path"]
    snapshot.write_bytes(b"tampering with historical snapshot")
    with pytest.raises(ValueError, match="Frozen snapshot changed"):
        verify_frozen_inputs(root, output / "manifest.json")


def test_final_freeze_still_rejects_source_code_changes(tmp_path: Path) -> None:
    root, output, _ = final_fixture(tmp_path)
    (root / "src" / "model.py").write_text("value = 99\n")
    with pytest.raises(ValueError, match="Frozen input changed"):
        verify_frozen_inputs(root, output / "manifest.json")


def test_final_freeze_requires_archived_data_record(tmp_path: Path) -> None:
    root, output, manifest = final_fixture(tmp_path)
    del manifest["data_files"][0]["snapshot_path"]
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Frozen snapshot required"):
        verify_frozen_inputs(root, output / "manifest.json")


def test_protocol_stage_rejects_updated_original_data_even_with_intact_archive(tmp_path: Path) -> None:
    root, config = fixture_project(tmp_path)
    output = tmp_path / "protocol-only"
    freeze_protocol(root, config, output)
    (root / "data" / "qqq.parquet").write_bytes(b"updated live cache")
    with pytest.raises(ValueError, match="Frozen input changed"):
        verify_frozen_inputs(root, output / "manifest.json")


def test_column_timestamp_parquet_has_real_rows_and_endpoints(tmp_path: Path) -> None:
    root, config = fixture_project(tmp_path)
    pd.DataFrame({"date": pd.to_datetime(["2026-10-01 10:00", "2026-10-02 10:00"]), "close": [1.0, 1.1]}).to_parquet(root / "data" / "qqq.parquet", index=False)
    manifest = freeze_protocol(root, config, tmp_path / "column-timestamp")
    record = manifest["data_files"][0]
    assert record["rows"] == 2
    assert record["index_end"] == "2026-10-02 10:00:00"
    assert record["timestamp_column"] == "date"
    assert len(manifest["known_local_bar_endpoints"]) == 1


def test_snapshot_copy_must_match_initial_hash(tmp_path: Path, monkeypatch) -> None:
    root, config = fixture_project(tmp_path)
    real_copy = research_protocol.shutil.copyfile

    def corrupt_copy(source, destination):
        result = real_copy(source, destination)
        Path(destination).write_bytes(b"inconsistent copy")
        return result

    monkeypatch.setattr(research_protocol.shutil, "copyfile", corrupt_copy)
    with pytest.raises(ValueError, match="changed during snapshot"):
        freeze_protocol(root, config, tmp_path / "bad-copy")
    assert not (tmp_path / "bad-copy" / "manifest.json").exists()


def test_production_final_freeze_timestamp_is_after_snapshot_completion(tmp_path: Path, monkeypatch) -> None:
    root, config = fixture_project(tmp_path)
    calendar = tmp_path / "sessions.csv"
    pd.DataFrame({"market_open": ["2026-10-02T13:30Z", "2026-10-05T13:30Z"]}).to_csv(calendar, index=False)
    times = iter([pd.Timestamp("2026-10-02T13:29:59Z"), pd.Timestamp("2026-10-02T13:30:01Z")])
    monkeypatch.setattr(research_protocol, "_utc_now", lambda: next(times))
    manifest = freeze_protocol(root, config, tmp_path / "clock-boundary", stage="final_code", final_code_ready=True, session_calendar_path=calendar)
    assert manifest["frozen_at_utc"] == "2026-10-02T13:30:01+00:00"
    assert manifest["genuine_holdout_start_utc"] == "2026-10-05T13:30:00+00:00"
