"""Bounded 2018 price preparation, never an evaluation or OOS permission.

The older packet's directory name is provenance, not the new research role.
Only its metadata is read before selection is sealed. CSVs are streamed in
chronological order; the session marker of the first 2019 row stops the reader
BEFORE any other field in that row is converted. Included numbers are checked
against the original independently generated per-session unit hashes. Neither
old results nor full raw-source/CSV arrays are loaded or re-hashed.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from trend_following import research_alpha_reconciliation_v5 as data
from trend_following import research_hourly_data_v5 as hourly
from trend_following import research_intersection_validation as intersection
from trend_following import research_quality_protocol_v5 as protocol

Error = protocol.ProtocolError
END_SESSION = "2018-12-31"
START_SESSION = "1999-11-01"
NUMERIC_FILES = ("daily.csv", "bars.csv", "calendar.csv", "coverage.csv", "blackouts.csv")
PACKET_FILES = (*NUMERIC_FILES, "unit_proof.json")
BINDINGS = {
    "daily.csv": ("daily_session_hashes", data.DAILY_BINDING_COLUMNS),
    "bars.csv": ("bar_session_hashes", data.BAR_BINDING_COLUMNS),
    "calendar.csv": ("calendar_session_hashes", data.CALENDAR_BINDING_COLUMNS),
    "coverage.csv": ("coverage_session_hashes", data.COVERAGE_BINDING_COLUMNS),
}
EXTRA_COLUMNS = {"daily.csv": {"date"}, "bars.csv": {"volume_reported"}}
TIMES = {
    "daily_price_available_at",
    "bar_start",
    "bar_end",
    "market_open",
    "market_close",
    "start",
    "end",
}
BOOLEANS = {"is_full_hour", "is_early_close", "signal_eligible", "fill_eligible"}
TEXT = {"session", "date", "status", "kind", "reason", "source_url"}


def _safe_directory(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise Error("Symlink prefix directory or ancestor prohibited")
    return path.resolve()


def _json_file(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise Error("Required nonsymlink metadata file absent")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise Error("Prefix metadata must be a JSON mapping")
    return value


def preseal_metadata(source_study: str | Path, config_path: str | Path) -> dict:
    """Verify custody metadata only, without opening price CSVs or old outcomes."""
    source = _safe_directory(source_study)
    config_path = Path(config_path)
    config = protocol.load_config(config_path)
    registration = protocol._verify_registration(source, config, config_path)
    prepared = protocol._read_seal(source / "prepared/freeze.json")
    if (
        prepared.get("registration_sha256") != protocol.canonical_hash(registration)
        or prepared.get("quality_policy_sha256") != registration["quality_policy_sha256"]
        or prepared.get("quote_roles_sha256") != registration["quote_roles_sha256"]
        or prepared.get("gap_authorization") != registration["gap_authorization"]
    ):
        raise Error("Original prepared provenance changed")
    root = _safe_directory(source / "prepared/historical-oos")
    manifest = _json_file(root / "manifest.json")
    record = prepared.get("prefixes", {}).get("historical-oos", {})
    if (
        protocol.canonical_hash(manifest) != record.get("manifest_sha256")
        or manifest.get("files") != record.get("files")
        or record.get("end") != "2026-05-28"
        or manifest.get("historical_ready") is not True
        or manifest.get("quality_profile") != prepared.get("quality_profile")
        or not set(PACKET_FILES).issubset(record.get("files", {}))
    ):
        raise Error("Original price-only provenance/readiness binding changed")
    protocol._require_alpha_price_source(manifest)
    protocol._require_half_hour_clock(manifest)
    if manifest.get("implementation_hashes") != data._implementation_hashes():
        raise Error("Original price implementation provenance changed")
    proof_path = root / "unit_proof.json"
    if protocol.sha256_file(proof_path) != manifest["unit_proof_sha256"]:
        raise Error("Original unit-proof metadata custody changed")
    proof = _json_file(proof_path)
    data._validate_unit_proof(proof)
    return {
        "original_registration_sha256": protocol.canonical_hash(registration),
        "original_prepared_metadata_sha256": protocol.canonical_hash(prepared),
        "original_manifest_sha256": protocol.canonical_hash(manifest),
        "original_unit_proof_sha256": manifest["unit_proof_sha256"],
        "manifest": manifest,
    }


def _session(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise Error("ISO session marker required before numeric parsing")
    try:
        if pd.Timestamp(value).strftime("%Y-%m-%d") != value:
            raise ValueError(value)
    except ValueError as error:
        raise Error("Invalid session date marker") from error
    return value


def _convert(value: str, column: str) -> Any:
    if column in TEXT:
        return value
    if column in TIMES:
        return hourly.utc(value, column)
    if column in BOOLEANS:
        if value not in ("True", "False"):
            raise Error(f"Explicit source boolean required: {column}")
        return value == "True"
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Error(f"Invalid bounded numeric field: {column}") from error
    if not math.isfinite(result):
        raise Error(f"Nonfinite bounded numeric field: {column}")
    return result


def _stream_frame(path: Path, name: str, *, end_session: str) -> pd.DataFrame:
    """Read only the permitted chronological prefix; never parse later numbers."""
    if path.is_symlink() or not path.is_file():
        raise Error("Missing or unsafe original price-prefix CSV")
    columns = data.BLACKOUT_BINDING_COLUMNS if name == "blackouts.csv" else BINDINGS[name][1]
    allowed = set(columns) | EXTRA_COLUMNS.get(name, set())
    rows = []
    previous = None
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if (
            reader.fieldnames is None
            or len(set(reader.fieldnames)) != len(reader.fieldnames)
            or set(reader.fieldnames) != allowed
        ):
            raise Error(f"Unexpected source-prefix columns: {name}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise Error(f"Malformed source-prefix row: {name}")
            if name == "blackouts.csv":
                # The boundary date is metadata. Do not parse the future reason,
                # action or any other field once this first future date is found.
                marker = _session(row["start"][:10])
                if marker > end_session:
                    break
                key = hourly.utc(row["start"], "blackout start")
                end = hourly.utc(row["end"], "blackout end")
                if end.tz_convert("America/New_York").strftime("%Y-%m-%d") > end_session:
                    raise Error("Blackout crosses the authorized validation endpoint")
            else:
                marker = _session(row["session"])
                if marker > end_session:
                    break
                key = (
                    hourly.utc(row["bar_start"]) if name in ("bars.csv", "coverage.csv") else marker
                )
            if marker < START_SESSION or (previous is not None and key <= previous):
                raise Error(f"Nonchronological or pre-initialization prefix: {name}")
            previous = key
            converted = {column: _convert(value, column) for column, value in row.items()}
            if name == "daily.csv" and converted["date"] != marker:
                raise Error("Daily date and bound session marker disagree")
            if name == "bars.csv" and converted["volume_reported"] != converted["volume"]:
                raise Error("Reported volume differs from the bound unused vendor volume")
            rows.append(converted)
    return pd.DataFrame(rows, columns=reader.fieldnames)


def _verify_rows(frames: dict[str, pd.DataFrame], proof: dict, *, end_session: str) -> dict:
    """Require every original <= endpoint session, not merely matching present rows."""
    bindings = proof["data_bindings"]
    bounded = {}
    for name, (key, columns) in BINDINGS.items():
        actual = data._session_hashes(frames[name], columns)
        expected = {
            session: digest
            for session, digest in bindings[key].items()
            if START_SESSION <= _session(session) <= end_session
        }
        if not actual or actual != expected:
            raise Error(f"Missing, changed or extra bounded source sessions: {key}")
        bounded[key] = actual
    blackouts = frames["blackouts.csv"]
    hashes = [data._row_hash(row, data.BLACKOUT_BINDING_COLUMNS) for _, row in blackouts.iterrows()]
    if len(set(hashes)) != len(hashes) or any(
        value not in bindings["blackout_row_hashes"] for value in hashes
    ):
        raise Error("Bounded blackout rows differ from original source proof")
    bounded["blackout_row_hashes"] = hashes
    return bounded


def _validate_frames(frames: dict[str, pd.DataFrame], manifest: dict, end_session: str) -> dict:
    daily, bars, calendar, coverage, blackouts = (frames[name] for name in NUMERIC_FILES)
    calendar.attrs["calendar_provenance"] = manifest.get("calendar_provenance", {})
    calendar = hourly.validate_calendar(calendar)
    if (
        daily.empty
        or daily.session.tolist() != calendar.session.tolist()
        or daily.session.iloc[0] != START_SESSION
        or daily.session.iloc[-1] != end_session
        or bars.empty
        or coverage.empty
        or bars.session.gt(end_session).any()
        or coverage.session.gt(end_session).any()
        or not bars.duration_minutes.eq(30).all()
        or bars.is_full_hour.any()
    ):
        raise Error("Bounded feature/calendar/bar periods or half-hour identity invalid")
    expected_available = pd.DatetimeIndex(
        [
            pd.Timestamp(session + " 17:00", tz="America/New_York").tz_convert("UTC")
            for session in daily.session
        ]
    )
    if not pd.DatetimeIndex(daily.daily_price_available_at).equals(expected_available):
        raise Error("Bounded provider daily availability differs from frozen 17:00 clock")
    for name in ("daily.csv", "bars.csv"):
        frame = frames[name]
        if (
            frame[["open", "high", "low", "close"]].le(0).any().any()
            or frame.high.lt(frame[["open", "close", "low"]].max(axis=1)).any()
            or frame.low.gt(frame[["open", "close", "high"]].min(axis=1)).any()
        ):
            raise Error("Invalid positive bounded OHLC structure")
    if daily.split_coefficient.le(0).any() or daily.dividend_amount.lt(0).any():
        raise Error("Invalid bounded corporate-action units")
    grid = hourly.canonical_half_hour_grid(
        calendar.loc[calendar.session.ge(coverage.session.iloc[0])].reset_index(drop=True)
    )
    if (
        not coverage[["session", "bar_start", "bar_end"]]
        .reset_index(drop=True)
        .equals(grid[["session", "bar_start", "bar_end"]])
    ):
        raise Error("Bounded coverage omits scheduled slots or changes early-close grid")
    calculated, expected_blackouts = data.classify_observation_coverage(grid, bars)
    if not calculated[list(data.COVERAGE_BINDING_COLUMNS)].equals(
        coverage[list(data.COVERAGE_BINDING_COLUMNS)].reset_index(drop=True)
    ):
        raise Error("Bounded coverage eligibility/source-presence mismatch")
    actual_blackout_hashes = [
        data._row_hash(row, data.BLACKOUT_BINDING_COLUMNS) for _, row in blackouts.iterrows()
    ]
    expected_blackout_hashes = [
        data._row_hash(row, data.BLACKOUT_BINDING_COLUMNS)
        for _, row in expected_blackouts.iterrows()
    ]
    if actual_blackout_hashes != expected_blackout_hashes:
        raise Error("Bounded blackout rows do not match missing slots or the known halt")
    matched = bars.merge(
        calculated[["bar_start", "status", "signal_eligible", "fill_eligible"]],
        on="bar_start",
        suffixes=("", "_expected"),
        validate="one_to_one",
    )
    if len(matched) != len(bars) or any(
        not matched[field].equals(matched[field + "_expected"])
        for field in ("status", "signal_eligible", "fill_eligible")
    ):
        raise Error("Bounded bars are off-grid or their execution eligibility changed")
    scored = calendar.session.ge("2012-01-01")
    return {
        "daily_sessions": len(daily),
        "observed_bars": len(bars),
        "scheduled_bars": len(coverage),
        "validation_sessions": int(scored.sum()),
        "first_validation_session": calendar.loc[scored, "session"].iloc[0],
        "last_validation_session": calendar.loc[scored, "session"].iloc[-1],
        "unknown_missing_slots": int(coverage.status.eq("unknown_missing").sum()),
        "validation_unknown_missing_slots": int(
            coverage.loc[coverage.session.ge("2012-01-01"), "status"].eq("unknown_missing").sum()
        ),
        "validation_known_halt_slots": int(
            coverage.loc[coverage.session.ge("2012-01-01"), "status"].eq("known_halt_overlap").sum()
        ),
        "prices_synthesized": False,
        "OOS_numeric_rows_parsed": 0,
    }


def _selection_guard(path: Path, expected_sha256: str, source: Path) -> dict:
    selection = protocol._read_seal(path)
    if (
        protocol.canonical_hash(selection) != expected_sha256
        or selection.get("schema") != "trend_following_resplit_intersection_validation_v1"
        or selection.get("stage") != "validation"
        or selection.get("period") != {"start": "2012-01-01", "end": END_SESSION}
        or selection.get("historical_oos_period") != {"start": "2019-01-01", "end": "2026-05-28"}
        or selection.get("historical_oos_authorized") is not False
        or selection.get("validation_only") is not True
        or selection.get("configuration_count_per_wait") != 159
        or selection.get("configuration_count") != 477
        or selection.get("wait_hours") != [2.0, 3.0, 4.0]
        or selection.get("source_study") != str(source)
        or len(selection.get("pairs", [])) != 159
        or len(selection.get("candidates", [])) != 477
        or protocol.canonical_hash(selection["pairs"]) != selection.get("pairs_sha256")
        or protocol.canonical_hash(selection["candidates"]) != selection.get("candidate_set_sha256")
    ):
        raise Error(
            "Sealed development intersection and validation-only split required BEFORE prices"
        )
    sealed_at = hourly.utc(selection.get("candidate_set_sealed_at"), "selection sealed_at")
    if sealed_at > pd.Timestamp.now(tz="UTC"):
        raise Error("Candidate seal is future-dated")
    canonical = [
        {"candidate_id": candidate.candidate_id, **candidate.to_dict()}
        for hours in (2.0, 3.0, 4.0)
        for candidate in intersection.candidates_from_pairs(selection["pairs"], hours)
    ]
    if selection["candidates"] != canonical:
        raise Error("Sealed candidates differ from canonical pairs at equal 2/3/4-hour waits")
    return selection


def prepare_validation_prefix(
    source_study: str | Path,
    output: str | Path,
    *,
    selection_seal: str | Path,
    expected_selection_sha256: str,
    end_session: str = END_SESSION,
) -> dict:
    """Prepare a new immutable physical price prefix only after candidate sealing."""
    if end_session != END_SESSION:
        raise Error("This validation-only reader is bounded exactly at December 31, 2018")
    source, output = _safe_directory(source_study), _safe_directory(output)
    if output == source or source in output.parents or output in source.parents or output.exists():
        raise Error("Prepared prefix must be new and outside the protected original study")
    selection = _selection_guard(Path(selection_seal), expected_selection_sha256, source)
    metadata = preseal_metadata(source, selection["config_path"])
    if selection.get("original_manifest_sha256") != metadata["original_manifest_sha256"]:
        raise Error("Source manifest differs from the pre-price candidate freeze")
    original = metadata["manifest"]
    root = source / "prepared/historical-oos"
    proof = _json_file(root / "unit_proof.json")
    frames = {
        name: _stream_frame(root / name, name, end_session=end_session) for name in NUMERIC_FILES
    }
    bounded_bindings = _verify_rows(frames, proof, end_session=end_session)
    calendar_provenance = {**original["calendar_provenance"], "requested_end_session": end_session}
    source_metadata = {**original["source_metadata"], "coverage_end_session": end_session}
    prefix_manifest = {
        "schema": "trend_following_validation_2018_prefix_v1",
        "prepared_at": protocol._utc_now(),
        "selection_sha256": expected_selection_sha256,
        "candidate_set_sealed_at": selection["candidate_set_sealed_at"],
        "feature_history_start": START_SESSION,
        "end_session": end_session,
        "stage": "validation",
        "historical_oos_authorized": False,
        "calendar_provenance": calendar_provenance,
        "source_metadata": source_metadata,
        "daily_source_metadata": original["daily_source_metadata"],
        "original_manifest_sha256": metadata["original_manifest_sha256"],
        "original_unit_proof_sha256": metadata["original_unit_proof_sha256"],
        "original_csv_hashes_recorded_not_rehashed": {
            name: original["files"][name]["sha256"] for name in NUMERIC_FILES
        },
        "implementation_hashes": original["implementation_hashes"],
        "quality_profile": original["quality_profile"],
        "gap_authorization": original["gap_authorization"],
        "metadata_custody_only_for_unopened_later_sources": True,
        "retrospective_history_previously_examined": True,
    }
    prefix_manifest["audit"] = _validate_frames(frames, prefix_manifest, end_session)
    # Global source-proof summary counts remain custody metadata, NOT newly
    # computed prefix-level anchor statistics. Only bounded bindings reach use.
    prefix_proof = {
        **proof,
        "data_bindings": bounded_bindings,
        "cutoff_session": end_session,
        "bounded_prefix_attestation": {
            "original_unit_proof_sha256": metadata["original_unit_proof_sha256"],
            "global_summary_statistics_are_original_custody_metadata": True,
            "numeric_rows_verified_only_through": end_session,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    protocol._write_json(output / "unit_proof.json", prefix_proof)
    prefix_manifest["files"] = {
        name: {
            "sha256": protocol.sha256_file(output / name),
            "bytes": (output / name).stat().st_size,
        }
        for name in PACKET_FILES
    }
    protocol._write_json(output / "manifest.json", prefix_manifest)
    return prefix_manifest


def verify_validation_prefix(root: str | Path, *, expected_manifest_sha256: str) -> dict:
    """Verify only the NEW bounded file bytes, without numerical array parsing."""
    root = _safe_directory(root)
    manifest = _json_file(root / "manifest.json")
    if (
        protocol.canonical_hash(manifest) != expected_manifest_sha256
        or manifest.get("schema") != "trend_following_validation_2018_prefix_v1"
        or manifest.get("end_session") != END_SESSION
        or manifest.get("stage") != "validation"
        or manifest.get("historical_oos_authorized") is not False
        or set(manifest.get("files", {})) != set(PACKET_FILES)
        or hourly.utc(manifest["prepared_at"]) < hourly.utc(manifest["candidate_set_sealed_at"])
    ):
        raise Error("New bounded validation manifest changed or predates candidate seal")
    for name, record in manifest["files"].items():
        path = root / name
        if protocol.sha256_file(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise Error(f"New bounded prefix bytes changed: {name}")
    return manifest


def read_validation_prefix(root: str | Path, *, expected_manifest_sha256: str) -> tuple:
    """Return verified bounded frames; never follow original source-cache paths."""
    root = _safe_directory(root)
    manifest = verify_validation_prefix(root, expected_manifest_sha256=expected_manifest_sha256)
    frames = {
        name: _stream_frame(root / name, name, end_session=END_SESSION) for name in NUMERIC_FILES
    }
    proof = _json_file(root / "unit_proof.json")
    data._validate_unit_proof(proof)
    _verify_rows(frames, proof, end_session=END_SESSION)
    if _validate_frames(frames, manifest, END_SESSION) != manifest["audit"]:
        raise Error("New bounded numerical audit changed")
    daily, bars, calendar, coverage, blackouts = (frames[name] for name in NUMERIC_FILES)
    daily.attrs["source_metadata"] = manifest["daily_source_metadata"]
    bars.attrs["source_metadata"] = manifest["source_metadata"]
    bars.attrs["unit_proof"] = proof
    calendar.attrs["calendar_provenance"] = manifest["calendar_provenance"]
    return daily, bars, calendar, coverage, blackouts, manifest
