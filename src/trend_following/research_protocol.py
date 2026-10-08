"""Immutable research-protocol snapshots without secret/environment capture.

A protocol-only snapshot precedes evaluation, but does not claim the code is
ready for prospective testing. A separate final-code snapshot must follow code
repair and validation; genuine holdout timing is computed from supplied exchange
session opens, never inferred by backdating an already observed history.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

SAFE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".csv", ".parquet", ".json"}
SECRET_PARTS = {".env", "secrets", "credentials", ".ssh", ".aws", ".git"}
DEPENDENCIES = ["pandas", "numpy", "pyarrow", "PyYAML", "requests", "yfinance", "pytest", "ruff", "tabulate", "tzdata"]


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return (
        not path.is_symlink()
        and path.suffix in SAFE_SUFFIXES
        and not any(part.lower() in SECRET_PARTS or part.lower().startswith(".env") for part in relative.parts)
        and not any(token in path.name.lower() for token in ("secret", "credential", "private_key"))
    )


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text())
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("Expected protocol mapping schema_version 1")
    required = {"protocol_id", "candidates", "selection", "evaluation", "costs", "provenance"}
    if not required.issubset(protocol):
        raise ValueError(f"Missing protocol keys: {sorted(required - protocol.keys())}")
    if "cash" not in protocol["candidates"] or "b5" not in protocol["candidates"]:
        raise ValueError("Protocol must retain cash fallback and fixed B5 track")
    previous_end = None
    for fold in protocol["evaluation"]["outer_folds"]:
        train_end = pd.Timestamp(fold["train_end"])
        start, end = pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])
        if train_end >= start or start > end or (previous_end is not None and start != previous_end + pd.Timedelta(days=1)):
            raise ValueError("Invalid, overlapping, or noncontiguous outer folds")
        previous_end = end
    if protocol["selection"]["minimum_training_years"] < 3:
        raise ValueError("At least three training years are required")
    return protocol


def next_complete_session(
    frozen_at: pd.Timestamp, session_opens: pd.DatetimeIndex
) -> pd.Timestamp:
    """First supplied exchange session that has not begun at the freeze time."""
    frozen_at = pd.Timestamp(frozen_at)
    opens = pd.DatetimeIndex(session_opens)
    if frozen_at.tzinfo is None or opens.tz is None:
        raise ValueError("Freeze and exchange session opens must be timezone-aware")
    if opens.hasnans or opens.has_duplicates or not opens.is_monotonic_increasing:
        raise ValueError("Exchange session opens must be sorted, unique, nonmissing")
    eligible = opens[opens.tz_convert("UTC") > frozen_at.tz_convert("UTC")]
    if len(eligible) == 0:
        raise ValueError("No future exchange session provided; cannot set holdout")
    return eligible[0]


def _git(root: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True, check=False
    )
    return result.stdout if result.returncode == 0 else ""


def _redact_diff(text: str) -> str:
    """Defense in depth: only allowlisted code is diffed; credential lines elided."""
    pattern = re.compile(r"(?i)(api[_-]?key|access[_-]?token|password|secret|authorization|bearer|private[_-]?key)")
    return "\n".join(
        "[REDACTED credential-related diff line]" if pattern.search(line) else line
        for line in text.splitlines()
    ) + "\n"


def _file_record(root: Path, path: Path, *, endpoints: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if endpoints and path.suffix == ".parquet":
        try:
            parquet = pq.ParquetFile(path)
            record["rows"] = parquet.metadata.num_rows
            timestamp_column = next(
                (column for column in ["date", "timestamp", "datetime", "Date", "__index_level_0__"] if column in parquet.schema.names),
                None,
            )
            if timestamp_column is not None:
                values = parquet.read(columns=[timestamp_column]).column(timestamp_column).to_pandas()
                timestamps = pd.to_datetime(values, errors="raise")
                if timestamps.isna().any() or timestamps.empty:
                    raise ValueError("Missing or empty timestamp column")
                record.update(index_start=str(timestamps.min()), index_end=str(timestamps.max()), timestamp_column=timestamp_column)
            else:
                frame = pd.read_parquet(path)
                if isinstance(frame.index, pd.DatetimeIndex) and len(frame):
                    record.update(index_start=str(frame.index.min()), index_end=str(frame.index.max()), timestamp_column="pandas_datetime_index")
                else:
                    record["endpoint_status"] = "timestamp_not_identified_requires_source_adapter"
        except (ValueError, OSError, ImportError) as error:
            record["endpoint_status"] = type(error).__name__
    return record


def freeze_protocol(
    root: Path,
    protocol_path: Path,
    output_dir: Path,
    *,
    stage: str = "protocol_only",
    final_code_ready: bool = False,
    session_calendar_path: Path | None = None,
    frozen_at: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Snapshot params/code/data and dirty diffs, refusing any existing output.

    ``frozen_at`` is injectable for deterministic tests. The CLI always uses now.
    The manifest honestly records a dirty tree; it does not mutate Git or data.
    No environment variables, .env files, credential stores or Git config read.
    """
    root = root.resolve()
    protocol_path = protocol_path.resolve()
    protocol = load_protocol(protocol_path)
    if stage not in {"protocol_only", "final_code"}:
        raise ValueError("stage must be protocol_only or final_code")
    now = _utc_now() if frozen_at is None else pd.Timestamp(frozen_at)
    if now.tzinfo is None:
        raise ValueError("frozen_at must be timezone-aware")
    now = now.tz_convert("UTC")
    holdout = None
    calendar_record = None
    opens = None
    if stage == "final_code":
        if not final_code_ready or session_calendar_path is None:
            raise ValueError("Final freeze requires readiness attestation and exchange session calendar")
        calendar_frame = pd.read_csv(session_calendar_path)
        if any(pd.Timestamp(value).tzinfo is None for value in calendar_frame["market_open"]):
            raise ValueError("Exchange session calendar timestamps must be timezone-aware")
        opens = pd.DatetimeIndex(pd.to_datetime(calendar_frame["market_open"], utc=True))
        calendar_record = {"sha256": sha256_file(session_calendar_path), "path": str(session_calendar_path.resolve())}
    code_paths = set()
    for relative in protocol["provenance"]["code_roots"]:
        base = root / relative
        if base.exists():
            code_paths.update(path for path in base.rglob("*") if path.is_file() and _safe_path(root, path))
    code_paths.update(path for path in [root / "pyproject.toml", root / "README.md"] if path.exists())
    code_records = [_file_record(root, path) for path in sorted(code_paths)]
    data_paths = set()
    missing_patterns = []
    for pattern in protocol["provenance"]["data_globs"]:
        matches = [path for path in root.glob(pattern) if path.is_file() and _safe_path(root, path)]
        data_paths.update(matches)
        if not matches:
            missing_patterns.append(pattern)
    data_records = [_file_record(root, path, endpoints=True) for path in sorted(data_paths)]
    research_inventory = []
    for relative in protocol["provenance"]["research_inventory_roots"]:
        base = root / relative
        if base.exists():
            research_inventory.extend(
                {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
                for path in sorted(base.rglob("*")) if path.is_file() and _safe_path(root, path)
            )
    # Individual allowlisted paths prevent .env/secrets from entering git diff.
    pathspecs = [record["path"] for record in code_records]
    diff = _redact_diff(_git(root, ["diff", "--no-ext-diff", "--no-textconv", "HEAD", "--", *pathspecs]))
    versions = {"python": platform.python_version()}
    for package in DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    parameters_canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol_bytes = protocol_path.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    if json.dumps(yaml.safe_load(protocol_bytes), sort_keys=True, separators=(",", ":")) != parameters_canonical:
        raise ValueError("Protocol changed during provenance capture")
    # Claim no completed freeze until every exact input has an immutable copy.
    # A crash leaves a partial directory without manifest.json and cannot be
    # silently resumed/overwritten as if a complete freeze had succeeded.
    output_dir.mkdir(parents=True, exist_ok=False)
    for record in code_records + data_records:
        relative = Path("snapshot") / record["path"]
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(root / record["path"], destination)
        if sha256_file(destination) != record["sha256"]:
            raise ValueError(f"Input changed during snapshot: {record['path']}")
        record["snapshot_path"] = relative.as_posix()
    for record in code_records:
        if sha256_file(root / record["path"]) != record["sha256"]:
            raise ValueError(f"Code changed during snapshot: {record['path']}")
    if sha256_file(protocol_path) != protocol_hash:
        raise ValueError("Protocol changed during snapshot")
    (output_dir / "protocol.yaml").write_bytes(protocol_bytes)
    (output_dir / "dirty_tree_redacted.patch").write_text(diff)
    # A production freeze crossing a session open must choose the NEXT session.
    # Deterministic test timestamps retain their explicitly injected value.
    if frozen_at is None:
        now = _utc_now().tz_convert("UTC")
    if stage == "final_code":
        if sha256_file(session_calendar_path) != calendar_record["sha256"]:
            raise ValueError("Exchange calendar changed during snapshot")
        holdout = next_complete_session(now, opens).isoformat()
        for record in data_records:
            if "index_end" in record:
                endpoint = pd.Timestamp(record["index_end"])
                if endpoint.tzinfo is None:
                    endpoint = endpoint.tz_localize(protocol.get("strategy", {}).get("timezone", "America/New_York"))
                if endpoint.tz_convert("UTC") >= pd.Timestamp(holdout):
                    raise ValueError("Proposed holdout overlaps already known local bars")
    manifest = {
        "schema_version": 1,
        "snapshot_layout_version": 1,
        "stage": stage,
        "frozen_at_utc": now.isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "parameters_sha256": hashlib.sha256(parameters_canonical.encode()).hexdigest(),
        "git_head": _git(root, ["rev-parse", "HEAD"]).strip() or None,
        "reviewed_state": "file hashes plus dirty-tree diff, not git HEAD alone",
        "dirty_tree_diff": "dirty_tree_redacted.patch",
        "dirty_tree_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "code_files": code_records,
        "data_files": data_records,
        "unmatched_data_patterns": missing_patterns,
        "dependencies": versions,
        "research_inventory": research_inventory,
        "inventory_limitation": "filenames are not independent trials or a complete search count",
        "historical_evidence": "all currently known local history is retrospective, including bars after June 2026",
        "known_local_bar_endpoints": [r for r in data_records if "index_end" in r],
        "genuine_holdout_start_utc": holdout,
        "exchange_calendar": calendar_record,
        "prospective_eligible": stage == "final_code",
        "prospective_limitation": "future results only; code/protocol changes require a new freeze",
        "verification_policy": (
            "original code plus archived code/data; original market caches may update"
            if stage == "final_code" else "original and archived code/data must all match"
        ),
    }
    # Manifest is the last completion marker, after exact archive verification.
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_frozen_inputs(root: Path, manifest_path: Path) -> None:
    """Verify snapshots; protocol runs also bind originals, forward runs code only.

    Final-code data must be archived so routine new market observations cannot
    invalidate the old development-history record. Retrospective runners must
    separately bind their actual runtime data against the manifest records.
    """
    manifest = json.loads(manifest_path.read_text())
    final = manifest.get("stage") == "final_code"
    archive_required = manifest.get("snapshot_layout_version") == 1
    for category in ["code_files", "data_files"]:
        for record in manifest[category]:
            snapshot = record.get("snapshot_path")
            if snapshot is not None:
                path = manifest_path.parent / snapshot
                try:
                    path.resolve().relative_to((manifest_path.parent / "snapshot").resolve())
                except ValueError as error:
                    raise ValueError("Snapshot path escapes archive") from error
                if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
                    raise ValueError(f"Frozen snapshot changed or missing: {record['path']}")
            elif archive_required or (final and category == "data_files"):
                raise ValueError(f"Frozen snapshot required: {record['path']}")
            if category == "code_files" or not final:
                path = root / record["path"]
                if not path.is_file() or sha256_file(path) != record["sha256"]:
                    raise ValueError(f"Frozen input changed or missing: {record['path']}")
