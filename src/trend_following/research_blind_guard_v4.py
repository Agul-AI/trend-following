"""Mechanical, fail-closed data partitioning and macOS child-process custody.

This module contains no strategy selection or performance calculations. Custodian
ownership means coordinator control, NOT Unix uid 0. Same-user unsandboxed tools
can still read or modify files; fresh context and an explicit tool allowlist are
an additional procedural boundary. The sandbox only governs spawned processes.
Python audit events supplement, but do not replace, kernel enforcement and do
not exhaustively trace native-library file access.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import json
import os
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
# Prepare-time raw custody is explicit/configurable; prepared manifests bind the
# canonical actual source paths, independent of this default on later runs.
ROOT = Path(os.environ.get("QQQ_BLIND_RAW_PROJECT", str(REPO))).expanduser().resolve()
DEFAULT_STUDY = REPO / "reports/research_validation/20261006_v4_blind_private"
PYTHON = Path.home() / ".venvs/myenv/bin/python"
SANDBOX = Path("/usr/bin/sandbox-exec")
SOURCES = {
    "qqq": ROOT / "data/raw/alpha_vantage_daily_adjusted/QQQ.parquet",
    "sessions": ROOT / "data/raw/yfinance_daily/QQQ.parquet",
    **{
        n: ROOT / "data/raw/research_rates/20261002_research_v1" / n
        for n in (
            "DFF_raw.csv",
            "DTB3_raw.csv",
            "DFF_available.csv",
            "DTB3_available.csv",
            "metadata.json",
        )
    },
}
STAGES = {
    "development": ("1999-11-01", "2011-12-31", "2002-01-01"),
    "validation": ("2012-01-01", "2015-12-31", "2012-01-01"),
    "test": ("2016-01-01", "2026-06-01", "2016-01-01"),
}
COLUMNS = ["session", "close", "dividend_amount", "split_coefficient"]
RATE_COLUMNS = [
    "observation_date",
    "quoted_percent",
    "available_at",
    "annual_rate",
    "series_id",
    "source_url",
    "vintage_type",
    "availability_assumption",
]


class GuardError(RuntimeError):
    """An integrity check or mandatory kernel boundary failed."""


def _locked(function):
    """Serialize every run/probe/seal; a competing invocation fails closed."""

    @functools.wraps(function)
    def wrapped(study_path, *args, **kwargs):
        import fcntl

        study = _canonical(study_path)
        lock = study / "custodian/operation.lock"
        with open(lock, "a+b") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise GuardError("Custodian study operation already active") from exc
            try:
                return function(study, *args, **kwargs)
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    return wrapped


def _canonical(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def sha256(path: Path | str) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _exclusive(path: Path, data: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, mode)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _files(base: Path) -> list[Path]:
    if not base.exists():
        return []
    paths = sorted(base.rglob("*"))
    if any(p.is_symlink() for p in paths) or base.is_symlink():
        raise GuardError(f"Symlinks forbidden in custody/allowed trees: {base}")
    if any(not (stat.S_ISREG(p.lstat().st_mode) or stat.S_ISDIR(p.lstat().st_mode)) for p in paths):
        raise GuardError(f"Special filesystem nodes forbidden: {base}")
    files = [p for p in paths if p.is_file()]
    if any(p.stat().st_nlink != 1 for p in files):
        raise GuardError(f"Hard links forbidden in custody/allowed trees: {base}")
    return files


def _hash_tree(base: Path) -> dict[str, str]:
    return {str(p.relative_to(base)): sha256(p) for p in _files(base)}


def _assert_hash_tree(base: Path, expected: Mapping[str, str]) -> None:
    if _hash_tree(base) != expected:
        raise GuardError(f"Protected content changed: {base}")


def _append_audit(study: Path, event: dict[str, Any]) -> dict[str, Any]:
    log = study / "custodian/audit.jsonl"
    previous = "0" * 64
    if log.exists():
        lines = log.read_text().splitlines()
        if lines:
            previous = json.loads(lines[-1])["event_sha256"]
    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "previous_sha256": previous,
        **event,
    }
    digest = hashlib.sha256(_json_bytes(record)).hexdigest()
    record["event_sha256"] = digest
    with open(log, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(log, 0o600)
    return record


def _check_audit(study: Path) -> str:
    previous = "0" * 64
    for line in (study / "custodian/audit.jsonl").read_text().splitlines():
        row = json.loads(line)
        digest = row.pop("event_sha256")
        if (
            row["previous_sha256"] != previous
            or hashlib.sha256(_json_bytes(row)).hexdigest() != digest
        ):
            raise GuardError("Custodian audit hash chain is broken")
        previous = digest
    return previous


def _load(study: Path | str) -> tuple[Path, dict[str, Any]]:
    study = _canonical(study)
    manifest = json.loads((study / "custodian/manifest.json").read_text())
    if manifest["study"] != str(study):
        raise GuardError("Study moved: path binding no longer matches")
    _assert_hash_tree(study / "developer/input", manifest["developer_input_hashes"])
    _assert_hash_tree(study / "developer/engine", manifest["engine_hashes"])
    _assert_hash_tree(study / "held", manifest["held_hashes"])
    _assert_hash_tree(study / "custodian/rate_provenance", manifest["rate_provenance_hashes"])
    for tree in ("developer/workspace", "developer/output", "custodian"):
        _files(study / tree)
    _check_audit(study)
    first_event = json.loads((study / "custodian/audit.jsonl").read_text().splitlines()[0])
    if first_event.get("event") != "prepare" or first_event["manifest_sha256"] != sha256(
        study / "custodian/manifest.json"
    ):
        raise GuardError("Manifest no longer matches preparation custody binding")
    if sha256(study / "custodian/runner.py") != manifest["runner_sha256"]:
        raise GuardError("Audit runner changed")
    return study, manifest


def _validate_rates(sources: Mapping[str, Path]):
    import numpy as np
    import pandas as pd
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay

    offset = CustomBusinessDay(2, calendar=USFederalHolidayCalendar())
    frames = {}
    for series in ("DFF", "DTB3"):
        raw = pd.read_csv(sources[f"{series}_raw.csv"], na_values=["."])
        if set(raw.columns) != {"observation_date", series}:
            raise GuardError(f"{series} raw schema changed")
        raw["observation_date"] = pd.to_datetime(raw["observation_date"], errors="raise")
        raw[series] = pd.to_numeric(raw[series], errors="coerce")
        if raw["observation_date"].duplicated().any():
            raise GuardError(f"Duplicate {series} observations")
        normalized = pd.read_csv(sources[f"{series}_available.csv"])
        if set(normalized.columns) != set(RATE_COLUMNS):
            raise GuardError(f"{series} available schema changed; unexpected columns forbidden")
        normalized = normalized[RATE_COLUMNS]
        normalized["observation_date"] = pd.to_datetime(
            normalized["observation_date"], errors="raise"
        )
        normalized["available_at"] = pd.to_datetime(
            normalized["available_at"], utc=True, errors="raise"
        )
        if normalized["observation_date"].duplicated().any():
            raise GuardError(f"Duplicate normalized {series} observations")
        available = raw.dropna(subset=[series]).sort_values("observation_date")
        normalized = normalized.sort_values("observation_date").reset_index(drop=True)
        if available["observation_date"].tolist() != normalized["observation_date"].tolist():
            raise GuardError(f"{series} quote coverage does not match raw nonmissing observations")
        quotes = available[series].to_numpy(float)
        if not np.allclose(
            quotes, normalized["quoted_percent"].to_numpy(float), rtol=0, atol=1e-12
        ):
            raise GuardError(f"{series} quotes changed during normalization")
        dates = [x + offset for x in normalized["observation_date"]]
        expected = (
            pd.DatetimeIndex(dates).tz_localize("America/New_York") + pd.Timedelta(hours=18)
        ).tz_convert("UTC")
        if not (expected == normalized["available_at"]).all():
            raise GuardError(
                f"{series} availability is not the documented two-federal-business-day lag at 18:00 NY"
            )
        fraction = quotes / 100
        annual = (
            fraction if series == "DFF" else fraction * (365 / 360) / (1 - fraction * (91 / 360))
        )
        if not np.allclose(annual, normalized["annual_rate"].to_numpy(float), rtol=0, atol=1e-12):
            raise GuardError(f"{series} annual-rate normalization failed")
        if not (normalized["series_id"] == series).all():
            raise GuardError(f"{series} identifiers inconsistent")
        frames[series] = normalized
    return frames


def source_map_from_json(path: Path | str) -> dict[str, Path]:
    """Read an explicit coordinator-supplied JSON role -> absolute path mapping."""
    value = json.loads(_canonical(path).read_text())
    if not isinstance(value, dict) or set(value) != set(SOURCES):
        raise GuardError("Source map must include exactly all required source roles")
    result = {}
    for role, name in value.items():
        if not isinstance(name, str) or not Path(name).expanduser().is_absolute():
            raise GuardError(f"Source role {role} must have an absolute filesystem path")
        result[role] = _canonical(name)
    return result


def prepare_study(
    study_path: Path | str = DEFAULT_STUDY,
    trusted_engine_dir: Path | str | None = None,
    *,
    sources: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Export mechanically partitioned raw data. Exclusive creation; never overwrite.

    ``sources`` exists for synthetic tests; production CLI fixes the source allowlist.
    The engine must be a newly supplied generic numerical/accounting directory.
    """
    import numpy as np
    import pandas as pd

    study = _canonical(study_path)
    if study.exists():
        raise GuardError("Study already exists; prepare never overwrites input or custody")
    source = {k: _canonical(v) for k, v in (sources or SOURCES).items()}
    if set(source) != set(SOURCES):
        raise GuardError("Source inventory mismatch")
    source_hashes = {k: sha256(v) for k, v in source.items()}
    qqq = pd.read_parquet(
        source["qqq"], columns=["date", "close", "dividend_amount", "split_coefficient"]
    )
    reference = pd.read_parquet(
        source["sessions"], columns=["date"]
    )  # DATES ONLY, never quote columns.
    qqq = qqq.rename(columns={"date": "session"})
    qqq["session"] = pd.to_datetime(qqq["session"]).dt.normalize()
    ref_dates = pd.to_datetime(reference["date"]).dt.normalize()
    if qqq["session"].duplicated().any() or ref_dates.duplicated().any():
        raise GuardError("Duplicate raw/reference session dates")
    qqq = qqq.sort_values("session").reset_index(drop=True)
    qqq = qqq[(qqq.session >= STAGES["development"][0]) & (qqq.session <= STAGES["test"][1])]
    reference_dates = set(
        ref_dates[(ref_dates >= STAGES["development"][0]) & (ref_dates <= STAGES["test"][1])]
    )
    if set(qqq.session) != reference_dates:
        raise GuardError("Raw AV session dates differ from independent date-only reference")
    values = qqq[["close", "dividend_amount", "split_coefficient"]].to_numpy(float)
    if (
        not np.isfinite(values).all()
        or (values[:, 0] <= 0).any()
        or (values[:, 1] < 0).any()
        or (values[:, 2] <= 0).any()
    ):
        raise GuardError("Invalid raw close/dividend/split values")
    rates = _validate_rates(source)
    engine = _canonical(trusted_engine_dir) if trusted_engine_dir else None
    if engine is not None:
        engine_files = _files(engine)
        if not engine_files or not (engine / "engine.py").is_file():
            raise GuardError("Trusted engine directory must include engine.py")
    else:
        engine_files = []
    study.mkdir(parents=True, exist_ok=False)
    try:
        for tree in (
            "developer/input",
            "developer/engine",
            "developer/workspace",
            "developer/output",
            "held/validation",
            "held/test",
            "custodian/rate_provenance",
            "custodian/profiles",
            "custodian/snapshots",
            "custodian/seals",
            "custodian/runs",
        ):
            (study / tree).mkdir(parents=True, exist_ok=True)
        os.chmod(study, 0o700)
        for name in (
            "DFF_raw.csv",
            "DTB3_raw.csv",
            "DFF_available.csv",
            "DTB3_available.csv",
            "metadata.json",
        ):
            _exclusive(study / "custodian/rate_provenance" / name, source[name].read_bytes())
        for path in engine_files:
            _exclusive(study / "developer/engine" / path.relative_to(engine), path.read_bytes())
        stage_records = {}
        metadata = json.loads(source["metadata.json"].read_text())
        for stage, (start, end, scoring) in STAGES.items():
            target = study / ("developer/input" if stage == "development" else f"held/{stage}")
            rows = qqq[(qqq.session >= start) & (qqq.session <= end)].copy()
            if rows.empty:
                raise GuardError(f"Empty {stage} partition")
            rows["session"] = rows["session"].dt.strftime("%Y-%m-%d")
            _exclusive(
                target / "QQQ.csv", rows[COLUMNS].to_csv(index=False, float_format="%.17g").encode()
            )
            cutoff = pd.Timestamp(f"{end} 16:00", tz="America/New_York").tz_convert("UTC")
            rate_counts = {}
            for series, frame in rates.items():
                subset = frame[frame.available_at <= cutoff].copy()
                subset["observation_date"] = subset["observation_date"].dt.strftime("%Y-%m-%d")
                subset["available_at"] = subset["available_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                _exclusive(
                    target / f"{series}_available.csv",
                    subset.to_csv(index=False, float_format="%.17g").encode(),
                )
                rate_counts[series] = len(subset)
            record = {
                "stage": stage,
                "feature_start": start,
                "scoring_start": scoring,
                "end": end,
                "rate_cutoff_utc": cutoff.isoformat(),
                "session_count": len(rows),
                "session_first": rows.session.iloc[0],
                "session_last": rows.session.iloc[-1],
                "raw_price_columns": COLUMNS,
                "rate_rows": rate_counts,
                "rate_limitation": metadata.get("vintage"),
                "availability_limitation": metadata.get("availability_limitation"),
                "reference_use": "independent session-date coverage only; quote columns never loaded",
            }
            _exclusive(target / "input_manifest.json", _json_bytes(record))
            stage_records[stage] = record
        runner = _RUNNER.encode()
        _exclusive(study / "custodian/runner.py", runner)
        manifest = {
            "schema": "qqq-blind-custody-v4",
            "study": str(study),
            "stages": stage_records,
            "sources": {k: {"path": str(source[k]), "sha256": v} for k, v in source_hashes.items()},
            "developer_input_hashes": _hash_tree(study / "developer/input"),
            "engine_hashes": _hash_tree(study / "developer/engine"),
            "held_hashes": _hash_tree(study / "held"),
            "rate_provenance_hashes": _hash_tree(study / "custodian/rate_provenance"),
            "runner_sha256": sha256(study / "custodian/runner.py"),
            "custodian_model": "coordinator-controlled, not Unix uid 0; outside child write allowance",
            "tool_limit": "Unsandboxed same-user agent tools are not kernel-restricted; explicit allowlist and fresh context required",
            "audit_limit": "Supplemental Python open/import audit is not a complete native syscall trace",
        }
        _exclusive(study / "custodian/manifest.json", _json_bytes(manifest))
        _append_audit(
            study,
            {
                "event": "prepare",
                "manifest_sha256": sha256(study / "custodian/manifest.json"),
                "source_hashes": source_hashes,
                "partition_hashes": manifest["held_hashes"],
            },
        )
        return manifest
    except BaseException as exc:
        _exclusive(study / "custodian/PREPARE_FAILED", str(exc).encode(), 0o600)
        raise


def _quote(path: Path | str) -> str:
    return json.dumps(str(path))


def _profile(
    study: Path,
    snapshot: Path,
    *,
    stage: str = "development",
    input_override: Path | None = None,
    output_override: Path | None = None,
    extra_read_files: Sequence[Path] = (),
) -> str:
    if stage != "development" and (input_override is None or output_override is None):
        raise GuardError(
            "Held stages require an explicit sealed released packet and operator output"
        )
    input_dir = input_override or study / "developer/input"
    output_dir = output_override or study / "developer/output"
    readable = [
        input_dir,
        study / "developer/engine",
        output_dir,
        snapshot,
        Path("/usr/local"),
        Path.home() / ".venvs/myenv",
        Path("/System/Library"),
        Path("/usr/lib"),
        Path("/Library/Apple"),
        Path("/private/var/db/dyld"),
        Path("/usr/share/zoneinfo"),
        Path("/usr/share/zoneinfo").resolve(),
    ]
    if stage == "development":
        readable.append(study / "developer/workspace")
    literals = [
        *extra_read_files,
        Path("/"),
        study / "custodian/runner.py",
        Path("/dev/null"),
        Path("/dev/random"),
        Path("/dev/urandom"),
        Path("/private/etc/localtime"),
        Path("/etc/localtime"),
    ]
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec)",
        "(deny process-fork)",
        "(allow sysctl-read)",
        "(deny network*)",
        "(allow file-read-metadata)",
        "(allow signal (target self))",
    ]
    lines += [f"(allow file-read* (subpath {_quote(p)}))" for p in readable]
    lines += [f"(allow file-read* (literal {_quote(p)}))" for p in literals]
    lines += [
        '(allow file-read* file-write-data (regex #"^/dev/fd/[0-9]+$"))',
        '(allow file-write-data (literal "/dev/null"))',
    ]
    write_bases = (
        (study / "developer/workspace", output_dir) if stage == "development" else (output_dir,)
    )
    for base in write_bases:
        lines.append(f"(allow file-write* (subpath {_quote(base)}))")
        _files(base)
        for existing in [base, *sorted(base.rglob("*"))]:
            lines.append(f"(deny file-write* (literal {_quote(existing)}))")
    bound_sources = json.loads((study / "custodian/manifest.json").read_text())["sources"]
    for record in bound_sources.values():
        lines.append(f"(deny file-read-data (literal {_quote(record['path'])}))")
    for path in (ROOT / "data/raw", ROOT / "reports", REPO / "reports"):
        # Do not deny study's own allowed descendants via broad REPO/reports rule.
        if not study.is_relative_to(path):
            lines.append(f"(deny file-read-data (subpath {_quote(path)}))")
    lines.append(f"(deny file-read-data (subpath {_quote(study / 'held')}))")
    if stage != "development":
        lines.append(f"(deny file-write* (subpath {_quote(study / 'developer/workspace')}))")
        lines.append(f"(deny file-write* (subpath {_quote(study / 'developer/output')}))")
    lines.append(f"(deny file-write* (subpath {_quote(study / 'released')}))")
    lines += [
        f"(deny file-write* (subpath {_quote(study / 'developer/input')}))",
        f"(deny file-write* (subpath {_quote(study / 'developer/engine')}))",
        f"(deny file-write* (subpath {_quote(study / 'held')}))",
        f"(deny file-write* (subpath {_quote(study / 'custodian')}))",
    ]
    return "\n".join(lines) + "\n"


_RUNNER = r"""import hashlib, json, os, runpy, sys
prefix, script, engine, workspace, input_dir, output_dir, *args = sys.argv[1:]
busy = False
counter = 0
def hook(event, arguments):
    global busy, counter
    if busy or event not in ('open', 'import', 'subprocess.Popen', 'socket.connect', 'socket.__new__'):
        return
    busy = True
    try:
        # Path names only; no values, input bytes, or quote statistics.
        value = arguments[0] if arguments else None
        if isinstance(value, (str, int)):
            counter += 1
            row = {'event': event, 'target': value, 'sequence': counter}
            sys.__stderr__.write(prefix + json.dumps(row, sort_keys=True) + '\n')
            sys.__stderr__.flush()
    finally:
        busy = False
sys.addaudithook(hook)
sys.path[:0] = [engine, workspace]
os.environ['BLIND_INPUT_DIR'] = input_dir
os.environ['BLIND_OUTPUT_DIR'] = output_dir
sys.argv = [script] + args
runpy.run_path(script, run_name='__main__')
"""


def _launch(
    study: Path,
    snapshot: Path,
    args: Sequence[str],
    timeout: float,
    *,
    stage: str = "development",
    input_override: Path | None = None,
    output_override: Path | None = None,
    extra_read_files: Sequence[Path] = (),
) -> dict[str, Any]:
    if not SANDBOX.is_file() or not os.access(SANDBOX, os.X_OK):
        raise GuardError("macOS sandbox-exec unavailable: refusing unsandboxed fallback")
    if not PYTHON.is_file():
        raise GuardError("Required virtual-environment Python unavailable")
    identifier = f"{time.time_ns()}-{secrets.token_hex(6)}"
    profile = _profile(
        study,
        snapshot.parent,
        stage=stage,
        input_override=input_override,
        output_override=output_override,
        extra_read_files=extra_read_files,
    )
    profile_path = study / "custodian/profiles" / f"{identifier}.sb"
    _exclusive(profile_path, profile.encode())
    prefix = f"BLIND_AUDIT_{secrets.token_hex(24)} "
    input_dir = input_override or study / "developer/input"
    output_dir = output_override or study / "developer/output"
    command = [
        str(SANDBOX),
        "-f",
        str(profile_path),
        str(PYTHON),
        "-I",
        "-B",
        str(study / "custodian/runner.py"),
        prefix,
        str(snapshot),
        str(study / "developer/engine"),
        str(study / "developer/workspace") if stage == "development" else str(snapshot.parent),
        str(input_dir),
        str(output_dir),
        *map(str, args),
    ]
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(study / "developer/workspace") if stage == "development" else str(output_dir),
        "TMPDIR": str(output_dir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(output_dir / "matplotlib"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "LC_ALL": "C",
        "BLIND_INPUT_DIR": str(input_dir),
        "BLIND_OUTPUT_DIR": str(output_dir),
    }
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=study / "developer/workspace" if stage == "development" else output_dir,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    output_limit_exceeded = False
    capture_limit = 8 * 1024 * 1024  # per stream; bounded custody memory
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    terminated = False
    try:
        while selector.get_map():
            if time.monotonic() - started > timeout and not terminated:
                timed_out = True
                terminated = True
                os.killpg(process.pid, signal.SIGKILL)
            for key, _ in selector.select(timeout=0.1):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                remaining_capacity = capture_limit - len(captured[key.data])
                captured[key.data].extend(chunk[: max(0, remaining_capacity)])
                if len(chunk) > remaining_capacity and not terminated:
                    output_limit_exceeded = True
                    terminated = True
                    os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    finally:
        selector.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    stdout, stderr = bytes(captured["stdout"]), bytes(captured["stderr"])
    events, remaining = [], []
    for line in stderr.decode("utf-8", "replace").splitlines():
        if line.startswith(prefix):
            try:
                events.append(json.loads(line[len(prefix) :]))
            except json.JSONDecodeError:
                remaining.append(line)
        else:
            remaining.append(line)
    result = {
        "run_id": identifier,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "output_limit_exceeded": output_limit_exceeded,
        "capture_limit_bytes_per_stream": capture_limit,
        "duration_seconds": time.monotonic() - started,
        "stdout": stdout.decode("utf-8", "replace"),
        "stderr": "\n".join(remaining),
        "input_access_audit": events,
        "audit_scope": "supplemental Python events; kernel deny-default policy governs all syscalls",
        "profile_sha256": sha256(profile_path),
        "script_sha256": sha256(snapshot),
        "stage": stage,
    }
    _exclusive(study / "custodian/runs" / f"{identifier}.json", _json_bytes(result), 0o600)
    return result


@_locked
def probe_guard(study_path: Path | str, prior_report_path: Path | str) -> dict[str, Any]:
    """Actual subprocess probes must all pass before any development run.

    The prior report is opened inside the denied sandbox, never read by custody.
    """
    study, manifest = _load(study_path)
    prior = _canonical(prior_report_path)
    if not prior.is_file():
        raise GuardError("Prior report probe must target a real existing file")
    if (
        prior.is_relative_to(study)
        or prior.is_relative_to(Path("/usr/local"))
        or prior.is_relative_to(Path.home() / ".venvs/myenv")
    ):
        raise GuardError("Prior report probe must be outside all research read allowances")
    blocked = {
        "held_validation": str(study / "held/validation/QQQ.csv"),
        "held_test": str(study / "held/test/QQQ.csv"),
        "original_raw": manifest["sources"]["qqq"]["path"],
        "prior_report": str(prior),
        "input_write": str(study / "developer/input/QQQ.csv"),
        "engine_write": str(study / "developer/engine/engine.py"),
        "custodian_write": str(study / "custodian/audit.jsonl"),
    }
    code = """import json, os, socket, sys\nresults={}\n"""
    code += "targets=" + repr(blocked) + "\n"
    code += """for name, path in targets.items():
    try:
        fd=os.open(path, os.O_WRONLY if name.endswith('_write') else os.O_RDONLY)
        os.close(fd)
        results[name]=False
    except PermissionError:
        results[name]=True
    except Exception as exc:
        results[name]='unexpected:'+repr(exc)
try:
    sock=socket.socket(); sock.settimeout(1); sock.connect(('1.1.1.1',443)); sock.close()
    results['network']=False
except PermissionError:
    results['network']=True
except Exception as exc:
    results['network']='unexpected:'+repr(exc)
try:
    pid=os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid,0)
    results['fork']=False
except PermissionError:
    results['fork']=True
except Exception as exc:
    results['fork']='unexpected:'+repr(exc)
try:
    pid=os.posix_spawn(sys.executable,[sys.executable,'-I','-c','pass'],dict(os.environ))
    os.waitpid(pid,0)
    results['spawn']=False
except PermissionError:
    results['spawn']=True
except Exception as exc:
    results['spawn']='unexpected:'+repr(exc)
try:
    import pandas as pd
    assert pd.Timestamp('2011-01-03').tz_localize('UTC').tz_convert('America/New_York') is not None
    frame=pd.read_csv(os.environ['BLIND_INPUT_DIR']+'/QQQ.csv')
    assert list(frame.columns)==['session','close','dividend_amount','split_coefficient']
    results['allowed_input_and_pandas']=True
except Exception as exc:
    results['allowed_input_and_pandas']='unexpected:'+repr(exc)
print(json.dumps(results,sort_keys=True))
if not all(v is True for v in results.values()):
    raise SystemExit(99)
"""
    snap = study / "custodian/snapshots" / f"probe-{time.time_ns()}" / "probe.py"
    _exclusive(snap, code.encode())
    result = _launch(study, snap, [], 90)
    okay = result["returncode"] == 0
    try:
        checks = json.loads(result["stdout"].strip())
        expected_checks = set(blocked) | {"network", "fork", "spawn", "allowed_input_and_pandas"}
        okay = (
            okay
            and set(checks) == expected_checks
            and all(value is True for value in checks.values())
        )
    except (ValueError, TypeError):
        checks, okay = {}, False
    record = _append_audit(
        study,
        {
            "event": "kernel_probe",
            "passed": okay,
            "checks": checks,
            "run_id": result["run_id"],
            "returncode": result["returncode"],
            "profile_sha256": result["profile_sha256"],
            "prior_probe_path": str(prior),
            "manifest_sha256": sha256(study / "custodian/manifest.json"),
            "guard_source_sha256": sha256(Path(__file__)),
            "python_binary_sha256": sha256(PYTHON.resolve()),
            "sandbox_binary_sha256": sha256(SANDBOX),
        },
    )
    if okay:
        _exclusive(
            study / "custodian" / f"PROBE_PASSED_{record['event_sha256']}.json", _json_bytes(record)
        )
    else:
        raise GuardError(
            f"Kernel guard probe failed closed: {result['stderr']}\n{result['stdout']}"
        )
    return {
        "passed": True,
        "checks": checks,
        "run_id": result["run_id"],
        "audit_sha256": record["event_sha256"],
    }


def _require_probe(study: Path) -> None:
    log = [json.loads(x) for x in (study / "custodian/audit.jsonl").read_text().splitlines()]
    probes = [row for row in log if row.get("event") == "kernel_probe"]
    if (
        not probes
        or not probes[-1]["passed"]
        or probes[-1]["manifest_sha256"] != sha256(study / "custodian/manifest.json")
        or probes[-1]["guard_source_sha256"] != sha256(Path(__file__))
        or probes[-1]["python_binary_sha256"] != sha256(PYTHON.resolve())
        or not SANDBOX.is_file()
        or probes[-1]["sandbox_binary_sha256"] != sha256(SANDBOX)
    ):
        raise GuardError("A successful bound kernel probe is required; no unsandboxed fallback")


@_locked
def run_guarded(
    study_path: Path | str,
    script_path: Path | str,
    args: Sequence[str] = (),
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    """Run development code only, after kernel probe. Custody records every exit."""
    study, manifest = _load(study_path)
    _require_probe(study)
    if (study / "custodian/DEVELOPMENT_SEALED.json").exists():
        raise GuardError("Development sealed; new research execution forbidden")
    script = _canonical(script_path)
    if not (
        script.is_relative_to(study / "developer/workspace")
        or script.is_relative_to(study / "developer/engine")
    ):
        raise GuardError("Script must be in isolated developer workspace or trusted engine")
    if not script.is_file() or script.suffix != ".py":
        raise GuardError("Guard runs only existing Python source files")
    if sha256(study / "custodian/runner.py") != manifest["runner_sha256"]:
        raise GuardError("Audit runner changed")
    source_before = _hash_tree(study / "developer/workspace")
    output_before = _hash_tree(study / "developer/output")
    source_bytes = script.read_bytes()
    snap = (
        study / "custodian/snapshots" / f"run-{time.time_ns()}-{secrets.token_hex(6)}" / script.name
    )
    _exclusive(snap, source_bytes)
    started = _append_audit(
        study,
        {
            "event": "run_start",
            "script": str(script),
            "script_sha256": sha256(snap),
            "source_hashes": source_before,
            "engine_hashes": manifest["engine_hashes"],
            "input_hashes": manifest["developer_input_hashes"],
            "args": list(args),
        },
    )
    try:
        result = _launch(study, snap, args, timeout_seconds)
        _load(study)
        source_after = _hash_tree(study / "developer/workspace")
        output_after = _hash_tree(study / "developer/output")
        existing_unchanged = all(
            source_after.get(k) == v for k, v in source_before.items()
        ) and all(output_after.get(k) == v for k, v in output_before.items())
        if not existing_unchanged:
            raise GuardError("Existing workspace/output artifact overwritten")
        _append_audit(
            study,
            {
                "event": "run_exit",
                "run_id": result["run_id"],
                "start_event_sha256": started["event_sha256"],
                "script_sha256": result["script_sha256"],
                "profile_sha256": result["profile_sha256"],
                "returncode": result["returncode"],
                "timed_out": result["timed_out"],
                "output_limit_exceeded": result["output_limit_exceeded"],
                "new_source_hashes": {
                    k: v for k, v in source_after.items() if k not in source_before
                },
                "new_output_hashes": {
                    k: v for k, v in output_after.items() if k not in output_before
                },
                "input_access_events": result["input_access_audit"],
            },
        )
        return result
    except BaseException as exc:
        _append_audit(
            study,
            {
                "event": "run_error",
                "start_event_sha256": started["event_sha256"],
                "error": str(exc),
            },
        )
        raise


@_locked
def seal_study(study_path: Path | str) -> dict[str, Any]:
    """Freeze development source/data hashes; exclusive, append-only custody seal."""
    study, manifest = _load(study_path)
    _require_probe(study)
    seal = {
        "type": "development_lock_v4",
        "manifest_sha256": sha256(study / "custodian/manifest.json"),
        "source_hashes": _hash_tree(study / "developer/workspace"),
        "output_hashes": _hash_tree(study / "developer/output"),
        "engine_hashes": manifest["engine_hashes"],
        "development_input_hashes": manifest["developer_input_hashes"],
        "held_hashes": manifest["held_hashes"],
        "guard_source_sha256": sha256(Path(__file__)),
        "audit_head_sha256": _check_audit(study),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _exclusive(study / "custodian/DEVELOPMENT_SEALED.json", _json_bytes(seal))
    _append_audit(
        study,
        {
            "event": "development_seal",
            "seal_sha256": sha256(study / "custodian/DEVELOPMENT_SEALED.json"),
        },
    )
    return seal


@_locked
def gate_release(
    study_path: Path | str, stage: str, stage_lock_path: Path | str, timeout_seconds: float = 900
) -> dict[str, Any]:
    """Coordinator-only one-attempt held evaluation; deliberately absent from CLI.

    Lock schema: stage, development_seal_sha256, proposal_path/proposal_sha256,
    request_path, files={absolute_path: sha256}. Test also binds
    validation_lock_sha256 and selection_receipt_path/selection_receipt_sha256.
    Opaque policy/proposal files are hashed, never interpreted by this guard.
    """
    import pandas as pd

    if stage not in ("validation", "test"):
        raise GuardError("Release stage must be validation or test")
    study, manifest = _load(study_path)
    _require_probe(study)
    development_seal_path = study / "custodian/DEVELOPMENT_SEALED.json"
    if not development_seal_path.is_file():
        raise GuardError("Held release requires development seal")
    development_seal = json.loads(development_seal_path.read_text())
    if (
        development_seal["manifest_sha256"] != sha256(study / "custodian/manifest.json")
        or development_seal["engine_hashes"] != _hash_tree(study / "developer/engine")
        or development_seal["source_hashes"] != _hash_tree(study / "developer/workspace")
        or development_seal["output_hashes"] != _hash_tree(study / "developer/output")
    ):
        raise GuardError("Development seal no longer matches frozen source/output/data")
    lock_path = _canonical(stage_lock_path)
    if not lock_path.is_relative_to(study / "custodian/seals") or not lock_path.is_file():
        raise GuardError("Stage lock must be an existing custodian/seals file")
    lock = json.loads(lock_path.read_text())
    if lock.get("stage") != stage or lock.get("development_seal_sha256") != sha256(
        development_seal_path
    ):
        raise GuardError("Stage lock does not bind this development seal")
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise GuardError("Stage lock must declare exact file hashes")
    declared = {}
    for name, digest in files.items():
        path = _canonical(name)
        if str(path) != name or not any(
            path.is_relative_to(study / base) for base in ("custodian/proposal", "custodian/seals")
        ):
            raise GuardError("Declared files must be canonical custodian proposal/seals paths")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or sha256(path) != digest
        ):
            raise GuardError(f"Frozen stage file hash mismatch: {path}")
        declared[path] = digest
    proposal = _canonical(lock["proposal_path"])
    request_path = _canonical(lock["request_path"])
    if declared.get(proposal) != lock.get("proposal_sha256") or request_path not in declared:
        raise GuardError("Proposal/request missing from exact lock-declared file bindings")
    if stage == "test":
        validation_release_path = study / "custodian/STAGE_VALIDATION_COMPLETED.json"
        if not validation_release_path.is_file():
            raise GuardError("Test requires a successful completed validation evaluation")
        validation_release = json.loads(validation_release_path.read_text())
        if validation_release["lock_sha256"] != lock.get("validation_lock_sha256"):
            raise GuardError("Test does not bind the completed validation lock")
        receipt = _canonical(lock["selection_receipt_path"])
        if declared.get(receipt) != lock.get("selection_receipt_sha256"):
            raise GuardError("Test selection receipt not included in frozen bindings")
    request = json.loads(request_path.read_text())
    input_dir = study / f"released/{stage}/input"
    output_dir = study / f"operator_output/{stage}"
    results_dir = output_dir / "results"
    if (
        _canonical(request["input_dir"]) != input_dir
        or _canonical(request["output"]) != results_dir
        or request["start"] != STAGES[stage][2]
        or request["end"] != STAGES[stage][1]
        or _canonical(request["policy_path"]) not in declared
        or results_dir.exists()
    ):
        raise GuardError("Request paths/scoring dates/policy do not match held stage contract")
    script = study / "developer/engine/engine.py"
    if not script.is_file():
        raise GuardError("Only sealed generic engine.py may execute a held stage")
    # Prestart state and audit are persisted BEFORE price-file materialization.
    started = {
        "stage": stage,
        "lock_path": str(lock_path),
        "lock_sha256": sha256(lock_path),
        "request_sha256": sha256(request_path),
        "proposal_sha256": lock["proposal_sha256"],
        "development_seal_sha256": sha256(development_seal_path),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "declared_file_hashes": {str(p): v for p, v in declared.items()},
    }
    _exclusive(study / f"custodian/STAGE_{stage.upper()}_STARTED.json", _json_bytes(started))
    _append_audit(study, {"event": "held_stage_start", **started})
    try:
        input_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)
        partitions = [study / "developer/input/QQQ.csv", study / "held/validation/QQQ.csv"]
        if stage == "test":
            partitions.append(study / "held/test/QQQ.csv")
        bodies = []
        dates = []
        expected_header = ",".join(COLUMNS).encode()
        for path in partitions:
            header, body = path.read_bytes().split(b"\n", 1)
            if header != expected_header or not body.endswith(b"\n"):
                raise GuardError("Cumulative packet raw-column byte contract failed")
            bodies.append(body)
            dates.extend(
                pd.read_csv(path, usecols=["session"], dtype={"session": str}).session.tolist()
            )
        if dates != sorted(set(dates)):
            raise GuardError("Cumulative feature-history dates are duplicated or out of order")
        # Preserve raw-export numeric text bytes exactly; no parse/re-roundtrip.
        _exclusive(input_dir / "QQQ.csv", expected_header + b"\n" + b"".join(bodies))
        for name in ("DFF_available.csv", "DTB3_available.csv"):
            _exclusive(input_dir / name, (study / f"held/{stage}" / name).read_bytes())
        packet = {
            "stage": stage,
            "feature_start": STAGES["development"][0],
            "scoring_start": STAGES[stage][2],
            "end": STAGES[stage][1],
            "raw_columns": COLUMNS,
            "source_partitions": {str(p): sha256(p) for p in partitions},
            "rate_cutoff_utc": manifest["stages"][stage]["rate_cutoff_utc"],
        }
        _exclusive(input_dir / "input_manifest.json", _json_bytes(packet))
        packet_hashes = _hash_tree(input_dir)
        _append_audit(
            study, {"event": "held_packet_sealed", "stage": stage, "input_hashes": packet_hashes}
        )
        snapshot = study / "custodian/snapshots" / f"{stage}-{time.time_ns()}" / "engine.py"
        _exclusive(snapshot, script.read_bytes())
        result = _launch(
            study,
            snapshot,
            ["--request", str(request_path)],
            timeout_seconds,
            stage=stage,
            input_override=input_dir,
            output_override=output_dir,
            extra_read_files=list(declared),
        )
        _load(study)
        _assert_hash_tree(input_dir, packet_hashes)
        if any(sha256(p) != digest for p, digest in declared.items()):
            raise GuardError("Frozen proposal/policy/request changed during held evaluation")
        completion = {
            **started,
            "run_id": result["run_id"],
            "returncode": result["returncode"],
            "timed_out": result["timed_out"],
            "input_hashes": packet_hashes,
            "output_hashes": _hash_tree(output_dir),
            "profile_sha256": result["profile_sha256"],
            "script_sha256": result["script_sha256"],
        }
        _append_audit(
            study,
            {
                "event": "held_stage_exit",
                **completion,
                "input_access_events": result["input_access_audit"],
            },
        )
        if result["returncode"] != 0 or result["timed_out"] or result["output_limit_exceeded"]:
            raise GuardError(
                f"Held evaluation failed closed; single attempt consumed: {result['stderr']}"
            )
        _exclusive(
            study / f"custodian/STAGE_{stage.upper()}_COMPLETED.json", _json_bytes(completion)
        )
        return result
    except BaseException as exc:
        _append_audit(
            study,
            {
                "event": "held_stage_error",
                "stage": stage,
                "error": str(exc),
                "lock_sha256": started["lock_sha256"],
            },
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--study", default=str(DEFAULT_STUDY))
    prepare.add_argument("--trusted-engine-dir", required=True)
    prepare.add_argument(
        "--source-map",
        help="JSON file containing every required role mapped to an absolute source path; overrides QQQ_BLIND_RAW_PROJECT",
    )
    probe = sub.add_parser("probe")
    probe.add_argument("--study", required=True)
    probe.add_argument("--prior-report", required=True)
    run = sub.add_parser("run")
    run.add_argument("--study", required=True)
    run.add_argument("--script", required=True)
    run.add_argument("--timeout", type=float, default=300)
    run.add_argument("args", nargs=argparse.REMAINDER)
    seal = sub.add_parser("seal")
    seal.add_argument("--study", required=True)
    opts = parser.parse_args(argv)
    try:
        if opts.command == "prepare":
            value = prepare_study(
                opts.study,
                opts.trusted_engine_dir,
                sources=source_map_from_json(opts.source_map) if opts.source_map else None,
            )
            print(
                json.dumps(
                    {
                        "study": value["study"],
                        "prepared": True,
                        "development_input": value["developer_input_hashes"],
                    },
                    sort_keys=True,
                )
            )
        elif opts.command == "probe":
            print(json.dumps(probe_guard(opts.study, opts.prior_report), sort_keys=True))
        elif opts.command == "run":
            args = opts.args[1:] if opts.args[:1] == ["--"] else opts.args
            value = run_guarded(opts.study, opts.script, args, opts.timeout)
            sys.stdout.write(value["stdout"])
            sys.stderr.write(value["stderr"] + "\n" if value["stderr"] else "")
            return value["returncode"] if value["returncode"] >= 0 else 128 - value["returncode"]
        else:
            print(json.dumps(seal_study(opts.study), sort_keys=True))
    except (GuardError, FileExistsError) as exc:
        print(f"GUARD CLOSED: {exc}", file=sys.stderr)
        return 125
    return 0
