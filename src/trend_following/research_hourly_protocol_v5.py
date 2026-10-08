"""Fail-closed, staged v5 research protocol; never imports or overwrites v4 results.

Preparation alone may inspect a complete verified input packet. It installs
physically separate historical prefixes, and performance stages load ONLY their
own prefix. Missing raw data produces a data-not-ready receipt, not a backtest.
All artifacts are private by default; failures and partially completed attempts
are retained. Historical evaluation is retrospective, not an untouched holdout.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = WORKSPACE / "configs/qqq_daily_hourly_v5.yaml"
DEFAULT_STUDY_DIR = WORKSPACE / ".cache/qqq_daily_halfhour_v5_alpha_only"
STUDY_ID = "qqq_daily_halfhour_v5_alpha_only"
CONFIG_SCHEMA = "qqq_daily_halfhour_v5_config_v2"
SAMPLING_INTERVAL_MINUTES = 30
CONFIRMATION_HOURS = [n / 2 for n in range(14)]
CANDIDATE_COUNT = 17 * 17 * 14 * 14
PAPER_STATE_SCHEMA = "qqq_halfhour_v5_paper_state_v2"
COUNTER_UNIT = "half_hour_checks"
PRICE_PROVIDER = "Alpha Vantage"
HISTORICAL_CUTOFF = "2026-05-28"
HISTORICAL_STAGES = ("development", "validation", "historical-oos")
FAMILIES = ("SMA", "EMA", "MACD")
MA_PERIODS = [10, 20, 50, 100, 200, 250]
MACD_PARAMETERS = [[6, 13, 5], [12, 26, 9], [24, 52, 18], [48, 104, 36], [96, 208, 72]]
SPLITS = {
    "warmup": {"start": "1999-11-01", "end": "2000-12-31"},
    "development": {"start": "2001-01-01", "end": "2011-12-31"},
    "validation": {"start": "2012-01-01", "end": "2015-12-31"},
    "historical-oos": {"start": "2016-01-01", "end": HISTORICAL_CUTOFF},
}
HISTORICAL_LABEL = "retrospective_previously_examined_history_not_untouched_oos"
SOURCE_FILES = (
    "src/trend_following/research_hourly_protocol_v5.py",
    "src/trend_following/research_daily_hourly_v5.py",
    "src/trend_following/research_hourly_data_v5.py",
    "src/trend_following/research_rates.py",
    "scripts/run_qqq_daily_hourly_v5.py",
)


class ProtocolError(ValueError):
    """A frozen contract, sequence, or integrity gate failed."""


def _engine():
    return importlib.import_module("trend_following.research_daily_hourly_v5")


def _data():
    return importlib.import_module("trend_following.research_hourly_data_v5")


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Noncanonical JSON value: {type(value).__name__}")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ProtocolError(f"Required nonsymlink file absent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    path = path if path.is_absolute() else WORKSPACE / path
    if path.is_symlink():
        raise ProtocolError(f"Top-level symlink path prohibited: {path}")
    return path.resolve()


def _write_json(path: Path, payload: dict, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (exclusive and path.exists()):
        raise ProtocolError(f"Immutable artifact already exists/unsafe: {path}")
    text = (
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n"
    )
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text)
        if exclusive:
            # Hard-link installation is atomic AND never replaces an existing seal.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _seal(path: Path, payload: dict) -> dict:
    sealed = {"payload": payload, "payload_sha256": canonical_hash(payload)}
    _write_json(path, sealed)
    return sealed


def _read_seal(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"Required prior sealed artifact missing: {path}")
    try:
        sealed = json.loads(path.read_text())
        payload = sealed["payload"]
        if canonical_hash(payload) != sealed["payload_sha256"]:
            raise ProtocolError(f"Seal hash mismatch: {path}")
    except (KeyError, json.JSONDecodeError, TypeError) as error:
        raise ProtocolError(f"Invalid sealed artifact: {path}") from error
    return payload


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    if path.is_symlink():
        raise ProtocolError("Symlink configuration prohibited")
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ProtocolError("Configuration must be a mapping")
    fixed = {
        "schema": CONFIG_SCHEMA,
        "study_id": STUDY_ID,
        "security": "QQQ",
        "exposure": "unlevered_long_or_cash",
        "historical_label": HISTORICAL_LABEL,
    }
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ProtocolError(f"Frozen configuration differs: {key}")
    grid = config.get("grid", {})
    expected_grid = {
        "indicator_unit": "daily_sessions",
        "ma_families": ["SMA", "EMA"],
        "macd_family": "standalone_EMA_MACD_histogram",
        "confirmation_unit": "trading_half_hour_checks_wait_after_first_hit",
        "ma_periods": MA_PERIODS,
        "macd_parameters": MACD_PARAMETERS,
        "entry_confirmation_hours": CONFIRMATION_HOURS,
        "exit_confirmation_hours": CONFIRMATION_HOURS,
        "independent_entry_exit_rules": True,
        "entry_rule_count": 17,
        "exit_rule_count": 17,
        "entry_duration_count": 14,
        "exit_duration_count": 14,
        "entry_rule_duration_count": 238,
        "exit_rule_duration_count": 238,
        "candidate_count": CANDIDATE_COUNT,
    }
    for key, value in expected_grid.items():
        if grid.get(key) != value:
            raise ProtocolError(f"Frozen grid differs: {key}")
    if "entry_confirmations" in grid or "exit_confirmations" in grid:
        raise ProtocolError("Legacy ambiguous hour-count confirmation keys are prohibited")
    for key in ("entry_confirmation_hours", "exit_confirmation_hours"):
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in grid[key]
        ):
            raise ProtocolError(
                "Confirmation durations must be explicit numeric hours, never booleans"
            )
    for name, split in SPLITS.items():
        if config.get("segments", {}).get(name) != split:
            raise ProtocolError(f"Frozen dates differ: {name}")
    if config["segments"].get("feature_history_start") != SPLITS["warmup"]["start"]:
        raise ProtocolError("Daily feature seed must remain 1999-11-01")
    if (
        config["segments"].get("portfolio_reset")
        != "flat_cash_no_pending_order_no_streak_at_each_scored_segment"
    ):
        raise ProtocolError("Each scored segment must start flat with reset order/streak state")
    expected_signals = {
        "indicator_input": "forward_built_total_return_index",
        "daily_state_update": "once_at_official_daily_close",
        "half_hour_preview": "prior_final_daily_state_plus_current_completed_30min_close",
        "sma_ema_support": "provisional_daily_value_above_its_moving_average",
        "macd_support": "EMA_MACD_histogram_strictly_positive",
        "flat_entry": "entry_support_and_exit_support_for_1_plus_2_entry_hours_checks",
        "long_exit": "exit_support_false_for_1_plus_2_exit_hours_checks",
        "required_checks_formula": "1_plus_2_times_confirmation_hours",
        "zero_hour_duration": "first_qualifying_completed_half_hour_close",
        "equality": "support_false",
        "streak_carries_across_sessions": True,
        "reset_streaks": ["condition_failure", "fill", "segment_start"],
        "all_half_hours_eligible": True,
        "session_tail": "final_15_30_to_16_00_half_hour_is_eligible",
        "decisions": "after_each_completed_30min_close",
        "fill": "next_half_hour_bar_open_or_next_session_open",
        "no_same_bar_close_execution": True,
    }
    if config.get("signals") != expected_signals:
        raise ProtocolError("Frozen signal/preview/streak/execution contract differs")
    accounting = config.get("accounting", {})
    for key, expected in {
        "initial_cash": 1000.0,
        "commission_bps_per_order": 1.0,
        "slippage_bps_scenarios": [1.0, 3.0, 10.0, 25.0],
        "baseline_slippage_bps": 3.0,
        "tax_rate": 0.0,
        "financing_rate": 0.0,
        "extra_expense_ratio": 0.0,
        "cash_series": "DTB3",
        "cash_rate_units": "annual_decimal",
        "cash_day_count": 365.0,
        "cash_availability_column": "available_at",
        "cash_duplicate_availability": "preserve_rows_latest_observation_date_at_same_available_at",
        "corporate_actions": "raw_share_split_then_ex_date_open_dividend_DRIP_proxy",
        "dividends": "pre_open_holders_only_post_split_per_share_credit",
        "dividend_reinvestment_cost": "zero_proxy_not_payment_date_ledger",
        "terminal": "liquidate_at_final_session_close_with_same_transition_costs",
        "benchmarks": ["QQQ_buy_and_hold", "causal_DTB3_cash"],
        "matching_benchmark_start_and_terminal": True,
    }.items():
        if accounting.get(key) != expected:
            raise ProtocolError(f"Frozen accounting differs: {key}")
    selection = config.get("selection", {})
    for key, expected in {
        "primary_metric": "cash_excess_sharpe",
        "return_frequency": "daily_session_returns",
        "annualization": 252,
        "standard_deviation_ddof": 1,
        "finite_only": True,
        "tie_tolerance": 1e-10,
        "tie_break": "lexicographically_smallest_stable_candidate_id",
        "finalist_count": 9,
        "stress_scenarios_may_change_winner": False,
        "development_finalists": "one_per_entry_exit_family_cell",
        "validation_candidates": "sealed_development_finalists_only",
        "historical_oos_candidates": "sealed_validation_winner_only",
    }.items():
        if selection.get(key) != expected:
            raise ProtocolError(f"Frozen selection differs: {key}")
    cells = [f"{entry}/{exit_rule}" for entry in FAMILIES for exit_rule in FAMILIES]
    if selection.get("family_cells") != cells:
        raise ProtocolError("All nine ordered entry/exit family cells must be declared")
    contract = config.get("data_contract", {})
    for key, expected in {
        "price_provider": PRICE_PROVIDER,
        "cutoff_policy": "common_available_daily_and_30min_alpha_coverage",
        "no_other_price_provider": True,
        "price_basis": "raw_as_traded",
        "provider_subbar_minutes": 30,
        "sampling_interval_minutes": SAMPLING_INTERVAL_MINUTES,
        "timestamp_label": "start",
        "calendar": "XNYS",
        "timezone": "America/New_York",
        "bar_start_session": "2001-01-02",
        "daily_close_rtol": 0.001,
        "daily_close_atol": 0.02,
        "bar_anchor": "actual_XNYS_session_open_normally_09_30",
    }.items():
        if contract.get(key) != expected:
            raise ProtocolError(f"Frozen data contract differs: {key}")
    expected_protocol = {
        "stage_order": ["prepare", "development", "validation", "historical-oos", "monitor"],
        "hash_before_performance": [
            "configuration",
            "grid",
            "code_sources",
            "data",
            "accounting",
            "selection",
        ],
        "physical_stage_prefix_packets": True,
        "preserve_failed_attempts": True,
        "performance_requires_verified_data": True,
        "monitor_read_only": True,
        "monitor_suppress_actions_if": ["stale", "unfinished", "unverified", "absent_winner"],
    }
    if config.get("protocol") != expected_protocol:
        raise ProtocolError("Frozen stage/failure/data/monitor protocol differs")
    # All other declared semantics are hash-bound, not implicitly reinterpreted.
    canonical_hash(config)
    return config


def _candidate_table(config: dict) -> tuple[list, dict[str, Any]]:
    candidates = _engine().generate_candidates()
    table = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidates) != CANDIDATE_COUNT or len(table) != CANDIDATE_COUNT:
        raise ProtocolError(
            "Engine grid must contain exactly 56,644 unique half-hour duration candidates"
        )
    if any(not key.startswith("v5_30m__") for key in table):
        raise ProtocolError("Old hourly candidate identities cannot enter the new half-hour grid")
    rules = {canonical_hash(c.entry_rule.to_dict()) for c in candidates}
    if len(rules) != 17:
        raise ProtocolError("Engine grid must contain all 17 independent indicator rules")
    expected_rules = {
        (family, period, None, None, None)
        for family in ("SMA", "EMA")
        for period in config["grid"]["ma_periods"]
    }
    expected_rules |= {
        ("MACD", None, fast, slow, signal)
        for fast, slow, signal in config["grid"]["macd_parameters"]
    }
    expected = {
        (entry, exit_rule, en, ex)
        for entry in expected_rules
        for exit_rule in expected_rules
        for en in config["grid"]["entry_confirmation_hours"]
        for ex in config["grid"]["exit_confirmation_hours"]
    }
    actual = {
        (
            (
                c.entry_rule.family,
                c.entry_rule.period,
                c.entry_rule.fast,
                c.entry_rule.slow,
                c.entry_rule.signal,
            ),
            (
                c.exit_rule.family,
                c.exit_rule.period,
                c.exit_rule.fast,
                c.exit_rule.slow,
                c.exit_rule.signal,
            ),
            c.entry_confirmation_hours,
            c.exit_confirmation_hours,
        )
        for c in candidates
    }
    if actual != expected:
        raise ProtocolError("Engine grid membership differs from frozen independent Cartesian grid")
    if any(
        c.entry_required_checks != 1 + int(2 * c.entry_confirmation_hours)
        or c.exit_required_checks != 1 + int(2 * c.exit_confirmation_hours)
        for c in candidates
    ):
        raise ProtocolError(
            "Confirmation hours must wait after the first hit: required_checks = 1 + 2 * hours"
        )
    return candidates, table


def _source_hashes() -> dict:
    return {name: sha256_file(WORKSPACE / name) for name in SOURCE_FILES}


def _runtime() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "PyYAML", "exchange_calendars"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return {"python": platform.python_version(), "packages": versions}


def _registration_payload(config: dict, config_path: Path) -> dict:
    candidates, _ = _candidate_table(config)
    grid = [{"candidate_id": c.candidate_id, **c.to_dict()} for c in candidates]
    return {
        **_source_contract_summary(),
        "schema": "qqq_daily_halfhour_v5_registration_v2",
        "configuration": config,
        "configuration_sha256": canonical_hash(config),
        "config_file_sha256": sha256_file(config_path),
        "grid": grid,
        "grid_sha256": canonical_hash(grid),
        "candidate_count": len(grid),
        "source_hashes": _source_hashes(),
        "runtime": _runtime(),
        "accounting_sha256": canonical_hash(config["accounting"]),
        "selection_sha256": canonical_hash(config["selection"]),
        "historical_label": HISTORICAL_LABEL,
    }


def _register_or_verify(study: Path, config: dict, config_path: Path) -> dict:
    payload = _registration_payload(config, config_path)
    path = study / "registration.json"
    if path.exists():
        previous = _read_seal(path)
        if previous != payload:
            raise ProtocolError(
                "Frozen config/grid/code/runtime/accounting/selection changed; use a new preregistration, never overwrite this study"
            )
        return previous
    _seal(path, payload)
    return payload


def _verify_registration(study: Path, config: dict, config_path: Path) -> dict:
    previous = _read_seal(study / "registration.json")
    if previous != _registration_payload(config, config_path):
        raise ProtocolError("Frozen config/grid/code/runtime/accounting/selection changed")
    return previous


@contextmanager
def _attempt(study: Path, stage: str):
    study.mkdir(parents=True, exist_ok=True)
    if study.is_symlink():
        raise ProtocolError("Symlink study directory prohibited")
    attempt_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex}"
    attempt_dir = study / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    start = {
        "attempt_id": attempt_id,
        "stage": stage,
        "started_at": _utc_now(),
        "status": "started",
    }
    _write_json(attempt_dir / "attempt.json", start)
    lock_path = study / ".stage.lock"
    acquired = False
    try:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise ProtocolError(
                "Another/stale stage lock exists; no concurrent research stage allowed"
            ) from error
        with os.fdopen(descriptor, "w") as stream:
            stream.write(attempt_id + "\n")
        acquired = True
        yield attempt_dir
    except BaseException as error:
        outcome = {
            **start,
            "finished_at": _utc_now(),
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_json(attempt_dir / "failure.json", outcome)
        raise
    finally:
        if acquired:
            lock_path.unlink(missing_ok=True)
        outcome_path = attempt_dir / "failure.json"
        if not outcome_path.exists():
            outcome_path = attempt_dir / "outcome.json"
            if not outcome_path.exists():
                _write_json(
                    outcome_path, {**start, "finished_at": _utc_now(), "status": "completed"}
                )
        line = json.dumps(json.loads(outcome_path.read_text()), sort_keys=True) + "\n"
        # O_APPEND retains every failed, blocked, successful and repeated attempt.
        descriptor = os.open(
            study / "attempts.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        with os.fdopen(descriptor, "w") as stream:
            stream.write(line)


def _context(config_path: str | Path, study_dir: str | Path | None) -> tuple[Path, Path]:
    return _resolve_path(config_path), _resolve_path(study_dir or DEFAULT_STUDY_DIR)


def _read_cash(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"available_at", "annual_rate", "observation_date", "series_id"}
    if not required.issubset(frame) or frame.empty:
        raise ProtocolError(
            "Causal DTB3 CSV requires available_at, annual_rate, observation_date, series_id"
        )
    timestamps = pd.DatetimeIndex(frame.available_at)
    if timestamps.tz is None or timestamps.hasnans:
        raise ProtocolError("Cash available_at must be timezone-aware")
    frame["available_at"] = timestamps.tz_convert("UTC")
    if not timestamps.is_monotonic_increasing:
        raise ProtocolError("Cash availability timestamps must be chronologically ordered")
    dates = pd.to_datetime(frame.observation_date, errors="raise").dt.strftime("%Y-%m-%d")
    if dates.isna().any() or dates.duplicated().any():
        raise ProtocolError("Cash source must uniquely retain every dated DTB3 observation")
    if (dates > frame.available_at.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")).any():
        raise ProtocolError("Cash rate marked available before its observation date")
    frame["observation_date"] = dates
    if not frame[["available_at", "observation_date"]].equals(
        frame.sort_values(["available_at", "observation_date"])[
            ["available_at", "observation_date"]
        ]
    ):
        raise ProtocolError(
            "Equal cash availability times require chronological observation-date tie ordering"
        )
    rates = pd.to_numeric(frame.annual_rate, errors="coerce")
    if not np.isfinite(rates).all() or (rates <= -1).any() or not frame.series_id.eq("DTB3").all():
        raise ProtocolError("DTB3 rates must be finite annual decimals and greater than -1")
    frame["annual_rate"] = rates
    return frame


def _session_end_utc(session: str) -> pd.Timestamp:
    return (
        pd.Timestamp(session, tz="America/New_York")
        + pd.Timedelta(days=1)
        - pd.Timedelta(nanoseconds=1)
    )


def _verify_source_receipts(manifest: dict) -> None:
    sources = manifest.get("source_files", [])
    if not sources:
        raise ProtocolError(
            "Verified packet must retain source-file provenance, not just an unsubstantiated raw label"
        )
    for item in sources:
        if sha256_file(item["path"]) != item["sha256"]:
            raise ProtocolError(f"Source-file provenance hash changed: {item['path']}")


def _source_contract_summary() -> dict:
    return {
        "study_id": STUDY_ID,
        "price_provider": PRICE_PROVIDER,
        "historical_cutoff": HISTORICAL_CUTOFF,
        "cutoff_policy": "common_available_daily_and_30min_alpha_coverage",
        "sampling_interval_minutes": SAMPLING_INTERVAL_MINUTES,
        "confirmation_counter_unit": COUNTER_UNIT,
        "confirmation_hours_semantics": "wait_after_first_hit",
        "required_checks_formula": "1 + 2 * confirmation_hours",
    }


def _require_alpha_price_source(manifest: dict) -> None:
    """Provider identity is a source constraint, never inferred from price patterns.

    DTB3/FRED observations are a cash-yield input, not a second price provider.
    Historical and prospective monitoring prices must come from Alpha Vantage.
    """
    metadata = manifest.get("source_metadata", {})
    if metadata.get("provider") != PRICE_PROVIDER:
        raise ProtocolError(
            "Price source provider must be exactly Alpha Vantage; other price providers are prohibited"
        )
    if metadata.get("symbol") != "QQQ":
        raise ProtocolError("Only actual unlevered QQQ source data is permitted")
    daily_metadata = manifest.get("daily_source_metadata")
    if (
        not isinstance(daily_metadata, dict)
        or daily_metadata.get("provider") != PRICE_PROVIDER
        or daily_metadata.get("symbol") != "QQQ"
    ):
        raise ProtocolError(
            "Daily price source must be independently attested as exactly Alpha Vantage QQQ; no provider default is inferred"
        )
    if (
        daily_metadata.get("price_basis") != "raw_as_traded"
        or not str(daily_metadata.get("verification", "")).strip()
    ):
        raise ProtocolError(
            "Daily Alpha Vantage prices require explicit raw_as_traded basis and nonempty verification"
        )


def _require_half_hour_clock(manifest: dict) -> None:
    audit = manifest.get("audit", {})
    if audit.get("sampling_interval_minutes") != SAMPLING_INTERVAL_MINUTES:
        raise ProtocolError(
            "New half-hour study requires an explicitly declared 30-minute sampling packet; legacy 60-minute packets are prohibited"
        )
    if manifest.get("sampling_interval_minutes", 30) != 30:
        raise ProtocolError("Top-level packet cadence conflicts with the half-hour study")
    if manifest.get("source_metadata", {}).get("interval_minutes") != 30:
        raise ProtocolError(
            "Half-hour study must use actual Alpha Vantage 30-minute source observations"
        )
    if audit.get("observation_interval_minutes", 30) != 30:
        raise ProtocolError("Packet observation cadence must match its 30-minute sampling contract")


def _check_historical_cutoff(
    daily: pd.DataFrame, bars: pd.DataFrame, calendar: pd.DataFrame
) -> None:
    for name, frame in (("daily", daily), ("bars", bars), ("calendar", calendar)):
        if frame.empty or frame.session.astype(str).gt(HISTORICAL_CUTOFF).any():
            raise ProtocolError(
                f"{name} prices/calendar must not extend beyond the frozen Alpha-only cutoff {HISTORICAL_CUTOFF}"
            )


def _check_prefix(
    daily: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: pd.DataFrame,
    cash: pd.DataFrame,
    stage: str,
    config: dict,
) -> None:
    end = config["segments"][stage]["end"]
    seed = config["segments"]["feature_history_start"]
    for name, frame in (("daily", daily), ("bars", bars), ("calendar", calendar)):
        if frame.empty or not frame.session.astype(str).le(end).all():
            raise ProtocolError(f"{stage} {name} packet has missing/future rows beyond {end}")
    if str(daily.session.iloc[0]) != seed or str(calendar.session.iloc[0]) != seed:
        raise ProtocolError(
            "Stage must retain the full frozen daily feature history from 1999-11-01"
        )
    if cash.empty or cash.available_at.gt(_session_end_utc(end)).any():
        raise ProtocolError("Stage cash packet includes unavailable future observations")
    if cash.available_at.iloc[0] > bars.bar_start.iloc[0]:
        raise ProtocolError("Cash packet lacks a causal predecessor at its first bar open")
    expected = _data().build_xnys_calendar(seed, end)
    if list(calendar.session) != list(expected.session):
        raise ProtocolError("Stage calendar coverage does not reach its exact frozen endpoint")
    if daily.session.iloc[-1] != expected.session.iloc[-1]:
        raise ProtocolError("Daily stage endpoint coverage incomplete")


def prepare_study(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    study_dir: str | Path | None = None,
    input_packet: str | Path | None = None,
    cash_file: str | Path | None = None,
) -> dict:
    """Register the full grid; prepare immutable prefixes, or honestly report blocked data.

    This performs no downloads, candidate evaluation or portfolio simulation.
    The unavailable-data receipt is retained even if preparation later succeeds.
    """
    config_path, study = _context(config_path, study_dir)
    with _attempt(study, "prepare") as attempt:
        config = load_config(config_path)
        registration = _register_or_verify(study, config, config_path)
        frozen_path = study / "prepared" / "freeze.json"
        if frozen_path.exists():
            frozen = _verify_prepared(study, registration)
            result = {
                **_source_contract_summary(),
                "status": "data_ready",
                "prepared_sha256": canonical_hash(frozen),
                "historical_label": HISTORICAL_LABEL,
                "performance_run": False,
            }
            _write_json(
                attempt / "outcome.json", {**result, "stage": "prepare", "finished_at": _utc_now()}
            )
            return result
        packet = _resolve_path(input_packet or config["paths"]["input_packet"])
        cash_path = _resolve_path(cash_file or config["paths"]["cash_observations"])
        if not (packet / "manifest.json").is_file() or not cash_path.is_file():
            result = {
                **_source_contract_summary(),
                "status": "data_not_ready",
                "performance_run": False,
                "registered_candidates": registration["candidate_count"],
                "reason": "verified Alpha Vantage raw daily/30-minute historical packet and causal DTB3 observations required",
                "declared_blocker": config["data_contract"].get("blocked_reason"),
                "input_packet": str(packet),
                "cash_file": str(cash_path),
                "historical_label": HISTORICAL_LABEL,
            }
            _write_json(attempt / "readiness.json", result)
            _write_json(
                attempt / "outcome.json", {**result, "stage": "prepare", "finished_at": _utc_now()}
            )
            _write_json(study / "readiness.json", result, exclusive=False)
            return result
        header_path = packet / "manifest.json"
        sha256_file(header_path)  # Require a regular receipt before parsing source metadata.
        header = json.loads(header_path.read_text())
        _require_alpha_price_source(header)
        _require_half_hour_clock(header)
        daily, bars, calendar, manifest = _data().read_input_packet(packet)
        _require_alpha_price_source(manifest)
        _require_half_hour_clock(manifest)
        _verify_source_receipts(manifest)
        _check_historical_cutoff(daily, bars, calendar)
        declaration = manifest.get("audit", {}).get("bar_start_session")
        if declaration != config["data_contract"]["bar_start_session"]:
            raise ProtocolError(
                "Packet must explicitly declare the frozen 2001-01-02 half-hour coverage start"
            )
        cash = _read_cash(cash_path)
        final_end = config["segments"]["historical-oos"]["end"]
        if str(calendar.session.iloc[-1]) < final_end:
            raise ProtocolError(
                "Master packet lacks complete frozen historical evaluation coverage"
            )
        metadata_values = dict(manifest["source_metadata"])
        metadata_values["source_files"] = tuple(metadata_values.get("source_files", ()))
        metadata = _data().SourceMetadata(**metadata_values)
        staged = Path(tempfile.mkdtemp(prefix=".prepared-", dir=attempt))
        try:
            prefixes = {}
            for stage in HISTORICAL_STAGES:
                end = config["segments"][stage]["end"]
                prefix_daily = daily.loc[daily.session.le(end)].copy()
                prefix_bars = bars.loc[bars.session.le(end)].copy()
                prefix_calendar = calendar.loc[calendar.session.le(end)].copy()
                prefix_cash = cash.loc[cash.available_at.le(_session_end_utc(end))].copy()
                _check_prefix(
                    prefix_daily, prefix_bars, prefix_calendar, prefix_cash, stage, config
                )
                prefix_manifest = _data().prepare_input_packet(
                    staged / stage,
                    prefix_daily,
                    prefix_bars,
                    prefix_calendar,
                    metadata,
                    daily_source_metadata=manifest["daily_source_metadata"],
                    sampling_interval_minutes=SAMPLING_INTERVAL_MINUTES,
                    as_of=manifest["as_of"],
                    cash_observations=prefix_cash,
                    cash_source_file=cash_path,
                    bar_start_session=declaration,
                    close_rtol=config["data_contract"]["daily_close_rtol"],
                    close_atol=config["data_contract"]["daily_close_atol"],
                )
                prefixes[stage] = {
                    "end": end,
                    "manifest_sha256": canonical_hash(prefix_manifest),
                    "files": prefix_manifest["files"],
                }
            freeze = {
                "schema": "qqq_daily_halfhour_v5_data_freeze_v2",
                "registration_sha256": canonical_hash(registration),
                "prefixes": prefixes,
                "master_manifest_sha256": sha256_file(packet / "manifest.json"),
                "master_packet": str(packet),
                "master_files": manifest["files"],
                "cash_source": str(cash_path),
                "cash_source_sha256": sha256_file(cash_path),
                "master_source_files": manifest["source_files"],
                "historical_label": HISTORICAL_LABEL,
            }
            _seal(staged / "freeze.json", freeze)
            target = study / "prepared"
            if target.exists():
                raise ProtocolError("Prepared packets already exist; never overwrite frozen data")
            os.rename(staged, target)
        except BaseException:
            # Retain partial prefix files inside the failed attempt for audit.
            raise
        result = {
            **_source_contract_summary(),
            "status": "data_ready",
            "prepared_sha256": canonical_hash(freeze),
            "registered_candidates": CANDIDATE_COUNT,
            "performance_run": False,
            "historical_label": HISTORICAL_LABEL,
        }
        _write_json(study / "readiness.json", result, exclusive=False)
        _write_json(
            attempt / "outcome.json", {**result, "stage": "prepare", "finished_at": _utc_now()}
        )
        return result


def _verify_prepared(study: Path, registration: dict) -> dict:
    freeze = _read_seal(study / "prepared" / "freeze.json")
    if freeze.get("registration_sha256") != canonical_hash(registration):
        raise ProtocolError("Prepared data is not bound to this frozen registration")
    if sha256_file(freeze["cash_source"]) != freeze["cash_source_sha256"]:
        raise ProtocolError("Original causal cash source changed")
    if (
        sha256_file(Path(freeze["master_packet"]) / "manifest.json")
        != freeze["master_manifest_sha256"]
    ):
        raise ProtocolError("Master input manifest changed")
    for name, expected in freeze["master_files"].items():
        if sha256_file(Path(freeze["master_packet"]) / name) != expected["sha256"]:
            raise ProtocolError(f"Original master data changed: {name}")
    _verify_source_receipts({"source_files": freeze["master_source_files"]})
    for stage in HISTORICAL_STAGES:
        root = study / "prepared" / stage
        manifest = json.loads((root / "manifest.json").read_text())
        _require_alpha_price_source(manifest)
        _require_half_hour_clock(manifest)
        item = freeze["prefixes"][stage]
        if (
            canonical_hash(manifest) != item["manifest_sha256"]
            or manifest["files"] != item["files"]
        ):
            raise ProtocolError(f"{stage} prefix manifest changed")
        for name, expected in item["files"].items():
            if sha256_file(root / name) != expected["sha256"]:
                raise ProtocolError(f"{stage} prefix data changed: {name}")
    return freeze


def _load_stage(study: Path, registration: dict, config: dict, stage: str):
    freeze = _verify_prepared(study, registration)
    root = study / "prepared" / stage
    daily, bars, calendar, manifest = _data().read_input_packet(root)
    _require_alpha_price_source(manifest)
    _require_half_hour_clock(manifest)
    _check_historical_cutoff(daily, bars, calendar)
    cash = _read_cash(root / "DTB3_available.csv")
    _check_prefix(daily, bars, calendar, cash, stage, config)
    if manifest["audit"].get("bar_start_session") != config["data_contract"]["bar_start_session"]:
        raise ProtocolError("Stage half-hour coverage declaration changed")
    return daily, bars, calendar, cash, freeze


def _finite_winner(metrics: pd.DataFrame, candidates: dict, tolerance: float = 1e-10) -> Any:
    if "candidate_id" not in metrics or "cash_excess_sharpe" not in metrics:
        raise ProtocolError("Selection requires stable candidate IDs and cash_excess_sharpe")
    if metrics.candidate_id.duplicated().any() or not metrics.candidate_id.isin(candidates).all():
        raise ProtocolError("Metric candidate IDs duplicated or outside permitted sealed set")
    scores = pd.to_numeric(metrics.cash_excess_sharpe, errors="coerce")
    finite = metrics.loc[np.isfinite(scores)].copy()
    if finite.empty:
        raise ProtocolError("No finite primary score; refusing to invent a winner")
    finite["cash_excess_sharpe"] = scores[np.isfinite(scores)]
    best = finite.cash_excess_sharpe.max()
    ties = finite.loc[finite.cash_excess_sharpe.ge(best - tolerance), "candidate_id"]
    return candidates[min(ties)]


def select_family_finalists(
    metrics: pd.DataFrame, candidates: list, *, tolerance: float = 1e-10
) -> list:
    """Exactly one finite-score finalist per ordered family cell; ties use ID only."""
    table = {c.candidate_id: c for c in candidates}
    if (
        len(table) != CANDIDATE_COUNT
        or not isinstance(metrics, pd.DataFrame)
        or "candidate_id" not in metrics
        or len(metrics) != len(table)
        or set(metrics.candidate_id) != set(table)
    ):
        raise ProtocolError(
            "Development must evaluate the complete preregistered 56,644-grid exactly once"
        )
    finalists = []
    for entry in FAMILIES:
        for exit_rule in FAMILIES:
            cell = {
                key: c
                for key, c in table.items()
                if c.entry_rule.family == entry and c.exit_rule.family == exit_rule
            }
            subset = metrics.loc[metrics.candidate_id.isin(cell)]
            try:
                finalists.append(_finite_winner(subset, cell, tolerance))
            except ProtocolError as error:
                raise ProtocolError(
                    f"No valid finite finalist for {entry}/{exit_rule}: {error}"
                ) from error
    return finalists


def _engine_kwargs(
    config: dict,
    stage: str,
    cash: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    slippage_bps: float | None = None,
) -> dict:
    return {
        **config["segments"][stage],
        "cash_observations": cash,
        "calendar": calendar,
        "commission_bps": config["accounting"]["commission_bps_per_order"],
        "slippage_bps": config["accounting"]["baseline_slippage_bps"]
        if slippage_bps is None
        else slippage_bps,
        "initial_cash": config["accounting"]["initial_cash"],
    }


def _audit_metrics(metrics: pd.DataFrame, allowed: list) -> None:
    ids = [c.candidate_id for c in allowed]
    if (
        not isinstance(metrics, pd.DataFrame)
        or "candidate_id" not in metrics
        or len(metrics) != len(ids)
        or metrics.candidate_id.duplicated().any()
        or set(metrics.candidate_id) != set(ids)
    ):
        raise ProtocolError("Stage metric output did not match the exact permitted candidate set")
    if "cash_excess_sharpe" not in metrics:
        raise ProtocolError("Engine did not return the frozen primary selection metric")


def _artifact_hashes(directory: Path) -> dict:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "stage_seal.json"
    }


def _install_stage(study: Path, stage: str, result_dir: Path, summary: dict) -> dict:
    summary = {
        **summary,
        "historical_label": HISTORICAL_LABEL,
        "artifact_hashes": _artifact_hashes(result_dir),
    }
    _seal(result_dir / "stage_seal.json", summary)
    destination = study / "stages" / stage
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ProtocolError(
            "Completed stage already exists; no performance reruns/overwrite allowed"
        )
    os.rename(result_dir, destination)
    return summary


def _verify_stage(study: Path, stage: str, registration: dict, freeze: dict) -> dict:
    root = study / "stages" / stage
    summary = _read_seal(root / "stage_seal.json")
    if summary.get("registration_sha256") != canonical_hash(registration) or summary.get(
        "data_freeze_sha256"
    ) != canonical_hash(freeze):
        raise ProtocolError(f"{stage} result belongs to a different frozen study/data")
    for name, expected in summary["artifact_hashes"].items():
        if sha256_file(root / name) != expected:
            raise ProtocolError(f"Prior {stage} artifact changed: {name}")
    if _artifact_hashes(root) != summary["artifact_hashes"]:
        raise ProtocolError(f"Prior {stage} artifact set changed")
    return summary


def _base_summary(registration: dict, freeze: dict, stage: str) -> dict:
    return {
        **_source_contract_summary(),
        "stage": stage,
        "status": "completed",
        "registration_sha256": canonical_hash(registration),
        "data_freeze_sha256": canonical_hash(freeze),
        "primary_metric": "cash_excess_sharpe",
        "scored_segment": SPLITS[stage],
        "feature_history_start": "1999-11-01",
        "portfolio_initial_state": "flat_cash_no_pending_order_no_streak",
    }


def _write_candidates(path: Path, candidates: list, **extra) -> dict:
    payload = {
        "candidates": [{"candidate_id": c.candidate_id, **c.to_dict()} for c in candidates],
        **extra,
    }
    _seal(path, payload)
    return payload


def _load_candidates(path: Path, table: dict, expected_count: int) -> tuple[list, dict]:
    payload = _read_seal(path)
    values = payload.get("candidates", [])
    if len(values) != expected_count:
        raise ProtocolError("Prior sealed candidate count differs")
    candidates = []
    for value in values:
        candidate_id = value["candidate_id"]
        if candidate_id not in table or table[candidate_id].to_dict() != {
            k: v for k, v in value.items() if k != "candidate_id"
        }:
            raise ProtocolError("Sealed candidate identity/parameters are not in the frozen grid")
        candidates.append(table[candidate_id])
    if len({c.candidate_id for c in candidates}) != expected_count:
        raise ProtocolError("Duplicate sealed candidates")
    return candidates, payload


def run_development(
    config_path: str | Path = DEFAULT_CONFIG, *, study_dir: str | Path | None = None
) -> dict:
    config_path, study = _context(config_path, study_dir)
    with _attempt(study, "development") as attempt:
        config = load_config(config_path)
        registration = _verify_registration(study, config, config_path)
        freeze = _verify_prepared(study, registration)
        if (study / "stages/development").exists():
            return _verify_stage(study, "development", registration, freeze)
        if any((study / "stages" / later).exists() for later in ("validation", "historical-oos")):
            raise ProtocolError("Later-stage artifact exists without a valid development stage")
        daily, bars, calendar, cash, freeze = _load_stage(
            study, registration, config, "development"
        )
        candidates, _ = _candidate_table(config)
        result_dir = attempt / "results"
        result_dir.mkdir()
        metrics = _engine().evaluate_grid(
            daily,
            bars,
            candidates=candidates,
            **_engine_kwargs(config, "development", cash, calendar),
        )
        _audit_metrics(metrics, candidates)
        metrics.to_csv(result_dir / "metrics.csv", index=False)
        finalists = select_family_finalists(
            metrics, candidates, tolerance=config["selection"]["tie_tolerance"]
        )
        finalist_payload = _write_candidates(
            result_dir / "finalists.json",
            finalists,
            selection_sha256=registration["selection_sha256"],
            development_metrics_sha256=sha256_file(result_dir / "metrics.csv"),
        )
        summary = {
            **_base_summary(registration, freeze, "development"),
            "candidate_count": CANDIDATE_COUNT,
            "finalist_count": 9,
            "finalists_sha256": canonical_hash(finalist_payload),
            "slippage_bps": config["accounting"]["baseline_slippage_bps"],
        }
        return _install_stage(study, "development", result_dir, summary)


def run_validation(
    config_path: str | Path = DEFAULT_CONFIG, *, study_dir: str | Path | None = None
) -> dict:
    config_path, study = _context(config_path, study_dir)
    with _attempt(study, "validation") as attempt:
        config = load_config(config_path)
        registration = _verify_registration(study, config, config_path)
        freeze = _verify_prepared(study, registration)
        development = _verify_stage(study, "development", registration, freeze)
        _, table = _candidate_table(config)
        finalists, finalist_payload = _load_candidates(
            study / "stages/development/finalists.json", table, 9
        )
        if canonical_hash(finalist_payload) != development["finalists_sha256"]:
            raise ProtocolError("Finalist seal does not match development")
        if len({c.family_cell for c in finalists}) != 9:
            raise ProtocolError("Development finalists do not represent all nine family cells")
        if (study / "stages/validation").exists():
            return _verify_stage(study, "validation", registration, freeze)
        if (study / "stages/historical-oos").exists():
            raise ProtocolError("Historical evaluation cannot precede sealed validation selection")
        daily, bars, calendar, cash, freeze = _load_stage(study, registration, config, "validation")
        result_dir = attempt / "results"
        result_dir.mkdir()
        metrics = _engine().evaluate_grid(
            daily,
            bars,
            candidates=finalists,
            **_engine_kwargs(config, "validation", cash, calendar),
        )
        _audit_metrics(metrics, finalists)
        metrics.to_csv(result_dir / "metrics.csv", index=False)
        winner = _finite_winner(
            metrics, {c.candidate_id: c for c in finalists}, config["selection"]["tie_tolerance"]
        )
        winner_payload = _write_candidates(
            result_dir / "winner.json",
            [winner],
            development_finalists_sha256=canonical_hash(finalist_payload),
            validation_metrics_sha256=sha256_file(result_dir / "metrics.csv"),
            selection_sha256=registration["selection_sha256"],
            sealed_before_historical_oos=True,
        )
        summary = {
            **_base_summary(registration, freeze, "validation"),
            "candidate_count": 9,
            "winner_id": winner.candidate_id,
            "winner_sha256": canonical_hash(winner_payload),
            "slippage_bps": config["accounting"]["baseline_slippage_bps"],
        }
        return _install_stage(study, "validation", result_dir, summary)


def _write_simulation(directory: Path, result) -> None:
    directory.mkdir(parents=True)
    pd.DataFrame([result.metrics]).to_csv(directory / "metrics.csv", index=False)
    pd.concat(
        [
            result.daily_returns.rename("return"),
            result.daily_equity.rename("equity"),
            result.daily_cash_returns.rename("cash_return"),
            result.daily_exposure.rename("exposure"),
        ],
        axis=1,
    ).to_csv(directory / "daily.csv")
    result.trades.to_csv(directory / "trades.csv", index=False)
    result.diagnostics.to_csv(directory / "diagnostics.csv", index=False)


def run_historical_oos(
    config_path: str | Path = DEFAULT_CONFIG, *, study_dir: str | Path | None = None
) -> dict:
    """Evaluate ONLY the already sealed winner; stress costs cannot select a new one."""
    config_path, study = _context(config_path, study_dir)
    with _attempt(study, "historical-oos") as attempt:
        config = load_config(config_path)
        registration = _verify_registration(study, config, config_path)
        freeze = _verify_prepared(study, registration)
        _verify_stage(study, "development", registration, freeze)
        validation = _verify_stage(study, "validation", registration, freeze)
        _, table = _candidate_table(config)
        winners, winner_payload = _load_candidates(
            study / "stages/validation/winner.json", table, 1
        )
        winner = winners[0]
        if (
            not winner_payload.get("sealed_before_historical_oos")
            or canonical_hash(winner_payload) != validation["winner_sha256"]
            or winner.candidate_id != validation["winner_id"]
        ):
            raise ProtocolError("Winner was not frozen before historical evaluation")
        if (study / "stages/historical-oos").exists():
            return _verify_stage(study, "historical-oos", registration, freeze)
        daily, bars, calendar, cash, freeze = _load_stage(
            study, registration, config, "historical-oos"
        )
        result_dir = attempt / "results"
        result_dir.mkdir()
        cache = _engine().build_support_cache(daily, bars, calendar)
        rows = []
        for slip in config["accounting"]["slippage_bps_scenarios"]:
            kwargs = _engine_kwargs(config, "historical-oos", cash, calendar, slippage_bps=slip)
            result = _engine().simulate_candidate(
                daily, bars, winner, support_cache=cache, **kwargs
            )
            if result.metrics.get("candidate_id", winner.candidate_id) != winner.candidate_id:
                raise ProtocolError("Historical evaluation attempted an unsealed candidate")
            folder = result_dir / f"slippage_{float(slip):g}bps"
            _write_simulation(folder / "winner", result)
            rows.append({"policy": winner.candidate_id, "slippage_bps": slip, **result.metrics})
            benchmarks = _engine().simulate_benchmarks(daily, bars, support_cache=cache, **kwargs)
            if set(benchmarks) != {"cash", "buy_and_hold"}:
                raise ProtocolError(
                    "Both matched-start/terminal QQQ and cash benchmarks are required"
                )
            for name, benchmark in benchmarks.items():
                if not result.daily_equity.index.equals(benchmark.daily_equity.index):
                    raise ProtocolError("Winner and benchmark scored endpoints differ")
                _write_simulation(folder / name, benchmark)
                rows.append({"policy": name, "slippage_bps": slip, **benchmark.metrics})
        pd.DataFrame(rows).to_csv(result_dir / "cost_scenarios.csv", index=False)
        summary = {
            **_base_summary(registration, freeze, "historical-oos"),
            "candidate_count": 1,
            "winner_id": winner.candidate_id,
            "winner_sha256": canonical_hash(winner_payload),
            "slippage_bps_scenarios": config["accounting"]["slippage_bps_scenarios"],
            "benchmarks": ["cash", "buy_and_hold"],
            "selection_after_evaluation": False,
        }
        return _install_stage(study, "historical-oos", result_dir, summary)


def monitor_winner(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    study_dir: str | Path | None = None,
    input_packet: str | Path | None = None,
    as_of: Any = None,
    state: dict | None = None,
    live: bool = False,
) -> dict:
    """Read-only winner-policy monitor; no performance ranking, orders, or disk writes.

    A monitor-only packet contains completed prior daily history plus completed
    current bars and independently verified current corporate actions. It is not
    usable as a historical packet. Missing paper state yields conditions only.
    """
    config_path, study = _context(config_path, study_dir)
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if now.tzinfo is None:
        raise ProtocolError("Monitor as_of must be timezone-aware")
    now = now.tz_convert("UTC")
    output = {
        **_source_contract_summary(),
        "stage": "monitor",
        "read_only": True,
        "as_of": now.isoformat(),
        "action": None,
        "actionable": False,
        "research_signal_not_financial_advice": True,
    }
    try:
        config = load_config(config_path)
        registration = _verify_registration(study, config, config_path)
        freeze = _verify_prepared(study, registration)
        validation = _verify_stage(study, "validation", registration, freeze)
        _, table = _candidate_table(config)
        winners, winner_payload = _load_candidates(
            study / "stages/validation/winner.json", table, 1
        )
        if canonical_hash(winner_payload) != validation["winner_sha256"]:
            raise ProtocolError("Winner seal changed")
        winner = winners[0]
        output["winner_id"] = winner.candidate_id
        packet = _resolve_path(input_packet or config["paths"]["input_packet"])
        header_path = packet / "manifest.json"
        sha256_file(header_path)
        header = json.loads(header_path.read_text())
        _require_alpha_price_source(header)
        _require_half_hour_clock(header)
        reader = _data().read_monitor_packet if live else _data().read_input_packet
        daily, bars, calendar, manifest = reader(packet)
        _require_alpha_price_source(manifest)
        _require_half_hour_clock(manifest)
        _verify_source_receipts(manifest)
        if bars.bar_end.gt(now).any():
            output["suppressed_reason"] = "unfinished_or_future_bar"
            return output
        if live:
            metadata_values = dict(manifest["source_metadata"])
            metadata_values["source_files"] = tuple(metadata_values.get("source_files", ()))
            metadata = _data().SourceMetadata(**metadata_values)
            monitored = monitor_live_policy(
                daily,
                bars,
                calendar,
                winner,
                metadata,
                as_of=now,
                state=state,
                bar_start_session=manifest["audit"]["bar_start_session"],
                daily_source_metadata=manifest["daily_source_metadata"],
                close_rtol=config["data_contract"]["daily_close_rtol"],
                close_atol=config["data_contract"]["daily_close_atol"],
            )
            return {**output, **monitored}
        local_day = now.tz_convert("America/New_York").strftime("%Y-%m-%d")
        expected_calendar = _data().build_xnys_calendar("1999-11-01", local_day)
        expected_grid = _data().canonical_interval_grid(expected_calendar, interval_minutes=30)
        completed = expected_grid.loc[expected_grid.bar_end.le(now)]
        if completed.empty or bars.bar_end.iloc[-1] != completed.bar_end.iloc[-1]:
            output["suppressed_reason"] = "stale_or_missing_completed_bar"
            return output
        latest = bars.iloc[-1]
        output["latest_completed_bar_end"] = latest.bar_end.isoformat()
        # Complete-history packet inspection cannot invent an actual holding or
        # carry state. Use the distinct live contract for a queued policy action.
        output["suppressed_reason"] = "needs_live_packet_and_explicit_paper_state"
        return output
    except (ProtocolError, ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        output["suppressed_reason"] = "absent_winner_unverified_stale_or_unfinished_input"
        output["detail"] = str(error)
        return output


def _paper_state(state: dict, candidate, grid: pd.DataFrame, bars: pd.DataFrame) -> dict:
    required = {
        "schema",
        "sampling_interval_minutes",
        "counter_unit",
        "candidate_id",
        "position",
        "entry_count",
        "exit_count",
        "pending_target",
        "last_bar_end",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise ProtocolError(
            "Paper state requires explicit half-hour v2 schema, sampling_interval_minutes, counter_unit, candidate_id, position, entry_count, exit_count, pending_target, last_bar_end"
        )
    if (
        state["schema"] != PAPER_STATE_SCHEMA
        or state["sampling_interval_minutes"] != SAMPLING_INTERVAL_MINUTES
        or state["counter_unit"] != COUNTER_UNIT
    ):
        raise ProtocolError(
            "Legacy/ambiguous paper states cannot be reused: require half-hour-check v2 schema and 30-minute sampling"
        )
    for name in ("position", "entry_count", "exit_count"):
        if not isinstance(state[name], int) or isinstance(state[name], bool):
            raise ProtocolError(f"Paper state {name} must be an integer")
    position, entry_count, exit_count = state["position"], state["entry_count"], state["exit_count"]
    pending = state["pending_target"]
    if (
        position not in (0, 1)
        or not 0 <= entry_count <= candidate.entry_required_checks
        or not 0 <= exit_count <= candidate.exit_required_checks
    ):
        raise ProtocolError("Paper state position/count outside frozen policy bounds")
    if (
        (position == 0 and exit_count)
        or (position == 1 and entry_count)
        or (
            pending is not None
            and (
                not isinstance(pending, int)
                or isinstance(pending, bool)
                or pending not in (0, 1)
                or pending == position
            )
        )
    ):
        raise ProtocolError("Paper state has inconsistent position/streak/pending order")
    if pending is not None and (
        (pending == 1 and entry_count != candidate.entry_required_checks)
        or (pending == 0 and exit_count != candidate.exit_required_checks)
    ):
        raise ProtocolError("Pending paper order does not match a confirmed streak")
    if pending is None and (
        entry_count == candidate.entry_required_checks
        or exit_count == candidate.exit_required_checks
    ):
        raise ProtocolError("Confirmed paper streak is missing its pending target")
    if state["candidate_id"] != candidate.candidate_id:
        raise ProtocolError("Paper state belongs to a different winner policy")
    last = pd.Timestamp(state["last_bar_end"])
    if last.tzinfo is None:
        raise ProtocolError("Paper state last_bar_end must be timezone-aware")
    last = last.tz_convert("UTC")
    if last > bars.bar_end.iloc[-1] or not grid.bar_end.eq(last).any():
        raise ProtocolError("Paper state is ahead of data or not on the canonical grid")
    expected = grid.loc[grid.bar_end.gt(last) & grid.bar_end.le(bars.bar_end.iloc[-1]), "bar_end"]
    observed = bars.loc[bars.bar_end.gt(last), "bar_end"]
    if list(expected) != list(observed):
        raise ProtocolError(
            "Paper state replay has an unobserved bar gap; cannot silently skip half-hour checks"
        )
    return {
        "schema": PAPER_STATE_SCHEMA,
        "sampling_interval_minutes": SAMPLING_INTERVAL_MINUTES,
        "counter_unit": COUNTER_UNIT,
        "entry_required_checks": candidate.entry_required_checks,
        "exit_required_checks": candidate.exit_required_checks,
        "position": position,
        "entry_count": entry_count,
        "exit_count": exit_count,
        "pending_target": pending,
        "last_bar_end": last.isoformat(),
        "candidate_id": candidate.candidate_id,
        "decision_at": state.get("decision_at"),
    }


def monitor_live_policy(
    daily_completed: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: pd.DataFrame,
    candidate,
    metadata,
    *,
    as_of: Any,
    state: dict | None = None,
    bar_start_session: str | None = None,
    close_rtol: float = 0.001,
    close_atol: float = 0.02,
    daily_source_metadata: dict | None = None,
) -> dict:
    """Pure, state-aware live policy preview. No files, prices downloaded, or P&L.

    An unfinished *input* bar or a stale completed grid fails closed. Merely
    being within the next half hour is NOT a reason to suppress valid evidence
    from the latest completed half hour. No actual live holding is inferred.
    Duration hours mean waiting after the first hit: 1 + 2 * hours checks.
    """
    now = pd.Timestamp(as_of)
    if now.tzinfo is None:
        raise ProtocolError("Monitor as_of must be timezone-aware")
    now = now.tz_convert("UTC")
    output = {
        **_source_contract_summary(),
        "stage": "monitor",
        "read_only": True,
        "as_of": now.isoformat(),
        "winner_id": candidate.candidate_id,
        "action": None,
        "actionable": False,
        "research_signal_not_financial_advice": True,
        "brokerage_order_placed": False,
    }
    try:
        _require_alpha_price_source(
            {
                "source_metadata": {"provider": metadata.provider, "symbol": metadata.symbol},
                "daily_source_metadata": daily_source_metadata
                if daily_source_metadata is not None
                else daily_completed.attrs.get("source_metadata"),
            }
        )
        _data().validate_monitor_inputs(
            daily_completed,
            bars,
            calendar,
            metadata,
            as_of=now,
            bar_start_session=bar_start_session,
            close_rtol=close_rtol,
            close_atol=close_atol,
            sampling_interval_minutes=SAMPLING_INTERVAL_MINUTES,
        )
        if not bars.duration_minutes.eq(30).all():
            raise ProtocolError("Every monitor observation must be a completed 30-minute check")
        if str(daily_completed.session.iloc[0]) != "1999-11-01":
            raise ProtocolError(
                "Monitor must retain finalized daily feature seed history from 1999-11-01"
            )
        grid = _data().canonical_interval_grid(calendar, interval_minutes=30)
        completed = grid.loc[grid.bar_end.le(now)]
        if completed.empty or bars.bar_end.iloc[-1] != completed.bar_end.iloc[-1]:
            output["suppressed_reason"] = "stale_or_missing_completed_bar"
            return output
        latest = bars.iloc[-1]
        output["latest_completed_bar_end"] = latest.bar_end.isoformat()
        actions = metadata.current_session_actions
        today = now.tz_convert("America/New_York").strftime("%Y-%m-%d")

        def margins_for(row):
            history = daily_completed.loc[daily_completed.session.lt(row.session)]
            if row.session == today:
                action = actions
            else:
                records = daily_completed.loc[daily_completed.session.eq(row.session)]
                if len(records) != 1:
                    raise ProtocolError("Replay session lacks explicit corporate actions")
                action = records.iloc[0]
            margins = _engine().preview_daily_margins(
                history,
                float(row.close),
                split_coefficient=float(action["split_coefficient"]),
                dividend_amount=float(action["dividend_amount"]),
                rules=tuple(dict.fromkeys((candidate.entry_rule, candidate.exit_rule))),
            )
            if margins.attrs.get("prior_session_count", len(history)) < _engine().WARMUP_SESSIONS:
                raise ProtocolError("Monitor finalized daily warm-up history is insufficient")
            return margins

        latest_margins = margins_for(latest)
        entry_value = float(latest_margins[candidate.entry_rule.rule_id])
        exit_value = float(latest_margins[candidate.exit_rule.rule_id])
        if not np.isfinite([entry_value, exit_value]).all():
            raise ProtocolError("Monitor indicator state is not eligible/finite")
        output["entry_support"] = entry_value > 0
        output["exit_support"] = exit_value > 0
        output["check_half_hour"] = True
        output["entry_confirmation_hours"] = candidate.entry_confirmation_hours
        output["exit_confirmation_hours"] = candidate.exit_confirmation_hours
        output["entry_required_checks"] = candidate.entry_required_checks
        output["exit_required_checks"] = candidate.exit_required_checks
        if state is None:
            output["status"] = "needs_state"
            output["suppressed_reason"] = "condition_only_no_actual_or_paper_holding_assumed"
            return output
        next_state = _paper_state(state, candidate, grid, bars)
        last_processed = pd.Timestamp(next_state["last_bar_end"])
        new_bars = bars.loc[bars.bar_end.gt(last_processed)]
        if new_bars.empty:
            output.update(
                {
                    "next_state": next_state,
                    "status": "no_new_bar",
                    "suppressed_reason": "no_new_completed_half_hour_decision",
                }
            )
            return output
        last_new_decision = None
        for row in new_bars.itertuples(index=False):
            # The preceding confirmed target fills at THIS half-hour bar's open,
            # including the first bar of a later session; reset on fill.
            if next_state["pending_target"] is not None:
                next_state.update(
                    position=next_state["pending_target"],
                    pending_target=None,
                    entry_count=0,
                    exit_count=0,
                    decision_at=None,
                )
            margins = margins_for(row)
            entry_value = float(margins[candidate.entry_rule.rule_id])
            exit_value = float(margins[candidate.exit_rule.rule_id])
            if not np.isfinite([entry_value, exit_value]).all():
                raise ProtocolError("Replay half-hour indicator state is not eligible/finite")
            entry_support, exit_support = entry_value > 0, exit_value > 0
            if next_state["position"]:
                next_state["entry_count"] = 0
                next_state["exit_count"] = next_state["exit_count"] + 1 if not exit_support else 0
                if next_state["exit_count"] == candidate.exit_required_checks:
                    next_state["pending_target"] = 0
                    next_state["decision_at"] = row.bar_end.isoformat()
                    last_new_decision = row.bar_end
            else:
                next_state["exit_count"] = 0
                next_state["entry_count"] = (
                    next_state["entry_count"] + 1 if entry_support and exit_support else 0
                )
                if next_state["entry_count"] == candidate.entry_required_checks:
                    next_state["pending_target"] = 1
                    next_state["decision_at"] = row.bar_end.isoformat()
                    last_new_decision = row.bar_end
            # Every completed 30-minute interval, including 15:30-16:00, counts.
            next_state["last_bar_end"] = row.bar_end.isoformat()
        output["next_state"] = next_state
        output["status"] = "paper_state_preview"
        queued_now = (
            last_new_decision == latest.bar_end and next_state["pending_target"] is not None
        )
        if queued_now:
            output["actionable"] = True
            output["action"] = "buy" if next_state["pending_target"] else "sell"
            output["decision_at"] = latest.bar_end.isoformat()
            output["execution_rule"] = "paper_policy_next_bar_open_only_not_a_brokerage_order"
        else:
            output["suppressed_reason"] = "no_new_confirmed_half_hour_order"
        return output
    except (ProtocolError, ValueError, KeyError) as error:
        output["suppressed_reason"] = "unverified_stale_unfinished_or_invalid_paper_state"
        output["detail"] = str(error)
        return output
