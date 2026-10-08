"""Explicit quality-v3 study; never rewrites previous protocols or registries.

Preparation alone may inspect a reconciled, hash-proven Alpha input packet. It installs
physically separate historical prefixes, and performance stages load ONLY their
own prefix. Missing raw data produces a data-not-ready receipt, not a backtest.
All artifacts are private by default; failures and partially completed attempts
are retained. The alternate missing-quote policy requires a human receipt;
classified blackouts are NOT repaired prices. Historical evaluation is retrospective.
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
DEFAULT_CONFIG = WORKSPACE / "configs/qqq_daily_halfhour_v5_quality.yaml"
DEFAULT_STUDY_DIR = WORKSPACE / ".cache/qqq_daily_halfhour_v5_alpha_quality"
STUDY_ID = "qqq_daily_halfhour_v5_alpha_quality"
CONFIG_SCHEMA = "qqq_daily_halfhour_v5_quality_config_v3"
SAMPLING_INTERVAL_MINUTES = 30
CONFIRMATION_HOURS = [n / 2 for n in range(14)]
CANDIDATE_COUNT = 17 * 17 * 14 * 14
PAPER_STATE_SCHEMA = "qqq_quality_halfhour_v5_paper_state_v3"
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
    "src/trend_following/research_quality_protocol_v5.py",
    "src/trend_following/research_quality_engine_v5.py",
    "src/trend_following/research_alpha_reconciliation_v5.py",
    "src/trend_following/research_daily_hourly_v5.py",
    "src/trend_following/research_hourly_data_v5.py",
    "src/trend_following/research_rates.py",
    "scripts/run_qqq_reconciled_v5.py",
    "scripts/run_qqq_gap_flags_v5.py",
)


class ProtocolError(ValueError):
    """A frozen contract, sequence, or integrity gate failed."""


def _engine():
    return importlib.import_module("trend_following.research_quality_engine_v5")


def _data():
    return importlib.import_module("trend_following.research_alpha_reconciliation_v5")


def _calendar_data():
    return importlib.import_module("trend_following.research_hourly_data_v5")


def _core():
    return importlib.import_module("trend_following.research_daily_hourly_v5")


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
        "daily_state_update": "once_at_provider_daily_quote_availability_17_NY",
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
        "all_half_hours_eligible_except_explicit_blackout_flags": True,
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
        "corporate_actions": "calendar_open_split_dividend_credit_DRIP_first_safe_observed_open",
        "dividends": "pre_open_holders_only_post_split_per_share_credit",
        "dividend_reinvestment_cost": "zero_proxy_not_payment_date_ledger",
        "terminal": "last_safe_observed_RTH_close_then_cash_accrual_to_17",
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
        "max_finalist_count": 9,
        "all_undefined_action": "stop_no_default_winner",
        "stress_scenarios_may_change_winner": False,
        "development_finalists": "one_finite_per_entry_exit_family_cell_up_to_nine",
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
        "price_basis": "derived_as_traded_from_Alpha_adjustment_factors_NOT_native_raw_capture",
        "provider_subbar_minutes": 30,
        "sampling_interval_minutes": SAMPLING_INTERVAL_MINUTES,
        "timestamp_label": "start",
        "calendar": "XNYS",
        "timezone": "America/New_York",
        "bar_start_session": "2001-01-02",
        "unit_anchor_minimum_days_per_month": 10,
        "daily_intraday_closing_identity_required": False,
        "bar_anchor": "actual_XNYS_session_open_normally_09_30",
    }.items():
        if contract.get(key) != expected:
            raise ProtocolError(f"Frozen data contract differs: {key}")
    quality = config.get("quality_policy", {})
    if quality.get("profile") not in {"strict_tradable_v1", "observed_only_blackout_v1"}:
        raise ProtocolError(
            "Declare an explicit supported quality profile; no automatic gap fallback"
        )
    for key, value in {
        "valuation_hour_ny": 17,
        "unknown_missing_profiles": ["strict_tradable_v1", "observed_only_blackout_v1"],
        "strict_unknown_missing": "reject_before_performance",
        "observed_only_unknown_missing": "hold_no_orders_cancel_pending_reset_streaks_not_a_price_repair",
        "observed_only_requires_explicit_human_receipt": True,
        "known_halts": "exact_regulatory_metadata_hold_cancel_pending_reset_streaks",
    }.items():
        if quality.get(key) != value:
            raise ProtocolError(f"Frozen quality policy differs: {key}")
    expected_roles = {
        "role_profile": "distinct_daily_feature_and_rth_book_quotes",
        "daily_feature_quote": "Alpha_provider_daily_raw_close_next_session_only",
        "provisional_quote": "observed_reconstructed_Alpha_30min_as_traded_close",
        "fill_quote": "observed_tradable_reconstructed_Alpha_30min_open",
        "daily_valuation_quote": "Alpha_provider_daily_raw_close_available_17_NY_assumption",
        "daily_quote_is_NOCP_guaranteed": False,
        "daily_source_is_point_in_time_guaranteed": False,
        "intraday_is_native_raw_capture": False,
        "daily_intraday_closing_identity_required": False,
    }
    if config.get("quote_roles") != expected_roles:
        raise ProtocolError("Frozen independent quote-role assumptions differ")
    _diagnostics_settings(config)
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
        "schema": "qqq_daily_halfhour_v5_quality_registration_v3",
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
        "quality_policy": _quality_policy(config).to_dict(),
        "quality_policy_sha256": canonical_hash(config["quality_policy"]),
        "quote_roles_sha256": canonical_hash(config["quote_roles"]),
        "gap_authorization": _authorization(config),
        "diagnostics_policy": _diagnostics_settings(config),
        "diagnostics_policy_sha256": canonical_hash(_diagnostics_settings(config)),
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
    path = _resolve_path(config_path)
    if study_dir is None:
        declared = yaml.safe_load(path.read_text())
        study_dir = declared.get("paths", {}).get("private_output", DEFAULT_STUDY_DIR)
    return path, _resolve_path(study_dir)


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
    """Require independently attested Alpha provenance and explicit derived units."""
    derived = "derived_as_traded_from_Alpha_adjustment_factors_NOT_native_raw_capture"
    if (
        manifest.get("schema") != "qqq_alpha_30m_reconciled_packet_v1"
        or manifest.get("provider") != "AlphaVantage"
        or manifest.get("symbol") != "QQQ"
        or manifest.get("price_basis") != derived
        or manifest.get("native_raw_capture") is not False
        or manifest.get("quote_role_profile") != "distinct_daily_feature_and_rth_book_quotes"
    ):
        raise ProtocolError(
            "Quality packet must identify AlphaVantage QQQ, derived as-traded units, NOT native raw capture, and independently approved quote roles"
        )
    meta = manifest.get("source_metadata", {})
    if (
        meta.get("provider") != "AlphaVantage"
        or meta.get("symbol") != "QQQ"
        or meta.get("price_origin") != derived
    ):
        raise ProtocolError(
            "Intraday source identity/derived price origin is not independently attested"
        )
    daily = manifest.get("daily_source_metadata", {})
    if (
        daily.get("provider") != "AlphaVantage"
        or daily.get("symbol") != "QQQ"
        or daily.get("price_basis") != "raw_as_traded"
        or daily.get("price_field") != "provider_daily_close"
        or not str(daily.get("verification", "")).strip()
    ):
        raise ProtocolError(
            "Daily AlphaVantage source must independently attest native raw provider daily close; no NOCP guarantee is inferred"
        )
    proof_record = manifest.get("files", {}).get("unit_proof.json", {})
    if (
        not proof_record.get("sha256")
        or manifest.get("unit_proof_sha256") != proof_record["sha256"]
    ):
        raise ProtocolError("Source-unit proof must be hash-bound before performance")


def _require_half_hour_clock(manifest: dict) -> None:
    if (
        manifest.get("sampling_interval_minutes") != 30
        or manifest.get("observation_interval_minutes") != 30
        or manifest.get("audit", {}).get("sampling_interval_minutes") != 30
    ):
        raise ProtocolError(
            "Quality study requires explicitly frozen 30-minute observation/sampling clocks"
        )


def _quality_policy(config: dict):
    q = config["quality_policy"]
    diagnostics = _diagnostics_settings(config)
    return _engine().QualityPolicy(
        profile=q["profile"],
        valuation_hour_ny=q["valuation_hour_ny"],
        post_gap_diagnostics=diagnostics["enabled"],
        post_gap_event_limit=diagnostics["event_limit"],
    )


def _diagnostics_settings(config: dict) -> dict:
    provided = config.get("diagnostics")
    if provided is None:
        return {
            "enabled": False,
            "event_limit": 10000,
            "descriptive_only": True,
            "affects_selection": False,
        }
    expected = {
        "enabled": True,
        "event_limit": 2500000,
        "scope": "unknown_missing",
        "window": "after_gap_end_through_next_trading_session_close",
        "event_kind": "confirmed_buy_sell_signals_not_fills",
        "held_position": "last_observed_filled_position_not_pending_target",
        "persist_all_candidate_events": True,
        "persist_all_candidate_flags": True,
        "descriptive_only": True,
        "affects_selection": False,
    }
    if provided != expected:
        raise ProtocolError(
            "Gap flags must use the frozen descriptive-only next-trading-close diagnostic contract; they cannot filter or rank candidates"
        )
    if config["quality_policy"]["profile"] != "observed_only_blackout_v1":
        raise ProtocolError(
            "Post-gap diagnostic study requires the explicitly authorized observed-only profile"
        )
    return dict(provided)


def _authorization(config: dict) -> dict | None:
    """Read a human decision receipt; this tool never generates or infers consent."""
    q = config["quality_policy"]
    receipt_path = q.get("gap_authorization_receipt")
    if q["profile"] == "strict_tradable_v1":
        if receipt_path is not None:
            raise ProtocolError(
                "Strict profile cannot silently consume a missing-quote authorization"
            )
        return None
    if not receipt_path:
        raise ProtocolError(
            "observed_only_blackout_v1 requires an explicit recorded HUMAN decision; changing profile alone is not consent"
        )
    path = _resolve_path(receipt_path)
    digest = sha256_file(path)
    receipt = json.loads(path.read_text())
    if (
        receipt.get("schema") != "qqq_quality_profile_authorization_v1"
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("quality_profile") != "observed_only_blackout_v1"
        or receipt.get("explicit_user_approval") is not True
        or receipt.get("authorized_by") != "human_user"
        or receipt.get("acknowledges_unrepaired_source_gaps") is not True
        or not str(receipt.get("instruction", "")).strip()
    ):
        raise ProtocolError(
            "Gap policy receipt does not record explicit human consent to unrepaired-source no-trade blackouts"
        )
    at = pd.Timestamp(receipt.get("approved_at"))
    if pd.isna(at) or at.tzinfo is None or at.tz_convert("UTC") > pd.Timestamp.now(tz="UTC"):
        raise ProtocolError("Gap authorization approved_at must be a nonfuture aware timestamp")
    return {"path": str(path), "sha256": digest, "payload": receipt}


def _quality_counts(coverage: pd.DataFrame) -> dict:
    required = {"session", "bar_start", "bar_end", "status", "signal_eligible", "fill_eligible"}
    if coverage.empty or not required.issubset(coverage):
        raise ProtocolError(
            "Every scheduled half-hour needs explicit coverage status and eligibility metadata"
        )
    if not coverage.status.isin({"observed", "unknown_missing", "known_halt_overlap"}).all():
        raise ProtocolError("Unknown coverage classification cannot be guessed as a tradable quote")
    return {
        "unknown_missing_count": int(coverage["status"].eq("unknown_missing").sum()),
        "known_halt_overlap_count": int(coverage.status.eq("known_halt_overlap").sum()),
        "observed_count": int(coverage.status.eq("observed").sum()),
    }


def _quality_gate(config: dict, coverage: pd.DataFrame, authorization: dict | None) -> dict:
    counts = _quality_counts(coverage)
    if (
        config["quality_policy"]["profile"] == "strict_tradable_v1"
        and counts["unknown_missing_count"]
    ):
        raise ProtocolError(
            "strict_tradable_v1 rejects unknown missing-source intervals before performance; known regulatory halts are separate and allowed"
        )
    if config["quality_policy"]["profile"] == "observed_only_blackout_v1" and authorization is None:
        raise ProtocolError("Missing recorded human gap-policy authorization")
    return counts


def _check_unit_proof(proof: dict) -> None:
    if (
        proof.get("unit_mapping_proof_passed") is not True
        or proof.get("quote_relationship_reconciled_by_explicit_roles") is not True
        or proof.get("native_raw_capture") is not False
    ):
        raise ProtocolError(
            "Independent unit mapping/role proof has not passed; restored quotes must not be relabelled native raw"
        )
    anchors = proof.get("independent_price_anchor_proof", {})
    for key in ("minimum_required_anchor_days_per_month", "minimum_anchor_days_per_month"):
        count = anchors.get(key)
        if (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, float, np.integer, np.floating))
            or not np.isfinite(count)
            or float(count) != int(count)
            or count < 10
        ):
            raise ProtocolError("Unit-anchor policy counts must be finite integers of at least 10")
    if (
        anchors.get("unanchored_months") != []
        or anchors.get("daily_close_not_used_as_an_anchor") is not True
    ):
        raise ProtocolError(
            "Each provider month requires at least 10 independently audited nonclosing price anchors"
        )


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
    if cash.empty or cash.available_at.gt(_valuation_end_utc(end)).any():
        raise ProtocolError("Stage cash packet includes unavailable future observations")
    if cash.available_at.iloc[0] > bars.bar_start.iloc[0]:
        raise ProtocolError("Cash packet lacks a causal predecessor at its first bar open")
    expected = _calendar_data().build_xnys_calendar(seed, end)
    if list(calendar.session) != list(expected.session):
        raise ProtocolError("Stage calendar coverage does not reach its exact frozen endpoint")
    if daily.session.iloc[-1] != expected.session.iloc[-1]:
        raise ProtocolError("Daily stage endpoint coverage incomplete")


def prepare_quality_study(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    study_dir: str | Path | None = None,
    input_packet: str | Path | None = None,
    cash_file: str | Path | None = None,
) -> dict:
    """Freeze structural/role/proof choices; no downloads or portfolio simulation."""
    config_path, study = _context(config_path, study_dir)
    with _attempt(study, "prepare") as attempt:
        config = load_config(config_path)
        registration = _register_or_verify(study, config, config_path)
        authorization = registration["gap_authorization"]
        profile = config["quality_policy"]["profile"]
        summary = {
            **_source_contract_summary(),
            "quality_profile": profile,
            "price_origin": config["data_contract"]["price_basis"],
            "native_raw_capture": False,
            "daily_valuation_hour_ny": 17,
            "unrepaired_source_gaps_not_price_repairs": True,
            "registered_candidates": CANDIDATE_COUNT,
            "performance_run": False,
            "historical_label": HISTORICAL_LABEL,
        }
        if (study / "prepared/freeze.json").exists():
            frozen = _verify_prepared(study, registration)
            result = {**summary, "status": "data_ready", "prepared_sha256": canonical_hash(frozen)}
            _write_json(
                attempt / "outcome.json", {**result, "stage": "prepare", "finished_at": _utc_now()}
            )
            return result
        packet = _resolve_path(input_packet or config["paths"]["input_packet"])
        cash_path = _resolve_path(cash_file or config["paths"]["cash_observations"])
        if not (packet / "manifest.json").is_file() or not cash_path.is_file():
            result = {
                **summary,
                "status": "data_not_ready",
                "reason": "hash-proven reconciled Alpha packet and causal DTB3 input required",
                "input_packet": str(packet),
                "cash_file": str(cash_path),
            }
            _record_readiness(study, attempt, result)
            return result
        header = json.loads((packet / "manifest.json").read_text())
        _require_alpha_price_source(header)
        _require_half_hour_clock(header)
        daily, bars, calendar, coverage, blackouts, manifest = _data().read_reconciled_packet(
            packet
        )
        _require_alpha_price_source(manifest)
        _verify_source_receipts(manifest)
        _check_historical_cutoff(daily, bars, calendar)
        proof = json.loads((packet / "unit_proof.json").read_text())
        _check_unit_proof(proof)
        counts = _quality_counts(coverage)
        if profile == "strict_tradable_v1" and counts["unknown_missing_count"]:
            result = {
                **summary,
                **counts,
                "status": "data_not_ready",
                "reason": "strict profile rejects unresolved unknown-source gaps; known halt metadata is allowed; no profile fallback or invented prices",
                "source_unit_proof_verified": True,
                "input_packet": str(packet),
            }
            _record_readiness(study, attempt, result)
            return result
        _quality_gate(config, coverage, authorization)
        if manifest["source_metadata"].get("required_study_start_session") != "2001-01-02":
            raise ProtocolError(
                "Explicit source coverage must begin 2001-01-02 with full daily seed history"
            )
        if str(calendar.session.iloc[-1]) != HISTORICAL_CUTOFF:
            raise ProtocolError("Master quality packet must end exactly at the common Alpha cutoff")
        cash = _read_cash(cash_path)
        staged = Path(tempfile.mkdtemp(prefix=".prepared-quality-", dir=attempt))
        prefixes = {}
        for stage in HISTORICAL_STAGES:
            end = config["segments"][stage]["end"]
            d = daily.loc[daily.session.le(end)].copy()
            b = bars.loc[bars.session.le(end)].copy()
            c = calendar.loc[calendar.session.le(end)].copy()
            cov = coverage.loc[coverage.session.le(end)].copy()
            cutoff = _valuation_end_utc(str(c.session.iloc[-1]))
            bl = blackouts.loc[blackouts.start.lt(cutoff)].copy()
            if not bl.empty and bl.end.gt(cutoff).any():
                raise ProtocolError(
                    "A source blackout straddles the stage cutoff; do not silently truncate metadata"
                )
            rates = cash.loc[cash.available_at.le(cutoff)].copy()
            _check_prefix(d, b, c, rates, stage, config)
            _quality_gate(config, cov, authorization)
            meta = {**manifest["source_metadata"], "coverage_end_session": str(c.session.iloc[-1])}
            bundle = _data().ReconciledBundle(
                d, b, c, cov, bl, proof, meta, manifest["source_files"]
            )
            prefix_manifest = _data().write_reconciled_packet(
                staged / stage,
                bundle,
                cash_observations=rates,
                cash_source_file=cash_path,
                quality_profile=profile,
                gap_authorization=None if authorization is None else authorization["payload"],
            )
            _require_alpha_price_source(prefix_manifest)
            if prefix_manifest.get("historical_ready") is not True:
                raise ProtocolError(
                    "Prefix source/quality-policy gate is not ready; no performance allowed"
                )
            prefixes[stage] = {
                "end": end,
                "manifest_sha256": canonical_hash(prefix_manifest),
                "files": prefix_manifest["files"],
            }
        freeze = {
            "schema": "qqq_daily_halfhour_v5_quality_data_freeze_v3",
            "registration_sha256": canonical_hash(registration),
            "prefixes": prefixes,
            "master_manifest_sha256": sha256_file(packet / "manifest.json"),
            "master_packet": str(packet),
            "master_files": manifest["files"],
            "master_source_files": manifest["source_files"],
            "master_unit_proof_sha256": sha256_file(packet / "unit_proof.json"),
            "quality_profile": profile,
            "quality_policy_sha256": registration["quality_policy_sha256"],
            "quote_roles_sha256": registration["quote_roles_sha256"],
            "gap_authorization": authorization,
            "cash_source": str(cash_path),
            "cash_source_sha256": sha256_file(cash_path),
            "historical_label": HISTORICAL_LABEL,
        }
        _seal(staged / "freeze.json", freeze)
        if (study / "prepared").exists():
            raise ProtocolError("Immutable quality preparation already exists; never overwrite")
        os.rename(staged, study / "prepared")
        result = {
            **summary,
            **counts,
            "status": "data_ready",
            "prepared_sha256": canonical_hash(freeze),
        }
        _record_readiness(study, attempt, result)
        return result


def _record_readiness(study: Path, attempt: Path, result: dict) -> None:
    _write_json(attempt / "readiness.json", result)
    _write_json(attempt / "outcome.json", {**result, "stage": "prepare", "finished_at": _utc_now()})
    _write_json(study / "readiness.json", result, exclusive=False)


def _valuation_end_utc(session: str) -> pd.Timestamp:
    return pd.Timestamp(session + " 17:00", tz="America/New_York").tz_convert("UTC")


def _verify_prepared(study: Path, registration: dict) -> dict:
    freeze = _read_seal(study / "prepared/freeze.json")
    if (
        freeze.get("registration_sha256") != canonical_hash(registration)
        or freeze.get("quality_policy_sha256") != registration["quality_policy_sha256"]
        or freeze.get("quote_roles_sha256") != registration["quote_roles_sha256"]
        or freeze.get("gap_authorization") != registration["gap_authorization"]
    ):
        raise ProtocolError(
            "Quality data/proof/roles/authorization do not match frozen registration"
        )
    if freeze["gap_authorization"] is not None:
        auth = freeze["gap_authorization"]
        if sha256_file(auth["path"]) != auth["sha256"]:
            raise ProtocolError("Human quality-policy authorization receipt changed")
    if sha256_file(freeze["cash_source"]) != freeze["cash_source_sha256"]:
        raise ProtocolError("Cash source changed")
    master = Path(freeze["master_packet"])
    if (
        sha256_file(master / "manifest.json") != freeze["master_manifest_sha256"]
        or sha256_file(master / "unit_proof.json") != freeze["master_unit_proof_sha256"]
    ):
        raise ProtocolError("Master reconciled source/proof manifest changed")
    for name, expected in freeze["master_files"].items():
        if sha256_file(master / name) != expected["sha256"]:
            raise ProtocolError(f"Master source data/proof changed: {name}")
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
            or manifest.get("historical_ready") is not True
            or manifest.get("quality_profile") != freeze["quality_profile"]
        ):
            raise ProtocolError(f"{stage} source/profile/role prefix manifest changed or not ready")
        for name, expected in item["files"].items():
            if sha256_file(root / name) != expected["sha256"]:
                raise ProtocolError(f"{stage} prefix data/proof changed: {name}")
    return freeze


def _load_stage(study: Path, registration: dict, config: dict, stage: str):
    freeze = _verify_prepared(study, registration)
    root = study / "prepared" / stage
    daily, bars, calendar, coverage, blackouts, manifest = _data().read_reconciled_packet(root)
    _require_alpha_price_source(manifest)
    _require_half_hour_clock(manifest)
    _check_historical_cutoff(daily, bars, calendar)
    cash = _read_cash(root / "DTB3_available.csv")
    _check_prefix(daily, bars, calendar, cash, stage, config)
    if coverage.session.gt(config["segments"][stage]["end"]).any() or (
        not blackouts.empty
        and blackouts.end.gt(_valuation_end_utc(config["segments"][stage]["end"])).any()
    ):
        raise ProtocolError("Future coverage/blackout metadata cannot enter a stage")
    _quality_gate(config, coverage, registration["gap_authorization"])
    return daily, bars, calendar, coverage, blackouts, cash, freeze


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
    """Up to nine finite family finalists; undefined cells excluded, ties use ID only."""
    table = {c.candidate_id: c for c in candidates}
    if (
        len(table) != CANDIDATE_COUNT
        or not isinstance(metrics, pd.DataFrame)
        or "candidate_id" not in metrics
        or "cash_excess_sharpe" not in metrics
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
            scores = pd.to_numeric(subset.get("cash_excess_sharpe"), errors="coerce")
            if not np.isfinite(scores).any():
                continue
            finalists.append(_finite_winner(subset, cell, tolerance))
    if not finalists:
        raise ProtocolError(
            "All development family scores are undefined; no finalist or default winner can be invented"
        )
    return finalists


def _engine_kwargs(
    config: dict,
    stage: str,
    cash: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    coverage: pd.DataFrame,
    blackouts: pd.DataFrame,
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
        "coverage": coverage,
        "blackouts": blackouts,
        "quality_policy": _quality_policy(config),
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


def _persist_post_gap_diagnostics(
    directory: Path, metrics: pd.DataFrame, candidates: list, config: dict, stage: str
) -> dict | None:
    """Persist every candidate flag and every decision-to-gap association, not rankings.

    Counts are DISTINCT candidate/side/decision timestamps. Association rows may
    repeat a decision for overlapping gap windows; this is intentionally retained.
    Truncation stops sealing, but all partial artifacts remain in the attempt.
    """
    policy = _diagnostics_settings(config)
    if not policy["enabled"]:
        return None
    events = metrics.attrs.get("post_gap_signal_events")
    metadata = metrics.attrs.get("post_gap_signal_event_metadata")
    flag_columns = [
        "candidate_id",
        "post_gap_confirmed_buy_count",
        "post_gap_confirmed_sell_count",
        "post_gap_confirmed_signal_count",
        "has_post_gap_confirmed_signal",
    ]
    event_columns = ["candidate_id", "side", "decision_at", "gap_start", "gap_end", "window_end"]
    if (
        not isinstance(events, pd.DataFrame)
        or not isinstance(metadata, dict)
        or not set(flag_columns).issubset(metrics)
        or not set(event_columns).issubset(events)
    ):
        raise ProtocolError(
            "Enabled gap diagnostics must return complete engine event metadata and per-candidate flags"
        )
    flags = metrics.loc[:, flag_columns].copy()
    events = events.loc[:, event_columns].copy()
    flags.to_csv(directory / "post_gap_candidate_flags.csv", index=False)
    events.to_csv(directory / "post_gap_signal_events.csv", index=False)
    explanatory = {
        **metadata,
        "diagnostics_policy": policy,
        "descriptive_only": True,
        "used_for_filtering_ranking_or_selection": False,
        "counts_are_distinct_confirmed_decisions": True,
        "event_rows_are_decision_to_gap_associations": True,
        "window_boundary": "gap_end < decision_at <= strictly_next_NY_trading_session_market_close",
        "held_position": "last_executed_observed_position_not_a_pending_target",
        "gap_prices_repaired": False,
        "stage": stage,
    }
    _write_json(directory / "post_gap_diagnostic_metadata.json", explanatory)
    if (
        metadata.get("enabled") is not True
        or metadata.get("event_limit") != policy["event_limit"]
        or not isinstance(metadata.get("truncated"), bool)
    ):
        raise ProtocolError(
            "Post-gap diagnostic metadata does not match the frozen enabled policy/event limit"
        )
    if metadata["truncated"]:
        raise ProtocolError(
            "Post-gap diagnostic event table was truncated; refusing to seal this study, partial artifacts retained"
        )
    if metadata.get("stored_event_rows") != len(events) or metadata.get("total_event_rows") != len(
        events
    ):
        raise ProtocolError("Post-gap event row counts incomplete or inconsistent")
    allowed_ids = {candidate.candidate_id for candidate in candidates}
    if (
        flags.candidate_id.duplicated().any()
        or set(flags.candidate_id) != allowed_ids
        or not events.candidate_id.isin(allowed_ids).all()
    ):
        raise ProtocolError("Every permitted candidate needs exactly one descriptive flag row")
    for column in flag_columns[1:4]:
        count = pd.to_numeric(flags[column], errors="coerce")
        if not np.isfinite(count).all() or count.lt(0).any() or count.ne(np.floor(count)).any():
            raise ProtocolError(
                "Post-gap confirmed signal counts must be finite nonnegative integers"
            )
    if (
        not flags.post_gap_confirmed_buy_count.add(flags.post_gap_confirmed_sell_count)
        .eq(flags.post_gap_confirmed_signal_count)
        .all()
        or not flags.has_post_gap_confirmed_signal.eq(
            flags.post_gap_confirmed_signal_count.gt(0)
        ).all()
    ):
        raise ProtocolError("Post-gap descriptive flags/counts disagree")
    if not events.side.isin(["buy", "sell"]).all():
        raise ProtocolError(
            "Post-gap events must identify confirmed buy/sell signals, not execution fills"
        )
    if not events.empty:
        for column in event_columns[2:]:
            stamps = pd.DatetimeIndex(events[column])
            if stamps.tz is None or stamps.hasnans:
                raise ProtocolError("Diagnostic clocks must be timezone-aware")
            events[column] = stamps.tz_convert("UTC")
        if (
            not events.decision_at.gt(events.gap_end).all()
            or not events.decision_at.le(events.window_end).all()
            or events.decision_at.gt(_valuation_end_utc(SPLITS[stage]["end"])).any()
        ):
            raise ProtocolError(
                "Confirmed signals lie outside their declared post-gap/stage window"
            )
    distinct = events.drop_duplicates(["candidate_id", "side", "decision_at"])
    counts = distinct.groupby(["candidate_id", "side"]).size()
    for row in flags.itertuples(index=False):
        if row.post_gap_confirmed_buy_count != int(
            counts.get((row.candidate_id, "buy"), 0)
        ) or row.post_gap_confirmed_sell_count != int(counts.get((row.candidate_id, "sell"), 0)):
            raise ProtocolError(
                "Per-candidate flags do not match all distinct retained confirmed-signal events"
            )
    return {
        "event_rows": len(events),
        "flagged_candidate_count": int(flags.has_post_gap_confirmed_signal.sum()),
        "candidate_flag_rows": len(flags),
        "truncated": False,
        "diagnostics_policy_sha256": canonical_hash(policy),
        "descriptive_only": True,
    }


def _selection_view(metrics: pd.DataFrame) -> pd.DataFrame:
    """Retain unchanged IDs/scores without copying multi-million-row event attrs.

    Every candidate remains. Flags cannot enter ranking, and the original full
    metrics and events remain intact for persistence and audit.
    """
    return pd.DataFrame(
        {
            "candidate_id": metrics.candidate_id.to_numpy(copy=True),
            "cash_excess_sharpe": metrics.cash_excess_sharpe.to_numpy(copy=True),
        },
        index=metrics.index.copy(),
    )


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
        "quality_profile": registration["configuration"]["quality_policy"]["profile"],
        "daily_valuation_hour_ny": 17,
        "quote_roles_sha256": registration["quote_roles_sha256"],
        "gap_authorization_sha256": None
        if registration["gap_authorization"] is None
        else registration["gap_authorization"]["sha256"],
        "unrepaired_source_gaps_not_price_repairs": True,
        "diagnostics_policy_sha256": registration["diagnostics_policy_sha256"],
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


def run_quality_development(
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
        daily, bars, calendar, coverage, blackouts, cash, freeze = _load_stage(
            study, registration, config, "development"
        )
        candidates, _ = _candidate_table(config)
        result_dir = attempt / "results"
        result_dir.mkdir()
        metrics = _engine().evaluate_quality_grid(
            daily,
            bars,
            candidates=candidates,
            **_engine_kwargs(
                config, "development", cash, calendar, coverage=coverage, blackouts=blackouts
            ),
        )
        _audit_metrics(metrics, candidates)
        metrics.to_csv(result_dir / "metrics.csv", index=False)
        diagnostics = _persist_post_gap_diagnostics(
            result_dir, metrics, candidates, config, "development"
        )
        finalists = select_family_finalists(
            _selection_view(metrics), candidates, tolerance=config["selection"]["tie_tolerance"]
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
            "finalist_count": len(finalists),
            "finalists_sha256": canonical_hash(finalist_payload),
            "slippage_bps": config["accounting"]["baseline_slippage_bps"],
            "post_gap_diagnostics": diagnostics,
        }
        return _install_stage(study, "development", result_dir, summary)


def run_quality_validation(
    config_path: str | Path = DEFAULT_CONFIG, *, study_dir: str | Path | None = None
) -> dict:
    config_path, study = _context(config_path, study_dir)
    with _attempt(study, "validation") as attempt:
        config = load_config(config_path)
        registration = _verify_registration(study, config, config_path)
        freeze = _verify_prepared(study, registration)
        development = _verify_stage(study, "development", registration, freeze)
        _, table = _candidate_table(config)
        count = development.get("finalist_count")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 9:
            raise ProtocolError("Development must seal between one and nine finite finalists")
        finalists, finalist_payload = _load_candidates(
            study / "stages/development/finalists.json", table, count
        )
        if canonical_hash(finalist_payload) != development["finalists_sha256"]:
            raise ProtocolError("Finalist seal does not match development")
        if len({c.family_cell for c in finalists}) != count:
            raise ProtocolError("Development finalists must represent distinct finite family cells")
        if (study / "stages/validation").exists():
            return _verify_stage(study, "validation", registration, freeze)
        if (study / "stages/historical-oos").exists():
            raise ProtocolError("Historical evaluation cannot precede sealed validation selection")
        daily, bars, calendar, coverage, blackouts, cash, freeze = _load_stage(
            study, registration, config, "validation"
        )
        result_dir = attempt / "results"
        result_dir.mkdir()
        metrics = _engine().evaluate_quality_grid(
            daily,
            bars,
            candidates=finalists,
            **_engine_kwargs(
                config, "validation", cash, calendar, coverage=coverage, blackouts=blackouts
            ),
        )
        _audit_metrics(metrics, finalists)
        metrics.to_csv(result_dir / "metrics.csv", index=False)
        diagnostics = _persist_post_gap_diagnostics(
            result_dir, metrics, finalists, config, "validation"
        )
        winner = _finite_winner(
            _selection_view(metrics),
            {c.candidate_id: c for c in finalists},
            config["selection"]["tie_tolerance"],
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
            "candidate_count": len(finalists),
            "winner_id": winner.candidate_id,
            "winner_sha256": canonical_hash(winner_payload),
            "slippage_bps": config["accounting"]["baseline_slippage_bps"],
            "post_gap_diagnostics": diagnostics,
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


def run_quality_historical_oos(
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
        daily, bars, calendar, coverage, blackouts, cash, freeze = _load_stage(
            study, registration, config, "historical-oos"
        )
        result_dir = attempt / "results"
        result_dir.mkdir()
        cache = _engine().build_quality_support_cache(
            daily,
            bars,
            calendar,
            coverage=coverage,
            blackouts=blackouts,
            quality_policy=_quality_policy(config),
        )
        rows = []
        for slip in config["accounting"]["slippage_bps_scenarios"]:
            kwargs = _engine_kwargs(
                config,
                "historical-oos",
                cash,
                calendar,
                coverage=coverage,
                blackouts=blackouts,
                slippage_bps=slip,
            )
            result = _engine().simulate_quality_candidate(
                daily, bars, winner, support_cache=cache, **kwargs
            )
            if result.metrics.get("candidate_id", winner.candidate_id) != winner.candidate_id:
                raise ProtocolError("Historical evaluation attempted an unsealed candidate")
            folder = result_dir / f"slippage_{float(slip):g}bps"
            _write_simulation(folder / "winner", result)
            rows.append({"policy": winner.candidate_id, "slippage_bps": slip, **result.metrics})
            benchmarks = _engine().simulate_quality_benchmarks(
                daily, bars, support_cache=cache, **kwargs
            )
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


def monitor_quality_winner(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    study_dir: str | Path | None = None,
    input_packet: str | Path | None = None,
    as_of: Any = None,
    state: dict | None = None,
) -> dict:
    """Read-only quality-policy monitor; no P&L ranking, orders or file writes."""
    config_path, study = _context(config_path, study_dir)
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if now.tzinfo is None:
        raise ProtocolError("Monitor as_of must be timezone-aware")
    now = now.tz_convert("UTC")
    out = {
        **_source_contract_summary(),
        "stage": "monitor",
        "read_only": True,
        "as_of": now.isoformat(),
        "action": None,
        "actionable": False,
        "research_signal_not_financial_advice": True,
        "brokerage_order_placed": False,
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
        packet = _resolve_path(input_packet or config["paths"]["input_packet"])
        header = json.loads((packet / "manifest.json").read_text())
        _require_alpha_price_source(header)
        _require_half_hour_clock(header)
        daily, bars, calendar, coverage, blackouts, manifest = _data().read_reconciled_packet(
            packet
        )
        return {
            **out,
            **monitor_quality_policy(
                daily,
                bars,
                calendar,
                coverage,
                blackouts,
                winners[0],
                quality_policy=_quality_policy(config),
                as_of=now,
                state=state,
                authorization=None
                if registration["gap_authorization"] is None
                else registration["gap_authorization"]["payload"],
            ),
        }
    except (ProtocolError, ValueError, OSError, KeyError) as error:
        return {
            **out,
            "suppressed_reason": "absent_winner_unverified_stale_or_unapproved_quality_input",
            "detail": str(error),
        }


def _paper_state(state: dict, candidate, grid: pd.DataFrame, policy) -> dict:
    required = {
        "schema",
        "sampling_interval_minutes",
        "counter_unit",
        "quality_profile",
        "candidate_id",
        "position",
        "entry_count",
        "exit_count",
        "pending_target",
        "last_bar_end",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise ProtocolError(
            "Paper state needs explicit quality v3 schema/profile, 30-minute check units, winner, position, counts, pending target and last bar end"
        )
    if (
        state["schema"] != PAPER_STATE_SCHEMA
        or state["sampling_interval_minutes"] != 30
        or state["counter_unit"] != COUNTER_UNIT
        or state["quality_profile"] != policy.profile
        or state["candidate_id"] != candidate.candidate_id
    ):
        raise ProtocolError("Old/other-policy paper state cannot be guessed or migrated")
    for key in ("position", "entry_count", "exit_count"):
        if not isinstance(state[key], int) or isinstance(state[key], bool):
            raise ProtocolError("Paper position/counts must be integer check units")
    pos, en, ex, pending = (
        state["position"],
        state["entry_count"],
        state["exit_count"],
        state["pending_target"],
    )
    if (
        pos not in (0, 1)
        or not 0 <= en <= candidate.entry_required_checks
        or not 0 <= ex <= candidate.exit_required_checks
        or (pos == 0 and ex != 0)
        or (pos == 1 and en != 0)
    ):
        raise ProtocolError("Invalid paper position/check count")
    if pending is not None and (
        isinstance(pending, bool)
        or not isinstance(pending, int)
        or pending not in (0, 1)
        or pending == pos
    ):
        raise ProtocolError("Invalid pending paper target")
    if (
        (pending == 1 and en != candidate.entry_required_checks)
        or (pending == 0 and ex != candidate.exit_required_checks)
        or (
            pending is None
            and (en == candidate.entry_required_checks or ex == candidate.exit_required_checks)
        )
    ):
        raise ProtocolError("Pending paper target/check threshold mismatch")
    last = pd.Timestamp(state["last_bar_end"])
    if last.tzinfo is None or not grid.bar_end.eq(last.tz_convert("UTC")).any():
        raise ProtocolError("Paper last_bar_end must be an aware canonical half-hour endpoint")
    return {
        "schema": PAPER_STATE_SCHEMA,
        "sampling_interval_minutes": 30,
        "counter_unit": COUNTER_UNIT,
        "quality_profile": policy.profile,
        "candidate_id": candidate.candidate_id,
        "position": pos,
        "entry_count": en,
        "exit_count": ex,
        "pending_target": pending,
        "last_bar_end": last.tz_convert("UTC").isoformat(),
        "decision_at": state.get("decision_at"),
    }


def monitor_quality_policy(
    daily: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: pd.DataFrame,
    coverage: pd.DataFrame,
    blackouts: pd.DataFrame,
    candidate,
    *,
    quality_policy,
    as_of: Any,
    state: dict | None = None,
    authorization: dict | None = None,
) -> dict:
    """Pure binary-policy replay: consume only prior daily feature quotes and safe half hours.

    Unknown/known blackout starts cancel pending targets and reset both counts,
    but do not manufacture prices or liquidate existing holdings. Daily marks are
    not execution quotes. No returns, fees or brokerage holdings are inferred.
    """
    now = pd.Timestamp(as_of)
    if now.tzinfo is None:
        raise ProtocolError("Monitor as_of must be aware")
    now = now.tz_convert("UTC")
    out = {
        **_source_contract_summary(),
        "stage": "monitor",
        "read_only": True,
        "quality_profile": quality_policy.profile,
        "as_of": now.isoformat(),
        "winner_id": candidate.candidate_id,
        "action": None,
        "actionable": False,
        "brokerage_order_placed": False,
        "unrepaired_source_gaps_not_price_repairs": True,
    }
    try:
        if not bars.empty and bars.bar_end.gt(now).any():
            raise ProtocolError("Unfinished observed half hour cannot create a signal")
        local_day = now.tz_convert("America/New_York").strftime("%Y-%m-%d")
        full_cal = _calendar_data().build_xnys_calendar("1999-11-01", local_day)
        grid = _calendar_data().canonical_interval_grid(full_cal, interval_minutes=30)
        complete = grid.loc[grid.bar_end.le(now)]
        cov = coverage.loc[coverage.bar_end.le(now)].copy()
        if complete.empty or cov.empty or cov.bar_end.iloc[-1] != complete.bar_end.iloc[-1]:
            raise ProtocolError("Stale completed coverage; no current action")
        if (
            quality_policy.profile == "strict_tradable_v1"
            and cov["status"].eq("unknown_missing").any()
        ):
            raise ProtocolError("Strict quality monitor rejects unknown source gaps")
        if quality_policy.profile == "observed_only_blackout_v1" and (
            not isinstance(authorization, dict)
            or authorization.get("explicit_user_approval") is not True
            or authorization.get("quality_profile") != quality_policy.profile
        ):
            raise ProtocolError("Observed-only monitor requires recorded human authorization")
        if state is None:
            return {
                **out,
                "status": "needs_state",
                "suppressed_reason": "no_holding_or_confirmation_state_is_assumed",
            }
        next_state = _paper_state(state, candidate, grid, quality_policy)
        last = pd.Timestamp(next_state["last_bar_end"])
        if last > cov.bar_end.iloc[-1]:
            raise ProtocolError("Paper state is ahead of completed input")
        steps = cov.loc[cov.bar_end.gt(last)]
        expected = complete.loc[complete.bar_end.gt(last), "bar_end"]
        if list(steps.bar_end) != list(expected):
            raise ProtocolError("Missing coverage metadata; cannot silently skip a check")
        observed = bars.set_index("bar_start")
        pending_decision = None
        active_blackouts = blackouts.loc[
            blackouts.start.ge(last) & blackouts.start.le(now)
        ].sort_values("start")
        handled = set()

        def process_blackouts(until):
            for index, event in active_blackouts.iterrows():
                if index not in handled and event.start <= until:
                    next_state.update(
                        pending_target=None, entry_count=0, exit_count=0, decision_at=None
                    )
                    handled.add(index)

        for row in steps.itertuples(index=False):
            process_blackouts(row.bar_start)
            exists = row.bar_start in observed.index
            if next_state["pending_target"] is not None and bool(row.fill_eligible) and exists:
                next_state.update(
                    position=next_state["pending_target"],
                    pending_target=None,
                    entry_count=0,
                    exit_count=0,
                    decision_at=None,
                )
            process_blackouts(row.bar_end)
            if bool(row.signal_eligible):
                if not exists:
                    raise ProtocolError("Signal-eligible metadata has no observed price")
                raw = observed.loc[row.bar_start]
                history = daily.loc[daily.session.lt(row.session)]
                if (
                    "daily_price_available_at" in history
                    and history.daily_price_available_at.gt(row.bar_end).any()
                ):
                    raise ProtocolError("Future daily quote cannot enter provisional features")
                action = daily.loc[daily.session.eq(row.session)]
                if len(action) != 1:
                    raise ProtocolError("Current session split/dividend metadata missing")
                margins = _core().preview_daily_margins(
                    history,
                    float(raw.close),
                    split_coefficient=float(action.split_coefficient.iloc[0]),
                    dividend_amount=float(action.dividend_amount.iloc[0]),
                    rules=tuple(dict.fromkeys((candidate.entry_rule, candidate.exit_rule))),
                )
                if margins.attrs.get("prior_session_count", len(history)) < _core().WARMUP_SESSIONS:
                    raise ProtocolError("Insufficient finalized DAILY warm-up state")
                values = [
                    float(margins[candidate.entry_rule.rule_id]),
                    float(margins[candidate.exit_rule.rule_id]),
                ]
                if not np.isfinite(values).all():
                    raise ProtocolError("Nonfinite provisional indicator state")
                en, ex = values[0] > 0, values[1] > 0
                if next_state["position"]:
                    next_state["entry_count"] = 0
                    next_state["exit_count"] = next_state["exit_count"] + 1 if not ex else 0
                    if next_state["exit_count"] == candidate.exit_required_checks:
                        next_state.update(pending_target=0, decision_at=row.bar_end.isoformat())
                        pending_decision = row.bar_end
                else:
                    next_state["exit_count"] = 0
                    next_state["entry_count"] = next_state["entry_count"] + 1 if en and ex else 0
                    if next_state["entry_count"] == candidate.entry_required_checks:
                        next_state.update(pending_target=1, decision_at=row.bar_end.isoformat())
                        pending_decision = row.bar_end
            next_state["last_bar_end"] = row.bar_end.isoformat()
        process_blackouts(now)
        out["next_state"] = next_state
        latest = cov.iloc[-1]
        if (
            pending_decision == latest.bar_end
            and next_state["pending_target"] is not None
            and bool(latest.signal_eligible)
        ):
            out.update(
                action="buy" if next_state["pending_target"] else "sell",
                actionable=True,
                decision_at=latest.bar_end.isoformat(),
                check_half_hour=True,
                execution_rule="next_safe_observed_open_not_daily_quote_or_brokerage_order",
            )
        else:
            out["suppressed_reason"] = "blackout_or_no_new_confirmed_order"
        return out
    except (ProtocolError, ValueError, KeyError) as error:
        return {
            **out,
            "suppressed_reason": "stale_unfinished_unapproved_or_invalid_quality_state",
            "detail": str(error),
        }
