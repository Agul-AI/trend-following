"""Development-only common-wait diagnostic under an assumed 3% cash rate.

This is a NEW, separately sealed accounting experiment, not a rewrite of any
prior DTB3 study. It reads only the already-frozen 1999--2011 development packet.
No validation/OOS market file, outcome, master price source or rate source is
opened. Repeated daily rate entries are an adapter for the frozen accrual API's
freshness check; the fixed modeling assumption is known before the first clock
instant, not a sequence of observed or released Treasury yields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trend_following import research_alpha_reconciliation_v5 as data
from trend_following import research_daily_hourly_v5 as core
from trend_following import research_quality_engine_v5 as engine
from trend_following import research_quality_protocol_v5 as protocol
from trend_following.research_rates import rate_accrual

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = WORKSPACE / "configs/qqq_daily_halfhour_v5_gap_flags_v2.yaml"
DEFAULT_SOURCE_STUDY = WORKSPACE / ".cache/qqq_daily_halfhour_v5_alpha_gap_flags_v2"
DEFAULT_SIDECAR = WORKSPACE / ".cache/constant_cash_equal_wait_20261008"
WAIT_HOURS = tuple(n / 2 for n in range(13))
ANNUAL_CASH_RATE = 0.03
COMMISSION_BPS = 1.0
SLIPPAGE_BPS = 3.0
BATCH_SIZE = 256
STAGE = "development"
PACKET_FILES = (
    "daily.csv",
    "bars.csv",
    "calendar.csv",
    "coverage.csv",
    "blackouts.csv",
    "unit_proof.json",
)
SOURCE_FILES = (*protocol.SOURCE_FILES, "src/trend_following/research_constant_cash.py")
Error = protocol.ProtocolError


def cash_policy() -> dict[str, Any]:
    """A nominal annual simple accrual rate, NOT a 3% effective/APY promise."""
    return {
        "annual_rate": ANNUAL_CASH_RATE,
        "rate_type": "fixed_nominal_annual_research_assumption",
        "day_count": "ACT/365",
        "per_clock_interval_return": "0.03 * elapsed_calendar_seconds / (365 * 86400)",
        "wealth": "multiply_one_plus_interval_return_on_each_frozen_event_clock_interval",
        "includes_closures": True,
        "effective_annual_cash_growth": "slightly_above_3pct_due_to_interval_compounding",
        "known_at": "before_first_supplied_clock_instant",
        "source": "user_assumption_not_DTB3_or_observed_market_yield",
        "freshness_adapter": "repeat_identical_preknown_assumption_at_daily_UTC_boundaries",
        "risk_free_reference": "same_assumed_cash_clock_returns_for_all_candidates_and_BH",
    }


def constant_cash_observations(first: Any, last: Any) -> pd.DataFrame:
    """Daily repeated model entries satisfy freshness without fake measurements.

    An entry dated inside an interval does not change its rate; no future
    observations are needed to know any value. ``series_id`` is deliberately
    absent: the frozen engine reserves that optional field for actual DTB3.
    """
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    if first.tzinfo is None or last.tzinfo is None or last < first:
        raise Error("Constant cash endpoints must be ordered timezone-aware instants")
    first, last = first.tz_convert("UTC"), last.tz_convert("UTC")
    known_from = first.normalize() - pd.Timedelta(days=1)
    entries = pd.date_range(known_from, last.normalize(), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "available_at": entries,
            "annual_rate": ANNUAL_CASH_RATE,
            "rate_model": "fixed_3pct_nominal_ACT365_assumption",
            "known_from": known_from,
            "representation": "daily_repeated_preknown_assumption_not_provider_observation",
        }
    )


def candidates_for_wait(hours: float) -> list[core.Candidate]:
    if isinstance(hours, bool) or hours not in WAIT_HOURS:
        raise Error("Diagnostic wait must be one of 0, 0.5, ..., 6 trading hours")
    rules = core.indicator_rules()
    candidates = [core.Candidate(en, ex, hours, hours) for en in rules for ex in rules]
    if len(rules) != 17 or len({c.candidate_id for c in candidates}) != 289:
        raise Error("Each diagnostic wait requires exactly 17 x 17 unique indicator pairs")
    return candidates


def _grid() -> list[dict]:
    return [
        {"candidate_id": c.candidate_id, **c.to_dict()}
        for hours in WAIT_HOURS
        for c in candidates_for_wait(hours)
    ]


def _safe_root(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_symlink():
        raise Error("Symlink roots prohibited")
    return path.resolve()


def _assert_separate(sidecar: Path, source_study: Path, authorization: Path | None = None) -> None:
    protected = {source_study, _safe_root(DEFAULT_SOURCE_STUDY)}
    for source in protected:
        if sidecar == source or sidecar in source.parents or source in sidecar.parents:
            raise Error(
                "Diagnostic must use a separate tree, never original/source ancestor or descendant"
            )
        if authorization is not None and authorization.is_relative_to(source):
            raise Error("New authorization must not reside inside original/source study")
    if authorization is not None and authorization.is_relative_to(sidecar):
        raise Error("New authorization must be outside diagnostic study tree")


def _read_metadata(source_study: Path, config_path: Path) -> tuple[dict, dict, dict, dict]:
    """Read sealed provenance metadata and DEVELOPMENT bytes only.

    Do not replace with protocol._verify_prepared/_load_stage or
    data.read_reconciled_packet: they re-hash later/master market sources.
    """
    config = protocol.load_config(config_path)
    registration = protocol._verify_registration(source_study, config, config_path)
    prepared = protocol._read_seal(source_study / "prepared/freeze.json")
    if (
        prepared.get("registration_sha256") != protocol.canonical_hash(registration)
        or prepared.get("quality_policy_sha256") != registration["quality_policy_sha256"]
        or prepared.get("quote_roles_sha256") != registration["quote_roles_sha256"]
        or prepared.get("gap_authorization") != registration["gap_authorization"]
    ):
        raise Error("Prepared provenance metadata differs from frozen registration")
    root = source_study / "prepared/development"
    if root.is_symlink() or not root.is_dir() or (root / "manifest.json").is_symlink():
        raise Error("Unsafe/missing development-only packet")
    manifest = json.loads((root / "manifest.json").read_text())
    item = prepared["prefixes"][STAGE]
    if (
        protocol.canonical_hash(manifest) != item["manifest_sha256"]
        or manifest.get("files") != item["files"]
        or item.get("end") != "2011-12-31"
        or manifest.get("historical_ready") is not True
        or manifest.get("quality_profile") != prepared["quality_profile"]
    ):
        raise Error("Development-only manifest/readiness binding changed")
    protocol._require_alpha_price_source(manifest)
    protocol._require_half_hour_clock(manifest)
    if manifest.get("implementation_hashes") != data._implementation_hashes():
        raise Error("Development implementation provenance changed")
    for name in PACKET_FILES:
        if protocol.sha256_file(root / name) != item["files"][name]["sha256"]:
            raise Error(f"Development packet hash changed: {name}")
    return config, registration, prepared, manifest


def _authorization(authorization: Path) -> dict:
    if authorization.is_symlink() or not authorization.is_file():
        raise Error("Explicit nonsymlink human authorization receipt required")
    value = json.loads(authorization.read_text())
    required = {
        "schema": "trend_following_development_equal_wait_constant_cash_authorization_v1",
        "authorized_stages": ["development_equal_wait_diagnostic"],
        "explicit_user_approval": True,
        "scope": "diagnostic_not_production_freeze",
        "known_global_wait_value_pending": True,
        "authorized_by": "human_user",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise Error(f"Human development-only authorization invalid: {key}")
    for key in ("explicit_user_approval", "known_global_wait_value_pending"):
        if value.get(key) is not True:
            raise Error(f"Human authorization requires explicit boolean {key}")
    optional_contract = {
        "cash_nominal_annual_rate": ANNUAL_CASH_RATE,
        "diagnostic_wait_hours": list(WAIT_HOURS),
        "entry_wait_equals_exit_wait": True,
        "validation_authorized": False,
        "historical_oos_authorized": False,
    }
    for key, expected in optional_contract.items():
        if key in value and value[key] != expected:
            raise Error(f"Human development-only authorization differs: {key}")
    requests = value.get("user_requests")
    expected_requests = {
        "for each choice of time, how many out of 289 configs, beat BH' sharpe ratio after cost.",
        "you shall set 3% cash interest rate for all",
    }
    if (
        not isinstance(requests, list)
        or not all(isinstance(request, str) for request in requests)
        or not expected_requests.issubset(set(requests))
    ):
        raise Error("Human authorization must retain both exact user requests")
    return value


def _binding_payload(source_study: Path, config_path: Path, authorization: Path) -> dict:
    _authorization(authorization)
    config, registration, prepared, manifest = _read_metadata(source_study, config_path)
    grid = _grid()
    return {
        "schema": "trend_following_constant_cash_equal_wait_development_v1",
        "scope": "development_only_diagnostic_not_global_wait_selection_or_new_winner",
        "config_path": str(config_path),
        "source_study": str(source_study),
        "authorization": {
            "path": str(authorization),
            "sha256": protocol.sha256_file(authorization),
        },
        "registration_sha256": protocol.canonical_hash(registration),
        "prepared_metadata_sha256": protocol.canonical_hash(prepared),
        "development_manifest_sha256": protocol.canonical_hash(manifest),
        "development_files": {
            name: prepared["prefixes"][STAGE]["files"][name] for name in PACKET_FILES
        },
        "source_hashes": {name: protocol.sha256_file(WORKSPACE / name) for name in SOURCE_FILES},
        "runtime": protocol._runtime(),
        "cash": cash_policy(),
        "wait_hours": list(WAIT_HOURS),
        "candidate_count_per_wait": 289,
        "candidate_count": len(grid),
        "grid_sha256": protocol.canonical_hash(grid),
        "period": config["segments"][STAGE],
        "initial_cash": config["accounting"]["initial_cash"],
        "commission_bps_per_order": COMMISSION_BPS,
        "slippage_bps_per_order": SLIPPAGE_BPS,
        "taxes_financing_separate_fund_expenses": "disabled",
        "same_start_terminal_and_cash_reference_for_strategy_BH_cash": True,
        "quality_policy": protocol._quality_policy(config).to_dict(),
        "comparison": "finite_candidate_cash_excess_Sharpe_greater_than_BH_plus_1e-10",
        "primary_score": "daily_cash_excess_Sharpe_252_annualization_sample_sd_ddof_1",
        "tie_diagnostic": "absolute_Sharpe_difference_at_most_1e-10; strict_gt_count_reported_separately",
        "later_stages": "prohibited_no_read_no_evaluate_no_select",
    }


def seal_study(
    sidecar: str | Path = DEFAULT_SIDECAR,
    *,
    authorization: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    source_study: str | Path = DEFAULT_SOURCE_STUDY,
) -> dict:
    sidecar, authorization = _safe_root(sidecar), _safe_root(authorization)
    config_path, source_study = _safe_root(config_path), _safe_root(source_study)
    _assert_separate(sidecar, source_study, authorization)
    with protocol._attempt(sidecar, "seal"):
        payload = _binding_payload(source_study, config_path, authorization)
        target = sidecar / "freeze.json"
        if target.exists():
            if protocol._read_seal(target) != payload:
                raise Error("Sidecar already sealed differently; preserve it and use a new sidecar")
        else:
            protocol._seal(target, payload)
        return payload


def verify_study(sidecar: str | Path = DEFAULT_SIDECAR) -> dict:
    sidecar = _safe_root(sidecar)
    payload = protocol._read_seal(sidecar / "freeze.json")
    _assert_separate(
        sidecar, _safe_root(payload["source_study"]), _safe_root(payload["authorization"]["path"])
    )
    expected = _binding_payload(
        _safe_root(payload["source_study"]),
        _safe_root(payload["config_path"]),
        _safe_root(payload["authorization"]["path"]),
    )
    if payload != expected:
        raise Error("Frozen constant-cash diagnostic inputs/code/assumptions changed")
    return payload


def load_development_packet(payload: dict):
    """Own reader: parse only hash-bound development-prefix price arrays."""
    root = _safe_root(payload["source_study"]) / "prepared/development"
    if set(payload.get("development_files", {})) != set(PACKET_FILES):
        raise Error("Incomplete/unsafe development-only file bindings")
    if (
        protocol.canonical_hash(json.loads((root / "manifest.json").read_text()))
        != payload["development_manifest_sha256"]
    ):
        raise Error("Development manifest changed before numeric reads")
    for name, expected in payload["development_files"].items():
        if name not in PACKET_FILES or protocol.sha256_file(root / name) != expected["sha256"]:
            raise Error("Unsafe/changed development-only file binding")
    frames = {
        name: pd.read_csv(root / name, float_precision="round_trip", dtype={"session": str})
        for name in PACKET_FILES
        if name.endswith(".csv")
    }
    daily, bars, calendar, coverage, blackouts = (
        frames[name]
        for name in ("daily.csv", "bars.csv", "calendar.csv", "coverage.csv", "blackouts.csv")
    )
    for frame, columns in (
        (daily, ("daily_price_available_at",)),
        (bars, ("bar_start", "bar_end")),
        (calendar, ("market_open", "market_close")),
        (coverage, ("bar_start", "bar_end")),
        (blackouts, ("start", "end")),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    if "source_url" in blackouts:
        blackouts["source_url"] = blackouts["source_url"].fillna("")
    for frame in (daily, bars, calendar, coverage):
        if frame.empty or frame.session.gt("2011-12-31").any():
            raise Error("Development reader refuses empty/future sessions")
    if not blackouts.empty and blackouts.end.gt(protocol._valuation_end_utc("2011-12-31")).any():
        raise Error("Development reader refuses future blackout metadata")
    manifest = json.loads((root / "manifest.json").read_text())
    proof = json.loads((root / "unit_proof.json").read_text())
    data._validate_unit_proof(proof)  # Validates metadata only; never follows source paths.
    bindings = proof["data_bindings"]
    for key, frame, columns in (
        ("daily_session_hashes", daily, data.DAILY_BINDING_COLUMNS),
        ("bar_session_hashes", bars, data.BAR_BINDING_COLUMNS),
        ("calendar_session_hashes", calendar, data.CALENDAR_BINDING_COLUMNS),
        ("coverage_session_hashes", coverage, data.COVERAGE_BINDING_COLUMNS),
    ):
        actual = data._session_hashes(frame, columns)
        if any(bindings[key].get(session) != value for session, value in actual.items()):
            raise Error(f"Development numeric proof changed: {key}")
    if any(
        data._row_hash(row, data.BLACKOUT_BINDING_COLUMNS) not in bindings["blackout_row_hashes"]
        for _, row in blackouts.iterrows()
    ):
        raise Error("Development blackout proof changed")
    calendar.attrs["calendar_provenance"] = manifest.get("calendar_provenance", {})
    daily.attrs["source_metadata"] = manifest["daily_source_metadata"]
    bars.attrs["source_metadata"] = manifest["source_metadata"]
    bars.attrs["unit_proof"] = proof
    config = protocol.load_config(payload["config_path"])
    cash = constant_cash_observations(
        calendar.market_open.iloc[0], daily.daily_price_available_at.iloc[-1]
    )
    protocol._check_prefix(daily, bars, calendar, cash, STAGE, config)
    protocol._quality_gate(
        config,
        coverage,
        protocol._read_seal(Path(payload["source_study"]) / "registration.json")[
            "gap_authorization"
        ],
    )
    return daily, bars, calendar, coverage, blackouts, cash, config


def counts_against_bh(metrics: pd.DataFrame, benchmark_sharpe: float) -> pd.DataFrame:
    """Count EVERY pair per wait; undefined scores stay in the denominator."""
    if not np.isfinite(benchmark_sharpe):
        raise Error("Cannot count Sharpe beats against an undefined BH Sharpe")
    all_permitted = {c.candidate_id for hours in WAIT_HOURS for c in candidates_for_wait(hours)}
    if len(metrics) != 3757 or set(metrics.candidate_id) != all_permitted:
        raise Error("Counts require exactly the 3,757 authorized configurations")
    rows = []
    for hours in WAIT_HOURS:
        permitted = {c.candidate_id for c in candidates_for_wait(hours)}
        subset = metrics.loc[metrics.entry_confirmation_hours.eq(hours)].copy()
        if (
            len(subset) != 289
            or subset.candidate_id.duplicated().any()
            or set(subset.candidate_id) != permitted
            or not subset.exit_confirmation_hours.eq(hours).all()
        ):
            raise Error("Counts require every one of the 289 equal-wait candidates exactly once")
        scores = pd.to_numeric(subset.cash_excess_sharpe, errors="coerce").to_numpy(float)
        finite = np.isfinite(scores)
        beats = finite & (scores > benchmark_sharpe + 1e-10)
        equal = finite & (np.abs(scores - benchmark_sharpe) <= 1e-10)
        rows.append(
            {
                "wait_hours": hours,
                "required_consecutive_hits": 1 + int(2 * hours),
                "configuration_count": 289,
                "finite_Sharpe_count": int(finite.sum()),
                "undefined_Sharpe_count": int((~finite).sum()),
                "beat_BH_Sharpe_count": int(beats.sum()),
                "beat_BH_Sharpe_pct": float(100 * beats.sum() / 289),
                "tie_BH_Sharpe_count": int(equal.sum()),
                "strict_gt_BH_Sharpe_count": int((finite & (scores > benchmark_sharpe)).sum()),
                "below_BH_Sharpe_count": int((finite & (scores < benchmark_sharpe - 1e-10)).sum()),
                "near_tie_within_1e_10_count": int(
                    (finite & (np.abs(scores - benchmark_sharpe) <= 1e-10)).sum()
                ),
                "BH_cash_excess_Sharpe": float(benchmark_sharpe),
                "median_finite_Sharpe": float(np.median(scores[finite]))
                if finite.any()
                else np.nan,
                "best_finite_Sharpe": float(np.max(scores[finite])) if finite.any() else np.nan,
                "cash_nominal_annual_rate": ANNUAL_CASH_RATE,
                "commission_bps": COMMISSION_BPS,
                "slippage_bps": SLIPPAGE_BPS,
            }
        )
    return pd.DataFrame(rows)


def _write_curve(path: Path, result) -> None:
    pd.DataFrame(
        {
            "timestamp": result.daily_equity.index,
            "return": result.daily_returns.to_numpy(),
            "equity": result.daily_equity.to_numpy(),
            "cash_return": result.daily_cash_returns.to_numpy(),
            "exposure": result.daily_exposure.to_numpy(),
        }
    ).to_csv(path, index=False)


def evaluate_study(sidecar: str | Path = DEFAULT_SIDECAR) -> dict:
    """No ranking/selection: retain all 289 pairs at each of 13 common waits."""
    sidecar = _safe_root(sidecar)
    with protocol._attempt(sidecar, "development_equal_wait") as attempt:
        payload = verify_study(sidecar)  # Seal before any market arrays load.
        started_at = json.loads((attempt / "attempt.json").read_text())["started_at"]
        if (sidecar / "results").exists():
            raise Error("Diagnostic results already exist; use verify, never overwrite/rerun")
        daily, bars, calendar, coverage, blackouts, cash, config = load_development_packet(payload)
        policy = protocol._quality_policy(config)
        cache = engine.build_quality_support_cache(
            daily, bars, calendar, coverage=coverage, blackouts=blackouts, quality_policy=policy
        )
        selected_calendar, events, clock, terminal = engine._segment_events(
            cache, **payload["period"]
        )
        accrued = rate_accrual(clock, cash, interval_start=clock[0])
        kwargs = {
            "start": payload["period"]["start"],
            "end": payload["period"]["end"],
            "cash_observations": cash,
            "commission_bps": COMMISSION_BPS,
            "slippage_bps": SLIPPAGE_BPS,
            "initial_cash": payload["initial_cash"],
        }
        output = attempt / "results"
        output.mkdir()
        benchmark_rows = []
        benchmark_sharpe = None
        dummy = candidates_for_wait(0)[0]
        for mode in ("cash", "buy_and_hold"):
            result = engine._result(engine._run(cache, [dummy], mode=mode, record=False, **kwargs))
            benchmark_rows.append({"portfolio": mode, **result.metrics})
            _write_curve(output / f"{mode}_daily.csv", result)
            if mode == "buy_and_hold":
                benchmark_sharpe = float(result.metrics["cash_excess_sharpe"])
        pd.DataFrame(benchmark_rows).to_csv(output / "benchmark_metrics.csv", index=False)
        all_rows = []
        post_context = engine._post_gap_context(cache, selected_calendar)
        post_rows = []
        post_total_rows = 0
        for hours in WAIT_HOURS:
            candidates = candidates_for_wait(hours)
            parts = []
            for offset in range(0, len(candidates), BATCH_SIZE):
                part = engine._batch(
                    cache,
                    selected_calendar,
                    events,
                    clock,
                    terminal,
                    candidates[offset : offset + BATCH_SIZE],
                    cash,
                    COMMISSION_BPS,
                    SLIPPAGE_BPS,
                    payload["initial_cash"],
                    clock_returns=accrued,
                    post_gap_context=post_context,
                    post_gap_event_budget=cache.policy.post_gap_event_limit - len(post_rows),
                )
                if part[-1] is not None:
                    post_rows.extend(part[-1]["rows"])
                    post_total_rows += part[-1]["total_rows"]
                parts.append(part)
                all_rows.extend(part[0])
            times = parts[0][1]
            if any(not pd.DatetimeIndex(part[1]).equals(pd.DatetimeIndex(times)) for part in parts):
                raise Error("Candidate batch clocks disagree")
            np.savez_compressed(
                output / f"wait_{str(hours).replace('.', 'p')}_daily.npz",
                timestamp=np.asarray([time.isoformat() for time in times], dtype=str),
                candidate_id=np.asarray([c.candidate_id for c in candidates], dtype=str),
                equity=np.concatenate([part[2] for part in parts], axis=1),
                returns=np.concatenate([part[3] for part in parts], axis=1),
                cash_returns=parts[0][4],
                exposure=np.concatenate([part[5] for part in parts], axis=1),
            )
            print(f"Completed common wait {hours:g} h: 289 pairs", flush=True)
        metrics = pd.DataFrame(all_rows)
        if post_context is not None:
            metrics.attrs["post_gap_signal_events"] = engine._post_gap_table(post_rows)
            metrics.attrs["post_gap_signal_event_metadata"] = engine._post_gap_metadata(
                post_context, cache.policy, post_total_rows, len(post_rows), all_rows
            )
            protocol._persist_post_gap_diagnostics(
                output,
                metrics,
                [c for hours in WAIT_HOURS for c in candidates_for_wait(hours)],
                config,
                STAGE,
            )
        metrics.to_csv(output / "candidate_metrics.csv", index=False)
        counts = counts_against_bh(metrics, benchmark_sharpe)
        counts.to_csv(output / "counts_by_wait.csv", index=False)
        verify_study(sidecar)  # Recheck frozen bytes after calculation, before sealing outputs.
        files = {path.name: protocol.sha256_file(path) for path in sorted(output.iterdir())}
        result = {
            "schema": "trend_following_constant_cash_equal_wait_results_v1",
            "freeze_sha256": protocol.canonical_hash(payload),
            "waits_evaluated": len(WAIT_HOURS),
            "configurations_per_wait": 289,
            "configurations_evaluated": len(metrics),
            "stage": STAGE,
            "started_at": started_at,
            "finished_at": protocol._utc_now(),
            "period": payload["period"],
            "cash_policy_sha256": protocol.canonical_hash(payload["cash"]),
            "grid_sha256": payload["grid_sha256"],
            "commission_bps_per_order": COMMISSION_BPS,
            "slippage_bps_per_order": SLIPPAGE_BPS,
            "development_manifest_sha256": payload["development_manifest_sha256"],
            "files": files,
            "BH_cash_excess_Sharpe": benchmark_sharpe,
            "no_winner_selected": True,
            "no_later_market_files_loaded": True,
        }
        protocol._seal(output / "seal.json", result)
        output.rename(sidecar / "results")
        return result


def verify_results(sidecar: str | Path = DEFAULT_SIDECAR) -> dict:
    sidecar = _safe_root(sidecar)
    payload = verify_study(sidecar)
    result = protocol._read_seal(sidecar / "results/seal.json")
    expected_contract = {
        "schema": "trend_following_constant_cash_equal_wait_results_v1",
        "freeze_sha256": protocol.canonical_hash(payload),
        "waits_evaluated": len(WAIT_HOURS),
        "configurations_per_wait": 289,
        "configurations_evaluated": 3757,
        "stage": STAGE,
        "period": payload["period"],
        "cash_policy_sha256": protocol.canonical_hash(payload["cash"]),
        "grid_sha256": payload["grid_sha256"],
        "commission_bps_per_order": COMMISSION_BPS,
        "slippage_bps_per_order": SLIPPAGE_BPS,
        "development_manifest_sha256": payload["development_manifest_sha256"],
        "no_winner_selected": True,
        "no_later_market_files_loaded": True,
    }
    if any(result.get(key) != expected for key, expected in expected_contract.items()):
        raise Error("Result seal differs from frozen assumptions/stage/count/cost contract")
    started, finished = pd.Timestamp(result["started_at"]), pd.Timestamp(result["finished_at"])
    if started.tzinfo is None or finished.tzinfo is None or finished < started:
        raise Error("Result chronology invalid")
    required = {
        "benchmark_metrics.csv",
        "buy_and_hold_daily.csv",
        "cash_daily.csv",
        "candidate_metrics.csv",
        "counts_by_wait.csv",
        *[f"wait_{str(hours).replace('.', 'p')}_daily.npz" for hours in WAIT_HOURS],
    }
    if payload["quality_policy"]["post_gap_diagnostics"]:
        required |= {
            "post_gap_candidate_flags.csv",
            "post_gap_signal_events.csv",
            "post_gap_diagnostic_metadata.json",
        }
    actual = {path.name for path in (sidecar / "results").iterdir()} - {"seal.json"}
    if set(result.get("files", {})) != required or actual != required:
        raise Error("Unexpected/missing/untracked diagnostic result artifacts")
    if not np.isfinite(result.get("BH_cash_excess_Sharpe", np.nan)):
        raise Error("Result benchmark Sharpe undefined")
    for name, expected in result["files"].items():
        if Path(name).name != name or protocol.sha256_file(sidecar / "results" / name) != expected:
            raise Error("Derived result bytes changed")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "evaluate", "verify"))
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-study", type=Path, default=DEFAULT_SOURCE_STUDY)
    args = parser.parse_args(argv)
    if args.command == "seal":
        if args.authorization is None:
            parser.error("seal requires an explicit human authorization receipt path")
        result = seal_study(
            args.sidecar,
            authorization=args.authorization,
            config_path=args.config,
            source_study=args.source_study,
        )
    elif args.command == "evaluate":
        result = evaluate_study(args.sidecar)
    else:
        result = (
            verify_results(args.sidecar)
            if (args.sidecar / "results").exists()
            else verify_study(args.sidecar)
        )
    print(
        json.dumps(
            result, indent=2, sort_keys=True, default=protocol._json_default, allow_nan=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
