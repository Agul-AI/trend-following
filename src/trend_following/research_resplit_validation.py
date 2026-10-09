"""Separately sealed 2012–2018 validation with 2019–2026 historical OOS closed.

The same 159 development-only indicator pairs are retained at all three shared
2/3/4-hour waits. This resplit does not erase prior history exposure. It selects
no winner, does not reselect pairs, and never evaluates the future OOS period.
Original sources, config, candidate freezes and old outcomes remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trend_following import research_constant_cash as fixed
from trend_following import research_intersection_validation as original
from trend_following import research_quality_engine_v5 as engine
from trend_following import research_quality_protocol_v5 as protocol
from trend_following import research_validation_prefix as prefix
from trend_following.research_rates import rate_accrual

WORKSPACE = fixed.WORKSPACE
DEFAULT_CONFIG = fixed.DEFAULT_CONFIG
DEFAULT_SOURCE_STUDY = fixed.DEFAULT_SOURCE_STUDY
DEFAULT_DEVELOPMENT_SIDECAR = fixed.DEFAULT_SIDECAR
DEFAULT_ORIGINAL_INTERSECTION = original.DEFAULT_SIDECAR
DEFAULT_SIDECAR = WORKSPACE / ".cache/intersection_validation_2018_20261008"
WAIT_HOURS = original.WAIT_HOURS
STAGE = "validation"
PERIOD = {"start": "2012-01-01", "end": "2018-12-31"}
HISTORICAL_OOS_PERIOD = {"start": "2019-01-01", "end": "2026-05-28"}
ANNUAL_CASH_RATE = fixed.ANNUAL_CASH_RATE
COMMISSION_BPS = fixed.COMMISSION_BPS
SLIPPAGE_BPS = fixed.SLIPPAGE_BPS
PAIR_COUNT = 159
CANDIDATE_COUNT = 477
PAIR_SHA256 = "c6a7c803561fcf7073b50b8e40496a2304c57ba9b587f526124816f08f73cbe6"
SOURCE_FILES = (
    *original.SOURCE_FILES,
    "src/trend_following/research_validation_prefix.py",
    "src/trend_following/research_resplit_validation.py",
)
USER_REQUEST = "use 2012-2018 as validation period, 2019-2026 as OOS, redo the validation"
Error = protocol.ProtocolError
candidates_from_pairs = original.candidates_from_pairs
_chronology = original._chronology


def _authorization(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise Error("Explicit nonsymlink human validation-only authorization required")
    value = json.loads(path.read_text())
    required = {
        "schema": "trend_following_resplit_intersection_validation_authorization_v1",
        "authorized_stages": ["validation_only"],
        "validation_only": True,
        "explicit_user_approval": True,
        "authorized_by": "human_user",
        "user_request": USER_REQUEST,
        "historical_oos_authorized": False,
        "shared_wait_hours": list(WAIT_HOURS),
        "cash_nominal_annual_rate": ANNUAL_CASH_RATE,
        "validation_period": PERIOD,
        "historical_oos_period": HISTORICAL_OOS_PERIOD,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise Error(f"Human resplit validation authorization invalid: {key}")
    for key in ("validation_only", "explicit_user_approval"):
        if value[key] is not True:
            raise Error(f"Authorization requires an explicit boolean {key}")
    if value["historical_oos_authorized"] is not False:
        raise Error("Historical OOS remains closed")
    return value


def _safe_roots(
    sidecar: Path, source: Path, development: Path, prior: Path, authorization: Path | None = None
) -> None:
    original._safe_roots(sidecar, source, development, authorization)
    for root in {prior, fixed._safe_root(DEFAULT_ORIGINAL_INTERSECTION)}:
        if sidecar == root or sidecar in root.parents or root in sidecar.parents:
            raise Error("Resplit requires a new nonoverlapping private tree")
        if authorization is not None and authorization.is_relative_to(root):
            raise Error("Authorization cannot reside inside any preserved study")


def _original_selection(
    prior: Path, pairs: list[dict], candidates: list[dict], dev_freeze: dict, dev_result: dict
) -> dict:
    """Read only prior SELECTION metadata, not prior validation outcomes."""
    previous = protocol._read_seal(prior / "freeze.json")
    expected = {
        "schema": "trend_following_constant_cash_intersection_validation_v1",
        "pairs": pairs,
        "pairs_sha256": protocol.canonical_hash(pairs),
        "candidates": candidates,
        "candidate_set_sha256": protocol.canonical_hash(candidates),
        "configuration_count_per_wait": PAIR_COUNT,
        "configuration_count": CANDIDATE_COUNT,
        "wait_hours": list(WAIT_HOURS),
        "development_freeze_sha256": protocol.canonical_hash(dev_freeze),
        "development_result_sha256": protocol.canonical_hash(dev_result),
        "prior_validation_or_OOS_outcomes_used_for_selection": False,
        "historical_oos_authorized": False,
    }
    if any(previous.get(key) != value for key, value in expected.items()):
        raise Error("Resplit must preserve the exact original development-only candidate set")
    if previous.get("pairs_sha256") != PAIR_SHA256:
        raise Error("The originally authorized 159 development pairs changed")
    for name, digest in previous["source_hashes"].items():
        if protocol.sha256_file(WORKSPACE / name) != digest:
            raise Error(f"Preserved original selection implementation changed: {name}")
    if previous.get("runtime") != protocol._runtime():
        raise Error("Original selection runtime changed")
    _chronology(previous["candidate_set_sealed_at"])
    return previous


def _binding_payload(
    source: Path, config_path: Path, authorization: Path, development: Path, prior: Path
) -> dict:
    _authorization(authorization)
    # These verification APIs read DEVELOPMENT only. No prior validation metrics.
    dev_result = fixed.verify_results(development)
    dev_freeze = fixed.verify_study(development)
    if dev_freeze["source_study"] != str(source) or dev_freeze["config_path"] != str(config_path):
        raise Error("Development inputs must originate from the same preserved source/config")
    metrics = pd.read_csv(
        development / "results/candidate_metrics.csv", float_precision="round_trip"
    )
    counts = pd.read_csv(development / "results/counts_by_wait.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(
        counts,
        fixed.counts_against_bh(metrics, dev_result["BH_cash_excess_Sharpe"]),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-14,
    )
    pairs, selection = original.select_shared_pairs(metrics, dev_result["BH_cash_excess_Sharpe"])
    candidates = [c for hours in WAIT_HOURS for c in candidates_from_pairs(pairs, hours)]
    candidate_set = [{"candidate_id": c.candidate_id, **c.to_dict()} for c in candidates]
    if len(pairs) != PAIR_COUNT or len(candidates) != CANDIDATE_COUNT:
        raise Error("Keep exactly the previously frozen 159 pairs and 477 timing instances")
    previous = _original_selection(prior, pairs, candidate_set, dev_freeze, dev_result)
    selected_rows = (
        metrics.set_index("candidate_id", drop=False)
        .loc[[c.candidate_id for c in candidates], list(original.DEV_EXPORT_COLUMNS)]
        .to_dict("records")
    )
    metadata = prefix.preseal_metadata(source, config_path)  # Provenance JSON ONLY.
    config = protocol.load_config(config_path)
    return {
        "schema": "trend_following_resplit_intersection_validation_v1",
        "scope": "validation_only_same_development_pairs_new_time_split_no_reselection",
        "source_study": str(source),
        "config_path": str(config_path),
        "development_sidecar": str(development),
        "original_intersection_sidecar": str(prior),
        "authorization": {
            "path": str(authorization),
            "sha256": protocol.sha256_file(authorization),
        },
        "original_selection_freeze_sha256": protocol.canonical_hash(previous),
        "original_selection_sealed_at": previous["candidate_set_sealed_at"],
        "development_freeze_sha256": protocol.canonical_hash(dev_freeze),
        "development_result_sha256": protocol.canonical_hash(dev_result),
        "development_selection_files": {
            name: dev_result["files"][name]
            for name in ("candidate_metrics.csv", "counts_by_wait.csv", "benchmark_metrics.csv")
        },
        "source_metadata_binding": metadata,
        "original_manifest_sha256": metadata["original_manifest_sha256"],
        "source_hashes": {name: protocol.sha256_file(WORKSPACE / name) for name in SOURCE_FILES},
        "runtime": protocol._runtime(),
        "selection": selection,
        "pairs": pairs,
        "pairs_sha256": protocol.canonical_hash(pairs),
        "candidates": candidate_set,
        "candidate_set_sha256": protocol.canonical_hash(candidate_set),
        "development_selection_rows": selected_rows,
        "development_selection_rows_sha256": protocol.canonical_hash(selected_rows),
        "wait_hours": list(WAIT_HOURS),
        "configuration_count_per_wait": PAIR_COUNT,
        "configuration_count": CANDIDATE_COUNT,
        "stage": STAGE,
        "period": PERIOD,
        "historical_oos_period": HISTORICAL_OOS_PERIOD,
        "historical_oos_authorized": False,
        "validation_only": True,
        "initial_cash": config["accounting"]["initial_cash"],
        "cash": fixed.cash_policy(),
        "commission_bps_per_order": COMMISSION_BPS,
        "slippage_bps_per_order": SLIPPAGE_BPS,
        "taxes_financing_separate_fund_expenses": "disabled",
        "quality_policy": protocol._quality_policy(config).to_dict(),
        "same_start_terminal_and_cash_reference_for_strategy_BH_cash": True,
        "global_wait_selection": "pending_three_uniform_cohorts_not_strategy_specific_tuning",
        "primary_score": "daily_cash_excess_Sharpe_252_annualization_sample_sd_ddof_1",
        "retain_every_frozen_pair_at_every_wait_even_if_validation_Sharpe_undefined": True,
        "no_validation_selection_or_replacement": True,
        "historical_label": "retrospective_resplit_validation_prior_history_previously_examined_not_untouched",
        "prior_validation_or_OOS_outcomes_used_for_selection": False,
    }


def seal_study(
    sidecar: str | Path = DEFAULT_SIDECAR,
    *,
    authorization: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    source_study: str | Path = DEFAULT_SOURCE_STUDY,
    development_sidecar: str | Path = DEFAULT_DEVELOPMENT_SIDECAR,
    original_intersection_sidecar: str | Path = DEFAULT_ORIGINAL_INTERSECTION,
) -> dict:
    sidecar, authorization, config_path, source, development, prior = (
        fixed._safe_root(path)
        for path in (
            sidecar,
            authorization,
            config_path,
            source_study,
            development_sidecar,
            original_intersection_sidecar,
        )
    )
    _safe_roots(sidecar, source, development, prior, authorization)
    with protocol._attempt(sidecar, "seal_same_development_intersection_new_split"):
        binding = _binding_payload(source, config_path, authorization, development, prior)
        target = sidecar / "freeze.json"
        if target.exists():
            payload = protocol._read_seal(target)
            if {k: v for k, v in payload.items() if k != "candidate_set_sealed_at"} != binding:
                raise Error("Preserve differently sealed trees; never overwrite")
            _chronology(payload["candidate_set_sealed_at"])
        else:
            payload = {**binding, "candidate_set_sealed_at": protocol._utc_now()}
            protocol._seal(target, payload)
        return payload


def verify_study(sidecar: str | Path = DEFAULT_SIDECAR) -> dict:
    sidecar = fixed._safe_root(sidecar)
    payload = protocol._read_seal(sidecar / "freeze.json")
    source, config, authorization, development, prior = (
        fixed._safe_root(path)
        for path in (
            payload["source_study"],
            payload["config_path"],
            payload["authorization"]["path"],
            payload["development_sidecar"],
            payload["original_intersection_sidecar"],
        )
    )
    _safe_roots(sidecar, source, development, prior, authorization)
    expected = _binding_payload(source, config, authorization, development, prior)
    if {k: v for k, v in payload.items() if k != "candidate_set_sealed_at"} != expected:
        raise Error("Resplit candidate/code/runtime/provenance contract changed")
    _chronology(payload["original_selection_sealed_at"], payload["candidate_set_sealed_at"])
    return payload


def prepare_study(sidecar: str | Path = DEFAULT_SIDECAR) -> dict:
    """First permitted numeric price access occurs only AFTER selection freeze."""
    sidecar = fixed._safe_root(sidecar)
    with protocol._attempt(sidecar, "prepare_bounded_1999_2018_prefix") as attempt:
        payload = verify_study(sidecar)
        if (sidecar / "prepared").exists():
            return verify_prepared(sidecar)
        output = attempt / "prepared"
        manifest = prefix.prepare_validation_prefix(
            Path(payload["source_study"]),
            output,
            selection_seal=sidecar / "freeze.json",
            expected_selection_sha256=protocol.canonical_hash(payload),
            end_session=PERIOD["end"],
        )
        prepared = {
            "schema": "trend_following_resplit_validation_prepared_v1",
            "selection_sha256": protocol.canonical_hash(payload),
            "candidate_set_sha256": payload["candidate_set_sha256"],
            "manifest_sha256": protocol.canonical_hash(manifest),
            "period": PERIOD,
            "feature_history_start": "1999-11-01",
            "prepared_at": manifest["prepared_at"],
            "historical_oos_authorized": False,
        }
        _chronology(payload["candidate_set_sealed_at"], prepared["prepared_at"])
        protocol._seal(output / "freeze.json", prepared)
        verify_study(sidecar)
        output.rename(sidecar / "prepared")
        return prepared


def verify_prepared(sidecar: str | Path = DEFAULT_SIDECAR) -> dict:
    sidecar = fixed._safe_root(sidecar)
    payload = verify_study(sidecar)
    prepared = protocol._read_seal(sidecar / "prepared/freeze.json")
    expected = {
        "schema": "trend_following_resplit_validation_prepared_v1",
        "selection_sha256": protocol.canonical_hash(payload),
        "candidate_set_sha256": payload["candidate_set_sha256"],
        "period": PERIOD,
        "feature_history_start": "1999-11-01",
        "historical_oos_authorized": False,
    }
    if any(prepared.get(k) != v for k, v in expected.items()):
        raise Error("Prepared resplit prefix selection/period binding changed")
    manifest = prefix.verify_validation_prefix(
        sidecar / "prepared", expected_manifest_sha256=prepared["manifest_sha256"]
    )
    if manifest.get("prepared_at") != prepared["prepared_at"] or manifest.get(
        "selection_sha256"
    ) != protocol.canonical_hash(payload):
        raise Error("Prepared prefix timestamp/selection provenance changed")
    _chronology(payload["candidate_set_sealed_at"], prepared["prepared_at"])
    return prepared


def load_validation_packet(sidecar: Path, payload: dict):
    if (
        payload.get("stage") != STAGE
        or payload.get("validation_only") is not True
        or payload.get("historical_oos_authorized") is not False
        or payload.get("period") != PERIOD
    ):
        raise Error("Only the sealed 2012–2018 validation period is permitted")
    prepared = verify_prepared(sidecar)
    daily, bars, calendar, coverage, blackouts, manifest = prefix.read_validation_prefix(
        sidecar / "prepared", expected_manifest_sha256=prepared["manifest_sha256"]
    )
    config = copy.deepcopy(protocol.load_config(payload["config_path"]))
    config["segments"][STAGE] = dict(PERIOD)
    config["segments"]["historical-oos"] = dict(HISTORICAL_OOS_PERIOD)
    cash = fixed.constant_cash_observations(
        calendar.market_open.iloc[0], daily.daily_price_available_at.iloc[-1]
    )
    # The original helper is parameterized by the NEW copied period, not 2015.
    protocol._check_prefix(daily, bars, calendar, cash, STAGE, config)
    registration = protocol._read_seal(Path(payload["source_study"]) / "registration.json")
    protocol._quality_gate(config, coverage, registration["gap_authorization"])
    return daily, bars, calendar, coverage, blackouts, cash, config


def counts_against_bh(
    metrics: pd.DataFrame, benchmark_sharpe: float, pairs: list[dict]
) -> pd.DataFrame:
    counts = original.counts_against_bh(metrics, benchmark_sharpe, pairs)
    for index, hours in enumerate(WAIT_HOURS):
        candidates = candidates_from_pairs(pairs, hours)
        rows = original._checked_metrics(
            metrics, [c for h in WAIT_HOURS for c in candidates_from_pairs(pairs, h)]
        )
        subset = rows.loc[[c.candidate_id for c in candidates]]
        for field in ("cagr", "max_drawdown", "annualized_volatility", "exposure_fraction"):
            if field in subset:
                values = pd.to_numeric(subset[field], errors="coerce").to_numpy(float)
                if not np.isfinite(values).all():
                    raise Error(
                        f"Nonfinite validation {field} cannot be silently excluded from median"
                    )
                counts.loc[index, "median_" + field] = float(np.median(values))
    return counts


def evaluate_study(sidecar: str | Path = DEFAULT_SIDECAR) -> dict:
    sidecar = fixed._safe_root(sidecar)
    with protocol._attempt(sidecar, "validation_only") as attempt:
        payload = verify_study(sidecar)  # Complete candidate set sealed before any price arrays.
        prepared = verify_prepared(sidecar)
        if (sidecar / "results").exists():
            raise Error("Validation already completed; verify it, never overwrite/rerun")
        started_at = protocol._utc_now()
        _chronology(payload["candidate_set_sealed_at"], prepared["prepared_at"], started_at)
        protocol._write_json(
            attempt / "market_access_started.json",
            {
                "candidate_set_sha256": payload["candidate_set_sha256"],
                "candidate_set_sealed_at": payload["candidate_set_sealed_at"],
                "market_outcomes_started_at": started_at,
                "stage": STAGE,
                "validation_only": True,
                "historical_oos_authorized": False,
            },
        )
        daily, bars, calendar, coverage, blackouts, cash, config = load_validation_packet(
            sidecar, payload
        )
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
        benchmarks, benchmark_sharpe = [], None
        dummy = candidates_from_pairs(payload["pairs"], WAIT_HOURS[0])[0]
        for mode in ("cash", "buy_and_hold"):
            result = engine._result(engine._run(cache, [dummy], mode=mode, record=False, **kwargs))
            benchmarks.append({"portfolio": mode, **result.metrics})
            fixed._write_curve(output / f"{mode}_daily.csv", result)
            if mode == "buy_and_hold":
                benchmark_sharpe = float(result.metrics["cash_excess_sharpe"])
        pd.DataFrame(benchmarks).to_csv(output / "benchmark_metrics.csv", index=False)
        all_rows, post_rows, post_total_rows = [], [], 0
        post_context = engine._post_gap_context(cache, selected_calendar)
        for hours in WAIT_HOURS:
            candidates = candidates_from_pairs(payload["pairs"], hours)
            parts = []
            for offset in range(0, len(candidates), fixed.BATCH_SIZE):
                part = engine._batch(
                    cache,
                    selected_calendar,
                    events,
                    clock,
                    terminal,
                    candidates[offset : offset + fixed.BATCH_SIZE],
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
                raise Error("Validation cohort clocks disagree")
            np.savez_compressed(
                output / f"wait_{str(hours).replace('.', 'p')}_daily.npz",
                timestamp=np.asarray([time.isoformat() for time in times], dtype=str),
                candidate_id=np.asarray([c.candidate_id for c in candidates], dtype=str),
                equity=np.concatenate([part[2] for part in parts], axis=1),
                returns=np.concatenate([part[3] for part in parts], axis=1),
                cash_returns=parts[0][4],
                exposure=np.concatenate([part[5] for part in parts], axis=1),
            )
            print(
                f"Completed 2012–2018 validation common wait {hours:g} h: {len(candidates)} development-frozen pairs",
                flush=True,
            )
        metrics = pd.DataFrame(all_rows)
        if post_context is not None:
            metrics.attrs["post_gap_signal_events"] = engine._post_gap_table(post_rows)
            metrics.attrs["post_gap_signal_event_metadata"] = engine._post_gap_metadata(
                post_context, cache.policy, post_total_rows, len(post_rows), all_rows
            )
            protocol._persist_post_gap_diagnostics(
                output,
                metrics,
                [c for hours in WAIT_HOURS for c in candidates_from_pairs(payload["pairs"], hours)],
                config,
                STAGE,
            )
        metrics.to_csv(output / "candidate_metrics.csv", index=False)
        counts = counts_against_bh(metrics, benchmark_sharpe, payload["pairs"])
        counts.to_csv(output / "counts_by_wait.csv", index=False)
        pd.DataFrame(payload["development_selection_rows"]).to_csv(
            output / "development_selection_metrics.csv", index=False
        )
        verify_study(sidecar)
        result = {
            "schema": "trend_following_resplit_intersection_validation_results_v1",
            "freeze_sha256": protocol.canonical_hash(payload),
            "candidate_set_sha256": payload["candidate_set_sha256"],
            "pairs_sha256": payload["pairs_sha256"],
            "configuration_count": len(metrics),
            "configuration_count_per_wait": len(payload["pairs"]),
            "waits_evaluated": list(WAIT_HOURS),
            "stage": STAGE,
            "period": payload["period"],
            "cash_policy_sha256": protocol.canonical_hash(payload["cash"]),
            "commission_bps_per_order": COMMISSION_BPS,
            "slippage_bps_per_order": SLIPPAGE_BPS,
            "candidate_set_sealed_at": payload["candidate_set_sealed_at"],
            "market_outcomes_started_at": started_at,
            "finished_at": protocol._utc_now(),
            "BH_cash_excess_Sharpe": benchmark_sharpe,
            "validation_manifest_sha256": prepared["manifest_sha256"],
            "prepared_sha256": protocol.canonical_hash(prepared),
            "prepared_at": prepared["prepared_at"],
            "files": {path.name: protocol.sha256_file(path) for path in sorted(output.iterdir())},
            "no_winner_selected": True,
            "no_validation_filtering_or_replacement": True,
            "no_2019plus_numeric_prices_or_prior_validation_outcomes_loaded": True,
            "historical_oos_authorized": False,
        }
        _chronology(
            result["candidate_set_sealed_at"],
            result["prepared_at"],
            result["market_outcomes_started_at"],
            result["finished_at"],
        )
        protocol._seal(output / "seal.json", result)
        output.rename(sidecar / "results")
        return result


def verify_results(sidecar: str | Path = DEFAULT_SIDECAR) -> dict:
    sidecar = fixed._safe_root(sidecar)
    payload = verify_study(sidecar)
    prepared = verify_prepared(sidecar)
    result = protocol._read_seal(sidecar / "results/seal.json")
    expected = {
        "schema": "trend_following_resplit_intersection_validation_results_v1",
        "freeze_sha256": protocol.canonical_hash(payload),
        "candidate_set_sha256": payload["candidate_set_sha256"],
        "pairs_sha256": payload["pairs_sha256"],
        "configuration_count": payload["configuration_count"],
        "configuration_count_per_wait": len(payload["pairs"]),
        "waits_evaluated": list(WAIT_HOURS),
        "stage": STAGE,
        "period": payload["period"],
        "cash_policy_sha256": protocol.canonical_hash(payload["cash"]),
        "commission_bps_per_order": COMMISSION_BPS,
        "slippage_bps_per_order": SLIPPAGE_BPS,
        "candidate_set_sealed_at": payload["candidate_set_sealed_at"],
        "validation_manifest_sha256": prepared["manifest_sha256"],
        "prepared_sha256": protocol.canonical_hash(prepared),
        "prepared_at": prepared["prepared_at"],
        "no_winner_selected": True,
        "no_validation_filtering_or_replacement": True,
        "no_2019plus_numeric_prices_or_prior_validation_outcomes_loaded": True,
        "historical_oos_authorized": False,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise Error("Validation result contract/counts/identity/costs changed")
    _chronology(
        result["candidate_set_sealed_at"],
        result["prepared_at"],
        result["market_outcomes_started_at"],
        result["finished_at"],
    )
    required = {
        "benchmark_metrics.csv",
        "buy_and_hold_daily.csv",
        "cash_daily.csv",
        "candidate_metrics.csv",
        "counts_by_wait.csv",
        "development_selection_metrics.csv",
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
        raise Error("Unexpected/missing/untracked validation artifacts")
    if not np.isfinite(result.get("BH_cash_excess_Sharpe", np.nan)):
        raise Error("Undefined validation benchmark Sharpe")
    for name, expected_hash in result["files"].items():
        path = sidecar / "results" / name
        if (
            Path(name).name != name
            or path.is_symlink()
            or protocol.sha256_file(path) != expected_hash
        ):
            raise Error("Validation derived output bytes changed")
    return result


def verify_release(sidecar: str | Path = DEFAULT_SIDECAR) -> tuple[dict, dict]:
    """Intact seals before publication parses derived numeric outcomes."""
    result = verify_results(sidecar)
    return protocol._read_seal(fixed._safe_root(sidecar) / "freeze.json"), result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "prepare", "evaluate", "verify"))
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-study", type=Path, default=DEFAULT_SOURCE_STUDY)
    parser.add_argument("--development-sidecar", type=Path, default=DEFAULT_DEVELOPMENT_SIDECAR)
    parser.add_argument(
        "--original-intersection-sidecar", type=Path, default=DEFAULT_ORIGINAL_INTERSECTION
    )
    args = parser.parse_args(argv)
    if args.command == "seal":
        if args.authorization is None:
            parser.error("seal requires explicit human validation-only authorization")
        result = seal_study(
            args.sidecar,
            authorization=args.authorization,
            config_path=args.config,
            source_study=args.source_study,
            development_sidecar=args.development_sidecar,
            original_intersection_sidecar=args.original_intersection_sidecar,
        )
    elif args.command == "prepare":
        result = prepare_study(args.sidecar)
    elif args.command == "evaluate":
        result = evaluate_study(args.sidecar)
    else:
        result = (
            verify_results(args.sidecar)
            if (args.sidecar / "results").exists()
            else verify_prepared(args.sidecar)
            if (args.sidecar / "prepared").exists()
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
