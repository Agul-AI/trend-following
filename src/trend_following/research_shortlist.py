"""Separately sealed development shortlist; never replaces the original winner.

The original study's sources, configuration and sealed results are immutable.
This exploratory extension was requested after earlier validation/OOS exposure:
it cannot restore untouched out-of-sample evidence or select on later outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from trend_following import research_quality_engine_v5 as engine
from trend_following import research_quality_protocol_v5 as parent
from trend_following.research_shortlist_selection import select_development_shortlist

ROOT = parent.WORKSPACE
DEFAULT_CONFIG = ROOT / "configs/qqq_daily_halfhour_v5_gap_flags_v2.yaml"
DEFAULT_PARENT = ROOT / ".cache/qqq_daily_halfhour_v5_alpha_gap_flags_v2"
NEW_SOURCES = (
    "src/trend_following/research_shortlist.py",
    "src/trend_following/research_shortlist_selection.py",
    "scripts/run_qqq_research.py",
)
SCHEMA = "trend_following_exploratory_shortlist_registration_v1"
LABEL = "exploratory_retrospective_extension_after_prior_validation_and_historical_oos_exposure"
DAILY_COLUMNS = (
    "candidate_id",
    "timestamp",
    "return",
    "equity",
    "cash_return",
    "exposure",
    "commission_bps",
    "slippage_bps",
)
REQUIRED_AUTHORIZATION_STAGES = {"validation", "historical-oos"}


def _require(condition, message):
    if not condition:
        raise parent.ProtocolError(message)


def _source_hashes():
    return {name: parent.sha256_file(ROOT / name) for name in NEW_SOURCES}


def _paths(parent_study, config_path, study_dir):
    paths = tuple(Path(value) for value in (parent_study, config_path, study_dir))
    _require(not any(path.is_symlink() for path in paths), "Symlink research paths prohibited")
    original, config, extension = (path.resolve() for path in paths)
    _require(
        original != extension
        and original not in extension.parents
        and extension not in original.parents,
        "Shortlist must use a separate study directory, never the original study tree",
    )
    return original, config, extension


def _authorization(path):
    path = Path(path)
    _require(not path.is_symlink(), "Symlink authorization receipt prohibited")
    receipt = json.loads(path.read_text())
    _require(
        receipt.get("schema") == "trend_following_shortlist_authorization_v1"
        and receipt.get("project") == "Trend Following Study"
        and receipt.get("authorized_by") == "human_user"
        and receipt.get("explicit_user_approval") is True
        and set(receipt.get("authorized_stages", [])) == REQUIRED_AUTHORIZATION_STAGES
        and isinstance(receipt.get("user_request"), str)
        and bool(receipt["user_request"].strip()),
        "Shortlist requires an explicit direct human authorization receipt for both stages",
    )
    return {
        "path": str(path.resolve()),
        "sha256": parent.sha256_file(path),
        "user_request": receipt["user_request"],
        "authorized_stages": sorted(REQUIRED_AUTHORIZATION_STAGES),
    }


def _parent_context(original, config_path):
    config = parent.load_config(config_path)
    registration = parent._verify_registration(original, config, config_path)
    freeze = parent._verify_prepared(original, registration)
    development = parent._verify_stage(original, "development", registration, freeze)
    candidates, table = parent._candidate_table(config)
    return config, registration, freeze, development, candidates, table


def _parent_identity(original, config_path, context):
    config, registration, freeze, development, _, _ = context
    return {
        "parent_study": str(original),
        "config_path": str(config_path),
        "parent_registration_sha256": parent.canonical_hash(registration),
        "prepared_sha256": parent.canonical_hash(freeze),
        "development_stage_sha256": parent.canonical_hash(development),
        "development_metrics_sha256": parent.sha256_file(
            original / "stages/development/metrics.csv"
        ),
        "original_grid_sha256": registration["grid_sha256"],
        "original_grid_count": registration["candidate_count"],
        "original_source_hashes": registration["source_hashes"],
        "original_config_sha256": parent.sha256_file(config_path),
        "accounting_sha256": parent.canonical_hash(config["accounting"]),
        "segments": config["segments"],
    }


def _selection(original, context, per_family):
    config, _, _, _, candidates, _ = context
    metrics = pd.read_csv(original / "stages/development/metrics.csv", float_precision="round_trip")
    selected = select_development_shortlist(
        metrics, candidates, per_family=per_family, tolerance=config["selection"]["tie_tolerance"]
    )
    _require(
        len(selected) == 9 * per_family,
        "Every family cell needs the requested finite distinct pairs",
    )
    return selected


def _candidate_payload(candidates):
    return [
        {"candidate_id": candidate.candidate_id, **candidate.to_dict()} for candidate in candidates
    ]


def _selection_contract(per_family, tolerance):
    _require(
        not isinstance(per_family, bool) and per_family in (1, 5, 10),
        "Use 1, 5 or 10 per family",
    )
    return {
        "per_family": per_family,
        "candidate_count": 9 * per_family,
        "distinctness": "entry_indicator_exit_indicator_pair_ignoring_confirmation_durations",
        "primary_metric": "development_daily_cash_excess_sharpe_after_1bp_commission_3bp_slippage",
        "tie_tolerance": tolerance,
        "tie_break": "lexicographically_smallest_stable_candidate_id",
        "later_stage_selection": False,
        "gap_flags_affect_selection": False,
        "all_selected_candidates_retained_in_both_later_stages": True,
    }


def seal_shortlist(
    *,
    parent_study=DEFAULT_PARENT,
    config_path=DEFAULT_CONFIG,
    study_dir,
    authorization_receipt,
    per_family=5,
):
    _require(
        not isinstance(per_family, bool) and per_family in (1, 5, 10), "Use 1, 5 or 10 per family"
    )
    original, config_path, study = _paths(parent_study, config_path, study_dir)
    _require(not study.exists(), "Never overwrite or extend an existing shortlist study directory")
    authorization = _authorization(authorization_receipt)
    # Only development numeric outcomes are opened; later parent outcomes are never inputs.
    context = _parent_context(original, config_path)
    selected = _selection(original, context, per_family)
    registration = {
        "schema": SCHEMA,
        "project": "Trend Following Study",
        "created_at": parent._utc_now(),
        "historical_label": LABEL,
        **_parent_identity(original, config_path, context),
        "new_source_hashes": _source_hashes(),
        "authorization": authorization,
        "selection": _selection_contract(per_family, context[0]["selection"]["tie_tolerance"]),
        "validation_slippage_bps": [3.0],
        "historical_oos_slippage_bps": [1.0, 3.0, 10.0, 25.0],
    }
    with parent._attempt(study, "shortlist-seal") as attempt:
        parent._seal(study / "registration.json", registration)
        candidate_set = {
            "schema": "trend_following_development_shortlist_set_v1",
            "registration_sha256": parent.canonical_hash(registration),
            "candidates": _candidate_payload(selected),
            "sealed_before_extension_validation": True,
            "prior_validation_and_historical_oos_already_examined": True,
            "sealed_at": parent._utc_now(),
        }
        parent._seal(study / "candidate_set.json", candidate_set)
        outcome = {
            "status": "completed",
            "project": "Trend Following Study",
            "candidate_count": len(selected),
            "candidate_set_sha256": parent.canonical_hash(candidate_set),
            "historical_label": LABEL,
            "performance_run": False,
            "stage": "shortlist-seal",
            "started_at": json.loads((attempt / "attempt.json").read_text())["started_at"],
            "finished_at": parent._utc_now(),
        }
        parent._write_json(attempt / "outcome.json", outcome)
        return outcome


def _verify_context(original, config_path, study):
    registration = parent._read_seal(study / "registration.json")
    _require(
        registration.get("schema") == SCHEMA and registration.get("historical_label") == LABEL,
        "Wrong exploratory shortlist registration",
    )
    _require(
        registration["new_source_hashes"] == _source_hashes(),
        "Shortlist implementation changed after seal",
    )
    _require(
        registration["authorization"] == _authorization(registration["authorization"]["path"]),
        "Direct human authorization receipt changed",
    )
    context = _parent_context(original, config_path)
    identity = _parent_identity(original, config_path, context)
    _require(
        all(registration.get(key) == value for key, value in identity.items()),
        "Original development, frozen sources, data, grid or accounting changed",
    )
    selection = registration["selection"]
    _require(
        selection
        == _selection_contract(selection["per_family"], context[0]["selection"]["tie_tolerance"]),
        "Invalid sealed retention rule",
    )
    candidate_set = parent._read_seal(study / "candidate_set.json")
    _require(
        candidate_set["registration_sha256"] == parent.canonical_hash(registration)
        and candidate_set["sealed_before_extension_validation"] is True
        and candidate_set["prior_validation_and_historical_oos_already_examined"] is True,
        "Candidate set is not linked to its prior exploratory registration",
    )
    selected = _selection(original, context, selection["per_family"])
    _require(
        candidate_set["candidates"] == _candidate_payload(selected),
        "Shortlist must exactly equal deterministic development-only selection",
    )
    _require(
        registration["validation_slippage_bps"] == [3.0]
        and registration["historical_oos_slippage_bps"] == [1.0, 3.0, 10.0, 25.0],
        "Frozen shortlist execution costs changed",
    )
    return registration, candidate_set, context, selected


def verify_shortlist(*, parent_study=DEFAULT_PARENT, config_path=DEFAULT_CONFIG, study_dir):
    """Read-only publication gate; no validation or OOS numerical outcomes opened."""
    original, config_path, study = _paths(parent_study, config_path, study_dir)
    registration, candidate_set, context, candidates = _verify_context(original, config_path, study)
    return {
        "candidates": candidates,
        "original_config": context[0],
        "config": context[0],
        "registration": registration,
        "candidate_set": candidate_set,
        "shortlist": candidate_set,
    }


def verify_stage(study_dir, stage, registration):
    """Verify fixed-set result hashes before any publication helper reads outcomes."""
    _require(
        stage in ("validation", "historical-oos"), "Only validation and historical-oos are allowed"
    )
    study = Path(study_dir)
    _require(not study.is_symlink(), "Symlink shortlist study prohibited")
    candidate_set = parent._read_seal(study / "candidate_set.json")
    _require(
        candidate_set["registration_sha256"] == parent.canonical_hash(registration),
        "Shortlist set does not match supplied registration",
    )
    return _verify_stage(study, stage, registration, candidate_set)


def _verify_stage(study, stage, registration, candidate_set):
    root = study / "stages" / stage
    summary = parent._read_seal(root / "stage_seal.json")
    expected_ids = [value["candidate_id"] for value in candidate_set["candidates"]]
    _require(
        summary.get("stage") == stage
        and summary.get("registration_sha256") == parent.canonical_hash(registration)
        and summary.get("candidate_set_sha256") == parent.canonical_hash(candidate_set)
        and summary.get("candidate_ids") == expected_ids
        and summary.get("candidate_count") == len(expected_ids)
        and summary.get("historical_label") == LABEL
        and summary.get("slippage_bps_scenarios")
        == registration[
            "validation_slippage_bps" if stage == "validation" else "historical_oos_slippage_bps"
        ]
        and summary.get("commission_bps") == 1.0
        and summary.get("scored_segment") == registration["segments"][stage]
        and summary.get("selection_after_evaluation") is False,
        "Stage must retain the complete unchanged development shortlist",
    )
    _require(
        parent._artifact_hashes(root) == summary.get("artifact_hashes"),
        "Shortlist stage artifacts changed",
    )
    return summary


def batch_derived(
    cache,
    candidates,
    *,
    start,
    end,
    cash_observations,
    commission_bps=1.0,
    slippage_bps=3.0,
    initial_cash=1000.0,
):
    """Use the unchanged kernel's vectorized batch, retaining derived curves only."""
    _require(0 < len(candidates) <= 256, "Shortlist must fit one unique nonempty vectorized batch")
    _require(
        len({candidate.candidate_id for candidate in candidates}) == len(candidates),
        "Duplicate candidate IDs",
    )
    parts = engine._run(
        cache,
        candidates,
        start=start,
        end=end,
        cash_observations=cash_observations,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        initial_cash=initial_cash,
        record=False,
    )
    metrics = pd.DataFrame(parts[0])
    parent._audit_metrics(metrics, candidates)
    metrics["return_type"] = "research_simulation"
    times, equity, returns, cash_returns, exposure = parts[1:6]
    curves = pd.concat(
        [
            pd.DataFrame(
                {
                    "candidate_id": candidate.candidate_id,
                    "timestamp": times,
                    "return": returns[:, index],
                    "equity": equity[:, index],
                    "cash_return": cash_returns,
                    "exposure": exposure[:, index],
                    "commission_bps": commission_bps,
                    "slippage_bps": slippage_bps,
                }
            )
            for index, candidate in enumerate(candidates)
        ],
        ignore_index=True,
    )
    _require(tuple(curves.columns) == DAILY_COLUMNS, "Derived daily publication whitelist differs")
    _require(
        not curves[["candidate_id", "timestamp"]].duplicated().any(),
        "Duplicate daily portfolio observations",
    )
    _require(
        np.isfinite(curves[["return", "equity", "cash_return", "exposure"]]).all().all(),
        "Derived portfolio values must be finite",
    )
    return metrics, curves


def _benchmark_tables(results, slip):
    _require(set(results) == {"buy_and_hold", "cash"}, "Both matched passive controls are required")
    metrics, curves = [], []
    for name, result in results.items():
        row = {
            **result.metrics,
            "candidate_id": name,
            "policy": name,
            "return_type": "research_simulation",
        }
        metrics.append(row)
        curves.append(
            pd.DataFrame(
                {
                    "candidate_id": name,
                    "timestamp": result.daily_equity.index,
                    "return": result.daily_returns.to_numpy(),
                    "equity": result.daily_equity.to_numpy(),
                    "cash_return": result.daily_cash_returns.to_numpy(),
                    "exposure": result.daily_exposure.to_numpy(),
                    "commission_bps": 1.0,
                    "slippage_bps": slip,
                }
            )
        )
    return pd.DataFrame(metrics), pd.concat(curves, ignore_index=True)


def run_shortlist_stage(
    stage, *, parent_study=DEFAULT_PARENT, config_path=DEFAULT_CONFIG, study_dir
):
    _require(
        stage in ("validation", "historical-oos"), "Only validation and historical-oos are allowed"
    )
    original, config_path, study = _paths(parent_study, config_path, study_dir)
    with parent._attempt(study, "shortlist-" + stage) as attempt:
        # Integrity checks precede ANY numeric stage prefix read; failures remain logged.
        registration, candidate_set, context, candidates = _verify_context(
            original, config_path, study
        )
        config, original_registration, _, _, _, _ = context
        if stage == "historical-oos":
            _verify_stage(study, "validation", registration, candidate_set)
        if (study / "stages" / stage).exists():
            summary = _verify_stage(study, stage, registration, candidate_set)
            parent._write_json(
                attempt / "outcome.json",
                {
                    **summary,
                    "verified_existing_stage": True,
                    "started_at": json.loads((attempt / "attempt.json").read_text())["started_at"],
                    "finished_at": parent._utc_now(),
                },
            )
            return summary
        if stage == "validation":
            _require(
                not (study / "stages/historical-oos").exists(),
                "Historical stage cannot precede validation",
            )
        daily, bars, calendar, coverage, blackouts, cash, _ = parent._load_stage(
            original, original_registration, config, stage
        )
        cache = engine.build_quality_support_cache(
            daily,
            bars,
            calendar,
            coverage=coverage,
            blackouts=blackouts,
            quality_policy=parent._quality_policy(config),
        )
        slips = (
            registration["validation_slippage_bps"]
            if stage == "validation"
            else registration["historical_oos_slippage_bps"]
        )
        metrics, curves, benchmarks, benchmark_curves = [], [], [], []
        for slip in slips:
            kwargs = parent._engine_kwargs(
                config,
                stage,
                cash,
                calendar,
                coverage=coverage,
                blackouts=blackouts,
                slippage_bps=slip,
            )
            result, derived = batch_derived(
                cache,
                candidates,
                **{
                    key: kwargs[key]
                    for key in (
                        "start",
                        "end",
                        "cash_observations",
                        "commission_bps",
                        "slippage_bps",
                        "initial_cash",
                    )
                },
            )
            controls = engine.simulate_quality_benchmarks(
                daily, bars, support_cache=cache, **kwargs
            )
            for control in controls.values():
                _require(
                    pd.DatetimeIndex(
                        derived.loc[
                            derived.candidate_id.eq(candidates[0].candidate_id), "timestamp"
                        ]
                    ).equals(control.daily_equity.index),
                    "Shortlist and passive control clocks differ",
                )
            control_metrics, control_curves = _benchmark_tables(controls, slip)
            metrics.append(result)
            curves.append(derived)
            benchmarks.append(control_metrics)
            benchmark_curves.append(control_curves)
        result_dir = attempt / "results"
        result_dir.mkdir()
        pd.concat(metrics, ignore_index=True).to_csv(result_dir / "metrics.csv", index=False)
        pd.concat(curves, ignore_index=True).to_csv(
            result_dir / "daily.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
        )
        pd.concat(benchmarks, ignore_index=True).to_csv(result_dir / "benchmarks.csv", index=False)
        pd.concat(benchmark_curves, ignore_index=True).to_csv(
            result_dir / "benchmark_daily.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        # Recheck every dependency after computation, before installing immutable results.
        _verify_context(original, config_path, study)
        summary = {
            "project": "Trend Following Study",
            "started_at": json.loads((attempt / "attempt.json").read_text())["started_at"],
            "finished_at": parent._utc_now(),
            "status": "completed",
            "stage": stage,
            "historical_label": LABEL,
            "registration_sha256": parent.canonical_hash(registration),
            "candidate_set_sha256": parent.canonical_hash(candidate_set),
            "candidate_count": len(candidates),
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "slippage_bps_scenarios": slips,
            "commission_bps": 1.0,
            "scored_segment": config["segments"][stage],
            "selection_after_evaluation": False,
            "derived_curves_only": True,
            "raw_quote_values_or_trades_exported": False,
            "artifact_hashes": parent._artifact_hashes(result_dir),
        }
        parent._seal(result_dir / "stage_seal.json", summary)
        destination = study / "stages" / stage
        destination.parent.mkdir(exist_ok=True)
        _require(not destination.exists(), "Completed shortlist results cannot be overwritten")
        os.rename(result_dir, destination)
        parent._write_json(attempt / "outcome.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("seal", "validation", "historical-oos"))
    parser.add_argument("--parent-study", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--per-family", type=int, choices=(1, 5, 10), default=None)
    parser.add_argument("--authorization-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "seal" and args.authorization_receipt is None:
        parser.error("seal requires --authorization-receipt with direct human approval")
    if args.stage != "seal" and (
        args.per_family is not None or args.authorization_receipt is not None
    ):
        parser.error(
            "--per-family and --authorization-receipt are seal-only; later stages cannot change the set"
        )
    common = {
        "parent_study": args.parent_study,
        "config_path": args.config,
        "study_dir": args.study_dir,
    }
    try:
        result = (
            seal_shortlist(
                **common,
                authorization_receipt=args.authorization_receipt,
                per_family=5 if args.per_family is None else args.per_family,
            )
            if args.stage == "seal"
            else run_shortlist_stage(args.stage, **common)
        )
    except (parent.ProtocolError, ValueError, OSError, KeyError) as error:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "status": "rejected",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0
