"""Separately sealed QQQ hybrid allocation development/retrospective validation.

Existing verifiers execute unchanged once at registration. A read-only profiling
observer records their exact source, seal, bounded-packet and derived-result file
closure. Subsequent checks rehash that closed graph directly; no verifier is
monkeypatched, skipped during registration, or memoized across unverified inputs.
No OOS numeric price reader or execution route is exposed.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from trend_following import research_constant_cash as fixed
from trend_following import research_daily_hourly_v5 as trend
from trend_following import research_mean_reversion as mr
from trend_following import research_mean_reversion_engine as mr_engine
from trend_following import research_mean_reversion_validation as mr_val
from trend_following import research_multi_scale_protocol as multi
from trend_following import research_portfolio_engine as portfolio
from trend_following import research_portfolio_protocol as preserved
from trend_following import research_quality_engine_v5 as quality
from trend_following import research_quality_protocol_v5 as custody
from trend_following import research_validation_prefix as prefix

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/qqq_hybrid_allocation_research.yaml"
DEFAULT_SIDECAR = ROOT / ".cache/hybrid_allocation_research_20261010"
DEFAULT_REPORT = ROOT / "reports/hybrid_allocation_research/20261010"
DEFAULT_MULTI_SCALE = multi.DEFAULT_SIDECAR
DEFAULT_PROTECTED = (
    ROOT / ".cache/hybrid_allocation_implementation_20261010/protected_tracked_hashes.json"
)
PERIODS = copy.deepcopy(multi.PERIODS)
SLIPPAGES = multi.SLIPPAGES
ALLOCATIONS = (0.5, 0.7, 0.9)
TOLERANCE = 1e-10
GATED_CONTROL_ID = "hybrid_gated_mr"
REGIME_ID = "hybrid_regime_er20_025_035"
PURE_TREND_ID = "portfolio_tf_100"
APPROVAL_EXCERPT = (
    "PLEASE IMPLEMENT THIS PLAN:\n# Trend Core, Tactical Mean Reversion and Adaptive Allocation"
)
REGIME_RULE = {
    "period_days": 20,
    "lower": 0.25,
    "upper": 0.35,
    "availability": "finalized_daily_17_NY",
    "initial_mode": "TF",
}
Error = custody.ProtocolError
safe_path = multi.safe_path
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *multi.SOURCE_FILES,
            "src/trend_following/research_hybrid_allocation_engine.py",
            "src/trend_following/research_hybrid_allocation_ledger.py",
            "src/trend_following/research_hybrid_allocation_protocol.py",
            "src/trend_following/research_hybrid_allocation_report.py",
            "scripts/run_hybrid_allocation_research.py",
            "configs/qqq_hybrid_allocation_research.yaml",
        )
    )
)


def principal_definitions():
    return (
        [
            {
                "portfolio_id": f"hybrid_core_{int(a * 100):03d}",
                "architecture": "A",
                "trend_allocation": a,
            }
            for a in ALLOCATIONS
        ]
        + [
            {
                "portfolio_id": f"hybrid_blend_{int(a * 100):03d}",
                "architecture": "B",
                "trend_allocation": a,
            }
            for a in ALLOCATIONS
        ]
        + [
            {"portfolio_id": REGIME_ID, "architecture": "C", "trend_allocation": None},
            {"portfolio_id": PURE_TREND_ID, "architecture": "pure_trend", "trend_allocation": 1.0},
        ]
    )


def approved_config():
    return {
        "schema": "qqq_hybrid_allocation_research",
        "study_name": "Trend Following Study — Trend Core, Tactical Mean Reversion and Adaptive Allocation",
        "security": "QQQ",
        "historical_label": "retrospective_hindsight_conditioned_membership",
        "membership": {
            "trend_count": 12,
            "mean_reversion_count": 32,
            "pair_count": 384,
            "equal_initial_capital_per_pair": True,
            "pair_selection": False,
            "unchanged_constituents": True,
        },
        "portfolio": {
            "initial_cash": 1000.0,
            "allocations": list(ALLOCATIONS),
            "pure_trend_fallback": True,
            "principal_count": 8,
            "pair_result_count_per_cost": 2688,
            "fractional_shares": True,
            "cross_account_netting": False,
            "capital_redistribution": False,
        },
        "core": {
            "funded_trend_accounts": 12,
            "funded_tactical_accounts": 384,
            "rebalance": False,
            "MR_gate": "paired_TF_actual_filled_long_state",
            "TF_exit_forces_MR_at_same_safe_open": True,
        },
        "blend": {
            "funded_accounts_per_allocation": 384,
            "target": "alpha_sTF_plus_one_minus_alpha_sMRgated",
            "rebalance": "only_confirmed_target_changes",
            "sizing": "post_cost_equity_at_execution_raw_open",
            "atomic_simultaneous_orders": True,
            "maintenance_rebalance": False,
        },
        "regime": {
            **REGIME_RULE,
            "observations": 21,
            "zero_denominator": 0.0,
            "unknown": "retain_mode_cancel_switch_suppress_risk_adds_allow_reductions",
            "confirmation_hours": 3.0,
            "required_checks": 7,
            "shadow_states": "warm_within_TF_filled_permission",
            "reconcile_after_unknown": True,
            "threshold_selection": False,
        },
        "execution": {
            "indicator_units": "daily_sessions",
            "shared_confirmation_hours": 3.0,
            "required_checks": 7,
            "TF_exit_priority": True,
            "next_safe_observed_open": True,
            "MR_fresh_after_TF_entry": True,
            "gaps": "hold_cancel_pending_reset_streaks",
            "post_gap_flags": "descriptive_through_next_session_close",
        },
        "periods": {"initialization": {"start": "1999-11-01", "end": "2000-12-31"}, **PERIODS},
        "accounting": {
            "cash_nominal_annual_rate": 0.03,
            "cash_day_count": "ACT/365",
            "commission_bps": 1.0,
            "baseline_slippage_bps": 3.0,
            "slippage_bps_scenarios": list(SLIPPAGES),
            "taxes": False,
            "financing": False,
            "separate_fund_expenses": False,
            "matched_terminal_liquidation": True,
            "gross_shadow": "same_commands_own_wealth_quantities",
        },
        "selection": {
            "metric": "daily_cash_excess_sharpe",
            "annualization": 252,
            "standard_deviation_ddof": 1,
            "finite_only": True,
            "baseline_only": True,
            "tie_tolerance": TOLERANCE,
            "tie_order": "stable_portfolio_id",
            "one_lock_per_architecture": ["A", "B"],
            "filters": False,
            "C_fixed_no_selection": True,
        },
        "controls": {
            "reused_independent_trend_allocations": [0.5, 0.7, 0.9],
            "pure_trend": True,
            "ungated_MR": True,
            "new_TF_state_gated_MR": True,
            "buy_and_hold": True,
            "cash": True,
            "reuse_existing_seals": True,
        },
        "bootstrap": {
            "replicates": 2000,
            "block_lengths": [5, 20, 60],
            "seed": 20261009,
            "confidence_level": 0.95,
            "paired": True,
            "deduplicate_identical_comparison_paths_keep_aliases": True,
        },
        "protocol": {
            "authorized_stages": ["development", "validation"],
            "historical_oos_authorized": False,
            "fresh_segment_state": True,
            "hindsight_conditioned_membership": True,
            "no_validation_reselection": True,
        },
    }


def load_config(path=DEFAULT_CONFIG):
    value = yaml.safe_load(safe_path(path).read_text())
    if custody.canonical_hash(value) != custody.canonical_hash(approved_config()):
        raise Error("Hybrid configuration differs from exact approved contract")
    return value


def authorization(path):
    value = json.loads(safe_path(path).read_text())
    expected = {
        "schema": "qqq_hybrid_allocation_authorization",
        "authorized_by": "human_user",
        "explicit_user_approval": True,
        "approval_excerpt": APPROVAL_EXCERPT,
        "authorized_stages": ["development", "validation"],
        "trend_count": 12,
        "mean_reversion_count": 32,
        "pair_count": 384,
        "principal_count": 8,
        "shared_confirmation_hours": 3.0,
        "required_checks": 7,
        "development_period": PERIODS["development"],
        "validation_period": PERIODS["validation"],
        "initial_trend_allocations": [0.5, 0.7, 0.9, 1.0],
        "architectures": ["A", "B", "C"],
        "regime_rule": REGIME_RULE,
        "historical_oos_authorized": False,
        "no_constituent_retuning_or_replacement": True,
        "reuse_sealed_controls": True,
    }
    if any(
        custody.canonical_hash(value.get(k)) != custody.canonical_hash(v)
        for k, v in expected.items()
    ):
        raise Error("Explicit matching human hybrid development/validation authorization required")
    return value


verify_authorization = authorization


def _protected(path):
    values = json.loads(safe_path(path).read_text())
    if (
        not isinstance(values, dict)
        or not values
        or any(n in values for n in ("AGENTS.md", "README.md"))
    ):
        raise Error("Protected tracked-file snapshot must omit only authorized metadata files")
    for name, digest in values.items():
        p = Path(name)
        if (
            p.is_absolute()
            or ".." in p.parts
            or len(digest) != 64
            or custody.sha256_file(safe_path(ROOT / p)) != digest
        ):
            raise Error("Existing protected tracked source/artifact changed")
    return values


def _separate(sidecar, source, auth=None):
    if sidecar.parent != ROOT / ".cache" or not sidecar.name.startswith(
        "hybrid_allocation_research_"
    ):
        raise Error("New separate top-level hybrid_allocation_research_ sidecar required")
    if sidecar == source or sidecar in source.parents or source in sidecar.parents:
        raise Error("New sidecar must not overlap preserved studies")
    if auth is not None and (auth.is_relative_to(sidecar) or auth.is_relative_to(source)):
        raise Error("Authorization must stay outside study trees")


def _capture_verified_closure(source):
    """Observe, never replace, the complete immutable preserved verification.

    Only current-worktree files read by named verification/hash/metadata APIs
    are recorded. OOS JSON provenance is allowed; OOS CSV/NPZ prices are not.
    The pre-existing Python profiler is restored even when verification fails.
    """
    captured = set()
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event != "call":
            return
        name = frame.f_code.co_name
        value = None
        if name in {"sha256_file", "_read_seal", "_json_file"}:
            value = frame.f_locals.get("path", frame.f_locals.get("root"))
        elif name == "read_text" and "pathlib" in frame.f_code.co_filename:
            value = frame.f_locals.get("self")
        elif name == "read_csv":
            value = frame.f_locals.get("filepath_or_buffer")
        if not isinstance(value, (str, Path)):
            return
        p = safe_path(value)
        if not p.is_relative_to(ROOT) or not p.is_file():
            return
        if p.suffix.lower() in {".csv", ".npz", ".parquet"} and any(
            x in p.parts for x in ("historical-oos", "historical_oos")
        ):
            raise Error("Preserved verification unexpectedly attempted later numeric price files")
        if (
            p.name == ".env"
            or "credential" in p.name.lower()
            or p.suffix.lower() in {".pem", ".key"}
        ):
            raise Error("Unrelated credential file must never enter study bindings")
        captured.add(p)

    try:
        sys.setprofile(observe)
        validation = multi.verify_results(source, "validation")
    finally:
        sys.setprofile(previous)
    # verify_results(validation) has transitively checked its freeze, development
    # seal/selection locks, all portfolio parent sources/stages and bounded inputs.
    frozen = custody._read_seal(source / "freeze.json")
    dev = custody._read_seal(source / "development/seal.json")
    lock = custody._read_seal(source / "locked_selections.json")
    old_root = safe_path(frozen["preserved_portfolio"])
    old = custody._read_seal(old_root / "freeze.json")
    old_lock = custody._read_seal(old_root / "locked_allocation.json")
    for p in (
        source / "freeze.json",
        source / "development/seal.json",
        source / "validation/seal.json",
        source / "locked_selections.json",
        old_root / "freeze.json",
        old_root / "development/seal.json",
        old_root / "validation/seal.json",
        old_root / "locked_allocation.json",
    ):
        captured.add(safe_path(p))
    return (
        frozen,
        old,
        old_lock,
        {"development": dev, "validation": validation, "locks": lock},
        captured,
    )


def _file_bindings(paths):
    return [
        {
            "path": str(safe_path(p)),
            "bytes": safe_path(p).stat().st_size,
            "sha256": custody.sha256_file(safe_path(p)),
        }
        for p in sorted(set(paths), key=str)
    ]


def _verify_files(rows):
    if not rows or len({r.get("path") for r in rows}) != len(rows):
        raise Error("Complete unique immutable dependency closure required")
    for row in rows:
        if set(row) != {"path", "bytes", "sha256"}:
            raise Error("Dependency file binding changed")
        p = safe_path(row["path"])
        if (
            not p.is_relative_to(ROOT)
            or not p.is_file()
            or p.stat().st_size != row["bytes"]
            or custody.sha256_file(p) != row["sha256"]
        ):
            raise Error("Captured protected source/metadata/result/data file changed")


def _pair_definitions(membership):
    from trend_following import research_hybrid_allocation_engine as engine

    tf = [trend.Candidate.from_dict(r["parameters"]) for r in membership[:12]]
    mrs = [mr.Candidate.from_dict(r["parameters"]) for r in membership[12:]]
    pairs = engine.pair_grid(tf, mrs)
    if len(pairs) != 384 or len({p.pair_id for p in pairs}) != 384:
        raise Error("Exactly 384 unchanged Cartesian pairs required")
    return [
        {
            "pair_id": p.pair_id,
            "tf_candidate_id": p.tf.candidate_id,
            "mr_candidate_id": p.mr.candidate_id,
            "tf_index": p.tf_index,
            "mr_index": p.mr_index,
        }
        for p in pairs
    ]


def freeze_study(
    sidecar=DEFAULT_SIDECAR,
    *,
    authorization,
    config_path=DEFAULT_CONFIG,
    multi_scale=DEFAULT_MULTI_SCALE,
    protected=DEFAULT_PROTECTED,
):
    sidecar, auth, config, source, protected_path = map(
        safe_path, (sidecar, authorization, config_path, multi_scale, protected)
    )
    _separate(sidecar, source, auth)
    load_config(config)
    verify_authorization(auth)
    protected_rows = _protected(protected_path)
    with custody._attempt(sidecar, "freeze"):
        if (sidecar / "freeze.json").exists():
            return verify_study(sidecar)
        inherited, old, old_lock, seals, paths = _capture_verified_closure(source)
        membership = preserved.validate_membership(old["membership"])
        if inherited["membership"] != membership:
            raise Error("Preserved portfolio and multi-scale membership differ")
        devroot = safe_path(old["development_adapter"]["source_study"]) / "prepared/development"
        valroot = safe_path(old["validation_packet"])
        # Explicit allowlisted physically bounded packets, never a cache scan.
        paths.update(devroot / n for n in (*fixed.PACKET_FILES, "manifest.json"))
        paths.update(valroot / n for n in (*prefix.PACKET_FILES, "manifest.json", "freeze.json"))
        paths.update(ROOT / n for n in protected_rows)
        paths.update(ROOT / n for n in SOURCE_FILES)
        paths.update((auth, config, protected_path))
        pairs = _pair_definitions(membership)
        payload = {
            "schema": "qqq_hybrid_allocation_freeze",
            "config_path": str(config),
            "config_sha256": custody.sha256_file(config),
            "configuration_sha256": custody.canonical_hash(approved_config()),
            "authorization": {"path": str(auth), "sha256": custody.sha256_file(auth)},
            "protected_snapshot": {
                "path": str(protected_path),
                "sha256": custody.sha256_file(protected_path),
                "count": len(protected_rows),
            },
            "preserved_multi_scale": str(source),
            "preserved_portfolio": inherited["preserved_portfolio"],
            "preserved_multi_scale_seals": {k: custody.canonical_hash(v) for k, v in seals.items()},
            "preserved_portfolio_freeze_sha256": custody.canonical_hash(old),
            "independent_locked_allocation": old_lock["allocation"],
            "membership": membership,
            "membership_sha256": custody.canonical_hash(membership),
            "pairs": pairs,
            "pairs_sha256": custody.canonical_hash(pairs),
            "principal_definitions": principal_definitions(),
            "fixed_regime_rule": REGIME_RULE,
            "controls": control_ids(),
            "development_adapter": old["development_adapter"],
            "validation_packet": old["validation_packet"],
            "validation_metadata": old["validation_metadata"],
            "source_hashes": {n: custody.sha256_file(ROOT / n) for n in SOURCE_FILES},
            "dependency_files": _file_bindings(paths),
            "runtime": custody._runtime(),
            "cash": fixed.cash_policy(),
            "periods": PERIODS,
            "initial_cash": 1000.0,
            "commission_bps": 1.0,
            "slippage_bps_scenarios": list(SLIPPAGES),
            "hindsight_conditioned_membership": True,
            "historical_oos_authorized": False,
            "OOS_numeric_rows_parsed": 0,
            "closure_observer": "read_only_profile_named_hash_metadata_and_derived_result_reads_restored_finally",
            "frozen_at": custody._utc_now(),
        }
        _protected(protected_path)
        custody._seal(sidecar / "freeze.json", payload)
        return payload


def verify_study(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    frozen = custody._read_seal(sidecar / "freeze.json")
    auth, config, source, protected = map(
        safe_path,
        (
            frozen["authorization"]["path"],
            frozen["config_path"],
            frozen["preserved_multi_scale"],
            frozen["protected_snapshot"]["path"],
        ),
    )
    _separate(sidecar, source, auth)
    load_config(config)
    authorization(auth)
    _protected(protected)
    _verify_files(frozen["dependency_files"])
    old = custody._read_seal(safe_path(frozen["preserved_portfolio"]) / "freeze.json")
    for key in ("membership", "development_adapter", "validation_packet", "validation_metadata"):
        if custody.canonical_hash(frozen[key]) != custody.canonical_hash(old[key]):
            raise Error("New packet/membership binding differs from verified preserved portfolio")
    for path_key, digest_key in (("config_path", "config_sha256"),):
        if custody.sha256_file(safe_path(frozen[path_key])) != frozen[digest_key]:
            raise Error("Frozen configuration file binding changed")
    if (
        custody.sha256_file(auth) != frozen["authorization"]["sha256"]
        or custody.sha256_file(protected) != frozen["protected_snapshot"]["sha256"]
    ):
        raise Error("Frozen authorization/protected snapshot file changed")
    expected = {
        "schema": "qqq_hybrid_allocation_freeze",
        "configuration_sha256": custody.canonical_hash(approved_config()),
        "principal_definitions": principal_definitions(),
        "fixed_regime_rule": REGIME_RULE,
        "controls": control_ids(),
        "runtime": custody._runtime(),
        "cash": fixed.cash_policy(),
        "periods": PERIODS,
        "initial_cash": 1000.0,
        "commission_bps": 1.0,
        "slippage_bps_scenarios": list(SLIPPAGES),
        "historical_oos_authorized": False,
        "OOS_numeric_rows_parsed": 0,
        "hindsight_conditioned_membership": True,
    }
    if any(
        custody.canonical_hash(frozen.get(k)) != custody.canonical_hash(v)
        for k, v in expected.items()
    ):
        raise Error("Frozen hybrid configuration/cash/runtime/stage contract changed")
    preserved.validate_membership(frozen["membership"])
    if (
        frozen["membership_sha256"] != custody.canonical_hash(frozen["membership"])
        or frozen["pairs"] != _pair_definitions(frozen["membership"])
        or frozen["pairs_sha256"] != custody.canonical_hash(frozen["pairs"])
    ):
        raise Error("Frozen hybrid membership/Cartesian pairs changed")
    for name, digest in frozen["source_hashes"].items():
        if name not in SOURCE_FILES or custody.sha256_file(safe_path(ROOT / name)) != digest:
            raise Error("New frozen hybrid source changed")
    if set(frozen["source_hashes"]) != set(SOURCE_FILES):
        raise Error("Incomplete hybrid source closure")
    mr_val._chronology(frozen["frozen_at"])
    return frozen


def load_packet(frozen, stage, *, sidecar=DEFAULT_SIDECAR):
    if stage not in PERIODS:
        raise Error("Only development and validation; 2019 onward remains closed")
    verified = verify_study(sidecar)
    if custody.canonical_hash(frozen) != custody.canonical_hash(verified):
        raise Error("Caller packet contract differs from verified active freeze")
    if stage == "development":
        packet = fixed.load_development_packet(frozen["development_adapter"])
    else:
        verify_locked_selections(sidecar)  # Before any new validation numeric price read.
        daily, bars, calendar, coverage, blackouts, _ = prefix.read_validation_prefix(
            safe_path(frozen["validation_packet"]),
            expected_manifest_sha256=frozen["validation_metadata"]["manifest_sha256"],
        )
        config = copy.deepcopy(custody.load_config(frozen["development_adapter"]["config_path"]))
        config["segments"]["validation"] = dict(PERIODS["validation"])
        cash = fixed.constant_cash_observations(
            calendar.market_open.iloc[0], daily.daily_price_available_at.iloc[-1]
        )
        custody._check_prefix(daily, bars, calendar, cash, "validation", config)
        custody._quality_gate(
            config, coverage, frozen["validation_metadata"]["manifest"]["gap_authorization"]
        )
        packet = (daily, bars, calendar, coverage, blackouts, cash, config)
    _assert_packet_bounds(packet, stage)
    return packet


def _assert_packet_bounds(packet, stage):
    if stage not in PERIODS:
        raise Error("No later numeric packet allowed")
    daily, bars, calendar, coverage, blackouts, cash, _ = packet
    end = PERIODS[stage]["end"]
    for frame in (daily, bars, calendar, coverage):
        if (
            frame.empty
            or "session" not in frame
            or frame.session.astype(str).max() > end
            or frame.session.astype(str).min() < "1999-11-01"
        ):
            raise Error(
                "Physically bounded packet has a forbidden later or pre-initialization session"
            )
    cutoff = (
        pd.Timestamp(end, tz="America/New_York") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    ).tz_convert("UTC")
    for frame, columns in (
        (daily, ("daily_price_available_at",)),
        (bars, ("bar_start", "bar_end")),
        (calendar, ("market_open", "market_close")),
        (coverage, ("bar_start", "bar_end")),
        (blackouts, ("start", "end")),
    ):
        if frame is None or frame.empty:
            continue
        for column in columns:
            if column in frame and pd.to_datetime(frame[column], utc=True).max() > cutoff:
                raise Error("Forbidden later numeric timestamp in bounded research packet")
    if cash.empty or not np.isfinite(cash.annual_rate).all() or not cash.annual_rate.eq(0.03).all():
        raise Error("Fixed 3% nominal cash assumption required")


def control_ids():
    return [
        "portfolio_tf_050",
        "portfolio_tf_070",
        "portfolio_tf_090",
        PURE_TREND_ID,
        "portfolio_tf_000",
        "buy_and_hold",
        "cash",
        GATED_CONTROL_ID,
    ]


def select_architecture_winners(metrics):
    base = metrics.loc[metrics.scenario_slippage_bps.eq(3.0)]
    definitions = {r["portfolio_id"]: r for r in principal_definitions()}
    if (
        len(base) != 8
        or base.portfolio_id.duplicated().any()
        or set(base.portfolio_id) != set(definitions)
    ):
        raise Error("Selection requires all eight unique predefined baseline portfolios")
    for r in base.itertuples(index=False):
        d = definitions[r.portfolio_id]
        if r.architecture != d["architecture"] or (
            d["trend_allocation"] is not None and r.trend_allocation != d["trend_allocation"]
        ):
            raise Error("Architecture allocation metadata changed")
    selected = []
    for architecture in ("A", "B"):
        ids = [
            r["portfolio_id"]
            for r in principal_definitions()
            if r["architecture"] in (architecture, "pure_trend")
        ]
        finite = base.loc[base.portfolio_id.isin(ids) & np.isfinite(base.cash_excess_sharpe)]
        if finite.empty:
            raise Error(f"No finite baseline development score for architecture {architecture}")
        best = finite.cash_excess_sharpe.max()
        pid = min(finite.loc[finite.cash_excess_sharpe.ge(best - TOLERANCE), "portfolio_id"])
        selected.append(
            {
                "architecture": architecture,
                "portfolio_id": pid,
                "trend_allocation": definitions[pid]["trend_allocation"],
            }
        )
    return selected


def seal_locked_selections(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    dev = verify_results(sidecar, "development")
    selections = select_architecture_winners(
        pd.read_csv(sidecar / "development/metrics.csv", float_precision="round_trip")
    )
    payload = {
        "schema": "qqq_hybrid_allocation_locks",
        "development_result_sha256": custody.canonical_hash(dev),
        "selections": selections,
        "fixed_regime_rule": REGIME_RULE,
        "baseline_slippage_bps": 3.0,
        "commission_bps": 1.0,
        "validation_reselection": False,
        "historical_oos_authorized": False,
    }
    path = sidecar / "locked_selections.json"
    if path.exists():
        return verify_locked_selections(sidecar)
    payload["sealed_at"] = custody._utc_now()
    mr_val._chronology(dev["finished_at"], payload["sealed_at"])
    custody._seal(path, payload)
    return payload


def verify_locked_selections(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    dev = verify_results(sidecar, "development")
    lock = custody._read_seal(sidecar / "locked_selections.json")
    expected = {
        "schema": "qqq_hybrid_allocation_locks",
        "development_result_sha256": custody.canonical_hash(dev),
        "selections": select_architecture_winners(
            pd.read_csv(sidecar / "development/metrics.csv", float_precision="round_trip")
        ),
        "fixed_regime_rule": REGIME_RULE,
        "baseline_slippage_bps": 3.0,
        "commission_bps": 1.0,
        "validation_reselection": False,
        "historical_oos_authorized": False,
    }
    if {k: v for k, v in lock.items() if k != "sealed_at"} != expected:
        raise Error("Hybrid locks/fixed ER rule differ from sealed development contract")
    mr_val._chronology(dev["finished_at"], lock["sealed_at"])
    return lock


def _load_preserved_sleeves(directory, slip, frozen):
    """Restore already sealed event-aware paths; never rerun an old strategy."""
    metrics = pd.read_csv(
        directory / f"constituent_metrics_s{slip:g}.csv", float_precision="round_trip"
    )
    records = pd.read_csv(directory / f"records_s{slip:g}.csv", float_precision="round_trip")
    if not records.empty:
        records["timestamp"] = pd.to_datetime(records.timestamp, utc=True)
        records["phase"] = np.where(records.reason.eq("terminal_liquidation"), "terminal", "open")
    with np.load(directory / f"constituent_s{slip:g}_daily.npz", allow_pickle=False) as z:
        ids = tuple(z["candidate_id"])
        if list(ids) != [r["candidate_id"] for r in frozen["membership"]] or list(
            metrics.candidate_id
        ) != list(ids):
            raise Error("Reused recorded constituent membership changed")
        fills = []
        for j, (timestamp, phase) in enumerate(
            zip(z["fill_timestamp"], z["fill_phase"], strict=True)
        ):
            timestamp = pd.Timestamp(str(timestamp))
            frame = records.loc[records.timestamp.eq(timestamp) & records.phase.eq(phase)].copy()
            if frame.empty:
                raise Error("Preserved simultaneous fill group lacks exact private records")
            fills.append(
                portfolio.FillGroup(
                    timestamp,
                    str(phase),
                    float(z["fill_reference_price"][j]),
                    z["fill_equity_before"][j].copy(),
                    frame,
                )
            )
        result = portfolio.SleevePaths(
            ids,
            pd.DatetimeIndex(pd.to_datetime(z["timestamp"], utc=True)),
            z["equity"].copy(),
            z["returns"].copy(),
            z["cash_returns"].copy(),
            z["gross_terminal_equity"].copy(),
            pd.DatetimeIndex(pd.to_datetime(z["event_timestamp"], utc=True)),
            z["event_equity"].copy(),
            z["event_invested_notional"].copy(),
            z["event_long"].copy(),
            tuple(fills),
            pd.Timestamp(str(z["segment_start"])),
            pd.Timestamp(str(z["segment_end"])),
            float(z["initial_cash"]),
            tuple(metrics.to_dict("records")),
        )
    portfolio._validate_paths(result)
    return result


def _read_existing_controls(frozen, stage, slip, times, cash):
    directory = safe_path(frozen["preserved_portfolio"]) / stage
    sleeves = _load_preserved_sleeves(directory, slip, frozen)
    if not sleeves.times.equals(times) or not np.array_equal(sleeves.cash_returns, cash):
        raise Error("Reused controls need exact daily clocks and cash references")
    allowed = control_ids()[:-1]
    old_metrics = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    old_metrics = old_metrics.loc[old_metrics.scenario_slippage_bps.eq(slip)]
    rows, paths = [], {}
    with np.load(directory / f"allocation_s{slip:g}_daily.npz", allow_pickle=False) as saved:
        if not pd.DatetimeIndex(pd.to_datetime(saved["timestamp"], utc=True)).equals(
            times
        ) or not np.array_equal(saved["cash_returns"], cash):
            raise Error("Saved independent allocation clocks/cash differ")
        saved_ids = list(saved["portfolio_id"])
        for cid in allowed[:5]:
            weight = int(cid.rsplit("_", 1)[1]) / 100
            allocation = portfolio.Allocation(cid, weight)
            result = portfolio.evaluate_portfolio(sleeves, allocation)
            if cid in saved_ids:
                j = saved_ids.index(cid)
                for k, value in (
                    ("equity", result.daily_equity),
                    ("returns", result.daily_returns),
                    ("exposure", result.daily_exposure),
                ):
                    multi._assert_close(
                        saved[k][:, j], value, f"Reused {cid} {k} differs from sealed paths"
                    )
                prior = old_metrics.loc[old_metrics.portfolio_id.eq(cid)]
                if len(prior) != 1:
                    raise Error("One preserved allocation metric per saved ID required")
                for k in (
                    "cash_excess_sharpe",
                    "cagr",
                    "max_drawdown",
                    "exposure_fraction",
                    "annualized_turnover",
                    "terminal_equity",
                    "cost_drag_return",
                ):
                    multi._assert_close(
                        prior.iloc[0][k], result.metrics[k], f"Reused {cid} {k} differs"
                    )
            row = {
                **result.metrics,
                "control_origin": "sealed_44_path_arithmetic_no_old_signal_rerun",
                "historically_saved_allocation": cid in saved_ids,
            }
            active = [
                m for m, w in zip(sleeves.metrics, allocation.weights(), strict=True) if w > 0
            ]
            trips = sum(m["trade_count"] for m in active)
            row["average_gross_sleeve_holding_hours"] = (
                sum(m["trade_count"] * m["average_holding_hours"] for m in active) / trips
                if trips
                else 0.0
            )
            rows.append(row)
            paths[cid] = {
                "equity": result.daily_equity.to_numpy(),
                "returns": result.daily_returns.to_numpy(),
                "exposure": result.daily_exposure.to_numpy(),
            }
    benchmarks = pd.read_csv(directory / "benchmark_metrics.csv", float_precision="round_trip")
    for mode in ("buy_and_hold", "cash"):
        frame = pd.read_csv(directory / f"{mode}_s{slip:g}_daily.csv", float_precision="round_trip")
        if not pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)).equals(times):
            raise Error("Passive reused clock mismatch")
        multi._assert_close(frame.cash_return, cash, "Passive reused cash mismatch")
        paths[mode] = {
            "equity": frame.equity.to_numpy(),
            "returns": frame["return"].to_numpy(),
            "exposure": frame.exposure.to_numpy(),
        }
        selected = benchmarks.loc[
            benchmarks.portfolio_id.eq(mode) & benchmarks.scenario_slippage_bps.eq(slip)
        ]
        if len(selected) != 1:
            raise Error("Exactly one passive outcome per cost required")
        rows.extend(selected.to_dict("records"))
    if len(rows) != 7 or list(paths) != allowed:
        raise Error("Exactly seven predefined read-only controls required")
    return rows, paths


def _baseline_parity(cache, shadows, tf, mrs, frozen, stage, cash):
    kwargs = dict(
        **PERIODS[stage],
        cash_observations=cash,
        commission_bps=1.0,
        slippage_bps=3.0,
        initial_cash=1000.0,
        record=True,
    )
    tf_parts = quality._run(cache.tf, tf, **kwargs)
    mr_parts = mr_engine.run(cache.mr, mrs, **kwargs)
    old = safe_path(frozen["preserved_portfolio"]) / stage / "constituent_s3_daily.npz"
    multi._compare_preserved_sleeves(tf_parts, tf, old)
    multi._compare_preserved_sleeves(mr_parts, mrs, old)
    # Validate cost-independent TF filled states at each next safe opening time.
    trades = pd.DataFrame(tf_parts[6])
    state = np.zeros(len(tf), dtype=np.int8)
    expected = []
    lookup = {c.candidate_id: j for j, c in enumerate(tf)}
    for timestamp in shadows.open_times:
        at = (
            trades.loc[pd.to_datetime(trades.timestamp, utc=True).eq(timestamp)]
            if not trades.empty
            else trades
        )
        for row in at.itertuples(index=False):
            state[lookup[row.candidate_id]] = 1 if row.side == "buy" else 0
        expected.append(state.copy())
    if not np.array_equal(np.asarray(expected), shadows.tf_open_state):
        raise Error("Hybrid TF shadow filled states differ from unchanged TF engine")
    return {
        "baseline_all12_TF_all32_MR_paths_match": True,
        "TF_shadow_open_states_match": True,
        "new_outcomes_not_yet_published": True,
    }


def _save_npz(path, ids, times, equity, returns, cash, exposure):
    np.savez_compressed(
        path,
        portfolio_id=np.array(ids, dtype=str),
        timestamp=np.array([t.isoformat() for t in times], dtype=str),
        equity=equity,
        returns=returns,
        cash_returns=cash,
        exposure=exposure,
    )


def _save_pair_paths(path, result, frozen):
    """Explicit private pair wealth; A paths are attributed, not extra funded cores."""
    pairs = frozen["pairs"]
    n = len(pairs)
    tf_indices = np.array([r["tf_index"] for r in pairs])
    core = result.account_equity[:, tf_indices]
    tactical = result.account_equity[:, 12 : 12 + n]
    core_notional = core * result.account_exposure[:, tf_indices]
    tactical_notional = tactical * result.account_exposure[:, 12 : 12 + n]
    groups = np.asarray(result.account_groups)
    eq, exp, ids, pair_ids, roles = [], [], [], [], []
    for a in ALLOCATIONS:
        wealth = core + (1 - a) * (tactical - core)
        notional = a * core_notional + (1 - a) * tactical_notional
        eq.append(wealth)
        exp.append(notional / wealth)
        ids.extend([f"hybrid_core_{int(a * 100):03d}"] * n)
        pair_ids.extend(r["pair_id"] for r in pairs)
        roles.extend(["attributed_pair_path_shared_12_funded_core_accounts"] * n)
    for group, pid in (
        ("B050", "hybrid_blend_050"),
        ("B070", "hybrid_blend_070"),
        ("B090", "hybrid_blend_090"),
        ("C", REGIME_ID),
    ):
        columns = np.flatnonzero(groups == group)
        if len(columns) != n:
            raise Error("Paired financial account group changed frozen count")
        eq.append(result.account_equity[:, columns])
        exp.append(result.account_exposure[:, columns])
        ids.extend([pid] * n)
        pair_ids.extend(r["pair_id"] for r in pairs)
        roles.extend(["actually_funded_pair_account"] * n)
    wealth = np.column_stack(eq)
    returns = wealth / np.vstack([np.full(7 * n, 1000.0), wealth[:-1]]) - 1
    np.savez_compressed(
        path,
        portfolio_id=np.array(ids, dtype=str),
        pair_id=np.array(pair_ids, dtype=str),
        path_role=np.array(roles, dtype=str),
        timestamp=np.array([t.isoformat() for t in result.daily_times], dtype=str),
        equity=wealth,
        returns=returns,
        exposure=np.column_stack(exp),
        cash_returns=result.cash_returns,
    )


def bootstrap_comparisons(comparisons, cash, *, replicates=2000, blocks=(5, 20, 60), seed=20261009):
    """Common circular block draws; duplicate path-pairs retain semantic aliases."""
    cash = np.asarray(cash, dtype=float)
    unique = []
    arrays = []

    def column(values):
        values = np.asarray(values, dtype=float)
        if values.shape != cash.shape or not np.isfinite(values).all():
            raise Error("Bootstrap requires finite exact matching daily paths")
        for i, a in enumerate(arrays):
            if np.array_equal(a, values):
                return i
        arrays.append(values)
        return len(arrays) - 1

    for c in comparisons:
        s, r = column(c["subject_returns"]), column(c["reference_returns"])
        found = next((x for x in unique if x["columns"] == (s, r)), None)
        alias = {
            "architecture": c["architecture"],
            "subject_id": c["subject_id"],
            "reference_id": c["reference_id"],
        }
        if found is None:
            unique.append({"columns": (s, r), "aliases": [alias]})
        else:
            found["aliases"].append(alias)
    if not arrays or len(cash) < 2 or not np.isfinite(cash).all() or replicates <= 0:
        raise Error("Valid nonempty paired bootstrap inputs required")
    matrix = np.column_stack(arrays)
    observed = np.array(
        [portfolio.cash_excess_sharpe(matrix[:, j], cash) for j in range(matrix.shape[1])]
    )
    out = []
    n = len(cash)
    for block in blocks:
        if block <= 0:
            raise Error("Positive bootstrap block required")
        rng = np.random.default_rng(np.random.SeedSequence([seed, block]))
        draws = np.empty((replicates, len(arrays)))
        for i in range(replicates):
            starts = rng.integers(0, n, size=int(np.ceil(n / block)))
            ix = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
            excess = matrix[ix] - cash[ix, None]
            sd = np.std(excess, axis=0, ddof=1)
            draws[i] = np.divide(
                np.sqrt(252) * np.mean(excess, axis=0),
                sd,
                out=np.full_like(sd, np.nan),
                where=sd > 0,
            )
        for c in unique:
            s, r = c["columns"]
            diff = draws[:, s] - draws[:, r]
            finite = diff[np.isfinite(diff)]
            low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
            first = c["aliases"][0]
            out.append(
                {
                    **first,
                    "comparison_aliases": json.dumps(c["aliases"], sort_keys=True),
                    "block_length": block,
                    "replicates": replicates,
                    "finite_replicates": len(finite),
                    "seed": seed,
                    "observed_sharpe_difference": observed[s] - observed[r],
                    "ci_lower": low,
                    "ci_upper": high,
                    "confidence_level": 0.95,
                }
            )
    return pd.DataFrame(out)


def _required_files(stage):
    files = {
        "metrics.csv",
        "controls_metrics.csv",
        "annual_returns.csv",
        "correlations.csv",
        "preflight.json",
        "regime_history.csv",
        "shadow_daily.npz",
        "shadow_trades.csv",
        "shadow_diagnostics.csv",
        "shadow_trace.npz",
    }
    for slip in SLIPPAGES:
        files |= {
            f"{kind}_s{slip:g}{suffix}"
            for kind, suffix in (
                ("principal", "_daily.npz"),
                ("controls", "_daily.npz"),
                ("account", "_daily.npz"),
                ("pair", "_daily.npz"),
                ("account_metrics", ".csv"),
                ("pair_metrics", ".csv"),
                ("records", ".csv"),
                ("commands", ".csv"),
                ("diagnostics", ".csv"),
                ("event_portfolios", ".npz"),
            )
        }
    if stage == "validation":
        files.add("bootstrap.csv")
    return files


def verify_results(sidecar=DEFAULT_SIDECAR, stage="development"):
    if stage not in PERIODS:
        raise Error("No OOS result route")
    sidecar = safe_path(sidecar)
    frozen = verify_study(sidecar)
    seal = custody._read_seal(sidecar / stage / "seal.json")
    lock = verify_locked_selections(sidecar) if stage == "validation" else None
    expected = {
        "schema": "qqq_hybrid_allocation_stage_results",
        "stage": stage,
        "freeze_sha256": custody.canonical_hash(frozen),
        "membership_sha256": frozen["membership_sha256"],
        "pairs_sha256": frozen["pairs_sha256"],
        "period": PERIODS[stage],
        "principal_ids": [r["portfolio_id"] for r in principal_definitions()],
        "control_ids": control_ids(),
        "pair_count": 384,
        "principal_count": 8,
        "pair_result_count_per_cost": 2688,
        "costs": list(SLIPPAGES),
        "frozen_at": frozen["frozen_at"],
        "historical_oos_authorized": False,
        "OOS_numeric_rows_parsed": 0,
        "original_baseline_parity_verified": True,
    }
    if lock:
        expected.update(
            selection_lock_sha256=custody.canonical_hash(lock),
            selection_sealed_at=lock["sealed_at"],
        )
    if any(
        custody.canonical_hash(seal.get(k)) != custody.canonical_hash(v)
        for k, v in expected.items()
    ):
        raise Error("Hybrid sealed stage contract changed")
    mr_val._chronology(
        frozen["frozen_at"],
        *([lock["sealed_at"]] if lock else []),
        seal["started_at"],
        seal["finished_at"],
    )
    required = _required_files(stage)
    if set(seal.get("files", {})) != required or {
        p.name for p in (sidecar / stage).iterdir()
    } != required | {"seal.json"}:
        raise Error("Missing or extra sealed hybrid artifacts")
    for name, digest in seal["files"].items():
        if custody.sha256_file(safe_path(sidecar / stage / name)) != digest:
            raise Error("Sealed hybrid artifact changed")
    return seal


def _validation_comparisons(lock, principal, controls):
    combined = {**controls, **principal}
    out = []
    for selected in lock["selections"]:
        arch, pid, a = (
            selected["architecture"],
            selected["portfolio_id"],
            selected["trend_allocation"],
        )
        matching = (
            f"portfolio_tf_{int(a * 100):03d}"
            if arch == "A"
            else (PURE_TREND_ID if a == 1 else f"hybrid_core_{int(a * 100):03d}")
        )
        for ref in (PURE_TREND_ID, "buy_and_hold", matching):
            out.append(
                {
                    "architecture": arch,
                    "subject_id": pid,
                    "reference_id": ref,
                    "subject_returns": combined[pid]["returns"],
                    "reference_returns": combined[ref]["returns"],
                }
            )
    for ref in (PURE_TREND_ID, "buy_and_hold", GATED_CONTROL_ID):
        out.append(
            {
                "architecture": "C",
                "subject_id": REGIME_ID,
                "reference_id": ref,
                "subject_returns": combined[REGIME_ID]["returns"],
                "reference_returns": combined[ref]["returns"],
            }
        )
    return out


def _evaluate_stage(sidecar, stage):
    from trend_following import research_hybrid_allocation_engine as engine

    sidecar = safe_path(sidecar)
    if stage not in PERIODS:
        raise Error("No historical OOS execution route")
    with custody._attempt(sidecar, stage) as attempt:
        frozen = verify_study(sidecar)
        lock = verify_locked_selections(sidecar) if stage == "validation" else None
        if (sidecar / stage).exists():
            raise Error("Stage already sealed; never overwrite or repeat")
        started = custody._utc_now()
        mr_val._chronology(frozen["frozen_at"], *([lock["sealed_at"]] if lock else []), started)
        daily, bars, calendar, coverage, blackouts, cash, config = load_packet(
            frozen, stage, sidecar=sidecar
        )
        cache = engine.build_cache(
            daily,
            bars,
            calendar,
            coverage=coverage,
            blackouts=blackouts,
            quality_policy=custody._quality_policy(config),
        )
        if cache.test_gate_override is not None:
            raise Error("Production cannot use synthetic gate overrides")
        tf = [trend.Candidate.from_dict(r["parameters"]) for r in frozen["membership"][:12]]
        mrs = [mr.Candidate.from_dict(r["parameters"]) for r in frozen["membership"][12:]]
        shadows = engine.build_shadow_paths(cache, tf, mrs, **PERIODS[stage], record=True)
        if shadows.test_only:
            raise Error("Production cannot use synthetic shadow paths")
        if list(shadows.pair_ids) != [r["pair_id"] for r in frozen["pairs"]]:
            raise Error("Shadow pair membership changed")
        parity = _baseline_parity(cache, shadows, tf, mrs, frozen, stage, cash)
        output = attempt / stage
        output.mkdir()
        shadows.shadow_trades.to_csv(output / "shadow_trades.csv", index=False)
        shadows.diagnostics.to_csv(output / "shadow_diagnostics.csv", index=False)
        np.savez_compressed(output / "shadow_trace.npz", **shadows.trace)
        np.savez_compressed(
            output / "shadow_daily.npz",
            pair_id=np.array(shadows.pair_ids, dtype=str),
            open_timestamp=np.array([t.isoformat() for t in shadows.open_times], dtype=str),
            tf_open_state=shadows.tf_open_state,
            mr_open_state=shadows.mr_open_state,
            mode_open_state=shadows.mode_open_state,
            er_open_valid=shadows.er_open_valid,
            open_bar_indices=shadows.open_bar_indices,
            tf_candidate_id=np.array(shadows.tf_ids, dtype=str),
            mr_candidate_id=np.array(shadows.mr_ids, dtype=str),
            event_timestamp=np.array([t.isoformat() for t in shadows.clock], dtype=str),
            tf_open_decision_at=shadows.tf_open_decision_at,
            mr_open_decision_at=shadows.mr_open_decision_at,
            tf_open_decision_bar=shadows.tf_open_decision_bar,
            mr_open_decision_bar=shadows.mr_open_decision_bar,
            mode_open_decision_at=shadows.mode_open_decision_at,
            mode_open_decision_bar=shadows.mode_open_decision_bar,
            mr_open_cause=shadows.mr_open_cause,
            mode_event_state=shadows.mode_event_state,
            er_event_valid=shadows.er_event_valid,
        )
        cache.finalized_er.rename("efficiency_ratio").rename_axis("session").reset_index().to_csv(
            output / "regime_history.csv", index=False
        )
        rows, control_rows, annual = [], [], []
        baseline = None
        for slip in SLIPPAGES:
            result = engine.run(
                cache,
                shadows,
                cash_observations=cash,
                commission_bps=1.0,
                slippage_bps=slip,
                initial_cash=1000.0,
                record=True,
            )
            ids = list(result.portfolio_ids)
            if (
                set(ids)
                != {r["portfolio_id"] for r in principal_definitions()} | {GATED_CONTROL_ID}
                or len(ids) != 9
            ):
                raise Error("Exactly eight principals plus the pure gated MR control required")
            old_tf = quality._run(
                cache.tf,
                tf,
                **PERIODS[stage],
                cash_observations=cash,
                commission_bps=1.0,
                slippage_bps=slip,
                initial_cash=1000.0,
                record=True,
            )
            multi._compare_preserved_sleeves(
                old_tf,
                tf,
                safe_path(frozen["preserved_portfolio"])
                / stage
                / f"constituent_s{slip:g}_daily.npz",
            )
            multi._assert_close(
                result.account_equity[:, :12],
                old_tf[2],
                "Funded twelve core accounts differ from unchanged TF engine",
            )
            pd.DataFrame(result.account_metrics).to_csv(
                output / f"account_metrics_s{slip:g}.csv", index=False
            )
            pair_metrics = result.pair_metrics.copy()
            pair_metrics["pair_path_role"] = np.where(
                pair_metrics.architecture.eq("A"),
                "attributed_pair_path_shared_12_funded_core_accounts",
                "actually_funded_pair_account",
            )
            pair_metrics["scenario_slippage_bps"], pair_metrics["stage"] = slip, stage
            pair_metrics.to_csv(output / f"pair_metrics_s{slip:g}.csv", index=False)
            _save_pair_paths(output / f"pair_s{slip:g}_daily.npz", result, frozen)
            result.trade_rows.to_csv(output / f"records_s{slip:g}.csv", index=False)
            result.command_rows.to_csv(output / f"commands_s{slip:g}.csv", index=False)
            result.diagnostics.to_csv(output / f"diagnostics_s{slip:g}.csv", index=False)
            np.savez_compressed(
                output / f"account_s{slip:g}_daily.npz",
                account_id=np.array(result.account_ids, dtype=str),
                account_group=np.array(result.account_groups, dtype=str),
                timestamp=np.array([t.isoformat() for t in result.daily_times], dtype=str),
                equity=result.account_equity,
                returns=result.account_returns,
                exposure=result.account_exposure,
                gross_terminal_equity=result.account_gross_terminal,
                gross_equity=result.account_gross_equity,
                cash_returns=result.cash_returns,
            )
            np.savez_compressed(
                output / f"event_portfolios_s{slip:g}.npz",
                portfolio_id=np.array(ids, dtype=str),
                timestamp=np.array([t.isoformat() for t in result.event_times], dtype=str),
                equity=result.event_equity,
                exposure=result.event_exposure,
            )
            principal_ids = [r["portfolio_id"] for r in principal_definitions()]
            positions = [ids.index(pid) for pid in principal_ids]
            _save_npz(
                output / f"principal_s{slip:g}_daily.npz",
                principal_ids,
                result.daily_times,
                result.daily_equity[:, positions],
                result.daily_returns[:, positions],
                result.cash_returns,
                result.daily_exposure[:, positions],
            )
            by_id = {r["portfolio_id"]: r for r in result.metrics}
            for d in principal_definitions():
                rows.append(
                    {**by_id[d["portfolio_id"]], **d, "scenario_slippage_bps": slip, "stage": stage}
                )
            old_rows, controls = _read_existing_controls(
                frozen, stage, slip, result.daily_times, result.cash_returns
            )
            pure = ids.index(PURE_TREND_ID)
            multi._assert_close(
                result.daily_equity[:, pure],
                controls[PURE_TREND_ID]["equity"],
                "Pure trend fallback differs from sealed unchanged basket",
            )
            gated = ids.index(GATED_CONTROL_ID)
            controls[GATED_CONTROL_ID] = {
                "equity": result.daily_equity[:, gated],
                "returns": result.daily_returns[:, gated],
                "exposure": result.daily_exposure[:, gated],
            }
            control_rows.extend(
                [{**r, "scenario_slippage_bps": slip, "stage": stage} for r in old_rows]
            )
            control_rows.append(
                {**by_id[GATED_CONTROL_ID], "scenario_slippage_bps": slip, "stage": stage}
            )
            order = control_ids()
            _save_npz(
                output / f"controls_s{slip:g}_daily.npz",
                order,
                result.daily_times,
                np.column_stack([controls[p]["equity"] for p in order]),
                np.column_stack([controls[p]["returns"] for p in order]),
                result.cash_returns,
                np.column_stack([controls[p]["exposure"] for p in order]),
            )
            for pid, path in {
                **controls,
                **{
                    pid: {
                        "equity": result.daily_equity[:, ids.index(pid)],
                        "returns": result.daily_returns[:, ids.index(pid)],
                    }
                    for pid in principal_ids
                },
            }.items():
                table = portfolio.annual_return_table(
                    pd.Series(path["equity"], index=result.daily_times)
                )
                table["portfolio_id"], table["scenario_slippage_bps"], table["stage"] = (
                    pid,
                    slip,
                    stage,
                )
                annual.append(table)
            if slip == 3:
                principal = {
                    pid: {
                        "equity": result.daily_equity[:, ids.index(pid)],
                        "returns": result.daily_returns[:, ids.index(pid)],
                    }
                    for pid in principal_ids
                }
                baseline = (principal, controls, result.cash_returns.copy())
                corr = pd.DataFrame(
                    {
                        pid: path["returns"] - result.cash_returns
                        for pid, path in {**controls, **principal}.items()
                    }
                ).corr()
                corr.rename_axis("portfolio_id").reset_index().to_csv(
                    output / "correlations.csv", index=False
                )
            print(
                f"Completed hybrid {stage}: eight principals, 384 pairs; 1-bp commission + {slip:g}-bps slippage",
                flush=True,
            )
        metrics = pd.DataFrame(rows)
        controls_metrics = pd.DataFrame(control_rows)
        for slip in SLIPPAGES:
            bh = controls_metrics.loc[
                controls_metrics.portfolio_id.eq("buy_and_hold")
                & controls_metrics.scenario_slippage_bps.eq(slip),
                "cash_excess_sharpe",
            ]
            if len(bh) != 1:
                raise Error("Exactly one matched BH per cost required")
            for frame in (metrics, controls_metrics):
                mask = frame.scenario_slippage_bps.eq(slip)
                frame.loc[mask, "BH_cash_excess_sharpe"] = float(bh.iloc[0])
                frame.loc[mask, "Sharpe_difference_vs_BH"] = frame.loc[
                    mask, "cash_excess_sharpe"
                ] - float(bh.iloc[0])
        metrics.to_csv(output / "metrics.csv", index=False)
        controls_metrics.to_csv(output / "controls_metrics.csv", index=False)
        pd.concat(annual, ignore_index=True).to_csv(output / "annual_returns.csv", index=False)
        (output / "preflight.json").write_text(json.dumps(parity, indent=2, allow_nan=False) + "\n")
        if lock:
            bootstrap_comparisons(_validation_comparisons(lock, *baseline[:2]), baseline[2]).to_csv(
                output / "bootstrap.csv", index=False
            )
        else:
            select_architecture_winners(metrics)
        review_derived_results(output, frozen, stage)
        verify_study(sidecar)
        finished = custody._utc_now()
        seal = {
            "schema": "qqq_hybrid_allocation_stage_results",
            "stage": stage,
            "freeze_sha256": custody.canonical_hash(frozen),
            "membership_sha256": frozen["membership_sha256"],
            "pairs_sha256": frozen["pairs_sha256"],
            "period": PERIODS[stage],
            "principal_ids": [r["portfolio_id"] for r in principal_definitions()],
            "control_ids": control_ids(),
            "pair_count": 384,
            "principal_count": 8,
            "pair_result_count_per_cost": 2688,
            "costs": list(SLIPPAGES),
            "frozen_at": frozen["frozen_at"],
            "started_at": started,
            "finished_at": finished,
            "historical_oos_authorized": False,
            "OOS_numeric_rows_parsed": 0,
            "original_baseline_parity_verified": True,
            "files": {p.name: custody.sha256_file(p) for p in sorted(output.iterdir())},
        }
        if lock:
            seal.update(
                selection_lock_sha256=custody.canonical_hash(lock),
                selection_sealed_at=lock["sealed_at"],
            )
        custody._seal(output / "seal.json", seal)
        output.rename(sidecar / stage)
        if stage == "development":
            seal_locked_selections(sidecar)
        return seal


def evaluate_development(sidecar=DEFAULT_SIDECAR):
    return _evaluate_stage(sidecar, "development")


def evaluate_validation(sidecar=DEFAULT_SIDECAR):
    return _evaluate_stage(sidecar, "validation")


def _principal_weights(groups):
    groups = np.asarray(groups)
    weights = []
    for d in principal_definitions() + [
        {"portfolio_id": GATED_CONTROL_ID, "architecture": "gated", "trend_allocation": 0.0}
    ]:
        w = np.zeros(len(groups))
        if d["architecture"] == "A":
            w[groups == "TF"] = d["trend_allocation"] / 12
            w[groups == "GMR"] = (1 - d["trend_allocation"]) / 384
        else:
            label = (
                "B" + d["portfolio_id"].rsplit("_", 1)[1]
                if d["architecture"] == "B"
                else "C"
                if d["architecture"] == "C"
                else "TF"
                if d["architecture"] == "pure_trend"
                else "GMR"
            )
            required = 12 if label == "TF" else 384
            if np.count_nonzero(groups == label) != required:
                raise Error("Actual funded account group count changed")
            w[groups == label] = 1 / required
        if (w < 0).any() or not np.isclose(w.sum(), 1, rtol=0, atol=1e-14):
            raise Error("Frozen initial portfolio capital weights invalid")
        weights.append(w)
    return np.asarray(weights)


def _review_account_aggregation(directory, slip, metrics, frozen):
    """Independent fixed-capital sums and common pre-fill cost/turnover arithmetic."""
    with (
        np.load(directory / f"account_s{slip:g}_daily.npz", allow_pickle=False) as accounts,
        np.load(directory / f"principal_s{slip:g}_daily.npz", allow_pickle=False) as daily,
        np.load(directory / f"event_portfolios_s{slip:g}.npz", allow_pickle=False) as events,
    ):
        weights = _principal_weights(accounts["account_group"])
        ids = [d["portfolio_id"] for d in principal_definitions()] + [GATED_CONTROL_ID]
        if len(accounts["account_id"]) != 1932 or len(set(accounts["account_id"])) != 1932:
            raise Error("Exactly 1,932 unique normalized funded streams required")
        if list(accounts["account_id"][:12]) != [
            r["candidate_id"] for r in frozen["membership"][:12]
        ]:
            raise Error("The twelve actual funded cores changed unchanged TF identity")
        event_times = pd.DatetimeIndex(pd.to_datetime(events["timestamp"], utc=True))
        seconds = np.diff(event_times.asi8) / 1e9
        years = seconds.sum() / (365 * 86400)
        event_ids = list(events["portfolio_id"])
        stats = pd.read_csv(
            directory / f"account_metrics_s{slip:g}.csv", float_precision="round_trip"
        )
        records = pd.read_csv(directory / f"records_s{slip:g}.csv", float_precision="round_trip")
        if not records.empty:
            records["timestamp"] = pd.to_datetime(records.timestamp, utc=True)
        lookup = {str(cid): j for j, cid in enumerate(accounts["account_id"])}
        grouped = list(records.groupby(["timestamp", "phase"]))
        for j, pid in enumerate(ids[:8]):
            w = weights[j]
            total = accounts["equity"] @ w
            multi._assert_close(
                total, daily["equity"][:, j], "Independent actual funded-account wealth sum differs"
            )
            wealth = daily["equity"][:, j]
            ret = daily["returns"][:, j]
            row = metrics.loc[
                metrics.portfolio_id.eq(pid) & metrics.scenario_slippage_bps.eq(slip)
            ].iloc[0]
            event_column = event_ids.index(pid)
            expected = {
                "cagr": (wealth[-1] / 1000.0) ** (1 / years) - 1,
                "max_drawdown": (
                    np.r_[1000.0, wealth] / np.maximum.accumulate(np.r_[1000.0, wealth]) - 1
                ).min(),
                "annualized_volatility": np.std(ret, ddof=1) * np.sqrt(252),
                "exposure_fraction": events["exposure"][:-1, event_column]
                @ seconds
                / seconds.sum(),
                "cost_drag_return": (accounts["gross_terminal_equity"] @ w - wealth[-1]) / 1000.0,
                "gross_sleeve_order_count": stats.loc[w > 0, "order_count"].sum(),
                "total_commission": stats.total_commission.to_numpy() @ w,
                "total_slippage": stats.total_slippage.to_numpy() @ w,
            }
            turnover = 0.0
            timestamps = set()
            account_weights = {cid: w[k] for cid, k in lookup.items()}
            for (timestamp, _phase), group in grouped:
                gw = group.account_id.map(account_weights).to_numpy()
                if not (gw > 0).any():
                    continue
                location = event_times.get_indexer([timestamp])[0]
                if location < 0:
                    raise Error("Private execution absent from exact event clock")
                costs = (group.commission.to_numpy() + group.slippage.to_numpy()) @ gw
                before = events["equity"][location, event_column] + costs
                if before <= 0:
                    raise Error("Invalid common pre-fill portfolio wealth")
                turnover += group.reference_notional.to_numpy() @ gw / before
                timestamps.add(timestamp)
            expected["annualized_turnover"] = turnover / years
            expected["distinct_execution_timestamp_count"] = len(timestamps)
            for field, value in expected.items():
                multi._assert_close(
                    row[field], value, f"Independent {field} differs from hybrid account arithmetic"
                )


def review_derived_results(directory, frozen, stage):
    if stage not in PERIODS:
        raise Error("Derived review rejects later stages")
    directory = safe_path(directory)
    metrics = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    controls = pd.read_csv(directory / "controls_metrics.csv", float_precision="round_trip")
    for frame, ids in (
        (metrics, [r["portfolio_id"] for r in principal_definitions()]),
        (controls, control_ids()),
    ):
        if (
            len(frame) != 32
            or frame.duplicated(["portfolio_id", "scenario_slippage_bps"]).any()
            or set(zip(frame.portfolio_id, frame.scenario_slippage_bps, strict=True))
            != {(p, s) for p in ids for s in SLIPPAGES}
        ):
            raise Error("All predefined portfolio/cost outcomes must remain present")
    for slip in SLIPPAGES:
        _review_account_aggregation(directory, slip, metrics, frozen)
        for kind, frame, ids in (
            ("principal", metrics, [r["portfolio_id"] for r in principal_definitions()]),
            ("controls", controls, control_ids()),
        ):
            with np.load(directory / f"{kind}_s{slip:g}_daily.npz", allow_pickle=False) as z:
                if list(z["portfolio_id"]) != ids:
                    raise Error("Saved portfolio identities differ from frozen controls")
                times = pd.DatetimeIndex(pd.to_datetime(z["timestamp"], utc=True))
                ny = times.tz_convert("America/New_York")
                if (
                    times.has_duplicates
                    or not times.is_monotonic_increasing
                    or ny[0].strftime("%Y-%m-%d") < PERIODS[stage]["start"]
                    or ny[-1].strftime("%Y-%m-%d") > PERIODS[stage]["end"]
                    or any((t.hour, t.minute) != (17, 0) for t in ny)
                ):
                    raise Error("Derived daily clock violates authorized segment or 17:00 NY marks")
                for j, pid in enumerate(ids):
                    equity = z["equity"][:, j]
                    returns = equity / np.r_[1000.0, equity[:-1]] - 1
                    multi._assert_close(
                        z["returns"][:, j], returns, "Saved hybrid return/wealth identity differs"
                    )
                    row = frame.loc[
                        frame.portfolio_id.eq(pid) & frame.scenario_slippage_bps.eq(slip)
                    ].iloc[0]
                    multi._assert_close(
                        row.terminal_equity, equity[-1], "Hybrid terminal wealth differs"
                    )
                    multi._assert_close(
                        row.cash_excess_sharpe,
                        portfolio.cash_excess_sharpe(returns, z["cash_returns"]),
                        "Hybrid daily cash-excess Sharpe differs",
                    )
        pairs = pd.read_csv(directory / f"pair_metrics_s{slip:g}.csv", float_precision="round_trip")
        pair_ids = {r["pair_id"] for r in frozen["pairs"]}
        expected_pairs = {
            (d["portfolio_id"], pid) for d in principal_definitions()[:7] for pid in pair_ids
        }
        if (
            len(pairs) != 2688
            or pairs.duplicated(["portfolio_id", "pair_id"]).any()
            or set(zip(pairs.portfolio_id, pairs.pair_id, strict=True)) != expected_pairs
        ):
            raise Error("Every predeclared architecture/allocation pair outcome required")
    return {
        "status": "passed",
        "principal_outcomes": 32,
        "control_outcomes": 32,
        "pair_outcomes": 10752,
        "historical_oos_authorized": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="QQQ hybrid allocation retrospective development/validation; no OOS"
    )
    parser.add_argument(
        "command", choices=("check", "freeze", "development", "validation", "report")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            load_config(args.config)
            result = (
                verify_study(args.sidecar)
                if (args.sidecar / "freeze.json").exists()
                else {
                    "status": "configuration_valid_unfrozen",
                    "pair_count": 384,
                    "principal_count": 8,
                    "market_evaluation_performed": False,
                    "historical_oos_authorized": False,
                }
            )
        elif args.command == "freeze":
            if args.authorization is None:
                parser.error("freeze requires --authorization")
            result = freeze_study(
                args.sidecar, authorization=args.authorization, config_path=args.config
            )
        elif args.command == "development":
            result = evaluate_development(args.sidecar)
        elif args.command == "validation":
            result = evaluate_validation(args.sidecar)
        else:
            from trend_following.research_hybrid_allocation_report import publish_report

            result = publish_report(args.sidecar, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (Error, ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Hybrid study stopped: {exc}\n")
