"""Additive, sealed mean-reversion validation of exactly 32 development cases.

Selection is derived only from the preserved 90-case development baseline.
A selection seal precedes any validation price read. A new 1999--2018 physical
packet and accounting/code freeze precede performance evaluation. No OOS route,
retuning, validation filtering, replacement, or selected validation winner exists.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from trend_following import research_constant_cash as fixed
from trend_following import research_mean_reversion as features
from trend_following import research_mean_reversion_engine as engine
from trend_following import research_mean_reversion_protocol as development
from trend_following import research_quality_engine_v5 as quality
from trend_following import research_quality_protocol_v5 as protocol
from trend_following import research_validation_prefix as prefix

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEVELOPMENT = development.DEFAULT_SIDECAR
DEFAULT_BOUNDED_PACKET = ROOT / ".cache/intersection_validation_2018_20261008/prepared"
DEFAULT_SIDECAR = ROOT / ".cache/mean_reversion_validation_20261009"
DEFAULT_REPORT = ROOT / "reports/mean_reversion_validation/20261009"
PERIOD = {"start": "2012-01-01", "end": "2018-12-31"}
HISTORICAL_OOS_PERIOD = {"start": "2019-01-01", "end": "2026-05-28"}
SLIPPAGES = development.SLIPPAGES
FAMILIES = development.FAMILIES
TOLERANCE = development.TOLERANCE
COUNT = 32
USER_REQUEST = "test those 32 cases in the validation period."
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *development.SOURCE_FILES,
            "src/trend_following/research_validation_prefix.py",
            "src/trend_following/research_mean_reversion_validation.py",
            "scripts/run_mean_reversion_validation.py",
        )
    )
)
Error = protocol.ProtocolError
safe_path = development.safe_path


def _chronology(*values):
    times = [pd.Timestamp(value) for value in values]
    if any(t.tzinfo is None or pd.isna(t) for t in times) or times != sorted(times):
        raise Error("Timezone-aware selection/preparation/freeze/execution chronology required")
    if any(t > pd.Timestamp.now(tz="UTC") for t in times):
        raise Error("Future-dated study chronology prohibited")
    return times


def _authorization(path: Path):
    value = json.loads(path.read_text())
    expected = {
        "schema": "mean_reversion_validation_authorization",
        "authorized_by": "human_user",
        "explicit_user_approval": True,
        "authorized_stages": ["validation"],
        "user_request": USER_REQUEST,
        "configuration_count": COUNT,
        "shared_confirmation_hours": 3.0,
        "validation_period": PERIOD,
        "historical_oos_authorized": False,
        "no_retuning_or_replacement": True,
    }
    if any(
        protocol.canonical_hash(value.get(k)) != protocol.canonical_hash(v)
        for k, v in expected.items()
    ):
        raise Error("Explicit matching human validation-only authorization required")
    return value


def _separate(sidecar, preserved, authorization=None):
    if sidecar.parent != ROOT / ".cache" or not sidecar.name.startswith(
        "mean_reversion_validation_"
    ):
        raise Error("New top-level mean_reversion_validation_ private sidecar required")
    for protected in preserved:
        if sidecar == protected or sidecar in protected.parents or protected in sidecar.parents:
            raise Error("Never overlap a preserved source, development or price-packet tree")
        if authorization is not None and authorization.is_relative_to(protected):
            raise Error("Authorization must be outside preserved trees")
    if authorization is not None and authorization.is_relative_to(sidecar):
        raise Error("Authorization must be outside the new study")


def candidates_from_selection(selection):
    rows = selection.get("candidates", [])
    if len(rows) != COUNT or protocol.canonical_hash(rows) != selection.get("candidate_set_sha256"):
        raise Error("Exactly 32 sealed candidate definitions required")
    grid = {c.candidate_id: c for c in features.candidate_grid()}
    ids = [row.get("candidate_id") for row in rows]
    if ids != sorted(set(ids)) or not set(ids).issubset(grid):
        raise Error("Canonical unique development candidate IDs required")
    expected = [{"candidate_id": grid[cid].candidate_id, **grid[cid].to_dict()} for cid in ids]
    if protocol.canonical_hash(rows) != protocol.canonical_hash(expected):
        raise Error("Sealed daily parameters or shared 3-hour wait changed")
    return [grid[cid] for cid in ids]


def select_development_cases(metrics, benchmark):
    """Strict existing development BH-beater rule, not the top-20 finalists."""
    grid = {c.candidate_id: c for c in features.candidate_grid()}
    if (
        len(metrics) != 90
        or metrics.candidate_id.duplicated().any()
        or set(metrics.candidate_id) != set(grid)
    ):
        raise Error("All preserved ninety development baseline rows are required")
    bhrows = benchmark.loc[
        benchmark.portfolio.eq("buy_and_hold") & benchmark.scenario_slippage_bps.eq(3)
    ]
    if len(bhrows) != 1 or not np.isfinite(bhrows.iloc[0].cash_excess_sharpe):
        raise Error("Exactly one finite baseline development BH score required")
    bh = float(bhrows.iloc[0].cash_excess_sharpe)
    for row in metrics.itertuples(index=False):
        candidate = grid[row.candidate_id]
        expected = {
            "family": candidate.family,
            "period": candidate.rule.period,
            "entry_threshold": candidate.entry_threshold,
            "exit_threshold": candidate.exit_threshold,
            "entry_confirmation_hours": 3.0,
            "exit_confirmation_hours": 3.0,
            "entry_required_checks": 7,
            "exit_required_checks": 7,
            "scenario_slippage_bps": 3.0,
            "commission_bps": 1.0,
            "slippage_bps": 3.0,
        }
        if any(getattr(row, name) != value for name, value in expected.items()):
            raise Error("Preserved development candidate metadata changed")
    delta = metrics.cash_excess_sharpe - bh
    beats = np.isfinite(metrics.cash_excess_sharpe) & (delta > TOLERANCE)
    if (
        not np.allclose(metrics.BH_cash_excess_sharpe, bh, atol=1e-12, rtol=0)
        or not np.allclose(
            metrics.Sharpe_difference_vs_BH, delta, atol=1e-12, rtol=0, equal_nan=True
        )
        or not np.array_equal(metrics.beats_BH_Sharpe, beats)
    ):
        raise Error("Preserved baseline BH comparisons changed")
    selected = metrics.loc[beats].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if len(selected) != COUNT:
        raise Error("The user authorized exactly the 32 existing baseline BH beaters")
    candidates = [grid[cid] for cid in selected.candidate_id]
    return candidates, selected, bh


def _metadata_binding(packet):
    """Manifest metadata only; no validation prices or earlier outcomes."""
    manifest = prefix._json_file(packet / "manifest.json")
    if (
        manifest.get("schema") != "trend_following_validation_2018_prefix_v1"
        or manifest.get("end_session") != PERIOD["end"]
        or manifest.get("feature_history_start") != prefix.START_SESSION
        or manifest.get("stage") != "validation"
        or manifest.get("historical_oos_authorized") is not False
        or set(manifest.get("files", {})) != set(prefix.PACKET_FILES)
        or manifest.get("audit", {}).get("OOS_numeric_rows_parsed") != 0
    ):
        raise Error("Existing physically bounded 1999--2018 price packet required")
    prepared = protocol._read_seal(packet / "freeze.json")
    expected = {
        "schema": "trend_following_resplit_validation_prepared_v1",
        "period": PERIOD,
        "feature_history_start": prefix.START_SESSION,
        "historical_oos_authorized": False,
        "manifest_sha256": protocol.canonical_hash(manifest),
        "prepared_at": manifest["prepared_at"],
    }
    if any(prepared.get(k) != v for k, v in expected.items()):
        raise Error("Preserved bounded preparation seal/manifest custody changed")
    return {
        "manifest_sha256": protocol.canonical_hash(manifest),
        "manifest": manifest,
        "prepared_seal_sha256": protocol.canonical_hash(prepared),
        "prepared_seal": prepared,
    }


def _selection_bindings(development_sidecar, bounded_packet, authorization):
    _authorization(authorization)
    dev_result = development.verify_results(development_sidecar)
    dev_freeze = development.verify_study(development_sidecar)
    metrics = pd.read_csv(
        development_sidecar / "results/candidate_metrics.csv", float_precision="round_trip"
    )
    benchmark = pd.read_csv(
        development_sidecar / "results/benchmark_metrics.csv", float_precision="round_trip"
    )
    candidates, selected, bh = select_development_cases(metrics, benchmark)
    metadata = _metadata_binding(bounded_packet)
    # Read source preparation custody JSON only, never original master/later
    # prices or original strategy results. MR verification already bound it.
    source_prepared = protocol._read_seal(
        safe_path(dev_freeze["source_study"]) / "prepared/freeze.json"
    )
    if protocol.canonical_hash(source_prepared) != dev_freeze["prepared_metadata_sha256"]:
        raise Error("Preserved development source provenance changed")
    original_record = source_prepared["prefixes"]["historical-oos"]
    original = {
        "prepared_metadata_sha256": protocol.canonical_hash(source_prepared),
        "original_manifest_sha256": original_record["manifest_sha256"],
        "original_unit_proof_sha256": original_record["files"]["unit_proof.json"]["sha256"],
    }
    if (
        metadata["manifest"]["original_manifest_sha256"] != original["original_manifest_sha256"]
        or metadata["manifest"]["original_unit_proof_sha256"]
        != original["original_unit_proof_sha256"]
    ):
        raise Error("Bounded packet and development must have common source provenance")
    canonical = [{"candidate_id": c.candidate_id, **c.to_dict()} for c in candidates]
    columns = (
        "candidate_id",
        "family",
        "period",
        "entry_threshold",
        "exit_threshold",
        "confirmation_hours",
        "cash_excess_sharpe",
        "cagr",
        "max_drawdown",
        "trade_count",
        "trades_per_year",
        "BH_cash_excess_sharpe",
        "Sharpe_difference_vs_BH",
    )
    rows = development._public_frame(selected, columns).to_dict("records")
    return {
        "schema": "mean_reversion_validation_selection",
        "stage": "validation",
        "period": PERIOD,
        "historical_oos_period": HISTORICAL_OOS_PERIOD,
        "historical_oos_authorized": False,
        "validation_only": True,
        "development_sidecar": str(development_sidecar),
        "bounded_packet": str(bounded_packet),
        "source_study": dev_freeze["source_study"],
        "source_config_path": dev_freeze["source_config_path"],
        "authorization": {
            "path": str(authorization),
            "sha256": protocol.sha256_file(authorization),
        },
        "development_freeze_sha256": protocol.canonical_hash(dev_freeze),
        "development_result_sha256": protocol.canonical_hash(dev_result),
        "development_finished_at": dev_result["finished_at"],
        "development_selection_file_hashes": {
            name: dev_result["files"][name]
            for name in ("candidate_metrics.csv", "benchmark_metrics.csv")
        },
        "development_BH_cash_excess_sharpe": bh,
        "selection_rule": "finite_development_baseline_cash_excess_Sharpe_minus_BH_greater_than_1e-10",
        "uses_top20_finalists": False,
        "development_selection_rows": rows,
        "development_selection_rows_sha256": protocol.canonical_hash(rows),
        "candidates": canonical,
        "candidate_set_sha256": protocol.canonical_hash(canonical),
        "configuration_count": COUNT,
        "shared_confirmation_hours": 3.0,
        "required_checks": 7,
        "bounded_packet_metadata": metadata,
        "source_metadata_binding": original,
        "source_hashes": {n: protocol.sha256_file(ROOT / n) for n in SOURCE_FILES},
        "runtime": protocol._runtime(),
        "cash": fixed.cash_policy(),
        "quality_policy": dev_freeze["quality_policy"],
        "initial_cash": 1000.0,
        "commission_bps": 1.0,
        "baseline_slippage_bps": 3.0,
        "slippage_bps_scenarios": list(SLIPPAGES),
        "no_validation_selection_or_replacement": True,
        "prior_validation_or_OOS_outcomes_used_for_selection": False,
        "retrospective_previously_examined_history": True,
    }


def seal_selection(
    sidecar=DEFAULT_SIDECAR,
    *,
    authorization,
    development_sidecar=DEFAULT_DEVELOPMENT,
    bounded_packet=DEFAULT_BOUNDED_PACKET,
):
    sidecar, authorization, dev, packet = map(
        safe_path, (sidecar, authorization, development_sidecar, bounded_packet)
    )
    _separate(sidecar, (dev, packet, safe_path(fixed.DEFAULT_SOURCE_STUDY)), authorization)
    with protocol._attempt(sidecar, "seal_development_only_selection"):
        bindings = _selection_bindings(dev, packet, authorization)
        _separate(sidecar, (dev, packet, safe_path(bindings["source_study"])), authorization)
        path = sidecar / "selection.json"
        if path.exists():
            old = protocol._read_seal(path)
            if {k: v for k, v in old.items() if k != "candidate_set_sealed_at"} != bindings:
                raise Error("Selection already sealed differently; preserve it")
            _chronology(old["development_finished_at"], old["candidate_set_sealed_at"])
            return old
        bindings["candidate_set_sealed_at"] = protocol._utc_now()
        _chronology(bindings["development_finished_at"], bindings["candidate_set_sealed_at"])
        protocol._seal(path, bindings)
        return bindings


def verify_selection(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    selection = protocol._read_seal(sidecar / "selection.json")
    dev, packet, authorization = map(
        safe_path,
        (
            selection["development_sidecar"],
            selection["bounded_packet"],
            selection["authorization"]["path"],
        ),
    )
    _separate(sidecar, (dev, packet, safe_path(selection["source_study"])), authorization)
    expected = _selection_bindings(dev, packet, authorization)
    if {k: v for k, v in selection.items() if k != "candidate_set_sealed_at"} != expected:
        raise Error("Selection/source/code/runtime/authorization changed")
    candidates_from_selection(selection)
    _chronology(selection["development_finished_at"], selection["candidate_set_sealed_at"])
    return selection


def _prepare_bounded_packet(selection, output):
    """Copy verified physically bounded prices only, never prior results or 2019+."""
    packet = safe_path(selection["bounded_packet"])
    original = prefix.verify_validation_prefix(
        packet, expected_manifest_sha256=selection["bounded_packet_metadata"]["manifest_sha256"]
    )
    # Parse the bounded source with its frozen hash/boundary validators BEFORE
    # accepting it. No broad or historical-OOS price reader is called.
    prefix.read_validation_prefix(
        packet, expected_manifest_sha256=selection["bounded_packet_metadata"]["manifest_sha256"]
    )
    output.mkdir(parents=True, exist_ok=False)
    for name in prefix.PACKET_FILES:
        shutil.copyfile(safe_path(packet / name), output / name)
    manifest = copy.deepcopy(original)
    manifest["prepared_at"] = protocol._utc_now()
    manifest["selection_sha256"] = protocol.canonical_hash(selection)
    manifest["candidate_set_sealed_at"] = selection["candidate_set_sealed_at"]
    manifest["source_bounded_manifest_sha256"] = protocol.canonical_hash(original)
    manifest["mean_reversion_validation_copy"] = True
    manifest["files"] = {
        name: {
            "sha256": protocol.sha256_file(output / name),
            "bytes": (output / name).stat().st_size,
        }
        for name in prefix.PACKET_FILES
    }
    if manifest["files"] != original["files"]:
        raise Error("New bounded packet copy changed source price/action bytes")
    protocol._write_json(output / "manifest.json", manifest)
    prefix.verify_validation_prefix(
        output, expected_manifest_sha256=protocol.canonical_hash(manifest)
    )
    return manifest


def freeze_study(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    with protocol._attempt(sidecar, "freeze_bounded_prices_and_accounting") as attempt:
        selection = verify_selection(sidecar)
        if (sidecar / "freeze.json").exists():
            return verify_study(sidecar)
        if (sidecar / "prepared").exists():
            raise Error("Unsealed prepared tree exists; do not overwrite")
        output = attempt / "prepared"
        manifest = _prepare_bounded_packet(selection, output)
        verify_selection(sidecar)
        frozen = {
            "schema": "mean_reversion_validation_freeze",
            "stage": "validation",
            "selection_sha256": protocol.canonical_hash(selection),
            "candidate_set_sha256": selection["candidate_set_sha256"],
            "configuration_count": COUNT,
            "period": PERIOD,
            "manifest_sha256": protocol.canonical_hash(manifest),
            "candidate_set_sealed_at": selection["candidate_set_sealed_at"],
            "prepared_at": manifest["prepared_at"],
            "frozen_at": protocol._utc_now(),
            "source_hashes": selection["source_hashes"],
            "runtime": selection["runtime"],
            "cash": selection["cash"],
            "initial_cash": 1000.0,
            "commission_bps": 1.0,
            "slippage_bps_scenarios": list(SLIPPAGES),
            "no_validation_selection_or_replacement": True,
            "historical_oos_authorized": False,
        }
        _chronology(frozen["candidate_set_sealed_at"], frozen["prepared_at"], frozen["frozen_at"])
        output.rename(sidecar / "prepared")
        protocol._seal(sidecar / "freeze.json", frozen)
        return frozen


def verify_study(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    selection = verify_selection(sidecar)
    frozen = protocol._read_seal(sidecar / "freeze.json")
    expected = {
        "schema": "mean_reversion_validation_freeze",
        "stage": "validation",
        "selection_sha256": protocol.canonical_hash(selection),
        "candidate_set_sha256": selection["candidate_set_sha256"],
        "configuration_count": COUNT,
        "period": PERIOD,
        "candidate_set_sealed_at": selection["candidate_set_sealed_at"],
        "source_hashes": selection["source_hashes"],
        "runtime": selection["runtime"],
        "cash": selection["cash"],
        "initial_cash": 1000.0,
        "commission_bps": 1.0,
        "slippage_bps_scenarios": list(SLIPPAGES),
        "no_validation_selection_or_replacement": True,
        "historical_oos_authorized": False,
    }
    if any(frozen.get(k) != v for k, v in expected.items()):
        raise Error("Validation freeze contract changed")
    manifest = prefix.verify_validation_prefix(
        sidecar / "prepared", expected_manifest_sha256=frozen["manifest_sha256"]
    )
    if (
        manifest["prepared_at"] != frozen["prepared_at"]
        or manifest["selection_sha256"] != protocol.canonical_hash(selection)
        or manifest.get("mean_reversion_validation_copy") is not True
        or manifest.get("source_bounded_manifest_sha256")
        != selection["bounded_packet_metadata"]["manifest_sha256"]
        or manifest["files"] != selection["bounded_packet_metadata"]["manifest"]["files"]
    ):
        raise Error("New physical validation packet changed selection/provenance binding")
    _chronology(frozen["candidate_set_sealed_at"], frozen["prepared_at"], frozen["frozen_at"])
    return frozen


def load_validation_packet(sidecar, selection):
    if (
        selection.get("stage") != "validation"
        or selection.get("period") != PERIOD
        or selection.get("historical_oos_authorized") is not False
        or selection.get("validation_only") is not True
    ):
        raise Error("Only frozen 2012--2018 validation may execute")
    frozen = verify_study(sidecar)
    if protocol.canonical_hash(selection) != frozen["selection_sha256"]:
        raise Error(
            "Caller selection differs from the verified frozen candidate/accounting contract"
        )
    daily, bars, calendar, coverage, blackouts, _ = prefix.read_validation_prefix(
        safe_path(sidecar) / "prepared", expected_manifest_sha256=frozen["manifest_sha256"]
    )
    config = copy.deepcopy(protocol.load_config(selection["source_config_path"]))
    config["segments"]["validation"] = dict(PERIOD)
    config["segments"]["historical-oos"] = dict(HISTORICAL_OOS_PERIOD)
    cash = fixed.constant_cash_observations(
        calendar.market_open.iloc[0], daily.daily_price_available_at.iloc[-1]
    )
    protocol._check_prefix(daily, bars, calendar, cash, "validation", config)
    protocol._quality_gate(
        config, coverage, selection["bounded_packet_metadata"]["manifest"]["gap_authorization"]
    )
    return daily, bars, calendar, coverage, blackouts, cash, config


def _result_files():
    names = {
        "candidate_metrics.csv",
        "candidate_metrics_all_costs.csv",
        "benchmark_metrics.csv",
        "development_selection_metrics.csv",
        "post_gap_candidate_flags.csv",
        "post_gap_signal_events.csv",
        "post_gap_diagnostic_metadata.json",
    }
    for slip in SLIPPAGES:
        names |= {
            f"slippage_{slip:g}_daily.npz",
            f"family_summary_{slip:g}.csv",
            f"cash_{slip:g}_daily.csv",
            f"buy_and_hold_{slip:g}_daily.csv",
        }
    return names


def evaluate_study(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    with protocol._attempt(sidecar, "validation") as attempt:
        selection = verify_selection(sidecar)
        frozen = verify_study(sidecar)
        if (sidecar / "results").exists():
            raise Error("Validation already sealed; never overwrite or rerun")
        started = protocol._utc_now()
        _chronology(frozen["frozen_at"], started)
        daily, bars, calendar, coverage, blackouts, cash, config = load_validation_packet(
            sidecar, selection
        )
        cache = engine.build_cache(
            daily,
            bars,
            calendar,
            coverage=coverage,
            blackouts=blackouts,
            quality_policy=protocol._quality_policy(config),
        )
        candidates = candidates_from_selection(selection)
        output = attempt / "results"
        output.mkdir()
        all_rows, benchmark_rows = [], []
        for slip in SLIPPAGES:
            kwargs = dict(
                start=PERIOD["start"],
                end=PERIOD["end"],
                cash_observations=cash,
                commission_bps=1.0,
                slippage_bps=slip,
                initial_cash=1000.0,
            )
            result = engine.run(cache, candidates, **kwargs)
            rows = pd.DataFrame(result[0])
            years = (
                result[1][-1]
                - calendar.loc[calendar.session.ge(PERIOD["start"]), "market_open"].iloc[0]
            ).total_seconds() / (365 * 86400)
            rows["trades_per_year"] = rows.trade_count / years
            rows["scenario_slippage_bps"] = slip
            development._array_file(output / f"slippage_{slip:g}_daily.npz", result, candidates)
            for mode in ("cash", "buy_and_hold"):
                passive = engine.run(cache, candidates[:1], mode=mode, **kwargs)
                stats = {"portfolio": mode, "scenario_slippage_bps": slip, **passive[0][0]}
                stats["trades_per_year"] = stats["trade_count"] / years
                benchmark_rows.append(stats)
                pd.DataFrame(
                    {
                        "timestamp": passive[1],
                        "equity": passive[2][:, 0],
                        "return": passive[3][:, 0],
                        "cash_return": passive[4],
                        "exposure": passive[5][:, 0],
                    }
                ).to_csv(output / f"{mode}_{slip:g}_daily.csv", index=False)
            bh = benchmark_rows[-1]["cash_excess_sharpe"]
            if not np.isfinite(bh):
                raise Error("Matched validation BH Sharpe is undefined")
            rows["BH_cash_excess_sharpe"] = bh
            rows["Sharpe_difference_vs_BH"] = rows.cash_excess_sharpe - bh
            rows["beats_BH_Sharpe"] = np.isfinite(rows.cash_excess_sharpe) & (
                rows.Sharpe_difference_vs_BH > TOLERANCE
            )
            all_rows.append(rows)
            development.family_summary(rows, bh).to_csv(
                output / f"family_summary_{slip:g}.csv", index=False
            )
            if slip == 3:
                baseline = rows
                post = result[-1]
                if post is None:
                    raise Error("Required descriptive post-gap diagnostic is absent")
                rows.attrs["post_gap_signal_events"] = quality._post_gap_table(post["rows"])
                rows.attrs["post_gap_signal_event_metadata"] = quality._post_gap_metadata(
                    post["context"], cache.policy, post["total_rows"], len(post["rows"]), result[0]
                )
                protocol._persist_post_gap_diagnostics(
                    output, rows, candidates, config, "validation"
                )
            print(
                f"Completed mean-reversion validation: 32 frozen configurations; 1-bp commission + {slip:g}-bps slippage",
                flush=True,
            )
        pd.concat(all_rows, ignore_index=True).to_csv(
            output / "candidate_metrics_all_costs.csv", index=False
        )
        baseline.to_csv(output / "candidate_metrics.csv", index=False)
        pd.DataFrame(benchmark_rows).to_csv(output / "benchmark_metrics.csv", index=False)
        pd.DataFrame(selection["development_selection_rows"]).to_csv(
            output / "development_selection_metrics.csv", index=False
        )
        review_derived_results(output, selection)
        verify_study(sidecar)
        result_seal = {
            "schema": "mean_reversion_validation_results",
            "stage": "validation",
            "freeze_sha256": protocol.canonical_hash(frozen),
            "selection_sha256": protocol.canonical_hash(selection),
            "candidate_set_sha256": selection["candidate_set_sha256"],
            "period": PERIOD,
            "configuration_count": COUNT,
            "configuration_scenario_count": COUNT * 4,
            "scenarios": list(SLIPPAGES),
            "candidate_set_sealed_at": selection["candidate_set_sealed_at"],
            "prepared_at": frozen["prepared_at"],
            "frozen_at": frozen["frozen_at"],
            "market_started_at": started,
            "finished_at": protocol._utc_now(),
            "no_validation_selection_or_replacement": True,
            "no_winner_selected": True,
            "historical_oos_authorized": False,
            "OOS_numeric_rows_parsed": 0,
            "files": {p.name: protocol.sha256_file(p) for p in sorted(output.iterdir())},
        }
        _chronology(
            result_seal["candidate_set_sealed_at"],
            result_seal["prepared_at"],
            result_seal["frozen_at"],
            result_seal["market_started_at"],
            result_seal["finished_at"],
        )
        protocol._seal(output / "seal.json", result_seal)
        output.rename(sidecar / "results")
        return result_seal


def verify_results(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    selection, frozen = verify_selection(sidecar), verify_study(sidecar)
    seal = protocol._read_seal(sidecar / "results/seal.json")
    expected = {
        "schema": "mean_reversion_validation_results",
        "stage": "validation",
        "freeze_sha256": protocol.canonical_hash(frozen),
        "selection_sha256": protocol.canonical_hash(selection),
        "candidate_set_sha256": selection["candidate_set_sha256"],
        "period": PERIOD,
        "configuration_count": COUNT,
        "configuration_scenario_count": COUNT * 4,
        "scenarios": list(SLIPPAGES),
        "candidate_set_sealed_at": selection["candidate_set_sealed_at"],
        "prepared_at": frozen["prepared_at"],
        "frozen_at": frozen["frozen_at"],
        "no_validation_selection_or_replacement": True,
        "no_winner_selected": True,
        "historical_oos_authorized": False,
        "OOS_numeric_rows_parsed": 0,
    }
    if any(seal.get(k) != v for k, v in expected.items()):
        raise Error("Validation results differ from frozen identity/accounting/scope")
    _chronology(
        seal["candidate_set_sealed_at"],
        seal["prepared_at"],
        seal["frozen_at"],
        seal["market_started_at"],
        seal["finished_at"],
    )
    actual = {p.name for p in (sidecar / "results").iterdir()} - {"seal.json"}
    if set(seal.get("files", {})) != _result_files() or actual != _result_files():
        raise Error("Missing or extra derived validation artifacts")
    for name, digest in seal["files"].items():
        if protocol.sha256_file(safe_path(sidecar / "results" / name)) != digest:
            raise Error("Sealed validation result bytes changed")
    return seal


def _assert_close(actual, expected, message, *, atol=1e-11):
    if not np.allclose(actual, expected, rtol=1e-10, atol=atol, equal_nan=True):
        raise Error(message)


def review_derived_results(private, selection):
    """Independent derived-path arithmetic audit, not historical price replay."""
    private = safe_path(private)
    candidates = {c.candidate_id: c for c in candidates_from_selection(selection)}
    metrics = pd.read_csv(private / "candidate_metrics_all_costs.csv", float_precision="round_trip")
    benchmark = pd.read_csv(private / "benchmark_metrics.csv", float_precision="round_trip")
    if (
        len(metrics) != COUNT * 4
        or metrics.duplicated(["candidate_id", "scenario_slippage_bps"]).any()
    ):
        raise Error("All 32 frozen candidates under all four costs must remain visible")
    if (
        len(benchmark) != 8
        or benchmark.duplicated(["portfolio", "scenario_slippage_bps"]).any()
        or set(zip(benchmark.portfolio, benchmark.scenario_slippage_bps, strict=True))
        != {(p, s) for p in ("cash", "buy_and_hold") for s in SLIPPAGES}
    ):
        raise Error("Eight unique cost-matched passive benchmarks required")
    audit = selection["bounded_packet_metadata"]["manifest"]["audit"]
    session_count = audit["validation_sessions"]
    first_session, last_session = (
        audit["first_validation_session"],
        audit["last_validation_session"],
    )
    if first_session != "2012-01-03" or last_session != "2018-12-31" or session_count < 2:
        raise Error("Frozen validation-session clock differs from authorized 2012--2018")
    # Pure frozen exchange-calendar generation uses no market prices or prior
    # outcomes; compare every session, not merely endpoint/count/17:00 labels.
    calendar = prefix.hourly.build_xnys_calendar(PERIOD["start"], PERIOD["end"])
    expected_times = pd.DatetimeIndex(
        [
            pd.Timestamp(session + " 17:00", tz="America/New_York").tz_convert("UTC")
            for session in calendar.session
        ]
    )
    if len(expected_times) != session_count:
        raise Error("Frozen XNYS calendar and packet session counts differ")
    common_times, common_cash = None, None
    for slip in SLIPPAGES:
        part = metrics.loc[metrics.scenario_slippage_bps.eq(slip)]
        if len(part) != COUNT or set(part.candidate_id) != set(candidates):
            raise Error("Validation candidate identity/coverage changed")
        for row in part.itertuples(index=False):
            candidate = candidates[row.candidate_id]
            identity = {
                "family": candidate.family,
                "period": candidate.rule.period,
                "indicator_units": "daily_sessions",
                "entry_threshold": candidate.entry_threshold,
                "exit_threshold": candidate.exit_threshold,
                "confirmation_hours": 3.0,
                "entry_confirmation_hours": 3.0,
                "exit_confirmation_hours": 3.0,
                "entry_required_checks": 7,
                "exit_required_checks": 7,
                "commission_bps": 1.0,
                "slippage_bps": slip,
                "sampling_minutes": 30,
                "start_session": first_session,
                "end_session": last_session,
                "scored_sessions": session_count,
            }
            if any(getattr(row, k) != v for k, v in identity.items()):
                raise Error("Validation candidate parameters, period or accounting changed")
            if (
                row.trade_count < 0
                or row.trade_count != int(row.trade_count)
                or row.order_count != 2 * row.trade_count
            ):
                raise Error("Completed round trips and one-way orders do not reconcile")
        with np.load(private / f"slippage_{slip:g}_daily.npz", allow_pickle=False) as arrays:
            ids = arrays["candidate_id"].tolist()
            times = pd.DatetimeIndex(pd.to_datetime(arrays["timestamp"], utc=True))
            equity, returns, cash_returns, exposure = (
                arrays["equity"],
                arrays["returns"],
                arrays["cash_returns"],
                arrays["exposure"],
            )
            local = times.tz_convert("America/New_York")
            if (
                len(ids) != COUNT
                or len(set(ids)) != COUNT
                or set(ids) != set(candidates)
                or equity.shape != (session_count, COUNT)
                or returns.shape != equity.shape
                or exposure.shape != equity.shape
                or cash_returns.shape != (session_count,)
                or len(times) != session_count
                or not times.equals(expected_times)
                or times.has_duplicates
                or not times.is_monotonic_increasing
                or not (local.hour == 17).all()
                or not (local.minute == 0).all()
                or not (local.second == 0).all()
                or (local.microsecond != 0).any()
                or local[0].strftime("%Y-%m-%d") != first_session
                or local[-1].strftime("%Y-%m-%d") != last_session
                or not np.isfinite(equity).all()
                or (equity <= 0).any()
                or not np.isfinite(returns).all()
                or not np.isfinite(cash_returns).all()
                or not np.isfinite(exposure).all()
                or (exposure < 0).any()
                or (exposure > 1).any()
            ):
                raise Error("Invalid validation-only derived dimensions/clock/wealth/exposure")
            if common_times is None:
                common_times, common_cash = times, cash_returns.copy()
            elif not times.equals(common_times) or not np.array_equal(cash_returns, common_cash):
                raise Error("Scenario clocks or preknown cash references differ")
            expected_returns = equity / np.vstack([np.full(COUNT, 1000.0), equity[:-1]]) - 1
            _assert_close(
                returns, expected_returns, "Daily returns do not reconcile to wealth", atol=1e-12
            )
            start = pd.Timestamp(first_session + " 09:30", tz="America/New_York").tz_convert("UTC")
            years = (times[-1] - start).total_seconds() / (365 * 86400)
            excess = returns - cash_returns[:, None]
            sd = excess.std(axis=0, ddof=1)
            scores = np.divide(
                np.sqrt(252) * excess.mean(axis=0), sd, out=np.full(COUNT, np.nan), where=sd > 0
            )
            wealth = np.vstack([np.full(COUNT, 1000.0), equity])
            aligned = part.set_index("candidate_id").loc[ids]
            expected = {
                "cash_excess_sharpe": scores,
                "terminal_equity": equity[-1],
                "cagr": (equity[-1] / 1000.0) ** (1 / years) - 1,
                "max_drawdown": (wealth / np.maximum.accumulate(wealth, axis=0) - 1).min(axis=0),
                "annualized_volatility": returns.std(axis=0, ddof=1) * np.sqrt(252),
                "total_return": equity[-1] / 1000.0 - 1,
                "trades_per_year": aligned.trade_count / years,
                "annualized_turnover": aligned.total_turnover / years,
                "cost_drag_return": (aligned.gross_terminal_equity - equity[-1]) / 1000.0,
            }
            for field, values in expected.items():
                _assert_close(aligned[field], values, f"Derived {field} differs from saved paths")
            for portfolio in ("cash", "buy_and_hold"):
                row = benchmark.loc[
                    benchmark.portfolio.eq(portfolio) & benchmark.scenario_slippage_bps.eq(slip)
                ].iloc[0]
                curve = pd.read_csv(
                    private / f"{portfolio}_{slip:g}_daily.csv", float_precision="round_trip"
                )
                if not pd.DatetimeIndex(pd.to_datetime(curve.timestamp, utc=True)).equals(times):
                    raise Error("Passive benchmark clock changed")
                if not np.isfinite(curve.equity).all() or curve.equity.le(0).any():
                    raise Error("Invalid passive benchmark wealth")
                _assert_close(
                    curve.cash_return, cash_returns, "Passive cash reference changed", atol=1e-12
                )
                _assert_close(
                    curve["return"],
                    curve.equity / np.r_[1000.0, curve.equity[:-1]] - 1,
                    "Passive returns differ from wealth",
                    atol=1e-12,
                )
                ex = curve["return"].to_numpy() - cash_returns
                score = np.sqrt(252) * ex.mean() / ex.std(ddof=1) if ex.std(ddof=1) > 0 else np.nan
                w = np.r_[1000.0, curve.equity]
                passive_stats = {
                    "cash_excess_sharpe": score,
                    "terminal_equity": curve.equity.iloc[-1],
                    "cagr": (curve.equity.iloc[-1] / 1000.0) ** (1 / years) - 1,
                    "max_drawdown": (w / np.maximum.accumulate(w) - 1).min(),
                    "annualized_volatility": curve["return"].std(ddof=1) * np.sqrt(252),
                    "total_return": curve.equity.iloc[-1] / 1000.0 - 1,
                    "trades_per_year": row.trade_count / years,
                }
                for field, value in passive_stats.items():
                    _assert_close(row[field], value, f"Passive {field} changed")
                if portfolio == "cash":
                    _assert_close(
                        curve["return"],
                        cash_returns,
                        "Cash benchmark is not matched reference",
                        atol=1e-12,
                    )
                    if row.trade_count != 0 or row.order_count != 0:
                        raise Error("Cash benchmark must not trade")
                    _assert_close(
                        aligned.cash_terminal_equity,
                        curve.equity.iloc[-1],
                        "Candidate cash reference changed",
                    )
                else:
                    if row.trade_count != 1 or row.order_count != 2:
                        raise Error("BH must have one common-start terminal-liquidated round trip")
            bh = (
                benchmark.loc[
                    benchmark.portfolio.eq("buy_and_hold")
                    & benchmark.scenario_slippage_bps.eq(slip)
                ]
                .iloc[0]
                .cash_excess_sharpe
            )
            if not np.isfinite(bh):
                raise Error("Undefined matched validation BH score")
            delta = aligned.cash_excess_sharpe - bh
            _assert_close(aligned.BH_cash_excess_sharpe, bh, "Cost-matched BH score changed")
            _assert_close(aligned.Sharpe_difference_vs_BH, delta, "BH Sharpe difference changed")
            if not np.array_equal(
                aligned.beats_BH_Sharpe, np.isfinite(delta) & (delta > TOLERANCE)
            ):
                raise Error("BH-beater flags changed")
            calculated = development.family_summary(part, bh)
            stored = pd.read_csv(
                private / f"family_summary_{slip:g}.csv", float_precision="round_trip"
            )
            pd.testing.assert_frame_equal(
                calculated, stored, check_dtype=False, rtol=1e-10, atol=1e-11
            )
    baseline = metrics.loc[metrics.scenario_slippage_bps.eq(3)].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        baseline,
        pd.read_csv(private / "candidate_metrics.csv", float_precision="round_trip"),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        pd.DataFrame(selection["development_selection_rows"]),
        pd.read_csv(private / "development_selection_metrics.csv", float_precision="round_trip"),
        check_dtype=False,
    )
    _review_gap_diagnostics(private, baseline, set(candidates))
    return metrics, baseline, benchmark


def _review_gap_diagnostics(private, baseline, ids):
    flags = pd.read_csv(private / "post_gap_candidate_flags.csv")
    events = pd.read_csv(private / "post_gap_signal_events.csv")
    metadata = json.loads((private / "post_gap_diagnostic_metadata.json").read_text())
    fields = [
        "candidate_id",
        "post_gap_confirmed_buy_count",
        "post_gap_confirmed_sell_count",
        "post_gap_confirmed_signal_count",
        "has_post_gap_confirmed_signal",
    ]
    if (
        len(flags) != COUNT
        or flags.candidate_id.duplicated().any()
        or set(flags.candidate_id) != ids
        or not set(events.candidate_id).issubset(ids)
        or not events.side.isin(["buy", "sell"]).all()
        or metadata.get("descriptive_only") is not True
        or metadata.get("used_for_filtering_ranking_or_selection") is not False
        or metadata.get("truncated") is not False
        or metadata.get("stored_event_rows") != len(events)
        or metadata.get("total_event_rows") != len(events)
    ):
        raise Error("Incomplete or selection-affecting post-gap diagnostics")
    pd.testing.assert_frame_equal(
        baseline[fields].sort_values("candidate_id").reset_index(drop=True),
        flags[fields].sort_values("candidate_id").reset_index(drop=True),
        check_dtype=False,
    )
    for cid in ids:
        part = events.loc[events.candidate_id.eq(cid)].drop_duplicates(["side", "decision_at"])
        row = flags.loc[flags.candidate_id.eq(cid)].iloc[0]
        buys, sells = int(part.side.eq("buy").sum()), int(part.side.eq("sell").sum())
        if (
            row.post_gap_confirmed_buy_count != buys
            or row.post_gap_confirmed_sell_count != sells
            or row.post_gap_confirmed_signal_count != buys + sells
            or bool(row.has_post_gap_confirmed_signal) != bool(buys + sells)
        ):
            raise Error("Distinct post-gap confirmations do not reconcile to flags")
    if len(events):
        times = {
            name: pd.to_datetime(events[name], utc=True)
            for name in ("decision_at", "gap_start", "gap_end", "window_end")
        }
        if (
            not (times["gap_start"] < times["gap_end"]).all()
            or not (times["gap_end"] < times["decision_at"]).all()
            or not (times["decision_at"] <= times["window_end"]).all()
            or times["decision_at"]
            .dt.tz_convert("America/New_York")
            .dt.strftime("%Y-%m-%d")
            .lt("2012-01-01")
            .any()
            or times["decision_at"]
            .dt.tz_convert("America/New_York")
            .dt.strftime("%Y-%m-%d")
            .gt("2018-12-31")
            .any()
        ):
            raise Error("Post-gap decision timestamps are outside frozen validation windows")


def report(sidecar=DEFAULT_SIDECAR, output=DEFAULT_REPORT):
    sidecar, output = map(safe_path, (sidecar, output))
    seal = verify_results(sidecar)
    selection = verify_selection(sidecar)
    if not output.is_relative_to(ROOT / "reports/mean_reversion_validation") or output.exists():
        raise Error("A new separate mean-reversion validation publication folder required")
    metrics, baseline, benchmark = review_derived_results(sidecar / "results", selection)
    bh = benchmark.loc[
        benchmark.portfolio.eq("buy_and_hold") & benchmark.scenario_slippage_bps.eq(3)
    ].iloc[0]
    summary = development.family_summary(baseline, bh.cash_excess_sharpe)
    output.mkdir(parents=True, exist_ok=False)
    development._public_frame(metrics, development.PUBLIC_METRICS).to_csv(
        output / "candidate_metrics_all_costs.csv", index=False
    )
    public_baseline = development._public_frame(baseline, development.PUBLIC_METRICS)
    public_baseline.to_csv(output / "candidate_metrics.csv", index=False)
    identity = {
        "candidate_id",
        "family",
        "period",
        "indicator_units",
        "entry_threshold",
        "exit_threshold",
        "confirmation_hours",
        "entry_confirmation_hours",
        "exit_confirmation_hours",
        "entry_required_checks",
        "exit_required_checks",
    }
    passive_columns = ("portfolio", *(n for n in development.PUBLIC_METRICS if n not in identity))
    development._public_frame(benchmark, passive_columns).to_csv(
        output / "benchmark_metrics.csv", index=False
    )
    summary.to_csv(output / "family_summary.csv", index=False)
    stress = []
    for slip in SLIPPAGES:
        part = metrics.loc[metrics.scenario_slippage_bps.eq(slip)]
        reference = benchmark.loc[
            benchmark.portfolio.eq("buy_and_hold") & benchmark.scenario_slippage_bps.eq(slip)
        ].iloc[0]
        table = development.family_summary(part, reference.cash_excess_sharpe)
        table["slippage_bps"] = slip
        stress.append(table)
    pd.concat(stress, ignore_index=True).to_csv(
        output / "family_summary_all_costs.csv", index=False
    )
    selected_development = pd.DataFrame(selection["development_selection_rows"])
    selected_development.to_csv(output / "development_selection_metrics.csv", index=False)
    compare_fields = [
        "candidate_id",
        "cash_excess_sharpe",
        "cagr",
        "max_drawdown",
        "trade_count",
        "trades_per_year",
        "BH_cash_excess_sharpe",
        "Sharpe_difference_vs_BH",
    ]
    comparison = public_baseline.merge(
        selected_development[compare_fields].rename(
            columns={n: "development_" + n for n in compare_fields if n != "candidate_id"}
        ),
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    comparison.to_csv(output / "development_vs_validation.csv", index=False)
    for name in ("post_gap_candidate_flags.csv", "post_gap_signal_events.csv"):
        frame = pd.read_csv(sidecar / "results" / name)
        development._public_frame(frame, frame.columns).to_csv(output / name, index=False)
    metadata = json.loads((sidecar / "results/post_gap_diagnostic_metadata.json").read_text())
    protocol._write_json(
        output / "gap_diagnostics.json",
        development._clean(
            {
                k: v
                for k, v in metadata.items()
                if k not in {"config_path", "source_study", "authorization"}
            }
        ),
    )
    public = {
        "study_name": "Mean-Reversion Study",
        "stage": "validation",
        "security": "QQQ",
        "period": PERIOD,
        "configuration_count": COUNT,
        "configuration_scenario_count": COUNT * 4,
        "candidate_set_sha256": selection["candidate_set_sha256"],
        "selection_sha256": seal["selection_sha256"],
        "freeze_sha256": seal["freeze_sha256"],
        "shared_confirmation_hours": 3.0,
        "required_consecutive_checks": 7,
        "selection_rule": selection["selection_rule"],
        "uses_top20_finalists": False,
        "initial_cash": 1000,
        "cash_policy": fixed.cash_policy(),
        "commission_bps": 1,
        "baseline_slippage_bps": 3,
        "slippage_bps_scenarios": list(SLIPPAGES),
        "development_BH_cash_excess_sharpe": selection["development_BH_cash_excess_sharpe"],
        "baseline_BH_Sharpe": bh.cash_excess_sharpe,
        "baseline_BH_CAGR": bh.cagr,
        "baseline_BH_max_drawdown": bh.max_drawdown,
        "candidate_set_sealed_at": seal["candidate_set_sealed_at"],
        "prepared_at": seal["prepared_at"],
        "frozen_at": seal["frozen_at"],
        "market_started_at": seal["market_started_at"],
        "finished_at": seal["finished_at"],
        "no_validation_selection_or_replacement": True,
        "no_winner_selected": True,
        "historical_oos_authorized": False,
        "historical_oos_period": HISTORICAL_OOS_PERIOD,
        "OOS_numeric_rows_parsed": 0,
        "all_32_in_denominator": True,
        "retrospective_previously_examined_history": True,
        "trade_definition": "completed_round_trips_including_terminal_liquidation",
        "drawdown_definition": "daily_provider_quote_17NY_marks_not_intraday",
        "exposure_definition": "elapsed_calendar_time_in_QQQ",
        "holding_duration": "elapsed_hours_including_closures",
        "derived_audit_scope": "independent_saved_path_arithmetic_not_independent_historical_price_replay",
    }
    protocol._write_json(output / "summary.json", development._clean(public))
    lines = [
        "# Mean-Reversion Study — Validation",
        "",
        "**2012–2018; exactly 32 development-selected configurations; shared 3-hour entry/exit wait (seven completed half-hour checks).**",
        "",
        "Membership was sealed before validation price access. These are all 32 baseline development cases beating QQQ buy-and-hold cash-excess Sharpe by more than 1e-10, not the separate top-20 finalist list. No parameter retuning, filtering, replacement or validation winner selection occurred.",
        "",
        f"Baseline cost-matched QQQ buy-and-hold: **{bh.cash_excess_sharpe:.4f} cash-excess Sharpe**, **{bh.cagr:.2%} CAGR**, **{-bh.max_drawdown:.2%} maximum daily drawdown loss**.",
        "",
        "| Family | Frozen cases | Beat BH Sharpe | All-case median Sharpe | BH-beater median Sharpe | BH-beater median CAGR | BH-beater median round trips | BH-beater median round trips/year |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def number(v, percent=False):
        return "—" if not np.isfinite(v) else (f"{v:.2%}" if percent else f"{v:.4f}")

    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.family} | {row.configurations} | {row.beat_BH_Sharpe_count} | {number(row.all_median_cash_excess_sharpe)} | {number(row.BH_beaters_median_cash_excess_sharpe)} | {number(row.BH_beaters_median_cagr, True)} | {number(row.BH_beaters_median_trade_count)} | {number(row.BH_beaters_median_trades_per_year)} |"
        )
    lines += [
        "",
        "## Execution-cost sensitivity",
        "",
        "| Slippage plus 1-bp commission | Cases beating matched BH Sharpe | All-case median Sharpe | BH Sharpe |",
        "|---|---:|---:|---:|",
    ]
    for table in stress:
        row = table.loc[table.family.eq("ALL")].iloc[0]
        lines.append(
            f"| {row.slippage_bps:g} + 1 bp | {row.beat_BH_Sharpe_count:g}/32 | {number(row.all_median_cash_excess_sharpe)} | {number(row.BH_Sharpe)} |"
        )
    lines += [
        "",
        "## Method and limitations",
        "",
        "- Every frozen candidate and all 128 candidate/cost outcomes remain visible, including any undefined Sharpe. Baseline is 1-bp commission plus 3-bps adverse slippage per one-way order; 1/10/25-bps slippage scenarios are hypothetical full-period execution stresses, without cost-specific reselection or hindsight labels.",
        "- Hold 100% unlevered QQQ or cash, starting at $1,000. Cash uses the unchanged 3% nominal ACT/365 assumption, compounded on the event clock; effective cash growth is not exactly 3% APY. Taxes, financing and separate fund-expense charges are disabled.",
        "- Same daily SMA/EMA-distance, sample-standard-deviation z-score and Wilder RSI rules as development. Persistent oversold entry and indicator-recovery exit use seven consecutive completed half-hour observations, with next-safe-bar-open execution. Preview daily indicators from yesterday's finalized state, never the previous preview; daily state finalizes once at assumed 17:00 New York availability.",
        "- Alpha-only intraday prices are reconstructed as-traded units, not guaranteed native raw or point-in-time captures. Corporate actions build the causal total-return signal index. Matched dividends/DRIP, common starts and terminal-liquidation conventions are unchanged. Provider daily marks are not guaranteed official or executable closes.",
        "- Initialization/development observations before 2012 provide indicator history only, no validation profits. Every strategy and benchmark starts fresh in cash; positions/counters reset at the segment boundary. Physically bounded input ends December 31, 2018; no 2019+ numerical price rows are opened or evaluated.",
        "- Missing intervals and halts preserve the last filled position, cancel pending orders and reset streaks. Post-gap flags extend through the next trading session close, are descriptive only and never select candidates.",
        "- Sharpe is daily cash-excess Sharpe with 252-session annualization and sample standard deviation (ddof=1). Drawdown is measured at daily 17:00 NY valuation marks, not intraday. Exposure is elapsed calendar-time in QQQ, not the fraction of trading checks. Holding duration includes overnight/weekend closures. Trades are completed round trips, including terminal liquidation; trades/year uses actual calendar years.",
        "- BH-beater medians condition on observed validation performance and are descriptive, not a new selection. A dash means the conditional group is empty or its score undefined. Whole-family medians and full denominators remain alongside them. Cost-specific BH-beater groups can differ; conditional medians are not a same-membership cost comparison.",
        "- This is retrospective validation on previously examined history, not genuinely untouched confirmation. There is no selected validation winner. Historical OOS (2019–May 28, 2026) remains closed and requires a later explicit instruction and sealed inputs.",
        "- Saved-path arithmetic is independently recomputed before sealing/publication; this is not an independent historical price/action/signal replay. Original frozen trend and mean-reversion development sources/results remain unchanged.",
        "",
        "## Evidence",
        "",
        "[All 32 baseline results](candidate_metrics.csv) · [All 128 cost outcomes](candidate_metrics_all_costs.csv) · [Development versus validation](development_vs_validation.csv) · [Frozen development selection](development_selection_metrics.csv) · [Family comparison](family_summary.csv) · [Cost comparisons](family_summary_all_costs.csv) · [Matched benchmarks](benchmark_metrics.csv) · [Protocol summary](summary.json) · [Gap flags](post_gap_candidate_flags.csv) · [Gap events](post_gap_signal_events.csv) · [Artifact hashes](artifact_hashes.json)",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    protocol._write_json(
        output / "artifact_hashes.json",
        {p.name: protocol.sha256_file(p) for p in sorted(output.iterdir())},
    )
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mean-Reversion Study — 32-case VALIDATION ONLY")
    parser.add_argument(
        "command", choices=("check", "seal-selection", "freeze", "validation", "report")
    )
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--development-sidecar", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--bounded-packet", type=Path, default=DEFAULT_BOUNDED_PACKET)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    if args.command == "check":
        # Readiness metadata/development only; no validation numeric data access.
        development.verify_results(safe_path(args.development_sidecar))
        _metadata_binding(safe_path(args.bounded_packet))
        result = {
            "stage": "validation",
            "period": PERIOD,
            "configuration_count": COUNT,
            "shared_confirmation_hours": 3,
            "historical_oos_authorized": False,
            "validation_price_rows_parsed": 0,
            "OOS_numeric_rows_parsed": 0,
        }
    elif args.command == "seal-selection":
        if args.authorization is None:
            parser.error(
                "seal-selection requires --authorization matching the explicit human instruction"
            )
        selection = seal_selection(
            args.sidecar,
            authorization=args.authorization,
            development_sidecar=args.development_sidecar,
            bounded_packet=args.bounded_packet,
        )
        result = {
            "selection_sha256": protocol.canonical_hash(selection),
            "candidate_set_sha256": selection["candidate_set_sha256"],
            "candidate_set_sealed_at": selection["candidate_set_sealed_at"],
            "configuration_count": COUNT,
        }
    elif args.command == "freeze":
        frozen = freeze_study(args.sidecar)
        result = {
            "freeze_sha256": protocol.canonical_hash(frozen),
            "frozen_at": frozen["frozen_at"],
            "manifest_sha256": frozen["manifest_sha256"],
            "historical_oos_authorized": False,
        }
    elif args.command == "validation":
        seal = evaluate_study(args.sidecar)
        result = {
            k: seal[k]
            for k in (
                "stage",
                "configuration_count",
                "configuration_scenario_count",
                "finished_at",
                "historical_oos_authorized",
                "no_winner_selected",
            )
        }
    else:
        result = report(args.sidecar, args.output)
    print(json.dumps(development._clean(result), indent=2, sort_keys=True, allow_nan=False))
    return 0
