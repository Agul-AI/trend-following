"""Separately sealed validation of development's shared 2/3/4-hour indicator pairs.

Selection reads only sealed fixed-3%-cash DEVELOPMENT outcomes. The complete
intersection is frozen before validation prices are parsed. Validation retains
all those pairs at each globally equal entry/exit wait; it never selects a new
winner or unlocks historical OOS. Prior examination makes this retrospective,
not genuinely untouched validation. All previous studies remain immutable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trend_following import research_alpha_reconciliation_v5 as data
from trend_following import research_constant_cash as fixed
from trend_following import research_daily_hourly_v5 as core
from trend_following import research_quality_engine_v5 as engine
from trend_following import research_quality_protocol_v5 as protocol
from trend_following.research_rates import rate_accrual

WORKSPACE = fixed.WORKSPACE
DEFAULT_CONFIG = fixed.DEFAULT_CONFIG
DEFAULT_SOURCE_STUDY = fixed.DEFAULT_SOURCE_STUDY
DEFAULT_DEVELOPMENT_SIDECAR = fixed.DEFAULT_SIDECAR
DEFAULT_SIDECAR = WORKSPACE / ".cache/constant_cash_intersection_validation_20261008"
WAIT_HOURS = (2.0, 3.0, 4.0)
TOLERANCE = 1e-10
STAGE = "validation"
PACKET_FILES = fixed.PACKET_FILES
SOURCE_FILES = (*fixed.SOURCE_FILES, "src/trend_following/research_intersection_validation.py")
COMMISSION_BPS = fixed.COMMISSION_BPS
SLIPPAGE_BPS = fixed.SLIPPAGE_BPS
ANNUAL_CASH_RATE = fixed.ANNUAL_CASH_RATE
Error = protocol.ProtocolError
USER_REQUEST = "keep the shared configs among them out of 289，and do validation"
DEV_EXPORT_COLUMNS = (
    "candidate_id",
    "entry_rule",
    "exit_rule",
    "entry_confirmation_hours",
    "exit_confirmation_hours",
    "cash_excess_sharpe",
    "cagr",
    "max_drawdown",
    "annualized_volatility",
    "exposure_fraction",
    "trade_count",
    "order_count",
    "annualized_turnover",
    "average_holding_hours",
    "has_post_gap_confirmed_signal",
)


def _pair_id(candidate: core.Candidate) -> str:
    return f"{candidate.entry_rule.rule_id}|{candidate.exit_rule.rule_id}"


def _checked_metrics(metrics: pd.DataFrame, candidates: list[core.Candidate]) -> pd.DataFrame:
    """Require complete IDs, identity fields and equal waits, not just row counts."""
    required = {
        "candidate_id",
        "entry_rule",
        "exit_rule",
        "entry_confirmation_hours",
        "exit_confirmation_hours",
        "cash_excess_sharpe",
    }
    permitted = {candidate.candidate_id: candidate for candidate in candidates}
    if (
        not required.issubset(metrics.columns)
        or len(metrics) != len(permitted)
        or metrics.candidate_id.duplicated().any()
        or set(metrics.candidate_id) != set(permitted)
    ):
        raise Error("Metrics must contain every authorized configuration exactly once")
    rows = metrics.set_index("candidate_id", drop=False)
    for candidate_id, candidate in permitted.items():
        row = rows.loc[candidate_id]
        if (
            row.entry_rule != candidate.entry_rule.rule_id
            or row.exit_rule != candidate.exit_rule.rule_id
            or isinstance(row.entry_confirmation_hours, (bool, np.bool_))
            or isinstance(row.exit_confirmation_hours, (bool, np.bool_))
            or row.entry_confirmation_hours != candidate.entry_confirmation_hours
            or row.exit_confirmation_hours != candidate.exit_confirmation_hours
        ):
            raise Error("Metric indicator identity or globally shared wait changed")
    return rows


def select_shared_pairs(metrics: pd.DataFrame, benchmark_sharpe: float) -> tuple[list[dict], dict]:
    """DEV-only intersection: finite Sharpe beats cost-matched BH at ALL three waits."""
    if not np.isfinite(benchmark_sharpe):
        raise Error("Development BH Sharpe must be finite")
    # Validate the entire sealed 13-cohort table before selecting its 3 cohorts.
    candidates = [c for hours in fixed.WAIT_HOURS for c in fixed.candidates_for_wait(hours)]
    rows = _checked_metrics(metrics, candidates)
    eligible, per_wait_counts = {}, {}
    scores = {}
    for hours in WAIT_HOURS:
        hits = set()
        for candidate in fixed.candidates_for_wait(hours):
            score = pd.to_numeric(
                pd.Series([rows.loc[candidate.candidate_id].cash_excess_sharpe]), errors="coerce"
            ).iloc[0]
            pair_id = _pair_id(candidate)
            scores[(pair_id, hours)] = float(score)
            if np.isfinite(score) and score > benchmark_sharpe + TOLERANCE:
                hits.add(pair_id)
        eligible[hours] = hits
        per_wait_counts[str(hours)] = len(hits)
    shared = set.intersection(*(eligible[hours] for hours in WAIT_HOURS))
    if not shared:
        raise Error("No shared development pairs; validation must not start")
    rules = {_pair_id(c): c for c in fixed.candidates_for_wait(WAIT_HOURS[0])}
    pairs = [
        {
            "pair_id": pair_id,
            "entry_rule": rules[pair_id].entry_rule.to_dict(),
            "exit_rule": rules[pair_id].exit_rule.to_dict(),
            "development_sharpe_by_wait": {
                str(hours): scores[(pair_id, hours)] for hours in WAIT_HOURS
            },
        }
        for pair_id in sorted(shared)
    ]
    return pairs, {
        "criterion": "finite_development_daily_cash_excess_Sharpe_above_cost_matched_BH_plus_1e-10_at_ALL_three_shared_waits",
        "development_period": {"start": "2001-01-01", "end": "2011-12-31"},
        "BH_cash_excess_Sharpe": float(benchmark_sharpe),
        "tolerance": TOLERANCE,
        "wait_hours": list(WAIT_HOURS),
        "per_wait_beating_BH_count": per_wait_counts,
        "full_indicator_pairs_per_wait": 289,
        "shared_pair_count": len(pairs),
        "candidate_count": len(pairs) * len(WAIT_HOURS),
        "uses_validation_or_OOS_outcomes": False,
        "filters_gap_flags": False,
        "no_top_N_ranking_or_family_quota": True,
    }


def candidates_from_pairs(pairs: list[dict], hours: float) -> list[core.Candidate]:
    if isinstance(hours, (bool, np.bool_)) or hours not in WAIT_HOURS:
        raise Error("Only globally equal 2, 3 or 4 trading-hour waits are authorized")
    if not isinstance(pairs, list) or not pairs:
        raise Error("Nonempty development-frozen intersection required")
    permitted = {_pair_id(c): c for c in fixed.candidates_for_wait(hours)}
    ids = []
    for pair in pairs:
        if not isinstance(pair, dict) or pair.get("pair_id") not in permitted:
            raise Error("Unrecognized frozen indicator pair")
        candidate = permitted[pair["pair_id"]]
        if (
            pair.get("entry_rule") != candidate.entry_rule.to_dict()
            or pair.get("exit_rule") != candidate.exit_rule.to_dict()
        ):
            raise Error("Frozen indicator pair identity differs from canonical daily rule")
        ids.append(pair["pair_id"])
    if ids != sorted(set(ids)):
        raise Error("Frozen pairs require unique stable sorted pair IDs")
    return [permitted[pair_id] for pair_id in ids]


def _authorization(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise Error("Explicit human authorization receipt required")
    value = json.loads(path.read_text())
    required = {
        "schema": "trend_following_intersection_validation_authorization_v1",
        "authorized_stages": ["validation_only"],
        "explicit_user_approval": True,
        "authorized_by": "human_user",
        "user_request": USER_REQUEST,
        "historical_oos_authorized": False,
        "shared_wait_hours": list(WAIT_HOURS),
        "cash_nominal_annual_rate": ANNUAL_CASH_RATE,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise Error(f"Human validation-only authorization invalid: {key}")
    if (
        value["explicit_user_approval"] is not True
        or value["historical_oos_authorized"] is not False
    ):
        raise Error("Authorization requires explicit true approval and false OOS permission")
    return value


def _safe_roots(sidecar: Path, source: Path, development: Path, authorization: Path | None = None):
    protected = {
        fixed._safe_root(DEFAULT_SOURCE_STUDY),
        source,
        fixed._safe_root(DEFAULT_DEVELOPMENT_SIDECAR),
        development,
    }
    for root in protected:
        if sidecar == root or sidecar in root.parents or root in sidecar.parents:
            raise Error("New validation must use a separate nonoverlapping sidecar tree")
        if authorization is not None and authorization.is_relative_to(root):
            raise Error("New authorization cannot reside inside any protected study")
    if authorization is not None and authorization.is_relative_to(sidecar):
        raise Error("Authorization must be outside the new sidecar")


def _read_validation_metadata(
    source_study: Path, config_path: Path
) -> tuple[dict, dict, dict, dict]:
    """Read provenance metadata only, never old validation/OOS outcomes or DTB3 bytes.

    Validation price hashes are copied from the sealed prepared manifest here;
    those six numeric files are checked and parsed only AFTER the candidate seal.
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
        raise Error("Prepared validation provenance changed")
    root = source_study / "prepared/validation"
    if root.is_symlink() or not root.is_dir() or (root / "manifest.json").is_symlink():
        raise Error("Unsafe/missing validation-only prefix")
    manifest = json.loads((root / "manifest.json").read_text())
    item = prepared["prefixes"][STAGE]
    if (
        protocol.canonical_hash(manifest) != item["manifest_sha256"]
        or manifest.get("files") != item["files"]
        or item.get("end") != "2015-12-31"
        or manifest.get("historical_ready") is not True
        or manifest.get("quality_profile") != prepared["quality_profile"]
    ):
        raise Error("Validation-only manifest/readiness binding changed")
    protocol._require_alpha_price_source(manifest)
    protocol._require_half_hour_clock(manifest)
    if manifest.get("implementation_hashes") != data._implementation_hashes():
        raise Error("Validation implementation provenance changed")
    if not set(PACKET_FILES).issubset(item["files"]):
        raise Error("Validation numeric prefix bindings incomplete")
    return config, registration, prepared, manifest


def _binding_payload(
    source_study: Path, config_path: Path, authorization: Path, development_sidecar: Path
) -> dict:
    _authorization(authorization)
    dev_result = fixed.verify_results(development_sidecar)  # Only DEVELOPMENT bytes.
    dev_freeze = fixed.verify_study(development_sidecar)
    config, registration, prepared, manifest = _read_validation_metadata(source_study, config_path)
    if dev_freeze["source_study"] != str(source_study) or dev_freeze["config_path"] != str(
        config_path
    ):
        raise Error("Development selection must originate from the same frozen source/config")
    dev_metrics = pd.read_csv(
        development_sidecar / "results/candidate_metrics.csv", float_precision="round_trip"
    )
    dev_counts = pd.read_csv(
        development_sidecar / "results/counts_by_wait.csv", float_precision="round_trip"
    )
    counts = fixed.counts_against_bh(dev_metrics, dev_result["BH_cash_excess_Sharpe"])
    pd.testing.assert_frame_equal(dev_counts, counts, check_dtype=False, rtol=1e-12, atol=1e-14)
    pairs, selection = select_shared_pairs(dev_metrics, dev_result["BH_cash_excess_Sharpe"])
    candidates = [c for hours in WAIT_HOURS for c in candidates_from_pairs(pairs, hours)]
    candidate_set = [{"candidate_id": c.candidate_id, **c.to_dict()} for c in candidates]
    selected_rows = (
        dev_metrics.set_index("candidate_id", drop=False)
        .loc[[c.candidate_id for c in candidates], list(DEV_EXPORT_COLUMNS)]
        .to_dict("records")
    )
    return {
        "schema": "trend_following_constant_cash_intersection_validation_v1",
        "scope": "validation_only_all_development_shared_pairs_no_reselection",
        "source_study": str(source_study),
        "config_path": str(config_path),
        "development_sidecar": str(development_sidecar),
        "authorization": {
            "path": str(authorization),
            "sha256": protocol.sha256_file(authorization),
        },
        "registration_sha256": protocol.canonical_hash(registration),
        "prepared_metadata_sha256": protocol.canonical_hash(prepared),
        "validation_manifest_sha256": protocol.canonical_hash(manifest),
        "validation_files": {
            name: prepared["prefixes"][STAGE]["files"][name] for name in PACKET_FILES
        },
        "development_freeze_sha256": protocol.canonical_hash(dev_freeze),
        "development_result_sha256": protocol.canonical_hash(dev_result),
        "development_selection_files": {
            name: dev_result["files"][name]
            for name in ("candidate_metrics.csv", "counts_by_wait.csv", "benchmark_metrics.csv")
        },
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
        "configuration_count_per_wait": len(pairs),
        "configuration_count": len(candidates),
        "stage": STAGE,
        "period": config["segments"][STAGE],
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
        "historical_oos_authorized": False,
        "historical_label": "retrospective_validation_history_previously_examined_not_untouched",
        "prior_validation_or_OOS_outcomes_used_for_selection": False,
    }


def seal_study(
    sidecar: str | Path = DEFAULT_SIDECAR,
    *,
    authorization: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    source_study: str | Path = DEFAULT_SOURCE_STUDY,
    development_sidecar: str | Path = DEFAULT_DEVELOPMENT_SIDECAR,
) -> dict:
    sidecar, authorization, config_path, source_study, development_sidecar = (
        fixed._safe_root(path)
        for path in (sidecar, authorization, config_path, source_study, development_sidecar)
    )
    _safe_roots(sidecar, source_study, development_sidecar, authorization)
    with protocol._attempt(sidecar, "seal_development_intersection"):
        binding = _binding_payload(source_study, config_path, authorization, development_sidecar)
        target = sidecar / "freeze.json"
        if target.exists():
            payload = protocol._read_seal(target)
            if {
                key: value for key, value in payload.items() if key != "candidate_set_sealed_at"
            } != binding:
                raise Error("Already sealed differently: preserve old tree; do not overwrite")
            _chronology(payload["candidate_set_sealed_at"])
        else:
            payload = {**binding, "candidate_set_sealed_at": protocol._utc_now()}
            protocol._seal(target, payload)
        return payload


def _chronology(*times: str):
    parsed = [pd.Timestamp(time) for time in times]
    if any(time.tzinfo is None for time in parsed) or parsed != sorted(parsed):
        raise Error("Seal/outcome chronology must be timezone-aware and monotonic")


def verify_study(sidecar: str | Path = DEFAULT_SIDECAR) -> dict:
    sidecar = fixed._safe_root(sidecar)
    payload = protocol._read_seal(sidecar / "freeze.json")
    source, config, authorization, development = (
        fixed._safe_root(path)
        for path in (
            payload["source_study"],
            payload["config_path"],
            payload["authorization"]["path"],
            payload["development_sidecar"],
        )
    )
    _safe_roots(sidecar, source, development, authorization)
    expected = _binding_payload(source, config, authorization, development)
    if {
        key: value for key, value in payload.items() if key != "candidate_set_sealed_at"
    } != expected:
        raise Error("Frozen intersection/code/runtime/data/assumptions changed")
    _chronology(payload["candidate_set_sealed_at"])
    return payload


def load_validation_packet(payload: dict):
    """Own bounded reader: ONLY prepared/validation six whitelisted price files."""
    if payload.get("stage") != STAGE or payload.get("historical_oos_authorized") is not False:
        raise Error("Validation-only candidate freeze required")
    if payload.get("period") != {"start": "2012-01-01", "end": "2015-12-31"}:
        raise Error("Validation period differs from explicit frozen endpoint")
    _chronology(payload["candidate_set_sealed_at"])
    root = fixed._safe_root(payload["source_study"]) / "prepared/validation"
    if root.is_symlink() or not root.is_dir() or (root / "manifest.json").is_symlink():
        raise Error("Unsafe validation-only packet root")
    if set(payload.get("validation_files", {})) != set(PACKET_FILES):
        raise Error("Unsafe validation-only whitelist")
    manifest = json.loads((root / "manifest.json").read_text())
    if protocol.canonical_hash(manifest) != payload["validation_manifest_sha256"]:
        raise Error("Validation manifest changed before numeric reads")
    for name, expected in payload["validation_files"].items():
        path = root / name
        if path.is_symlink() or protocol.sha256_file(path) != expected["sha256"]:
            raise Error(f"Validation-only packet bytes changed: {name}")
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
        if (
            frame.empty
            or frame.session.gt("2015-12-31").any()
            or frame.session.lt("1999-11-01").any()
        ):
            raise Error("Validation reader refuses empty/future/pre-seed sessions")
    if not blackouts.empty and blackouts.end.gt(protocol._valuation_end_utc("2015-12-31")).any():
        raise Error("Validation reader refuses future blackout metadata")
    proof = json.loads((root / "unit_proof.json").read_text())
    data._validate_unit_proof(proof)  # Metadata validation does not follow original source paths.
    bindings = proof["data_bindings"]
    for key, frame, columns in (
        ("daily_session_hashes", daily, data.DAILY_BINDING_COLUMNS),
        ("bar_session_hashes", bars, data.BAR_BINDING_COLUMNS),
        ("calendar_session_hashes", calendar, data.CALENDAR_BINDING_COLUMNS),
        ("coverage_session_hashes", coverage, data.COVERAGE_BINDING_COLUMNS),
    ):
        actual = data._session_hashes(frame, columns)
        if any(bindings[key].get(session) != value for session, value in actual.items()):
            raise Error(f"Validation numeric proof changed: {key}")
    if any(
        data._row_hash(row, data.BLACKOUT_BINDING_COLUMNS) not in bindings["blackout_row_hashes"]
        for _, row in blackouts.iterrows()
    ):
        raise Error("Validation blackout proof changed")
    calendar.attrs["calendar_provenance"] = manifest.get("calendar_provenance", {})
    daily.attrs["source_metadata"] = manifest["daily_source_metadata"]
    bars.attrs["source_metadata"] = manifest["source_metadata"]
    bars.attrs["unit_proof"] = proof
    config = protocol.load_config(payload["config_path"])
    cash = fixed.constant_cash_observations(
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


def counts_against_bh(
    metrics: pd.DataFrame, benchmark_sharpe: float, pairs: list[dict]
) -> pd.DataFrame:
    if not np.isfinite(benchmark_sharpe):
        raise Error("Validation BH Sharpe must be finite")
    permitted = [c for hours in WAIT_HOURS for c in candidates_from_pairs(pairs, hours)]
    rows = _checked_metrics(metrics, permitted)
    records = []
    for hours in WAIT_HOURS:
        subset = rows.loc[[c.candidate_id for c in candidates_from_pairs(pairs, hours)]]
        scores = pd.to_numeric(subset.cash_excess_sharpe, errors="coerce").to_numpy(float)
        finite = np.isfinite(scores)
        beat = finite & (scores > benchmark_sharpe + TOLERANCE)
        trades = pd.to_numeric(subset.trade_count, errors="coerce").to_numpy(float)
        if (
            not np.isfinite(trades).all()
            or (trades < 0).any()
            or (trades != np.floor(trades)).any()
        ):
            raise Error("Trade counts must be nonnegative completed round trips")
        records.append(
            {
                "wait_hours": hours,
                "required_consecutive_hits": 1 + int(2 * hours),
                "configuration_count": len(pairs),
                "finite_Sharpe_count": int(finite.sum()),
                "undefined_Sharpe_count": int((~finite).sum()),
                "beat_BH_Sharpe_count": int(beat.sum()),
                "beat_BH_Sharpe_pct": float(100 * beat.sum() / len(pairs)),
                "tie_BH_Sharpe_count": int(
                    (finite & (np.abs(scores - benchmark_sharpe) <= TOLERANCE)).sum()
                ),
                "below_BH_Sharpe_count": int(
                    (finite & (scores < benchmark_sharpe - TOLERANCE)).sum()
                ),
                "BH_cash_excess_Sharpe": float(benchmark_sharpe),
                "median_finite_Sharpe": float(np.median(scores[finite]))
                if finite.any()
                else np.nan,
                "best_finite_Sharpe": float(np.max(scores[finite])) if finite.any() else np.nan,
                "median_trade_count": float(np.median(trades)),
                "cash_nominal_annual_rate": ANNUAL_CASH_RATE,
                "commission_bps": COMMISSION_BPS,
                "slippage_bps": SLIPPAGE_BPS,
            }
        )
    return pd.DataFrame(records)


def evaluate_study(sidecar: str | Path = DEFAULT_SIDECAR) -> dict:
    sidecar = fixed._safe_root(sidecar)
    with protocol._attempt(sidecar, "validation_only") as attempt:
        payload = verify_study(sidecar)  # Complete candidate set sealed before any price arrays.
        if (sidecar / "results").exists():
            raise Error("Validation already completed; verify it, never overwrite/rerun")
        started_at = protocol._utc_now()
        _chronology(payload["candidate_set_sealed_at"], started_at)
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
        daily, bars, calendar, coverage, blackouts, cash, config = load_validation_packet(payload)
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
                f"Completed validation common wait {hours:g} h: {len(candidates)} development-frozen pairs",
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
            "schema": "trend_following_constant_cash_intersection_validation_results_v1",
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
            "validation_manifest_sha256": payload["validation_manifest_sha256"],
            "files": {path.name: protocol.sha256_file(path) for path in sorted(output.iterdir())},
            "no_winner_selected": True,
            "no_validation_filtering_or_replacement": True,
            "no_OOS_market_files_or_prior_validation_outcomes_loaded": True,
            "historical_oos_authorized": False,
        }
        _chronology(
            result["candidate_set_sealed_at"],
            result["market_outcomes_started_at"],
            result["finished_at"],
        )
        protocol._seal(output / "seal.json", result)
        output.rename(sidecar / "results")
        return result


def verify_results(sidecar: str | Path = DEFAULT_SIDECAR) -> dict:
    sidecar = fixed._safe_root(sidecar)
    payload = verify_study(sidecar)
    result = protocol._read_seal(sidecar / "results/seal.json")
    expected = {
        "schema": "trend_following_constant_cash_intersection_validation_results_v1",
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
        "validation_manifest_sha256": payload["validation_manifest_sha256"],
        "no_winner_selected": True,
        "no_validation_filtering_or_replacement": True,
        "no_OOS_market_files_or_prior_validation_outcomes_loaded": True,
        "historical_oos_authorized": False,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise Error("Validation result contract/counts/identity/costs changed")
    _chronology(
        result["candidate_set_sealed_at"],
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
    """Publication gate: never open numeric outputs before intact freeze/result seals."""
    result = verify_results(sidecar)
    return protocol._read_seal(fixed._safe_root(sidecar) / "freeze.json"), result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "evaluate", "verify"))
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-study", type=Path, default=DEFAULT_SOURCE_STUDY)
    parser.add_argument("--development-sidecar", type=Path, default=DEFAULT_DEVELOPMENT_SIDECAR)
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
