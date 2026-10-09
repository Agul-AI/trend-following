"""Sealed multi-scale QQQ research; bounded development and validation only.

The independent-sleeve experiment is reused unchanged. New filtered mean-
reversion paths are separately sealed and never authorize 2019+ price access.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from trend_following import research_constant_cash as fixed
from trend_following import research_mean_reversion as mr
from trend_following import research_mean_reversion_engine as mr_engine
from trend_following import research_mean_reversion_validation as mr_val
from trend_following import research_portfolio_engine as portfolio
from trend_following import research_portfolio_protocol as preserved
from trend_following import research_quality_engine_v5 as quality
from trend_following import research_quality_protocol_v5 as custody
from trend_following import research_validation_prefix as prefix

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/qqq_multi_scale_research.yaml"
DEFAULT_SIDECAR = ROOT / ".cache/multi_scale_research_20261009_checked"
DEFAULT_REPORT = ROOT / "reports/multi_scale_research/20261009"
DEFAULT_PORTFOLIO = preserved.DEFAULT_SIDECAR
PERIODS = copy.deepcopy(preserved.PERIODS)
SLIPPAGES = preserved.SLIPPAGES
POLICIES = ("entry_only", "recovery_or_trend_break")
GATE_PERIODS = (100, 200)
TOLERANCE = 1e-10
APPROVAL_EXCERPT = (
    "PLEASE IMPLEMENT THIS PLAN:\n# Long-Term Trend + Short-Term Mean-Reversion Comparison"
)
Error = custody.ProtocolError
safe_path = preserved.safe_path
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *preserved.SOURCE_FILES,
            "src/trend_following/research_multi_scale_engine.py",
            "src/trend_following/research_multi_scale_protocol.py",
            "src/trend_following/research_multi_scale_report.py",
            "scripts/run_multi_scale_research.py",
        )
    )
)
DIAGNOSTICS = (
    "blocked_oversold_checks",
    "trend_break_exit_count",
    "entry_cancellation_count",
    "exit_cancellation_count",
    "unknown_gate_reset_count",
    "cancelled_blackout_order_count",
    "cancelled_terminal_order_count",
)


def baskets():
    return [
        {
            "basket_id": f"multi_scale_sma{period}_{policy}",
            "gate_period": period,
            "exit_policy": policy,
        }
        for policy in POLICIES
        for period in GATE_PERIODS
    ]


def approved_config():
    return {
        "schema": "qqq_multi_scale_research",
        "study_name": "QQQ Multi-Scale Trend Following Study",
        "security": "QQQ",
        "historical_label": "retrospective_hindsight_conditioned_membership",
        "membership": {
            "trend_count": 12,
            "mean_reversion_count": 32,
            "total_count": 44,
            "reuse_exact_sealed_portfolio_membership": True,
            "filtered_rule_count": 128,
            "filtered_basket_count": 4,
            "shared_confirmation_hours": 3.0,
            "required_checks": 7,
        },
        "construction": {
            "initial_cash": 1000.0,
            "fixed_initial_capital_sleeves": True,
            "equal_initial_weights_within_basket": True,
            "portfolio_rebalancing": False,
            "capital_redistribution": False,
            "order_netting": False,
            "independent_portfolios": "reuse_existing_seals_without_extension",
        },
        "gate": {
            "family": "SMA",
            "periods_days": list(GATE_PERIODS),
            "availability": "finalized_daily_17_NY",
            "includes_finalized_observation": True,
            "support": "signal_value_strictly_above_SMA",
            "equality": "OFF",
            "missing_required_history": "UNKNOWN",
            "intraday_preview": False,
        },
        "execution": {
            "exit_policies": list(POLICIES),
            "separate_recovery_and_trend_break_streaks": True,
            "next_safe_observed_open": True,
            "cancel_unfilled_entry_when_gate_OFF": True,
            "cancel_trend_break_only_exit_when_gate_ON": True,
            "UNKNOWN_holds_cancels_resets": True,
            "gap_flags_selection_role": False,
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
        },
        "selection": {
            "metric": "daily_cash_excess_sharpe",
            "annualization": 252,
            "standard_deviation_ddof": 1,
            "finite_scores_only": True,
            "one_filter_per_exit_policy": True,
            "tie_tolerance": TOLERANCE,
            "tie_order": "stable_basket_id",
            "baseline_only": True,
            "trade_return_drawdown_BH_filters": False,
        },
        "validation": {
            "all_four_predeclared_filtered_baskets": True,
            "gate_only_controls": list(GATE_PERIODS),
            "reuse_old_A_controls_only": True,
            "no_reselection": True,
        },
        "bootstrap": {
            "replicates": 2000,
            "block_lengths": [5, 20, 60],
            "seed": 20261009,
            "paired": True,
            "confidence_level": 0.95,
            "references": ["locked_independent", "ungated_mean_reversion", "buy_and_hold"],
            "deduplicate_identical_reference_paths_keep_aliases": True,
        },
        "protocol": {
            "authorized_stages": ["development", "validation"],
            "historical_oos_authorized": False,
            "fresh_segment_cash_and_strategy_state": True,
            "hindsight_conditioned_membership": True,
        },
    }


def load_config(path=DEFAULT_CONFIG):
    value = yaml.safe_load(safe_path(path).read_text())
    if custody.canonical_hash(value) != custody.canonical_hash(approved_config()):
        raise Error("Multi-scale configuration differs from the exact approved contract")
    return value


def authorization(path):
    value = json.loads(safe_path(path).read_text())
    expected = {
        "schema": "qqq_multi_scale_authorization",
        "authorized_by": "human_user",
        "explicit_user_approval": True,
        "authorized_stages": ["development", "validation"],
        "approval_excerpt": APPROVAL_EXCERPT,
        "configuration_count": 44,
        "trend_count": 12,
        "mean_reversion_count": 32,
        "filtered_rule_count": 128,
        "filtered_basket_count": 4,
        "shared_confirmation_hours": 3.0,
        "development_period": PERIODS["development"],
        "validation_period": PERIODS["validation"],
        "historical_oos_authorized": False,
        "no_constituent_retuning_or_replacement": True,
        "reuse_sealed_independent_portfolios": True,
    }
    if any(
        custody.canonical_hash(value.get(k)) != custody.canonical_hash(v)
        for k, v in expected.items()
    ):
        raise Error(
            "Explicit matching human multi-scale development/validation authorization required"
        )
    return value


def _separate(sidecar, source, auth=None):
    if sidecar.parent != ROOT / ".cache" or not sidecar.name.startswith("multi_scale_research_"):
        raise Error("New top-level multi_scale_research_ private sidecar required")
    if sidecar == source or sidecar in source.parents or source in sidecar.parents:
        raise Error("New tree must not overlap preserved studies")
    if auth is not None and (auth.is_relative_to(source) or auth.is_relative_to(sidecar)):
        raise Error("Authorization must remain outside all study trees")


def _bindings(config_path, auth, source):
    from trend_following import research_multi_scale_engine as engine

    load_config(config_path)
    authorization(auth)
    old = preserved.verify_study(source)
    old_seals = {stage: preserved.verify_results(source, stage) for stage in PERIODS}
    old_lock = preserved.verify_locked_allocation(source)
    membership = preserved.validate_membership(old["membership"])
    candidates = [mr.Candidate.from_dict(r["parameters"]) for r in membership[12:]]
    gated = engine.candidate_grid(candidates)
    if len(gated) != 128 or len({c.candidate_id for c in gated}) != 128:
        raise Error("Exactly 128 unique filtered rules required")
    definitions = [
        {"candidate_id": c.candidate_id, "basket_id": c.basket_id, "parameters": c.to_dict()}
        for c in gated
    ]
    if {r["basket_id"] for r in definitions} != {b["basket_id"] for b in baskets()}:
        raise Error("Filtered rules changed the four predefined baskets")
    return {
        "schema": "qqq_multi_scale_freeze",
        "config_path": str(config_path),
        "config_sha256": custody.sha256_file(config_path),
        "configuration_sha256": custody.canonical_hash(approved_config()),
        "authorization": {"path": str(auth), "sha256": custody.sha256_file(auth)},
        "preserved_portfolio": str(source),
        "preserved_freeze_sha256": custody.canonical_hash(old),
        "preserved_result_hashes": {s: custody.canonical_hash(v) for s, v in old_seals.items()},
        "preserved_lock_sha256": custody.canonical_hash(old_lock),
        "independent_locked_allocation": old_lock["allocation"],
        "membership": membership,
        "membership_sha256": custody.canonical_hash(membership),
        "gated_candidates": definitions,
        "gated_candidates_sha256": custody.canonical_hash(definitions),
        "baskets": baskets(),
        "gate_only_controls": [c.to_dict() for c in engine.gate_only_candidates()],
        "development_adapter": old["development_adapter"],
        "validation_packet": old["validation_packet"],
        "validation_metadata": old["validation_metadata"],
        "source_hashes": {n: custody.sha256_file(ROOT / n) for n in SOURCE_FILES},
        "runtime": custody._runtime(),
        "cash": fixed.cash_policy(),
        "periods": PERIODS,
        "initial_cash": 1000.0,
        "commission_bps": 1.0,
        "slippage_bps_scenarios": list(SLIPPAGES),
        "hindsight_conditioned_membership": True,
        "historical_oos_authorized": False,
        "OOS_numeric_rows_parsed": 0,
    }


def freeze_study(
    sidecar=DEFAULT_SIDECAR,
    *,
    authorization,
    config_path=DEFAULT_CONFIG,
    preserved_portfolio=DEFAULT_PORTFOLIO,
):
    sidecar, auth, config, source = map(
        safe_path, (sidecar, authorization, config_path, preserved_portfolio)
    )
    _separate(sidecar, source, auth)
    with custody._attempt(sidecar, "freeze"):
        payload = _bindings(config, auth, source)
        target = sidecar / "freeze.json"
        if target.exists():
            old = custody._read_seal(target)
            if {k: v for k, v in old.items() if k != "frozen_at"} != payload:
                raise Error("Already frozen differently; never overwrite")
            return old
        payload["frozen_at"] = custody._utc_now()
        custody._seal(target, payload)
        return payload


def verify_study(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    old = custody._read_seal(sidecar / "freeze.json")
    auth, config, source = map(
        safe_path, (old["authorization"]["path"], old["config_path"], old["preserved_portfolio"])
    )
    _separate(sidecar, source, auth)
    if {k: v for k, v in old.items() if k != "frozen_at"} != _bindings(config, auth, source):
        raise Error("Frozen source/data/membership/runtime/authorization changed")
    mr_val._chronology(old["frozen_at"])
    return old


def load_packet(frozen, stage, *, sidecar=DEFAULT_SIDECAR):
    if stage not in PERIODS:
        raise Error("Only development/validation; 2019+ remains closed")
    verified = verify_study(sidecar)
    if custody.canonical_hash(frozen) != custody.canonical_hash(verified):
        raise Error("Caller packet contract differs from verified active freeze")
    if stage == "development":
        return fixed.load_development_packet(frozen["development_adapter"])
    verify_locked_selections(sidecar)  # Before any newly opened validation price rows.
    packet = safe_path(frozen["validation_packet"])
    daily, bars, calendar, coverage, blackouts, _ = prefix.read_validation_prefix(
        packet, expected_manifest_sha256=frozen["validation_metadata"]["manifest_sha256"]
    )
    config = copy.deepcopy(custody.load_config(frozen["development_adapter"]["config_path"]))
    config["segments"]["validation"] = dict(PERIODS["validation"])
    config["segments"]["historical-oos"] = {"start": "2019-01-01", "end": "2026-05-28"}
    cash = fixed.constant_cash_observations(
        calendar.market_open.iloc[0], daily.daily_price_available_at.iloc[-1]
    )
    custody._check_prefix(daily, bars, calendar, cash, "validation", config)
    custody._quality_gate(
        config, coverage, frozen["validation_metadata"]["manifest"]["gap_authorization"]
    )
    return daily, bars, calendar, coverage, blackouts, cash, config


def select_policy_winners(metrics):
    baseline = metrics.loc[metrics.scenario_slippage_bps.eq(3.0)].copy()
    canonical = {b["basket_id"]: b for b in baskets()}
    if (
        len(baseline) != 4
        or baseline.basket_id.duplicated().any()
        or set(baseline.basket_id) != set(canonical)
    ):
        raise Error("Selection requires exactly four baseline predefined baskets")
    for row in baseline.itertuples(index=False):
        expected = canonical[row.basket_id]
        if row.gate_period != expected["gate_period"] or row.exit_policy != expected["exit_policy"]:
            raise Error("Basket daily filter or exit policy changed")
    winners = []
    for policy in POLICIES:
        finite = baseline.loc[
            baseline.exit_policy.eq(policy) & np.isfinite(baseline.cash_excess_sharpe)
        ]
        if finite.empty:
            raise Error(f"No finite development score for {policy}; stop")
        best = finite.cash_excess_sharpe.max()
        cid = min(finite.loc[finite.cash_excess_sharpe.ge(best - TOLERANCE), "basket_id"])
        winners.append(canonical[cid])
    return winners


def seal_locked_selections(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    dev = verify_results(sidecar, "development")
    selections = select_policy_winners(
        pd.read_csv(sidecar / "development/metrics.csv", float_precision="round_trip")
    )
    payload = {
        "schema": "qqq_multi_scale_policy_locks",
        "development_result_sha256": custody.canonical_hash(dev),
        "selections": selections,
        "baseline_slippage_bps": 3.0,
        "commission_bps": 1.0,
        "historical_oos_authorized": False,
        "validation_reselection": False,
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
    old = custody._read_seal(sidecar / "locked_selections.json")
    expected = {
        "schema": "qqq_multi_scale_policy_locks",
        "development_result_sha256": custody.canonical_hash(dev),
        "selections": select_policy_winners(
            pd.read_csv(sidecar / "development/metrics.csv", float_precision="round_trip")
        ),
        "baseline_slippage_bps": 3.0,
        "commission_bps": 1.0,
        "historical_oos_authorized": False,
        "validation_reselection": False,
    }
    if {k: v for k, v in old.items() if k != "sealed_at"} != expected:
        raise Error("Policy locks differ from sealed development-only selections")
    mr_val._chronology(dev["finished_at"], old["sealed_at"])
    return old


def basket_weights(frozen, basket_id):
    ids = [r["basket_id"] for r in frozen["gated_candidates"]]
    weights = np.array([1 / 32 if cid == basket_id else 0 for cid in ids])
    if np.count_nonzero(weights) != 32 or not np.isclose(weights.sum(), 1, atol=1e-15, rtol=0):
        raise Error("Each predefined basket must have exactly thirty-two equal initial sleeves")
    return weights


def _assert_close(actual, expected, message):
    preserved._assert_close(actual, expected, message)


def _save_combined(path, ids, results, cash):
    np.savez_compressed(
        path,
        portfolio_id=np.array(ids, dtype=str),
        timestamp=np.array([t.isoformat() for t in results[0].daily_equity.index], dtype=str),
        equity=np.column_stack([r.daily_equity.to_numpy() for r in results]),
        returns=np.column_stack([r.daily_returns.to_numpy() for r in results]),
        cash_returns=cash,
        exposure=np.column_stack([r.daily_exposure.to_numpy() for r in results]),
    )


def expected_control_ids(frozen, stage):
    if stage not in PERIODS:
        raise Error("Only approved controls from development/validation")
    if stage == "development":
        independent = [a.portfolio_id for a in portfolio.allocation_grid()]
    else:
        independent = list(
            dict.fromkeys(
                [
                    frozen["independent_locked_allocation"]["portfolio_id"],
                    "portfolio_tf_100",
                    "portfolio_tf_000",
                    "portfolio_tf_050",
                    "portfolio_equal_44",
                ]
            )
        )
    return [
        *independent,
        "buy_and_hold",
        "cash",
        "multi_scale_sma100_only",
        "multi_scale_sma200_only",
    ]


def _read_old_controls(frozen, stage, slip, times, cash):
    source = safe_path(frozen["preserved_portfolio"])
    preserved.verify_results(source, stage)
    lock = preserved.verify_locked_allocation(source)
    allocations = (
        preserved.eligible_validation_allocations(lock)
        if stage == "validation"
        else portfolio.allocation_grid()
    )
    ids = [a.portfolio_id for a in allocations]
    directory = source / stage
    with np.load(directory / f"allocation_s{slip:g}_daily.npz", allow_pickle=False) as saved:
        if list(saved["portfolio_id"]) != ids or not pd.DatetimeIndex(
            pd.to_datetime(saved["timestamp"], utc=True)
        ).equals(times):
            raise Error("Existing independent portfolio IDs/clocks changed; no interpolation")
        _assert_close(saved["cash_returns"], cash, "Existing independent cash reference differs")
        paths = {
            pid: {
                "equity": saved["equity"][:, j].copy(),
                "returns": saved["returns"][:, j].copy(),
                "exposure": saved["exposure"][:, j].copy(),
            }
            for j, pid in enumerate(ids)
        }
    metrics = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    rows = metrics.loc[metrics.scenario_slippage_bps.eq(slip)].copy()
    if set(rows.portfolio_id) != set(ids) or len(rows) != len(ids):
        raise Error("Only existing sealed independent allocations may be reused")
    benchmarks = pd.read_csv(directory / "benchmark_metrics.csv", float_precision="round_trip")
    for mode in ("buy_and_hold", "cash"):
        frame = pd.read_csv(directory / f"{mode}_s{slip:g}_daily.csv", float_precision="round_trip")
        if not pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)).equals(times):
            raise Error("Existing passive benchmark clock changed")
        _assert_close(frame.cash_return, cash, "Existing passive cash reference changed")
        paths[mode] = {
            "equity": frame.equity.to_numpy(),
            "returns": frame["return"].to_numpy(),
            "exposure": frame.exposure.to_numpy(),
        }
    constituent_metrics = pd.read_csv(
        directory / f"constituent_metrics_s{slip:g}.csv", float_precision="round_trip"
    )
    if list(constituent_metrics.candidate_id) != [r["candidate_id"] for r in frozen["membership"]]:
        raise Error("Saved independent constituent metrics changed membership")
    for allocation in allocations:
        active = constituent_metrics.loc[allocation.weights() > 0]
        trips = int(active.trade_count.sum())
        total_hold = float((active.average_holding_hours * active.trade_count).sum())
        mask = rows.portfolio_id.eq(allocation.portfolio_id)
        rows.loc[mask, "gross_sleeve_completed_round_trips"] = trips
        rows.loc[mask, "average_gross_sleeve_holding_hours"] = total_hold / trips if trips else 0.0
        rows.loc[mask, "holding_duration_definition"] = (
            "mean_elapsed_calendar_hours_per_completed_gross_sleeve_round_trip"
        )
    benchmark_rows = benchmarks.loc[benchmarks.scenario_slippage_bps.eq(slip)].copy()

    return pd.concat([rows, benchmark_rows], ignore_index=True), paths


def _compare_preserved_sleeves(parts, candidates, path):
    """Read-only adapter for the sealed portfolio's event-aware sleeve archive.

    The old portfolio NPZ has no daily binary ``exposure`` field. Select its
    exact event-long rows at daily timestamps instead; never interpolate,
    forward-fill or confuse binary position with actual notional exposure.
    """
    with np.load(safe_path(path), allow_pickle=False) as saved:
        required = {
            "candidate_id",
            "timestamp",
            "equity",
            "returns",
            "cash_returns",
            "event_timestamp",
            "event_long",
        }
        if not required.issubset(saved.files):
            raise Error("Preserved event-aware sleeve archive lacks required fields")
        ids = list(saved["candidate_id"])
        if len(ids) != len(set(ids)) or any(c.candidate_id not in ids for c in candidates):
            raise Error("Preserved sleeve archive lacks exact unchanged constituent IDs")
        columns = [ids.index(c.candidate_id) for c in candidates]
        times = pd.DatetimeIndex(pd.to_datetime(saved["timestamp"], utc=True))
        events = pd.DatetimeIndex(pd.to_datetime(saved["event_timestamp"], utc=True))
        if (
            times.has_duplicates
            or not times.is_monotonic_increasing
            or events.has_duplicates
            or not events.is_monotonic_increasing
            or not times.equals(parts[1])
        ):
            raise Error("Preserved sleeve daily/event clocks changed")
        rows = events.get_indexer(times)
        if (rows < 0).any():
            raise Error("Every preserved daily mark must be an exact observed event timestamp")
        event_long = saved["event_long"]
        if event_long.shape != (len(events), len(ids)) or not np.isin(event_long, [0, 1]).all():
            raise Error("Preserved event-long positions must be aligned binary values")
        for key, index in (("equity", 2), ("returns", 3)):
            if saved[key].shape != (len(times), len(ids)):
                raise Error("Preserved sleeve daily arrays do not match the sealed clock")
            _assert_close(
                parts[index], saved[key][:, columns], f"MR replay differs from preserved {key}"
            )
        _assert_close(
            parts[4], saved["cash_returns"], "MR replay differs from preserved cash reference"
        )
        _assert_close(
            parts[5],
            event_long[rows][:, columns],
            "MR replay differs from preserved daily binary positions",
        )


def _parity_preflight(cache, candidates, kwargs, frozen, stage):
    from trend_following import research_multi_scale_engine as engine

    bases = [mr.Candidate.from_dict(r["parameters"]) for r in frozen["membership"][12:]]
    original = mr_engine.run(cache.base, bases, **kwargs)
    _compare_preserved_sleeves(
        original,
        bases,
        safe_path(frozen["preserved_portfolio"])
        / stage
        / f"constituent_s{kwargs['slippage_bps']:g}_daily.npz",
    )
    always = engine.run(engine.with_constant_gate(cache, 1), candidates, **kwargs)
    if not original[1].equals(always[1]):
        raise Error("Always-ON gate altered original daily clock")
    _assert_close(always[4], original[4], "Always-ON gate altered cash reference")
    left = pd.DataFrame(original[6]).copy()
    for basket in baskets():
        columns = [j for j, c in enumerate(candidates) if c.basket_id == basket["basket_id"]]
        members = [candidates[j] for j in columns]
        if len(members) != 32 or [c.base.candidate_id for c in members] != [
            c.candidate_id for c in bases
        ]:
            raise Error("Always-ON preflight requires all four unchanged 32-member baskets")
        for index in (2, 3, 5):
            _assert_close(
                always[index][:, columns],
                original[index],
                "Always-ON gate altered original MR paths",
            )
        right = pd.DataFrame(always[6]).copy()
        if not right.empty:
            mapping = {c.candidate_id: c.base.candidate_id for c in members}
            right = right.loc[right.candidate_id.isin(mapping)].copy()
            right["candidate_id"] = right.candidate_id.map(mapping)
        try:
            pd.testing.assert_frame_equal(
                left.reset_index(drop=True),
                right.reindex(columns=list(left.columns)).reset_index(drop=True),
                check_dtype=False,
                rtol=1e-10,
                atol=1e-10,
            )
        except AssertionError as exc:
            raise Error("Always-ON gate altered original MR fills or accounting") from exc
        for j, column in enumerate(columns):
            a, b = original[0][j], always[0][column]
            for field in (
                "terminal_equity",
                "gross_terminal_equity",
                "total_commission",
                "total_slippage",
                "order_count",
                "trade_count",
                "exposure_fraction",
                "average_holding_hours",
                "total_turnover",
                "cost_drag_return",
            ):
                _assert_close(a[field], b[field], f"Always-ON constituent {j} {field} differs")
    never = engine.run(engine.with_constant_gate(cache, 0), candidates, **kwargs)
    passive = mr_engine.run(cache.base, bases[:1], mode="cash", **kwargs)
    if not never[1].equals(passive[1]) or never[6]:
        raise Error("Always-OFF preflight must retain cash with no fills")
    for index in (2, 3):
        _assert_close(
            never[index],
            np.repeat(passive[index], len(candidates), axis=1),
            "Always-OFF cash path differs",
        )
    _assert_close(never[4], passive[4], "Always-OFF cash reference differs")
    _assert_close(never[5], np.zeros_like(never[5]), "Always-OFF cannot hold QQQ")
    for row in never[0]:
        if any(
            row[field] != 0
            for field in ("total_commission", "total_slippage", "order_count", "trade_count")
        ):
            raise Error("Always-OFF cash charged a trading cost")
    return {
        "scenario_slippage_bps": kwargs["slippage_bps"],
        "original_constituent_count": 32,
        "filtered_rule_count": 128,
        "basket_count": 4,
        "always_ON_paths_fills_charges_match": True,
        "always_OFF_cash_reconciles": True,
        "saved_preserved_paths_match": True,
    }


def paired_bootstrap(
    selected, references, cash, *, replicates=2000, block_lengths=(5, 20, 60), seed=20261009
):
    """Paired circular moving blocks; identical reference arrays retain aliases."""
    if not selected or not references or replicates <= 0:
        raise Error("Finite paired-bootstrap inputs required")
    ids = list(selected)
    values = [np.asarray(selected[x], dtype=float) for x in ids]
    cash = np.asarray(cash, dtype=float)
    unique = []
    for alias, value in references.items():
        value = np.asarray(value, dtype=float)
        match = next((u for u in unique if np.array_equal(u["values"], value)), None)
        if match is None:
            unique.append({"reference_id": alias, "aliases": [alias], "values": value})
        else:
            match["aliases"].append(alias)
    matrix = np.column_stack([*values, *[u["values"] for u in unique]])
    if (
        matrix.shape[0] != len(cash)
        or matrix.shape[0] < 2
        or not np.isfinite(matrix).all()
        or not np.isfinite(cash).all()
    ):
        raise Error("Bootstrap requires exact finite matching daily cash/return clocks")
    observed = [portfolio.cash_excess_sharpe(matrix[:, j], cash) for j in range(matrix.shape[1])]
    rows = []
    for block in block_lengths:
        if block <= 0:
            raise Error("Positive block length required")
        rng = np.random.default_rng(np.random.SeedSequence([seed, block]))
        draws = np.empty((replicates, matrix.shape[1]))
        n = len(cash)
        for r in range(replicates):
            starts = rng.integers(0, n, size=int(np.ceil(n / block)))
            ix = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
            excess = matrix[ix] - cash[ix, None]
            sd = np.std(excess, axis=0, ddof=1)
            draws[r] = np.divide(
                np.sqrt(252) * np.mean(excess, axis=0),
                sd,
                out=np.full_like(sd, np.nan),
                where=sd > 0,
            )
        for j, pid in enumerate(ids):
            for k, ref in enumerate(unique):
                diff = draws[:, j] - draws[:, len(ids) + k]
                finite = diff[np.isfinite(diff)]
                low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
                rows.append(
                    {
                        "basket_id": pid,
                        "reference_id": ref["reference_id"],
                        "reference_aliases": json.dumps(ref["aliases"]),
                        "block_length": block,
                        "replicates": replicates,
                        "finite_replicates": len(finite),
                        "seed": seed,
                        "observed_sharpe_difference": observed[j] - observed[len(ids) + k],
                        "ci_lower": low,
                        "ci_upper": high,
                        "confidence_level": 0.95,
                    }
                )
    return pd.DataFrame(rows)


def _evaluate_stage(sidecar, stage):
    from trend_following import research_multi_scale_engine as engine

    sidecar = safe_path(sidecar)
    if stage not in PERIODS:
        raise Error("No historical OOS route")
    with custody._attempt(sidecar, stage) as attempt:
        frozen = verify_study(sidecar)
        lock = verify_locked_selections(sidecar) if stage == "validation" else None
        if (sidecar / stage).exists():
            raise Error("Stage already sealed; no overwrite or repeat")
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
        if cache.gate_override is not None:
            raise Error("Production performance must use actual finalized daily gates")
        candidates = [
            engine.GatedCandidate.from_dict(r["parameters"]) for r in frozen["gated_candidates"]
        ]
        if [c.candidate_id for c in candidates] != [
            r["candidate_id"] for r in frozen["gated_candidates"]
        ]:
            raise Error("Frozen gated candidate identity changed")
        output = attempt / stage
        output.mkdir()
        cutoff = pd.Timestamp(PERIODS[stage]["end"], tz="America/New_York") + pd.Timedelta(hours=17)
        gates = cache.gates.loc[cache.available_at <= cutoff].copy()
        gates.columns = [f"SMA{n}_gate_state" for n in gates.columns]
        gates.insert(0, "available_at", cache.available_at[cache.available_at <= cutoff])
        gates.rename_axis("session").reset_index().to_csv(output / "gate_history.csv", index=False)
        rows, controls_rows, annual, parity = [], [], [], []
        baseline = None
        for slip in SLIPPAGES:
            kwargs = dict(
                **PERIODS[stage],
                cash_observations=cash,
                commission_bps=1.0,
                slippage_bps=slip,
                initial_cash=1000.0,
                record=True,
            )
            parity.append(_parity_preflight(cache, candidates, kwargs, frozen, stage))
            parts = engine.run(cache, candidates, **kwargs)
            replay = portfolio.replay_recorded(
                cache, parts, **PERIODS[stage], cash_observations=cash, initial_cash=1000.0
            )
            preserved._save_sleeves(output / f"constituent_s{slip:g}_daily.npz", replay)
            pd.DataFrame(parts[0]).to_csv(
                output / f"constituent_metrics_s{slip:g}.csv", index=False
            )
            pd.DataFrame(parts[6]).to_csv(output / f"records_s{slip:g}.csv", index=False)
            pd.DataFrame(parts[7]).to_csv(output / f"diagnostics_s{slip:g}.csv", index=False)
            quality._post_gap_table(parts[8]["rows"] if parts[8] else []).to_csv(
                output / f"post_gap_s{slip:g}.csv", index=False
            )
            results = []
            for b in baskets():
                result = engine.aggregate_basket(
                    replay, b["basket_id"], weights=basket_weights(frozen, b["basket_id"])
                )
                members = [
                    i
                    for i, r in enumerate(frozen["gated_candidates"])
                    if r["basket_id"] == b["basket_id"]
                ]
                row = {
                    **result.metrics,
                    **b,
                    "portfolio_id": b["basket_id"],
                    "scenario_slippage_bps": slip,
                    "stage": stage,
                }
                for field in DIAGNOSTICS:
                    row[field] = sum(parts[0][i].get(field, 0) for i in members)
                rows.append(row)
                table = result.annual_returns.copy()
                table["portfolio_id"], table["scenario_slippage_bps"], table["stage"] = (
                    b["basket_id"],
                    slip,
                    stage,
                )
                annual.append(table)
                results.append(result)
            _save_combined(
                output / f"basket_s{slip:g}_daily.npz",
                [b["basket_id"] for b in baskets()],
                results,
                replay.cash_returns,
            )
            old_rows, control_paths = _read_old_controls(
                frozen, stage, slip, replay.times, replay.cash_returns
            )
            old_rows["control_type"] = "reused_sealed_independent_or_passive"
            old_rows["basket_id"] = old_rows.portfolio_id
            controls_rows.extend(old_rows.to_dict("records"))
            gate_parts = engine.run(cache, engine.gate_only_candidates(), **kwargs)
            gate_paths = portfolio.replay_recorded(
                cache, gate_parts, **PERIODS[stage], cash_observations=cash, initial_cash=1000.0
            )
            preserved._save_sleeves(output / f"gate_only_s{slip:g}_daily.npz", gate_paths)
            pd.DataFrame(gate_parts[6]).to_csv(
                output / f"gate_only_records_s{slip:g}.csv", index=False
            )
            for j, c in enumerate(engine.gate_only_candidates()):
                weights = np.eye(2)[j]
                result = engine.aggregate_basket(gate_paths, c.candidate_id, weights=weights)
                row = {
                    **result.metrics,
                    "portfolio_id": c.candidate_id,
                    "gate_period": c.gate_period,
                    "scenario_slippage_bps": slip,
                    "stage": stage,
                    "control_type": "finalized_daily_SMA_gate_only",
                }
                controls_rows.append(row)
                control_paths[c.candidate_id] = {
                    "equity": result.daily_equity.to_numpy(),
                    "returns": result.daily_returns.to_numpy(),
                    "exposure": result.daily_exposure.to_numpy(),
                }
            control_ids = list(control_paths)
            np.savez_compressed(
                output / f"controls_s{slip:g}_daily.npz",
                portfolio_id=np.array(control_ids, dtype=str),
                timestamp=np.array([t.isoformat() for t in replay.times], dtype=str),
                cash_returns=replay.cash_returns,
                **{
                    name: np.column_stack([control_paths[cid][name] for cid in control_ids])
                    for name in ("equity", "returns", "exposure")
                },
            )
            for pid, path in control_paths.items():
                table = portfolio.annual_return_table(pd.Series(path["equity"], index=replay.times))
                table["portfolio_id"], table["scenario_slippage_bps"], table["stage"] = (
                    pid,
                    slip,
                    stage,
                )
                annual.append(table)
            if slip == 3:
                baseline = {
                    "results": results,
                    "controls": control_paths,
                    "cash": replay.cash_returns.copy(),
                }
            print(
                f"Completed multi-scale {stage}: four baskets, 128 constituents; 1-bp commission + {slip:g}-bps slippage",
                flush=True,
            )
        metrics = pd.DataFrame(rows)
        controls = pd.DataFrame(controls_rows)
        for slip in SLIPPAGES:
            bh = controls.loc[
                controls.portfolio_id.eq("buy_and_hold") & controls.scenario_slippage_bps.eq(slip),
                "cash_excess_sharpe",
            ]
            if len(bh) != 1:
                raise Error("Exactly one matched BH benchmark per cost required")
            for frame in (metrics, controls):
                mask = frame.scenario_slippage_bps.eq(slip)
                frame.loc[mask, "BH_cash_excess_sharpe"] = float(bh.iloc[0])
                frame.loc[mask, "Sharpe_difference_vs_BH"] = frame.loc[
                    mask, "cash_excess_sharpe"
                ] - float(bh.iloc[0])
                frame.loc[mask, "beats_BH_Sharpe"] = np.isfinite(
                    frame.loc[mask, "cash_excess_sharpe"]
                ) & (frame.loc[mask, "Sharpe_difference_vs_BH"] > TOLERANCE)
        metrics.to_csv(output / "metrics.csv", index=False)
        controls.to_csv(output / "controls_metrics.csv", index=False)
        pd.concat(annual, ignore_index=True).to_csv(output / "annual_returns.csv", index=False)
        (output / "preflight.json").write_text(json.dumps(parity, indent=2, allow_nan=False) + "\n")
        if stage == "development":
            select_policy_winners(metrics)
        else:
            result_by_id = {
                b["basket_id"]: r for b, r in zip(baskets(), baseline["results"], strict=True)
            }
            selected = {
                r["basket_id"]: result_by_id[r["basket_id"]].daily_returns.to_numpy()
                for r in lock["selections"]
            }
            old_id = frozen["independent_locked_allocation"]["portfolio_id"]
            refs = {
                "locked_independent": baseline["controls"][old_id]["returns"],
                "ungated_mean_reversion": baseline["controls"]["portfolio_tf_000"]["returns"],
                "buy_and_hold": baseline["controls"]["buy_and_hold"]["returns"],
            }
            paired_bootstrap(selected, refs, baseline["cash"]).to_csv(
                output / "bootstrap.csv", index=False
            )
        review_derived_results(output, frozen, stage)
        verify_study(sidecar)
        finished = custody._utc_now()
        seal = {
            "schema": "qqq_multi_scale_stage_results",
            "stage": stage,
            "freeze_sha256": custody.canonical_hash(frozen),
            "membership_sha256": frozen["membership_sha256"],
            "gated_candidates_sha256": frozen["gated_candidates_sha256"],
            "period": PERIODS[stage],
            "basket_ids": [b["basket_id"] for b in baskets()],
            "basket_count": 4,
            "control_ids": expected_control_ids(frozen, stage),
            "constituent_count": 128,
            "scenarios": list(SLIPPAGES),
            "scenario_count": 16,
            "frozen_at": frozen["frozen_at"],
            "started_at": started,
            "finished_at": finished,
            "hindsight_conditioned_membership": True,
            "historical_oos_authorized": False,
            "OOS_numeric_rows_parsed": 0,
            "always_ON_all128_allcost_parity_verified": True,
            "always_OFF_all128_cash_reconciles": True,
            "independent_A_reused_without_extension": True,
            "files": {p.name: custody.sha256_file(p) for p in sorted(output.iterdir())},
        }
        if lock:
            seal["selection_lock_sha256"], seal["selection_sealed_at"] = (
                custody.canonical_hash(lock),
                lock["sealed_at"],
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


def _required_files(stage):
    files = {
        "metrics.csv",
        "controls_metrics.csv",
        "annual_returns.csv",
        "preflight.json",
        "gate_history.csv",
    }
    for slip in SLIPPAGES:
        files |= {
            f"{prefix}_s{slip:g}{suffix}"
            for prefix, suffix in (
                ("constituent", "_daily.npz"),
                ("constituent_metrics", ".csv"),
                ("records", ".csv"),
                ("diagnostics", ".csv"),
                ("post_gap", ".csv"),
                ("basket", "_daily.npz"),
                ("controls", "_daily.npz"),
                ("gate_only", "_daily.npz"),
                ("gate_only_records", ".csv"),
            )
        }
    if stage == "validation":
        files.add("bootstrap.csv")
    return files


def verify_results(sidecar=DEFAULT_SIDECAR, stage="development"):
    if stage not in PERIODS:
        raise Error("No historical OOS verifier or execution route")
    sidecar = safe_path(sidecar)
    frozen = verify_study(sidecar)
    seal = custody._read_seal(sidecar / stage / "seal.json")
    lock = verify_locked_selections(sidecar) if stage == "validation" else None
    expected = {
        "schema": "qqq_multi_scale_stage_results",
        "stage": stage,
        "freeze_sha256": custody.canonical_hash(frozen),
        "membership_sha256": frozen["membership_sha256"],
        "gated_candidates_sha256": frozen["gated_candidates_sha256"],
        "period": PERIODS[stage],
        "basket_ids": [b["basket_id"] for b in baskets()],
        "basket_count": 4,
        "control_ids": expected_control_ids(frozen, stage),
        "constituent_count": 128,
        "scenarios": list(SLIPPAGES),
        "scenario_count": 16,
        "frozen_at": frozen["frozen_at"],
        "hindsight_conditioned_membership": True,
        "historical_oos_authorized": False,
        "OOS_numeric_rows_parsed": 0,
        "always_ON_all128_allcost_parity_verified": True,
        "always_OFF_all128_cash_reconciles": True,
        "independent_A_reused_without_extension": True,
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
        raise Error("Sealed multi-scale stage contract changed")
    mr_val._chronology(
        frozen["frozen_at"],
        *([lock["sealed_at"]] if lock else []),
        seal["started_at"],
        seal["finished_at"],
    )
    files = _required_files(stage)
    if set(seal.get("files", {})) != files or {
        p.name for p in (sidecar / stage).iterdir()
    } != files | {"seal.json"}:
        raise Error("Missing or extra sealed stage artifacts")
    for name, digest in seal["files"].items():
        if custody.sha256_file(safe_path(sidecar / stage / name)) != digest:
            raise Error("Sealed stage artifact changed")
    return seal


def review_derived_results(directory, frozen, stage):
    """Independent arithmetic check of saved combined wealth before sealing."""
    directory = safe_path(directory)
    if stage not in PERIODS:
        raise Error("Derived review refuses unauthorized stages")
    metrics = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    expected = {(b["basket_id"], s) for b in baskets() for s in SLIPPAGES}
    if (
        len(metrics) != 16
        or metrics.duplicated(["basket_id", "scenario_slippage_bps"]).any()
        or set(zip(metrics.basket_id, metrics.scenario_slippage_bps, strict=True)) != expected
    ):
        raise Error("All sixteen filtered basket/cost outcomes required")
    controls = pd.read_csv(directory / "controls_metrics.csv", float_precision="round_trip")
    common_times, common_cash = None, None
    for slip in SLIPPAGES:
        with (
            np.load(directory / f"constituent_s{slip:g}_daily.npz", allow_pickle=False) as sleeves,
            np.load(directory / f"basket_s{slip:g}_daily.npz", allow_pickle=False) as combined,
            np.load(directory / f"controls_s{slip:g}_daily.npz", allow_pickle=False) as control,
        ):
            if list(sleeves["candidate_id"]) != [
                r["candidate_id"] for r in frozen["gated_candidates"]
            ] or list(combined["portfolio_id"]) != [b["basket_id"] for b in baskets()]:
                raise Error("Saved identities changed frozen filtered universe")
            times = pd.DatetimeIndex(pd.to_datetime(sleeves["timestamp"], utc=True))
            cash = sleeves["cash_returns"]
            ny = times.tz_convert("America/New_York")
            if (
                times.has_duplicates
                or not times.is_monotonic_increasing
                or any((t.hour, t.minute, t.second) != (17, 0, 0) for t in ny)
                or ny[0].strftime("%Y-%m-%d") < PERIODS[stage]["start"]
                or ny[-1].strftime("%Y-%m-%d") > PERIODS[stage]["end"]
            ):
                raise Error("Daily clock exceeds authorized stage or differs from 17:00 NY")
            if common_times is None:
                common_times, common_cash = times, cash.copy()
            for packet in (combined, control):
                if not np.array_equal(
                    packet["timestamp"], sleeves["timestamp"]
                ) or not np.array_equal(packet["cash_returns"], cash):
                    raise Error("Exact clock/cash references required; no forward filling")
            if not common_times.equals(times) or not np.array_equal(common_cash, cash):
                raise Error("Cost scenario clocks/cash reference changed")
            for j, b in enumerate(baskets()):
                weights = basket_weights(frozen, b["basket_id"])
                _assert_close(
                    combined["equity"][:, j],
                    sleeves["equity"] @ weights,
                    "Independent scaled sleeve wealth sum differs",
                )
                wealth = combined["equity"][:, j]
                returns = wealth / np.r_[1000.0, wealth[:-1]] - 1
                _assert_close(
                    combined["returns"][:, j],
                    returns,
                    "Combined returns imply unaccounted rebalancing",
                )
                row = metrics.loc[
                    metrics.basket_id.eq(b["basket_id"]) & metrics.scenario_slippage_bps.eq(slip)
                ].iloc[0]
                event_times = pd.DatetimeIndex(pd.to_datetime(sleeves["event_timestamp"], utc=True))
                seconds = np.diff(event_times.asi8) / 1e9
                years = seconds.sum() / (365 * 86400)
                event_equity = sleeves["event_equity"] @ weights
                exposure = (sleeves["event_invested_notional"] @ weights) / event_equity
                expected_stats = {
                    "cash_excess_sharpe": portfolio.cash_excess_sharpe(returns, cash),
                    "terminal_equity": wealth[-1],
                    "cagr": (wealth[-1] / 1000) ** (1 / years) - 1,
                    "max_drawdown": (
                        np.r_[1000, wealth] / np.maximum.accumulate(np.r_[1000, wealth]) - 1
                    ).min(),
                    "annualized_volatility": np.std(returns, ddof=1) * np.sqrt(252),
                    "cost_drag_return": (sleeves["gross_terminal_equity"] @ weights - wealth[-1])
                    / 1000,
                    "exposure_fraction": exposure[:-1] @ seconds / seconds.sum(),
                }
                for field, value in expected_stats.items():
                    _assert_close(row[field], value, f"Independent {field} differs")
                _assert_close(
                    combined["exposure"][:, j],
                    exposure[event_times.get_indexer(times)],
                    "Daily exposure differs",
                )
            ids = list(control["portfolio_id"])
            if ids != expected_control_ids(frozen, stage):
                raise Error(
                    "Validation controls differ from frozen selections and predefined controls"
                )
            selected = controls.loc[controls.scenario_slippage_bps.eq(slip)]
            if (
                len(set(ids)) != len(ids)
                or len(selected) != len(ids)
                or set(selected.portfolio_id) != set(ids)
            ):
                raise Error("Control identities/metrics mismatch")
            for j, pid in enumerate(ids):
                wealth = control["equity"][:, j]
                returns = wealth / np.r_[1000.0, wealth[:-1]] - 1
                _assert_close(control["returns"][:, j], returns, "Control returns do not reconcile")
                row = selected.loc[selected.portfolio_id.eq(pid)].iloc[0]
                _assert_close(row.terminal_equity, wealth[-1], "Control terminal wealth differs")
                _assert_close(
                    row.cash_excess_sharpe,
                    portfolio.cash_excess_sharpe(returns, cash),
                    "Control Sharpe differs",
                )
    return {
        "status": "passed",
        "filtered_scenarios_checked": 16,
        "exact_clock_cash_and_sleeve_wealth": True,
        "historical_oos_authorized": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="QQQ multi-scale trend/mean-reversion; retrospective development/validation only"
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
            payload = (
                verify_study(args.sidecar)
                if (args.sidecar / "freeze.json").exists()
                else {
                    "status": "configuration_valid_unfrozen",
                    "membership_count": 44,
                    "filtered_rule_count": 128,
                    "filtered_basket_count": 4,
                    "market_evaluation_performed": False,
                    "historical_oos_authorized": False,
                }
            )
        elif args.command == "freeze":
            if args.authorization is None:
                parser.error("freeze requires --authorization")
            payload = freeze_study(
                args.sidecar, authorization=args.authorization, config_path=args.config
            )
        elif args.command == "development":
            payload = evaluate_development(args.sidecar)
        elif args.command == "validation":
            payload = evaluate_validation(args.sidecar)
        else:
            from trend_following.research_multi_scale_report import publish_report

            payload = publish_report(args.sidecar, args.output)
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    except (Error, ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Multi-scale study stopped: {exc}\n")
