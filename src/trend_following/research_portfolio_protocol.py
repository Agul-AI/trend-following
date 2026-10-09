"""Sealed QQQ timing-diversification portfolio research; development/validation only.

Constituent membership is hindsight-conditioned: twelve trend candidates were
identified using validation outcomes. No OOS route or claim of untouched data
exists. Existing strategies, price packets and numerical seals remain read-only.
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
from trend_following import research_daily_hourly_v5 as trend
from trend_following import research_mean_reversion as mr
from trend_following import research_mean_reversion_engine as mr_engine
from trend_following import research_mean_reversion_protocol as mr_dev
from trend_following import research_mean_reversion_validation as mr_val
from trend_following import research_quality_engine_v5 as quality
from trend_following import research_quality_protocol_v5 as custody
from trend_following import research_resplit_validation as tf_val
from trend_following import research_validation_prefix as prefix

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/qqq_portfolio_research.yaml"
DEFAULT_SIDECAR = ROOT / ".cache/portfolio_research_20261009"
DEFAULT_REPORT = ROOT / "reports/portfolio_research/20261009"
DEFAULT_TREND_DEVELOPMENT = fixed.DEFAULT_SIDECAR
DEFAULT_TREND_VALIDATION = tf_val.DEFAULT_SIDECAR
DEFAULT_MEAN_REVERSION_DEVELOPMENT = mr_dev.DEFAULT_SIDECAR
DEFAULT_MEAN_REVERSION_VALIDATION = mr_val.DEFAULT_SIDECAR
PERIODS = {
    "development": {"start": "2001-01-01", "end": "2011-12-31"},
    "validation": {"start": "2012-01-01", "end": "2018-12-31"},
}
SLIPPAGES = (1.0, 3.0, 10.0, 25.0)
TOLERANCE = 1e-10
APPROVAL_EXCERPT = "PLEASE IMPLEMENT THIS PLAN:\n# QQQ Trend + Mean-Reversion Portfolio Test"
Error = custody.ProtocolError
safe_path = mr_dev.safe_path
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *tf_val.SOURCE_FILES,
            *mr_val.SOURCE_FILES,
            "src/trend_following/research_portfolio_engine.py",
            "src/trend_following/research_portfolio_protocol.py",
            "src/trend_following/research_portfolio_report.py",
            "scripts/run_portfolio_research.py",
        )
    )
)


def approved_config():
    return {
        "schema": "qqq_strategy_portfolio_research",
        "study_name": "QQQ Strategy Portfolio Study",
        "security": "QQQ",
        "historical_label": "retrospective_hindsight_conditioned_membership",
        "membership": {
            "trend_count": 12,
            "mean_reversion_count": 32,
            "total_count": 44,
            "shared_confirmation_hours": 3.0,
            "required_checks": 7,
            "trend_rule": "2012_2018_validation_3h_finite_Sharpe_advantage_gt_1e_10",
            "mean_reversion_rule": "sealed_32_development_baseline_BH_Sharpe_beaters_not_top20",
        },
        "portfolio": {
            "initial_cash": 1000.0,
            "construction": "fixed_initial_capital_sleeves_no_rebalancing",
            "weights_within_style": "equal",
            "trend_allocations": [x / 10 for x in range(11)],
            "additional_allocation": "equal44",
            "allocation_count": 12,
            "allow_pure_style": True,
            "order_netting": False,
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
            "tie_tolerance": TOLERANCE,
            "tie_order": "closest_to_half_then_stable_portfolio_id",
            "minimum_trade_filter": False,
            "return_drawdown_filter": False,
            "baseline_only": True,
        },
        "validation": {
            "controls": [
                "pure_trend",
                "pure_mean_reversion",
                "50_50",
                "equal44",
                "buy_and_hold",
                "cash",
            ],
            "deduplicate_locked_control": True,
            "no_reselection": True,
        },
        "bootstrap": {
            "replicates": 2000,
            "block_lengths": [5, 20, 60],
            "seed": 20261009,
            "paired": True,
            "confidence_level": 0.95,
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
        raise Error("Portfolio configuration differs from the exact user-approved contract")
    return value


def authorization(path):
    value = json.loads(safe_path(path).read_text())
    expected = {
        "schema": "qqq_strategy_portfolio_authorization",
        "authorized_by": "human_user",
        "explicit_user_approval": True,
        "authorized_stages": ["development", "validation"],
        "approval_excerpt": APPROVAL_EXCERPT,
        "configuration_count": 44,
        "trend_count": 12,
        "mean_reversion_count": 32,
        "shared_confirmation_hours": 3.0,
        "development_period": PERIODS["development"],
        "validation_period": PERIODS["validation"],
        "historical_oos_authorized": False,
        "no_constituent_retuning_or_replacement": True,
    }
    if any(
        custody.canonical_hash(value.get(k)) != custody.canonical_hash(v)
        for k, v in expected.items()
    ):
        raise Error(
            "Explicit matching human development/validation portfolio authorization required"
        )
    return value


def validate_membership(rows):
    if len(rows) != 44 or len({r.get("candidate_id") for r in rows}) != 44:
        raise Error("Exactly 44 unique unchanged constituent definitions required")
    styles = [r.get("style") for r in rows]
    if styles != ["trend"] * 12 + ["mean_reversion"] * 32:
        raise Error("Canonical twelve trend then thirty-two mean-reversion ordering required")
    for style in ("trend", "mean_reversion"):
        subset = [r for r in rows if r["style"] == style]
        if [r["candidate_id"] for r in subset] != sorted(r["candidate_id"] for r in subset):
            raise Error("Constituent IDs must be lexically ordered within style")
        for row in subset:
            cls = trend.Candidate if style == "trend" else mr.Candidate
            if set(row) != {"candidate_id", "style", "parameters"}:
                raise Error("Unexpected constituent definition fields")
            try:
                c = cls.from_dict(row["parameters"])
            except (ValueError, TypeError, KeyError) as exc:
                raise Error("Invalid constituent definition") from exc
            if (
                c.candidate_id != row["candidate_id"]
                or c.entry_required_checks != 7
                or c.exit_required_checks != 7
            ):
                raise Error("Constituent ID/daily parameters/shared three-hour wait changed")
    return rows


def select_trend_members(metrics, canonical_candidates):
    grid = {c.candidate_id: c for c in canonical_candidates}
    if metrics.candidate_id.duplicated().any() or set(metrics.candidate_id) != set(grid):
        raise Error("All preserved validation trend candidates must remain intact")
    for row in metrics.itertuples(index=False):
        c = grid[row.candidate_id]
        if (
            row.entry_confirmation_hours != c.entry_confirmation_hours
            or row.exit_confirmation_hours != c.exit_confirmation_hours
        ):
            raise Error("Preserved trend timing metadata changed")
    selected = metrics.loc[
        metrics.entry_confirmation_hours.eq(3.0)
        & metrics.exit_confirmation_hours.eq(3.0)
        & np.isfinite(metrics.cash_excess_sharpe)
        & (metrics.Sharpe_difference_vs_BH > TOLERANCE)
    ].copy()
    if len(selected) != 12 or not np.allclose(
        selected.cash_excess_sharpe - selected.BH_cash_excess_sharpe,
        selected.Sharpe_difference_vs_BH,
        atol=1e-12,
        rtol=0,
    ):
        raise Error("Exactly the twelve current three-hour validation BH-Sharpe beaters required")
    return [grid[cid] for cid in sorted(selected.candidate_id)]


def select_mean_reversion_members(selection):
    return mr_val.candidates_from_selection(selection)


def _separate(sidecar, preserved, auth=None):
    if sidecar.parent != ROOT / ".cache" or not sidecar.name.startswith("portfolio_research_"):
        raise Error("New top-level portfolio_research_ private sidecar required")
    for source in preserved:
        if sidecar == source or sidecar in source.parents or source in sidecar.parents:
            raise Error("Portfolio tree must not overlap preserved studies/packets")
        if auth is not None and auth.is_relative_to(source):
            raise Error("Authorization must be outside preserved studies")
    if auth is not None and auth.is_relative_to(sidecar):
        raise Error("Authorization must be outside the portfolio tree")


def _bindings(config_path, auth, tf_dev_path, tf_val_path, mr_dev_path, mr_val_path):
    load_config(config_path)
    authorization(auth)
    td, tv, md, mv = (
        fixed.verify_study(tf_dev_path),
        tf_val.verify_study(tf_val_path),
        mr_dev.verify_study(mr_dev_path),
        mr_val.verify_selection(mr_val_path),
    )
    seals = {
        "trend_development": fixed.verify_results(tf_dev_path),
        "trend_validation": tf_val.verify_results(tf_val_path),
        "mean_reversion_development": mr_dev.verify_results(mr_dev_path),
        "mean_reversion_validation": mr_val.verify_results(mr_val_path),
    }
    canonical = [
        trend.Candidate.from_dict(
            {
                k: r[k]
                for k in (
                    "entry_rule",
                    "exit_rule",
                    "entry_confirmation_hours",
                    "exit_confirmation_hours",
                )
            }
        )
        for r in tv["candidates"]
    ]
    trend_metrics = pd.read_csv(
        tf_val_path / "results/candidate_metrics.csv", float_precision="round_trip"
    )
    trend_benchmark = pd.read_csv(
        tf_val_path / "results/benchmark_metrics.csv", float_precision="round_trip"
    )
    trend_bh = trend_benchmark.loc[
        trend_benchmark.portfolio.eq("buy_and_hold"), "cash_excess_sharpe"
    ]
    if len(trend_bh) != 1 or not np.isfinite(trend_bh.iloc[0]):
        raise Error("Exactly one finite preserved trend validation BH score required")
    trend_metrics["BH_cash_excess_sharpe"] = float(trend_bh.iloc[0])
    trend_metrics["Sharpe_difference_vs_BH"] = trend_metrics.cash_excess_sharpe - float(
        trend_bh.iloc[0]
    )
    selected_tf = select_trend_members(trend_metrics, canonical)
    selected_mr = select_mean_reversion_members(mv)
    rows = [
        {"candidate_id": c.candidate_id, "style": style, "parameters": c.to_dict()}
        for style, group in (("trend", selected_tf), ("mean_reversion", selected_mr))
        for c in group
    ]
    validate_membership(rows)
    if (
        td["source_study"] != md["source_study"]
        or td["development_files"] != md["development_files"]
        or td["development_manifest_sha256"] != md["development_manifest_sha256"]
    ):
        raise Error("Styles must share exact development price/calendar/action provenance")
    bound = mr_val._metadata_binding(safe_path(mv["bounded_packet"]))
    prepared = tf_val.verify_prepared(tf_val_path)
    if prepared["manifest_sha256"] != bound["manifest_sha256"]:
        raise Error("Styles must share the exact physically bounded validation packet")
    from trend_following import research_portfolio_engine as engine

    return {
        "schema": "qqq_strategy_portfolio_freeze",
        "config_path": str(config_path),
        "config_sha256": custody.sha256_file(config_path),
        "configuration_sha256": custody.canonical_hash(approved_config()),
        "authorization": {"path": str(auth), "sha256": custody.sha256_file(auth)},
        "source_paths": {
            "trend_development": str(tf_dev_path),
            "trend_validation": str(tf_val_path),
            "mean_reversion_development": str(mr_dev_path),
            "mean_reversion_validation": str(mr_val_path),
        },
        "source_seal_hashes": {k: custody.canonical_hash(v) for k, v in seals.items()},
        "membership": rows,
        "membership_sha256": custody.canonical_hash(rows),
        "allocations": [a.to_dict() for a in engine.allocation_grid()],
        "development_adapter": {
            k: td[k]
            for k in (
                "source_study",
                "config_path",
                "development_manifest_sha256",
                "development_files",
            )
        },
        "validation_packet": str(safe_path(mv["bounded_packet"])),
        "validation_metadata": bound,
        "source_hashes": {name: custody.sha256_file(ROOT / name) for name in SOURCE_FILES},
        "runtime": custody._runtime(),
        "cash": fixed.cash_policy(),
        "periods": PERIODS,
        "initial_cash": 1000.0,
        "commission_bps": 1.0,
        "slippage_bps_scenarios": list(SLIPPAGES),
        "hindsight_conditioned_membership": True,
        "trend_membership_uses_validation_performance": True,
        "mean_reversion_membership_uses_development_only": True,
        "historical_oos_authorized": False,
        "OOS_numeric_rows_parsed": 0,
    }


def freeze_study(
    sidecar=DEFAULT_SIDECAR,
    *,
    authorization,
    config_path=DEFAULT_CONFIG,
    trend_development=DEFAULT_TREND_DEVELOPMENT,
    trend_validation=DEFAULT_TREND_VALIDATION,
    mean_reversion_development=DEFAULT_MEAN_REVERSION_DEVELOPMENT,
    mean_reversion_validation=DEFAULT_MEAN_REVERSION_VALIDATION,
):
    sidecar, auth, config_path, td, tv, md, mv = map(
        safe_path,
        (
            sidecar,
            authorization,
            config_path,
            trend_development,
            trend_validation,
            mean_reversion_development,
            mean_reversion_validation,
        ),
    )
    _separate(sidecar, (td, tv, md, mv, safe_path(fixed.DEFAULT_SOURCE_STUDY)), auth)
    with custody._attempt(sidecar, "freeze"):
        payload = _bindings(config_path, auth, td, tv, md, mv)
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
    sources = old["source_paths"]
    auth, config = safe_path(old["authorization"]["path"]), safe_path(old["config_path"])
    paths = [
        safe_path(sources[k])
        for k in (
            "trend_development",
            "trend_validation",
            "mean_reversion_development",
            "mean_reversion_validation",
        )
    ]
    _separate(sidecar, (*paths, safe_path(fixed.DEFAULT_SOURCE_STUDY)), auth)
    expected = _bindings(config, auth, *paths)
    if {k: v for k, v in old.items() if k != "frozen_at"} != expected:
        raise Error("Portfolio frozen code/data/membership/runtime/authorization changed")
    mr_val._chronology(old["frozen_at"])
    return old


def load_packet(frozen, stage, *, sidecar=DEFAULT_SIDECAR):
    if stage not in PERIODS:
        raise Error("Only development and validation are authorized; 2019+ remains closed")
    verified = verify_study(sidecar)
    if custody.canonical_hash(frozen) != custody.canonical_hash(verified):
        raise Error("Caller packet contract differs from the verified active freeze")
    if stage == "development":
        return fixed.load_development_packet(frozen["development_adapter"])
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


def _candidates(frozen):
    validate_membership(frozen["membership"])
    return (
        [trend.Candidate.from_dict(r["parameters"]) for r in frozen["membership"][:12]],
        [mr.Candidate.from_dict(r["parameters"]) for r in frozen["membership"][12:]],
    )


def _assert_close(actual, expected, message):
    if np.shape(actual) != np.shape(expected) or not np.allclose(
        actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True
    ):
        raise Error(message)


def _compare_saved(parts, candidates, path):
    """Reproduce existing individual paths, never an already-combined portfolio."""
    with np.load(safe_path(path), allow_pickle=False) as saved:
        ids = list(saved["candidate_id"])
        if len(ids) != len(set(ids)) or any(c.candidate_id not in ids for c in candidates):
            raise Error("Preserved constituent daily paths lack exact candidate IDs")
        columns = [ids.index(c.candidate_id) for c in candidates]
        times = pd.DatetimeIndex(pd.to_datetime(saved["timestamp"], utc=True))
        if not times.equals(parts[1]):
            raise Error("Replay clock differs from preserved constituent daily clock")
        for key, index in (("equity", 2), ("returns", 3), ("exposure", 5)):
            _assert_close(
                parts[index], saved[key][:, columns], f"Baseline replay differs from sealed {key}"
            )
        _assert_close(parts[4], saved["cash_returns"], "Replay cash reference changed")


def select_allocation_metrics(metrics):
    from trend_following import research_portfolio_engine as engine

    baseline = metrics.loc[metrics.scenario_slippage_bps.eq(3.0)].copy()
    grid = {a.portfolio_id: a for a in engine.allocation_grid()}
    if (
        len(baseline) != 12
        or set(baseline.portfolio_id) != set(grid)
        or baseline.portfolio_id.duplicated().any()
    ):
        raise Error("Selection requires all twelve baseline development allocations")
    for row in baseline.itertuples(index=False):
        if (
            not hasattr(row, "trend_weight")
            or not hasattr(row, "mean_reversion_weight")
            or row.trend_weight != grid[row.portfolio_id].trend_weight
            or row.mean_reversion_weight != 1 - grid[row.portfolio_id].trend_weight
        ):
            raise Error("Development allocation weight metadata changed")
    finite = baseline.loc[np.isfinite(baseline.cash_excess_sharpe)]
    if finite.empty:
        raise Error("No finite development portfolio Sharpe; stop without selecting")
    best = finite.cash_excess_sharpe.max()
    ids = finite.loc[finite.cash_excess_sharpe.ge(best - TOLERANCE), "portfolio_id"]
    return min(
        (grid[cid] for cid in ids), key=lambda a: (abs(a.trend_weight - 0.5), a.portfolio_id)
    )


def eligible_validation_allocations(lock):
    from trend_following import research_portfolio_engine as engine

    grid = {a.portfolio_id: a for a in engine.allocation_grid()}
    row = lock.get("allocation", lock)
    if row.get("portfolio_id") not in grid or custody.canonical_hash(row) != custody.canonical_hash(
        grid[row["portfolio_id"]].to_dict()
    ):
        raise Error("Locked allocation is not an unchanged predefined grid member")
    ids = [
        row["portfolio_id"],
        "portfolio_tf_100",
        "portfolio_tf_000",
        "portfolio_tf_050",
        "portfolio_equal_44",
    ]
    return [grid[cid] for cid in dict.fromkeys(ids)]


def _save_sleeves(path, sleeves):
    np.savez_compressed(
        path,
        candidate_id=np.array(sleeves.candidate_ids, dtype=str),
        timestamp=np.array([t.isoformat() for t in sleeves.times], dtype=str),
        equity=sleeves.equity,
        returns=sleeves.returns,
        cash_returns=sleeves.cash_returns,
        gross_terminal_equity=sleeves.gross_terminal_equity,
        event_timestamp=np.array([t.isoformat() for t in sleeves.event_times], dtype=str),
        event_equity=sleeves.event_equity,
        event_invested_notional=sleeves.event_invested_notional,
        event_long=sleeves.event_long,
        fill_timestamp=np.array([g.timestamp.isoformat() for g in sleeves.fill_groups], dtype=str),
        fill_phase=np.array([g.phase for g in sleeves.fill_groups], dtype=str),
        fill_reference_price=np.array(
            [g.reference_price for g in sleeves.fill_groups], dtype=float
        ),
        fill_equity_before=np.array(
            [g.equity_before for g in sleeves.fill_groups], dtype=float
        ).reshape(-1, len(sleeves.candidate_ids)),
        segment_start=sleeves.segment_start.isoformat(),
        segment_end=sleeves.segment_end.isoformat(),
        initial_cash=sleeves.initial_cash,
    )


def _save_portfolios(path, results, cash_returns):
    np.savez_compressed(
        path,
        portfolio_id=np.array([r.allocation.portfolio_id for r in results], dtype=str),
        timestamp=np.array([t.isoformat() for t in results[0].daily_equity.index], dtype=str),
        equity=np.column_stack([r.daily_equity.to_numpy() for r in results]),
        returns=np.column_stack([r.daily_returns.to_numpy() for r in results]),
        cash_returns=cash_returns,
        exposure=np.column_stack([r.daily_exposure.to_numpy() for r in results]),
    )


def _evaluate_stage(sidecar, stage):
    from trend_following import research_portfolio_engine as engine

    sidecar = safe_path(sidecar)
    if stage not in PERIODS:
        raise Error("No historical OOS execution route")
    with custody._attempt(sidecar, stage) as attempt:
        frozen = verify_study(sidecar)
        lock = verify_locked_allocation(sidecar) if stage == "validation" else None
        if (sidecar / stage).exists():
            raise Error("Portfolio stage already sealed; no overwrite or repeat evaluation")
        started = custody._utc_now()
        mr_val._chronology(frozen["frozen_at"], *([lock["sealed_at"]] if lock else []), started)
        daily, bars, calendar, coverage, blackouts, cash, config = load_packet(
            frozen, stage, sidecar=sidecar
        )
        kwargs_cache = dict(
            coverage=coverage, blackouts=blackouts, quality_policy=custody._quality_policy(config)
        )
        tf_cache = quality.build_quality_support_cache(daily, bars, calendar, **kwargs_cache)
        mr_cache = mr_engine.build_cache(daily, bars, calendar, **kwargs_cache)
        tf_candidates, mr_candidates = _candidates(frozen)
        allocations = (
            eligible_validation_allocations(lock) if lock else list(engine.allocation_grid())
        )
        output = attempt / stage
        output.mkdir()
        all_rows, bench_rows, annual = [], [], []
        baseline_results = None
        baseline_bh = None
        baseline_cash_returns = None
        for slip in SLIPPAGES:
            kwargs = dict(
                **PERIODS[stage],
                cash_observations=cash,
                commission_bps=1.0,
                slippage_bps=slip,
                initial_cash=1000.0,
                record=True,
            )
            tf_parts = quality._run(tf_cache, tf_candidates, **kwargs)
            mr_parts = mr_engine.run(mr_cache, mr_candidates, **kwargs)
            sources = frozen["source_paths"]
            if slip == 3:
                tf_file = "wait_3p0_daily.npz"
                tf_source = safe_path(sources[f"trend_{stage}"]) / "results" / tf_file
                _compare_saved(tf_parts, tf_candidates, tf_source)
            mr_source = (
                safe_path(sources[f"mean_reversion_{stage}"])
                / "results"
                / f"slippage_{slip:g}_daily.npz"
            )
            _compare_saved(mr_parts, mr_candidates, mr_source)
            replay_kwargs = dict(**PERIODS[stage], cash_observations=cash, initial_cash=1000.0)
            tf_paths = engine.replay_recorded(tf_cache, tf_parts, **replay_kwargs)
            mr_paths = engine.replay_recorded(mr_cache, mr_parts, **replay_kwargs)
            sleeves = engine.merge_sleeves(tf_paths, mr_paths)
            if tuple(r["candidate_id"] for r in frozen["membership"]) != tuple(
                sleeves.candidate_ids
            ):
                raise Error("Replayed sleeves changed frozen constituent ordering")
            _save_sleeves(output / f"constituent_s{slip:g}_daily.npz", sleeves)
            records = pd.DataFrame([*tf_parts[6], *mr_parts[6]], columns=trend.TRADE_COLUMNS)
            records.to_csv(output / f"records_s{slip:g}.csv", index=False)
            pd.DataFrame([*tf_parts[0], *mr_parts[0]]).to_csv(
                output / f"constituent_metrics_s{slip:g}.csv", index=False
            )
            results = [engine.evaluate_portfolio(sleeves, allocation) for allocation in allocations]
            _save_portfolios(
                output / f"allocation_s{slip:g}_daily.npz", results, sleeves.cash_returns
            )
            for result in results:
                all_rows.append(
                    {
                        **result.metrics,
                        **result.allocation.to_dict(),
                        "scenario_slippage_bps": slip,
                        "stage": stage,
                    }
                )
                table = result.annual_returns.copy()
                table["portfolio_id"], table["scenario_slippage_bps"] = (
                    result.allocation.portfolio_id,
                    slip,
                )
                annual.append(table)
            for mode in ("buy_and_hold", "cash"):
                parts = mr_engine.run(mr_cache, mr_candidates[:1], mode=mode, **kwargs)
                if not pd.DatetimeIndex(parts[1]).equals(sleeves.times):
                    raise Error("Passive benchmark clock differs from portfolio sleeves")
                _assert_close(parts[4], sleeves.cash_returns, "Passive cash reference differs")
                benchmark_paths = engine.replay_recorded(mr_cache, parts, **replay_kwargs)
                interval_seconds = np.diff(benchmark_paths.event_times.asi8) / 1e9
                benchmark_exposure = (
                    benchmark_paths.event_invested_notional[:, 0]
                    / benchmark_paths.event_equity[:, 0]
                )
                stats = {
                    **parts[0][0],
                    "binary_long_time_fraction": parts[0][0]["exposure_fraction"],
                    "exposure_fraction": float(
                        benchmark_exposure[:-1] @ interval_seconds / interval_seconds.sum()
                    ),
                    "exposure_definition": "calendar_time_weighted_event_left_QQQ_notional_over_total_wealth",
                    "portfolio_id": mode,
                    "portfolio": mode,
                    "scenario_slippage_bps": slip,
                    "stage": stage,
                }
                for field in list(stats):
                    if field in {
                        "candidate_id",
                        "family",
                        "period",
                        "entry_threshold",
                        "exit_threshold",
                        "confirmation_hours",
                        "entry_confirmation_hours",
                        "exit_confirmation_hours",
                        "entry_required_checks",
                        "exit_required_checks",
                        "indicator_unit",
                    }:
                        del stats[field]
                bench_rows.append(stats)
                pd.DataFrame(
                    {
                        "timestamp": parts[1],
                        "equity": parts[2][:, 0],
                        "return": parts[3][:, 0],
                        "cash_return": parts[4],
                        "exposure": parts[5][:, 0],
                    }
                ).to_csv(output / f"{mode}_s{slip:g}_daily.csv", index=False)
                if slip == 3 and mode == "buy_and_hold":
                    baseline_bh = pd.Series(parts[3][:, 0], index=parts[1])
            if slip == 3:
                baseline_results = results
                baseline_cash_returns = sleeves.cash_returns.copy()
                for kind, table in engine.correlations(sleeves).items():
                    table.rename_axis(
                        "constituent_id" if kind == "constituent" else "style"
                    ).reset_index().to_csv(output / f"correlations_{kind}.csv", index=False)
            print(
                f"Completed portfolio {stage}: {len(results)} predefined portfolios; 1-bp commission + {slip:g}-bps slippage",
                flush=True,
            )
        metrics = pd.DataFrame(all_rows)
        benchmarks = pd.DataFrame(bench_rows)
        for slip in SLIPPAGES:
            bh = float(
                benchmarks.loc[
                    benchmarks.portfolio_id.eq("buy_and_hold")
                    & benchmarks.scenario_slippage_bps.eq(slip),
                    "cash_excess_sharpe",
                ].iloc[0]
            )
            mask = metrics.scenario_slippage_bps.eq(slip)
            metrics.loc[mask, "BH_cash_excess_sharpe"] = bh
            metrics.loc[mask, "Sharpe_difference_vs_BH"] = (
                metrics.loc[mask, "cash_excess_sharpe"] - bh
            )
            metrics.loc[mask, "beats_BH_Sharpe"] = np.isfinite(
                metrics.loc[mask, "cash_excess_sharpe"]
            ) & (metrics.loc[mask, "Sharpe_difference_vs_BH"] > TOLERANCE)
        metrics.to_csv(output / "metrics.csv", index=False)
        benchmarks.to_csv(output / "benchmark_metrics.csv", index=False)
        pd.concat(annual, ignore_index=True).to_csv(output / "annual_returns.csv", index=False)
        if stage == "development":
            selected = engine.select_allocation(baseline_results)
            if selected.to_dict() != select_allocation_metrics(metrics).to_dict():
                raise Error("Independent development allocation selection differs")
        else:
            by_id = {r.allocation.portfolio_id: r for r in baseline_results}
            engine.bootstrap_sharpe_differences(
                by_id[lock["allocation"]["portfolio_id"]].daily_returns,
                baseline_bh,
                by_id["portfolio_tf_050"].daily_returns,
                baseline_cash_returns,
                replicates=2000,
                block_lengths=(5, 20, 60),
                seed=20261009,
            ).to_csv(output / "bootstrap.csv", index=False)
        review_derived_results(output, frozen, stage, allocations)
        verify_study(sidecar)
        finished = custody._utc_now()
        result_seal = {
            "schema": "qqq_strategy_portfolio_stage_results",
            "stage": stage,
            "freeze_sha256": custody.canonical_hash(frozen),
            "membership_sha256": frozen["membership_sha256"],
            "period": PERIODS[stage],
            "portfolio_ids": [a.portfolio_id for a in allocations],
            "allocation_count": len(allocations),
            "scenario_count": len(allocations) * 4,
            "scenarios": list(SLIPPAGES),
            "frozen_at": frozen["frozen_at"],
            "started_at": started,
            "finished_at": finished,
            "hindsight_conditioned_membership": True,
            "historical_oos_authorized": False,
            "OOS_numeric_rows_parsed": 0,
            "baseline_replay_matches_preserved_constituents": True,
            "files": {p.name: custody.sha256_file(p) for p in sorted(output.iterdir())},
        }
        if lock:
            result_seal["allocation_lock_sha256"] = custody.canonical_hash(lock)
            result_seal["allocation_sealed_at"] = lock["sealed_at"]
        custody._seal(output / "seal.json", result_seal)
        output.rename(sidecar / stage)
        if stage == "development":
            seal_locked_allocation(sidecar, selected)
        return result_seal


def evaluate_development(sidecar=DEFAULT_SIDECAR):
    return _evaluate_stage(sidecar, "development")


def evaluate_validation(sidecar=DEFAULT_SIDECAR):
    return _evaluate_stage(sidecar, "validation")


def _required_files(stage):
    files = {
        "metrics.csv",
        "benchmark_metrics.csv",
        "annual_returns.csv",
        "correlations_constituent.csv",
        "correlations_style.csv",
    }
    for slip in SLIPPAGES:
        files |= {
            f"constituent_s{slip:g}_daily.npz",
            f"records_s{slip:g}.csv",
            f"constituent_metrics_s{slip:g}.csv",
            f"allocation_s{slip:g}_daily.npz",
            f"buy_and_hold_s{slip:g}_daily.csv",
            f"cash_s{slip:g}_daily.csv",
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
    from trend_following import research_portfolio_engine as engine

    lock = verify_locked_allocation(sidecar) if stage == "validation" else None
    allocations = eligible_validation_allocations(lock) if lock else engine.allocation_grid()
    expected = {
        "schema": "qqq_strategy_portfolio_stage_results",
        "stage": stage,
        "freeze_sha256": custody.canonical_hash(frozen),
        "membership_sha256": frozen["membership_sha256"],
        "period": PERIODS[stage],
        "portfolio_ids": [a.portfolio_id for a in allocations],
        "allocation_count": len(allocations),
        "scenario_count": len(allocations) * 4,
        "scenarios": list(SLIPPAGES),
        "frozen_at": frozen["frozen_at"],
        "hindsight_conditioned_membership": True,
        "historical_oos_authorized": False,
        "OOS_numeric_rows_parsed": 0,
        "baseline_replay_matches_preserved_constituents": True,
    }
    if lock:
        expected.update(
            allocation_lock_sha256=custody.canonical_hash(lock),
            allocation_sealed_at=lock["sealed_at"],
        )
    if any(
        custody.canonical_hash(seal.get(k)) != custody.canonical_hash(v)
        for k, v in expected.items()
    ):
        raise Error("Sealed portfolio stage contract changed")
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
        raise Error("Missing or extra sealed portfolio stage artifacts")
    for name, digest in seal["files"].items():
        if custody.sha256_file(safe_path(sidecar / stage / name)) != digest:
            raise Error("Sealed portfolio stage artifact changed")
    return seal


def seal_locked_allocation(sidecar=DEFAULT_SIDECAR, allocation=None):
    sidecar = safe_path(sidecar)
    dev = verify_results(sidecar, "development")
    metrics = pd.read_csv(sidecar / "development/metrics.csv", float_precision="round_trip")
    winner = select_allocation_metrics(metrics)
    if allocation is not None and allocation.to_dict() != winner.to_dict():
        raise Error("Only the development baseline winner may be locked")
    payload = {
        "schema": "qqq_strategy_portfolio_allocation_lock",
        "development_result_sha256": custody.canonical_hash(dev),
        "allocation": winner.to_dict(),
        "baseline_slippage_bps": 3.0,
        "commission_bps": 1.0,
        "historical_oos_authorized": False,
        "validation_reselection": False,
    }
    path = sidecar / "locked_allocation.json"
    if path.exists():
        return verify_locked_allocation(sidecar)
    payload["sealed_at"] = custody._utc_now()
    mr_val._chronology(dev["finished_at"], payload["sealed_at"])
    custody._seal(path, payload)
    return payload


def verify_locked_allocation(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    dev = verify_results(sidecar, "development")
    lock = custody._read_seal(sidecar / "locked_allocation.json")
    winner = select_allocation_metrics(
        pd.read_csv(sidecar / "development/metrics.csv", float_precision="round_trip")
    )
    expected = {
        "schema": "qqq_strategy_portfolio_allocation_lock",
        "development_result_sha256": custody.canonical_hash(dev),
        "allocation": winner.to_dict(),
        "baseline_slippage_bps": 3.0,
        "commission_bps": 1.0,
        "historical_oos_authorized": False,
        "validation_reselection": False,
    }
    if {k: v for k, v in lock.items() if k != "sealed_at"} != expected:
        raise Error("Locked allocation differs from the sealed development-only rule")
    mr_val._chronology(dev["finished_at"], lock["sealed_at"])
    return lock


locked_allocation = verify_locked_allocation


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="QQQ Strategy Portfolio Study; retrospective development/validation, no OOS route"
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
                    "configuration_count": 44,
                    "allocation_count": 12,
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
            from trend_following.research_portfolio_report import publish_report

            payload = publish_report(args.sidecar, args.output)
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    except (Error, ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Portfolio study stopped: {exc}\n")


def review_derived_results(directory, frozen, stage, allocations):
    """Independently recompute portfolio wealth/path statistics before sealing.

    This audit is derived arithmetic, not an independent market-price history
    reconstruction. Recorded constituent ledger replay provides that separate
    accounting check before these files exist.
    """
    from trend_following import research_portfolio_engine as engine

    directory = safe_path(directory)
    if stage not in PERIODS:
        raise Error("Derived audit refuses unauthorized stages")
    metrics = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    benchmarks = pd.read_csv(directory / "benchmark_metrics.csv", float_precision="round_trip")
    expected_pairs = {(a.portfolio_id, s) for a in allocations for s in SLIPPAGES}
    if (
        len(metrics) != len(expected_pairs)
        or metrics.duplicated(["portfolio_id", "scenario_slippage_bps"]).any()
        or set(zip(metrics.portfolio_id, metrics.scenario_slippage_bps, strict=True))
        != expected_pairs
    ):
        raise Error("Derived portfolio results must retain every permitted allocation/cost")
    if len(benchmarks) != 8 or set(
        zip(benchmarks.portfolio_id, benchmarks.scenario_slippage_bps, strict=True)
    ) != {(p, s) for p in ("buy_and_hold", "cash") for s in SLIPPAGES}:
        raise Error("Eight unique matched passive benchmark outcomes required")
    common_times, common_cash = None, None
    initial = 1000.0
    for slip in SLIPPAGES:
        with (
            np.load(directory / f"constituent_s{slip:g}_daily.npz", allow_pickle=False) as sleeves,
            np.load(directory / f"allocation_s{slip:g}_daily.npz", allow_pickle=False) as combined,
        ):
            ids = list(sleeves["candidate_id"])
            if ids != [r["candidate_id"] for r in frozen["membership"]]:
                raise Error("Derived constituent IDs changed frozen membership")
            times = pd.DatetimeIndex(pd.to_datetime(sleeves["timestamp"], utc=True))
            cash = sleeves["cash_returns"]
            if common_times is None:
                common_times, common_cash = times, cash.copy()
            if (
                not common_times.equals(times)
                or not np.array_equal(common_cash, cash)
                or not np.array_equal(combined["timestamp"], sleeves["timestamp"])
                or not np.array_equal(combined["cash_returns"], cash)
            ):
                raise Error("Exact common daily clock/cash reference required; no interpolation")
            ny = times.tz_convert("America/New_York")
            if (
                times.has_duplicates
                or not times.is_monotonic_increasing
                or any((x.hour, x.minute, x.second) != (17, 0, 0) for x in ny)
                or ny[0].strftime("%Y-%m-%d") < PERIODS[stage]["start"]
                or ny[-1].strftime("%Y-%m-%d") > PERIODS[stage]["end"]
            ):
                raise Error("Portfolio daily marks exceed authorized stage or differ from 17:00 NY")
            if list(combined["portfolio_id"]) != [a.portfolio_id for a in allocations]:
                raise Error("Derived portfolio IDs differ from permitted sealed allocations")
            for j, allocation in enumerate(allocations):
                weights = allocation.weights()
                if (weights < 0).any() or not np.isclose(weights.sum(), 1, atol=1e-15, rtol=0):
                    raise Error("Nonnegative unit capital weights required")
                wealth = np.sum(sleeves["equity"] * weights[None, :], axis=1)
                _assert_close(
                    combined["equity"][:, j], wealth, "Independent scaled sleeve wealth sum differs"
                )
                # Use the already-reconciled aggregate marks to preserve exact
                # identical-cash identity under floating-point sum order.
                wealth = combined["equity"][:, j]
                returns = wealth / np.r_[initial, wealth[:-1]] - 1
                _assert_close(
                    combined["returns"][:, j],
                    returns,
                    "Portfolio uses implicit rebalanced weighted returns",
                )
                row = metrics.loc[
                    metrics.portfolio_id.eq(allocation.portfolio_id)
                    & metrics.scenario_slippage_bps.eq(slip)
                ].iloc[0]
                years = row.elapsed_calendar_years
                if years <= 0:
                    raise Error("Positive authorized calendar-time denominator required")
                sd = np.std(returns - cash, ddof=1)
                expected = {
                    "cash_excess_sharpe": np.sqrt(252) * np.mean(returns - cash) / sd
                    if sd > 0
                    else np.nan,
                    "terminal_equity": wealth[-1],
                    "cagr": (wealth[-1] / initial) ** (1 / years) - 1,
                    "max_drawdown": (
                        np.r_[initial, wealth] / np.maximum.accumulate(np.r_[initial, wealth]) - 1
                    ).min(),
                    "annualized_volatility": np.std(returns, ddof=1) * np.sqrt(252),
                    "cost_drag_return": (sleeves["gross_terminal_equity"] @ weights - wealth[-1])
                    / initial,
                    "ending_trend_weight": (sleeves["equity"][-1, :12] @ weights[:12]) / wealth[-1],
                    "trend_weight": allocation.trend_weight,
                    "mean_reversion_weight": 1 - allocation.trend_weight,
                }
                for field, value in expected.items():
                    _assert_close(
                        row[field], value, f"Independent {field} differs from portfolio metric"
                    )
                event_times = pd.DatetimeIndex(pd.to_datetime(sleeves["event_timestamp"], utc=True))
                event_wealth = np.sum(sleeves["event_equity"] * weights, axis=1)
                event_exposure = (
                    np.sum(sleeves["event_invested_notional"] * weights, axis=1) / event_wealth
                )
                seconds = (event_times[-1] - event_times[0]).total_seconds()
                _assert_close(
                    row.exposure_fraction,
                    np.sum(event_exposure[:-1] * (np.diff(event_times.asi8) / 1e9)) / seconds,
                    "Independent event-time exposure differs",
                )
                _assert_close(
                    years, seconds / (365 * 86400), "Calendar elapsed-year denominator differs"
                )
                _assert_close(
                    combined["exposure"][:, j],
                    event_exposure[event_times.get_indexer(times)],
                    "Daily exposure marks differ",
                )
                if (
                    row.commission_bps != 1
                    or row.slippage_bps != slip
                    or row.scored_sessions != len(times)
                ):
                    raise Error("Derived portfolio accounting metadata changed")
            for mode in ("buy_and_hold", "cash"):
                frame = pd.read_csv(
                    directory / f"{mode}_s{slip:g}_daily.csv", float_precision="round_trip"
                )
                if not pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)).equals(times):
                    raise Error("Passive benchmark clock does not exactly match sleeves")
                _assert_close(frame.cash_return, cash, "Passive cash reference differs")
                returns = frame.equity.to_numpy() / np.r_[initial, frame.equity.to_numpy()[:-1]] - 1
                _assert_close(frame["return"], returns, "Passive curve returns do not reconcile")
                row = benchmarks.loc[
                    benchmarks.portfolio_id.eq(mode) & benchmarks.scenario_slippage_bps.eq(slip)
                ].iloc[0]
                _assert_close(
                    row.terminal_equity, frame.equity.iloc[-1], "Passive terminal wealth differs"
                )
                _assert_close(
                    row.cash_excess_sharpe,
                    engine.cash_excess_sharpe(returns, cash),
                    "Passive cash-excess Sharpe differs",
                )
    return {
        "status": "passed",
        "portfolio_scenarios_checked": len(metrics),
        "benchmark_scenarios_checked": len(benchmarks),
        "exact_clock_cash_and_scaled_sleeve_wealth": True,
        "historical_oos_authorized": False,
    }
