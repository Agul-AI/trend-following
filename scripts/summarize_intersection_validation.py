#!/usr/bin/env python3
"""Publish the sealed development intersection and its complete validation follow-up.

No market evaluations or raw packets are opened here. Both the selection/source
freeze and fresh validation result seal must pass before any numeric CSV reads.
All selected pairs are retained at 2, 3 and 4 hours, including undefined scores.
The output is diagnostic validation, not a winner selection or an OOS release.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import summarize_constant_cash as shared  # noqa: E402

from trend_following import research_constant_cash as cash_model  # noqa: E402
from trend_following import research_daily_hourly_v5 as core  # noqa: E402
from trend_following import research_intersection_validation as study  # noqa: E402
from trend_following import research_quality_protocol_v5 as protocol  # noqa: E402

DEFAULT_OUTPUT = ROOT / "reports/intersection_validation/20261008"
WAITS = (2.0, 3.0, 4.0)
SESSIONS = 1006
START = pd.Timestamp("2012-01-03 09:30", tz="America/New_York").tz_convert("UTC")
END = pd.Timestamp("2015-12-31 17:00", tz="America/New_York").tz_convert("UTC")
YEARS = (END - START).total_seconds() / (365 * 86400)
TOLERANCE = 1e-10
PAIR_COUNT = 159
CANDIDATE_COUNT = 3 * PAIR_COUNT


def pair_id(candidate):
    return f"{candidate.entry_rule.rule_id}|{candidate.exit_rule.rule_id}"


def frozen_candidates(payload):
    """Reconstruct canonical identities without accepting validation-derived selection."""
    records = payload.get("candidates", [])
    candidates = [
        core.Candidate.from_dict(
            {key: value for key, value in record.items() if key != "candidate_id"}
        )
        for record in records
    ]
    table = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidates) != CANDIDATE_COUNT or len(table) != CANDIDATE_COUNT:
        raise ValueError("Require exactly 477 unique frozen candidates from 159 pairs")
    for candidate, record in zip(candidates, records, strict=True):
        if record.get("candidate_id") != candidate.candidate_id:
            raise ValueError("Frozen candidate identity differs from canonical engine parameters")
        if (
            candidate.entry_confirmation_hours not in WAITS
            or candidate.exit_confirmation_hours != candidate.entry_confirmation_hours
        ):
            raise ValueError(
                "All entry/exit waits must be equal and restricted to 2, 3 and 4 hours"
            )
    sets = [
        {
            pair_id(candidate)
            for candidate in candidates
            if candidate.entry_confirmation_hours == hours
        }
        for hours in WAITS
    ]
    if not all(len(pairs) == PAIR_COUNT and pairs == sets[0] for pairs in sets):
        raise ValueError("Exactly the same 159 frozen pairs must appear at all three waits")
    pair_records = payload.get("pairs", [])
    if (
        len(pair_records) != PAIR_COUNT
        or len({record.get("pair_id") for record in pair_records}) != PAIR_COUNT
        or {record.get("pair_id") for record in pair_records} != sets[0]
    ):
        raise ValueError("Frozen pair table and candidate table must agree without attrition")
    canonical_rules = {rule.rule_id for rule in core.indicator_rules()}
    if any(
        candidate.entry_rule.rule_id not in canonical_rules
        or candidate.exit_rule.rule_id not in canonical_rules
        for candidate in candidates
    ):
        raise ValueError("Only the fixed 17 daily-session indicator choices are allowed")
    return table


def verify_release(sidecar):
    """Metadata/source gates precede all result and development numeric readers."""
    payload = study.verify_study(sidecar)
    result = study.verify_results(sidecar)
    table = frozen_candidates(payload)
    if payload.get("cash") != cash_model.cash_policy():
        raise ValueError("Release must bind fixed nominal 3% ACT/365 cash")
    for key, expected in (
        ("wait_hours", list(WAITS)),
        ("commission_bps_per_order", 1),
        ("slippage_bps_per_order", 3),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"Release must bind the common-wait accounting contract: {key}")
    if result.get("freeze_sha256") != protocol.canonical_hash(payload):
        raise ValueError("Validation results must bind the pre-evaluation frozen selection")
    if result.get("stage") != "validation" or result.get("no_winner_selected") is not True:
        raise ValueError(
            "Only complete diagnostic validation without winner selection is publishable"
        )
    selection = payload.get("selection", {})
    required_payload = {
        "stage": "validation",
        "period": {"start": "2012-01-01", "end": "2015-12-31"},
        "configuration_count_per_wait": PAIR_COUNT,
        "configuration_count": CANDIDATE_COUNT,
        "historical_oos_authorized": False,
        "prior_validation_or_OOS_outcomes_used_for_selection": False,
        "no_validation_selection_or_replacement": True,
    }
    required_result = {
        "configuration_count": CANDIDATE_COUNT,
        "configuration_count_per_wait": PAIR_COUNT,
        "waits_evaluated": list(WAITS),
        "no_validation_filtering_or_replacement": True,
        "no_OOS_market_files_or_prior_validation_outcomes_loaded": True,
        "historical_oos_authorized": False,
    }
    if (
        any(payload.get(key) != value for key, value in required_payload.items())
        or any(result.get(key) != value for key, value in required_result.items())
        or selection.get("uses_validation_or_OOS_outcomes") is not False
        or selection.get("wait_hours") != list(WAITS)
        or selection.get("shared_pair_count") != PAIR_COUNT
        or selection.get("full_indicator_pairs_per_wait") != 289
        or selection.get("tolerance") != TOLERANCE
    ):
        raise ValueError(
            "Release must bind development-only intersection and validation-only access"
        )
    for name in payload.get("source_hashes", {}):
        if Path(name).is_absolute() or ".." in Path(name).parts or not name.endswith(".py"):
            raise ValueError("Public source hashes must use safe repository-relative names")
    return payload, result, table


def validate_metrics(frame, table, *, benchmark=False):
    allowed = shared.METRICS | {"portfolio", "pair_id", "wait_hours"}
    required = shared.REQUIRED | ({"portfolio"} if benchmark else set(shared.IDENTITY))
    shared._schema(frame, allowed, required)
    if (
        frame.candidate_id.duplicated().any()
        or not frame.candidate_id.map(lambda x: isinstance(x, str)).all()
    ):
        raise ValueError("Unique string candidate identities required")
    constants = {
        "commission_bps": 1,
        "slippage_bps": 3,
        "sampling_minutes": 30,
        "scored_sessions": SESSIONS,
        "start_session": "2012-01-03",
        "end_session": "2015-12-31",
    }
    for key, expected in constants.items():
        if not frame[key].eq(expected).all():
            raise ValueError(f"Accounting/period contract differs: {key}")
    numeric = [
        "cash_excess_sharpe",
        "cagr",
        "max_drawdown",
        "terminal_equity",
        "cash_terminal_equity",
    ]
    for key in numeric:
        pd.to_numeric(frame[key], errors="raise")
    if "cash_excess_Sharpe" in frame and not np.array_equal(
        frame.cash_excess_sharpe, frame.cash_excess_Sharpe, equal_nan=True
    ):
        raise ValueError("Cash-excess Sharpe aliases disagree")
    for key in ("cagr", "max_drawdown", "terminal_equity", "cash_terminal_equity"):
        if not np.isfinite(frame[key].to_numpy(float)).all():
            raise ValueError("Non-score portfolio metrics must remain finite")
    if frame.terminal_equity.le(0).any() or frame.cash_terminal_equity.le(0).any():
        raise ValueError("Terminal portfolio and cash wealth must be positive")
    if benchmark:
        if (
            len(frame) != 2
            or set(frame.portfolio) != {"cash", "buy_and_hold"}
            or not frame.candidate_id.eq("benchmark_" + frame.portfolio).all()
        ):
            raise ValueError("Exactly cash and cost-matched QQQ buy-and-hold benchmarks required")
    else:
        if len(frame) != CANDIDATE_COUNT or set(frame.candidate_id) != set(table):
            raise ValueError("All 477 frozen validation candidates must remain; no attrition")
        for key, getter in shared.IDENTITY.items():
            expected = frame.candidate_id.map(
                {identity: getter(candidate) for identity, candidate in table.items()}
            )
            if not frame[key].eq(expected).all():
                raise ValueError(f"Candidate metric differs from frozen equal-wait identity: {key}")
        if (
            "pair_id" in frame
            and not frame.pair_id.eq(
                frame.candidate_id.map({k: pair_id(v) for k, v in table.items()})
            ).all()
        ):
            raise ValueError("Pair labels differ from canonical engine identities")
        if "wait_hours" in frame and not frame.wait_hours.eq(frame.entry_confirmation_hours).all():
            raise ValueError("Wait alias differs from shared entry/exit duration")
    result = frame.copy()
    if not benchmark:
        result["pair_id"] = result.candidate_id.map(
            {key: pair_id(candidate) for key, candidate in table.items()}
        )
        result["wait_hours"] = result.entry_confirmation_hours
        result = result.sort_values(["pair_id", "wait_hours"], kind="stable").reset_index(drop=True)
    return result


def validate_development_selection(frame, table, payload):
    """Sparse sealed development selection rows, never invented accounting results."""
    allowed = set(study.DEV_EXPORT_COLUMNS)
    shared._schema(frame, allowed, allowed)
    if (
        len(frame) != CANDIDATE_COUNT
        or frame.candidate_id.duplicated().any()
        or set(frame.candidate_id) != set(table)
    ):
        raise ValueError("All 477 development-selected instances must remain unchanged")
    frozen = pd.DataFrame(payload["development_selection_rows"])
    shared._schema(frozen, allowed, allowed)
    try:
        pd.testing.assert_frame_equal(
            frame.sort_values("candidate_id").reset_index(drop=True)[sorted(allowed)],
            frozen.sort_values("candidate_id").reset_index(drop=True)[sorted(allowed)],
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(
            "Development selection CSV differs from its pre-validation frozen rows"
        ) from exc
    result = frame.copy()
    for key, getter in shared.IDENTITY.items():
        expected = frame.candidate_id.map(
            {identity: getter(candidate) for identity, candidate in table.items()}
        )
        if key in frame and not frame[key].eq(expected).all():
            raise ValueError(
                "Development selection identity differs from canonical equal-wait grid"
            )
        result[key] = expected
    result["pair_id"] = frame.candidate_id.map(
        {identity: pair_id(candidate) for identity, candidate in table.items()}
    )
    result["wait_hours"] = result.entry_confirmation_hours
    return result.sort_values(["pair_id", "wait_hours"], kind="stable").reset_index(drop=True)


def counts_by_wait(metrics, bh_sharpe):
    if not np.isfinite(bh_sharpe):
        raise ValueError("Matched QQQ buy-and-hold Sharpe must be finite")
    rows = []
    for hours in WAITS:
        cohort = metrics.loc[metrics.wait_hours.eq(hours)]
        if len(cohort) != PAIR_COUNT or not cohort.exit_confirmation_hours.eq(hours).all():
            raise ValueError("Each wait must retain exactly 159 shared pairs")
        for key in ("trade_count", "order_count"):
            values = pd.to_numeric(cohort[key], errors="raise").to_numpy(float)
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or not np.equal(values, np.floor(values)).all()
            ):
                raise ValueError("Trade/order counts must be finite nonnegative integers")
        if not cohort.order_count.eq(2 * cohort.trade_count).all():
            raise ValueError("Terminal-reconciled one-way orders must equal twice round trips")
        scores = cohort.cash_excess_sharpe.to_numpy(float)
        finite = np.isfinite(scores)
        delta = scores - bh_sharpe
        beat = int((finite & (delta > TOLERANCE)).sum())
        rows.append(
            {
                "wait_hours": hours,
                "required_consecutive_hits": 1 + int(2 * hours),
                "configuration_count": PAIR_COUNT,
                "finite_Sharpe_count": int(finite.sum()),
                "undefined_Sharpe_count": int((~finite).sum()),
                "beat_BH_Sharpe_count": beat,
                "beat_BH_Sharpe_pct": 100 * beat / PAIR_COUNT,
                "tie_BH_Sharpe_count": int((finite & (np.abs(delta) <= TOLERANCE)).sum()),
                "strict_gt_BH_Sharpe_count": int((finite & (delta > 0)).sum()),
                "below_BH_Sharpe_count": int((finite & (delta < -TOLERANCE)).sum()),
                "BH_cash_excess_Sharpe": bh_sharpe,
                "median_finite_Sharpe": np.median(scores[finite]) if finite.any() else np.nan,
                "median_cagr": cohort.cagr.median(),
                "median_max_drawdown": cohort.max_drawdown.median(),
                "median_completed_round_trips": cohort.trade_count.median(),
                "median_one_way_orders": cohort.order_count.median(),
                "commission_bps": 1,
                "slippage_bps": 3,
                "cash_nominal_annual_rate": 0.03,
            }
        )
    return pd.DataFrame(rows)


def wide_pairs(development, validation, bh_dev, bh_val=None):
    """Canonical pair order, not validation performance ordering or new selection."""
    if not np.isfinite(bh_dev):
        raise ValueError("Development intersection must use finite development BH Sharpe")
    scores = development.cash_excess_sharpe.to_numpy(float)
    if not np.isfinite(scores).all() or not (scores > bh_dev + TOLERANCE).all():
        raise ValueError("Every selected pair must beat development BH at all three waits")
    columns = ["cash_excess_sharpe", "cagr", "max_drawdown", "trade_count", "order_count"]
    columns += [
        name
        for name in ("has_post_gap_confirmed_signal", "post_gap_confirmed_signal_count")
        if name in development and name in validation
    ]
    output = []
    for identity, dev in development.groupby("pair_id", sort=True):
        val = validation.loc[validation.pair_id.eq(identity)]
        if (
            set(dev.wait_hours) != set(WAITS)
            or set(val.wait_hours) != set(WAITS)
            or len(dev) != 3
            or len(val) != 3
        ):
            raise ValueError(
                "Complete three-wait development and validation pair coverage required"
            )
        row = {
            "pair_id": identity,
            "entry_rule": dev.entry_rule.iloc[0],
            "exit_rule": dev.exit_rule.iloc[0],
            "entry_family": dev.entry_family.iloc[0],
            "exit_family": dev.exit_family.iloc[0],
            "development_BH_cash_excess_Sharpe": bh_dev,
            "development_min_three_wait_Sharpe": dev.cash_excess_sharpe.min(),
            "selected_using_development_only": True,
        }
        for stage, frame in (("development", dev), ("validation", val)):
            for record in frame.itertuples(index=False):
                suffix = f"{int(record.wait_hours)}h"
                for key in columns:
                    row[f"{stage}_{key}_{suffix}"] = getattr(record, key)
        if bh_val is not None:
            if not np.isfinite(bh_val):
                raise ValueError("Validation BH Sharpe must be finite for descriptive pair flags")
            wins = []
            row["validation_BH_cash_excess_Sharpe"] = bh_val
            for hours in WAITS:
                score = val.loc[val.wait_hours.eq(hours), "cash_excess_sharpe"].iloc[0]
                beat = bool(np.isfinite(score) and score > bh_val + TOLERANCE)
                row[f"validation_beat_BH_Sharpe_{int(hours)}h"] = beat
                wins.append(beat)
            row["validation_beat_BH_Sharpe_all_three_waits"] = all(wins)
        output.append(row)
    if len(output) != PAIR_COUNT:
        raise ValueError("Pair comparison must contain all 159 frozen identities")
    return pd.DataFrame(output)


def validate_curves(bh, cash, benchmarks, initial_cash):
    """Reconcile small derived benchmark curves on the fresh validation clock."""
    for frame in (bh, cash):
        shared._schema(frame, shared.CURVE, shared.CURVE)
    if any(
        frame.timestamp.map(lambda value: pd.Timestamp(value).tzinfo is None).any()
        for frame in (bh, cash)
    ):
        raise ValueError("Benchmark valuation timestamps must explicitly carry a timezone")
    clocks = [pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)) for frame in (bh, cash)]
    clock = clocks[0]
    local = clock.tz_convert("America/New_York")
    first = pd.Timestamp("2012-01-03 17:00", tz="America/New_York").tz_convert("UTC")
    if (
        not clock.equals(clocks[1])
        or len(clock) != SESSIONS
        or clock.has_duplicates
        or clock.hasnans
        or not clock.is_monotonic_increasing
        or clock[0] != first
        or clock[-1] != END
        or not (local.hour == 17).all()
        or not (local.minute == 0).all()
        or not (local.second == 0).all()
        or not (local.dayofweek < 5).all()
    ):
        raise ValueError("Benchmarks require matched 1006 validation-only 17:00 NY clocks")
    cash_returns = cash["return"].to_numpy(float)
    dt = np.diff(np.r_[START.value, clock.asi8]) / 1e9 / (365 * 86400)
    if (cash_returns < 0.03 * dt - 2e-12).any() or (
        cash_returns > np.expm1(0.03 * dt) + 2e-12
    ).any():
        raise ValueError("Cash curve violates nominal 3% ACT/365 event-compounding bounds")
    statistics = {}
    for policy, frame in (("buy_and_hold", bh), ("cash", cash)):
        equity = frame.equity.to_numpy(float)
        returns = equity / np.r_[initial_cash, equity[:-1]] - 1
        if (
            not np.isfinite(equity).all()
            or (equity <= 0).any()
            or not np.isfinite(frame["return"].to_numpy(float)).all()
            or not np.allclose(frame["return"], returns, rtol=1e-10, atol=2e-12)
            or not np.allclose(frame.cash_return, cash_returns, rtol=1e-10, atol=2e-12)
        ):
            raise ValueError("Benchmark equity/returns must reconcile to the same cash reference")
        excess = returns - cash_returns
        sd = excess.std(ddof=1)
        sharpe = np.sqrt(252) * excess.mean() / sd if sd > 0 else np.nan
        if policy == "cash":
            if not np.allclose(excess, 0, rtol=0, atol=2e-12):
                raise ValueError("Cash must have zero cash-excess returns")
            sharpe = np.nan
        wealth = np.r_[initial_cash, equity]
        actual = {
            "terminal_equity": equity[-1],
            "cagr": (equity[-1] / initial_cash) ** (1 / YEARS) - 1,
            "cash_excess_sharpe": sharpe,
            "max_drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min(),
        }
        metric = benchmarks.loc[benchmarks.portfolio.eq(policy)].iloc[0]
        if any(
            not np.isclose(metric[key], value, rtol=1e-9, atol=1e-10, equal_nan=True)
            for key, value in actual.items()
        ):
            raise ValueError("Benchmark metric differs from independently recomputed derived curve")
        statistics[policy] = actual
    if not 0.03 < statistics["cash"]["cagr"] <= np.expm1(0.03) + 1e-12:
        raise ValueError("Nominal event-compounded cash CAGR is not exactly 3% APY")
    return statistics


def publish(sidecar, output):
    sidecar, output = Path(sidecar).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Never overwrite published validation evidence")
    if output == sidecar or output.is_relative_to(sidecar) or sidecar.is_relative_to(output):
        raise ValueError("Public output must be separate from private frozen input")
    payload, result, table = verify_release(sidecar)
    private = sidecar / "results"
    metrics = validate_metrics(shared._read(private / "candidate_metrics.csv"), table)
    benchmarks = validate_metrics(
        shared._read(private / "benchmark_metrics.csv"), table, benchmark=True
    )
    development = validate_development_selection(
        shared._read(private / "development_selection_metrics.csv"), table, payload
    )
    bh = benchmarks.loc[benchmarks.portfolio.eq("buy_and_hold")].iloc[0]
    cash = benchmarks.loc[benchmarks.portfolio.eq("cash")].iloc[0]
    if not np.isnan(cash.cash_excess_sharpe):
        raise ValueError("Cash-excess Sharpe of cash is undefined, not finite or zero")
    curves = {
        name: shared._read(private / f"{name}_daily.csv") for name in ("buy_and_hold", "cash")
    }
    statistics = validate_curves(
        curves["buy_and_hold"], curves["cash"], benchmarks, payload["initial_cash"]
    )
    if not np.allclose(metrics.cash_terminal_equity, cash.terminal_equity, rtol=1e-11, atol=1e-8):
        raise ValueError("All 477 candidates must share the new 3% cash benchmark")
    counts = counts_by_wait(metrics, bh.cash_excess_sharpe)
    pairs = wide_pairs(
        development, metrics, payload["selection"]["BH_cash_excess_Sharpe"], bh.cash_excess_sharpe
    )
    public_benchmarks = benchmarks.drop(
        columns=["candidate_id", *[c for c in shared.IDENTITY if c in benchmarks]]
    )
    public_benchmarks["portfolio_label"] = public_benchmarks.portfolio.map(
        {"buy_and_hold": "QQQ buy-and-hold", "cash": "Cash"}
    )
    combined = pd.concat(
        [frame.assign(portfolio=name) for name, frame in curves.items()], ignore_index=True
    )
    summary = {
        "project": "Trend Following Study",
        "scope": "development_BH_beating_intersection_with_diagnostic_validation_only",
        "development_period": {"start": "2001-01-01", "end": "2011-12-31"},
        "validation_period": {"start": "2012-01-01", "end": "2015-12-31"},
        "first_validation_session": "2012-01-03",
        "last_validation_session": "2015-12-31",
        "validation_sessions": SESSIONS,
        "selection_definition": "finite development cash-excess Sharpe > development BH Sharpe + 1e-10 at EVERY shared wait 2, 3 and 4 hours",
        "selection_inputs": "new_constant_3pct_cash_development_results_only_not_old_9_or_45_shortlists",
        "selected_pair_count": PAIR_COUNT,
        "instances_per_pair": 3,
        "validation_configurations_evaluated": CANDIDATE_COUNT,
        "all_selected_candidates_retained_including_undefined_scores": True,
        "common_confirmation_hours": list(WAITS),
        "same_entry_and_exit_wait": True,
        "required_consecutive_hits_formula": "1 + 2 * wait_hours",
        "indicator_unit": "daily_sessions",
        "cash_policy": payload["cash"],
        "effective_cash_cagr": statistics["cash"]["cagr"],
        "initial_capital_per_fresh_segment": payload["initial_cash"],
        "commission_bps_per_one_way_order": 1,
        "slippage_bps_per_one_way_order": 3,
        "total_bps_per_one_way_order": 4,
        "taxes_financing_separate_fund_expenses": False,
        "benchmark_statistics": statistics,
        "score_definition": "session_close_cash_excess_return_mean / sample_sd_ddof1 * sqrt(252)",
        "primary_beat_definition": "finite candidate cash-excess Sharpe > BH Sharpe + 1e-10",
        "undefined_scores_retained_in_159_denominator": True,
        "counts_independently_recomputed": True,
        "validation_pairs_beating_BH_Sharpe_at_all_three_waits": int(
            pairs.validation_beat_BH_Sharpe_all_three_waits.sum()
        ),
        "validation_all_three_beat_flag_is_descriptive_not_a_new_selection": True,
        "trade_medians_scope": "all_159_frozen_pairs_per_wait_not_only_validation_BH_beating_members",
        "trade_count_definition": "completed_buy_sell_round_trips_including_terminal_liquidation_not_trades_per_year",
        "winner_selected_after_validation": False,
        "global_wait_selected_after_validation": False,
        "historical_oos_authorized_or_opened_in_this_follow_up": False,
        "prior_validation_history_previously_examined": True,
        "validation_is_untouched_confirmation": False,
        "freeze_payload_sha256": protocol.canonical_hash(payload),
        "result_seal_payload_sha256": protocol.canonical_hash(result),
        "implementation_source_sha256": payload.get("source_hashes", {}),
        "publication_helper_sha256": shared.sha256(__file__),
        "publication_dependency_source_sha256": {
            "scripts/summarize_constant_cash.py": shared.sha256(shared.__file__)
        },
        "candidate_set_sha256": payload.get("candidate_set_sha256"),
        "pairs_sha256": payload.get("pairs_sha256"),
        "candidate_set_sealed_at": payload.get("candidate_set_sealed_at"),
        "validation_market_access_started_at": result.get("market_outcomes_started_at"),
        "derived_input_sha256": {
            name: value
            for name, value in result.get("files", {}).items()
            if name
            in {
                "candidate_metrics.csv",
                "benchmark_metrics.csv",
                "buy_and_hold_daily.csv",
                "cash_daily.csv",
                "development_selection_metrics.csv",
            }
        },
        "limitations": [
            "Exploratory retrospective validation on previously examined history, not genuinely untouched confirmation",
            "Development-based intersection is in-sample selection",
            "Hypothetical cash yield and execution costs",
            "No winner, final global wait or historical OOS selected or opened",
        ],
    }
    # Complete verification and privacy checks before creating even the output folder.
    output.mkdir(parents=True)
    metrics.to_csv(output / "validation_candidate_metrics.csv", index=False)
    development.to_csv(output / "development_selection_metrics.csv", index=False)
    pairs.to_csv(output / "pair_comparison.csv", index=False)
    counts.to_csv(output / "counts_by_wait.csv", index=False)
    public_benchmarks.to_csv(output / "benchmark_metrics.csv", index=False)
    combined.to_csv(output / "benchmark_daily_curves.csv", index=False)
    shared.write_json(output / "summary.json", summary)
    shared.write_json(
        output / "artifact_hashes.json",
        {path.name: shared.sha256(path) for path in sorted(output.iterdir())},
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, default=study.DEFAULT_SIDECAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    summary = publish(args.sidecar, args.output)
    print(json.dumps(shared.clean_json(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
