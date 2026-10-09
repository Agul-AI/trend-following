#!/usr/bin/env python3
"""Publish the complete separately sealed Trend Following Study shortlist.

No performance is evaluated, no member is replaced and no later-stage winner is
chosen. Every extension seal is checked before any later numerical CSV is read.
Only explicitly whitelisted metrics and optional derived daily curves leave the
private study; prices, trades, consent text and machine paths are never exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trend_following import research_shortlist as protocol  # noqa: E402

DEFAULT_DEVELOPMENT_REPORT = ROOT / "reports/development/20261008"
DEFAULT_OUTPUT = ROOT / "reports/shortlist/20261008"
COSTS = (1, 3, 10, 25)
BENCHMARKS = ("buy_and_hold", "cash")
INDICATOR_FIELDS = {
    "entry_family",
    "exit_family",
    "family_cell",
    "entry_rule",
    "exit_rule",
    "entry_confirmation_hours",
    "exit_confirmation_hours",
    "entry_required_checks",
    "exit_required_checks",
}
METRIC_COLUMNS = {
    "candidate_id",
    *INDICATOR_FIELDS,
    "cash_excess_sharpe",
    "cash_excess_Sharpe",
    "cagr",
    "max_drawdown",
    "annualized_volatility",
    "exposure_fraction",
    "total_turnover",
    "annualized_turnover",
    "trade_count",
    "order_count",
    "average_holding_hours",
    "total_commission",
    "total_slippage",
    "cost_drag_return",
    "terminal_equity",
    "gross_terminal_equity",
    "cash_terminal_equity",
    "total_return",
    "scored_sessions",
    "start_session",
    "end_session",
    "sampling_minutes",
    "commission_bps",
    "slippage_bps",
    "quality_profile",
    "daily_valuation_hour_ny",
    "daily_quote_kind",
    "unknown_missing_slots",
    "halt_affected_slots",
    "quality_classification_complete",
    "tradable_price_coverage_complete",
    "source_row_presence_complete",
    "raw_intraday_price_coverage_complete",
    "all_slots_signal_eligible",
    "cancelled_blackout_order_count",
    "cancelled_terminal_order_count",
    "post_gap_confirmed_buy_count",
    "post_gap_confirmed_sell_count",
    "post_gap_confirmed_signal_count",
    "has_post_gap_confirmed_signal",
    "policy",
    "return_type",
    "family_selection_rank",
    "development_shortlist_order",
}
CURVE_COLUMNS = {
    "candidate_id",
    "timestamp",
    "return",
    "equity",
    "cash_return",
    "exposure",
    "commission_bps",
    "slippage_bps",
}
SUMMARY_METRICS = (
    "cagr",
    "cash_excess_sharpe",
    "max_drawdown",
    "annualized_volatility",
    "exposure_fraction",
    "annualized_turnover",
    "trade_count",
    "average_holding_hours",
    "cost_drag_return",
)


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(clean_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def artifact_hashes(output: Path) -> dict:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }


def release_report(parent_study: Path, config_path: Path, study: Path) -> dict:
    """Integrity/retention gates, not a numerical results loader."""
    released = protocol.verify_shortlist(
        parent_study=parent_study, config_path=config_path, study_dir=study
    )
    registration = released["registration"]
    ids = [candidate.candidate_id for candidate in released["candidates"]]
    selection = registration["selection"]
    if (
        len(ids) != len(set(ids))
        or len(ids) != selection["candidate_count"]
        or len(ids) != 9 * selection["per_family"]
        or selection["all_selected_candidates_retained_in_both_later_stages"] is not True
        or selection["later_stage_selection"] is not False
        or selection["gap_flags_affect_selection"] is not False
    ):
        raise ValueError("Publication requires the complete fixed development shortlist")
    stages = {}
    for name, costs in (("validation", (3,)), ("historical-oos", COSTS)):
        seal = protocol.verify_stage(study, name, registration)
        if (
            seal["candidate_ids"] != ids
            or seal["candidate_count"] != len(ids)
            or seal["selection_after_evaluation"] is not False
            or tuple(seal["slippage_bps_scenarios"]) != costs
            or seal["commission_bps"] != 1.0
        ):
            raise ValueError("Later stages must retain every sealed member at the fixed costs")
        stages[name] = seal
    return {**released, "stage_seals": stages}


def public_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.duplicated().any() or set(frame.columns) - METRIC_COLUMNS:
        raise ValueError(
            "Metric publication permits only whitelisted derived fields, never paths/quotes/trades"
        )
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]):
        strings = result[column].dropna().astype(str)
        if strings.str.match(r"^(?:/|~/|file:|[A-Za-z]:[\\/])").any():
            raise ValueError("Metric text unexpectedly contains a private machine path")
    benchmark = result.candidate_id.isin(BENCHMARKS)
    for column in INDICATOR_FIELDS.intersection(result.columns):
        result.loc[benchmark, column] = None
    return result


def _metric_contract(frame: pd.DataFrame, ids: list[str], costs: tuple[int, ...]) -> pd.DataFrame:
    required = {
        "candidate_id",
        "commission_bps",
        "slippage_bps",
        "cagr",
        "cash_excess_sharpe",
        "max_drawdown",
        "terminal_equity",
        "scored_sessions",
    }
    if not required.issubset(frame):
        raise ValueError("Metric comparison lacks required derived quantities")
    public = public_metrics(frame)
    expected = {(identity, float(slip)) for slip in costs for identity in ids}
    keys = list(zip(public.candidate_id, public.slippage_bps, strict=True))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(
            "Metric comparison must include every sealed member exactly once at each cost"
        )
    if not pd.to_numeric(public.commission_bps, errors="coerce").eq(1.0).all():
        raise ValueError("Every policy must use the same one-bp commission")
    for column in ("cagr", "max_drawdown", "terminal_equity", "scored_sessions"):
        numeric = pd.to_numeric(public[column], errors="coerce")
        if not np.isfinite(numeric).all():
            raise ValueError(f"Metric {column} must be finite")
        public[column] = numeric
    if not public.terminal_equity.gt(0).all():
        raise ValueError("Terminal portfolio equity must be positive")
    if (
        public.scored_sessions.le(0).any()
        or not public.scored_sessions.eq(public.scored_sessions.astype(int)).all()
    ):
        raise ValueError("Scored sessions must be positive integers")
    # Undefined Sharpe is retained, never converted to zero or used to drop rows.
    scores = pd.to_numeric(public.cash_excess_sharpe, errors="coerce")
    public["cash_excess_sharpe"] = scores.where(np.isfinite(scores), np.nan)
    if "cash_excess_Sharpe" in public:
        alias = pd.to_numeric(public.cash_excess_Sharpe, errors="coerce")
        public["cash_excess_Sharpe"] = alias.where(np.isfinite(alias), np.nan)
    order = {
        (identity, float(slip)): rank
        for rank, (slip, identity) in enumerate(
            (slip, identity) for slip in costs for identity in ids
        )
    }
    return (
        public.assign(_order=[order[key] for key in keys])
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def development_controls(report: Path) -> tuple[pd.DataFrame, str]:
    source = report / "comparison.csv"
    manifest = json.loads((report / "artifact_hashes.json").read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if manifest.get(source.name) != digest:
        raise ValueError("Current development benchmark publication hash changed")
    frame = pd.read_csv(source, float_precision="round_trip")
    aliases = {"benchmark_buy_and_hold": "buy_and_hold", "benchmark_cash": "cash"}
    controls = frame.loc[frame.candidate_id.isin(aliases)].copy()
    controls["candidate_id"] = controls.candidate_id.map(aliases)
    controls["policy"] = controls.candidate_id
    return _metric_contract(controls, list(BENCHMARKS), (3,)), digest


def _descriptive_flags(development: pd.DataFrame) -> dict:
    if not {"has_post_gap_confirmed_signal", "post_gap_confirmed_signal_count"}.issubset(
        development
    ):
        raise ValueError("Development descriptive gap columns are required")
    flags = development.has_post_gap_confirmed_signal
    if not flags.isin([True, False]).all():
        raise ValueError("Development descriptive flags must be boolean")
    counts = pd.to_numeric(development.post_gap_confirmed_signal_count, errors="coerce")
    if (
        not np.isfinite(counts).all()
        or counts.lt(0).any()
        or not counts.eq(counts.astype(int)).all()
    ):
        raise ValueError("Development gap-event counts must be finite nonnegative integers")
    if not flags.astype(bool).eq(counts.gt(0)).all():
        raise ValueError("Development descriptive flags disagree with their event counts")
    return {
        "post_gap_flagged_candidates": int(flags.astype(bool).sum()),
        "post_gap_confirmed_signals": int(counts.sum()),
        "flags_used_for_selection": False,
    }


def scenario_summary(comparison: pd.DataFrame, ids: list[str], slip: int, tolerance=1e-10) -> dict:
    subset = comparison.loc[comparison.slippage_bps.eq(slip)]
    candidates = subset.loc[subset.candidate_id.isin(ids)]
    controls = subset.loc[subset.candidate_id.isin(BENCHMARKS)].set_index("candidate_id")
    statistics = {}
    for column in SUMMARY_METRICS:
        if column not in candidates:
            continue
        values = pd.to_numeric(candidates[column], errors="coerce")
        finite = values.loc[np.isfinite(values)]
        statistics[column] = {
            "finite_count": len(finite),
            "min": finite.min() if len(finite) else None,
            "median": finite.median() if len(finite) else None,
            "max": finite.max() if len(finite) else None,
        }
    versus = {}
    for identity in BENCHMARKS:
        control = controls.loc[identity]
        higher_cagr = candidates.cagr.gt(float(control.cagr) + tolerance)
        shallower_dd = candidates.max_drawdown.gt(float(control.max_drawdown) + tolerance)
        control_score = float(control.cash_excess_sharpe)
        higher_sharpe = candidates.cash_excess_sharpe.gt(control_score + tolerance)
        versus[identity] = {
            "higher_net_cagr": int(higher_cagr.sum()),
            "shallower_daily_drawdown": int(shallower_dd.sum()),
            "higher_cash_excess_sharpe": int(higher_sharpe.sum())
            if np.isfinite(control_score)
            else None,
            "better_all_three": int((higher_cagr & higher_sharpe & shallower_dd).sum())
            if np.isfinite(control_score)
            else None,
            "control_cagr": control.cagr,
            "control_cash_excess_sharpe": control.cash_excess_sharpe,
            "control_max_drawdown": control.max_drawdown,
        }
    finite_score = np.isfinite(pd.to_numeric(candidates.cash_excess_sharpe, errors="coerce"))
    return {
        "slippage_bps": slip,
        "commission_bps": 1,
        "total_one_way_bps": 1 + slip,
        "candidate_count": len(candidates),
        "finite_primary_scores": int(finite_score.sum()),
        "undefined_primary_scores_retained": int((~finite_score).sum()),
        "statistics": statistics,
        "versus": versus,
        "comparison_tolerance": tolerance,
    }


def public_curves(
    frame: pd.DataFrame, ids: list[str], costs: tuple[int, ...], metrics: pd.DataFrame
) -> pd.DataFrame:
    if frame.columns.duplicated().any() or set(frame) != CURVE_COLUMNS:
        raise ValueError(
            "Daily publication permits only derived portfolio fields, never quotes/trades/paths"
        )
    expected = {(identity, float(slip)) for slip in costs for identity in ids}
    if set(zip(frame.candidate_id, frame.slippage_bps, strict=True)) != expected:
        raise ValueError("Derived daily curves must retain every sealed member and control")
    if not frame.commission_bps.eq(1.0).all():
        raise ValueError("Derived curve commissions do not match the fixed accounting")
    values = frame[["return", "equity", "cash_return", "exposure"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(values).all().all() or values.equity.le(0).any():
        raise ValueError("Derived portfolio quantities must be finite with positive equity")
    reference = None
    pieces = []
    records = metrics.set_index(["candidate_id", "slippage_bps"])
    for slip in costs:
        for identity in ids:
            group = frame.loc[frame.candidate_id.eq(identity) & frame.slippage_bps.eq(slip)].copy()
            clock = pd.DatetimeIndex(group.timestamp)
            if (
                clock.tz is None
                or clock.hasnans
                or clock.has_duplicates
                or not clock.is_monotonic_increasing
            ):
                raise ValueError(
                    "Derived daily curve timestamps must be aware, unique and chronological"
                )
            clock = clock.tz_convert("UTC")
            if reference is None:
                reference = clock
            elif not reference.equals(clock):
                raise ValueError(
                    "All candidates, controls and costs must share the same daily valuation clock"
                )
            row = records.loc[(identity, float(slip))]
            if len(group) != int(row.scored_sessions) or not np.isclose(
                float(group.equity.iloc[-1]), float(row.terminal_equity), rtol=1e-10, atol=1e-10
            ):
                raise ValueError("Derived daily endpoints disagree with matching sealed metrics")
            group["timestamp"] = clock
            pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def baseline_wide(
    development: pd.DataFrame, validation: pd.DataFrame, historical: pd.DataFrame
) -> pd.DataFrame:
    identity = [
        "candidate_id",
        "development_shortlist_order",
        "family_selection_rank",
        *sorted(INDICATOR_FIELDS.intersection(development.columns)),
    ]
    result = development[identity].copy()
    for label, frame in (
        ("development", development),
        ("validation", validation),
        ("historical_oos", historical),
    ):
        selected = frame.loc[
            frame.candidate_id.isin(result.candidate_id) & frame.slippage_bps.eq(3)
        ]
        columns = [column for column in SUMMARY_METRICS if column in selected]
        right = selected[["candidate_id", *columns]].rename(
            columns={column: f"{label}_{column}" for column in columns}
        )
        result = result.merge(
            right, on="candidate_id", how="left", validate="one_to_one", sort=False
        )
    if (
        len(result) != len(development)
        or result.candidate_id.tolist() != development.candidate_id.tolist()
    ):
        raise ValueError("Cross-stage baseline report cannot drop or reorder sealed members")
    return result


def publish(
    *,
    parent_study: Path,
    config_path: Path,
    study: Path,
    output: Path,
    development_report: Path = DEFAULT_DEVELOPMENT_REPORT,
    include_daily: bool = False,
) -> dict:
    released = release_report(parent_study, config_path, study)  # Before ANY later numeric read.
    if output.exists():
        raise FileExistsError("Publication directory already exists; preserve completed artifacts")
    ids = [candidate.candidate_id for candidate in released["candidates"]]
    registration = released["registration"]
    development_all = pd.read_csv(
        parent_study / "stages/development/metrics.csv", float_precision="round_trip"
    )
    selected = development_all.set_index("candidate_id").loc[ids].reset_index()
    development = _metric_contract(selected, ids, (3,))
    ranks, cell_counts = [], {}
    for candidate in released["candidates"]:
        cell_counts[candidate.family_cell] = cell_counts.get(candidate.family_cell, 0) + 1
        ranks.append(cell_counts[candidate.family_cell])
    development.insert(0, "development_shortlist_order", range(1, len(ids) + 1))
    development.insert(1, "family_selection_rank", ranks)
    controls, benchmark_sha = development_controls(development_report)
    comparisons = {"development": pd.concat([development, controls], ignore_index=True)}
    curves = {}
    for stage, costs in (("validation", (3,)), ("historical-oos", COSTS)):
        root = study / "stages" / stage
        members = _metric_contract(
            pd.read_csv(root / "metrics.csv", float_precision="round_trip"), ids, costs
        )
        benchmarks = _metric_contract(
            pd.read_csv(root / "benchmarks.csv", float_precision="round_trip"),
            list(BENCHMARKS),
            costs,
        )
        comparison = _metric_contract(
            pd.concat([members, benchmarks], ignore_index=True), ids + list(BENCHMARKS), costs
        )
        comparisons[stage] = comparison
        if include_daily:
            joined = pd.concat(
                [
                    pd.read_csv(root / name, float_precision="round_trip")
                    for name in ("daily.csv.gz", "benchmark_daily.csv.gz")
                ],
                ignore_index=True,
            )
            curves[stage] = public_curves(joined, ids + list(BENCHMARKS), costs, comparison)
    wide = baseline_wide(development, comparisons["validation"], comparisons["historical-oos"])
    summary = {
        "project": "Trend Following Study",
        "scope": "fixed_development_shortlist_followup",
        "candidate_count": len(ids),
        "candidate_ids": ids,
        "per_family": registration["selection"]["per_family"],
        "family_cell_counts": cell_counts,
        "selection": registration["selection"],
        "all_selected_candidates_retained_in_both_later_stages": True,
        "later_stage_winner_selected": False,
        "post_evaluation_reselection": False,
        "historical_label": registration["historical_label"],
        "prior_validation_and_historical_oos_already_examined": True,
        "interpretation": "Exploratory retrospective extension after previous validation/historical OOS exposure; not untouched OOS, live evidence or a newly selected winner.",
        "daily_valuation_hour_ny": 17,
        "primary_metric": "daily_cash_excess_sharpe_252_ddof1",
        "costs_include_commission_and_adverse_slippage": True,
        "taxes_financing_and_separate_fund_expense_charges": False,
        "raw_quote_values_or_trades_exported": False,
        "private_paths_or_consent_text_exported": False,
        "daily_curve_stages": list(curves),
        "shortlist_registration_sha256": protocol.parent.canonical_hash(registration),
        "candidate_set_sha256": protocol.parent.canonical_hash(released["candidate_set"]),
        "parent_development_metrics_sha256": registration["development_metrics_sha256"],
        "development_benchmark_comparison_sha256": benchmark_sha,
        "descriptive_development_gap_flags": _descriptive_flags(development),
        "segments": released["config"]["segments"],
        "comparisons": {
            stage: [scenario_summary(frame, ids, slip) for slip in costs]
            for stage, frame, costs in (
                ("development", comparisons["development"], (3,)),
                ("validation", comparisons["validation"], (3,)),
                ("historical-oos", comparisons["historical-oos"], COSTS),
            )
        },
    }
    # Build and validate everything before creating a public output directory.
    output.mkdir(parents=True)
    development.to_csv(output / "development_shortlist.csv", index=False)
    comparisons["validation"].to_csv(output / "validation_comparison.csv", index=False)
    comparisons["historical-oos"].to_csv(output / "historical_cost_comparison.csv", index=False)
    wide.to_csv(output / "validation_and_historical_baseline.csv", index=False)
    for stage, frame in curves.items():
        frame.to_csv(
            output / f"{stage.replace('-', '_')}_daily_curves.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
    write_json(output / "summary.json", summary)
    write_json(output / "artifact_hashes.json", artifact_hashes(output))
    return clean_json(summary)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-study", type=Path, default=protocol.DEFAULT_PARENT)
    parser.add_argument("--config", type=Path, default=protocol.DEFAULT_CONFIG)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--development-report", type=Path, default=DEFAULT_DEVELOPMENT_REPORT)
    parser.add_argument("--include-daily", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            publish(
                parent_study=args.parent_study,
                config_path=args.config,
                study=args.study_dir,
                output=args.output,
                development_report=args.development_report,
                include_daily=args.include_daily,
            ),
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
