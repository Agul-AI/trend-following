#!/usr/bin/env python3
"""Publish sealed CURRENT validation/OOS metrics; never evaluate or read raw prices.

All release/selection seals verify before stage numeric CSVs are opened. Only
metrics and derived portfolio curves are exported; trades, quote data, consent
text, machine paths and other studies are excluded. Historical OOS is retrospective.
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

from trend_following import research_quality_protocol_v5 as protocol  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/qqq_daily_halfhour_v5_gap_flags_v2.yaml"
DEFAULT_STUDY = ROOT / ".cache/qqq_daily_halfhour_v5_alpha_gap_flags_v2"
COSTS = (1, 3, 10, 25)
PRIVATE_COLUMNS = {
    "source_path",
    "input_path",
    "receipt_path",
    "path",
    "raw_price",
    "price",
    "open",
    "high",
    "low",
    "close",
}
DUMMY_FIELDS = {
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


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(clean_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def release_report(config_path: Path, study: Path, stage: str) -> dict:
    """Custody/selection checks ONLY; no stage numeric price/return CSV reads."""
    config = protocol.load_config(config_path)
    registration = protocol._verify_registration(study, config, config_path)
    freeze = protocol._verify_prepared(study, registration)
    development = protocol._verify_stage(study, "development", registration, freeze)
    validation = protocol._verify_stage(study, "validation", registration, freeze)
    _, table = protocol._candidate_table(config)
    count = development.get("finalist_count")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 9:
        raise ValueError("Invalid sealed development finalist count")
    finalists, finalist_payload = protocol._load_candidates(
        study / "stages/development/finalists.json", table, count
    )
    winners, winner_payload = protocol._load_candidates(
        study / "stages/validation/winner.json", table, 1
    )
    winner = winners[0]
    if (
        protocol.canonical_hash(finalist_payload) != development["finalists_sha256"]
        or protocol.canonical_hash(winner_payload) != validation["winner_sha256"]
        or winner.candidate_id != validation["winner_id"]
        or winner.candidate_id not in {candidate.candidate_id for candidate in finalists}
        or winner_payload.get("sealed_before_historical_oos") is not True
    ):
        raise ValueError("Validation winner does not match prereleased frozen selection")
    summary = validation
    if stage == "historical-oos":
        summary = protocol._verify_stage(study, stage, registration, freeze)
        if (
            summary.get("winner_id") != winner.candidate_id
            or summary.get("winner_sha256") != validation["winner_sha256"]
            or summary.get("candidate_count") != 1
            or summary.get("selection_after_evaluation") is not False
        ):
            raise ValueError(
                "Historical report must use ONLY the previously sealed validation winner"
            )
    elif stage != "validation":
        raise ValueError("Only validation or historical-oos reports are supported")
    return {
        "config": config,
        "stage_summary": summary,
        "validation": validation,
        "winner": winner,
        "finalists": finalists,
        "winner_payload_sha256": protocol.canonical_hash(winner_payload),
    }


def rank_validation(metrics: pd.DataFrame, candidates: list, winner_id: str) -> pd.DataFrame:
    """Repeat frozen primary-score/ID ordering, not chained approximate tie groups."""
    table = {candidate.candidate_id: candidate for candidate in candidates}
    protocol._audit_metrics(metrics, candidates)
    winner = protocol._finite_winner(protocol._selection_view(metrics), table, 1e-10)
    if winner.candidate_id != winner_id:
        raise ValueError("Numerical validation scores do not reproduce the sealed winner")
    remaining = protocol._selection_view(metrics)
    finite = np.isfinite(pd.to_numeric(remaining.cash_excess_sharpe, errors="coerce"))
    order = []
    remaining = remaining.loc[finite].copy()
    while not remaining.empty:
        candidate = protocol._finite_winner(remaining, table, 1e-10)
        order.append(candidate.candidate_id)
        remaining = remaining.loc[remaining.candidate_id.ne(candidate.candidate_id)]
    order += sorted(set(table) - set(order))
    public = metrics.set_index("candidate_id").loc[order].reset_index()
    public.insert(0, "rank", range(1, len(public) + 1))
    public.insert(1, "selected_validation_winner", public.candidate_id.eq(winner_id))
    return _public_metrics(public)


def _public_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if PRIVATE_COLUMNS.intersection(frame.columns):
        raise ValueError(
            "Metric publication input unexpectedly contains private paths or raw quotes"
        )
    result = frame.copy()
    benchmark = (
        result.policy.isin(["cash", "buy_and_hold"])
        if "policy" in result
        else pd.Series(False, index=result.index)
    )
    for column in DUMMY_FIELDS.intersection(result.columns):
        result.loc[benchmark, column] = None
    return result


def read_public_daily(path: Path) -> pd.DataFrame:
    """Whitelist portfolio quantities; never copy trades/diagnostics or source OHLC."""
    frame = pd.read_csv(path)
    # Independent controls use explicit names for these same derived quantities.
    # Reject ambiguous duplicate schemas instead of silently choosing either series.
    aliases = {"strategy_return": "return", "valuation_exposure": "exposure"}
    for alias, canonical in aliases.items():
        if alias in frame and canonical in frame:
            raise ValueError("Ambiguous derived daily column aliases")
    frame = frame.rename(columns=aliases)
    time = (
        "timestamp"
        if "timestamp" in frame
        else "session"
        if "session" in frame
        else frame.columns[0]
    )
    allowed = {time, "return", "equity", "cash_return", "exposure"}
    if set(frame.columns) - allowed or "equity" not in frame:
        raise ValueError(
            "Daily publication permits only derived portfolio series, never quote/trade fields"
        )
    frame = frame.rename(columns={time: "timestamp"})
    clock = pd.DatetimeIndex(frame.timestamp)
    if (
        clock.tz is None
        or clock.hasnans
        or clock.has_duplicates
        or not clock.is_monotonic_increasing
    ):
        raise ValueError(
            "Derived daily curve requires unique chronological aware valuation timestamps"
        )
    frame["timestamp"] = clock.tz_convert("UTC")
    if (
        not np.isfinite(pd.to_numeric(frame.equity, errors="coerce")).all()
        or frame.equity.le(0).any()
    ):
        raise ValueError("Derived equity must be finite and positive")
    return frame


def _curves(root: Path, winner_id: str, slippage: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, curves, reference = [], [], None
    for policy in ("winner", "buy_and_hold", "cash"):
        metrics = pd.read_csv(root / policy / "metrics.csv")
        if len(metrics) != 1:
            raise ValueError("Each policy must supply exactly one matching-cost metric record")
        row = metrics.iloc[0].to_dict()
        if (
            float(row.get("commission_bps", -1)) != 1
            or float(row.get("slippage_bps", -1)) != slippage
        ):
            raise ValueError("Policy controls do not use matching commission/slippage")
        if policy == "winner" and row.get("candidate_id") != winner_id:
            raise ValueError("Derived curve is not the sealed validation winner")
        row["policy"] = winner_id if policy == "winner" else policy
        records.append(row)
        daily = read_public_daily(root / policy / "daily.csv")
        if reference is None:
            reference = daily.timestamp
        elif not daily.timestamp.equals(reference):
            raise ValueError(
                "Winner and benchmarks must have exactly matched daily valuation endpoints"
            )
        daily.insert(0, "policy", winner_id if policy == "winner" else policy)
        daily.insert(1, "slippage_bps", slippage)
        curves.append(daily)
    return _public_metrics(pd.DataFrame(records)), pd.concat(curves, ignore_index=True)


def publish(
    config_path: Path, study: Path, stage: str, output: Path, validation_curves: Path | None = None
) -> dict:
    released = release_report(config_path, study, stage)  # MUST precede any numeric CSV reads.
    winner = released["winner"]
    if output.exists():
        raise FileExistsError(
            "Publication directory already exists; preserve completed report artifacts"
        )
    output.mkdir(parents=True)
    summary = {
        "project": "Trend Following Study",
        "stage": stage,
        "selected_candidate_id": winner.candidate_id,
        "selected_candidate": winner.to_dict(),
        "winner_payload_sha256": released["winner_payload_sha256"],
        "costs_include_commission_and_adverse_slippage": True,
        "flags_used_for_selection": False,
        "daily_valuation_hour_ny": 17,
        "raw_quotes_or_trades_exported": False,
        "private_paths_or_consent_text_exported": False,
        "interval": released["config"]["segments"][stage],
        "primary_metric": "daily_cash_excess_sharpe_252_ddof1",
        "selection_tie_tolerance": 1e-10,
    }
    if stage == "validation":
        ranked = rank_validation(
            pd.read_csv(study / "stages/validation/metrics.csv"),
            released["finalists"],
            winner.candidate_id,
        )
        ranked.to_csv(output / "candidate_ranking.csv", index=False)
        summary.update(
            evaluated_candidates=len(ranked),
            finite_primary_scores=int(np.isfinite(ranked.cash_excess_sharpe).sum()),
            historical_oos_results_included=False,
            curve_source="not_supplied_metrics_only",
        )
        if validation_curves is not None:
            metrics, curves = _curves(validation_curves, winner.candidate_id, 3)
            selected = ranked.loc[ranked.selected_validation_winner].iloc[0]
            replay = metrics.loc[metrics.policy.eq(winner.candidate_id)].iloc[0]
            for column in ("cash_excess_sharpe", "cagr", "max_drawdown", "terminal_equity"):
                if not np.isclose(
                    float(selected[column]), float(replay[column]), rtol=1e-10, atol=1e-10
                ):
                    raise ValueError(
                        "Precomputed validation winner replay does not match sealed grid metrics"
                    )
            metrics.to_csv(output / "winner_benchmark_comparison.csv", index=False)
            curves.to_csv(output / "daily_portfolio_curves.csv", index=False)
            summary["curve_source"] = "precomputed_same_inputs_root_audit_no_helper_evaluation"
    else:
        scenario = pd.read_csv(study / "stages/historical-oos/cost_scenarios.csv")
        rows, curves = [], []
        for slip in COSTS:
            metrics, daily = _curves(
                study / f"stages/historical-oos/slippage_{slip}bps", winner.candidate_id, slip
            )
            rows.append(metrics)
            curves.append(daily)
        metrics = pd.concat(rows, ignore_index=True)
        if len(scenario) != 12 or not set(scenario.slippage_bps) == set(COSTS):
            raise ValueError(
                "Historical stage must contain the frozen four cost scenarios and three policies"
            )
        if set(metrics.policy) != {winner.candidate_id, "buy_and_hold", "cash"}:
            raise ValueError("Historical sensitivity cannot select a different candidate")
        metrics.to_csv(output / "cost_scenario_comparison.csv", index=False)
        pd.concat(curves, ignore_index=True).to_csv(
            output / "daily_portfolio_curves.csv", index=False
        )
        summary.update(
            evaluated_candidates=1,
            slippage_bps_scenarios=list(COSTS),
            same_winner_all_costs=True,
            historical_oos_interpretation="retrospective previously examined history, not untouched OOS or live evidence",
            post_evaluation_reselection=False,
        )
    _write_json(output / "summary.json", summary)
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    _write_json(output / "artifact_hashes.json", hashes)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["validation", "historical-oos"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-curves", type=Path)
    args = parser.parse_args(argv)
    result = publish(args.config, args.study, args.stage, args.output, args.validation_curves)
    print(json.dumps(clean_json(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
