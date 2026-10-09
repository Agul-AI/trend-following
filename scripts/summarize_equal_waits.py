#!/usr/bin/env python3
"""Describe equal entry/exit confirmation waits using saved development metrics only.

No backtest is run, no later-period file is opened and no new winner is selected.
The unchanged 45 development-selected indicator pairs are compared at every wait,
with all 289 indicator pairs as context. This is in-sample sensitivity, not OOS.
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

DEFAULT_DEVELOPMENT = ROOT / "reports/development/20261008"
DEFAULT_SHORTLIST = ROOT / "reports/development_shortlist_hold/20261008"
DEFAULT_CONFIG = ROOT / "configs/qqq_daily_halfhour_v5_gap_flags_v2.yaml"
WAIT_HOURS = tuple(tick / 2.0 for tick in range(13))
PINNED_INPUT_HASHES = {
    "metrics.csv.gz": "a51d1b252c996ac8a528e912e9530647e6e6fb0fc478fb07678ba103e6b900fe",
    "publication_receipt.json": "309a84c6427f27760bb08cbfd89a43fdc54be4fcde11c65863a2a480f2f2dcb9",
    "development_shortlist.csv": "5b5a1cefa54be4363b7411bd07cb3668002306b90301f319f83f23e3023dcbf2",
    "selection_and_hold.json": "21d3be83986975aaba410099a06d01fe276c8c12f04f6c3d34b801e26b94266e",
}
CONFIG_SHA256 = "2838a701f748e587922e9499aa6115307543b2d78b4cb52c73e393d2119e8d2c"
IDENTITY_FIELDS = {
    "entry_family": lambda c: c.entry_rule.family,
    "exit_family": lambda c: c.exit_rule.family,
    "family_cell": lambda c: c.family_cell,
    "entry_rule": lambda c: c.entry_rule.rule_id,
    "exit_rule": lambda c: c.exit_rule.rule_id,
    "entry_confirmation_hours": lambda c: c.entry_confirmation_hours,
    "exit_confirmation_hours": lambda c: c.exit_confirmation_hours,
    "entry_required_checks": lambda c: c.entry_required_checks,
    "exit_required_checks": lambda c: c.exit_required_checks,
}
METRIC_COLUMNS = {
    "candidate_id",
    *IDENTITY_FIELDS,
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
}
SHORTLIST_EXTRA = {"development_shortlist_order", "family_selection_rank"}
REQUIRED_COLUMNS = {
    "candidate_id",
    *IDENTITY_FIELDS,
    "cash_excess_sharpe",
    "cagr",
    "max_drawdown",
    "commission_bps",
    "slippage_bps",
    "sampling_minutes",
    "scored_sessions",
    "start_session",
    "end_session",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(clean_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def verify_public_inputs(development: Path, shortlist: Path, *, root=ROOT, anchors=None) -> dict:
    """Hash gates precede all numerical CSV readers; reads development artifacts only."""
    anchors = PINNED_INPUT_HASHES if anchors is None else anchors
    for directory, names in (
        (development, ("metrics.csv.gz", "publication_receipt.json")),
        (shortlist, ("development_shortlist.csv", "selection_and_hold.json")),
    ):
        manifest = json.loads((directory / "artifact_hashes.json").read_text())
        for name in names:
            if manifest.get(name) != anchors[name] or sha256(directory / name) != anchors[name]:
                raise ValueError(f"Frozen development artifact changed: {name}")
    receipt = json.loads((development / "publication_receipt.json").read_text())
    hold = json.loads((shortlist / "selection_and_hold.json").read_text())
    source_hashes = receipt["frozen_source_sha256"]
    if source_hashes != hold["source_hashes"] or len(source_hashes) != 8:
        raise ValueError("Development/hold frozen source contracts disagree")
    for name, digest in source_hashes.items():
        # Only repository-relative source files may be used, never external private paths.
        if Path(name).is_absolute() or ".." in Path(name).parts or not name.endswith(".py"):
            raise ValueError("Frozen source manifest contains an unsafe path")
        if sha256(root / name) != digest:
            raise ValueError(f"Frozen implementation source changed: {name}")
    if (
        sha256(root / "src/trend_following/research_shortlist_selection.py")
        != hold["selection_code_sha256"]
    ):
        raise ValueError("Frozen development shortlist selection source changed")
    if (
        hold["development_input_sha256"] != anchors["metrics.csv.gz"]
        or hold["candidate_count"] != 45
        or hold["development_period"] != {"start": "2001-01-01", "end": "2011-12-31"}
        or hold["later_stage_numerical_inputs_opened_for_this_selection"] is not False
        or hold["later_stage_performance_run_for_this_selection"] is not False
    ):
        raise ValueError("Hold must describe the frozen development-only 45-option selection")
    return {
        "input_artifact_sha256": dict(anchors),
        "frozen_source_sha256": source_hashes,
        "selection_code_sha256": hold["selection_code_sha256"],
        "hold": hold,
    }


def _safe_schema(frame: pd.DataFrame, *, shortlist=False) -> None:
    allowed = METRIC_COLUMNS | (SHORTLIST_EXTRA if shortlist else set())
    if (
        frame.columns.duplicated().any()
        or set(frame.columns) - allowed
        or not REQUIRED_COLUMNS <= set(frame.columns)
    ):
        raise ValueError("Only whitelisted derived development metric fields are allowed")
    for name in frame.select_dtypes(include=["object", "string"]):
        if frame[name].dropna().astype(str).str.match(r"^(?:/|~/|file:|[A-Za-z]:[\\/])").any():
            raise ValueError("Metric values must not contain private paths")


def validate_metrics(metrics: pd.DataFrame, table: dict, *, complete=True) -> pd.DataFrame:
    """Canonical identities, costs and scored period are mandatory; retain undefined Sharpe."""
    _safe_schema(metrics, shortlist=not complete)
    ids = metrics.candidate_id
    if ids.duplicated().any() or not ids.map(lambda item: isinstance(item, str)).all():
        raise ValueError("Candidate identities must be unique strings")
    if set(ids) - set(table) or (complete and (len(ids) != len(table) or set(ids) != set(table))):
        raise ValueError("Development rows must match the canonical candidate grid")
    result = metrics.copy()
    for name, getter in IDENTITY_FIELDS.items():
        expected = ids.map({identity: getter(candidate) for identity, candidate in table.items()})
        if not result[name].eq(expected).all():
            raise ValueError(f"Candidate field differs from canonical grid: {name}")
    constants = {
        "commission_bps": 1.0,
        "slippage_bps": 3.0,
        "sampling_minutes": 30,
        "scored_sessions": 2767,
        "start_session": "2001-01-02",
        "end_session": "2011-12-30",
    }
    for name, value in constants.items():
        if not result[name].eq(value).all():
            raise ValueError(f"Frozen development contract differs: {name}")
    for name in ("cash_excess_sharpe", "cagr", "max_drawdown"):
        result[name] = pd.to_numeric(result[name], errors="raise")
    if "cash_excess_Sharpe" in result and not np.array_equal(
        result.cash_excess_sharpe.to_numpy(), result.cash_excess_Sharpe.to_numpy(), equal_nan=True
    ):
        raise ValueError("Sharpe aliases disagree")
    return result


def _score_stats(scores: pd.Series) -> dict:
    finite = scores.loc[np.isfinite(scores)]
    return {
        "pair_count": int(len(scores)),
        "finite_sharpe_count": int(len(finite)),
        "nonfinite_sharpe_count": int(len(scores) - len(finite)),
        "mean_sharpe": finite.mean(),
        "median_sharpe": finite.median(),
        "q25_sharpe": finite.quantile(0.25),
        "q75_sharpe": finite.quantile(0.75),
        "sample_std_sharpe": finite.std(ddof=1),
        "min_sharpe": finite.min(),
        "max_sharpe": finite.max(),
    }


def equal_wait_analysis(
    metrics: pd.DataFrame,
    shortlist: pd.DataFrame,
    table: dict,
    *,
    waits=WAIT_HOURS,
    expected_shortlist_count=45,
    family_quota=5,
) -> dict:
    """Fixed-pair descriptive dependence; does not rank or replace shortlist members."""
    if tuple(waits) != WAIT_HOURS:
        raise ValueError("Requested common waits must be 0 through 6 hours in half-hour intervals")
    full = validate_metrics(metrics, table)
    selected = validate_metrics(shortlist, table, complete=False)
    if (
        len(selected) != expected_shortlist_count
        or selected[["entry_rule", "exit_rule"]].duplicated().any()
    ):
        raise ValueError("Frozen shortlist requires the exact number of distinct indicator pairs")
    expected_families = {candidate.family_cell for candidate in table.values()}
    if selected.groupby("family_cell").size().to_dict() != {
        family: family_quota for family in expected_families
    }:
        raise ValueError("Frozen shortlist family quotas are incomplete")
    saved = full.set_index("candidate_id", drop=False).loc[selected.candidate_id]
    # The old asymmetrical selection rows are verified, never rewritten or selected again.
    for name in REQUIRED_COLUMNS:
        a, b = (
            saved[name].to_numpy(),
            selected[name].to_numpy()
            if name != "candidate_id"
            else selected.candidate_id.to_numpy(),
        )
        if name == "candidate_id":
            a = saved.index.to_numpy()
        if name in {"cash_excess_sharpe", "cagr", "max_drawdown"}:
            agrees = np.array_equal(a, b, equal_nan=True)
        else:
            agrees = np.array_equal(a, b)
        if not agrees:
            raise ValueError(f"Frozen shortlist row no longer matches development source: {name}")
    selected_pairs = set(zip(selected.entry_rule, selected.exit_rule, strict=True))
    all_pairs = {
        (candidate.entry_rule.rule_id, candidate.exit_rule.rule_id) for candidate in table.values()
    }
    equal = full.loc[
        full.entry_confirmation_hours.eq(full.exit_confirmation_hours)
        & full.entry_confirmation_hours.isin(waits)
    ].copy()
    if len(equal) != len(all_pairs) * len(waits):
        raise ValueError("Equal-wait grid is incomplete")
    for _, rows in equal.groupby("entry_confirmation_hours"):
        if set(zip(rows.entry_rule, rows.exit_rule, strict=True)) != all_pairs:
            raise ValueError("The indicator-pair population changed across common waits")
    equal["common_confirmation_hours"] = equal.entry_confirmation_hours
    equal["pair_id"] = equal.entry_rule + "__" + equal.exit_rule
    equal["frozen_shortlist_member"] = [
        (a, b) in selected_pairs for a, b in zip(equal.entry_rule, equal.exit_rule, strict=True)
    ]
    equal["finite_sharpe"] = np.isfinite(equal.cash_excess_sharpe)
    equal = equal.sort_values(["common_confirmation_hours", "pair_id"], kind="stable").reset_index(
        drop=True
    )
    zero = equal.loc[equal.common_confirmation_hours.eq(0)].set_index("pair_id")
    changes = equal[
        [
            "candidate_id",
            "pair_id",
            "family_cell",
            "entry_rule",
            "exit_rule",
            "common_confirmation_hours",
            "frozen_shortlist_member",
            "cash_excess_sharpe",
        ]
    ].copy()
    changes["zero_wait_sharpe"] = changes.pair_id.map(zero.cash_excess_sharpe)
    changes["finite_paired_sharpe"] = np.isfinite(changes.cash_excess_sharpe) & np.isfinite(
        changes.zero_wait_sharpe
    )
    changes["delta_sharpe_vs_zero"] = (changes.cash_excess_sharpe - changes.zero_wait_sharpe).where(
        changes.finite_paired_sharpe
    )
    summaries, family_summaries = [], []
    for cohort, frame in (
        ("frozen_45_pairs", equal.loc[equal.frozen_shortlist_member]),
        ("all_289_pairs", equal),
    ):
        for wait, rows in frame.groupby("common_confirmation_hours", sort=True):
            deltas = changes.loc[
                changes.candidate_id.isin(rows.candidate_id), "delta_sharpe_vs_zero"
            ].dropna()
            summaries.append(
                {
                    "cohort": cohort,
                    "common_confirmation_hours": wait,
                    **_score_stats(rows.cash_excess_sharpe),
                    "finite_paired_count": len(deltas),
                    "mean_delta_sharpe_vs_zero": deltas.mean(),
                    "median_delta_sharpe_vs_zero": deltas.median(),
                    "positive_delta_count": int(deltas.gt(0).sum()),
                    "negative_delta_count": int(deltas.lt(0).sum()),
                    "unchanged_delta_count": int(deltas.eq(0).sum()),
                    "fraction_improved_vs_zero": deltas.gt(0).mean() if len(deltas) else np.nan,
                    "mean_trade_count": rows.trade_count.mean()
                    if "trade_count" in rows
                    else np.nan,
                    "median_trade_count": rows.trade_count.median()
                    if "trade_count" in rows
                    else np.nan,
                    "mean_annualized_turnover": rows.annualized_turnover.mean()
                    if "annualized_turnover" in rows
                    else np.nan,
                    "mean_cost_drag_return": rows.cost_drag_return.mean()
                    if "cost_drag_return" in rows
                    else np.nan,
                }
            )
        for (family, wait), rows in frame.groupby(
            ["family_cell", "common_confirmation_hours"], sort=True
        ):
            family_summaries.append(
                {
                    "cohort": cohort,
                    "family_cell": family,
                    "common_confirmation_hours": wait,
                    **_score_stats(rows.cash_excess_sharpe),
                }
            )
    ranges = []
    for pair_id, rows in equal.groupby("pair_id", sort=True):
        finite = rows.loc[rows.finite_sharpe, "cash_excess_sharpe"]
        first = rows.iloc[0]
        ranges.append(
            {
                "pair_id": pair_id,
                "entry_rule": first.entry_rule,
                "exit_rule": first.exit_rule,
                "family_cell": first.family_cell,
                "frozen_shortlist_member": bool(first.frozen_shortlist_member),
                "wait_count": len(rows),
                "finite_sharpe_count": len(finite),
                "minimum_sharpe_across_waits": finite.min(),
                "maximum_sharpe_across_waits": finite.max(),
                "sharpe_span_0_to_6_hours": finite.max() - finite.min(),
            }
        )
    return {
        "per_pair_wait_range": pd.DataFrame(ranges),
        "per_pair_equal_wait_metrics": equal,
        "per_wait_summary": pd.DataFrame(summaries),
        "paired_changes_vs_zero": changes,
        "family_wait_summary": pd.DataFrame(family_summaries),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--shortlist", type=Path, default=DEFAULT_SHORTLIST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Never overwrite published sensitivity evidence")
    provenance = verify_public_inputs(args.development, args.shortlist)
    if sha256(DEFAULT_CONFIG) != CONFIG_SHA256:
        raise ValueError("Frozen configuration changed")
    config = protocol.load_config(DEFAULT_CONFIG)
    _, table = protocol._candidate_table(config)
    metrics = pd.read_csv(args.development / "metrics.csv.gz", float_precision="round_trip")
    shortlist = pd.read_csv(
        args.shortlist / "development_shortlist.csv", float_precision="round_trip"
    )
    if shortlist.candidate_id.tolist() != provenance["hold"]["candidate_ids"]:
        raise ValueError("Shortlist rows do not match the frozen hold IDs/order")
    frames = equal_wait_analysis(metrics, shortlist, table)
    if (
        len(frames["per_pair_equal_wait_metrics"]) != 3757
        or int(frames["per_pair_equal_wait_metrics"].frozen_shortlist_member.sum()) != 585
    ):
        raise ValueError("Current analysis requires 289 and 45 unchanged pairs at all 13 waits")
    summary = {
        "project": "Trend Following Study",
        "scope": "development_only_equal_confirmation_wait_sensitivity",
        "development_period": {"start": "2001-01-01", "end": "2011-12-31"},
        "scored_sessions": 2767,
        "common_confirmation_hours": WAIT_HOURS,
        "confirmation_unit": "trading_hours_wait_after_first_qualifying_completed_30min_close",
        "required_checks_formula": "1 + 2 * common_confirmation_hours",
        "indicator_unit": "daily_sessions",
        "selection_score": "daily_cash_excess_sharpe_252_sessions_sample_std_ddof_1",
        "commission_bps_per_one_way_order": 1,
        "slippage_bps_per_one_way_order": 3,
        "total_cost_bps_per_one_way_order": 4,
        "taxes_financing_separate_fund_expenses": False,
        "all_indicator_pair_count": 289,
        "fixed_development_shortlist_pair_count": 45,
        "all_equal_wait_configuration_count": 3757,
        "fixed_shortlist_equal_wait_configuration_count": 585,
        "original_asymmetric_shortlist_unchanged": True,
        "wait_6p5_excluded_from_this_user_requested_analysis": True,
        "market_performance_recomputed": False,
        "validation_historical_oos_numerical_inputs_opened": False,
        "new_wait_or_strategy_selected": False,
        "paired_improvement_comparison": "strict_delta_above_zero_no_selection_tie_tolerance",
        "paired_improvement_comparison_tolerance": 0.0,
        "distribution_statistic_definition": "Mean/median/quantiles summarize individual configuration Sharpes, not the Sharpe of a combined portfolio; quartiles use linear interpolation and distribution sample std uses ddof=1.",
        "trade_count_unit": "completed_qqq_round_trips_in_entire_scored_development_period",
        "sharpe_span_definition": "Maximum minus minimum finite individual Sharpe across the 13 common waits, without choosing a wait.",
        "raw_quotes_or_trades_exported": False,
        "caveat": "In-sample dependence only; the 45 pairs were selected on development results with independently optimized original waits, so this subset is selection-biased. No evidence here establishes a best wait or future performance. Later-period work remains on hold.",
        "config_file_sha256": CONFIG_SHA256,
        **{key: value for key, value in provenance.items() if key != "hold"},
        "per_wait_summary": frames["per_wait_summary"].to_dict(orient="records"),
    }
    args.output.mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
    write_json(args.output / "summary.json", summary)
    write_json(
        args.output / "artifact_hashes.json",
        {path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()},
    )
    print(
        json.dumps(
            clean_json(
                {"scope": summary["scope"], "rows": 3757, "fixed_pairs": 45, "waits": WAIT_HOURS}
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
