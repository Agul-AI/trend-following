#!/usr/bin/env python3
"""Publish a sealed DEVELOPMENT-only 3% cash/common-wait diagnostic.

Read-only: no simulations, selections, later-period outcomes or price packets.
Both study and result seals are verified before any numerical CSV is parsed.
Only derived metrics and small benchmark curves are exported; the private
per-candidate array packets, authorizations and source paths are never copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trend_following import research_constant_cash as study  # noqa: E402
from trend_following import research_quality_protocol_v5 as protocol  # noqa: E402

DEFAULT_OUTPUT = ROOT / "reports/constant_cash_wait_counts/20261008"
START = pd.Timestamp("2001-01-02 09:30", tz="America/New_York").tz_convert("UTC")
END = pd.Timestamp("2011-12-30 17:00", tz="America/New_York").tz_convert("UTC")
YEARS = (END - START).total_seconds() / (365 * 86400)
SESSIONS = 2767
IDENTITY = {
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
METRICS = {
    "candidate_id",
    *IDENTITY,
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
REQUIRED = {
    "candidate_id",
    "cash_excess_sharpe",
    "cagr",
    "max_drawdown",
    "terminal_equity",
    "cash_terminal_equity",
    "scored_sessions",
    "start_session",
    "end_session",
    "sampling_minutes",
    "commission_bps",
    "slippage_bps",
}
COUNTS = {
    "wait_hours",
    "required_consecutive_hits",
    "configuration_count",
    "finite_Sharpe_count",
    "undefined_Sharpe_count",
    "beat_BH_Sharpe_count",
    "beat_BH_Sharpe_pct",
    "tie_BH_Sharpe_count",
    "strict_gt_BH_Sharpe_count",
    "below_BH_Sharpe_count",
    "near_tie_within_1e_10_count",
    "BH_cash_excess_Sharpe",
    "median_finite_Sharpe",
    "best_finite_Sharpe",
    "cash_nominal_annual_rate",
    "commission_bps",
    "slippage_bps",
}
CURVE = {"timestamp", "return", "equity", "cash_return", "exposure"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(clean_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _schema(frame, allowed, required):
    if frame.columns.duplicated().any() or set(frame) - allowed or not required <= set(frame):
        raise ValueError("Only whitelisted derived fields are publishable; no quotes/trades")
    for column in frame.select_dtypes(include=["object", "string"]):
        if (
            frame[column]
            .dropna()
            .astype(str)
            .str.contains(r"(?:^/|^~/|^file:|^[A-Za-z]:[\\/])", regex=True)
            .any()
        ):
            raise ValueError("Derived metrics must not contain private paths")


def _read(path):
    # Pandas can mangle duplicate headers; reject them before parsing numeric data.
    with Path(path).open(newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV headers are prohibited")
    return pd.read_csv(path, float_precision="round_trip")


def verify_release(sidecar):
    payload = study.verify_study(sidecar)
    result = study.verify_results(sidecar)
    if (
        payload.get("cash") != study.cash_policy()
        or payload.get("cash", {}).get("annual_rate") != 0.03
        or payload.get("period") != {"start": "2001-01-01", "end": "2011-12-31"}
        or payload.get("wait_hours") != list(study.WAIT_HOURS)
        or payload.get("candidate_count_per_wait") != 289
        or payload.get("candidate_count") != 3757
        or payload.get("commission_bps_per_order") != 1
        or payload.get("slippage_bps_per_order") != 3
        or result.get("freeze_sha256") != protocol.canonical_hash(payload)
        or result.get("cash_policy_sha256") != protocol.canonical_hash(payload["cash"])
        or result.get("stage") != "development"
        or result.get("no_winner_selected") is not True
        or result.get("no_later_market_files_loaded") is not True
    ):
        raise ValueError("Release must bind new 3% nominal cash, development grid and costs")
    for name in payload.get("source_hashes", {}):
        if Path(name).is_absolute() or ".." in Path(name).parts or not name.endswith(".py"):
            raise ValueError("Public source hashes must have safe repository-relative names")
    return payload, result


def validate_metrics(frame, *, candidates=True):
    allowed = METRICS if candidates else METRICS | {"portfolio"}
    required = REQUIRED | (set(IDENTITY) if candidates else {"portfolio"})
    _schema(frame, allowed, required)
    if (
        frame.candidate_id.duplicated().any()
        or not frame.candidate_id.map(lambda item: isinstance(item, str)).all()
    ):
        raise ValueError("Unique string candidate identities required")
    constants = {
        "commission_bps": 1,
        "slippage_bps": 3,
        "sampling_minutes": 30,
        "scored_sessions": SESSIONS,
        "start_session": "2001-01-02",
        "end_session": "2011-12-30",
    }
    for name, value in constants.items():
        if not frame[name].eq(value).all():
            raise ValueError(f"Development accounting contract differs: {name}")
    for name in (
        "cash_excess_sharpe",
        "cagr",
        "max_drawdown",
        "terminal_equity",
        "cash_terminal_equity",
    ):
        pd.to_numeric(frame[name], errors="raise")
    if "cash_excess_Sharpe" in frame and not np.array_equal(
        frame.cash_excess_sharpe, frame.cash_excess_Sharpe, equal_nan=True
    ):
        raise ValueError("Cash-excess Sharpe aliases disagree")
    if candidates:
        table = {
            c.candidate_id: c
            for hours in study.WAIT_HOURS
            for c in study.candidates_for_wait(hours)
        }
        if len(frame) != 3757 or set(frame.candidate_id) != set(table):
            raise ValueError("Require the same complete 289 indicator pairs at all 13 waits")
        for name, getter in IDENTITY.items():
            expected = frame.candidate_id.map({key: getter(c) for key, c in table.items()})
            if not frame[name].eq(expected).all():
                raise ValueError(f"Candidate differs from canonical equal-wait grid: {name}")
    elif (
        len(frame) != 2
        or set(frame.portfolio) != {"cash", "buy_and_hold"}
        or not frame.candidate_id.eq("benchmark_" + frame.portfolio).all()
    ):
        raise ValueError("Exactly cash and cost-matched QQQ buy-and-hold benchmarks required")
    return frame.copy()


def independent_counts(metrics, benchmark):
    if not np.isfinite(benchmark):
        raise ValueError("Buy-and-hold Sharpe must be finite")
    rows = []
    for hours in study.WAIT_HOURS:
        cohort = metrics.loc[metrics.entry_confirmation_hours.eq(hours)]
        scores = cohort.cash_excess_sharpe.to_numpy(float)
        if len(scores) != 289 or not cohort.exit_confirmation_hours.eq(hours).all():
            raise ValueError("Counts require 289 equal-wait configurations per cohort")
        finite = np.isfinite(scores)
        diff = scores - benchmark
        rows.append(
            {
                "wait_hours": hours,
                "required_consecutive_hits": 1 + int(hours * 2),
                "configuration_count": 289,
                "finite_Sharpe_count": int(finite.sum()),
                "undefined_Sharpe_count": int((~finite).sum()),
                "beat_BH_Sharpe_count": int((finite & (scores > benchmark + 1e-10)).sum()),
                "beat_BH_Sharpe_pct": 100
                * int((finite & (scores > benchmark + 1e-10)).sum())
                / 289,
                "tie_BH_Sharpe_count": int((finite & (np.abs(diff) <= 1e-10)).sum()),
                "strict_gt_BH_Sharpe_count": int((finite & (scores > benchmark)).sum()),
                "below_BH_Sharpe_count": int((finite & (scores < benchmark - 1e-10)).sum()),
                "near_tie_within_1e_10_count": int((finite & (np.abs(diff) <= 1e-10)).sum()),
                "BH_cash_excess_Sharpe": benchmark,
                "median_finite_Sharpe": float(np.median(scores[finite]))
                if finite.any()
                else np.nan,
                "best_finite_Sharpe": float(np.max(scores[finite])) if finite.any() else np.nan,
                "cash_nominal_annual_rate": 0.03,
                "commission_bps": 1,
                "slippage_bps": 3,
            }
        )
    return pd.DataFrame(rows)


def validate_counts(saved, independent):
    _schema(saved, COUNTS, COUNTS)
    if len(saved) != 13 or saved.wait_hours.duplicated().any():
        raise ValueError("Exactly 13 distinct wait-count rows required")
    saved = saved.sort_values("wait_hours").reset_index(drop=True)
    for column in independent:
        integer_column = column.endswith("_count") or column in {
            "required_consecutive_hits",
            "configuration_count",
        }
        agrees = (
            np.array_equal(saved[column], independent[column])
            if integer_column
            else np.allclose(
                saved[column], independent[column], rtol=1e-12, atol=1e-12, equal_nan=True
            )
        )
        if not agrees:
            raise ValueError(
                f"Saved counts differ from independently recomputed new-cash results: {column}"
            )
    return independent


def validate_curves(bh, cash, benchmarks, initial_cash):
    for frame in (bh, cash):
        _schema(frame, CURVE, CURVE)
    clocks = [pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)) for frame in (bh, cash)]
    clock = clocks[0]
    local = clock.tz_convert("America/New_York")
    if (
        not clock.equals(clocks[1])
        or len(clock) != SESSIONS
        or clock.has_duplicates
        or not clock.is_monotonic_increasing
        or clock[0] != START.normalize() + pd.Timedelta(hours=22)
        or clock[-1] != END
        or not (local.hour == 17).all()
        or not (local.minute == 0).all()
        or not (local.dayofweek < 5).all()
    ):
        raise ValueError("Benchmark curves require matched development-only 17:00 NY clocks")
    reference = cash["return"].to_numpy(float)
    cash_growth = cash.equity.to_numpy(float)
    if (
        not np.isfinite(cash_growth).all()
        or not np.isfinite(reference).all()
        or (reference < 0).any()
    ):
        raise ValueError("3% assumed cash wealth/returns must be finite and nonnegative")
    dt_years = np.diff(np.r_[START.value, clock.asi8]) / 1e9 / (365 * 86400)
    if (reference < 0.03 * dt_years - 2e-12).any() or (
        reference > np.expm1(0.03 * dt_years) + 2e-12
    ).any():
        raise ValueError("Cash curve violates fixed nominal 3% ACT/365 event-compounding bounds")
    if not np.allclose(
        cash_growth, initial_cash * np.cumprod(1 + reference), rtol=1e-11, atol=1e-8
    ):
        raise ValueError("Cash equity must reconcile to its compounded returns")
    statistics = {}
    for name, frame in (("buy_and_hold", bh), ("cash", cash)):
        equity = frame.equity.to_numpy(float)
        returns = equity / np.r_[initial_cash, equity[:-1]] - 1
        if (
            not np.isfinite(equity).all()
            or (equity <= 0).any()
            or not np.allclose(frame["return"], returns, rtol=1e-10, atol=2e-12)
            or not np.allclose(frame.cash_return, reference, rtol=1e-10, atol=2e-12)
        ):
            raise ValueError("Both benchmark returns/equity must share the new cash reference")
        excess = returns - reference
        sd = excess.std(ddof=1)
        sharpe = float(np.sqrt(252) * excess.mean() / sd) if sd > 0 else np.nan
        if name == "cash":
            # Small serialization roundoff must not create a fictitious finite cash Sharpe.
            if not np.allclose(excess, 0, rtol=0, atol=2e-12):
                raise ValueError("Cash benchmark must have zero cash-excess returns")
            sharpe = np.nan
        wealth = np.r_[initial_cash, equity]
        actual = {
            "cagr": (equity[-1] / initial_cash) ** (1 / YEARS) - 1,
            "cash_excess_sharpe": sharpe,
            "terminal_equity": equity[-1],
            "max_drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min(),
        }
        row = benchmarks.loc[benchmarks.portfolio.eq(name)].iloc[0]
        for key, value in actual.items():
            if not np.isclose(row[key], value, rtol=1e-9, atol=1e-10, equal_nan=True):
                raise ValueError(f"Benchmark metric does not reconcile to curve: {name}/{key}")
        statistics[name] = actual
    if not 0.03 < statistics["cash"]["cagr"] <= np.expm1(0.03) + 1e-12:
        raise ValueError("Nominal event-compounded cash CAGR is not exactly 3% APY")
    return statistics


def publish(sidecar, output):
    sidecar, output = Path(sidecar).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Never overwrite published diagnostic evidence")
    if output == sidecar or output.is_relative_to(sidecar) or sidecar.is_relative_to(output):
        raise ValueError("Public output must not modify or contain the private sealed study")
    payload, seal = verify_release(sidecar)  # BOTH gates precede every numeric reader.
    private = sidecar / "results"
    metrics = validate_metrics(_read(private / "candidate_metrics.csv"))
    benchmarks = validate_metrics(_read(private / "benchmark_metrics.csv"), candidates=False)
    bh = benchmarks.loc[benchmarks.portfolio.eq("buy_and_hold")].iloc[0]
    cash = benchmarks.loc[benchmarks.portfolio.eq("cash")].iloc[0]
    if not np.isclose(bh.cash_excess_sharpe, seal["BH_cash_excess_Sharpe"], rtol=0, atol=1e-12):
        raise ValueError("Benchmark metric differs from its sealed Sharpe")
    if not np.isnan(cash.cash_excess_sharpe):
        raise ValueError("Cash-excess Sharpe of cash is undefined, not zero or finite")
    curves = {name: _read(private / f"{name}_daily.csv") for name in ("buy_and_hold", "cash")}
    statistics = validate_curves(
        curves["buy_and_hold"], curves["cash"], benchmarks, payload["initial_cash"]
    )
    if not np.allclose(metrics.cash_terminal_equity, cash.terminal_equity, rtol=1e-11, atol=1e-8):
        raise ValueError("Candidate metrics must use the same new 3% cash benchmark")
    counts = validate_counts(
        _read(private / "counts_by_wait.csv"), independent_counts(metrics, bh.cash_excess_sharpe)
    )
    public_benchmarks = benchmarks.drop(
        columns=["candidate_id", *[c for c in IDENTITY if c in benchmarks]]
    )
    public_benchmarks["portfolio_label"] = public_benchmarks.portfolio.map(
        {"buy_and_hold": "QQQ buy-and-hold", "cash": "Cash"}
    )
    for frame in (metrics, public_benchmarks):
        frame["cash_nominal_annual_rate"] = 0.03
        frame["cash_convention"] = "nominal_ACT365_event_compounding_not_3pct_APY"
    combined = pd.concat(
        [frame.assign(portfolio=name) for name, frame in curves.items()], ignore_index=True
    )
    summary = {
        "project": "Trend Following Study",
        "scope": "development_only_3pct_cash_common_wait_diagnostic",
        "development_period": payload["period"],
        "first_scored_session": "2001-01-02",
        "last_scored_session": "2011-12-30",
        "scored_sessions": SESSIONS,
        "common_confirmation_hours": list(study.WAIT_HOURS),
        "configurations_per_wait": 289,
        "configurations_evaluated": 3757,
        "same_entry_and_exit_wait": True,
        "required_consecutive_hits_formula": "1 + 2 * wait_hours",
        "indicator_unit": "daily_sessions",
        "cash_policy": payload["cash"],
        "effective_cash_cagr": statistics["cash"]["cagr"],
        "commission_bps_per_one_way_order": 1,
        "slippage_bps_per_one_way_order": 3,
        "total_bps_per_one_way_order": 4,
        "taxes_financing_separate_fund_expenses": False,
        "initial_capital": payload["initial_cash"],
        "benchmark_statistics": statistics,
        "primary_beat_definition": "finite candidate cash-excess Sharpe > BH Sharpe + 1e-10",
        "strict_gt_count_also_reported": True,
        "undefined_scores_retained_in_289_denominator": True,
        "score_definition": "session_close_cash_excess_return_mean / sample_sd_ddof1 * sqrt(252)",
        "counts_independently_recomputed": True,
        "market_performance_recomputed_for_new_cash": True,
        "new_global_wait_or_strategy_selected": False,
        "later_stage_numerical_inputs_opened": False,
        "freeze_payload_sha256": protocol.canonical_hash(payload),
        "result_seal_payload_sha256": protocol.canonical_hash(seal),
        "development_manifest_sha256": payload["development_manifest_sha256"],
        "implementation_source_sha256": payload["source_hashes"],
        "derived_input_sha256": {
            name: seal["files"][name]
            for name in (
                "candidate_metrics.csv",
                "benchmark_metrics.csv",
                "counts_by_wait.csv",
                "cash_daily.csv",
                "buy_and_hold_daily.csv",
            )
        },
        "publication_helper_sha256": sha256(__file__),
        "limitations": [
            "In-sample development diagnostic, not OOS confirmation",
            "Hypothetical commission, slippage and cash yield",
            "No optimal wait chosen and no new candidate shortlist selected",
            "Prior sealed DTB3 studies remain unchanged, not relabeled as 3% results",
        ],
    }
    # Nothing is written until every release, schema, metric and count check passes.
    output.mkdir(parents=True)
    metrics.to_csv(output / "candidate_metrics.csv", index=False)
    public_benchmarks.to_csv(output / "benchmark_metrics.csv", index=False)
    counts.to_csv(output / "counts_by_wait.csv", index=False)
    combined.to_csv(output / "benchmark_daily_curves.csv", index=False)
    write_json(output / "summary.json", summary)
    write_json(
        output / "artifact_hashes.json",
        {path.name: sha256(path) for path in sorted(output.iterdir())},
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, default=study.DEFAULT_SIDECAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    summary = publish(args.sidecar, args.output)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
