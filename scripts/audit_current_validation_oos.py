#!/usr/bin/env python3
"""Current sealed validation/OOS audits only; never imports/evaluates a strategy.

The validation command cannot open OOS numeric outputs. The OOS command first
verifies the already sealed one-candidate winner and its selection provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if __package__:
    from . import audit_current_development as base
else:
    import audit_current_development as base

require = base.require


def stage_seal(study, name, registration, freeze):
    root = study / "stages" / name
    seal = base.read_seal(root / "stage_seal.json")
    require(seal["stage"] == name, "Wrong sealed current phase")
    require(
        seal["registration_sha256"] == base.canonical_hash(registration)
        and seal["data_freeze_sha256"] == base.canonical_hash(freeze),
        "Phase registration/data identity differs",
    )
    for path, expected in seal["artifact_hashes"].items():
        require(base.file_hash(root / path) == expected, f"Sealed {name} artifact changed: {path}")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "stage_seal.json"
    }
    require(actual == set(seal["artifact_hashes"]), "Sealed phase artifact set changed")
    return seal


def context(study, config_path):
    study = Path(study).resolve()
    config_path = Path(config_path).resolve()
    registration = base.read_seal(study / "registration.json")
    config = yaml.safe_load(config_path.read_text())
    expected = base.audit_contract(config, registration, config_path, config_path.parent.parent)
    freeze = base.read_seal(study / "prepared/freeze.json")
    development = stage_seal(study, "development", registration, freeze)
    finalists = base.read_seal(study / "stages/development/finalists.json")
    require(
        development["finalists_sha256"] == base.canonical_hash(finalists),
        "Development finalist seal identity changed",
    )
    require(
        len(finalists["candidates"]) == development["finalist_count"] == 9,
        "Expected exactly nine current development finalists",
    )
    require(
        len({row["candidate_id"] for row in finalists["candidates"]}) == 9,
        "Duplicate development finalists",
    )
    require(
        all(row == expected[row["candidate_id"]] for row in finalists["candidates"]),
        "Development finalist parameters differ from frozen grid",
    )
    return study, config, registration, freeze, expected, finalists


def check_metrics(metrics, calendar, dates, allowed, slip):
    require(
        len(metrics) == len(allowed)
        and not metrics.candidate_id.duplicated().any()
        and set(metrics.candidate_id) == set(allowed),
        "Phase candidate universe differs from its sealed allowed set",
    )
    require(
        metrics.commission_bps.eq(1).all() and metrics.slippage_bps.eq(slip).all(),
        "Wrong phase/scenario transaction costs",
    )
    require(
        metrics.quality_profile.eq("observed_only_blackout_v1").all()
        and metrics.daily_valuation_hour_ny.eq(17).all(),
        "Phase quality/quote clock changed",
    )
    cal = (
        calendar[calendar.session.astype(str).between(dates["start"], dates["end"])]
        .sort_values("session")
        .copy()
    )
    require(not cal.empty, "No current scored calendar")
    end = pd.Timestamp(cal.session.iloc[-1] + " 17:00", tz="America/New_York").tz_convert("UTC")
    years = (end - pd.Timestamp(cal.market_open.iloc[0])).total_seconds() / (365 * 86400)
    require(
        metrics.start_session.eq(cal.session.iloc[0]).all()
        and metrics.end_session.eq(cal.session.iloc[-1]).all()
        and metrics.scored_sessions.eq(len(cal)).all(),
        "Phase periods/clock/session counts differ",
    )
    require(
        np.allclose(
            metrics.cagr,
            (metrics.terminal_equity / 1000) ** (1 / years) - 1,
            rtol=1e-10,
            atol=1e-12,
        ),
        "Phase CAGR mismatch",
    )
    require(
        np.allclose(
            metrics.total_return, metrics.terminal_equity / 1000 - 1, rtol=1e-10, atol=1e-12
        ),
        "Phase terminal return mismatch",
    )
    require(
        np.allclose(
            metrics.cost_drag_return,
            (metrics.gross_terminal_equity - metrics.terminal_equity) / 1000,
            rtol=1e-10,
            atol=1e-12,
        ),
        "Phase gross/net drag mismatch",
    )
    require(
        np.allclose(
            metrics.annualized_turnover, metrics.total_turnover / years, rtol=1e-10, atol=1e-12
        ),
        "Phase turnover annualization mismatch",
    )
    require(
        np.allclose(
            metrics.total_turnover,
            metrics.trade_count * (1 + 1 / ((1 + slip / 10000) * 1.0001)),
            rtol=1e-9,
            atol=1e-7,
        ),
        "Phase one-way all-in turnover/cost accounting mismatch",
    )
    require(
        metrics.order_count.eq(metrics.trade_count * 2).all(),
        "Phase roundtrip/terminal order counts mismatch",
    )
    require(
        metrics.exposure_fraction.between(0, 1).all() and metrics.max_drawdown.between(-1, 0).all(),
        "Phase unlevered risk range invalid",
    )
    return cal, years


def pick_primary(metrics, tolerance=1e-10):
    finite = metrics[np.isfinite(pd.to_numeric(metrics.cash_excess_sharpe, errors="coerce"))]
    require(not finite.empty, "No finite primary score; cannot invent a winner")
    score = finite.cash_excess_sharpe.max()
    return finite.loc[finite.cash_excess_sharpe >= score - tolerance, "candidate_id"].min()


def curve_statistics(daily, metric, calendar):
    require("timestamp" in daily, "Missing current curve clock")
    stamps = pd.DatetimeIndex(pd.to_datetime(daily.timestamp, utc=True))
    expected = pd.DatetimeIndex(
        [
            pd.Timestamp(day + " 17:00", tz="America/New_York").tz_convert("UTC")
            for day in calendar.session
        ]
    )
    require(stamps.equals(expected), "Winner/BH/cash daily curve endpoints differ")
    ret = pd.to_numeric(daily["return" if "return" in daily else "strategy_return"], errors="raise")
    cash = pd.to_numeric(daily.cash_return, errors="raise")
    eq = pd.to_numeric(daily.equity, errors="raise")
    require(
        np.isfinite(ret).all() and np.isfinite(cash).all() and np.isfinite(eq).all(),
        "Nonfinite persisted daily curve",
    )
    require(
        np.allclose(ret, eq / np.r_[1000, eq.to_numpy()[:-1]] - 1, rtol=1e-10, atol=1e-12),
        "Persisted equity/return reconciliation failed",
    )
    excess = ret - cash
    sd = excess.std(ddof=1)
    sharpe = np.sqrt(252) * excess.mean() / sd if sd > 0 else np.nan
    vol = ret.std(ddof=1) * np.sqrt(252)
    wealth = np.r_[1000, eq.to_numpy()]
    dd = (wealth / np.maximum.accumulate(wealth) - 1).min()
    for actual, name in (
        (sharpe, "cash_excess_sharpe"),
        (vol, "annualized_volatility"),
        (dd, "max_drawdown"),
        (eq.iloc[-1], "terminal_equity"),
        (1000 * np.prod(1 + cash), "cash_terminal_equity"),
    ):
        require(
            np.isclose(actual, float(metric[name]), rtol=1e-10, atol=1e-9, equal_nan=True),
            f"Independent curve statistic mismatch: {name}",
        )
    return dict(
        cash_excess_sharpe=None if np.isnan(sharpe) else float(sharpe),
        volatility=float(vol),
        max_drawdown=float(dd),
        terminal_equity=float(eq.iloc[-1]),
    )


def validation_audit(study, config_path, benchmark_root=None):
    study, config, registration, freeze, expected, finalists = context(study, config_path)
    seal = stage_seal(study, "validation", registration, freeze)
    stage = study / "stages/validation"
    metrics = base.csv(stage / "metrics.csv")
    allowed = [row["candidate_id"] for row in finalists["candidates"]]
    calendar = base.csv(study / "prepared/validation/calendar.csv")
    cal, years = check_metrics(metrics, calendar, config["segments"]["validation"], allowed, 3)
    chosen = pick_primary(metrics, config["selection"]["tie_tolerance"])
    winner = base.read_seal(stage / "winner.json")
    require(
        len(winner["candidates"]) == 1 and winner["candidates"][0] == expected[chosen],
        "Sealed validation winner differs from independent primary-only selection",
    )
    require(
        winner["sealed_before_historical_oos"] is True, "Winner not explicitly locked before OOS"
    )
    require(
        winner["development_finalists_sha256"] == base.canonical_hash(finalists)
        and winner["validation_metrics_sha256"] == base.file_hash(stage / "metrics.csv")
        and winner["selection_sha256"] == registration["selection_sha256"],
        "Winner selection provenance changed",
    )
    require(
        seal["winner_id"] == chosen
        and seal["winner_sha256"] == base.canonical_hash(winner)
        and seal["candidate_count"] == 9,
        "Validation one-winner seal binding inconsistent",
    )
    flags = base.csv(stage / "post_gap_candidate_flags.csv")
    events = base.csv(stage / "post_gap_signal_events.csv")
    meta = json.loads((stage / "post_gap_diagnostic_metadata.json").read_text())
    require(
        set(flags.candidate_id) == set(allowed)
        and len(flags) == 9
        and not flags.candidate_id.duplicated().any(),
        "Validation discarded candidate flags",
    )
    require(
        events.empty
        and meta["total_event_rows"] == meta["stored_event_rows"] == 0
        and meta["truncated"] is False,
        "Expired old gap windows unexpectedly flagged validation",
    )
    require(
        metrics.post_gap_confirmed_signal_count.eq(0).all()
        and not base.bool_values(metrics.has_post_gap_confirmed_signal, "flag").any(),
        "Historical gap windows leaked into validation",
    )
    curves = {}
    if benchmark_root:
        for name in ("buy_and_hold", "cash"):
            folder = Path(benchmark_root) / name
            m = base.csv(folder / "metrics.csv")
            d = base.csv(folder / "daily.csv")
            check_metrics(m, calendar, config["segments"]["validation"], ["benchmark_" + name], 3)
            curves[name] = curve_statistics(d, m.iloc[0], cal)
    row = metrics.set_index("candidate_id").loc[chosen]
    return dict(
        status="pass",
        stage="validation",
        candidate_count=9,
        winner_id=chosen,
        winner_payload_sha256=base.canonical_hash(winner),
        seal_digests={
            "registration_payload_sha256": base.canonical_hash(registration),
            "data_freeze_payload_sha256": base.canonical_hash(freeze),
            "validation_stage_payload_sha256": base.canonical_hash(seal),
            "winner_payload_sha256": base.canonical_hash(winner),
        },
        independent_primary_only_selection=True,
        sealed_before_oos=True,
        start_session=cal.session.iloc[0],
        end_session=cal.session.iloc[-1],
        scored_sessions=len(cal),
        winner_metrics={
            key: float(row[key])
            for key in ("cagr", "cash_excess_sharpe", "max_drawdown", "terminal_equity")
        },
        benchmark_curves=curves,
        scope="Read CURRENT validation only; no OOS numerical artifacts opened.",
    )


def oos_audit(study, config_path):
    # Locks and exact validation provenance are verified BEFORE any OOS numeric file is opened.
    validation = validation_audit(study, config_path)
    study, config, registration, freeze, expected, finalists = context(study, config_path)
    locked = validation["winner_id"]
    seal = stage_seal(study, "historical-oos", registration, freeze)
    require(
        seal["winner_id"] == locked
        and seal["winner_sha256"] == validation["winner_payload_sha256"]
        and seal["candidate_count"] == 1
        and seal["selection_after_evaluation"] is False,
        "OOS winner replacement/reselection detected",
    )
    require(
        seal["slippage_bps_scenarios"] == [1.0, 3.0, 10.0, 25.0], "OOS stress scenario set changed"
    )
    root = study / "stages/historical-oos"
    calendar = base.csv(study / "prepared/historical-oos/calendar.csv")
    summaries = {}
    frames = {}
    timestamps = {}
    receipts = []
    for slip in (1.0, 3.0, 10.0, 25.0):
        folder = root / f"slippage_{slip:g}bps"
        per = {}
        for policy, key in (
            ("winner", locked),
            ("buy_and_hold", "benchmark_buy_and_hold"),
            ("cash", "benchmark_cash"),
        ):
            m = base.csv(folder / policy / "metrics.csv")
            d = base.csv(folder / policy / "daily.csv")
            cal, _ = check_metrics(m, calendar, config["segments"]["historical-oos"], [key], slip)
            stats = curve_statistics(d, m.iloc[0], cal)
            require(int(m.trade_count.iloc[0]) >= 0, "Invalid OOS trade count")
            per[policy] = {
                **stats,
                **{
                    field: float(m.iloc[0][field])
                    for field in (
                        "cagr",
                        "total_commission",
                        "total_slippage",
                        "cost_drag_return",
                        "gross_terminal_equity",
                    )
                },
            }
            frames[(slip, policy)] = d
            tr = base.csv(folder / policy / "trades.csv")
            require(len(tr) == int(m.order_count.iloc[0]), "Persisted ledger order count differs")
            if not tr.empty:
                require(
                    tr.candidate_id.eq(key).all(), "Trade ledger contains a replacement candidate"
                )
                expected_execution = tr.reference_price * np.where(
                    tr.side.eq("buy"), 1 + slip / 10000, 1 - slip / 10000
                )
                require(
                    np.allclose(tr.execution_price, expected_execution, rtol=1e-10, atol=1e-10),
                    "Adverse slippage execution-price formula differs",
                )
                require(
                    np.allclose(
                        tr.commission,
                        tr.quantity * tr.execution_price / 10000,
                        rtol=1e-10,
                        atol=1e-10,
                    ),
                    "Commission not charged exactly once at1bp",
                )
                require(
                    np.allclose(
                        tr.slippage,
                        tr.quantity * (tr.execution_price - tr.reference_price).abs(),
                        rtol=1e-10,
                        atol=1e-10,
                    ),
                    "Slippage not charged exactly once",
                )
                require(
                    np.isclose(
                        tr.commission.sum(), m.total_commission.iloc[0], rtol=1e-10, atol=1e-9
                    )
                    and np.isclose(
                        tr.slippage.sum(), m.total_slippage.iloc[0], rtol=1e-10, atol=1e-9
                    ),
                    "One-way commission/slippage charge reconciliation failed",
                )
            timestamps[(slip, policy)] = tr.loc[:, ["side", "timestamp", "reason"]]
            receipts.append((slip, policy, m.iloc[0].to_dict()))
        for policy in ("cash", "buy_and_hold"):
            require(
                frames[(slip, "winner")].timestamp.equals(frames[(slip, policy)].timestamp),
                "Winner and controls have mismatched curve clocks",
            )
            require(
                np.array_equal(
                    frames[(slip, "winner")].cash_return.to_numpy(),
                    frames[(slip, policy)].cash_return.to_numpy(),
                ),
                "Cash reference differs across winner/benchmarks",
            )
        summaries[f"{slip:g}bps"] = per
    for policy in ("winner", "buy_and_hold", "cash"):
        require(
            all(
                timestamps[(slip, policy)].equals(timestamps[(1.0, policy)])
                for slip in (3.0, 10.0, 25.0)
            ),
            "Cost stress changed orders/policy rather than only execution costs",
        )
        net = [
            summaries[f"{slip:g}bps"][policy]["terminal_equity"] for slip in (1.0, 3.0, 10.0, 25.0)
        ]
        gross = [
            summaries[f"{slip:g}bps"][policy]["gross_terminal_equity"]
            for slip in (1.0, 3.0, 10.0, 25.0)
        ]
        require(
            np.all(np.diff(net) <= 1e-9) and np.allclose(gross, gross[0], rtol=1e-10, atol=1e-9),
            "Cost drag/gross shadow inconsistent across scenarios",
        )
    table = base.csv(root / "cost_scenarios.csv")
    require(
        len(table) == 12 and set(table.slippage_bps) == {1.0, 3.0, 10.0, 25.0},
        "Combined OOS scenario table incomplete",
    )
    require(
        set(table.policy) == {locked, "cash", "buy_and_hold"},
        "OOS scenario table introduced another strategy",
    )
    expected_pairs = {
        (slip, policy)
        for slip in (1.0, 3.0, 10.0, 25.0)
        for policy in (locked, "cash", "buy_and_hold")
    }
    require(
        not table.duplicated(["slippage_bps", "policy"]).any()
        and set(zip(table.slippage_bps, table.policy, strict=True)) == expected_pairs,
        "Duplicate/missing scenario-policy pair",
    )
    for slip, policy, metric in receipts:
        label = locked if policy == "winner" else policy
        combined = table[(table.slippage_bps == slip) & table.policy.eq(label)].iloc[0]
        for field in (
            "cagr",
            "cash_excess_sharpe",
            "max_drawdown",
            "terminal_equity",
            "total_commission",
            "total_slippage",
            "cost_drag_return",
        ):
            require(
                np.isclose(
                    float(combined[field]),
                    float(metric[field]),
                    rtol=1e-10,
                    atol=1e-9,
                    equal_nan=True,
                ),
                "Combined scenario table differs from individually audited artifacts",
            )
    attempts = [
        json.loads(line)
        for line in (study / "attempts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    va = [
        row
        for row in attempts
        if row.get("stage") == "validation" and row.get("status") == "completed"
    ]
    oo = [
        row
        for row in attempts
        if row.get("stage") == "historical-oos" and row.get("status") == "completed"
    ]
    require(
        va and oo and pd.Timestamp(va[-1]["finished_at"]) < pd.Timestamp(oo[-1]["started_at"]),
        "Actual OOS attempt preceded validation completion/lock",
    )
    chronology = {
        "validation_finished_at": va[-1]["finished_at"],
        "oos_started_at": oo[-1]["started_at"],
        "winner_locked_before_oos": True,
    }
    return dict(
        status="pass",
        stage="historical-oos",
        winner_id=locked,
        winner_unchanged_all_four_costs=True,
        seal_digests={
            "registration_payload_sha256": base.canonical_hash(registration),
            "data_freeze_payload_sha256": base.canonical_hash(freeze),
            "historical_oos_stage_payload_sha256": base.canonical_hash(seal),
            "winner_payload_sha256": validation["winner_payload_sha256"],
        },
        chronology=chronology,
        commission_bps=1,
        slippage_bps=[1, 3, 10, 25],
        same_daily_clocks_and_cash_reference=True,
        same_decisions_across_costs=True,
        selection_after_evaluation=False,
        scenarios=summaries,
        scope="CURRENT already-sealed winner only; no parameter fitting or strategy execution performed by audit.",
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["validation", "oos"])
    p.add_argument("--study", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--benchmark-root")
    p.add_argument("--output")
    args = p.parse_args(argv)
    report = (
        validation_audit(args.study, args.config, args.benchmark_root)
        if args.stage == "validation"
        else oos_audit(args.study, args.config)
    )
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        out = Path(args.output).resolve()
        require(
            not out.is_relative_to(Path(args.study).resolve() / "stages"),
            "Audit output must not modify sealed stages",
        )
        require(not out.exists(), "Audit report already exists")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
