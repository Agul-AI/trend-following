#!/usr/bin/env python3
"""Independent CURRENT-development result audit; no strategy execution/import.

Reads the explicitly provided current config, seals, development-only results,
and TIME/eligibility metadata. Never imports a trading kernel, opens another
study, parses raw price inputs, or evaluates any candidate. Optional curve
checks consume already-generated CURRENT benchmark/candidate daily returns.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FAMILIES = ("SMA", "EMA", "MACD")
MA_PERIODS = (10, 20, 50, 100, 200, 250)
MACD = ((6, 13, 5), (12, 26, 9), (24, 52, 18), (48, 104, 36), (96, 208, 72))
HOURS = tuple(x / 2 for x in range(14))
FLAG_COLUMNS = (
    "candidate_id",
    "post_gap_confirmed_buy_count",
    "post_gap_confirmed_sell_count",
    "post_gap_confirmed_signal_count",
    "has_post_gap_confirmed_signal",
)
EVENT_COLUMNS = ("candidate_id", "side", "decision_at", "gap_start", "gap_end", "window_end")
COUNT = 56644


class AuditFailure(ValueError):
    pass


def require(condition, message):
    if not bool(condition):
        raise AuditFailure(message)


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing/unsafe current artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def read_seal(path):
    file_hash(path)
    sealed = json.loads(Path(path).read_text())
    require(set(sealed) >= {"payload", "payload_sha256"}, f"Missing seal fields: {path}")
    require(canonical_hash(sealed["payload"]) == sealed["payload_sha256"], f"Seal mismatch: {path}")
    return sealed["payload"]


def csv(path):
    file_hash(path)
    return pd.read_csv(path, float_precision="round_trip")


def independently_expected_grid():
    rules = []
    for family in ("SMA", "EMA"):
        for period in MA_PERIODS:
            rules.append(
                (
                    f"{family.lower()}_{period}",
                    dict(family=family, period=period, fast=None, slow=None, signal=None),
                )
            )
    for fast, slow, signal in MACD:
        rules.append(
            (
                f"macd_{fast}_{slow}_{signal}",
                dict(family="MACD", period=None, fast=fast, slow=slow, signal=signal),
            )
        )
    table = {}
    for (entry, er), (exit_rule, xr), en, xn in itertools.product(rules, rules, HOURS, HOURS):
        e = f"{en:g}".replace(".", "p")
        x = f"{xn:g}".replace(".", "p")
        key = f"v5_30m__entry_{entry}__exit_{exit_rule}__entry_hours_{e}__exit_hours_{x}"
        table[key] = dict(
            candidate_id=key,
            entry_rule=er,
            exit_rule=xr,
            entry_confirmation_hours=en,
            exit_confirmation_hours=xn,
        )
    return table


def audit_contract(config, registration, config_path, project_root, verify_sources=True):
    require(
        config == registration["configuration"], "Current config differs from sealed registration"
    )
    require(
        registration["configuration_sha256"] == canonical_hash(config),
        "Config canonical hash mismatch",
    )
    require(
        registration["config_file_sha256"] == file_hash(config_path), "Config file hash mismatch"
    )
    g = config["grid"]
    a = config["accounting"]
    s = config["selection"]
    d = config["diagnostics"]
    require(
        config["security"] == "QQQ" and config["exposure"] == "unlevered_long_or_cash",
        "Wrong current instrument/exposure",
    )
    require(
        g["ma_periods"] == list(MA_PERIODS) and g["macd_parameters"] == [list(x) for x in MACD],
        "Rule grid changed",
    )
    require(
        g["entry_confirmation_hours"] == list(HOURS)
        and g["exit_confirmation_hours"] == list(HOURS),
        "Independent duration grid changed",
    )
    require(
        g["candidate_count"] == COUNT and g["independent_entry_exit_rules"] is True,
        "Candidate budget changed",
    )
    require(
        a["initial_cash"] == 1000
        and a["commission_bps_per_order"] == 1
        and a["baseline_slippage_bps"] == 3,
        "Wrong baseline capital/costs",
    )
    require(
        all(a[k] == 0 for k in ("tax_rate", "financing_rate", "extra_expense_ratio")),
        "Forbidden tax/financing/extra expense",
    )
    require(a["cash_series"] == "DTB3" and a["cash_day_count"] == 365, "Wrong cash reference")
    require(
        s["primary_metric"] == "cash_excess_sharpe"
        and s["annualization"] == 252
        and s["standard_deviation_ddof"] == 1,
        "Primary Sharpe definition changed",
    )
    require(
        s["tie_tolerance"] == 1e-10
        and s["tie_break"] == "lexicographically_smallest_stable_candidate_id",
        "Tie rule changed",
    )
    require(
        s["finite_only"] is True and s["max_finalist_count"] == 9,
        "Finite-only family-selection contract changed",
    )
    require(
        config["segments"]["development"] == {"start": "2001-01-01", "end": "2011-12-31"},
        "Audit is CURRENT development2001-2011 only",
    )
    require(
        d["enabled"] is True
        and d["event_limit"] == 2500000
        and d["descriptive_only"] is True
        and d["affects_selection"] is False,
        "Flags must be complete descriptive-only diagnostics",
    )
    require(
        config["quality_policy"]["profile"] == "observed_only_blackout_v1"
        and config["quality_policy"]["valuation_hour_ny"] == 17,
        "Held-gap/17NY profile changed",
    )
    expected = independently_expected_grid()
    grid = registration["grid"]
    require(
        len(grid) == COUNT and len({x["candidate_id"] for x in grid}) == COUNT,
        "Registered grid is duplicated/incomplete",
    )
    require(registration["grid_sha256"] == canonical_hash(grid), "Registered grid hash mismatch")
    require(
        {x["candidate_id"]: x for x in grid} == expected,
        "Registration differs from independent unrestricted Cartesian grid",
    )
    if verify_sources:
        require(
            len(registration["source_hashes"]) == 8,
            "Expected exactly eight frozen current implementation files",
        )
        for name, sha in registration["source_hashes"].items():
            require(
                file_hash(Path(project_root) / name) == sha,
                f"Frozen current source changed: {name}",
            )
    return expected


def numbers(frame, column):
    require(column in frame, f"Missing metric: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def bool_values(series, label):
    if series.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        return series.astype(bool)
    require(series.astype(str).isin(["True", "False"]).all(), f"Nonboolean {label}")
    return series.astype(str).eq("True")


def audit_metric_universe(metrics, expected, calendar, config):
    require(
        len(metrics) == COUNT and not metrics.candidate_id.duplicated().any(),
        "Development rows missing, duplicated, or filtered",
    )
    require(
        set(metrics.candidate_id) == set(expected),
        "Candidate membership differs from preregistration",
    )
    require(
        numbers(metrics, "commission_bps").eq(1).all()
        and numbers(metrics, "slippage_bps").eq(3).all(),
        "Nonbaseline cost scenario entered comparison",
    )
    require(
        metrics.quality_profile.eq("observed_only_blackout_v1").all(), "Wrong gap policy in results"
    )
    require(
        numbers(metrics, "daily_valuation_hour_ny").eq(17).all()
        and numbers(metrics, "sampling_minutes").eq(30).all(),
        "Wrong valuation/sampling clock",
    )
    for row in metrics.itertuples(index=False):
        e = expected[row.candidate_id]

        def rule_id(rule):
            return (
                f"macd_{rule['fast']}_{rule['slow']}_{rule['signal']}"
                if rule["family"] == "MACD"
                else f"{rule['family'].lower()}_{rule['period']}"
            )

        require(
            row.entry_family == e["entry_rule"]["family"]
            and row.exit_family == e["exit_rule"]["family"],
            "Mislabeled family",
        )
        require(
            row.entry_rule == rule_id(e["entry_rule"]) and row.exit_rule == rule_id(e["exit_rule"]),
            "Mislabeled indicator rule",
        )
        require(
            row.entry_confirmation_hours == e["entry_confirmation_hours"]
            and row.exit_confirmation_hours == e["exit_confirmation_hours"],
            "Mislabeled independent confirmation duration",
        )
        require(
            row.entry_required_checks == 1 + 2 * row.entry_confirmation_hours
            and row.exit_required_checks == 1 + 2 * row.exit_confirmation_hours,
            "Confirmation check mapping changed",
        )
    c = calendar.copy()
    c["session"] = c.session.astype(str)
    c = c[c.session.between("2001-01-01", "2011-12-31")].sort_values("session")
    require(
        not c.empty and not c.session.duplicated().any(),
        "Missing/duplicate current development calendar times",
    )
    first = pd.Timestamp(c.market_open.iloc[0])
    end = pd.Timestamp(c.session.iloc[-1] + " 17:00", tz="America/New_York").tz_convert("UTC")
    require(first.tzinfo is not None, "Naive first-open clock")
    years = (end - first).total_seconds() / (365 * 86400)
    require(
        metrics.start_session.eq(c.session.iloc[0]).all()
        and metrics.end_session.eq(c.session.iloc[-1]).all(),
        "Scored start/end dates differ among candidates",
    )
    require(
        numbers(metrics, "scored_sessions").eq(len(c)).all(),
        "Scored session count differs among candidates",
    )
    terminal = numbers(metrics, "terminal_equity")
    gross = numbers(metrics, "gross_terminal_equity")
    require(np.isfinite(terminal).all() and terminal.gt(0).all(), "Nonpositive/nonfinite wealth")
    require(
        np.allclose(numbers(metrics, "total_return"), terminal / 1000 - 1, rtol=1e-10, atol=1e-12),
        "Terminal-return reconciliation failed",
    )
    require(
        np.allclose(
            numbers(metrics, "cagr"), (terminal / 1000) ** (1 / years) - 1, rtol=1e-10, atol=1e-12
        ),
        "ACT365 first-open-to17 CAGR mismatch",
    )
    require(
        np.allclose(
            numbers(metrics, "cost_drag_return"), (gross - terminal) / 1000, rtol=1e-10, atol=1e-12
        ),
        "Gross/net cost-drag reconciliation failed",
    )
    require((gross + 1e-9 >= terminal).all(), "Gross wealth below positive-cost net wealth")
    require(numbers(metrics, "max_drawdown").between(-1, 0).all(), "Invalid daily drawdown")
    require(numbers(metrics, "exposure_fraction").between(0, 1).all(), "Invalid unlevered exposure")
    require(numbers(metrics, "annualized_volatility").ge(0).all(), "Invalid volatility")
    for name in (
        "trade_count",
        "order_count",
        "cancelled_blackout_order_count",
        "cancelled_terminal_order_count",
    ):
        v = numbers(metrics, name)
        require(
            np.isfinite(v).all() and v.ge(0).all() and v.eq(np.floor(v)).all(), f"Invalid {name}"
        )
    require(
        numbers(metrics, "order_count").eq(2 * numbers(metrics, "trade_count")).all(),
        "Fresh-flat/terminal-flat roundtrip order counts inconsistent",
    )
    require(
        np.allclose(
            numbers(metrics, "annualized_turnover"),
            numbers(metrics, "total_turnover") / years,
            rtol=1e-10,
            atol=1e-12,
        ),
        "Annualized turnover mismatch",
    )
    expected_turnover = numbers(metrics, "trade_count") * (1 + 1 / (1.0003 * 1.0001))
    require(
        np.allclose(numbers(metrics, "total_turnover"), expected_turnover, rtol=1e-9, atol=1e-7),
        "All-in raw-price turnover/cost normalization inconsistent",
    )
    if "cash_excess_Sharpe" in metrics:
        require(
            np.allclose(
                numbers(metrics, "cash_excess_sharpe"),
                numbers(metrics, "cash_excess_Sharpe"),
                equal_nan=True,
            ),
            "Sharpe aliases disagree",
        )
    sizes = metrics.groupby(["entry_family", "exit_family"]).size()
    for e in FAMILIES:
        for x in FAMILIES:
            require(
                sizes[(e, x)] == (5 if e == "MACD" else 6) * (5 if x == "MACD" else 6) * 196,
                "Family grid cell was filtered",
            )
    return dict(
        scored_sessions=len(c),
        start_session=c.session.iloc[0],
        end_session=c.session.iloc[-1],
        elapsed_years=years,
        finite_primary_scores=int(np.isfinite(numbers(metrics, "cash_excess_sharpe")).sum()),
        undefined_primary_scores=int(numbers(metrics, "cash_excess_sharpe").isna().sum()),
    )


def independently_select_finalists(metrics, tolerance=1e-10):
    selected = []
    empty = []
    for e in FAMILIES:
        for x in FAMILIES:
            cell = metrics[(metrics.entry_family == e) & (metrics.exit_family == x)]
            finite = cell[np.isfinite(pd.to_numeric(cell.cash_excess_sharpe, errors="coerce"))]
            if finite.empty:
                empty.append(f"{e}/{x}")
                continue
            best = finite.cash_excess_sharpe.max()
            selected.append(
                finite.loc[finite.cash_excess_sharpe >= best - tolerance, "candidate_id"].min()
            )
    require(bool(selected), "All primary scores undefined; no sealed default finalist is permitted")
    return selected, empty


def audit_gap_diagnostics(metrics, flags, events, metadata, calendar, coverage, blackouts):
    require(tuple(events.columns) == EVENT_COLUMNS, "Event table contains unexpected/price fields")
    require(
        not flags.candidate_id.duplicated().any()
        and set(flags.candidate_id) == set(metrics.candidate_id),
        "Flags not retained for all candidates",
    )
    require(
        set(events.candidate_id).issubset(set(metrics.candidate_id)),
        "Event references an unregistered candidate",
    )
    require(
        metadata["enabled"] is True
        and metadata["event_limit"] == 2500000
        and metadata["truncated"] is False,
        "Truncated/disabled diagnostics cannot be sealed",
    )
    require(
        metadata["total_event_rows"] == metadata["stored_event_rows"] == len(events),
        "Retained association count mismatch",
    )
    require(
        metadata["descriptive_only"] is True
        and metadata["used_for_filtering_ranking_or_selection"] is False,
        "Flags used as a selection filter",
    )
    for frame in (metrics, flags):
        b = numbers(frame, "post_gap_confirmed_buy_count")
        s = numbers(frame, "post_gap_confirmed_sell_count")
        total = numbers(frame, "post_gap_confirmed_signal_count")
        require(
            np.isfinite(b).all()
            and np.isfinite(s).all()
            and b.ge(0).all()
            and s.ge(0).all()
            and b.eq(np.floor(b)).all()
            and s.eq(np.floor(s)).all(),
            "Invalid diagnostic count",
        )
        require(
            (b + s).eq(total).all()
            and bool_values(frame.has_post_gap_confirmed_signal, "flag").eq(total.gt(0)).all(),
            "Count/boolean disagreement",
        )
    a = metrics.set_index("candidate_id")
    f = flags.set_index("candidate_id").reindex(a.index)
    for column in FLAG_COLUMNS[1:]:
        require(
            a[column].equals(f[column]), "Metric flags differ from complete retained flags file"
        )
    for col in EVENT_COLUMNS[2:]:
        parsed = pd.to_datetime(events[col], utc=True, errors="raise")
        require(not parsed.isna().any(), "Missing event clock")
        events[col] = parsed
    require(events.side.isin(["buy", "sell"]).all(), "Non-confirmed side in diagnostic table")
    require(
        not events.duplicated(list(EVENT_COLUMNS)).any(), "Duplicated decision-to-gap association"
    )
    require(
        events.decision_at.gt(events.gap_end).all()
        and events.decision_at.le(events.window_end).all(),
        "Event outside positive-time inclusive-close window",
    )
    cal = calendar.copy()
    cal["session"] = cal.session.astype(str)
    cal["market_close"] = pd.to_datetime(cal.market_close, utc=True)
    unknown = blackouts[blackouts.kind.eq("unknown_missing")].copy()
    triples = set()
    for row in unknown.itertuples(index=False):
        gs = pd.Timestamp(row.start).tz_convert("UTC")
        ge = pd.Timestamp(row.end).tz_convert("UTC")
        day = ge.tz_convert("America/New_York").strftime("%Y-%m-%d")
        nxt = cal[cal.session.gt(day)].sort_values("session")
        require(
            not nxt.empty,
            "Audit requires explicit next-session calendar time for CURRENT internal development windows",
        )
        triples.add((gs, ge, nxt.market_close.iloc[0]))
    require(
        all(
            (r.gap_start, r.gap_end, r.window_end) in triples
            for r in events.itertuples(index=False)
        ),
        "Event associated with nonexistent/known-halt/wrong-next-session window",
    )
    safe = coverage[bool_values(coverage.signal_eligible, "coverage signal")].copy()
    safe_ends = set(pd.to_datetime(safe.bar_end, utc=True))
    require(
        events.decision_at.isin(safe_ends).all(),
        "Event did not occur at a safe observed completed30m close",
    )
    local = events.decision_at.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    require(local.between("2001-01-01", "2011-12-31").all(), "Nondevelopment diagnostic event")
    distinct = events.drop_duplicates(["candidate_id", "side", "decision_at"])
    require(
        not distinct.duplicated(["candidate_id", "decision_at"]).any(),
        "Same candidate confirmed both sides at one close",
    )
    counts = distinct.groupby(["candidate_id", "side"]).size()
    for row in flags.itertuples(index=False):
        require(
            row.post_gap_confirmed_buy_count == int(counts.get((row.candidate_id, "buy"), 0))
            and row.post_gap_confirmed_sell_count == int(counts.get((row.candidate_id, "sell"), 0)),
            "Distinct retained events disagree with candidate counts",
        )
    require(
        metadata["total_confirmed_decisions"] == len(distinct),
        "Global distinct decision metadata mismatch",
    )
    return dict(
        flagged_candidates=int(bool_values(flags.has_post_gap_confirmed_signal, "flag").sum()),
        event_associations=len(events),
        distinct_confirmed_decisions=len(distinct),
        unknown_windows=len(triples),
        truncated=False,
    )


def audit_curve(daily, metric, calendar):
    """Recompute CURRENT persisted curve summaries, never regenerate a strategy."""
    ret = pd.to_numeric(daily["return"], errors="raise")
    eq = pd.to_numeric(daily.equity, errors="raise")
    cash = pd.to_numeric(daily.cash_return, errors="raise")
    require(
        np.isfinite(ret).all() and np.isfinite(eq).all() and np.isfinite(cash).all(),
        "Invalid persisted curve",
    )
    require("timestamp" in daily, "Persisted curve lacks explicit valuation timestamps")
    stamps = pd.DatetimeIndex(pd.to_datetime(daily.timestamp, utc=True))
    cal = calendar.copy()
    cal["session"] = cal.session.astype(str)
    cal = cal[cal.session.between("2001-01-01", "2011-12-31")].sort_values("session")
    expected = pd.DatetimeIndex(
        [
            pd.Timestamp(day + " 17:00", tz="America/New_York").tz_convert("UTC")
            for day in cal.session
        ]
    )
    require(
        stamps.equals(expected),
        "Benchmark/candidate daily curve clock/period differs from current development",
    )
    require(
        str(metric["start_session"]) == cal.session.iloc[0]
        and str(metric["end_session"]) == cal.session.iloc[-1]
        and int(metric["scored_sessions"]) == len(cal),
        "Persisted curve metric period/session count differs",
    )
    years = (stamps[-1] - pd.Timestamp(cal.market_open.iloc[0])).total_seconds() / (365 * 86400)
    require(
        np.isclose(
            (eq.iloc[-1] / 1000) ** (1 / years) - 1, float(metric["cagr"]), rtol=1e-10, atol=1e-12
        ),
        "Persisted curve ACT365 CAGR mismatch",
    )
    previous = np.r_[1000.0, eq.to_numpy()[:-1]]
    require(
        np.allclose(ret, eq / previous - 1, rtol=1e-10, atol=1e-12), "Daily equity/return mismatch"
    )
    excess = ret - cash
    sd = excess.std(ddof=1)
    sharpe = np.sqrt(252) * excess.mean() / sd if sd > 0 else np.nan
    require(
        np.isclose(
            sharpe, float(metric["cash_excess_sharpe"]), equal_nan=True, rtol=1e-10, atol=1e-12
        ),
        "Daily cash-excess Sharpe/ddof mismatch",
    )
    vol = ret.std(ddof=1) * np.sqrt(252)
    require(
        np.isclose(
            vol, float(metric["annualized_volatility"]), equal_nan=True, rtol=1e-10, atol=1e-12
        ),
        "Daily volatility/ddof mismatch",
    )
    wealth = np.r_[1000.0, eq.to_numpy()]
    dd = (wealth / np.maximum.accumulate(wealth) - 1).min()
    require(
        np.isclose(dd, float(metric["max_drawdown"]), rtol=1e-10, atol=1e-12),
        "Daily drawdown/terminal mismatch",
    )
    require(
        np.isclose(eq.iloc[-1], float(metric["terminal_equity"]), rtol=1e-10, atol=1e-9),
        "Curve terminal equity mismatch",
    )
    require(
        np.isclose(
            1000 * np.prod(1 + cash), float(metric["cash_terminal_equity"]), rtol=1e-10, atol=1e-9
        ),
        "Cash-reference compounding mismatch",
    )
    return dict(
        curve_rows=len(daily),
        cash_excess_sharpe=None if np.isnan(sharpe) else float(sharpe),
        annualization=252,
        ddof=1,
    )


def audit_current_development(study, config_path, project_root=None, curve_dirs=()):
    study = Path(study).resolve()
    config_path = Path(config_path).resolve()
    root = Path(project_root or config_path.parent.parent).resolve()
    config = yaml.safe_load(config_path.read_text())
    registration = read_seal(study / "registration.json")
    expected = audit_contract(config, registration, config_path, root)
    freeze = read_seal(study / "prepared/freeze.json")
    stage = study / "stages/development"
    summary = read_seal(stage / "stage_seal.json")
    require(
        summary["stage"] == "development" and summary["candidate_count"] == COUNT,
        "Wrong completed stage",
    )
    require(
        summary["registration_sha256"] == canonical_hash(registration)
        and summary["data_freeze_sha256"] == canonical_hash(freeze),
        "Stage registration/data binding mismatch",
    )
    for name, sha in summary["artifact_hashes"].items():
        require(
            file_hash(stage / name) == sha, f"Sealed current development artifact changed: {name}"
        )
    actual = {
        str(p.relative_to(stage))
        for p in stage.rglob("*")
        if p.is_file() and p.name != "stage_seal.json"
    }
    require(actual == set(summary["artifact_hashes"]), "Sealed current artifact set changed")
    metrics = csv(stage / "metrics.csv")
    calendar = csv(study / "prepared/development/calendar.csv")
    population = audit_metric_universe(metrics, expected, calendar, config)
    flags = csv(stage / "post_gap_candidate_flags.csv")
    events = csv(stage / "post_gap_signal_events.csv")
    metadata = json.loads((stage / "post_gap_diagnostic_metadata.json").read_text())
    coverage = csv(study / "prepared/development/coverage.csv")
    blackouts = csv(study / "prepared/development/blackouts.csv")
    diagnostics = audit_gap_diagnostics(
        metrics, flags, events, metadata, calendar, coverage, blackouts
    )
    selected, undefined = independently_select_finalists(
        metrics, config["selection"]["tie_tolerance"]
    )
    finalists = read_seal(stage / "finalists.json")
    require(
        finalists["development_metrics_sha256"] == file_hash(stage / "metrics.csv"),
        "Finalist source metrics binding mismatch",
    )
    require(
        summary["finalists_sha256"] == canonical_hash(finalists), "Finalist seal not bound to stage"
    )
    ids = [row["candidate_id"] for row in finalists["candidates"]]
    require(
        ids == selected and len(ids) == summary["finalist_count"],
        "Sealed finalists differ from independent primary-only unfiltered selection",
    )
    require(
        all(row == expected[row["candidate_id"]] for row in finalists["candidates"]),
        "Finalist parameters differ from grid",
    )
    curves = {}
    for directory in curve_dirs:
        directory = Path(directory).resolve()
        m = csv(directory / "metrics.csv")
        d = csv(directory / "daily.csv")
        require(
            len(m) == 1
            and float(m.commission_bps.iloc[0]) == 1
            and float(m.slippage_bps.iloc[0]) == 3,
            "Current benchmark curve costs differ",
        )
        curves[str(directory)] = audit_curve(d, m.iloc[0], calendar)
    return dict(
        status="pass",
        stage="development",
        study=str(study),
        candidate_count=COUNT,
        no_gap_flag_filtering=True,
        selection="finite_cash_excess_sharpe_only_then_ID_within_1e-10",
        family_finalist_count=len(ids),
        family_finalists=ids,
        undefined_family_cells=undefined,
        population=population,
        diagnostics=diagnostics,
        persisted_curve_checks=curves,
        limitation="Grid does not persist all candidate daily curves; exact daily Sharpe/vol/DD recomputation is limited to supplied already-generated curves. No market strategy reruns performed.",
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--project-root")
    p.add_argument("--curve-dir", action="append", default=[])
    p.add_argument("--output")
    a = p.parse_args(argv)
    report = audit_current_development(a.study, a.config, a.project_root, a.curve_dir)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if a.output:
        target = Path(a.output).resolve()
        protected = Path(a.study).resolve() / "stages"
        require(
            not target.is_relative_to(protected),
            "Audit output must not alter sealed stage artifacts",
        )
        require(not target.exists(), "Independent audit output already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
