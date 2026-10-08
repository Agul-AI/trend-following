#!/usr/bin/env python3
"""Independently audit one sealed QQQ split-v3 result, not a generic backtester.

Requires the private study package and its licensed raw data. This script cannot
reproduce market results from the public repository alone. It reads but never
modifies the study; output must be a NEW directory outside the study tree.

Only the manifest/selection and costs of the published October 6, 2026 v3 run are
supported. Pretax paths use separately written arithmetic, importing no project
numerical helpers. Hypothetical taxes receive structural reconciliation only,
not an independent tax-lot engine or tax-compliance certification. This auditor
was added AFTER the numerical freeze and is not part of that frozen closure.

Example (paths supplied by the user):
    python scripts/audit_qqq_split_v3_result.py --study STUDY_DIR --output NEW_AUDIT_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXPECTED_MANIFEST_SHA256 = "9e8aad4c77c87d4f55d2d79a96f00bd2ac3a2928656ab1e8c0d9869dd985e671"
EXPECTED_SELECTION_SHA256 = "562c815ec9c06295dfc4f6c92e232e967d79f95d71316527c82e14fe7845f1de"
EXPECTED_SPEC = {
    "name": "ma200_alloc_3_3",
    "kind": "ma",
    "asset": "synthetic_3x",
    "ma_sessions": 200,
    "allocation": 1.0,
}
EXPECTED_COSTS = {
    "fee_bps": 1.0,
    "slippage_bps": 5.0,
    "financing_spread_bps": 50.0,
    "expense_ratio": 0.0095,
    "extra_drag": 0.005,
    "short_tax_rate": 0.24,
    "long_tax_rate": 0.15,
    "interest_tax_rate": 0.24,
    "wash_sales": True,
    "financing_day_count": 360.0,
    "expense_day_count": 365.0,
    "cash_day_count": 365.0,
    "max_rate_age_days": 10.0,
}


def require(condition: bool, message: str) -> None:
    """Do not disable integrity checks when Python is run with -O."""
    if not condition:
        raise ValueError(message)


def file_snapshot(study: Path) -> dict[str, str]:
    """Read-only byte hashes also detect accidental changes during this audit."""
    return {
        str(path.relative_to(study)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(study.rglob("*"))
        if path.is_file()
    }


def validate_supported_contract(study: Path, manifest: dict, selection: dict) -> None:
    """Refuse changed selection, economics or chronology instead of guessing."""
    config = yaml.safe_load((study / "snapshot/configs/qqq_split_v3.yaml").read_text())
    require(
        manifest["study_id"] == "qqq_split_v3" and manifest["schema_version"] == 3,
        "Unsupported study schema",
    )
    require(
        selection["selected_id"] == EXPECTED_SPEC["name"]
        and selection["selected_spec"] == EXPECTED_SPEC,
        "Unsupported selected candidate",
    )
    require(
        selection["manifest_sha256"] == EXPECTED_MANIFEST_SHA256,
        "Selection belongs to another protocol",
    )
    require(
        selection["final_test_used"] is False
        and selection["last_price_available_to_selection"] == "2015-12-31",
        "Unexpected recorded selection boundary",
    )
    require(
        config["segments"]
        == {
            "development": {"start": "2002-01-01", "end": "2011-12-31"},
            "validation": {"start": "2012-01-01", "end": "2015-12-31"},
            "test": {"start": "2016-01-01", "end": "2026-06-01"},
        },
        "Unsupported chronological split",
    )
    for name, value in EXPECTED_COSTS.items():
        require(config["costs"][name] == value, f"Unsupported cost assumption: {name}")
    require(
        config["policy"]["initial_cash"] == 1000.0
        and config["policy"]["execution_delay_sessions"] == 1
        and config["policy"]["product_leverage"] == 3.0
        and config["policy"]["allocation_mode"] == "on_target_change_only_with_drift",
        "Unsupported capital, leverage, delay or rebalancing policy",
    )
    require(
        config["evaluation"]["annualization_sessions"] == 252
        and config["evaluation"]["cost_layers"]
        == ["gross", "trading", "financing", "expense", "tax"],
        "Unsupported evaluation convention",
    )


def run_audit(study: Path, output: Path) -> None:
    """Recompute the specified eight pretax paths and reconcile tax outputs."""
    STUDY, OUT = study.resolve(), output.resolve()
    require(STUDY.is_dir(), "Study directory does not exist")
    require(not OUT.exists(), "Output directory must not already exist")
    require(
        OUT != STUDY and STUDY not in OUT.parents and OUT not in STUDY.parents,
        "Output must be outside the study tree and not its ancestor",
    )
    study_hashes = file_snapshot(STUDY)
    EVAL = STUDY / "evaluation"
    checks = []

    def check(name, actual, expected, tol=1e-8):
        error = float(
            np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)))
        )
        checks.append(dict(name=name, pass_check=error <= tol, max_abs_error=error, tolerance=tol))
        if error > tol:
            raise AssertionError((name, error, tol))

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = json.loads((STUDY / "manifest.json").read_text())
    require(
        sha(STUDY / "manifest.json") == EXPECTED_MANIFEST_SHA256,
        "Only the specified sealed v3 study is supported",
    )
    require(
        sha(STUDY / "manifest.json") == (STUDY / "manifest.sha256").read_text().strip(),
        "Manifest checksum mismatch",
    )
    for rec in manifest["input_files"]:
        require(sha(STUDY / rec["path"]) == rec["sha256"], "Input checksum mismatch")
    for rec in manifest["code_files"]:
        require(
            sha(STUDY / "snapshot" / rec["path"]) == rec["sha256"], "Snapshot checksum mismatch"
        )
    completion = json.loads((EVAL / "completion.json").read_text())
    for rec in completion["files"]:
        require(sha(EVAL / rec["path"]) == rec["sha256"], "Output checksum mismatch")
    selection = json.loads((STUDY / "selection.json").read_text())
    require(
        sha(STUDY / "selection.json")
        == EXPECTED_SELECTION_SHA256
        == (STUDY / "selection.sha256").read_text().strip(),
        "Unsupported or changed selection",
    )
    validate_supported_contract(STUDY, manifest, selection)
    start_selection = pd.Timestamp(
        json.loads((STUDY / "selection_started.json").read_text())["started_at_utc"]
    )
    start_eval = pd.Timestamp(
        json.loads((STUDY / "evaluation_started.json").read_text())["started_at_utc"]
    )
    require(
        pd.Timestamp(manifest["frozen_at_utc"])
        < start_selection
        < pd.Timestamp(selection["selected_at_utc"])
        < start_eval,
        "Invalid freeze/selection/evaluation chronology",
    )
    raw = pd.read_parquet(STUDY / "inputs/QQQ_daily.parquet")
    dates = pd.DatetimeIndex(pd.to_datetime(raw.date)).normalize() + pd.Timedelta(hours=16)
    dates = dates.tz_localize("America/New_York")
    raw.index = dates
    mask = (dates >= pd.Timestamp("2016-01-01", tz="America/New_York")) & (
        dates < pd.Timestamp("2026-06-02", tz="America/New_York")
    )
    ix = np.flatnonzero(mask)
    testdates = dates[mask]
    N = len(ix)
    require(
        N == 2617
        and str(testdates[0].date()) == "2016-01-04"
        and str(testdates[-1].date()) == "2026-06-01",
        "Unsupported test dates/count",
    )
    price = raw.close.to_numpy(float)
    split = raw.split_coefficient.to_numpy(float)
    div = raw.dividend_amount.to_numpy(float)
    tr = np.zeros(len(raw))
    tr[1:] = split[1:] * (price[1:] + div[1:]) / price[:-1] - 1
    TR = 100 * np.cumprod(1 + tr)
    ma = pd.Series(TR).rolling(200, min_periods=200).mean().to_numpy()
    decision = (TR[ix] > ma[ix]).astype(float)
    target = np.r_[0, decision[:-1]]
    clock = dates.tz_convert("UTC").asi8 / 1e9
    elapsed = np.r_[0, np.diff(clock)] / 86400

    def integrate(filename, denominator, spread=0.0):
        obs = pd.read_csv(STUDY / "inputs" / filename)
        obs["available_at"] = pd.to_datetime(obs.available_at, utc=True)
        obs["observation_date"] = pd.to_datetime(obs.observation_date)
        obs = obs.sort_values(["available_at", "observation_date"]).drop_duplicates(
            "available_at", keep="last"
        )
        available = pd.DatetimeIndex(obs.available_at).asi8 / 1e9
        rate = obs.annual_rate.to_numpy() + spread
        out = np.zeros(len(clock))
        for i in range(1, len(clock)):
            left, right = clock[i - 1 : i + 1]
            cuts = np.r_[left, available[(available > left) & (available < right)], right]
            pos = np.searchsorted(available, cuts[:-1], side="right") - 1
            require(min(pos) >= 0, "Missing available rate")
            require(np.max(cuts[1:] - available[pos]) <= 10 * 86400, "Stale available rate")
            out[i] = np.dot(np.diff(cuts), rate[pos]) / (86400 * denominator)
        return out

    fund = integrate("DFF_available.csv", 360, 0.005)
    cr = integrate("DTB3_available.csv", 365)
    cashret = cr[ix].copy()
    cashret[0] = 0.0
    rf = cashret.copy()
    metrics = pd.read_csv(EVAL / "test_metrics.csv")
    independent = []
    qaq = []
    for name, asset in [("ma200_alloc_3_3", "synthetic_3x"), ("qqq_buy_hold", "QQQ")]:
        folder = EVAL / "paths" / name
        targets = target if asset == "synthetic_3x" else np.r_[0, np.ones(N - 1)]
        exported_target = pd.read_csv(folder / "targets.csv").iloc[:, 1].to_numpy()
        check(name + " targets", targets, exported_target, 0)
        finals = json.loads((folder / "terminal.json").read_text())
        exported_daily = pd.read_csv(folder / "daily_returns.csv", index_col=0)
        for layer in ["gross", "trading", "financing", "expense"]:
            c = 0 if layer == "gross" else 0.0006
            synthetic_returns = 3 * tr.copy()
            if layer in ["financing", "expense"]:
                synthetic_returns -= 2 * fund
            if layer == "expense":
                synthetic_returns -= 0.0145 * elapsed / 365
            require(
                np.isfinite(synthetic_returns).all() and np.all(synthetic_returns > -1),
                "Nonpositive or nonfinite synthetic NAV",
            )
            all_marks = (
                100 * np.cumprod(1 + synthetic_returns) if asset == "synthetic_3x" else price
            )
            marks = all_marks[ix]
            C = 1000.0
            q = 0.0
            previous_target = None
            equities = []
            cashes = []
            qtys = []
            daily = []
            turnover = []
            trades = []
            fee = 0.0
            slippage = 0.0
            prior = 1000.0
            for j, i in enumerate(ix):
                interest = C * cashret[j]
                if asset == "QQQ":
                    q *= split[i]
                    C += q * div[i]
                C += interest
                rebalance = (
                    asset == "QQQ" or previous_target is None or targets[j] != previous_target
                )
                traded = 0.0
                if rebalance:
                    wealth = C + q * marks[j]
                    desired = targets[j] * wealth
                    delta = desired - q * marks[j]
                    if delta < -1e-11:
                        notional = -delta
                        q -= notional / marks[j]
                        C += notional * (1 - c)
                        traded += notional
                        trades.append((j, "sell", notional))
                    elif delta > 1e-11:
                        notional = min(delta, max(0.0, C) / (1 + c))
                        q += notional / marks[j]
                        C -= notional * (1 + c)
                        traded += notional
                        trades.append((j, "buy", notional))
                    if abs(q) < 1e-11:
                        q = 0.0
                    if abs(C) < 1e-10:
                        C = 0.0
                previous_target = targets[j]
                eq = C + q * marks[j]
                equities.append(eq)
                cashes.append(C)
                qtys.append(q)
                daily.append(eq / prior - 1)
                turnover.append(traded / prior)
                if layer != "gross":
                    fee += traded * 0.0001
                    slippage += traded * 0.0005
                prior = eq
            exitnotional = q * marks[-1]
            terminal = C + exitnotional * (1 - c)
            fees = fee + (exitnotional * 0.0001 if layer != "gross" else 0)
            slips = slippage + (exitnotional * 0.0005 if layer != "gross" else 0)
            account = pd.read_csv(folder / f"{layer}_account.csv", index_col=0)
            hold = pd.read_parquet(folder / f"{layer}_holdings.parquet")
            events = pd.read_csv(folder / f"{layer}_events.csv")
            orders = events[events.event.isin(["buy", "sell"])]
            check(name + " " + layer + " cash", cashes, account.cash, 1e-7)
            check(name + " " + layer + " equity", equities, account.equity, 1e-7)
            check(name + " " + layer + " quantity", qtys, hold.quantity, 1e-7)
            check(name + " " + layer + " raw_marks", marks, hold.price, 1e-7)
            check(name + " " + layer + " returns", daily, account.returns, 1e-11)
            check(name + " " + layer + " turnover", turnover, account.turnover, 1e-10)
            check(
                name + " " + layer + " terminal", terminal, finals[layer]["liquidated_equity"], 1e-7
            )
            check(
                name + " " + layer + " firstbuyindex",
                trades[0][0],
                np.where(
                    pd.to_datetime(account.index, utc=True)
                    == pd.to_datetime(
                        orders.loc[orders.event.eq("buy"), "timestamp"].iloc[0], utc=True
                    )
                )[0][0],
                0,
            )
            d = np.array(daily)
            d[-1] = (1 + d[-1]) * terminal / equities[-1] - 1
            check(name + " " + layer + " liquidated_daily_returns", d, exported_daily[layer], 1e-11)
            comp = np.cumprod(1 + d)
            ex = d - rf
            dd = comp / np.maximum(np.maximum.accumulate(comp), 1) - 1
            calc = dict(
                CAGR=(terminal / 1000) ** (252 / N) - 1,
                excess_return_Sharpe=ex.mean() / ex.std(ddof=1) * np.sqrt(252),
                zero_RF_Sharpe=d.mean() / d.std(ddof=1) * np.sqrt(252),
                annualized_volatility=d.std(ddof=1) * np.sqrt(252),
                max_drawdown=dd.min(),
                terminal_equity=terminal,
                total_return=terminal / 1000 - 1,
                total_trading_fees=fees,
                total_slippage=slips,
                filled_orders=len(trades) + (exitnotional > 1e-11),
                turnover=sum(turnover) + exitnotional / equities[-1],
                mean_actual_weight=np.mean(np.array(qtys) * marks / equities),
            )
            reported = metrics[(metrics.candidate_id == name) & (metrics.layer == layer)].iloc[0]
            for key, val in calc.items():
                check(
                    name + " " + layer + " metric " + key,
                    val,
                    reported[key],
                    1e-7
                    if key in ["terminal_equity", "total_trading_fees", "total_slippage"]
                    else 1e-9,
                )
            firsttr = trades[0][0]
            entrycost = trades[0][2] * c
            entryfactor = equities[firsttr] / (equities[firsttr] + entrycost)
            terminalfactor = terminal / equities[-1]
            clean = np.array(daily)
            clean[firsttr] = (1 + clean[firsttr]) / entryfactor - 1
            check(
                name + " " + layer + " bootstrap_boundary_identity",
                np.prod(1 + clean) * entryfactor * terminalfactor,
                terminal / 1000,
                1e-9,
            )
            independent.append(
                dict(
                    candidate_id=name,
                    layer=layer,
                    **{k: float(v) for k, v in calc.items()},
                    first_fill_session=str(testdates[firsttr].date()),
                    source_sessions=N,
                )
            )
    # Real taxable outputs: structural reconciliation, NOT an independent reimplementation of the tax law/lot engine.
    for name in ["ma200_alloc_3_3", "qqq_buy_hold", "qqq200ma", "cash"]:
        folder = EVAL / "paths" / name
        final = json.loads((folder / "terminal.json").read_text())["tax"]
        acc = pd.read_csv(folder / "tax_account.csv", index_col=0)
        ev = pd.read_csv(folder / "tax_events.csv")
        hold = pd.read_parquet(folder / "tax_holdings.parquet")
        dr = pd.read_csv(folder / "daily_returns.csv", index_col=0)["tax"]
        orders = ev[ev.event.isin(["buy", "sell"])]
        r = metrics[(metrics.candidate_id == name) & metrics.layer.eq("tax")].iloc[0]
        check(
            name + " tax holdings_to_equity",
            hold.quantity * hold.price + acc.cash.to_numpy(),
            acc.equity,
            1e-7,
        )
        check(
            name + " tax terminal_cost_tax_identity",
            final["marked_equity"]
            - final["exit_fees"]
            - final["exit_slippage"]
            - final["terminal_taxes_paid"],
            final["liquidated_equity"],
            1e-7,
        )
        check(
            name + " tax return_product", np.prod(1 + dr) * 1000, final["liquidated_equity"], 1e-7
        )
        check(name + " tax fees", orders.fee.sum() + final["exit_fees"], r.total_trading_fees, 1e-9)
        check(
            name + " tax slippage",
            orders.slippage.sum() + final["exit_slippage"],
            r.total_slippage,
            1e-9,
        )
        taxcash = -ev.loc[ev.event.isin(["tax_payment", "tax_refund"]), "cash_flow"].sum()
        check(name + " tax sums", taxcash + final["terminal_taxes_paid"], r.total_taxes_paid, 1e-8)
        check(
            name + " tax account_sums",
            acc.taxes_paid.sum() + final["terminal_taxes_paid"],
            r.total_taxes_paid,
            1e-8,
        )
        check(
            name + " tax event_cash_reconciliation",
            1000 + ev.cash_flow.cumsum().iloc[-1] if len(ev) else 1000,
            acc.cash.iloc[-1],
            1e-7,
        )
        qaq.append(
            dict(
                candidate_id=name,
                terminal_equity=float(r.terminal_equity),
                total_taxes_paid=float(r.total_taxes_paid),
                tax_funding_orders=int(r.tax_funding_orders),
                all_structural_checks_pass=True,
            )
        )
    summary = {
        "audit_status": "PASS",
        "scope": "Independent new arithmetic on all 2617 daily test rows for selected strategy and QQQ gross/trading/financing/expense, with separate structural hypothetical-tax reconciliation. No project evaluator, rate integration helper, or generic ledger imported or called.",
        "manifest_sha256": sha(STUDY / "manifest.json"),
        "selection_sha256": sha(STUDY / "selection.json"),
        "frozen_at_utc": manifest["frozen_at_utc"],
        "selection_started_at_utc": start_selection.isoformat(),
        "evaluation_started_at_utc": start_eval.isoformat(),
        "bound_input_files_verified": len(manifest["input_files"]),
        "bound_snapshot_code_files_verified": len(manifest["code_files"]),
        "completed_output_files_verified": len(completion["files"]),
        "selected_id": "ma200_alloc_3_3",
        "test_sessions": N,
        "independent_pretax_paths": 8,
        "checks": checks,
        "independent_metrics": independent,
        "tax_reconciliation": qaq,
        "limitations": [
            "Historical test years were previously explored; fresh split is not untouched historical information.",
            "Selected full synthetic3x strategy and QQQ do not have matched risk; larger returns do not establish alpha.",
            "Synthetic economics and dividend/tax timing are declared hypothetical daily assumptions, not actual TQQQ investor returns.",
            "Tax structural reconciliation does not independently reproduce every real tax lot or certify tax compliance; separate artificial hand-tax fixtures passed.",
        ],
    }
    require(
        len(checks) == 210 and all(row["pass_check"] for row in checks),
        "Unexpected numerical check count or failure",
    )
    require(study_hashes == file_snapshot(STUDY), "Study files changed while audit ran")
    summary["auditor_scope"] = (
        "Exact sealed qqq_split_v3 run only; not a generic strategy evaluator"
    )
    summary["study_files_unchanged"] = True
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / "final_accounting_audit.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        "PASS",
        len(checks),
        "checks",
        len(independent),
        "independent paths",
        "hashes",
        len(manifest["input_files"]) + len(manifest["code_files"]) + len(completion["files"]),
    )
    for row in independent:
        print(
            row["candidate_id"],
            row["layer"],
            "CAGR",
            row["CAGR"],
            "DD",
            row["max_drawdown"],
            "exS",
            row["excess_return_Sharpe"],
            "terminal",
            row["terminal_equity"],
            "firstfill",
            row["first_fill_session"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--study", type=Path, required=True, help="Complete sealed private v3 study"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="NEW audit directory outside study"
    )
    args = parser.parse_args()
    try:
        run_audit(args.study, args.output)
    except (ValueError, AssertionError, KeyError, OSError) as error:
        parser.exit(1, f"Audit failed: {error}\n")


if __name__ == "__main__":
    main()
