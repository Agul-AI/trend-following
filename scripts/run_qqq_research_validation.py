#!/usr/bin/env python
"""Run the frozen retrospective experiment; never claims untouched history.

No live broker or order submission. Existing reports are not overwritten.
Run a protocol freeze first. The final-code freeze and prospective ledger are
separate: a successful historical reconstruction cannot create a new holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trend_following.research_data import (
    audit_sessions,
    distribution_aware_panel,
    load_research_price,
)  # noqa:E402
from trend_following.research_evaluation import (
    ChronologicalFold,
    CostScenario,
    daily_returns,
    evaluate_path,  # noqa:E402
    matched_cash_benchmark,
    nested_schedule,
    research_metrics,
    waterfall_scenarios,
)
from trend_following.research_rates import rate_accrual  # noqa:E402

ROOT = Path(__file__).resolve().parents[1]


def export_run(output: Path, label: str, path, ledger, metrics: dict) -> None:
    directory = output / "paths" / label
    directory.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(
        {
            "return": ledger.returns,
            "equity": ledger.equity,
            "cash": ledger.cash,
            "turnover": ledger.turnover,
            "taxes_paid": ledger.taxes_paid,
            "cash_weight": ledger.cash_weight,
        }
    ).to_csv(directory / "account.csv")
    ledger.events.to_csv(directory / "ledger_events.csv", index=False)
    ledger.holdings.to_parquet(directory / "holdings.parquet", index=False)
    path.events.to_csv(directory / "execution_events.csv", index=False)
    path.decisions.to_parquet(directory / "decisions.parquet")
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in metrics.items()
            },
            indent=2,
            default=str,
            allow_nan=False,
        )
        + "\n"
    )


def fold_metrics(ledger, cash, folds, track, scenario):
    rows = []
    for fold in folds:
        dates = ledger.returns.index[
            (ledger.returns.index >= fold.test_start) & (ledger.returns.index <= fold.test_end)
        ]
        if not len(dates):
            continue
        metrics = research_metrics(
            ledger.returns.reindex(dates),
            (cash.reindex(dates).where(dates != ledger.returns.index[0], 0.0)),
            turnover=ledger.turnover.reindex(dates),
        )
        rows.append(
            dict(
                metrics,
                track=track,
                scenario=scenario,
                fold=fold.name,
                test_start=dates[0],
                test_end=dates[-1],
                accounting="continuous_account_slice_no_fold_liquidation",
                label="retrospective_not_independent_oos",
            )
        )
    return rows


def run(config_path: Path, rate_dir: Path, output: Path, manifest_path: Path):
    config = yaml.safe_load(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    from trend_following.research_protocol import sha256_file, verify_frozen_inputs

    verify_frozen_inputs(ROOT, manifest_path)
    if sha256_file(config_path) != manifest["protocol_sha256"]:
        raise ValueError("runtime config is not the frozen protocol")
    records = {str((ROOT / r["path"]).resolve()): r["sha256"] for r in manifest["data_files"]}
    required_prices = [
        ROOT / "data/raw/alpha_vantage_60min/QQQ.parquet",
        ROOT / "data/raw/alpha_vantage_60min/TQQQ.parquet",
        ROOT / "data/raw/synthetic_3x_60min/QQQ_3X_CALC.parquet",
        ROOT / "data/raw/alpha_vantage_daily_adjusted/QQQ.parquet",
        ROOT / "data/raw/alpha_vantage_daily_adjusted/TQQQ.parquet",
    ]
    for file in required_prices:
        if records.get(str(file.resolve())) != sha256_file(file):
            raise ValueError(f"runtime price input missing or differs from frozen files: {file}")
    for name in ["DFF_available.csv", "DTB3_available.csv", "metadata.json"]:
        actual = (rate_dir / name).resolve()
        if records.get(str(actual)) != sha256_file(actual):
            raise ValueError("runtime rate inputs are not the frozen files")
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.yaml").write_text(config_path.read_text())
    (output / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    qqq = load_research_price(ROOT / "data/raw/alpha_vantage_60min/QQQ.parquet")
    risk = load_research_price(ROOT / "data/raw/synthetic_3x_60min/QQQ_3X_CALC.parquet")
    dates = qqq.index.intersection(risk.index)
    if len(dates) != len(qqq):
        raise ValueError("QQQ and synthetic history must align; no silent drop of bars")
    qqq = qqq.reindex(dates)
    risk = risk.reindex(dates)
    audit_sessions(qqq).to_csv(output / "session_audit.csv", index=False)
    financing_obs = pd.read_csv(
        rate_dir / "DFF_available.csv", parse_dates=["available_at", "observation_date"]
    )
    cash_obs = pd.read_csv(
        rate_dir / "DTB3_available.csv", parse_dates=["available_at", "observation_date"]
    )
    financing = rate_accrual(dates, financing_obs, day_count=360, max_age_days=10)
    cash = rate_accrual(dates, cash_obs, day_count=365, max_age_days=10)
    pd.DataFrame({"financing_base_return": financing, "cash_return": cash}).to_csv(
        output / "rate_accrual.csv"
    )
    c = config["costs"]
    base = CostScenario(
        "primary",
        fee_bps=c["transaction_cost_bps"],
        slippage_bps=c["slippage_bps"],
        financing_spread_bps=c["financing_spread_bps"],
        expense_ratio=c["synthetic_fund_expense_annual"],
        extra_drag=c["synthetic_extra_tracking_drag_annual"],
    )
    taxable = replace(
        base, short_tax_rate=0.24, long_tax_rate=0.15, interest_tax_rate=0.24, wash_sales=True
    )
    candidates = config["candidates"]
    for name, spec in candidates.items():
        spec.setdefault("name", name)
    selection = config["selection"]
    bars = config["strategy"]["legacy_bars_per_day"]
    train_start = pd.Timestamp(config["evaluation"]["train_start"], tz="America/New_York")
    folds = []
    for declared in config["evaluation"]["outer_folds"]:
        cutoff = pd.Timestamp(declared["train_end"], tz="America/New_York") + pd.Timedelta(days=1)
        lower = pd.Timestamp(declared["test_start"], tz="America/New_York")
        upper = pd.Timestamp(declared["test_end"], tz="America/New_York") + pd.Timedelta(days=1)
        train = dates[dates < cutoff]
        test = dates[(dates >= lower) & (dates < upper)]
        if len(train) and len(test):
            folds.append(
                ChronologicalFold(
                    f"{lower.year}_{min(upper.year - 1, dates[-1].year)}",
                    train[-1],
                    test[0],
                    test[-1],
                )
            )
    if not folds or any(a.test_end >= b.test_start for a, b in zip(folds, folds[1:], strict=False)):
        raise ValueError("empty or overlapping declared evaluation folds")
    union = pd.DatetimeIndex(
        np.concatenate(
            [dates[(dates >= f.test_start) & (dates <= f.test_end)].to_numpy() for f in folds]
        )
    )
    if len(union) != len(dates[dates >= folds[0].test_start]):
        raise ValueError("declared outer folds omit return intervals")
    options = dict(
        minimum_training_years=selection["minimum_training_years"],
        first_validation_year=selection["inner_first_validation_year"],
        minimum_inner_observations=selection["minimum_inner_observations"],
        minimum_complete_validation_years=selection["minimum_complete_validation_years"],
        max_drawdown_limit=selection["max_inner_drawdown"],
        bars_per_day=bars,
        tie_tolerance=selection["tie_tolerance"],
    )
    print("Building training-only selections...", flush=True)
    schedule, selections, inner = nested_schedule(
        qqq, risk, financing, cash, candidates, folds, base, **options
    )
    selections.to_csv(output / "fold_selections.csv", index=False)
    inner.to_csv(output / "inner_validation.csv", index=False)
    rows = []
    foldrows = []
    tracks = {"fixed_b5": None, "train_selected_policy": schedule}
    start = folds[0].test_start
    for track, track_schedule in tracks.items():
        print("Evaluating", track, flush=True)
        # Fixed-target attribution holds primary pre-tax target allocations fixed;
        # tax-funding sales and cash-dependent share quantities remain ledger events.
        primary_path, _, _ = evaluate_path(
            qqq,
            risk,
            financing,
            cash,
            candidates["b5"],
            base,
            start=start,
            schedule=track_schedule,
            bars_per_day=bars,
        )
        for scenario in waterfall_scenarios(taxable):
            for attribution in ["end_to_end", "fixed_target_path"]:
                path, ledger, metrics = evaluate_path(
                    qqq,
                    risk,
                    financing,
                    cash,
                    candidates["b5"],
                    scenario,
                    start=start,
                    schedule=track_schedule,
                    bars_per_day=bars,
                    fixed_weights=primary_path.weights_at_close
                    if attribution == "fixed_target_path"
                    else None,
                )
                metrics.update(
                    track=track,
                    attribution=attribution,
                    tax_model="hypothetical_FIFO_ST24_LT15_cash24_same_security_wash_sales"
                    if scenario.short_tax_rate
                    else "none",
                    order_path_source="primary_net_cost_pretax_targets"
                    if attribution == "fixed_target_path"
                    else "scenario_specific",
                    price_tax_limitation="adjusted_total_return_proxy_no_separate_distributions",
                    protocol_id=manifest.get("protocol_id", config["protocol_id"]),
                )
                label = f"{track}__{scenario.name}__{attribution}"
                export_run(
                    output,
                    label,
                    primary_path if attribution == "fixed_target_path" else path,
                    ledger,
                    metrics,
                )
                rows.append(metrics)
                rf = matched_cash_benchmark(cash, ledger.returns.index, scenario.interest_tax_rate)
                foldrows.extend(
                    fold_metrics(ledger, rf, folds, track, scenario.name + "__" + attribution)
                )
    print("Matched benchmarks...", flush=True)
    for name, spec, price, multiplier in [
        ("cash", {"kind": "cash"}, qqq, 1),
        (
            "QQQ_buy_hold",
            {"kind": "buyhold", "product_leverage": 1, "effective_leverage": 1},
            qqq,
            1,
        ),
        ("QQQ_200MA_actual_vehicle", {"kind": "trend_1x", "product_leverage": 1}, qqq, 1),
        ("simple_1x_synthetic_sleeve", candidates["qqq200ma1x"], risk, 3),
        ("simple_2x_synthetic_sleeve", candidates["qqq200ma2x"], risk, 3),
        ("vol_state_anchor", candidates["anchor"], risk, 3),
    ]:
        for scenario in [base, taxable]:
            path, ledger, metrics = evaluate_path(
                qqq,
                price,
                financing,
                cash,
                spec,
                scenario,
                start=start,
                bars_per_day=bars,
                fund_costs_embedded=(multiplier == 1),
            )
            metrics.update(
                track=name, attribution="end_to_end", fund_costs_embedded=multiplier == 1
            )
            export_run(
                output,
                f"{name}__" + ("taxable" if scenario.short_tax_rate else "pretax"),
                path,
                ledger,
                metrics,
            )
            rows.append(metrics)
            foldrows.extend(
                fold_metrics(
                    ledger,
                    matched_cash_benchmark(cash, ledger.returns.index, scenario.interest_tax_rate),
                    folds,
                    name,
                    scenario.name + ("__taxable" if scenario.short_tax_rate else "__pretax"),
                )
            )
    # Risk matching uses only outer training observations and never test volatility.
    match_schedule = {}
    match_rows = []
    for fold in folds:
        _, training_ledger, _ = evaluate_path(
            qqq.loc[: fold.train_end],
            risk.loc[: fold.train_end],
            financing.loc[: fold.train_end],
            cash.loc[: fold.train_end],
            candidates["b5"],
            base,
            start=train_start,
            end=fold.train_end,
            bars_per_day=bars,
        )
        b5_daily = daily_returns(training_ledger.returns)
        q_daily = daily_returns(qqq.pct_change().fillna(0).reindex(training_ledger.returns.index))
        multiplier = min(1, float(q_daily.std() / b5_daily.std())) if b5_daily.std() > 0 else 0.0
        match_schedule[fold.test_start] = dict(candidates["b5"], allocation_multiplier=multiplier)
        match_rows.append(
            dict(
                fold=fold.name,
                train_end=fold.train_end,
                allocation_multiplier=multiplier,
                sizing="train_only_QQQ_vol_div_B5_vol_capped_at_1",
            )
        )
    pd.DataFrame(match_rows).to_csv(output / "risk_matching.csv", index=False)
    path, ledger, metrics = evaluate_path(
        qqq,
        risk,
        financing,
        cash,
        candidates["b5"],
        base,
        start=start,
        schedule=match_schedule,
        bars_per_day=bars,
    )
    metrics.update(track="train_risk_matched_b5", attribution="end_to_end")
    export_run(output, "train_risk_matched_b5", path, ledger, metrics)
    rows.append(metrics)
    foldrows.extend(
        fold_metrics(
            ledger,
            matched_cash_benchmark(cash, ledger.returns.index),
            folds,
            "train_risk_matched_b5",
            base.name,
        )
    )
    tqqq = load_research_price(ROOT / "data/raw/alpha_vantage_60min/TQQQ.parquet")
    common = qqq.index.intersection(tqqq.index)
    alignment_rows = []
    for stamp in qqq.index[(qqq.index >= common[0]) & (qqq.index <= common[-1])].difference(common):
        alignment_rows.append(
            dict(track="actual_TQQQ_overlap", timestamp=stamp, status="QQQ_bar_absent_in_TQQQ")
        )
    overlap_financing = rate_accrual(common, financing_obs, day_count=360, max_age_days=10)
    overlap_cash = rate_accrual(common, cash_obs, day_count=365, max_age_days=10)
    overlap_start = max(start, common[0])
    for name, spec in [
        ("actual_TQQQ_B5", candidates["b5"]),
        ("actual_TQQQ_buyhold", {"kind": "buyhold"}),
        (
            "QQQ_buy_hold_TQQQ_overlap",
            {"kind": "buyhold", "product_leverage": 1, "effective_leverage": 1},
        ),
    ]:
        for scenario in [base, taxable]:
            path, ledger, metrics = evaluate_path(
                qqq.reindex(common),
                qqq.reindex(common)
                if name == "QQQ_buy_hold_TQQQ_overlap"
                else tqqq.reindex(common),
                overlap_financing,
                overlap_cash,
                spec,
                scenario,
                start=overlap_start,
                bars_per_day=bars,
                fund_costs_embedded=True,
            )
            metrics.update(
                track=name,
                attribution="end_to_end",
                fund_costs_embedded=True,
                overlap_start=overlap_start,
                overlap_end=common[-1],
                alignment="intersection_only_missing_QQQ_bars_disclosed_indicator_clock_differs",
            )
            export_run(
                output,
                name + ("__taxable" if scenario.short_tax_rate else "__pretax"),
                path,
                ledger,
                metrics,
            )
            rows.append(metrics)
            foldrows.extend(
                fold_metrics(
                    ledger,
                    matched_cash_benchmark(
                        overlap_cash, ledger.returns.index, scenario.interest_tax_rate
                    ),
                    folds,
                    name,
                    scenario.name + ("__taxable" if scenario.short_tax_rate else "__pretax"),
                )
            )
    # Supplement adjusted-return proxy scenarios with distribution-aware actual
    # vehicles. Signals reference the adjusted total-return path; the executable
    # physical ledger uses unadjusted quotes and separately credited cash flows.
    corporate_info = []
    for ticker in ["QQQ", "TQQQ"]:
        physical, physical_ret, actions, info = distribution_aware_panel(
            ROOT / f"data/raw/alpha_vantage_60min/{ticker}.parquet",
            ROOT / f"data/raw/alpha_vantage_daily_adjusted/{ticker}.parquet",
        )
        common = qqq.index.intersection(physical.index)
        actual_adjusted = load_research_price(
            ROOT / f"data/raw/alpha_vantage_60min/{ticker}.parquet"
        ).reindex(common)
        physical_financing = rate_accrual(common, financing_obs, day_count=360, max_age_days=10)
        physical_cash = rate_accrual(common, cash_obs, day_count=365, max_age_days=10)
        spec = (
            candidates["b5"]
            if ticker == "TQQQ"
            else {"kind": "buyhold", "product_leverage": 1, "effective_leverage": 1}
        )
        for scenario in [base, taxable]:
            path, ledger, metrics = evaluate_path(
                qqq.reindex(common),
                actual_adjusted,
                physical_financing,
                physical_cash,
                spec,
                scenario,
                start=max(start, common[0]),
                bars_per_day=bars,
                fund_costs_embedded=True,
                physical_returns=physical_ret.reindex(common),
                physical_prices=physical.reindex(common),
                corporate_actions=actions,
            )
            track = f"{ticker}_distribution_aware"
            metrics.update(
                track=track,
                attribution="end_to_end",
                fund_costs_embedded=True,
                distribution_tax_character="ordinary_hypothetical",
                distribution_cash_timing="ex_date_not_paydate",
                trade_reference="adjusted_total_return_fill_quote_not_tax_lot_unrealized_gain",
                overlap_start=max(start, common[0]),
                overlap_end=common[-1],
            )
            export_run(
                output,
                track + ("__taxable" if scenario.short_tax_rate else "__pretax"),
                path,
                ledger,
                metrics,
            )
            rows.append(metrics)
            foldrows.extend(
                fold_metrics(
                    ledger,
                    matched_cash_benchmark(
                        physical_cash, ledger.returns.index, scenario.interest_tax_rate
                    ),
                    folds,
                    track,
                    scenario.name + ("__taxable" if scenario.short_tax_rate else "__pretax"),
                )
            )
        corporate_info.append(dict(info, ticker=ticker))
    (output / "corporate_action_provenance.json").write_text(
        json.dumps(corporate_info, indent=2) + "\n"
    )

    pd.DataFrame(alignment_rows, columns=["track", "timestamp", "status"]).to_csv(
        output / "alignment_exceptions.csv", index=False
    )

    # Locked schedule for selection-policy cost scenarios; no reselection on test outcomes.
    scenarios = []
    for slip in c["sensitivities"]["slippage_bps"]:
        scenarios.append(replace(base, name=f"slippage_{slip}", slippage_bps=slip))
    for spread in c["sensitivities"]["financing_spread_bps"]:
        scenarios.append(
            replace(base, name=f"financing_spread_{spread}", financing_spread_bps=spread)
        )
    for delay in c["sensitivities"]["extra_execution_delay_bars"]:
        scenarios.append(replace(base, name=f"delay_extra_{delay}", fill_delay_bars=1 + delay))
    scenarios += [replace(base, name="cash_0", cash_multiplier=0), taxable]
    stress = []
    for track, track_schedule in tracks.items():
        for scenario in scenarios:
            _, ledger, metrics = evaluate_path(
                qqq,
                risk,
                financing,
                cash,
                candidates["b5"],
                scenario,
                start=start,
                schedule=track_schedule,
                bars_per_day=bars,
            )
            stress.append(dict(metrics, track=track))
    table = pd.DataFrame(rows)
    for track in tracks:
        mask = table.track.eq(track)
        for attribution in ["end_to_end", "fixed_target_path"]:
            values = table.loc[mask & table.attribution.eq(attribution)]
            gross = values.loc[values.name.eq("gross")].iloc[0]
            table.loc[values.index, "CAGR_drag_vs_gross_pp"] = (gross.CAGR - values.CAGR) * 100
            table.loc[values.index, "wealth_drag_vs_gross"] = (
                gross.terminal_wealth - values.terminal_wealth
            )
    table.to_csv(output / "performance.csv", index=False)
    pd.DataFrame(foldrows).to_csv(output / "outer_fold_performance.csv", index=False)
    pd.DataFrame(stress).to_csv(output / "cost_sensitivity.csv", index=False)
    (output / "run_status.json").write_text(
        json.dumps(
            {
                "historical_validation": "complete",
                "bootstrap": "pending_separate_command",
                "prospective_evidence": "not_available",
                "resume_claim": "historical nested reconstruction only; no independent historical OOS",
                "known_limitations": [
                    "latest-vintage rate proxies",
                    "adjusted price tax scenarios omit separate fund distributions",
                    "synthetic and actual vehicles evaluated separately",
                    "fill-referenced execution differs from archived raw-overlay B5",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    summary = [
        "# Frozen QQQ retrospective validation",
        "",
        "Historical reconstruction only, not independent out-of-sample or live returns.",
        "Primary selector is pre-tax and net of trading/financing costs. Taxes are hypothetical scenarios.",
        "All previously examined history stays retrospective. No prospective performance exists yet.",
        "",
        "## Continuous accounts",
        table[
            [
                "track",
                "name",
                "attribution",
                "CAGR",
                "excess_return_Sharpe",
                "max_drawdown",
                "terminal_wealth",
            ]
        ].to_markdown(index=False),
        "",
        "## Selection by outer fold",
        selections.to_markdown(index=False),
        "",
        "## Limitations",
        "Latest-vintage rates with an assumed publication lag are not point-in-time vintages.",
        "Adjusted-price tax scenarios are not fully distribution-aware investor after-tax returns.",
        "Cost-free risk diagnostics from the execution engine are not authoritative portfolio exposures.",
        "Bootstrap and forward evidence are separate required artifacts; inspect run_status.json.",
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n")
    print("Historical run complete:", output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qqq_research_v1.yaml")
    parser.add_argument("--rate-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.rate_dir, args.output_dir, args.manifest)


if __name__ == "__main__":
    main()
