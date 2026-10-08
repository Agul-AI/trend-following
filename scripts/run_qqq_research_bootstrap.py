#!/usr/bin/env python
"""Paired underlying-path bootstrap of frozen B5, not saved-strategy returns.

Rebuilds observed-session-reset synthetic prices, rates, signals and ledger on
EVERY path. This conditional stress model is not a calibrated future loss model.
Writes a checkpoint after every completed simulation; resume never duplicates a
path and refuses changed frozen inputs/configuration. All block lengths remain
separate. Default precision target: 1,000 paths per block length.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trend_following.research_data import (
    calendar_year_fraction,
    load_research_price,
    stationary_day_bootstrap,
    synthetic_session_reset_price,
)  # noqa:E402
from trend_following.research_evaluation import (  # noqa:E402
    CostScenario,
    evaluate_path,
)
from trend_following.research_protocol import sha256_file, verify_frozen_inputs  # noqa:E402
from trend_following.research_rates import rate_accrual  # noqa:E402

ROOT = Path(__file__).resolve().parents[1]
_CONTEXT = {}


def _initialize(config_path: str, rate_dir: str):
    config = yaml.safe_load(Path(config_path).read_text())
    qqq = load_research_price(ROOT / "data/raw/alpha_vantage_60min/QQQ.parquet")
    rates = Path(rate_dir)
    financing = rate_accrual(
        qqq.index,
        pd.read_csv(rates / "DFF_available.csv", parse_dates=["available_at", "observation_date"]),
        day_count=360,
    )
    cash = rate_accrual(
        qqq.index,
        pd.read_csv(rates / "DTB3_available.csv", parse_dates=["available_at", "observation_date"]),
        day_count=365,
    )
    dt = calendar_year_fraction(qqq.index)
    finance_annual = financing.div(dt * 365 / 360).replace([np.inf, -np.inf], np.nan)
    cash_annual = cash.div(dt).replace([np.inf, -np.inf], np.nan)
    # First source day has no observed overnight interval; exclude it as a source
    # by sampling only later complete day-vectors below. First accrual is zero.
    finance_annual.iloc[0] = finance_annual.iloc[1]
    cash_annual.iloc[0] = cash_annual.iloc[1]
    c = config["costs"]
    scenario = CostScenario(
        "bootstrap_net_cost",
        fee_bps=c["transaction_cost_bps"],
        slippage_bps=c["slippage_bps"],
        financing_spread_bps=c["financing_spread_bps"],
        expense_ratio=c["synthetic_fund_expense_annual"],
        extra_drag=c["synthetic_extra_tracking_drag_annual"],
    )
    _CONTEXT.update(
        config=config,
        qqq=qqq,
        financing_annual=finance_annual,
        cash_annual=cash_annual,
        scenario=scenario,
    )


def _simulate(task):
    block, sim_id, seed = task
    qqq = _CONTEXT["qqq"]
    sampled, paired, source_indices = stationary_day_bootstrap(
        qqq,
        block_days=block,
        seed=seed,
        paired_series={"financing": _CONTEXT["financing_annual"], "cash": _CONTEXT["cash_annual"]},
    )
    raw = synthetic_session_reset_price(sampled)
    dt = calendar_year_fraction(sampled.index)
    financing = paired["financing"] * dt * 365 / 360
    cash = paired["cash"] * dt
    start = pd.Timestamp(_CONTEXT["config"]["evaluation"]["train_start"], tz="America/New_York")
    scenario = _CONTEXT["scenario"]
    _, ledger, metrics = evaluate_path(
        sampled,
        raw,
        financing,
        cash,
        _CONTEXT["config"]["candidates"]["b5"],
        scenario,
        start=start,
        bars_per_day=_CONTEXT["config"]["strategy"]["legacy_bars_per_day"],
    )
    _, qledger, qmetrics = evaluate_path(
        sampled,
        sampled,
        financing,
        cash,
        {"kind": "buyhold", "product_leverage": 1, "effective_leverage": 1},
        scenario,
        start=start,
        bars_per_day=6,
        fund_costs_embedded=True,
    )
    return dict(
        metrics,
        block_days=block,
        sim_id=sim_id,
        seed=seed,
        QQQ_CAGR=qmetrics["CAGR"],
        QQQ_max_drawdown=qmetrics["max_drawdown"],
        paired_CAGR_difference=metrics["CAGR"] - qmetrics["CAGR"],
        paired_max_drawdown_difference=metrics["max_drawdown"] - qmetrics["max_drawdown"],
        source_day_index_checksum=int(source_indices.sum()),
        method="underlying_paths_recomputed_observed_session_reset",
        label="conditional_stress_not_future_probability",
    )


def wilson_interval(k: int, n: int, z: float = 1.96):
    if not n:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return middle - radius, middle + radius


def summarize(table: pd.DataFrame):
    rows = []
    for block, part in table.groupby("block_days"):
        n = len(part)
        k = int(part.max_drawdown.lt(-0.6).sum())
        p = k / n
        low, high = wilson_interval(k, n)
        rows.append(
            dict(
                block_days=block,
                simulations=n,
                CAGR_p05=part.CAGR.quantile(0.05),
                CAGR_median=part.CAGR.median(),
                max_drawdown_p01=part.max_drawdown.quantile(0.01),
                max_drawdown_p05=part.max_drawdown.quantile(0.05),
                max_drawdown_median=part.max_drawdown.median(),
                drawdown_worse_60_frequency=p,
                frequency_MC_standard_error=math.sqrt(p * (1 - p) / n),
                frequency_Wilson95_low=low,
                frequency_Wilson95_high=high,
                paired_CAGR_difference_p05=part.paired_CAGR_difference.quantile(0.05),
                paired_CAGR_difference_median=part.paired_CAGR_difference.median(),
                frequency_interpretation="conditional_scenario_frequency_not_forecast",
            )
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qqq_research_v1.yaml")
    parser.add_argument("--rate-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--block-lengths", type=int, nargs="+", default=[60, 126, 252])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.n_sims < 1 or args.workers < 1:
        parser.error("n-sims and workers must be positive")
    if any(block < 1 for block in args.block_lengths) or len(set(args.block_lengths)) != len(
        args.block_lengths
    ):
        parser.error("block lengths must be positive and unique")
    verify_frozen_inputs(ROOT, args.manifest)
    manifest = json.loads(args.manifest.read_text())
    if sha256_file(args.config) != manifest["protocol_sha256"]:
        raise ValueError("runtime protocol differs from frozen input")
    frozen = {str((ROOT / r["path"]).resolve()): r["sha256"] for r in manifest["data_files"]}
    qqq_file = ROOT / "data/raw/alpha_vantage_60min/QQQ.parquet"
    if frozen.get(str(qqq_file.resolve())) != sha256_file(qqq_file):
        raise ValueError("bootstrap QQQ input missing or differs from frozen input")
    for name in ["DFF_available.csv", "DTB3_available.csv", "metadata.json"]:
        file = (args.rate_dir / name).resolve()
        if frozen.get(str(file)) != sha256_file(file):
            raise ValueError("runtime rate cache differs from frozen input")
    identity = {
        "protocol_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(args.manifest),
        "rates_metadata_sha256": sha256_file(args.rate_dir / "metadata.json"),
        "n_sims": args.n_sims,
        "block_lengths": args.block_lengths,
        "seed": args.seed,
        "method": "paired_underlying_paths_observed_session_reset_v1",
        "protocol_bootstrap_deviation": (
            args.n_sims
            != yaml.safe_load(args.config.read_text())["reporting"]["bootstrap"][
                "target_simulations_each"
            ]
            or args.block_lengths
            != yaml.safe_load(args.config.read_text())["reporting"]["bootstrap"][
                "block_lengths_trading_days"
            ]
            or args.seed
            != yaml.safe_load(args.config.read_text())["reporting"]["bootstrap"]["seed"]
        ),
    }
    if args.resume:
        if json.loads((args.output_dir / "identity.json").read_text()) != identity:
            raise ValueError("resume identity differs; never mix experiments")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    checkpoint = args.output_dir / "results.jsonl"
    records = (
        [json.loads(line) for line in checkpoint.read_text().splitlines()]
        if checkpoint.exists()
        else []
    )
    seen = {(row["block_days"], row["sim_id"]) for row in records}
    if len(seen) != len(records):
        raise ValueError("duplicate checkpoint paths")
    tasks = [
        (block, i, int(np.random.SeedSequence([args.seed, block, i]).generate_state(1)[0]))
        for block in args.block_lengths
        for i in range(args.n_sims)
        if (block, i) not in seen
    ]
    print(f"Bootstrap {len(tasks)} remaining paths; workers={args.workers}", flush=True)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize,
        initargs=(str(args.config), str(args.rate_dir)),
    ) as pool:
        jobs = {pool.submit(_simulate, task): task for task in tasks}
        for job in as_completed(jobs):
            record = job.result()
            with checkpoint.open("a") as stream:
                stream.write(json.dumps(record, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            records.append(record)
            if len(records) % 10 == 0 or len(records) == len(seen) + len(tasks):
                print(f"Completed {len(records)}/{len(seen) + len(tasks)}", flush=True)
                table = pd.DataFrame(records)
                table.to_csv(args.output_dir / "results.csv", index=False)
                summarize(table).to_csv(args.output_dir / "summary.csv", index=False)
    table = pd.DataFrame(records)
    table.to_csv(args.output_dir / "results.csv", index=False)
    summary = summarize(table)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.md").write_text(
        "# Underlying path stress results\n\n"
        "Conditional historical-distribution scenarios, not future loss probabilities. "
        "Signals and portfolios recomputed on every path; QQQ comparisons use paired samples. "
        "Observed-session reset at last cached bar differs from an actual ETF 16:00 reset. "
        "Tax-free net-cost results are primary. Latest-vintage rate regimes are sampled jointly.\n\n"
        + summary.to_markdown(index=False)
        + "\n"
    )
    (args.output_dir / "completion.json").write_text(
        json.dumps(
            {
                "complete": len(records) == args.n_sims * len(args.block_lengths),
                "paths_completed": len(records),
                "paths_requested": args.n_sims * len(args.block_lengths),
            },
            indent=2,
        )
        + "\n"
    )
    print("Bootstrap complete", flush=True)


if __name__ == "__main__":
    main()
