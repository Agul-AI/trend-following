#!/usr/bin/env python3
"""Publish CURRENT development-only metrics and the matching-cost QQQ comparison.

The full grid is not rerun. A single in-sample primary-score leader is replayed
with identical inputs only to produce a derived portfolio curve and verify its
saved metrics. No validation/OOS or other study's outcomes are opened.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trend_following import research_quality_engine_v5 as engine  # noqa: E402
from trend_following import research_quality_protocol_v5 as protocol  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/qqq_daily_halfhour_v5_gap_flags_v2.yaml"
DEFAULT_STUDY = ROOT / ".cache/qqq_daily_halfhour_v5_alpha_gap_flags_v2"
COUNT = 56_644
BENCHMARK_DUMMY_RULE_FIELDS = {
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


def benchmark_public_metrics(metrics: dict) -> dict:
    """A plain buy-and-hold/cash control has no selected indicator rule.

    The kernel's shared result schema carries inert dummy-rule fields; omit them
    from public control rows so they cannot be mistaken for a strategy filter.
    """
    return {key: value for key, value in metrics.items() if key not in BENCHMARK_DUMMY_RULE_FIELDS}


def compare_grid(metrics: pd.DataFrame, benchmark: dict, tolerance: float = 1e-10) -> dict:
    """Compare all rows, not just finalists; flags never affect these counts."""
    if len(metrics) != COUNT or metrics.candidate_id.nunique() != COUNT:
        raise ValueError("Exactly 56,644 distinct current configurations required")
    if not metrics.commission_bps.eq(1.0).all() or not metrics.slippage_bps.eq(3.0).all():
        raise ValueError("Only the frozen 1-bp commission/3-bp slippage baseline is comparable")
    finite = np.isfinite(pd.to_numeric(metrics.cash_excess_sharpe, errors="coerce"))
    if not finite.any():
        raise ValueError("No finite current primary score")
    scores = metrics.loc[finite, ["candidate_id", "cash_excess_sharpe"]]
    best = scores.cash_excess_sharpe.max()
    leader = min(scores.loc[scores.cash_excess_sharpe.ge(best - tolerance), "candidate_id"])
    cagr = metrics.cagr.gt(benchmark["cagr"] + tolerance)
    sharpe = metrics.cash_excess_sharpe.gt(benchmark["cash_excess_sharpe"] + tolerance)
    # Engine drawdown is negative: a larger value is a shallower loss.
    shallower = metrics.max_drawdown.gt(benchmark["max_drawdown"] + tolerance)
    flags = metrics.has_post_gap_confirmed_signal.astype(bool)
    return {
        "configurations": len(metrics),
        "finite_primary_scores": int(finite.sum()),
        "higher_net_cagr_than_qqq_bh": int(cagr.sum()),
        "higher_cash_excess_sharpe_than_qqq_bh": int(sharpe.sum()),
        "shallower_daily_drawdown_than_qqq_bh": int(shallower.sum()),
        "better_all_three_than_qqq_bh": int((cagr & sharpe & shallower).sum()),
        "post_gap_flagged_configurations": int(flags.sum()),
        "post_gap_confirmed_signals": int(metrics.post_gap_confirmed_signal_count.sum()),
        "comparison_tolerance": tolerance,
        "primary_score_leader_id": leader,
        "flags_used_for_selection": False,
    }


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(clean_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def compress_copy(source: Path, target: Path) -> None:
    with source.open("rb") as incoming, target.open("wb") as outgoing:
        with gzip.GzipFile(filename="", mode="wb", fileobj=outgoing, mtime=0) as zipped:
            shutil.copyfileobj(incoming, zipped)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Do not overwrite published current-run evidence")
    config_path = args.config.resolve()
    study = args.study.resolve()
    config = protocol.load_config(config_path)
    registration = protocol._verify_registration(study, config, config_path)
    freeze = protocol._verify_prepared(study, registration)
    stage = protocol._verify_stage(study, "development", registration, freeze)
    source = study / "stages/development"
    metrics = pd.read_csv(source / "metrics.csv", float_precision="round_trip")
    receipt = json.loads((args.benchmarks / "receipt.json").read_text())
    if receipt["stage"] != "development" or receipt[
        "registration_sha256"
    ] != protocol.canonical_hash(registration):
        raise ValueError("Benchmark receipt must belong to the same current development freeze")
    for name, sha in receipt["files"].items():
        if protocol.sha256_file(args.benchmarks / name) != sha:
            raise ValueError(f"Current benchmark artifact changed: {name}")
    bh = receipt["benchmark_metrics"]["buy_and_hold"]
    cash_metrics = receipt["benchmark_metrics"]["cash"]
    comparison = compare_grid(metrics, bh, config["selection"]["tie_tolerance"])
    candidates, table = protocol._candidate_table(config)
    if set(metrics.candidate_id) != set(table):
        raise ValueError("Current comparison omitted/added a configuration")
    leader = table[comparison["primary_score_leader_id"]]
    d, b, calendar, coverage, blackouts, rates, _ = protocol._load_stage(
        study, registration, config, "development"
    )
    replay = engine.simulate_quality_candidate(
        d,
        b,
        leader,
        **protocol._engine_kwargs(
            config, "development", rates, calendar, coverage=coverage, blackouts=blackouts
        ),
    )
    saved = metrics.set_index("candidate_id").loc[leader.candidate_id]
    for key in (
        "cagr",
        "cash_excess_sharpe",
        "max_drawdown",
        "terminal_equity",
        "exposure_fraction",
        "total_commission",
        "total_slippage",
    ):
        if not np.isclose(saved[key], replay.metrics[key], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Current leader replay differs from sealed grid: {key}")
    for key in ("trade_count", "order_count", "post_gap_confirmed_signal_count"):
        if int(saved[key]) != int(replay.metrics[key]):
            raise ValueError(f"Current leader replay count differs: {key}")
    args.output.mkdir(parents=True)
    for name in ("metrics.csv", "post_gap_candidate_flags.csv", "post_gap_signal_events.csv"):
        compress_copy(source / name, args.output / (name + ".gz"))
    shutil.copy2(source / "post_gap_diagnostic_metadata.json", args.output / "gap_diagnostics.json")
    finals = json.loads((source / "finalists.json").read_text())["payload"]["candidates"]
    finalist_ids = [x["candidate_id"] for x in finals]
    metrics.set_index("candidate_id").loc[finalist_ids].reset_index().to_csv(
        args.output / "family_finalists.csv", index=False
    )
    public_bh = benchmark_public_metrics(bh)
    public_cash = benchmark_public_metrics(cash_metrics)
    pd.DataFrame([replay.metrics, public_bh, public_cash]).to_csv(
        args.output / "comparison.csv", index=False
    )
    curves = pd.DataFrame(
        {
            "timestamp": replay.daily_equity.index,
            "leader_equity": replay.daily_equity.to_numpy(),
            "leader_return": replay.daily_returns.to_numpy(),
            "leader_cash_return": replay.daily_cash_returns.to_numpy(),
            "leader_exposure": replay.daily_exposure.to_numpy(),
        }
    )
    for label, name in (("qqq_bh", "buy_and_hold"), ("cash", "cash")):
        daily = pd.read_csv(args.benchmarks / name / "daily.csv", float_precision="round_trip")
        if not pd.DatetimeIndex(pd.to_datetime(daily.timestamp, utc=True)).equals(
            replay.daily_equity.index
        ):
            raise ValueError("Benchmark/leader daily clocks differ")
        curves[f"{label}_equity"] = daily.equity
        curves[f"{label}_return"] = daily["return"]
    curves.to_csv(args.output / "daily_portfolio_curves.csv", index=False)
    summary = {
        "scope": "current_development_only",
        "development_requested_period": config["segments"]["development"],
        "first_scored_session": replay.metrics["start_session"],
        "last_scored_session": replay.metrics["end_session"],
        "scored_sessions": replay.metrics["scored_sessions"],
        "one_way_cost_bps": 4,
        "comparison": comparison,
        "development_leader": replay.metrics,
        "qqq_buy_and_hold": public_bh,
        "cash": public_cash,
        "leader_replay_matches_sealed_grid": True,
        "finalist_count": len(finals),
        "registration_sha256": protocol.canonical_hash(registration),
        "prepared_sha256": protocol.canonical_hash(freeze),
        "development_stage_sha256": protocol.canonical_hash(stage),
        "validation_or_historical_oos_evaluated": False,
        "raw_quote_values_exported": False,
        "caveat": "Development-selected/in-sample results after a 56,644-configuration search; not evidence of out-of-sample superiority.",
    }
    write_json(args.output / "summary.json", summary)
    write_json(
        args.output / "artifact_hashes.json",
        {p.name: protocol.sha256_file(p) for p in sorted(args.output.iterdir()) if p.is_file()},
    )
    print(json.dumps(clean_json(comparison), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
