"""Separately sealed mean-reversion DEVELOPMENT; later market stages are closed.

Only original provenance metadata and hash-bound 1999--2011 development bytes
are read. No broad preparation verifier, original results, master prices, or
validation/OOS reader is used. Public output contains whitelisted derived
statistics, not raw quotes, private paths, authorization or full freezes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from trend_following import research_constant_cash as fixed
from trend_following import research_mean_reversion as features
from trend_following import research_mean_reversion_engine as engine
from trend_following import research_quality_engine_v5 as quality
from trend_following import research_quality_protocol_v5 as protocol

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/qqq_mean_reversion.yaml"
DEFAULT_SIDECAR = ROOT / ".cache/mean_reversion_development_20261009"
DEFAULT_REPORT = ROOT / "reports/mean_reversion_development/20261009"
FAMILIES = ("SMA_DISTANCE", "EMA_DISTANCE", "ZSCORE", "RSI")
SLIPPAGES = (1.0, 3.0, 10.0, 25.0)
PERIOD = {"start": "2001-01-01", "end": "2011-12-31"}
TOLERANCE = 1e-10
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *fixed.SOURCE_FILES,
            "src/trend_following/research_mean_reversion.py",
            "src/trend_following/research_mean_reversion_engine.py",
            "src/trend_following/research_mean_reversion_protocol.py",
            "scripts/run_mean_reversion_research.py",
        )
    )
)
Error = protocol.ProtocolError
APPROVAL_EXCERPT = "PLEASE IMPLEMENT THIS PLAN:\n# Separate Mean-Reversion Study — Development Plan"


def approved_config() -> dict:
    """Exact user-approved contract; changed grids need a separate proposal/freeze."""
    return {
        "schema": "qqq_mean_reversion_development",
        "study_name": "Mean-Reversion Study",
        "security": "QQQ",
        "exposure": "unlevered_long_or_cash",
        "historical_label": "retrospective_previously_examined_history",
        "grid": {
            "ma_periods": [10, 20, 50],
            "ma_entry_thresholds": [-0.02, -0.04, -0.06],
            "ma_exit_thresholds": [0.0, 0.01],
            "zscore_periods": [10, 20, 50],
            "zscore_entry_thresholds": [-1.5, -2.0, -2.5],
            "zscore_exit_thresholds": [-0.5, 0.0, 0.5],
            "rsi_periods": [2, 5, 14],
            "rsi_entry_thresholds": [10.0, 20.0, 30.0],
            "rsi_exit_thresholds": [50.0, 60.0, 70.0],
            "candidate_count": 90,
            "same_family_and_period_for_entry_exit": True,
        },
        "signals": {
            "indicator_unit": "daily_sessions",
            "sampling_minutes": 30,
            "confirmation_hours": 3.0,
            "required_consecutive_checks": 7,
            "entry": "finite_feature_less_than_or_equal_to_entry_threshold_while_cash",
            "exit": "finite_feature_greater_than_or_equal_to_exit_threshold_while_long",
            "entry_mode": "persistent_oversold_not_confirmed_recovery",
            "daily_finalize_hour_ny": 17,
            "zscore_ddof": 1,
            "rsi_smoothing": "Wilder",
            "extra_risk_exits": False,
        },
        "periods": {
            "initialization": {"start": "1999-11-01", "end": "2000-12-31"},
            "development": PERIOD,
            "validation": {"start": "2012-01-01", "end": "2018-12-31"},
            "historical_oos": {"start": "2019-01-01", "end": "2026-05-28"},
        },
        "accounting": {
            "initial_cash": 1000.0,
            "cash_nominal_annual_rate": 0.03,
            "cash_day_count": "ACT/365",
            "commission_bps": 1.0,
            "baseline_slippage_bps": 3.0,
            "slippage_bps_scenarios": list(SLIPPAGES),
            "taxes": False,
            "financing": False,
            "separate_fund_expenses": False,
        },
        "selection": {
            "metric": "daily_cash_excess_sharpe",
            "annualization": 252,
            "standard_deviation_ddof": 1,
            "finalists_per_family": 5,
            "maximum_finalists": 20,
            "finite_scores_only": True,
            "tie_tolerance": TOLERANCE,
            "minimum_trade_filter": False,
            "maximum_trade_filter": False,
            "must_beat_buy_and_hold": False,
        },
        "protocol": {
            "authorized_stage": "development",
            "validation_authorized": False,
            "historical_oos_authorized": False,
            "gap_policy": "hold_cancel_pending_reset_streaks",
            "post_gap_flags": "descriptive_never_selection_filters",
        },
    }


def safe_path(path: str | Path) -> Path:
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise Error("Symlink paths/ancestors prohibited")
    return path.resolve()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    value = yaml.safe_load(safe_path(path).read_text())
    # Canonical hashes distinguish bool from int and reject NaN/extra fields.
    if protocol.canonical_hash(value) != protocol.canonical_hash(approved_config()):
        raise Error("Mean-reversion config differs from the approved development-only contract")
    return value


def _authorization(path: Path, config: dict) -> dict:
    value = json.loads(path.read_text())
    if (
        value.get("authorized_by") != "human_user"
        or value.get("explicit_user_approval") is not True
        or value.get("authorized_stages") != ["development"]
        or value.get("validation_authorized") is not False
        or value.get("historical_oos_authorized") is not False
        or value.get("approved_config_sha256") != protocol.canonical_hash(config)
        or value.get("user_request_excerpt") != APPROVAL_EXCERPT
    ):
        raise Error("Explicit matching human development-only authorization required")
    return value


def _separate(sidecar: Path, source: Path, authorization: Path):
    if sidecar.parent != ROOT / ".cache" or not sidecar.name.startswith("mean_reversion_"):
        raise Error("Separate mean_reversion_ private sidecar required")
    if sidecar == source or sidecar in source.parents or source in sidecar.parents:
        raise Error("Never write in or above original frozen study")
    if authorization.is_relative_to(sidecar) or authorization.is_relative_to(source):
        raise Error("Authorization must be outside study/source trees")


def _bindings(config_path: Path, source: Path, source_config: Path, authorization: Path) -> dict:
    config = load_config(config_path)
    _authorization(authorization, config)
    _, registration, prepared, manifest = fixed._read_metadata(source, source_config)
    candidates = [
        {"candidate_id": c.candidate_id, **c.to_dict()} for c in features.candidate_grid()
    ]
    if len(candidates) != 90 or len({c["candidate_id"] for c in candidates}) != 90:
        raise Error("Exactly ninety unique approved candidates required")
    return {
        "schema": "mean_reversion_development_freeze",
        "stage": "development",
        "config_path": str(config_path),
        "config_file_sha256": protocol.sha256_file(config_path),
        "config": config,
        "source_study": str(source),
        "source_config_path": str(source_config),
        "authorization": {
            "path": str(authorization),
            "sha256": protocol.sha256_file(authorization),
        },
        "registration_sha256": protocol.canonical_hash(registration),
        "prepared_metadata_sha256": protocol.canonical_hash(prepared),
        "development_manifest_sha256": protocol.canonical_hash(manifest),
        "development_files": {
            n: prepared["prefixes"]["development"]["files"][n] for n in fixed.PACKET_FILES
        },
        "source_hashes": {n: protocol.sha256_file(ROOT / n) for n in SOURCE_FILES},
        "runtime": protocol._runtime(),
        "cash": fixed.cash_policy(),
        "quality_policy": protocol._quality_policy(protocol.load_config(source_config)).to_dict(),
        "candidates": candidates,
        "grid_sha256": protocol.canonical_hash(candidates),
        "period": PERIOD,
        "initial_cash": 1000.0,
        "confirmation_hours": 3.0,
        "commission_bps": 1.0,
        "slippage_bps_scenarios": list(SLIPPAGES),
        "validation_authorized": False,
        "historical_oos_authorized": False,
        "no_prior_strategy_outcomes_used": True,
    }


def seal_study(
    sidecar=DEFAULT_SIDECAR,
    *,
    authorization,
    config_path=DEFAULT_CONFIG,
    source_study=fixed.DEFAULT_SOURCE_STUDY,
    source_config=fixed.DEFAULT_CONFIG,
):
    sidecar, authorization = safe_path(sidecar), safe_path(authorization)
    source, config_path, source_config = map(safe_path, (source_study, config_path, source_config))
    _separate(sidecar, source, authorization)
    with protocol._attempt(sidecar, "freeze"):
        bindings = _bindings(config_path, source, source_config, authorization)
        target = sidecar / "freeze.json"
        if target.exists():
            previous = protocol._read_seal(target)
            if {k: v for k, v in previous.items() if k != "sealed_at"} != bindings:
                raise Error("Study already sealed differently; preserve and use a new sidecar")
            return previous
        bindings["sealed_at"] = protocol._utc_now()
        protocol._seal(target, bindings)
        return bindings


def verify_study(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    old = protocol._read_seal(sidecar / "freeze.json")
    source, authorization = safe_path(old["source_study"]), safe_path(old["authorization"]["path"])
    _separate(sidecar, source, authorization)
    expected = _bindings(
        safe_path(old["config_path"]), source, safe_path(old["source_config_path"]), authorization
    )
    if {k: v for k, v in old.items() if k != "sealed_at"} != expected:
        raise Error("Frozen mean-reversion inputs/code/runtime changed")
    timestamp = pd.Timestamp(old["sealed_at"])
    if timestamp.tzinfo is None:
        raise Error("Freeze chronology must be timezone-aware")
    return old


def load_development(payload):
    adapter = {
        k: payload[k] for k in ("source_study", "development_manifest_sha256", "development_files")
    }
    adapter["config_path"] = payload["source_config_path"]
    return fixed.load_development_packet(adapter)


def select_finalists(metrics: pd.DataFrame) -> pd.DataFrame:
    """Development-only, finite-only; no BH or trading-frequency eligibility gate."""
    picked = []
    for family in FAMILIES:
        remaining = metrics.loc[
            metrics.family.eq(family) & np.isfinite(metrics.cash_excess_sharpe)
        ].copy()
        for rank in range(1, 6):
            if remaining.empty:
                break
            maximum = remaining.cash_excess_sharpe.max()
            tied = remaining.loc[maximum - remaining.cash_excess_sharpe <= TOLERANCE]
            row = tied.sort_values("candidate_id", kind="stable").iloc[0].copy()
            row["family_rank"] = rank
            picked.append(row)
            remaining = remaining.loc[remaining.candidate_id.ne(row.candidate_id)]
    if not picked:
        return metrics.iloc[:0].assign(family_rank=pd.Series(dtype=int))
    return pd.DataFrame(picked).reset_index(drop=True)


def family_summary(metrics, bh):
    rows = []
    for family in (*FAMILIES, "ALL"):
        part = metrics if family == "ALL" else metrics.loc[metrics.family.eq(family)]
        finite = np.isfinite(part.cash_excess_sharpe)
        winners = part.loc[finite & (part.cash_excess_sharpe - bh > TOLERANCE)]
        row = {
            "family": family,
            "configurations": len(part),
            "finite_Sharpe_count": int(finite.sum()),
            "undefined_Sharpe_count": int((~finite).sum()),
            "beat_BH_Sharpe_count": len(winners),
            "beat_BH_Sharpe_pct": 100 * len(winners) / len(part),
            "BH_Sharpe": bh,
        }
        for label, subset in (("all", part), ("BH_beaters", winners)):
            for field in (
                "cash_excess_sharpe",
                "cagr",
                "max_drawdown",
                "trade_count",
                "trades_per_year",
            ):
                values = subset[field].to_numpy(float)
                values = values[np.isfinite(values)]
                row[f"{label}_median_{field}"] = float(np.median(values)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _array_file(path, result, candidates):
    np.savez_compressed(
        path,
        timestamp=np.array([x.isoformat() for x in result[1]], dtype=str),
        candidate_id=np.array([c.candidate_id for c in candidates], dtype=str),
        equity=result[2],
        returns=result[3],
        cash_returns=result[4],
        exposure=result[5],
    )


def evaluate_study(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    with protocol._attempt(sidecar, "development") as attempt:
        frozen = verify_study(sidecar)
        if (sidecar / "results").exists():
            raise Error("Sealed results cannot be overwritten or rerun")
        started = protocol._utc_now()
        daily, bars, calendar, coverage, blackouts, cash, original_config = load_development(frozen)
        cache = engine.build_cache(
            daily,
            bars,
            calendar,
            coverage=coverage,
            blackouts=blackouts,
            quality_policy=protocol._quality_policy(original_config),
        )
        candidates = list(features.candidate_grid())
        output = attempt / "results"
        output.mkdir()
        all_rows, benchmark_rows = [], []
        for slip in SLIPPAGES:
            kwargs = dict(
                start=PERIOD["start"],
                end=PERIOD["end"],
                cash_observations=cash,
                commission_bps=1.0,
                slippage_bps=slip,
                initial_cash=1000.0,
            )
            result = engine.run(cache, candidates, **kwargs)
            rows = pd.DataFrame(result[0])
            years = (
                result[1][-1]
                - calendar.loc[calendar.session.ge("2001-01-01"), "market_open"].iloc[0]
            ).total_seconds() / (365 * 86400)
            rows["trades_per_year"] = rows.trade_count / years
            rows["scenario_slippage_bps"] = slip
            _array_file(output / f"slippage_{slip:g}_daily.npz", result, candidates)
            for mode in ("cash", "buy_and_hold"):
                benchmark = engine.run(cache, candidates[:1], mode=mode, **kwargs)
                stats = {"portfolio": mode, "scenario_slippage_bps": slip, **benchmark[0][0]}
                stats["trades_per_year"] = stats["trade_count"] / years
                benchmark_rows.append(stats)
                pd.DataFrame(
                    {
                        "timestamp": benchmark[1],
                        "equity": benchmark[2][:, 0],
                        "return": benchmark[3][:, 0],
                        "cash_return": benchmark[4],
                        "exposure": benchmark[5][:, 0],
                    }
                ).to_csv(output / f"{mode}_{slip:g}_daily.csv", index=False)
            bh = benchmark_rows[-1]["cash_excess_sharpe"]
            if not np.isfinite(bh):
                raise Error("Matched development BH Sharpe is undefined")
            rows["BH_cash_excess_sharpe"] = bh
            rows["Sharpe_difference_vs_BH"] = rows.cash_excess_sharpe - bh
            rows["beats_BH_Sharpe"] = np.isfinite(rows.cash_excess_sharpe) & (
                rows.Sharpe_difference_vs_BH > TOLERANCE
            )
            all_rows.append(rows)
            family_summary(rows, bh).to_csv(output / f"family_summary_{slip:g}.csv", index=False)
            if slip == 3:
                baseline = rows
                post = result[-1]
                if post is not None:
                    rows.attrs["post_gap_signal_events"] = quality._post_gap_table(post["rows"])
                    rows.attrs["post_gap_signal_event_metadata"] = quality._post_gap_metadata(
                        post["context"],
                        cache.policy,
                        post["total_rows"],
                        len(post["rows"]),
                        result[0],
                    )
                    protocol._persist_post_gap_diagnostics(
                        output, rows, candidates, original_config, "development"
                    )
            print(
                f"Completed mean-reversion development: 90 configurations; 1-bp commission + {slip:g}-bps slippage",
                flush=True,
            )
        pd.concat(all_rows, ignore_index=True).to_csv(
            output / "candidate_metrics_all_costs.csv", index=False
        )
        baseline.to_csv(output / "candidate_metrics.csv", index=False)
        pd.DataFrame(benchmark_rows).to_csv(output / "benchmark_metrics.csv", index=False)
        finalists = select_finalists(baseline)
        finalists.to_csv(output / "development_finalists.csv", index=False)
        selection = {
            "schema": "mean_reversion_development_finalists",
            "freeze_sha256": protocol.canonical_hash(frozen),
            "selected_at": protocol._utc_now(),
            "family_limit": 5,
            "finalist_count": len(finalists),
            "candidate_ids": finalists.candidate_id.tolist(),
            "uses_validation_or_oos": False,
            "baseline_slippage_bps": 3.0,
            "finite_only": True,
            "no_trade_or_BH_gate": True,
        }
        protocol._seal(output / "finalists.json", selection)
        review_derived_results(output)
        verify_study(sidecar)
        result_seal = {
            "schema": "mean_reversion_development_results",
            "stage": "development",
            "freeze_sha256": protocol.canonical_hash(frozen),
            "grid_sha256": frozen["grid_sha256"],
            "period": PERIOD,
            "sealed_at": frozen["sealed_at"],
            "market_started_at": started,
            "finished_at": protocol._utc_now(),
            "configuration_count": 90,
            "scenarios": list(SLIPPAGES),
            "configuration_scenario_count": 360,
            "finalist_count": len(finalists),
            "validation_authorized": False,
            "historical_oos_authorized": False,
            "no_prior_strategy_outcomes_used": True,
            "files": {p.name: protocol.sha256_file(p) for p in sorted(output.iterdir())},
        }
        protocol._seal(output / "seal.json", result_seal)
        output.rename(sidecar / "results")
        return result_seal


def verify_results(sidecar=DEFAULT_SIDECAR):
    sidecar = safe_path(sidecar)
    frozen = verify_study(sidecar)
    seal = protocol._read_seal(sidecar / "results/seal.json")
    expected = {
        "schema": "mean_reversion_development_results",
        "stage": "development",
        "freeze_sha256": protocol.canonical_hash(frozen),
        "grid_sha256": frozen["grid_sha256"],
        "period": PERIOD,
        "sealed_at": frozen["sealed_at"],
        "configuration_count": 90,
        "scenarios": list(SLIPPAGES),
        "configuration_scenario_count": 360,
        "validation_authorized": False,
        "historical_oos_authorized": False,
        "no_prior_strategy_outcomes_used": True,
    }
    if any(seal.get(k) != v for k, v in expected.items()):
        raise Error("Derived result contract differs from development freeze")
    times = [pd.Timestamp(seal[k]) for k in ("sealed_at", "market_started_at", "finished_at")]
    if any(t.tzinfo is None for t in times) or times != sorted(times):
        raise Error("Freeze must precede market execution and completion")
    required = {
        "candidate_metrics_all_costs.csv",
        "candidate_metrics.csv",
        "benchmark_metrics.csv",
        "development_finalists.csv",
        "finalists.json",
        "post_gap_candidate_flags.csv",
        "post_gap_signal_events.csv",
        "post_gap_diagnostic_metadata.json",
    }
    for slip in SLIPPAGES:
        required |= {
            f"slippage_{slip:g}_daily.npz",
            f"family_summary_{slip:g}.csv",
            f"cash_{slip:g}_daily.csv",
            f"buy_and_hold_{slip:g}_daily.csv",
        }
    actual = {p.name for p in (sidecar / "results").iterdir()} - {"seal.json"}
    if set(seal.get("files", {})) != required or actual != required:
        raise Error("Missing/extra result artifacts")
    for name, digest in seal["files"].items():
        if protocol.sha256_file(sidecar / "results" / name) != digest:
            raise Error("Derived result bytes changed")
    finalists = protocol._read_seal(sidecar / "results/finalists.json")
    if (
        finalists["freeze_sha256"] != seal["freeze_sha256"]
        or finalists["uses_validation_or_oos"] is not False
        or len(finalists["candidate_ids"]) != seal["finalist_count"]
        or len(set(finalists["candidate_ids"])) != seal["finalist_count"]
        or not 0 <= seal["finalist_count"] <= 20
        or finalists.get("baseline_slippage_bps") != 3.0
        or finalists.get("finite_only") is not True
        or finalists.get("no_trade_or_BH_gate") is not True
    ):
        raise Error("Finalist seal changed")
    selected = pd.Timestamp(finalists["selected_at"])
    if selected.tzinfo is None or not times[1] <= selected <= times[2]:
        raise Error("Finalists must be selected inside the development evaluation chronology")
    return seal


PUBLIC_METRICS = (
    "candidate_id",
    "family",
    "period",
    "indicator_units",
    "entry_threshold",
    "exit_threshold",
    "confirmation_hours",
    "entry_confirmation_hours",
    "exit_confirmation_hours",
    "entry_required_checks",
    "exit_required_checks",
    "cash_excess_sharpe",
    "cagr",
    "max_drawdown",
    "annualized_volatility",
    "exposure_fraction",
    "total_turnover",
    "annualized_turnover",
    "trade_count",
    "trades_per_year",
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
    "unknown_missing_slots",
    "halt_affected_slots",
    "cancelled_blackout_order_count",
    "cancelled_terminal_order_count",
    "post_gap_confirmed_buy_count",
    "post_gap_confirmed_sell_count",
    "post_gap_confirmed_signal_count",
    "has_post_gap_confirmed_signal",
    "scenario_slippage_bps",
    "BH_cash_excess_sharpe",
    "Sharpe_difference_vs_BH",
    "beats_BH_Sharpe",
)


def _public_frame(frame, columns):
    result = frame.loc[:, [n for n in columns if n in frame]].copy()
    for values in result.select_dtypes(include=["object", "string"]).values.flatten():
        if isinstance(values, str) and (
            "/Users/" in values or ".cache/" in values or "@" in values
        ):
            raise Error("Private text rejected by public projection")
    return result


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def review_derived_results(private):
    """Recompute published scores from newly sealed derived arrays, not prices."""
    private = Path(private)
    metrics = pd.read_csv(private / "candidate_metrics_all_costs.csv", float_precision="round_trip")
    benchmark = pd.read_csv(private / "benchmark_metrics.csv", float_precision="round_trip")
    candidates = {c.candidate_id: c for c in features.candidate_grid()}
    if len(metrics) != 360 or metrics.duplicated(["candidate_id", "scenario_slippage_bps"]).any():
        raise Error("All ninety configurations at all four costs are required")
    if (
        len(benchmark) != 8
        or benchmark.duplicated(["portfolio", "scenario_slippage_bps"]).any()
        or set(zip(benchmark.portfolio, benchmark.scenario_slippage_bps, strict=True))
        != {(p, s) for p in ("cash", "buy_and_hold") for s in SLIPPAGES}
    ):
        raise Error("Exactly eight matched passive benchmark/scenario rows required")
    for slip in SLIPPAGES:
        part = metrics.loc[metrics.scenario_slippage_bps.eq(slip)]
        if len(part) != 90 or set(part.candidate_id) != set(candidates):
            raise Error("Candidate identity/coverage changed")
        for row in part.itertuples(index=False):
            candidate = candidates[row.candidate_id]
            identity = {
                "family": candidate.family,
                "period": candidate.rule.period,
                "entry_threshold": candidate.entry_threshold,
                "exit_threshold": candidate.exit_threshold,
                "entry_confirmation_hours": 3.0,
                "exit_confirmation_hours": 3.0,
                "entry_required_checks": 7,
                "exit_required_checks": 7,
                "commission_bps": 1.0,
                "slippage_bps": slip,
                "sampling_minutes": 30,
                "start_session": "2001-01-02",
                "end_session": "2011-12-30",
                "scored_sessions": 2767,
            }
            if any(getattr(row, k) != value for k, value in identity.items()):
                raise Error("Published candidate identity, period or accounting changed")
            if (
                row.trade_count < 0
                or row.trade_count != int(row.trade_count)
                or row.order_count != 2 * row.trade_count
            ):
                raise Error("Round trips/one-way orders do not reconcile")
        with np.load(private / f"slippage_{slip:g}_daily.npz", allow_pickle=False) as arrays:
            ids = arrays["candidate_id"].tolist()
            times = pd.DatetimeIndex(pd.to_datetime(arrays["timestamp"], utc=True))
            equity, returns, cash_returns = (
                arrays["equity"],
                arrays["returns"],
                arrays["cash_returns"],
            )
            if (
                len(ids) != 90
                or len(set(ids)) != 90
                or set(ids) != set(candidates)
                or equity.shape != (2767, 90)
                or returns.shape != equity.shape
                or cash_returns.shape != (2767,)
                or len(times) != 2767
                or not times.is_monotonic_increasing
                or times.has_duplicates
                or not (times.tz_convert("America/New_York").hour == 17).all()
                or times[0].tz_convert("America/New_York").strftime("%Y-%m-%d") != "2001-01-02"
                or times[-1].tz_convert("America/New_York").strftime("%Y-%m-%d") != "2011-12-30"
                or not np.isfinite(equity).all()
                or (equity <= 0).any()
            ):
                raise Error("Invalid development-only derived array dimensions/clock/wealth")
            expected_returns = equity / np.vstack([np.full(90, 1000.0), equity[:-1]]) - 1
            if not np.allclose(returns, expected_returns, rtol=0, atol=1e-12):
                raise Error("Daily returns do not reconcile to wealth")
            start = pd.Timestamp("2001-01-02 09:30", tz="America/New_York").tz_convert("UTC")
            years = (times[-1] - start).total_seconds() / (365 * 86400)
            excess = returns - cash_returns[:, None]
            sd = excess.std(axis=0, ddof=1)
            sr = np.divide(
                np.sqrt(252) * excess.mean(axis=0), sd, out=np.full(90, np.nan), where=sd > 0
            )
            wealth = np.vstack([np.full(90, 1000.0), equity])
            expected = {
                "cash_excess_sharpe": sr,
                "terminal_equity": equity[-1],
                "cagr": (equity[-1] / 1000.0) ** (1 / years) - 1,
                "max_drawdown": (wealth / np.maximum.accumulate(wealth, axis=0) - 1).min(axis=0),
                "annualized_volatility": returns.std(axis=0, ddof=1) * np.sqrt(252),
            }
            aligned = part.set_index("candidate_id").loc[ids]
            for field, values in expected.items():
                if not np.allclose(aligned[field], values, rtol=1e-10, atol=1e-11, equal_nan=True):
                    raise Error(f"Derived {field} differs from saved wealth/return paths")
            if not np.allclose(aligned.trades_per_year, aligned.trade_count / years, atol=1e-12):
                raise Error("Annualized trade count changed")
            for portfolio in ("cash", "buy_and_hold"):
                row = benchmark.loc[
                    benchmark.portfolio.eq(portfolio) & benchmark.scenario_slippage_bps.eq(slip)
                ].iloc[0]
                curve = pd.read_csv(
                    private / f"{portfolio}_{slip:g}_daily.csv", float_precision="round_trip"
                )
                if (
                    not pd.DatetimeIndex(pd.to_datetime(curve.timestamp, utc=True)).equals(times)
                    or not np.allclose(curve.cash_return, cash_returns, atol=1e-12)
                    or not np.allclose(
                        curve["return"],
                        curve.equity / np.r_[1000.0, curve.equity[:-1]] - 1,
                        atol=1e-12,
                    )
                ):
                    raise Error("Passive curve clock/wealth/cash reference changed")
                ex = curve["return"].to_numpy() - cash_returns
                score = (
                    float(np.sqrt(252) * ex.mean() / ex.std(ddof=1))
                    if ex.std(ddof=1) > 0
                    else np.nan
                )
                if not np.isclose(
                    row.cash_excess_sharpe, score, rtol=1e-10, atol=1e-11, equal_nan=True
                ):
                    raise Error("Passive Sharpe changed")
                if not np.isclose(row.terminal_equity, curve.equity.iloc[-1], atol=1e-9):
                    raise Error("Passive terminal wealth changed")
                passive_wealth = np.r_[1000.0, curve.equity]
                passive_stats = {
                    "cagr": (curve.equity.iloc[-1] / 1000.0) ** (1 / years) - 1,
                    "max_drawdown": (
                        passive_wealth / np.maximum.accumulate(passive_wealth) - 1
                    ).min(),
                    "annualized_volatility": curve["return"].std(ddof=1) * np.sqrt(252),
                }
                for field, value in passive_stats.items():
                    if not np.isclose(row[field], value, rtol=1e-10, atol=1e-11, equal_nan=True):
                        raise Error(f"Passive {field} changed")
            bh = (
                benchmark.loc[
                    benchmark.portfolio.eq("buy_and_hold")
                    & benchmark.scenario_slippage_bps.eq(slip)
                ]
                .iloc[0]
                .cash_excess_sharpe
            )
            delta = aligned.cash_excess_sharpe - bh
            if (
                not np.allclose(aligned.BH_cash_excess_sharpe, bh, atol=1e-12)
                or not np.allclose(
                    aligned.Sharpe_difference_vs_BH, delta, atol=1e-12, equal_nan=True
                )
                or not np.array_equal(
                    aligned.beats_BH_Sharpe, np.isfinite(delta) & (delta > TOLERANCE)
                )
            ):
                raise Error("Cost-matched BH comparison changed")
    baseline = metrics.loc[metrics.scenario_slippage_bps.eq(3)].copy()
    stored = pd.read_csv(private / "candidate_metrics.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(baseline.reset_index(drop=True), stored.reset_index(drop=True))
    finalists = select_finalists(baseline)
    stored_finalists = pd.read_csv(
        private / "development_finalists.csv", float_precision="round_trip"
    )
    pd.testing.assert_frame_equal(
        finalists.reset_index(drop=True), stored_finalists.reset_index(drop=True), check_dtype=False
    )
    return metrics, baseline, benchmark, finalists


def report(sidecar=DEFAULT_SIDECAR, output=DEFAULT_REPORT):
    sidecar, output = safe_path(sidecar), safe_path(output)
    seal = verify_results(sidecar)
    if not output.is_relative_to(ROOT / "reports/mean_reversion_development") or output.exists():
        raise Error("A new separate mean-reversion publication folder is required")
    private = sidecar / "results"
    metrics, baseline, benchmark, finalists = review_derived_results(private)
    expected = {c.candidate_id for c in features.candidate_grid()}
    if len(metrics) != 360 or len(baseline) != 90:
        raise Error("Public comparison requires all ninety at all four scenarios")
    for slip in SLIPPAGES:
        part = metrics.loc[metrics.scenario_slippage_bps.eq(slip)]
        if (
            len(part) != 90
            or set(part.candidate_id) != expected
            or part.candidate_id.duplicated().any()
        ):
            raise Error("Candidate coverage/identity changed")
    finalists = select_finalists(baseline)
    sealed_finalists = protocol._read_seal(private / "finalists.json")
    if finalists.candidate_id.tolist() != sealed_finalists["candidate_ids"]:
        raise Error("Published finalists differ from sealed development selection")
    bh = benchmark.loc[
        benchmark.portfolio.eq("buy_and_hold") & benchmark.scenario_slippage_bps.eq(3)
    ].iloc[0]
    summary = family_summary(baseline, bh.cash_excess_sharpe)
    output.mkdir(parents=True)
    _public_frame(metrics, PUBLIC_METRICS).to_csv(
        output / "candidate_metrics_all_costs.csv", index=False
    )
    _public_frame(baseline, PUBLIC_METRICS).to_csv(output / "candidate_metrics.csv", index=False)
    _public_frame(finalists, (*PUBLIC_METRICS, "family_rank")).to_csv(
        output / "development_finalists.csv", index=False
    )
    benchmark_columns = tuple(
        n
        for n in PUBLIC_METRICS
        if n
        not in {
            "candidate_id",
            "family",
            "period",
            "indicator_units",
            "entry_threshold",
            "exit_threshold",
            "confirmation_hours",
            "entry_confirmation_hours",
            "exit_confirmation_hours",
            "entry_required_checks",
            "exit_required_checks",
        }
    )
    _public_frame(benchmark, ("portfolio", *benchmark_columns)).to_csv(
        output / "benchmark_metrics.csv", index=False
    )
    summary.to_csv(output / "family_summary.csv", index=False)
    stress = []
    for slip in SLIPPAGES:
        part = metrics.loc[metrics.scenario_slippage_bps.eq(slip)]
        reference = benchmark.loc[
            benchmark.portfolio.eq("buy_and_hold") & benchmark.scenario_slippage_bps.eq(slip)
        ].iloc[0]
        table = family_summary(part, reference.cash_excess_sharpe)
        table["slippage_bps"] = slip
        stress.append(table)
    pd.concat(stress, ignore_index=True).to_csv(
        output / "family_summary_all_costs.csv", index=False
    )
    for name in ("post_gap_candidate_flags.csv", "post_gap_signal_events.csv"):
        frame = pd.read_csv(private / name)
        _public_frame(frame, frame.columns).to_csv(output / name, index=False)
    metadata = json.loads((private / "post_gap_diagnostic_metadata.json").read_text())
    protocol._write_json(
        output / "gap_diagnostics.json",
        _clean(
            {
                k: v
                for k, v in metadata.items()
                if k not in ("config_path", "source_study", "authorization")
            }
        ),
    )
    public_summary = {
        "study_name": "Mean-Reversion Study",
        "security": "QQQ",
        "period": PERIOD,
        "configuration_count": 90,
        "configuration_scenario_count": 360,
        "families": list(FAMILIES),
        "shared_confirmation_hours": 3,
        "entry_mode": "persistent_oversold_not_rebound_confirmation",
        "initial_cash": 1000,
        "cash_policy": fixed.cash_policy(),
        "commission_bps": 1,
        "slippage_bps_scenarios": list(SLIPPAGES),
        "baseline_slippage_bps": 3,
        "freeze_sha256": seal["freeze_sha256"],
        "grid_sha256": seal["grid_sha256"],
        "freeze_at": seal["sealed_at"],
        "market_started_at": seal["market_started_at"],
        "finished_at": seal["finished_at"],
        "finalist_count": len(finalists),
        "finalists_sha256": protocol.canonical_hash(sealed_finalists),
        "baseline_BH_Sharpe": bh.cash_excess_sharpe,
        "baseline_BH_CAGR": bh.cagr,
        "baseline_BH_max_drawdown": bh.max_drawdown,
        "validation_authorized": False,
        "historical_oos_authorized": False,
        "no_prior_strategy_outcomes_used": True,
        "all_90_in_denominator": True,
        "trade_definition": "completed_round_trips_including_terminal_liquidation",
        "finalist_eligibility": "finite_development_baseline_Sharpe_only_no_trade_or_BH_gate",
        "retrospective_previously_examined_history": True,
    }
    protocol._write_json(output / "summary.json", _clean(public_summary))
    lines = [
        "# Mean-Reversion Study — Development Only",
        "",
        "**2001–2011; 90 configurations; shared 3-hour entry/exit confirmation.**",
        "Validation (2012–2018) and historical OOS (2019–May 28, 2026) remain closed.",
        "",
        f"Baseline QQQ buy-and-hold: **{bh.cash_excess_sharpe:.4f} cash-excess Sharpe**, **{bh.cagr:.2%} CAGR**, **{-bh.max_drawdown:.2%} maximum daily drawdown loss**.",
        "",
        "| Family | Configs | Beat BH Sharpe | All-config median Sharpe | BH-beater median Sharpe | BH-beater median CAGR | BH-beater median round trips |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def number(v, percent=False):
        return "undefined" if not np.isfinite(v) else (f"{v:.2%}" if percent else f"{v:.4f}")

    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.family} | {row.configurations} | {row.beat_BH_Sharpe_count} | {number(row.all_median_cash_excess_sharpe)} | {number(row.BH_beaters_median_cash_excess_sharpe)} | {number(row.BH_beaters_median_cagr, True)} | {number(row.BH_beaters_median_trade_count)} |"
        )
    lines += [
        "",
        "## Method and limits",
        "",
        "- Persistent weakness is the entry hypothesis, not evidence of a confirmed rebound. Cash enters below its threshold; long exits above its recovery threshold. Equality qualifies. Seven consecutive completed half-hour observations are required on either side, with next-safe-bar-open execution.",
        "- SMA/EMA distance uses 10/20/50 daily sessions, entry −2/−4/−6%, exit 0/+1%. Z-score uses 10/20/50 sessions with sample SD (ddof=1), entry −1.5/−2/−2.5, exit −0.5/0/+0.5. RSI uses Wilder smoothing with 2/5/14-session lookbacks, entry 10/20/30, exit 50/60/70. Bollinger-equivalent thresholds are not duplicate families.",
        "- Features use a causal forward-built total-return signal index and pure provisional daily previews; finalized daily state advances once at the provider quote's assumed 17:00 NY availability. Alpha quotes are reconstructed as-traded units, not guaranteed native raw or point-in-time captures; provider daily marks are not guaranteed official/executable closes.",
        "- Fixed 3% nominal ACT/365 cash accrues over calendar time and compounds on the event clock. Effective cash growth is slightly above 3%, not exactly 3% APY. Every one-way order pays 1 bp commission plus the scenario's adverse slippage; taxes, leverage, shorting, financing and separate fund-expense charges are disabled.",
        "- Hypothetical slippage 1/3/10/25 bps applies across the whole period. No hindsight regime labels or cost-specific reselection. Cost drag compares net wealth against a parallel gross ledger with the same decisions and fills.",
        "- Dividends credit pre-open holders after splits and reinvest at the first safe observed open, using the same cost-free DRIP proxy for strategies and BH. Terminal positions liquidate at the last safe RTH close, followed by cash accrual to the 17:00 mark.",
        "- Missing development intervals hold positions without fabricated observations or orders; pending orders are cancelled and streaks reset. Post-gap flags extend through the next session close and never affect ranking.",
        "- Returns and Sharpe use daily valuation marks, 252 annualization and sample standard deviation. Drawdown is daily, not intraday. Round trips include terminal liquidation; trades/year divides round trips by actual calendar years; holding duration includes market closures.",
        f"- All 90 configurations and 360 scenario outcomes remain visible. Up to five finite-scoring development candidates per family are retained ({len(finalists)} total), with no minimum/maximum trade filter or BH eligibility requirement. Ties within 1e-10 resolve by stable ID. No validation winner was selected.",
        "- BH-beater medians condition on observed development Sharpe; they are not whole-family or independent confirmation statistics. Parameter searching and finalist selection are in-sample. Previously examined history is retrospective, not untouched or live evidence.",
        "",
        "## Evidence",
        "",
        "[All 90 baseline results](candidate_metrics.csv) · [All 360 cost outcomes](candidate_metrics_all_costs.csv) · [Family comparison](family_summary.csv) · [Cost comparisons](family_summary_all_costs.csv) · [Matched benchmarks](benchmark_metrics.csv) · [Frozen development finalists](development_finalists.csv) · [Protocol summary](summary.json) · [Gap flags](post_gap_candidate_flags.csv) · [Gap events](post_gap_signal_events.csv) · [Artifact hashes](artifact_hashes.json)",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    protocol._write_json(
        output / "artifact_hashes.json",
        {p.name: protocol.sha256_file(p) for p in sorted(output.iterdir())},
    )
    return public_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Separate Mean-Reversion Study — DEVELOPMENT ONLY")
    parser.add_argument("command", choices=("check", "freeze", "development", "report"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--source-study", type=Path, default=fixed.DEFAULT_SOURCE_STUDY)
    parser.add_argument("--source-config", type=Path, default=fixed.DEFAULT_CONFIG)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    if args.command == "check":
        load_config(args.config)
        fixed._read_metadata(safe_path(args.source_study), safe_path(args.source_config))
        result = {
            "stage": "development",
            "configurations": len(features.candidate_grid()),
            "shared_wait_hours": 3,
            "validation_authorized": False,
            "historical_oos_authorized": False,
        }
    elif args.command == "freeze":
        if args.authorization is None:
            parser.error("freeze requires --authorization with the matching human instruction")
        result = seal_study(
            args.sidecar,
            authorization=args.authorization,
            config_path=args.config,
            source_study=args.source_study,
            source_config=args.source_config,
        )
        result = {
            "freeze_sha256": protocol.canonical_hash(result),
            "sealed_at": result["sealed_at"],
        }
    elif args.command == "development":
        result = evaluate_study(args.sidecar)
        result = {
            k: result[k]
            for k in (
                "stage",
                "configuration_count",
                "configuration_scenario_count",
                "finalist_count",
                "finished_at",
            )
        }
    else:
        result = report(args.sidecar, args.output)
    print(json.dumps(_clean(result), indent=2, sort_keys=True, allow_nan=False))
    return 0
