"""Readiness, immutable final freeze and embargoed scoring for the OOS study.

No scheduler, broker, notifications, old-paper backfill or pretend future returns.
A freeze is a completion marker written only AFTER strict data/calibration/code
checks. Runtime scoring is bound to the archived source/parameters/environment.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from trend_following.research_execution import normalize_spec
from trend_following.research_oos_data import (
    coverage_report,
    file_hash,
    merge_packages,
    read_calendar,
    read_input_package,
    safe_file,
    utc,
    verify_package_derivation,
)
from trend_following.research_oos_evaluation import (
    checkpoint_end_for_months,
    evaluate_oos_checkpoint,
    fit_pretrain_multiplier,
)

RUNTIME_PACKAGES = (
    "numpy",
    "pandas",
    "PyYAML",
    "pyarrow",
    "yfinance",
    "requests",
    "tzdata",
    "pytz",
    "python-dateutil",
)
REQUIRED_MODULES = (
    "trend_following",
    "trend_following.metrics",
    "trend_following.research_ledger",
    "trend_following.research_execution",
    "trend_following.research_rates",
    "trend_following.research_oos_accounting",
    "trend_following.research_oos_data",
    "trend_following.research_oos_evaluation",
    "trend_following.research_oos_protocol",
)
REQUIRED_CODE_PATHS = {
    "src/trend_following/__init__.py",
    "pyproject.toml",
    "scripts/run_qqq_oos_study.py",
    *("src/" + name.replace(".", "/") + ".py" for name in REQUIRED_MODULES[1:]),
}


def verify_runtime_origins(root: Path) -> None:
    for name in REQUIRED_MODULES:
        module = importlib.import_module(name)
        relative = (
            "src/trend_following/__init__.py"
            if name == "trend_following"
            else "src/" + name.replace(".", "/") + ".py"
        )
        if Path(module.__file__).resolve() != (root / relative).resolve():
            raise ValueError("executing modules do not originate from the verified checkout root")


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        **{name: _version(name) for name in RUNTIME_PACKAGES},
    }


def load_oos_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != 2
        or config.get("study_id") != "qqq_oos_v2"
    ):
        raise ValueError("Expected qqq_oos_v2 schema_version2")
    required = {
        "policy_spec",
        "calendar",
        "distribution_calendar",
        "calibration",
        "execution",
        "costs",
        "evaluation",
        "bootstrap",
        "frozen_code_paths",
        "data_contract",
    }
    if not required.issubset(config):
        raise ValueError("incomplete OOS configuration")
    spec = normalize_spec(config["policy_spec"])
    if (
        spec["kind"] != "b5"
        or spec["allocation_multiplier"] != 1
        or spec["max_changes_per_session"] != 1
        or spec["delay_bars"] != 1
    ):
        raise ValueError("Only the predeclared fixed unscaled B5 calibration is supported")
    if config["calibration"] != {
        "sessions": 252,
        "warmup": "features_only",
        "refit": "never",
    } or config["execution"] != {
        "initial_cash": 1000.0,
        "cash_yield": 0.0,
        "bars_per_normal_session": 7,
        "bars_per_early_close": 4,
        "benchmark_entry": "second_completed_bar_independent_of_B5",
    }:
        raise ValueError("changed calibration/execution contract requires a new version")
    if config["costs"] != {
        "fee_bps": 1.0,
        "baseline_slippage_bps": 5.0,
        "sensitivity_slippage_bps": [10.0, 20.0, 30.0],
        "tax_short_rate": 0.24,
        "tax_long_rate": 0.15,
        "tax_interest_rate": 0.24,
        "fund_costs": "embedded_actual_TQQQ_no_duplicate_charge",
    }:
        raise ValueError("changed cost/tax contract requires a new version")
    if config["evaluation"] != {
        "checkpoint_months": [12, 24],
        "primary_months": 24,
        "endpoint": "net_total_return_delta_percentage_points_scaled_B5_minus_QQQ",
        "cutoff": "last_session_close_before_local_anniversary",
        "performance_peeking": "no_retuning_no_early_success_stopping",
    }:
        raise ValueError("changed outcome/horizon requires a new version")
    if config["bootstrap"] != {
        "simulations": 5000,
        "seed": 42,
        "primary_block_sessions": 20,
        "sensitivity_block_sessions": 60,
    }:
        raise ValueError("changed uncertainty specification requires a new version")
    contract = config["data_contract"]
    if (
        contract.get("missing_bars") != "fail_no_fill_no_day_deletion"
        or contract.get("revision_policy") != "first_vintage_wins_report_overlap_revisions"
        or contract.get("signal_basis")
        != "causal_cumulative_split_normalized_price_no_dividend_reinvestment"
        or contract.get("action_availability") != "assumed_public_by_ex_date_not_live_observation"
        or contract.get("dividend_cash_availability")
        != "16:00_America_New_York_on_declared_pay_date"
    ):
        raise ValueError("unsupported data/availability contract")
    paths = config["frozen_code_paths"]
    if not isinstance(paths, list) or not paths or len(paths) != len(set(paths)):
        raise ValueError("unique explicit frozen code closure required")
    if not REQUIRED_CODE_PATHS.issubset(paths):
        raise ValueError("configuration omits required runtime code closure")
    for relative in [*paths, config["calendar"], config["distribution_calendar"]]:
        p = Path(relative)
        if (
            p.is_absolute()
            or ".." in p.parts
            or p.suffix not in {".py", ".csv", ".json", ".yaml", ".toml", ".md"}
        ):
            raise ValueError("unsafe frozen input path")
    return config


def assess_readiness(root: Path, config_path: Path, warmup: Path, *, as_of=None) -> dict:
    now = now_utc() if as_of is None else utc(as_of)
    config = load_oos_config(config_path)
    calendar_path = safe_file(root, config["calendar"])
    calendar = read_calendar(calendar_path)
    prices, actions, package = read_input_package(warmup)
    verify_package_derivation(warmup, calendar)
    if utc(package["observed_at_utc"]) > now:
        raise ValueError("warmup acquisition occurs in the future")
    report = coverage_report(prices, calendar, as_of=now)
    known_completed = calendar.loc[calendar.market_close <= now]
    if known_completed.empty:
        raise ValueError("calendar has no completed training session")
    last_close = known_completed.iloc[-1].market_close
    open_session = calendar.loc[(calendar.market_open <= now) & (calendar.market_close > now)]
    reasons = []
    if not open_session.empty:
        reasons.append("freeze must be completed outside a scheduled open session")
    if prices.index[-1] != last_close:
        reasons.append("warmup does not end at latest complete scheduled session")
    if not report["complete"]:
        reasons.append("missing_or_extra_scheduled_warmup_bars")
    calibration = None
    if not reasons:
        try:
            calibration = fit_pretrain_multiplier(
                prices,
                calendar,
                policy_spec=config["policy_spec"],
                training_end=last_close,
                corporate_actions=actions,
            )
        except ValueError as error:
            reasons.append(str(error))
    future = calendar.loc[calendar.market_open > now]
    prospective_start = future.iloc[0].market_open if not future.empty else None
    checkpoints = {}
    if prospective_start is not None:
        for months in [12, 24]:
            try:
                checkpoints[str(months)] = checkpoint_end_for_months(
                    calendar, prospective_start, months
                ).isoformat()
            except ValueError as error:
                reasons.append(str(error))
    else:
        reasons.append("no future complete session in calendar")
    for relative in config["frozen_code_paths"]:
        safe_file(root, relative)
    return {
        "study_id": config["study_id"],
        "ready": not reasons,
        "reasons": reasons,
        "checked_at_utc": now.isoformat(),
        "quality": report,
        "calibration": calibration,
        "candidate_holdout_start_utc": prospective_start.isoformat()
        if prospective_start is not None
        else None,
        "candidate_checkpoints": checkpoints,
        "state": "ready_for_final_freeze"
        if not reasons
        else "not_frozen_data_or_readiness_failure",
        "warning": "candidate dates are not a declared holdout until final freeze completes",
    }


def _protect_inputs_from_output(output: Path, inputs: list[Path]):
    destination = output.resolve()
    for source in inputs:
        origin = source.resolve()
        if destination == origin or destination.is_relative_to(origin):
            raise ValueError("output cannot be inside immutable input/freeze directory")


def freeze_oos_study(
    root: Path,
    config_path: Path,
    warmup: Path,
    output: Path,
    *,
    attest_ready: bool = False,
    _test_now=None,
) -> dict:
    """Finalize a fresh trial only after copying and rechecking exact inputs.

    The clock injection is a Python unit-test hook, never exposed by the CLI.
    Failed readiness does not create the output directory or any start claim.
    """
    _protect_inputs_from_output(output, [warmup])
    if not attest_ready:
        raise ValueError("explicit final readiness attestation required")
    if _test_now is None:
        verify_runtime_origins(root)
    config = load_oos_config(config_path)
    now = now_utc() if _test_now is None else utc(_test_now)
    readiness = assess_readiness(root, config_path, warmup, as_of=now)
    if not readiness["ready"]:
        raise ValueError("Study not ready: " + "; ".join(readiness["reasons"]))
    # All paths are explicit, safe and reviewed; no dirty-tree dump/environment secrets.
    config_relative = config_path.resolve().relative_to(root.resolve()).as_posix()
    code_paths = [
        config_relative,
        *config["frozen_code_paths"],
        config["calendar"],
        config["distribution_calendar"],
    ]
    records = [
        {
            "path": p,
            "sha256": file_hash(safe_file(root, p)),
            "bytes": safe_file(root, p).stat().st_size,
        }
        for p in dict.fromkeys(code_paths)
    ]
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "snapshot"
    for record in records:
        dest = snapshot / record["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / record["path"], dest)
        if (
            file_hash(dest) != record["sha256"]
            or file_hash(root / record["path"]) != record["sha256"]
        ):
            raise ValueError("source changed during snapshot")
    shutil.copytree(warmup, output / "warmup")
    read_input_package(output / "warmup")
    if file_hash(warmup / "metadata.json") != file_hash(output / "warmup" / "metadata.json"):
        raise ValueError("warmup changed during snapshot")
    final_readiness = assess_readiness(
        root,
        config_path,
        output / "warmup",
        as_of=now_utc() if _test_now is None else utc(_test_now),
    )
    if not final_readiness["ready"]:
        raise ValueError("snapshot crossed readiness boundary; failed freeze must not be resumed")
    for record in records:
        if file_hash(root / record["path"]) != record["sha256"]:
            raise ValueError("source changed before completion")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dependencies = runtime_versions()
    # Expensive copy, calibration, hash and environment work is now finished.
    calendar = read_calendar(snapshot / config["calendar"])
    completed_at = now_utc() if _test_now is None else utc(_test_now)
    past = calendar.loc[calendar.market_close <= completed_at]
    in_session = calendar.loc[
        (calendar.market_open <= completed_at) & (calendar.market_close > completed_at)
    ]
    future = calendar.loc[calendar.market_open > completed_at]
    if (
        not in_session.empty
        or past.empty
        or future.empty
        or past.iloc[-1].market_close.isoformat() != final_readiness["calibration"]["training_end"]
        or future.iloc[0].market_open.isoformat() != final_readiness["candidate_holdout_start_utc"]
    ):
        raise ValueError("final freeze crossed a market/readiness boundary; no completed manifest")
    manifest = {
        "schema_version": 2,
        "study_id": config["study_id"],
        "stage": "final_code",
        "status": "frozen_awaiting_genuinely_future_data_no_performance_observations",
        "frozen_at_utc": completed_at.isoformat(),
        "genuine_holdout_start_utc": final_readiness["candidate_holdout_start_utc"],
        "checkpoints": final_readiness["candidate_checkpoints"],
        "calibration": final_readiness["calibration"],
        "git_head": head or None,
        "code_revision_authority": "exact file hashes not git head alone",
        "files": records,
        "config_path": config_relative,
        "warmup_metadata_sha256": file_hash(output / "warmup" / "metadata.json"),
        "dependencies": dependencies,
        "cash_initial": 1000.0,
        "holdings_initial": {"QQQ": 0.0, "TQQQ": 0.0},
        "returns_observed": 0,
        "simulation_only": True,
        "action_availability": config["data_contract"]["action_availability"],
        "integrity_limit": "local timestamped hash snapshot not independent notarization",
        "previous_paper_pilot": "missed_start_preserved_not_backfilled_or_imported",
    }
    # Last file is the only completion marker. Never rewrite it as a new start.
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    import hashlib

    (output / "receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "start_utc": manifest["genuine_holdout_start_utc"],
                "local_integrity_only": True,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "manifest.json").write_bytes(manifest_bytes)
    after_write = now_utc() if _test_now is None else utc(_test_now)
    if after_write >= utc(manifest["genuine_holdout_start_utc"]):
        (output / "manifest.json").unlink()
        raise ValueError(
            "completion marker crossed session opening; partial snapshot is not a freeze"
        )
    return manifest


def verify_oos_freeze(root: Path, manifest_path: Path, *, verify_runtime: bool = True) -> dict:
    if manifest_path.is_symlink():
        raise ValueError("manifest cannot be a symlink")
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 2
        or manifest.get("study_id") != "qqq_oos_v2"
        or manifest.get("stage") != "final_code"
    ):
        raise ValueError("unsupported/incomplete freeze")
    frozen, start = utc(manifest["frozen_at_utc"]), utc(manifest["genuine_holdout_start_utc"])
    if start <= frozen or frozen > now_utc():
        raise ValueError("holdout must start after completed freeze")
    receipt = json.loads(safe_file(manifest_path.parent, "receipt.json").read_text())
    if receipt.get("manifest_sha256") != file_hash(manifest_path):
        raise ValueError("manifest differs from its separate freeze receipt")
    records = manifest["files"]
    if len(records) != len({r["path"] for r in records}):
        raise ValueError("duplicate frozen records")
    for record in records:
        for folder in [root, manifest_path.parent / "snapshot"]:
            path = safe_file(folder, record["path"])
            if file_hash(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise ValueError("frozen code/config/calendar changed")
    config = load_oos_config(manifest_path.parent / "snapshot" / manifest["config_path"])
    expected = {
        manifest["config_path"],
        *config["frozen_code_paths"],
        config["calendar"],
        config["distribution_calendar"],
    }
    if expected != {r["path"] for r in records}:
        raise ValueError("freeze inventory does not bind the entire declared runtime closure")
    package = manifest_path.parent / "warmup"
    if file_hash(safe_file(package, "metadata.json")) != manifest["warmup_metadata_sha256"]:
        raise ValueError("frozen warmup metadata changed")
    prices, actions, metadata = read_input_package(package)
    if prices.index[-1] >= start or utc(metadata["observed_at_utc"]) > frozen:
        raise ValueError("warmup overlaps holdout or was acquired after freeze")
    if verify_runtime and runtime_versions() != manifest["dependencies"]:
        raise ValueError("runtime dependency versions differ from final freeze")
    if verify_runtime:
        verify_runtime_origins(root)
    calendar = read_calendar(manifest_path.parent / "snapshot" / config["calendar"])
    for months in [12, 24]:
        if (
            checkpoint_end_for_months(calendar, start, months).isoformat()
            != manifest["checkpoints"][str(months)]
        ):
            raise ValueError("checkpoint calendar/manifest disagreement")
    # Re-derive timing and calibration from frozen time, never editable cutoffs.
    reconstructed = assess_readiness(
        root, manifest_path.parent / "snapshot" / manifest["config_path"], package, as_of=frozen
    )
    if (
        not reconstructed["ready"]
        or reconstructed["calibration"] != manifest["calibration"]
        or reconstructed["candidate_holdout_start_utc"] != manifest["genuine_holdout_start_utc"]
        or reconstructed["candidate_checkpoints"] != manifest["checkpoints"]
    ):
        raise ValueError(
            "freeze timing/calibration differs from independently reconstructed readiness"
        )
    return manifest


def score_oos_study(
    root: Path,
    manifest_path: Path,
    future_packages: list[Path],
    months: int,
    output: Path,
    *,
    _test_now=None,
) -> dict:
    _protect_inputs_from_output(output, [manifest_path.parent, *future_packages])
    now = now_utc() if _test_now is None else utc(_test_now)
    manifest = verify_oos_freeze(root, manifest_path)
    if months not in [12, 24] or now < utc(manifest["checkpoints"][str(months)]):
        raise ValueError("checkpoint not yet elapsed; no premature/placeholder performance output")
    config = load_oos_config(manifest_path.parent / "snapshot" / manifest["config_path"])
    calendar = read_calendar(manifest_path.parent / "snapshot" / config["calendar"])
    for package in future_packages:
        _, _, metadata = read_input_package(package)
        verify_package_derivation(package, calendar)
        if utc(metadata["observed_at_utc"]) > now:
            raise ValueError(
                "future package preparation timestamp is later than actual scoring time"
            )
        observed = utc(metadata["source_vintage_at_utc"])
        if observed < utc(manifest["frozen_at_utc"]) or observed > now:
            raise ValueError("future packages must be acquired after freeze and not in the future")
    prices, actions, provenance = merge_packages(
        [manifest_path.parent / "warmup", *future_packages]
    )
    result = evaluate_oos_checkpoint(
        prices,
        calendar,
        policy_spec=config["policy_spec"],
        frozen_multiplier=manifest["calibration"]["multiplier"],
        holdout_start=manifest["genuine_holdout_start_utc"],
        checkpoint_months=months,
        checkpoint_end=manifest["checkpoints"][str(months)],
        as_of=now,
        corporate_actions=actions,
    )
    output.mkdir(parents=True, exist_ok=False)
    for name in [
        "metrics",
        "attribution_metrics",
        "daily_returns",
        "bootstrap_summary",
        "bootstrap_samples",
    ]:
        getattr(result, name).to_csv(output / f"{name}.csv", index=name == "daily_returns")
    paths = output / "paths"
    paths.mkdir()
    for key, ledger_result in result.paths.items():
        pd.DataFrame(
            {
                "equity": ledger_result.equity,
                "cash": ledger_result.cash,
                "returns": ledger_result.returns,
                "turnover": ledger_result.turnover,
            }
        ).to_csv(paths / f"{key}.csv")
        ledger_result.events.to_csv(paths / f"{key}_events.csv", index=False)
    terminal = {
        k: {field: value for field, value in v.items() if field != "ledger"}
        for k, v in result.terminal_snapshots.items()
    }
    metadata = {
        **result.metadata,
        "manifest_sha256": file_hash(manifest_path),
        "provenance": provenance,
        "terminal_snapshots": terminal,
        "raw_prices_publication": "not_included",
    }
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    (output / "completion.json").write_text(
        json.dumps(
            {
                "complete": True,
                "checkpoint_months": months,
                "files": [
                    {
                        "path": p.relative_to(output).as_posix(),
                        "sha256": file_hash(p),
                        "bytes": p.stat().st_size,
                    }
                    for p in sorted(output.rglob("*"))
                    if p.is_file()
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return metadata
