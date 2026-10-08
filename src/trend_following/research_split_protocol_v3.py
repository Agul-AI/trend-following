"""An isolated chronological rebuild: seal -> select -> evaluate, never retune.

Locks are local reproducibility evidence, not independent notarization. Historical
research exposure is disclosed. No old signals, synthetic caches or result tables
are inputs to this experiment. All numerical output is generated after a new seal.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import shutil
from dataclasses import asdict, fields, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_FILES = (
    "pyproject.toml",
    "configs/qqq_split_v3.yaml",
    "scripts/run_qqq_split_study.py",
    "src/trend_following/__init__.py",
    "src/trend_following/research_split_v3.py",
    "src/trend_following/research_split_protocol_v3.py",
    "src/trend_following/research_ledger.py",
    "src/trend_following/research_rates.py",
    "tests/test_research_split_v3.py",
    "tests/test_research_split_protocol_v3.py",
)
PACKAGES = ("numpy", "pandas", "PyYAML", "pyarrow", "tzdata", "pytz", "python-dateutil")


def digest(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Regular nonsymlink file required: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime() -> dict:
    return {
        "python": platform.python_version(),
        **{p: importlib.metadata.version(p) for p in PACKAGES},
    }


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def clock() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != 3 or config.get("study_id") != "qqq_split_v3":
        raise ValueError("Unsupported split-study protocol")
    last = None
    for name in ("development", "validation", "test"):
        segment = config["segments"][name]
        start, end = date.fromisoformat(segment["start"]), date.fromisoformat(segment["end"])
        if start > end or (last is not None and start != last + timedelta(days=1)):
            raise ValueError("Segments must be chronological, nonoverlapping and contiguous")
        last = end
    policy = config["policy"]
    windows, allocations = policy["ma_sessions"], policy["allocations"]
    if (
        not windows
        or any(isinstance(x, bool) or int(x) != x or x < 2 for x in windows)
        or len(set(windows)) != len(windows)
        or not allocations
        or not np.isfinite(allocations).all()
        or any(not 0 < x <= 1 for x in allocations)
        or len(set(allocations)) != len(allocations)
        or policy["candidate_count"] != len(windows) * len(allocations)
    ):
        raise ValueError("Invalid predefined candidate grid")
    if (
        policy["allocation_mode"] != "on_target_change_only_with_drift"
        or policy["execution_delay_sessions"] != 1
        or policy["product_leverage"] != 3
        or config["selection"]["stage"] != "validation_only"
        or config["selection"]["layer"] != "expense"
        or config["selection"]["drawdown_screen"] is not None
        or config["evaluation"]["cost_layers"]
        != ["gross", "trading", "financing", "expense", "tax"]
        or config["evaluation"]["benchmarks"] != ["qqq_buy_hold", "cash", "qqq200ma"]
        or config["evaluation"]["primary_layer"] != "expense"
        or config["evaluation"]["retraining_during_test"]
        or config["evaluation"]["expand_grid_after_test"]
    ):
        raise ValueError("Unsupported rule, selection or evaluation semantics")
    if (
        windows != [50, 100, 200]
        or allocations != [1 / 3, 2 / 3, 1.0]
        or config["selection"]["tie_tolerance"] != 1e-10
        or config["costs"]["financing_day_count"] != 360.0
        or config["costs"]["expense_day_count"] != 365.0
        or config["costs"]["cash_day_count"] != 365.0
    ):
        raise ValueError(
            "Only the implemented fixed grid, tie rule and day-count conventions are supported"
        )
    if (
        not np.isfinite(config["selection"]["tie_tolerance"])
        or config["selection"]["tie_tolerance"] < 0
    ):
        raise ValueError("Invalid tie tolerance")
    return config


def candidates(config: dict) -> dict:
    result = {}
    for window in config["policy"]["ma_sessions"]:
        for i, allocation in enumerate(config["policy"]["allocations"], 1):
            key = f"ma{window}_alloc_{i}_3"
            result[key] = dict(
                name=key, kind="ma", asset="synthetic_3x", ma_sessions=window, allocation=allocation
            )
    return result


def costs(config: dict):
    from trend_following.research_split_v3 import SplitCosts

    names = {field.name for field in fields(SplitCosts)}
    values = {k: v for k, v in config["costs"].items() if k in names}
    return SplitCosts(initial_cash=config["policy"]["initial_cash"], **values)


def _date_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if "date" in frame:
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    else:
        dates = pd.DatetimeIndex(frame.index)
    if dates.tz is not None:
        dates = dates.tz_convert("America/New_York").tz_localize(None)
    dates = dates.normalize()
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Source session dates must be unique, sorted and nonmissing")
    return dates


def source_quality(qqq: Path, reference: Path, config: dict) -> dict:
    source = pd.read_parquet(qqq)
    other = pd.read_parquet(reference)
    dates, control = _date_index(source), _date_index(other)
    if not {"close", "adj_close", "dividend_amount", "split_coefficient"}.issubset(source):
        raise ValueError("Raw daily prices and corporate actions are required")
    values = source[["close", "dividend_amount", "split_coefficient"]].to_numpy(float)
    if (
        not np.isfinite(values).all()
        or (values[:, 0] <= 0).any()
        or (values[:, 1] < 0).any()
        or (values[:, 2] <= 0).any()
    ):
        raise ValueError("Invalid raw close, distribution or split")
    rows = []
    for name, bounds in config["segments"].items():
        start, end = pd.Timestamp(bounds["start"]), pd.Timestamp(bounds["end"])
        a, b = (
            dates[(dates >= start) & (dates <= end)],
            control[(control >= start) & (control <= end)],
        )
        if not len(a) or not a.equals(b):
            raise ValueError(f"Daily session coverage differs from independent source: {name}")
        rows.append(
            dict(
                segment=name,
                sessions=len(a),
                first_session=str(a[0].date()),
                last_session=str(a[-1].date()),
                missing_relative_to_reference=0,
            )
        )
    warmup = dates[dates < pd.Timestamp(config["segments"]["development"]["start"])]
    if len(warmup) < max(config["policy"]["ma_sessions"]):
        raise ValueError("Insufficient feature-only warmup before development")
    return dict(
        rows=rows,
        feature_only_warmup_sessions=len(warmup),
        daily_source_rows=len(source),
        reference_scope="Session dates only; not price validation",
        selected_distributions=int(
            (
                source.loc[
                    (dates >= pd.Timestamp(rows[0]["first_session"]))
                    & (dates <= pd.Timestamp(rows[-1]["last_session"])),
                    "dividend_amount",
                ]
                > 0
            ).sum()
        ),
    )


def audit_rates(directory: Path) -> dict:
    from trend_following.research_rates import prepare_rate_observations

    result = {}
    for name in ("DFF", "DTB3"):
        raw = pd.read_csv(directory / f"{name}_raw.csv", float_precision="round_trip")
        derived = prepare_rate_observations(
            raw, name, publication_lag_business_days=2, publication_hour=18
        )
        supplied = pd.read_csv(directory / f"{name}_available.csv", float_precision="round_trip")
        if len(derived) != len(supplied) or not {
            "observation_date",
            "available_at",
            "annual_rate",
            "series_id",
        }.issubset(supplied):
            raise ValueError("Prepared rate source does not match raw observations")
        dates = pd.to_datetime(supplied.observation_date)
        stamps = pd.to_datetime(supplied.available_at, utc=True)
        if (
            not pd.DatetimeIndex(dates).equals(pd.DatetimeIndex(derived.observation_date))
            or not pd.DatetimeIndex(stamps).equals(pd.DatetimeIndex(derived.available_at))
            or not supplied.series_id.eq(name).all()
            or not np.allclose(supplied.annual_rate, derived.annual_rate, rtol=1e-12, atol=1e-15)
        ):
            raise ValueError(
                "Prepared rate lag/conversion differs from the declared raw-source transformation"
            )
        result[name] = dict(
            rows=len(derived), availability_rederived=True, annual_rate_rederived=True
        )
    return result


def _origins(root: Path) -> None:
    for name in (
        "research_split_protocol_v3",
        "research_split_v3",
        "research_ledger",
        "research_rates",
    ):
        module = importlib.import_module(f"trend_following.{name}")
        if Path(module.__file__).resolve() != root / "src" / "trend_following" / f"{name}.py":
            raise ValueError(f"Loaded module does not belong to verified checkout: {name}")


def freeze_split_study(
    root: Path, config_path: Path, qqq: Path, reference: Path, rate_dir: Path, output: Path
) -> dict:
    root = root.resolve()
    _origins(root)
    config = read_config(config_path)
    if config_path.resolve() != root / "configs/qqq_split_v3.yaml":
        raise ValueError("Use the versioned v3 config in the executing checkout")
    input_paths = {
        "QQQ_daily.parquet": qqq,
        "QQQ_reference_daily.parquet": reference,
        **{
            name: rate_dir / name
            for name in (
                "DFF_available.csv",
                "DTB3_available.csv",
                "DFF_raw.csv",
                "DTB3_raw.csv",
                "metadata.json",
            )
        },
    }
    originals = {name: digest(path) for name, path in input_paths.items()}
    quality = source_quality(qqq, reference, config)
    rate_quality = audit_rates(rate_dir)
    costs(config)  # Reject impossible assumptions before creating a study.
    output = output.resolve()
    if output.exists() or any(
        output == p.resolve().parent or p.resolve().parent in output.parents
        for p in input_paths.values()
    ):
        raise ValueError("New study output must be outside original input directories")
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    for name, source in input_paths.items():
        shutil.copyfile(source, output / "inputs" / name)
        if digest(output / "inputs" / name) != originals[name] or digest(source) != originals[name]:
            raise ValueError("Input changed during freezing; preserve this incomplete attempt")
    code_records = []
    for relative in CODE_FILES:
        source = root / relative
        destination = output / "snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = digest(source)
        shutil.copyfile(source, destination)
        if digest(destination) != original or digest(source) != original:
            raise ValueError("Code changed during freezing")
        code_records.append(dict(path=relative, sha256=original, bytes=source.stat().st_size))
    inputs = [
        dict(path=f"inputs/{name}", sha256=sha, bytes=(output / "inputs" / name).stat().st_size)
        for name, sha in originals.items()
    ]
    manifest = dict(
        schema_version=3,
        study_id="qqq_split_v3",
        frozen_at_utc=clock(),
        status="protocol_and_inputs_sealed_before_parameter_selection",
        prior_research_disclosure=config["prior_research_disclosure"],
        config_path="configs/qqq_split_v3.yaml",
        code_files=code_records,
        input_files=inputs,
        dependencies=runtime(),
        source_quality=quality,
        rate_quality=rate_quality,
    )
    write_json(output / "manifest.json", manifest)
    (output / "manifest.sha256").write_text(digest(output / "manifest.json") + "\n")
    verify_split_study(root, output)
    return manifest


def verify_split_study(root: Path, study: Path) -> dict:
    root, study = root.resolve(), study.resolve()
    _origins(root)
    manifest_path = study / "manifest.json"
    if digest(manifest_path) != (study / "manifest.sha256").read_text().strip():
        raise ValueError("Protocol manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 3 or manifest.get("study_id") != "qqq_split_v3":
        raise ValueError("Unsupported manifest")
    if {r["path"] for r in manifest["code_files"]} != set(CODE_FILES):
        raise ValueError("Incomplete numerical code closure")
    expected_snapshots = set()
    for record in manifest["code_files"]:
        relative = record["path"]
        expected_snapshots.add(relative)
        for path in (root / relative, study / "snapshot" / relative):
            if digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise ValueError(f"Frozen code changed: {relative}")
    for dirname, records in (
        ("snapshot", manifest["code_files"]),
        ("inputs", manifest["input_files"]),
    ):
        base = study / dirname
        if base.is_symlink() or any(p.is_symlink() for p in base.rglob("*")):
            raise ValueError("Frozen directories may not contain symlinks")
        actual = {p.relative_to(study).as_posix() for p in base.rglob("*") if p.is_file()}
        expected = {
            f"snapshot/{r['path']}" if dirname == "snapshot" else r["path"] for r in records
        }
        if actual != expected:
            raise ValueError("Frozen file inventory changed")
    for record in manifest["input_files"]:
        path = study / record["path"]
        if digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Frozen raw input changed")
    if runtime() != manifest["dependencies"]:
        raise ValueError("Frozen Python dependency versions changed")
    config = read_config(study / "snapshot" / manifest["config_path"])
    if (
        source_quality(
            study / "inputs/QQQ_daily.parquet", study / "inputs/QQQ_reference_daily.parquet", config
        )
        != manifest["source_quality"]
    ):
        raise ValueError("Derived source coverage differs from freeze")
    if audit_rates(study / "inputs") != manifest["rate_quality"]:
        raise ValueError("Derived rates differ from the declared frozen transformation")
    return manifest


def load_stage_inputs(study: Path, config: dict, stage: str):
    from trend_following.research_split_v3 import prepare_daily_panel

    end = pd.Timestamp(config["segments"][stage]["end"])
    frame = pd.read_parquet(study / "inputs/QQQ_daily.parquet")
    dates = _date_index(frame)
    lower = pd.Timestamp(config["source_contract"]["feature_warmup_start"])
    # Truncate before normalization/features/selection. No final test prices enter
    # development or validation computation, even if cached alongside them.
    frame = frame.loc[(dates >= lower) & (dates <= end)].copy()
    panel = prepare_daily_panel(frame)
    rates = []
    cutoff = (end + pd.Timedelta(hours=16)).tz_localize("America/New_York").tz_convert("UTC")
    for name in ("DFF", "DTB3"):
        obs = pd.read_csv(study / f"inputs/{name}_available.csv")
        obs["available_at"] = pd.to_datetime(obs["available_at"], utc=True)
        if "observation_date" in obs:
            obs["observation_date"] = pd.to_datetime(obs["observation_date"])
        rates.append(obs.loc[obs.available_at <= cutoff].copy())
    return panel, *rates


def _bounds(config: dict, stage: str) -> dict:
    return {k: v for k, v in config["segments"][stage].items() if k in ("start", "end")}


def _save_evaluation(directory: Path, result) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    result.metrics.to_csv(directory / "metrics.csv", index=False)
    result.daily_returns.to_csv(directory / "daily_returns.csv")
    result.target_weights.to_csv(directory / "targets.csv")
    write_json(directory / "metadata.json", result.metadata)
    terminals = {
        k: {field: value for field, value in terminal.items() if field != "ledger"}
        for k, terminal in result.terminal_snapshots.items()
    }
    write_json(directory / "terminal.json", terminals)
    for layer, path in result.paths.items():
        pd.DataFrame(
            dict(
                equity=path.equity,
                cash=path.cash,
                returns=path.returns,
                turnover=path.turnover,
                taxes_paid=path.taxes_paid,
            )
        ).to_csv(directory / f"{layer}_account.csv")
        path.events.to_csv(directory / f"{layer}_events.csv", index=False)
        path.holdings.to_parquet(directory / f"{layer}_holdings.parquet", index=False)


def select_split_candidate(root: Path, study: Path) -> dict:
    from trend_following.research_split_v3 import (
        evaluate_daily_segment,
        select_validation_candidate,
    )

    manifest = verify_split_study(root, study)
    config = read_config(study / "snapshot" / manifest["config_path"])
    if (study / "selection.json").exists() or (study / "selection_started.json").exists():
        raise ValueError("Selection already attempted; never silently overwrite a research run")
    write_json(
        study / "selection_started.json",
        dict(started_at_utc=clock(), manifest_sha256=digest(study / "manifest.json")),
    )
    grid, base = candidates(config), costs(config)
    # Development is diagnostic only, and remains frozen before any results.
    panel, financing, cash = load_stage_inputs(study, config, "development")
    development = []
    for spec in grid.values():
        result = evaluate_daily_segment(
            panel,
            financing,
            cash,
            spec,
            costs=base,
            layers=("expense",),
            **_bounds(config, "development"),
        )
        development.append(result.metrics)
    pd.concat(development, ignore_index=True).to_csv(study / "development_metrics.csv", index=False)
    panel, financing, cash = load_stage_inputs(study, config, "validation")
    selected, table = select_validation_candidate(
        panel,
        financing,
        cash,
        grid,
        costs=base,
        tie_tolerance=config["selection"]["tie_tolerance"],
        **_bounds(config, "validation"),
    )
    table.to_csv(study / "validation_candidates.csv", index=False)
    selection = dict(
        schema_version=3,
        selected_at_utc=clock(),
        selected_id=selected,
        selected_spec=grid[selected],
        manifest_sha256=digest(study / "manifest.json"),
        validation_table_sha256=digest(study / "validation_candidates.csv"),
        candidate_count=len(grid),
        selection=config["selection"],
        last_price_available_to_selection=config["segments"]["validation"]["end"],
        final_test_used=False,
        prior_research_disclosure=config["prior_research_disclosure"],
    )
    write_json(study / "selection.json", selection)
    (study / "selection.sha256").write_text(digest(study / "selection.json") + "\n")
    return selection


def verify_selection(root: Path, study: Path) -> tuple[dict, dict]:
    manifest = verify_split_study(root, study)
    selection_path = study / "selection.json"
    if digest(selection_path) != (study / "selection.sha256").read_text().strip():
        raise ValueError("Selection receipt changed")
    selection = json.loads(selection_path.read_text())
    config = read_config(study / "snapshot" / manifest["config_path"])
    if (
        selection["manifest_sha256"] != digest(study / "manifest.json")
        or selection["validation_table_sha256"] != digest(study / "validation_candidates.csv")
        or selection["selected_spec"] != candidates(config).get(selection["selected_id"])
        or selection["final_test_used"]
        or selection["selection"] != config["selection"]
    ):
        raise ValueError("Selection is inconsistent with the sealed protocol")
    table = pd.read_csv(study / "validation_candidates.csv", float_precision="round_trip")
    grid = candidates(config)
    if set(table.candidate_id) != set(grid) or table.candidate_id.duplicated().any():
        raise ValueError("Validation table does not cover the sealed candidate grid")
    for row in table.itertuples():
        spec = grid[row.candidate_id]
        if row.ma_sessions != spec["ma_sessions"] or row.allocation != spec["allocation"]:
            raise ValueError("Validation-table parameters differ from sealed candidates")
    finite = table.loc[np.isfinite(table.excess_return_Sharpe)]
    if finite.empty:
        raise ValueError("No finite validation candidate")
    best = float(finite.excess_return_Sharpe.max())
    tied = finite.loc[best - finite.excess_return_Sharpe <= config["selection"]["tie_tolerance"]]
    expected = (
        tied.sort_values(
            ["allocation", "ma_sessions", "candidate_id"], ascending=[True, False, True]
        )
        .iloc[0]
        .candidate_id
    )
    if selection["selected_id"] != expected:
        raise ValueError("Selected candidate does not follow the sealed validation-only rule")
    return config, selection


def paired_uncertainty(returns: pd.DataFrame, factors: tuple[float, float], spec: dict):
    values = returns.to_numpy(float)
    boundaries = np.asarray(factors, float)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or len(values) < 2
        or not np.isfinite(values).all()
        or (values <= -1).any()
        or not np.isfinite(boundaries).all()
        or (boundaries <= 0).any()
        or not returns.index.is_unique
        or not returns.index.is_monotonic_increasing
    ):
        raise ValueError("Finite aligned paired return streams and positive boundaries required")
    n, logs = len(values), np.log1p(values)
    observed = (np.prod(1 + values, axis=0) * boundaries - 1) * 100
    summaries, samples = [], []
    for block in spec["mean_block_sessions"]:
        count, seed = spec["simulations_per_block"], spec["seed"]
        if isinstance(block, bool) or int(block) != block or block < 1 or count < 1 or seed < 0:
            raise ValueError("Invalid fixed bootstrap settings")
        rng = np.random.default_rng(np.random.SeedSequence([seed, block]))
        positions = rng.integers(0, n, size=count)
        accumulated = logs[positions].copy()
        for _ in range(1, n):
            positions = np.where(
                rng.random(count) < 1 / block, rng.integers(0, n, size=count), (positions + 1) % n
            )
            accumulated += logs[positions]
        wealth = np.exp(accumulated) * boundaries
        delta = (wealth[:, 0] - wealth[:, 1]) * 100
        lo, median, hi = np.quantile(delta, [0.025, 0.5, 0.975])
        summaries.append(
            dict(
                block_sessions=block,
                simulations=count,
                seed=seed,
                observed_delta_pct_points=float(observed[0] - observed[1]),
                delta_percentile_025=float(lo),
                delta_median=float(median),
                delta_percentile_975=float(hi),
                role="primary" if block == 20 else "dependence_sensitivity",
                interpretation=spec["limitation"],
            )
        )
        samples.append(
            pd.DataFrame(
                dict(block_sessions=block, simulation=np.arange(count), delta_pct_points=delta)
            )
        )
    return pd.DataFrame(summaries), pd.concat(samples, ignore_index=True)


def evaluate_split_study(root: Path, study: Path) -> dict:
    from trend_following.research_split_v3 import bootstrap_components, evaluate_daily_segment

    config, selection = verify_selection(root, study)
    output = study / "evaluation"
    if output.exists() or (study / "evaluation_started.json").exists():
        raise ValueError("Final evaluation already attempted; existing artifacts are preserved")
    write_json(
        study / "evaluation_started.json",
        dict(started_at_utc=clock(), selection_sha256=digest(study / "selection.json")),
    )
    output.mkdir(exist_ok=False)
    panel, financing, cash = load_stage_inputs(study, config, "test")
    base = costs(config)
    specifications = {
        selection["selected_id"]: selection["selected_spec"],
        "qqq_buy_hold": dict(name="qqq_buy_hold", kind="buy_hold", asset="QQQ", allocation=1.0),
        "cash": dict(name="cash", kind="cash", asset="QQQ", allocation=0.0),
        "qqq200ma": dict(name="qqq200ma", kind="ma", asset="QQQ", allocation=1.0, ma_sessions=200),
    }
    results, metrics, annual = {}, [], []
    for key, specification in specifications.items():
        result = evaluate_daily_segment(
            panel,
            financing,
            cash,
            specification,
            costs=base,
            layers=tuple(config["evaluation"]["cost_layers"]),
            **_bounds(config, "test"),
        )
        results[key] = result
        _save_evaluation(output / "paths" / key, result)
        metrics.append(result.metrics)
        for layer in result.daily_returns:
            series = result.daily_returns[layer]
            for year, daily in series.groupby(series.index.year):
                wealth = (1 + daily).cumprod()
                peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
                annual.append(
                    dict(
                        candidate_id=key,
                        layer=layer,
                        year=int(year),
                        sessions=len(daily),
                        total_return_pct=float((wealth.iloc[-1] - 1) * 100),
                        max_drawdown=float(np.min(wealth.to_numpy() / peak - 1)),
                        period_type="partial_year"
                        if year == series.index[-1].year
                        else "calendar_year",
                    )
                )
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_metrics.to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame(annual).to_csv(output / "test_annual.csv", index=False)
    sensitivity = []
    for field, values in (
        ("slippage_bps", config["evaluation"]["slippage_sensitivity_bps"]),
        ("financing_spread_bps", config["evaluation"]["financing_spread_sensitivity_bps"]),
    ):
        for value in values:
            altered = replace(base, **{field: value})
            for key in (selection["selected_id"], "qqq_buy_hold"):
                result = evaluate_daily_segment(
                    panel,
                    financing,
                    cash,
                    specifications[key],
                    costs=altered,
                    layers=("expense",),
                    **_bounds(config, "test"),
                )
                rows = result.metrics.copy()
                rows["changed_assumption"], rows["assumption_value"] = field, value
                sensitivity.append(rows)
    pd.concat(sensitivity, ignore_index=True).to_csv(output / "cost_sensitivity.csv", index=False)
    a, af, _ = bootstrap_components(results[selection["selected_id"]], layer="expense")
    b, bf, _ = bootstrap_components(results["qqq_buy_hold"], layer="expense")
    if not a.index.equals(b.index):
        raise ValueError("Paired strategy and benchmark sessions differ")
    bootstrap, draws = paired_uncertainty(
        pd.DataFrame({"selected": a, "QQQ": b}), (af, bf), config["uncertainty"]
    )
    bootstrap.to_csv(output / "bootstrap_summary.csv", index=False)
    draws.to_csv(output / "bootstrap_samples.csv", index=False)
    selected_row = all_metrics.loc[
        (all_metrics.candidate_id == selection["selected_id"]) & (all_metrics.layer == "expense")
    ].iloc[0]
    benchmark_row = all_metrics.loc[
        (all_metrics.candidate_id == "qqq_buy_hold") & (all_metrics.layer == "expense")
    ].iloc[0]
    delta = float(selected_row.total_return_pct - benchmark_row.total_return_pct)
    if not np.isclose(delta, bootstrap.observed_delta_pct_points.iloc[0], atol=1e-8):
        raise ValueError("Bootstrap boundaries fail held-out terminal-wealth reconciliation")
    summary = dict(
        schema_version=3,
        study_id="qqq_split_v3",
        completed_at_utc=clock(),
        manifest_sha256=digest(study / "manifest.json"),
        selection_sha256=digest(study / "selection.json"),
        selected_id=selection["selected_id"],
        selected_spec=selection["selected_spec"],
        periods=config["segments"],
        primary_endpoint=config["evaluation"]["primary_endpoint"],
        primary_delta_pct_points=delta,
        source_quality=json.loads((study / "manifest.json").read_text())["source_quality"],
        test_metrics_rows=len(all_metrics),
        validation_configurations=len(candidates(config)),
        bootstrap_draws=len(draws),
        prior_research_disclosure=config["prior_research_disclosure"],
        interpretation="Retrospective held-out diagnostic, not live or previously untouched evidence",
        cost_assumptions=asdict(base),
        limitations=config["source_contract"],
    )
    write_json(output / "summary.json", summary)
    files = [
        dict(path=p.relative_to(output).as_posix(), sha256=digest(p), bytes=p.stat().st_size)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    write_json(output / "completion.json", dict(complete=True, files=files))
    return summary
