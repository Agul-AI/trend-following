"""Coordinator semantics: source-only proposal -> validation -> sealed winner -> test.

No signal choices are supplied here. The only policy/candidate input is the fresh
development agent's finished handoff. Held-out outcomes cannot feed that agent or
change the frozen grid. Kernel custody complements, not replaces, this protocol.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from trend_following import research_blind_guard_v4 as guard

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/qqq_blind_v4.yaml"
CODE = (
    "configs/qqq_blind_v4.yaml",
    "src/trend_following/research_blind_engine_v4.py",
    "src/trend_following/research_blind_guard_v4.py",
    "src/trend_following/research_blind_protocol_v4.py",
    "src/trend_following/research_ledger.py",
    "src/trend_following/research_rates.py",
    "scripts/run_qqq_blind_study.py",
    "scripts/run_qqq_blind_guard.py",
    "tests/test_research_blind_engine_v4.py",
    "tests/test_research_blind_guard_v4.py",
    "tests/test_research_blind_protocol_v4.py",
)
SAFE_ATTRIBUTES = {
    "iloc",
    "loc",
    "iat",
    "index",
    "columns",
    "values",
    "shape",
    "size",
    "empty",
    "dtype",
    "close",
    "dividend",
    "split",
    "total_return",
    "total_return_index",
    "price_return",
    "mean",
    "std",
    "var",
    "median",
    "min",
    "max",
    "sum",
    "prod",
    "abs",
    "diff",
    "shift",
    "pct_change",
    "rolling",
    "ewm",
    "expanding",
    "cumprod",
    "cumsum",
    "cummax",
    "cummin",
    "quantile",
    "percentile",
    "nanpercentile",
    "nanquantile",
    "nanmean",
    "nanstd",
    "nanmedian",
    "sqrt",
    "log",
    "log1p",
    "exp",
    "expm1",
    "power",
    "clip",
    "isfinite",
    "isnan",
    "where",
    "maximum",
    "minimum",
    "sign",
    "array",
    "asarray",
    "dot",
    "einsum",
    "diag",
    "linalg",
    "solve",
    "lstsq",
    "pinv",
    "norm",
    "correlate",
    "convolve",
    "corr",
    "corrcoef",
    "copy",
    "tail",
    "head",
    "to_numpy",
    "astype",
    "fillna",
    "dropna",
    "notna",
    "isna",
    "all",
    "any",
    "item",
    "get",
    "items",
    "keys",
    "ravel",
    "reshape",
    "flatten",
    "nan",
    "inf",
    "pi",
    "float64",
    "int64",
    "float32",
    "int32",
    "float",
    "int",
    "bool",
    "floor",
    "ceil",
    "isclose",
    "round",
    "isinf",
    "log10",
    "fabs",
    "copysign",
    "sin",
    "cos",
    "tan",
    "atan",
    "erf",
    "tanh",
    "degrees",
    "radians",
    "e",
    "tau",
    "year",
    "month",
    "day",
    "dayofweek",
    "weekday",
}
FORBIDDEN_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "breakpoint",
    "vars",
    "dir",
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return guard.sha256(Path(path))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def hash_value(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    def bad(value):
        raise ValueError(f"Nonfinite JSON value is forbidden: {value}")

    return json.loads(Path(path).read_text(), parse_constant=bad)


def write(path, value):
    guard._exclusive(Path(path), json.dumps(value, indent=2, allow_nan=False).encode() + b"\n")


def runtime():
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "PyYAML",
                "pyarrow",
                "tzdata",
                "pytz",
                "python-dateutil",
            )
        },
    }


def config():
    value = yaml.safe_load(CONFIG.read_text())
    if value["schema_version"] != 4 or value["study_id"] != "qqq_blind_v4":
        raise ValueError("Unsupported staged protocol")
    if value["periods"] != {
        "development": {"start": "2002-01-01", "end": "2011-12-31"},
        "validation": {"start": "2012-01-01", "end": "2015-12-31"},
        "test": {"start": "2016-01-01", "end": "2026-06-01"},
    }:
        raise ValueError("Fixed chronological split changed")
    if (
        value["selection"]["objective"]
        != "Pooled validation expense-inclusive pretax daily cash-excess Sharpe"
        or value["selection"]["tie_tolerance"] != 1e-10
    ):
        raise ValueError("Selection objective/tie rule changed")
    if value["evaluation"]["benchmarks"] != ["QQQ_buy_hold", "synthetic3x_buy_hold", "cash"]:
        raise ValueError("Predeclared controls changed")
    return value


def validate_policy(path: Path):
    """Positive pure-numerics surface; no mutable module state, I/O or randomness."""

    def immutable(x):
        return (
            isinstance(x, (str, int, float, bool, type(None)))
            or isinstance(x, tuple)
            and all(immutable(y) for y in x)
        )

    tree = ast.parse(path.read_text())
    found = False
    for top in tree.body:
        if isinstance(top, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            if isinstance(top, ast.FunctionDef) and top.name == "target":
                found = True
                if (
                    len(top.args.args) != 2
                    or top.args.defaults
                    or top.args.kwonlyargs
                    or top.args.vararg
                    or top.args.kwarg
                ):
                    raise ValueError(
                        "Target must take exactly history and parameters, without persistent defaults"
                    )
        elif (
            isinstance(top, ast.Expr)
            and isinstance(top.value, ast.Constant)
            and isinstance(top.value.value, str)
        ):
            pass
        elif isinstance(top, (ast.Assign, ast.AnnAssign)):
            try:
                value = ast.literal_eval(top.value)
            except (ValueError, TypeError) as exc:
                raise ValueError("Policy module constants must be immutable literals") from exc

            if not immutable(value):
                raise ValueError("Policy may not maintain mutable module-level state")
        else:
            raise ValueError("Policy top level must be declarations, not executable code")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.decorator_list:
                raise ValueError("Policy decorators/state wrappers are forbidden")
            for default in [
                *node.args.defaults,
                *[x for x in node.args.kw_defaults if x is not None],
            ]:
                try:
                    value = ast.literal_eval(default)
                except (ValueError, TypeError) as exc:
                    raise ValueError("Policy defaults must be immutable literals") from exc
                if not immutable(value):
                    raise ValueError("Policy defaults may not retain mutable state")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef)):
            raise ValueError("Policy must be stateless plain functions")
        if isinstance(node, ast.Import):
            if any(alias.name not in {"numpy", "pandas", "math"} for alias in node.names):
                raise ValueError("Policy import outside pure numeric libraries")
        if isinstance(node, ast.ImportFrom):
            if node.module not in {"numpy", "pandas", "math"} or any(
                alias.name not in SAFE_ATTRIBUTES for alias in node.names
            ):
                raise ValueError("Policy imported function outside numeric allowlist")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise ValueError("Policy I/O or introspection is forbidden")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_ATTRIBUTES:
            raise ValueError(f"Policy attribute outside pure numeric allowlist: {node.attr}")
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            raise ValueError("Policy may not mutate module/library attribute state")
    if not found:
        raise ValueError("Policy must define target(history, parameters)")


def validate_proposal(value: dict) -> dict:
    if value.get("schema_version") != 4 or not isinstance(value.get("candidates"), list):
        raise ValueError("Fresh handoff requires schema_version4 and a candidate list")
    rows = value["candidates"]
    if not 1 <= len(rows) <= 12:
        raise ValueError("Frozen candidate budget is 1..12")
    names = []
    for row in rows:
        identifier = row["candidate_id"]
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,79}", identifier):
            raise ValueError("Candidate ID must be a safe stable filename")
        if identifier in ("cash", "QQQ_buy_hold", "synthetic3x_buy_hold"):
            raise ValueError("Candidate cannot shadow a benchmark")
        names.append(identifier)
        window, rank = row["history_sessions"], row["complexity_rank"]
        if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= 1260:
            raise ValueError("Invalid declared history window")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("Invalid predeclared complexity tie-break")
        if not isinstance(row["parameters"], dict):
            raise ValueError("Parameters must be a fixed JSON object")
        canonical(row["parameters"])
    if len(set(names)) != len(names):
        raise ValueError("Duplicate frozen candidate IDs")
    return value


def verify_protocol(study: Path) -> dict:
    study, _ = guard._load(study)
    lock = read_json(study / "custodian/seals/protocol.json")
    _bind_event(
        study, "proposal_lock", "protocol_sha256", sha(study / "custodian/seals/protocol.json")
    )
    for record in lock["code_files"]:
        current, snapshot = (
            ROOT / record["path"],
            study / "custodian/code_snapshot" / record["path"],
        )
        if sha(current) != record["sha256"] or sha(snapshot) != record["sha256"]:
            raise ValueError("Pre-validation numerical source/configuration changed")
    if {record["path"] for record in lock["code_files"]} != set(CODE) or runtime() != lock[
        "dependencies"
    ]:
        raise ValueError("Numerical closure or environment changed")
    development = read_json(study / "custodian/DEVELOPMENT_SEALED.json")
    if sha(study / "custodian/DEVELOPMENT_SEALED.json") != lock["development_seal_sha256"]:
        raise ValueError("Development seal changed")
    for relative, expected in development["source_hashes"].items():
        if sha(study / "developer/workspace" / relative) != expected:
            raise ValueError("Research source changed after handoff")
    for relative, expected in development["output_hashes"].items():
        if sha(study / "developer/output" / relative) != expected:
            raise ValueError("Research output changed after handoff")
    for name, expected in lock["proposal_files"].items():
        if sha(study / "custodian/proposal" / name) != expected:
            raise ValueError("Sealed development proposal changed")
    proposal = validate_proposal(read_json(study / "custodian/proposal/proposal.json"))
    validate_policy(study / "custodian/proposal/policy.py")
    if hash_value(proposal["candidates"]) != lock["candidate_grid_sha256"]:
        raise ValueError("Candidate grid changed after validation lock")
    return lock


def _bind_event(study: Path, event: str, field: str, expected: str):
    guard._check_audit(study)
    matches = [
        json.loads(line)
        for line in (study / "custodian/audit.jsonl").read_text().splitlines()
        if json.loads(line).get("event") == event
    ]
    if len(matches) != 1 or matches[0].get(field) != expected:
        raise ValueError(f"Immutable custody event no longer binds {event}")


def _completed_outputs(study: Path, stage: str):
    completed = read_json(study / f"custodian/STAGE_{stage.upper()}_COMPLETED.json")
    guard._check_audit(study)
    events = [
        json.loads(line)
        for line in (study / "custodian/audit.jsonl").read_text().splitlines()
        if json.loads(line).get("event") == "held_stage_exit"
        and json.loads(line).get("stage") == stage
    ]
    if len(events) != 1 or any(
        completed.get(key) != events[0].get(key)
        for key in (
            "run_id",
            "returncode",
            "output_hashes",
            "input_hashes",
            "lock_sha256",
            "request_sha256",
            "proposal_sha256",
        )
    ):
        raise ValueError("Completed stage receipt differs from its custody event")
    if (
        completed["returncode"] != 0
        or guard._hash_tree(study / "operator_output" / stage) != completed["output_hashes"]
    ):
        raise ValueError("Completed held-stage output bytes changed")
    return completed


def lock_proposal(study: Path, policy: Path, proposal: Path, development_log: Path) -> dict:
    study, custody = guard._load(study)
    rules = config()
    workspace = study / "developer/workspace"
    for path in (policy, proposal, development_log):
        if not path.resolve().is_relative_to(workspace) or path.is_symlink():
            raise ValueError("Handoff must originate in the fresh developer workspace")
    value = validate_proposal(read_json(proposal))
    validate_policy(policy)
    # The research log is retained in full. Coordinator verifies coverage/budget;
    # it never recommends replacement signals based on any prior/later results.
    research_log = read_json(development_log)
    trials = research_log.get(
        "configurations", research_log.get("trials", research_log.get("experiments", []))
    )
    if not isinstance(trials, list) or not 1 <= len(trials) <= 32:
        raise ValueError("Handoff must enumerate all explored development configurations (<=32)")
    executable_keys = ("candidate_id", "parameters", "history_sessions", "complexity_rank")
    trial_keys = {hash_value({key: row[key] for key in executable_keys}) for row in trials}
    if not all(
        hash_value({key: row[key] for key in executable_keys}) in trial_keys
        for row in value["candidates"]
    ):
        raise ValueError("Frozen candidates must be accounted for in the development ledger")
    if value.get("policy_sha256", sha(policy)) != sha(policy) or value.get(
        "development_log_sha256", sha(development_log)
    ) != sha(development_log):
        raise ValueError("Research handoff references different policy/log bytes")
    guard.seal_study(study)
    folder = study / "custodian/proposal"
    folder.mkdir(exist_ok=False)
    for source, name in (
        (policy, "policy.py"),
        (proposal, "proposal.json"),
        (development_log, "development_log.json"),
    ):
        guard._exclusive(folder / name, source.read_bytes())
    records = []
    for relative in CODE:
        source = ROOT / relative
        destination = study / "custodian/code_snapshot" / relative
        guard._exclusive(destination, source.read_bytes())
        records.append(dict(path=relative, sha256=sha(source)))
    engine = custody["engine_hashes"]
    for source_name, seed_name in (
        ("research_blind_engine_v4.py", "engine.py"),
        ("research_ledger.py", "ledger_kernel.py"),
        ("research_rates.py", "rate_kernel.py"),
    ):
        if sha(ROOT / "src/trend_following" / source_name) != engine[seed_name]:
            raise ValueError("Research engine is not the numerical source being sealed")
    lock = dict(
        schema_version=4,
        locked_at_utc=now(),
        researcher_context="fresh fork_turns none",
        development_seal_sha256=sha(study / "custodian/DEVELOPMENT_SEALED.json"),
        candidate_grid_sha256=hash_value(value["candidates"]),
        proposal_files={
            name: sha(folder / name)
            for name in ("policy.py", "proposal.json", "development_log.json")
        },
        code_files=records,
        dependencies=runtime(),
        selection=rules["selection"],
        explored_configurations=len(trials),
        frozen_candidates=len(value["candidates"]),
        guard_scope=custody["tool_limit"],
    )
    write(study / "custodian/seals/protocol.json", lock)
    guard._append_audit(
        study,
        {
            "event": "proposal_lock",
            "protocol_sha256": sha(study / "custodian/seals/protocol.json"),
            "candidate_grid_sha256": lock["candidate_grid_sha256"],
        },
    )
    verify_protocol(study)
    return lock


def expected_winner(table: pd.DataFrame, proposal: dict):
    grid = {row["candidate_id"]: row for row in proposal["candidates"]}
    if (
        set(table.candidate_id) != set(grid)
        or table.candidate_id.duplicated().any()
        or not table.layer.eq("expense").all()
    ):
        raise ValueError("Validation metrics do not cover exactly the sealed grid/layer")
    scores = table.loc[np.isfinite(table.cash_excess_Sharpe)].copy()
    if scores.empty:
        raise ValueError("No finite validation scores; do not invent fallback or peek at test")
    best = float(scores.cash_excess_Sharpe.max())
    tied = scores.loc[best - scores.cash_excess_Sharpe <= 1e-10].copy()
    tied["complexity_rank"] = tied.candidate_id.map(
        {key: row["complexity_rank"] for key, row in grid.items()}
    )
    winner = tied.sort_values(["complexity_rank", "candidate_id"]).iloc[0].candidate_id
    return grid[winner]


def _request(study: Path, stage: str, candidates: list, layers: list, benchmarks: bool):
    rules = config()
    cost_keys = (
        "fee_bps",
        "slippage_bps",
        "financing_spread_bps",
        "expense_ratio",
        "extra_drag",
        "short_tax_rate",
        "long_tax_rate",
        "interest_tax_rate",
        "wash_sales",
    )
    return dict(
        input_dir=str(study / "released" / stage / "input"),
        policy_path=str(study / "custodian/proposal/policy.py"),
        output=str(study / "operator_output" / stage / "results"),
        start=rules["periods"][stage]["start"],
        end=rules["periods"][stage]["end"],
        candidates=candidates,
        layers=layers,
        benchmarks=benchmarks,
        costs={
            "initial_cash": rules["execution"]["initial_cash"],
            **{name: rules["costs"][name] for name in cost_keys},
        },
    )


def _gate_lock(study: Path, stage: str, request_path: Path, selection_path: Path | None = None):
    files = {
        str(study / "custodian/proposal" / name): sha(study / "custodian/proposal" / name)
        for name in ("policy.py", "proposal.json", "development_log.json")
    }
    files[str(request_path)] = sha(request_path)
    lock = dict(
        stage=stage,
        development_seal_sha256=sha(study / "custodian/DEVELOPMENT_SEALED.json"),
        proposal_path=str(study / "custodian/proposal/proposal.json"),
        proposal_sha256=sha(study / "custodian/proposal/proposal.json"),
        request_path=str(request_path),
        files=files,
    )
    if stage == "test":
        files[str(selection_path)] = sha(selection_path)
        lock.update(
            validation_lock_sha256=sha(study / "custodian/seals/validation_lock.json"),
            selection_receipt_path=str(selection_path),
            selection_receipt_sha256=sha(selection_path),
        )
    path = study / "custodian/seals" / f"{stage}_lock.json"
    write(path, lock)
    return path


def run_validation(study: Path) -> dict:
    study = study.resolve()
    protocol = verify_protocol(study)
    proposal = read_json(study / "custodian/proposal/proposal.json")
    request = _request(study, "validation", proposal["candidates"], ["expense"], False)
    request_path = study / "custodian/seals/validation_request.json"
    write(request_path, request)
    gate = _gate_lock(study, "validation", request_path)
    result = guard.gate_release(study, "validation", gate, timeout_seconds=1800)
    if result["returncode"] != 0:
        raise ValueError("Validation failed; retain attempt, do not change rules or open test")
    table_path = study / "operator_output/validation/results/all_metrics.csv"
    table = pd.read_csv(table_path, float_precision="round_trip")
    selected = expected_winner(table, proposal)
    selection = dict(
        schema_version=4,
        selected_at_utc=now(),
        selected=selected,
        protocol_sha256=sha(study / "custodian/seals/protocol.json"),
        candidate_grid_sha256=protocol["candidate_grid_sha256"],
        policy_sha256=sha(study / "custodian/proposal/policy.py"),
        validation_table_sha256=sha(table_path),
        validation_gate_sha256=sha(gate),
        selection_rule=config()["selection"],
        test_used=False,
    )
    path = study / "custodian/seals/selection.json"
    write(path, selection)
    guard._append_audit(
        study,
        {
            "event": "winner_lock",
            "selection_sha256": sha(path),
            "selected_id": selected["candidate_id"],
        },
    )
    verify_selection(study)
    return selection


def verify_selection(study: Path):
    protocol = verify_protocol(study)
    selection = read_json(study / "custodian/seals/selection.json")
    _bind_event(
        study, "winner_lock", "selection_sha256", sha(study / "custodian/seals/selection.json")
    )
    _completed_outputs(study, "validation")
    table_path = study / "operator_output/validation/results/all_metrics.csv"
    proposal = read_json(study / "custodian/proposal/proposal.json")
    if (
        selection["protocol_sha256"] != sha(study / "custodian/seals/protocol.json")
        or selection["candidate_grid_sha256"] != protocol["candidate_grid_sha256"]
        or selection["policy_sha256"] != sha(study / "custodian/proposal/policy.py")
        or selection["validation_table_sha256"] != sha(table_path)
        or selection["validation_gate_sha256"]
        != sha(study / "custodian/seals/validation_lock.json")
        or selection["selection_rule"] != config()["selection"]
        or selection["test_used"]
    ):
        raise ValueError("Selection or proposal changed after validation")
    expected = expected_winner(pd.read_csv(table_path, float_precision="round_trip"), proposal)
    if selection["selected"] != expected:
        raise ValueError("Winner does not belong to/follow the original sealed candidate grid")
    return selection


def run_test(study: Path) -> dict:
    study = study.resolve()
    selection = verify_selection(study)
    request = _request(
        study, "test", [selection["selected"]], config()["evaluation"]["cost_layers"], True
    )
    request_path = study / "custodian/seals/test_request.json"
    write(request_path, request)
    gate = _gate_lock(study, "test", request_path, study / "custodian/seals/selection.json")
    result = guard.gate_release(study, "test", gate, timeout_seconds=1800)
    if result["returncode"] != 0:
        raise ValueError("Test failed; preserve attempt without retuning/reselection")
    metrics_path = study / "operator_output/test/results/all_metrics.csv"
    metrics = pd.read_csv(metrics_path, float_precision="round_trip")
    net = metrics.loc[metrics.layer.eq("expense")].set_index("candidate_id")
    identifier = selection["selected"]["candidate_id"]
    summary = dict(
        schema_version=4,
        completed_at_utc=now(),
        selected=selection["selected"],
        protocol_sha256=sha(study / "custodian/seals/protocol.json"),
        selection_sha256=sha(study / "custodian/seals/selection.json"),
        test_metrics_sha256=sha(metrics_path),
        primary_cash_excess_sharpe_difference=float(
            net.loc[identifier, "cash_excess_Sharpe"]
            - net.loc["QQQ_buy_hold", "cash_excess_Sharpe"]
        ),
        test_sessions=int(net.loc[identifier, "sessions"]),
        development_context="fresh agent, no inherited project turns, only development packet and generic accounting kernel",
        no_test_retuning=True,
        scope="Historical OOS with staged file/process restrictions; not live performance or erased human background knowledge",
    )
    write(study / "custodian/seals/completion.json", summary)
    guard._append_audit(
        study,
        {
            "event": "test_complete",
            "summary_sha256": sha(study / "custodian/seals/completion.json"),
        },
    )
    return summary
