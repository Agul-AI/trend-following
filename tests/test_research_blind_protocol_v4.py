"""Pure-source semantics and staged end-to-end tests on fabricated data only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend_following import research_blind_guard_v4 as guard
from trend_following import research_blind_protocol_v4 as protocol


def proposal():
    return dict(
        schema_version=4,
        candidates=[
            dict(
                candidate_id="a", parameters={"weight": 0.1}, history_sessions=1, complexity_rank=1
            ),
            dict(
                candidate_id="b", parameters={"weight": 0.2}, history_sessions=1, complexity_rank=2
            ),
        ],
    )


def test_candidate_grid_finite_safe_budget():
    value = proposal()
    assert protocol.validate_proposal(value) == value
    for mutation in [
        lambda v: v.update(schema_version=3),
        lambda v: v.update(candidates=[]),
        lambda v: v["candidates"][0].update(candidate_id="../bad"),
        lambda v: v["candidates"][0].update(history_sessions=0),
        lambda v: v["candidates"][0].update(complexity_rank=-1),
    ]:
        copy = json.loads(json.dumps(value))
        mutation(copy)
        with pytest.raises(ValueError):
            protocol.validate_proposal(copy)


@pytest.mark.parametrize(
    "body",
    [
        "import os",
        "import random",
        "import pathlib",
        "pd.read_json('x')",
        "np.fromfile('x')",
        "np.random.normal()",
        "open('x')",
        "x.__class__",
        "global x",
        "nonlocal x",
    ],
)
def test_pure_policy_cannot_read_future_files_clock_or_mutable_state(tmp_path, body):
    file = tmp_path / "policy.py"
    file.write_text(
        "import numpy as np\nimport pandas as pd\ndef target(history,parameters):\n    "
        + body
        + "\n    return 0.\n"
    )
    with pytest.raises(ValueError):
        protocol.validate_policy(file)


def test_policy_top_level_mutable_state_rejected(tmp_path):
    file = tmp_path / "policy.py"
    file.write_text('CACHE={}\ndef target(history,parameters):\n    CACHE["x"]=1\n    return 0.\n')
    with pytest.raises(ValueError):
        protocol.validate_policy(file)


@pytest.mark.parametrize(
    "extra",
    [
        "def helper(state=[]):\n    state.append(1)\n    return 0.\n",
        "def helper(): return 0.\n@helper\ndef decorated(): return 0.\n",
        "import numpy as np\ndef helper():\n    np.sqrt = float\n    return 0.\n",
    ],
)
def test_persistent_defaults_decorators_library_mutation_rejected(tmp_path, extra):
    file = tmp_path / "policy.py"
    file.write_text(extra + "def target(history,parameters): return 0.\n")
    with pytest.raises(ValueError):
        protocol.validate_policy(file)


def test_data_only_numeric_policy_is_accepted(tmp_path):
    file = tmp_path / "policy.py"
    file.write_text(
        'import numpy as np\nVERSION=1\ndef target(history,parameters):\n    values=history["total_return"].to_numpy()\n    return float(np.clip(values.mean(),0.,1.))\n'
    )
    protocol.validate_policy(file)


def test_negative_validation_best_and_declared_tie_not_drawdown_rescue():
    rows = pd.DataFrame(
        [
            dict(candidate_id="a", layer="expense", cash_excess_Sharpe=-2.0, CAGR=-0.8),
            dict(candidate_id="b", layer="expense", cash_excess_Sharpe=-1.0, CAGR=-0.9),
        ]
    )
    assert protocol.expected_winner(rows, proposal())["candidate_id"] == "b"
    rows.cash_excess_Sharpe = -1.0
    assert protocol.expected_winner(rows, proposal())["candidate_id"] == "a"
    rows.cash_excess_Sharpe = np.nan
    with pytest.raises(ValueError, match="No finite"):
        protocol.expected_winner(rows, proposal())


def test_validation_cannot_introduce_new_candidate():
    rows = pd.DataFrame(
        [
            dict(candidate_id="a", layer="expense", cash_excess_Sharpe=0.1),
            dict(candidate_id="new", layer="expense", cash_excess_Sharpe=2.0),
        ]
    )
    with pytest.raises(ValueError):
        protocol.expected_winner(rows, proposal())


@pytest.mark.skipif(
    not Path("/usr/bin/sandbox-exec").is_file(), reason="requires macOS numerical sandbox"
)
def test_actual_coordinator_seals_before_validation_and_selected_before_test(tmp_path, monkeypatch):
    from trend_following.research_rates import prepare_rate_observations

    real_root = protocol.ROOT
    fake_root = tmp_path / "checkout"
    for relative in protocol.CODE:
        destination = fake_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(real_root / relative, destination)
    monkeypatch.setattr(protocol, "ROOT", fake_root)
    monkeypatch.setattr(protocol, "CONFIG", fake_root / "configs/qqq_blind_v4.yaml")
    seed = tmp_path / "engine"
    seed.mkdir()
    for source, name in [
        ("research_blind_engine_v4.py", "engine.py"),
        ("research_ledger.py", "ledger_kernel.py"),
        ("research_rates.py", "rate_kernel.py"),
    ]:
        shutil.copyfile(real_root / "src/trend_following" / source, seed / name)
    original = tmp_path / "source"
    original.mkdir()
    dates = pd.to_datetime(
        ["2002-01-02", "2002-01-03", "2012-01-03", "2012-01-04", "2016-01-04", "2016-01-05"]
    )
    raw = pd.DataFrame(
        {"date": dates, "close": 100.0, "dividend_amount": 0.0, "split_coefficient": 1.0}
    )
    raw.to_parquet(original / "q.parquet")
    raw[["date"]].to_parquet(original / "dates.parquet")
    sources = {"qqq": original / "q.parquet", "sessions": original / "dates.parquet"}
    for name in ["DFF", "DTB3"]:
        obs = pd.DataFrame(
            {"observation_date": pd.bdate_range("1999-01-01", "2016-01-05"), name: 2.0}
        )
        obs.to_csv(original / f"{name}_raw.csv", index=False)
        prepared = prepare_rate_observations(obs, name)
        prepared.to_csv(original / f"{name}_available.csv", index=False)
        for suffix in ["raw", "available"]:
            sources[f"{name}_{suffix}.csv"] = original / f"{name}_{suffix}.csv"
    (original / "metadata.json").write_text("{}\n")
    sources["metadata.json"] = original / "metadata.json"
    study = tmp_path / "blind"
    guard.prepare_study(study, seed, sources=sources)
    prior = tmp_path / "prior.txt"
    prior.write_text("forbidden synthetic artifact")
    guard.probe_guard(study, prior)
    workspace = study / "developer/workspace"
    policy = workspace / "policy.py"
    policy.write_text('def target(history, parameters):\n    return float(parameters["weight"])\n')
    handoff = workspace / "proposal.json"
    handoff.write_text(json.dumps(proposal()))
    log = workspace / "development_log.json"
    log.write_text(json.dumps({"experiments": proposal()["candidates"]}))
    protocol.lock_proposal(study, policy, handoff, log)
    assert not (study / "released/validation").exists()
    selection = protocol.run_validation(study)
    assert selection["test_used"] is False
    assert not (study / "released/test").exists()
    assert selection["selected"] in proposal()["candidates"]
    summary = protocol.run_test(study)
    assert summary["test_sessions"] == 2
    assert summary["no_test_retuning"] is True
    protocol.verify_selection(study)
    with pytest.raises((ValueError, guard.GuardError, FileExistsError)):
        protocol.run_validation(study)
    with pytest.raises((ValueError, guard.GuardError, FileExistsError)):
        protocol.run_test(study)
    # Rewriting a locally self-consistent receipt is rejected against the earlier
    # coordinator custody anchor, even before checking rank/parameter membership.
    receipt = study / "custodian/seals/selection.json"
    saved = receipt.read_bytes()
    receipt.chmod(0o600)
    amended = json.loads(saved)
    amended["selected_at_utc"] = "2099-01-01T00:00:00Z"
    receipt.write_text(json.dumps(amended))
    with pytest.raises(ValueError, match="custody event"):
        protocol.verify_selection(study)
    receipt.write_bytes(saved)
    protocol.verify_selection(study)
    # Completed-stage data cannot silently be replaced while keeping selection
    # metadata ostensibly unchanged.
    table_path = study / "operator_output/validation/results/all_metrics.csv"
    saved_table = table_path.read_bytes()
    table_path.write_bytes(saved_table + b"\n")
    with pytest.raises(ValueError, match="output bytes"):
        protocol.verify_selection(study)
    table_path.write_bytes(saved_table)
    # Alter a frozen candidate after seeing results: it must fail before reexecution.
    file = study / "custodian/proposal/proposal.json"
    file.chmod(0o600)
    value = json.loads(file.read_text())
    value["candidates"][0]["parameters"]["weight"] = 0.9
    file.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        protocol.verify_protocol(study)
