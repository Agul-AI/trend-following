"""Split-before-selection integrity and boundary tests; artificial inputs only."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from trend_following import research_split_protocol_v3 as protocol

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return yaml.safe_load((ROOT / "configs/qqq_split_v3.yaml").read_text())


def test_declared_split_is_contiguous_and_candidates_are_new(tmp_path):
    config = configuration()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    assert protocol.read_config(path) == config
    grid = protocol.candidates(config)
    assert len(grid) == 9
    assert {spec["ma_sessions"] for spec in grid.values()} == {50, 100, 200}
    assert all(spec["kind"] == "ma" for spec in grid.values())
    assert not any("b5" in str(spec).lower() for spec in grid.values())


@pytest.mark.parametrize(
    "defect", ["overlap", "gap", "duplicate_window", "future_retraining", "drawdown_screen"]
)
def test_invalid_protocol_fails(tmp_path, defect):
    config = configuration()
    if defect == "overlap":
        config["segments"]["test"]["start"] = "2015-12-31"
    elif defect == "gap":
        config["segments"]["test"]["start"] = "2016-01-02"
    elif defect == "duplicate_window":
        config["policy"]["ma_sessions"] = [50, 50, 200]
    elif defect == "future_retraining":
        config["evaluation"]["retraining_during_test"] = True
    else:
        config["selection"]["drawdown_screen"] = -0.5
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        protocol.read_config(path)


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    root, inputs, study = tmp_path / "checkout", tmp_path / "original", tmp_path / "study"
    config = configuration()
    config["segments"] = {
        "development": dict(start="2002-01-01", end="2002-01-31"),
        "validation": dict(start="2002-02-01", end="2002-02-28"),
        "test": dict(start="2002-03-01", end="2002-03-31"),
    }
    for relative in protocol.CODE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# artificial hash-bound source\n")
    config_path = root / "configs/qqq_split_v3.yaml"
    config_path.write_text(yaml.safe_dump(config))
    inputs.mkdir()
    dates = pd.bdate_range("2000-01-01", "2002-03-31")
    data = pd.DataFrame(
        dict(
            date=dates,
            close=np.linspace(100, 120, len(dates)),
            adj_close=np.linspace(100, 120, len(dates)),
            dividend_amount=0.0,
            split_coefficient=1.0,
        )
    )
    data.to_parquet(inputs / "QQQ.parquet")
    data.to_parquet(inputs / "reference.parquet")
    from trend_following.research_rates import prepare_rate_observations

    for name in ("DFF", "DTB3"):
        raw = pd.DataFrame({"DATE": pd.bdate_range("1999-12-01", "2002-03-31"), name: 2.0})
        raw.to_csv(inputs / f"{name}_raw.csv", index=False)
        prepare_rate_observations(raw, name).to_csv(inputs / f"{name}_available.csv", index=False)
    (inputs / "metadata.json").write_text("{}\n")
    monkeypatch.setattr(protocol, "_origins", lambda root: None)
    protocol.freeze_split_study(
        root, config_path, inputs / "QQQ.parquet", inputs / "reference.parquet", inputs, study
    )
    return root, inputs, study, config


def test_freeze_has_no_selection_or_test_results(sealed):
    root, _, study, _ = sealed
    manifest = protocol.verify_split_study(root, study)
    assert manifest["status"] == "protocol_and_inputs_sealed_before_parameter_selection"
    assert not (study / "selection.json").exists()
    assert not (study / "evaluation").exists()


@pytest.mark.parametrize("target", ["code", "snapshot", "input", "manifest", "extra", "symlink"])
def test_post_freeze_changes_fail(sealed, target):
    root, _, study, _ = sealed
    if target == "code":
        (root / "scripts/run_qqq_split_study.py").write_text("# changed\n")
    elif target == "snapshot":
        (study / "snapshot/pyproject.toml").write_text("# changed\n")
    elif target == "input":
        (study / "inputs/DFF_available.csv").write_text("changed\n")
    elif target == "manifest":
        path = study / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["frozen_at_utc"] = "2000-01-01T00:00:00Z"
        path.write_text(json.dumps(manifest))
    elif target == "extra":
        (study / "inputs/extra.csv").write_text("1\n")
    else:
        (study / "inputs/link").symlink_to(study / "inputs/DFF_available.csv")
    with pytest.raises(ValueError):
        protocol.verify_split_study(root, study)


def test_stage_loader_excludes_final_test_before_features(sealed, monkeypatch):
    from trend_following import research_split_v3

    _, _, study, config = sealed
    monkeypatch.setattr(research_split_v3, "prepare_daily_panel", lambda frame: frame)
    frame, _, _ = protocol.load_stage_inputs(study, config, "validation")
    assert pd.to_datetime(frame.date).max() == pd.Timestamp("2002-02-28")
    assert not (pd.to_datetime(frame.date) >= pd.Timestamp("2002-03-01")).any()


def test_selection_never_receives_test_data_and_refuses_overwrite(sealed, monkeypatch):
    from trend_following import research_split_v3

    root, _, study, _ = sealed
    monkeypatch.setattr(research_split_v3, "prepare_daily_panel", lambda frame: frame)

    def evaluate(panel, *args, **kwargs):
        assert pd.to_datetime(panel.date).max() <= pd.Timestamp("2002-01-31")
        return SimpleNamespace(
            metrics=pd.DataFrame([dict(candidate_id=args[2]["name"], score=0.0)])
        )

    def select(panel, *args, **kwargs):
        assert pd.to_datetime(panel.date).max() == pd.Timestamp("2002-02-28")
        grid = args[2]
        key = "ma200_alloc_1_3"
        return key, pd.DataFrame(
            [
                dict(
                    candidate_id=k,
                    ma_sessions=s["ma_sessions"],
                    allocation=s["allocation"],
                    excess_return_Sharpe=0.0,
                )
                for k, s in grid.items()
            ]
        )

    monkeypatch.setattr(research_split_v3, "evaluate_daily_segment", evaluate)
    monkeypatch.setattr(research_split_v3, "select_validation_candidate", select)
    result = protocol.select_split_candidate(root, study)
    assert result["final_test_used"] is False
    assert result["last_price_available_to_selection"] == "2002-02-28"
    protocol.verify_selection(root, study)
    with pytest.raises(ValueError, match="already attempted"):
        protocol.select_split_candidate(root, study)


def test_source_date_gap_fails_before_performance(tmp_path):
    config = configuration()
    frame = pd.DataFrame(
        dict(
            date=pd.bdate_range("1999-11-01", "2026-06-01"),
            close=100.0,
            adj_close=100.0,
            dividend_amount=0.0,
            split_coefficient=1.0,
        )
    )
    frame.to_parquet(tmp_path / "qqq.parquet")
    frame.loc[frame.date != pd.Timestamp("2004-11-30")].to_parquet(tmp_path / "reference.parquet")
    with pytest.raises(ValueError, match="coverage differs"):
        protocol.source_quality(tmp_path / "qqq.parquet", tmp_path / "reference.parquet", config)


def test_rehashed_prepared_rate_with_wrong_lag_still_fails(sealed):
    root, _, study, _ = sealed
    file = study / "inputs/DFF_available.csv"
    supplied = pd.read_csv(file)
    supplied["available_at"] = pd.to_datetime(supplied.available_at, utc=True) - pd.Timedelta(
        hours=1
    )
    supplied.to_csv(file, index=False)
    manifest_path = study / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = next(r for r in manifest["input_files"] if r["path"] == "inputs/DFF_available.csv")
    record.update(sha256=protocol.digest(file), bytes=file.stat().st_size)
    manifest_path.write_text(json.dumps(manifest))
    (study / "manifest.sha256").write_text(protocol.digest(manifest_path) + "\n")
    with pytest.raises(ValueError, match="lag/conversion"):
        protocol.verify_split_study(root, study)


def test_resealed_valid_candidate_that_loses_tie_rule_is_rejected(sealed):
    root, _, study, config = sealed
    grid = protocol.candidates(config)
    table = pd.DataFrame(
        [
            dict(
                candidate_id=k,
                ma_sessions=s["ma_sessions"],
                allocation=s["allocation"],
                excess_return_Sharpe=0.0,
            )
            for k, s in grid.items()
        ]
    )
    table.to_csv(study / "validation_candidates.csv", index=False)
    wrong = "ma50_alloc_1_3"  # All scores tied; the declared winner must be MA200.
    selection = dict(
        selected_id=wrong,
        selected_spec=grid[wrong],
        manifest_sha256=protocol.digest(study / "manifest.json"),
        validation_table_sha256=protocol.digest(study / "validation_candidates.csv"),
        selection=config["selection"],
        final_test_used=False,
    )
    protocol.write_json(study / "selection.json", selection)
    (study / "selection.sha256").write_text(protocol.digest(study / "selection.json") + "\n")
    with pytest.raises(ValueError, match="validation-only rule"):
        protocol.verify_selection(root, study)


def test_paired_bootstrap_identity_and_boundaries_once():
    dates = pd.date_range("2020-01-01", periods=20, tz="UTC")
    returns = pd.DataFrame({"strategy": 0.001, "benchmark": 0.001}, index=dates)
    spec = dict(
        mean_block_sessions=[20, 60], simulations_per_block=100, seed=42, limitation="conditional"
    )
    summary, draws = protocol.paired_uncertainty(returns, (0.999, 0.999), spec)
    assert np.allclose(summary.observed_delta_pct_points, 0.0)
    assert np.allclose(draws.delta_pct_points, 0.0)
    altered, samples = protocol.paired_uncertainty(returns, (0.99, 1.0), spec)
    expected = (1.001**20) * -0.01 * 100
    assert np.allclose(altered.observed_delta_pct_points, expected)
    assert np.allclose(samples.delta_pct_points, expected)
