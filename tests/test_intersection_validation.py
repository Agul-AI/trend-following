"""Synthetic intersection-selection and validation-only protocol tests."""

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend_following import research_constant_cash as fixed
from trend_following import research_intersection_validation as study


def synthetic_metrics():
    return pd.DataFrame(
        {
            "candidate_id": candidate.candidate_id,
            "entry_rule": candidate.entry_rule.rule_id,
            "exit_rule": candidate.exit_rule.rule_id,
            "entry_confirmation_hours": hours,
            "exit_confirmation_hours": hours,
            "cash_excess_sharpe": 0.6,
            "trade_count": index,
        }
        for hours in fixed.WAIT_HOURS
        for index, candidate in enumerate(fixed.candidates_for_wait(hours))
    )


def authorization(tmp_path, **changes):
    value = {
        "schema": "trend_following_intersection_validation_authorization_v1",
        "authorized_stages": ["validation_only"],
        "explicit_user_approval": True,
        "authorized_by": "human_user",
        "user_request": study.USER_REQUEST,
        "historical_oos_authorized": False,
        "shared_wait_hours": [2.0, 3.0, 4.0],
        "cash_nominal_annual_rate": 0.03,
        **changes,
    }
    path = tmp_path / "human_authorization.json"
    path.write_text(json.dumps(value))
    return path


def pairs():
    return study.select_shared_pairs(synthetic_metrics(), 0.5)[0]


def test_select_shared_pairs_preserves_entire_289_and_equal_waits():
    selected, summary = study.select_shared_pairs(synthetic_metrics(), 0.5)
    assert len(selected) == 289
    assert summary["candidate_count"] == 867
    assert summary["per_wait_beating_BH_count"] == {"2.0": 289, "3.0": 289, "4.0": 289}
    assert summary["uses_validation_or_OOS_outcomes"] is False
    assert summary["filters_gap_flags"] is False
    all_ids = set()
    for hours in study.WAIT_HOURS:
        candidates = study.candidates_from_pairs(selected, hours)
        assert len(candidates) == 289
        assert all(
            c.entry_confirmation_hours == c.exit_confirmation_hours == hours for c in candidates
        )
        assert all(
            c.entry_required_checks == c.exit_required_checks == 1 + 2 * hours for c in candidates
        )
        all_ids.update(c.candidate_id for c in candidates)
    assert len(all_ids) == 867


def test_intersection_requires_all_three_scores_not_union_or_any():
    metrics = synthetic_metrics()
    for hours, index in zip(study.WAIT_HOURS, (0, 1, 2), strict=True):
        candidate = fixed.candidates_for_wait(hours)[index]
        metrics.loc[metrics.candidate_id.eq(candidate.candidate_id), "cash_excess_sharpe"] = 0.4
    selected, summary = study.select_shared_pairs(metrics, 0.5)
    assert len(selected) == 286
    assert set(summary["per_wait_beating_BH_count"].values()) == {288}
    excluded = {study._pair_id(c) for c in fixed.candidates_for_wait(2)[:3]}
    assert excluded.isdisjoint({pair["pair_id"] for pair in selected})


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf, 0.5, 0.5 + 5e-11, 0.5 - 1e-9])
def test_finiteness_and_strict_tolerance_on_each_wait(score):
    metrics = synthetic_metrics()
    candidate = fixed.candidates_for_wait(3)[0]
    metrics.loc[metrics.candidate_id.eq(candidate.candidate_id), "cash_excess_sharpe"] = score
    selected, _ = study.select_shared_pairs(metrics, 0.5)
    assert len(selected) == 288
    assert study._pair_id(candidate) not in {pair["pair_id"] for pair in selected}


def test_scores_outside_retained_waits_do_not_affect_intersection():
    metrics = synthetic_metrics()
    metrics.loc[
        ~metrics.entry_confirmation_hours.isin(study.WAIT_HOURS), "cash_excess_sharpe"
    ] = -9999
    assert len(study.select_shared_pairs(metrics, 0.5)[0]) == 289


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "extra", "unequal_wait", "wrong_rule", "wrong_id"]
)
def test_selection_rejects_partial_or_wrong_canonical_grid(corruption):
    metrics = synthetic_metrics()
    if corruption == "missing":
        metrics = metrics.iloc[1:]
    elif corruption == "duplicate":
        metrics.loc[1, "candidate_id"] = metrics.loc[0, "candidate_id"]
    elif corruption == "extra":
        metrics = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    elif corruption == "unequal_wait":
        metrics.loc[0, "exit_confirmation_hours"] = 3
    elif corruption == "wrong_rule":
        metrics.loc[0, "entry_rule"] = "ema_200"
    else:
        metrics.loc[0, "candidate_id"] = "unrecognized"
    with pytest.raises(study.Error):
        study.select_shared_pairs(metrics, 0.5)


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf])
def test_undefined_development_benchmark_rejected(score):
    with pytest.raises(study.Error):
        study.select_shared_pairs(synthetic_metrics(), score)


def test_empty_intersection_stops_before_validation():
    metrics = synthetic_metrics()
    metrics["cash_excess_sharpe"] = 0.1
    with pytest.raises(study.Error, match="No shared"):
        study.select_shared_pairs(metrics, 0.5)


@pytest.mark.parametrize("invalid", [False, True, 0, 1, 2.5, 6, 6.5, np.nan])
def test_no_unretained_or_asymmetric_waits(invalid):
    with pytest.raises(study.Error):
        study.candidates_from_pairs(pairs(), invalid)


@pytest.mark.parametrize(
    "corruption", ["unsorted", "duplicate", "wrong_id", "wrong_parameters", "empty"]
)
def test_frozen_pairs_must_be_canonical_unique_sorted(corruption):
    selected = pairs()
    if corruption == "unsorted":
        selected = selected[::-1]
    elif corruption == "duplicate":
        selected.append(selected[-1])
    elif corruption == "wrong_id":
        selected[0]["pair_id"] = "bad"
    elif corruption == "wrong_parameters":
        selected[0]["entry_rule"]["period"] = 10
    else:
        selected = []
    with pytest.raises(study.Error):
        study.candidates_from_pairs(selected, 2)


def test_authorization_accepts_exact_current_human_request(tmp_path):
    assert study._authorization(authorization(tmp_path))["authorized_stages"] == ["validation_only"]


@pytest.mark.parametrize(
    "changes",
    [
        {"authorized_stages": ["validation_only", "historical_oos"]},
        {"historical_oos_authorized": True},
        {"historical_oos_authorized": 0},
        {"explicit_user_approval": 1},
        {"explicit_user_approval": False},
        {"cash_nominal_annual_rate": 0.02},
        {"shared_wait_hours": [2, 3]},
        {"user_request": "run historical OOS"},
        {"authorized_by": "agent"},
    ],
)
def test_authorization_forbids_wrong_stage_cash_wait_and_implicit_approval(tmp_path, changes):
    with pytest.raises(study.Error):
        study._authorization(authorization(tmp_path, **changes))


def test_validation_counts_keep_undefined_in_full_denominator_and_median_all_pairs():
    selected = pairs()[:5]
    metrics = pd.DataFrame(
        {
            "candidate_id": c.candidate_id,
            "entry_rule": c.entry_rule.rule_id,
            "exit_rule": c.exit_rule.rule_id,
            "entry_confirmation_hours": hours,
            "exit_confirmation_hours": hours,
            "cash_excess_sharpe": [0.7, np.nan, np.inf, 0.5 + 5e-11, 0.3][index],
            "trade_count": index * 10,
        }
        for hours in study.WAIT_HOURS
        for index, c in enumerate(study.candidates_from_pairs(selected, hours))
    )
    counts = study.counts_against_bh(metrics, 0.5, selected)
    assert counts.configuration_count.eq(5).all()
    assert counts.finite_Sharpe_count.eq(3).all()
    assert counts.undefined_Sharpe_count.eq(2).all()
    assert counts.beat_BH_Sharpe_count.eq(1).all()
    assert counts.beat_BH_Sharpe_pct.eq(20).all()
    assert counts.tie_BH_Sharpe_count.eq(1).all()
    assert counts.median_trade_count.eq(20).all()
    assert len(counts) == 3


@pytest.mark.parametrize("count", [-1, 0.5, np.nan, np.inf])
def test_validation_counts_reject_invalid_trades(count):
    selected = pairs()[:1]
    metrics = pd.DataFrame(
        {
            "candidate_id": c.candidate_id,
            "entry_rule": c.entry_rule.rule_id,
            "exit_rule": c.exit_rule.rule_id,
            "entry_confirmation_hours": hours,
            "exit_confirmation_hours": hours,
            "cash_excess_sharpe": 0.7,
            "trade_count": count,
        }
        for hours in study.WAIT_HOURS
        for c in study.candidates_from_pairs(selected, hours)
    )
    with pytest.raises(study.Error):
        study.counts_against_bh(metrics, 0.5, selected)


@pytest.mark.parametrize(
    "times",
    [
        ("2026-10-08",),
        ("2026-10-08T18:00Z", "2026-10-08T17:00Z"),
        ("2026-10-08T18:00Z", "2026-10-08"),
    ],
)
def test_chronology_rejects_naive_or_reversed_receipts(times):
    with pytest.raises(study.Error):
        study._chronology(*times)


def test_study_rejects_original_or_development_tree_overlap(tmp_path):
    original = tmp_path / "original"
    development = tmp_path / "development"
    for target in (original, original / "nested", development, development / "nested", tmp_path):
        with pytest.raises(study.Error):
            study._safe_roots(target, original, development)
    study._safe_roots(tmp_path / "separate", original, development)


def test_evaluate_requires_valid_candidate_seal_before_loading_prices(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        study,
        "verify_study",
        lambda *_: (_ for _ in ()).throw(study.Error("missing candidate seal")),
    )
    monkeypatch.setattr(study, "load_validation_packet", lambda *_: calls.append("unsafe_read"))
    with pytest.raises(study.Error, match="missing candidate seal"):
        study.evaluate_study(tmp_path / "sidecar")
    assert calls == []
    failures = list((tmp_path / "sidecar/attempts").glob("*/failure.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["status"] == "failed"


def test_public_release_requires_result_seal_before_return(tmp_path, monkeypatch):
    monkeypatch.setattr(
        study,
        "verify_results",
        lambda *_: (_ for _ in ()).throw(study.Error("missing result seal")),
    )
    with pytest.raises(study.Error, match="missing result seal"):
        study.verify_release(tmp_path)


def test_bound_reader_and_cli_have_no_oos_or_full_master_route():
    source = inspect.getsource(study)
    reader = inspect.getsource(study.load_validation_packet)
    metadata = inspect.getsource(study._read_validation_metadata)
    assert '"prepared/validation"' in reader
    assert "2015-12-31" in reader
    assert "DTB3_available.csv" not in source
    assert "_verify_prepared(" not in source
    assert "_load_stage(" not in source
    assert "read_reconciled_packet(" not in source
    assert "pd.read_csv" not in metadata
    assert "sha256_file" not in metadata
    assert 'choices=("seal", "evaluate", "verify")' in inspect.getsource(study.main)
    assert Path(study.DEFAULT_SIDECAR).name == "constant_cash_intersection_validation_20261008"
