"""Synthetic equal-wait summaries never load market or later-phase data."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend_following.research_daily_hourly_v5 import Candidate, IndicatorRule

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "equal_wait_summary", ROOT / "scripts/summarize_equal_waits.py"
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


@pytest.fixture
def toy():
    rules = [
        IndicatorRule("SMA", period=10),
        IndicatorRule("EMA", period=20),
        IndicatorRule("MACD", fast=6, slow=13, signal=5),
    ]
    table, records = {}, []
    for i, entry in enumerate(rules):
        for j, exit_rule in enumerate(rules):
            for en in np.arange(0, 7, 0.5):
                for ex in np.arange(0, 7, 0.5):
                    candidate = Candidate(entry, exit_rule, en, ex)
                    table[candidate.candidate_id] = candidate
                    records.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            **{
                                name: getter(candidate)
                                for name, getter in helper.IDENTITY_FIELDS.items()
                            },
                            "cash_excess_sharpe": (3 * i + j + 1) / 10 + en / 10 - ex / 50,
                            "cagr": 0.08,
                            "max_drawdown": -0.2,
                            "commission_bps": 1.0,
                            "slippage_bps": 3.0,
                            "sampling_minutes": 30,
                            "scored_sessions": 2767,
                            "start_session": "2001-01-02",
                            "end_session": "2011-12-30",
                            "trade_count": 110,
                            "annualized_turnover": 20.0,
                            "cost_drag_return": 0.1,
                        }
                    )
    metrics = pd.DataFrame(records)
    selected = metrics.loc[
        metrics.entry_confirmation_hours.eq(0) & metrics.exit_confirmation_hours.eq(0.5)
    ].copy()
    return metrics, selected, table


def analyze(toy):
    metrics, selected, table = toy
    return helper.equal_wait_analysis(
        metrics, selected, table, expected_shortlist_count=9, family_quota=1
    )


def test_complete_fixed_pair_population_and_original_selection_unchanged(toy):
    metrics, selected, _ = toy
    before, shortlist_before = metrics.copy(deep=True), selected.copy(deep=True)
    result = analyze(toy)
    rows = result["per_pair_equal_wait_metrics"]
    assert len(rows) == 9 * 13
    assert rows.groupby("common_confirmation_hours").size().tolist() == [9] * 13
    assert (
        rows.groupby("pair_id")
        .common_confirmation_hours.agg(tuple)
        .map(lambda values: values == helper.WAIT_HOURS)
        .all()
    )
    assert rows.entry_confirmation_hours.eq(rows.exit_confirmation_hours).all()
    assert rows.frozen_shortlist_member.all()
    assert rows.entry_required_checks.eq(1 + 2 * rows.common_confirmation_hours).all()
    assert set(rows.common_confirmation_hours) == set(np.arange(0, 6.5, 0.5))
    pd.testing.assert_frame_equal(metrics, before)
    pd.testing.assert_frame_equal(selected, shortlist_before)


def test_aggregate_math_paired_deltas_counts_and_family_cohort(toy):
    result = analyze(toy)
    summary = (
        result["per_wait_summary"]
        .query("cohort == 'frozen_45_pairs'")
        .set_index("common_confirmation_hours")
    )
    assert summary.loc[0, "median_sharpe"] == pytest.approx(0.5)
    assert summary.loc[0, "mean_sharpe"] == pytest.approx(0.5)
    assert summary.loc[0, "q25_sharpe"] == pytest.approx(0.3)
    assert summary.loc[0, "q75_sharpe"] == pytest.approx(0.7)
    assert summary.loc[0, "sample_std_sharpe"] == pytest.approx(
        np.std(np.arange(0.1, 1.0, 0.1), ddof=1)
    )
    assert summary.loc[6, "mean_delta_sharpe_vs_zero"] == pytest.approx(0.48)
    assert summary.loc[6, "fraction_improved_vs_zero"] == 1
    assert summary.loc[0, "unchanged_delta_count"] == 9
    assert summary.loc[0, "fraction_improved_vs_zero"] == 0
    assert summary.loc[0, "median_trade_count"] == 110
    assert summary.loc[0, "mean_trade_count"] == 110
    assert result["per_pair_wait_range"].sharpe_span_0_to_6_hours.eq(0.48).all() or np.allclose(
        result["per_pair_wait_range"].sharpe_span_0_to_6_hours, 0.48
    )
    assert len(result["family_wait_summary"]) == 2 * 9 * 13


def test_nonfinite_sharpes_are_retained_and_explicitly_counted(toy):
    metrics, _, _ = toy
    mask = (
        metrics.entry_rule.eq("sma_10")
        & metrics.exit_rule.eq("sma_10")
        & metrics.entry_confirmation_hours.eq(0)
        & metrics.exit_confirmation_hours.eq(0)
    )
    metrics.loc[mask, "cash_excess_sharpe"] = np.nan
    mask2 = mask.copy()
    mask2 = (
        metrics.entry_rule.eq("ema_20")
        & metrics.exit_rule.eq("ema_20")
        & metrics.entry_confirmation_hours.eq(6)
        & metrics.exit_confirmation_hours.eq(6)
    )
    metrics.loc[mask2, "cash_excess_sharpe"] = np.inf
    result = analyze(toy)
    assert len(result["per_pair_equal_wait_metrics"]) == 117
    summary = (
        result["per_wait_summary"]
        .query("cohort == 'frozen_45_pairs'")
        .set_index("common_confirmation_hours")
    )
    assert summary.loc[0, "finite_sharpe_count"] == 8
    assert summary.loc[0, "nonfinite_sharpe_count"] == 1
    assert summary.loc[6, "finite_sharpe_count"] == 8
    assert summary.loc[6, "finite_paired_count"] == 7
    rows = result["per_pair_equal_wait_metrics"]
    assert len(rows.loc[~rows.finite_sharpe]) == 2


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "bad_id", "bad_rule", "bad_wait", "bad_checks"]
)
def test_canonical_grid_malformed_missing_or_duplicate_fails(toy, change):
    metrics, selected, table = toy
    if change == "missing":
        metrics = metrics.iloc[1:]
    elif change == "duplicate":
        metrics = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    else:
        name, value = {
            "bad_id": ("candidate_id", "made_up"),
            "bad_rule": ("entry_rule", "ema_999"),
            "bad_wait": ("exit_confirmation_hours", 0.25),
            "bad_checks": ("entry_required_checks", 0),
        }[change]
        metrics.loc[0, name] = value
    with pytest.raises(ValueError):
        helper.equal_wait_analysis(
            metrics, selected, table, expected_shortlist_count=9, family_quota=1
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("commission_bps", 0),
        ("slippage_bps", 10),
        ("start_session", "2012-01-03"),
        ("end_session", "2015-12-31"),
        ("scored_sessions", 1006),
        ("sampling_minutes", 60),
    ],
)
def test_period_and_cost_contract_fails_closed(toy, name, value):
    toy[0].loc[0, name] = value
    with pytest.raises(ValueError, match="development contract"):
        analyze(toy)


def test_shortlist_pair_diversity_and_source_metrics_preserved(toy):
    metrics, selected, table = toy
    changed = selected.copy()
    changed.loc[changed.index[0], "cash_excess_sharpe"] += 0.01
    with pytest.raises(ValueError, match="no longer matches"):
        helper.equal_wait_analysis(
            metrics, changed, table, expected_shortlist_count=9, family_quota=1
        )
    duplicate_pair = pd.concat(
        [
            selected.iloc[:-1],
            metrics.loc[
                metrics.entry_confirmation_hours.eq(1) & metrics.exit_confirmation_hours.eq(1)
            ].iloc[[0]],
        ]
    )
    with pytest.raises(ValueError, match="distinct indicator pairs"):
        helper.equal_wait_analysis(
            metrics, duplicate_pair, table, expected_shortlist_count=9, family_quota=1
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("close", 123.4),
        ("reference_price", 123.4),
        ("private_path", "/secret"),
        ("quality_profile", "/Users/private"),
    ],
)
def test_privacy_schema_rejects_quotes_trades_and_paths(toy, name, value):
    toy[0][name] = value
    with pytest.raises(ValueError):
        analyze(toy)


def test_only_half_hour_wait_range_is_accepted(toy):
    with pytest.raises(ValueError, match="half-hour intervals"):
        helper.equal_wait_analysis(
            *toy, waits=(0, 0.25, 1), expected_shortlist_count=9, family_quota=1
        )


def test_tampered_public_hash_gate_precedes_every_numerical_reader(tmp_path, monkeypatch):
    development = tmp_path / "development"
    shortlist = tmp_path / "shortlist"
    development.mkdir()
    shortlist.mkdir()
    (development / "artifact_hashes.json").write_text(json.dumps({"metrics.csv.gz": "wrong"}))
    monkeypatch.setattr(
        helper.pd, "read_csv", lambda *args, **kwargs: pytest.fail("CSV read before hash gate")
    )
    with pytest.raises(ValueError, match="Frozen development artifact changed"):
        helper.main(
            [
                "--development",
                str(development),
                "--shortlist",
                str(shortlist),
                "--output",
                str(tmp_path / "output"),
            ]
        )
    assert not (tmp_path / "output").exists()


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(
        helper,
        "verify_public_inputs",
        lambda *args, **kwargs: pytest.fail("Inputs must not be opened"),
    )
    with pytest.raises(FileExistsError):
        helper.main(["--output", str(tmp_path)])


def test_primary_subset_does_not_expand_to_new_high_scoring_pairs(toy):
    metrics, selected, table = toy
    extra = IndicatorRule("SMA", period=20)
    rules = [
        extra,
        IndicatorRule("SMA", period=10),
        IndicatorRule("EMA", period=20),
        IndicatorRule("MACD", fast=6, slow=13, signal=5),
    ]
    rows = []
    base = metrics.iloc[0].to_dict()
    for entry in rules:
        for exit_rule in rules:
            if entry != extra and exit_rule != extra:
                continue
            for en in np.arange(0, 7, 0.5):
                for ex in np.arange(0, 7, 0.5):
                    candidate = Candidate(entry, exit_rule, en, ex)
                    table[candidate.candidate_id] = candidate
                    rows.append(
                        {
                            **base,
                            "candidate_id": candidate.candidate_id,
                            **{
                                name: getter(candidate)
                                for name, getter in helper.IDENTITY_FIELDS.items()
                            },
                            "cash_excess_sharpe": 1000.0,
                        }
                    )
    expanded = pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)
    result = helper.equal_wait_analysis(
        expanded, selected, table, expected_shortlist_count=9, family_quota=1
    )
    equal = result["per_pair_equal_wait_metrics"]
    assert len(equal) == 16 * 13
    assert equal.frozen_shortlist_member.sum() == 9 * 13
    assert not equal.loc[equal.frozen_shortlist_member, "cash_excess_sharpe"].eq(1000.0).any()
    assert equal.loc[~equal.frozen_shortlist_member, "cash_excess_sharpe"].eq(1000.0).all()


def test_valid_hash_gate_and_modified_implementation_failure(tmp_path):
    development, shortlist = tmp_path / "dev", tmp_path / "hold"
    development.mkdir()
    shortlist.mkdir()
    source_hashes = {}
    for index in range(8):
        name = f"src/frozen_{index}.py"
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"# Immutable synthetic source {index}\n")
        source_hashes[name] = helper.sha256(path)
    selection_name = "src/trend_following/research_shortlist_selection.py"
    selection = tmp_path / selection_name
    selection.parent.mkdir()
    selection.write_text("# Frozen synthetic selector\n")
    (development / "metrics.csv.gz").write_bytes(b"synthetic bytes; no reader")
    (shortlist / "development_shortlist.csv").write_text("synthetic shortlist; no reader\n")
    receipt = {"frozen_source_sha256": source_hashes}
    helper.write_json(development / "publication_receipt.json", receipt)
    hold = {
        "source_hashes": source_hashes,
        "selection_code_sha256": helper.sha256(selection),
        "development_input_sha256": helper.sha256(development / "metrics.csv.gz"),
        "candidate_count": 45,
        "development_period": {"start": "2001-01-01", "end": "2011-12-31"},
        "later_stage_numerical_inputs_opened_for_this_selection": False,
        "later_stage_performance_run_for_this_selection": False,
    }
    helper.write_json(shortlist / "selection_and_hold.json", hold)
    anchors = {
        path.name: helper.sha256(path)
        for directory in (development, shortlist)
        for path in directory.iterdir()
    }
    for directory in (development, shortlist):
        helper.write_json(
            directory / "artifact_hashes.json",
            {path.name: helper.sha256(path) for path in directory.iterdir()},
        )
    verified = helper.verify_public_inputs(development, shortlist, root=tmp_path, anchors=anchors)
    assert verified["input_artifact_sha256"] == anchors
    (tmp_path / "src/frozen_0.py").write_text("# Modified\n")
    with pytest.raises(ValueError, match="implementation source changed"):
        helper.verify_public_inputs(development, shortlist, root=tmp_path, anchors=anchors)
