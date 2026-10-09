"""Shortlist publication uses fabricated sealed contexts and derived numbers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend_following.research_daily_hourly_v5 import Candidate, IndicatorRule

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "shortlist_summary", ROOT / "scripts/summarize_shortlist.py"
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def models():
    rules = [
        IndicatorRule("SMA", period=10),
        IndicatorRule("EMA", period=20),
        IndicatorRule("MACD", fast=6, slow=13, signal=5),
    ]
    return [Candidate(entry, exit_rule, 0, 0) for entry in rules for exit_rule in rules]


def released(candidates):
    return {
        "candidates": candidates,
        "registration": {
            "selection": {
                "per_family": 1,
                "candidate_count": 9,
                "all_selected_candidates_retained_in_both_later_stages": True,
                "later_stage_selection": False,
                "gap_flags_affect_selection": False,
            },
            "historical_label": "exploratory_retrospective_extension_after_prior_exposure",
            "development_metrics_sha256": "a" * 64,
        },
        "candidate_set": {"synthetic": True},
        "config": {
            "segments": {
                "development": {"start": "2001-01-01", "end": "2011-12-31"},
                "validation": {"start": "2012-01-01", "end": "2015-12-31"},
                "historical-oos": {"start": "2016-01-01", "end": "2026-05-28"},
            }
        },
    }


def row(identity, slip=3, score=0.5):
    return {
        "candidate_id": identity,
        "commission_bps": 1.0,
        "slippage_bps": slip,
        "cagr": 0.1,
        "cash_excess_sharpe": score,
        "max_drawdown": -0.1,
        "terminal_equity": 1100.0,
        "scored_sessions": 2,
        "annualized_volatility": 0.2,
        "exposure_fraction": 0.5,
        "annualized_turnover": 2.0,
        "trade_count": 1,
        "average_holding_hours": 10.0,
        "cost_drag_return": 0.01,
        "entry_rule": "synthetic_dummy",
        "post_gap_confirmed_signal_count": 0,
        "has_post_gap_confirmed_signal": False,
    }


def daily(rows, start):
    output = []
    for record in rows:
        for stamp, equity in (
            (f"{start}T22:00:00+00:00", 1000.0),
            (f"{start[:-2]}04T22:00:00+00:00", 1100.0),
        ):
            output.append(
                {
                    "candidate_id": record["candidate_id"],
                    "timestamp": stamp,
                    "return": 0.0 if equity == 1000 else 0.1,
                    "equity": equity,
                    "cash_return": 0.001,
                    "exposure": 0.5,
                    "commission_bps": 1.0,
                    "slippage_bps": record["slippage_bps"],
                }
            )
    return pd.DataFrame(output)


@pytest.fixture
def publication_inputs(tmp_path, monkeypatch):
    candidates = models()
    context = released(candidates)
    monkeypatch.setattr(helper, "release_report", lambda *args: context)
    original, study, public = tmp_path / "parent", tmp_path / "study", tmp_path / "dev_public"
    dev_path = original / "stages/development"
    dev_path.mkdir(parents=True)
    development = pd.DataFrame([row(candidate.candidate_id) for candidate in candidates])
    development.loc[0, ["has_post_gap_confirmed_signal", "post_gap_confirmed_signal_count"]] = [
        True,
        2,
    ]
    development.to_csv(dev_path / "metrics.csv", index=False)
    public.mkdir()
    control = pd.DataFrame(
        [row("benchmark_buy_and_hold", score=0.4), row("benchmark_cash", score=np.nan)]
    )
    control.to_csv(public / "comparison.csv", index=False)
    helper.write_json(
        public / "artifact_hashes.json",
        {"comparison.csv": hashlib.sha256((public / "comparison.csv").read_bytes()).hexdigest()},
    )
    for stage, costs, start in (
        ("validation", (3,), "2012-01-03"),
        ("historical-oos", helper.COSTS, "2016-01-03"),
    ):
        folder = study / "stages" / stage
        folder.mkdir(parents=True)
        members = [
            row(candidate.candidate_id, slip, np.nan if index == 0 else 0.5)
            for slip in costs
            for index, candidate in enumerate(candidates)
        ]
        controls = [
            row(identity, slip, np.nan if identity == "cash" else 0.4)
            for slip in costs
            for identity in helper.BENCHMARKS
        ]
        pd.DataFrame(members).to_csv(folder / "metrics.csv", index=False)
        pd.DataFrame(controls).to_csv(folder / "benchmarks.csv", index=False)
        daily(members, start).to_csv(folder / "daily.csv.gz", index=False)
        daily(controls, start).to_csv(folder / "benchmark_daily.csv.gz", index=False)
    return original, study, public, context


def test_missing_release_stops_before_any_numeric_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(
        helper, "release_report", lambda *args: (_ for _ in ()).throw(ValueError("unsealed"))
    )
    monkeypatch.setattr(
        pd, "read_csv", lambda *args, **kwargs: pytest.fail("CSV before seal verification")
    )
    with pytest.raises(ValueError, match="unsealed"):
        helper.publish(
            parent_study=tmp_path, config_path=tmp_path, study=tmp_path, output=tmp_path / "output"
        )
    assert not (tmp_path / "output").exists()


def test_both_stage_retention_gates_run_without_numeric_reads(tmp_path, monkeypatch):
    context = released(models())
    monkeypatch.setattr(helper.protocol, "verify_shortlist", lambda **kwargs: context)
    calls = []

    def stage(study, name, registration):
        calls.append(name)
        return {
            "candidate_ids": [candidate.candidate_id for candidate in context["candidates"]],
            "candidate_count": 9,
            "selection_after_evaluation": False,
            "commission_bps": 1.0,
            "slippage_bps_scenarios": [3] if name == "validation" else list(helper.COSTS),
        }

    monkeypatch.setattr(helper.protocol, "verify_stage", stage)
    monkeypatch.setattr(
        pd, "read_csv", lambda *args, **kwargs: pytest.fail("Numeric CSV inside gates")
    )
    result = helper.release_report(tmp_path, tmp_path, tmp_path)
    assert calls == ["validation", "historical-oos"]
    assert set(result["stage_seals"]) == set(calls)


def test_changed_stage_candidate_set_rejected_before_any_csv(tmp_path, monkeypatch):
    context = released(models())
    monkeypatch.setattr(helper.protocol, "verify_shortlist", lambda **kwargs: context)
    monkeypatch.setattr(
        helper.protocol,
        "verify_stage",
        lambda *args: {
            "candidate_ids": ["wrong"],
            "candidate_count": 1,
            "selection_after_evaluation": False,
            "commission_bps": 1,
            "slippage_bps_scenarios": [3],
        },
    )
    monkeypatch.setattr(
        pd, "read_csv", lambda *args, **kwargs: pytest.fail("CSV before retained-set gate")
    )
    with pytest.raises(ValueError, match="retain every"):
        helper.release_report(tmp_path, tmp_path, tmp_path)


def test_full_fixed_set_publication_retains_undefined_and_joins_without_attrition(
    publication_inputs, tmp_path
):
    original, study, public, context = publication_inputs
    output = tmp_path / "output"
    summary = helper.publish(
        parent_study=original,
        config_path=tmp_path / "config",
        study=study,
        output=output,
        development_report=public,
        include_daily=True,
    )
    assert summary["candidate_count"] == 9
    assert summary["later_stage_winner_selected"] is False
    assert summary["all_selected_candidates_retained_in_both_later_stages"]
    assert summary["descriptive_development_gap_flags"] == {
        "post_gap_flagged_candidates": 1,
        "post_gap_confirmed_signals": 2,
        "flags_used_for_selection": False,
    }
    validation = pd.read_csv(output / "validation_comparison.csv")
    historical = pd.read_csv(output / "historical_cost_comparison.csv")
    wide = pd.read_csv(output / "validation_and_historical_baseline.csv")
    assert len(validation) == 11 and len(historical) == 44 and len(wide) == 9
    assert wide.candidate_id.tolist() == [item.candidate_id for item in context["candidates"]]
    assert np.isnan(wide.validation_cash_excess_sharpe.iloc[0])
    assert (
        validation.loc[validation.candidate_id.isin(helper.BENCHMARKS), "entry_rule"].isna().all()
    )
    stats = summary["comparisons"]["validation"][0]
    assert stats["undefined_primary_scores_retained"] == 1
    assert stats["finite_primary_scores"] == 8
    assert stats["versus"]["cash"]["higher_cash_excess_sharpe"] is None
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert "artifact_hashes.json" not in hashes
    assert hashes == helper.artifact_hashes(output)
    text = (output / "summary.json").read_text()
    assert str(tmp_path) not in text and "NaN" not in text
    assert "untouched OOS" in text


@pytest.mark.parametrize(
    "field", ["close", "raw_price", "execution_price", "quantity", "source_path", "consent"]
)
def test_metrics_privacy_whitelist_rejects_quote_trade_or_consent(field):
    frame = pd.DataFrame([row("cash")]).assign(**{field: "private"})
    with pytest.raises(ValueError, match="whitelisted"):
        helper.public_metrics(frame)


def test_path_in_allowed_metric_text_is_rejected():
    frame = pd.DataFrame([row("cash")]).assign(return_type="/Users/person/private.txt")
    with pytest.raises(ValueError, match="machine path"):
        helper.public_metrics(frame)


def test_all_nonfinite_primary_scores_are_retained_as_undefined_not_zero():
    frame = pd.DataFrame([row("one", score=np.inf), row("two", score="undefined")])
    result = helper._metric_contract(frame, ["one", "two"], (3,))
    assert result.candidate_id.tolist() == ["one", "two"]
    assert result.cash_excess_sharpe.isna().all()


def test_fractional_scored_session_count_rejected():
    frame = pd.DataFrame([row("one")]).assign(scored_sessions=2.5)
    with pytest.raises(ValueError, match="positive integers"):
        helper._metric_contract(frame, ["one"], (3,))


@pytest.mark.parametrize("damage", ["missing", "duplicate", "foreign", "wrong_cost"])
def test_comparison_counts_ids_and_costs_are_exact(damage):
    frame = pd.DataFrame([row("one"), row("two")])
    if damage == "missing":
        frame = frame.iloc[:-1]
    elif damage == "duplicate":
        frame.loc[1, "candidate_id"] = "one"
    elif damage == "foreign":
        frame.loc[1, "candidate_id"] = "other"
    else:
        frame.loc[1, "slippage_bps"] = 25
    with pytest.raises(ValueError, match="every sealed member"):
        helper._metric_contract(frame, ["one", "two"], (3,))


def test_daily_privacy_and_unfinished_naive_or_mismatched_clock_rejected():
    records = [row("one"), row("cash", score=np.nan)]
    frame = daily(records, "2012-01-03")
    metrics = pd.DataFrame(records)
    with pytest.raises(ValueError, match="never quotes"):
        helper.public_curves(frame.assign(close=100), ["one", "cash"], (3,), metrics)
    naive = frame.copy()
    naive["timestamp"] = naive.timestamp.str.replace("+00:00", "", regex=False)
    with pytest.raises(ValueError, match="aware"):
        helper.public_curves(naive, ["one", "cash"], (3,), metrics)
    mismatched = frame.copy()
    mismatched.loc[3, "timestamp"] = "2012-01-05T22:00:00+00:00"
    with pytest.raises(ValueError, match="same daily valuation clock"):
        helper.public_curves(mismatched, ["one", "cash"], (3,), metrics)


def test_development_benchmark_manifest_gate_precedes_csv(publication_inputs, monkeypatch):
    _, _, public, _ = publication_inputs
    (public / "comparison.csv").write_text("changed")
    monkeypatch.setattr(
        pd, "read_csv", lambda *args, **kwargs: pytest.fail("CSV before dev hash gate")
    )
    with pytest.raises(ValueError, match="hash changed"):
        helper.development_controls(public)


def test_bad_later_metric_does_not_create_partial_output(publication_inputs, tmp_path):
    original, study, public, _ = publication_inputs
    path = study / "stages/historical-oos/metrics.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="every sealed member"):
        helper.publish(
            parent_study=original,
            config_path=tmp_path / "config",
            study=study,
            output=output,
            development_report=public,
        )
    assert not output.exists()
