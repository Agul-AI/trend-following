"""Independent shortlist checks on fabricated data only, never market outcomes."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trend_following import research_shortlist as protocol
from trend_following.research_daily_hourly_v5 import Candidate, indicator_rules
from trend_following.research_shortlist_selection import select_development_shortlist


def _toy_universe():
    # Cover every indicator pair, but only two fabricated confirmation variants.
    return [
        Candidate(entry, exit_rule, wait, 0)
        for entry in indicator_rules()
        for exit_rule in indicator_rules()
        for wait in (0, 0.5)
    ]


def _independent_oracle(candidates, scores, count):
    """Pure-Python reference: maximum band, stable ID, then remove whole pair."""
    answer = []
    for entry_family in ("SMA", "EMA", "MACD"):
        for exit_family in ("SMA", "EMA", "MACD"):
            eligible = [
                candidate
                for candidate in candidates
                if candidate.entry_rule.family == entry_family
                and candidate.exit_rule.family == exit_family
                and math.isfinite(scores[candidate.candidate_id])
            ]
            for _ in range(count):
                if not eligible:
                    break
                maximum = max(scores[candidate.candidate_id] for candidate in eligible)
                chosen = min(
                    (
                        candidate
                        for candidate in eligible
                        if scores[candidate.candidate_id] >= maximum - 1e-10
                    ),
                    key=lambda candidate: candidate.candidate_id,
                )
                answer.append(chosen.candidate_id)
                eligible = [
                    candidate
                    for candidate in eligible
                    if (candidate.entry_rule, candidate.exit_rule)
                    != (chosen.entry_rule, chosen.exit_rule)
                ]
    return answer


@pytest.mark.parametrize("seed", [7, 101, 999])
@pytest.mark.parametrize("count", [1, 5, 10])
def test_randomized_selection_matches_independent_maximum_band_oracle(seed, count):
    candidates = _toy_universe()
    rng = np.random.default_rng(seed)
    # Discrete scores create real ties and independently exclude nonfinite values.
    values = rng.integers(-7, 8, size=len(candidates)).astype(float)
    values[::31] = np.nan
    values[::47] = np.inf
    values[::59] = -np.inf
    scores = dict(zip((item.candidate_id for item in candidates), values, strict=True))
    frame = pd.DataFrame({"candidate_id": list(scores), "cash_excess_sharpe": values})
    frame = frame.sample(frac=1, random_state=seed)
    selected = select_development_shortlist(frame, list(reversed(candidates)), per_family=count)
    assert [item.candidate_id for item in selected] == _independent_oracle(
        candidates, scores, count
    )
    assert len(selected) == 9 * count
    for family in {item.family_cell for item in selected}:
        subset = [item for item in selected if item.family_cell == family]
        assert len(subset) == count
        assert len({(item.entry_rule.rule_id, item.exit_rule.rule_id) for item in subset}) == count


def test_selection_is_pure_preserves_inputs_and_cannot_open_future_results(monkeypatch):
    candidates = _toy_universe()
    frame = pd.DataFrame(
        {
            "candidate_id": [item.candidate_id for item in candidates],
            "cash_excess_sharpe": np.arange(len(candidates), dtype=float),
            "validation_score": np.arange(len(candidates), dtype=float)[::-1],
            "historical_oos_score": np.arange(len(candidates), dtype=float)[::-1],
            "has_post_gap_confirmed_signal": True,
        }
    )
    before = frame.copy(deep=True)
    original_ids = [item.candidate_id for item in candidates]

    def denied(*args, **kwargs):
        raise AssertionError("Development selector must not read any file")

    monkeypatch.setattr("builtins.open", denied)
    selected = select_development_shortlist(frame, candidates, per_family=5)
    assert len(selected) == 45
    pd.testing.assert_frame_equal(frame, before)
    assert [item.candidate_id for item in candidates] == original_ids
    assert all(any(item is original for original in candidates) for item in selected)


def test_negative_scores_are_relative_shortlist_not_profitability_filter():
    candidates = _toy_universe()
    frame = pd.DataFrame(
        {
            "candidate_id": [item.candidate_id for item in candidates],
            "cash_excess_sharpe": -np.arange(1, len(candidates) + 1, dtype=float),
        }
    )
    selected = select_development_shortlist(frame, candidates, per_family=5)
    assert len(selected) == 45
    assert (
        frame.set_index("candidate_id")
        .loc[[item.candidate_id for item in selected], "cash_excess_sharpe"]
        .lt(0)
        .all()
    )


@pytest.mark.parametrize(
    "damage",
    ["drop", "reorder", "duplicate", "reselection", "wrong_period", "commission", "slippage"],
)
def test_stage_resealed_metadata_still_cannot_change_set_period_or_cost(tmp_path, damage):
    # Fabricated artifacts and seals only: no real study or numeric file is read.
    registration = {
        "segments": {"validation": {"start": "2012-01-01", "end": "2015-12-31"}},
        "validation_slippage_bps": [3.0],
        "historical_oos_slippage_bps": [1.0, 3.0, 10.0, 25.0],
    }
    candidate_set = {
        "candidates": [{"candidate_id": "synthetic_a"}, {"candidate_id": "synthetic_b"}]
    }
    root = tmp_path / "stages/validation"
    root.mkdir(parents=True)
    (root / "SYNTHETIC.txt").write_text("FABRICATED ARTIFACT ONLY")
    summary = {
        "stage": "validation",
        "registration_sha256": protocol.parent.canonical_hash(registration),
        "candidate_set_sha256": protocol.parent.canonical_hash(candidate_set),
        "candidate_ids": ["synthetic_a", "synthetic_b"],
        "candidate_count": 2,
        "historical_label": protocol.LABEL,
        "slippage_bps_scenarios": [3.0],
        "commission_bps": 1.0,
        "scored_segment": registration["segments"]["validation"].copy(),
        "selection_after_evaluation": False,
        "artifact_hashes": protocol.parent._artifact_hashes(root),
    }
    if damage == "drop":
        summary.update(candidate_ids=["synthetic_a"], candidate_count=1)
    elif damage == "reorder":
        summary["candidate_ids"].reverse()
    elif damage == "duplicate":
        summary["candidate_ids"] = ["synthetic_a", "synthetic_a"]
    elif damage == "reselection":
        summary["selection_after_evaluation"] = True
    elif damage == "wrong_period":
        summary["scored_segment"]["end"] = "2026-05-28"
    elif damage == "commission":
        summary["commission_bps"] = 0.0
    else:
        summary["slippage_bps_scenarios"] = [25.0]
    protocol.parent._seal(root / "stage_seal.json", summary)
    with pytest.raises(protocol.parent.ProtocolError, match="complete unchanged"):
        protocol._verify_stage(tmp_path, "validation", registration, candidate_set)
