"""Shortlist governance tests use fabricated scores, never market outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_following.research_daily_hourly_v5 import Candidate, IndicatorRule, indicator_rules
from trend_following.research_shortlist_selection import (
    FAMILIES,
    ShortlistSelectionError,
    select_development_shortlist,
)


def candidate(entry=10, exit_period=10, entry_hours=0.0, exit_hours=0.0):
    return Candidate(
        IndicatorRule("SMA", period=entry),
        IndicatorRule("EMA", period=exit_period),
        entry_hours,
        exit_hours,
    )


def metrics(candidates, values):
    return pd.DataFrame(
        {"candidate_id": [item.candidate_id for item in candidates], "cash_excess_sharpe": values}
    )


def test_best_confirmation_only_and_distinct_indicator_pairs():
    variants = [candidate(entry_hours=float(tick) / 2) for tick in range(14)]
    distinct = [candidate(entry=period) for period in (20, 50, 100, 200, 250)]
    candidates = variants + distinct
    scores = [20 - tick / 10 for tick in range(14)] + [5, 4, 3, 2, 1]
    selected = select_development_shortlist(metrics(candidates, scores), candidates)
    assert selected == [variants[0], *distinct[:4]]
    assert len({(item.entry_rule.rule_id, item.exit_rule.rule_id) for item in selected}) == 5


@pytest.mark.parametrize("per_family, expected", [(1, 9), (5, 45), (10, 90)])
def test_requested_count_and_stable_family_order(per_family, expected):
    rules = indicator_rules()
    candidates = [Candidate(entry, exit_rule, 0, 0) for entry in rules for exit_rule in rules]
    data = metrics(candidates, np.arange(len(candidates), dtype=float))
    chosen = select_development_shortlist(data, candidates, per_family=per_family)
    assert len(chosen) == expected
    assert [item.family_cell for item in chosen] == [
        f"{entry}/{exit_family}"
        for entry in FAMILIES
        for exit_family in FAMILIES
        for _ in range(per_family)
    ]
    shuffled = data.sample(frac=1, random_state=19)
    assert chosen == select_development_shortlist(
        shuffled, list(reversed(candidates)), per_family=per_family
    )


def test_ties_use_maximum_band_then_lexical_id_not_pairwise_chains():
    candidates = [candidate(entry=period) for period in (10, 20, 50)]
    # ID10 is too far below the maximum, even though close to ID20. ID20 lies
    # within the maximum's tolerance and precedes ID50 lexicographically.
    data = metrics(candidates, [1.0, 1.0 + 0.75e-10, 1.0 + 1.5e-10])
    assert select_development_shortlist(data, candidates, per_family=1) == [candidates[1]]


def test_lower_finite_negative_scores_remain_eligible_and_undefined_pairs_do_not():
    candidates = [candidate(entry=period) for period in (10, 20, 50, 100, 200, 250)]
    data = metrics(candidates, [np.nan, np.inf, -np.inf, -2, -3, "undefined"])
    assert select_development_shortlist(data, candidates) == [candidates[3], candidates[4]]
    with pytest.raises(ShortlistSelectionError, match="All development scores are undefined"):
        select_development_shortlist(metrics(candidates, [np.nan] * 6), candidates)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "foreign", "engine_duplicate"])
def test_metrics_must_cover_authoritative_candidates_exactly_once(damage):
    candidates = [candidate(entry=period) for period in (10, 20, 50)]
    data = metrics(candidates, [3, 2, 1])
    if damage == "missing":
        data = data.iloc[:-1]
    elif damage == "duplicate":
        data.loc[2, "candidate_id"] = data.loc[1, "candidate_id"]
    elif damage == "foreign":
        data.loc[2, "candidate_id"] = "unregistered_lookalike"
    else:
        candidates = [candidates[0], candidates[0], candidates[2]]
    with pytest.raises(ShortlistSelectionError, match="exactly once|unique"):
        select_development_shortlist(data, candidates)


@pytest.mark.parametrize(
    "field, value",
    [
        ("entry_family", "EMA"),
        ("exit_family", "MACD"),
        ("family_cell", "EMA/MACD"),
        ("entry_rule", "sma_250"),
        ("exit_rule", "ema_20"),
        ("entry_confirmation_hours", 6.5),
        ("exit_confirmation_hours", 1.5),
        ("entry_required_checks", 14),
        ("exit_required_checks", 4),
        ("commission_bps", 0),
        ("slippage_bps", 25),
        ("sampling_minutes", 60),
    ],
)
def test_optional_parameter_columns_cannot_conflict_with_engine_contract(field, value):
    candidates = [candidate()]
    data = metrics(candidates, [1.0])
    data[field] = value
    with pytest.raises(ShortlistSelectionError, match=field):
        select_development_shortlist(data, candidates)


def test_matching_parameter_columns_are_checked_but_never_reconstruct_candidates():
    candidates = [candidate()]
    data = metrics(candidates, [1.0]).assign(
        entry_family="SMA",
        exit_family="EMA",
        family_cell="SMA/EMA",
        entry_rule="sma_10",
        exit_rule="ema_10",
        entry_confirmation_hours=0,
        exit_confirmation_hours=0,
        entry_required_checks=1,
        exit_required_checks=1,
        commission_bps=1,
        slippage_bps=3,
        sampling_minutes=30,
    )
    assert select_development_shortlist(data, candidates)[0] is candidates[0]


def test_other_performance_and_post_gap_columns_cannot_change_selection():
    candidates = [candidate(entry=period) for period in (10, 20, 50)]
    data = metrics(candidates, [3, 2, 1])
    expected = select_development_shortlist(data, candidates)
    extended = data.assign(
        cagr=[-999, 999, 999],
        max_drawdown=[-1, 0, 0],
        trade_count=[10000, 0, 0],
        has_post_gap_confirmed_signal=[True, False, False],
        validation_sharpe=[-999, 999, 999],
        historical_oos_sharpe=[-999, 999, 999],
    )
    assert select_development_shortlist(extended, candidates) == expected


@pytest.mark.parametrize("value", [0, 2, 3, 4, 6, 11, True, "5", 5.0])
def test_invalid_shortlist_size_rejected(value):
    candidates = [candidate()]
    with pytest.raises(ShortlistSelectionError, match="per_family"):
        select_development_shortlist(metrics(candidates, [1]), candidates, per_family=value)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, True, "1e-10"])
def test_invalid_tie_tolerance_rejected(value):
    candidates = [candidate()]
    with pytest.raises(ShortlistSelectionError, match="tolerance"):
        select_development_shortlist(metrics(candidates, [1]), candidates, tolerance=value)


def test_empty_bad_schema_and_noncanonical_candidates_rejected():
    canonical = candidate()
    with pytest.raises(ShortlistSelectionError, match="require"):
        select_development_shortlist(pd.DataFrame(), [canonical])
    with pytest.raises(ShortlistSelectionError, match="canonical"):
        select_development_shortlist(metrics([canonical], [1]), [])
    with pytest.raises(ShortlistSelectionError, match="canonical"):
        select_development_shortlist(metrics([canonical], [1]), [canonical.to_dict()])
    doubled = pd.concat([metrics([canonical], [1]), metrics([canonical], [1])], axis=1)
    with pytest.raises(ShortlistSelectionError, match="columns must be unique"):
        select_development_shortlist(doubled, [canonical])
