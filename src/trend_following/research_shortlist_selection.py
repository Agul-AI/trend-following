"""Pure development-only selection for the Trend Following Study shortlist.

This module never opens files or evaluates market performance. Supplied engine
candidate objects are the authority for identities and indicator diversity;
optional metric identity columns are checked, never trusted as parameter inputs.
The new shortlist is exploratory retrospective research, not a replacement for
the immutable previously completed study or a restoration of untouched OOS.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from trend_following.research_daily_hourly_v5 import SAMPLING_MINUTES, Candidate

FAMILIES = ("SMA", "EMA", "MACD")
ALLOWED_PER_FAMILY = (1, 5, 10)
TIE_TOLERANCE = 1e-10


class ShortlistSelectionError(ValueError):
    """The supplied development universe or selection inputs are invalid."""


def _selection_inputs(metrics: pd.DataFrame, candidates: Sequence[Candidate]) -> pd.DataFrame:
    if not isinstance(metrics, pd.DataFrame) or not {
        "candidate_id",
        "cash_excess_sharpe",
    }.issubset(metrics.columns):
        raise ShortlistSelectionError(
            "Development metrics require candidate_id and cash_excess_sharpe"
        )
    if metrics.columns.duplicated().any():
        raise ShortlistSelectionError("Development metric columns must be unique")
    if not candidates or not all(isinstance(candidate, Candidate) for candidate in candidates):
        raise ShortlistSelectionError("Selection requires canonical engine Candidate objects")
    table = {candidate.candidate_id: candidate for candidate in candidates}
    if len(table) != len(candidates):
        raise ShortlistSelectionError("Engine candidate identities must be unique")
    ids = metrics["candidate_id"]
    if (
        not ids.map(lambda value: isinstance(value, str)).all()
        or ids.duplicated().any()
        or len(metrics) != len(table)
        or set(ids) != set(table)
    ):
        raise ShortlistSelectionError(
            "Development metrics must match the complete supplied engine universe exactly once"
        )

    indexed = metrics.set_index("candidate_id", drop=False).copy()
    expected_text = {
        "entry_family": lambda candidate: candidate.entry_rule.family,
        "exit_family": lambda candidate: candidate.exit_rule.family,
        "family_cell": lambda candidate: candidate.family_cell,
        "entry_rule": lambda candidate: candidate.entry_rule.rule_id,
        "exit_rule": lambda candidate: candidate.exit_rule.rule_id,
    }
    expected_numeric = {
        "entry_confirmation_hours": lambda candidate: candidate.entry_confirmation_hours,
        "exit_confirmation_hours": lambda candidate: candidate.exit_confirmation_hours,
        "entry_required_checks": lambda candidate: candidate.entry_required_checks,
        "exit_required_checks": lambda candidate: candidate.exit_required_checks,
        "commission_bps": lambda candidate: 1.0,
        "slippage_bps": lambda candidate: 3.0,
        "sampling_minutes": lambda candidate: SAMPLING_MINUTES,
    }
    for column, getter in expected_text.items():
        if column in indexed:
            expected = indexed["candidate_id"].map(
                {identity: getter(candidate) for identity, candidate in table.items()}
            )
            if not indexed[column].eq(expected).all():
                raise ShortlistSelectionError(
                    f"Metric {column} disagrees with canonical engine candidates"
                )
    for column, getter in expected_numeric.items():
        if column in indexed:
            expected = indexed["candidate_id"].map(
                {identity: getter(candidate) for identity, candidate in table.items()}
            )
            numeric = pd.to_numeric(indexed[column], errors="coerce")
            if not np.isfinite(numeric).all() or not numeric.eq(expected).all():
                raise ShortlistSelectionError(
                    f"Metric {column} disagrees with the fixed engine/baseline contract"
                )

    # Selection reads no other performance, gap-flag, validation or OOS column.
    indexed["cash_excess_sharpe"] = pd.to_numeric(indexed["cash_excess_sharpe"], errors="coerce")
    return indexed[["candidate_id", "cash_excess_sharpe"]]


def select_development_shortlist(
    metrics: pd.DataFrame,
    candidates: Sequence[Candidate],
    *,
    per_family: int = 5,
    tolerance: float = TIE_TOLERANCE,
) -> list[Candidate]:
    """Choose up to 1, 5 or 10 distinct indicator pairs in every family cell.

    Repeatedly find the maximum finite development cash-excess Sharpe, then
    choose the lexicographically smallest ID within ``tolerance`` of that
    maximum. Remove every confirmation variant of the chosen entry/exit
    indicator pair before selecting the next member. The tie rule matches the
    completed study's finite-winner semantics and avoids nontransitive pairwise
    comparisons. Undefined pairs are omitted; all-undefined selection stops.

    Family order is SMA, EMA, MACD for entry, then that same order for exit;
    within a family cell the list follows iterative selection rank. There is
    deliberately no positivity, return, turnover, drawdown or gap-flag filter.
    Callers seal the set before evaluation and verify the expected actual count.
    """
    if (
        isinstance(per_family, (bool, np.bool_))
        or not isinstance(per_family, (int, np.integer))
        or per_family not in ALLOWED_PER_FAMILY
    ):
        raise ShortlistSelectionError("per_family must be 1, 5 or 10")
    if (
        isinstance(tolerance, (bool, np.bool_))
        or not isinstance(tolerance, (int, float, np.integer, np.floating))
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ShortlistSelectionError("Tie tolerance must be finite and nonnegative")

    scores = _selection_inputs(metrics, candidates)
    table = {candidate.candidate_id: candidate for candidate in candidates}
    selected: list[Candidate] = []
    for entry_family in FAMILIES:
        for exit_family in FAMILIES:
            remaining = {
                identity: candidate
                for identity, candidate in table.items()
                if candidate.entry_rule.family == entry_family
                and candidate.exit_rule.family == exit_family
            }
            for _ in range(per_family):
                subset = scores.loc[scores.index.isin(remaining)]
                finite = subset.loc[np.isfinite(subset["cash_excess_sharpe"])]
                if finite.empty:
                    break
                maximum = finite["cash_excess_sharpe"].max()
                tied = finite.loc[
                    finite["cash_excess_sharpe"].ge(maximum - tolerance), "candidate_id"
                ]
                chosen = remaining[min(tied)]
                selected.append(chosen)
                pair = chosen.entry_rule.rule_id, chosen.exit_rule.rule_id
                remaining = {
                    identity: candidate
                    for identity, candidate in remaining.items()
                    if (candidate.entry_rule.rule_id, candidate.exit_rule.rule_id) != pair
                }
    if not selected:
        raise ShortlistSelectionError(
            "All development scores are undefined; no shortlist can be invented"
        )
    return selected
