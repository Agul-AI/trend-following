from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tqqq_cash_yield_candidate_comparison import (  # noqa: E402
    simulate_after_tax_portfolio_with_cash_yield,
)


def _simulate(
    returns: list[float],
    weights: list[float],
    index: pd.DatetimeIndex,
    *,
    cash_yield: float = 0.0,
    annualization: int = 252,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    return_frame = pd.DataFrame({"X": returns}, index=index)
    weight_frame = pd.DataFrame({"X": weights}, index=index)
    return simulate_after_tax_portfolio_with_cash_yield(
        return_frame,
        weight_frame,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        tax_rate=0.24,
        cash_annual_yield=cash_yield,
        annualization=annualization,
        cash_interest_tax_rate=0.24,
    )


def test_profitable_sells_are_taxed_only_at_year_end() -> None:
    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-03", "2020-12-31"])
    _, taxes, _, _, _ = _simulate([0.10, 0.0, 0.10, 0.0], [1.0, 0.0, 1.0, 0.0], index)

    assert taxes.iloc[:-1].eq(0.0).all()
    assert taxes.iloc[-1] > 0.0


def test_loss_before_gain_offsets_same_year_taxable_income() -> None:
    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-03", "2020-12-31"])
    _, taxes, _, _, _ = _simulate([-0.10, 0.0, 0.10, 0.0], [1.0, 0.0, 1.0, 0.0], index)

    assert np.isclose(taxes.sum(), 0.0)


def test_net_loss_carryforward_offsets_next_year_gain() -> None:
    index = pd.to_datetime(["2020-01-02", "2020-12-31", "2021-01-04", "2021-12-31"])
    _, taxes, _, _, _ = _simulate([-0.10, 0.0, 0.20, 0.0], [1.0, 0.0, 1.0, 0.0], index)

    assert taxes.iloc[1] == 0.0
    assert np.isclose(taxes.iloc[-1], 0.24 * 0.08)


def test_cash_interest_is_taxed_at_year_end() -> None:
    index = pd.to_datetime(["2020-01-02", "2020-12-31"])
    _, taxes, _, cash_interest, _ = _simulate(
        [0.0, 0.0],
        [0.0, 0.0],
        index,
        cash_yield=0.10,
        annualization=2,
    )

    assert taxes.iloc[0] == 0.0
    assert np.isclose(taxes.iloc[-1], 0.24 * cash_interest.sum())
