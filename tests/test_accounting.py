from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_following.accounting import (
    annual_yield_to_period_return,
    cash_returns_for_mode,
    financing_returns_for_mode,
    simulate_after_tax_portfolio_with_cash_returns,
    synthetic_leveraged_return_variant,
    validate_cash_mode,
)


def test_cash_modes_are_cash_only_and_reject_risky_assets() -> None:
    validate_cash_mode("cash_0pct")
    validate_cash_mode("historical_tbill_yield_minus_50bps")

    with pytest.raises(ValueError, match="cash/T-bills"):
        validate_cash_mode("TLT_rotation")


def test_constant_cash_modes() -> None:
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    zero, zero_info = cash_returns_for_mode(index, "cash_0pct", annualization=252)
    cash3, cash3_info = cash_returns_for_mode(index, "cash_3pct_constant", annualization=252)

    assert zero.tolist() == [0.0, 0.0, 0.0]
    assert zero_info.source == "constant"
    assert cash3.iloc[0] == pytest.approx((1.03 ** (1 / 252)) - 1)
    assert cash3_info.mode == "cash_3pct_constant"


def test_historical_tbill_proxy_haircut_and_missing_fill() -> None:
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    proxy = pd.Series([0.001, np.nan, 0.002], index=index)
    cash, info = cash_returns_for_mode(
        index,
        "historical_tbill_yield_minus_25bps",
        annualization=252,
        tbill_proxy_returns=proxy,
        missing_fill="zero",
    )

    haircut = annual_yield_to_period_return(0.0025, 252)
    assert cash.iloc[0] == pytest.approx(0.001 - haircut)
    assert cash.iloc[1] == pytest.approx(-haircut)
    assert info.coverage == pytest.approx(2 / 3)
    assert info.haircut_bps == pytest.approx(25.0)


def test_financing_proxy_plus_spread() -> None:
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    proxy = pd.Series([0.0001, 0.0002], index=index)
    financing = financing_returns_for_mode(
        index,
        "historical_financing_proxy_plus_50bps",
        annualization=252,
        financing_proxy_returns=proxy,
    )

    assert financing.iloc[0] == pytest.approx(0.0001 + annual_yield_to_period_return(0.005, 252))


def test_synthetic_variants_subtract_expected_drags() -> None:
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    raw = pd.Series([0.03, -0.06], index=index)
    financing = pd.Series([0.001, 0.001], index=index)

    raw_variant = synthetic_leveraged_return_variant(raw, variant="qqq_3x_raw")
    adjusted = synthetic_leveraged_return_variant(
        raw,
        variant="qqq_3x_financing_adjusted",
        financing_returns=financing,
    )
    conservative = synthetic_leveraged_return_variant(
        raw,
        variant="qqq_3x_conservative",
        financing_returns=financing,
    )

    assert raw_variant.iloc[0] == pytest.approx(0.03)
    assert adjusted.iloc[0] < raw_variant.iloc[0]
    assert conservative.iloc[0] < adjusted.iloc[0]


def test_after_tax_simulator_uses_dynamic_cash_returns() -> None:
    index = pd.date_range("2020-12-30", periods=3, freq="D")
    returns = pd.DataFrame({"asset": [0.0, 0.10, 0.0]}, index=index)
    weights = pd.DataFrame({"asset": [0.0, 0.0, 0.0]}, index=index)
    cash_returns = pd.Series([0.001, 0.001, 0.001], index=index)

    after_tax, taxes, turnover, cash_interest, cash_weight = simulate_after_tax_portfolio_with_cash_returns(
        returns,
        weights,
        cash_returns=cash_returns,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        tax_rate=0.24,
        cash_interest_tax_rate=0.0,
    )

    assert turnover.sum() == pytest.approx(0.0)
    assert taxes.sum() == pytest.approx(0.0)
    assert cash_weight.mean() == pytest.approx(1.0)
    assert cash_interest.sum() > 0
    assert (1.0 + after_tax).prod() == pytest.approx((1.001) ** 3)
