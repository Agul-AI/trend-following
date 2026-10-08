from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_following.research_rates import (
    financing_cash_accrual,
    prepare_rate_observations,
    rate_accrual,
)


def curve() -> pd.DataFrame:
    return pd.DataFrame({
        "available_at": pd.to_datetime(["2024-01-05T12:00Z", "2024-01-08T18:00Z"]),
        "annual_rate": [0.05, 0.10],
    })


def test_weekend_accrues_calendar_time_not_one_bar() -> None:
    index = pd.to_datetime(["2024-01-05T21:00Z", "2024-01-08T21:00Z"])
    # Monday 18:00 change: 69 hours at 5%, then 3 hours at 10%.
    result = rate_accrual(index, curve())
    assert result.iloc[0] == 0
    assert result.iloc[1] == pytest.approx((0.05 * 69 + 0.10 * 3) / (24 * 365))


def test_future_rate_perturbations_do_not_change_earlier_accrual() -> None:
    index = pd.to_datetime(["2024-01-05T21:00Z", "2024-01-08T17:00Z", "2024-01-08T21:00Z"])
    original = rate_accrual(index, curve())
    changed = curve()
    changed.loc[1, "annual_rate"] = 0.99
    updated = rate_accrual(index, changed)
    pd.testing.assert_series_equal(original.iloc[:2], updated.iloc[:2])
    assert original.iloc[2] != updated.iloc[2]


def test_rate_release_at_endpoint_cannot_change_past_interval() -> None:
    index = pd.to_datetime(["2024-01-08T17:00Z", "2024-01-08T18:00Z", "2024-01-08T19:00Z"])
    result = rate_accrual(index, curve())
    assert result.iloc[1] == pytest.approx(0.05 / (24 * 365))
    assert result.iloc[2] == pytest.approx(0.10 / (24 * 365))


def test_missing_or_stale_rate_errors_not_zero_fill() -> None:
    with pytest.raises(ValueError, match="No published rate"):
        rate_accrual(pd.to_datetime(["2024-01-04T21:00Z", "2024-01-05T21:00Z"]), curve())
    with pytest.raises(ValueError, match="Stale rate"):
        rate_accrual(pd.to_datetime(["2024-01-08T21:00Z", "2024-02-08T21:00Z"]), curve())


def test_explicit_first_interval_and_separate_day_counts() -> None:
    index = pd.to_datetime(["2024-01-05T21:00Z"])
    financing = curve()
    cash = curve().assign(annual_rate=0.03)
    result = financing_cash_accrual(
        index, financing, cash, financing_spread_bps=50,
        interval_start=pd.Timestamp("2024-01-05T12:00Z"),
    )
    assert result.financing_return.iloc[0] == pytest.approx(0.055 * 9 / (24 * 360))
    assert result.cash_return.iloc[0] == pytest.approx(0.03 * 9 / (24 * 365))


def test_prepare_availability_lag_holiday_and_units() -> None:
    raw = pd.DataFrame({"observation_date": ["2024-01-12", "2024-01-16"], "DFF": [5.0, "."]})
    obs = prepare_rate_observations(raw, "DFF")
    # Monday Jan15 is a Federal holiday; two business days from Friday -> Wed.
    assert obs.available_at.iloc[0] == pd.Timestamp("2024-01-17T23:00Z")
    assert obs.annual_rate.iloc[0] == 0.05
    assert len(obs) == 1
    assert obs.vintage_type.iloc[0] == "latest_vintage_not_point_in_time"
    with pytest.raises(ValueError, match="at least one"):
        prepare_rate_observations(raw, "DFF", publication_lag_business_days=0)


def test_discount_cash_quote_conversion() -> None:
    raw = pd.DataFrame({"DATE": ["2024-01-05"], "DTB3": [5.0]})
    value = prepare_rate_observations(raw, "DTB3").annual_rate.iloc[0]
    assert value == pytest.approx(0.05 * (365 / 360) / (1 - 0.05 * 91 / 360))


def test_ambiguous_timezone_and_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        rate_accrual(pd.date_range("2024-01-05", periods=2), curve())
    with pytest.raises(ValueError, match="sorted, unique"):
        rate_accrual(pd.to_datetime(["2024-01-05T21:00Z"] * 2), curve())
    with pytest.raises(ValueError, match="invalid annual rates"):
        rate_accrual(pd.to_datetime(["2024-01-05T21:00Z"]), curve().assign(annual_rate=np.nan))
    with pytest.raises(ValueError, match="Invalid nonmissing rate"):
        prepare_rate_observations(pd.DataFrame({"DATE": ["2024-01-05"], "DFF": ["bad"]}), "DFF")
    with pytest.raises(ValueError, match="Invalid accrual"):
        rate_accrual(pd.to_datetime(["2024-01-05T21:00Z"]), curve(), day_count=np.nan)
