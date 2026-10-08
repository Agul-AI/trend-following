"""Timestamp-available annual rates, with explicit elapsed-calendar accrual.

FRED downloads are latest-vintage historical observations. Assigned availability
lags reduce publication-time lookahead but do NOT establish point-in-time vintages.
No borrowing rate is inferred from BIL/ETF total returns; missing coverage fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

RATE_SOURCES = {
    "DFF": "https://fred.stlouisfed.org/series/DFF",
    "DTB3": "https://fred.stlouisfed.org/series/DTB3",
}
PUBLICATION_SOURCES = [
    "https://www.federalreserve.gov/releases/h15/",
    "https://www.newyorkfed.org/markets/reference-rates/effr",
]


def prepare_rate_observations(
    raw: pd.DataFrame,
    series_id: str,
    *,
    publication_lag_business_days: int = 2,
    publication_hour: int = 18,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Convert FRED percent quotes to decimal rates and modeled availability.

    Input: ``observation_date`` (or FRED's ``DATE``) and the series-ID column.
    A deliberately conservative two-US-federal-business-day, 18:00 local lag is
    a research assumption, not reconstructed historical publication timestamps.
    DTB3 bank-discount quotes are converted to a simple ACT/365 investment-yield
    proxy assuming 91-day maturity. This is not a traded bill total-return series.
    """
    if publication_lag_business_days < 1 or not 0 <= publication_hour <= 23:
        raise ValueError("Require at least one business-day lag and a valid hour")
    if series_id not in RATE_SOURCES:
        raise ValueError(f"Unsupported research rate series: {series_id}")
    date_column = "observation_date" if "observation_date" in raw else "DATE"
    dates = pd.to_datetime(raw[date_column], errors="raise")
    quotes = pd.to_numeric(raw[series_id], errors="coerce")
    missing_marker = raw[series_id].isna() | raw[series_id].astype(str).str.strip().isin([".", ""])
    if (quotes.isna() & ~missing_marker).any():
        raise ValueError("Invalid nonmissing rate quote")
    if dates.isna().any() or getattr(dates.dt, "tz", None) is not None:
        raise ValueError("FRED observation dates must be nonmissing naive dates")
    # FRED '.' means missing. It is removed, never replaced by zero or backfilled.
    valid = quotes.notna()
    result = pd.DataFrame({"observation_date": dates[valid], "quoted_percent": quotes[valid]})
    if result.empty or result["observation_date"].duplicated().any():
        raise ValueError("Rate observations must be nonempty with unique dates")
    if not np.isfinite(result["quoted_percent"]).all():
        raise ValueError("Nonfinite rate quote")
    result = result.sort_values("observation_date").reset_index(drop=True)
    offset = CustomBusinessDay(
        n=publication_lag_business_days, calendar=USFederalHolidayCalendar()
    )
    result["available_at"] = pd.DatetimeIndex(
        [date.normalize() + offset + pd.Timedelta(hours=publication_hour)
         for date in result["observation_date"]]
    ).tz_localize(timezone).tz_convert("UTC")
    rate = result["quoted_percent"] / 100.0
    if series_id == "DTB3":
        denominator = 1.0 - rate * 91.0 / 360.0
        if (denominator <= 0).any():
            raise ValueError("Invalid discount quote for assumed 91-day bill")
        rate = rate * (365.0 / 360.0) / denominator
    result["annual_rate"] = rate
    result["series_id"] = series_id
    result["source_url"] = RATE_SOURCES[series_id]
    result["vintage_type"] = "latest_vintage_not_point_in_time"
    result["availability_assumption"] = (
        f"observation_date_plus_{publication_lag_business_days}_US_federal_business_days_"
        f"at_{publication_hour:02d}:00_{timezone}"
    )
    return result


def _utc_index(index: pd.DatetimeIndex, label: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index)
    if index.tz is None:
        raise ValueError(f"{label} must be timezone-aware; localize explicitly")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{label} must be sorted, unique, and nonmissing")
    return index.tz_convert("UTC")


def rate_accrual(
    index: pd.DatetimeIndex,
    observations: pd.DataFrame,
    *,
    interval_start: pd.Timestamp | None = None,
    spread_bps: float = 0.0,
    day_count: float = 365.0,
    max_age_days: float = 10.0,
) -> pd.Series:
    """Integrate a known-at-the-time simple annual rate over each bar interval.

    Result t accrues from index[t-1] to index[t], including nights/weekends.
    A release inside an interval affects only the remainder of that interval;
    an observation released at its right endpoint affects the NEXT interval.
    The first result is zero unless ``interval_start`` explicitly precedes it.
    Rates are quoted decimal annual rates; financing spreads are basis points.
    Stale/missing coverage raises, never silently fills with zero.
    """
    original_index = pd.DatetimeIndex(index)
    ends = _utc_index(original_index, "index")
    if not np.isfinite([day_count, max_age_days, spread_bps]).all() or day_count <= 0 or max_age_days <= 0:
        raise ValueError("Invalid accrual parameters")
    if len(ends) == 0:
        return pd.Series(index=original_index, dtype=float, name="rate_accrual")
    obs = observations.copy()
    required = {"available_at", "annual_rate"}
    if not required.issubset(obs):
        raise ValueError(f"Rate observations require {required}")
    availability = pd.DatetimeIndex(obs["available_at"])
    if availability.tz is None:
        raise ValueError("available_at must be timezone-aware")
    obs["available_at"] = availability.tz_convert("UTC")
    sort_columns = ["available_at"] + (["observation_date"] if "observation_date" in obs else [])
    obs = obs.sort_values(sort_columns).drop_duplicates("available_at", keep="last")
    known = _utc_index(pd.DatetimeIndex(obs["available_at"]), "available_at")
    rates = pd.to_numeric(obs["annual_rate"], errors="raise").to_numpy(float)
    if len(known) == 0 or not np.isfinite(rates).all():
        raise ValueError("Empty or invalid annual rates")
    start = ends[0] if interval_start is None else pd.Timestamp(interval_start)
    if start.tzinfo is None:
        raise ValueError("interval_start must be timezone-aware")
    start = start.tz_convert("UTC")
    if start > ends[0]:
        raise ValueError("interval_start is after the first endpoint")
    out = []
    for end in ends:
        if end == start:
            out.append(0.0)
            continue
        position = int(known.searchsorted(start, side="right") - 1)
        if position < 0:
            raise ValueError(f"No published rate available at {start}")
        cursor, accrued = start, 0.0
        while cursor < end:
            next_release = known[position + 1] if position + 1 < len(known) else end
            stop = min(end, next_release)
            if stop - known[position] > pd.Timedelta(days=max_age_days):
                raise ValueError(f"Stale rate coverage at {stop}; latest availability {known[position]}")
            seconds = (stop - cursor).total_seconds()
            accrued += (rates[position] + spread_bps / 10000.0) * seconds / (86400.0 * day_count)
            cursor = stop
            if cursor < end:
                position += 1
        out.append(accrued)
        start = end
    return pd.Series(out, index=original_index, name="rate_accrual")


def financing_cash_accrual(
    index: pd.DatetimeIndex,
    financing_observations: pd.DataFrame,
    cash_observations: pd.DataFrame,
    *,
    financing_spread_bps: float = 50.0,
    cash_spread_bps: float = 0.0,
    interval_start: pd.Timestamp | None = None,
    max_age_days: float = 10.0,
) -> pd.DataFrame:
    """Separate borrowing and cash-investment assumptions on the same intervals."""
    return pd.DataFrame({
        "financing_return": rate_accrual(
            index, financing_observations, interval_start=interval_start,
            spread_bps=financing_spread_bps, day_count=360.0, max_age_days=max_age_days,
        ),
        "cash_return": rate_accrual(
            index, cash_observations, interval_start=interval_start,
            spread_bps=cash_spread_bps, day_count=365.0, max_age_days=max_age_days,
        ),
    })
