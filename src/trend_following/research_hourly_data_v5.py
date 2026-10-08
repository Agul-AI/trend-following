"""Fail-closed v5 raw-price preparation; no legacy-cache mutation or backtests.

Explicit 30m mode preserves all 13 regular-session completed observations,
including 15:30–16:00. Legacy 60m mode retains six full hours plus a marking
half-hour. The literal ``is_full_hour`` flag never defines eligibility in 30m
mode. Corporate actions remain in raw physical-share units; retrospective
provider retrieval is not point-in-time availability, and adjusted caches are
never silently relabelled as raw.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OHLCV = ("open", "high", "low", "close", "volume")
DAILY_COLUMNS = ("session", "close", "dividend_amount", "split_coefficient")
CALENDAR_COLUMNS = ("session", "market_open", "market_close", "is_early_close")
BAR_COLUMNS = ("bar_start", "bar_end", "session", *OHLCV, "is_full_hour", "duration_minutes")
PRICE_BASIS = "raw_as_traded"
AV_URL = "https://www.alphavantage.co/query"
AV_DOCUMENTATION = "https://www.alphavantage.co/documentation/#intraday"
SPECIAL_OPEN_SOURCE = "https://www.nasdaqtrader.com/Content/NewsAlerts/ISEMics/MIC-2002-03%24Market_Opening_Time_-_September_11_2002%2420020909.pdf"
SPECIAL_OPEN_SESSION = "2002-09-11"


class DataContractError(ValueError):
    """Input cannot support the frozen raw-price/bar-time research contract."""


@dataclass(frozen=True)
class SourceMetadata:
    provider: str
    symbol: str
    interval_minutes: int
    timestamp_label: str
    timezone: str
    price_basis: str
    adjusted: bool
    observed_at: str
    verification: str
    source_files: tuple[str, ...] = field(default_factory=tuple)
    current_session_actions: dict[str, Any] | None = None

    def require_raw_start(self, allowed_intervals: tuple[int, ...] = (30,)) -> None:
        if self.adjusted is not False or self.price_basis != PRICE_BASIS:
            raise DataContractError(
                "verified RAW as-traded price basis required; adjusted=true or "
                "unknown caches cannot be implicitly treated as raw"
            )
        if self.timestamp_label != "start" or not self.verification.strip():
            raise DataContractError(
                "verified start-labelled bars required; end/unknown labels rejected"
            )
        if self.interval_minutes not in allowed_intervals:
            raise DataContractError(
                f"verified {allowed_intervals}-minute subbars required; "
                "lagging Alpha Vantage 60-minute bars must not be relabelled"
            )
        if not self.provider.strip() or not self.symbol.strip() or not self.timezone:
            raise DataContractError("provider, symbol and timezone provenance are required")
        utc(self.observed_at, "source observed_at")


def utc(value: Any, name: str = "timestamp") -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise DataContractError(f"{name} must be a nonmissing timezone-aware timestamp")
    return timestamp.tz_convert("UTC")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _session_strings(values: Any, name: str = "session") -> pd.Series:
    result = []
    for value in values:
        if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            date = pd.Timestamp(value)
            if pd.isna(date):
                raise DataContractError(f"invalid {name}: {value!r}")
            result.append(value)
            continue
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise DataContractError(f"missing {name}")
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("America/New_York")
        if stamp != stamp.normalize():
            raise DataContractError(f"{name} must be a calendar date, not an intraday timestamp")
        result.append(stamp.strftime("%Y-%m-%d"))
    return pd.Series(result, dtype="str")


def _required(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise DataContractError(f"{name} missing columns: {missing}")
    if frame.empty:
        raise DataContractError(f"{name} must not be empty")


def _ordered_unique(values: pd.Series | pd.DatetimeIndex, name: str) -> None:
    if pd.Index(values).has_duplicates or not pd.Index(values).is_monotonic_increasing:
        raise DataContractError(f"{name} must be unique and chronologically ordered")


def build_xnys_calendar(start_session: str, end_session: str) -> pd.DataFrame:
    """Generate the exact XNYS schedule, including historical special closures.

    Freeze the resulting CSV and package/version provenance; do not regenerate
    a frozen packet using a newer calendar dependency behind the user's back.
    """
    dates = _session_strings([start_session, end_session])
    if dates.iloc[0] > dates.iloc[1]:
        raise DataContractError("calendar start must not follow end")
    try:
        import exchange_calendars as xc
    except ImportError as error:
        raise DataContractError(
            "exchange_calendars is required to generate frozen XNYS sessions"
        ) from error
    # The dependency requires start < end, even for a one-session request.
    schedule_start = pd.Timestamp(dates.iloc[0]) - pd.Timedelta(days=7)
    schedule_end = pd.Timestamp(dates.iloc[1]) + pd.Timedelta(days=7)
    calendar = xc.get_calendar("XNYS", start=schedule_start, end=schedule_end)
    schedule = calendar.schedule.loc[dates.iloc[0] : dates.iloc[1]]
    if schedule.empty:
        raise DataContractError("calendar range has no XNYS sessions")
    open_column = "open" if "open" in schedule else "market_open"
    close_column = "close" if "close" in schedule else "market_close"
    out = pd.DataFrame(
        {
            "session": schedule.index.strftime("%Y-%m-%d"),
            "market_open": pd.to_datetime(schedule[open_column].to_numpy(), utc=True),
            "market_close": pd.to_datetime(schedule[close_column].to_numpy(), utc=True),
        }
    )
    # Calendar-library exception corrected from an official exchange bulletin,
    # never inferred from absent provider trades. This is the ANNOUNCED 11:00
    # scheduled-open assumption, not evidence of an actual observed first trade.
    exception = out.session.eq(SPECIAL_OPEN_SESSION)
    if exception.any():
        out.loc[exception, "market_open"] = pd.Timestamp(
            "2002-09-11 11:00", tz="America/New_York"
        ).tz_convert("UTC")
    out["is_early_close"] = out.market_close.dt.tz_convert("America/New_York").dt.hour.lt(16)
    out.attrs["calendar_provenance"] = {
        "name": "XNYS",
        "package": "exchange_calendars",
        "package_version": importlib.metadata.version("exchange_calendars"),
        "requested_start_session": dates.iloc[0],
        "requested_end_session": dates.iloc[1],
        "availability": "retrospectively_generated_schedule_not_point_in_time_evidence",
        "bar_anchor": "session_open_normally_09:30_ET_except_documented_special_open",
        "scheduled_open_overrides": [
            {
                "session": SPECIAL_OPEN_SESSION,
                "scheduled_open_local": "11:00 America/New_York",
                "source_url": SPECIAL_OPEN_SOURCE,
                "basis": "official_announced_scheduled_open_assumption_not_actual_first_trade_measurement",
                "calendar_library_issue": "XNYS_4.13.2_generates_09:30_for_delayed_open",
            }
        ]
        if exception.any()
        else [],
    }
    return out


def validate_calendar(calendar: pd.DataFrame, *, verify_xnys: bool = True) -> pd.DataFrame:
    _required(calendar, CALENDAR_COLUMNS, "calendar")
    out = calendar.loc[:, CALENDAR_COLUMNS].copy().reset_index(drop=True)
    out["session"] = _session_strings(out.session)
    _ordered_unique(out.session, "calendar sessions")
    for column in ("market_open", "market_close"):
        out[column] = pd.to_datetime([utc(value, column) for value in out[column]], utc=True)
    if not out.is_early_close.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise DataContractError("calendar is_early_close must contain booleans")
    for row in out.itertuples():
        op = row.market_open.tz_convert("America/New_York")
        cl = row.market_close.tz_convert("America/New_York")
        if (
            op.strftime("%Y-%m-%d") != row.session
            or cl.date() != op.date()
            or op.strftime("%H:%M:%S")
            != ("11:00:00" if row.session == SPECIAL_OPEN_SESSION else "09:30:00")
            or cl <= op
            or cl > op.normalize() + pd.Timedelta(hours=16)
            or bool(row.is_early_close) != (cl.hour < 16)
        ):
            raise DataContractError(f"inconsistent calendar open/close: {row.session}")
    if verify_xnys:
        frozen_version = calendar.attrs.get("calendar_provenance", {}).get("package_version")
        if frozen_version is not None and frozen_version != importlib.metadata.version(
            "exchange_calendars"
        ):
            raise DataContractError(
                "calendar dependency version changed from frozen provenance; explicit re-audit required"
            )
        reference = build_xnys_calendar(out.session.iloc[0], out.session.iloc[-1])
        if not out.equals(reference):
            raise DataContractError(
                "calendar differs from XNYS: missing sessions or incorrect historical open/close"
            )
        out.attrs["calendar_provenance"] = reference.attrs["calendar_provenance"]
    out.attrs.update(calendar.attrs)
    return out


def _sampling_interval(value: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value not in (30, 60)
    ):
        raise DataContractError("sampling_interval_minutes must be explicitly 30 or 60")
    return int(value)


def canonical_interval_grid(calendar: pd.DataFrame, interval_minutes: int = 30) -> pd.DataFrame:
    """Session-open anchored grid with explicit observation interval.

    A normal 30m session has 13 complete observations, including 15:30–16:00;
    a 13:00 early close has seven. ``is_full_hour`` is retained as a legacy
    literal duration flag, NOT the eligibility rule for the 30m study.
    """
    interval_minutes = _sampling_interval(interval_minutes)
    calendar = validate_calendar(calendar)
    rows = []
    for row in calendar.itertuples():
        for start in pd.date_range(
            row.market_open, row.market_close, freq=f"{interval_minutes}min", inclusive="left"
        ):
            end = min(start + pd.Timedelta(minutes=interval_minutes), row.market_close)
            minutes = (end - start).total_seconds() / 60
            rows.append(
                {
                    "bar_start": start,
                    "bar_end": end,
                    "session": row.session,
                    "is_full_hour": minutes == 60,
                    "duration_minutes": minutes,
                }
            )
    return pd.DataFrame(rows)


def canonical_hour_grid(calendar: pd.DataFrame) -> pd.DataFrame:
    """Legacy/generic 60m grid; residual bars are marking-only in that mode."""
    return canonical_interval_grid(calendar, 60)


def canonical_half_hour_grid(calendar: pd.DataFrame) -> pd.DataFrame:
    """Every completed 30m observation; no final half-hour exclusion."""
    return canonical_interval_grid(calendar, 30)


def normalize_daily_actions(frame: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Use DAILY_ADJUSTED ``close`` (raw), never ``adj_close``.

    Daily data may cover more dates than the frozen calendar; input rows remain
    ordered/unique and the selected calendar range must be complete.
    """
    _required(frame, ("close", "dividend_amount", "split_coefficient"), "daily actions")
    out = frame.copy().reset_index(drop=True)
    date_column = "session" if "session" in out else "date" if "date" in out else None
    if date_column is None:
        raise DataContractError("daily data requires session or date column")
    out["session"] = _session_strings(out[date_column])
    _ordered_unique(out.session, "daily sessions")
    for column in DAILY_COLUMNS[1:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    values = out[list(DAILY_COLUMNS[1:])].to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or out.close.le(0).any()
        or out.dividend_amount.lt(0).any()
        or out.split_coefficient.le(0).any()
    ):
        raise DataContractError(
            "daily raw close/actions must be finite, close/split positive, dividend nonnegative"
        )
    cal = validate_calendar(calendar)
    selected = out[out.session.isin(cal.session)].loc[:, DAILY_COLUMNS].reset_index(drop=True)
    missing = sorted(set(cal.session).difference(selected.session))
    if missing:
        raise DataContractError(
            f"daily corporate-action coverage missing {len(missing)} XNYS sessions: {missing[:10]}"
        )
    return selected


def _provider_bars(
    frame: pd.DataFrame, metadata: SourceMetadata, *, check_order: bool = True
) -> pd.DataFrame:
    _required(frame, OHLCV, "provider bars")
    out = frame.copy().reset_index(drop=True)
    if "bar_start" in out:
        values = out.bar_start
    elif "date" in out:
        values = out.date
    elif "timestamp" in out:
        values = out.timestamp
    elif isinstance(frame.index, pd.DatetimeIndex):
        values = frame.index
    else:
        raise DataContractError("provider bars require date/timestamp/bar_start or DatetimeIndex")
    starts = pd.DatetimeIndex(pd.to_datetime(values))
    if starts.isna().any():
        raise DataContractError("provider timestamps must not be missing")
    if starts.tz is None:
        try:
            starts = starts.tz_localize(metadata.timezone, ambiguous="raise", nonexistent="raise")
        except Exception as error:
            raise DataContractError(
                "cannot localize provider timestamps under source timezone"
            ) from error
    starts = starts.tz_convert("UTC")
    if check_order:
        _ordered_unique(starts, "provider timestamps")
    out["bar_start"] = starts
    for column in OHLCV:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    values = out[list(OHLCV)].to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or out[list(OHLCV[:4])].le(0).any().any()
        or out.volume.lt(0).any()
    ):
        raise DataContractError(
            "provider OHLCV must be finite; prices positive and volume nonnegative"
        )
    if (
        out.high.lt(out[["open", "close", "low"]].max(axis=1)).any()
        or out.low.gt(out[["open", "close", "high"]].min(axis=1)).any()
    ):
        raise DataContractError("provider OHLC inconsistent high/low envelope")
    return out[["bar_start", *OHLCV]]


def _subbar_grid(calendar: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    if not isinstance(interval_minutes, int) or interval_minutes <= 0:
        raise DataContractError("positive integer provider interval_minutes required")
    rows = []
    for row in validate_calendar(calendar).itertuples():
        for start in pd.date_range(
            row.market_open, row.market_close, freq=f"{interval_minutes}min", inclusive="left"
        ):
            rows.append(
                {
                    "bar_start": start,
                    "bar_end": min(
                        start + pd.Timedelta(minutes=interval_minutes), row.market_close
                    ),
                    "session": row.session,
                }
            )
    return pd.DataFrame(rows)


def audit_source_coverage(
    frame: pd.DataFrame, calendar: pd.DataFrame, metadata: SourceMetadata
) -> pd.DataFrame:
    """Read-only gap report; coverage does not verify price basis or labels.

    An adjusted source can be audited but cannot pass normalization. Calendar
    controls the whole intended range, including entirely absent sessions.
    """
    source = _provider_bars(frame, metadata, check_order=False)
    grid = _subbar_grid(calendar, metadata.interval_minutes)
    actual = pd.DatetimeIndex(source.bar_start)
    expected = pd.DatetimeIndex(grid.bar_start)
    rows = []
    for kind, stamps in (
        ("missing_subbar", expected.difference(actual)),
        ("extra_or_off_grid_subbar", actual.difference(expected)),
        ("duplicate_timestamp", actual[actual.duplicated()]),
    ):
        for stamp in stamps:
            rows.append(
                {
                    "issue": kind,
                    "session": stamp.tz_convert("America/New_York").strftime("%Y-%m-%d"),
                    "timestamp_utc": stamp.isoformat(),
                }
            )
    if not actual.is_monotonic_increasing:
        rows.append({"issue": "unordered_timestamps", "session": "", "timestamp_utc": ""})
    out = pd.DataFrame(rows, columns=["issue", "session", "timestamp_utc"])
    out.attrs["coverage"] = {
        "expected_subbars": len(expected),
        "observed_subbars": len(actual),
        "complete": out.empty,
        "price_basis": metadata.price_basis,
        "adjusted": metadata.adjusted,
        "timestamp_label": metadata.timestamp_label,
        "basis_verified": metadata.adjusted is False and metadata.price_basis == PRICE_BASIS,
        "warning": "timestamp coverage alone never verifies raw price basis or bar-label semantics",
    }
    return out


def normalize_raw_subbars(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    metadata: SourceMetadata,
    *,
    as_of: Any,
    sampling_interval_minutes: int = 60,
) -> pd.DataFrame:
    """Normalize verified raw 30m: direct observations in 30m mode, aggregate in legacy 60m mode.

    Missing/extra subbars and unfinished bars raise. No forward fill, daily-bar
    substitution, backward shift or retrospective adjustment reconstruction.
    """
    sampling_interval_minutes = _sampling_interval(sampling_interval_minutes)
    metadata.require_raw_start()
    now = utc(as_of, "as_of")
    if utc(metadata.observed_at, "source observed_at") > now:
        raise DataContractError("source observed_at exceeds preparation as_of")
    source = _provider_bars(frame, metadata)
    report = audit_source_coverage(frame, calendar, metadata)
    if not report.empty:
        sample = report.head(8).to_dict(orient="records")
        raise DataContractError(
            f"unrepaired provider-grid gaps/extra bars ({len(report)} issues): {sample}"
        )
    subgrid = _subbar_grid(calendar, metadata.interval_minutes)
    unfinished = subgrid.bar_end.gt(now)
    if unfinished.any():
        raise DataContractError(
            f"unfinished/future subbars as of {now.isoformat()}: {subgrid.loc[unfinished, 'bar_end'].iloc[0]}"
        )
    indexed = source.set_index("bar_start")
    rows = []
    for row in canonical_interval_grid(calendar, sampling_interval_minutes).itertuples():
        constituents = indexed.loc[(indexed.index >= row.bar_start) & (indexed.index < row.bar_end)]
        rows.append(
            {
                "bar_start": row.bar_start,
                "bar_end": row.bar_end,
                "session": row.session,
                "open": float(constituents.open.iloc[0]),
                "high": float(constituents.high.max()),
                "low": float(constituents.low.min()),
                "close": float(constituents.close.iloc[-1]),
                "volume": float(constituents.volume.sum()),
                "is_full_hour": row.is_full_hour,
                "duration_minutes": row.duration_minutes,
            }
        )
    bars = pd.DataFrame(rows, columns=BAR_COLUMNS)
    bars.attrs["sampling_interval_minutes"] = sampling_interval_minutes
    bars.attrs["source_metadata"] = asdict(metadata)
    return bars


def normalize_provider_present_day(
    frame: pd.DataFrame, calendar: pd.DataFrame, metadata: SourceMetadata, *, as_of: Any
) -> pd.DataFrame:
    """Separate Yahoo/current-session monitor adapter; never historical splicing.

    Yahoo ``auto_adjust=False`` is not, alone, physical-share verification. The
    metadata must explicitly verify raw present-day units and start labelling.
    Completed 09:30-anchored 60m bars are accepted only for the as_of NY session.
    A moving unfinished bar raises instead of producing a research signal.
    """
    metadata.require_raw_start((60,))
    now = utc(as_of, "as_of")
    if utc(metadata.observed_at, "source observed_at") > now:
        raise DataContractError("source observed_at exceeds monitor as_of")
    cal = validate_calendar(calendar)
    today = now.tz_convert("America/New_York").strftime("%Y-%m-%d")
    if len(cal) != 1 or cal.session.iloc[0] != today:
        raise DataContractError("present-day monitor cannot splice historical sessions")
    source = _provider_bars(frame, metadata)
    grid = canonical_hour_grid(cal)
    lookup = grid.set_index("bar_start")
    if not source.bar_start.isin(grid.bar_start).all():
        raise DataContractError(
            "monitor bars do not match verified 09:30-anchored canonical starts"
        )
    ends = source.bar_start.map(lookup.bar_end)
    if ends.gt(now).any():
        raise DataContractError("unfinished monitor bar rejected at moving as_of")
    expected = grid.loc[grid.bar_end.le(now), "bar_start"]
    if set(source.bar_start) != set(expected):
        raise DataContractError("missing completed present-day monitor bars")
    bars = source.merge(grid, on="bar_start", validate="one_to_one")[list(BAR_COLUMNS)]
    bars.attrs["source_metadata"] = asdict(metadata)
    bars.attrs["role"] = "monitor_only_not_historical_research_splice"
    return bars


def validate_research_inputs(
    daily: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    as_of: Any,
    close_rtol: float = 0.001,
    close_atol: float = 0.02,
    bar_start_session: str | None = None,
    sampling_interval_minutes: int = 60,
) -> dict[str, Any]:
    """Strict completeness and corporate-action checks, with explicit close tolerance.

    Daily official/auction closes can differ from the last intraday trade; the
    frozen tolerance is recorded, not relaxed automatically upon a failure.
    """
    if (
        not np.isfinite(close_rtol)
        or not np.isfinite(close_atol)
        or min(close_rtol, close_atol) < 0
    ):
        raise DataContractError("close matching tolerances must be finite and nonnegative")
    sampling_interval_minutes = _sampling_interval(sampling_interval_minutes)
    now = utc(as_of, "as_of")
    cal = validate_calendar(calendar)
    _required(daily, DAILY_COLUMNS, "daily actions")
    normalized_daily = normalize_daily_actions(daily, cal)
    daily_sessions = _session_strings(daily.session)
    if list(daily_sessions) != list(normalized_daily.session):
        raise DataContractError("daily session coverage must match the frozen calendar exactly")
    _required(bars, BAR_COLUMNS, "canonical bars")
    out = bars.loc[:, BAR_COLUMNS].copy().reset_index(drop=True)
    for column in ("bar_start", "bar_end"):
        out[column] = pd.to_datetime([utc(value, column) for value in out[column]], utc=True)
        _ordered_unique(out[column], column)
    out["session"] = _session_strings(out.session)
    out["duration_minutes"] = pd.to_numeric(out.duration_minutes, errors="coerce").astype(float)
    if not out.is_full_hour.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise DataContractError("is_full_hour must contain booleans")
    declared_bar_start = (
        cal.session.iloc[0]
        if bar_start_session is None
        else _session_strings([bar_start_session]).iloc[0]
    )
    if declared_bar_start not in set(cal.session):
        raise DataContractError("declared bar_start_session must be a frozen XNYS session")
    bar_calendar = cal.loc[cal.session.ge(declared_bar_start)].reset_index(drop=True)
    grid = canonical_interval_grid(bar_calendar, sampling_interval_minutes)
    shape_columns = ["bar_start", "bar_end", "session", "is_full_hour", "duration_minutes"]
    if not out[shape_columns].equals(grid[shape_columns]):
        raise DataContractError(
            "canonical grid mismatch: gaps, missing sessions, incorrect bar ends/full-hour flags"
        )
    if out.bar_end.gt(now).any():
        raise DataContractError("unfinished/future canonical bars rejected at moving as_of")
    if cal.market_close.gt(now).any():
        raise DataContractError("historical research packet contains an unfinished daily session")
    _provider_bars(
        out,
        SourceMetadata(
            "canonical",
            "QQQ",
            30,
            "start",
            "UTC",
            PRICE_BASIS,
            False,
            now.isoformat(),
            "canonical validation",
        ),
    )
    final = out.groupby("session", sort=False).last()
    compare = normalized_daily.set_index("session").close.reindex(bar_calendar.session)
    matches = np.isclose(
        final.close.to_numpy(), compare.to_numpy(), rtol=close_rtol, atol=close_atol
    )
    if not matches.all():
        bad = (
            pd.DataFrame(
                {
                    "session": compare.index[~matches],
                    "intraday_last_close": final.close.to_numpy()[~matches],
                    "daily_raw_close": compare.to_numpy()[~matches],
                }
            )
            .head(8)
            .to_dict(orient="records")
        )
        raise DataContractError(
            f"bar-end close/daily RAW close mismatch; check price basis/closing coverage: {bad}"
        )
    return {
        "complete": True,
        "sessions": len(cal),
        "bars": len(out),
        "full_hours": int(out.is_full_hour.sum()),
        "partial_marking_bars": int(out.duration_minutes.lt(sampling_interval_minutes).sum()),
        "full_observations": int(out.duration_minutes.eq(sampling_interval_minutes).sum()),
        "sampling_interval_minutes": sampling_interval_minutes,
        "observation_interval_minutes": sampling_interval_minutes,
        "start_session": cal.session.iloc[0],
        "end_session": cal.session.iloc[-1],
        "bar_start_session": declared_bar_start,
        "as_of": now.isoformat(),
        "close_rtol": close_rtol,
        "close_atol": close_atol,
    }


def normalize_monitor_subbars(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    metadata: SourceMetadata,
    *,
    as_of: Any,
    sampling_interval_minutes: int = 30,
) -> pd.DataFrame:
    """Normalize completed raw 30m monitor data at an explicitly frozen interval.

    Unlike the strict historical normalizer this accepts a currently incomplete
    SESSION, but not an incomplete provider subbar. In 30m mode EVERY completed
    subbar is emitted, even within an unfinished clock hour. Only legacy 60m
    mode excludes completed subbars awaiting completion of the canonical hour.
    """
    sampling_interval_minutes = _sampling_interval(sampling_interval_minutes)
    metadata.require_raw_start()
    now = utc(as_of, "as_of")
    if utc(metadata.observed_at) > now:
        raise DataContractError("source observed_at exceeds monitor as_of")
    source = _provider_bars(frame, metadata)
    subgrid = _subbar_grid(calendar, 30)
    known = subgrid.loc[subgrid.bar_end.le(now)]
    if not source.bar_start.isin(known.bar_start).all():
        raise DataContractError("unfinished or off-grid monitor subbar rejected")
    if set(source.bar_start) != set(known.bar_start):
        raise DataContractError("missing completed monitor subbars; no padding permitted")
    indexed = source.set_index("bar_start")
    full_grid = canonical_interval_grid(calendar, sampling_interval_minutes)
    complete_grid = full_grid.loc[full_grid.bar_end.le(now)]
    rows, consumed = [], 0
    for row in complete_grid.itertuples():
        part = indexed.loc[(indexed.index >= row.bar_start) & (indexed.index < row.bar_end)]
        consumed += len(part)
        rows.append(
            {
                "bar_start": row.bar_start,
                "bar_end": row.bar_end,
                "session": row.session,
                "open": float(part.open.iloc[0]),
                "high": float(part.high.max()),
                "low": float(part.low.min()),
                "close": float(part.close.iloc[-1]),
                "volume": float(part.volume.sum()),
                "is_full_hour": row.is_full_hour,
                "duration_minutes": row.duration_minutes,
            }
        )
    bars = pd.DataFrame(rows, columns=BAR_COLUMNS)
    bars.attrs["sampling_interval_minutes"] = sampling_interval_minutes
    bars.attrs["source_metadata"] = asdict(metadata)
    bars.attrs["role"] = "monitor_only_not_historical_research_splice"
    bars.attrs["completed_subbars_within_unfinished_hour_excluded"] = len(source) - consumed
    return bars


def validate_monitor_inputs(
    daily_completed: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: pd.DataFrame,
    metadata: SourceMetadata,
    *,
    as_of: Any,
    bar_start_session: str | None = None,
    close_rtol: float = 0.001,
    close_atol: float = 0.02,
    sampling_interval_minutes: int = 30,
) -> dict[str, Any]:
    """Separate monitoring contract: completed bars, no fabricated daily close.

    Calendar ends on today's NY session. Daily history stops at the prior
    session; today's verified actions are separately available by the open.
    All historical bars and completed current bars remain gap-free. No assumed
    flat paper state or executable action is created by this data adapter.
    """
    sampling_interval_minutes = _sampling_interval(sampling_interval_minutes)
    metadata.require_raw_start((30,) if sampling_interval_minutes == 30 else (30, 60))
    now = utc(as_of, "as_of")
    if utc(metadata.observed_at) > now:
        raise DataContractError("source observed_at exceeds monitor as_of")
    if (
        not np.isfinite(close_rtol)
        or not np.isfinite(close_atol)
        or min(close_rtol, close_atol) < 0
    ):
        raise DataContractError("close matching tolerances must be finite and nonnegative")
    cal = validate_calendar(calendar)
    today = now.tz_convert("America/New_York").strftime("%Y-%m-%d")
    if cal.session.iloc[-1] != today:
        raise DataContractError("monitor calendar must end on the current NY trading session")
    actions = metadata.current_session_actions
    required = {"session", "dividend_amount", "split_coefficient", "available_at", "verification"}
    if not isinstance(actions, dict) or not required.issubset(actions):
        raise DataContractError(
            "monitor requires separately verified current-session corporate actions"
        )
    if actions["session"] != today or not str(actions["verification"]).strip():
        raise DataContractError("current-session action verification/session mismatch")
    dividend, split = float(actions["dividend_amount"]), float(actions["split_coefficient"])
    if not np.isfinite([dividend, split]).all() or dividend < 0 or split <= 0:
        raise DataContractError("invalid current-session dividend/split")
    today_open = cal.market_open.iloc[-1]
    if utc(actions["available_at"], "current actions available_at") > min(today_open, now):
        raise DataContractError(
            "current-session corporate actions were not known by the session open"
        )
    previous_calendar = cal[cal.session.lt(today)].reset_index(drop=True)
    _required(daily_completed, DAILY_COLUMNS, "completed daily history")
    if previous_calendar.empty:
        raise DataContractError("monitor requires completed prior daily history")
    normalized_daily = normalize_daily_actions(daily_completed, previous_calendar)
    if list(_session_strings(daily_completed.session)) != list(normalized_daily.session):
        raise DataContractError(
            "monitor daily history must exactly cover prior sessions and exclude today's close"
        )
    _required(bars, BAR_COLUMNS, "completed monitor bars")
    out = bars.loc[:, BAR_COLUMNS].copy().reset_index(drop=True)
    for column in ("bar_start", "bar_end"):
        out[column] = pd.to_datetime([utc(value, column) for value in out[column]], utc=True)
        _ordered_unique(out[column], column)
    out["session"] = _session_strings(out.session)
    out["duration_minutes"] = pd.to_numeric(out.duration_minutes, errors="coerce").astype(float)
    if not out.is_full_hour.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise DataContractError("monitor is_full_hour must contain booleans")
    if out.bar_end.gt(now).any():
        raise DataContractError("unfinished monitor bar rejected at moving as_of")
    declared = (
        cal.session.iloc[0]
        if bar_start_session is None
        else _session_strings([bar_start_session]).iloc[0]
    )
    if declared not in set(cal.session):
        raise DataContractError("declared monitor bar_start_session must be a frozen XNYS session")
    grid = canonical_interval_grid(
        cal[cal.session.ge(declared)].reset_index(drop=True), sampling_interval_minutes
    )
    grid = grid[grid.bar_end.le(now)].reset_index(drop=True)
    columns = ["bar_start", "bar_end", "session", "is_full_hour", "duration_minutes"]
    if not out[columns].equals(grid[columns]):
        raise DataContractError(
            "completed monitor grid mismatch: historical/current gaps or incorrect labels"
        )
    _provider_bars(out, metadata)
    previous_bars = out[out.session.lt(today)]
    if not previous_bars.empty:
        final = previous_bars.groupby("session", sort=False).last().close
        daily_closes = normalized_daily.set_index("session").close.reindex(final.index)
        if not np.isclose(
            final.to_numpy(), daily_closes.to_numpy(), rtol=close_rtol, atol=close_atol
        ).all():
            raise DataContractError("completed prior bar-end close/daily RAW close mismatch")
    return {
        "complete": True,
        "role": "monitor_only_not_historical_research_splice",
        "sessions": len(cal),
        "completed_daily_sessions": len(normalized_daily),
        "bars": len(out),
        "full_hours": int(out.is_full_hour.sum()),
        "partial_marking_bars": int(out.duration_minutes.lt(sampling_interval_minutes).sum()),
        "full_observations": int(out.duration_minutes.eq(sampling_interval_minutes).sum()),
        "sampling_interval_minutes": sampling_interval_minutes,
        "observation_interval_minutes": sampling_interval_minutes,
        "start_session": cal.session.iloc[0],
        "end_session": today,
        "bar_start_session": declared,
        "as_of": now.isoformat(),
        "close_rtol": close_rtol,
        "close_atol": close_atol,
        "current_session_actions": actions,
    }


def validate_cash_observations(
    frame: pd.DataFrame, *, as_of: Any, first_bar_start: Any | None = None
) -> pd.DataFrame:
    """Preserve timestamped cash-yield observations; never use undated yields."""
    _required(frame, ("available_at", "annual_rate"), "cash observations")
    out = frame.copy().reset_index(drop=True)
    out["available_at"] = pd.to_datetime(
        [utc(value, "cash available_at") for value in out.available_at], utc=True
    )
    if not out.available_at.is_monotonic_increasing:
        raise DataContractError("cash available_at must be chronologically ordered")
    if out.available_at.duplicated().any() and "observation_date" not in out:
        raise DataContractError(
            "tied cash available_at requires observation_date for explicit latest-observation tie policy"
        )
    out["annual_rate"] = pd.to_numeric(out.annual_rate, errors="coerce")
    if not np.isfinite(out.annual_rate).all() or out.annual_rate.le(-1).any():
        raise DataContractError("cash annual_rate must be finite and greater than -1")
    if out.available_at.gt(utc(as_of, "as_of")).any():
        raise DataContractError("cash observations include unavailable/future rates at as_of")
    if first_bar_start is not None and not out.available_at.le(utc(first_bar_start)).any():
        raise DataContractError("cash observations lack an available rate at first bar start")
    if "observation_date" in out:
        dates = _session_strings(out.observation_date, "cash observation_date")
        if (
            dates > out.available_at.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        ).any():
            raise DataContractError("cash rate marked available before its observation date")
        out["observation_date"] = dates
        pairs = pd.MultiIndex.from_arrays([out.available_at, out.observation_date])
        if pairs.has_duplicates or not pairs.is_monotonic_increasing:
            raise DataContractError(
                "cash (available_at, observation_date) must be unique and ordered"
            )
    out.attrs["availability_tie_policy"] = "latest_observation_date_at_same_available_at"
    return out


def _daily_source_attestation(value: Any, *, as_of: Any) -> dict[str, Any] | None:
    """Preserve independent daily-source identity, never infer it from hourly."""
    if value is None:
        return None
    if not isinstance(value, dict) or not value.get("provider") or not value.get("symbol"):
        raise DataContractError(
            "daily_source_metadata must explicitly identify provider and symbol"
        )
    if value.get("price_basis", PRICE_BASIS) != PRICE_BASIS:
        raise DataContractError("daily source attestation must identify raw as-traded price basis")
    if "observed_at" in value and utc(value["observed_at"], "daily source observed_at") > utc(
        as_of
    ):
        raise DataContractError("daily source observed_at exceeds packet as_of")
    return json.loads(json.dumps(value))


def prepare_input_packet(
    output_dir: str | Path,
    daily: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: pd.DataFrame,
    metadata: SourceMetadata,
    *,
    as_of: Any,
    close_rtol: float = 0.001,
    close_atol: float = 0.02,
    cash_observations: pd.DataFrame | None = None,
    cash_source_file: str | Path | None = None,
    bar_start_session: str | None = None,
    packet_kind: str = "historical",
    daily_source_metadata: dict[str, Any] | None = None,
    sampling_interval_minutes: int = 60,
) -> dict[str, Any]:
    """Stage and atomically install an immutable hash-bound input packet."""
    sampling_interval_minutes = _sampling_interval(sampling_interval_minutes)
    if packet_kind not in ("historical", "monitor"):
        raise DataContractError("packet_kind must be historical or monitor")
    metadata.require_raw_start(
        (30, 60) if packet_kind == "monitor" and sampling_interval_minutes == 60 else (30,)
    )
    now = utc(as_of, "as_of")
    if utc(metadata.observed_at, "source observed_at") > now:
        raise DataContractError("source observed_at exceeds packet as_of")
    daily_attestation = _daily_source_attestation(
        daily_source_metadata
        if daily_source_metadata is not None
        else daily.attrs.get("source_metadata"),
        as_of=as_of,
    )
    if daily_attestation is not None and daily_attestation["symbol"] != metadata.symbol:
        raise DataContractError("daily and intraday source symbols must match")
    if bars.attrs.get("source_metadata") != asdict(metadata):
        raise DataContractError(
            "canonical bars must carry matching verified normalization provenance"
        )
    if packet_kind == "historical":
        audit = validate_research_inputs(
            daily,
            bars,
            calendar,
            as_of=as_of,
            close_rtol=close_rtol,
            close_atol=close_atol,
            bar_start_session=bar_start_session,
            sampling_interval_minutes=sampling_interval_minutes,
        )
    else:
        audit = validate_monitor_inputs(
            daily,
            bars,
            calendar,
            metadata,
            as_of=as_of,
            close_rtol=close_rtol,
            close_atol=close_atol,
            bar_start_session=bar_start_session,
            sampling_interval_minutes=sampling_interval_minutes,
        )
    cash = (
        None
        if cash_observations is None
        else validate_cash_observations(
            cash_observations, as_of=as_of, first_bar_start=bars.bar_start.iloc[0]
        )
    )
    if cash_source_file is not None and cash is None:
        raise DataContractError("cash source provenance supplied without cash observations")
    sources = []
    source_paths = (*metadata.source_files, *((str(cash_source_file),) if cash_source_file else ()))
    for path_string in source_paths:
        path = Path(path_string)
        if not path.is_file() or path.is_symlink():
            raise DataContractError(f"source provenance missing/symlink file: {path}")
        sources.append(
            {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"immutable input packet already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        daily.loc[:, DAILY_COLUMNS].to_csv(stage / "daily.csv", index=False)
        bars.loc[:, BAR_COLUMNS].to_csv(stage / "bars.csv", index=False)
        calendar.loc[:, CALENDAR_COLUMNS].to_csv(stage / "calendar.csv", index=False)
        file_names = ["daily.csv", "bars.csv", "calendar.csv"]
        if cash is not None:
            cash.to_csv(stage / "DTB3_available.csv", index=False)
            file_names.append("DTB3_available.csv")
        manifest = {
            "schema": "qqq_hourly_v5_monitor_packet_v1"
            if packet_kind == "monitor"
            else "qqq_hourly_v5_input_packet_v1",
            "price_basis": PRICE_BASIS,
            "sampling_interval_minutes": sampling_interval_minutes,
            "observation_interval_minutes": sampling_interval_minutes,
            "data_role": "monitor_only_not_historical_research_splice"
            if packet_kind == "monitor"
            else "historical_research_not_point_in_time_feed",
            "as_of": now.isoformat(),
            "source_metadata": asdict(metadata),
            "source_files": sources,
            "calendar_provenance": validate_calendar(calendar).attrs["calendar_provenance"],
            "audit": audit,
            "files": {
                name: {"sha256": sha256_file(stage / name), "bytes": (stage / name).stat().st_size}
                for name in file_names
            },
        }
        if daily_attestation is not None:
            manifest["daily_source_metadata"] = daily_attestation
        if packet_kind == "monitor":
            manifest["monitor_normalization"] = {
                "completed_subbars_within_unfinished_hour_excluded": int(
                    bars.attrs.get("completed_subbars_within_unfinished_hour_excluded", 0)
                ),
                "completed_partial_bar_role": "every_completed_30min_bar_is_confirmation_eligible"
                if sampling_interval_minutes == 30
                else "paper_fill_and_mark_only_not_signal_confirmation",
            }
        if cash is not None:
            manifest["cash_observations"] = {
                "filename": "DTB3_available.csv",
                "rows": len(cash),
                "columns": list(cash.columns),
                "availability_column": "available_at",
                "rate_column": "annual_rate",
                "rate_units": "annual_decimal",
                "tie_policy": "latest_observation_date_at_same_available_at",
                "availability": "source_explicit_availability_assumption_not_verified_point_in_time",
            }
        manifest = json.loads(json.dumps(manifest))
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # rename is atomic within the same filesystem; do not replace an existing packet.
        if target.exists():
            raise FileExistsError(f"immutable input packet already exists: {target}")
        os.rename(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def prepare_monitor_packet(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Write a distinct monitor-only packet; historical readers reject it."""
    kwargs["packet_kind"] = "monitor"
    kwargs.setdefault("sampling_interval_minutes", 30)
    return prepare_input_packet(*args, **kwargs)


def read_input_packet(
    root: str | Path, *, packet_kind: str = "historical"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Verify frozen file hashes and revalidate the packet before use."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise DataContractError("packet root must be an existing nonsymlink directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise DataContractError("symlink manifest prohibited")
    manifest = json.loads(manifest_path.read_text())
    if packet_kind not in ("historical", "monitor"):
        raise DataContractError("packet_kind must be historical or monitor")
    expected_schema = (
        "qqq_hourly_v5_monitor_packet_v1"
        if packet_kind == "monitor"
        else "qqq_hourly_v5_input_packet_v1"
    )
    if manifest.get("schema") != expected_schema or manifest.get("price_basis") != PRICE_BASIS:
        raise DataContractError("unsupported packet schema/price basis")
    allowed_files = {"daily.csv", "bars.csv", "calendar.csv", "DTB3_available.csv"}
    if not {"daily.csv", "bars.csv", "calendar.csv"}.issubset(manifest.get("files", {})):
        raise DataContractError("packet lacks required input file hashes")
    if set(manifest["files"]).difference(allowed_files):
        raise DataContractError("packet manifest contains unexpected/path-escaping files")
    for name in manifest["files"]:
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != manifest["files"][name]["sha256"]
        ):
            raise DataContractError(f"input hash mismatch or unsafe file: {name}")
    daily = pd.read_csv(root / "daily.csv", dtype={"session": str})
    bars = pd.read_csv(root / "bars.csv", dtype={"session": str})
    cal = pd.read_csv(root / "calendar.csv", dtype={"session": str})
    for frame, columns in (
        (bars, ("bar_start", "bar_end")),
        (cal, ("market_open", "market_close")),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    metadata = SourceMetadata(
        **{
            **manifest["source_metadata"],
            "source_files": tuple(manifest["source_metadata"].get("source_files", ())),
        }
    )
    metadata.require_raw_start((30, 60) if packet_kind == "monitor" else (30,))
    if utc(metadata.observed_at, "source observed_at") > utc(manifest["as_of"]):
        raise DataContractError("packet source observation exceeds as_of")
    if "cash_observations" in manifest:
        cash_info = manifest["cash_observations"]
        if (
            cash_info.get("filename") != "DTB3_available.csv"
            or "DTB3_available.csv" not in manifest["files"]
        ):
            raise DataContractError("cash observations require hash-bound DTB3_available.csv")
        cash = validate_cash_observations(
            pd.read_csv(root / "DTB3_available.csv"),
            as_of=manifest["as_of"],
            first_bar_start=bars.bar_start.iloc[0],
        )
        if cash_info.get("rows") != len(cash) or cash_info.get("columns") != list(cash.columns):
            raise DataContractError("cash observations metadata does not match frozen file")
    elif "DTB3_available.csv" in manifest["files"]:
        raise DataContractError("cash file lacks cash observations metadata")
    if "daily_source_metadata" in manifest:
        daily.attrs["source_metadata"] = _daily_source_attestation(
            manifest["daily_source_metadata"], as_of=manifest["as_of"]
        )
    bars.attrs["source_metadata"] = asdict(metadata)
    cal.attrs["calendar_provenance"] = manifest.get("calendar_provenance", {})
    audit = manifest["audit"]
    sampling_interval_minutes = _sampling_interval(audit.get("sampling_interval_minutes", 60))
    if (
        "sampling_interval_minutes" in audit
        and audit.get("observation_interval_minutes") != sampling_interval_minutes
    ):
        raise DataContractError(
            "packet observation interval differs from declared sampling interval"
        )
    if "sampling_interval_minutes" not in audit and "observation_interval_minutes" in audit:
        raise DataContractError("packet observation interval lacks explicit sampling mode")
    for mode_field in ("sampling_interval_minutes", "observation_interval_minutes"):
        if mode_field in manifest and manifest[mode_field] != sampling_interval_minutes:
            raise DataContractError(
                "packet top-level interval disagrees with declared audit sampling mode"
            )
    bars.attrs["sampling_interval_minutes"] = sampling_interval_minutes
    if packet_kind == "historical":
        actual_audit = validate_research_inputs(
            daily,
            bars,
            cal,
            as_of=manifest["as_of"],
            close_rtol=audit["close_rtol"],
            close_atol=audit["close_atol"],
            bar_start_session=audit.get("bar_start_session"),
            sampling_interval_minutes=sampling_interval_minutes,
        )
    else:
        actual_audit = validate_monitor_inputs(
            daily,
            bars,
            cal,
            metadata,
            as_of=manifest["as_of"],
            close_rtol=audit["close_rtol"],
            close_atol=audit["close_atol"],
            bar_start_session=audit.get("bar_start_session"),
            sampling_interval_minutes=sampling_interval_minutes,
        )
    if "sampling_interval_minutes" not in audit:
        actual_audit = {key: value for key, value in actual_audit.items() if key in audit}
    if actual_audit != audit:
        raise DataContractError("packet audit metadata differs from recomputed validation")
    return daily, bars, cal, manifest


def read_monitor_packet(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Verify monitor-only receipt without treating it as a historical dataset."""
    return read_input_packet(root, packet_kind="monitor")


def download_alpha_vantage_raw_month(
    month: str, cache_dir: str | Path, *, api_key: str, symbol: str = "QQQ", timeout: float = 60
) -> dict[str, Any]:
    """Explicit one-month raw download; no subscriptions or implicit full sweep.

    Response and credential-free request metadata are immutable isolated cache
    files. The label starts UNVERIFIED; response availability is not sufficient
    evidence of bar-label semantics. Provider denials are saved and raised.
    """
    import requests

    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month) or month < "2000-01":
        raise DataContractError("Alpha Vantage month must be YYYY-MM at or after 2000-01")
    if not api_key or not re.fullmatch(r"[A-Za-z0-9.\-]+", symbol):
        raise DataContractError("authorized API key and valid symbol required")
    target = Path(cache_dir) / symbol / month
    if target.exists():
        raise FileExistsError(f"immutable monthly raw cache exists: {target}")
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "30min",
        "month": month,
        "adjusted": "false",
        "extended_hours": "false",
        "outputsize": "full",
        "datatype": "csv",
    }
    try:
        response = requests.get(AV_URL, params={**params, "apikey": api_key}, timeout=timeout)
    except requests.RequestException:
        # A requests exception may embed the full credential-bearing URL.
        raise DataContractError(
            "Alpha Vantage request transport failure (credential redacted)"
        ) from None
    body = response.text.replace(api_key, "[REDACTED]")
    csv_response = response.status_code == 200 and body.lstrip().startswith("timestamp,")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{month}.staging-", dir=target.parent))
    try:
        name = "response.csv" if csv_response else "provider_error.json"
        (stage / name).write_text(body)
        info = {
            "provider": "Alpha Vantage",
            "symbol": symbol,
            "request": params,
            "price_basis": PRICE_BASIS,
            "adjusted": False,
            "timestamp_label": "unverified",
            "timezone": "America/New_York",
            "observed_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "documentation_url": AV_DOCUMENTATION,
            "http_status": response.status_code,
            "response_status": "csv" if csv_response else "provider_error",
            "response_file": name,
            "sha256": sha256_file(stage / name),
        }
        if csv_response:
            frame = pd.read_csv(StringIO(body)).rename(columns={"timestamp": "date"})
            frame = frame.sort_values("date").reset_index(drop=True)
            unverified = SourceMetadata(
                "Alpha Vantage",
                symbol,
                30,
                "unverified",
                "America/New_York",
                PRICE_BASIS,
                False,
                info["observed_at"],
                "",
            )
            _provider_bars(frame, unverified)
            frame.to_parquet(stage / "raw_30min.parquet", index=False)
            info["normalized_sha256"] = sha256_file(stage / "raw_30min.parquet")
            info["rows"] = len(frame)
        (stage / "metadata.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
        if target.exists():
            raise FileExistsError(f"immutable monthly raw cache exists: {target}")
        os.rename(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    if not csv_response:
        raise DataContractError(
            f"Alpha Vantage denied/raw data unavailable for {symbol} {month}; "
            f"diagnostic saved at {target}; no paid subscription was created"
        )
    return info


def audit_alpha_only_cache(
    source_root: str | Path,
    *,
    bar_start_session: str = "2001-01-02",
    close_rtol: float = 0.001,
    close_atol: float = 0.02,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Read-only Alpha-only AV30 feasibility; no API or prepared price packet.

    The selected terminal date is the intersection of Alpha daily and 30m
    coverage, never Yahoo, native AV60, or a new quote source. Multiplying
    adjusted OHLC by daily.close/daily.adj_close is assessed in memory as a
    retrospective reconstruction CANDIDATE, not silently declared verified raw.
    No reconstructed quotes are returned or exported. Volume units, official
    versus intraday closes, internal gaps and halt semantics remain explicit.
    """
    if (
        not np.isfinite(close_rtol)
        or not np.isfinite(close_atol)
        or min(close_rtol, close_atol) < 0
    ):
        raise DataContractError("Alpha-only comparison tolerances must be finite and nonnegative")
    bar_start_session = _session_strings([bar_start_session]).iloc[0]
    root = Path(source_root)
    daily_path = root / "data/raw/alpha_vantage_daily_adjusted/QQQ.parquet"
    primary_path = root / "data/raw/alpha_vantage_30min/QQQ.parquet"
    secondary_path = root / "data/raw/alpha_vantage_15min/QQQ.parquet"
    source_daily = pd.read_parquet(daily_path)
    source30 = pd.read_parquet(primary_path)
    _required(
        source_daily,
        ("date", "close", "adj_close", "dividend_amount", "split_coefficient"),
        "Alpha daily",
    )
    _required(source30, ("date", *OHLCV), "Alpha30 cache")
    daily_dates = _session_strings(source_daily.date)
    _ordered_unique(daily_dates, "Alpha daily sessions")
    starts = pd.DatetimeIndex(pd.to_datetime(source30.date))
    _ordered_unique(starts, "Alpha30 cache timestamps")
    cutoff = min(daily_dates.iloc[-1], starts[-1].strftime("%Y-%m-%d"))
    calendar = build_xnys_calendar(daily_dates.iloc[0], cutoff)
    daily = normalize_daily_actions(source_daily, calendar)
    if bar_start_session not in set(calendar.session):
        raise DataContractError(
            "declared Alpha-only bar_start_session must be a frozen XNYS session"
        )
    daily_factor = pd.to_numeric(source_daily.close, errors="coerce") / pd.to_numeric(
        source_daily.adj_close, errors="coerce"
    )
    if not np.isfinite(daily_factor).all() or daily_factor.le(0).any():
        raise DataContractError("Alpha daily adjustment factors must be finite and positive")
    factor_by_session = pd.Series(daily_factor.to_numpy(), index=daily_dates)
    observed_at = pd.Timestamp.now(tz="UTC").isoformat()
    metadata = SourceMetadata(
        "AlphaVantage",
        "QQQ",
        30,
        "start",
        "America/New_York",
        "split_dividend_adjusted",
        True,
        observed_at,
        "legacy downloader requested adjusted=true; raw basis is NOT verified",
        (str(primary_path.resolve()), str(daily_path.resolve())),
    )
    source30 = source30.loc[
        pd.to_datetime(source30.date).dt.strftime("%Y-%m-%d").le(cutoff)
    ].reset_index(drop=True)
    provider = _provider_bars(source30, metadata)
    provider["session"] = provider.bar_start.dt.tz_convert("America/New_York").dt.strftime(
        "%Y-%m-%d"
    )
    source_start = provider.session.iloc[0]
    source_calendar = calendar.loc[calendar.session.ge(source_start)].reset_index(drop=True)
    gaps = audit_source_coverage(source30, source_calendar, metadata)
    gaps["classification"] = gaps.issue.map(
        {
            "missing_subbar": "unresolved_expected_timestamp_gap",
            "extra_or_off_grid_subbar": "outside_regular_trading_hours_or_off_grid",
        }
    ).fillna("unresolved_input_quality_issue")
    halted = gaps.session.eq("2013-08-22") & gaps.issue.eq("missing_subbar")
    gaps.loc[halted, "classification"] = "known_halt_consistent_not_established_missing_trades"
    gaps["in_required_study_scope"] = gaps.session.ge(bar_start_session)
    indexed_calendar = calendar.set_index("session")
    opens = provider.session.map(indexed_calendar.market_open)
    closes = provider.session.map(indexed_calendar.market_close)
    rth = provider.bar_start.ge(opens) & provider.bar_start.lt(closes)
    prices = provider.loc[rth].copy()
    factors = prices.session.map(factor_by_session)
    if factors.isna().any():
        raise DataContractError("Alpha30 rows lack same-provider daily adjustment-factor coverage")
    # This multiplication is diagnostic only, not a raw-source declaration.
    prices[list(OHLCV[:4])] = prices[list(OHLCV[:4])].mul(factors, axis=0)
    aggregates = prices.groupby("session", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        rows=("bar_start", "size"),
    )
    daily_ohlc = source_daily.copy()
    daily_ohlc["session"] = daily_dates
    _required(daily_ohlc, (*OHLCV, "session"), "Alpha daily OHLCV")
    daily_ohlc = daily_ohlc.set_index("session").reindex(aggregates.index)

    def price_quality(selected: pd.DataFrame) -> dict[str, Any]:
        reference = daily_ohlc.reindex(selected.index)
        matches = np.isclose(
            selected[list(OHLCV[:4])], reference[list(OHLCV[:4])], rtol=close_rtol, atol=close_atol
        )
        fields = {}
        for i, column in enumerate(OHLCV[:4]):
            relative = (selected[column] / reference[column] - 1).abs()
            fields[column] = {
                "pass_count": int(matches[:, i].sum()),
                "mismatch_count": int((~matches[:, i]).sum()),
                "median_absolute_error_percent": float(relative.median() * 100),
                "p99_absolute_error_percent": float(relative.quantile(0.99) * 100),
                "max_absolute_error_percent": float(relative.max() * 100),
            }
        return {
            "compared_sessions": len(selected),
            "all_four_ohlc_pass_count": int(matches.all(axis=1).sum()),
            "fields": fields,
            "rtol": close_rtol,
            "atol": close_atol,
        }

    required_aggregates = aggregates.loc[aggregates.index >= bar_start_session]
    if required_aggregates.empty:
        raise DataContractError("declared Alpha-only bar_start_session is outside source coverage")
    expected_last = _subbar_grid(
        calendar.loc[calendar.session.eq(cutoff)].reset_index(drop=True), 30
    )
    actual_last = provider.loc[provider.session.eq(cutoff), "bar_start"]
    terminal_complete = set(expected_last.bar_start) == set(actual_last)
    source_rows = [
        {
            "role": "primary_adjusted_30min",
            "provider": "AlphaVantage",
            "path": str(primary_path.resolve()),
            "sha256": sha256_file(primary_path),
            "interval_minutes": 30,
            "adjusted": True,
            "price_basis": "split_dividend_adjusted",
            "source_end_session": starts[-1].strftime("%Y-%m-%d"),
        },
        {
            "role": "daily_raw_close_actions_and_adjustment_factor",
            "provider": "AlphaVantage",
            "path": str(daily_path.resolve()),
            "sha256": sha256_file(daily_path),
            "source_end_session": daily_dates.iloc[-1],
        },
    ]
    secondary = {
        "available": secondary_path.is_file(),
        "role": "alignment_QA_only_not_automatic_gap_repair",
    }
    if secondary_path.is_file():
        fifteen = pd.read_parquet(secondary_path)
        fifteen_metadata = SourceMetadata(
            "AlphaVantage",
            "QQQ",
            15,
            "start",
            "America/New_York",
            "split_dividend_adjusted",
            True,
            observed_at,
            "legacy adjusted=true; alignment QA only",
            (str(secondary_path.resolve()),),
        )
        fif = _provider_bars(fifteen, fifteen_metadata)
        fif["bucket"] = fif.bar_start.dt.floor("30min")
        grouped = fif.groupby("bucket").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            count=("bar_start", "size"),
        )
        pairs = grouped.loc[grouped["count"].eq(2)].join(
            provider.set_index("bar_start")[list(OHLCV[:4])], how="inner", rsuffix="_30"
        )
        values = pairs[list(OHLCV[:4])].to_numpy()
        reference = pairs[[f"{name}_30" for name in OHLCV[:4]]].to_numpy()
        secondary.update(
            {
                "source_end_session": pd.Timestamp(fif.bar_start.iloc[-1])
                .tz_convert("America/New_York")
                .strftime("%Y-%m-%d"),
                "compared_30min_windows": len(pairs),
                "all_four_ohlc_match_count": int(
                    np.isclose(values, reference, rtol=1e-6, atol=1e-4).all(axis=1).sum()
                ),
                "alignment_hypothesis": "two_start_labelled_15min_subbars_per_same_label_30min_bar",
                "rtol": 1e-6,
                "atol": 1e-4,
            }
        )
        source_rows.append(
            {
                "role": "secondary_adjusted_15min_alignment_QA",
                "provider": "AlphaVantage",
                "path": str(secondary_path.resolve()),
                "sha256": sha256_file(secondary_path),
                "adjusted": True,
                "price_basis": "split_dividend_adjusted",
            }
        )
    halt_start = pd.Timestamp("2013-08-22T12:23:00", tz="America/New_York").tz_convert("UTC")
    halt_end = pd.Timestamp("2013-08-22T15:25:00", tz="America/New_York").tz_convert("UTC")
    halt_rows = provider.loc[
        provider.bar_start.ge(halt_start)
        & (provider.bar_start + pd.Timedelta(minutes=30)).le(halt_end)
    ]
    flat = halt_rows[list(OHLCV[:4])].nunique(axis=1).eq(1)
    first_split = source_daily.loc[source_daily.split_coefficient.ne(1), "date"]
    split_day = pd.Timestamp(first_split.iloc[0]).strftime("%Y-%m-%d") if len(first_split) else None
    volume_quality = {
        "raw_volume_basis_verified": False,
        "warning": "price factor does not restore volume; daily and intraday volume coverage differ",
    }
    if split_day:
        before = aggregates.index < split_day
        split_ratio = float(
            source_daily.loc[
                source_daily.date.eq(pd.Timestamp(split_day)), "split_coefficient"
            ].iloc[0]
        )
        volume_quality.update(
            {
                "first_split_session": split_day,
                "split_ratio": split_ratio,
                "pre_split_intraday_to_daily_volume_median": float(
                    (aggregates.loc[before, "volume"] / daily_ohlc.loc[before, "volume"]).median()
                ),
                "pre_split_reverse_split_to_daily_volume_median": float(
                    (
                        aggregates.loc[before, "volume"]
                        / split_ratio
                        / daily_ohlc.loc[before, "volume"]
                    ).median()
                ),
                "post_split_intraday_to_daily_volume_median": float(
                    (aggregates.loc[~before, "volume"] / daily_ohlc.loc[~before, "volume"]).median()
                ),
            }
        )
    source_dates_after30 = daily_dates.gt(starts[-1].strftime("%Y-%m-%d"))
    actions_after30 = source_dates_after30 & (
        source_daily.dividend_amount.ne(0) | source_daily.split_coefficient.ne(1)
    )
    required_gaps = gaps.loc[gaps.in_required_study_scope & gaps.issue.eq("missing_subbar")]
    report = {
        "schema": "qqq_alpha_only_cached_feasibility_v1",
        "provider_policy": "AlphaVantage_only",
        "provider_policy_scope": "security_prices_only_cash_benchmark_separate",
        "price_source_selection": "existing_30min_direct_completed_half_hours_no_hour_aggregation",
        "sampling_interval_minutes": 30,
        "observation_interval_minutes": 30,
        "normal_session_observations": 13,
        "13_00_early_close_observations": 7,
        "last_15_30_to_16_00_observation_eligible": True,
        "cutoff_session": cutoff,
        "daily_warmup_start_session": daily_dates.iloc[0],
        "required_bar_start_session": bar_start_session,
        "source_30min_start_session": source_start,
        "terminal_session_complete": terminal_complete,
        "historical_ready": False,
        "performance_generated": False,
        "prepared_raw_packet_generated": False,
        "new_provider_calls": 0,
        "source_files": source_rows,
        "daily_source_metadata": {
            "provider": "AlphaVantage",
            "symbol": "QQQ",
            "price_basis": PRICE_BASIS,
            "price_field": "close",
            "verification": "same-provider TIME_SERIES_DAILY_ADJUSTED raw close field; native adjusted close is used only to audit unit-restoration candidate",
            "source_path": str(daily_path.resolve()),
            "sha256": sha256_file(daily_path),
            "observed_at": observed_at,
            "availability": "retrospective_cache_not_point_in_time_evidence",
        },
        "daily": {
            "complete_calendar_sessions": len(daily),
            "uses_raw_close_not_adj_close": True,
            "source_rows_after_cutoff_excluded_from_analysis": int(daily_dates.gt(cutoff).sum()),
        },
        "calendar_provenance": calendar.attrs["calendar_provenance"],
        "reconstruction_candidate": {
            "status": "retrospective_reconstruction_candidate_not_verified_raw_as_traded",
            "quote_relationship_status": "as_not_yet_reconciled_quote_relationship",
            "close_mismatch_interpretation": "a frozen tolerance failure is not by itself corrupt quotes or a bad adjustment factor; official auction versus last intraday trade, endpoint and vintage semantics require reconciliation",
            "further_verification": "original cached Alpha data may be reviewed further without implying a paid redownload is necessary",
            "ohlc_formula": "adjusted_intraday_OHLC * (same_session_daily_close / daily_adj_close)",
            "action_timing": "ex_post_unit_normalization_not_intraday_indicator_input_or_point_in_time_evidence",
            "all_daily_factors_finite_positive": True,
            "corporate_actions_after_30min_source_end_through_daily_source_end": int(
                actions_after30.sum()
            ),
            "all_source_session_quality": price_quality(aggregates),
            "required_study_session_quality": price_quality(required_aggregates),
            "volume_quality": volume_quality,
            "warning": "daily official/auction prices and intraday trades can differ; mismatch is not automatically an adjustment-factor error",
        },
        "rth_selection": {
            "retained_30min_rows": len(prices),
            "outside_rth_rows_reported_and_excluded_for_diagnostic_only": int((~rth).sum()),
            "source_caches_modified": False,
        },
        "missing_expected_timestamps": {
            "all_source_scope_count": int(gaps.issue.eq("missing_subbar").sum()),
            "required_study_scope_count": len(required_gaps),
            "required_by_session": required_gaps.groupby("session").size().to_dict(),
            "known_halt_consistent_count": int(
                required_gaps.classification.eq(
                    "known_halt_consistent_not_established_missing_trades"
                ).sum()
            ),
            "unresolved_nonhalt_count": int(
                required_gaps.classification.ne(
                    "known_halt_consistent_not_established_missing_trades"
                ).sum()
            ),
        },
        "known_halt_quality": {
            "session": "2013-08-22",
            "regulatory_halt_complete_at": halt_start.isoformat(),
            "trading_resumed_at": halt_end.isoformat(),
            "source_url": "https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2013-77",
            "fully_halted_vendor_rows": len(halt_rows),
            "flat_ohlc_vendor_rows": int(flat.sum()),
            "flat_ohlc_rows_with_nonzero_volume": int((flat & halt_rows.volume.gt(0)).sum()),
            "classification": "not_verified_executable_quotes; halt_aware_confirmation_and_fill_policy_required",
            "price_or_fill_fabrication_performed": False,
        },
        "secondary_15min_alignment": secondary,
        "blocking_conditions": [
            "legacy adjusted=true cache is not a verified native raw source",
            "raw volume basis remains unverified",
        ],
    }
    if report["reconstruction_candidate"]["required_study_session_quality"]["fields"]["close"][
        "mismatch_count"
    ]:
        report["blocking_conditions"].append(
            "retrospective reconstruction does not satisfy frozen daily-close matching on every session"
        )
    if report["missing_expected_timestamps"]["unresolved_nonhalt_count"]:
        report["blocking_conditions"].append(
            "unresolved opening-subbar and whole-session gaps cannot be silently skipped"
        )
    if len(halt_rows) or halted.any():
        report["blocking_conditions"].append(
            "known trading halt needs explicit nontradable-bar and fill-time semantics"
        )
    if not terminal_complete:
        report["blocking_conditions"].append("common terminal source session is incomplete")
    cash_path = root / "data/raw/research_rates/20261002_research_v1/DTB3_available.csv"
    report["cash_benchmark"] = {
        "provider": "FRED",
        "series_id": "DTB3",
        "role": "non_price_cash_yield_benchmark_not_a_security_quote_source",
        "path": str(cash_path.resolve()),
        "available": cash_path.is_file(),
    }
    if cash_path.is_file():
        original_cash = pd.read_csv(cash_path)
        available = pd.to_datetime(original_cash.available_at, utc=True)
        cutoff_close = calendar.market_close.iloc[-1]
        eligible = original_cash.loc[available.le(cutoff_close)].reset_index(drop=True)
        validated = validate_cash_observations(
            eligible, as_of=cutoff_close, first_bar_start=calendar.market_open.iloc[0]
        )
        report["cash_benchmark"].update(
            {
                "sha256": sha256_file(cash_path),
                "schema_valid": True,
                "source_observations": len(original_cash),
                "eligible_observations_at_cutoff": len(validated),
                "future_available_observations_excluded": int(available.gt(cutoff_close).sum()),
                "last_eligible_available_at": validated.available_at.iloc[-1].isoformat(),
                "availability_tie_policy": "latest_observation_date_at_same_available_at",
                "availability": "explicit_source_assumption_latest_vintage_not_point_in_time",
            }
        )
    return report, gaps, calendar
