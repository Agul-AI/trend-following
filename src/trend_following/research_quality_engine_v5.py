"""Opt-in, observed-only quality kernel; never repairs prices or opens files.

The strict-tradable profile rejects unknown missing-source blackouts, while
allowing explicitly documented market halts. The alternate profile is an
EXPLICIT research assumption, not a data repair: hold holdings/cash, cancel
pending orders and reset confirmations at blackout start, then require fresh
observed confirmation. No OHLC row is synthesized for an absent interval.

Provider daily raw closes are available at a conservative ASSUMED 17:00 New
York clock, not asserted to be Nasdaq Official Closing Prices. Those quotes
mark each calendar session and finalize DAILY indicator state for the next
session only. Intraday decisions/fills retain observed raw RTH quotes. Terminal
liquidation requires a safe observed RTH close on the final session; its cash
then accrues to the same 17:00 valuation endpoint as both benchmarks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trend_following import research_daily_hourly_v5 as core
from trend_following.research_rates import rate_accrual

Candidate = core.Candidate
IndicatorRule = core.IndicatorRule
SimulationResult = core.SimulationResult
generate_candidates = core.generate_candidates
CONFIRMATION_HOURS = core.CONFIRMATION_HOURS
COVERAGE_COLUMNS = ("session", "bar_start", "bar_end", "status", "signal_eligible", "fill_eligible")
STATUS_VALUES = {"observed", "unknown_missing", "known_halt_overlap"}
PROFILE_VALUES = {"strict_tradable_v1", "observed_only_blackout_v1"}
MAX_POST_GAP_EVENTS = 2_500_000
POST_GAP_EVENT_COLUMNS = (
    "candidate_id",
    "side",
    "decision_at",
    "gap_start",
    "gap_end",
    "window_end",
)


@dataclass(frozen=True)
class QualityPolicy:
    profile: str = "strict_tradable_v1"
    valuation_hour_ny: int = 17
    post_gap_diagnostics: bool = False
    post_gap_event_limit: int = 10_000

    def __post_init__(self):
        if (
            self.profile not in PROFILE_VALUES
            or self.valuation_hour_ny != 17
            or isinstance(self.valuation_hour_ny, bool)
        ):
            raise ValueError(
                "Declare strict_tradable_v1 or observed_only_blackout_v1 with the frozen 17:00 NY valuation clock"
            )

        if not isinstance(self.post_gap_diagnostics, bool):
            raise ValueError("post_gap_diagnostics must be an explicit boolean")
        if self.post_gap_diagnostics and self.profile != "observed_only_blackout_v1":
            raise ValueError(
                "Post-gap diagnostics require the explicitly authorized observed-only profile"
            )
        if (
            not core._positive_integer(self.post_gap_event_limit)
            or self.post_gap_event_limit > MAX_POST_GAP_EVENTS
        ):
            raise ValueError("post_gap_event_limit must be an integer in 1..2500000")
        object.__setattr__(self, "post_gap_event_limit", int(self.post_gap_event_limit))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "valuation_hour_ny": self.valuation_hour_ny,
            "blackout_behavior": "hold_cancel_pending_reset_both_streaks",
            "unknown_source_policy": "reject"
            if self.profile == "strict_tradable_v1"
            else "hold_cancel_pending_reset_both_streaks",
            "known_halt_policy": "hold_cancel_pending_reset_both_streaks",
            "missing_open_dividends": "credit_at_calendar_open_DRIP_first_safe_observed_open",
            "daily_close_kind": "provider_daily_raw_close_not_NOCP",
            "daily_availability": "17:00_America_New_York_conservative_assumption",
            "terminal": "last_safe_observed_RTH_close_then_cash_accrual_to_17",
            "post_gap_diagnostics": self.post_gap_diagnostics,
            "post_gap_event_limit": self.post_gap_event_limit,
            "post_gap_window": "gap_end_exclusive_through_strictly_next_NY_session_date_market_close_inclusive",
        }


DEFAULT_POLICY = QualityPolicy()


@dataclass(frozen=True)
class QualitySupportCache:
    daily: pd.DataFrame
    bars: pd.DataFrame
    calendar: pd.DataFrame
    coverage: pd.DataFrame
    blackouts: pd.DataFrame
    policy: QualityPolicy
    rules: tuple[IndicatorRule, ...]
    margins: np.ndarray
    provisional_total_return_index: np.ndarray
    finalized_total_return_index: pd.Series
    finalized_daily_margins: pd.DataFrame
    prior_session_counts: np.ndarray
    signal_eligible: np.ndarray
    fill_eligible: np.ndarray
    bar_rows: tuple
    input_fingerprint: str = ""
    feature_fingerprint: str = ""


def _hash_frame(digest, label, frame, columns):
    columns = tuple(columns)
    digest.update(json.dumps([label, len(frame), columns], separators=(",", ":")).encode())
    times = {
        "bar_start",
        "bar_end",
        "market_open",
        "market_close",
        "daily_price_available_at",
        "start",
        "end",
    }
    texts = {"session", "status", "kind", "reason", "source_url"}
    booleans = {
        "signal_eligible",
        "fill_eligible",
        "is_completed",
        "is_full_hour",
        "is_early_close",
    }
    for column in columns:
        values = frame[column]
        if column in times:
            index = pd.DatetimeIndex(values)
            if len(index):
                index = index.tz_convert("UTC").as_unit("ns")
            digest.update(np.asarray(index.asi8, dtype="<i8").tobytes())
        elif column in texts:
            for value in values:
                encoded = str(value).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
        elif column in booleans:
            digest.update(np.asarray(values, dtype=np.uint8).tobytes())
        else:
            digest.update(np.asarray(values, dtype="<f8").tobytes())


def _input_fingerprint(daily, bars, calendar, coverage, blackouts, policy):
    digest = hashlib.sha256()
    digest.update(json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":")).encode())
    dcols = ["session", "close", "dividend_amount", "split_coefficient"]
    dcols += [
        x
        for x in ("open", "high", "low", "adj_close", "volume", "daily_price_available_at")
        if x in daily
    ]
    bcols = [
        "session",
        "bar_start",
        "bar_end",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "duration_minutes",
    ]
    bcols += [
        x
        for x in ("is_completed", "is_full_hour", "status", "signal_eligible", "fill_eligible")
        if x in bars
    ]
    ccols = ["session", "market_open", "market_close"] + (
        ["is_early_close"] if "is_early_close" in calendar else []
    )
    blcols = ["start", "end", "kind", "reason"] + (
        ["source_url"] if "source_url" in blackouts else []
    )
    for label, frame, columns in (
        ("daily", daily, dcols),
        ("observed", bars, bcols),
        ("calendar", calendar, ccols),
        ("coverage", coverage, COVERAGE_COLUMNS),
        ("blackouts", blackouts, blcols),
    ):
        _hash_frame(digest, label, frame, columns)
    return digest.hexdigest()


def _feature_fingerprint(cache):
    digest = hashlib.sha256()
    digest.update(json.dumps([r.to_dict() for r in cache.rules], sort_keys=True).encode())
    for label in (
        "margins",
        "provisional_total_return_index",
        "prior_session_counts",
        "signal_eligible",
        "fill_eligible",
    ):
        values = np.asarray(getattr(cache, label))
        digest.update(json.dumps([label, values.shape, str(values.dtype)]).encode())
        digest.update(np.ascontiguousarray(values).tobytes())
    final = pd.DataFrame(
        {
            "session": cache.finalized_total_return_index.index,
            "value": cache.finalized_total_return_index.to_numpy(),
        }
    )
    _hash_frame(digest, "finalized_features", final, ("session", "value"))
    _hash_frame(
        digest,
        "daily_margins",
        cache.finalized_daily_margins,
        cache.finalized_daily_margins.columns,
    )
    rows = pd.DataFrame(
        [(r.session, r.bar_start, r.bar_end, r.open, r.close) for r in cache.bar_rows],
        columns=["session", "bar_start", "bar_end", "open", "close"],
    )
    _hash_frame(digest, "execution_rows", rows, rows.columns)
    return digest.hexdigest()


def _validate_runtime_cache(cache, daily, bars, calendar, coverage, blackouts, policy):
    """Recheck profile and normalized source/derived identities once per public call.

    This detects accidental mutation/relabeling; it is not an OS sandbox or a
    cryptographic authorization boundary against a caller fabricating hashes.
    Cached frames remain immutable-by-contract even though pandas arrays are
    mutable. Missing kwargs use the cache's already-bound coverage metadata.
    """
    if not isinstance(cache, QualitySupportCache) or cache.policy != policy:
        raise ValueError("Cached quality policy differs from the explicitly declared policy")
    if policy.profile == "strict_tradable_v1" and cache.coverage.status.eq("unknown_missing").any():
        raise ValueError(
            "Strict-tradable cache rejects unknown missing-source intervals regardless of its label"
        )
    actual = _input_fingerprint(
        cache.daily, cache.bars, cache.calendar, cache.coverage, cache.blackouts, cache.policy
    )
    if not cache.input_fingerprint or actual != cache.input_fingerprint:
        raise ValueError("Cached source/coverage identity changed after validation")
    if not cache.feature_fingerprint or _feature_fingerprint(cache) != cache.feature_fingerprint:
        raise ValueError(
            "Cached derived features/eligibility/execution rows changed after validation"
        )
    frames = _prepare(
        daily,
        bars,
        calendar,
        cache.coverage if coverage is None else coverage,
        cache.blackouts if blackouts is None else blackouts,
        policy,
    )
    if _input_fingerprint(*frames, policy) != cache.input_fingerprint:
        raise ValueError("Caller inputs differ from the cached normalized source identity")
    return cache


def _aware(values, label):
    return core._utc(values, label)


def _bool_column(values, label):
    if not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{label} must contain actual booleans")
    return values.astype(bool)


def _valuation_clock(session, policy):
    return (
        (pd.Timestamp(session) + pd.Timedelta(hours=policy.valuation_hour_ny))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )


def _grid(calendar):
    rows = []
    for row in calendar.itertuples(index=False):
        elapsed = (row.market_close - row.market_open).total_seconds() / 60
        if (
            elapsed <= 0
            or elapsed % 30 != 0
            or row.market_close >= _valuation_clock(row.session, DEFAULT_POLICY)
        ):
            raise ValueError(
                "Calendar must contain positive complete 30-minute RTH intervals before 17:00"
            )
        for start in pd.date_range(
            row.market_open, row.market_close, freq="30min", inclusive="left"
        ):
            rows.append(
                dict(session=row.session, bar_start=start, bar_end=start + pd.Timedelta(minutes=30))
            )
    return pd.DataFrame(rows)


def _normalize_blackouts(blackouts, coverage):
    if blackouts is None:
        if coverage.status.eq("known_halt_overlap").any():
            raise ValueError(
                "Known halt classification requires exact documented blackout start/end"
            )
        rows = []
        for row in coverage.loc[coverage.status.eq("unknown_missing")].itertuples():
            if rows and rows[-1]["end"] == row.bar_start:
                rows[-1]["end"] = row.bar_end
            else:
                rows.append(
                    dict(
                        start=row.bar_start,
                        end=row.bar_end,
                        kind="unknown_missing",
                        reason="declared_source_blackout",
                    )
                )
        return pd.DataFrame(rows, columns=["start", "end", "kind", "reason"])
    out = blackouts.copy(deep=True).reset_index(drop=True)
    if not {"start", "end", "kind", "reason"}.issubset(out):
        raise ValueError("Blackout metadata requires start, end, kind and reason")
    if out.empty:
        out["start"], out["end"] = pd.to_datetime([], utc=True), pd.to_datetime([], utc=True)
        return out
    out["start"], out["end"] = _aware(out.start, "blackout start"), _aware(out.end, "blackout end")
    out = out.sort_values(["start", "end"]).reset_index(drop=True)
    if (out.end <= out.start).any() or out[["start", "end"]].duplicated().any():
        raise ValueError("Blackouts require unique positive intervals")
    if (
        not out.kind.isin(["unknown_missing", "known_halt"]).all()
        or out.reason.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Blackouts require explicit kind and reason")
    return out


def _prepare(daily, bars, calendar, coverage, blackouts, policy):
    if not isinstance(policy, QualityPolicy):
        raise ValueError("quality_policy must be an explicit QualityPolicy")
    daily, bars, calendar = daily.copy(deep=True), bars.copy(deep=True), calendar.copy(deep=True)
    if (
        not {"session", "close", "dividend_amount", "split_coefficient"}.issubset(daily)
        or daily.empty
    ):
        raise ValueError(
            "Complete raw daily closes/actions required independently of intraday availability"
        )
    daily["session"] = core._sessions(daily.session)
    daily = daily.sort_values("session").reset_index(drop=True)
    if daily.session.duplicated().any():
        raise ValueError("Duplicate daily sessions")
    numbers = daily[["close", "dividend_amount", "split_coefficient"]].to_numpy(float)
    if (
        not np.isfinite(numbers).all()
        or (numbers[:, 0] <= 0).any()
        or (numbers[:, 1] < 0).any()
        or (numbers[:, 2] <= 0).any()
    ):
        raise ValueError("Invalid raw daily prices/actions")
    if not {"session", "market_open", "market_close"}.issubset(calendar) or calendar.empty:
        raise ValueError("Explicit calendar required even for wholly missing intraday sessions")
    calendar["session"] = core._sessions(calendar.session)
    calendar["market_open"], calendar["market_close"] = (
        _aware(calendar.market_open, "market_open"),
        _aware(calendar.market_close, "market_close"),
    )
    calendar = calendar.sort_values("session").reset_index(drop=True)
    if calendar.session.duplicated().any() or not calendar.session.isin(daily.session).all():
        raise ValueError("Unique calendar sessions require complete daily action/valuation records")
    if "daily_price_available_at" in daily:
        available = _aware(daily.daily_price_available_at, "daily_price_available_at")
        expected = pd.DatetimeIndex([_valuation_clock(x, policy) for x in daily.session])
        if not available.equals(expected):
            raise ValueError(
                "Daily quotes must use the frozen conservative 17:00 NY availability assumption"
            )
    required = {
        "session",
        "bar_start",
        "bar_end",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "duration_minutes",
    }
    if not required.issubset(bars):
        raise ValueError(
            "Observed intraday rows require actual raw 30-minute OHLCV, not placeholder marks"
        )
    bars["session"] = core._sessions(bars.session)
    bars["bar_start"], bars["bar_end"] = (
        _aware(bars.bar_start, "bar_start"),
        _aware(bars.bar_end, "bar_end"),
    )
    bars = bars.sort_values("bar_start").reset_index(drop=True)
    if bars.bar_start.duplicated().any() or not bars.session.isin(calendar.session).all():
        raise ValueError("Observed rows require unique canonical starts and calendar sessions")
    values = bars[["open", "high", "low", "close", "volume", "duration_minutes"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Invalid observed raw OHLCV")
    if (bars.high < bars[["open", "close", "low"]].max(axis=1)).any() or (
        bars.low > bars[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("Invalid observed OHLC envelope")
    if not (
        (bars.bar_end - bars.bar_start).dt.total_seconds().eq(1800).all()
        and bars.duration_minutes.eq(30).all()
    ):
        raise ValueError("Observed sample rows must be actual complete 30-minute intervals")
    if "is_completed" in bars and (not _bool_column(bars.is_completed, "is_completed").all()):
        raise ValueError("Unfinished observations cannot generate signals or fills")
    if "is_full_hour" in bars and _bool_column(bars.is_full_hour, "is_full_hour").any():
        raise ValueError("No 30-minute observation is a full hour")
    if coverage is None:
        if policy.profile != "strict_tradable_v1" or bars.empty:
            raise ValueError(
                "An opt-in observed-only profile requires explicit full-slot coverage metadata"
            )
        start_session = bars.session.iloc[0]
        coverage = _grid(calendar.loc[calendar.session.ge(start_session)]).assign(
            status="observed", signal_eligible=True, fill_eligible=True
        )
    else:
        coverage = coverage.copy(deep=True).reset_index(drop=True)
        if not set(COVERAGE_COLUMNS).issubset(coverage) or coverage.empty:
            raise ValueError(
                "Explicit quality coverage must classify every scheduled slot without OHLC placeholders"
            )
        coverage["session"] = core._sessions(coverage.session)
        coverage["bar_start"], coverage["bar_end"] = (
            _aware(coverage.bar_start, "coverage start"),
            _aware(coverage.bar_end, "coverage end"),
        )
        coverage = coverage.sort_values("bar_start").reset_index(drop=True)
    expected = _grid(calendar.loc[calendar.session.ge(coverage.session.iloc[0])])
    if not coverage[["session", "bar_start", "bar_end"]].equals(
        expected[["session", "bar_start", "bar_end"]]
    ):
        raise ValueError(
            "Coverage metadata must exactly classify the full canonical grid, including missing sessions"
        )
    coverage["signal_eligible"] = _bool_column(coverage.signal_eligible, "signal_eligible")
    coverage["fill_eligible"] = _bool_column(coverage.fill_eligible, "fill_eligible")
    if not coverage.status.isin(STATUS_VALUES).all():
        raise ValueError("Unknown coverage status")
    indexed = coverage.set_index("bar_start")
    if not bars.bar_start.isin(indexed.index).all():
        raise ValueError("Observed rows cannot lie outside the frozen coverage grid")
    shapes = indexed.loc[bars.bar_start]
    if not np.array_equal(shapes.bar_end.to_numpy(), bars.bar_end.to_numpy()) or not np.array_equal(
        shapes.session.to_numpy(), bars.session.to_numpy()
    ):
        raise ValueError("Observed row labels/ends disagree with coverage")
    present = coverage.bar_start.isin(bars.bar_start)
    if (~present & coverage.status.eq("observed")).any() or (
        present & coverage.status.eq("unknown_missing")
    ).any():
        raise ValueError("Missing-slot classification must match actual source-row presence")
    if ((coverage.signal_eligible | coverage.fill_eligible) & ~present).any():
        raise ValueError("Absent rows cannot be eligible for signal or execution")
    blackouts = _normalize_blackouts(blackouts, coverage)
    known_overlap = np.zeros(len(coverage), dtype=bool)
    unknown_overlap = np.zeros(len(coverage), dtype=bool)
    opens_blocked = np.zeros(len(coverage), dtype=bool)
    for row in blackouts.itertuples(index=False):
        overlap = coverage.bar_start.lt(row.end) & coverage.bar_end.gt(row.start)
        within = coverage.bar_start.ge(row.start) & coverage.bar_start.lt(row.end)
        if not overlap.any():
            raise ValueError(
                "Blackout metadata cannot classify unobserved overnight/closed-market intervals as a price gap"
            )
        if row.kind == "unknown_missing":
            affected = coverage.loc[overlap]
            if (
                row.start != affected.bar_start.iloc[0]
                or row.end != affected.bar_end.iloc[-1]
                or not np.array_equal(
                    affected.bar_start.iloc[1:].to_numpy(), affected.bar_end.iloc[:-1].to_numpy()
                )
            ):
                raise ValueError(
                    "Unknown source blackout bounds must exactly cover contiguous missing scheduled slots"
                )
        if row.kind == "known_halt":
            known_overlap |= overlap.to_numpy()
        else:
            unknown_overlap |= overlap.to_numpy()
        opens_blocked |= within.to_numpy()
    if not np.array_equal(known_overlap, coverage.status.eq("known_halt_overlap").to_numpy()):
        raise ValueError("Known-halt coverage must match exact documented halt intervals")
    if not np.array_equal(unknown_overlap, coverage.status.eq("unknown_missing").to_numpy()):
        raise ValueError("Unknown missing coverage must match declared blackout intervals")
    if (coverage.signal_eligible.to_numpy() & (known_overlap | unknown_overlap)).any() or (
        coverage.fill_eligible.to_numpy() & opens_blocked
    ).any():
        raise ValueError("Blackout-overlapping signals or opens inside a blackout are prohibited")
    observed = coverage.status.eq("observed")
    if (observed & (~coverage.signal_eligible | ~coverage.fill_eligible)).any():
        raise ValueError(
            "Observed normal slots must be eligible; exclusions require explicit blackout classification"
        )
    if policy.profile == "strict_tradable_v1" and coverage.status.eq("unknown_missing").any():
        raise ValueError(
            "Strict-tradable quality profile rejects unknown missing-source intervals; documented halts alone are permitted"
        )
    return daily, bars, calendar, coverage, blackouts


def build_quality_support_cache(
    daily: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    coverage: pd.DataFrame | None = None,
    blackouts: pd.DataFrame | None = None,
    quality_policy: QualityPolicy = DEFAULT_POLICY,
    rules: tuple[IndicatorRule, ...] | None = None,
) -> QualitySupportCache:
    daily, bars, calendar, coverage, blackouts = _prepare(
        daily, bars, calendar, coverage, blackouts, quality_policy
    )
    rules = core.indicator_rules() if rules is None else tuple(rules)
    if (
        not rules
        or len(set(rules)) != len(rules)
        or not all(isinstance(r, IndicatorRule) for r in rules)
    ):
        raise ValueError("Unique IndicatorRule collection required")
    eligibility = coverage.set_index("bar_start").loc[bars.bar_start]
    signal = eligibility.signal_eligible.to_numpy(bool)
    fill = eligibility.fill_eligible.to_numpy(bool)
    margins = np.full((len(bars), len(rules)), np.nan)
    preview = np.full(len(bars), np.nan)
    counts = np.zeros(len(bars), dtype=int)
    state = core._DailyState(rules)
    prior_price, prior_index = None, 100.0
    finalized, matured = [], []
    groups = bars.groupby("session", sort=False).indices
    for row in daily.itertuples(index=False):
        for i in groups.get(row.session, []):
            counts[i] = state.count
            if prior_price is not None:
                value = (
                    prior_index
                    * row.split_coefficient
                    * (bars.close.iloc[i] + row.dividend_amount)
                    / prior_price
                )
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("Nonfinite/exhausted provisional total-return feature")
                preview[i] = value
                if signal[i]:
                    margins[i] = state.preview(value)[0]
        value = (
            100.0
            if prior_price is None
            else prior_index
            * row.split_coefficient
            * (row.close + row.dividend_amount)
            / prior_price
        )
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Nonfinite/exhausted finalized total-return feature")
        matured.append(state.preview(value)[0])
        state.finalize(value)  # Available at 17:00, and used only on later sessions.
        finalized.append(value)
        prior_price, prior_index = row.close, value
    index = pd.Index(daily.session, name="session")
    cache = QualitySupportCache(
        daily,
        bars,
        calendar,
        coverage,
        blackouts,
        quality_policy,
        rules,
        margins,
        preview,
        pd.Series(finalized, index=index, name="total_return_index"),
        pd.DataFrame(matured, index=index, columns=[r.rule_id for r in rules]),
        counts,
        signal,
        fill,
        tuple(bars.itertuples(index=False)),
    )
    object.__setattr__(
        cache,
        "input_fingerprint",
        _input_fingerprint(daily, bars, calendar, coverage, blackouts, quality_policy),
    )
    object.__setattr__(cache, "feature_fingerprint", _feature_fingerprint(cache))
    return cache


def _segment_events(cache, start, end):
    lower, upper = pd.Timestamp(start).strftime("%Y-%m-%d"), pd.Timestamp(end).strftime("%Y-%m-%d")
    calendar = cache.calendar.loc[cache.calendar.session.between(lower, upper)]
    if lower > upper or calendar.empty or not calendar.session.isin(cache.coverage.session).all():
        raise ValueError("Each scored session must have explicit coverage and a daily valuation")
    first, last = (
        calendar.market_open.iloc[0],
        _valuation_clock(calendar.session.iloc[-1], cache.policy),
    )
    positions = np.flatnonzero(cache.bars.session.isin(calendar.session).to_numpy())
    final_session = calendar.session.iloc[-1]
    terminal = np.flatnonzero(
        (
            cache.bars.session.eq(final_session)
            & cache.bars.bar_end.eq(calendar.market_close.iloc[-1])
        ).to_numpy()
    )
    if (
        len(terminal) != 1
        or not cache.signal_eligible[terminal[0]]
        or not cache.fill_eligible[terminal[0]]
    ):
        raise ValueError(
            "Terminal liquidation requires a safe observed final-session RTH closing quote; daily valuation is never a substitute fill"
        )
    events = []
    for i in positions:
        row = cache.bar_rows[i]
        events.extend([(row.bar_start, 4, "bar_open", i), (row.bar_end, 0, "bar_close", i)])
    for row in calendar.itertuples(index=False):
        events.extend(
            [
                (row.market_open, 3, "actions", row.session),
                (_valuation_clock(row.session, cache.policy), 5, "valuation", row.session),
            ]
        )
    for i, row in cache.blackouts.iterrows():
        if row.end > first and row.start <= last:
            events.extend(
                [
                    (max(first, row.start), 1, "blackout_start", i),
                    (min(last, row.end), 2, "blackout_end", i),
                ]
            )
    events.sort(key=lambda value: (value[0], value[1], str(value[3])))
    clock = pd.DatetimeIndex(sorted({x[0] for x in events}))
    return calendar, events, clock, int(terminal[0])


def _post_gap_context(cache, scored_calendar):
    if not cache.policy.post_gap_diagnostics:
        return None
    first = scored_calendar.market_open.iloc[0]
    scored_end = scored_calendar.market_close.iloc[-1]
    staged_end = cache.calendar.market_close.iloc[-1]
    full_calendar = cache.calendar.sort_values("session")
    windows = []
    metadata = []
    for row in cache.blackouts.loc[cache.blackouts.kind.eq("unknown_missing")].itertuples(
        index=False
    ):
        day = row.end.tz_convert("America/New_York").strftime("%Y-%m-%d")
        later = full_calendar.loc[full_calendar.session.gt(day)]
        time_source = "input_calendar"
        if later.empty:
            # A last-stage window may extend beyond supplied quotes. Only a
            # pinned library's next SESSION TIME is generated, never a price.
            from trend_following.research_hourly_data_v5 import build_xnys_calendar

            begin = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            stop = (pd.Timestamp(day) + pd.Timedelta(days=31)).strftime("%Y-%m-%d")
            later = build_xnys_calendar(begin, stop)
            time_source = "pinned_XNYS_time_only_calendar_helper"
        window_end = later.market_close.iloc[0]
        windows.append((row.start, row.end, window_end))
        overlaps = row.end < scored_end and window_end >= first
        metadata.append(
            {
                "gap_start": row.start.isoformat(),
                "gap_end": row.end.isoformat(),
                "window_end": window_end.isoformat(),
                "next_session": str(later.session.iloc[0]),
                "next_session_time_source": time_source,
                "overlaps_scored_segment": bool(overlaps),
                "right_censored": bool(window_end > scored_end),
                "right_censored_by_scored_end": bool(window_end > scored_end),
                "right_censored_by_staged_end": bool(window_end > staged_end),
            }
        )
    by_bar = {}
    ends = cache.bars.bar_end
    scored = cache.bars.session.isin(scored_calendar.session)
    for j, (_, gap_end, window_end) in enumerate(windows):
        for i in np.flatnonzero((scored & ends.gt(gap_end) & ends.le(window_end)).to_numpy()):
            by_bar.setdefault(int(i), []).append(j)
    return {
        "windows": tuple(windows),
        "metadata_windows": metadata,
        "policy": cache.policy,
        "by_bar": {key: tuple(value) for key, value in by_bar.items()},
    }


def _post_gap_table(rows):
    frame = pd.DataFrame(rows, columns=POST_GAP_EVENT_COLUMNS)
    for column in ("decision_at", "gap_start", "gap_end", "window_end"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.sort_values(
        ["decision_at", "candidate_id", "gap_start", "side"], kind="stable"
    ).reset_index(drop=True)


def _post_gap_metadata(context, policy, total_rows, stored_rows, metrics):
    return {
        "enabled": True,
        "event_limit": policy.post_gap_event_limit,
        "total_event_rows": int(total_rows),
        "stored_event_rows": int(stored_rows),
        "truncated": bool(total_rows > stored_rows),
        "counting_unit": "distinct_confirmed_decisions_per_candidate",
        "event_table_unit": "confirmed_decision_gap_window_associations",
        "total_confirmed_buy_decisions": int(
            sum(x.get("post_gap_confirmed_buy_count", 0) for x in metrics)
        ),
        "total_confirmed_sell_decisions": int(
            sum(x.get("post_gap_confirmed_sell_count", 0) for x in metrics)
        ),
        "total_confirmed_decisions": int(
            sum(x.get("post_gap_confirmed_signal_count", 0) for x in metrics)
        ),
        "window_count": len(context["windows"]),
        "active_window_count": sum(
            x["overlaps_scored_segment"] for x in context["metadata_windows"]
        ),
        "windows": context["metadata_windows"],
    }


def _batch(
    cache,
    calendar,
    events,
    clock,
    terminal_index,
    candidates,
    cash_observations,
    commission_bps,
    slippage_bps,
    initial_cash,
    *,
    mode="candidate",
    record=False,
    clock_returns=None,
    post_gap_context=None,
    post_gap_event_budget=None,
):
    fee, slip = core._costs(commission_bps, slippage_bps, initial_cash)
    if "series_id" in cash_observations and not cash_observations.series_id.eq("DTB3").all():
        raise ValueError("Cash observations must be causal DTB3")
    accrued = (
        rate_accrual(clock, cash_observations, interval_start=clock[0])
        if clock_returns is None
        else clock_returns
    )
    if (accrued <= -1).any():
        raise ValueError("Cash wealth exhausted")
    factor = dict(zip(clock, accrued.to_numpy(), strict=True))
    n = len(candidates)
    ids = [c.candidate_id for c in candidates] if mode == "candidate" else [f"benchmark_{mode}"]
    lookup = {r: i for i, r in enumerate(cache.rules)}
    ec = np.asarray([lookup[c.entry_rule] for c in candidates])
    xc = np.asarray([lookup[c.exit_rule] for c in candidates])
    en = np.asarray([c.entry_required_checks for c in candidates])
    xn = np.asarray([c.exit_required_checks for c in candidates])
    cash = np.full(n, initial_cash, dtype=float)
    gross_cash = cash.copy()
    shares = np.zeros(n)
    gross_shares = np.zeros(n)
    dividend_cash = np.zeros(n)
    gross_dividend_cash = np.zeros(n)
    long = np.zeros(n, dtype=bool)
    pending = np.zeros(n, dtype=np.int8)
    entry_count = np.zeros(n, dtype=int)
    exit_count = np.zeros(n, dtype=int)
    pending_at = np.full(n, np.nan)
    entered_at = np.full(n, np.nan)
    commission = np.zeros(n)
    slippage = np.zeros(n)
    turnover = np.zeros(n)
    trades = np.zeros(n, dtype=int)
    orders = np.zeros(n, dtype=int)
    holds = np.zeros(n)
    held_seconds = np.zeros(n)
    cancelled = np.zeros(n, dtype=int)
    terminal_cancelled = np.zeros(n, dtype=int)
    cash_reference = initial_cash
    previous = clock[0]
    daily = {row.session: row for row in cache.daily.itertuples(index=False)}
    day_equity = []
    day_gross = []
    day_cash = []
    day_exposure = []
    day_times = []
    trade_rows = []
    diagnostics = []
    benchmark_bought = False
    post_enabled = mode == "candidate" and cache.policy.post_gap_diagnostics
    if post_enabled and post_gap_context is None:
        post_gap_context = _post_gap_context(cache, calendar)
    post_buy = np.zeros(n, dtype=np.int64)
    post_sell = np.zeros(n, dtype=np.int64)
    post_rows = []
    post_total_rows = 0
    post_limit = (
        cache.policy.post_gap_event_limit
        if post_gap_event_budget is None
        else max(0, post_gap_event_budget)
    )

    def fill(mask, side, price, timestamp, session, reason):
        if not mask.any():
            return
        equity = cash + shares * price
        execution = price * (1 + slip if side == "buy" else 1 - slip)
        if side == "buy":
            qty = cash[mask] / (execution * (1 + fee))
            charged = qty * execution * fee
            cash[mask] = 0
            dividend_cash[mask] = 0
            shares[mask] = qty
            gross_shares[mask] = gross_cash[mask] / price
            gross_cash[mask] = 0
            gross_dividend_cash[mask] = 0
            long[mask] = True
            entered_at[mask] = timestamp.value / 1e9
            holding = np.zeros(mask.sum())
        else:
            qty = shares[mask].copy()
            charged = qty * execution * fee
            cash[mask] += qty * execution - charged
            shares[mask] = 0
            dividend_cash[mask] = 0
            gross_cash[mask] += gross_shares[mask] * price
            gross_shares[mask] = 0
            gross_dividend_cash[mask] = 0
            long[mask] = False
            holding = (timestamp.value / 1e9 - entered_at[mask]) / 3600
            holds[mask] += holding
            trades[mask] += 1
        notional = qty * price
        adverse = qty * abs(execution - price)
        commission[mask] += charged
        slippage[mask] += adverse
        turnover[mask] += notional / equity[mask]
        orders[mask] += 1
        if record:
            for k, j in enumerate(np.flatnonzero(mask)):
                decision = (
                    None
                    if np.isnan(pending_at[j])
                    else pd.Timestamp(pending_at[j], unit="s", tz="UTC")
                )
                trade_rows.append(
                    dict(
                        candidate_id=ids[j],
                        side=side,
                        decision_at=decision,
                        timestamp=timestamp,
                        session=session,
                        reason=reason,
                        reference_price=price,
                        execution_price=execution,
                        quantity=qty[k],
                        reference_notional=notional[k],
                        commission=charged[k],
                        slippage=adverse[k],
                        equity_before=equity[j],
                        holding_hours=holding[k],
                    )
                )
        pending[mask] = 0
        pending_at[mask] = np.nan
        entry_count[mask] = 0
        exit_count[mask] = 0

    for timestamp, _, kind, item in events:
        if timestamp != previous:
            elapsed = (timestamp - previous).total_seconds()
            held_seconds += long * elapsed
            cash *= 1 + factor[timestamp]
            gross_cash *= 1 + factor[timestamp]
            dividend_cash *= 1 + factor[timestamp]
            gross_dividend_cash *= 1 + factor[timestamp]
            cash_reference *= 1 + factor[timestamp]
            previous = timestamp
        if kind == "blackout_start":
            cancelled += pending != 0
            pending[:] = 0
            pending_at[:] = np.nan
            entry_count[:] = 0
            exit_count[:] = 0
            if record:
                diagnostics.append(
                    dict(
                        timestamp=timestamp,
                        event="blackout_start",
                        reason=cache.blackouts.iloc[item].reason,
                        entry_count=0,
                        exit_count=0,
                        position=int(long[0]),
                        pending=0,
                    )
                )
        elif kind == "actions":
            action = daily[item]
            shares *= action.split_coefficient
            gross_shares *= action.split_coefficient
            distribution = shares * action.dividend_amount
            gross_distribution = gross_shares * action.dividend_amount
            cash += distribution
            dividend_cash += distribution
            gross_cash += gross_distribution
            gross_dividend_cash += gross_distribution
        elif kind == "bar_open":
            row = cache.bar_rows[item]
            if cache.fill_eligible[item]:
                mask = long & (dividend_cash > 0)
                shares[mask] += dividend_cash[mask] / row.open
                cash[mask] -= dividend_cash[mask]
                dividend_cash[mask] = 0
                gross_mask = long & (gross_dividend_cash > 0)
                gross_shares[gross_mask] += gross_dividend_cash[gross_mask] / row.open
                gross_cash[gross_mask] -= gross_dividend_cash[gross_mask]
                gross_dividend_cash[gross_mask] = 0
                if mode == "buy_and_hold" and not benchmark_bought:
                    fill(
                        np.ones(n, dtype=bool),
                        "buy",
                        row.open,
                        timestamp,
                        row.session,
                        "benchmark_entry",
                    )
                    benchmark_bought = True
                else:
                    fill(pending == -1, "sell", row.open, timestamp, row.session, "confirmed_exit")
                    fill(pending == 1, "buy", row.open, timestamp, row.session, "confirmed_entry")
        elif kind == "bar_close":
            row = cache.bar_rows[item]
            if mode == "candidate" and cache.signal_eligible[item]:
                margin = cache.margins[item]
                entry_ok = margin[ec] > 0
                exit_ok = margin[xc] > 0
                exit_valid = np.isfinite(margin[xc])
                entry_count[:] = np.where(~long & entry_ok & exit_ok, entry_count + 1, 0)
                exit_count[:] = np.where(long & exit_valid & ~exit_ok, exit_count + 1, 0)
                buys = ~long & (entry_count >= en)
                sells = long & (exit_count >= xn)
                new_buys = buys & (pending != 1)
                new_sells = sells & (pending != -1)
                if post_enabled and item in post_gap_context["by_bar"]:
                    # Count distinct newly CONFIRMED decisions, never mere support,
                    # fills, pre-gap cancellation, or terminal liquidation.
                    post_buy += new_buys
                    post_sell += new_sells
                    decisions = np.flatnonzero(new_buys | new_sells)
                    for gap_index in post_gap_context["by_bar"][item]:
                        gap_start, gap_end, window_end = post_gap_context["windows"][gap_index]
                        post_total_rows += len(decisions)
                        available = max(0, post_limit - len(post_rows))
                        for j in decisions[:available]:
                            post_rows.append(
                                (
                                    ids[j],
                                    "buy" if new_buys[j] else "sell",
                                    timestamp,
                                    gap_start,
                                    gap_end,
                                    window_end,
                                )
                            )
                pending[buys] = 1
                pending[sells] = -1
                pending_at[buys | sells] = timestamp.value / 1e9
            if record:
                diagnostics.append(
                    dict(
                        timestamp=timestamp,
                        event="bar_close",
                        session=row.session,
                        signal_eligible=bool(cache.signal_eligible[item]),
                        fill_eligible=bool(cache.fill_eligible[item]),
                        entry_count=int(entry_count[0]),
                        exit_count=int(exit_count[0]),
                        position=int(long[0]),
                        pending=int(pending[0]),
                    )
                )
            if item == terminal_index:
                terminal_cancelled[:] = pending != 0
                pending[:] = 0
                pending_at[:] = np.nan
                fill(long.copy(), "sell", row.close, timestamp, row.session, "terminal_liquidation")
        elif kind == "valuation":
            price = float(daily[item].close)
            equity = cash + shares * price
            gross = gross_cash + gross_shares * price
            if not np.isfinite(equity).all() or (equity <= 0).any() or not np.isfinite(gross).all():
                raise ValueError("Account wealth is nonfinite or exhausted")
            day_equity.append(equity.copy())
            day_gross.append(gross.copy())
            day_cash.append(cash_reference)
            day_exposure.append(long.astype(float))
            day_times.append(timestamp)
    times = pd.DatetimeIndex(day_times, name="timestamp")
    equity = np.asarray(day_equity)
    gross = np.asarray(day_gross)
    returns = equity / np.vstack([np.full(n, initial_cash), equity[:-1]]) - 1
    ref = np.asarray(day_cash)
    cash_returns = ref / np.r_[initial_cash, ref[:-1]] - 1
    seconds = (clock[-1] - clock[0]).total_seconds()
    years = seconds / (365 * 86400)
    metrics = []
    cov = cache.coverage.loc[cache.coverage.session.isin(calendar.session)]
    unknown_slots = int(cov.status.eq("unknown_missing").sum())
    halt_slots = int(cov.status.eq("known_halt_overlap").sum())
    source_presence_complete = bool(cov.bar_start.isin(cache.bars.bar_start).all())
    all_signals_eligible = bool(cov.signal_eligible.all())
    raw_price_complete = bool(cov.status.eq("observed").all()) and source_presence_complete
    for j, c in enumerate(candidates):
        ret = returns[:, j]
        excess = ret - cash_returns
        sd = np.std(excess, ddof=1) if len(excess) > 1 else np.nan
        sharpe = float(np.sqrt(252) * np.mean(excess) / sd) if sd > 0 else np.nan
        wealth = np.r_[initial_cash, equity[:, j]]
        metrics.append(
            dict(
                candidate_id=ids[j],
                entry_family=c.entry_rule.family,
                exit_family=c.exit_rule.family,
                family_cell=c.family_cell,
                entry_rule=c.entry_rule.rule_id,
                exit_rule=c.exit_rule.rule_id,
                entry_confirmation_hours=c.entry_confirmation_hours,
                exit_confirmation_hours=c.exit_confirmation_hours,
                entry_required_checks=c.entry_required_checks,
                exit_required_checks=c.exit_required_checks,
                cash_excess_sharpe=sharpe,
                cash_excess_Sharpe=sharpe,
                cagr=float((equity[-1, j] / initial_cash) ** (1 / years) - 1) if years else np.nan,
                max_drawdown=float((wealth / np.maximum.accumulate(wealth) - 1).min()),
                annualized_volatility=float(np.std(ret, ddof=1) * np.sqrt(252))
                if len(ret) > 1
                else np.nan,
                exposure_fraction=float(held_seconds[j] / seconds) if seconds else 0,
                total_turnover=float(turnover[j]),
                annualized_turnover=float(turnover[j] / years) if years else np.nan,
                trade_count=int(trades[j]),
                order_count=int(orders[j]),
                average_holding_hours=float(holds[j] / trades[j]) if trades[j] else 0,
                total_commission=float(commission[j]),
                total_slippage=float(slippage[j]),
                cost_drag_return=float((gross[-1, j] - equity[-1, j]) / initial_cash),
                terminal_equity=float(equity[-1, j]),
                gross_terminal_equity=float(gross[-1, j]),
                cash_terminal_equity=float(ref[-1]),
                total_return=float(equity[-1, j] / initial_cash - 1),
                scored_sessions=len(times),
                start_session=calendar.session.iloc[0],
                end_session=calendar.session.iloc[-1],
                sampling_minutes=30,
                commission_bps=float(commission_bps),
                slippage_bps=float(slippage_bps),
                quality_profile=cache.policy.profile,
                daily_valuation_hour_ny=17,
                daily_quote_kind="provider_daily_raw_close_not_NOCP",
                unknown_missing_slots=unknown_slots,
                halt_affected_slots=halt_slots,
                quality_classification_complete=True,
                tradable_price_coverage_complete=unknown_slots == 0,
                source_row_presence_complete=source_presence_complete,
                raw_intraday_price_coverage_complete=raw_price_complete,
                all_slots_signal_eligible=all_signals_eligible,
                cancelled_blackout_order_count=int(cancelled[j]),
                cancelled_terminal_order_count=int(terminal_cancelled[j]),
            )
        )
    post_payload = None
    if post_enabled:
        for j, stats in enumerate(metrics):
            stats.update(
                post_gap_confirmed_buy_count=int(post_buy[j]),
                post_gap_confirmed_sell_count=int(post_sell[j]),
                post_gap_confirmed_signal_count=int(post_buy[j] + post_sell[j]),
                has_post_gap_confirmed_signal=bool(post_buy[j] + post_sell[j]),
            )
        if record:
            for candidate_id, side, decision_at, gap_start, gap_end, window_end in post_rows:
                diagnostics.append(
                    dict(
                        event="post_gap_confirmed_signal",
                        timestamp=decision_at,
                        candidate_id=candidate_id,
                        side=side,
                        decision_at=decision_at,
                        gap_start=gap_start,
                        gap_end=gap_end,
                        window_end=window_end,
                    )
                )
        post_payload = {
            "rows": post_rows,
            "total_rows": post_total_rows,
            "context": post_gap_context,
        }
    return (
        metrics,
        times,
        equity,
        returns,
        cash_returns,
        np.asarray(day_exposure),
        trade_rows,
        diagnostics,
        post_payload,
    )


def _run(
    cache,
    candidates,
    *,
    start,
    end,
    cash_observations,
    commission_bps,
    slippage_bps,
    initial_cash,
    mode="candidate",
    record=False,
):
    calendar, events, clock, terminal = _segment_events(cache, start, end)
    return _batch(
        cache,
        calendar,
        events,
        clock,
        terminal,
        candidates,
        cash_observations,
        commission_bps,
        slippage_bps,
        initial_cash,
        mode=mode,
        record=record,
    )


def _result(parts):
    metrics, times, equity, returns, cash_returns, exposure, trades, diag, post_payload = parts
    diagnostics = pd.DataFrame(diag)
    if post_payload is not None:
        context = post_payload["context"]
        diagnostics.attrs["post_gap_signal_event_metadata"] = _post_gap_metadata(
            context,
            context["policy"],
            post_payload["total_rows"],
            len(post_payload["rows"]),
            metrics,
        )
        diagnostics.attrs["post_gap_signal_events"] = _post_gap_table(post_payload["rows"])
    return SimulationResult(
        metrics[0],
        pd.Series(returns[:, 0], index=times, name="strategy_return"),
        pd.Series(equity[:, 0], index=times, name="equity"),
        pd.DataFrame(trades, columns=core.TRADE_COLUMNS),
        pd.Series(cash_returns, index=times, name="cash_return"),
        pd.Series(exposure[:, 0], index=times, name="valuation_exposure"),
        diagnostics,
    )


def simulate_quality_candidate(
    daily,
    bars,
    candidate,
    *,
    start,
    end,
    cash_observations,
    calendar,
    coverage=None,
    blackouts=None,
    quality_policy=DEFAULT_POLICY,
    commission_bps=1,
    slippage_bps=3,
    initial_cash=1000,
    support_cache=None,
):
    rules = tuple(
        dict.fromkeys((*core.indicator_rules(), candidate.entry_rule, candidate.exit_rule))
    )
    cache = (
        _validate_runtime_cache(
            support_cache, daily, bars, calendar, coverage, blackouts, quality_policy
        )
        if support_cache is not None
        else None
    )
    cache = cache or build_quality_support_cache(
        daily,
        bars,
        calendar,
        coverage=coverage,
        blackouts=blackouts,
        quality_policy=quality_policy,
        rules=rules,
    )
    return _result(
        _run(
            cache,
            [candidate],
            start=start,
            end=end,
            cash_observations=cash_observations,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            initial_cash=initial_cash,
            record=True,
        )
    )


def evaluate_quality_grid(
    daily,
    bars,
    *,
    start,
    end,
    cash_observations,
    calendar,
    candidates=None,
    coverage=None,
    blackouts=None,
    quality_policy=DEFAULT_POLICY,
    commission_bps=1,
    slippage_bps=3,
    initial_cash=1000,
    batch_size=256,
    support_cache=None,
):
    candidates = generate_candidates() if candidates is None else list(candidates)
    if not candidates or len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("Nonempty unique candidate grid required")
    if not core._positive_integer(batch_size) or batch_size > 256:
        raise ValueError("Quality-grid batch_size must be an integer in 1..256")
    rules = tuple(
        dict.fromkeys(
            (*core.indicator_rules(), *(r for c in candidates for r in (c.entry_rule, c.exit_rule)))
        )
    )
    cache = (
        _validate_runtime_cache(
            support_cache, daily, bars, calendar, coverage, blackouts, quality_policy
        )
        if support_cache is not None
        else None
    )
    cache = cache or build_quality_support_cache(
        daily,
        bars,
        calendar,
        coverage=coverage,
        blackouts=blackouts,
        quality_policy=quality_policy,
        rules=rules,
    )
    calendar, events, clock, terminal = _segment_events(cache, start, end)
    clock_returns = rate_accrual(clock, cash_observations, interval_start=clock[0])
    rows = []
    context = _post_gap_context(cache, calendar)
    event_rows = []
    total_event_rows = 0
    for offset in range(0, len(candidates), batch_size):
        parts = _batch(
            cache,
            calendar,
            events,
            clock,
            terminal,
            candidates[offset : offset + batch_size],
            cash_observations,
            commission_bps,
            slippage_bps,
            initial_cash,
            clock_returns=clock_returns,
            post_gap_context=context,
            post_gap_event_budget=cache.policy.post_gap_event_limit - len(event_rows),
        )
        rows.extend(parts[0])
        if parts[-1] is not None:
            event_rows.extend(parts[-1]["rows"])
            total_event_rows += parts[-1]["total_rows"]
    frame = pd.DataFrame(rows)
    if context is not None:
        frame.attrs["post_gap_signal_events"] = _post_gap_table(event_rows)
        frame.attrs["post_gap_signal_event_metadata"] = _post_gap_metadata(
            context, cache.policy, total_event_rows, len(event_rows), rows
        )
    return frame


def simulate_quality_benchmarks(
    daily,
    bars,
    *,
    start,
    end,
    cash_observations,
    calendar,
    coverage=None,
    blackouts=None,
    quality_policy=DEFAULT_POLICY,
    commission_bps=1,
    slippage_bps=3,
    initial_cash=1000,
    support_cache=None,
):
    cache = (
        _validate_runtime_cache(
            support_cache, daily, bars, calendar, coverage, blackouts, quality_policy
        )
        if support_cache is not None
        else None
    )
    cache = cache or build_quality_support_cache(
        daily, bars, calendar, coverage=coverage, blackouts=blackouts, quality_policy=quality_policy
    )
    dummy = Candidate(cache.rules[0], cache.rules[0], 0, 0)
    return {
        mode: _result(
            _run(
                cache,
                [dummy],
                start=start,
                end=end,
                cash_observations=cash_observations,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                initial_cash=initial_cash,
                mode=mode,
                record=True,
            )
        )
        for mode in ("cash", "buy_and_hold")
    }
