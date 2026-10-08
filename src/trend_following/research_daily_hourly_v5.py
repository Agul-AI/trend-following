"""Isolated v5 daily-indicator/half-hour-confirmation research kernel.

This module never opens market/result files. Callers supply stage-limited frames.
EMA states advance once per official daily close, not once per preview half-hour.
Signals use a forward-built total-return index; orders use raw tradable prices.
All scored segments start in cash, with no inherited order or confirmation count.

Corporate-action convention: splits and ex-date dividend credits precede the
session-open order. Only pre-open holders receive dividends (per post-split
share); distributions immediately DRIP at that raw open without transition
fees. This ex-date/open reinvestment proxy is not an actual payment-date ledger.
Ordinary entry/exit and terminal orders each pay commission and adverse slippage
once. No tax, leverage, borrowing, synthetic security, or extra expense exists.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from trend_following.research_rates import rate_accrual

MA_PERIODS = (10, 20, 50, 100, 200, 250)
MACD_PARAMETERS = ((6, 13, 5), (12, 26, 9), (24, 52, 18), (48, 104, 36), (96, 208, 72))
CONFIRMATION_HOURS = tuple(tick / 2.0 for tick in range(14))
SAMPLING_MINUTES = 30
FEATURE_SEED_DATE = "1999-11-01"
WARMUP_SESSIONS = 295
OFFICIAL_CLOSE_RTOL = 0.001
OFFICIAL_CLOSE_ATOL = 0.02
TRADE_COLUMNS = (
    "candidate_id",
    "side",
    "decision_at",
    "timestamp",
    "session",
    "reason",
    "reference_price",
    "execution_price",
    "quantity",
    "reference_notional",
    "commission",
    "slippage",
    "equity_before",
    "holding_hours",
)


@dataclass(frozen=True)
class IndicatorRule:
    family: str
    period: int | None = None
    fast: int | None = None
    slow: int | None = None
    signal: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "family", str(self.family).upper())
        if self.family in {"SMA", "EMA"}:
            if not _positive_integer(self.period) or any(
                value is not None for value in (self.fast, self.slow, self.signal)
            ):
                raise ValueError("An MA rule requires only a positive integer period")
        elif self.family == "MACD":
            if (
                self.period is not None
                or not all(_positive_integer(x) for x in (self.fast, self.slow, self.signal))
                or self.fast >= self.slow
            ):
                raise ValueError("MACD requires positive integer fast < slow and signal")
        else:
            raise ValueError("Rule family must be SMA, EMA, or MACD")
        for field in ("period", "fast", "slow", "signal"):
            if getattr(self, field) is not None:
                object.__setattr__(self, field, int(getattr(self, field)))

    @property
    def rule_id(self) -> str:
        if self.family == "MACD":
            return f"macd_{self.fast}_{self.slow}_{self.signal}"
        return f"{self.family.lower()}_{self.period}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IndicatorRule:
        return cls(**value)


def _positive_integer(value) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value > 0
    )


def _confirmation_hours(value) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or float(value) not in CONFIRMATION_HOURS
    ):
        raise ValueError(
            "Confirmation hours must be 0, 0.5, ..., 6.5; bool/nonfinite/quarter-hour values are invalid"
        )
    return float(value)


def _hours_id(hours: float) -> str:
    return f"{hours:g}".replace(".", "p")


@dataclass(frozen=True)
class Candidate:
    """Wait h TRADING hours after the first qualifying completed 30m close.

    h=0 requires one check; h=0.5 two; h=1 three; h=6.5 fourteen.
    Closures consume no confirmation time, so streaks carry between sessions.
    """

    entry_rule: IndicatorRule
    exit_rule: IndicatorRule
    entry_confirmation_hours: float
    exit_confirmation_hours: float

    def __post_init__(self):
        if not isinstance(self.entry_rule, IndicatorRule) or not isinstance(
            self.exit_rule, IndicatorRule
        ):
            raise ValueError("Candidate entry/exit rules must be IndicatorRule objects")
        for field in ("entry_confirmation_hours", "exit_confirmation_hours"):
            object.__setattr__(self, field, _confirmation_hours(getattr(self, field)))

    @property
    def entry_required_checks(self) -> int:
        return 1 + int(2 * self.entry_confirmation_hours)

    @property
    def exit_required_checks(self) -> int:
        return 1 + int(2 * self.exit_confirmation_hours)

    @property
    def candidate_id(self) -> str:
        return (
            f"v5_30m__entry_{self.entry_rule.rule_id}__exit_{self.exit_rule.rule_id}"
            f"__entry_hours_{_hours_id(self.entry_confirmation_hours)}"
            f"__exit_hours_{_hours_id(self.exit_confirmation_hours)}"
        )

    @property
    def family_cell(self) -> str:
        return f"{self.entry_rule.family}/{self.exit_rule.family}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_rule": self.entry_rule.to_dict(),
            "exit_rule": self.exit_rule.to_dict(),
            "entry_confirmation_hours": self.entry_confirmation_hours,
            "exit_confirmation_hours": self.exit_confirmation_hours,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Candidate:
        expected = {
            "entry_rule",
            "exit_rule",
            "entry_confirmation_hours",
            "exit_confirmation_hours",
        }
        if set(value) != expected:
            raise ValueError(
                "Candidate schema requires explicit confirmation_hours; old integer confirmation fields are not compatible"
            )
        return cls(
            IndicatorRule.from_dict(value["entry_rule"]),
            IndicatorRule.from_dict(value["exit_rule"]),
            value["entry_confirmation_hours"],
            value["exit_confirmation_hours"],
        )


def indicator_rules() -> tuple[IndicatorRule, ...]:
    return tuple(
        IndicatorRule(family, period=n) for family in ("SMA", "EMA") for n in MA_PERIODS
    ) + tuple(IndicatorRule("MACD", fast=f, slow=s, signal=g) for f, s, g in MACD_PARAMETERS)


def generate_candidates() -> list[Candidate]:
    """The complete, ordered 17 x 17 x 14 x 14 = 56,644 frozen search grid."""
    return [
        Candidate(entry, exit_rule, en, ex)
        for entry in indicator_rules()
        for exit_rule in indicator_rules()
        for en in CONFIRMATION_HOURS
        for ex in CONFIRMATION_HOURS
    ]


class _DailyState:
    """Continuous recursive EMA seeded at first observation, with full minperiods."""

    def __init__(self, rules: tuple[IndicatorRule, ...]):
        self.rules = rules
        self.count = 0
        periods = {r.period for r in rules if r.family in {"SMA", "EMA"}}
        periods |= {p for r in rules if r.family == "MACD" for p in (r.fast, r.slow)}
        self.ema = dict.fromkeys(periods)
        self.history = deque(maxlen=max(periods, default=1))
        self.macd_signal = {r: None for r in rules if r.family == "MACD"}
        self.macd_count = dict.fromkeys(self.macd_signal, 0)

    def preview(self, close: float) -> tuple[np.ndarray, dict, dict, dict]:
        count = self.count + 1
        ema = {
            p: close if old is None else old + 2 / (p + 1) * (close - old)
            for p, old in self.ema.items()
        }
        signals, counts = self.macd_signal.copy(), self.macd_count.copy()
        values = []
        history = list(self.history)
        for rule in self.rules:
            if rule.family == "SMA":
                value = (
                    close - (sum(history[-(rule.period - 1) :]) + close) / rule.period
                    if count >= rule.period
                    else np.nan
                )
                # A one-period fabricated rule has no previous observations.
                if rule.period == 1:
                    value = 0.0
            elif rule.family == "EMA":
                value = close - ema[rule.period] if count >= rule.period else np.nan
            elif count >= rule.slow:
                macd = ema[rule.fast] - ema[rule.slow]
                old = signals[rule]
                signals[rule] = macd if old is None else old + 2 / (rule.signal + 1) * (macd - old)
                counts[rule] += 1
                value = macd - signals[rule] if counts[rule] >= rule.signal else np.nan
            else:
                value = np.nan
            values.append(value)
        return np.asarray(values, dtype=float), ema, signals, counts

    def finalize(self, close: float) -> None:
        _, self.ema, self.macd_signal, self.macd_count = self.preview(close)
        self.history.append(close)
        self.count += 1


@dataclass(frozen=True)
class SupportCache:
    """Reusable causal features; NaN margin means the rule is not yet eligible."""

    daily: pd.DataFrame
    bars: pd.DataFrame
    calendar: pd.DataFrame
    rules: tuple[IndicatorRule, ...]
    margins: np.ndarray
    provisional_total_return_index: np.ndarray
    finalized_total_return_index: pd.Series
    finalized_daily_margins: pd.DataFrame
    prior_session_counts: np.ndarray

    @property
    def support(self) -> np.ndarray:
        return self.margins > 0.0


def _sessions(values) -> pd.Series:
    stamps = pd.to_datetime(values, errors="raise")
    if getattr(stamps.dtype, "tz", None) is not None:
        raise ValueError("Session dates must be timezone-naive")
    if stamps.isna().any():
        raise ValueError("Missing session")
    return stamps.dt.strftime("%Y-%m-%d")


def _utc(values, label: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    if index.tz is None or index.hasnans:
        raise ValueError(f"{label} must be nonmissing timezone-aware timestamps")
    return index.tz_convert("UTC")


def _prepare_frames(daily, bars, calendar):
    daily, bars = daily.copy(deep=True), bars.copy(deep=True)
    if not {"session", "close", "dividend_amount", "split_coefficient"}.issubset(daily):
        raise ValueError("Raw daily session, close, dividend_amount, split_coefficient required")
    required = {
        "bar_start",
        "bar_end",
        "session",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "duration_minutes",
    }
    if not required.issubset(bars) or daily.empty or bars.empty:
        raise ValueError("Nonempty normalized daily and intraday frames required")
    daily["session"], bars["session"] = _sessions(daily.session), _sessions(bars.session)
    daily = daily.sort_values("session").reset_index(drop=True)
    bars["bar_start"], bars["bar_end"] = (
        _utc(bars.bar_start, "bar_start"),
        _utc(bars.bar_end, "bar_end"),
    )
    bars = bars.sort_values("bar_start").reset_index(drop=True)
    if daily.session.duplicated().any() or bars.bar_start.duplicated().any():
        raise ValueError("Duplicate daily sessions or intraday starts")
    numeric = daily[["close", "dividend_amount", "split_coefficient"]].to_numpy(float)
    prices = bars[["open", "high", "low", "close", "volume", "duration_minutes"]].to_numpy(float)
    if (
        not np.isfinite(numeric).all()
        or not np.isfinite(prices).all()
        or (numeric[:, 0] <= 0).any()
        or (numeric[:, 1] < 0).any()
        or (numeric[:, 2] <= 0).any()
        or (prices[:, :4] <= 0).any()
        or (prices[:, 4] < 0).any()
    ):
        raise ValueError("Invalid raw prices or corporate actions")
    if (bars.high < bars[["open", "close", "low"]].max(axis=1)).any() or (
        bars.low > bars[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("Incoherent OHLC bounds")
    elapsed = (bars.bar_end - bars.bar_start).dt.total_seconds() / 60
    if (elapsed <= 0).any() or not np.allclose(elapsed, bars.duration_minutes):
        raise ValueError("Duration must match a positive bar interval")
    if (
        not elapsed.eq(SAMPLING_MINUTES).all()
        or not bars.duration_minutes.eq(SAMPLING_MINUTES).all()
    ):
        raise ValueError("Research sampling requires every completed bar to be exactly 30 minutes")
    if "is_full_hour" in bars and (
        not bars.is_full_hour.map(lambda x: isinstance(x, (bool, np.bool_))).all()
        or bars.is_full_hour.any()
    ):
        raise ValueError("is_full_hour, if retained, must be false for 30-minute bars")
    if (bars.bar_start.iloc[1:].to_numpy() < bars.bar_end.iloc[:-1].to_numpy()).any():
        raise ValueError("Overlapping intraday bars")
    if not bars.session.isin(daily.session).all():
        raise ValueError("Every intraday session needs a raw daily action record")
    if "is_completed" in bars and not bars.is_completed.eq(True).all():
        raise ValueError("Incomplete bars cannot be used")
    if calendar is None:
        calendar = (
            bars.groupby("session", sort=True)
            .agg(market_open=("bar_start", "min"), market_close=("bar_end", "max"))
            .reset_index()
        )
    else:
        calendar = calendar.copy(deep=True)
        if not {"session", "market_open", "market_close"}.issubset(calendar):
            raise ValueError("Calendar requires session, market_open and market_close")
        calendar["session"] = _sessions(calendar.session)
        calendar["market_open"], calendar["market_close"] = (
            _utc(calendar.market_open, "market_open"),
            _utc(calendar.market_close, "market_close"),
        )
        calendar = calendar.sort_values("session").reset_index(drop=True)
    if calendar.session.duplicated().any() or not bars.session.isin(calendar.session).all():
        raise ValueError("Unique calendar rows required for every intraday session")
    schedule = calendar.set_index("session")
    for session, frame in bars.groupby("session", sort=False):
        row = schedule.loc[session]
        if (
            row.market_close <= row.market_open
            or frame.bar_start.iloc[0] != row.market_open
            or frame.bar_end.iloc[-1] != row.market_close
            or not np.array_equal(
                frame.bar_start.iloc[1:].to_numpy(), frame.bar_end.iloc[:-1].to_numpy()
            )
        ):
            raise ValueError(f"Missing bar or incomplete session grid: {session}")
        raw_close = daily.loc[daily.session.eq(session), "close"].iloc[0]
        if not np.isclose(
            frame.close.iloc[-1], raw_close, rtol=OFFICIAL_CLOSE_RTOL, atol=OFFICIAL_CLOSE_ATOL
        ):
            raise ValueError(f"Official close and final intraday close disagree: {session}")
    return daily, bars, calendar


def build_support_cache(
    daily: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: pd.DataFrame | None = None,
    *,
    rules: tuple[IndicatorRule, ...] | None = None,
) -> SupportCache:
    """Preview each completed half-hour from the preceding daily state, then finalize.

    Raw same-session split/dividend metadata is an ex-open action assumption;
    neither today's official close nor tomorrow's open enters an earlier signal.
    Official-close comparisons use rtol=0.001, atol=0.02 without changing
    observed bar quotes. Features finalize from the official daily close, while
    portfolio marks and terminal fills use the observed last intraday close.
    Calendar-less operation is for fabricated fixtures only: production callers
    must validate an authoritative calendar and exact grid before calling.
    """
    daily, bars, calendar = _prepare_frames(daily, bars, calendar)
    rules = indicator_rules() if rules is None else tuple(rules)
    if (
        not rules
        or len(set(rules)) != len(rules)
        or not all(isinstance(r, IndicatorRule) for r in rules)
    ):
        raise ValueError("Unique nonempty IndicatorRule collection required")
    state = _DailyState(rules)
    margins = np.full((len(bars), len(rules)), np.nan)
    provisional, prior_counts = np.empty(len(bars)), np.empty(len(bars), dtype=int)
    tr_values, final_margins = [], []
    groups = bars.groupby("session", sort=False).indices
    previous_close, previous_index = None, 100.0
    for row in daily.itertuples(index=False):
        if previous_close is None:
            finalized = 100.0
        else:
            finalized = (
                previous_index
                * row.split_coefficient
                * (row.close + row.dividend_amount)
                / previous_close
            )
        if not np.isfinite(finalized) or finalized <= 0:
            raise ValueError("Total-return feature wealth is nonfinite or exhausted")
        for i in groups.get(row.session, []):
            value = (
                np.nan
                if previous_close is None
                else previous_index
                * row.split_coefficient
                * (bars.close.iloc[i] + row.dividend_amount)
                / previous_close
            )
            if previous_close is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("Provisional total-return wealth is nonfinite or exhausted")
            # First seed-session bars cannot be causal without a preceding raw close.
            if previous_close is not None:
                margins[i] = state.preview(value)[0]
            provisional[i], prior_counts[i] = value, state.count
        final_margins.append(state.preview(finalized)[0])
        state.finalize(finalized)
        tr_values.append(finalized)
        previous_close, previous_index = row.close, finalized
    session_index = pd.Index(daily.session, name="session")
    return SupportCache(
        daily,
        bars,
        calendar,
        rules,
        margins,
        provisional,
        pd.Series(tr_values, index=session_index, name="total_return_index"),
        pd.DataFrame(final_margins, index=session_index, columns=[r.rule_id for r in rules]),
        prior_counts,
    )


def preview_daily_margins(
    daily_completed: pd.DataFrame,
    raw_close: float,
    *,
    split_coefficient: float = 1.0,
    dividend_amount: float = 0.0,
    rules: tuple[IndicatorRule, ...] | None = None,
) -> pd.Series:
    """Pure current-session preview from completed DAILY history only.

    No current-day daily row, calendar, fabricated final close, or incomplete bar
    is needed. Same-day action values must be explicitly known at ex-open. The
    returned margins use the same seed/EMA/MACD rules as historical simulation;
    the completed state is never advanced with the current provisional value.
    """
    frame = daily_completed.copy(deep=True)
    required = {"session", "close", "dividend_amount", "split_coefficient"}
    if frame.empty or not required.issubset(frame):
        raise ValueError("Nonempty completed raw daily history required")
    frame["session"] = _sessions(frame.session)
    frame = frame.sort_values("session").reset_index(drop=True)
    numbers = frame[["close", "dividend_amount", "split_coefficient"]].to_numpy(float)
    if (
        frame.session.duplicated().any()
        or not np.isfinite(numbers).all()
        or (numbers[:, 0] <= 0).any()
        or (numbers[:, 1] < 0).any()
        or (numbers[:, 2] <= 0).any()
        or not np.isfinite([raw_close, split_coefficient, dividend_amount]).all()
        or raw_close <= 0
        or split_coefficient <= 0
        or dividend_amount < 0
    ):
        raise ValueError("Invalid completed history or current raw quote/actions")
    rules = indicator_rules() if rules is None else tuple(rules)
    if (
        not rules
        or len(set(rules)) != len(rules)
        or not all(isinstance(r, IndicatorRule) for r in rules)
    ):
        raise ValueError("Unique nonempty IndicatorRule collection required")
    state = _DailyState(rules)
    previous_close, previous_index = None, 100.0
    for row in frame.itertuples(index=False):
        finalized = (
            100.0
            if previous_close is None
            else previous_index
            * row.split_coefficient
            * (row.close + row.dividend_amount)
            / previous_close
        )
        if not np.isfinite(finalized) or finalized <= 0:
            raise ValueError("Total-return feature wealth is nonfinite or exhausted")
        state.finalize(finalized)
        previous_close, previous_index = row.close, finalized
    provisional = (
        previous_index * split_coefficient * (raw_close + dividend_amount) / previous_close
    )
    if not np.isfinite(provisional) or provisional <= 0:
        raise ValueError("Provisional total-return wealth is nonfinite or exhausted")
    result = pd.Series(
        state.preview(provisional)[0], index=[r.rule_id for r in rules], name="margin"
    )
    result.attrs.update(
        provisional_total_return_index=provisional,
        prior_session_count=state.count,
        last_completed_session=frame.session.iloc[-1],
    )
    return result


@dataclass(frozen=True)
class SimulationResult:
    metrics: dict[str, Any]
    daily_returns: pd.Series
    daily_equity: pd.Series
    trades: pd.DataFrame
    daily_cash_returns: pd.Series
    daily_exposure: pd.Series
    diagnostics: pd.DataFrame


def _segment(cache: SupportCache, start, end):
    start, end = pd.Timestamp(start).strftime("%Y-%m-%d"), pd.Timestamp(end).strftime("%Y-%m-%d")
    if start > end:
        raise ValueError("Segment start must not exceed end")
    positions = np.flatnonzero(cache.bars.session.between(start, end).to_numpy())
    scored = cache.daily.loc[cache.daily.session.between(start, end), "session"]
    if not len(positions) or not scored.isin(cache.bars.session.iloc[positions]).all():
        raise ValueError("Every scored session must have a complete intraday grid")
    return positions


def _costs(commission_bps, slippage_bps, initial_cash):
    if (
        not np.isfinite([commission_bps, slippage_bps, initial_cash]).all()
        or not 0 <= commission_bps < 10000
        or not 0 <= slippage_bps < 10000
        or initial_cash <= 0
    ):
        raise ValueError("Invalid nonnegative commission/slippage or initial cash")
    return commission_bps / 10000.0, slippage_bps / 10000.0


def _clock_accrual(frame: pd.DataFrame, observations: pd.DataFrame):
    # Unique endpoint union ensures release-at-endpoint behavior and zero-length
    # close/open boundaries without feeding duplicate times into rate_accrual.
    clock = pd.DatetimeIndex(sorted(set(frame.bar_start) | set(frame.bar_end)))
    accrued = rate_accrual(clock, observations, interval_start=clock[0], day_count=365.0)
    factors = accrued.to_numpy()
    if (factors <= -1).any():
        raise ValueError("Cash accrual exhausted account wealth")
    lookup = {timestamp: i for i, timestamp in enumerate(clock)}
    # There are no events between an earlier close and next open, or a bar's
    # open and close, under the complete-grid contract.
    starts = np.asarray([factors[lookup[x]] for x in frame.bar_start])
    ends = np.asarray([factors[lookup[x]] for x in frame.bar_end])
    # A touching open is the SAME endpoint as the preceding close: do not accrue twice.
    starts[1:][frame.bar_start.iloc[1:].to_numpy() == frame.bar_end.iloc[:-1].to_numpy()] = 0.0
    return starts, ends


def _simulate_batch(
    cache,
    positions,
    candidates,
    observations,
    commission_bps,
    slippage_bps,
    initial_cash,
    *,
    record=False,
    mode="candidate",
    clock_accrual=None,
):
    fee, slip = _costs(commission_bps, slippage_bps, initial_cash)
    frame = cache.bars.iloc[positions].reset_index(drop=True)
    if "series_id" in observations and not observations.series_id.eq("DTB3").all():
        raise ValueError("Cash observations must be DTB3")
    start_accrual, end_accrual = (
        _clock_accrual(frame, observations) if clock_accrual is None else clock_accrual
    )
    n = len(candidates)
    ids = [c.candidate_id for c in candidates]
    if mode != "candidate":
        ids = [f"benchmark_{mode}"]
    rule_lookup = {r: i for i, r in enumerate(cache.rules)}
    entry_cols = np.asarray([rule_lookup[c.entry_rule] for c in candidates])
    exit_cols = np.asarray([rule_lookup[c.exit_rule] for c in candidates])
    entry_n = np.asarray([c.entry_required_checks for c in candidates])
    exit_n = np.asarray([c.exit_required_checks for c in candidates])
    long = np.zeros(n, dtype=bool)
    pending = np.zeros(n, dtype=np.int8)  # +1 entry, -1 exit
    entry_count, exit_count = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    cash, gross_cash = np.full(n, initial_cash, dtype=float), np.full(n, initial_cash, dtype=float)
    shares, gross_shares = np.zeros(n), np.zeros(n)
    commission, slippage, turnover = np.zeros(n), np.zeros(n), np.zeros(n)
    trade_count, orders = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    held_seconds, total_holding, completed_holds = np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)
    entered_at = np.full(n, np.nan)
    pending_at = np.full(n, np.nan)
    trade_rows, diagnostics = [], []
    day_equity, day_gross, day_cash, day_exposure, day_timestamps = [], [], [], [], []
    cash_benchmark = initial_cash
    actions = cache.daily.set_index("session")
    schedule = cache.calendar.set_index("session")
    split_values = frame.session.map(actions.split_coefficient).to_numpy(float)
    dividend_values = frame.session.map(actions.dividend_amount).to_numpy(float)
    closes_session = frame.bar_end.eq(frame.session.map(schedule.market_close)).to_numpy()
    previous_event = frame.bar_start.iloc[0]
    first_session = None
    cancelled_terminal_orders = np.zeros(n, dtype=int)

    def accrue(timestamp, factor):
        nonlocal cash, gross_cash, cash_benchmark, previous_event
        elapsed = (timestamp - previous_event).total_seconds()
        held_seconds[:] += long * elapsed
        cash *= 1 + factor
        gross_cash *= 1 + factor
        cash_benchmark *= 1 + factor
        previous_event = timestamp

    def fill(which, side, price, timestamp, session, reason):
        nonlocal cash, shares, gross_cash, gross_shares
        if not which.any():
            return
        equity = cash + shares * price
        execution = price * (1 + slip if side == "buy" else 1 - slip)
        if side == "buy":
            qty = cash[which] / (execution * (1 + fee))
            notional = qty * price
            charged = qty * execution * fee
            cash[which] = 0.0
            shares[which] = qty
            gross_shares[which] = gross_cash[which] / price
            gross_cash[which] = 0.0
            entered_at[which] = timestamp.value / 1e9
            holding = np.zeros(which.sum())
            long[which] = True
        else:
            qty = shares[which].copy()
            notional = qty * price
            charged = qty * execution * fee
            cash[which] += qty * execution - charged
            shares[which] = 0.0
            gross_cash[which] += gross_shares[which] * price
            gross_shares[which] = 0.0
            holding = (timestamp.value / 1e9 - entered_at[which]) / 3600
            total_holding[which] += holding
            completed_holds[which] += 1
            trade_count[which] += 1
            long[which] = False
        adverse = qty * abs(execution - price)
        commission[which] += charged
        slippage[which] += adverse
        turnover[which] += notional / equity[which]
        orders[which] += 1
        if record:
            for local, j in enumerate(np.flatnonzero(which)):
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
                        quantity=qty[local],
                        reference_notional=notional[local],
                        commission=charged[local],
                        slippage=adverse[local],
                        equity_before=equity[j],
                        holding_hours=holding[local],
                    )
                )
        pending[which], entry_count[which], exit_count[which] = 0, 0, 0
        pending_at[which] = np.nan

    for k, row in enumerate(frame.itertuples(index=False)):
        accrue(row.bar_start, start_accrual[k])
        if row.session != first_session:
            shares *= split_values[k]
            gross_shares *= split_values[k]
            # Open-price DRIP for pre-open holders; never also add an adjusted return.
            shares *= 1 + dividend_values[k] / row.open
            gross_shares *= 1 + dividend_values[k] / row.open
            first_session = row.session
        if mode == "buy_and_hold" and k == 0:
            fill(
                np.ones(n, dtype=bool),
                "buy",
                row.open,
                row.bar_start,
                row.session,
                "benchmark_entry",
            )
        else:
            fill(pending == -1, "sell", row.open, row.bar_start, row.session, "confirmed_exit")
            fill(pending == 1, "buy", row.open, row.bar_start, row.session, "confirmed_entry")
        accrue(row.bar_end, end_accrual[k])
        if mode == "candidate":
            margin = cache.margins[positions[k]]
            entry_ok = margin[entry_cols] > 0
            exit_valid = np.isfinite(margin[exit_cols])
            exit_ok = margin[exit_cols] > 0
            eligible_entry = entry_ok & exit_ok
            entry_count[:] = np.where(~long & eligible_entry, entry_count + 1, 0)
            exit_count[:] = np.where(long & exit_valid & ~exit_ok, exit_count + 1, 0)
            buys = ~long & (entry_count >= entry_n)
            sells = long & (exit_count >= exit_n)
            pending[buys], pending[sells] = 1, -1
            pending_at[buys | sells] = row.bar_end.value / 1e9
            if record:
                diagnostics.append(
                    dict(
                        timestamp=row.bar_end,
                        session=row.session,
                        entry_support=bool(entry_ok[0]),
                        exit_support=bool(exit_ok[0]),
                        entry_count=int(entry_count[0]),
                        exit_count=int(exit_count[0]),
                        position=int(long[0]),
                        pending=int(pending[0]),
                    )
                )
        if closes_session[k]:
            terminal = k == len(frame) - 1
            closing_exposure = long.astype(float).copy()
            if terminal:
                cancelled_terminal_orders[:] = pending != 0
                if record and diagnostics:
                    diagnostics[-1]["terminal_order_cancelled"] = bool(pending[0])
                pending[:] = 0
                pending_at[:] = np.nan
                # Final-close liquidation is included in the last scored daily return.
                fill(
                    long.copy(), "sell", row.close, row.bar_end, row.session, "terminal_liquidation"
                )
            day_equity.append((cash + shares * row.close).copy())
            day_gross.append((gross_cash + gross_shares * row.close).copy())
            day_cash.append(cash_benchmark)
            day_exposure.append(closing_exposure)
            day_timestamps.append(row.bar_end)
    times = pd.DatetimeIndex(day_timestamps, name="timestamp")
    equity, gross = np.asarray(day_equity), np.asarray(day_gross)
    returns = equity / np.vstack([np.full(n, initial_cash), equity[:-1]]) - 1
    cash_curve = np.asarray(day_cash)
    cash_returns = cash_curve / np.r_[initial_cash, cash_curve[:-1]] - 1
    total_seconds = (previous_event - frame.bar_start.iloc[0]).total_seconds()
    years = total_seconds / (365.0 * 86400.0)
    metrics = []
    for j, candidate in enumerate(candidates):
        daily = returns[:, j]
        excess = daily - cash_returns
        sd = np.std(excess, ddof=1) if len(excess) > 1 else np.nan
        sharpe = float(np.sqrt(252) * np.mean(excess) / sd) if sd > 0 else np.nan
        wealth = np.r_[initial_cash, equity[:, j]]
        drawdown = wealth / np.maximum.accumulate(wealth) - 1
        terminal = equity[-1, j]
        stats = dict(
            candidate_id=ids[j],
            entry_family=candidate.entry_rule.family,
            exit_family=candidate.exit_rule.family,
            family_cell=candidate.family_cell,
            entry_rule=candidate.entry_rule.rule_id,
            exit_rule=candidate.exit_rule.rule_id,
            entry_confirmation_hours=candidate.entry_confirmation_hours,
            exit_confirmation_hours=candidate.exit_confirmation_hours,
            entry_required_checks=candidate.entry_required_checks,
            exit_required_checks=candidate.exit_required_checks,
            cancelled_terminal_order_count=int(cancelled_terminal_orders[j]),
            sampling_minutes=SAMPLING_MINUTES,
            cash_excess_sharpe=sharpe,
            cash_excess_Sharpe=sharpe,
            cagr=float((terminal / initial_cash) ** (1 / years) - 1) if years > 0 else np.nan,
            max_drawdown=float(drawdown.min()),
            annualized_volatility=float(np.std(daily, ddof=1) * np.sqrt(252))
            if len(daily) > 1
            else np.nan,
            exposure_fraction=float(held_seconds[j] / total_seconds) if total_seconds else 0.0,
            total_turnover=float(turnover[j]),
            annualized_turnover=float(turnover[j] / years) if years > 0 else np.nan,
            trade_count=int(trade_count[j]),
            order_count=int(orders[j]),
            average_holding_hours=float(total_holding[j] / completed_holds[j])
            if completed_holds[j]
            else 0.0,
            total_commission=float(commission[j]),
            total_slippage=float(slippage[j]),
            cost_drag_return=float((gross[-1, j] - terminal) / initial_cash),
            total_return=float(terminal / initial_cash - 1),
            terminal_equity=float(terminal),
            gross_terminal_equity=float(gross[-1, j]),
            cash_terminal_equity=float(cash_curve[-1]),
            scored_sessions=len(times),
            start_session=frame.session.iloc[0],
            end_session=frame.session.iloc[-1],
            commission_bps=float(commission_bps),
            slippage_bps=float(slippage_bps),
        )
        metrics.append(stats)
    return (
        metrics,
        times,
        equity,
        returns,
        cash_returns,
        np.asarray(day_exposure),
        trade_rows,
        diagnostics,
    )


def simulate_candidate(
    daily: pd.DataFrame,
    bars: pd.DataFrame,
    candidate: Candidate,
    *,
    start,
    end,
    cash_observations: pd.DataFrame,
    commission_bps: float = 1,
    slippage_bps: float = 3,
    initial_cash: float = 1000,
    calendar: pd.DataFrame | None = None,
    support_cache: SupportCache | None = None,
) -> SimulationResult:
    """Simulate a single segment with a fresh cash account and empty counters."""
    cache = support_cache or build_support_cache(
        daily,
        bars,
        calendar,
        rules=tuple(dict.fromkeys((*indicator_rules(), candidate.entry_rule, candidate.exit_rule))),
    )
    positions = _segment(cache, start, end)
    metrics, times, equity, returns, cash_returns, exposure, trades, diag = _simulate_batch(
        cache,
        positions,
        [candidate],
        cash_observations,
        commission_bps,
        slippage_bps,
        initial_cash,
        record=True,
    )
    return SimulationResult(
        metrics[0],
        pd.Series(returns[:, 0], index=times, name="strategy_return"),
        pd.Series(equity[:, 0], index=times, name="equity"),
        pd.DataFrame(trades, columns=TRADE_COLUMNS),
        pd.Series(cash_returns, index=times, name="cash_return"),
        pd.Series(exposure[:, 0], index=times, name="close_exposure"),
        pd.DataFrame(diag),
    )


def evaluate_grid(
    daily: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    start,
    end,
    cash_observations: pd.DataFrame,
    candidates: list[Candidate] | None = None,
    commission_bps: float = 1,
    slippage_bps: float = 3,
    initial_cash: float = 1000,
    calendar: pd.DataFrame | None = None,
    batch_size: int = 256,
    support_cache: SupportCache | None = None,
) -> pd.DataFrame:
    """Evaluate an explicit grid in bounded-memory vectorized ledger batches.

    Daily indicators/supports are built once and reused. Only current/past
    observations affect signals, despite vectorized execution. This function
    does not select candidates, read files, or evaluate another split implicitly.
    """
    candidates = generate_candidates() if candidates is None else list(candidates)
    if not candidates or len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("A nonempty unique candidate list is required")
    if not _positive_integer(batch_size):
        raise ValueError("batch_size must be a positive integer")
    rules = tuple(
        dict.fromkeys(
            (*indicator_rules(), *(r for c in candidates for r in (c.entry_rule, c.exit_rule)))
        )
    )
    cache = support_cache or build_support_cache(daily, bars, calendar, rules=rules)
    positions = _segment(cache, start, end)
    rows = []
    clock_accrual = _clock_accrual(cache.bars.iloc[positions], cash_observations)
    for offset in range(0, len(candidates), batch_size):
        rows.extend(
            _simulate_batch(
                cache,
                positions,
                candidates[offset : offset + batch_size],
                cash_observations,
                commission_bps,
                slippage_bps,
                initial_cash,
                clock_accrual=clock_accrual,
            )[0]
        )
    return pd.DataFrame(rows)


def simulate_benchmarks(
    daily: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    start,
    end,
    cash_observations: pd.DataFrame,
    commission_bps: float = 1,
    slippage_bps: float = 3,
    initial_cash: float = 1000,
    calendar: pd.DataFrame | None = None,
    support_cache: SupportCache | None = None,
) -> dict[str, SimulationResult]:
    """Cash and QQQ buy-and-hold with identical clock/action/terminal conventions."""
    cache = support_cache or build_support_cache(daily, bars, calendar)
    positions = _segment(cache, start, end)
    dummy = Candidate(cache.rules[0], cache.rules[0], 0, 0)
    out = {}
    clock_accrual = _clock_accrual(cache.bars.iloc[positions], cash_observations)
    for mode in ("cash", "buy_and_hold"):
        metrics, times, equity, returns, cash_returns, exposure, trades, diag = _simulate_batch(
            cache,
            positions,
            [dummy],
            cash_observations,
            commission_bps,
            slippage_bps,
            initial_cash,
            record=True,
            mode=mode,
            clock_accrual=clock_accrual,
        )
        out[mode] = SimulationResult(
            metrics[0],
            pd.Series(returns[:, 0], index=times, name="strategy_return"),
            pd.Series(equity[:, 0], index=times, name="equity"),
            pd.DataFrame(trades, columns=TRADE_COLUMNS),
            pd.Series(cash_returns, index=times, name="cash_return"),
            pd.Series(exposure[:, 0], index=times, name="close_exposure"),
            pd.DataFrame(diag),
        )
    return out


def shortlist_family_cells(metrics: pd.DataFrame, *, tolerance: float = 1e-10) -> pd.DataFrame:
    """One development winner per cell: finite Sharpe, then stable candidate ID.

    Sharpe differences no larger than ``tolerance`` are ties. No CAGR, drawdown,
    turnover, or another split is a secondary selection objective.
    """
    required = {"entry_family", "exit_family", "cash_excess_sharpe", "candidate_id"}
    if not required.issubset(metrics) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Missing shortlist columns or invalid tie tolerance")
    eligible = metrics[np.isfinite(metrics.cash_excess_sharpe)].copy()
    winners = []
    for _, cell in eligible.groupby(["entry_family", "exit_family"], sort=True):
        best = cell.cash_excess_sharpe.max()
        winners.append(
            cell.loc[cell.cash_excess_sharpe >= best - tolerance]
            .sort_values("candidate_id", kind="stable")
            .iloc[0]
        )
    return pd.DataFrame(winners, columns=metrics.columns).reset_index(drop=True)
