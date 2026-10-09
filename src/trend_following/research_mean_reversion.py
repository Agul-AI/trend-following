"""Independent daily mean-reversion features and the fixed 90-candidate grid.

This module opens no market or result files. Completed half-hour observations
preview one provisional daily value from yesterday's finalized state. Daily
history advances only in ``finalize``. Raw trade prices never replace the
forward-built corporate-action-adjusted signal index.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from trend_following import research_quality_engine_v5 as quality

FAMILIES = ("SMA_DISTANCE", "EMA_DISTANCE", "ZSCORE", "RSI")
DISTANCE_PERIODS = (10, 20, 50)
RSI_PERIODS = (2, 5, 14)
CONFIRMATION_HOURS = 3.0
REQUIRED_CHECKS = 7
DAILY_INDICATOR_UNITS = "sessions"


def _positive_integer(value: Any) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value > 0
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )


@dataclass(frozen=True)
class Rule:
    family: str
    period: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", str(self.family).upper())
        if self.family not in FAMILIES or not _positive_integer(self.period):
            raise ValueError(
                "Mean-reversion rule requires a known family and positive daily period"
            )
        object.__setattr__(self, "period", int(self.period))

    @property
    def rule_id(self) -> str:
        return f"{self.family.lower()}_{self.period}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Rule:
        return cls(**value)


def _threshold_id(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Candidate:
    """One family/period, oversold entry, recovery exit and one fixed wait."""

    rule: Rule
    entry_threshold: float
    exit_threshold: float
    confirmation_hours: float = CONFIRMATION_HOURS

    def __post_init__(self) -> None:
        if not isinstance(self.rule, Rule):
            raise ValueError("Candidate rule must be a mean-reversion Rule")
        if (
            not _finite_number(self.confirmation_hours)
            or float(self.confirmation_hours) != CONFIRMATION_HOURS
        ):
            raise ValueError("The shared confirmation wait is fixed at 3 trading hours")
        if (
            not _finite_number(self.entry_threshold)
            or not _finite_number(self.exit_threshold)
            or self.entry_threshold >= self.exit_threshold
        ):
            raise ValueError("Finite entry threshold must be below the finite exit threshold")
        for field in ("entry_threshold", "exit_threshold", "confirmation_hours"):
            value = float(getattr(self, field))
            object.__setattr__(self, field, 0.0 if value == 0 else value)

    @property
    def candidate_id(self) -> str:
        return (
            f"mr_30m__{self.rule.rule_id}__entry_{_threshold_id(self.entry_threshold)}"
            f"__exit_{_threshold_id(self.exit_threshold)}__shared_hours_3"
        )

    @property
    def family(self) -> str:
        return self.rule.family

    @property
    def family_cell(self) -> str:
        return self.family

    @property
    def required_checks(self) -> int:
        return REQUIRED_CHECKS

    @property
    def entry_required_checks(self) -> int:
        return self.required_checks

    @property
    def exit_required_checks(self) -> int:
        return self.required_checks

    def entry_support(self, value: float) -> bool:
        return bool(np.isfinite(value) and value <= self.entry_threshold)

    def exit_support(self, value: float) -> bool:
        return bool(np.isfinite(value) and value >= self.exit_threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule.to_dict(),
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "confirmation_hours": self.confirmation_hours,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Candidate:
        if set(value) != {"rule", "entry_threshold", "exit_threshold", "confirmation_hours"}:
            raise ValueError(
                "Candidate schema requires one rule, thresholds and explicit shared wait"
            )
        return cls(
            Rule.from_dict(value["rule"]),
            value["entry_threshold"],
            value["exit_threshold"],
            value["confirmation_hours"],
        )


def feature_rules() -> tuple[Rule, ...]:
    return tuple(
        Rule(family, period)
        for family in FAMILIES
        for period in (RSI_PERIODS if family == "RSI" else DISTANCE_PERIODS)
    )


def candidate_grid() -> list[Candidate]:
    thresholds = {
        "SMA_DISTANCE": ((-0.02, -0.04, -0.06), (0.0, 0.01)),
        "EMA_DISTANCE": ((-0.02, -0.04, -0.06), (0.0, 0.01)),
        "ZSCORE": ((-1.5, -2.0, -2.5), (-0.5, 0.0, 0.5)),
        "RSI": ((10.0, 20.0, 30.0), (50.0, 60.0, 70.0)),
    }
    return [
        Candidate(rule, entry, exit_threshold)
        for rule in feature_rules()
        for entry in thresholds[rule.family][0]
        for exit_threshold in thresholds[rule.family][1]
    ]


class DailyState:
    """Pure daily previews; EMA/RSI recursion advances once per finalized day."""

    def __init__(self, rules: tuple[Rule, ...] | None = None):
        self.rules = feature_rules() if rules is None else tuple(rules)
        if (
            not self.rules
            or not all(isinstance(rule, Rule) for rule in self.rules)
            or len(set(self.rules)) != len(self.rules)
        ):
            raise ValueError("Unique nonempty mean-reversion rules are required")
        self.count = 0
        self.history: deque[float] = deque(maxlen=max(rule.period for rule in self.rules))
        self.ema = dict.fromkeys(
            rule.period for rule in self.rules if rule.family == "EMA_DISTANCE"
        )
        self.rsi_gain = dict.fromkeys(
            (rule.period for rule in self.rules if rule.family == "RSI"), 0.0
        )
        self.rsi_loss = self.rsi_gain.copy()
        self.previous: float | None = None

    def _transition(self, value: float) -> tuple[dict[str, float], dict, dict, dict]:
        if not _finite_number(value) or value <= 0:
            raise ValueError("Daily signal value must be finite and positive")
        value = float(value)
        count = self.count + 1
        ema = {
            period: value if old is None else old + 2 / (period + 1) * (value - old)
            for period, old in self.ema.items()
        }
        gains, losses = self.rsi_gain.copy(), self.rsi_loss.copy()
        if self.previous is not None:
            change = value - self.previous
            gain, loss = max(change, 0.0), max(-change, 0.0)
            # Before N changes, store running sums; on change N convert to the
            # initial average, then use Wilder's 1/N recursive update.
            prior_changes = self.count - 1
            for period in gains:
                if prior_changes < period:
                    gains[period] += gain
                    losses[period] += loss
                    if prior_changes + 1 == period:
                        gains[period] /= period
                        losses[period] /= period
                else:
                    gains[period] += (gain - gains[period]) / period
                    losses[period] += (loss - losses[period]) / period
        values = {}
        history = list(self.history)
        for rule in self.rules:
            period = rule.period
            result = np.nan
            if rule.family == "RSI":
                if self.previous is not None and self.count >= period:
                    avg_gain, avg_loss = gains[period], losses[period]
                    if avg_gain == 0.0 and avg_loss == 0.0:
                        result = 50.0
                    elif avg_loss == 0.0:
                        result = 100.0
                    elif avg_gain == 0.0:
                        result = 0.0
                    else:
                        result = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            elif count >= period:
                if rule.family == "EMA_DISTANCE":
                    result = value / ema[period] - 1.0
                else:
                    window = np.asarray(
                        ([] if period == 1 else history[-(period - 1) :]) + [value],
                        dtype=float,
                    )
                    # Center around the provisional value before averaging.
                    # This keeps constant nonbinary prices exactly flat (e.g.
                    # 100.2): averaging absolute prices can manufacture a tiny
                    # deviation/variance and break an equality-at-zero exit.
                    offsets = window - value
                    mean_offset = float(np.mean(offsets))
                    mean = value + mean_offset
                    difference = -mean_offset
                    if rule.family == "SMA_DISTANCE":
                        result = difference / mean
                    elif period > 1 and np.ptp(offsets) > 0.0:
                        std = float(np.std(offsets, ddof=1))
                        if std > 0.0:
                            result = difference / std
            values[rule.rule_id] = float(result)
        return values, ema, gains, losses

    def preview(self, value: float) -> dict[str, float]:
        """Return features without mutating finalized counts/history/recursion."""
        return self._transition(value)[0]

    def finalize(self, value: float) -> None:
        _, self.ema, self.rsi_gain, self.rsi_loss = self._transition(value)
        self.history.append(float(value))
        self.previous = float(value)
        self.count += 1


@dataclass(frozen=True)
class MeanReversionCache(quality.QualitySupportCache):
    """Quality-validated cache; ``margins`` contains raw MR indicator values.

    The inherited field names permit read-only reuse of frozen event/accounting
    validation utilities, not reuse of the trend-following signal predicates.
    """

    rules: tuple[Rule, ...]


def build_feature_cache(
    daily: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    coverage: pd.DataFrame | None = None,
    blackouts: pd.DataFrame | None = None,
    quality_policy: quality.QualityPolicy = quality.DEFAULT_POLICY,
    rules: tuple[Rule, ...] | None = None,
) -> MeanReversionCache:
    """Assemble causal MR features from supplied, stage-bounded raw frames.

    The provider daily quote is assumed available at 17:00 New York. Same-day
    intraday previews use only prior completed daily state plus one provisional
    observation; the session's final daily quote affects subsequent sessions.
    Known ex-open splits/dividends update the signal index forward, never by a
    retrospectively adjusted full-history series.
    """
    daily, bars, calendar, coverage, blackouts = quality._prepare(
        daily, bars, calendar, coverage, blackouts, quality_policy
    )
    state = DailyState(rules)
    rules = state.rules
    eligibility = coverage.set_index("bar_start").loc[bars.bar_start]
    signal = eligibility.signal_eligible.to_numpy(bool)
    fill = eligibility.fill_eligible.to_numpy(bool)
    margins = np.full((len(bars), len(rules)), np.nan)
    preview = np.full(len(bars), np.nan)
    counts = np.zeros(len(bars), dtype=int)
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
                    values = state.preview(value)
                    margins[i] = [values[rule.rule_id] for rule in rules]
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
        values = state.preview(value)
        matured.append([values[rule.rule_id] for rule in rules])
        state.finalize(value)
        finalized.append(value)
        prior_price, prior_index = row.close, value
    index = pd.Index(daily.session, name="session")
    cache = MeanReversionCache(
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
        pd.DataFrame(matured, index=index, columns=[rule.rule_id for rule in rules]),
        counts,
        signal,
        fill,
        tuple(bars.itertuples(index=False)),
    )
    object.__setattr__(
        cache,
        "input_fingerprint",
        quality._input_fingerprint(daily, bars, calendar, coverage, blackouts, quality_policy),
    )
    object.__setattr__(cache, "feature_fingerprint", quality._feature_fingerprint(cache))
    return cache
