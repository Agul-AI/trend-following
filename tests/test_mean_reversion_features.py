"""Synthetic-only independent mean-reversion grid and causal feature checks."""

import json
from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from trend_following import research_mean_reversion as mr
from trend_following import research_quality_engine_v5 as quality


def packet(specs, history=(100.0, 101.0, 99.0, 102.0, 101.0, 100.0)):
    daily = [
        dict(
            session=date.strftime("%Y-%m-%d"),
            close=price,
            dividend_amount=0.0,
            split_coefficient=1.0,
        )
        for date, price in zip(
            pd.bdate_range("1999-11-01", periods=len(history)), history, strict=True
        )
    ]
    bars, calendar, coverage = [], [], []
    for i, spec in enumerate(specs):
        session = (pd.Timestamp("2000-01-03") + pd.offsets.BDay(i)).strftime("%Y-%m-%d")
        opening = pd.Timestamp(f"{session} 09:30", tz="America/New_York").tz_convert("UTC")
        closes = spec["closes"]
        missing = set(spec.get("missing", []))
        for k, close in enumerate(closes):
            start = opening + pd.Timedelta(minutes=30 * k)
            end = start + pd.Timedelta(minutes=30)
            coverage.append(
                dict(
                    session=session,
                    bar_start=start,
                    bar_end=end,
                    status="unknown_missing" if k in missing else "observed",
                    signal_eligible=k not in missing,
                    fill_eligible=k not in missing,
                )
            )
            if k not in missing:
                bars.append(
                    dict(
                        session=session,
                        bar_start=start,
                        bar_end=end,
                        open=close,
                        high=close,
                        low=close,
                        close=close,
                        volume=100.0,
                        duration_minutes=30.0,
                        is_completed=True,
                        is_full_hour=False,
                    )
                )
        calendar.append(
            dict(
                session=session,
                market_open=opening,
                market_close=opening + pd.Timedelta(minutes=30 * len(closes)),
            )
        )
        daily.append(
            dict(
                session=session,
                close=spec.get("daily_close", closes[-1]),
                dividend_amount=spec.get("dividend", 0.0),
                split_coefficient=spec.get("split", 1.0),
            )
        )
    daily = pd.DataFrame(daily)
    daily["daily_price_available_at"] = [
        pd.Timestamp(f"{session} 17:00", tz="America/New_York").tz_convert("UTC")
        for session in daily.session
    ]
    return daily, pd.DataFrame(bars), pd.DataFrame(calendar), pd.DataFrame(coverage)


def build(data, rules=None, *, observed_only=False):
    daily, bars, calendar, coverage = data
    return mr.build_feature_cache(
        daily,
        bars,
        calendar,
        coverage=coverage,
        rules=rules,
        quality_policy=quality.QualityPolicy(
            "observed_only_blackout_v1" if observed_only else "strict_tradable_v1"
        ),
    )


def reference_rsi(values, period):
    differences = np.diff(np.asarray(values, dtype=float))
    if len(differences) < period:
        return np.nan
    gains = np.maximum(differences, 0.0)
    losses = np.maximum(-differences, 0.0)
    average_gain = float(np.mean(gains[:period]))
    average_loss = float(np.mean(losses[:period]))
    for gain, loss in zip(gains[period:], losses[period:], strict=True):
        average_gain = ((period - 1) * average_gain + gain) / period
        average_loss = ((period - 1) * average_loss + loss) / period
    if average_gain == average_loss == 0:
        return 50.0
    if average_loss == 0:
        return 100.0
    if average_gain == 0:
        return 0.0
    return 100 * average_gain / (average_gain + average_loss)


def reference_value(rule, values):
    if rule.family == "RSI":
        return reference_rsi(values, rule.period)
    if len(values) < rule.period:
        return np.nan
    if rule.family == "EMA_DISTANCE":
        average = pd.Series(values).ewm(span=rule.period, adjust=False).mean().iloc[-1]
        return values[-1] / average - 1
    window = values[-rule.period :]
    average = sum(window) / rule.period
    if rule.family == "SMA_DISTANCE":
        return values[-1] / average - 1
    variance = sum((value - average) ** 2 for value in window) / (rule.period - 1)
    return np.nan if variance == 0 else (values[-1] - average) / np.sqrt(variance)


def test_exact_90_immutable_unique_grid_daily_units_shared_3h_and_serialization():
    candidates = mr.candidate_grid()
    assert len(candidates) == len({candidate.candidate_id for candidate in candidates}) == 90
    assert Counter(candidate.family for candidate in candidates) == {
        "SMA_DISTANCE": 18,
        "EMA_DISTANCE": 18,
        "ZSCORE": 27,
        "RSI": 27,
    }
    assert len(mr.feature_rules()) == 12
    assert mr.DAILY_INDICATOR_UNITS == "sessions"
    assert all(candidate.confirmation_hours == 3.0 for candidate in candidates)
    assert all(
        candidate.entry_required_checks == candidate.exit_required_checks == 7
        for candidate in candidates
    )
    assert mr.candidate_grid() == candidates
    assert mr.Candidate.from_dict(json.loads(json.dumps(candidates[0].to_dict()))) == candidates[0]
    assert mr.Rule("rsi", np.int64(14)).to_dict() == {"family": "RSI", "period": 14}
    assert all(
        candidate.rule.period in (2, 5, 14) for candidate in candidates if candidate.family == "RSI"
    )
    assert all(
        candidate.rule.period in (10, 20, 50)
        for candidate in candidates
        if candidate.family != "RSI"
    )
    with pytest.raises(FrozenInstanceError):
        candidates[0].confirmation_hours = 2.0


@pytest.mark.parametrize("bad", [True, np.bool_(False), np.nan, np.inf, -1, 0, 2.0, "2", None])
def test_rule_rejects_invalid_daily_periods(bad):
    with pytest.raises(ValueError, match="positive daily period"):
        mr.Rule("SMA_DISTANCE", bad)


@pytest.mark.parametrize("bad", [True, np.bool_(False), np.nan, np.inf, 0.0, 2.0, 4.0, "3", None])
def test_wait_is_one_explicit_3h_policy(bad):
    with pytest.raises(ValueError, match="fixed at 3"):
        mr.Candidate(mr.Rule("SMA_DISTANCE", 10), -0.02, 0.0, bad)


def test_invalid_rule_threshold_schema_and_equality_support():
    with pytest.raises(ValueError, match="known family"):
        mr.Rule("MACD", 10)
    with pytest.raises(ValueError, match="Finite entry threshold"):
        mr.Candidate(mr.Rule("RSI", 2), 20, 20)
    with pytest.raises(ValueError, match="Finite entry threshold"):
        mr.Candidate(mr.Rule("RSI", 2), False, 20)
    with pytest.raises(ValueError, match="explicit shared wait"):
        mr.Candidate.from_dict({"rule": {"family": "RSI", "period": 2}})
    candidate = mr.Candidate(mr.Rule("SMA_DISTANCE", 10), -0.02, 0.0)
    assert candidate.entry_support(-0.02)
    assert candidate.exit_support(0.0)
    assert not candidate.entry_support(-0.019)
    assert not candidate.exit_support(-0.001)
    assert not candidate.entry_support(np.nan)
    assert not candidate.exit_support(np.inf)


@pytest.mark.parametrize("rule", mr.feature_rules())
def test_final_and_provisional_values_match_independent_reference(rule):
    values = [100.0 + 0.13 * i + 4.0 * np.sin(i / 2.1) for i in range(90)]
    state = mr.DailyState((rule,))
    for i, value in enumerate(values):
        before = deepcopy(state.__dict__)
        result = state.preview(value)[rule.rule_id]
        expected = reference_value(rule, values[: i + 1])
        assert result == pytest.approx(expected) if np.isfinite(expected) else np.isnan(result)
        assert state.__dict__ == before
        state.finalize(value)
    assert state.count == 90


def test_ema_and_rsi_repeated_previews_never_advance_finalized_state():
    rules = (mr.Rule("EMA_DISTANCE", 3), mr.Rule("RSI", 2))
    state = mr.DailyState(rules)
    for value in (100.0, 105.0, 102.0):
        state.finalize(value)
    original = deepcopy(state.__dict__)
    first = state.preview(99.0)
    for value in (110.0, 115.0, 90.0, 101.0, 99.0):
        state.preview(value)
    assert state.preview(99.0) == first
    assert state.__dict__ == original
    assert first["ema_distance_3"] == pytest.approx(reference_value(rules[0], [100, 105, 102, 99]))
    assert first["rsi_2"] == pytest.approx(reference_rsi([100, 105, 102, 99], 2))
    state.finalize(101.0)
    assert state.count == 4
    assert state.preview(98)["rsi_2"] == pytest.approx(reference_rsi([100, 105, 102, 101, 98], 2))


@pytest.mark.parametrize(
    "values,expected",
    [([100, 101, 102, 103], 100.0), ([103, 102, 101, 100], 0.0), ([100] * 4, 50.0)],
)
def test_rsi_exact_zero_cases_and_n_plus_one_initialization(values, expected):
    state = mr.DailyState((mr.Rule("RSI", 3),))
    for value in values[:3]:
        assert np.isnan(state.preview(value)["rsi_3"])
        state.finalize(value)
    assert state.preview(values[-1])["rsi_3"] == expected


def test_sma_ema_mature_on_n_and_zscore_uses_sample_not_population_std():
    rules = tuple(mr.Rule(family, 3) for family in ("SMA_DISTANCE", "EMA_DISTANCE", "ZSCORE"))
    state = mr.DailyState(rules)
    for value in (100.0, 102.0):
        assert all(np.isnan(result) for result in state.preview(value).values())
        state.finalize(value)
    values = state.preview(98.0)
    assert values["sma_distance_3"] == pytest.approx(-0.02)
    assert values["zscore_3"] == pytest.approx(-1.0)
    flat = mr.DailyState((mr.Rule("ZSCORE", 3),))
    flat.finalize(100)
    flat.finalize(100)
    assert np.isnan(flat.preview(100)["zscore_3"])


@pytest.mark.parametrize("rule", mr.feature_rules())
def test_constant_nonbinary_price_has_exact_zero_ma_and_undefined_zscore(rule):
    state = mr.DailyState((rule,))
    for _ in range(60):
        state.finalize(100.2)
    value = state.preview(100.2)[rule.rule_id]
    if rule.family == "ZSCORE":
        assert np.isnan(value)
    elif rule.family == "RSI":
        assert value == 50.0
    else:
        assert value == 0.0
        assert mr.Candidate(rule, -0.02, 0.0).exit_support(value)


@pytest.mark.parametrize("bad", [True, np.bool_(False), np.nan, np.inf, 0.0, -2.0, "100", None])
def test_signal_state_rejects_invalid_values_without_advancing(bad):
    state = mr.DailyState()
    with pytest.raises(ValueError, match="finite and positive"):
        state.finalize(bad)
    assert state.count == 0


def test_cache_forward_split_dividend_index_and_daily_state_count():
    data = packet(
        [
            {"closes": [49.0, 50.0, 51.0], "split": 2.0, "dividend": 1.0},
            {"closes": [50.0, 51.0, 52.0], "dividend": 0.5},
        ],
        history=(100.0,),
    )
    before = tuple(frame.copy(deep=True) for frame in data)
    cache = build(data, (mr.Rule("SMA_DISTANCE", 2), mr.Rule("RSI", 2)))
    assert isinstance(cache, quality.QualitySupportCache)
    np.testing.assert_allclose(cache.provisional_total_return_index[:3], [100, 102, 104])
    np.testing.assert_allclose(
        cache.provisional_total_return_index[3:], 104 * np.array([50.5, 51.5, 52.5]) / 51
    )
    np.testing.assert_array_equal(cache.prior_session_counts, [1, 1, 1, 2, 2, 2])
    assert cache.finalized_total_return_index.iloc[1] == pytest.approx(104)
    assert cache.finalized_total_return_index.iloc[2] == pytest.approx(104 * 52.5 / 51)
    assert cache.margins[1, 0] == pytest.approx(102 / 101 - 1)
    assert cache.margins[4, 1] == pytest.approx(reference_rsi([100, 104, 104 * 51.5 / 51], 2))
    for original, after in zip(before, data, strict=True):
        pd.testing.assert_frame_equal(original, after)
    quality._validate_runtime_cache(cache, *data[:3], data[3], None, cache.policy)


def test_future_quotes_and_today_final_quote_do_not_change_earlier_previews():
    rules = (mr.Rule("EMA_DISTANCE", 3), mr.Rule("RSI", 2), mr.Rule("ZSCORE", 3))
    data = packet([{"closes": [99, 98, 97]}, {"closes": [103, 104, 105]}])
    cache = build(data, rules)
    changed = tuple(frame.copy(deep=True) for frame in data)
    changed[0].loc[changed[0].session.eq("2000-01-03"), "close"] = 90.0
    last = changed[1].session.eq("2000-01-04")
    changed[1].loc[last, ["open", "high", "low", "close"]] *= 2
    changed[0].loc[changed[0].session.eq("2000-01-04"), "close"] = 210.0
    altered = build(changed, rules)
    np.testing.assert_array_equal(cache.margins[:3], altered.margins[:3])
    np.testing.assert_array_equal(
        cache.provisional_total_return_index[:3], altered.provisional_total_return_index[:3]
    )
    assert not np.array_equal(cache.margins[3:], altered.margins[3:])
    # Prefix-only recomputation produces identical previous-session values.
    prefix = (data[0].iloc[:-1], data[1].iloc[:3], data[2].iloc[:1], data[3].iloc[:3])
    np.testing.assert_array_equal(cache.margins[:3], build(prefix, rules).margins)


def test_missing_halfhours_not_invented_and_17_clock_is_required():
    data = packet([{"closes": [99, 98, 97], "missing": [1]}])
    cache = build(data, (mr.Rule("EMA_DISTANCE", 3),), observed_only=True)
    assert len(cache.bars) == len(cache.margins) == 2
    assert len(cache.coverage) == 3
    assert len(cache.blackouts) == 1
    bad = tuple(frame.copy(deep=True) for frame in data)
    bad[0]["daily_price_available_at"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="17:00"):
        build(bad, observed_only=True)


def test_unfinished_bar_and_changed_cache_are_rejected():
    data = packet([{"closes": [99, 98, 97]}])
    bad = tuple(frame.copy(deep=True) for frame in data)
    bad[1].loc[0, "is_completed"] = False
    with pytest.raises(ValueError, match="Unfinished"):
        build(bad)
    cache = build(data)
    cache.margins[0, 0] = 42
    with pytest.raises(ValueError, match="derived features"):
        quality._validate_runtime_cache(cache, *data[:3], data[3], None, cache.policy)
