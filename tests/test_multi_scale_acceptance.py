"""Independent synthetic transition acceptance tests; no market input files."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trend_following import research_mean_reversion as mr
from trend_following import research_mean_reversion_engine as original
from trend_following import research_multi_scale_engine as engine
from trend_following import research_quality_engine_v5 as quality


def packet(specs):
    history = pd.bdate_range("1999-01-01", periods=250)
    daily = [
        dict(
            session=x.strftime("%Y-%m-%d"), close=100.0, dividend_amount=0.0, split_coefficient=1.0
        )
        for x in history
    ]
    bars = []
    calendar = []
    coverage = []
    for j, spec in enumerate(specs):
        day = (pd.Timestamp("2001-01-02") + pd.offsets.BDay(j)).strftime("%Y-%m-%d")
        start = pd.Timestamp(day + " 09:30", tz="America/New_York").tz_convert("UTC")
        closes = spec.get("closes", [90.0] * 13)
        opens = spec.get("opens", [100.0] * len(closes))
        for k, (opening, close) in enumerate(zip(opens, closes, strict=True)):
            left = start + pd.Timedelta(minutes=30 * k)
            right = left + pd.Timedelta(minutes=30)
            missing = k in spec.get("missing", [])
            coverage.append(
                dict(
                    session=day,
                    bar_start=left,
                    bar_end=right,
                    status="unknown_missing" if missing else "observed",
                    signal_eligible=not missing,
                    fill_eligible=not missing,
                )
            )
            if not missing:
                bars.append(
                    dict(
                        session=day,
                        bar_start=left,
                        bar_end=right,
                        open=opening,
                        close=close,
                        high=max(opening, close),
                        low=min(opening, close),
                        volume=100.0,
                        duration_minutes=30.0,
                        is_completed=True,
                        is_full_hour=False,
                    )
                )
        calendar.append(
            dict(
                session=day,
                market_open=start,
                market_close=start + pd.Timedelta(minutes=30 * len(closes)),
            )
        )
        daily.append(
            dict(
                session=day,
                close=spec.get("daily_close", closes[-1]),
                dividend_amount=spec.get("dividend", 0.0),
                split_coefficient=spec.get("split", 1.0),
            )
        )
    return tuple(pd.DataFrame(x) for x in (daily, bars, calendar, coverage))


def base():
    return mr.Candidate(mr.Rule("SMA_DISTANCE", 10), -0.02, 0.0)


def cache(specs, values=None, final_gates=None):
    daily, bars, calendar, coverage = packet(specs)
    result = engine.build_cache(
        daily,
        bars,
        calendar,
        coverage=coverage,
        quality_policy=quality.QualityPolicy("observed_only_blackout_v1"),
    )
    if values is not None:
        margins = result.base.margins.copy()
        margins[:, result.rules.index(base().rule)] = values
        b = replace(result.base, margins=margins, feature_fingerprint="")
        b = replace(b, feature_fingerprint=quality._feature_fingerprint(b))
        result = replace(result, base=b)
    gates = result.gates * 0 + engine.ON
    for session, state in zip(
        result.calendar.session, final_gates or [engine.ON] * len(specs), strict=True
    ):
        gates.loc[session, :] = state
    result = replace(result, gates=gates, gate_override="synthetic_transition", gate_fingerprint="")
    return replace(result, gate_fingerprint=engine._gate_fingerprint(result))


def observations(c, annual=0.0):
    return pd.DataFrame(
        {
            "available_at": [c.calendar.market_open.iloc[0] - pd.Timedelta(days=1)],
            "annual_rate": [annual],
        }
    )


def run(c, policy="recovery_or_trend_break", annual=0.0, fee=0.0, slip=0.0):
    return engine.result(
        engine.run(
            c,
            [engine.GatedCandidate(base(), 100, policy)],
            start=c.calendar.session.iloc[0],
            end=c.calendar.session.iloc[-1],
            cash_observations=observations(c, annual),
            commission_bps=fee,
            slippage_bps=slip,
            record=True,
        )
    )


def test_independent_final_gate_sma_reference_and_future_prefix_invariance():
    values = np.linspace(80.0, 120.0, 260)
    values[215] = np.nan
    actual = engine.finalized_gates(values)
    for period in (100, 200):
        expected = []
        for j, value in enumerate(values):
            window = values[max(0, j - period + 1) : j + 1]
            expected.append(
                -1
                if len(window) != period or not np.isfinite(window).all()
                else int(value > sum(window) / period)
            )
        assert actual[period].tolist() == expected
    prefix = engine.finalized_gates(values[:210])
    assert prefix.equals(actual.iloc[:210])
    flat = engine.finalized_gates(np.full(210, 0.1))
    assert flat[100].iloc[99:].eq(engine.OFF).all()


def test_gate_finalization_does_not_increment_a_confirmation():
    c = cache(
        [{}, {}],
        values=[-0.01] * 13 + [-0.02] * 7 + [-0.01] * 6,
        final_gates=[engine.ON, engine.ON],
    )
    result = run(c)
    buy = result.trades.iloc[0]
    assert buy.decision_at.tz_convert("America/New_York") == pd.Timestamp(
        "2001-01-03 13:00", tz="America/New_York"
    )
    updates = result.diagnostics.loc[result.diagnostics.event.eq("gate_update")]
    assert (
        pd.to_datetime(updates.timestamp, utc=True).dt.tz_convert("America/New_York").dt.hour == 17
    ).all()


def test_overnight_off_gate_cancels_unfilled_entry():
    c = cache(
        [{}, {}],
        values=[-0.01] * 6 + [-0.02] * 7 + [-0.02] * 13,
        final_gates=[engine.OFF, engine.ON],
    )
    result = run(c)
    assert result.trades.empty
    assert result.metrics["entry_cancellation_count"] == 1
    assert result.metrics["blocked_oversold_checks"] == 13


def test_recovery_and_break_streaks_cannot_pool_across_daily_interruptions():
    c = cache(
        [{}, {"closes": [90.0] * 6}, {"closes": [90.0]}, {"closes": [90.0]}],
        values=[-0.02] * 7 + [-0.01] * 6 + [0.0] * 6 + [-0.01] + [0.0],
        final_gates=[engine.OFF, engine.ON, engine.OFF, engine.ON],
    )
    result = run(c)
    assert result.trades.reason.tolist() == ["confirmed_entry", "terminal_liquidation"]
    checks = result.diagnostics.loc[result.diagnostics.event.eq("bar_close")]
    assert checks.exit_count.max() == 6
    assert checks.trend_break_count.max() == 6


@pytest.mark.parametrize(
    "recovery,expected",
    [
        (False, ["confirmed_entry", "terminal_liquidation"]),
        (True, ["confirmed_entry", "confirmed_exit"]),
    ],
)
def test_on_gate_cancels_only_trend_break_pending_not_recovery_cause(recovery, expected):
    c = cache(
        [{}, {"closes": [90.0] * 7}, {}],
        values=[-0.02] * 7 + [-0.01] * 6 + ([0.0] * 7 if recovery else [-0.01] * 7) + [-0.01] * 13,
        final_gates=[engine.OFF, engine.ON, engine.ON],
    )
    result = run(c)
    assert result.trades.reason.tolist() == expected
    assert result.metrics["exit_cancellation_count"] == int(not recovery)
    if recovery:
        sale = result.trades.iloc[1]
        assert sale.exit_causes == "recovery+trend_break"
        assert sale.timestamp.tz_convert("America/New_York") == pd.Timestamp(
            "2001-01-04 09:30", tz="America/New_York"
        )


def test_entry_only_does_not_turn_off_gate_into_an_exit():
    c = cache(
        [{}, {}],
        values=[-0.02] * 7 + [-0.01] * 6 + [-0.01] * 13,
        final_gates=[engine.OFF, engine.OFF],
    )
    result = run(c, policy="entry_only")
    assert result.trades.reason.tolist() == ["confirmed_entry", "terminal_liquidation"]
    assert result.metrics["trend_break_exit_count"] == 0


def test_unknown_gate_cancels_and_holds_position_without_recovery_exit():
    c = cache(
        [{}, {}],
        values=[-0.02] * 7 + [-0.01] * 6 + [0.0] * 13,
        final_gates=[engine.UNKNOWN, engine.ON],
    )
    result = run(c)
    assert result.trades.reason.tolist() == ["confirmed_entry", "terminal_liquidation"]
    assert result.metrics["unknown_gate_reset_count"] >= 1


def test_always_on_all_32_members_reproduce_original_wealth_fills_costs():
    c = cache([{}, {}, {}])
    c = engine.with_constant_gate(c, engine.ON)
    members = sorted(mr.candidate_grid(), key=lambda x: x.candidate_id)[:32]
    wrapped = [engine.GatedCandidate(m, 100, "entry_only") for m in members]
    kwargs = dict(
        start=c.calendar.session.iloc[0],
        end=c.calendar.session.iloc[-1],
        cash_observations=observations(c, 0.03),
        commission_bps=1.0,
        slippage_bps=3.0,
        record=True,
    )
    old = original.run(c.base, members, **kwargs)
    new = engine.run(c, wrapped, **kwargs)
    for j in (2, 3, 4, 5):
        np.testing.assert_array_equal(old[j], new[j])
    for left, right in zip(old[0], new[0], strict=True):
        for key in (
            "terminal_equity",
            "total_commission",
            "total_slippage",
            "trade_count",
            "order_count",
        ):
            assert left[key] == right[key]
    a = pd.DataFrame(old[6])
    b = pd.DataFrame(new[6])
    b["candidate_id"] = b.candidate_id.str.split("__", n=1).str[1]
    pd.testing.assert_frame_equal(a, b[a.columns])


def test_always_off_independent_nominal_event_cash_reconciliation():
    c = engine.with_constant_gate(cache([{}, {}]), engine.OFF)
    result = run(c, annual=0.03, fee=1.0, slip=25.0)
    events = set(c.bars.bar_start) | set(c.bars.bar_end) | set(c.calendar.market_open)
    events |= {
        pd.Timestamp(day + " 17:00", tz="America/New_York").tz_convert("UTC")
        for day in c.calendar.session
    }
    clock = sorted(events)
    value = 1000.0
    for previous, current in zip(clock[:-1], clock[1:], strict=True):
        value *= 1 + 0.03 * (current - previous).total_seconds() / (365 * 86400)
    assert result.metrics["terminal_equity"] == pytest.approx(value, abs=1e-10)
    assert result.metrics["total_commission"] == result.metrics["total_slippage"] == 0
    assert result.trades.empty and np.isnan(result.metrics["cash_excess_sharpe"])
