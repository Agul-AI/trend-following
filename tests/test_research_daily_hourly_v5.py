"""Fabricated 30m/daily-state mechanics only; never read prices or old results."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from trend_following import research_daily_hourly_v5 as engine


def packet(sessions, *, historical=(100.0,) * 15, dividends=None, splits=None):
    history = pd.bdate_range("1999-11-01", periods=len(historical))
    daily = [
        dict(session=d.strftime("%Y-%m-%d"), close=p, dividend_amount=0.0, split_coefficient=1.0)
        for d, p in zip(history, historical, strict=True)
    ]
    bars, calendar = [], []
    for i, spec in enumerate(sessions):
        session = spec.get(
            "session", (pd.Timestamp("2000-01-03") + pd.offsets.BDay(i)).strftime("%Y-%m-%d")
        )
        closes = spec["closes"]
        opens = spec.get("opens", [closes[0]] + closes[:-1])
        durations = spec.get("durations", [30] * len(closes))
        start = (
            (pd.Timestamp(session) + pd.Timedelta(hours=9, minutes=30))
            .tz_localize("America/New_York")
            .tz_convert("UTC")
        )
        opening = start
        for close, open_price, duration in zip(closes, opens, durations, strict=True):
            end = start + pd.Timedelta(minutes=duration)
            bars.append(
                dict(
                    bar_start=start,
                    bar_end=end,
                    session=session,
                    open=open_price,
                    high=max(close, open_price),
                    low=min(close, open_price),
                    close=close,
                    volume=100,
                    is_full_hour=False,
                    duration_minutes=duration,
                )
            )
            start = end
        calendar.append(dict(session=session, market_open=opening, market_close=start))
        daily.append(
            dict(
                session=session,
                close=spec.get("official_close", closes[-1]),
                dividend_amount=0.0 if dividends is None else dividends[i],
                split_coefficient=1.0 if splits is None else splits[i],
            )
        )
    return pd.DataFrame(daily), pd.DataFrame(bars), pd.DataFrame(calendar)


def observations(bars, annual_rate=0.0):
    available = pd.date_range(
        bars.bar_start.iloc[0].normalize() - pd.Timedelta(days=1),
        bars.bar_end.iloc[-1].normalize() + pd.Timedelta(days=1),
        tz="UTC",
    )
    return pd.DataFrame(
        {"available_at": available, "annual_rate": annual_rate, "series_id": "DTB3"}
    )


def candidate(entry=None, exit_rule=None, en=0.5, ex=0.5):
    rule = engine.IndicatorRule("SMA", period=2)
    return engine.Candidate(entry or rule, exit_rule or rule, en, ex)


def run(data, c=None, *, commission=0.0, slip=0.0, rate=0.0, start=None, end=None):
    daily, bars, calendar = data
    return engine.simulate_candidate(
        daily,
        bars,
        c or candidate(),
        start=start or bars.session.iloc[0],
        end=end or bars.session.iloc[-1],
        cash_observations=observations(bars, rate),
        commission_bps=commission,
        slippage_bps=slip,
        calendar=calendar,
    )


def test_complete_56644_grid_unique_immutable_serializable_and_family_counts():
    grid, rules = engine.generate_candidates(), engine.indicator_rules()
    assert len(rules) == 17
    assert len(grid) == len({c.candidate_id for c in grid}) == 56644
    assert engine.CONFIRMATION_HOURS == tuple(i / 2 for i in range(14))
    assert grid == engine.generate_candidates()
    assert all(c.candidate_id.startswith("v5_30m__") for c in grid)
    assert (
        len({c.candidate_id for c in grid if c.entry_rule == rules[0] and c.exit_rule == rules[0]})
        == 196
    )
    for entry in ("SMA", "EMA", "MACD"):
        for exit_rule in ("SMA", "EMA", "MACD"):
            expected = (5 if entry == "MACD" else 6) * (5 if exit_rule == "MACD" else 6) * 196
            assert sum(c.family_cell == f"{entry}/{exit_rule}" for c in grid) == expected
    assert engine.Candidate.from_dict(json.loads(json.dumps(grid[0].to_dict()))) == grid[0]
    numpy_types = candidate(engine.IndicatorRule("EMA", period=np.int64(10)), en=np.float64(0.5))
    assert json.loads(json.dumps(numpy_types.to_dict()))["entry_confirmation_hours"] == 0.5
    with pytest.raises(FrozenInstanceError):
        grid[0].entry_confirmation_hours = 6.5


@pytest.mark.parametrize(
    "bad", [True, False, np.bool_(False), np.nan, np.inf, -np.inf, -0.5, 0.25, 0.75, 7, "0.5", None]
)
def test_duration_validation_rejects_boolean_nonfinite_quarterhours_and_ambiguous_schema(bad):
    rule = engine.IndicatorRule("SMA", period=2)
    with pytest.raises(ValueError, match="Confirmation hours"):
        engine.Candidate(rule, rule, bad, 0)
    with pytest.raises(ValueError, match="Confirmation hours"):
        engine.Candidate(rule, rule, 0, bad)
    with pytest.raises(ValueError, match="old integer"):
        engine.Candidate.from_dict(
            dict(
                entry_rule=rule.to_dict(),
                exit_rule=rule.to_dict(),
                entry_confirmation=2,
                exit_confirmation=2,
            )
        )


@pytest.mark.parametrize(
    "hours,checks,clock", [(0, 1, "10:00"), (0.5, 2, "10:30"), (1, 3, "11:00")]
)
def test_wait_after_first_sample_hours_zero_half_one(hours, checks, clock):
    data = packet([{"closes": [110] * 13, "opens": [100] * 13}])
    c = candidate(en=hours, ex=6.5)
    result = run(data, c)
    buy = result.trades.iloc[0]
    assert c.entry_required_checks == checks
    assert buy.decision_at.tz_convert("America/New_York").strftime("%H:%M") == clock
    assert buy.timestamp == buy.decision_at == data[1].bar_start.iloc[checks]
    assert result.diagnostics.entry_count.iloc[checks - 1] == checks
    assert buy.reference_price == data[1].open.iloc[checks]


def test_six_and_half_hours_requires_fourteen_checks_and_carries_overnight():
    data = packet(
        [
            {"session": "2000-01-07", "closes": [110] * 13},
            {"session": "2000-01-10", "closes": [120] * 13},
        ]
    )
    result = run(data, candidate(en=6.5))
    assert len(result.diagnostics[result.diagnostics.session.eq("2000-01-07")]) == 13
    assert result.diagnostics.entry_count.iloc[12] == 13
    assert result.diagnostics.entry_count.iloc[13] == 14
    buy = result.trades.iloc[0]
    assert buy.timestamp.tz_convert("America/New_York") == pd.Timestamp(
        "2000-01-10 10:00", tz="America/New_York"
    )
    assert result.metrics["entry_required_checks"] == 14
    assert result.metrics["entry_confirmation_hours"] == 6.5


def test_early_close_has_seven_checks_then_carries_without_counting_closed_hours():
    data = packet(
        [
            {"closes": [110] * 7},  # 09:30-13:00, all seven half-hours qualify.
            {"closes": [120] * 13},
        ]
    )
    result = run(data, candidate(en=4))  # Nine checks, not four hourly rows.
    assert result.diagnostics.entry_count.iloc[:9].tolist() == list(range(1, 10))
    assert result.trades.iloc[0].timestamp.tz_convert("America/New_York") == pd.Timestamp(
        "2000-01-04 10:30", tz="America/New_York"
    )


def test_former_partial_tail_counts_and_1600_order_fills_next_session_open():
    data = packet(
        [
            {"closes": [100] * 12 + [110]},
            {"closes": [110] * 13, "opens": [123] + [110] * 12},
        ]
    )
    result = run(data, candidate(en=0, ex=6.5))
    buy = result.trades.iloc[0]
    assert buy.decision_at.tz_convert("America/New_York") == pd.Timestamp(
        "2000-01-03 16:00", tz="America/New_York"
    )
    assert buy.timestamp.tz_convert("America/New_York") == pd.Timestamp(
        "2000-01-04 09:30", tz="America/New_York"
    )
    assert buy.reference_price == 123
    assert result.diagnostics.iloc[12].pending == 1


def test_terminal_pending_entry_is_cancelled_without_a_future_open_fill():
    data = packet([{"closes": [100] * 12 + [110]}])
    result = run(data, candidate(en=0))
    assert result.trades.empty
    assert result.metrics["terminal_equity"] == 1000
    assert result.metrics["cancelled_terminal_order_count"] == 1
    assert bool(result.diagnostics.iloc[-1].terminal_order_cancelled)


def test_ema_seed_minperiod_and_30m_previews_never_advance_daily_state():
    ema = engine.IndicatorRule("EMA", period=9)
    data = packet([{"closes": [120, 130] + [110] * 11}])
    cache = engine.build_support_cache(data[0], data[1], data[2], rules=(ema,))
    np.testing.assert_allclose(cache.margins[:2, 0], [120 - 104, 130 - 106])
    np.testing.assert_allclose(cache.margins[2:, 0], [110 - 102] * 11)
    assert cache.finalized_daily_margins.iloc[-1, 0] == pytest.approx(8)
    assert cache.prior_session_counts.tolist() == [15] * 13
    state = engine._DailyState((engine.IndicatorRule("EMA", period=3),))
    assert np.isnan(state.preview(100)[0][0])
    state.finalize(100)
    state.finalize(110)
    assert state.preview(120)[0][0] == pytest.approx(120 - 112.5)


def test_macd_seeds_first_valid_slow_full_signal_period_and_preview_is_pure():
    rule = engine.IndicatorRule("MACD", fast=2, slow=3, signal=2)
    state = engine._DailyState((rule,))
    state.finalize(100)
    state.finalize(110)
    assert np.isnan(state.preview(120)[0][0])
    state.finalize(120)
    fast4 = state.ema[2] + 2 / 3 * (130 - state.ema[2])
    slow4 = state.ema[3] + 0.5 * (130 - state.ema[3])
    macd4 = fast4 - slow4
    seeded = state.macd_signal[rule]
    assert state.preview(130)[0][0] == pytest.approx(macd4 - (seeded + 2 / 3 * (macd4 - seeded)))
    before = state.macd_signal.copy()
    state.preview(140)
    assert state.macd_signal == before
    long_rule = engine.IndicatorRule("MACD", fast=96, slow=208, signal=72)
    long_state = engine._DailyState((long_rule,))
    for i in range(278):
        assert np.isnan(long_state.preview(100 + i)[0][0])
        long_state.finalize(100 + i)
    assert np.isfinite(long_state.preview(378)[0][0])


def test_zero_hour_uses_completed_close_not_current_open_and_costs_charge_once():
    data = packet([{"closes": [110, 120], "opens": [100, 130]}])
    result = run(data, candidate(en=0, ex=6.5), commission=1, slip=3)
    buy, sell = result.trades.iloc[0], result.trades.iloc[1]
    assert buy.timestamp == data[1].bar_start.iloc[1]
    assert buy.decision_at == data[1].bar_end.iloc[0]
    assert buy.reference_price == 130
    assert sell.reason == "terminal_liquidation" and pd.isna(sell.decision_at)
    qty = 1000 / (130 * 1.0003 * 1.0001)
    expected = qty * 120 * 0.9997 * 0.9999
    assert result.metrics["terminal_equity"] == pytest.approx(expected)
    assert result.metrics["total_commission"] == pytest.approx(result.trades.commission.sum())
    assert result.metrics["total_slippage"] == pytest.approx(result.trades.slippage.sum())
    assert np.prod(1 + result.daily_returns) * 1000 == pytest.approx(expected)
    assert result.metrics["order_count"] == 2


def test_independent_both_entry_gate_exit_only_and_opposite_checks_reset():
    gate_data = packet([{"closes": [110, 120, 130, 140]}], historical=(100,) * 13 + (200, 100))
    blocked = run(gate_data, candidate(exit_rule=engine.IndicatorRule("SMA", period=3), en=0))
    assert blocked.diagnostics.entry_support.all()
    assert not blocked.diagnostics.exit_support.any()
    assert blocked.trades.empty
    data = packet([{"closes": [160, 160, 120, 100, 100, 100]}], historical=(100,) * 13 + (200, 100))
    held = run(data, candidate(entry=engine.IndicatorRule("SMA", period=3)))
    assert not bool(held.diagnostics.iloc[2].entry_support)
    assert bool(held.diagnostics.iloc[2].exit_support)
    assert held.diagnostics.exit_count.tolist() == [0, 0, 0, 1, 2, 0]
    assert held.trades.iloc[1].timestamp == data[1].bar_start.iloc[5]
    intermittent = run(packet([{"closes": [110, 90, 110, 90]}]))
    assert intermittent.diagnostics.entry_count.tolist() == [1, 0, 1, 0]
    assert intermittent.trades.empty


def test_independent_exit_hours_zero_half_and_one_need_one_two_three_checks():
    data = packet([{"closes": [110, 90, 90, 90, 90, 90], "opens": [100, 110, 90, 90, 90, 90]}])
    for hours, sell_index in [(0, 2), (0.5, 3), (1, 4)]:
        result = run(data, candidate(en=0, ex=hours))
        assert result.trades.iloc[1].reason == "confirmed_exit"
        assert result.trades.iloc[1].timestamp == data[1].bar_start.iloc[sell_index]


def test_segment_starts_cash_with_empty_streak_and_no_warmup_holds():
    data = packet(
        [
            {"closes": [110, 110, 100]},
            {"closes": [110, 100]},
        ]
    )
    assert run(data).metrics["trade_count"] == 1
    segment = run(data, candidate(en=0.5), start="2000-01-04")
    assert segment.trades.empty
    assert segment.diagnostics.entry_count.tolist() == [1, 0]


def test_split_dividend_drip_conservation_benchmark_cost_matching_no_double_count():
    data = packet(
        [
            {"closes": [100, 100]},
            {"closes": [49.5, 49.5]},
            {"closes": [55, 55]},
        ],
        dividends=[0, 0.5, 0],
        splits=[1, 2, 1],
    )
    daily, bars, cal = data
    result = engine.simulate_benchmarks(
        daily,
        bars,
        start=bars.session.iloc[0],
        end=bars.session.iloc[-1],
        cash_observations=observations(bars),
        commission_bps=0,
        slippage_bps=0,
        calendar=cal,
    )
    held = result["buy_and_hold"]
    assert held.daily_equity.iloc[:2].tolist() == pytest.approx([1000, 1000])
    assert held.metrics["terminal_equity"] == pytest.approx(1000 * 55 / 49.5)
    assert held.trades.side.tolist() == ["buy", "sell"]
    cache = engine.build_support_cache(daily, bars, cal)
    assert cache.finalized_total_return_index.loc["2000-01-04"] == pytest.approx(100)
    assert cache.finalized_total_return_index.loc["2000-01-05"] == pytest.approx(100 * 55 / 49.5)
    assert result["cash"].metrics["terminal_equity"] == 1000


def test_ex_date_buyer_gets_no_prior_holder_dividend_credit():
    data = packet(
        [
            {"closes": [110]},
            {"closes": [109, 109]},
        ],
        dividends=[0, 1],
    )
    result = run(data, candidate(en=0, ex=6.5))
    assert result.trades.iloc[0].timestamp == data[1].bar_start.iloc[1]
    assert result.trades.iloc[0].quantity == pytest.approx(1000 / 109)
    assert result.metrics["terminal_equity"] == pytest.approx(1000)


def test_cash_clock_no_duplicate_boundary_accrual_includes_weekends_and_future_rates_causal():
    data = packet(
        [
            {"session": "2000-01-07", "closes": [100] * 13},
            {"session": "2000-01-10", "closes": [100] * 13},
        ]
    )
    result = run(data, rate=0.10)
    bars = data[1]
    clock = pd.DatetimeIndex(sorted(set(bars.bar_start) | set(bars.bar_end)))
    expected = 1000 * np.prod(
        1 + engine.rate_accrual(clock, observations(bars, 0.10), interval_start=clock[0])
    )
    assert result.metrics["terminal_equity"] == pytest.approx(expected)
    assert np.prod(1 + result.daily_cash_returns) * 1000 == pytest.approx(expected)
    assert np.isnan(result.metrics["cash_excess_sharpe"])
    assert expected > 1000 * (1 + 0.10 * 3 / 365)
    rates = observations(bars, 0.10)
    later = rates.copy()
    later.loc[later.available_at >= pd.Timestamp("2000-01-10", tz="UTC"), "annual_rate"] = 0.50
    kwargs = dict(start="2000-01-07", end="2000-01-10", calendar=data[2])
    changed = engine.simulate_candidate(
        data[0], bars, candidate(), cash_observations=later, **kwargs
    )
    assert changed.daily_cash_returns.iloc[0] == result.daily_cash_returns.iloc[0]
    assert changed.daily_cash_returns.iloc[-1] > result.daily_cash_returns.iloc[-1]


def test_bad_frames_stale_unknown_rates_and_old_hourly_clock_are_rejected():
    daily, bars, cal = packet([{"closes": [100, 100]}])
    kwargs = dict(start="2000-01-03", end="2000-01-03", calendar=cal)
    stale = pd.DataFrame(
        {"available_at": [bars.bar_start.iloc[0] - pd.Timedelta(days=20)], "annual_rate": [0.01]}
    )
    with pytest.raises(ValueError, match="Stale rate"):
        engine.simulate_candidate(daily, bars, candidate(), cash_observations=stale, **kwargs)
    with pytest.raises(ValueError, match="DTB3"):
        engine.simulate_candidate(
            daily,
            bars,
            candidate(),
            cash_observations=observations(bars).assign(series_id="DFF"),
            **kwargs,
        )
    naive = bars.copy()
    naive["bar_start"] = naive.bar_start.dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        engine.build_support_cache(daily, naive, cal)
    incomplete = bars.assign(is_completed=[True, False])
    with pytest.raises(ValueError, match="Incomplete"):
        engine.build_support_cache(daily, incomplete, cal)
    wrong = packet([{"closes": [100, 100], "durations": [60, 30]}])
    with pytest.raises(ValueError, match="exactly 30"):
        engine.build_support_cache(*wrong[:2], wrong[2])
    gap = bars.copy()
    gap.loc[1, "bar_start"] += pd.Timedelta(minutes=30)
    gap.loc[1, "bar_end"] += pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="Missing bar"):
        engine.build_support_cache(daily, gap, cal)


def test_future_prices_actions_and_next_open_do_not_change_prior_signals():
    a = packet(
        [{"closes": [110, 110, 100]}, {"closes": [110, 110, 100]}, {"closes": [110, 110, 100]}]
    )
    b = packet(
        [
            {"closes": [110, 110, 100]},
            {"closes": [110, 110, 100]},
            {"closes": [1000, 2000, 3000], "opens": [999, 1999, 2999]},
        ],
        dividends=[0, 0, 5],
        splits=[1, 1, 2],
    )
    ar, br = run(a), run(b)
    pd.testing.assert_frame_equal(ar.diagnostics.iloc[:6], br.diagnostics.iloc[:6])
    pd.testing.assert_series_equal(ar.daily_equity.iloc[:2], br.daily_equity.iloc[:2])
    ac, bc = engine.build_support_cache(*a[:2], a[2]), engine.build_support_cache(*b[:2], b[2])
    np.testing.assert_allclose(ac.margins[:6], bc.margins[:6], equal_nan=True)
    changed = a[1].copy()
    changed.loc[2, "open"] = 1000
    changed.loc[2, "high"] = 1000
    cr = run((a[0], changed, a[2]))
    assert cr.trades.iloc[0].decision_at == ar.trades.iloc[0].decision_at
    assert cr.trades.iloc[0].reference_price != ar.trades.iloc[0].reference_price


def test_daily_cash_excess_sharpe_ddof1_terminal_return_and_cost_drag():
    data = packet(
        [
            {"closes": [110, 110, 100], "opens": [100, 110, 100]},
            {"closes": [110, 110, 120]},
            {"closes": [125, 125, 130]},
        ]
    )
    result = run(data, commission=1, slip=3, rate=0.01)
    excess = result.daily_returns - result.daily_cash_returns
    assert result.metrics["cash_excess_sharpe"] == pytest.approx(
        np.sqrt(252) * excess.mean() / excess.std(ddof=1)
    )
    assert result.metrics["cost_drag_return"] > 0
    assert result.metrics["exposure_fraction"] > 0
    assert result.daily_exposure.iloc[-1] == 1


def test_batch_fixed256_cache_reuse_matches_single_and_full_grid_smoke():
    data = packet([{"closes": [110, 110, 90, 90, 100]}, {"closes": [110, 110, 120]}])
    daily, bars, cal = data
    cs = [candidate(), candidate(en=1), candidate(ex=1)]
    cache = engine.build_support_cache(
        daily, bars, cal, rules=tuple(dict.fromkeys((*engine.indicator_rules(), cs[0].entry_rule)))
    )
    rows = engine.evaluate_grid(
        daily,
        bars,
        start="2000-01-03",
        end="2000-01-04",
        cash_observations=observations(bars),
        candidates=cs,
        calendar=cal,
        support_cache=cache,
        batch_size=256,
    )
    for i, c in enumerate(cs):
        single = run(data, c, commission=1, slip=3)
        for key in ("terminal_equity", "cash_excess_sharpe", "total_turnover", "exposure_fraction"):
            assert rows.iloc[i][key] == pytest.approx(single.metrics[key], nan_ok=True)
    # All real rules are warmed on invented constant prices; no market data is read.
    invented = packet(
        [
            {"session": "2001-01-02", "closes": [110] * 13},
            {"session": "2001-01-03", "closes": [120] * 13},
        ],
        historical=(100.0,) * 295,
    )
    grid = engine.evaluate_grid(
        invented[0],
        invented[1],
        start="2001-01-02",
        end="2001-01-03",
        cash_observations=observations(invented[1]),
        calendar=invented[2],
        batch_size=256,
    )
    assert len(grid) == grid.candidate_id.nunique() == 56644
    assert set(grid.entry_confirmation_hours) == set(engine.CONFIRMATION_HOURS)


def test_pure_live_preview_no_current_daily_close_matches_historical_and_actions():
    data = packet([{"closes": [120, 130, 110]}])
    rules = (
        engine.IndicatorRule("EMA", period=9),
        engine.IndicatorRule("SMA", period=3),
        engine.IndicatorRule("MACD", fast=2, slow=3, signal=2),
    )
    completed = data[0].iloc[:-1].copy()
    before = completed.copy(deep=True)
    first = engine.preview_daily_margins(completed, 120, rules=rules)
    cache = engine.build_support_cache(data[0], data[1], data[2], rules=rules)
    np.testing.assert_allclose(first.to_numpy(), cache.margins[0], equal_nan=True)
    assert first.attrs["prior_session_count"] == len(completed)
    assert first.loc["ema_9"] == pytest.approx(16)
    assert engine.preview_daily_margins(completed, 130, rules=rules).loc["ema_9"] == pytest.approx(
        24
    )
    pd.testing.assert_frame_equal(completed, before)
    live = engine.preview_daily_margins(
        completed, 49.5, split_coefficient=2, dividend_amount=0.5, rules=rules
    )
    assert live.attrs["provisional_total_return_index"] == pytest.approx(100)


def test_close_tolerance_preserves_observed_quote_and_special_open_is_accepted():
    data = packet([{"closes": [110, 110, 100], "official_close": 100.05}])
    cache = engine.build_support_cache(data[0], data[1], data[2])
    assert cache.bars.close.iloc[-1] == 100
    assert cache.finalized_total_return_index.iloc[-1] == pytest.approx(100.05)
    bars, cal = data[1].copy(), data[2].copy()
    offset = pd.Timedelta(hours=1, minutes=30)
    bars["bar_start"] += offset
    bars["bar_end"] += offset
    cal["market_open"] += offset
    cal["market_close"] += offset
    assert (
        engine.build_support_cache(data[0], bars, cal)
        .bars.bar_start.iloc[0]
        .tz_convert("America/New_York")
        .hour
        == 11
    )


def test_shortlist_uses_only_finite_primary_tolerance_then_stable_id():
    rows = pd.DataFrame(
        [
            dict(
                entry_family="SMA",
                exit_family="EMA",
                cash_excess_sharpe=1.0,
                candidate_id="b",
                cagr=999,
            ),
            dict(
                entry_family="SMA",
                exit_family="EMA",
                cash_excess_sharpe=1.0 - 5e-11,
                candidate_id="a",
                cagr=-999,
            ),
            dict(
                entry_family="EMA",
                exit_family="MACD",
                cash_excess_sharpe=np.nan,
                candidate_id="c",
                cagr=999,
            ),
        ]
    )
    assert engine.shortlist_family_cells(rows).candidate_id.tolist() == ["a"]
