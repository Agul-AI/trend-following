"""Synthetic observed-only blackout mechanics; no market files or API calls."""

import numpy as np
import pandas as pd
import pytest

from trend_following import research_daily_hourly_v5 as core
from trend_following import research_quality_engine_v5 as quality


def fixture(specs, history=(100.0,) * 15):
    historical = pd.bdate_range("1999-11-01", periods=len(history))
    daily = [
        dict(session=x.strftime("%Y-%m-%d"), close=p, dividend_amount=0.0, split_coefficient=1.0)
        for x, p in zip(historical, history, strict=True)
    ]
    bars = []
    coverage = []
    calendar = []
    for i, spec in enumerate(specs):
        session = spec.get(
            "session", (pd.Timestamp("2000-01-03") + pd.offsets.BDay(i)).strftime("%Y-%m-%d")
        )
        opening = pd.Timestamp(f"{session} 09:30", tz="America/New_York").tz_convert("UTC")
        closes = spec.get("closes", [110] * 13)
        opens = spec.get("opens", [100] * len(closes))
        missing = set(spec.get("missing", []))
        for k, (op, cl) in enumerate(zip(opens, closes, strict=True)):
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
                        open=op,
                        close=cl,
                        high=max(op, cl),
                        low=min(op, cl),
                        volume=100.0,
                        duration_minutes=30.0,
                        is_full_hour=False,
                        is_completed=True,
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
    return pd.DataFrame(daily), pd.DataFrame(bars), pd.DataFrame(calendar), pd.DataFrame(coverage)


def c(en=0, ex=6.5):
    rule = core.IndicatorRule("SMA", period=2)
    return core.Candidate(rule, rule, en, ex)


def rates(calendar, annual=0.0):
    dates = pd.date_range(
        calendar.market_open.iloc[0].normalize() - pd.Timedelta(days=1),
        calendar.market_close.iloc[-1].normalize() + pd.Timedelta(days=2),
        tz="UTC",
    )
    return pd.DataFrame({"available_at": dates, "annual_rate": annual, "series_id": "DTB3"})


def run(
    data,
    candidate=None,
    *,
    profile="observed_only_blackout_v1",
    blackouts=None,
    annual=0.0,
    fee=0.0,
    slip=0.0,
):
    daily, bars, calendar, coverage = data
    return quality.simulate_quality_candidate(
        daily,
        bars,
        candidate or c(),
        start=calendar.session.iloc[0],
        end=calendar.session.iloc[-1],
        cash_observations=rates(calendar, annual),
        calendar=calendar,
        coverage=coverage,
        blackouts=blackouts,
        quality_policy=quality.QualityPolicy(profile),
        commission_bps=fee,
        slippage_bps=slip,
    )


def halt_fixture():
    data = fixture(
        [
            {"session": "2013-08-22", "closes": [100, 100, 100, 110, 110] + [5000] * 7 + [110]},
            {"session": "2013-08-23", "closes": [120] * 13},
        ]
    )
    daily, bars, calendar, coverage = data
    start = pd.Timestamp("2013-08-22 12:23", tz="America/New_York").tz_convert("UTC")
    end = pd.Timestamp("2013-08-22 15:25", tz="America/New_York").tz_convert("UTC")
    overlap = coverage.bar_start.lt(end) & coverage.bar_end.gt(start)
    coverage.loc[overlap, ["status", "signal_eligible", "fill_eligible"]] = [
        "known_halt_overlap",
        False,
        False,
    ]
    # A genuine 12:00 opening precedes the halt, though this half-hour cannot confirm.
    coverage.loc[
        coverage.bar_start.eq(
            pd.Timestamp("2013-08-22 12:00", tz="America/New_York").tz_convert("UTC")
        ),
        "fill_eligible",
    ] = True
    blackouts = pd.DataFrame(
        {
            "start": [start],
            "end": [end],
            "kind": ["known_halt"],
            "reason": ["documented_exchange_halt"],
        }
    )
    return (daily, bars, calendar, coverage), blackouts


def test_default_strict_tradable_rejects_unknown_but_allows_documented_halt():
    assert quality.DEFAULT_POLICY.profile == "strict_tradable_v1"
    gap = fixture([{"missing": [0]}])
    with pytest.raises(ValueError, match="Strict-tradable"):
        run(gap, profile="strict_tradable_v1")
    data, blackouts = halt_fixture()
    result = run(data, c(en=1), profile="strict_tradable_v1", blackouts=blackouts)
    assert result.metrics["halt_affected_slots"] == 7
    assert result.metrics["unknown_missing_slots"] == 0
    assert result.metrics["tradable_price_coverage_complete"] is True
    assert result.metrics["all_slots_signal_eligible"] is False


def test_first_missing_open_cancels_old_order_and_zero_needs_fresh_completed_close():
    data = fixture(
        [
            {"closes": [100] * 12 + [110], "daily_close": 100},
            {"missing": [0], "closes": [110] * 13, "opens": [100, 101, 130] + [100] * 10},
        ]
    )
    result = run(data)
    buy = result.trades.iloc[0]
    assert buy.decision_at.tz_convert("America/New_York") == pd.Timestamp(
        "2000-01-04 10:30", tz="America/New_York"
    )
    assert buy.timestamp == buy.decision_at
    assert buy.reference_price == 130  # Next row open, never first post-gap 10:00 open.
    assert result.metrics["cancelled_blackout_order_count"] == 1
    assert result.metrics["raw_intraday_price_coverage_complete"] is False
    assert result.metrics["quality_classification_complete"] is True


def test_wholly_missing_day_still_values_actions_and_defers_dividend_drip_without_fake_price():
    data = fixture(
        [
            {"closes": [110] * 13, "daily_close": 100},
            {
                "missing": list(range(13)),
                "closes": [49.5] * 13,
                "daily_close": 49.5,
                "split": 2,
                "dividend": 0.5,
            },
            {"closes": [55] * 13, "opens": [50] * 13, "daily_close": 60},
        ]
    )
    result = run(data)
    assert len(result.daily_returns) == 3
    assert result.daily_equity.iloc[:2].tolist() == pytest.approx([1000, 1000])
    # 10 old shares -> 20 new shares, $10 cash held without a made-up gap-day open.
    # Next observed safe open50 DRIPs .2 shares; liquidation uses observed55, not daily60.
    assert result.trades.iloc[-1].quantity == pytest.approx(20.2)
    assert result.metrics["terminal_equity"] == pytest.approx(20.2 * 55)
    assert result.metrics["unknown_missing_slots"] == 13
    assert result.daily_equity.index.tz_convert("America/New_York").hour.tolist() == [17] * 3
    assert result.trades.iloc[-1].timestamp.tz_convert("America/New_York").hour == 16


def test_dividend_credit_does_not_earn_prior_overnight_interest():
    data = fixture(
        [
            {"closes": [110] * 13, "daily_close": 100},
            {"missing": list(range(13)), "closes": [99] * 13, "daily_close": 99, "dividend": 1},
            {"closes": [100] * 13, "opens": [100] * 13, "daily_close": 100},
        ]
    )
    result = run(data, annual=0.10)
    qty = result.trades.iloc[0].quantity
    rate = 0.10 / (365 * 24)
    # The distribution arrives at 09:30. Only 09:30->16 and 16->17 compound that day.
    dividend_at_17 = qty * (1 + rate * 6.5) * (1 + rate * 1)
    assert result.daily_equity.iloc[1] == pytest.approx(qty * 99 + dividend_at_17)


def test_halt_straddle_close_never_confirms_or_fills_inside_halt_and_streak_resets():
    data, blackouts = halt_fixture()
    cache = quality.build_quality_support_cache(
        *data[:3],
        coverage=data[3],
        blackouts=blackouts,
        quality_policy=quality.QualityPolicy("strict_tradable_v1"),
        rules=(c().entry_rule,),
    )
    assert np.isnan(cache.margins[5:12]).all()
    assert cache.fill_eligible[5]  # 12:00 is before 12:23.
    assert not cache.fill_eligible[11]  # 15:00 open is not the first trade at 15:25.
    assert cache.fill_eligible[12] and cache.signal_eligible[12]
    result = run(data, c(en=1), profile="strict_tradable_v1", blackouts=blackouts)
    # Two pre-halt hits reset at12:23;16:00 becomes hit1, next10:00 hit2,10:30 hit3.
    assert result.trades.iloc[0].timestamp.tz_convert("America/New_York") == pd.Timestamp(
        "2013-08-23 10:30", tz="America/New_York"
    )
    assert not result.trades.timestamp.between(
        blackouts.start.iloc[0], blackouts.end.iloc[0], inclusive="left"
    ).any()


def test_pre_halt_twelve_open_can_fill_already_confirmed_signal():
    data, blackouts = halt_fixture()
    # Only the11:30->12:00 close qualifies, so no earlier entry can fill.
    data[1].loc[
        data[1].bar_end.lt(
            pd.Timestamp("2013-08-22 12:00", tz="America/New_York").tz_convert("UTC")
        ),
        "close",
    ] = 100
    data[1]["low"] = data[1][["low", "close"]].min(axis=1)
    result = run(data, c(en=0), profile="strict_tradable_v1", blackouts=blackouts)
    assert result.trades.iloc[0].timestamp.tz_convert("America/New_York") == pd.Timestamp(
        "2013-08-22 12:00", tz="America/New_York"
    )


def test_intraday_previews_do_not_consume_current_provider_daily_price_until_next_session():
    data = fixture(
        [{"closes": [120, 130] + [110] * 11, "daily_close": 111}, {"closes": [120] * 13}]
    )
    changed = (data[0].copy(), data[1].copy(), data[2].copy(), data[3].copy())
    changed[0].loc[changed[0].session.eq("2000-01-03"), "close"] = 999
    rules = (core.IndicatorRule("EMA", period=9),)
    a = quality.build_quality_support_cache(*data[:3], coverage=data[3], rules=rules)
    b = quality.build_quality_support_cache(*changed[:3], coverage=changed[3], rules=rules)
    np.testing.assert_allclose(a.margins[:13], b.margins[:13], equal_nan=True)
    assert a.margins[0, 0] == pytest.approx(16)
    assert a.margins[1, 0] == pytest.approx(24)
    assert a.prior_session_counts[:13].tolist() == [15] * 13
    assert not np.array_equal(a.margins[13:], b.margins[13:])
    assert a.finalized_total_return_index.loc["2000-01-03"] == pytest.approx(111)
    assert b.finalized_total_return_index.loc["2000-01-03"] == pytest.approx(999)


def test_terminal_requires_observed_safe_rth_close_and_accrues_liquidated_cash_to_17():
    bad = fixture([{"missing": [12]}])
    with pytest.raises(ValueError, match="Terminal liquidation requires"):
        run(bad)
    data = fixture([{"closes": [110] * 13, "daily_close": 999}])
    result = run(data, annual=0.10, fee=1, slip=3)
    terminal = result.trades.iloc[-1]
    after16 = terminal.quantity * terminal.execution_price - terminal.commission
    expected = after16 * (1 + 0.10 / (365 * 24))
    assert result.metrics["terminal_equity"] == pytest.approx(expected)
    assert terminal.reference_price == 110  # Provider999 is not a16:00 execution quote.
    assert result.daily_returns.index[-1].tz_convert("America/New_York").hour == 17
    assert result.metrics["total_commission"] == pytest.approx(result.trades.commission.sum())
    assert result.metrics["total_slippage"] == pytest.approx(result.trades.slippage.sum())


def test_benchmarks_and_cash_match_17_clock_corporate_actions_and_terminal_prices():
    data = fixture(
        [
            {"closes": [100] * 13, "daily_close": 101},
            {
                "missing": list(range(13)),
                "closes": [49.5] * 13,
                "daily_close": 49.5,
                "split": 2,
                "dividend": 0.5,
            },
            {"closes": [55] * 13, "opens": [50] * 13, "daily_close": 60},
        ]
    )
    daily, bars, cal, cov = data
    out = quality.simulate_quality_benchmarks(
        daily,
        bars,
        start="2000-01-03",
        end="2000-01-05",
        calendar=cal,
        coverage=cov,
        quality_policy=quality.QualityPolicy("observed_only_blackout_v1"),
        cash_observations=rates(cal),
        commission_bps=0,
        slippage_bps=0,
    )
    assert set(out) == {"cash", "buy_and_hold"}
    assert out["buy_and_hold"].daily_equity.iloc[1] == pytest.approx(1000)
    assert out["buy_and_hold"].metrics["terminal_equity"] == pytest.approx(1111)
    assert out["cash"].metrics["terminal_equity"] == 1000
    assert out["cash"].daily_equity.index.equals(out["buy_and_hold"].daily_equity.index)


def test_quality_batch_cache_reuses_core_candidates_and_matches_single():
    data = fixture([{"missing": [2, 3], "daily_close": 100}, {"closes": [120] * 13}])
    daily, bars, cal, cov = data
    candidates = [c(0), c(0.5), c(1)]
    policy = quality.QualityPolicy("observed_only_blackout_v1")
    cache = quality.build_quality_support_cache(
        daily,
        bars,
        cal,
        coverage=cov,
        quality_policy=policy,
        rules=tuple(dict.fromkeys((*core.indicator_rules(), c().entry_rule))),
    )
    rows = quality.evaluate_quality_grid(
        daily,
        bars,
        start="2000-01-03",
        end="2000-01-04",
        calendar=cal,
        coverage=cov,
        quality_policy=policy,
        candidates=candidates,
        cash_observations=rates(cal),
        support_cache=cache,
        batch_size=2,
        commission_bps=1,
        slippage_bps=3,
    )
    for i, candidate in enumerate(candidates):
        result = run(data, candidate, fee=1, slip=3)
        for field in (
            "terminal_equity",
            "cash_excess_sharpe",
            "total_turnover",
            "exposure_fraction",
        ):
            assert rows.iloc[i][field] == pytest.approx(result.metrics[field], nan_ok=True)
    with pytest.raises(ValueError, match="1..256"):
        quality.evaluate_quality_grid(
            daily,
            bars,
            start="2000-01-03",
            end="2000-01-04",
            calendar=cal,
            candidates=candidates,
            cash_observations=rates(cal),
            support_cache=cache,
            batch_size=257,
        )
    assert quality.generate_candidates is core.generate_candidates


def test_metadata_cannot_forge_observations_fills_or_daily_available_clock():
    data = fixture([{"missing": [0]}])
    forged = data[3].copy()
    forged.loc[0, "fill_eligible"] = True
    with pytest.raises(ValueError, match="Absent rows"):
        quality.build_quality_support_cache(
            *data[:3],
            coverage=forged,
            quality_policy=quality.QualityPolicy("observed_only_blackout_v1"),
        )
    data, blackouts = halt_fixture()
    forged = data[3].copy()
    forged.loc[
        forged.bar_start.eq(
            pd.Timestamp("2013-08-22 15:00", tz="America/New_York").tz_convert("UTC")
        ),
        "fill_eligible",
    ] = True
    with pytest.raises(ValueError, match="inside a blackout"):
        quality.build_quality_support_cache(*data[:3], coverage=forged, blackouts=blackouts)
    complete = fixture([{}])
    complete[0]["daily_price_available_at"] = [
        pd.Timestamp(f"{day} 16:00", tz="America/New_York").tz_convert("UTC")
        for day in complete[0].session
    ]
    with pytest.raises(ValueError, match="17:00"):
        quality.build_quality_support_cache(*complete[:3], coverage=complete[3])


def test_opt_in_cache_cannot_silently_bypass_default_strict_policy():
    data = fixture([{"missing": [0]}])
    policy = quality.QualityPolicy("observed_only_blackout_v1")
    cache = quality.build_quality_support_cache(*data[:3], coverage=data[3], quality_policy=policy)
    with pytest.raises(ValueError, match="Cached quality policy differs"):
        quality.simulate_quality_candidate(
            data[0],
            data[1],
            c(),
            start="2000-01-03",
            end="2000-01-03",
            cash_observations=rates(data[2]),
            calendar=data[2],
            support_cache=cache,
        )


def test_unknown_gap_bounds_cannot_consume_regular_overnight_closures():
    data = fixture([{"missing": [0]}, {}])
    opening = data[2].market_open.iloc[0]
    fabricated = pd.DataFrame(
        {
            "start": [opening - pd.Timedelta(hours=2)],
            "end": [opening + pd.Timedelta(minutes=30)],
            "kind": ["unknown_missing"],
            "reason": ["bad_bounds"],
        }
    )
    with pytest.raises(ValueError, match="exactly cover contiguous"):
        quality.build_quality_support_cache(
            *data[:3],
            coverage=data[3],
            blackouts=fabricated,
            quality_policy=quality.QualityPolicy("observed_only_blackout_v1"),
        )


def test_full_56644_grid_bounded256_batch_smoke_on_invented_prices_only():
    data = fixture(
        [
            {"session": "2001-01-02", "closes": [110] * 13},
            {"session": "2001-01-03", "closes": [120] * 13},
        ],
        history=(100.0,) * 295,
    )
    rows = quality.evaluate_quality_grid(
        data[0],
        data[1],
        start="2001-01-02",
        end="2001-01-03",
        cash_observations=rates(data[2]),
        calendar=data[2],
        coverage=data[3],
        batch_size=256,
    )
    assert len(rows) == rows.candidate_id.nunique() == 56644
    assert rows.quality_profile.eq("strict_tradable_v1").all()
    assert rows.raw_intraday_price_coverage_complete.all()


def test_relabelled_observed_only_cache_cannot_claim_strict_readiness():
    from dataclasses import replace

    data = fixture([{"missing": [0]}])
    opted = quality.build_quality_support_cache(
        *data[:3],
        coverage=data[3],
        quality_policy=quality.QualityPolicy("observed_only_blackout_v1"),
        rules=(c().entry_rule,),
    )
    relabelled = replace(opted, policy=quality.DEFAULT_POLICY)
    with pytest.raises(ValueError, match="regardless of its label"):
        quality.simulate_quality_candidate(
            data[0],
            data[1],
            c(),
            start="2000-01-03",
            end="2000-01-03",
            cash_observations=rates(data[2]),
            calendar=data[2],
            support_cache=relabelled,
        )


def test_cached_input_identity_rejects_altered_caller_quotes():
    data = fixture([{}])
    cache = quality.build_quality_support_cache(
        *data[:3], coverage=data[3], rules=(c().entry_rule,)
    )
    changed = data[1].copy()
    changed.loc[1, "open"] = 105
    with pytest.raises(ValueError, match="Caller inputs differ"):
        quality.simulate_quality_candidate(
            data[0],
            changed,
            c(),
            start="2000-01-03",
            end="2000-01-03",
            cash_observations=rates(data[2]),
            calendar=data[2],
            support_cache=cache,
        )
    daily = data[0].copy()
    daily.loc[daily.index[-1], "close"] = 111
    with pytest.raises(ValueError, match="Caller inputs differ"):
        quality.simulate_quality_candidate(
            daily,
            data[1],
            c(),
            start="2000-01-03",
            end="2000-01-03",
            cash_observations=rates(data[2]),
            calendar=data[2],
            support_cache=cache,
        )


def test_mutated_cache_coverage_features_and_execution_rows_fail_closed():
    from dataclasses import replace

    data = fixture([{}])
    kwargs = dict(
        start="2000-01-03", end="2000-01-03", cash_observations=rates(data[2]), calendar=data[2]
    )
    cache = quality.build_quality_support_cache(
        *data[:3], coverage=data[3], rules=(c().entry_rule,)
    )
    changed = cache.coverage.copy()
    changed.loc[0, "fill_eligible"] = False
    with pytest.raises(ValueError, match="Cached source/coverage identity changed"):
        quality.simulate_quality_candidate(
            data[0], data[1], c(), support_cache=replace(cache, coverage=changed), **kwargs
        )
    features = cache.margins.copy()
    features[0, 0] = -100
    with pytest.raises(ValueError, match="Cached derived features"):
        quality.simulate_quality_candidate(
            data[0], data[1], c(), support_cache=replace(cache, margins=features), **kwargs
        )
    rows = list(cache.bar_rows)
    rows[1] = rows[1]._replace(open=1000)
    with pytest.raises(ValueError, match="Cached derived features"):
        quality.simulate_quality_candidate(
            data[0], data[1], c(), support_cache=replace(cache, bar_rows=tuple(rows)), **kwargs
        )


def diagnostic_run(data, candidate=None, *, limit=10000, blackouts=None, start=None, end=None):
    daily, bars, calendar, coverage = data
    return quality.simulate_quality_candidate(
        daily,
        bars,
        candidate or c(),
        start=start or calendar.session.iloc[0],
        end=end or calendar.session.iloc[-1],
        cash_observations=rates(calendar),
        calendar=calendar,
        coverage=coverage,
        blackouts=blackouts,
        quality_policy=quality.QualityPolicy(
            "observed_only_blackout_v1", post_gap_diagnostics=True, post_gap_event_limit=limit
        ),
        commission_bps=0,
        slippage_bps=0,
    )


def test_post_gap_diagnostics_are_opt_in_and_do_not_change_financial_results():
    data = fixture([{"missing": [0]}, {}])
    off = run(data)
    on = diagnostic_run(data)
    assert not any(
        key.startswith("post_gap") or key == "has_post_gap_confirmed_signal" for key in off.metrics
    )
    assert "post_gap_signal_event_metadata" not in off.diagnostics.attrs
    pd.testing.assert_series_equal(off.daily_equity, on.daily_equity)
    pd.testing.assert_series_equal(off.daily_returns, on.daily_returns)
    pd.testing.assert_series_equal(off.daily_cash_returns, on.daily_cash_returns)
    pd.testing.assert_frame_equal(off.trades, on.trades)
    for key in off.metrics:
        if isinstance(off.metrics[key], float):
            assert off.metrics[key] == pytest.approx(on.metrics[key], nan_ok=True)
        else:
            assert off.metrics[key] == on.metrics[key]
    assert on.metrics["post_gap_confirmed_buy_count"] == 1
    assert (
        on.metrics["post_gap_confirmed_sell_count"] == 0
    )  # Forced terminal liquidation is not a signal.


def test_post_gap_flag_uses_positive_time_fresh_confirmation_not_initial_support_or_cancelled_order():
    data = fixture(
        [
            {"closes": [100] * 12 + [110], "daily_close": 100},
            {"missing": [0], "closes": [110] * 13},
            {"closes": [110] * 13},
        ]
    )
    result = diagnostic_run(data)
    events = result.diagnostics.attrs["post_gap_signal_events"]
    assert result.metrics["cancelled_blackout_order_count"] == 1
    assert len(events) == 1
    assert events.decision_at.iloc[0].tz_convert("America/New_York") == pd.Timestamp(
        "2000-01-04 10:30", tz="America/New_York"
    )
    assert events.gap_end.iloc[0].tz_convert("America/New_York") == pd.Timestamp(
        "2000-01-04 10:00", tz="America/New_York"
    )
    assert (events.decision_at > events.gap_end).all()
    waited = diagnostic_run(data, c(en=1))
    assert waited.diagnostics.attrs["post_gap_signal_events"].decision_at.iloc[0].tz_convert(
        "America/New_York"
    ) == pd.Timestamp("2000-01-04 11:30", tz="America/New_York")
    assert (
        not waited.diagnostics.attrs["post_gap_signal_events"]
        .decision_at.eq(pd.Timestamp("2000-01-04 10:30", tz="America/New_York").tz_convert("UTC"))
        .any()
    )


def test_next_session_close_inclusive_decision_can_fill_later_without_a_trade_flag_requirement():
    data = fixture(
        [
            {"missing": [0], "closes": [100] * 13},
            {"closes": [100] * 12 + [110]},
            {"closes": [110] * 13},
        ]
    )
    result = diagnostic_run(data)
    events = result.diagnostics.attrs["post_gap_signal_events"]
    assert len(events) == 1
    assert events.decision_at.iloc[0] == events.window_end.iloc[0]
    assert events.decision_at.iloc[0].tz_convert("America/New_York") == pd.Timestamp(
        "2000-01-04 16:00", tz="America/New_York"
    )
    assert result.trades.iloc[0].timestamp.tz_convert("America/New_York") == pd.Timestamp(
        "2000-01-05 09:30", tz="America/New_York"
    )
    assert result.metrics["post_gap_confirmed_signal_count"] == 1


def test_confirmed_buy_and_sell_counts_include_only_decisions_within_unknown_gap_window():
    data = fixture(
        [
            {"missing": [0], "closes": [110] * 3 + [90] * 10, "daily_close": 90},
            {"closes": [90] * 13},
        ]
    )
    result = diagnostic_run(data, c(en=0, ex=0))
    events = result.diagnostics.attrs["post_gap_signal_events"]
    assert events.side.tolist() == ["buy", "sell"]
    assert result.metrics["post_gap_confirmed_buy_count"] == 1
    assert result.metrics["post_gap_confirmed_sell_count"] == 1
    assert result.metrics["post_gap_confirmed_signal_count"] == 2
    late = fixture(
        [{"missing": [0], "closes": [100] * 13}, {"closes": [100] * 13}, {"closes": [110] * 13}]
    )
    after = diagnostic_run(late)
    assert after.metrics["post_gap_confirmed_signal_count"] == 0
    assert after.trades.iloc[0].reason == "confirmed_entry"


def test_weekend_and_early_close_boundary_uses_next_trading_session_not_24_hours_or_17_mark():
    data = fixture(
        [
            {"session": "2024-11-27", "missing": [0], "closes": [100] * 13},
            {
                "session": "2024-11-29",
                "closes": [100] * 6 + [110],
            },  # Actual next XNYS session is an early close.
            {"session": "2024-12-02", "closes": [110] * 13},
        ]
    )
    result = diagnostic_run(data)
    event = result.diagnostics.attrs["post_gap_signal_events"].iloc[0]
    assert event.window_end.tz_convert("America/New_York") == pd.Timestamp(
        "2024-11-29 13:00", tz="America/New_York"
    )
    assert event.decision_at == event.window_end
    assert event.window_end - event.gap_end > pd.Timedelta(hours=24)
    assert result.trades.iloc[0].timestamp.tz_convert("America/New_York") == pd.Timestamp(
        "2024-12-02 09:30", tz="America/New_York"
    )


def test_old_gap_does_not_expand_into_later_scored_segment_when_full_calendar_has_next_session():
    data = fixture(
        [
            {"missing": [0], "closes": [100] * 13},
            {"closes": [100] * 13},
            {"closes": [110] * 13},
            {"closes": [120] * 13},
        ]
    )
    result = diagnostic_run(data, start="2000-01-05", end="2000-01-06")
    assert not result.metrics["has_post_gap_confirmed_signal"]
    meta = result.diagnostics.attrs["post_gap_signal_event_metadata"]
    assert meta["active_window_count"] == 0
    assert meta["windows"][0]["next_session"] == "2000-01-04"
    assert result.trades.iloc[0].side == "buy"


def test_known_halt_never_creates_post_unknown_gap_flags():
    data, blackouts = halt_fixture()
    result = diagnostic_run(data, blackouts=blackouts)
    assert not result.trades.empty
    assert result.metrics["post_gap_confirmed_signal_count"] == 0
    assert result.diagnostics.attrs["post_gap_signal_event_metadata"]["window_count"] == 0
    assert result.diagnostics.attrs["post_gap_signal_events"].empty


def test_overlapping_windows_deduplicate_candidate_counts_but_store_explicit_gap_associations():
    data = fixture([{"missing": [0, 2]}, {}])
    result = diagnostic_run(data, c(en=1))
    meta = result.diagnostics.attrs["post_gap_signal_event_metadata"]
    events = result.diagnostics.attrs["post_gap_signal_events"]
    assert result.metrics["post_gap_confirmed_signal_count"] == 1
    assert len(events) == meta["total_event_rows"] == 2
    assert meta["total_confirmed_decisions"] == 1
    assert events.decision_at.nunique() == 1 and events.gap_start.nunique() == 2


def test_grid_record_false_counts_every_candidate_and_global_event_cap_reports_truncation():
    data = fixture([{"missing": [0]}, {}])
    candidates = [c(en=0), c(en=0.5), c(en=1)]
    policy = quality.QualityPolicy(
        "observed_only_blackout_v1", post_gap_diagnostics=True, post_gap_event_limit=1
    )
    table = quality.evaluate_quality_grid(
        data[0],
        data[1],
        start="2000-01-03",
        end="2000-01-04",
        cash_observations=rates(data[2]),
        calendar=data[2],
        coverage=data[3],
        candidates=candidates,
        quality_policy=policy,
        batch_size=1,
        commission_bps=0,
        slippage_bps=0,
    )
    assert table.post_gap_confirmed_signal_count.tolist() == [1, 1, 1]
    assert table.has_post_gap_confirmed_signal.all()
    meta = table.attrs["post_gap_signal_event_metadata"]
    assert meta["total_event_rows"] == meta["total_confirmed_decisions"] == 3
    assert meta["stored_event_rows"] == 1 and meta["truncated"]
    assert len(table.attrs["post_gap_signal_events"]) == 1
    assert tuple(table.attrs["post_gap_signal_events"].columns) == quality.POST_GAP_EVENT_COLUMNS
    assert not any("price" in column for column in table.attrs["post_gap_signal_events"])


def test_last_staged_date_generates_calendar_time_only_and_reports_right_censoring():
    data = fixture([{"missing": [0]}])
    result = diagnostic_run(data)
    window = result.diagnostics.attrs["post_gap_signal_event_metadata"]["windows"][0]
    assert window["next_session"] == "2000-01-04"
    assert window["next_session_time_source"] == "pinned_XNYS_time_only_calendar_helper"
    assert window["right_censored_by_scored_end"] and window["right_censored_by_staged_end"]
    assert result.metrics["post_gap_confirmed_signal_count"] == 1


@pytest.mark.parametrize("bad", [0, -1, True, np.nan, 2500001, 1.5])
def test_diagnostic_event_limit_is_bounded_and_strict_does_not_enable_new_flags(bad):
    with pytest.raises(ValueError, match="post_gap_event_limit"):
        quality.QualityPolicy(post_gap_event_limit=bad)
    with pytest.raises(ValueError, match="authorized observed-only"):
        quality.QualityPolicy(post_gap_diagnostics=True)
