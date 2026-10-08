"""Independent cross-module acceptance checks on fabricated data only."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from trend_following import research_daily_hourly_v5 as engine
from trend_following import research_hourly_data_v5 as data
from trend_following import research_quality_protocol_v5 as protocol


def fabricated_panel():
    calendar = data.build_xnys_calendar("1999-11-01", "2001-01-04")
    daily = pd.DataFrame(
        {
            "session": calendar.session,
            "close": 100.0 + np.arange(len(calendar)) * 0.01,
            "dividend_amount": 0.0,
            "split_coefficient": 1.0,
        }
    )
    grid = data.canonical_half_hour_grid(calendar.loc[calendar.session.ge("2001-01-02")])
    records = []
    for session, group in grid.groupby("session", sort=True):
        position = int(daily.index[daily.session.eq(session)][0])
        previous = float(daily.close.iloc[position - 1])
        opening = previous
        for number, row in enumerate(group.itertuples(index=False)):
            close = previous + 0.1 + number * 0.025
            records.append(
                {
                    "bar_start": row.bar_start,
                    "bar_end": row.bar_end,
                    "session": session,
                    "open": opening,
                    "high": max(opening, close),
                    "low": min(opening, close),
                    "close": close,
                    "volume": 100,
                    "is_full_hour": row.is_full_hour,
                    "duration_minutes": row.duration_minutes,
                }
            )
            opening = close
        daily.loc[position, "close"] = opening
    bars = pd.DataFrame(records)
    timestamps = pd.date_range(
        bars.bar_start.iloc[0].normalize() - pd.Timedelta(days=1),
        bars.bar_end.iloc[-1].normalize(),
        freq="D",
    )
    cash = pd.DataFrame({"available_at": timestamps, "annual_rate": 0.025, "series_id": "DTB3"})
    return daily, bars, calendar, cash


def test_all_family_cells_and_units_match_frozen_configuration():
    config = protocol.load_config(
        protocol.WORKSPACE / "configs/qqq_daily_halfhour_v5_gap_flags_v2.yaml"
    )
    candidates = engine.generate_candidates()
    cells = Counter(candidate.family_cell for candidate in candidates)
    assert cells == {
        "SMA/SMA": 7056,
        "SMA/EMA": 7056,
        "SMA/MACD": 5880,
        "EMA/SMA": 7056,
        "EMA/EMA": 7056,
        "EMA/MACD": 5880,
        "MACD/SMA": 5880,
        "MACD/EMA": 5880,
        "MACD/MACD": 4900,
    }
    assert config["grid"]["indicator_unit"] == "daily_sessions"
    assert config["grid"]["entry_confirmation_hours"] == [i / 2 for i in range(14)]
    assert config["grid"]["exit_confirmation_hours"] == [i / 2 for i in range(14)]
    assert config["data_contract"]["sampling_interval_minutes"] == 30
    assert config["grid"]["candidate_count"] == sum(cells.values()) == 56644
    assert 300 not in config["grid"]["ma_periods"]
    assert config["accounting"]["commission_bps_per_order"] == 1
    assert config["accounting"]["baseline_slippage_bps"] == 3
    assert config["accounting"]["tax_rate"] == 0


def test_entire_grid_batch_and_single_ledger_reconcile():
    daily, bars, calendar, cash = fabricated_panel()
    cache = engine.build_support_cache(daily, bars, calendar)
    candidates = engine.generate_candidates()
    kwargs = dict(
        start="2001-01-01",
        end="2001-01-04",
        cash_observations=cash,
        commission_bps=1,
        slippage_bps=3,
        calendar=calendar,
        support_cache=cache,
    )
    table = engine.evaluate_grid(daily, bars, candidates=candidates, batch_size=171, **kwargs)
    assert len(table) == table.candidate_id.nunique() == 56644
    assert table.terminal_equity.gt(0).all()
    assert table.scored_sessions.eq(3).all()
    for candidate in (candidates[0], candidates[2345], candidates[-1]):
        result = engine.simulate_candidate(daily, bars, candidate, **kwargs)
        row = table.set_index("candidate_id").loc[candidate.candidate_id]
        for key in ("terminal_equity", "total_commission", "total_slippage", "total_turnover"):
            assert row[key] == pytest.approx(result.metrics[key], abs=1e-10)
        assert np.prod(1 + result.daily_returns) * 1000 == pytest.approx(
            result.metrics["terminal_equity"]
        )
        if np.isfinite(result.metrics["cash_excess_sharpe"]):
            excess = result.daily_returns - result.daily_cash_returns
            assert row.cash_excess_sharpe == pytest.approx(
                np.sqrt(252) * excess.mean() / excess.std(ddof=1)
            )


def test_same_day_future_official_quote_cannot_change_earlier_hourly_preview():
    daily, bars, calendar, _ = fabricated_panel()
    first = engine.build_support_cache(daily, bars, calendar)
    altered = daily.copy(deep=True)
    altered.loc[altered.session.eq("2001-01-02"), "close"] += 0.01
    second = engine.build_support_cache(altered, bars, calendar)
    today = bars.session.eq("2001-01-02")
    np.testing.assert_allclose(first.margins[today], second.margins[today], equal_nan=True)
    # Tomorrow may change, because that daily quote has then actually finalized.
    assert not np.allclose(first.margins[~today], second.margins[~today], equal_nan=True)
    pd.testing.assert_frame_equal(daily, first.daily)


def test_live_preview_and_full_historical_cache_agree_without_today_daily_quote():
    daily, bars, calendar, _ = fabricated_panel()
    cache = engine.build_support_cache(daily, bars, calendar)
    completed = daily.loc[daily.session.lt("2001-01-02")].copy()
    before = completed.copy(deep=True)
    for position in range(13):
        preview = engine.preview_daily_margins(completed, float(bars.close.iloc[position]))
        np.testing.assert_allclose(preview.to_numpy(), cache.margins[position], equal_nan=True)
        assert preview.attrs["prior_session_count"] == 295
    pd.testing.assert_frame_equal(completed, before)


def test_verified_announced_late_open_is_not_filled_with_nonexistent_preopen_hours():
    calendar = data.build_xnys_calendar("2002-09-11", "2002-09-11")
    opening = calendar.market_open.iloc[0].tz_convert("America/New_York")
    assert (opening.hour, opening.minute) == (11, 0)
    grid = data.canonical_half_hour_grid(calendar)
    assert len(grid) == 10
    assert not grid.is_full_hour.any()
    assert grid.duration_minutes.eq(30).all()
    assert grid.bar_start.iloc[0] == calendar.market_open.iloc[0]
    assert grid.bar_end.iloc[-1] == calendar.market_close.iloc[0]
