"""Full-grid flag coverage on fabricated prices, not historical performance."""

import pandas as pd

from trend_following import research_quality_engine_v5 as engine


def test_every_configuration_gets_an_exact_post_gap_flag_without_record_mode():
    daily = [
        dict(
            session=day.strftime("%Y-%m-%d"),
            close=100.0,
            dividend_amount=0.0,
            split_coefficient=1.0,
        )
        for day in pd.bdate_range("1999-11-01", periods=300)
    ]
    calendar, bars, coverage = [], [], []
    for session in ("2002-01-02", "2002-01-03"):
        start = pd.Timestamp(f"{session} 09:30", tz="America/New_York").tz_convert("UTC")
        calendar.append(
            dict(session=session, market_open=start, market_close=start + pd.Timedelta(hours=6.5))
        )
        daily.append(dict(session=session, close=110.0, dividend_amount=0.0, split_coefficient=1.0))
        for i in range(13):
            opening = start + pd.Timedelta(minutes=30 * i)
            closing = opening + pd.Timedelta(minutes=30)
            absent = session == "2002-01-02" and i == 0
            coverage.append(
                dict(
                    session=session,
                    bar_start=opening,
                    bar_end=closing,
                    status="unknown_missing" if absent else "observed",
                    signal_eligible=not absent,
                    fill_eligible=not absent,
                )
            )
            if not absent:
                bars.append(
                    dict(
                        session=session,
                        bar_start=opening,
                        bar_end=closing,
                        open=100.0,
                        high=110.0,
                        low=100.0,
                        close=110.0,
                        volume=1.0,
                        duration_minutes=30,
                        is_completed=True,
                        is_full_hour=False,
                    )
                )
    daily, bars, calendar, coverage = map(pd.DataFrame, (daily, bars, calendar, coverage))
    rates = pd.DataFrame(
        dict(
            available_at=[calendar.market_open.iloc[0] - pd.Timedelta(days=1)],
            annual_rate=[0.0],
            series_id=["DTB3"],
        )
    )
    candidates = engine.generate_candidates()
    metrics = engine.evaluate_quality_grid(
        daily,
        bars,
        calendar=calendar,
        coverage=coverage,
        start="2002-01-02",
        end="2002-01-03",
        cash_observations=rates,
        candidates=candidates,
        quality_policy=engine.QualityPolicy(
            "observed_only_blackout_v1", post_gap_diagnostics=True, post_gap_event_limit=2_500_000
        ),
    )
    assert len(metrics) == 56_644
    assert set(metrics.candidate_id) == {candidate.candidate_id for candidate in candidates}
    assert metrics.has_post_gap_confirmed_signal.all()
    assert metrics.post_gap_confirmed_buy_count.eq(1).all()
    assert metrics.post_gap_confirmed_sell_count.eq(0).all()
    assert metrics.post_gap_confirmed_signal_count.eq(1).all()
    events = metrics.attrs["post_gap_signal_events"]
    metadata = metrics.attrs["post_gap_signal_event_metadata"]
    assert len(events) == events.candidate_id.nunique() == 56_644
    assert metadata["total_event_rows"] == metadata["stored_event_rows"] == 56_644
    assert metadata["truncated"] is False
    assert events.decision_at.gt(events.gap_end).all()
    assert events.decision_at.le(events.window_end).all()
    assert events.window_end.eq(calendar.market_close.iloc[-1]).all()
