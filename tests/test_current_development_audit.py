"""Fabricated tests for the independent audit; imports no trading kernel."""

import numpy as np
import pandas as pd
import pytest
from scripts import audit_current_development as audit


def metric_flags(ids, counts):
    return pd.DataFrame(
        [
            dict(
                candidate_id=key,
                post_gap_confirmed_buy_count=count,
                post_gap_confirmed_sell_count=0,
                post_gap_confirmed_signal_count=count,
                has_post_gap_confirmed_signal=count > 0,
            )
            for key, count in zip(ids, counts, strict=True)
        ]
    )


def event_inputs():
    g0 = pd.Timestamp("2001-01-02 09:30", tz="America/New_York").tz_convert("UTC")
    g1 = g0 + pd.Timedelta(minutes=30)
    window = pd.Timestamp("2001-01-03 16:00", tz="America/New_York").tz_convert("UTC")
    decision = g1 + pd.Timedelta(minutes=30)
    flags = metric_flags(["a", "b"], [1, 0])
    events = pd.DataFrame([("a", "buy", decision, g0, g1, window)], columns=audit.EVENT_COLUMNS)
    metadata = dict(
        enabled=True,
        event_limit=2500000,
        truncated=False,
        total_event_rows=1,
        stored_event_rows=1,
        descriptive_only=True,
        used_for_filtering_ranking_or_selection=False,
        total_confirmed_decisions=1,
    )
    cal = pd.DataFrame(
        {
            "session": ["2001-01-02", "2001-01-03"],
            "market_close": [window - pd.Timedelta(days=1), window],
        }
    )
    coverage = pd.DataFrame({"bar_end": [decision], "signal_eligible": [True]})
    blackout = pd.DataFrame({"start": [g0], "end": [g1], "kind": ["unknown_missing"]})
    return flags, events, metadata, cal, coverage, blackout


def test_independent_unrestricted_grid():
    grid = audit.independently_expected_grid()
    assert len(grid) == 56644
    assert len({(x["entry_rule"]["family"], x["exit_rule"]["family"]) for x in grid.values()}) == 9
    assert (
        sum(
            x["entry_rule"]["family"] == "MACD" and x["exit_rule"]["family"] == "MACD"
            for x in grid.values()
        )
        == 4900
    )
    assert (
        sum(
            x["entry_rule"]["family"] == "SMA" and x["exit_rule"]["family"] == "EMA"
            for x in grid.values()
        )
        == 7056
    )


def test_selector_does_not_filter_flagged_winner_or_use_secondary_cagr():
    frame = pd.DataFrame(
        [
            dict(
                candidate_id="b",
                entry_family="SMA",
                exit_family="EMA",
                cash_excess_sharpe=1.0,
                cagr=100,
                has_post_gap_confirmed_signal=False,
            ),
            dict(
                candidate_id="a",
                entry_family="SMA",
                exit_family="EMA",
                cash_excess_sharpe=1.0 - 5e-11,
                cagr=-100,
                has_post_gap_confirmed_signal=True,
            ),
            dict(
                candidate_id="c",
                entry_family="EMA",
                exit_family="SMA",
                cash_excess_sharpe=np.nan,
                cagr=999,
                has_post_gap_confirmed_signal=False,
            ),
        ]
    )
    selected, empty = audit.independently_select_finalists(frame)
    assert selected == ["a"] and "EMA/SMA" in empty


def test_complete_diagnostics_and_deleted_event_detection():
    flags, events, meta, cal, coverage, blackout = event_inputs()
    result = audit.audit_gap_diagnostics(
        flags.copy(), flags.copy(), events.copy(), meta, cal, coverage, blackout
    )
    assert result["flagged_candidates"] == 1 and result["distinct_confirmed_decisions"] == 1
    bad = events.iloc[:0].copy()
    changed = {**meta, "total_event_rows": 0, "stored_event_rows": 0}
    with pytest.raises(audit.AuditFailure, match="retained events"):
        audit.audit_gap_diagnostics(
            flags.copy(), flags.copy(), bad, changed, cal, coverage, blackout
        )


def test_duplicate_gap_association_and_wrong_boundary_are_rejected():
    flags, events, meta, cal, coverage, blackout = event_inputs()
    repeated = pd.concat([events, events], ignore_index=True)
    with pytest.raises(audit.AuditFailure, match="Duplicated"):
        audit.audit_gap_diagnostics(
            flags.copy(),
            flags.copy(),
            repeated,
            {**meta, "total_event_rows": 2, "stored_event_rows": 2},
            cal,
            coverage,
            blackout,
        )
    wrong = events.copy()
    wrong["decision_at"] = wrong.gap_end
    with pytest.raises(audit.AuditFailure, match="positive-time"):
        audit.audit_gap_diagnostics(
            flags.copy(), flags.copy(), wrong, meta, cal, coverage, blackout
        )


def test_truncation_unsafe_signal_and_known_halt_association_are_rejected():
    flags, events, meta, cal, coverage, blackout = event_inputs()
    with pytest.raises(audit.AuditFailure, match="Truncated"):
        audit.audit_gap_diagnostics(
            flags.copy(),
            flags.copy(),
            events.copy(),
            {**meta, "truncated": True},
            cal,
            coverage,
            blackout,
        )
    with pytest.raises(audit.AuditFailure, match="safe observed"):
        audit.audit_gap_diagnostics(
            flags.copy(),
            flags.copy(),
            events.copy(),
            meta,
            cal,
            coverage.assign(signal_eligible=False),
            blackout,
        )
    with pytest.raises(audit.AuditFailure, match="known-halt"):
        audit.audit_gap_diagnostics(
            flags.copy(),
            flags.copy(),
            events.copy(),
            meta,
            cal,
            coverage,
            blackout.assign(kind="known_halt"),
        )


def test_persisted_curve_independent_ddof_and_clock_validation():
    dates = ["2001-01-02", "2001-01-03", "2001-01-04"]
    stamps = [
        pd.Timestamp(day + " 17:00", tz="America/New_York").tz_convert("UTC") for day in dates
    ]
    ret = pd.Series([0.1, -0.1, 0.2])
    cash = pd.Series([0.001, 0.001, 0.001])
    eq = 1000 * (1 + ret).cumprod()
    frame = pd.DataFrame(dict(timestamp=stamps, **{"return": ret}, equity=eq, cash_return=cash))
    cal = pd.DataFrame(
        dict(
            session=dates,
            market_open=[
                pd.Timestamp(day + " 09:30", tz="America/New_York").tz_convert("UTC")
                for day in dates
            ],
        )
    )
    excess = ret - cash
    years = (stamps[-1] - cal.market_open.iloc[0]).total_seconds() / (365 * 86400)
    wealth = np.r_[1000.0, eq]
    dd = (wealth / np.maximum.accumulate(wealth) - 1).min()
    metric = dict(
        cash_excess_sharpe=np.sqrt(252) * excess.mean() / excess.std(ddof=1),
        annualized_volatility=np.sqrt(252) * ret.std(ddof=1),
        max_drawdown=dd,
        terminal_equity=eq.iloc[-1],
        cash_terminal_equity=1000 * np.prod(1 + cash),
        start_session=dates[0],
        end_session=dates[-1],
        scored_sessions=3,
        cagr=(eq.iloc[-1] / 1000) ** (1 / years) - 1,
    )
    assert audit.audit_curve(frame, metric, cal)["ddof"] == 1
    with pytest.raises(audit.AuditFailure, match="Sharpe/ddof"):
        audit.audit_curve(
            frame,
            {**metric, "cash_excess_sharpe": np.sqrt(252) * excess.mean() / excess.std(ddof=0)},
            cal,
        )
    shifted = frame.copy()
    shifted["timestamp"] = pd.to_datetime(shifted.timestamp) + pd.Timedelta(hours=1)
    with pytest.raises(audit.AuditFailure, match="clock/period"):
        audit.audit_curve(shifted, metric, cal)
