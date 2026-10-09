"""Fabricated tests only; never import a strategy or access a real study."""

import numpy as np
import pandas as pd
import pytest
from scripts import audit_current_development as base
from scripts import audit_current_validation_oos as audit


def test_finite_primary_tie_rule_ignores_flags_and_cagr():
    frame = pd.DataFrame(
        [
            dict(
                candidate_id="b",
                cash_excess_sharpe=1.0,
                cagr=99,
                has_post_gap_confirmed_signal=False,
            ),
            dict(
                candidate_id="a",
                cash_excess_sharpe=1.0 - 5e-11,
                cagr=-99,
                has_post_gap_confirmed_signal=True,
            ),
            dict(
                candidate_id="c",
                cash_excess_sharpe=np.nan,
                cagr=999,
                has_post_gap_confirmed_signal=False,
            ),
        ]
    )
    assert audit.pick_primary(frame) == "a"
    with pytest.raises(base.AuditFailure, match="No finite"):
        audit.pick_primary(frame.iloc[2:])


def test_oos_numeric_reads_cannot_precede_winner_lock_check(monkeypatch):
    def denied(*args, **kwargs):
        raise base.AuditFailure("missing winner lock")

    def forbidden(*args, **kwargs):
        raise AssertionError("numeric file read before winner lock")

    monkeypatch.setattr(audit, "validation_audit", denied)
    monkeypatch.setattr(base, "csv", forbidden)
    with pytest.raises(base.AuditFailure, match="missing winner lock"):
        audit.oos_audit("fake-study", "fake-config")


def test_wrong_validation_winner_fails_before_any_oos_read(tmp_path, monkeypatch):
    ids = [f"a{x}" for x in range(9)]
    metric = pd.DataFrame(dict(candidate_id=ids, cash_excess_sharpe=list(range(9, 0, -1))))
    finalists = {"candidates": [dict(candidate_id=x) for x in ids]}
    expected = {x: dict(candidate_id=x) for x in ids}
    config = dict(
        segments={"validation": dict(start="2012-01-01", end="2015-12-31")},
        selection=dict(tie_tolerance=1e-10),
    )
    monkeypatch.setattr(
        audit, "context", lambda *args: (tmp_path, config, {}, {}, expected, finalists)
    )
    monkeypatch.setattr(audit, "stage_seal", lambda *args: {})
    monkeypatch.setattr(audit, "check_metrics", lambda *args: (pd.DataFrame(), 4.0))
    seen = []

    def reader(path):
        seen.append(str(path))
        assert "historical-oos" not in str(path)
        return metric if str(path).endswith("metrics.csv") else pd.DataFrame()

    monkeypatch.setattr(base, "csv", reader)
    monkeypatch.setattr(base, "read_seal", lambda *args: {"candidates": [dict(candidate_id="a1")]})
    with pytest.raises(base.AuditFailure, match="independent primary-only"):
        audit.validation_audit(tmp_path, "fake-config")
    assert all("historical-oos" not in x for x in seen)


def test_curve_math_uses_cash_excess_ddof1_and_matching17_clock():
    sessions = ["2016-01-04", "2016-01-05", "2016-01-06"]
    calendar = pd.DataFrame(dict(session=sessions))
    returns = pd.Series([0.02, -0.01, 0.03])
    cash = pd.Series([0.001, 0.001, 0.001])
    equity = 1000 * (1 + returns).cumprod()
    dates = [
        pd.Timestamp(day + " 17:00", tz="America/New_York").tz_convert("UTC") for day in sessions
    ]
    frame = pd.DataFrame(
        dict(timestamp=dates, strategy_return=returns, cash_return=cash, equity=equity)
    )
    excess = returns - cash
    wealth = np.r_[1000, equity.to_numpy()]
    metric = dict(
        cash_excess_sharpe=np.sqrt(252) * excess.mean() / excess.std(ddof=1),
        annualized_volatility=returns.std(ddof=1) * np.sqrt(252),
        max_drawdown=(wealth / np.maximum.accumulate(wealth) - 1).min(),
        terminal_equity=equity.iloc[-1],
        cash_terminal_equity=1000 * np.prod(1 + cash),
    )
    assert audit.curve_statistics(frame, metric, calendar)["terminal_equity"] == pytest.approx(
        equity.iloc[-1]
    )
    with pytest.raises(base.AuditFailure, match="cash_excess_sharpe"):
        audit.curve_statistics(
            frame,
            {**metric, "cash_excess_sharpe": np.sqrt(252) * excess.mean() / excess.std(ddof=0)},
            calendar,
        )
    shifted = frame.copy()
    shifted.timestamp += pd.Timedelta(hours=1)
    with pytest.raises(base.AuditFailure, match="endpoints"):
        audit.curve_statistics(shifted, metric, calendar)


def test_replacement_candidate_and_wrong_cost_fail_before_metrics_math():
    frame = pd.DataFrame(
        dict(candidate_id=["replacement"], commission_bps=[1.0], slippage_bps=[3.0])
    )
    with pytest.raises(base.AuditFailure, match="sealed allowed set"):
        audit.check_metrics(
            frame, pd.DataFrame(), dict(start="2016-01-01", end="2026-05-28"), ["locked"], 3
        )
    frame.candidate_id = "locked"
    frame.slippage_bps = 10.0
    with pytest.raises(base.AuditFailure, match="transaction costs"):
        audit.check_metrics(
            frame, pd.DataFrame(), dict(start="2016-01-01", end="2026-05-28"), ["locked"], 3
        )
