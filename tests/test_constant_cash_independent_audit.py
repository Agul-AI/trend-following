"""Independent derived-series math checks using fabricated observations only.

The helper below can also audit already-released DEVELOPMENT-only arrays. It
does not call simulation, selection, data loading, or production score helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_following import research_constant_cash as diagnostic


def independent_nominal_cash_ledger(clock, *, initial_cash=1000.0):
    """Fixed 3% nominal ACT/365 ledger; no rate provider or accounting API."""
    clock = pd.DatetimeIndex(clock)
    if (
        clock.tz is None
        or clock.hasnans
        or clock.has_duplicates
        or not clock.is_monotonic_increasing
        or len(clock) == 0
        or not np.isfinite(initial_cash)
        or initial_cash <= 0
    ):
        raise ValueError("Require an ordered unique timezone-aware event clock")
    seconds = np.r_[0.0, np.diff(clock.asi8).astype(float) / 1e9]
    return initial_cash * np.cumprod(1.0 + 0.03 * seconds / (365.0 * 86400.0))


def independent_daily_statistics(equity, cash_returns, *, initial_cash, elapsed_years):
    """Recompute session-close statistics from derived portfolio values only."""
    wealth = np.asarray(equity, dtype=float)
    reference_returns = np.asarray(cash_returns, dtype=float)
    if wealth.ndim == 1:
        wealth = wealth[:, None]
    if (
        wealth.ndim != 2
        or wealth.shape[0] < 2
        or reference_returns.shape != (wealth.shape[0],)
        or not np.isfinite(wealth).all()
        or not np.isfinite(reference_returns).all()
        or (wealth <= 0).any()
        or not np.isfinite(initial_cash)
        or initial_cash <= 0
        or not np.isfinite(elapsed_years)
        or elapsed_years <= 0
    ):
        raise ValueError("Require finite positive derived wealth and a matching cash clock")
    previous = np.vstack([np.full(wealth.shape[1], initial_cash), wealth[:-1]])
    returns = wealth / previous - 1.0
    excess = returns - reference_returns[:, None]
    sample_sd = excess.std(axis=0, ddof=1)
    sharpe = np.full(wealth.shape[1], np.nan)
    np.divide(np.sqrt(252.0) * excess.mean(axis=0), sample_sd, out=sharpe, where=sample_sd > 0)
    full_wealth = np.vstack([np.full(wealth.shape[1], initial_cash), wealth])
    drawdown = full_wealth / np.maximum.accumulate(full_wealth, axis=0) - 1
    return {
        "returns": returns,
        "cash_excess_sharpe": sharpe,
        "cagr": (wealth[-1] / initial_cash) ** (1.0 / elapsed_years) - 1.0,
        "max_drawdown": drawdown.min(axis=0),
        "annualized_volatility": returns.std(axis=0, ddof=1) * np.sqrt(252.0),
        "terminal_equity": wealth[-1],
    }


def independent_beat_counts(scores, benchmark):
    """Explicit full-denominator comparison independent of publication code."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (289,) or not np.isfinite(benchmark):
        raise ValueError("Expected all 289 pairs and a finite cost-matched benchmark")
    finite = np.isfinite(scores)
    return {
        "configuration_count": 289,
        "finite_Sharpe_count": sum(bool(x) for x in finite),
        "undefined_Sharpe_count": sum(not bool(x) for x in finite),
        "beat_BH_Sharpe_count": sum(
            bool(valid and score - benchmark > 1e-10)
            for valid, score in zip(finite, scores, strict=True)
        ),
        "strict_gt_BH_Sharpe_count": sum(
            bool(valid and score > benchmark) for valid, score in zip(finite, scores, strict=True)
        ),
        "tie_BH_Sharpe_count": sum(
            bool(valid and abs(score - benchmark) <= 1e-10)
            for valid, score in zip(finite, scores, strict=True)
        ),
        "below_BH_Sharpe_count": sum(
            bool(valid and benchmark - score > 1e-10)
            for valid, score in zip(finite, scores, strict=True)
        ),
    }


def test_independent_cash_ledger_includes_weekend_and_dst_seconds():
    clock = pd.DatetimeIndex(
        [
            pd.Timestamp("2001-04-06 16:00", tz="America/New_York"),
            pd.Timestamp("2001-04-09 09:30", tz="America/New_York"),
            pd.Timestamp("2001-04-09 10:00", tz="America/New_York"),
        ]
    )
    ledger = independent_nominal_cash_ledger(clock)
    expected = 1000 * (1 + 0.03 * 65.5 / (365 * 24)) * (1 + 0.03 * 0.5 / (365 * 24))
    assert ledger[0] == 1000
    assert ledger[-1] == pytest.approx(expected)
    # A weekend with a daylight-saving transition is one calendar hour shorter.
    transition_clock = pd.DatetimeIndex(
        [
            pd.Timestamp("2001-03-30 16:00", tz="America/New_York"),
            pd.Timestamp("2001-04-02 09:30", tz="America/New_York"),
        ]
    )
    transition = independent_nominal_cash_ledger(transition_clock)
    assert transition[-1] == pytest.approx(1000 * (1 + 0.03 * 64.5 / (365 * 24)))


def test_independent_nominal_cash_is_not_exact_three_percent_apy():
    clock = pd.date_range("2001-01-01 00:00Z", "2002-01-01 00:00Z", freq="D")
    ledger = independent_nominal_cash_ledger(clock)
    assert ledger[-1] == pytest.approx(1000 * (1 + 0.03 / 365) ** 365)
    assert 3.04 < 100 * (ledger[-1] / ledger[0] - 1) < 3.05


def test_independent_cash_excess_sample_sd_and_first_return():
    returns = np.array([0.01, -0.01, 0.03, 0.02])
    cash = np.array([0.001, 0.003, 0.001, 0.001])
    equity = 1000 * np.cumprod(1 + returns)
    result = independent_daily_statistics(equity, cash, initial_cash=1000, elapsed_years=1)
    expected_excess = np.array([0.009, -0.013, 0.029, 0.019])
    expected = np.sqrt(252) * expected_excess.mean() / expected_excess.std(ddof=1)
    assert result["cash_excess_sharpe"][0] == pytest.approx(expected)
    np.testing.assert_allclose(result["returns"][:, 0], returns, atol=1e-15)
    # Neither total-return Sharpe nor subtracting 0.03/252 implements the ledger.
    assert result["cash_excess_sharpe"][0] != pytest.approx(
        np.sqrt(252) * returns.mean() / returns.std(ddof=1)
    )
    assert result["cash_excess_sharpe"][0] != pytest.approx(
        np.sqrt(252) * (returns - 0.03 / 252).mean() / (returns - 0.03 / 252).std(ddof=1)
    )


def test_initial_cash_is_included_in_drawdown_peak():
    result = independent_daily_statistics(
        np.array([900, 810, 850]), np.zeros(3), initial_cash=1000, elapsed_years=1
    )
    assert result["max_drawdown"][0] == pytest.approx(-0.19)
    assert result["cagr"][0] == pytest.approx(-0.15)
    assert result["terminal_equity"][0] == 850


def test_cash_benchmark_excess_is_undefined_not_zero():
    equity = np.array([1001.0, 1001.25, 1001.5])
    cash = equity / np.r_[1000, equity[:-1]] - 1
    result = independent_daily_statistics(equity, cash, initial_cash=1000, elapsed_years=1)
    assert np.isnan(result["cash_excess_sharpe"][0])
    assert result["max_drawdown"][0] == 0


def test_independent_statistics_support_all_289_in_a_single_vectorized_cohort():
    returns = np.array([[0.01, 0.02], [-0.01, 0.01], [0.02, -0.01]])
    equity = 1000 * np.cumprod(1 + returns, axis=0)
    many = np.tile(equity[:, :1], (1, 289))
    result = independent_daily_statistics(many, np.zeros(3), initial_cash=1000, elapsed_years=2)
    assert result["cash_excess_sharpe"].shape == (289,)
    assert np.ptp(result["cash_excess_sharpe"]) == 0
    np.testing.assert_allclose(result["returns"], np.tile(returns[:, :1], (1, 289)))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1.0, 0.0])
def test_independent_math_rejects_undefined_or_exhausted_wealth(invalid):
    with pytest.raises(ValueError, match="finite positive"):
        independent_daily_statistics([1000, invalid], [0, 0], initial_cash=1000, elapsed_years=1)


def test_independent_counts_keep_nan_pairs_and_distinguish_rounded_near_ties():
    scores = np.full(289, 0.4)
    scores[:6] = [0.8, 0.5 + 5e-11, 0.5, 0.5 - 5e-11, np.nan, np.inf]
    result = independent_beat_counts(scores, 0.5)
    assert result == {
        "configuration_count": 289,
        "finite_Sharpe_count": 287,
        "undefined_Sharpe_count": 2,
        "beat_BH_Sharpe_count": 1,
        "strict_gt_BH_Sharpe_count": 2,
        "tie_BH_Sharpe_count": 3,
        "below_BH_Sharpe_count": 283,
    }
    assert (
        sum(
            result[key]
            for key in (
                "undefined_Sharpe_count",
                "beat_BH_Sharpe_count",
                "tie_BH_Sharpe_count",
                "below_BH_Sharpe_count",
            )
        )
        == 289
    )


@pytest.mark.parametrize("scores,benchmark", [(np.zeros(288), 0.5), (np.zeros(289), np.nan)])
def test_independent_counts_reject_incomplete_cohort_or_undefined_benchmark(scores, benchmark):
    with pytest.raises(ValueError, match="289"):
        independent_beat_counts(scores, benchmark)


def test_public_counting_agrees_with_independent_math_for_fabricated_cohorts():
    rows, expected = [], []
    for hours in diagnostic.WAIT_HOURS:
        scores = np.linspace(-0.5, 1.5, 289) + hours / 20
        scores[0] = np.nan
        scores[1] = 0.5 + 5e-11
        scores[2] = 0.5 - 5e-11
        expected.append(independent_beat_counts(scores, 0.5))
        for candidate, score in zip(diagnostic.candidates_for_wait(hours), scores, strict=True):
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "entry_confirmation_hours": hours,
                    "exit_confirmation_hours": hours,
                    "cash_excess_sharpe": score,
                }
            )
    actual = diagnostic.counts_against_bh(pd.DataFrame(rows), 0.5)
    for index, independently_counted in enumerate(expected):
        for key, value in independently_counted.items():
            assert actual.iloc[index][key] == value
