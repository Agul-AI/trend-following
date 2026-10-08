from __future__ import annotations

import numpy as np
import pandas as pd

from trend_following.metrics import annualized_return, calculate_metrics, max_drawdown, sharpe_ratio


def test_max_drawdown_known_case() -> None:
    returns = pd.Series([0.10, -0.10, -0.10])
    assert np.isclose(max_drawdown(returns), -0.19)


def test_drawdown_includes_initial_capital() -> None:
    from trend_following.monte_carlo_robustness import drawdown_series

    returns = pd.Series([-0.10, 0.0])
    assert np.isclose(max_drawdown(returns), -0.10)
    assert np.isclose(drawdown_series(returns).min(), -0.10)


def test_sharpe_uses_aligned_excess_returns() -> None:
    returns = pd.Series([0.02, 0.01, -0.01])
    cash = pd.Series([0.002, 0.003, 0.001])
    excess = returns - cash
    expected = excess.mean() / excess.std(ddof=0) * np.sqrt(252)
    assert np.isclose(sharpe_ratio(returns, risk_free_returns=cash), expected)


def test_annualized_return_one_year_constant_daily_return() -> None:
    returns = pd.Series([0.001] * 252)
    expected = (1.001**252) - 1.0
    assert np.isclose(annualized_return(returns), expected)


def test_calculate_metrics_turnover_and_exposure() -> None:
    returns = pd.Series([0.0, 0.01, -0.01, 0.02])
    turnover = pd.Series([0.0, 1.0, 0.0, 1.0], index=returns.index)
    weights = pd.DataFrame({"X": [0.0, 1.0, 1.0, 0.0]}, index=returns.index)
    metrics = calculate_metrics(returns, turnover=turnover, weights=weights)
    assert metrics["number_of_trades"] == 2
    assert np.isclose(metrics["average_daily_turnover"], 0.5)
    assert np.isclose(metrics["exposure_percentage"], 0.5)
