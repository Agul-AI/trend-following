"""Performance metrics for daily return streams."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    """Compute a cumulative equity curve from simple returns."""
    clean = returns.fillna(0.0).astype(float)
    return initial_capital * (1.0 + clean).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdowns as negative percentages from running peak."""
    equity = equity_curve(returns)
    running_peak = equity.cummax()
    return equity / running_peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Return the most negative drawdown."""
    if returns.empty:
        return math.nan
    return float(drawdown_series(returns).min())


def annualized_return(returns: pd.Series, annualization: int = 252) -> float:
    """Geometric annualized return."""
    clean = returns.dropna().astype(float)
    if clean.empty:
        return math.nan
    total = float((1.0 + clean).prod())
    if total <= 0:
        return math.nan
    return total ** (annualization / len(clean)) - 1.0


def annualized_volatility(returns: pd.Series, annualization: int = 252) -> float:
    """Annualized daily-return volatility."""
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return math.nan
    return float(clean.std(ddof=0) * np.sqrt(annualization))


def sharpe_ratio(returns: pd.Series, annualization: int = 252) -> float:
    """Annualized Sharpe ratio with zero risk-free rate."""
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return math.nan
    std = clean.std(ddof=0)
    if std == 0 or np.isnan(std):
        return math.nan
    return float(clean.mean() / std * np.sqrt(annualization))


def calculate_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    weights: pd.Series | pd.DataFrame | None = None,
    annualization: int = 252,
) -> dict[str, Any]:
    """Calculate core performance and trading metrics."""
    clean = returns.fillna(0.0).astype(float)
    if clean.empty:
        return {
            "cumulative_return": math.nan,
            "annualized_return": math.nan,
            "annualized_volatility": math.nan,
            "sharpe_ratio": math.nan,
            "max_drawdown": math.nan,
            "calmar_ratio": math.nan,
            "hit_rate": math.nan,
            "average_daily_turnover": math.nan,
            "number_of_trades": 0,
            "exposure_percentage": math.nan,
            "observations": 0,
        }

    cumulative = float((1.0 + clean).prod() - 1.0)
    ann_return = annualized_return(clean, annualization=annualization)
    ann_vol = annualized_volatility(clean, annualization=annualization)
    sharpe = sharpe_ratio(clean, annualization=annualization)
    mdd = max_drawdown(clean)
    calmar = ann_return / abs(mdd) if mdd < 0 and not np.isnan(ann_return) else math.nan
    hit_rate = float((clean > 0).mean())

    if turnover is None:
        avg_turnover = math.nan
        trades = 0
    else:
        aligned_turnover = turnover.reindex(clean.index).fillna(0.0)
        avg_turnover = float(aligned_turnover.mean())
        trades = int((aligned_turnover > 1e-12).sum())

    if weights is None:
        exposure = math.nan
    else:
        if isinstance(weights, pd.Series):
            exposure_series = weights.reindex(clean.index).abs().fillna(0.0)
        else:
            exposure_series = weights.reindex(clean.index).abs().sum(axis=1).fillna(0.0)
        exposure = float(exposure_series.mean())

    return {
        "cumulative_return": cumulative,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "calmar_ratio": calmar,
        "hit_rate": hit_rate,
        "average_daily_turnover": avg_turnover,
        "number_of_trades": trades,
        "exposure_percentage": exposure,
        "observations": int(len(clean)),
    }


def metrics_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert metric dictionaries into a consistently ordered DataFrame."""
    preferred = [
        "name",
        "strategy",
        "segment",
        "parameters",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "hit_rate",
        "average_daily_turnover",
        "number_of_trades",
        "exposure_percentage",
        "observations",
    ]
    frame = pd.DataFrame(rows)
    ordered = [column for column in preferred if column in frame.columns]
    remaining = [column for column in frame.columns if column not in ordered]
    return frame[ordered + remaining]
