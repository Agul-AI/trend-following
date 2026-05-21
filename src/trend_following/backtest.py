"""Transparent vectorized backtester."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Container for daily backtest outputs."""

    daily_returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        """Return a compact daily result table."""
        return pd.DataFrame(
            {
                "gross_return": self.gross_returns,
                "cost": self.costs,
                "net_return": self.daily_returns,
                "turnover": self.turnover,
                "equity": self.equity_curve,
            }
        )


def _to_frame(data: pd.Series | pd.DataFrame, name: str = "asset") -> pd.DataFrame:
    if isinstance(data, pd.Series):
        return data.to_frame(name=data.name or name)
    return data.copy()


def backtest(
    returns: pd.Series | pd.DataFrame,
    weights: pd.Series | pd.DataFrame,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """Backtest daily portfolio returns after turnover costs.

    Parameters
    ----------
    returns:
        Daily asset returns labeled by the end date of the return interval.
    weights:
        Portfolio weights held during each return interval. These should already
        be shifted by the signal module to avoid look-ahead bias.
    transaction_cost_bps, slippage_bps:
        One-way costs charged on absolute weight changes.
    """
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("Costs and slippage must be non-negative")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    returns_df = _to_frame(returns, name="asset").sort_index()
    weights_df = _to_frame(
        weights, name=returns_df.columns[0] if len(returns_df.columns) == 1 else "asset"
    )
    weights_df = weights_df.sort_index()

    common_columns = [column for column in returns_df.columns if column in weights_df.columns]
    if not common_columns:
        raise ValueError("returns and weights must share at least one asset column")
    common_index = returns_df.index.intersection(weights_df.index)
    if common_index.empty:
        raise ValueError("returns and weights must share at least one date")

    returns_df = returns_df.loc[common_index, common_columns]
    weights_df = weights_df.loc[common_index, common_columns].fillna(0.0)

    problematic_missing = returns_df.isna() & weights_df.abs().gt(1e-12)
    if problematic_missing.any().any():
        first_bad = problematic_missing.stack()[lambda s: s].index[0]
        raise ValueError(f"Return is missing for a non-zero weight at {first_bad}")
    returns_df = returns_df.fillna(0.0)

    gross_returns = (weights_df * returns_df).sum(axis=1)
    turnover = weights_df.diff().abs().sum(axis=1)
    if not weights_df.empty:
        turnover.iloc[0] = weights_df.iloc[0].abs().sum()
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0
    costs = turnover * cost_rate
    net_returns = gross_returns - costs
    equity_curve = initial_capital * (1.0 + net_returns).cumprod()

    return BacktestResult(
        daily_returns=net_returns.rename("net_return"),
        gross_returns=gross_returns.rename("gross_return"),
        costs=costs.rename("cost"),
        turnover=turnover.rename("turnover"),
        equity_curve=equity_curve.rename("equity"),
        weights=weights_df,
    )


def buy_and_hold_weights(returns: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    """Create constant equal-weight buy-and-hold benchmark weights."""
    selected = tickers or list(returns.columns)
    if not selected:
        raise ValueError("No tickers available for benchmark")
    missing = [ticker for ticker in selected if ticker not in returns.columns]
    if missing:
        raise ValueError(f"Benchmark tickers missing from returns: {missing}")
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    weights.loc[:, selected] = 1.0 / len(selected)
    return weights[selected]
