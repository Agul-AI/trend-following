from __future__ import annotations

import pandas as pd

from trend_following.backtest import backtest
from trend_following.signals import make_executable_positions


def test_shifted_positions_do_not_capture_same_day_clairvoyant_returns() -> None:
    index = pd.bdate_range("2020-01-01", periods=6)
    returns = pd.Series([0.0, 0.20, -0.20, 0.20, -0.20, 0.20], index=index, name="X")

    # This raw signal is intentionally illegal because it knows same-day returns.
    # If used unshifted, it captures every positive return. The executable shift
    # prevents the backtester from applying it to unavailable future returns.
    clairvoyant_raw_signal = (returns > 0).astype(float)
    shifted_positions = make_executable_positions(clairvoyant_raw_signal, execution_delay_days=1)

    safe = backtest(returns, shifted_positions, transaction_cost_bps=0, slippage_bps=0)
    cheating = backtest(returns, clairvoyant_raw_signal, transaction_cost_bps=0, slippage_bps=0)

    assert shifted_positions.iloc[1] == 0.0
    assert shifted_positions.iloc[2] == 0.0
    assert safe.equity_curve.iloc[-1] < cheating.equity_curve.iloc[-1]
