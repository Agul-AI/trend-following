"""Stateless two-horizon gross-compounding agreement policy.

The modeled instrument resets its leverage daily. Its recent *gross* path is
therefore computed from daily returns, not three times an endpoint QQQ gain.
Allocation is a fixed weight when eligible; it is not a volatility target and
the accounting engine is permitted to let fractional holdings drift.
"""
import numpy as np


def target(history, parameters):
    """Use only the supplied past/current history; insufficient history is cash."""
    if parameters['mode'] == 'constant':
        return float(parameters['cap'])

    long_window = int(parameters['long_window'])
    short_window = int(parameters['short_window'])
    vol_window = int(parameters['vol_window'])
    required = max(long_window, short_window, vol_window) + 1
    if len(history) < required:
        return 0.0

    daily = history['total_return'].to_numpy(dtype=float)
    leveraged = 3.0 * daily[-long_window:]
    if not np.isfinite(leveraged).all() or np.any(leveraged <= -1.0):
        return 0.0
    long_log_growth = float(np.log1p(leveraged).sum())
    short_log_growth = float(np.log1p(leveraged[-short_window:]).sum())
    if long_log_growth <= 0.0 or short_log_growth <= 0.0:
        return 0.0

    ceiling = parameters['volatility_ceiling']
    if ceiling is not None:
        realized = float(np.std(daily[-vol_window:], ddof=1) * np.sqrt(252.0))
        if not np.isfinite(realized) or realized > float(ceiling):
            return 0.0
    return float(parameters['cap'])
