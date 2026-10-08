"""Monte Carlo robustness tools for strategy return panels.

The primary workflow here is a paired, time-series-aware stationary block
bootstrap over daily rows.  The same sampled row indices are applied to strategy
and benchmark series, so crisis/bull/chop regimes remain paired across columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class MetricsConfig:
    """Settings for return metric calculations."""

    annualization: int = TRADING_DAYS_PER_YEAR
    risk_free_rate: float = 0.0


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Return growth of ``initial`` from simple returns."""
    clean = returns.fillna(0.0).astype(float)
    return (initial * (1.0 + clean).cumprod()).rename("equity")


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Return drawdown series from simple returns."""
    equity = equity_curve(returns)
    peak = equity.cummax().clip(lower=1.0)
    return (equity / peak - 1.0).rename("drawdown")


def drawdown_episode_count(returns: pd.Series, threshold: float) -> int:
    """Count distinct drawdown episodes crossing a negative threshold."""
    if threshold >= 0:
        raise ValueError("threshold must be negative")
    dd = drawdown_series(returns)
    count = 0
    active = False
    for value in dd:
        if value <= threshold + 1e-12 and not active:
            count += 1
            active = True
        elif value >= -1e-12:
            active = False
    return count


def longest_underwater_period(returns: pd.Series) -> int:
    """Longest number of consecutive observations below the prior equity peak."""
    dd = drawdown_series(returns)
    longest = 0
    current = 0
    for value in dd:
        if value < -1e-12:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _rolling_compounded_returns(returns: pd.Series, window: int) -> np.ndarray:
    """Fast rolling compounded simple returns for complete windows."""
    if window <= 0:
        raise ValueError("window must be positive")
    clean = returns.fillna(0.0).astype(float).to_numpy(dtype=float)
    if len(clean) < window:
        return np.array([], dtype=float)
    gross = 1.0 + clean
    cumulative = np.concatenate([[1.0], np.cumprod(gross)])
    return cumulative[window:] / cumulative[:-window] - 1.0


def worst_window_return(returns: pd.Series, window: int) -> float:
    """Worst compounded return over a rolling window."""
    rolling = _rolling_compounded_returns(returns, window)
    if len(rolling) == 0:
        return float("nan")
    return float(np.nanmin(rolling))


def _rolling_return_summary(returns: pd.Series, window: int, prefix: str) -> dict[str, float]:
    rolling = _rolling_compounded_returns(returns, window)
    if len(rolling) == 0:
        return {
            f"{prefix}_return_min": float("nan"),
            f"{prefix}_return_p05": float("nan"),
            f"{prefix}_return_median": float("nan"),
        }
    return {
        f"{prefix}_return_min": float(np.nanmin(rolling)),
        f"{prefix}_return_p05": float(np.nanquantile(rolling, 0.05)),
        f"{prefix}_return_median": float(np.nanmedian(rolling)),
    }


def performance_metrics(
    returns: pd.Series,
    *,
    exposure: pd.Series | None = None,
    config: MetricsConfig | None = None,
) -> dict[str, Any]:
    """Compute robust performance and drawdown metrics for simple daily returns."""
    cfg = config or MetricsConfig()
    clean = returns.dropna().astype(float)
    if clean.empty:
        return {
            "observations": 0,
            "final_multiple": float("nan"),
            "total_return": float("nan"),
            "cagr": float("nan"),
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "sortino_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "calmar_ratio": float("nan"),
        }

    final_multiple = float((1.0 + clean).prod())
    total_return = final_multiple - 1.0
    years = len(clean) / cfg.annualization
    cagr = final_multiple ** (1.0 / years) - 1.0 if years > 0 and final_multiple > 0 else float("nan")
    vol = float(clean.std(ddof=0) * np.sqrt(cfg.annualization)) if len(clean) > 1 else float("nan")
    excess_daily = clean - cfg.risk_free_rate / cfg.annualization
    sharpe = (
        float(excess_daily.mean() / clean.std(ddof=0) * np.sqrt(cfg.annualization))
        if len(clean) > 1 and clean.std(ddof=0) > 0
        else float("nan")
    )
    downside = clean.loc[clean < 0]
    sortino = (
        float(excess_daily.mean() / downside.std(ddof=0) * np.sqrt(cfg.annualization))
        if len(downside) > 1 and downside.std(ddof=0) > 0
        else float("nan")
    )
    max_dd = float(drawdown_series(clean).min())
    calmar = cagr / abs(max_dd) if max_dd < 0 and np.isfinite(cagr) else float("nan")

    out: dict[str, Any] = {
        "observations": int(len(clean)),
        "final_multiple": final_multiple,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_return": cagr,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "longest_underwater_days": longest_underwater_period(clean),
        "worst_1d_return": worst_window_return(clean, 1),
        "worst_5d_return": worst_window_return(clean, 5),
        "worst_20d_return": worst_window_return(clean, 20),
        "worst_63d_return": worst_window_return(clean, 63),
        "worst_252d_return": worst_window_return(clean, 252),
    }
    for threshold in (20, 30, 40, 50, 60, 70):
        out[f"drawdown_episodes_gt_{threshold}pct"] = drawdown_episode_count(
            clean, -threshold / 100.0
        )
    out.update(_rolling_return_summary(clean, 252, "rolling_1y"))
    out.update(_rolling_return_summary(clean, 3 * 252, "rolling_3y"))
    out.update(_rolling_return_summary(clean, 5 * 252, "rolling_5y"))

    if exposure is not None:
        aligned = exposure.reindex(clean.index).astype(float)
        out["exposure_percentage"] = float(aligned.abs().mean())
    else:
        out["exposure_percentage"] = float("nan")
    return out


PERFORMANCE_TABLE_COLUMNS = [
    "strategy_name",
    "start_date",
    "end_date",
    "first_executable_trade_date",
    "years",
    "final_multiple",
    "total_return",
    "CAGR",
    "annualized_return",
    "annualized_volatility",
    "Sharpe",
    "Sortino",
    "Calmar",
    "MAR",
    "Omega",
    "max_drawdown",
    "average_drawdown",
    "median_drawdown",
    "ulcer_index",
    "recovery_factor",
    "longest_underwater_days",
    "longest_underwater_years",
    "time_underwater_pct",
    "current_drawdown",
    "dd_count_10",
    "dd_count_20",
    "dd_count_30",
    "dd_count_40",
    "dd_count_50",
    "dd_count_60",
    "dd_count_70",
    "worst_1_day_return",
    "worst_5_day_return",
    "worst_20_day_return",
    "worst_63_day_return",
    "worst_252_day_return",
    "worst_month",
    "worst_year",
    "worst_rolling_1y_return",
    "worst_rolling_3y_return",
    "worst_rolling_5y_return",
    "median_rolling_1y_return",
    "median_rolling_3y_return",
    "median_rolling_5y_return",
    "rolling_1y_sharpe_min",
    "rolling_3y_sharpe_min",
    "rolling_5y_sharpe_min",
    "VaR_95",
    "VaR_99",
    "CVaR_95",
    "CVaR_99",
    "skewness",
    "kurtosis",
    "exposure_pct",
    "return_per_exposure",
    "trades",
    "trades_per_year",
    "win_rate",
    "avg_trade_return",
    "median_trade_return",
    "avg_winner",
    "avg_loser",
    "payoff_ratio",
    "profit_factor",
    "expectancy_per_trade",
    "max_consecutive_losses",
    "avg_holding_days",
    "median_holding_days",
    "turnover",
    "alpha_vs_QQQ",
    "beta_vs_QQQ",
    "correlation_vs_QQQ",
    "tracking_error_vs_QQQ",
    "information_ratio_vs_QQQ",
    "upside_capture_vs_QQQ",
    "downside_capture_vs_QQQ",
    "excess_CAGR_vs_QQQ",
    "relative_final_multiple_vs_QQQ",
    "alpha_vs_SPY",
    "beta_vs_SPY",
    "correlation_vs_SPY",
    "information_ratio_vs_SPY",
    "excess_CAGR_vs_SPY",
]


def _empty_row_value() -> float:
    return float("nan")


def _as_daily_returns(returns: pd.Series) -> pd.Series:
    clean = returns.dropna().astype(float).sort_index()
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index)
    return clean


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _rolling_sharpe_min(
    returns: pd.Series,
    window: int,
    *,
    annualization: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    clean = returns.fillna(0.0).astype(float)
    if len(clean) < window:
        return float("nan")
    excess = clean - risk_free_rate / annualization
    rolling_mean = excess.rolling(window).mean()
    rolling_std = clean.rolling(window).std(ddof=0)
    sharpe = rolling_mean / rolling_std * np.sqrt(annualization)
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan).dropna()
    return float(sharpe.min()) if not sharpe.empty else float("nan")


def _worst_period_return(returns: pd.Series, freq: str) -> tuple[float, str | None]:
    clean = _as_daily_returns(returns)
    if clean.empty:
        return float("nan"), None
    period_returns = (1.0 + clean).resample(freq).prod() - 1.0
    period_returns = period_returns.dropna()
    if period_returns.empty:
        return float("nan"), None
    idx = period_returns.idxmin()
    label = idx.strftime("%Y-%m") if freq.upper().startswith("M") else idx.strftime("%Y")
    return float(period_returns.loc[idx]), label


def _historical_var_cvar(returns: pd.Series, probability: float) -> tuple[float, float]:
    """Return historical left-tail VaR/CVaR as return quantiles, usually negative."""
    clean = returns.dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan")
    tail_probability = 1.0 - probability
    var = float(clean.quantile(tail_probability))
    tail = clean.loc[clean <= var]
    cvar = float(tail.mean()) if not tail.empty else float("nan")
    return var, cvar


def _trade_stats_from_position(
    returns: pd.Series,
    position: pd.Series,
    *,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, Any]:
    ret = _as_daily_returns(returns)
    pos = position.reindex(ret.index).fillna(0.0).astype(float)
    in_market = pos.gt(0.0)
    starts = in_market & ~in_market.shift(1, fill_value=False)
    trade_id = starts.cumsum().where(in_market, 0).astype(int)
    trade_returns: list[float] = []
    holding_days: list[int] = []
    for tid in [int(x) for x in sorted(trade_id.unique()) if int(x) > 0]:
        group_returns = ret.loc[trade_id.eq(tid)]
        if group_returns.empty:
            continue
        trade_returns.append(float((1.0 + group_returns).prod() - 1.0))
        holding_days.append(int(len(group_returns)))

    trades = len(trade_returns)
    years = len(ret) / annualization if len(ret) else float("nan")
    if trades == 0:
        return {
            "trades": 0,
            "trades_per_year": 0.0 if years and years > 0 else float("nan"),
            "win_rate": float("nan"),
            "avg_trade_return": float("nan"),
            "median_trade_return": float("nan"),
            "avg_winner": float("nan"),
            "avg_loser": float("nan"),
            "payoff_ratio": float("nan"),
            "profit_factor": float("nan"),
            "expectancy_per_trade": float("nan"),
            "max_consecutive_losses": 0,
            "avg_holding_days": float("nan"),
            "median_holding_days": float("nan"),
            "turnover": float(pos.diff().abs().fillna(pos.abs()).sum() / years) if years and years > 0 else float("nan"),
        }

    trade_arr = np.asarray(trade_returns, dtype=float)
    winners = trade_arr[trade_arr > 0]
    losers = trade_arr[trade_arr < 0]
    max_consec = 0
    current = 0
    for value in trade_arr:
        if value < 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0
    avg_winner = float(winners.mean()) if len(winners) else float("nan")
    avg_loser = float(losers.mean()) if len(losers) else float("nan")
    gross_profit = float(winners.sum()) if len(winners) else 0.0
    gross_loss = float(abs(losers.sum())) if len(losers) else 0.0
    turnover = float(pos.diff().abs().fillna(pos.abs()).sum() / years) if years and years > 0 else float("nan")
    return {
        "trades": int(trades),
        "trades_per_year": _safe_ratio(float(trades), years),
        "win_rate": float((trade_arr > 0).mean()),
        "avg_trade_return": float(trade_arr.mean()),
        "median_trade_return": float(np.median(trade_arr)),
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "payoff_ratio": _safe_ratio(avg_winner, abs(avg_loser)),
        "profit_factor": _safe_ratio(gross_profit, gross_loss),
        "expectancy_per_trade": float(trade_arr.mean()),
        "max_consecutive_losses": int(max_consec),
        "avg_holding_days": float(np.mean(holding_days)),
        "median_holding_days": float(np.median(holding_days)),
        "turnover": turnover,
    }


def _relative_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    prefix: str,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    keys = [
        f"alpha_vs_{prefix}",
        f"beta_vs_{prefix}",
        f"correlation_vs_{prefix}",
        f"tracking_error_vs_{prefix}",
        f"information_ratio_vs_{prefix}",
        f"upside_capture_vs_{prefix}",
        f"downside_capture_vs_{prefix}",
        f"excess_CAGR_vs_{prefix}",
        f"relative_final_multiple_vs_{prefix}",
    ]
    out = {key: float("nan") for key in keys}
    if benchmark_returns is None:
        return out
    aligned = pd.concat(
        [returns.rename("strategy").astype(float), benchmark_returns.rename("benchmark").astype(float)],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 2:
        return out
    strat = aligned["strategy"]
    bench = aligned["benchmark"]
    bench_var = float(bench.var(ddof=0))
    beta = float(strat.cov(bench, ddof=0) / bench_var) if bench_var > 0 else float("nan")
    alpha_daily = float(strat.mean() - beta * bench.mean()) if np.isfinite(beta) else float("nan")
    active = strat - bench
    tracking_error = float(active.std(ddof=0) * np.sqrt(annualization)) if len(active) > 1 else float("nan")
    information = (
        float(active.mean() / active.std(ddof=0) * np.sqrt(annualization))
        if len(active) > 1 and active.std(ddof=0) > 0
        else float("nan")
    )
    strat_final = float((1.0 + strat).prod())
    bench_final = float((1.0 + bench).prod())
    years = len(aligned) / annualization
    strat_cagr = strat_final ** (1.0 / years) - 1.0 if years > 0 and strat_final > 0 else float("nan")
    bench_cagr = bench_final ** (1.0 / years) - 1.0 if years > 0 and bench_final > 0 else float("nan")
    up = aligned.loc[bench > 0]
    down = aligned.loc[bench < 0]
    out.update(
        {
            f"alpha_vs_{prefix}": float(alpha_daily * annualization) if np.isfinite(alpha_daily) else float("nan"),
            f"beta_vs_{prefix}": beta,
            f"correlation_vs_{prefix}": float(strat.corr(bench)),
            f"tracking_error_vs_{prefix}": tracking_error,
            f"information_ratio_vs_{prefix}": information,
            f"upside_capture_vs_{prefix}": _safe_ratio(float(up["strategy"].mean()), float(up["benchmark"].mean())) if not up.empty else float("nan"),
            f"downside_capture_vs_{prefix}": _safe_ratio(float(down["strategy"].mean()), float(down["benchmark"].mean())) if not down.empty else float("nan"),
            f"excess_CAGR_vs_{prefix}": strat_cagr - bench_cagr if np.isfinite(strat_cagr) and np.isfinite(bench_cagr) else float("nan"),
            f"relative_final_multiple_vs_{prefix}": _safe_ratio(strat_final, bench_final),
        }
    )
    return out


def expanded_performance_row(
    strategy_name: str,
    returns: pd.Series,
    *,
    position: pd.Series | None = None,
    qqq_returns: pd.Series | None = None,
    spy_returns: pd.Series | None = None,
    config: MetricsConfig | None = None,
) -> dict[str, Any]:
    """Build an interview-style historical performance row with rich risk stats.

    The table uses daily simple returns. Drawdown and tail metrics are reported
    as decimal returns (for example, ``-0.50`` means ``-50%``). ``trades`` are
    in-market segments/round trips inferred from transitions from zero exposure
    to positive exposure; an open final segment is included mark-to-market.
    """
    cfg = config or MetricsConfig()
    clean = _as_daily_returns(returns)
    if position is None:
        pos = pd.Series(1.0, index=clean.index, name="position")
    else:
        pos = position.reindex(clean.index).fillna(0.0).astype(float).rename("position")
    metrics = performance_metrics(clean, exposure=pos, config=cfg)
    dd = drawdown_series(clean) if not clean.empty else pd.Series(dtype=float)
    underwater = dd.loc[dd < -1e-12]
    years = len(clean) / cfg.annualization if len(clean) else float("nan")
    final_multiple = metrics.get("final_multiple", float("nan"))
    total_return = metrics.get("total_return", float("nan"))
    max_dd = metrics.get("max_drawdown", float("nan"))
    cagr = metrics.get("cagr", float("nan"))
    exposure_pct = metrics.get("exposure_percentage", float("nan"))
    var95, cvar95 = _historical_var_cvar(clean, 0.95)
    var99, cvar99 = _historical_var_cvar(clean, 0.99)
    worst_month, worst_month_period = _worst_period_return(clean, "ME")
    worst_year, worst_year_period = _worst_period_return(clean, "YE")
    first_trade = pos.loc[pos > 0].index.min() if not pos.loc[pos > 0].empty else pd.NaT
    gains = clean.loc[clean > 0].sum()
    losses = abs(clean.loc[clean < 0].sum())

    row: dict[str, Any] = {
        "strategy_name": strategy_name,
        "start_date": clean.index.min().date().isoformat() if not clean.empty else None,
        "end_date": clean.index.max().date().isoformat() if not clean.empty else None,
        "first_executable_trade_date": first_trade.date().isoformat() if pd.notna(first_trade) else None,
        "years": years,
        "final_multiple": final_multiple,
        "total_return": total_return,
        "CAGR": cagr,
        "annualized_return": metrics.get("annualized_return", float("nan")),
        "annualized_volatility": metrics.get("annualized_volatility", float("nan")),
        "Sharpe": metrics.get("sharpe_ratio", float("nan")),
        "Sortino": metrics.get("sortino_ratio", float("nan")),
        "Calmar": metrics.get("calmar_ratio", float("nan")),
        "MAR": metrics.get("calmar_ratio", float("nan")),
        "Omega": _safe_ratio(float(gains), float(losses)),
        "max_drawdown": max_dd,
        "average_drawdown": float(underwater.mean()) if not underwater.empty else 0.0,
        "median_drawdown": float(underwater.median()) if not underwater.empty else 0.0,
        "ulcer_index": float(np.sqrt(np.nanmean(np.square(dd.to_numpy(dtype=float))))) if not dd.empty else float("nan"),
        "recovery_factor": _safe_ratio(total_return, abs(max_dd)),
        "longest_underwater_days": metrics.get("longest_underwater_days", float("nan")),
        "longest_underwater_years": _safe_ratio(float(metrics.get("longest_underwater_days", np.nan)), cfg.annualization),
        "time_underwater_pct": float(dd.lt(-1e-12).mean()) if not dd.empty else float("nan"),
        "current_drawdown": float(dd.iloc[-1]) if not dd.empty else float("nan"),
        "dd_count_10": drawdown_episode_count(clean, -0.10) if not clean.empty else 0,
        "dd_count_20": metrics.get("drawdown_episodes_gt_20pct", 0),
        "dd_count_30": metrics.get("drawdown_episodes_gt_30pct", 0),
        "dd_count_40": metrics.get("drawdown_episodes_gt_40pct", 0),
        "dd_count_50": metrics.get("drawdown_episodes_gt_50pct", 0),
        "dd_count_60": metrics.get("drawdown_episodes_gt_60pct", 0),
        "dd_count_70": metrics.get("drawdown_episodes_gt_70pct", 0),
        "worst_1_day_return": metrics.get("worst_1d_return", float("nan")),
        "worst_5_day_return": metrics.get("worst_5d_return", float("nan")),
        "worst_20_day_return": metrics.get("worst_20d_return", float("nan")),
        "worst_63_day_return": metrics.get("worst_63d_return", float("nan")),
        "worst_252_day_return": metrics.get("worst_252d_return", float("nan")),
        "worst_month": worst_month,
        "worst_year": worst_year,
        "worst_rolling_1y_return": metrics.get("rolling_1y_return_min", float("nan")),
        "worst_rolling_3y_return": metrics.get("rolling_3y_return_min", float("nan")),
        "worst_rolling_5y_return": metrics.get("rolling_5y_return_min", float("nan")),
        "median_rolling_1y_return": metrics.get("rolling_1y_return_median", float("nan")),
        "median_rolling_3y_return": metrics.get("rolling_3y_return_median", float("nan")),
        "median_rolling_5y_return": metrics.get("rolling_5y_return_median", float("nan")),
        "rolling_1y_sharpe_min": _rolling_sharpe_min(clean, 252, annualization=cfg.annualization, risk_free_rate=cfg.risk_free_rate),
        "rolling_3y_sharpe_min": _rolling_sharpe_min(clean, 3 * 252, annualization=cfg.annualization, risk_free_rate=cfg.risk_free_rate),
        "rolling_5y_sharpe_min": _rolling_sharpe_min(clean, 5 * 252, annualization=cfg.annualization, risk_free_rate=cfg.risk_free_rate),
        "VaR_95": var95,
        "VaR_99": var99,
        "CVaR_95": cvar95,
        "CVaR_99": cvar99,
        "skewness": float(clean.skew()) if len(clean) > 2 else float("nan"),
        "kurtosis": float(clean.kurtosis()) if len(clean) > 3 else float("nan"),
        "exposure_pct": exposure_pct,
        "return_per_exposure": _safe_ratio(cagr, exposure_pct),
    }
    row.update(_trade_stats_from_position(clean, pos, annualization=cfg.annualization))
    row.update(_relative_metrics(clean, qqq_returns, prefix="QQQ", annualization=cfg.annualization))
    row.update(_relative_metrics(clean, spy_returns, prefix="SPY", annualization=cfg.annualization))
    row["worst_month_period"] = worst_month_period
    row["worst_year_period"] = worst_year_period
    return row


def order_performance_table_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Put requested performance-table columns first while preserving extras."""
    for column in PERFORMANCE_TABLE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    ordered = PERFORMANCE_TABLE_COLUMNS + [c for c in frame.columns if c not in PERFORMANCE_TABLE_COLUMNS]
    return frame[ordered]


def stationary_bootstrap_indices(
    n_obs: int,
    avg_block_length: float,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate stationary bootstrap indices of length ``n_obs``.

    Blocks have geometrically distributed lengths with mean
    ``avg_block_length``.  Indices wrap around at the end of the source sample.
    """
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if avg_block_length <= 0:
        raise ValueError("avg_block_length must be positive")
    if rng is not None and seed is not None:
        raise ValueError("Pass rng or seed, not both")
    if rng is None:
        rng = np.random.default_rng(seed)

    restart_probability = min(max(1.0 / float(avg_block_length), 0.0), 1.0)
    indices = np.empty(n_obs, dtype=int)
    current = int(rng.integers(0, n_obs))
    for i in range(n_obs):
        if i == 0 or float(rng.random()) < restart_probability:
            current = int(rng.integers(0, n_obs))
        else:
            current = (current + 1) % n_obs
        indices[i] = current
    return indices


def paired_bootstrap_sample(panel: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    """Sample all columns of a panel with identical bootstrap indices."""
    if len(indices) <= 0:
        raise ValueError("indices must not be empty")
    if indices.min() < 0 or indices.max() >= len(panel):
        raise ValueError("indices are out of bounds for panel")
    sample = panel.iloc[indices].copy()
    if len(sample) == len(panel):
        sample.index = panel.index
    else:
        sample.index = pd.RangeIndex(len(sample))
    return sample


def aggregate_daily_returns(hourly_returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compound intraday/hourly returns into daily returns."""
    frame = hourly_returns.sort_index().fillna(0.0)
    grouped = (1.0 + frame).groupby(frame.index.normalize()).prod() - 1.0
    grouped.index.name = "date"
    return grouped


def daily_last(series_or_frame: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Take the last intraday observation for each calendar date."""
    data = series_or_frame.sort_index()
    out = data.groupby(data.index.normalize()).last()
    out.index.name = "date"
    return out


def daily_mean(series_or_frame: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Take the mean intraday observation for each calendar date."""
    data = series_or_frame.sort_index()
    out = data.groupby(data.index.normalize()).mean()
    out.index.name = "date"
    return out


def derive_trade_markers(position: pd.Series) -> pd.DataFrame:
    """Derive daily entry/exit markers and trade IDs from a position series."""
    pos = position.fillna(0.0).astype(float)
    prev = pos.shift(1).fillna(0.0)
    entry = pos.gt(0.0) & prev.le(0.0)
    exit_ = pos.le(0.0) & prev.gt(0.0)
    trade_id = entry.cumsum().where(pos.gt(0.0), 0).astype(int)
    return pd.DataFrame(
        {
            "entry_marker": entry.astype(int),
            "exit_marker": exit_.astype(int),
            "trade_id": trade_id,
        },
        index=pos.index,
    )


def qqq_200ma_timing_return(
    qqq_price: pd.Series,
    qqq_return: pd.Series,
    *,
    execution_delay_days: int = 1,
    cash_annual_yield: float = 0.03,
    annualization: int = 252,
) -> tuple[pd.Series, pd.Series]:
    """Build a no-lookahead QQQ 200-day moving-average timing benchmark."""
    price = qqq_price.sort_index().astype(float)
    returns = qqq_return.reindex(price.index).fillna(0.0).astype(float)
    raw = price.gt(price.rolling(200, min_periods=200).mean()).astype(float)
    # Same close-to-close convention as project strategy: close-t signal cannot
    # earn return label t and next-close execution first earns t+2.
    position = raw.shift(execution_delay_days + 1).fillna(0.0)
    cash_return = (1.0 + cash_annual_yield) ** (1.0 / annualization) - 1.0
    strategy_return = position * returns + (1.0 - position) * cash_return
    return strategy_return.rename("qqq_200ma_timing_return"), position.rename(
        "qqq_200ma_timing_position"
    )


def benchmark_probability_rows(
    mc_results: pd.DataFrame,
    *,
    strategy_name: str,
    benchmark_names: list[str],
) -> pd.DataFrame:
    """Compute benchmark outperformance probabilities from long MC results."""
    rows: list[dict[str, Any]] = []
    key_cols = ["window", "block_length", "sim_id"]
    strategy = mc_results.loc[mc_results["series"].eq(strategy_name)].copy()
    for benchmark in benchmark_names:
        bench = mc_results.loc[mc_results["series"].eq(benchmark)].copy()
        if strategy.empty or bench.empty:
            continue
        merged = strategy.merge(bench, on=key_cols, suffixes=("_strategy", "_benchmark"))
        if merged.empty:
            continue
        for (window, block), group in merged.groupby(["window", "block_length"]):
            rows.append(
                {
                    "window": window,
                    "block_length": block,
                    "benchmark": benchmark,
                    "simulations": int(len(group)),
                    "p_strategy_beats_final_multiple": float(
                        group["final_multiple_strategy"].gt(group["final_multiple_benchmark"]).mean()
                    ),
                    "p_strategy_beats_cagr": float(
                        group["cagr_strategy"].gt(group["cagr_benchmark"]).mean()
                    ),
                    "p_strategy_lower_max_dd": float(
                        group["max_drawdown_strategy"].gt(group["max_drawdown_benchmark"]).mean()
                    ),
                    "p_strategy_beats_and_lower_dd": float(
                        (
                            group["final_multiple_strategy"].gt(group["final_multiple_benchmark"])
                            & group["max_drawdown_strategy"].gt(group["max_drawdown_benchmark"])
                        ).mean()
                    ),
                    "p_strategy_underperforms_by_more_than_25pct": float(
                        (
                            group["final_multiple_strategy"]
                            < 0.75 * group["final_multiple_benchmark"]
                        ).mean()
                    ),
                    "median_relative_final_multiple": float(
                        (group["final_multiple_strategy"] / group["final_multiple_benchmark"]).median()
                    ),
                    "median_relative_cagr": float(
                        (group["cagr_strategy"] - group["cagr_benchmark"]).median()
                    ),
                    "median_relative_max_dd": float(
                        (group["max_drawdown_strategy"] - group["max_drawdown_benchmark"]).median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def mc_percentile_table(mc_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize MC metric percentiles by window/block/series."""
    metrics = ["final_multiple", "cagr", "max_drawdown", "calmar_ratio", "longest_underwater_days"]
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    rows: list[dict[str, Any]] = []
    for (window, block, series), group in mc_results.groupby(["window", "block_length", "series"]):
        row: dict[str, Any] = {"window": window, "block_length": block, "series": series}
        for metric in metrics:
            for q in quantiles:
                row[f"{metric}_p{int(q * 100):02d}"] = float(group[metric].quantile(q))
        rows.append(row)
    return pd.DataFrame(rows)


def drawdown_probability_table(mc_results: pd.DataFrame, *, strategy_name: str) -> pd.DataFrame:
    """Summarize strategy drawdown and loss probabilities by MC setting."""
    rows: list[dict[str, Any]] = []
    strategy = mc_results.loc[mc_results["series"].eq(strategy_name)].copy()
    for (window, block), group in strategy.groupby(["window", "block_length"]):
        rows.append(
            {
                "window": window,
                "block_length": block,
                "simulations": int(len(group)),
                "p_max_dd_worse_than_40pct": float(group["max_drawdown"].le(-0.40).mean()),
                "p_max_dd_worse_than_50pct": float(group["max_drawdown"].le(-0.50).mean()),
                "p_max_dd_worse_than_60pct": float(group["max_drawdown"].le(-0.60).mean()),
                "p_max_dd_worse_than_70pct": float(group["max_drawdown"].le(-0.70).mean()),
                "p_losing_full_period": float(group["total_return"].lt(0.0).mean()),
                "p_longest_underwater_gt_1y": float(
                    group["longest_underwater_days"].gt(252).mean()
                ),
                "p_longest_underwater_gt_2y": float(
                    group["longest_underwater_days"].gt(504).mean()
                ),
                "p_longest_underwater_gt_3y": float(
                    group["longest_underwater_days"].gt(756).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def simulate_trade_level_bootstrap(
    trade_returns: pd.Series,
    trade_durations: pd.Series,
    *,
    n_sims: int,
    seed: int,
    annualization: int = 252,
) -> pd.DataFrame:
    """Bootstrap completed trade returns as a secondary, lower-confidence test."""
    clean_returns = trade_returns.dropna().astype(float)
    clean_durations = trade_durations.reindex(clean_returns.index).fillna(1).astype(float)
    if clean_returns.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    n_trades = len(clean_returns)
    for sim_id in range(n_sims):
        idx = rng.integers(0, n_trades, size=n_trades)
        sampled_returns = pd.Series(clean_returns.to_numpy()[idx])
        sampled_days = float(clean_durations.to_numpy()[idx].sum())
        metrics = performance_metrics(sampled_returns, config=MetricsConfig(annualization=n_trades))
        final_multiple = float((1.0 + sampled_returns).prod())
        years = sampled_days / annualization if sampled_days > 0 else np.nan
        cagr = final_multiple ** (1.0 / years) - 1.0 if years and years > 0 else np.nan
        metrics.update(
            {
                "sim_id": sim_id,
                "n_trades": n_trades,
                "sampled_trade_days": sampled_days,
                "final_multiple": final_multiple,
                "total_return": final_multiple - 1.0,
                "cagr": cagr,
                "annualized_return": cagr,
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def apply_entry_exit_delay_to_position(
    position: pd.Series,
    *,
    entry_delay_days: int,
    exit_delay_days: int,
) -> pd.Series:
    """Approximate daily entry/exit delays from a desired position series."""
    if entry_delay_days < 0 or exit_delay_days < 0:
        raise ValueError("delays must be non-negative")
    pos = position.fillna(0.0).astype(float).reset_index(drop=True)
    events: dict[int, float] = {0: 0.0}
    previous = 0.0
    for i, desired in enumerate(pos):
        if np.isclose(desired, previous):
            continue
        delay = entry_delay_days if desired > previous else exit_delay_days
        effective = min(i + delay, len(pos) - 1)
        events[effective] = float(desired)
        previous = float(desired)
    current = 0.0
    values: list[float] = []
    for i in range(len(pos)):
        if i in events:
            current = events[i]
        values.append(current)
    return pd.Series(values, index=position.index, name=position.name)


def strategy_return_from_position(
    asset_returns: pd.Series,
    position: pd.Series,
    *,
    slippage_bps: float = 0.0,
    transaction_cost_bps: float = 0.0,
    cash_annual_yield: float = 0.03,
    annualization: int = 252,
) -> pd.Series:
    """Approximate daily strategy return from daily position and asset returns."""
    returns = asset_returns.fillna(0.0).astype(float)
    pos = position.reindex(returns.index).fillna(0.0).clip(0.0, 1.0).astype(float)
    cash = (1.0 + cash_annual_yield) ** (1.0 / annualization) - 1.0
    cost_rate = (slippage_bps + transaction_cost_bps) / 10_000.0
    turnover = pos.diff().abs().fillna(pos.abs())
    out = pos * returns + (1.0 - pos) * cash - turnover * cost_rate
    return out.rename("approx_strategy_return")
