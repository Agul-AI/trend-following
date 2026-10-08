"""Accounting helpers for cash/T-bill sleeves and synthetic leveraged returns.

This module intentionally keeps risk-off assets restricted to cash-like returns:
fixed cash assumptions or a documented T-bill / short-Treasury proxy.  It also
creates documented synthetic leveraged-return variants so the preferred QQQ /
synthetic-TQQQ strategy can be audited before signal optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

TRADING_BARS_PER_YEAR = 1512

CashMode = Literal[
    "cash_0pct",
    "cash_3pct_constant",
    "historical_tbill_yield",
    "historical_tbill_yield_minus_25bps",
    "historical_tbill_yield_minus_50bps",
]

FinancingMode = Literal[
    "no_financing_cost",
    "historical_financing_proxy",
    "historical_financing_proxy_plus_50bps",
    "historical_financing_proxy_plus_100bps",
]

SyntheticVariant = Literal[
    "qqq_3x_raw",
    "qqq_3x_financing_adjusted",
    "qqq_3x_conservative",
]

ALLOWED_CASH_MODES: tuple[str, ...] = (
    "cash_0pct",
    "cash_3pct_constant",
    "historical_tbill_yield",
    "historical_tbill_yield_minus_25bps",
    "historical_tbill_yield_minus_50bps",
)

DISALLOWED_RISK_OFF_HINTS = (
    "TLT",
    "IEF",
    "GLD",
    "SQQQ",
    "inverse",
    "gold",
    "bond_rotation",
    "managed_futures",
    "high_yield",
    "corporate_bond",
    "options",
    "volatility",
)


@dataclass(frozen=True)
class SyntheticAccountingAssumptions:
    """Assumptions for a synthetic leveraged-return series."""

    leverage: float = 3.0
    expense_ratio: float = 0.0095
    conservative_extra_drag: float = 0.0050
    annualization: int = TRADING_BARS_PER_YEAR


@dataclass(frozen=True)
class CashSeriesInfo:
    """Metadata for a generated cash-return series."""

    mode: str
    annualization: int
    source: str
    missing_fill: str
    coverage: float
    haircut_bps: float


def validate_cash_mode(mode: str) -> None:
    """Validate that a risk-off mode is cash/T-bill-only."""
    if mode not in ALLOWED_CASH_MODES:
        lowered = mode.lower()
        if any(hint.lower() in lowered for hint in DISALLOWED_RISK_OFF_HINTS):
            raise ValueError(
                f"Risk-off mode {mode!r} is not allowed; risk-off is restricted to cash/T-bills."
            )
        raise ValueError(f"Unknown cash mode {mode!r}; allowed modes are {ALLOWED_CASH_MODES}")


def annual_yield_to_period_return(annual_yield: float, annualization: int) -> float:
    """Convert an annual simple yield assumption to a compounded period return."""
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    return float((1.0 + annual_yield) ** (1.0 / annualization) - 1.0)


def cash_returns_for_mode(
    index: pd.Index,
    mode: CashMode | str,
    *,
    annualization: int = TRADING_BARS_PER_YEAR,
    tbill_proxy_returns: pd.Series | None = None,
    missing_fill: Literal["zero", "cash_3pct_constant"] = "zero",
) -> tuple[pd.Series, CashSeriesInfo]:
    """Return per-bar cash/T-bill sleeve returns for the requested mode.

    Historical modes use a supplied T-bill/short-Treasury proxy return series,
    typically BIL adjusted-price returns in this repository.  Missing proxy bars
    are filled conservatively by default with 0% cash return; callers can request
    a 3% fixed-cash fallback for sensitivity checks.
    """
    validate_cash_mode(str(mode))
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    out_index = pd.Index(index)
    if mode == "cash_0pct":
        series = pd.Series(0.0, index=out_index, name=str(mode))
        info = CashSeriesInfo(str(mode), annualization, "constant", missing_fill, 1.0, 0.0)
        return series, info
    if mode == "cash_3pct_constant":
        period = annual_yield_to_period_return(0.03, annualization)
        series = pd.Series(period, index=out_index, name=str(mode))
        info = CashSeriesInfo(str(mode), annualization, "constant", missing_fill, 1.0, 0.0)
        return series, info

    if tbill_proxy_returns is None:
        raise ValueError(f"{mode} requires tbill_proxy_returns")
    proxy = tbill_proxy_returns.sort_index().astype(float).reindex(out_index)
    coverage = float(proxy.notna().mean()) if len(proxy) else 0.0
    if missing_fill == "zero":
        filled = proxy.fillna(0.0)
    elif missing_fill == "cash_3pct_constant":
        filled = proxy.fillna(annual_yield_to_period_return(0.03, annualization))
    else:
        raise ValueError(f"unsupported missing_fill: {missing_fill}")

    haircut = {
        "historical_tbill_yield": 0.0,
        "historical_tbill_yield_minus_25bps": 0.0025,
        "historical_tbill_yield_minus_50bps": 0.0050,
    }[str(mode)]
    adjusted = filled - annual_yield_to_period_return(haircut, annualization)
    adjusted = adjusted.clip(lower=-0.999).rename(str(mode))
    info = CashSeriesInfo(
        str(mode),
        annualization,
        "historical_tbill_proxy",
        missing_fill,
        coverage,
        haircut * 10_000.0,
    )
    return adjusted, info


def financing_returns_for_mode(
    index: pd.Index,
    mode: FinancingMode | str,
    *,
    annualization: int = TRADING_BARS_PER_YEAR,
    financing_proxy_returns: pd.Series | None = None,
    missing_fill: Literal["zero", "cash_3pct_constant"] = "zero",
) -> pd.Series:
    """Return per-bar financing-rate proxy returns for borrowed notional."""
    out_index = pd.Index(index)
    if mode == "no_financing_cost":
        return pd.Series(0.0, index=out_index, name=str(mode))
    spread = {
        "historical_financing_proxy": 0.0,
        "historical_financing_proxy_plus_50bps": 0.0050,
        "historical_financing_proxy_plus_100bps": 0.0100,
    }.get(str(mode))
    if spread is None:
        raise ValueError(f"Unknown financing mode {mode!r}")
    if financing_proxy_returns is None:
        raise ValueError(f"{mode} requires financing_proxy_returns")
    proxy = financing_proxy_returns.sort_index().astype(float).reindex(out_index)
    if missing_fill == "zero":
        proxy = proxy.fillna(0.0)
    elif missing_fill == "cash_3pct_constant":
        proxy = proxy.fillna(annual_yield_to_period_return(0.03, annualization))
    else:
        raise ValueError(f"unsupported missing_fill: {missing_fill}")
    return (proxy + annual_yield_to_period_return(spread, annualization)).rename(str(mode))


def synthetic_leveraged_return_variant(
    raw_leveraged_returns: pd.Series,
    *,
    variant: SyntheticVariant | str,
    financing_returns: pd.Series | None = None,
    assumptions: SyntheticAccountingAssumptions | None = None,
) -> pd.Series:
    """Create a documented synthetic leveraged-return variant.

    Parameters
    ----------
    raw_leveraged_returns:
        Existing raw synthetic leveraged returns.  In this project this is
        normally the return series from `QQQ_3X_CALC`, whose daily close-to-close
        path follows the perfect daily-reset +3x rule.
    variant:
        - `qqq_3x_raw`: no additional accounting drag.
        - `qqq_3x_financing_adjusted`: subtract financing on borrowed notional
          plus expense ratio.
        - `qqq_3x_conservative`: financing-adjusted plus extra tracking drag.
    financing_returns:
        Per-bar financing proxy returns.  Required for adjusted/conservative
        variants; use all-zero series for the explicit no-financing case.
    """
    assumptions = assumptions or SyntheticAccountingAssumptions()
    if assumptions.leverage <= 0:
        raise ValueError("leverage must be positive")
    if assumptions.annualization <= 0:
        raise ValueError("annualization must be positive")
    raw = raw_leveraged_returns.astype(float).sort_index().fillna(0.0)
    if variant == "qqq_3x_raw":
        return raw.rename(str(variant))
    if variant not in {"qqq_3x_financing_adjusted", "qqq_3x_conservative"}:
        raise ValueError(f"Unknown synthetic variant {variant!r}")
    if financing_returns is None:
        raise ValueError(f"{variant} requires financing_returns")

    financing = financing_returns.reindex(raw.index).fillna(0.0).astype(float)
    borrowed_notional = max(float(assumptions.leverage) - 1.0, 0.0)
    expense = annual_yield_to_period_return(float(assumptions.expense_ratio), assumptions.annualization)
    adjusted = raw - borrowed_notional * financing - expense
    if variant == "qqq_3x_conservative":
        extra = annual_yield_to_period_return(
            float(assumptions.conservative_extra_drag), assumptions.annualization
        )
        adjusted = adjusted - extra
    return adjusted.clip(lower=-0.999).rename(str(variant))


def price_from_returns(returns: pd.Series, *, initial_price: float = 100.0) -> pd.Series:
    """Create a synthetic price path from simple returns."""
    if initial_price <= 0:
        raise ValueError("initial_price must be positive")
    clean = returns.fillna(0.0).astype(float)
    return (initial_price * (1.0 + clean).cumprod()).rename("synthetic_price")


def simulate_after_tax_portfolio_with_cash_returns(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    cash_returns: pd.Series,
    transaction_cost_bps: float,
    slippage_bps: float,
    tax_rate: float,
    cash_interest_tax_rate: float | None = None,
    liquidate_at_end: bool = True,
    rebalance_on_weight_change_only: bool = True,
    ledger=None,
    finalize_tax_at_end: bool | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compatibility facade over the shared, self-financing research ledger.

    Legacy inputs are beginning-of-interval/return-aligned targets. The facade
    retains intersection/zero-fill behavior, but the new ledger itself rejects
    missing data. Both capital-gain categories use ``tax_rate`` here; use
    ``research_ledger.simulate_ledger`` for separate rates, wash-sale scenarios,
    dated holdings, and order events. These are hypothetical tax scenarios.

    Costs are cash-inclusive, terminal sales incur costs, and annual taxes are
    paid at the first event of the next year (before its return). A terminal run
    also settles the current year. To continue a fold, supply a persistent
    PortfolioLedger and set liquidate_at_end=False; this defaults to no terminal
    tax settlement as well. Existing report files are not rewritten.
    """
    from trend_following.research_ledger import LedgerConfig, simulate_ledger

    if cash_interest_tax_rate is None:
        cash_interest_tax_rate = tax_rate
    config = LedgerConfig(
        fee_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        short_tax_rate=tax_rate,
        long_tax_rate=tax_rate,
        interest_tax_rate=cash_interest_tax_rate,
        rebalance_policy="on_target_change" if rebalance_on_weight_change_only else "every_bar",
    )
    common = returns.index.intersection(target_weights.index).intersection(cash_returns.index)
    columns = [column for column in returns.columns if column in target_weights.columns]
    if len(common) == 0 or len(columns) == 0:
        return tuple(pd.Series(index=common, dtype=float, name=name) for name in
                     ("after_tax_return", "tax_paid", "turnover", "cash_interest", "cash_weight"))
    result = simulate_ledger(
        returns.loc[common, columns].fillna(0.0).astype(float),
        target_weights.loc[common, columns].fillna(0.0).astype(float),
        cash_returns.loc[common].fillna(0.0).astype(float),
        config if ledger is None else ledger.config,
        liquidate_at_end=liquidate_at_end,
        finalize_tax_at_end=(liquidate_at_end if finalize_tax_at_end is None else finalize_tax_at_end),
        ledger=ledger,
        targets_at="interval_start",
    )
    return (result.returns.rename("after_tax_return"), result.taxes_paid.rename("tax_paid"),
            result.turnover, result.cash_interest, result.cash_weight)
