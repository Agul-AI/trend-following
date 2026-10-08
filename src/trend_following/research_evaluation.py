"""Train-only chronological selection and matched continuous-ledger evaluation.

All historical reconstruction is retrospective, even with nested splits.
There is no automatic live execution or optimizer of outer-test performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from trend_following.metrics import max_drawdown, sharpe_ratio
from trend_following.research_data import calendar_year_fraction


@dataclass(frozen=True)
class CostScenario:
    name: str
    fee_bps: float = 1.0
    slippage_bps: float = 5.0
    financing_spread_bps: float = 50.0
    expense_ratio: float = 0.0095
    extra_drag: float = 0.0050
    short_tax_rate: float = 0.0
    long_tax_rate: float = 0.0
    interest_tax_rate: float = 0.0
    wash_sales: bool = False
    fill_delay_bars: int = 1
    cash_multiplier: float = 1.0
    include_financing: bool = True
    financing_day_count: float = 360.0


@dataclass(frozen=True)
class ChronologicalFold:
    name: str
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def chronological_folds(
    index: pd.DatetimeIndex, *, first_test_year: int = 2009, test_years: int = 3
) -> list[ChronologicalFold]:
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("unique chronological timestamps required")
    folds = []
    for year in range(first_test_year, index[-1].year + 1, test_years):
        mask = (index.year >= year) & (index.year < year + test_years)
        dates = index[mask]
        train = index[index < dates[0]] if len(dates) else index[:0]
        if len(dates) and len(train):
            folds.append(
                ChronologicalFold(
                    f"{year}_{min(year + test_years - 1, index[-1].year)}",
                    train[-1],
                    dates[0],
                    dates[-1],
                )
            )
    return folds


def daily_returns(series: pd.Series) -> pd.Series:
    return series.add(1).groupby(series.index.normalize()).prod().sub(1)


def research_metrics(
    returns: pd.Series,
    cash: pd.Series,
    *,
    turnover: pd.Series | None = None,
    actual_exposure: pd.Series | None = None,
) -> dict[str, float]:
    daily = daily_returns(returns)
    aligned_cash = cash.reindex(returns.index)
    if aligned_cash.isna().any():
        raise ValueError("missing matched cash benchmark")
    rf = daily_returns(aligned_cash)
    years = len(daily) / 252
    wealth = float((1 + daily).prod())
    return {
        "CAGR": wealth ** (1 / years) - 1 if years else np.nan,
        "excess_return_Sharpe": sharpe_ratio(daily, risk_free_returns=rf),
        "zero_RF_Sharpe": sharpe_ratio(daily),
        "max_drawdown": max_drawdown(daily),
        "terminal_wealth": wealth,
        "observations": len(returns),
        "sessions": len(daily),
        "turnover": float(turnover.sum()) if turnover is not None else np.nan,
        "position_changes": int((turnover > 1e-12).sum()) if turnover is not None else 0,
        "mean_effective_exposure": float(actual_exposure.mean())
        if actual_exposure is not None
        else np.nan,
    }


def matched_cash_benchmark(
    market_cash: pd.Series, dates: pd.DatetimeIndex, interest_tax_rate: float = 0.0
) -> pd.Series:
    """Common market cash opportunity cost, with the same tax settlement model.

    The zero-cash-yield sensitivity does not change this benchmark. Taxable
    comparisons use an actual all-cash ledger, not immediate-tax accruals.
    """
    cash = market_cash.reindex(dates).copy()
    if cash.isna().any():
        raise ValueError("missing market cash benchmark")
    cash.iloc[0] = 0.0
    if not interest_tax_rate:
        return cash
    from trend_following.research_ledger import LedgerConfig, simulate_ledger

    zeros = pd.DataFrame({"risk": 0.0}, index=dates)
    baseline = simulate_ledger(
        zeros,
        zeros,
        cash,
        LedgerConfig(interest_tax_rate=interest_tax_rate),
        targets_at="close",
        liquidate_at_end=True,
        finalize_tax_at_end=True,
    )
    return baseline.returns


def scenario_asset_returns(
    raw_price: pd.Series,
    financing_rate_returns: pd.Series,
    scenario: CostScenario,
    *,
    fund_costs_embedded: bool = False,
    leverage: float = 3,
) -> pd.Series:
    returns = raw_price.pct_change().fillna(0)
    if fund_costs_embedded:
        return returns.rename("risk")
    accrual = financing_rate_returns.reindex(returns.index)
    if accrual.isna().any():
        raise ValueError("financing series does not cover prices")
    dt = calendar_year_fraction(returns.index)
    financing = (
        accrual + scenario.financing_spread_bps / 10000 * dt * 365 / scenario.financing_day_count
    )
    drag = (
        max(leverage - 1, 0) * financing
        if scenario.include_financing
        else pd.Series(0.0, index=returns.index)
    )
    drag += (scenario.expense_ratio + scenario.extra_drag) * dt
    adjusted = returns - drag
    if adjusted.le(-1).any():
        raise ValueError("nonpositive leveraged NAV; bankruptcy cannot be clipped away")
    return adjusted.rename("risk")


def evaluate_path(
    qqq: pd.Series,
    raw_risk: pd.Series,
    financing: pd.Series,
    cash: pd.Series,
    spec: dict[str, Any],
    scenario: CostScenario,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    schedule: dict[pd.Timestamp, dict[str, Any]] | None = None,
    bars_per_day: int = 6,
    fund_costs_embedded: bool = False,
    fixed_weights: pd.Series | None = None,
    physical_returns: pd.Series | None = None,
    physical_prices: pd.Series | None = None,
    corporate_actions: pd.DataFrame | None = None,
):
    from trend_following.research_execution import build_execution_path
    from trend_following.research_ledger import LedgerConfig, simulate_ledger

    end = end or qqq.index[-1]
    all_dates = qqq.index[qqq.index <= end]
    qqq = qqq.reindex(all_dates)
    raw_risk = raw_risk.reindex(all_dates)
    risk_returns = scenario_asset_returns(
        raw_risk, financing, scenario, fund_costs_embedded=fund_costs_embedded
    )
    modeled_price = 100 * (1 + risk_returns).cumprod()
    effective_spec = dict(
        spec, delay_bars=scenario.fill_delay_bars, fill_delay_bars=scenario.fill_delay_bars
    )
    effective_schedule = {
        date: dict(
            value, delay_bars=scenario.fill_delay_bars, fill_delay_bars=scenario.fill_delay_bars
        )
        for date, value in (schedule or {}).items()
    }
    path = build_execution_path(
        qqq,
        modeled_price,
        effective_spec,
        evaluate_start=start,
        bars_per_day=bars_per_day,
        candidate_schedule=effective_schedule or None,
        slippage_bps=scenario.slippage_bps,
        transaction_cost_bps=scenario.fee_bps,
    )
    dates = all_dates[all_dates >= start]
    targets = (
        path.weights_at_close.reindex(dates)
        if fixed_weights is None
        else fixed_weights.reindex(dates)
    )
    config = LedgerConfig(
        fee_bps=scenario.fee_bps,
        slippage_bps=scenario.slippage_bps,
        short_tax_rate=scenario.short_tax_rate,
        long_tax_rate=scenario.long_tax_rate,
        interest_tax_rate=scenario.interest_tax_rate,
        wash_sales=scenario.wash_sales,
        rebalance_policy="on_target_change",
        exposure_multipliers={"risk": float(spec.get("product_leverage", 3))},
    )
    account_cash = cash.reindex(dates).copy() * scenario.cash_multiplier
    account_cash.iloc[0] = 0.0  # no pre-evaluation warm-up cash P&L
    physical_initial = None
    if physical_prices is not None:
        previous = physical_prices.loc[physical_prices.index < dates[0]]
        if len(previous):
            physical_initial = float(previous.iloc[-1])
        else:
            first_split = 1.0
            if corporate_actions is not None:
                first_actions = corporate_actions.loc[corporate_actions.timestamp.eq(dates[0])]
                if len(first_actions):
                    first_split = float(first_actions.iloc[0].split_ratio)
            physical_initial = (
                float(physical_prices.loc[dates[0]])
                * first_split
                / (1 + float(physical_returns.loc[dates[0]]))
            )
    ledger = simulate_ledger(
        (risk_returns if physical_returns is None else physical_returns)
        .reindex(dates)
        .to_frame("risk"),
        targets.to_frame("risk"),
        account_cash,
        config,
        liquidate_at_end=True,
        targets_at="close",
        initial_prices={"risk": physical_initial} if physical_initial is not None else None,
        corporate_actions=corporate_actions.loc[corporate_actions.timestamp.isin(dates)]
        if corporate_actions is not None
        else None,
        price_basis="unadjusted" if corporate_actions is not None else "total_return",
    )
    rf = matched_cash_benchmark(cash, dates, scenario.interest_tax_rate)
    exposure = ledger.holdings.groupby("timestamp").effective_exposure.sum().reindex(dates)
    metrics = research_metrics(
        ledger.returns, rf, turnover=ledger.turnover, actual_exposure=exposure
    )
    trades = ledger.events.loc[ledger.events.event.isin(["buy", "sell"])]
    metrics["filled_orders"] = len(trades)
    metrics["tax_funding_orders"] = int(trades.reason.eq("tax_funding").sum())
    metrics["total_trading_fees"] = float(trades.fee.sum())
    metrics["total_slippage"] = float(trades.slippage.sum())
    metrics["total_taxes_paid"] = float(ledger.taxes_paid.sum())
    return path, ledger, dict(metrics, **asdict(scenario), evaluation_label="retrospective")


def inner_validation_windows(
    index: pd.DatetimeIndex,
    train_end: pd.Timestamp,
    *,
    minimum_training_years: int = 3,
    maximum_validation_years: int | None = None,
    first_validation_year: int = 2005,
    minimum_inner_observations: int = 200,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    training = index[index <= train_end]
    first_year = max(first_validation_year, training[0].year + minimum_training_years)
    windows = []
    for year in range(first_year, train_end.year + 1):
        dates = training[training.year == year]
        # Only completed calendar-year inner blocks; no outer observations.
        if len(dates.normalize().unique()) >= minimum_inner_observations and (
            year < train_end.year or train_end.month == 12
        ):
            windows.append((dates[0], dates[-1]))
    return windows[-maximum_validation_years:] if maximum_validation_years else windows


def select_candidate(
    qqq: pd.Series,
    risk: pd.Series,
    financing: pd.Series,
    cash: pd.Series,
    candidates: dict[str, dict[str, Any]],
    fold: ChronologicalFold,
    scenario: CostScenario,
    *,
    minimum_training_years: int = 3,
    maximum_validation_years: int | None = None,
    max_drawdown_limit: float = -0.45,
    bars_per_day: int = 6,
    cache: dict | None = None,
    first_validation_year: int = 2005,
    minimum_inner_observations: int = 200,
    minimum_complete_validation_years: int = 2,
    tie_tolerance: float = 1e-8,
) -> tuple[str, pd.DataFrame]:
    windows = inner_validation_windows(
        qqq.index,
        fold.train_end,
        minimum_training_years=minimum_training_years,
        maximum_validation_years=maximum_validation_years,
        first_validation_year=first_validation_year,
        minimum_inner_observations=minimum_inner_observations,
    )
    rows = []
    memo = cache if cache is not None else {}
    for candidate, spec in candidates.items():
        for start, end in windows:
            key = (
                candidate,
                start.isoformat(),
                end.isoformat(),
                repr(scenario),
                repr(sorted(spec.items())),
            )
            if key not in memo:
                _, _, metrics = evaluate_path(
                    qqq.loc[:end],
                    risk.loc[:end],
                    financing.loc[:end],
                    cash.loc[:end],
                    spec,
                    scenario,
                    start=start,
                    end=end,
                    bars_per_day=bars_per_day,
                )
                memo[key] = metrics
            rows.append(
                dict(
                    memo[key],
                    candidate_id=candidate,
                    validation_start=start,
                    validation_end=end,
                    outer_fold=fold.name,
                    selection_cutoff=fold.train_end,
                )
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return "cash", table
    ranking = []
    for candidate, part in table.groupby("candidate_id"):
        if candidate == "cash":
            continue
        scores = part.excess_return_Sharpe
        qualifies = (
            len(part) >= minimum_complete_validation_years
            and part.max_drawdown.ge(max_drawdown_limit).all()
            and np.isfinite(scores).all()
            and scores.median() > 0
        )
        if qualifies:
            ranking.append(
                (
                    float(scores.median()),
                    int(candidates[candidate].get("complexity_rank", 100)),
                    candidate,
                )
            )
    if ranking:
        best = max(row[0] for row in ranking)
        tied = [row for row in ranking if best - row[0] <= tie_tolerance]
        chosen = min(tied, key=lambda row: (row[1], row[2]))[2]
    else:
        chosen = "cash"
    table["selected_candidate"] = chosen
    return chosen, table


def nested_schedule(
    qqq: pd.Series,
    risk: pd.Series,
    financing: pd.Series,
    cash: pd.Series,
    candidates: dict[str, dict[str, Any]],
    folds: list[ChronologicalFold],
    scenario: CostScenario,
    **selection_options,
):
    schedule = {}
    selections = []
    inner = []
    cache = {}
    for fold in folds:
        chosen, table = select_candidate(
            qqq, risk, financing, cash, candidates, fold, scenario, cache=cache, **selection_options
        )
        schedule[fold.test_start] = (
            candidates[chosen] if chosen != "cash" else {"id": "cash", "kind": "cash"}
        )
        selections.append(
            dict(
                asdict(fold),
                selected_candidate=chosen,
                candidate_pool=",".join(candidates),
                evaluation_label="retrospective_nested",
            )
        )
        inner.append(table)
    return (
        schedule,
        pd.DataFrame(selections),
        pd.concat(inner, ignore_index=True) if inner else pd.DataFrame(),
    )


def waterfall_scenarios(base: CostScenario) -> list[CostScenario]:
    from dataclasses import replace

    return [
        replace(
            base,
            name="gross",
            fee_bps=0,
            slippage_bps=0,
            financing_spread_bps=0,
            expense_ratio=0,
            extra_drag=0,
            short_tax_rate=0,
            long_tax_rate=0,
            interest_tax_rate=0,
            include_financing=False,
        ),
        replace(
            base,
            name="trading_costs",
            financing_spread_bps=0,
            expense_ratio=0,
            extra_drag=0,
            short_tax_rate=0,
            long_tax_rate=0,
            interest_tax_rate=0,
            include_financing=False,
        ),
        replace(
            base, name="financing_expenses", short_tax_rate=0, long_tax_rate=0, interest_tax_rate=0
        ),
        replace(base, name="hypothetical_taxes"),
    ]
