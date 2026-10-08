"""Fresh daily MA family for a sealed chronological research split.

This module does not import the B5 engine or old synthetic cache. Signals use
causal QQQ total returns and complete daily observations. NY16:00 is a declared
daily accounting clock, NOT a claim about intraday availability or early closes.
Old market history remains retrospective research, even with a held-out split.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from trend_following.research_ledger import (
    LedgerConfig,
    LedgerResult,
    PortfolioLedger,
    simulate_ledger,
)
from trend_following.research_rates import rate_accrual

LAYERS = ("gross", "trading", "financing", "expense", "tax")


@dataclass(frozen=True)
class SplitCosts:
    initial_cash: float = 1000.0
    fee_bps: float = 1.0
    slippage_bps: float = 5.0
    financing_spread_bps: float = 50.0
    expense_ratio: float = 0.0095
    extra_drag: float = 0.0050
    short_tax_rate: float = 0.24
    long_tax_rate: float = 0.15
    interest_tax_rate: float = 0.24
    wash_sales: bool = True
    max_rate_age_days: float = 10.0

    def __post_init__(self):
        numeric = {k: v for k, v in asdict(self).items() if k != "wash_sales"}
        if not np.isfinite(list(numeric.values())).all():
            raise ValueError("Cost assumptions must be finite")
        if self.initial_cash <= 0 or self.max_rate_age_days <= 0:
            raise ValueError("Positive capital and maximum rate age required")
        if min(self.expense_ratio, self.extra_drag) < 0:
            raise ValueError("Fund expense and extra drag must be nonnegative")
        LedgerConfig(
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
            short_tax_rate=self.short_tax_rate,
            long_tax_rate=self.long_tax_rate,
            interest_tax_rate=self.interest_tax_rate,
            wash_sales=self.wash_sales,
        )


DEFAULT_COSTS = SplitCosts()


@dataclass
class DailyPanel:
    data: pd.DataFrame
    metadata: dict[str, Any]


@dataclass
class SplitEvaluation:
    metrics: pd.DataFrame
    daily_returns: pd.DataFrame
    paths: dict[str, LedgerResult]
    terminal_snapshots: dict[str, dict]
    target_weights: pd.Series
    metadata: dict[str, Any]


class _DailyDistributionLedger(PortfolioLedger):
    """Same-close dividend cash cannot earn the preceding interval's cash rate.

    The frozen generic ledger processes corporate actions before cash accrual.
    This local adapter retains its FIFO/tax/DRIP mechanics but temporarily sets
    aside newly credited dividend cash when accruing the ending cash interval.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._same_close_distribution = 0.0

    def credit_distribution(self, *args, **kwargs):
        received = super().credit_distribution(*args, **kwargs)
        self._same_close_distribution += received
        return received

    def accrue_cash(self, period_return, timestamp):
        new_distribution = self._same_close_distribution
        self.cash -= new_distribution
        try:
            return super().accrue_cash(period_return, timestamp)
        finally:
            self.cash += new_distribution
            self._same_close_distribution = 0.0


def _day_clock(value: Any) -> pd.Timestamp:
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError("Missing session date")
    if date.tzinfo is not None:
        date = date.tz_convert("America/New_York").tz_localize(None)
    return (date.normalize() + pd.Timedelta(hours=16)).tz_localize("America/New_York")


def prepare_daily_panel(raw_daily: pd.DataFrame, *, end: Any | None = None) -> DailyPanel:
    """Build causal daily TR and split-price indices from raw daily action data.

    Dividend units are POST-split shares. TR[t] =
    split[t]*(close[t]+dividend[t])/close[t-1]-1. The initial row is an index
    normalization, not an invented prehistory return. Adjusted close is retained
    for source QA only; it does not supply signals or economic returns.
    """
    frame = raw_daily.copy()
    required = {"close", "adj_close", "dividend_amount", "split_coefficient"}
    if not required.issubset(frame):
        raise ValueError("Raw daily close/adj_close/dividend_amount/split_coefficient required")
    dates = frame.pop("date") if "date" in frame else frame.index
    frame.index = pd.DatetimeIndex([_day_clock(x) for x in dates], name="timestamp")
    if end is not None:
        frame = frame.loc[frame.index <= _day_clock(end)].copy()
    if (
        frame.empty
        or not frame.index.is_unique
        or not frame.index.is_monotonic_increasing
        or (frame.index.dayofweek >= 5).any()
    ):
        raise ValueError("Nonempty chronological unique weekday sessions required")
    numeric = frame[list(required)].astype(float)
    if (
        not np.isfinite(numeric.to_numpy()).all()
        or numeric[["close", "adj_close", "split_coefficient"]].le(0).any().any()
        or numeric.dividend_amount.lt(0).any()
    ):
        raise ValueError("Finite positive prices/splits and nonnegative dividends required")
    out = pd.DataFrame(
        {
            "raw_close": numeric.close,
            "adj_close": numeric.adj_close,
            "dividend": numeric.dividend_amount,
            "split_ratio": numeric.split_coefficient,
        }
    )
    out["qqq_total_return"] = (
        out.raw_close + out.dividend
    ) * out.split_ratio / out.raw_close.shift(1) - 1
    out.iloc[0, out.columns.get_loc("qqq_total_return")] = 0.0
    if out.qqq_total_return.le(-1).any():
        raise ValueError("Nonpositive underlying total-return wealth")
    out["total_return_index"] = 100 * (1 + out.qqq_total_return).cumprod()
    out["split_price_index"] = out.raw_close * out.split_ratio.cumprod()
    out["split_neutral_price_return"] = out.raw_close * out.split_ratio / out.raw_close.shift(1) - 1
    out.iloc[0, out.columns.get_loc("split_neutral_price_return")] = 0.0
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Daily normalized index overflow")
    return DailyPanel(
        out,
        {
            "signal_basis": "causal_QQQ_total_return_index_from_raw_close_dividends_splits",
            "daily_clock": "16:00_America_New_York_accounting_only_not_actual_early_close_availability",
            "adjusted_close_role": "source_QA_only_not_signal_or_ledger_input",
            "distribution_units": "post_split_share_units",
            "session_coverage": "caller_must_audit_complete_exchange_sessions_not_inferred_from_weekdays",
            "source_vintage": "retrospective_source_not_point_in_time_vintage",
        },
    )


def default_candidates() -> dict[str, dict]:
    return {
        f"ma{window}_alloc_{units}_3": {
            "name": f"ma{window}_alloc_{units}_3",
            "kind": "ma",
            "asset": "synthetic_3x",
            "ma_sessions": window,
            "allocation": units / 3,
        }
        for window in (50, 100, 200)
        for units in (1, 2, 3)
    }


def benchmark_specs() -> dict[str, dict]:
    return {
        "qqq_buy_hold": {
            "name": "qqq_buy_hold",
            "kind": "buy_hold",
            "asset": "QQQ",
            "allocation": 1.0,
        },
        "cash": {"name": "cash", "kind": "cash", "asset": "QQQ", "allocation": 0.0},
        "qqq200ma": {
            "name": "qqq200ma",
            "kind": "ma",
            "asset": "QQQ",
            "ma_sessions": 200,
            "allocation": 1.0,
        },
    }


def _spec(spec: dict) -> dict:
    p = dict(spec)
    if not {"name", "kind", "asset", "allocation"}.issubset(p):
        raise ValueError("Strategy requires name/kind/asset/allocation")
    if p["kind"] not in {"ma", "buy_hold", "cash"} or p["asset"] not in {"QQQ", "synthetic_3x"}:
        raise ValueError("Only declared daily MA/buyhold/cash and QQ/synthetic3x assets supported")
    if not np.isfinite(p["allocation"]) or not 0 <= p["allocation"] <= 1:
        raise ValueError("Allocation must lie in [0,1]")
    if p["asset"] == "QQQ" and p["allocation"] not in (0.0, 1.0):
        raise ValueError(
            "Physical QQQ controls require full 0/1 targets, not fractional rebalancing"
        )
    if p["kind"] == "ma" and p.get("ma_sessions") not in (50, 100, 200):
        raise ValueError("Only predeclared 50/100/200 completed-session MAs supported")
    return p


def _through(panel: DailyPanel, end: Any) -> pd.DataFrame:
    data = panel.data.loc[panel.data.index <= _day_clock(end)].copy()
    required = {
        "raw_close",
        "dividend",
        "split_ratio",
        "total_return_index",
        "qqq_total_return",
        "split_neutral_price_return",
    }
    if not required.issubset(data) or data.empty:
        raise ValueError("Empty or incomplete daily panel")
    if (
        not isinstance(data.index, pd.DatetimeIndex)
        or data.index.tz is None
        or not data.index.is_unique
        or not data.index.is_monotonic_increasing
    ):
        raise ValueError("Daily panel requires aware chronological unique timestamps")
    if not np.isfinite(data[list(required)].to_numpy()).all():
        raise ValueError("Nonfinite daily panel through requested cutoff")
    return data


def build_daily_ma_targets(panel: DailyPanel, spec: dict, *, start: Any, end: Any) -> pd.DataFrame:
    """One completed-session lag; warmup supplies no positions or pending orders."""
    p = _spec(spec)
    data = _through(panel, end)
    dates = data.index[data.index >= _day_clock(start)]
    if len(dates) < 2:
        raise ValueError("At least two in-segment completed sessions required")
    if p["kind"] == "ma":
        ma = data.total_return_index.rolling(p["ma_sessions"], min_periods=p["ma_sessions"]).mean()
        if not np.isfinite(ma.loc[dates[0]]):
            raise ValueError("Insufficient completed-session feature-only warmup")
        desired = (
            data.total_return_index.loc[dates].gt(ma.loc[dates]).astype(float) * p["allocation"]
        )
    else:
        ma = pd.Series(np.nan, index=data.index)
        desired = pd.Series(0.0 if p["kind"] == "cash" else p["allocation"], index=dates)
    targets = desired.shift(1, fill_value=0.0)
    changes = targets.diff().fillna(targets.iloc[0]).abs().gt(1e-12)
    decisions = pd.DataFrame(
        {
            "signal_index": data.total_return_index.loc[dates],
            "moving_average": ma.loc[dates],
            "decision_target": desired,
            "weights_at_close": targets,
            "target_change": changes,
        }
    )
    decisions["decision_timestamp_for_fill"] = pd.Series(dates, index=dates).shift(1)
    decisions["fill_timestamp"] = pd.Series(dates, index=dates).where(changes)
    return decisions


def _known_rates(observations: pd.DataFrame, end: pd.Timestamp, series_id: str) -> pd.DataFrame:
    if not {"available_at", "annual_rate"}.issubset(observations):
        raise ValueError("Rates require available_at and annual_rate; no implicit zero rate")
    availability = pd.DatetimeIndex(observations.available_at)
    if availability.tz is None:
        raise ValueError("Rate availability timestamps must be timezone-aware")
    known = observations.loc[availability <= end].copy()
    if "series_id" in known and not known.series_id.eq(series_id).all():
        raise ValueError(f"Expected independent {series_id} rate source")
    return known


def synthetic_daily_returns(
    panel: DailyPanel,
    financing_observations: pd.DataFrame,
    *,
    end: Any,
    costs: SplitCosts = DEFAULT_COSTS,
    layer: str = "expense",
) -> pd.DataFrame:
    """Fresh 3x daily-reset total-return instrument, never historical TQQQ.

    Return = 3*underlyingTR - 2*(DFF+spread)*ACT/360 -
    (expense+extra_drag)*ACT/365, charged once on previous NAV. No leverage
    clipping or arbitrary -50% floor: zero/nonpositive NAV raises bankruptcy.
    """
    if layer not in LAYERS:
        raise ValueError("Unknown cost layer")
    data = _through(panel, end)
    gross = 3 * data.qqq_total_return
    borrow = pd.Series(0.0, index=data.index)
    expense = pd.Series(0.0, index=data.index)
    if layer in {"financing", "expense", "tax"}:
        borrow = 2 * rate_accrual(
            data.index,
            _known_rates(financing_observations, data.index[-1], "DFF"),
            spread_bps=costs.financing_spread_bps,
            day_count=360.0,
            max_age_days=costs.max_rate_age_days,
        )
    if layer in {"expense", "tax"}:
        dt = data.index.to_series().diff().dt.total_seconds().fillna(0) / (365 * 86400)
        expense = (costs.expense_ratio + costs.extra_drag) * dt
    modeled = gross - borrow - expense
    if not np.isfinite(modeled).all() or modeled.le(-1).any():
        raise ValueError("Nonpositive synthetic NAV: modeled bankruptcy, never clipped")
    nav = 100 * (1 + modeled).cumprod()
    if not np.isfinite(nav).all() or nav.le(0).any():
        raise ValueError("Synthetic NAV overflow/exhaustion")
    return pd.DataFrame(
        {
            "gross_return": gross,
            "financing_drag": borrow,
            "expense_tracking_drag": expense,
            "modeled_return": modeled,
            "synthetic_nav": nav,
        }
    )


def _terminal(ledger, stamp: pd.Timestamp, *, taxable: bool) -> dict:
    clone = copy.deepcopy(ledger)
    original_equity = float(clone.equity)
    offset = len(clone.events)
    clone.liquidate(stamp)
    if taxable:
        clone.expire_wash_losses(stamp, no_future_replacements=True)
        clone.settle_taxes(stamp, close_year=stamp.year)
    events = clone.events[offset:]
    orders = [row for row in events if row["event"] in {"buy", "sell"}]
    return {
        "marked_equity": original_equity,
        "liquidated_equity": float(clone.equity),
        "exit_fees": float(sum(row["fee"] for row in orders)),
        "exit_slippage": float(sum(row["slippage"] for row in orders)),
        "terminal_taxes_paid": float(
            -sum(
                row["cash_flow"] for row in events if row["event"] in {"tax_payment", "tax_refund"}
            )
        ),
        "terminal_filled_orders": len(orders),
        "terminal_turnover_notional": float(sum(row["turnover_notional"] for row in orders)),
        "ledger": clone,
    }


def _ledger_config(costs: SplitCosts, layer: str, asset: str) -> LedgerConfig:
    return LedgerConfig(
        fee_bps=0.0 if layer == "gross" else costs.fee_bps,
        slippage_bps=0.0 if layer == "gross" else costs.slippage_bps,
        short_tax_rate=costs.short_tax_rate if layer == "tax" else 0.0,
        long_tax_rate=costs.long_tax_rate if layer == "tax" else 0.0,
        interest_tax_rate=costs.interest_tax_rate if layer == "tax" else 0.0,
        wash_sales=costs.wash_sales if layer == "tax" else False,
        # QQQ controls have ONLY 0/1 targets: maintaining the full invested target
        # reinvests ex-date distribution cash at that close, with investor costs.
        # Synthetic fractional allocations retain target-change-only drift.
        rebalance_policy="every_bar" if asset == "QQQ" else "on_target_change",
        exposure_multipliers={asset: 3.0 if asset == "synthetic_3x" else 1.0},
    )


def _with_terminal(ledger: LedgerResult, terminal: dict) -> pd.Series:
    returns = ledger.returns.copy()
    returns.iloc[-1] = (1 + returns.iloc[-1]) * terminal["liquidated_equity"] / terminal[
        "marked_equity"
    ] - 1
    return returns


def _entry_factor(ledger: LedgerResult) -> float:
    buys = ledger.events.loc[ledger.events.event.eq("buy")]
    if buys.empty:
        return 1.0
    first = buys.iloc[0]
    cost = float(first.fee + first.slippage)
    # Before the first purchase the account may have earned cash interest.
    # Normalize by actual pre-purchase wealth, NOT blindly by initial capital.
    after = float(ledger.equity.loc[pd.Timestamp(first.timestamp)])
    return after / (after + cost)


def evaluate_daily_segment(
    panel: DailyPanel,
    financing_observations: pd.DataFrame,
    cash_observations: pd.DataFrame,
    spec: dict,
    *,
    start: Any,
    end: Any,
    costs: SplitCosts = DEFAULT_COSTS,
    layers: tuple[str, ...] = LAYERS,
) -> SplitEvaluation:
    """Flat independent segment; one unchanged intended target path for all costs."""
    if not layers or len(layers) != len(set(layers)) or any(x not in LAYERS for x in layers):
        raise ValueError("Unique declared cost layers required")
    p = _spec(spec)
    data = _through(panel, end)
    decisions = build_daily_ma_targets(panel, p, start=start, end=end)
    dates, targets = decisions.index, decisions.weights_at_close
    cash_all = rate_accrual(
        data.index,
        _known_rates(cash_observations, data.index[-1], "DTB3"),
        day_count=365.0,
        max_age_days=costs.max_rate_age_days,
    )
    account_cash = cash_all.loc[dates].copy()
    account_cash.iloc[0] = 0.0
    actions = None
    if p["asset"] == "QQQ":
        action_rows = data.loc[dates].loc[lambda x: x.dividend.ne(0) | x.split_ratio.ne(1)]
        actions = pd.DataFrame(
            {
                "timestamp": action_rows.index,
                "asset": "QQQ",
                "split_ratio": action_rows.split_ratio.to_numpy(),
                "distribution_per_share": action_rows.dividend.to_numpy(),
                "tax_kind": "ordinary",
            }
        )
    rows, paths, snapshots, daily = [], {}, {}, {}
    for layer in layers:
        config = _ledger_config(costs, layer, p["asset"])
        if p["asset"] == "synthetic_3x":
            synthetic = synthetic_daily_returns(
                panel, financing_observations, end=end, costs=costs, layer=layer
            )
            returns = synthetic.modeled_return.loc[dates]
            previous = synthetic.synthetic_nav.loc[synthetic.index < dates[0]]
            initial_price = float(previous.iloc[-1]) if len(previous) else 100.0
        else:
            returns = data.split_neutral_price_return.loc[dates]
            previous = data.raw_close.loc[data.index < dates[0]]
            initial_price = (
                float(previous.iloc[-1])
                if len(previous)
                else float(data.raw_close.iloc[0] * data.split_ratio.iloc[0])
            )
        initial_ledger = _DailyDistributionLedger(
            config, costs.initial_cash, {p["asset"]: initial_price}
        )
        ledger = simulate_ledger(
            returns.to_frame(p["asset"]),
            targets.to_frame(p["asset"]),
            account_cash,
            config,
            targets_at="close",
            liquidate_at_end=False,
            finalize_tax_at_end=False,
            ledger=initial_ledger,
            corporate_actions=actions,
            price_basis="unadjusted" if p["asset"] == "QQQ" else "total_return",
        )
        terminal = _terminal(ledger.ledger, dates[-1], taxable=layer == "tax")
        dr = _with_terminal(ledger, terminal)
        rf_config = LedgerConfig(
            interest_tax_rate=costs.interest_tax_rate if layer == "tax" else 0.0
        )
        zero = pd.DataFrame({"cash_reference": 0.0}, index=dates)
        rf_ledger = simulate_ledger(
            zero,
            zero,
            account_cash,
            rf_config,
            targets_at="close",
            initial_cash=costs.initial_cash,
            liquidate_at_end=False,
            finalize_tax_at_end=False,
        )
        rf_terminal = _terminal(rf_ledger.ledger, dates[-1], taxable=layer == "tax")
        rf = _with_terminal(rf_ledger, rf_terminal)
        excess = dr - rf
        volatility, excess_vol = float(dr.std(ddof=1)), float(excess.std(ddof=1))
        wealth = float(terminal["liquidated_equity"]) / costs.initial_cash
        compounded = (1 + dr).cumprod()
        dd = compounded / compounded.cummax().clip(lower=1.0) - 1
        if not np.isclose(float((1 + dr).prod()), wealth, rtol=1e-10):
            raise ValueError("Terminal wealth/return reconciliation failed")
        orders = ledger.orders
        continuing_buys = orders.loc[
            orders.event.eq("buy")
            & ~orders.timestamp.map(decisions.target_change).fillna(False).astype(bool)
        ]
        distribution_dates = ledger.events.loc[
            ledger.events.event.eq("distribution") & ledger.events.cash_flow.gt(0), "timestamp"
        ]
        entry_factor, terminal_factor = (
            _entry_factor(ledger),
            terminal["liquidated_equity"] / terminal["marked_equity"],
        )
        rows.append(
            {
                "candidate_id": p["name"],
                "layer": layer,
                "total_return": wealth - 1,
                "total_return_pct": (wealth - 1) * 100,
                "terminal_equity": terminal["liquidated_equity"],
                "marked_equity": terminal["marked_equity"],
                "CAGR": wealth ** (252 / len(dr)) - 1,
                "excess_return_Sharpe": float(excess.mean() / excess_vol * np.sqrt(252))
                if excess_vol > 1e-12
                else np.nan,
                "zero_RF_Sharpe": float(dr.mean() / volatility * np.sqrt(252))
                if volatility > 1e-12
                else np.nan,
                "annualized_volatility": volatility * np.sqrt(252),
                "max_drawdown": float(dd.min()),
                "sessions": len(dr),
                "evaluation_start": dates[0].isoformat(),
                "evaluation_end": dates[-1].isoformat(),
                "turnover": float(ledger.turnover.sum())
                + terminal["terminal_turnover_notional"] / terminal["marked_equity"],
                "filled_orders": len(orders) + terminal["terminal_filled_orders"],
                "strategy_target_changes": int(decisions.target_change.sum()),
                "signal_crosses": int(
                    decisions.decision_target.diff().fillna(0).abs().gt(1e-12).sum()
                ),
                "terminal_filled_orders": terminal["terminal_filled_orders"],
                "tax_funding_orders": int(orders.reason.eq("tax_funding").sum()),
                "continuing_target_buy_orders": len(continuing_buys),
                "dividend_reinvestment_orders": int(
                    continuing_buys.timestamp.isin(distribution_dates).sum()
                ),
                "mean_effective_exposure": float(ledger.holdings.effective_exposure.mean()),
                "mean_actual_weight": float(ledger.holdings.actual_weight.mean()),
                "mean_cash_weight": float((ledger.cash / ledger.equity).mean()),
                "total_trading_fees": float(orders.fee.sum()) + terminal["exit_fees"],
                "total_slippage": float(orders.slippage.sum()) + terminal["exit_slippage"],
                "total_taxes_paid": float(ledger.taxes_paid.sum())
                + terminal["terminal_taxes_paid"],
                "first_entry_factor": entry_factor,
                "terminal_factor": terminal_factor,
                "ma_sessions": p.get("ma_sessions"),
                "allocation": p["allocation"],
                "asset": p["asset"],
            }
        )
        paths[layer], snapshots[layer], daily[layer] = ledger, terminal, dr
    return SplitEvaluation(
        pd.DataFrame(rows),
        pd.DataFrame(daily),
        paths,
        snapshots,
        targets,
        {
            **panel.metadata,
            "candidate": p,
            "costs": asdict(costs),
            "evaluation_label": "retrospective_chronological_split_not_untouched_market_history",
            "segment_state": "flat_cash_reset_no_inherited_positions_pending_orders_or_warmup_PnL",
            "signal": "current_completed_QQQ_TR_close_above_trailing_MA_strictly_gt_exit_lte",
            "execution": "first_segment_close_decision_next_session_close_fill_no_same_close_return",
            "allocation": "synthetic_fractional_targets_transition_only_drift_not_daily_portfolio_rebalancing_or_exposure_cap",
            "QQQ_dividend_reinvestment": "physical_full_0_or_1_controls_reinvest_ex_date_cash_at_same_accounting_close_with_costs_when_target_invested",
            "synthetic": "fresh_3x_daily_reset_QQQ_total_return_minus_2x_borrowing_minus_expense_tracking_once",
            "cost_comparison": "same_intended_target_allocations_not_fixed_share_orders",
            "terminal": "full_liquidation_and_tax_clone_included_once_in_daily_returns_continuous_paths_unchanged",
            "physical_QQQ_tax": "FIFO_hypothetical_ST24_LT15_distribution_ex_date_cash_credit_ordinary_scenario_not_actual_investor_tax",
            "synthetic_tax": "hypothetical_proxy_investor_realizations_not_actual_ETF_tax",
            "risk_free": "separate_causal_DTB3_ACT365_same_cash_account_tax_timing_no_zero_fill",
            "dividend_interest": "new_same_close_distribution_cash_excluded_from_preceding_interval_cash_interest",
            "annualization": "252_sessions_CAGR_and_sample_daily_vol_ddof1_pooled_cash_excess_Sharpe",
            "maximum_input_timestamp_used": data.index[-1].isoformat(),
        },
    )


def select_validation_candidate(
    panel: DailyPanel,
    financing_observations: pd.DataFrame,
    cash_observations: pd.DataFrame,
    candidates: dict[str, dict] | None = None,
    *,
    start: Any,
    end: Any,
    costs: SplitCosts = DEFAULT_COSTS,
    tie_tolerance: float = 1e-10,
) -> tuple[str, pd.DataFrame]:
    """Only validation expense-inclusive pretax pooled daily excess Sharpe ranks.

    No DD/positive-score hurdle or fallback selected after seeing results. If
    every score is undefined, selection fails rather than inventing a winner.
    """
    candidates = default_candidates() if candidates is None else candidates
    if set(candidates) != set(default_candidates()) or candidates != default_candidates():
        raise ValueError("Exactly the frozen nine MA/allocation candidates are required")
    if tie_tolerance != 1e-10:
        raise ValueError("Frozen tie tolerance is 1e-10")
    rows = []
    for candidate_id, spec in candidates.items():
        result = evaluate_daily_segment(
            panel,
            financing_observations,
            cash_observations,
            spec,
            start=start,
            end=end,
            costs=costs,
            layers=("expense",),
        )
        row = result.metrics.iloc[0].to_dict()
        row.update(
            candidate_id=candidate_id,
            selection_cutoff=_day_clock(end).isoformat(),
            selection_basis="validation_only_expense_inclusive_pretax_pooled_daily_cash_excess_Sharpe",
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    finite = table.loc[np.isfinite(table.excess_return_Sharpe)]
    if finite.empty:
        raise ValueError("No finite validation excess Sharpe; selection cannot invent fallback")
    best = float(finite.excess_return_Sharpe.max())
    tied = finite.loc[best - finite.excess_return_Sharpe <= tie_tolerance]
    winner = (
        tied.sort_values(
            ["allocation", "ma_sessions", "candidate_id"], ascending=[True, False, True]
        )
        .iloc[0]
        .candidate_id
    )
    table["selected"] = table.candidate_id.eq(winner)
    table["selected_candidate"] = winner
    return str(winner), table


def bootstrap_components(
    result: SplitEvaluation, layer: str = "expense"
) -> tuple[pd.Series, float, dict]:
    """Strip first-entry/terminal costs, to reapply once per conditional draw.

    Ongoing turnover and cash carry remain in observed returns. This is NOT a
    re-executed policy bootstrap. Identity samples recover exact terminal wealth.
    """
    ledger = result.paths[layer]
    terminal = result.terminal_snapshots[layer]
    returns = ledger.returns.copy()  # already excludes cloned terminal liquidation
    entry_factor = _entry_factor(ledger)
    buys = ledger.orders.loc[ledger.orders.event.eq("buy")]
    if len(buys):
        stamp = pd.Timestamp(buys.iloc[0].timestamp)
        returns.loc[stamp] = (1 + returns.loc[stamp]) / entry_factor - 1
    terminal_factor = terminal["liquidated_equity"] / terminal["marked_equity"]
    factor = entry_factor * terminal_factor
    initial_cash = result.metadata["costs"]["initial_cash"]
    if not np.isclose(
        float((1 + returns).prod()) * factor,
        terminal["liquidated_equity"] / initial_cash,
        rtol=1e-10,
    ):
        raise ValueError("Bootstrap boundary reconciliation failed")
    return (
        returns,
        float(factor),
        {"first_entry_factor": entry_factor, "terminal_factor": terminal_factor},
    )
