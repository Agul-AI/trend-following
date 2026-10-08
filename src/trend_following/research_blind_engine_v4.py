"""Signal-agnostic daily research kernel for isolated, staged data packages.

No strategy family, optimized parameter or old market-result input lives here.
A policy receives a copy of current/past observations ONLY. Today’s decision
fills next session's close and cannot earn the return ending at that fill.
Standalone sandbox bundles contain this file and two mechanical accounting/rate
kernels; they do not contain other project modules or previous result files.
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from trend_following.research_ledger import LedgerConfig, PortfolioLedger, simulate_ledger
    from trend_following.research_rates import rate_accrual
else:
    from ledger_kernel import LedgerConfig, PortfolioLedger, simulate_ledger
    from rate_kernel import rate_accrual


@dataclass(frozen=True)
class Costs:
    initial_cash: float = 1000.0
    fee_bps: float = 1.0
    slippage_bps: float = 5.0
    financing_spread_bps: float = 50.0
    expense_ratio: float = 0.0095
    extra_drag: float = 0.005
    short_tax_rate: float = 0.24
    long_tax_rate: float = 0.15
    interest_tax_rate: float = 0.24
    wash_sales: bool = True

    def __post_init__(self):
        numbers = [value for key, value in asdict(self).items() if key != "wash_sales"]
        if (
            not np.isfinite(numbers).all()
            or self.initial_cash <= 0
            or self.expense_ratio < 0
            or self.extra_drag < 0
        ):
            raise ValueError("Invalid fixed accounting assumptions")
        LedgerConfig(
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
            short_tax_rate=self.short_tax_rate,
            long_tax_rate=self.long_tax_rate,
            interest_tax_rate=self.interest_tax_rate,
            wash_sales=self.wash_sales,
        )


DEFAULT_COSTS = Costs()


@dataclass
class Market:
    data: pd.DataFrame
    borrowing: pd.Series
    cash_returns: pd.Series


@dataclass
class Result:
    metrics: pd.DataFrame
    daily_returns: pd.DataFrame
    targets: pd.Series
    paths: dict
    terminals: dict


class DistributionLedger(PortfolioLedger):
    """Do not pay prior-interval interest on same-close dividend cash."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_distribution = 0.0

    def credit_distribution(self, *args, **kwargs):
        received = super().credit_distribution(*args, **kwargs)
        self.new_distribution += received
        return received

    def accrue_cash(self, period_return, timestamp):
        distribution = self.new_distribution
        self.cash -= distribution
        try:
            return super().accrue_cash(period_return, timestamp)
        finally:
            self.cash += distribution
            self.new_distribution = 0.0


def daily_clock(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    return (stamp.normalize() + pd.Timedelta(hours=16)).tz_localize("America/New_York")


def load_market(input_dir: str | Path, costs: Costs = DEFAULT_COSTS) -> Market:
    """Read only a caller-provided, physically stage-limited raw packet."""
    directory = Path(input_dir)
    raw = pd.read_csv(directory / "QQQ.csv", float_precision="round_trip")
    if not {"session", "close", "dividend_amount", "split_coefficient"}.issubset(raw):
        raise ValueError("Raw daily sessions, closes and actions required")
    index = pd.DatetimeIndex([daily_clock(x) for x in raw.session], name="timestamp")
    if index.has_duplicates or not index.is_monotonic_increasing or (index.dayofweek >= 5).any():
        raise ValueError("Unique ordered weekday observations required")
    data = pd.DataFrame(
        {
            "close": raw.close.to_numpy(float),
            "dividend": raw.dividend_amount.to_numpy(float),
            "split": raw.split_coefficient.to_numpy(float),
        },
        index=index,
    )
    if (
        not np.isfinite(data.to_numpy()).all()
        or data.close.le(0).any()
        or data.dividend.lt(0).any()
        or data.split.le(0).any()
    ):
        raise ValueError("Invalid raw prices or actions")
    data["total_return"] = data.split * (data.close + data.dividend) / data.close.shift(1) - 1
    data.iloc[0, data.columns.get_loc("total_return")] = 0.0
    if data.total_return.le(-1).any():
        raise ValueError("Underlying total-return wealth exhausted")
    data["total_return_index"] = 100 * (1 + data.total_return).cumprod()
    data["price_return"] = data.split * data.close / data.close.shift(1) - 1
    data.iloc[0, data.columns.get_loc("price_return")] = 0.0
    rate_values = {}
    for name, denominator, spread in (
        ("DFF", 360.0, costs.financing_spread_bps),
        ("DTB3", 365.0, 0.0),
    ):
        observations = pd.read_csv(
            directory / f"{name}_available.csv", float_precision="round_trip"
        )
        observations["available_at"] = pd.to_datetime(observations.available_at, utc=True)
        if "observation_date" in observations:
            observations["observation_date"] = pd.to_datetime(observations.observation_date)
        if "series_id" in observations and not observations.series_id.eq(name).all():
            raise ValueError("Incorrect rate source")
        rate_values[name] = rate_accrual(
            index, observations, spread_bps=spread, day_count=denominator, max_age_days=10.0
        )
    return Market(data, rate_values["DFF"], rate_values["DTB3"])


def build_targets(
    market: Market,
    target: Callable,
    parameters: dict,
    history_sessions: int,
    *,
    start: str,
    end: str,
) -> pd.Series:
    """Every callback gets an independent past-only window and parameter copy."""
    if (
        isinstance(history_sessions, bool)
        or not isinstance(history_sessions, int)
        or not 1 <= history_sessions <= 1260
    ):
        raise ValueError("Declare a bounded integer history window before evaluation")
    lower, upper = daily_clock(start), daily_clock(end)
    data = market.data.loc[market.data.index <= upper]
    positions = np.flatnonzero(data.index >= lower)
    if len(positions) < 2:
        raise ValueError("At least two scored sessions required")
    decisions = []
    for i in positions:
        history = data.iloc[max(0, i - history_sessions + 1) : i + 1].copy(deep=True)
        value = target(history, copy.deepcopy(parameters))
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isscalar(value)
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Policy must return a finite target allocation in [0,1]")
        decisions.append(float(value))
    # No inherited pending order at a segment boundary.
    return pd.Series(np.r_[0.0, decisions[:-1]], index=data.index[positions], name="target")


def _terminal(ledger, stamp: pd.Timestamp, taxable: bool) -> dict:
    clone = copy.deepcopy(ledger)
    before = float(clone.equity)
    offset = len(clone.events)
    clone.liquidate(stamp)
    if taxable:
        clone.expire_wash_losses(stamp, no_future_replacements=True)
        clone.settle_taxes(stamp, close_year=stamp.year)
    events = clone.events[offset:]
    orders = [x for x in events if x["event"] in {"buy", "sell"}]
    return dict(
        marked_equity=before,
        liquidated_equity=float(clone.equity),
        fees=sum(x["fee"] for x in orders),
        slippage=sum(x["slippage"] for x in orders),
        taxes=-sum(x["cash_flow"] for x in events if x["event"] in {"tax_payment", "tax_refund"}),
        orders=len(orders),
        turnover=sum(x["turnover_notional"] for x in orders),
    )


def _returns_with_terminal(path, terminal):
    returns = path.returns.copy()
    returns.iloc[-1] = (1 + returns.iloc[-1]) * terminal["liquidated_equity"] / terminal[
        "marked_equity"
    ] - 1
    return returns


def evaluate_targets(
    market: Market,
    targets: pd.Series,
    *,
    candidate_id: str,
    asset: str = "synthetic3x",
    costs: Costs = DEFAULT_COSTS,
    layers: tuple[str, ...] = ("expense",),
) -> Result:
    """Only price/cash accounting, never signal selection or parameter fitting."""
    if (
        asset not in ("synthetic3x", "QQQ")
        or not layers
        or len(set(layers)) != len(layers)
        or any(layer not in ("gross", "trading", "financing", "expense", "tax") for layer in layers)
    ):
        raise ValueError("Unsupported asset/layer")
    dates = targets.index
    if (
        not isinstance(dates, pd.DatetimeIndex)
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
        or not dates.isin(market.data.index).all()
        or not np.isfinite(targets).all()
        or targets.lt(0).any()
        or targets.gt(1).any()
        or targets.iloc[0] != 0
    ):
        raise ValueError("Aligned targets must begin flat and be finite allocations")
    data = market.data.loc[market.data.index <= dates[-1]]
    cash = market.cash_returns.loc[dates].copy()
    cash.iloc[0] = 0.0
    elapsed = data.index.to_series().diff().dt.total_seconds().fillna(0.0) / (365 * 86400)
    rows, daily, paths, terminals = [], {}, {}, {}
    for layer in layers:
        config = LedgerConfig(
            fee_bps=0.0 if layer == "gross" else costs.fee_bps,
            slippage_bps=0.0 if layer == "gross" else costs.slippage_bps,
            short_tax_rate=costs.short_tax_rate if layer == "tax" else 0.0,
            long_tax_rate=costs.long_tax_rate if layer == "tax" else 0.0,
            interest_tax_rate=costs.interest_tax_rate if layer == "tax" else 0.0,
            wash_sales=costs.wash_sales if layer == "tax" else False,
            rebalance_policy="every_bar" if asset == "QQQ" else "on_target_change",
            exposure_multipliers={asset: 1.0 if asset == "QQQ" else 3.0},
        )
        actions = None
        if asset == "synthetic3x":
            returns = 3 * data.total_return
            if layer in ("financing", "expense", "tax"):
                returns = returns - 2 * market.borrowing.loc[data.index]
            if layer in ("expense", "tax"):
                returns = returns - (costs.expense_ratio + costs.extra_drag) * elapsed
            if returns.le(-1).any():
                raise ValueError("Nonpositive synthetic NAV; bankruptcy is not clipped")
            prices = 100 * (1 + returns).cumprod()
        else:
            returns, prices = data.price_return, data.close
            source = data.loc[dates].loc[lambda x: x.dividend.ne(0) | x.split.ne(1)]
            actions = pd.DataFrame(
                dict(
                    timestamp=source.index,
                    asset=asset,
                    split_ratio=source.split.to_numpy(),
                    distribution_per_share=source.dividend.to_numpy(),
                    tax_kind="ordinary",
                )
            )
        before = prices.loc[prices.index < dates[0]]
        initial_price = float(before.iloc[-1]) if len(before) else float(prices.iloc[0])
        account = DistributionLedger(
            config, initial_cash=costs.initial_cash, initial_prices={asset: initial_price}
        )
        path = simulate_ledger(
            returns.loc[dates].to_frame(asset),
            targets.to_frame(asset),
            cash,
            config,
            ledger=account,
            targets_at="close",
            liquidate_at_end=False,
            finalize_tax_at_end=False,
            corporate_actions=actions,
            price_basis="unadjusted" if asset == "QQQ" else "total_return",
        )
        terminal = _terminal(path.ledger, dates[-1], layer == "tax")
        dr = _returns_with_terminal(path, terminal)
        rf_config = LedgerConfig(
            interest_tax_rate=costs.interest_tax_rate if layer == "tax" else 0.0
        )
        zeros = pd.DataFrame({"cash": 0.0}, index=dates)
        reference = simulate_ledger(
            zeros,
            zeros,
            cash,
            rf_config,
            initial_cash=costs.initial_cash,
            targets_at="close",
            liquidate_at_end=False,
            finalize_tax_at_end=False,
        )
        rf_terminal = _terminal(reference.ledger, dates[-1], layer == "tax")
        rf = _returns_with_terminal(reference, rf_terminal)
        wealth = float(terminal["liquidated_equity"]) / costs.initial_cash
        if not np.isclose(float((1 + dr).prod()), wealth, rtol=1e-10):
            raise ValueError("Terminal return/equity reconciliation failed")
        curve = (1 + dr).cumprod()
        drawdown = curve / curve.cummax().clip(lower=1.0) - 1
        excess = dr - rf
        excess_vol, vol = float(excess.std(ddof=1)), float(dr.std(ddof=1))
        orders = path.orders
        exposure = path.holdings.groupby("timestamp").effective_exposure.sum().reindex(dates)
        rows.append(
            dict(
                candidate_id=candidate_id,
                layer=layer,
                asset=asset,
                sessions=len(dates),
                CAGR=wealth ** (252 / len(dates)) - 1,
                total_return_pct=(wealth - 1) * 100,
                terminal_equity=terminal["liquidated_equity"],
                max_drawdown=float(drawdown.min()),
                annualized_volatility=vol * np.sqrt(252),
                cash_excess_Sharpe=float(excess.mean() / excess_vol * np.sqrt(252))
                if excess_vol > 1e-12
                else np.nan,
                zero_RF_Sharpe=float(dr.mean() / vol * np.sqrt(252)) if vol > 1e-12 else np.nan,
                target_changes=int(targets.diff().fillna(0).abs().gt(1e-12).sum()),
                filled_orders=len(orders) + terminal["orders"],
                fees=float(orders.fee.sum()) + terminal["fees"],
                slippage=float(orders.slippage.sum()) + terminal["slippage"],
                taxes=float(path.taxes_paid.sum()) + terminal["taxes"],
                mean_effective_exposure=float(exposure.mean()),
                max_effective_exposure=float(exposure.max()),
                mean_cash_weight=float((path.cash / path.equity).mean()),
                first_session=dates[0].isoformat(),
                last_session=dates[-1].isoformat(),
            )
        )
        daily[layer], paths[layer], terminals[layer] = dr, path, terminal
    return Result(pd.DataFrame(rows), pd.DataFrame(daily), targets, paths, terminals)


def run_candidate(
    input_dir,
    target: Callable,
    parameters: dict,
    history_sessions: int,
    *,
    candidate_id: str,
    start: str,
    end: str,
    costs: Costs = DEFAULT_COSTS,
    layers: tuple[str, ...] = ("expense",),
) -> Result:
    market = load_market(input_dir, costs)
    targets = build_targets(market, target, parameters, history_sessions, start=start, end=end)
    return evaluate_targets(market, targets, candidate_id=candidate_id, costs=costs, layers=layers)


def load_policy(path: Path) -> Callable:
    """Reviewable pure-policy surface: no imports that expose files/network/clock."""
    tree = ast.parse(path.read_text())
    forbidden = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "breakpoint",
        "input",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name not in ("numpy", "pandas", "math") for alias in node.names):
                raise ValueError("Policy imports are restricted to numpy, pandas and math")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ("numpy", "pandas", "math"):
                raise ValueError("Policy import is outside the pure numerical interface")
        elif isinstance(node, ast.Name) and node.id in forbidden:
            raise ValueError("Policy may not perform I/O, inspect globals or execute code")
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("__")
            or node.attr
            in {
                "read_csv",
                "read_parquet",
                "read_pickle",
                "load",
                "loadtxt",
                "genfromtxt",
                "save",
                "savetxt",
                "to_csv",
                "to_parquet",
                "to_pickle",
            }
        ):
            raise ValueError("Policy may not perform external I/O or introspection")
    spec = importlib.util.spec_from_file_location("frozen_blind_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "target", None)):
        raise ValueError("Policy must expose target(history, parameters)")
    return module.target


def export_result(directory: Path, result: Result):
    directory.mkdir(parents=True, exist_ok=False)
    result.metrics.to_csv(directory / "metrics.csv", index=False)
    result.daily_returns.to_csv(directory / "daily_returns.csv")
    result.targets.to_csv(directory / "targets.csv")
    (directory / "terminal.json").write_text(json.dumps(result.terminals, indent=2))
    for layer, path in result.paths.items():
        pd.DataFrame(
            dict(
                equity=path.equity,
                cash=path.cash,
                returns=path.returns,
                taxes=path.taxes_paid,
                turnover=path.turnover,
            )
        ).to_csv(directory / f"{layer}_account.csv")
        path.events.to_csv(directory / f"{layer}_events.csv", index=False)
        path.holdings.to_parquet(directory / f"{layer}_holdings.parquet", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    policy = load_policy(Path(request["policy_path"]))
    market = load_market(request["input_dir"], Costs(**request.get("costs", {})))
    directory = Path(request["output"])
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for candidate in request["candidates"]:
        targets = build_targets(
            market,
            policy,
            candidate["parameters"],
            candidate["history_sessions"],
            start=request["start"],
            end=request["end"],
        )
        result = evaluate_targets(
            market,
            targets,
            candidate_id=candidate["candidate_id"],
            costs=Costs(**request.get("costs", {})),
            layers=tuple(request["layers"]),
        )
        export_result(directory / candidate["candidate_id"], result)
        rows.append(result.metrics)
    if request.get("benchmarks", False):
        dates = targets.index
        for name, asset, weight in (
            ("QQQ_buy_hold", "QQQ", 1.0),
            ("cash", "QQQ", 0.0),
            ("synthetic3x_buy_hold", "synthetic3x", 1.0),
        ):
            values = pd.Series(weight, index=dates)
            values.iloc[0] = 0.0
            result = evaluate_targets(
                market,
                values,
                candidate_id=name,
                asset=asset,
                costs=Costs(**request.get("costs", {})),
                layers=tuple(request["layers"]),
            )
            export_result(directory / name, result)
            rows.append(result.metrics)
    pd.concat(rows, ignore_index=True).to_csv(directory / "all_metrics.csv", index=False)
    print(
        json.dumps(
            {
                "status": "completed",
                "candidate_count": len(request["candidates"]),
                "output": str(directory),
            }
        )
    )


if __name__ == "__main__":
    main()
