#!/usr/bin/env python
"""Apply the current preferred hourly-200MA/MACD strategy to synthetic +3x S&P 500.

This is the S&P 500 analogue of the preferred QQQ/synthetic-TQQQ rule:
- Signal source: SPY hourly close.
- Target exposure: synthetic SPY_3X_CALC.
- Entry: SPY hourly MACD histogram > 0 and SPY hourly close > SPY hourly 200-day MA.
- Exit: SPY hourly close < SPY hourly 200-day MA.
- No daily regime gate, no profit lock, max one trade per day via executable_weights.
- Strategy rows include costs, slippage, short-term tax approximation, and 3% cash yield.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tqqq_cash_yield_candidate_comparison import (
    simulate_after_tax_portfolio_with_cash_yield,  # noqa: E402
)
from run_tqqq_daily_gate_ablation import no_daily_gate_hourly_ma_gate_signal  # noqa: E402
from run_tqqq_entry_signal_comparison import _returns_from_prices, executable_weights  # noqa: E402
from run_tqqq_position_risk_sizing_experiments import drawdown_episode_count  # noqa: E402
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.metrics import calculate_metrics  # noqa: E402
from trend_following.utils import ensure_directory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--raw-dir", default="data/raw/alpha_vantage_60min")
    parser.add_argument("--synthetic-raw-dir", default="data/raw/synthetic_3x_60min")
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-term-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-interest-tax-rate", type=float, default=0.24)
    parser.add_argument("--cash-annual-yield", type=float, default=0.03)
    parser.add_argument("--bars-per-day", type=int, default=6)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--entry-confirm-bars", type=int, default=2)
    parser.add_argument("--exit-confirm-bars", type=int, default=3)
    parser.add_argument("--exit-ma-days", type=float, default=200.0)
    parser.add_argument("--output-prefix", default="spy_3x_preferred_strategy")
    return parser.parse_args()


def load_close(path: Path, name: str) -> pd.Series:
    return read_price_file(path).sort_index()["adj_close"].astype(float).rename(name)


def add_dd_counts(metrics: dict[str, Any], returns: pd.Series) -> None:
    counts = {
        threshold: drawdown_episode_count(returns, threshold=-(threshold / 100.0))
        for threshold in (20, 30, 40, 50)
    }
    metrics["dd_20_30_40_50"] = f"{counts[20]}/{counts[30]}/{counts[40]}/{counts[50]}"


def strategy_returns(
    *,
    config: Any,
    signal_price: pd.Series,
    target_price: pd.Series,
    target_ticker: str,
    common: pd.DatetimeIndex,
    args: argparse.Namespace,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    raw, _ = no_daily_gate_hourly_ma_gate_signal(
        entry_price=signal_price,
        exit_price=signal_price,
        output_index=common,
        bars_per_day=args.bars_per_day,
        average_type=args.average_type,
        macd_unit=args.macd_unit,
        entry_confirm_bars=args.entry_confirm_bars,
        exit_confirm_bars=args.exit_confirm_bars,
        exit_ma_days=args.exit_ma_days,
    )
    raw_weights = raw.rename(target_ticker).to_frame(target_ticker)
    weights = executable_weights(raw_weights, config=config).reindex(common).fillna(0.0)
    returns = _returns_from_prices(target_price.loc[common].to_frame(target_ticker))
    after_tax, _, turnover, _, _ = simulate_after_tax_portfolio_with_cash_yield(
        returns,
        weights,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        tax_rate=args.short_term_tax_rate,
        cash_annual_yield=args.cash_annual_yield,
        annualization=config.backtest.annualization,
        cash_interest_tax_rate=args.cash_interest_tax_rate,
    )
    return after_tax, weights[target_ticker], turnover


def row_for_returns(
    *,
    name: str,
    kind: str,
    returns: pd.Series,
    config: Any,
    turnover: pd.Series | None = None,
    weights: pd.Series | None = None,
) -> dict[str, Any]:
    metrics = calculate_metrics(
        returns,
        turnover=turnover,
        weights=weights,
        annualization=config.backtest.annualization,
    )
    add_dd_counts(metrics, returns)
    return {
        "name": name,
        "kind": kind,
        "final_return": metrics["cumulative_return"],
        "annualized_return": metrics["annualized_return"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "number_of_trades": metrics["number_of_trades"],
        "exposure_percentage": metrics["exposure_percentage"],
        "dd_20_30_40_50": metrics["dd_20_30_40_50"],
        "observations": metrics["observations"],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    raw_dir = Path(args.raw_dir)
    synth_dir = Path(args.synthetic_raw_dir)

    spy = load_close(raw_dir / "SPY.parquet", "SPY")
    qqq = load_close(raw_dir / "QQQ.parquet", "QQQ")
    spy_3x = load_close(synth_dir / "SPY_3X_CALC.parquet", "SPY_3X_CALC")
    qqq_3x = load_close(synth_dir / "QQQ_3X_CALC.parquet", "QQQ_3X_CALC")

    common = spy.index.intersection(qqq.index).intersection(spy_3x.index).intersection(qqq_3x.index)
    common = common.sort_values()

    spy_strategy, spy_weights, spy_turnover = strategy_returns(
        config=config,
        signal_price=spy,
        target_price=spy_3x,
        target_ticker="SPY_3X_CALC",
        common=common,
        args=args,
    )
    qqq_strategy, qqq_weights, qqq_turnover = strategy_returns(
        config=config,
        signal_price=qqq,
        target_price=qqq_3x,
        target_ticker="QQQ_3X_CALC",
        common=common,
        args=args,
    )

    benchmark_returns = {
        "SPY_3X_CALC_BH": _returns_from_prices(spy_3x.loc[common].to_frame("SPY_3X_CALC"))["SPY_3X_CALC"],
        "SPY_BH": _returns_from_prices(spy.loc[common].to_frame("SPY"))["SPY"],
        "QQQ_BH": _returns_from_prices(qqq.loc[common].to_frame("QQQ"))["QQQ"],
    }

    rows = [
        row_for_returns(
            name="SPY_3X preferred strategy",
            kind="strategy_after_tax_cash_yield",
            returns=spy_strategy,
            config=config,
            turnover=spy_turnover,
            weights=spy_weights,
        ),
        row_for_returns(
            name="QQQ_3X preferred strategy reference",
            kind="strategy_after_tax_cash_yield",
            returns=qqq_strategy,
            config=config,
            turnover=qqq_turnover,
            weights=qqq_weights,
        ),
    ]
    for name, returns in benchmark_returns.items():
        rows.append(
            row_for_returns(
                name=name,
                kind="buy_hold_raw",
                returns=returns,
                config=config,
                turnover=None,
                weights=pd.Series(1.0, index=returns.index),
            )
        )

    table = pd.DataFrame(rows)
    table.insert(0, "period_start", common[0])
    table.insert(1, "period_end", common[-1])
    table = table.sort_values("annualized_return", ascending=False)

    returns_out = pd.DataFrame(
        {
            "SPY_3X_preferred_strategy": spy_strategy,
            "QQQ_3X_preferred_strategy_reference": qqq_strategy,
            **benchmark_returns,
        }
    )
    weights_out = pd.DataFrame(
        {
            "SPY_3X_preferred_strategy": spy_weights,
            "QQQ_3X_preferred_strategy_reference": qqq_weights,
        }
    )

    tables_dir = Path("reports/tables")
    ensure_directory(tables_dir)
    table_path = tables_dir / f"{args.output_prefix}_comparison.csv"
    returns_path = tables_dir / f"{args.output_prefix}_returns.csv"
    weights_path = tables_dir / f"{args.output_prefix}_weights.csv"
    table.to_csv(table_path, index=False)
    returns_out.to_csv(returns_path, index_label="date")
    weights_out.to_csv(weights_path, index_label="date")

    print(f"Saved comparison to {table_path}")
    print(f"Saved returns to {returns_path}")
    print(f"Saved weights to {weights_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
