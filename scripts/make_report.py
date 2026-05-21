#!/usr/bin/env python
"""Generate a concise markdown research memo from saved outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.config import load_config
from trend_following.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    return parser.parse_args()


def _df_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored markdown table without tabulate."""
    if frame.empty:
        return ""
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.astype(object).values.tolist()]

    def row_line(values: list[str]) -> str:
        return "| " + " | ".join(values) + " |"

    separator = ["---"] * len(headers)
    return "\n".join([row_line(headers), row_line(separator), *[row_line(row) for row in rows]])


def _format_metric_table(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "No metrics table found yet. Run `scripts/run_backtest.py` first."
    display_cols = [
        "name",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "average_daily_turnover",
        "exposure_percentage",
    ]
    available = [col for col in display_cols if col in metrics.columns]
    table = metrics[available].copy()
    for col in available:
        if col != "name":
            table[col] = pd.to_numeric(table[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.4f}"
            )
    return _df_to_markdown(table)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics_files = sorted(config.reports.tables_dir.glob("*_metrics.csv"))
    metrics = (
        pd.concat([pd.read_csv(path) for path in metrics_files], ignore_index=True)
        if metrics_files
        else pd.DataFrame()
    )

    validation_path = config.reports.tables_dir / "data_validation.csv"
    validation_text = "No validation report found yet."
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        validation_text = _df_to_markdown(
            validation[["ticker", "status", "rows", "start_date", "end_date", "messages"]]
        )

    sweep_path = config.reports.tables_dir / "parameter_sweep.csv"
    sweep_text = "No parameter sweep found yet."
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        oos = sweep[sweep["segment"] == "out_of_sample"].copy()
        if not oos.empty:
            oos = oos.sort_values(["strategy", "sharpe_ratio"], ascending=[True, False])
            sweep_table = (
                oos.groupby("strategy")
                .head(3)[
                    ["strategy", "parameters", "annualized_return", "sharpe_ratio", "max_drawdown"]
                ]
                .copy()
            )
            for col in ["annualized_return", "sharpe_ratio", "max_drawdown"]:
                sweep_table[col] = pd.to_numeric(sweep_table[col], errors="coerce").map(
                    lambda x: "" if pd.isna(x) else f"{x:.4f}"
                )
            sweep_text = _df_to_markdown(sweep_table)

    content = f"""# Trend-Following Research Memo

## Dataset description

Data source: `{config.data.source}` via `yfinance`.

Configured universe: {", ".join(config.data.tickers)}.

Configured range: `{config.data.start_date}` to `{config.data.end_date or "latest available"}`.

Raw data is cached under `{config.data.raw_dir}` and processed adjusted-price panels are stored under `{config.data.processed_dir}`.

## Data validation summary

{validation_text}

## Strategy definitions

- **SMA trend:** long if adjusted close is above its moving average; otherwise cash.
- **SMA crossover:** long if short SMA is above long SMA; otherwise cash.
- **Time-series momentum:** long if the past lookback-day return is positive; otherwise cash.

## Assumptions

- Long-only, cash earns zero.
- Adjusted close prices are used for returns.
- Signal timing uses no-lookahead shifting. With `execution_delay_days = {config.backtest.execution_delay_days}`, a close-date signal first earns returns after the configured execution delay.
- Transaction cost = `{config.backtest.transaction_cost_bps}` bps and slippage = `{config.backtest.slippage_bps}` bps on one-way turnover.
- Multi-asset strategy uses `{config.backtest.portfolio_mode}` portfolio construction.
- Train/test split date: `{config.backtest.train_end_date}`.

## Main results

{_format_metric_table(metrics)}

## Parameter sensitivity

Top out-of-sample rows by strategy, if the sweep has been run:

{sweep_text}

## Comparison to buy-and-hold

The backtest scripts compare each strategy to buy-and-hold SPY and equal-weight buy-and-hold over the configured universe. These are simple reference benchmarks rather than investable recommendations.

## What worked

- The pipeline separates raw and processed data.
- Signals are shifted before returns are applied.
- Costs, slippage, turnover, exposure, and benchmark comparisons are explicit.
- Parameter sweeps report in-sample and out-of-sample performance separately.

## What failed or remains uncertain

- Yahoo Finance data quality can vary by asset and date.
- Close-to-close execution is a simplifying assumption.
- Fixed ETF universe does not test broader universe construction or delisting effects.
- Current v1 does not model cash yield, borrow constraints, taxes, or intraday liquidity.

## Limitations

Survivorship bias, ETF inception-date differences, adjusted-price assumptions, close-to-close execution, and no borrow/short constraints should be discussed explicitly before interpreting results.

## Next steps

1. Add open-to-open or close-to-next-open execution with carefully adjusted open prices.
2. Add walk-forward parameter selection.
3. Add portfolio-level volatility targeting and risk budgeting.
4. Add richer data vendor support and vendor reconciliation checks.
5. Add vectorbt later as a speed comparison while preserving this transparent reference backtester.
"""
    ensure_directory(config.reports.summary_path.parent)
    config.reports.summary_path.write_text(content, encoding="utf-8")
    print(f"Summary written to {config.reports.summary_path}")


if __name__ == "__main__":
    main()
