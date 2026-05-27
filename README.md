# trend-following-real-data

A professional, interview-ready Python research pipeline for transparent trend-following strategies on real daily market data.

This project is **not** a live trading system. It is designed to demonstrate a clean quantitative research workflow: data ingestion, validation, adjusted-price processing, signal generation, no-lookahead backtesting, transaction costs, parameter sensitivity, out-of-sample evaluation, plots, and a short research memo.

## Why trend following?

Trend following is a simple and well-studied family of rules that can be implemented transparently and audited for common backtesting mistakes. The strategies here are intentionally basic so the research pipeline, assumptions, and validation choices remain easy to explain in interviews.

## Initial universe

The default config uses liquid ETFs:

- SPY
- UPRO
- QQQ
- TQQQ
- IWM
- URTY
- TLT
- TMF
- GLD
- EFA
- EEM
- EDC
- VNQ
- DRN

The date range defaults to `2005-01-01` through the most recent data available from Yahoo Finance via `yfinance`. Note that several leveraged ETFs began trading later than the unlevered ETFs, so the default inner alignment will start the multi-asset panel at the first date where all selected tickers have data.

## Leveraged ETF note

The default universe includes the available **long +3x** ETF candidates that closely map to the current unlevered universe: UPRO for S&P 500/SPY, TQQQ for Nasdaq-100/QQQ, URTY for Russell 2000/IWM, TMF for 20+ year Treasuries/TLT, EDC for emerging markets/EEM, and DRN for real estate/VNQ. GLD and EFA are left without a default +3x ETF counterpart because there is no clean current broad +3x ETF match in the same style.

Leveraged ETFs target daily multiples and can experience path-dependent compounding and volatility drag, so they should be analyzed separately from unlevered buy-and-hold ETFs and are not assumed to be appropriate long-term holdings. Product availability can change, so verify tickers before using them in research.

## Data source and limitations

Data is downloaded with `yfinance`, then cached locally as parquet:

- Raw downloaded files: `data/raw/{ticker}.parquet`
- Processed panels: `data/processed/*.parquet`

Important limitations:

- Yahoo Finance data can contain revisions, missing values, ticker changes, and corporate-action issues.
- ETF inception dates differ, so some assets may not have history from the requested start date.
- The default universe is composed of current ETFs, so a broader stock universe would require explicit survivorship-bias controls.
- Adjusted close is used for return calculations when available; adjusted opens are approximated using the close adjustment factor when needed later.
- No borrow costs, short constraints, margin model, tax model, or intraday execution model are implemented in v1.

## Installation

```bash
cd /Users/cosdis/Desktop/job/quant_projects/trend_following
source ~/.venvs/myenv/bin/activate
pip install -e '.[dev]'
```

## Quick start

```bash
python scripts/download_data.py --config configs/default.yaml
python scripts/run_backtest.py --config configs/default.yaml --strategy sma_trend
python scripts/run_backtest.py --config configs/default.yaml --strategy sma_crossover
python scripts/run_backtest.py --config configs/default.yaml --strategy tsmom
python scripts/run_parameter_sweep.py --config configs/default.yaml
python scripts/make_report.py --config configs/default.yaml
pytest
```

For a first smaller run using only the initial deliverable tickers:

```bash
python scripts/download_data.py --config configs/default.yaml --tickers SPY QQQ IWM TLT GLD
python scripts/run_backtest.py --config configs/default.yaml --strategy sma_trend --tickers SPY QQQ IWM TLT GLD
```

## Backtesting assumptions

The first version is deliberately simple and auditable:

- Long-only strategies; signal values are in `[0, 1]`.
- Cash earns zero return.
- Daily returns are computed from adjusted close prices.
- The default return convention is close-to-close.
- A signal computed using the close on day `t` is **not** allowed to earn day `t` or day `t+1` close-to-close returns when the execution assumption is next-close execution.
- With the default `execution_delay_days = 1`, a close-`t` signal is executed at the close of `t+1`, so P&L starts with the close-to-close return ending on `t+2`. In code this is implemented as `raw_signal.shift(execution_delay_days + 1)` before multiplying by returns.
- Transaction costs and slippage are charged in basis points on one-way turnover.
- Multi-asset strategy portfolios use equal capital sleeves by default: each asset receives `1/N` when its signal is long and cash otherwise.
- Equal-weight buy-and-hold benchmark is modeled as a daily rebalanced equal-weight universe with zero costs.

## Strategies

1. **Simple moving average trend**
   - Long if adjusted close is above its SMA.
   - Default window: 200 days.

2. **Moving average crossover**
   - Long if short SMA is above long SMA.
   - Default short/long windows: 50/200.

3. **Time-series momentum**
   - Long if past `N`-day return is positive.
   - Default lookback: 252 days.

4. **Optional volatility targeting**
   - Can scale positions to a target annualized volatility.
   - Default target: 10% annualized volatility.
   - Leverage is capped at 1.0 in v1.

## Parameter experiments

The parameter sweep evaluates in-sample and out-of-sample performance without selecting parameters based on full-sample results:

- SMA windows: `[50, 100, 150, 200, 250]`
- Time-series momentum lookbacks: `[63, 126, 189, 252]`
- Moving-average crossover short windows: `[20, 50, 100]`
- Moving-average crossover long windows: `[100, 150, 200, 250]`

Results are saved to `reports/tables/parameter_sweep.csv` and plots/tables are saved under `reports/figures` and `reports/tables`.

## Common pitfalls addressed

- **Look-ahead bias:** Signals are shifted into executable positions before returns are applied.
- **Data leakage:** Parameter sweeps report in-sample and out-of-sample metrics separately.
- **Survivorship bias:** Documented as a limitation for any current-constituent stock universe; ETF universe is fixed and explicit.
- **Transaction costs:** Costs and slippage are charged on turnover.
- **Parameter overfitting:** The sweep reports sensitivity rather than auto-selecting best full-sample parameters.
- **Adjusted prices:** Adjusted close is used for return calculations when available.
- **Data quality:** Missing values, duplicate dates, invalid prices, non-monotonic dates, and suspicious gaps are reported.

## Outputs

- Metrics: `reports/tables/*_metrics.csv`
- Daily backtest data: `reports/tables/*_daily_results.csv`
- Parameter sweep: `reports/tables/parameter_sweep.csv`
- Figures: `reports/figures/*.png`
- Research memo: `reports/summary.md`

## Example interview explanation

> This project demonstrates an end-to-end research pipeline using real market data: ingestion, validation, feature/signal construction, backtesting, transaction costs, parameter sensitivity, and out-of-sample evaluation.

## Future extensions

- Add a richer data vendor and compare vendor discrepancies.
- Add open-to-open or close-to-next-open execution once adjusted open returns are modeled carefully.
- Add risk-free cash returns and financing costs.
- Add shorting, borrow constraints, and portfolio-level volatility targeting.
- Add walk-forward parameter selection.
- Add vectorbt later for performance comparison, while keeping the transparent from-scratch backtester as the reference implementation.
