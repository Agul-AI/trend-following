# Trend-Following Research Memo

## Dataset description

Data source: `yfinance` via `yfinance`.

Configured universe: SPY, QQQ, IWM, TLT, GLD, EFA, EEM, VNQ.

Configured range: `2005-01-01` to `latest available`.

Raw data is cached under `/Users/cosdis/Desktop/job/quant_projects/trend_following/data/raw` and processed adjusted-price panels are stored under `/Users/cosdis/Desktop/job/quant_projects/trend_following/data/processed`.

## Data validation summary

| ticker | status | rows | start_date | end_date | messages |
| --- | --- | --- | --- | --- | --- |
| SPY | pass | 5375 | 2005-01-03 | 2026-05-14 | nan |
| QQQ | pass | 5375 | 2005-01-03 | 2026-05-14 | nan |
| IWM | pass | 5375 | 2005-01-03 | 2026-05-14 | nan |
| TLT | pass | 5375 | 2005-01-03 | 2026-05-14 | nan |
| GLD | pass | 5375 | 2005-01-03 | 2026-05-14 | nan |
| EFA | pass | 5375 | 2005-01-03 | 2026-05-14 | nan |
| EEM | pass | 5375 | 2005-01-03 | 2026-05-14 | nan |
| VNQ | pass | 5375 | 2005-01-03 | 2026-05-14 | nan |

## Strategy definitions

- **SMA trend:** long if adjusted close is above its moving average; otherwise cash.
- **SMA crossover:** long if short SMA is above long SMA; otherwise cash.
- **Time-series momentum:** long if the past lookback-day return is positive; otherwise cash.

## Assumptions

- Long-only, cash earns zero.
- Adjusted close prices are used for returns.
- Signal timing uses no-lookahead shifting. With `execution_delay_days = 1`, a close-date signal first earns returns after the configured execution delay.
- Transaction cost = `1.0` bps and slippage = `1.0` bps on one-way turnover.
- Multi-asset strategy uses `equal_sleeves` portfolio construction.
- Train/test split date: `2018-12-31`.

## Main results

| name | cumulative_return | annualized_return | annualized_volatility | sharpe_ratio | max_drawdown | average_daily_turnover | exposure_percentage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sma_crossover | 2.7592 | 0.0641 | 0.1025 | 0.6573 | -0.2546 | 0.0053 | 0.6718 |
| Buy & Hold SPY | 8.1931 | 0.1096 | 0.1897 | 0.6433 | -0.5519 | 0.0002 | 1.0000 |
| Equal-Weight Buy & Hold | 6.6599 | 0.1002 | 0.1604 | 0.6756 | -0.4434 | 0.0002 | 1.0000 |
| sma_trend | 2.9993 | 0.0671 | 0.0881 | 0.7818 | -0.1385 | 0.0323 | 0.6724 |
| Buy & Hold SPY | 8.1931 | 0.1096 | 0.1897 | 0.6433 | -0.5519 | 0.0002 | 1.0000 |
| Equal-Weight Buy & Hold | 6.6599 | 0.1002 | 0.1604 | 0.6756 | -0.4434 | 0.0002 | 1.0000 |
| tsmom | 2.9938 | 0.0671 | 0.0963 | 0.7222 | -0.1520 | 0.0221 | 0.6835 |
| Buy & Hold SPY | 8.1931 | 0.1096 | 0.1897 | 0.6433 | -0.5519 | 0.0002 | 1.0000 |
| Equal-Weight Buy & Hold | 6.6599 | 0.1002 | 0.1604 | 0.6756 | -0.4434 | 0.0002 | 1.0000 |

## Parameter sensitivity

Top out-of-sample rows by strategy, if the sweep has been run:

| strategy | parameters | annualized_return | sharpe_ratio | max_drawdown |
| --- | --- | --- | --- | --- |
| sma_crossover | {"long_window": 100, "short_window": 20} | 0.0784 | 0.8616 | -0.1523 |
| sma_crossover | {"long_window": 200, "short_window": 100} | 0.0958 | 0.8345 | -0.2569 |
| sma_crossover | {"long_window": 250, "short_window": 100} | 0.0982 | 0.8344 | -0.2569 |
| sma_trend | {"window": 150} | 0.0864 | 0.9877 | -0.1157 |
| sma_trend | {"window": 100} | 0.0819 | 0.9611 | -0.1165 |
| sma_trend | {"window": 200} | 0.0817 | 0.9322 | -0.1385 |
| tsmom | {"lookback": 252} | 0.0943 | 1.0050 | -0.1520 |
| tsmom | {"lookback": 126} | 0.0861 | 0.9645 | -0.1286 |
| tsmom | {"lookback": 189} | 0.0845 | 0.9254 | -0.1372 |

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
