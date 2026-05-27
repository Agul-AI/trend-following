# Trend-Following Research Memo

## Dataset description

Data source: `yfinance` via `yfinance`.

Configured universe: SPY, UPRO, QQQ, TQQQ, IWM, URTY, TLT, TMF, GLD, EFA, EEM, EDC, VNQ, DRN.

Configured range: `2005-01-01` to `latest available`.

Raw data is cached under `/Users/cosdis/Desktop/job/quant_projects/trend_following/data/raw` and processed adjusted-price panels are stored under `/Users/cosdis/Desktop/job/quant_projects/trend_following/data/processed`.

## Data validation summary

| ticker | status | rows | start_date | end_date | messages |
| --- | --- | --- | --- | --- | --- |
| SPY | pass | 5382 | 2005-01-03 | 2026-05-26 | nan |
| UPRO | pass | 4255 | 2009-06-25 | 2026-05-26 | nan |
| QQQ | pass | 5382 | 2005-01-03 | 2026-05-26 | nan |
| TQQQ | pass | 4096 | 2010-02-11 | 2026-05-26 | nan |
| IWM | pass | 5382 | 2005-01-03 | 2026-05-26 | nan |
| URTY | pass | 4096 | 2010-02-11 | 2026-05-26 | nan |
| TLT | pass | 5382 | 2005-01-03 | 2026-05-26 | nan |
| TMF | pass | 4304 | 2009-04-16 | 2026-05-26 | nan |
| GLD | pass | 5382 | 2005-01-03 | 2026-05-26 | nan |
| EFA | pass | 5382 | 2005-01-03 | 2026-05-26 | nan |
| EEM | pass | 5382 | 2005-01-03 | 2026-05-26 | nan |
| EDC | pass | 4377 | 2008-12-30 | 2026-05-26 | nan |
| VNQ | pass | 5382 | 2005-01-03 | 2026-05-26 | nan |
| DRN | pass | 4241 | 2009-07-16 | 2026-05-26 | nan |

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
| sma_crossover | 2.2262 | 0.0747 | 0.1750 | 0.5010 | -0.4681 | 0.0055 | 0.6363 |
| Buy & Hold SPY | 8.2774 | 0.1469 | 0.1712 | 0.8866 | -0.3372 | 0.0002 | 1.0000 |
| Equal-Weight Buy & Hold | 12.5338 | 0.1738 | 0.2640 | 0.7402 | -0.4884 | 0.0002 | 1.0000 |
| sma_trend | 2.7940 | 0.0855 | 0.1390 | 0.6602 | -0.1884 | 0.0318 | 0.6320 |
| Buy & Hold SPY | 8.2774 | 0.1469 | 0.1712 | 0.8866 | -0.3372 | 0.0002 | 1.0000 |
| Equal-Weight Buy & Hold | 12.5338 | 0.1738 | 0.2640 | 0.7402 | -0.4884 | 0.0002 | 1.0000 |
| tsmom | 2.1510 | 0.0732 | 0.1526 | 0.5395 | -0.2497 | 0.0269 | 0.6305 |
| Buy & Hold SPY | 8.2774 | 0.1469 | 0.1712 | 0.8866 | -0.3372 | 0.0002 | 1.0000 |
| Equal-Weight Buy & Hold | 12.5338 | 0.1738 | 0.2640 | 0.7402 | -0.4884 | 0.0002 | 1.0000 |

## Parameter sensitivity

Top out-of-sample rows by strategy, if the sweep has been run:

| strategy | parameters | annualized_return | sharpe_ratio | max_drawdown |
| --- | --- | --- | --- | --- |
| sma_crossover | {"long_window": 200, "short_window": 100} | 0.1099 | 0.5957 | -0.4706 |
| sma_crossover | {"long_window": 100, "short_window": 20} | 0.0868 | 0.5802 | -0.2885 |
| sma_crossover | {"long_window": 250, "short_window": 20} | 0.0839 | 0.5518 | -0.3530 |
| sma_trend | {"window": 250} | 0.1091 | 0.7704 | -0.2099 |
| sma_trend | {"window": 200} | 0.1085 | 0.7661 | -0.1884 |
| sma_trend | {"window": 50} | 0.0976 | 0.6810 | -0.2765 |
| tsmom | {"lookback": 189} | 0.1110 | 0.7535 | -0.2544 |
| tsmom | {"lookback": 252} | 0.1068 | 0.7205 | -0.2411 |
| tsmom | {"lookback": 126} | 0.1036 | 0.7076 | -0.2332 |

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
