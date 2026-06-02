# Trend-Following Research Memo

## Dataset description

Data source: `yfinance` via `yfinance`.

Configured universe: SPY, UPRO, QQQ, TQQQ, IWM, URTY, TLT, TMF, GLD, EFA, EEM, EDC, VNQ, DRN.

Configured range: `2017-01-01` to `latest available`.

Raw data is cached under `/Users/cosdis/Desktop/job/quant_projects/trend_following/data/raw` and processed adjusted-price panels are stored under `/Users/cosdis/Desktop/job/quant_projects/trend_following/data/processed`.

## Data validation summary

| ticker | status | rows | start_date | end_date | messages |
| --- | --- | --- | --- | --- | --- |
| SPY | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |
| UPRO | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |
| QQQ | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |
| TQQQ | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |
| IWM | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |
| URTY | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |
| TLT | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |
| TMF | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |
| GLD | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |
| EFA | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |
| EEM | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |
| EDC | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |
| VNQ | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |
| DRN | pass | 2361 | 2017-01-03 | 2026-05-26 | nan |

## Strategy definitions

- **SMA trend:** long if adjusted close is above its moving average; otherwise cash.
- **SMA crossover:** long if short SMA is above long SMA; otherwise cash.
- **Time-series momentum:** long if the past lookback-day return is positive; otherwise cash.
- **Donchian breakout:** long after an entry-window high breakout until an exit-window low is hit.
- **Regression slope:** long when the rolling log-price regression slope is positive.
- **Kalman trend:** long when a local-linear Kalman filter estimates a positive latent trend slope.
- **Cross-sectional momentum:** long the strongest assets by trailing return, optionally requiring positive absolute momentum.

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
| cross_sectional_momentum | 2.8095 | 0.1534 | 0.3548 | 0.5813 | -0.6000 | 0.1502 | 0.9030 |
| Buy & Hold SPY | 2.8597 | 0.1551 | 0.1830 | 0.8799 | -0.3372 | 0.0004 | 1.0000 |
| Equal-Weight Buy & Hold | 2.9751 | 0.1587 | 0.2779 | 0.6704 | -0.4884 | 0.0004 | 1.0000 |
| donchian_breakout | 0.5554 | 0.0483 | 0.1294 | 0.4295 | -0.2109 | 0.0040 | 0.4296 |
| Buy & Hold SPY | 2.8597 | 0.1551 | 0.1830 | 0.8799 | -0.3372 | 0.0004 | 1.0000 |
| Equal-Weight Buy & Hold | 2.9751 | 0.1587 | 0.2779 | 0.6704 | -0.4884 | 0.0004 | 1.0000 |
| kalman_trend | 1.2401 | 0.0899 | 0.1509 | 0.6463 | -0.2597 | 0.0294 | 0.6098 |
| Buy & Hold SPY | 2.8597 | 0.1551 | 0.1830 | 0.8799 | -0.3372 | 0.0004 | 1.0000 |
| Equal-Weight Buy & Hold | 2.9751 | 0.1587 | 0.2779 | 0.6704 | -0.4884 | 0.0004 | 1.0000 |
| regression_slope | 0.8855 | 0.0700 | 0.1817 | 0.4652 | -0.4127 | 0.0069 | 0.6043 |
| Buy & Hold SPY | 2.8597 | 0.1551 | 0.1830 | 0.8799 | -0.3372 | 0.0004 | 1.0000 |
| Equal-Weight Buy & Hold | 2.9751 | 0.1587 | 0.2779 | 0.6704 | -0.4884 | 0.0004 | 1.0000 |
| sma_crossover | 0.6156 | 0.0525 | 0.1884 | 0.3685 | -0.4681 | 0.0054 | 0.5895 |
| Buy & Hold SPY | 2.8597 | 0.1551 | 0.1830 | 0.8799 | -0.3372 | 0.0004 | 1.0000 |
| Equal-Weight Buy & Hold | 2.9751 | 0.1587 | 0.2779 | 0.6704 | -0.4884 | 0.0004 | 1.0000 |
| sma_trend | 1.1560 | 0.0855 | 0.1422 | 0.6481 | -0.1884 | 0.0317 | 0.5832 |
| Buy & Hold SPY | 2.8597 | 0.1551 | 0.1830 | 0.8799 | -0.3372 | 0.0004 | 1.0000 |
| Equal-Weight Buy & Hold | 2.9751 | 0.1587 | 0.2779 | 0.6704 | -0.4884 | 0.0004 | 1.0000 |
| tsmom | 0.8353 | 0.0670 | 0.1512 | 0.5046 | -0.2411 | 0.0234 | 0.5570 |
| Buy & Hold SPY | 2.8597 | 0.1551 | 0.1830 | 0.8799 | -0.3372 | 0.0004 | 1.0000 |
| Equal-Weight Buy & Hold | 2.9751 | 0.1587 | 0.2779 | 0.6704 | -0.4884 | 0.0004 | 1.0000 |

## Parameter sensitivity

Top out-of-sample rows by strategy, if the sweep has been run:

| strategy | parameters | annualized_return | sharpe_ratio | max_drawdown |
| --- | --- | --- | --- | --- |
| cross_sectional_momentum | {"lookback": 252, "portfolio_mode": "active_equal", "require_positive": true, "top_n": 5} | 0.2082 | 0.8005 | -0.4709 |
| cross_sectional_momentum | {"lookback": 252, "portfolio_mode": "active_equal", "require_positive": true, "top_n": 2} | 0.2710 | 0.7959 | -0.5221 |
| cross_sectional_momentum | {"lookback": 252, "portfolio_mode": "active_equal", "require_positive": true, "top_n": 3} | 0.2273 | 0.7685 | -0.4965 |
| donchian_breakout | {"entry_lookback": 126, "exit_lookback": 126} | 0.1001 | 0.7014 | -0.2109 |
| donchian_breakout | {"entry_lookback": 252, "exit_lookback": 126} | 0.0709 | 0.5820 | -0.2109 |
| donchian_breakout | {"entry_lookback": 63, "exit_lookback": 126} | 0.0902 | 0.5716 | -0.3757 |
| kalman_trend | {"min_periods": 20, "observation_var": 0.001, "process_level_var": 1e-05, "process_trend_var": 1e-07} | 0.1072 | 0.7130 | -0.2597 |
| regression_slope | {"min_r_squared": 0.0, "window": 252} | 0.0878 | 0.5140 | -0.4357 |
| regression_slope | {"min_r_squared": 0.0, "window": 189} | 0.0838 | 0.5008 | -0.4619 |
| regression_slope | {"min_r_squared": 0.0, "window": 126} | 0.0746 | 0.4712 | -0.4127 |
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
