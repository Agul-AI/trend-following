# Trend-Following Research Pipeline with QQQ / Synthetic-TQQQ Case Study

A professional, interview-ready Python research pipeline for transparent trend-following strategies on real market data. The repo is intentionally structured in two layers: a **general ETF research framework** and a deeper **QQQ / synthetic-TQQQ flagship case study**.

This project is **not** a live trading system. It is designed to demonstrate a clean quantitative research workflow: data ingestion, validation, adjusted-price processing, signal generation, no-lookahead backtesting, transaction costs, parameter sensitivity, out-of-sample evaluation, plots, research logs, and documented limitations.

## How to read this repo

| Layer | Purpose | Main files |
|---|---|---|
| General ETF pipeline | Download, validate, process, and backtest ETF strategies from scratch | `src/trend_following/`, `scripts/run_backtest.py`, `configs/default.yaml` |
| Strategy library | Transparent trend-following rules such as SMA, crossover, momentum, regime, breakout, regression, and Kalman-style trend | `src/trend_following/signals.py`, `src/trend_following/regime.py` |
| Flagship case study | Focused QQQ-driven synthetic-TQQQ research thread | `docs/qqq_tqqq_case_study.md`, `reports/preferred_strategy_rules.md`, `reports/project_log.md` |
| Curated outputs | Compact result tables and drawdown figures for the current preferred strategy | `reports/tables/`, `reports/figures/` |

## Current flagship case study: QQQ-driven synthetic TQQQ

The main research application is a long-only strategy that uses **QQQ as the signal asset** and **synthetic +3x QQQ exposure** as the risk asset. The current preferred rule is:

- Signal source: QQQ.
- Exposure: synthetic TQQQ, calculated as `QQQ_3X_CALC`.
- Entry: QQQ hourly MACD histogram > 0.
- Entry gate: QQQ hourly close > QQQ hourly 200-day moving average.
- Exit: QQQ hourly close < QQQ hourly 200-day moving average.
- No daily regime gate.
- No profit lock.
- Max one trade per day.
- Out-of-market cash earns 3% annualized in evaluation.

Detailed case-study documentation and retained-candidate performance tables: [`docs/qqq_tqqq_case_study.md`](docs/qqq_tqqq_case_study.md).

Curated performance CSVs:

- [`reports/tables/qqq_tqqq_retained_candidate_performance_synthetic.csv`](reports/tables/qqq_tqqq_retained_candidate_performance_synthetic.csv)
- [`reports/tables/qqq_tqqq_retained_candidate_performance_actual_tqqq.csv`](reports/tables/qqq_tqqq_retained_candidate_performance_actual_tqqq.csv)

Confirmed preferred rules: [`reports/preferred_strategy_rules.md`](reports/preferred_strategy_rules.md).

Research history, including failed experiments: [`reports/project_log.md`](reports/project_log.md).

## Why trend following?

Trend following is a simple and well-studied family of rules that can be implemented transparently and audited for common backtesting mistakes. The initial strategy set is intentionally broad so the research pipeline, assumptions, and validation choices remain easy to explain in interviews. The QQQ/synthetic-TQQQ case study then shows how the same framework can support a deeper, more focused research investigation.

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

The default config downloads **daily** data using `yfinance`. Long-history intraday data is configured separately in `configs/alpha_vantage_max_history.yaml` and requires an Alpha Vantage API key with historical intraday entitlement.

## Leveraged ETF note

The default universe includes the available **long +3x** ETF candidates that closely map to the current unlevered universe: UPRO for S&P 500/SPY, TQQQ for Nasdaq-100/QQQ, URTY for Russell 2000/IWM, TMF for 20+ year Treasuries/TLT, EDC for emerging markets/EEM, and DRN for real estate/VNQ. GLD and EFA are left without a default +3x ETF counterpart because there is no clean current broad +3x ETF match in the same style.

Leveraged ETFs target daily multiples and can experience path-dependent compounding and volatility drag, so they should be analyzed separately from unlevered buy-and-hold ETFs and are not assumed to be appropriate long-term holdings. Product availability can change, so verify tickers before using them in research.

## Data source and limitations

Data is downloaded with the configured source (`yfinance` by default for daily data, `stooq` for long-history hourly CSVs when a Stooq apikey is available, or `alpha_vantage` for long-history intraday data when entitled), then cached locally as parquet:

- Raw downloaded yfinance daily files: `data/raw/{ticker}.parquet`
- Processed yfinance daily panels: `data/processed/*.parquet`
- Raw downloaded Stooq hourly files: `data/raw/stooq_hourly/{ticker}.parquet`
- Processed Stooq hourly panels: `data/processed/stooq_hourly/*.parquet`
- Raw downloaded Alpha Vantage hourly files: `data/raw/alpha_vantage_hourly/{ticker}.parquet`
- Processed Alpha Vantage hourly panels: `data/processed/alpha_vantage_hourly/*.parquet`

Important limitations:

- Yahoo Finance data can contain revisions, missing values, ticker changes, and corporate-action issues.
- Yahoo Finance intraday/hourly history is limited to roughly 730 days; use `configs/stooq_hourly.yaml` or `configs/alpha_vantage_hourly.yaml` for longer hourly history when the required vendor apikey/entitlement is available.
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

General ETF pipeline commands:

```bash
python scripts/download_data.py --config configs/default.yaml
python scripts/run_backtest.py --config configs/default.yaml --strategy sma_trend
python scripts/run_backtest.py --config configs/default.yaml --strategy sma_crossover
python scripts/run_backtest.py --config configs/default.yaml --strategy tsmom
python scripts/run_backtest.py --config configs/default.yaml --strategy donchian_breakout
python scripts/run_backtest.py --config configs/default.yaml --strategy regression_slope
python scripts/run_backtest.py --config configs/default.yaml --strategy kalman_trend
python scripts/run_backtest.py --config configs/default.yaml --strategy cross_sectional_momentum
python scripts/run_backtest.py --config configs/default.yaml --strategy regime_switch --tickers QQQ
python scripts/run_mixed_frequency_backtest.py --config configs/regime_hourly_qqq.yaml
python scripts/run_parameter_sweep.py --config configs/default.yaml
python scripts/make_report.py --config configs/default.yaml
pytest
```

Current QQQ/synthetic-TQQQ case-study commands, assuming the required Alpha Vantage and synthetic data caches already exist:

```bash
python scripts/run_tqqq_cash_yield_candidate_comparison.py
python scripts/plot_preferred_worst_drawdowns_hiking.py
```

For long-history hourly Alpha Vantage data, store `ALPHA_VANTAGE_API_KEY` in `.env`, confirm your plan includes the historical intraday endpoint, then run:

```bash
python scripts/download_data.py --config configs/alpha_vantage_hourly.yaml --force
```

This can require thousands of monthly API calls for the full ETF universe, so set `ALPHA_VANTAGE_PAUSE_SECONDS` according to your plan's rate limit.

For a one-month Alpha Vantage Premium bootstrap that maximizes useful history for the expanded regular + leveraged ETF universe, use:

```bash
export ALPHA_VANTAGE_PAUSE_SECONDS=0.85  # about 75 requests/minute with a small buffer

python scripts/download_alpha_vantage_bulk.py \
  --config configs/alpha_vantage_max_history.yaml \
  --intervals 15min 30min 60min 1d \
  --pause-seconds 0.85
```

The bulk script downloads daily adjusted data first, uses each ticker's actual first available daily date to avoid pre-inception intraday month calls, and then downloads 15-minute, 30-minute, and 60-minute bars into separate caches:

- `data/raw/alpha_vantage_15min/{ticker}.parquet`
- `data/raw/alpha_vantage_30min/{ticker}.parquet`
- `data/raw/alpha_vantage_60min/{ticker}.parquet`
- `data/raw/alpha_vantage_daily_adjusted/{ticker}.parquet`

To estimate request counts without downloading:

```bash
python scripts/download_alpha_vantage_bulk.py \
  --config configs/alpha_vantage_max_history.yaml \
  --intervals 15min 30min 60min 1d \
  --pause-seconds 0.85 \
  --dry-run
```

To create a separate synthetic daily-reset 3x TQQQ-style series from QQQ, without
using the actual TQQQ history:

```bash
python scripts/create_synthetic_tqqq.py --intervals 1d 15min 30min 60min
```

This writes `TQQQ_CALC.parquet` under `data/raw/synthetic_tqqq_1d/`,
`data/raw/synthetic_tqqq_15min/`, `data/raw/synthetic_tqqq_30min/`, and
`data/raw/synthetic_tqqq_60min/`. The synthetic close return is exactly
`3 * QQQ adjusted close return` each day before fees, financing, and tracking
error. Intraday synthetic bars are mapped from QQQ bars relative to the prior
daily close, preserving the daily-reset convention.

For long-history hourly Stooq data, first obtain a Stooq CSV apikey by opening a URL like `https://stooq.com/q/d/?s=spy.us&get_apikey`, completing Stooq's captcha flow, and copying the `apikey` value from the generated CSV link into `.env` as `STOOQ_API_KEY=...`. Then run:

```bash
python scripts/download_data.py --config configs/stooq_hourly.yaml --force
```

To compare the overlapping Stooq cache with the recent Yahoo hourly cache:

```bash
python scripts/compare_data_sources.py \
  --config configs/default.yaml \
  --left-dir data/raw/hourly \
  --right-dir data/raw/stooq_hourly \
  --left-label yahoo \
  --right-label stooq \
  --right-timezone America/New_York
```

The comparison report is saved to `reports/tables/data_source_comparison_yahoo_vs_stooq.csv`.

For a first smaller run using only the initial deliverable tickers:

```bash
python scripts/download_data.py --config configs/default.yaml --tickers SPY QQQ IWM TLT GLD
python scripts/run_backtest.py --config configs/default.yaml --strategy sma_trend --tickers SPY QQQ IWM TLT GLD
```

## Backtesting assumptions

The first version is deliberately simple and auditable:

- Long-only strategies; signal values are in `[0, 1]`.
- General daily ETF examples assume cash earns zero return unless a script explicitly adds a cash yield.
- The current QQQ/synthetic-TQQQ case-study evaluation assumes out-of-market cash earns 3% annualized.
- Daily returns are computed from adjusted close prices.
- The default return convention is close-to-close.
- A signal computed using the close on day `t` is **not** allowed to earn day `t` or day `t+1` close-to-close returns when the execution assumption is next-close execution.
- With the default `execution_delay_days = 1`, a close-`t` signal is executed at the close of `t+1`, so P&L starts with the close-to-close return ending on `t+2`. In code this is implemented as `raw_signal.shift(execution_delay_days + 1)` before multiplying by returns.
- Intraday backtests additionally enforce `max_trades_per_day = 1` by default: after positions are shifted into executable weights, the simulator accepts only the first position change per calendar day and ignores later same-day changes. This prevents buy+sell round trips on the same day.
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

4. **Donchian breakout**
   - Long after price makes an entry-window high, and exit after an exit-window low.
   - Default entry/exit lookbacks: 252/126 days.

5. **Regression slope trend**
   - Long if a rolling regression slope on log adjusted prices is positive.
   - Default window: 126 days.

6. **Kalman/state-estimated trend**
   - Long if a transparent local-linear Kalman filter estimates a positive latent trend slope.
   - Default minimum warm-up: 20 observations.

7. **Cross-sectional momentum**
   - Long the strongest assets by trailing return across the universe.
   - Default lookback/top-N: 126 days / top 3 assets, with positive absolute momentum required.

8. **QQQ regime switch**
   - Classifies QQQ into `trend`, `mean_reversion`, `risk_off`, or `neutral` using only close/return features.
   - The default `trend` regime is QQQ above a rising 200-day SMA; variance ratio is not required for trend by default, but it remains available for mean-reversion classification.
   - In trend regimes it applies a 50/200 SMA crossover; in mean-reversion regimes it buys oversold QQQ pullbacks; otherwise it holds cash.
   - Outputs a regime table to `reports/tables/regime_switch_regimes.csv` and a regime diagnostic plot.

9. **Mixed-frequency QQQ regime + hourly trend**
   - Keeps the regime classifier on daily QQQ adjusted closes.
   - Trades an hourly QQQ crossover only when yesterday's daily regime estimate is `trend`.
   - The crossover keeps the classic 50/200 **trading-day** reference by converting to hourly bars; for 60-minute Alpha Vantage data this defaults to 300/1200 hourly bars using 6 bars per day.
   - Uses `configs/regime_hourly_qqq.yaml` and writes outputs with the `regime_switch_hourly_daily_regime` prefix.

10. **Optional volatility targeting**
   - Can scale positions to a target annualized volatility.
   - Default target: 10% annualized volatility.
   - Leverage is capped at 1.0 in v1.

## Parameter experiments

The parameter sweep evaluates in-sample and out-of-sample performance without selecting parameters based on full-sample results:

- SMA windows: `[50, 100, 150, 200, 250]`
- Time-series momentum lookbacks: `[63, 126, 189, 252]`
- Moving-average crossover short windows: `[20, 50, 100]`
- Moving-average crossover long windows: `[100, 150, 200, 250]`
- Donchian entry lookbacks: `[63, 126, 252]`
- Donchian exit lookbacks: `[21, 63, 126]`
- Regression slope windows: `[63, 126, 189, 252]`
- Cross-sectional momentum lookbacks: `[63, 126, 252]`
- Cross-sectional momentum top-N values: `[2, 3, 5]`

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

- QQQ/synthetic-TQQQ case study: `docs/qqq_tqqq_case_study.md`
- Current preferred strategy rules: `reports/preferred_strategy_rules.md`
- Research/project log: `reports/project_log.md`
- Metrics: `reports/tables/*_metrics.csv`
- Daily backtest data: `reports/tables/*_daily_results.csv`
- Parameter sweep: `reports/tables/parameter_sweep.csv`
- Figures: `reports/figures/*.png`
- Research memo: `reports/summary.md`

## Example interview explanation

Framework explanation:

> This project demonstrates an end-to-end research pipeline using real market data: ingestion, validation, feature/signal construction, backtesting, transaction costs, parameter sensitivity, and out-of-sample evaluation.

Case-study explanation:

> I first built a general ETF trend-following framework, then used it to deeply investigate a QQQ-driven synthetic-TQQQ strategy with explicit no-lookahead timing, cash-yield assumptions, drawdown analysis, and a research log of both failed and successful variants.

## Future extensions

- Add a richer data vendor and compare vendor discrepancies.
- Add open-to-open or close-to-next-open execution once adjusted open returns are modeled carefully.
- Add risk-free cash returns and financing costs.
- Add shorting, borrow constraints, and portfolio-level volatility targeting.
- Add walk-forward parameter selection.
- Add vectorbt later for performance comparison, while keeping the transparent from-scratch backtester as the reference implementation.
- Convert the current QQQ/synthetic-TQQQ case-study scripts into a more unified configurable experiment runner.
