# Mean-Reversion Study — Validation

**2012–2018; exactly 32 development-selected configurations; shared 3-hour entry/exit wait (seven completed half-hour checks).**

Membership was sealed before validation price access. These are all 32 baseline development cases beating QQQ buy-and-hold cash-excess Sharpe by more than 1e-10, not the separate top-20 finalist list. No parameter retuning, filtering, replacement or validation winner selection occurred.

Baseline cost-matched QQQ buy-and-hold: **0.8562 cash-excess Sharpe**, **16.59% CAGR**, **22.79% maximum daily drawdown loss**.

| Family | Frozen cases | Beat BH Sharpe | All-case median Sharpe | BH-beater median Sharpe | BH-beater median CAGR | BH-beater median round trips | BH-beater median round trips/year |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMA_DISTANCE | 6 | 0 | 0.4947 | — | — | — | — |
| EMA_DISTANCE | 4 | 0 | 0.4510 | — | — | — | — |
| ZSCORE | 5 | 1 | 0.4716 | 0.8744 | 8.15% | 29.0000 | 4.1440 |
| RSI | 17 | 0 | 0.5079 | — | — | — | — |
| ALL | 32 | 1 | 0.5029 | 0.8744 | 8.15% | 29.0000 | 4.1440 |

## Baseline interpretation

Only **1/32** cases beats matched buy-and-hold Sharpe; **0/32** beats its CAGR.
The only Sharpe beater is **10-day z-score, entry ≤ −2, exit ≥ 0**, with the
unchanged shared 3-hour wait: **0.87435 Sharpe, 8.15% CAGR, 8.83% maximum daily
drawdown loss, 29 completed round trips (4.14/year)**. Its small Sharpe advantage
is not a validation winner selection, and it disappears at 10/25-bps slippage.

Across all 32 cases, median CAGR is **6.57%**, median round trips **14.5**
(**2.07/year**) and median daily drawdown loss **9.10%**. Median Sharpe **0.50290**
uses the **27 finite scores**; five zero-trade cash-only cases have undefined
Sharpe and remain in every 32-case denominator and non-Sharpe median. All 32
have shallower daily drawdown than buy-and-hold, but none matches its net growth.
These are separate strategy distributions, not an ensemble portfolio.

## Execution-cost sensitivity

| Slippage plus 1-bp commission | Cases beating matched BH Sharpe | All-case median Sharpe | BH Sharpe |
|---|---:|---:|---:|
| 1 + 1 bp | 1/32 | 0.5532 | 0.8566 |
| 3 + 1 bp | 1/32 | 0.5029 | 0.8562 |
| 10 + 1 bp | 0/32 | 0.3330 | 0.8550 |
| 25 + 1 bp | 0/32 | 0.2472 | 0.8523 |

## Method and limitations

- Every frozen candidate and all 128 candidate/cost outcomes remain visible, including any undefined Sharpe. Baseline is 1-bp commission plus 3-bps adverse slippage per one-way order; 1/10/25-bps slippage scenarios are hypothetical full-period execution stresses, without cost-specific reselection or hindsight labels.
- Hold 100% unlevered QQQ or cash, starting at $1,000. Cash uses the unchanged 3% nominal ACT/365 assumption, compounded on the event clock; effective cash growth is not exactly 3% APY. Taxes, financing and separate fund-expense charges are disabled.
- Same daily SMA/EMA-distance, sample-standard-deviation z-score and Wilder RSI rules as development. Persistent oversold entry and indicator-recovery exit use seven consecutive completed half-hour observations, with next-safe-bar-open execution. Preview daily indicators from yesterday's finalized state, never the previous preview; daily state finalizes once at assumed 17:00 New York availability.
- Alpha-only intraday prices are reconstructed as-traded units, not guaranteed native raw or point-in-time captures. Corporate actions build the causal total-return signal index. Matched dividends/DRIP, common starts and terminal-liquidation conventions are unchanged. Provider daily marks are not guaranteed official or executable closes.
- Initialization/development observations before 2012 provide indicator history only, no validation profits. Every strategy and benchmark starts fresh in cash; positions/counters reset at the segment boundary. Physically bounded input ends December 31, 2018; no 2019+ numerical price rows are opened or evaluated.
- Missing intervals and halts preserve the last filled position, cancel pending orders and reset streaks. Post-gap flags extend through the next trading session close, are descriptive only and never select candidates.
- Sharpe is daily cash-excess Sharpe with 252-session annualization and sample standard deviation (ddof=1). Drawdown is measured at daily 17:00 NY valuation marks, not intraday. Exposure is elapsed calendar-time in QQQ, not the fraction of trading checks. Holding duration includes overnight/weekend closures. Trades are completed round trips, including terminal liquidation; trades/year uses actual calendar years.
- BH-beater medians condition on observed validation performance and are descriptive, not a new selection. A dash means the conditional group is empty or its score undefined. Whole-family medians and full denominators remain alongside them. Cost-specific BH-beater groups can differ; conditional medians are not a same-membership cost comparison.
- This is retrospective validation on previously examined history, not genuinely untouched confirmation. There is no selected validation winner. Historical OOS (2019–May 28, 2026) remains closed and requires a later explicit instruction and sealed inputs.
- Saved-path arithmetic is independently recomputed before sealing/publication; this is not an independent historical price/action/signal replay. Original frozen trend and mean-reversion development sources/results remain unchanged.

## Evidence

[All 32 baseline results](candidate_metrics.csv) · [All 128 cost outcomes](candidate_metrics_all_costs.csv) · [Development versus validation](development_vs_validation.csv) · [Frozen development selection](development_selection_metrics.csv) · [Family comparison](family_summary.csv) · [Cost comparisons](family_summary_all_costs.csv) · [Matched benchmarks](benchmark_metrics.csv) · [Protocol summary](summary.json) · [Gap flags](post_gap_candidate_flags.csv) · [Gap events](post_gap_signal_events.csv) · [Independent arithmetic audit](independent_audit.json) · [Artifact hashes](artifact_hashes.json)
