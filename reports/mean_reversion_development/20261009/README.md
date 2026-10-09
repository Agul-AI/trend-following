# Mean-Reversion Study — Development Only

**2001–2011; 90 configurations; shared 3-hour entry/exit confirmation.**
Validation (2012–2018) and historical OOS (2019–May 28, 2026) remain closed.

Baseline QQQ buy-and-hold: **0.0433 cash-excess Sharpe**, **-0.04% CAGR**, **70.40% maximum daily drawdown loss**.

| Family | Configs | Beat BH Sharpe | All-config median Sharpe | BH-beater median Sharpe | BH-beater median CAGR | BH-beater median round trips |
|---|---:|---:|---:|---:|---:|---:|
| SMA_DISTANCE | 18 | 6 | -0.0645 | 0.2057 | 5.15% | 39 |
| EMA_DISTANCE | 18 | 4 | -0.0395 | 0.1309 | 3.92% | 19 |
| ZSCORE | 27 | 5 | -0.0852 | 0.1496 | 3.91% | 25 |
| RSI | 27 | 17 | 0.1591 | 0.1922 | 4.89% | 113 |
| ALL | 90 | 32 | -0.0383 | 0.1737 | 4.48% | 38.5 |

## Development interpretation

There are **84 finite Sharpe scores and six undefined scores from configurations
with no completed trades**. Undefined cases remain in the 90-configuration
denominator. The overall finite-score median is **−0.0383**, so the best case
should not be treated as representative of the grid.

The highest development Sharpe is **SMA10 distance**, entry **≤−6%**, exit **≥0%**,
with the common 3-hour wait: **0.4013 Sharpe, 8.02% CAGR, 20.14% drawdown loss**,
and **20 completed round trips (1.82/year)**. This is an in-sample ranking, not a
validated or production winner. Up to five candidates per family were frozen;
validation and historical OOS have not been opened.

| Commission + slippage assumption | Configurations beating matched BH Sharpe /90 |
|---|---:|
| 1 + 1 bps | 32 |
| 1 + 3 bps — baseline | 32 |
| 1 + 10 bps | 26 |
| 1 + 25 bps | 14 |

These conditional counts do not show the same membership at every cost. All
configurations are retained, and the 20 finalists are chosen using only the
predeclared baseline development score.

## Method and limits

- Persistent weakness is the entry hypothesis, not evidence of a confirmed rebound. Cash enters below its threshold; long exits above its recovery threshold. Equality qualifies. Seven consecutive completed half-hour observations are required on either side, with next-safe-bar-open execution.
- SMA/EMA distance uses 10/20/50 daily sessions, entry −2/−4/−6%, exit 0/+1%. Z-score uses 10/20/50 sessions with sample SD (ddof=1), entry −1.5/−2/−2.5, exit −0.5/0/+0.5. RSI uses Wilder smoothing with 2/5/14-session lookbacks, entry 10/20/30, exit 50/60/70. Bollinger-equivalent thresholds are not duplicate families.
- Features use a causal forward-built total-return signal index and pure provisional daily previews; finalized daily state advances once at the provider quote's assumed 17:00 NY availability. Alpha quotes are reconstructed as-traded units, not guaranteed native raw or point-in-time captures; provider daily marks are not guaranteed official/executable closes.
- Fixed 3% nominal ACT/365 cash accrues over calendar time and compounds on the event clock. Effective cash growth is slightly above 3%, not exactly 3% APY. Every one-way order pays 1 bp commission plus the scenario's adverse slippage; taxes, leverage, shorting, financing and separate fund-expense charges are disabled.
- Hypothetical slippage 1/3/10/25 bps applies across the whole period. No hindsight regime labels or cost-specific reselection. Cost drag compares net wealth against a parallel gross ledger with the same decisions and fills.
- Dividends credit pre-open holders after splits and reinvest at the first safe observed open, using the same cost-free DRIP proxy for strategies and BH. Terminal positions liquidate at the last safe RTH close, followed by cash accrual to the 17:00 mark.
- Missing development intervals hold positions without fabricated observations or orders; pending orders are cancelled and streaks reset. Post-gap flags extend through the next session close and never affect ranking.
- Returns and Sharpe use daily valuation marks, 252 annualization and sample standard deviation. Drawdown is daily, not intraday. Round trips include terminal liquidation; trades/year divides round trips by actual calendar years; holding duration includes market closures.
- All 90 configurations and 360 scenario outcomes remain visible. Up to five finite-scoring development candidates per family are retained (20 total), with no minimum/maximum trade filter or BH eligibility requirement. Ties within 1e-10 resolve by stable ID. No validation winner was selected.
- BH-beater medians condition on observed development Sharpe; they are not whole-family or independent confirmation statistics. Parameter searching and finalist selection are in-sample. Previously examined history is retrospective, not untouched or live evidence.

## Evidence

[All 90 baseline results](candidate_metrics.csv) · [All 360 cost outcomes](candidate_metrics_all_costs.csv) · [Family comparison](family_summary.csv) · [Cost comparisons](family_summary_all_costs.csv) · [Matched benchmarks](benchmark_metrics.csv) · [Frozen development finalists](development_finalists.csv) · [Protocol summary](summary.json) · [Gap flags](post_gap_candidate_flags.csv) · [Gap events](post_gap_signal_events.csv) · [Artifact hashes](artifact_hashes.json)

[Independent derived-results audit](independent_audit.json) rederived all 360 candidate wealth-path statistics and eight passive paths, reproduced the 20 finalists and verified the preserved source hashes. This is arithmetic/control validation, not an independent historical price or signal implementation.
