# Trend Following Study — Beat-BH Counts with Fixed 3% Cash

**Development only: January 2, 2001–December 30, 2011; 2,767 daily marks.**
This separately sealed diagnostic replaces the variable cash-yield proxy with
**3% nominal annual cash interest**, using simple ACT/365 accrual over elapsed
calendar time and compounding on the existing event clock. All cash balances
and the cash-excess Sharpe reference use this same predeclared assumption.

Each cohort contains **all 17 × 17 = 289 daily-indicator pairs**, with a shared
entry/exit confirmation wait of 0, 0.5, …, 6 trading hours. No pair is dropped,
no preferred wait or strategy is selected, and the global production wait still
awaits the user's choice. The 13 cohorts are development timing diagnostics,
not permission to unlock validation or historical OOS.

## Counts after commission and slippage

Both candidates and QQQ buy-and-hold include **1 bp commission + 3 bps adverse
slippage per one-way order**. Buy-and-hold trades only at the common start and
terminal liquidation; matched per-order assumptions do not imply equal total
fees. The cost-matched BH **daily cash-excess Sharpe is 0.0433298242**.

A beat requires a finite candidate score greater than the BH score by more than
`1e-10`. All 289 scores are finite at every wait, with no ties; strict `>` gives
the same counts here. These are the new 3%-cash results, not relabeled results
from the earlier variable-rate model.

| Equal entry/exit wait (trading hours) | Required consecutive half-hour hits | Beat BH Sharpe / 289 | Percentage |
|---:|---:|---:|---:|
| 0 | 1 | 200 | 69.2% |
| 0.5 | 2 | 208 | 72.0% |
| 1 | 3 | 193 | 66.8% |
| 1.5 | 4 | 192 | 66.4% |
| 2 | 5 | 187 | 64.7% |
| 2.5 | 6 | 180 | 62.3% |
| 3 | 7 | 181 | 62.6% |
| 3.5 | 8 | 173 | 59.9% |
| 4 | 9 | 169 | 58.5% |
| 4.5 | 10 | 171 | 59.2% |
| 5 | 11 | 169 | 58.5% |
| 5.5 | 12 | 159 | 55.0% |
| 6 | 13 | 165 | 57.1% |

The largest development count occurs at 0.5 hours, **208/289 (72.0%)**. That is
a descriptive result, **not an established optimal future wait**. Beating this
low-development-Sharpe benchmark does not establish superiority on CAGR,
drawdown, other costs or later performance. No significance claim is made.

## What “3%” means in this diagnostic

For each event-clock interval, the cash factor is
`1 + 0.03 × elapsed_calendar_seconds / (365 × 86400)`. It includes nights,
weekends and holidays. Cash observations in the adapter are repeated entries
of this foreknown research assumption, not recovered DTB3 observations or a
promised deposit yield.

Compounding means the effective cash-only CAGR is **3.04527%**, not exactly
3% APY. The optional annual-rate convention choice remains open; exactly 3%
effective APY would be a differently sealed calculation, not a relabeling of
these nominal-rate results. The same assumption applies to residual cash in
QQQ buy-and-hold, strategy accounts and the cash-excess reference.

The new accounting was recomputed for all **3,757 configurations**, not obtained
by merely subtracting 3% from old Sharpe scores. Cash accrual changes account
wealth, position quantities, fee amounts and daily returns. Cash's own
cash-excess Sharpe is undefined because its excess return is zero.

## Signals, data and boundaries

- SMA/EMA/MACD parameters remain **daily sessions**. The common wait `h` requires
  `1 + 2h` consecutive completed half-hour observations after the first qualifying
  hit. It is confirmation, not an unconditional execution delay. Zero still
  requires a completed observation and the subsequent safe observed bar open.
- Entry and exit use the same wait within each cohort. Indicator pairs remain
  identical across all 13 cohorts. Normal overnight closures do not advance
  confirmation counters; cash still accrues during those closures.
- Prices and actions come solely from the existing hash-bound development Alpha
  packet. Initialization history is 1999–2000; no initialization profits accrue.
  The development-only reader does not follow original master-price paths or
  open validation/OOS numeric files.
- The 14 missing development half-hours remain disclosed no-trade blackouts:
  last filled holdings persist, pending orders cancel and streaks reset. All
  candidate flags and confirmed post-gap decisions are retained, never used to
  filter/rank the counts. No quotes were invented or gaps repaired.
- Initial capital, corporate actions, terminal liquidation and daily 17:00 NY
  valuation match across candidate, QQQ and cash accounts. No taxes, financing,
  leverage, shorting or separate fund-expense charges are modeled.
- Source intraday prices are verified reconstructed as-traded Alpha values, not
  guaranteed native raw captures or point-in-time history. Provider daily quotes
  at assumed 17:00 availability are not guaranteed official/executable closes.
- The fixed 3% rate is hypothetical. Commission/slippage are research execution
  assumptions, not measured historical brokerage fills. Prior historical outputs
  are preserved with their original cash model and cannot become untouched again.

## Audit and evidence

The independent auditor rederived all 3,757 candidate scores from saved daily
returns/equity arrays and built its own 41,346-event 3% cash ledger. The cash
ledger matched daily values within **$5.14×10⁻¹¹**; all 13 count rows and all
200 post-gap event associations reconciled. This checks arithmetic, custody and
the cash ledger; it is not an independent signal/strategy reimplementation.
Private candidate arrays and exact authorization receipts are not published.

- [All 13 counts](counts_by_wait.csv).
- [All 3,757 candidate development metrics](candidate_metrics.csv), including poor results.
- [Cost-matched QQQ and cash metrics](benchmark_metrics.csv).
- [Derived benchmark daily curves](benchmark_daily_curves.csv).
- [Scope, rate convention and hashes](summary.json).
- [Independent audit](independent_audit.json).
- [Artifact hashes](artifact_hashes.json).
- [Shared future-study assumptions](../../../configs/shared_research_assumptions.yaml).

**Validation and historical OOS remain on hold.** No new 45-option shortlist,
winning wait or monitored trading strategy was selected. Previously frozen
sources, configurations and results remain unchanged.
