# Trend Following Study — Revised 2012–2018 Validation

## Revised periods, unchanged development selection

| Stage | Period | Status |
|---|---|---|
| Development | 2001–2011 | Existing fixed-3%-cash results and selection unchanged |
| Validation | 2012–2018 | Recomputed here for all 159 shared pairs at 2/3/4-hour waits |
| Historical OOS | 2019–May 28, 2026 | Defined only; not authorized or evaluated in this follow-up |

May 28, 2026 is the verified Alpha half-hour endpoint, not the end of calendar
2026. Earlier split/source/result records are preserved, not overwritten.

The same **159 indicator pairs** were selected using development only: each
beats development QQQ BH cash-excess Sharpe at **every 2-, 3- and 4-hour common
entry/exit wait**, after the same 1-bp commission/3-bps slippage and 3% nominal
cash interest. Their full **477 timing instances** are identical to the earlier
selection, with unchanged canonical pair/candidate hashes. No earlier validation
or OOS outcome entered numerical membership selection.

Membership was sealed before a **new physically bounded 1999–2018 price/feature
prefix** was prepared, which preceded performance evaluation. Original rows were
verified against preexisting session/row hashes. The CSV reader stops at the
first future-session marker before converting 2019+ price/action fields, and no
such fields enter the prepared frames. Whole-source custody metadata is reused;
original master prices and earlier outcome files are not followed or reopened.

## Fresh validation results

**January 3, 2012–December 31, 2018: 1,760 daily 17:00 NY valuation marks.**
Each segment starts with $1,000 cash and reset order/streak state; prior daily
prices initialize indicators, not profits. Cash, corporate actions, costs,
quote roles and terminal conventions match across strategies, QQQ and cash.

QQQ buy-and-hold after costs:

- Cash-excess Sharpe: **0.85621**.
- Net CAGR: **16.59%**.
- Daily maximum drawdown loss: **22.79%**.

| Shared entry/exit wait | Beat BH Sharpe /159 | Percentage | Median cash-excess Sharpe (BH beaters) | Median net CAGR (BH beaters) | Median drawdown loss (BH beaters) | Median completed round trips (BH beaters) |
|---|---:|---:|---:|---:|---:|---:|
| 2 hours | 9 | 5.7% | 0.876 | 14.89% | 13.67% | 10 |
| 3 hours | 12 | 7.5% | 0.887 | 15.00% | 14.08% | 13 |
| 4 hours | 7 | 4.4% | 0.885 | 14.88% | 15.06% | 12 |

The displayed medians include **only the configurations beating validation BH
cash-excess Sharpe at that particular wait**: 9, 12 and 7 respectively. They do
not use all 159 pairs or the three pairs beating BH at all three waits. The counts
and 159-pair denominators are unchanged. Original whole-cohort medians remain in
the `median_*` columns of `counts_by_wait.csv`; displayed values use the explicit
`beating_BH_median_*` columns.

These are **conditional summaries selected on the already observed validation
scores**, not independent performance estimates or a new candidate selection.
All 477 outcomes remain published. Most configurations did not beat BH Sharpe;
the winning-subset median CAGRs remain below BH's 16.59% CAGR. Beating Sharpe does
not imply beating return. No replacement, global-wait decision or OOS access
occurred to change this table.

Trades are completed buy–sell round trips across the **whole seven-year validation
segment**, including terminal liquidation—not annual counts; one-way execution
medians are twice these values. These medians are individual-strategy statistics,
not performance of an ensemble portfolio.

The descriptive number of pairs beating validation BH at **all three waits** is
**3**. This report does not select a new shortlist from that flag, replace any
candidate, choose a global wait or promote a winning strategy. All 477 outcomes,
including weak cases, remain in the published files.

## Scope and interpretation limits

- The split was changed **after earlier validation and historical OOS history had
  been examined**. This is an exploratory, retrospective resplit—not genuinely
  untouched confirmation. Seals establish unchanged numerical membership and
  chronology for this follow-up, not an absence of prior familiarity or
  multiple-comparison risk. Moving 2016–2018 into validation does not erase
  its earlier use as historical OOS in a different protocol.
- **Historical OOS is now 2019–May 28, 2026 and remains closed.** Defining this
  interval is not approval to run it. No new production wait or strategy was
  selected. The original 2012–2015 validation results remain intact.
- Fixed **3% nominal annual cash interest** uses simple ACT/365 accrual over
  elapsed calendar intervals compounded on the existing event clock, including
  overnight/weekend closures. Cash-only effective CAGR is **3.04527%**, not
  exactly 3% APY. The deterministic schedule is a user research assumption,
  not observed Treasury/deposit yields or a forward-filled DTB3 data series.
- Commission is 1 bp and adverse slippage 3 bps per actual one-way order. These
  are hypothetical execution assumptions, not measured historical fills. No
  taxes, leverage, shorting, financing or separate fund expenses are modeled.
- Daily-session SMA/EMA/MACD lookbacks and signal definitions remain unchanged.
  Provisional EMA/MACD previews use yesterday's finalized state; daily state
  finalizes once at the provider quote's assumed 17:00 availability. Common
  2/3/4-hour waits require 5/7/9 consecutive completed half-hour observations
  after the first hit—not an unconditional order delay. Orders fill at the
  subsequent safe observed intraday open. Normal market closures do not count
  as confirmation hours.
- Original development missing intervals remain disclosed no-trade/carry/cancel/
  reset assumptions, not repaired prices. Their post-gap windows have expired
  in validation, so validation flags are zero. Scored validation has no unknown
  missing slots; seven half-hours overlap the documented 2013 halt and suppress
  unsafe confirmations/fills. Flags never filter or rank the selection.
- Alpha-only reconstructed as-traded quotes are not guaranteed native raw
  captures or point-in-time vintages. Daily provider marks are not guaranteed
  official/executable closes. Maximum drawdown uses daily marks, not intraday lows.

## Audit and inspectable evidence

- [Unchanged 477 development selection instances](development_selection_metrics.csv).
- [Every 477 revised validation result](validation_candidate_metrics.csv).
- [159 pairs joined across development and revised validation](pair_comparison.csv).
- [Three beat-count, risk and trading summaries](counts_by_wait.csv).
- [Fresh matched QQQ/cash metrics](benchmark_metrics.csv) and [derived curves](benchmark_daily_curves.csv).
- [Bounded prefix, candidate identities, periods and chronology](summary.json).
- [Independent 477-path and passive-ledger audit](independent_audit.json).
- [Artifact hashes](artifact_hashes.json).
- [Preserved previous 2012–2015 validation](../../intersection_validation/20261008/README.md).

The independent auditor rederived all strategy statistics from saved derived
arrays and built its own raw-share QQQ/cash ledger on the new bounded prefix.
All 1,760 passive wealth marks matched exactly (29 dividends, no splits), and
all 477 paths, fees/order relationships and 9/12/7 counts reconciled. This is
arithmetic/control-ledger auditing, not an independent signal implementation.
Licensed raw prices, private candidate arrays, authorization receipts and full
input/source freezes are not published.
