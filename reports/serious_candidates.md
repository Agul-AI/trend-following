# Serious Strategy Candidates

This file tracks candidates that are strong enough to keep testing. Promoted candidates remain documented here for provenance.

## Bear re-entry filter on top of q100 preferred strategy

**Short label:** `bear_reentry_buf1_slope20_20gt50`

**Status:** earlier serious candidate; superseded by promoted preferred variant `robust_buf010bp_slope30_20gt50` with q100 activation at +110%.

**Base strategy:** current q100 preferred strategy:

- QQQ hourly MACD + QQQ hourly 200MA gate/exit.
- Synthetic `QQQ_3X_CALC` exposure.
- `+300% -> 75%`, `+400% -> 50%` profit lock.
- Dynamic q100 trim after `+100%` trade gain.
- 40% synthetic trade-peak stop.
- Max one trade/day and 3% cash assumption in evaluation.

**Additional rule:**

```text
If QQQ hourly 200-day MA slope is negative:
    allow a new entry only if all are true:
        QQQ close is at least 1% above its hourly 200-day MA
        QQQ hourly 50MA slope over 20 trading days is positive
        QQQ hourly 20MA > QQQ hourly 50MA
```

**Full-sample comparison:**

| Strategy | Annualized return | Sharpe | Max DD | 2007-2009 DD | Trades/year | DD >20/30/40/50 |
|---|---:|---:|---:|---:|---:|---:|
| q100 preferred baseline | 27.11% | 0.805 | -56.36% | -56.36% | 5.70 | 25/10/6/4 |
| bear re-entry serious candidate | 28.75% | 0.844 | -52.15% | -51.01% | 5.09 | 24/8/5/3 |

**Years with blocked-entry triggers:** 2002, 2003, 2005, 2008, 2011, 2012, 2016, 2023, 2025.

**Main observed benefit:** blocks failed re-entry attempts during bear/weak-trend recoveries, especially August 2008.

**Main caution:** the worst drawdown shifts to 2010 and remains above -50%, so this is an improvement but not a complete drawdown solution.

## Promoted best robustness bear-filter preferred variant

**Short label:** `preferred_qqq_hourly_200ma_macd_profit_lock_300_400_stop40_q110_best_bear_filter`

**Status:** promoted to current preferred rule on 2026-06-04.

**Changes versus the earlier serious candidate:**

- q100 activation moved from `+100%` to `+110%`.
- Bear re-entry filter uses the 30-trading-day QQQ 50MA slope confirmation instead of the earlier 20-trading-day confirmation.
- The 1% buffer above the QQQ hourly 200MA and the QQQ 20MA > 50MA requirement are unchanged.

**Promoted bear-filter rule:**

```text
If QQQ hourly 200-day MA slope is negative:
    allow a new entry only if all are true:
        QQQ close is at least 1% above its hourly 200-day MA
        QQQ hourly 50MA slope over 30 trading days is positive
        QQQ hourly 20MA > QQQ hourly 50MA
```

**Full-sample comparison:**

| Strategy | Annualized return | Sharpe | Max DD | 2007-2009 DD | Trades/year | DD >20/30/40/50 |
|---|---:|---:|---:|---:|---:|---:|
| q100 preferred baseline | 27.11% | 0.805 | -56.36% | -56.36% | 5.70 | 25/10/6/4 |
| serious candidate, q100 + slope20 | 28.75% | 0.844 | -52.15% | -51.01% | 5.09 | 24/8/5/3 |
| **promoted preferred, q110 + slope30** | **29.41%** | **0.857** | **-52.15%** | **-51.12%** | **4.86** | **23/8/5/3** |

**Main caution:** this is the best current return/turnover robustness compromise, but it still has a worst drawdown above -50%.
