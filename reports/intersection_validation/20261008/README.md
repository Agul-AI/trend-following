# Trend Following Study — Shared Development Pairs, Validation Only

## Selection locked before validation

Keep an indicator pair only if its **2001–2011 development cash-excess Sharpe
beats QQQ buy-and-hold at all three shared entry/exit waits: 2, 3 and 4 hours**.
Use the fixed **3% nominal annual cash assumption** and **1 bp commission +
3 bps adverse slippage per one-way order**. A beat must exceed the development
BH score (0.0433298242) by more than `1e-10`; undefined scores fail this selection.

Development beat sets contained 187, 181 and 169 pairs respectively. Their
intersection contains **159 of the 289 indicator pairs**. These are matching
entry/exit *indicator settings*, not identical full candidate IDs, since the
wait changes the candidate ID. All 159 pairs were retained at each of the three
uniform waits: **477 validation instances**, not 477 different indicator pairs.
No gap-flag, CAGR or drawdown filter was added.

The complete membership and timing variants were sealed before validation array
access. The old 45-option shortlist, original study, cash diagnostic, sources and
all previous sealed records remain unchanged. No original validation outcome or
historical OOS result was used to construct this intersection.

## Validation results: 2012–2015

**January 3, 2012–December 31, 2015; 1,006 daily valuation sessions.** Cash,
price/signal logic and costs are the same as selection. Each account restarts
with $1,000 in cash and reset orders/streaks; daily indicators retain causal prior
history. QQQ and cash share the same starting/terminal/valuation conventions.

Cost-matched QQQ buy-and-hold:

- Cash-excess Sharpe: **1.09027**.
- Net CAGR: **19.88%**.
- Daily maximum drawdown loss: **13.94%**.

| Shared entry/exit wait | Beat BH Sharpe /159 | Median cash-excess Sharpe | Median net CAGR | Median drawdown loss | Median completed round trips |
|---|---:|---:|---:|---:|---:|
| 2 hours | 0 | 0.703 | 11.37% | 13.35% | 9 |
| 3 hours | 0 | 0.646 | 10.66% | 14.11% | 8 |
| 4 hours | 0 | 0.743 | 11.80% | 13.74% | 8 |

**None of the retained pairs beats BH Sharpe in validation at any of the three
waits.** The development intersection therefore did not establish transferable
risk-adjusted superiority in this validation period. This is a research finding,
not a reason to replace losers, search the remaining pairs using validation, or
promote the best validation row as a validated trading recommendation.

Medians include **all 159 pairs per wait**, not just winners. Trades are completed
buy–sell round trips across the whole four-year validation segment, including
terminal liquidation; one-way execution counts are twice those figures. Medians
of individual strategy statistics are not portfolio/ensemble statistics.
All 159 scores were finite at every wait; no row was removed or replaced.
The descriptive count of pairs beating BH at **all three validation waits** is
zero; that flag did not change the frozen membership.

## Scope and limitations

- **Only validation is authorized by the latest user instruction. Historical OOS
  remains closed.** No global production wait, replacement configuration or new
  winner was selected after validation.
- This is **retrospective historical validation**, not genuinely untouched
  confirmation: the 2012–2015 history had been examined previously. New seals
  prove the numerical development-only selection and ordering of this follow-up,
  not an absence of earlier researcher familiarity or multiple-comparison risk.
- The 3% interest is a nominal ACT/365 research assumption, compounded over
  elapsed calendar time on the existing event clock. Cash-only effective CAGR
  is **3.04527%**, not exactly 3% APY. It is not a measured DTB3/deposit
  return series. Nights, weekends and holidays accrue cash interest.
- Indicators use daily-session lookbacks; waits are additional trading hours
  after the first qualifying completed half-hour close. Required consecutive
  observations are 5, 7 and 9 for 2, 3 and 4 hours. The wait is not an unconditional
  execution delay. Flat entry requires both rule supports; long exit requires
  consecutive failures of exit-side support. Interrupted streaks reset; normal
  closures do not count as confirmation hours. Fills use the next safe observed
  intraday open. Daily EMA/MACD state finalizes once at the assumed 17:00 quote.
- Prices/actions use the verified Alpha-only validation prefix with initialization
  and prior feature history from 1999 onward, bounded at December 31, 2015.
  No master/OOS prices, DTB3 numerical inputs or earlier validation outcome files
  were loaded. Quotes are reconstructed as-traded units, not guaranteed native
  raw captures or point-in-time source history. Daily marks are not guaranteed
  official closing or executable quotes.
- Scored validation has no unknown missing slots. Seven half-hours overlap the
  August 22, 2013 regulatory halt; unsafe signals/fills are suppressed. Original
  development source gaps remain disclosed hold/no-fill/reset assumptions.
  Their post-gap windows have expired in validation, so all validation flags are
  zero—not evidence that the development gaps were harmless. Flags never filter.
- Costs are hypothetical, not measured historical fills. Holdings are 100% QQQ
  or cash, without leverage, shorting, taxes, financing or separate fund expense.
  Corporate actions, dividend reinvestment, terminal liquidation and 17:00 NY
  valuation follow the unchanged original accounting conventions.

## Inspectable evidence and audit

- [All 477 development selection instances](development_selection_metrics.csv).
- [All 477 validation outcomes](validation_candidate_metrics.csv), including poor results.
- [All 159 pairs joined across development/validation and three waits](pair_comparison.csv).
- [Three wait-count and trade-median summaries](counts_by_wait.csv).
- [Matched QQQ/cash metrics](benchmark_metrics.csv) and [derived daily curves](benchmark_daily_curves.csv).
- [Scope, frozen set identity and chronology](summary.json).
- [Independent validation audit](independent_audit.json).
- [Public integrity hashes](artifact_hashes.json).

The independent auditor rederived every strategy's daily returns, CAGR,
cash-excess Sharpe, volatility and drawdown from the saved derived arrays, and
replayed a separate raw-share passive ledger. Every QQQ/cash daily balance
matched exactly (17 dividends, no splits; BH two orders/cash zero). Maximum
Sharpe discrepancy was below `4e-15`. The membership seal preceded validation
access; all 477 stayed in the report. This is arithmetic/control-ledger validation,
not an independent strategy implementation. Private source prices, candidate
arrays, authorization receipts and full freezes are not published.
