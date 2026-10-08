# QQQ v5 Alpha-only data-quality correction — October 8, 2026

## Result

**False price-role gates corrected; genuine missing data still blocks strict backtesting.**
Actual default preparation verifies the source-unit proof and stops before any
portfolio calculation. It reports 14 unknown missing half-hours, not an
unverified raw-price/closing-equality failure. No v5 market scores were generated.

## Corrections

- Restore adjusted Alpha intraday OHLC to verified as-traded units, explicitly
  labeled **derived, not native raw capture**. All 317 months have independent
  nonclosing anchors; 6,685 daily observations match native JSON across eight
  fields; 86,170 comparable 15/30-minute OHLC windows agree.
- Preserve 558 daily-versus-last-intraday close differences. Daily feature/valuation
  quotes and executable RTH quotes have distinct roles; no prices or tolerances
  were altered to force equality. Daily quotes are assumed available at 17:00 NY,
  not certified official-auction prices or point-in-time vendor captures.
- Treat the documented 2013 halt as a tradability blackout, not an ordinary
  missing-source defect. Block unsafe signals/fills and reset/cancel at halt start.
- Fail closed on all supplied unknown coverage; bind source proofs, input/cache
  identities, coverage, accounting and policy before performance. Reject malformed
  proof counts and wrong cash-series identities.
- Seal up to nine finite development finalists; exclude undefined family cells,
  then select and seal the validation winner before retrospective historical OOS.

## Unrecovered source holes

| Session | New York interval | Missing half-hours |
|---|---|---:|
| March 29, 2001 | 09:30–10:00 | 1 |
| November 30, 2004 | 09:30–16:00 | 13 |

Alpha caches/backups do not recover these observations; a fresh same-provider
request was premium-denied. No other provider or synthetic OHLC was substituted.
Default `strict_tradable_v1` remains blocked. An alternative observed-only policy
requires **explicit human approval** and a fresh freeze: hold existing positions,
make no missing-interval decisions/fills, cancel pending orders, reset confirmations,
and value every session using real daily quotes. This is not a repair of source data.
No such approval has been received or inferred.

See [the implementation and operating policy](../../../docs/research_validation/alpha_quality_reconciliation_v5.md)
and `completion_receipt.json`. The 56,644-candidate grid, chronological periods,
May 28, 2026 endpoint and 2/4/11/26-bp one-way execution scenarios remain unchanged.
All 425 checked v4/legacy artifacts and three prior 13-file v5 source snapshots
remain unchanged. Resume files and existing monitors were not edited. Changes are
local, uncommitted and unpushed; the public repository does not yet contain this fix.

## Software verification

**589 tests passed in 229.90 seconds**, including 79 new quality-specific
fabricated/mocked tests. Repository-wide Ruff, seven new Python-file format
checks and `git diff --check` passed. These are software checks, not market
performance evidence.

## Publication status note

This is a dated pre-publication audit milestone. Any `committed: false`,
`pushed: false` or local/unpushed wording describes the state when this receipt
was recorded, not the current repository status. V5 is published to
[Agul-quant/trend-following](https://github.com/Agul-quant/trend-following).
Original numerical evidence and JSON receipt status fields are unchanged.
