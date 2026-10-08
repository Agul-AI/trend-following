# Alpha Vantage only QQQ research revision

All daily and intraday QQQ price inputs must be independently attested as
**Alpha Vantage**. Yahoo prices and other-provider stitching are rejected,
including in prospective monitoring. The previously agreed FRED-derived DTB3
cash yield remains a separately identified non-price benchmark.

## Scope and cutoff

The user selected the unchanged 09:30-anchored confirmation clock constructed
from Alpha Vantage 30-minute data. Its available endpoint is **May 28, 2026**.
The longer native hourly cache is not used to extend the tail. Initialization
1999–2000, development 2001–2011, validation 2012–2015 and all **7,225**
configurations remain unchanged. Historical OOS is January 4, 2016–May 28, 2026:
**2,615 daily sessions**. May 29 and June 1 are no longer required.

Daily SMA/EMA/MACD periods, independent 2–6-hour streaks, next-bar-open fills,
1-bp commission and 1/3/10/25-bp slippage scenarios are unchanged. No taxes.

## Verification and preservation

**479 tests pass**, including **100 v5 tests**. Repository Ruff, new Python-file
formatting and whitespace checks pass. All **425 preserved v4/legacy files**
match their original hashes, and all **13 archived prior-v5 source files** match
their archive hashes. The earlier registration/implementation receipt is not
overwritten; this revision uses a new study identity and private cache.

## Remaining data preparation

No API request, source-cache edit, market backtest or quote export was performed
by this revision. The audit reads only Alpha Vantage price caches.

Within the chosen endpoint, **16 expected half-hour timestamps** are absent:
14 unresolved internal gaps (the March 29, 2001 opening half-hour and the entire
November 30, 2004 session) and two August 22, 2013 halt-consistent exceptions.
Those halt intervals and other rows during the halt require no-trade/execution
classification, not invented prices. The earlier trailing 26-slot issue is gone.

A diagnostic restoration from Alpha daily raw/adjusted factors was assessed.
All **86,170** comparable 15-to-30-minute OHLC windows align, but **558**
required-scope reconstructed closes remain outside the existing daily-close
tolerance. These differences are **not proof of corrupt prices or a requirement
for paid redownloads**; auction endpoints/vintages and quote relationships still
need reconciliation. An unverified restoration is not labeled raw.

Actual prepare reports `data_not_ready`; the development guard rejects the
missing prepared-data seal. **No performance results were generated.**

See the [runbook](../../../docs/research_validation/daily_hourly_v5.md),
[Alpha-only configuration](../../../configs/qqq_daily_hourly_v5.yaml), and
[revision receipt](revision_receipt.json). These changes have not been pushed.

## Publication status note

This is a dated pre-publication audit milestone. Any `committed: false`,
`pushed: false` or local/unpushed wording describes the state when this receipt
was recorded, not the current repository status. V5 is published to
[Agul-quant/trend-following](https://github.com/Agul-quant/trend-following).
Original numerical evidence and JSON receipt status fields are unchanged.
