# QQQ daily indicators with hourly confirmation implementation

The software is implemented locally, with daily SMA/EMA/MACD periods and
independent 2–6 completed-trading-hour entry/exit confirmations. Its full grid
contains **7,225 configurations**. Confirmations carry across trading sessions;
fills use the next bar open, including a partial closing bar. Commission is
1 bp and slippage scenarios are 1, 3, 10 and 25 bps. Taxes are disabled.

## Verification

- **465 tests passed**, including **86 new tests**.
- Repository-wide Ruff checks, new-file formatting and whitespace checks passed.
- **425 preserved v4/frozen/legacy files** matched their pre-implementation hashes.
- A fabricated-data full-grid evaluation and single-versus-batch ledger checks
  passed; these are software tests, not trading-performance results.
- Actual preparation returned `data_not_ready` (exit 3). Development correctly
  rejected the missing prepared-data seal (exit 2), and monitoring emitted no action.

## Market evaluation is blocked

**No development, validation or historical-OOS market performance has been
produced.** The configured key was denied for raw intraday requests for November
2004 and June 2026 as premium-only. No subscription or alternative data was bought.
The existing adjusted caches cannot be silently relabeled as raw as-traded prices.

After the documented scheduled-open correction for September 11, 2002, the
legacy 30-minute cache still has **42 missing subbars and 271 extra/off-grid
subbars** in its audited 2000–June 2026 coverage. Verified raw source acquisition,
bar-timing proof and required gap repair must precede market scoring.

Historical dates remain retrospective, previously examined history. No v4
reports, frozen source packets, existing monitors, prospective studies or resumes
were updated with hypothetical results. These changes have not been pushed.

See the [runbook](../../../docs/research_validation/daily_hourly_v5.md),
[configuration](../../../configs/qqq_daily_hourly_v5.yaml), and
[implementation receipt](implementation_receipt.json). Private market packets,
source snapshots, provider diagnostics and attempt logs stay in ignored caches.

## Publication status note

This is a dated pre-publication audit milestone. Any `committed: false`,
`pushed: false` or local/unpushed wording describes the state when this receipt
was recorded, not the current repository status. This project is published to
[Agul-quant/trend-following](https://github.com/Agul-quant/trend-following).
Original numerical evidence and JSON receipt status fields are unchanged.
