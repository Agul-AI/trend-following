# QQQ daily indicators with half hour confirmation revision

Triggers are now checked at every **completed 30-minute close**, including
15:30–16:00. Daily SMA/EMA/MACD periods stay unchanged. Prices remain Alpha
Vantage only and historical evaluation stops May 28, 2026.

## Confirmation waits and grid

Entry and exit independently test **0, 0.5, 1, …, 6.5 trading hours** after the
first qualifying completed half-hour close. Required checks are **1 + 2h**.
For a first hit at 10:00: zero confirms at 10:00; 0.5 hours at 10:30; 1 hour at
11:00. A 6.5-hour wait requires **14 checks**, carrying overnight into the next
normal session's 10:00 close. Nights and weekends do not count as trading time.
This is sampled-close confirmation, not proof of continuous intrabar support.

Each side has 17 daily indicator rules × 14 durations = **238 choices**. The
full ordered grid contains **56,644 configurations**. Independent rule families,
all six SMA/EMA periods and five standalone MACD tuples are retained.

Fills use the next half-hour bar's Open, with adverse slippage and 1-bp
commission. Zero confirmation never uses an unfinished bar or the confirming
bar's earlier Open. A 16:00 confirmation queues the next session's Open; it
cannot fill retroactively or outside the final evaluation segment. Terminal
liquidation of existing holdings remains a separately disclosed convention.

## Verification and data status

**510 tests pass**, including **131 tests**. The core/data/independent
acceptance subset passed 83 tests, and staged-protocol tests passed 48. Ruff,
new-file formatting and whitespace checks pass. All 425 preserved v4/legacy
files and both 13-file earlier-source archives match their original hashes.
Old hour-only packet/state schemas are rejected by the explicit half-hour
v2 study; earlier registrations remain unchanged.

**No market performance results were generated.** Existing Alpha intraday
quote-relationship checks, 14 unresolved missing slots and two halt-consistent
slots remain pending. Actual preparation returns data_not_ready and the
development guard rejects a missing verified-data seal before evaluation.
No new API requests, source-cache changes, subscriptions or broker orders occur.

Costs remain 1-bp commission plus 1/3/10/25-bp slippage, with no taxes, leverage
financing or additional fund expenses. The FRED-derived cash yield remains a
non-price benchmark. Selection still uses development 2001–2011, validation
2012–2015 and a locked single-winner retrospective historical OOS test.

See the [runbook](../../../docs/research_validation/daily_hourly_v5.md),
[configuration](../../../configs/qqq_daily_hourly_v5.yaml), and
[revision receipt](revision_receipt.json). Changes are local and not pushed.

## Publication status note

This is a dated pre-publication audit milestone. Any `committed: false`,
`pushed: false` or local/unpushed wording describes the state when this receipt
was recorded, not the current repository status. This project is published to
[Agul-quant/trend-following](https://github.com/Agul-quant/trend-following).
Original numerical evidence and JSON receipt status fields are unchanged.
