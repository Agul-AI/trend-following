# QQQ gap-held positions and development flags — October 8, 2026

**Implemented and data-ready under the explicitly authorized hold-position policy.**
The 14 original Alpha missing half-hours remain missing; no prices or confirmations
were invented. No market development, validation or historical OOS run occurred.

- During gaps, preserve the simulator's last filled QQQ/cash holding—not a queued
  target. Cancel pending orders, reset confirmations, keep causal cash/corporate
  actions and daily valuation, then require fresh completed observations.
- For all 56,644 development configurations, flag confirmed buy/sell decisions
  after the source gap through the next NY trading-session date's actual close.
  March 29 flags end March 30, 2001 at 16:00; November 30 flags end December 1, 2004 at 16:00.
- Persist each candidate's boolean flag/buy/sell/counts, the decision-to-gap event
  table and explicit window/retention metadata. Full retention is frozen; any
  truncation prevents sealing. Mere support, fills and forced liquidation are
  not new confirmed signals. Known 2013 halt intervals do not create these flags.
- Flags are descriptive only: no filtering, score adjustment, ranking changes
  or replacement of development finalists/validation winners.

## Verification

**614 tests pass**, including a fabricated-price acceptance test covering every
56,644 candidate. Ruff, formatting and diff checks pass. The production
**development loader** verified its own numerical prefix ending December 30, 2011:
3,062 daily rows, 39,074 observed half-hours, 14 missing slots all nontradable.
All registration/prepared hashes verified. Source custody/preparation used the
full master; this is not a claim that preparation avoided future-price parsing.
No strategy returns or actual flagged-candidate counts have been calculated.

A pandas metadata-copy bottleneck was fixed without changing row/hash encoding;
every original numeric session hash and master unit-proof hash remain identical.
The interrupted first preparation, partial prefixes, original config and source
snapshot are preserved. V2 changes only serialization and artifact paths.
All 425 checked v4/legacy files and all five archived source snapshots are unchanged.
Existing monitors and resumes were not edited; changes are local and unpushed.

See [the operating policy](../../../docs/research_validation/gap_hold_flags_v5.md)
and `completion_receipt.json`. To start development explicitly, use the publication
checkout and `~/.venvs/myenv/bin/python scripts/run_qqq_gap_flags_v5.py development`.

## Publication status note

This is a dated pre-publication audit milestone. Any `committed: false`,
`pushed: false` or local/unpushed wording describes the state when this receipt
was recorded, not the current repository status. V5 is published to
[Agul-quant/trend-following](https://github.com/Agul-quant/trend-following).
Original numerical evidence and JSON receipt status fields are unchanged.
