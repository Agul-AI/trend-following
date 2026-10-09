# Trend Following Study — Retained 2/3/4-Hour Wait Choices

Retain only **2, 3 and 4 trading hours** for development comparison. At each
choice, that one duration applies to both entry and exit for every indicator
pair. **No final global duration has been selected.** Do not combine different
waits across configurations or reopen validation/historical OOS.

These are subsets of the already completed fixed-3%-cash development diagnostic,
not new market simulations. The original 13-cohort records and frozen sources
remain unchanged. Each retained cohort has the same 289 daily-indicator pairs,
for 867 development configurations in total.

**2001–2011; fixed 3% nominal cash interest; 1 bp commission + 3 bps adverse
slippage per one-way order.** BH cash-excess Sharpe: 0.0433298242.

| Shared entry/exit wait | Consecutive half-hour hits | Beat BH Sharpe /289 | Percentage | Median completed round trips |
|---|---:|---:|---:|---:|
| 2 hours | 5 | 187 | 64.7% | 62 |
| 3 hours | 7 | 181 | 62.6% | 55 |
| 4 hours | 9 | 169 | 58.5% | 51 |

Median trades are computed across **all 289 configurations**, not just those
beating BH, over the full development period—not per year. They count completed
buy–sell round trips, including terminal liquidation; one-way order medians are
twice these figures. Waiting requires consecutive qualifying completed closes,
not an unconditional order delay. Nights/weekends do not add confirmation hours;
cash still accrues across closures. The next safe observed bar open fills orders.

These three development choices reflect the user's preference to reduce trading
without keeping the longest waits. The table shows an aggregate tradeoff, not a
guarantee that longer waits worsen every individual strategy or that any duration
will work in future data. No optimal wait, production study or new 45-member
selection was created. Prior familiarity with history cannot be undone.

- [Three retained count rows](counts_by_wait.csv).
- [All 867 corresponding development configurations](candidate_metrics.csv).
- [Scope and source hashes](summary.json).
- [Integrity hashes](artifact_hashes.json).
- [Original fixed-cash diagnostic](../../constant_cash_wait_counts/20261008/README.md).

The 3% rate is nominal ACT/365 with event compounding, not exactly 3% APY;
other original data/gap/valuation/accounting assumptions remain unchanged.
**Validation and historical OOS remain on hold.**
