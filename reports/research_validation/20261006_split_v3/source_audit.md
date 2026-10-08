# Fresh split v3 — daily source audit

## Approved scope

Daily primary experiment, with partitions fixed before new research: **2002–2011 development; 2012–2015 validation; 2016–June 1, 2026 test**. This audit inspected only source coverage, transformations and accounting requirements. No old performance conclusions or new strategy/test outcomes were read or calculated.

`daily_source_audit.json` is publication-safe aggregate evidence: relative source identifiers, hashes, counts, assumptions and limitations, without raw quotes or private paths. The broader private inventory records exact local paths separately.

## Data readiness

- AV daily QQQ raw/adjusted/action data cover **2,519 / 1,006 / 2,617 sessions** respectively, totaling **6,142**, without duplicate dates or missing cells.
- Independent Yahoo daily dates exactly equal AV dates in every partition. That file is for source-coverage verification, not another strategy outcome or candidate-selection input.
- AV daily provides **28 / 17 / 42 distributions** and no splits in these partitions. Earlier observations are available for feature-only warmup.
- Daily primary avoids the unresolved intraday missing session, start-label and after-hours issues documented in the broader audit. No hourly row is relabeled as a daily bar.
- DFF/DTB3 modeled availability can be reconstructed exactly from the bound raw sources. All 6,142 daily accounting intervals have finite non-stale coverage. DTB3 includes nine negative-yield intervals, retained rather than floored.

## Explicit fresh accounting contract

The daily accounting clock is **16:00 America/New_York on every observed session**, not a claim that early-close days actually traded until 16:00. Delayed close execution must be enforced: a target based on today's finished close cannot earn the return that ends at that close.

Build the leveraged total-return proxy causally from daily raw close and supplied dividends/splits. At each daily reset borrow twice prior synthetic NAV for three-times risky exposure. Charge DFF plus 50 bp on that debt ACT/360; charge explicitly hypothetical 95 bp expense plus 50 bp additional drag ACT/365 once. This is a **3× QQQ total-return proxy**, not actual TQQQ. Never run already-funded synthetic returns through another funding/expense subtraction.

Investor cash uses the declared DTB3 91-day investment-yield conversion, ACT/365, including negative yields. Synthetic investor positions rebalance **only when their target allocation changes**; an unchanged fractional target may drift with prices and cash. Root-approved physical QQQ full 0/1 controls instead reinvest ex-date distribution cash at the same accounting close, subject to costs. The synthetic instrument's internal daily leverage reset is distinct from investor rebalancing. Fees, slippage and terminal liquidation must be explicit, matched across strategy and controls.

Physical QQQ distributions use the root-declared **ex-date immediate cash-credit approximation**, not verified payment-date cash. Taxed distributions use an ordinary-income hypothetical scenario, not proven tax qualification. A hypothetical synthetic-unit tax scenario is not actual leveraged-fund investor taxation. Adjusted total returns must never also receive those explicit cash dividends.

## Limits and sealing

Existing sources are retrospective/latest vintage. A modeled two-business-day publication lag is not a point-in-time data archive. Prior exposure to the historical test years cannot be erased: describe the rerun as a newly protocol-separated historical study, not a never-seen holdout.

Original data files remain mutable. Copy approved bytes into a new immutable package, verify the JSON hashes, and seal source/feature/selection/cost rules before candidate selection or test performance. The audit supports data readiness, **not strategy merit**. Code-level unchanged-target drift and physical QQQ reinvestment were independently confirmed using artificial fixtures in `daily_implementation_hand_audit.md`. Independent winner-equity reconciliation remains a separate check after sealed results exist.
