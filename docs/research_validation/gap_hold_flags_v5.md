# QQQ daily indicators, gap-held positions and post-gap flags

This is the current 56,644-configuration QQQ/cash operating contract. Prices are
Alpha Vantage only; the fixed historical endpoint is May 28, 2026. The
2001–2011 development sweep has completed; independent QA of the current
results and cost-matched QQQ buy-and-hold comparison is underway. Validation and historical OOS have not been run.

## Daily rules and half-hour confirmations

SMA/EMA periods are 10, 20, 50, 100, 200 and 250 **daily trading sessions**.
Standalone EMA-MACD tuples are `(6,13,5)`, `(12,26,9)`, `(24,52,18)`, `(48,104,36)`
and `(96,208,72)`. Each side independently chooses one of 17 rules and one of 14
waits: 0, 0.5, …, 6.5 trading hours after the first qualifying completed close.
Required checks are `1 + 2 * hours`; 238 choices per side give **56,644 pairs**.

At a first hit at 10:00 NY, zero confirms at 10:00, 0.5 hours at 10:30 and one
hour at 11:00. A 6.5-hour wait requires 14 qualifying checks, crossing into the
next session. Nights, weekends and holidays are not confirmation samples.
Daily state updates once at the provider daily quote's assumed 17:00 availability;
every intraday preview is recomputed from the prior finalized daily state.

While in cash, both entry-rule and exit-rule support must qualify for the entry
streak. While long, loss of exit-rule support drives the exit streak. Conditions
are strict positive support; equality is unsupported. Streaks carry across
sessions except on condition failure, fills, segment resets, or blackouts.
Pending orders fill only at a safe observed next bar Open. All completed
half-hours, including 15:30–16:00, may confirm when their eligibility metadata
allows. No zero-wait decision uses an unfinished bar or its earlier Open.

## Pricing, valuation and accounting

Intraday OHLC is reconstructed into as-traded units from Alpha adjustment
factors, validated independently against native daily JSON fields, nonclosing
monthly anchors and same-provider 15/30-minute windows. It is **derived, not a
native adjusted=false capture**. Source snapshots are latest-vintage, not
certified point-in-time. Vendor-reported volume is not certified physical-share
volume and is unused by the fixed indicator/cost model.

Provider daily raw closes are assumed available at 17:00 New York. They value
holdings and finalize features for the next session; they are not guaranteed
Nasdaq Official Closing Prices. A provider daily close need not equal the last
16:00 intraday trade. Neither price is overwritten to force equality.
Executable prices remain safe observed intraday quotes. Terminal liquidation
uses the last verified tradable RTH close and then cash accrual to the same
17:00 endpoint as the benchmarks; it never invents a fill at the daily mark.

Baseline per-order costs are 1 bp commission plus 3 bps adverse slippage;
slippage stress is 1/3/10/25 bps. Taxes, financing and extra expenses are zero.
QQQ buy-and-hold and causal FRED-derived DTB3 cash share the same start,
accounting, valuation and terminal conventions. Splits/dividend entitlements
remain effective; a missing opening quote defers dividend reinvestment until a
safe observed opening quote. Cash observations are timestamped, not undated
future yields.

## Missing observations and tradability

Keep the simulator's **last filled QQQ/cash position**, not a pending target or
an indicator suggestion. Do not trade, impute quotes, or count confirmations
inside an unknown source gap. Cancel unfilled orders at gap start and reset both
streaks; fresh completed observations are required afterward. Holdings still
receive legitimate corporate actions and daily valuations; cash still accrues.

Fourteen source intervals remain absent: March 29, 2001, 09:30–10:00 (one
half-hour) and November 30, 2004, 09:30–16:00 (13 half-hours). Their disclosed
no-trade treatment is a user-authorized assumption, **not a data repair**.

The documented August 22, 2013 regulatory halt spans 12:23–15:25 NY. The 12:00
Open precedes the halt and may fill an already confirmed order, but that
half-hour cannot confirm at its Close. A 15:00 label cannot backdate a quote
whose post-halt trading begins at 15:25; the first fully safe post-halt interval
is 15:30–16:00. Halt handling does not fabricate absent prices. Known halts are
separate from unknown source gaps and do not produce the unknown-gap flags.

## Descriptive post-gap flags

Flag confirmed **buy/sell decisions** from every development configuration,
including configurations not selected as finalists. Mere support, early streak
observations, fills, forced liquidation and cancelled pre-gap orders are not
new confirmed signals. No flag establishes that missing data had no effect.

For a source gap ending at `g`, flag a decision at `t` when:
`g < t <= the strictly next NY trading-session date's actual market close`.
This is not a 24-hour window. It also includes valid decisions later on the gap's
ending day; early closes use the frozen exchange calendar.

| Source gap | Flagged decision window, New York time |
|---|---|
| March 29, 2001, 09:30–10:00 | After March 29 at 10:00 through March 30 at 16:00 |
| November 30, 2004, 09:30–16:00 | After November 30 at 16:00 through December 1 at 16:00 |

Persist each candidate's flag/buy/sell/counts, confirmed-signal event table, and
window metadata. Counts represent distinct decisions; association rows can map
one decision to multiple overlapping gap windows. The frozen event limit is
2,500,000, exceeding the two actual windows' 2,152,472 maximum associations.
Truncated event records prevent sealing; incomplete attempts are retained.
**Flags cannot filter, rank, alter Sharpe, or replace any selection.**

## Staged research and software

Daily initialization is 1999–2000; development 2001–2011; validation 2012–2015;
historical OOS 2016–May 28, 2026. Historical OOS is retrospective, not untouched
or live evidence. Each scored segment starts with cash and reset orders/streaks.
Full preparation is source custody/QA, not a claim of avoiding future source
parsing. A market stage itself receives only its physically bounded numerical
prefix. Source, proof, config, grid, accounting, consent and selection hashes
must verify before performance.

Development scores every configuration by finite daily cash-excess Sharpe
(252 annualization, sample standard deviation). Keep up to one finite finalist
per entry/exit family cell, maximum nine; undefined cells are excluded, and an
all-undefined grid stops. Differences within 1e-10 tie on stable candidate ID.
Only sealed finalists may enter validation; only its sealed winner may enter
historical OOS. No validation/OOS market evaluation has occurred.

Install `.[dev,research]`, then use `python scripts/run_qqq_research.py --help`.
The active configuration is `configs/qqq_daily_halfhour_v5_gap_flags_v2.yaml` and
private study `.cache/qqq_daily_halfhour_v5_alpha_gap_flags_v2`. Licensed inputs,
private human consent and prepared freezes are not committed. Missing inputs or
changed hashes fail closed. Internal versioned identifiers remain unchanged;
the public study and command use no version-number branding.
