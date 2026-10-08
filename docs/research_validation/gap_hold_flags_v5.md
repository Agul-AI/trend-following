# QQQ gap-held positions and post-gap development flags

The user authorized this operating policy on October 8, 2026. It is a new,
separately frozen study, not a repair of the Alpha source history or a change to
previously registered results. The earlier strict study and its source snapshot
remain preserved. Prices remain Alpha-only and historical evaluation ends May
28, 2026. There are still 56,644 daily-indicator/half-hour-confirmation candidates.

## Positions during missing intervals

Keep the **simulator's last filled QQQ/cash position**, not a pending target or an
indicator's latest suggestion. Do not trade, impute prices or count confirmations
inside an unknown source gap. Cancel any unfilled order at gap start and reset
both confirmation streaks; fresh completed observations are required afterward.

“Same position” does not mean artificially constant portfolio value. Real
provider daily quotes still value QQQ at the frozen 17:00 NY clock, cash accrues
causally, and split/dividend entitlements remain in effect. If an opening quote
is missing, dividend reinvestment waits for a safe observed opening quote.

## Which signals are flagged

Flag **confirmed buy/sell decisions** from every configuration, including those
not selected as finalists. Mere indicator support, initial streak observations,
executed fills, forced terminal liquidation and cancelled pre-gap orders are not
new confirmed signals. A flagged decision is not evidence that a missing-bar
signal would have occurred; it is a diagnostic of proximity to incomplete data.
Conversely, the absence of a flag does not establish that missing data had no
effect on a configuration.

The user explicitly selected **through the next trading session's close**, not
24 elapsed hours. For an unknown gap ending at `g`, flag a confirmed decision at
`t` when `g < t <= the next NY trading-session date's actual market close`. This
includes valid decisions later on the gap's ending day. Early closes use the
frozen exchange calendar; nights, weekends and holidays do not create samples.

| Unknown source gap | Post-gap flagged decision window (New York time) |
|---|---|
| March 29, 2001, 09:30–10:00 | After March 29 at 10:00 through March 30 at 16:00 |
| November 30, 2004, 09:30–16:00 | After November 30 at 16:00 through December 1 at 16:00 |

The documented 2013 regulatory halt is handled as before, but is not an unknown
source gap and does not create these development flags.

## Reporting and selection

Each development candidate receives a `has_post_gap_confirmed_signal` flag and
exact buy/sell/total post-gap confirmed-signal counts. A separate event table
records candidate ID, side, decision timestamp, source-gap bounds and window end;
it does not export raw prices. Event retention is frozen before performance.
The two actual windows contain 38 possible completed half-hour decision closes,
so 56,644 candidates can generate at most 2,152,472 signal-window associations.
The configured limit is 2,500,000. Incomplete/truncated event records cannot be
silently sealed as a complete diagnostic study.

**Flags are descriptive only.** They do not alter trades, return accounting,
Sharpe scores, ranking, family-finalist eligibility or validation selection. No
candidate may be excluded or replaced because a flag appears. Development still
searches the full grid; only its sealed finite family finalists may enter
validation, and historical OOS still requires a sealed validation winner.

## Separate entry point

```bash
cd /path/to/trend-following
# Activate your environment with the dev and research dependencies installed.
python scripts/run_qqq_gap_flags_v5.py prepare
# Explicitly request/start this market run after preparation is verified:
python scripts/run_qqq_gap_flags_v5.py development
```

The current config is `configs/qqq_daily_halfhour_v5_gap_flags_v2.yaml`; its private
study is `.cache/qqq_daily_halfhour_v5_alpha_gap_flags_v2`. Human consent is retained
as a hash-bound receipt. The original strict configuration is not overwritten
or reinterpreted. Preparation and software tests are not market performance.

The first gap-flags preparation attempt was interrupted after a pandas metadata
copying bottleneck was confirmed. Its registration, failure receipt, partial
prefixes and a 16-file source snapshot remain preserved. V2 changes only
proof serialization and artifact location; numeric hashes, source prices and
all research/position/flag/selection rules are unchanged.

The public repository contains code and curated audit metadata, not licensed
Alpha price caches, private authorization receipts or prepared packets. A fresh
clone can run the fabricated software tests; historical preparation requires
verified local inputs and a locally recorded authorization. The preparation
CLI must fail closed when those private inputs are absent.
