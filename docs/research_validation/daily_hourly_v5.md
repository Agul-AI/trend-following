# QQQ daily indicators with half hour confirmation

This revision checks triggers at every **completed 30-minute close** using daily
SMA/EMA/MACD lookbacks. All daily and intraday QQQ price inputs must come from
**Alpha Vantage**. The chosen half-hour cache fixes historical evaluation at
**May 28, 2026**. Yahoo prices, other-provider stitching and trailing extensions
are prohibited. No strategy profitability is assumed.

## Daily indicators and confirmation durations

SMA and EMA lookbacks are **10, 20, 50, 100, 200, 250 trading days**. Standalone
MACD uses daily `(fast EMA, slow EMA, signal EMA)` tuples **(6,13,5), (12,26,9),
(24,52,18), (48,104,36), (96,208,72)**. There is no separate EMA price gate inside
MACD. The 12/26/9 reference is conventional; the other scales are research
hypotheses, not proven optima.

Entry and exit independently choose one of 17 indicators and a confirmation
wait from **0, 0.5, 1, 1.5, …, 6.5 trading hours**: 14 durations per side.
There are **238 choices per side and 56,644 ordered configurations**.

A wait starts at the **first qualifying completed half-hour close**. It is not
an assumption that the condition was continuously true before that first sample.
For a first qualifying close at 10:00 New York time:

| Confirmation wait | Consecutive qualifying closes required | Confirmation time, if uninterrupted |
|---|---:|---|
| 0 hours | 1 | 10:00 |
| 0.5 hours | 2 | 10:30 |
| 1 hour | 3 | 11:00 |
| 1.5 hours | 4 | 11:30 |
| 6.5 hours | 14 | Next normal session at 10:00 |

The formula is **required checks = 1 + 2 × confirmation hours**. Thus zero and
0.5 hours are distinct: zero means no additional wait, not use of an unfinished
bar. A normal session provides 13 closing samples; a 6.5-hour wait from its first
10:00 sample necessarily crosses a session boundary. Holidays, nights and
weekends do not advance the confirmation clock. Opposite conditions reset it.
This is persistence at sampled closes, not proof of continuous intrabar support.

At each half-hour close, the observed raw price forms **one provisional daily
observation** using the causal split/dividend total-return index. Every preview
is calculated from yesterday's finalized daily state. EMA/MACD do not advance
recursively 13 times per day; official daily state advances once at that
session's actual close. Full-window eligibility and first-observation seeding
are fixed and tested.

Support is strict price-above-MA for SMA/EMA, or strict positive MACD histogram.
While cash, both selected entry and exit supports must qualify for the entry
wait. While long, exit support must fail for the exit wait. Equality fails
support; entry failure alone does not force exit. Fills and scored-segment
boundaries reset counters. All completed 30-minute bars count, including
**15:30–16:00**. The old hour-only exclusion of this bar no longer applies.

## Bar boundaries and fills

The cached Alpha Vantage timestamp labels a bar's **start**. The 09:30 row is
09:30–10:00 and completes at 10:00. Its final Close cannot be known at 09:30.
After confirmation, the **next bar's Open** supplies the modeled fill, plus
adverse slippage and commission. There is no additional half-hour wait.
The previous Close and next Open may differ despite sharing a nominal boundary.
This zero-latency boundary convention is a historical approximation, not a
promise of an executable live price.

Normal session bars run 09:30–10:00 through 15:30–16:00: **13 observations**.
A 13:00 early close provides seven. The frozen calendar also records the
announced 11:00 opening on September 11, 2002; no nonexistent pre-open bars are
invented. [Exchange circular](https://www.nasdaqtrader.com/Content/NewsAlerts/ISEMics/MIC-2002-03%24Market_Opening_Time_-_September_11_2002%2420020909.pdf)

A confirmation at 16:00 queues the next session's opening fill, not a retroactive
15:30 fill. If the segment ends that day, the pending order is canceled rather
than filled outside the dataset. Existing positions receive the separately
specified final-close liquidation used consistently for benchmarks.

## Accounting and chronological research

The portfolio is 100% unlevered QQQ or cash and begins each scored segment with
1,000 units of cash, no position, pending order or inherited streak. Raw splits
change physical shares. Only pre-open holders receive the modeled ex-date
dividend, reinvested at that raw open without separate transition charges.
This explicit proxy is not a broker payment-date/DRIP ledger. New distributions
never receive retroactive cash interest.

The previously agreed causal DTB3 cash-yield proxy remains a **non-price**
benchmark from FRED, separately identified from Alpha Vantage prices. Its
publication lag is modeled and its observations are latest-vintage, not
reconstructed point-in-time vintages. No taxes, synthetic financing or additional
fund expense is deducted; observed QQQ prices already embody fund expenses.

| Execution scenario | Commission | Slippage | Total one-way cost |
|---|---:|---:|---:|
| Smooth | 1 bp | 1 bp | 2 bps |
| Moderate selection baseline | 1 bp | 3 bps | 4 bps |
| Volatile stress | 1 bp | 10 bps | 11 bps |
| Severe stress | 1 bp | 25 bps | 26 bps |

These are hypothetical constant scenarios across the full chronology, not
measured QQQ costs or historical broker commissions. Slippage changes execution
price; commission is a separate cash charge. Neither is charged twice.

| Stage | Calendar period | Daily observations |
|---|---|---:|
| Feature initialization only | Available November 1, 1999–December 31, 2000 | 295 |
| Development | 2001–2011 | 2,767 |
| Validation | 2012–2015 | 1,006 |
| Historical OOS | January 2016–May 28, 2026 | 2,615 |

Counts describe daily coverage, not verified tradable half-hour coverage.
Initialization earns no profits. Its 295 observations satisfy MA250 and the
longest MACD's 279-observation availability minimum, but do not eliminate EMA
initialization sensitivity.

1. Freeze the full grid, code, configuration, verified data, accounting and
   selection rules before market performance calculations.
2. Search all **56,644 configurations on development only** under 1-bp
   commission and 3-bp slippage. Freeze one finite-scoring finalist per ordered
   entry/exit family cell, up to nine.
3. Validation ranks only those finalists and seals one winner before numerical
   historical-OOS rows are released.
4. Historical OOS evaluates only that winner under all four cost scenarios, with
   matched QQQ buy-and-hold and cash controls. No stress-cost reselection,
   parameter retuning or winner replacement is permitted.

Both selection steps use **daily cash-excess Sharpe**, 252-session annualization
and sample standard deviation. Differences within `1e-10` tie by stable candidate
ID only. Undefined scores are excluded; no finite candidate stops the stage.
QQQ buy-and-hold enters at the first scored session's open. Reports expose CAGR,
daily-close drawdown/volatility, turnover, completed trades and cost drag.
**Confirmation waits** count trading intervals; **holding hours/exposure** count
actual elapsed calendar time, including overnight/weekend risk.

History examined previously is retrospective evidence, not a genuinely untouched
holdout or live performance. Code/hash gates are not an OS sandbox for all
same-user tools. Fresh prospective confirmation starts only after the final
freeze. Larger parameter searches do not imply stronger evidence.

## Data gate and running the study

Use the requested environment from this checkout:

```bash
source ~/.venvs/myenv/bin/activate
pip install -e '.[dev,research]'
python -m pytest
python scripts/prepare_qqq_hourly_v5_data.py audit-alpha-only
python scripts/prepare_qqq_hourly_v5_data.py prepare --help
python scripts/run_qqq_daily_hourly_v5.py prepare
```

The last command registers the new half-hour study and emits `data_not_ready`
if verified inputs are absent. It does not download or backtest. The earlier
hourly and Alpha-only source snapshots/registrations remain archived; their
sealed data/parameters/paper states cannot be silently reused for this cadence.

Historical and live packet manifests bind **sampling interval 30 minutes**.
The preparation CLI defaults to `--sampling-interval-minutes 30`. Generic
legacy data helpers can still read declared 60-minute packets, but the current
study rejects them. Historical input uses direct verified raw 30-minute bars,
not hour aggregation. Both intraday and independent `daily_source_metadata`
attestations must explicitly name Alpha Vantage, QQQ and raw-as-traded pricing;
missing/other-provider provenance is rejected. No Alpha provider is inferred.

```bash
python scripts/run_qqq_daily_hourly_v5.py prepare \
  --study-dir .cache/qqq_daily_halfhour_v5_alpha_only \
  --input-packet /absolute/path/to/verified_alpha_halfhour_packet \
  --cash-file /absolute/path/to/DTB3_available.csv
python scripts/run_qqq_daily_hourly_v5.py development
python scripts/run_qqq_daily_hourly_v5.py validation
python scripts/run_qqq_daily_hourly_v5.py historical-oos
```

The May 28 cutoff is user-approved and determined by source coverage, not returns.
May 29/June 1 tail data are not required. Internal gaps cannot be filled with
daily candles or silently skipped: 14 unresolved half-hour slots and two
August 22, 2013 halt-consistent slots remain. Other vendor rows during that halt
also require no-trade/fill classification. A diagnostic raw-price restoration
has unreconciled last-intraday versus official-daily quotes under the predefined
relative `0.001`/absolute `0.02` check. This is not proof of corrupt prices or a
requirement to buy a new dataset. Unverified conversions are not labeled raw;
**no market backtest is run until these checks pass**.

Prepared stage packets contain only the permitted chronological prefix, with
full daily seed history. Prior selection seals are checked before held-out
numerical rows are loaded. Changed sources/code/configuration fail closed and
require a new study directory; failed attempts are retained. Original caches,
v4 artifacts, existing monitors, prospective studies and resumes are unchanged.

## Read only half hour monitoring

Prepare a distinct live packet from completed prior daily history, completed
Alpha Vantage raw half-hours and current corporate actions known by session open.
An unfinished current 30-minute bar is not evidence. At 10:00, the completed
09:30–10:00 bar is usable immediately without waiting until 10:30.

```bash
python scripts/prepare_qqq_hourly_v5_data.py prepare-monitor \
  --raw-30min /absolute/path/to/raw_alpha_subbars.csv \
  --metadata /absolute/path/to/verified_alpha_metadata.json \
  --daily /absolute/path/to/completed_prior_daily_history.csv \
  --bar-start 2026-10-07 --as-of "$AS_OF" \
  --sampling-interval-minutes 30 \
  --output /absolute/path/to/new_verified_monitor_packet
python scripts/run_qqq_daily_hourly_v5.py monitor --live \
  --input-packet /absolute/path/to/verified_monitor_packet \
  --state /absolute/path/to/paper_state.json --as-of "$AS_OF"
```

Use a current timezone-aware `AS_OF`, not a backdated time for data obtained later.
Monitoring requires the sealed winner and explicit versioned half-hour paper
state. No actual holding is assumed if state is absent: only conditions are
reported. Required state binds the winner ID, sampling interval, half-hour-check
counter unit, integer position, streak counters, pending target and canonical
last processed bar end. Old hour-only states and inconsistent pending counts
are rejected. The returned next state is printed only, not saved or submitted
as brokerage orders. Stale, unfinished, missing or unverified data suppresses
new actions. No recurring jobs or notifications are created.
