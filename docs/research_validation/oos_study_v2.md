# Prospective QQQ out of sample study

The study asks whether a fixed QQQ leverage-rotation rule adds value on genuinely
future prices after costs. It locks the code, position sizing and reporting dates
before the market period begins. It is a simulation, not a live trade record or a
claim that the strategy works.

**Frozen October 5, 2026 at 22:35:46 New York time.** The holdout starts October
6, 2026 at 09:30. Its preliminary and principal checkpoints are October 5, 2027
and October 5, 2028 at the scheduled 16:00 close. The [public freeze receipt](../../reports/research_validation/20261005_oos_v2_freeze_receipt.json)
records the private manifest digest and exact source revision. All 284 software
tests pass; independent freeze verification passed. **OOS observations are zero
at initialization and returns are unavailable**, not a reported zero-percent
return. The previous missed-start manual-paper pilot is preserved separately and
is never backfilled.

## Locked comparison

- Fixed B5 rule with seven completed observations on full sessions and four on
  early closes. Parameter windows count observations, not exact calendar days.
- Scale each nonzero target allocation using only the previous 252 complete
  sessions. Calibration is retrospective, not an OOS result. The prelaunch
  multiplier is **0.5949310662**; future volatility is not guaranteed to match QQQ.
- Compare against QQQ purchased at the fixed second completed holdout bar, whether
  or not B5 buys. Both start with $1,000 cash and no warmup holdings or profits.
- Primary costs are 1 bp fees and 5 bps adverse slippage; cash interest is zero.
  Actual TQQQ prices already include fund economics—no duplicate financing charge.
- Preliminary report after 12 months; principal report after 24 months. End at
  the last scheduled session close before the local anniversary. Never retune
  after a favorable or unfavorable preliminary result.

The primary endpoint is net total-return difference, scaled B5 minus QQQ, in
percentage points. Positive return difference with no worse observed drawdown is
only provisional support. An uncertainty interval spanning zero is inconclusive;
no outcome alone authorizes deployment.

## Data readiness and accounting

The prelaunch package covers October 7, 2024 through October 5, 2026 with **3,485
scheduled observations**, zero gaps, and 17 corporate actions. The original Yahoo
request missed 14 observations per symbol. Seven whole incomplete sessions were
replaced with existing licensed Alpha Vantage subbar closes—not interpolation.
Every selected 30-minute close is checked against preserved 15-minute evidence,
physical share prices are restored with matching daily factors, and after-close
rows are excluded. All raw inputs and exact transformation hashes stay private.
See the [source audit](../../reports/research_validation/20261005_oos_v2_source_audit.md).

Daily action-source coverage is also required: a missing split or distribution
row cannot silently disappear. Splits preserve quantities, basis and policy
references. Dividends create an ex-date receivable; spendable cash is released at
16:00 New York on the publisher's stated pay date, a declared conservative
simulation convention rather than verified broker availability. QQQ's 2026
reference dates are official estimated schedules, not confirmed receipts.
Unknown tax character remains provisional in hypothetical tax results.

This deliberately replaces the old manual prototype's 15-minute recording gate.
Later ingestion of genuinely future prices is allowed, but does not establish
contemporaneous signal availability or executable fills. Archive at least monthly
so provider retention does not erase early hourly data. No recurring job has been
created by this implementation.

## Run the pipeline

Use a Python environment with the dependencies in `pyproject.toml`. The local
implementation uses the requested virtual environment. Commands below run from
this checkout, not the original frozen working tree.

```bash
~/.venvs/myenv/bin/python scripts/run_qqq_oos_study.py --help
~/.venvs/myenv/bin/python -m pytest
~/.venvs/myenv/bin/python -m ruff check .
```

Acquire into a **new** package directory. A renamed or overwritten package is not
an immutable new vintage. Provide a separately reviewed future ex/pay reference
file when new distributions occur; never edit the frozen historical CSV. Each
new reference file must cover every distribution in the downloaded window, not
just the newest quarter.

```bash
~/.venvs/myenv/bin/python scripts/run_qqq_oos_study.py acquire \
  --output data/raw/research_oos/NEW_VINTAGE \
  --distribution-calendar /absolute/path/to/new_verified_ex_pay_dates.csv \
  --fallback-30min-dir /absolute/path/to/licensed_AV_30min_cache \
  --fallback-15min-dir /absolute/path/to/licensed_AV_15min_cache \
  --fallback-daily-dir /absolute/path/to/matching_AV_daily_cache
```

Fallback inputs may be omitted only when Yahoo has every required scheduled
observation. If a fallback is needed, preserve complete observed subbars and
source-specific timing evidence. No daily-price substitution, padding, day
removal or zero financing/risk-free imputation is allowed.

`prepare` accepts already acquired `QQQ_hourly.parquet`, `QQQ_daily.parquet`,
`TQQQ_hourly.parquet`, `TQQQ_daily.parquet` and original request/acquisition JSON
records. It uses the same normalization and accepts the same fallback flags.
Production readiness and scoring independently replay normalized prices/actions
from those raw inputs. Self-hashed CSVs alone do not qualify.

Before launch:

```bash
~/.venvs/myenv/bin/python scripts/run_qqq_oos_study.py check \
  --warmup data/raw/research_oos/20261005_v2_final_warmup
~/.venvs/myenv/bin/python scripts/run_qqq_oos_study.py freeze \
  --warmup data/raw/research_oos/20261005_v2_final_warmup \
  --output reports/research_validation/NEW_oos_freeze --attest-ready
```

A failed readiness check creates no study start. A final freeze must finish
outside an open session, with warmup ending at the latest completed scheduled
close. It copies exact inputs, binds dependencies and actual loaded module paths,
then records the first still-unstarted session. A clock crossing the market open
invalidates the incomplete attempt. Never resume or overwrite a partial freeze.

At a fully elapsed checkpoint:

```bash
~/.venvs/myenv/bin/python scripts/run_qqq_oos_study.py verify \
  --manifest /absolute/path/to/frozen/manifest.json
~/.venvs/myenv/bin/python scripts/run_qqq_oos_study.py score \
  --manifest /absolute/path/to/frozen/manifest.json \
  --future-package /absolute/path/to/first_future_package \
  --future-package /absolute/path/to/next_future_package \
  --months 12 --output reports/research_validation/NEW_oos_private
```

There is no CLI time override. Scoring early, missing a scheduled observation,
changing frozen code, altering calibration or using a foreign checkout fails.
Use the archived snapshot and recorded dependencies if the development checkout
later changes. Keep output directories outside immutable input/freeze directories.

The earliest source vintage wins for overlapping quotes; later changes are
retained and reported, not silently substituted. Late-added, withdrawn or revised
historical actions stop evaluation for explicit correction review. For a hybrid
vintage, timing includes the local AV-cache read at assembly; it is not a claim of
point-in-time retrieval of the old cache. Exceptional calendar changes require
review, not invented scheduled bars.

## Reports and limitations

The evaluator exports **36** primary/secondary scenario rows, **18** separate
matched-target attribution rows, daily paths and terminal clone summaries. It
includes unscaled B5, QQQ, TQQQ, cash and actual-QQQ 200-day trend comparisons;
zero-investor-cost paths still contain embedded fund economics. Higher slippage
and hypothetical 24% short-term/15% long-term tax scenarios never replace the
primary comparison. Matched-target attribution is not fixed-share-order
attribution: quantities and tax funding sales can differ.

Checkpoint liquidation uses a clone with exit fees and taxes; it does not reset
the continuing account. The paired bootstrap performs **5,000** resamples per
mean block length of 20 and 60 sessions, keeping both portfolios together and
reapplying boundary costs once. Its intervals are conditional estimates, not
future crash probabilities, independent market trials or the earlier 3,000
underlying-price-path stress experiment.

A separate synthetic financing API is implemented. Its predeclared spread grid
is 0/50/100/200 bps above DFF, with explicit hypothetical fund-drag assumptions.
It requires an independently causal QQQ total-return index and timestamp-available
DFF; no empirical financing output is fabricated before those inputs exist. The
model is not actual TQQQ, isolated measured financing cost or a substitute primary
strategy result. Treasury-yield cash sensitivities likewise require real causal
inputs; the primary cash assumption remains zero.

All public numbers here describe implementation, calibration or scheduled coverage.
They do not describe realized out-of-sample performance. Raw licensed prices,
private freezes, daily account paths and original source documents are excluded
from Git publication. Resume wording must not claim completed OOS evidence yet.

## Local initialization record

The completed private freeze is `reports/research_validation/20261005_v2_oos_freeze/manifest.json`.
It and the licensed warmup are intentionally ignored by Git. The public receipt
is not a substitute manifest and cannot initialize or score a replica without
the exact private inputs. A fresh download requires its own readiness checks and
future freeze; do not fabricate the original digest.
