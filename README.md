# Trend Following Study

**Repository:** [Agul-quant/trend-following](https://github.com/Agul-quant/trend-following).
The **Trend Following Study** tests **56,644 configurations** of daily indicators
and half-hour confirmation on unlevered QQQ/cash. This tree contains only the
current study, its dependencies and software tests. Git history is preserved.

## Current status: intersection validation complete; historical OOS closed

**Latest resplit:** validation is now **2012–2018**; future historical OOS is
**2019–May 28, 2026**, the verified Alpha endpoint. The same frozen 159 pairs at
all three shared waits were reevaluated with unchanged 3% nominal cash/costs.
At 2/3/4 hours, **9/12/7 of 159** beat revised validation BH cash-excess Sharpe
**0.85621**; no candidate was removed, replaced or promoted. OOS was not run.
This split change is after prior history examination, not untouched confirmation.
[Revised validation report](reports/resplit_validation/20261008/README.md).

**Preserved earlier follow-up: 2012–2015 shared-pair validation.** The 159 indicator
pairs beating development BH Sharpe at **all three 2/3/4-hour waits** were sealed
before validation access, then evaluated at every retained wait in 2012–2015.
All **477 validation instances** remain; none beats the fresh cost-matched BH
cash-excess Sharpe **1.09027**. No winner, replacement or global wait was selected.
**Historical OOS remains closed.** [Intersection selection and validation report](reports/intersection_validation/20261008/README.md).

**Cash assumption for all new/restarted studies:** fixed **3% nominal annual
interest**, accrued ACT/365 and compounded on the existing event clock, replaces
the variable DTB3 proxy. Cash-only effective CAGR is slightly above 3%, not exactly
3% APY; the optional convention choice remains open. Old sealed results below
retain their original cash model and must not be relabeled as 3%-cash results.

The [new development beat-BH counts](reports/constant_cash_wait_counts/20261008/README.md)
recompute all 289 indicator pairs at each equal entry/exit wait under this cash
assumption. These 13 timing diagnostics do not choose the global production wait
or authorize validation/OOS. [Shared future assumptions](configs/shared_research_assumptions.yaml).

The earlier **45-option development-only hold** remains a preserved record, not
the selection used for this intersection follow-up. Validation authorization
covered only the sealed 159-pair comparison above; other validation work and
historical evaluation require new explicit user instructions. The old 45-option
criterion was baseline net cash-excess Sharpe with five distinct indicator pairs
per family cell and confirmation waits chosen using development only.

Earlier evaluations shown below already occurred and are retained as historical
records; this hold cannot make their periods genuinely untouched again. See the
[development-only shortlist and hold](reports/development_shortlist_hold/20261008/README.md).
The hold is operational, not a newly installed runtime interlock.

**Timing policy for new/restarted studies:** one globally fixed wait, equal for
entry and exit and shared by every configuration. Retain **2, 3 and 4 trading
hours** as development choices; one final global value is still awaiting the
user's choice. Do not tune waits independently or treat this as a new executed
study. Previously sealed configurations remain unchanged.

[Retained 2/3/4-hour comparison](reports/retained_wait_choices/20261008/README.md):
289 indicator pairs at each choice, 867 total development configurations. No
new market simulation, shortlist selection or later-stage access was performed.

The [equal-wait development diagnostic](reports/equal_wait_sensitivity/20261008/README.md)
shows Sharpe dependence for the same 45 indicator pairs and all 289 pairs at
common waits of 0–6 hours. It reuses saved development results only; no later
period was opened and no new optimal wait was selected.

## Diversified shortlist follow-up: keep 45 options, not one winner

The follow-up selects **five distinct indicator pairs per entry/exit family
cell from development only**, retaining their development-selected confirmation
waits. The same **45 options** entered validation and all four historical cost
scenarios without eliminating or replacing any member.

At 1 bp commission + 3 bps adverse slippage per order:

| Comparison across the fixed set | Validation 2012–2015 | Historical 2016–May 28, 2026 |
|---|---:|---:|
| Median candidate CAGR | 5.73% | 14.32% |
| QQQ buy-and-hold CAGR | 19.88% | 20.96% |
| Candidates with higher net CAGR than QQQ | 0/45 | 0/45 |
| Candidates with higher cash-excess Sharpe than QQQ | 0/45 | 22/45 |
| Candidates with shallower daily drawdown than QQQ | 13/45 | 45/45 |

**This is exploratory retrospective research after the earlier phase outputs
were examined, not untouched OOS.** The median describes separate candidates,
not an ensemble portfolio. The original single-winner results and monitor are
unchanged. [All 45 options, results and costs](reports/shortlist/20261008/README.md)
· [Follow-up design](docs/research_validation/shortlist_followup.md).

## Current development results (2001–2011)

All **56,644** configurations have now been backtested once after **1 bp commission
+ 3 bps slippage per order**. **47,526** beat QQQ buy-and-hold on net CAGR;
**35,529** beat its cash-excess Sharpe and all three comparison metrics together.

| Portfolio | Net CAGR | Maximum drawdown loss | Cash-excess Sharpe |
|---|---:|---:|---:|
| Development primary-score leader | **8.86%** | **14.17%** | **0.583** |
| QQQ buy-and-hold | **−0.04%** | **70.40%** | **0.078** |
| Causal cash proxy | 2.00% | 0.00% | — |

These are **in-sample, development-selected results**, not proof of a strategy
that works out of sample. The nine sealed family finalists subsequently entered
validation; its single sealed winner then entered historical OOS. Gap flags do
not filter or rank any candidate.

[Full current report, every configuration and post-gap flags](reports/development/20261008/README.md).

## Completed validation and historical comparison

Validation selected **daily EMA200 entry with a 5.5-hour confirmation wait and
daily EMA100 exit with a 2.5-hour wait**. Waits are trading hours after the first
qualifying half-hour hit. Only **nine** finalists were evaluated in validation;
only this **one locked winner** was evaluated in historical OOS, unchanged across
all four slippage scenarios.

Baseline results include **1 bp commission + 3 bps adverse slippage per order**:

| Segment / portfolio | Net CAGR | Maximum drawdown loss | Cash-excess Sharpe |
|---|---:|---:|---:|
| Validation 2012–2015: selected winner | 11.43% | 13.96% | 0.930 |
| Validation: QQQ buy-and-hold | 19.88% | 13.94% | 1.288 |
| Historical 2016–May 28, 2026: locked winner | 13.81% | 23.08% | 0.795 |
| Historical: QQQ buy-and-hold | 20.96% | 35.12% | 0.868 |

The winner had lower historical drawdown and volatility, but **lower CAGR and
cash-excess Sharpe than buy-and-hold in both segments**. Lower risk does not
demonstrate overall superiority. Validation is selection data; historical OOS
is **retrospective, previously examined history**, not untouched or live evidence.

[Validation ranking and comparison](reports/validation/20261008/README.md) ·
[Historical comparison and all four costs](reports/historical_oos/20261008/README.md).

## Frozen research contract

- **Daily indicators:** SMA and EMA periods 10, 20, 50, 100, 200, 250 trading
  sessions; five standalone EMA-MACD tuples `(6,13,5)`, `(12,26,9)`, `(24,52,18)`,
  `(48,104,36)`, `(96,208,72)`.
- **Independent entry/exit:** 17 indicator rules × 14 confirmation waits per
  side. Waits are 0, 0.5, …, 6.5 trading hours **after the first qualifying hit**,
  requiring `1 + 2 * hours` completed half-hour checks. All 56,644 pairs are
  included; zero confirms at the first qualifying close, not an unfinished Open.
- **Execution:** next safe observed half-hour Open; no leverage or shorting.
  Baseline costs are **1 bp commission + 3 bps adverse slippage per order**.
  Stress slippage is 1/3/10/25 bps; modeled taxes, financing and extra expenses
  are zero.
- **Prices:** Alpha Vantage only, ending May 28, 2026. Intraday prices are
  reconstructed as-traded units with independent source proof, **not native raw
  capture or guaranteed point-in-time data**. Provider daily closes are assumed
  available at 17:00 New York for valuation and next-session daily features;
  they are not guaranteed Nasdaq Official Closing Prices or execution quotes.
- **Missing observations:** 14 source half-hours remain absent. Hold the last
  filled QQQ/cash position, cancel pending orders, reset confirmations and make
  no missing-interval trades or fabricated quotes. Documented regulatory halts
  also suppress unsafe signals/fills.
- **Descriptive gap flags:** retain every candidate's confirmed buy/sell signals
  after each source gap through the **next trading session's close**. Flags never
  filter candidates, adjust scores, rank, or replace selections.
- **Chronology:** daily initialization 1999–2000; development 2001–2011;
  validation 2012–2015; historical OOS 2016–May 28, 2026. Historical OOS is
  retrospective, not untouched or live evidence. Each scored segment starts
  with cash and reset confirmation/order state.

See the [current operating contract](docs/research_validation/gap_hold_flags_v5.md)
for source roles, blackout handling, diagnostics and staged release gates.

## Setup and public entry point

```bash
python -m pip install -e '.[dev,research]'
python -m pytest
python scripts/run_qqq_research.py --help
```

Licensed market-price inputs, credentials, human authorization receipts and
prepared freezes remain private and are **not included**. Synthetic software
tests require no market-price inputs. Market preparation fails closed when
verified local inputs or required authorization are missing.

Publication checks passed **641 synthetic software tests**, Ruff lint and
whitespace checks. These checks are separate from market performance; the
eight frozen implementation files and immutable configuration remain unchanged.

```bash
python scripts/run_qqq_research.py prepare
# Stage commands require private inputs, authorization and intact frozen seals.
# Development, validation and historical OOS are already completed for this freeze.
python scripts/run_qqq_research.py development
python scripts/run_qqq_research.py validation
python scripts/run_qqq_research.py historical-oos
# Separately sealed, user-authorized exploratory fixed-set comparison:
python scripts/run_qqq_research.py shortlist --help
```

Do not change parameters and present a rerun as the same locked evaluation.
Future market runs still require the appropriate authorization and stage gates.

The active immutable configuration is
`configs/qqq_daily_halfhour_v5_gap_flags_v2.yaml`. The public command and study
are unversioned; internal technical filenames and schema IDs remain unchanged
because they are bound in the prepared source freeze.
