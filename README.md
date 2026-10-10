# Trend Following Study

**Repository:** [Agul-quant/trend-following](https://github.com/Agul-quant/trend-following).
The **Trend Following Study** tests **56,644 configurations** of daily indicators
and half-hour confirmation on unlevered QQQ/cash. The tree also contains the
separate current mean-reversion, fixed-capital portfolio, multi-scale and
core/tactical adaptive-allocation studies, shared dependencies and software tests.
Git history is preserved; archived studies are not reintroduced.

## Trend core, tactical mean reversion and adaptive allocation

This separate study retains the exact **12 trend and 32 MR rules** and all
**384 predefined pairings**, equally weighted initially, without pair selection or indicator changes.
It compares fixed-initial-capital core/tactical sleeves, signal blending with
actual partial-position trades, and a fixed causal ER20 regime selector.
Core/blend allocations are **50/70/90% trend**, with **pure trend eligible**.
The paired unchanged trend state gates MR; a trend exit forces tactical exits
at the same next safe open. Tactical recovery sales never sell a funded core.

Fixed-capital portfolios fund twelve trend cores and 384 tactical MR sleeves;
these sleeves drift without rebalancing. Blended accounts restore their target
only on confirmed model-state changes, paying costs on actual partial trades.
Regime switching uses finalized daily ER20 (MR at <=0.25, trend at >=0.35),
seven half-hour confirmations and warm incoming shadow states. The selector
starts in trend mode each segment; it is a frozen research assumption, not a
proven market classifier. All indicator lookbacks remain daily sessions.

Development is **2001–2011**, followed by separately sealed **2012–2018
retrospective validation**. This remains hindsight-conditioned because the
trend members were identified using that validation period. **2019 onward is
closed**, and the original sources/results, monitors and resumes are preserved.
Cash earns 3% nominal ACT/365 with event-clock compounding, not exactly 3% APY.
Execution scenarios use 1-bp commission plus 1/3/10/25-bp hypothetical slippage;
3 bps is the selection baseline. There is no cross-account order netting.

```bash
~/.venvs/myenv/bin/python scripts/run_hybrid_allocation_research.py check
~/.venvs/myenv/bin/python scripts/run_hybrid_allocation_research.py freeze --authorization PRIVATE_AUTH.json
~/.venvs/myenv/bin/python scripts/run_hybrid_allocation_research.py development
~/.venvs/myenv/bin/python scripts/run_hybrid_allocation_research.py validation
~/.venvs/myenv/bin/python scripts/run_hybrid_allocation_research.py report
```

The CLI exposes no OOS route and refuses overwriting sealed stages. Prices,
private freezes, approval receipts and detailed execution ledgers stay private.
Public comparison results appear in
[the separate derived report](reports/hybrid_allocation_research/20261010/README.md).

Development locked 50/50 for both new core/tactical and blending constructions.
At baseline costs in retrospective validation, blended Sharpe was 0.908 versus
0.898 for pure trend, but CAGR fell to 10.75% from 15.08%; the small Sharpe
difference was not supported by the paired bootstrap intervals and reversed
under higher slippage. The fixed ER regime route scored 0.685. See the report
for exposure, drawdown, trading, all allocations and cost scenarios.

## Long-term trend + short-term mean-reversion comparison

The separate multi-scale comparison retains all **12 trend and 32 mean-reversion
members**. It reuses the independent fixed-capital portfolio seals unchanged,
and tests **128 filtered rules in four equal-initial-capital MR baskets**:
SMA100/SMA200, each with recovery-only or recovery-or-confirmed-trend-break exits.
Finalized daily gates become available at assumed **17:00 New York**; intraday
checks never update the gate. The shared wait is three trading hours/seven
completed half-hour checks. There is no rebalancing or order netting.

Development (2001–2011) locked **SMA100 with recovery-only exits** and
**SMA200 with recovery-or-trend-break exits** before validation. At 1-bp commission
plus 3-bps slippage, their development cash-excess Sharpes were **0.3635** and
**0.4340**, respectively, versus **0.1954** for the unchanged ungated MR basket.
The old independent experiment's locked allocation remains **0% trend / 100% MR**;
its validation controls are not new replacement candidates.

Validation used the two locked baskets and the other two predefined filtered
baskets as controls, without reselection. Baseline costs remain 1-bp one-way
commission plus 3-bps adverse slippage, with fixed 3% nominal ACT/365 cash.

| Validation 2012–2018 | Cash-excess Sharpe | Net CAGR | Daily max drawdown | Exposure |
|---|---:|---:|---:|---:|
| Locked independent / ungated MR | 0.5992 | 6.46% | -7.55% | 12.93% |
| SMA100 gate; recovery only (locked) | 0.4962 | 5.03% | -4.90% | 9.36% |
| SMA200 gate; recovery or trend break (locked) | 0.5680 | 5.22% | -4.34% | 9.43% |
| 50/50 independent sleeves (control) | 0.8994 | 11.27% | -8.29% | 55.97% |
| QQQ buy-and-hold | 0.8562 | 16.59% | -22.79% | 100.00% |

**Finding:** neither locked filtered basket improved validation Sharpe or growth
against the ungated MR basket or QQQ buy-and-hold. Their shallower drawdowns
come with lower exposure and growth. The 50/50 sleeve control has higher
validation Sharpe than BH but lower growth; it was **not** the development-selected
allocation and must not be promoted using this validation comparison. All twelve
paired bootstrap Sharpe-difference intervals include zero. At 25-bps slippage,
both locked filtered baskets have negative cash-excess Sharpe and CAGR below cash.

[Full multi-scale comparison, all 128 constituents, cost scenarios, diagnostics
and bootstrap intervals](reports/multi_scale_research/20261009/README.md).

**Retrospective and hindsight-conditioned:** the twelve trend members were
identified using validation performance. Neither development fitting nor paired
bootstrap intervals restore independent confirmation. Some trend entries are
short-horizon; three MR members use fifty days, so the retained universe is not
strictly separated into long and short horizons. **2019 onward remains closed.**

The separate CLI exposes `check`, `freeze`, `development`, `validation` and
`report`, but no OOS route. It requires intact private input seals and matching
explicit authorization; licensed prices and private receipts are not published.

```bash
~/.venvs/myenv/bin/python scripts/run_multi_scale_research.py check
~/.venvs/myenv/bin/python scripts/run_multi_scale_research.py freeze --authorization PRIVATE_AUTH.json
~/.venvs/myenv/bin/python scripts/run_multi_scale_research.py development
~/.venvs/myenv/bin/python scripts/run_multi_scale_research.py validation
~/.venvs/myenv/bin/python scripts/run_multi_scale_research.py report
```

The reviewed wealth/exposure chart layout is reproducible separately:

```bash
~/.venvs/myenv/bin/python scripts/plot_multi_scale_report.py
```

This presentation-only helper reads sealed derived arrays, fixes labels/layout,
and records exact plot-data digests; it cannot backtest, fit parameters or open
price packets. The frozen performance/report engines stay unchanged.

Sealed stages and published reports cannot be overwritten or rerun. The first
technical attempt stopped on a legacy archive-schema mismatch before any new
filtered performance result; its private record was preserved, the read-only
adapter was corrected without changing strategy rules, and a fresh freeze was
used. Existing study sources/results, monitors and resumes remain unchanged.

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

**Timing policy for new/restarted studies:** the user fixed **3 trading hours**,
equal for entry and exit and shared by every configuration: seven consecutive
half-hour checks. The earlier **2/3/4-hour** comparison remains a frozen record,
not independently tunable waits for new studies. Previously sealed
configurations remain unchanged.

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

### Separate mean-reversion development study

The additive **Mean-Reversion Study** tests 90 QQQ/cash configurations on
2001–2011 only: SMA/EMA percentage distance, sample-standard-deviation z-score,
and Wilder RSI. It uses a shared 3-hour entry/exit confirmation, fixed 3%
nominal cash interest and cost-matched QQQ buy-and-hold. Persistent oversold
entry is not confirmation of a rebound; exits require indicator recovery.
There are no stop-loss, holding-limit or trade-frequency filters.

```bash
python scripts/run_mean_reversion_research.py --help
python scripts/run_mean_reversion_research.py check
# Freeze requires private inputs and the matching human development authorization.
python scripts/run_mean_reversion_research.py freeze --authorization /path/to/private-receipt.json
python scripts/run_mean_reversion_research.py development
python scripts/run_mean_reversion_research.py report
```

[Development results and full assumptions](reports/mean_reversion_development/20261009/README.md).
The separate command offers no validation or OOS execution route. Existing trend
sources, frozen results and monitors remain unchanged. All future new/restarted
studies use a fixed shared 3-hour wait; old freezes are not rewritten.

### Separate mean-reversion validation

The user authorized **all 32 baseline development cases beating cost-matched
QQQ buy-and-hold Sharpe**, not the separate 20-finalist list, for **2012–2018**
validation. Membership is sealed before reading a physically bounded
1999–2018 input packet; a new data/code/accounting freeze precedes execution.
Every candidate keeps its daily parameters, shared 3-hour wait and cash/cost
assumptions. There is no retuning, replacement or validation-winner selection.
Historical OOS **2019–May 28, 2026 remains closed**.

Completed validation: **1/32** beats buy-and-hold cash-excess Sharpe at baseline
1+3-bp costs; **0/32** beats its CAGR. The only Sharpe beater is 10-day z-score,
entry ≤−2 and exit ≥0: **0.87435 Sharpe, 8.15% CAGR, 8.83% daily drawdown loss**,
versus BH **0.85621, 16.59%, 22.79%**. It completed **29 round trips (4.14/year)**;
no case beats matched BH Sharpe at 10/25-bp slippage.
[All 32 validation results and four costs](reports/mean_reversion_validation/20261009/README.md).

```bash
python scripts/run_mean_reversion_validation.py check
python scripts/run_mean_reversion_validation.py seal-selection --authorization /path/to/private-receipt.json
python scripts/run_mean_reversion_validation.py freeze
python scripts/run_mean_reversion_validation.py validation
python scripts/run_mean_reversion_validation.py report
```

This separate command exposes no historical OOS route. The original development
command and all frozen trend and mean-reversion development sources/results
remain unchanged. Previously examined history is retrospective research, not
untouched confirmation.

### QQQ strategy portfolio study

The separate portfolio experiment assigns fixed initial capital to the same
**12 three-hour trend validation Sharpe beaters and 32 mean-reversion development
Sharpe beaters**. It tests trend allocations of 0–100% in 10% steps plus equal
capital across all 44 candidates. Weights within each style are initially equal;
cash stays in its sleeve and capital weights drift. No portfolio rebalancing,
order netting, leverage, shorts or constituent retuning is assumed.

Select one allocation using **2001–2011 baseline net cash-excess Sharpe**, seal
it, then test it and predefined controls on **2012–2018** at all four costs.
The universe is already hindsight-conditioned by the twelve trend members'
validation selection; this is retrospective research, not independent
confirmation. **2019 onward remains closed**.

```bash
python scripts/run_portfolio_research.py check
python scripts/run_portfolio_research.py freeze --authorization /path/to/private-receipt.json
python scripts/run_portfolio_research.py development
python scripts/run_portfolio_research.py validation
python scripts/run_portfolio_research.py report
```

Completed: development selected **0% trend / 100% mean reversion** (Sharpe
**0.1954**). Its validation Sharpe was **0.5992**, below BH **0.8562**.
The predefined initial **50/50 control** achieved **0.8994 Sharpe, 11.27% CAGR
and 8.29% daily drawdown loss**, versus QQQ **0.8562, 16.59%, 22.79%**;
it was **not substituted for the locked allocation**.
[Full portfolio results, chart, four costs and bootstrap evidence](reports/portfolio_research/20261009/README.md).

The portfolio command exposes no OOS route. All earlier frozen studies, monitors
and resumes remain unchanged. It combines **wealth**, not standalone Sharpe
ratios or fixed-weight daily returns; gross sleeve orders are not net portfolio
round trips.

### Frozen trend-following entry point

```bash
python -m pip install -e '.[dev,research]'
python -m pytest
python scripts/run_qqq_research.py --help
```

Licensed market-price inputs, credentials, human authorization receipts and
prepared freezes remain private and are **not included**. Synthetic software
tests require no market-price inputs. Market preparation fails closed when
verified local inputs or required authorization are missing.

Publication checks passed **1039 synthetic software tests**, Ruff lint and
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
