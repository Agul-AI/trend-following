# QQQ leverage-rotation research

A Python research project on QQQ and synthetic leveraged exposure. The current
project is a **fresh chronological rebuild**, not a continuation of the earlier
full-sample-selected performance claim. Its strength is an inspectable research
process, cost modeling and honest risk diagnosis—not a claim that the strategy
works.

## What I rebuilt

- Specified a new simple moving-average/allocation family and a fixed split:
  **2002–2011 development, 2012–2015 parameter selection, 2016–June 1, 2026
  final evaluation**. Previous B5 rules, optimized parameters, synthetic caches
  and performance tables are not inputs.
- Selected among **nine** settings using only the selection period, then
  evaluated that choice on **2,617 subsequent daily sessions**, without refitting
  or choosing a different winner after seeing test results.
- Used raw daily QQQ closes, distributions and splits to rebuild causal returns;
  matched all **6,142** selected session dates against an independent source.
  Indicators count actual daily observations, not approximated hourly windows.
- Tracked cash, holdings, delayed execution, dividend reinvestment, funding,
  fund-drag assumptions and hypothetical taxes; compared five cost layers and
  predefined controls.
- Reconciled results with an independent arithmetic implementation and reported
  conditional paired-return uncertainty rather than claiming statistical proof.

The historical years were previously researched. They are held out from **this
parameter-selection procedure**, not newly untouched research data. Synthetic
3× QQQ is not actual TQQQ execution or live performance.

## What the final period showed

January 4, 2016–June 1, 2026, with the same $1,000 starting cash and uniform
hypothetical terminal liquidation:

| Expense-inclusive pretax result | CAGR | Maximum drawdown | Annualized volatility | Cash-excess Sharpe |
|---|---:|---:|---:|---:|
| Validation-selected synthetic-3× rule | 38.95% | 59.44% | 49.67% | 0.870 |
| QQQ buy-and-hold with reinvestment | 21.12% | 35.12% | 22.18% | 0.872 |
| QQQ 200-session trend control | 17.10% | 23.20% | 16.56% | 0.898 |

The selected rule's CAGR fell from **47.77% before added costs to 38.95% after
trading, financing and fund assumptions**, and **33.85% under hypothetical taxes**.
After-tax maximum drawdown was **63.05%**. Gross-to-after-tax annualized return
declined **13.92 percentage points**.

Higher return came with much greater leverage and risk, while cash-excess Sharpe
was slightly below QQQ's. There is no risk-matched comparison or demonstrated
timing alpha. Both prespecified conditional bootstrap intervals cross zero.
This is useful evidence about research assumptions and downside risk, **not proof
of a profitable edge**.

## Evidence for a recruiter or reviewer

- [New results, exact test dates, costs and limitations](reports/research_validation/20261006_split_v3/README.md)
- [Protocol fixed before this run's selection](configs/qqq_split_v3.yaml)
- [Raw daily signal and accounting implementation](src/trend_following/research_split_v3.py)
- [Separate freeze, selection and evaluation stages](scripts/run_qqq_split_study.py)
- [All nine validation configurations](reports/research_validation/20261006_split_v3/validation_candidates.csv)
- [Independent result reconciliation](reports/research_validation/20261006_split_v3/independent_accounting_audit.md)
- [Independent arithmetic audit CLI](scripts/audit_qqq_split_v3_result.py)
- [Run receipt with source/config/data hashes](reports/research_validation/20261006_split_v3/run_receipt.json)

The source implementation was committed and published before new parameter
selection. Local receipt times/hashes improve reproducibility; they are not
independent notarization. Raw licensed prices, private portfolio paths, personal
resumes and the executable private freeze are not published.

## Run the software checks

```bash
python -m pytest
python -m ruff check .
python scripts/run_qqq_split_study.py --help
```

**327 tests pass**, including **43 new split-specific tests**. These are software
checks on artificial inputs, not market trials. Use an environment matching the
receipt to reproduce an already sealed numerical run.

With licensed inputs, create a new study:

```bash
python scripts/run_qqq_split_study.py freeze \
  --qqq-daily /path/to/raw_qqq_daily_with_actions.parquet \
  --reference-daily /path/to/independent_qqq_daily_dates.parquet \
  --rate-dir /path/to/raw_and_prepared_DFF_DTB3 \
  --output /path/to/NEW_split_private
python scripts/run_qqq_split_study.py select --study /path/to/NEW_split_private
python scripts/run_qqq_split_study.py evaluate --study /path/to/NEW_split_private
```

The commands refuse to overwrite an earlier attempt or accept altered sealed
code, inputs, rate transformations or selections. A code-only checkout does not
contain the licensed data needed to recreate the published market experiment.

The separate arithmetic checker is scoped to the exact published private run,
not arbitrary new experiments:

```bash
python scripts/audit_qqq_split_v3_result.py \
  --study /path/to/published_private_study --output /path/to/NEW_audit_output
```

## Prior work is preserved, not used to sell the rebuild

- [Previous project overview and numbers](ARCHIVED_RESEARCH_README.md)
- [Earlier research tables and assumptions](reports/research_validation/README_pre_split_v3_20261006.md)
- [Original generic ETF framework](LEGACY_README.md)

The **separate prospective OOS study remains frozen and unchanged**. Its future
observations and 12/24-month reports must not be mixed with this retrospective
rebuild. [Prospective study operating guide](docs/research_validation/oos_study_v2.md).
Neither workflow places real orders or creates an unattended data collector.
