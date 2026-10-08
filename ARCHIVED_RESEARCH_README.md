# QQQ leverage-rotation research

A Python research project on QQQ and simulated leveraged exposure. The goal is
to test research assumptions and diagnose costs and risk—not to advertise a
profitable trading strategy.

## What I built

- Selected trading rules using earlier data, then evaluated them across **six
  later historical periods**. Selection never uses the next test period's data.
- Tracked trades, holdings, cash and hypothetical taxes in **one continuous
  portfolio**, including costs when the selected rule changed.
- Compared results before and after trading costs, financing, fund expenses and
  hypothetical taxes, against simple QQQ benchmarks.
- Rebuilt prices, signals and portfolios for **3,000 resampled market scenarios**,
  rather than just resampling the strategy's saved returns.

## What I learned

Costs and taxes can materially weaken attractive historical results. On the
January 2002–June 2026 reconstruction, modeled annualized return was **37.52% gross,
30.53% after trading/financing/fund drag and 25.78% under hypothetical taxes**.
These scenarios can also change trade decisions; their differences are not pure
fixed-trade cost attribution.

Leveraged exposure also retained substantial downside risk. Across three
resampling specifications, median worst peak-to-trough losses were **65%–75%**
over roughly 24-year simulated histories. These are conditional stress results,
**not annual loss probabilities or forecasts**.

All previously explored history remains retrospective—even with selection and
testing separated. Neither a profitable edge nor independent unseen performance
has been established. A manual paper-testing module is provided, but its
initialization is not a performance record; unsupported actions or missing/stale
observations stop that version.

## Evidence for a recruiter or reviewer

| Evidence | Period | Main risk result |
| --- | --- | --- |
| Fixed-rule cost reconstruction | Jan 10, 2002–Jun 1, 2026 | Maximum drawdown 38.39% after trading/financing/fund costs; 41.70% with hypothetical taxes |
| Continuous selected-rule evaluation | Jan 2, 2009–Jun 1, 2026 | Maximum drawdown 41.77% before hypothetical taxes; six chronological periods, final period partial |
| Underlying-path stress diagnostic | Approximately 24-year simulated histories | 1,000 paths per mean block length of 60/126/252 sessions; median drawdowns 74.86%/70.07%/65.14% |

The original resume's 41.16% drawdown is a **different legacy accounting version**;
it is not silently replaced or treated as equivalent. The public tables contain
aggregate research statistics, not real brokerage returns.

- [Published results and assumptions](reports/research_validation/README.md)
- [Research workflow and reproduction boundaries](docs/research_validation/README.md)
- [Continuous portfolio and tax accounting](src/trend_following/research_ledger.py)
- [Historical selection and evaluation](scripts/run_qqq_research_validation.py)
- [Full-path stress tests](scripts/run_qqq_research_bootstrap.py)
- [Manual paper-testing limitations](docs/research_validation/forward_paper_contract.md)
- [New prospective OOS study and operating guide](docs/research_validation/oos_study_v2.md)

## Prospective study

A separate frozen-rule OOS pipeline is implemented. It sizes positions using only
the previous 252 complete sessions, then locks the rules and costs before a
future period begins. Twelve months is preliminary and 24 months is the principal
comparison against QQQ. The actual freeze completed October 5, 2026, before the October 6 market-open
holdout. Its initialization is not an OOS performance result.
See the [study specification](docs/research_validation/out_of_sample_study_design.md)
and [operating guide](docs/research_validation/oos_study_v2.md) for source checks,
actual freeze status and interpretation. No real trades or recurring collector
are created by the code.

## Run the software checks

In a Python virtual environment on macOS or Linux:

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
```

The initial scoped publication had **193 passing tests**. The OOS extension brings
the public suite to **284 passing tests**, adding source, timing, accounting and
evaluation checks using artificial inputs;
no private price cache or network request is needed for those checks. The broader
original local project's 313 tests are a different suite, not additional market
trials. Source
code/test reproduction is not the same as rerunning the market-data results.
The manual recorder currently uses POSIX file locking; Windows is not claimed
as supported.

Raw licensed price data, private freezes/warmup, personal resumes, account records
and notification helpers are deliberately omitted. To rerun a market experiment,
obtain appropriately licensed inputs and create a **new** protocol/data/code
freeze. Published code is not the original private freeze, and old paper events
must never be backdated into a new experiment.

The original generic ETF framework and its commands are retained in
[the archived overview](LEGACY_README.md). New result tables above are the
October 2026 research implementation, not a deployment recommendation.
