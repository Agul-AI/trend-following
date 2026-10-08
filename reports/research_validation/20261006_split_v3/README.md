# A fresh chronological study shows higher returns with much larger losses

## Executive summary

- A **new daily study** was specified and its source code published before this
  run's parameter selection. The earlier B5 rules, synthetic caches, selected
  parameters and performance tables were not inputs. The fixed split is
  **2002–2011 development, 2012–2015 selection, and 2016–June 1, 2026 evaluation**.
- Selection among **nine** moving-average/allocation settings chose a 200-session
  trend rule with full allocation to modeled daily-reset 3× QQQ when the signal
  is positive. The subsequent **2,617-session** evaluation produced **38.95% CAGR
  after modeled trading, financing and fund costs**, but **59.44% maximum
  drawdown**, versus QQQ's **21.12% CAGR and 35.12% drawdown**.
- Higher return did **not** establish a risk-adjusted advantage: daily cash-excess
  Sharpe was **0.870 versus 0.872** for QQQ, and annualized volatility was
  **49.67% versus 22.18%**. Both conditional paired-bootstrap intervals cross zero.
- **The historical years were previously researched.** They are excluded from
  this selection procedure, not a newly untouched research holdout. This is a
  retrospective simulation, not actual TQQQ fills, live returns or evidence that
  the strategy should be deployed.

## Split and selection

| Stage | Actual source dates | Sessions | Role |
|---|---|---:|---|
| Development | Jan 2, 2002–Dec 30, 2011 | 2,519 | Source/method diagnostics; no changes to the sealed grid after results |
| Parameter selection | Jan 3, 2012–Dec 31, 2015 | 1,006 | Choose one setting using expense-inclusive pretax daily cash-excess Sharpe |
| Final evaluation | Jan 4, 2016–Jun 1, 2026 | 2,617 | Evaluate that same setting and predefined controls, without refitting |

The candidate grid is **50/100/200 completed-session simple moving averages ×
1/3, 2/3 or full allocation** to the simulated 3× instrument. The signal uses a
causal QQQ total-return index reconstructed from raw closes, distributions and
split coefficients. It does not use the vendor's back-adjusted price as a signal.
The old MACD/B5 rules and optimization results are set aside, not relabeled as new
OOS evidence. Nine correlated configurations are not nine independent trials.

Rank the pooled selection-period **net daily return minus a matched cash return**
by its mean divided by sample standard deviation, annualized by sqrt(252).
Undefined candidates are reported and excluded; no finite candidate means failure.
Ties within 1e-10 favor lower allocation, then a longer moving average, then the
stable candidate ID. There is no after-the-fact drawdown or positive-return screen.
The selected validation Sharpe was **0.880248**. The complete nine-row selection
table is [inspectable](validation_candidates.csv); none of the final-test
candidates were ranked or reselected after evaluating the winner.

All segment accounts begin with **$1,000 cash**, no inherited positions, orders,
tax basis or warmup profit. Earlier observations warm the indicators only. A
completed daily close generates a decision; execution is at the **next session's
close**, never before the return ending at that fill. Both the selected strategy
and QQQ buy-and-hold first purchased on **January 5, 2016**.

Fractional synthetic allocations rebalance only on target changes and can drift;
they are not constant volatility matches or leverage caps. Physical QQQ controls
maintain full invested targets to reinvest eligible distribution cash, with
costs. Dividend-reinvestment orders are distinct from changes in the trading
signal. These are modeled daily-close fills at a declared 16:00 New York
accounting clock, not evidence of executable early-close or live prices.

## Final evaluation after each cost layer

Maximum drawdown is shown as the magnitude of the worst loss from a previous
account peak. All rows include hypothetical terminal liquidation and any
outstanding modeled taxes; continuous preterminal accounts are retained privately.

| Selected 200-session rule | CAGR | Maximum drawdown | Cash-excess Sharpe |
|---|---:|---:|---:|
| Before added trading/funding/fund costs | 47.77% | 58.40% | 0.994 |
| After 1 bp fees + 5 bps slippage | 47.31% | 58.70% | 0.987 |
| Also after borrowing costs | 40.61% | 59.26% | 0.893 |
| Also after fund expense/tracking assumptions | **38.95%** | **59.44%** | **0.870** |
| Also under hypothetical investor taxes | **33.85%** | **63.05%** | **0.804** |

The gross-to-pretax-net CAGR difference is **8.82 percentage points**; the
gross-to-hypothetical-after-tax difference is **13.92 points**. These are differences
between matched intended allocation paths, **not pure fixed-share-order cost
attribution**. Funding, tax payments and price levels can change share quantities.
The same selected rule is used in every layer; it is never reselected for a more
favorable cost assumption. QQQ expenses already embedded in its market prices
remain present in the gross source returns.

### Like-period controls

| Expense-inclusive pretax comparison | CAGR | Maximum drawdown | Annualized volatility | Cash-excess Sharpe |
|---|---:|---:|---:|---:|
| Selected synthetic-3× rule | 38.95% | 59.44% | 49.67% | 0.870 |
| Physical QQQ buy-and-hold with reinvestment | 21.12% | 35.12% | 22.18% | 0.872 |
| Physical QQQ 200-session trend control | 17.10% | 23.20% | 16.56% | 0.898 |
| Cash proxy | 2.32% | 0.0005% | 0.17% | Undefined by construction |

The primary endpoint, declared before selection, is the selected-minus-QQQ
**compounded net total-return difference**, not the annualized-return difference:
**+2,312.92 percentage points**. Modeled terminal wealth is approximately $30,445
versus $7,316 from $1,000. This comparison is **not risk matched**. The selected
rule averaged approximately **2.45× effective exposure**, so the large return gap
cannot isolate timing skill from leveraged exposure. No matched-leverage control
was declared, and none is added after seeing the results to rescue or overturn
the primary comparison.

The selected rule changed its intended target **53 times** (approximately 5.1 per
252-session year), with **54 pretax filled orders including terminal liquidation**.
It lost **48.70% in calendar 2022**, versus QQQ's 32.58% loss; its overall net
drawdown trough was March 10, 2023. The higher-Sharpe QQQ trend control remains a
control, not a replacement chosen from final-test results. See the
[annual table](test_annual.csv); **2026 is partial-year**, not a full-year forecast.

### Prespecified cost sensitivities

| Changed assumption, all others fixed | Selected CAGR | Selected drawdown |
|---|---:|---:|
| Slippage 10 bps | 38.59% | 59.68% |
| Slippage 20 bps | 37.87% | 60.16% |
| Slippage 30 bps | 37.15% | 60.64% |
| Borrowing spread 100 bps over DFF | 37.80% | 59.57% |
| Borrowing spread 200 bps over DFF | 35.53% | 59.81% |

These ten strategy/QQQ rows were declared before selection and use the same
selected parameters. [Exact sensitivity results](cost_sensitivity.csv).

## Uncertainty and what the study does not establish

Paired circular stationary resampling keeps the selected strategy and QQQ on the
same sampled dates. **5,000 draws at each of two predefined mean block lengths**
remove observed initial-entry and terminal multipliers, then apply each once per
draw. Ongoing turnover remains in the observed return series. Identity
reconstruction reproduces the exact original terminal wealth.

The conditional 2.5–97.5 percentile interval for the **compounded total-return
difference in percentage points** is **−183.80 to +50,369.23** at 20 sessions;
the 60-session sensitivity is **−134.08 to +40,928.43**. Both cross zero. The very
wide, asymmetric ranges concern multi-year compounded wealth, not a one-year
return or a loss probability. They resample observed returns; they do **not**
reexecute the rule, repeat selection, correct all past researcher choices or
provide confidence bounds for future markets. These are not the old 3,000
underlying-market stress scenarios. [Exact summaries](bootstrap_summary.csv).

There is therefore **no demonstrated risk-adjusted outperformance or profitable
edge**. Higher return comes with much greater volatility and drawdown, while the
overall strategy family still had prior exposure to the historical test years.
The already frozen prospective study remains separate and unchanged.

## Inputs, financing and hypothetical taxes

- Daily raw/action data are used because the historic intraday caches have a
  missing session and source-specific timestamp issues. **All 6,142 selected
  daily session dates match an independent Yahoo daily cache**, with no duplicates
  or missing required daily fields. This is a date-coverage cross-check, not
  independent validation of every price. There are **87 distributions**; 543
  preceding daily observations are features-only warmup.
- Rebuild 3× QQQ **total-return** exposure each day. Borrow two units per unit of
  prior NAV; deduct DFF + a declared 50 bp spread on ACT/360, and hypothetical
  95 bp expense plus 50 bp additional tracking drag on ACT/365, **once**. This is
  not actual TQQQ or a replication of its financing contract.
- Cash uses the DTB3 91-day investment-yield conversion on ACT/365, retaining
  negative observations. Raw-to-prepared rates are independently rederived at
  freeze/verification. Their modeled availability is two US federal business
  days after observation at 18:00 New York. Latest-vintage historical rates and
  assumed release lags are **not a point-in-time archive**.
- Hypothetical tax uses FIFO, 24% short-term and 15% long-term rates, 24% ordinary
  income and same-security wash-loss accounting. QQQ cash dividends are credited
  and reinvested at an **ex-date accounting close**, not verified pay dates; tax
  character is modeled ordinary, not verified qualification. New same-close
  buyers get no dividend and new dividend cash earns no prior-interval interest.
  Synthetic fund-unit realizations are a tax proxy, not actual leveraged-ETF
  investor taxation. Other accounts, state taxes, qualification, rate changes
  and actual tax-filing treatment are not modeled.

Source documentation: [Alpha Vantage daily-adjusted schema](https://www.alphavantage.co/documentation/),
[DFF](https://fred.stlouisfed.org/series/DFF),
[DTB3 discount-basis definition](https://fred.stlouisfed.org/series/DTB3).

## Reproducibility and independent checks

- [Versioned protocol](../../../configs/qqq_split_v3.yaml) and
  [phase-separated CLI](../../../scripts/run_qqq_split_study.py).
- [Public run receipt](run_receipt.json): source revision, file digests, dependency
  versions and the actual freeze → selection → evaluation sequence. A local
  receipt is not independent notarization and is not a substitute private manifest.
- **327 software tests pass**, including **43 new split-specific tests**. Tests
  use artificial inputs; they are not independent market trials.
- [Source audit](source_audit.md) and
  [artificial hand-accounting audit](artificial_accounting_audit.md).
- [Independent real-path audit](independent_accounting_audit.md): **210 numerical
  checks**, eight independently reconstructed pretax paths × all 2,617 test days,
  and 103 bound source/snapshot/output hashes. No project evaluator, ledger or
  rate helper is used by that reconstruction. Four tax paths have structural
  reconciliation; this is not a separate full real-world tax-lot implementation.
- [Exact aggregate metrics](test_metrics.csv), [annual returns](test_annual.csv)
  and [machine-readable summary](summary.json).

Raw licensed inputs, account paths, bootstrap draws and the private executable
freeze are excluded from Git. To reproduce a market-data run, use licensed daily
QQQ/action data, an independent session-date reference and raw/prepared DFF/DTB3
inputs. Create a **new** output directory and run `freeze`, `select`, then
`evaluate`; existing attempts are never silently overwritten. Use the saved
snapshot and exact recorded dependencies if the development checkout changes.

Previous results are [archived separately](../README_pre_split_v3_20261006.md) and
do not carry the rebuilt project or its resume description.
