# The data-only restart underperformed QQQ in its historical out-of-sample test

## Executive summary

This experiment starts with raw data and a fresh strategy-development agent,
not the earlier project rules or results. The data were physically split first.
The developer received **only observations through 2011** and generic accounting
tools; it could not read later prices or earlier result files in its numerical
processes. It logged **28 development configurations**, handed off **12 fixed
candidates**, and stopped before validation. No later outcomes were sent back to it.

The prescribed validation ranking selected a 252/63-observation momentum-agreement
rule with a two-thirds nominal allocation to a hypothetical daily-reset 3× QQQ
instrument when eligible, otherwise cash. The one final historical OOS test,
**January 4, 2016–June 1, 2026**, produced **16.22% annualized return after modeled
costs**, **40.43% maximum drawdown** and **0.585 cash-excess Sharpe**. QQQ returned
**21.12%**, with **35.12% drawdown** and **0.872 Sharpe** over the same period.

**The hypothesis is not supported by this comparison.** The selected rule had
lower return and risk-adjusted performance, with greater volatility/drawdown than
QQQ. It is not replaced with another rule after viewing these outcomes. Historical
OOS means later prices are excluded from this development/selection procedure;
it is not a claim of live fills, future returns or erased human background knowledge.

## What was actually enforced

| Stage | Scored dates | Sessions | What could be used |
|---|---|---:|---|
| Development | Jan 2, 2002–Dec 30, 2011 | 2,519 | Only earlier raw QQQ and available-rate data; 543 pre-2002 observations for feature warmup |
| Validation and selection | Jan 3, 2012–Dec 31, 2015 | 1,006 | All 12 already sealed candidates, unchanged code and a fixed ranking rule |
| Final historical OOS evaluation | Jan 4, 2016–Jun 1, 2026 | 2,617 | One already selected configuration plus predefined controls; no refitting |

1. **Split before research.** Separate read-only raw-data packets were created.
   Future-back-adjusted price columns were excluded. An independent source checked
   session dates only, not another strategy's performance.
2. **Fresh context.** A new agent with no inherited project turns was the only
   strategy developer. Earlier rules, optimization results, reports and later
   observations were outside its authorized scopes.
3. **Numerical process boundary.** Deny-default macOS child sandboxes permitted
   only the development packet, generic engine and work/output directories.
   Actual probes verified denied reads of validation/test data, original raw
   sources and a previous result file, as well as denied network access, protected
   writes and process spawning. **11 production probe checks passed.**
4. **Immutable handoff.** Policy source, the entire candidate grid, complete
   development ledger, accounting engine and selection criteria were hash-bound
   before validation was released. Source/proposal were published before release
   in [commit ee2a8db](https://github.com/Agul-AI/trend-following/commit/ee2a8db9cbe157a3b2c9475167571ec249e30d53).
5. **Fixed selection.** Rank pooled validation expense-inclusive pretax daily
   cash-excess Sharpe, with sample standard deviation and sqrt(252). Ties within
   1e-10 use lower declared complexity, then stable ID. Undefined scores are
   reported/excluded; no finite score means failure. No positive-score hurdle,
   after-the-fact drawdown screen or manual winner replacement is introduced.
6. **One held test.** The selected member of the original grid, exact policy bytes
   and costs were locked before test release. The pure policy receives only a
   deep-copied current/past window; code cannot read outside files, inspect globals
   or retain mutable candidate-order state. Today’s decision fills **next session's
   close** and earns no return ending at that fill. No inherited portfolio, order,
   tax basis or warmup profit enters a new segment.

The numerical sandbox is a real kernel boundary, but the research agent's other
same-user tools are **instruction-restricted, not OS-sandboxed**. Custodian
partitioning, winner ranking and report arithmetic run outside the numerical
children by design. Python open/import logs supplement kernel enforcement; they
are not an exhaustive native-syscall trace. “Custodian-owned” means coordinator
control, not Unix root ownership. These limits are disclosed rather than claiming
that every tool, process or person was rendered incapable of knowing history.

## The independently developed rule

The fresh developer observed large losses and volatile episodes in its earlier
data, explored a small self-authored family, and documented unsuccessful settings.
The strongest development result was modest and episodic; the selected final
configuration is **not** automatically the best development configuration.

Eligibility requires both trailing **252-session and 63-session** gross 3×
compounding paths to be positive:

`sum(log(1 + 3 × QQQ daily total return)) > 0` over each window.

If both qualify, the desired nominal synthetic allocation is **2/3**; otherwise
it is zero. This is not three times an endpoint price move. The frozen grid has
six window pairs × two allocations; the final grid has no volatility ceiling.
The complete [proposal](proposal.json), [exact policy](policy.py),
[development ledger](development_log.json) and [12 validation results](validation_metrics.csv)
remain inspectable. All development configurations, negative results and the two
technical failures are retained. The final grid was declared before the last
eight development configurations were evaluated; no held result shaped it.

The validation winner had **0.647566** pooled cash-excess Sharpe. The same
configuration subsequently had **0.584725** in the test. There is no replacement
after viewing that deterioration or its benchmark comparison.

## Same-period results and implementation costs

All rows start with **$1,000 cash**, use the same scored dates, and include cloned
terminal liquidation and outstanding modeled taxes. CAGR uses 252 sessions per
year. Drawdown is the magnitude of peak-to-trough loss, not a future crash probability.

| Selected rule cost layer | Annualized return | Maximum drawdown | Cash-excess Sharpe |
|---|---:|---:|---:|
| Before added trading/funding/fund costs | 20.62% | 38.85% | 0.711 |
| After 1 bp fees + 5 bps slippage | 20.03% | 38.91% | 0.694 |
| Also after borrowing costs | 16.99% | 39.91% | 0.607 |
| Also after fund expense/tracking assumptions | **16.22%** | **40.43%** | **0.585** |
| Also under hypothetical investor taxes | **12.93%** | **43.98%** | **0.503** |

The gross-to-pretax-net CAGR decline is **4.40 percentage points**. Gross to the
hypothetical after-tax scenario declines **7.69 points**. These are matched
intended target paths, not pure fixed-share-order attribution: quantities,
costs, funding and tax-payment flows can differ. No cost layer reselects parameters.
QQQ's own embedded fund economics remain in its market returns even in the gross row.

| Expense-inclusive pretax control | Annualized return | Drawdown | Annualized volatility | Cash-excess Sharpe |
|---|---:|---:|---:|---:|
| Selected rule | 16.22% | 40.43% | 29.15% | 0.585 |
| Physical QQQ buy-and-hold with reinvestment | 21.12% | 35.12% | 22.18% | 0.872 |
| Hypothetical 3× buy-and-hold | 42.55% | 81.45% | 66.55% | 0.835 |
| Cash proxy | 2.32% | 0.0005% | 0.17% | Undefined by construction |

The predeclared primary endpoint, selected-minus-QQQ **cash-excess Sharpe
difference**, is **−0.287204**. The net CAGR difference is **−4.90 percentage
points per year**. These are descriptive historical outcomes, not significance
tests or forecasts. Mean effective selected exposure was **1.37×**, not a fixed
2× or a risk match to QQQ/3× buy-and-hold. Lower drawdown than full 3× exposure
alone does not demonstrate timing alpha or a deployable strategy.

Nominal targets use **pre-trade marked equity**; fees/slippage are paid from
cash, and full-investment purchases are scaled to what cash can afford. They are
not exact maintained post-cost weights. Synthetic holdings rebalance only on
target change and may drift. QQQ full-weight controls reinvest eligible dividend
cash with trading costs. The [annual table](annual_metrics.csv) flags **2026 as
partial-year**; terminal taxes/liquidation are included in its last observation.

## Accounting and data limitations

- Raw close, split and dividend inputs produce a causal underlying total-return
  index. No old synthetic price cache, adjusted-price signal or earlier strategy
  is imported. The simulated product resets daily, not hourly.
- Borrow two units per unit of prior synthetic NAV; charge DFF + a declared 50 bp
  spread on ACT/360 **once**, then hypothetical 95 bp expense plus 50 bp extra
  tracking drag on ACT/365. The product is not actual TQQQ or its financing contract.
- Cash uses the declared DTB3 91-day investment-yield proxy on ACT/365, including
  negative values. Raw-to-prepared rates are checked; known-rate integration uses
  elapsed UTC seconds and only timestamp-available observations. The two federal
  business-day/18:00 publication lag is modeled, and data are latest vintage—not
  independently reconstructed point-in-time records.
- The 16:00 New York clock is a daily accounting convention, not actual early-close
  fill availability. Prices/slippage are hypothetical execution assumptions.
- FIFO taxes, 24% short-term/ordinary and 15% long-term rates, wash-loss rules and
  funding sales are scenarios. QQQ dividends use ex-date cash credit/reinvestment
  and ordinary character, not verified pay dates or qualification. New same-close
  buyers receive no dividend; new cash earns no preceding-interval interest.
  Synthetic-unit gains are not actual leveraged-ETF investor tax filings. Other
  accounts, state taxes and investor-specific tax treatment are excluded.

## Verification, custody caveat and reproduction

**379 software tests passed**, including **52 new v4 tests** on fabricated inputs.
Actual kernel probes also passed. Software checks are not independent market trials.

The [independent post-test audit](independent_audit.md) rebuilt **16 pretax account
paths × 2,617 dates = 41,872 daily states**, without project engine/ledger/rate
helpers. Daily prices, targets, cash, quantities, equity, returns, turnover and
terminal metrics reconcile. Four tax paths have structural/event/metric checks;
this is not a second complete real-world tax-lot implementation.

**716/717 audit checks passed; financial arithmetic passed.** The one remaining
check is a preserved metadata defect: each held-stage exit payload repeats its
start timestamp. Those values must not be represented as completion times.
Hash-chain append order, proposal/selection seals, test-start receipt and the
separate parent completion timestamp establish the correct phase order. No frozen
record or financial result was rewritten to make the audit appear fully green.
Technical audit v1/v2/v3 corrections concerned schema, clock and nominal sizing
clarification only; they changed no primary policy, parameter, cost or data bytes.

- [Pre-validation source/grid receipt](pre_validation_receipt.json)
- [Final projected run receipt](final_run_receipt.json)
- [Selection receipt](selection_receipt.json) and [completion receipt](completion_receipt.json)
- [Independent machine-readable evidence](independent_audit.json) and [exact post-test checker](independent_audit.py)
- [Exact 20 scenario/control rows](test_metrics.csv)
- [Coordinator phase locks](../../../scripts/run_qqq_blind_study.py),
  [kernel custody](../../../scripts/run_qqq_blind_guard.py) and
  [versioned protocol](../../../configs/qqq_blind_v4.yaml)

Raw licensed inputs, developer workspaces, private syscall logs, daily accounts
and executable freezes are not in Git. Public receipts/logs are safe projections,
not replacements for the exact private manifests. Reproduction requires licensed
raw daily/action data and FRED raw/prepared inputs, the same software environment
and a **new** custody directory; macOS sandbox-exec is required and missing support
fails closed. Old and completed attempts cannot be silently overwritten.

Earlier project results are [archived](../README_pre_blind_v4.md), not inputs to
this development. The separately frozen prospective study is preserved unchanged.
