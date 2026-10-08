# QQQ out of sample study design

**Study specification dated October 5, 2026. The pipeline is implemented; actual freeze and observation status are recorded in the [operating guide](oos_study_v2.md) and public freeze receipt. Future results remain unavailable until the declared checkpoints.**

We can test the QQQ leverage-rotation rule on genuinely new data. The recommended
study locks the entire experiment before a future trading session, then measures
performance over the following 12 and 24 months. It does not require real money
or a person recording every bar in real time. The purpose is to test whether the
rule adds value after costs, not to establish that the strategy works.

## What qualifies as new evidence

The existing six-period walk-forward study selects rules using earlier data and
evaluates them on later data. That is out of sample relative to each selection
step. However, the rule family and research choices were developed with knowledge
of the wider history. Splitting that same history again cannot make it an
untouched research holdout. This distinction addresses research selection bias,
not a failure of the chronological code. See the
[backtest overfitting research](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
and [time-ordered validation documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

Use three separate labels:

| Study | What it can establish | Current status |
| --- | --- | --- |
| Historical walk-forward | Whether selection uses only earlier data; historical robustness | Implemented across six periods |
| Prospective frozen-rule simulation | How the locked rule performs on subsequent market data under declared fill assumptions | Proposed here |
| Contemporaneous paper execution | Whether quotes, signals and hypothetical fills were actually recorded on time | Separate prototype; no performance observations recorded |

The original manual-paper version scheduled October 5 as its start. As of the
October 5 evening audit, its ledger contained only an initialization header and
zero observations. Its first recording deadline had expired. Preserve that as a
missed operational start; do not backfill it into a successful paper record.
That missed deadline does not prevent a new frozen-rule future-data study.

## Keep the historical selection test

Retain the existing five candidates: cash, the simple 200-day QQQ rule at 1x and
2x exposure, the anchor rule, and B5. Before each test period, use only completed
earlier annual validation windows to select a candidate. The existing objective
is median net cash-excess Sharpe, with a 45% worst annual validation drawdown
limit, simpler-rule tie breaking, and cash fallback when eligible scores are not
positive. Sharpe is a research diagnostic here, not a resume selling point.

| Earlier data through | Later evaluation period |
| --- | --- |
| December 2008 | 2009 to 2011 |
| December 2011 | 2012 to 2014 |
| December 2014 | 2015 to 2017 |
| December 2017 | 2018 to 2020 |
| December 2020 | 2021 to 2023 |
| December 2023 | 2024 to June 1, 2026, the actual data end |

Report every period and the continuous portfolio, including switching costs and
tax basis. Do not reset capital between periods, discard losing periods, or pick
new splits because their results are more attractive. Trailing feature history
is allowed; selection may not use returns or fills occurring after its cutoff.
The final partial period must stay explicitly labeled.

## Lock the prospective comparison

**Primary question:** Does B5, sized using only earlier volatility, earn a higher
net total return than buying and holding QQQ over the new period?

- Use the existing fixed B5 rule, not whichever candidate wins the new test.
  Freeze its full parameter specification and seven-observation session cadence.
- Use actual QQQ for signals and actual TQQQ for the leveraged instrument.
  There are seven completed bars on a normal session, including the last
  30-minute interval, and four on a scheduled early close.
- Before the holdout, simulate the same implementation over the last 252 complete
  trading sessions with unscaled B5 (`m=1`), the frozen cost/cash conventions,
  and enough preceding history to initialize every feature.
  Freeze an allocation multiplier
  `m = min(1, sd(QQQ daily returns) / sd(net B5 daily returns))`.
  Invalid or zero calibration volatility fails readiness rather than inviting
  a discretionary substitution. Use identical corporate-action conventions.
- Apply this multiplier to every nonzero B5 target weight throughout the holdout.
  Never recompute it using test-period volatility. This is **training-volatility
  scaling**, not a guarantee that future risk will match QQQ.
- Both portfolios start with $1,000 hypothetical cash, no inherited holdings,
  orders, tax lots or warmup profits. Buy QQQ at the fixed first protocol-eligible
  delayed fill timestamp, regardless of when or whether B5 signals a purchase.
  Include benchmark trading costs.
- Primary cash interest is explicitly zero. A Treasury-rate cash scenario is
  secondary and requires timestamp-available rate data. Never silently assume a
  risk-free series of zero when reporting excess Sharpe.

Unscaled B5, cash, TQQQ buy-and-hold and the simple QQQ 200-day rule are secondary
comparisons. Report all of them; none can replace the primary comparison after
results are seen. Adaptive candidate selection is outside this primary study.

## Dates and sample size

The start is the **first complete scheduled trading session after the final
implementation, data, protocol and environment freeze has finished**. The freeze
must record its actual timestamp, code revision, input hashes, dependency
versions, calibration multiplier and future evaluation cutoffs. Preserve a
timestamped copy for the mentor or a public commit without licensed raw data.

For illustration only, a study ready before October 6 could cover October 6,
2026 through October 5, 2027. The existing scheduled calendar contains **251
sessions and 1,751 completed bars** for that interval: 249 normal sessions times
seven bars plus two early closes times four bars. Exceptional closures could
change those counts. This date is not a promise or a declared actual start.

- **12 months:** preliminary descriptive report, even if results are poor.
- **24 months:** principal final report of the same unchanged experiment.
  Extend the verified calendar before freezing; current calendar coverage is
  insufficient for two years.
- No early success stopping or retuning after the first report. Any material
  change starts a separately registered experiment and retains the original one.

The old estimate of approximately 4.7 position changes per year would correspond
to roughly five changes in one year and nine in two years if that rate persisted.
This is a sample-size illustration, not a forecast for the new cadence. Neither
1,751 hourly bars nor 251 daily marks represent that many independent decisions.

## Data and execution conventions

Acquire appropriately licensed hourly prices, distributions and split records.
Archive timestamped downloads and revisions immutably; retain the original and
any corrected record. Freeze raw versus adjusted price conventions before the
study. Adjust split quantities and policy price references consistently. Record
distribution entitlement as an ex-date receivable and release spendable cash on
the payment date; this requires an upgrade from the ledger's current ex-date cash
credit. Never count the same dividend through both adjusted prices and a cash
payment.

Generate signals only from completed bars. Fill a simulated order no earlier than
the next tradable bar close, and earn returns only after the fill. Preserve the
one-position-change-per-session limit and target-weight-then-drift convention.
Late downloading of a genuinely future bar is permitted for this batch study;
it is not evidence that an order or executable quote was observed in real time.
Do not substitute daily bars for missing hourly decisions.

The predeclared source hierarchy uses Yahoo regular-session closes, with full-session Alpha Vantage 30-minute fallback when either symbol is missing an expected observation. Restore physical prices with the matching daily adjustment factor and verify every fallback close against preserved 15-minute nested-close evidence. Reject missing subbars and exclude after-close rows. Archive all raw inputs and replay the normalization before freezing or scoring.

Predeclare the outage handling and correction policy before
launch. Missing critical bars or unresolved actions invalidate the affected
primary evaluation rather than allowing silent deletion, forward filling, or
period selection. Archive regularly enough that provider retention limits do not
erase early hourly history. Monitoring data integrity is allowed; using emerging
performance to change this experiment is not.

## Cost and tax reporting

| Comparison | Frozen assumption or interpretation |
| --- | --- |
| Before investor trading costs | Actual fund prices, already reflecting fund economics |
| Primary net trading result | 1 bp fee plus 5 bps adverse slippage per traded notional |
| Trading sensitivity | 10, 20 and 30 bps slippage, same declared fee |
| Financing sensitivity | Separate synthetic 3x daily-reset companion; DFF plus 0, 50, 100 and 200 bps, with explicit fund-drag assumptions |
| Hypothetical tax scenarios | Tax-free versus declared 24% short-term and 15% long-term rates; explicit distributions, lots, losses and settlement conventions |

Actual TQQQ prices already reflect financing, fund expenses and tracking effects;
do not subtract synthetic financing or an expense ratio a second time. Its
prospectus also distinguishes financing costs from the headline operating
expense table. See the [ProShares summary prospectus](https://www.proshares.com/globalassets/proshares/prospectuses/tqqq_summary_prospectus.pdf).

The synthetic companion diagnoses model assumptions. Its difference from actual
TQQQ is not a measured pure financing cost. Historical synthetic assumptions are
not statements of the fund's current actual expense ratio.

Taxes are hypothetical investor scenarios, not a haircut on all returns or a
statement of personal liability. Require FIFO lots, holding periods, distributions,
loss carryforwards, wash-sale treatment, and the same terminal-liquidation and
tax-reserve convention for strategy and benchmarks. Show marked wealth before
liquidation separately from hypothetical liquidated after-tax wealth. Declare
unknown distribution classifications rather than assuming a favorable tax rate.
Unknown tax character makes taxable results provisional, not the otherwise valid
pretax path when cash amounts and entitlements are known. The accounting design
should follow the relevant conventions in [IRS Publication 550](https://www.irs.gov/publications/p550),
without treating this isolated simulation as a complete investor tax return.
Tax scenarios must be specified before launch, although they are secondary and
never affect selection of the primary rule.

Separate matched-target-allocation-path attribution from end-to-end reruns where
costs or tax cash flows can change later policy decisions. Quantities and funding
sales can still differ under the matched-target convention; do not call it a
fixed-share-order experiment or present an end-to-end CAGR gap as a pure
accounting contribution.

## Scorecard and interpretation

The primary endpoint is the **net total-return difference in percentage points**
between scaled B5 and QQQ at 24 months, including hypothetical exit fees and
slippage for both portfolios. Report the same comparison at 12 months as
preliminary. Checkpoint liquidation and taxes use a cloned account; the original
account continues without resetting holdings, capital or tax basis. Also show
cumulative return, annualized volatility, maximum drawdown,
worst month, cash/exposure weights, position changes, turnover and costs. Sharpe
and CAGR are secondary; annualization conventions must be explicit.

Freeze a paired daily-return stationary-bootstrap analysis with 5,000 resamples,
seed 42, mean block length 20 sessions and a 60-session sensitivity. Resample the
strategy and benchmark together and recompute their compounded return difference,
retaining the declared initial and terminal trading-cost convention.
Report 95% conditional intervals and sensitivity, not annual crash probabilities
or a guarantee of statistical power. This return-comparison analysis is distinct
from the existing 3,000 underlying-price-path stress test. A short record with
few changes and regimes can remain inconclusive even after two years.

Research success means a valid frozen protocol, complete auditable data and
reproducible reporting. Provisional performance support requires a positive net
return difference **and no larger observed maximum drawdown than QQQ**. If return
improves but downside worsens, report a trade-off, not improved downside control.
An interval spanning zero is inconclusive; a favorable interval alone does not
prove durable edge or authorize deployment. Retain failures and negative results.

## Implementation sequence

1. Preserve the existing missed-start paper record and historical freezes.
2. Add a separate prospective batch configuration and evaluator using the existing
   execution engine and distribution-aware ledger. Do not reuse the manual
   recorder's 15-minute freshness gate as a statistical requirement.
3. Test split/dividend continuity, information cutoffs, delayed fills, benchmark
   alignment, cash/tax reconciliation, missing-data failure and unchanged results
   after append-only future data. Extend and verify the calendar.
4. Calibrate the multiplier on pre-holdout data with the exact new conventions.
   Freeze code, inputs, cost/tax scenarios and both reporting dates only after
   readiness checks pass. Begin at the next complete future session.
5. Collect and preserve future data under the declared contract. Produce the
   fixed 12- and 24-month reports without selecting a better-performing version.

Optional contemporaneous paper monitoring needs additional reviewed work: the
collector currently hard-codes the old version, and the bridge deliberately
halts on distributions/splits. A renamed YAML alone is not a launch-ready system.
Keep any real-time execution-quality evidence separate from batch performance.

No real trades or recurring automation are part of this study. A final freeze creates a prospective protocol, not completed out-of-sample performance. The operating guide and freeze receipt distinguish implementation, frozen timing and genuinely observed results.
