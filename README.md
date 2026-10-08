# QQQ v5 daily-indicator research

**Repository:** [Agul-quant/trend-following](https://github.com/Agul-quant/trend-following).
This is the publishing destination for v5 and subsequent updates. Earlier code
and v4 reference evidence are retained for compatibility and audit, not presented
as results of the current v5 study.

## Public setup

Install the project in a Python environment with the `dev` and `research` extras:

```bash
python -m pip install -e '.[dev,research]'
python -m pytest
python scripts/run_qqq_gap_flags_v5.py --help
```

Market-price caches, credentials, private authorizations and prepared freezes
are **not included**. The fabricated tests do not need these inputs. Historical
preparation requires verified local Alpha data and a local authorization receipt;
it intentionally stops when those are missing. Use `--source-root` explicitly
when auditing price caches outside this checkout. Dated v5 receipts retain their
pre-publication status fields as historical milestones.

## New v5 implementation

The latest [gap-held development variant](docs/research_validation/gap_hold_flags_v5.md)
uses the user-authorized no-trading rule for missing intervals: keep the last
filled QQQ/cash holding, cancel pending orders, and restart confirmations.
Preparation is **data-ready under that disclosed policy**, not repaired complete
intraday history. Every development configuration receives descriptive flags for
confirmed signals after a source gap through the next trading session's close;
flags never change ranking or selection. Use `scripts/run_qqq_gap_flags_v5.py`.
**No v5 market-performance run has been started.**

The separate [daily-indicator/half-hour-confirmation study](docs/research_validation/daily_hourly_v5.md)
implements **56,644** unlevered QQQ/cash configurations using **Alpha Vantage
prices only**, with historical evaluation ending **May 28, 2026**. Yahoo prices
are not used for stitching or gap repair. SMA, EMA and MACD periods
are **daily trading sessions**; independent 0–6.5-trading-hour entry/exit confirmation waits
are checked at every completed half-hour close. Zero means the first qualifying
completed bar; 0.5 hours requires one additional qualifying close. Fills use the next bar's open. The study charges 1 bp
commission plus 1/3/10/25 bps slippage, with **no modeled taxes**.

**No v5 market-performance results are available.** The new
[Alpha quote-reconciliation profile](docs/research_validation/alpha_quality_reconciliation_v5.md)
verifies reconstructed as-traded units, separates provider daily closes from
intraday trading quotes, and excludes documented halt intervals from signals
and unsafe fills. It does not claim native raw capture or overwrite quotes.
The strict profile still rejects **14 unknown missing half-hours**: the
March 29, 2001 opening interval and all November 30, 2004 intervals. A disclosed
no-trading blackout fallback requires explicit human approval before performance
runs; it is not a repair of missing prices. The user-approved cutoff no longer
requires a June 1 tail.
The completed v4 evidence below, its frozen artifacts, existing monitors and
personal resumes remain unchanged.

## Archived v4 study (not v5 results)

That archived study was a **fresh data-only restart**. Data were split before
research, and a new development agent received only raw observations through
2011—not earlier strategies/results or later prices. Its numerical processes
ran in a deny-default sandbox. The developer stopped after handing off a frozen
policy/grid and was never given validation or test outcomes.

## The enforced workflow

1. Development **2002–2011**: 2,519 scored sessions plus 543 earlier feature-only
   observations; log all **28** explored configurations, including weak results.
2. Seal policy source, **12** candidates, code, costs and a fixed selection rule.
3. Validation **2012–2015**: rank those unchanged candidates using net daily
   cash-excess Sharpe; no manual replacement or performance-dependent risk screen.
4. Lock the selected rule before releasing test prices.
5. Historical OOS **January 4, 2016–June 1, 2026**: evaluate one choice on **2,617
   sessions** against predefined QQQ, hypothetical 3× and cash controls.
6. Keep the result, without feeding it back to the developer or retuning.

Actual kernel probes denied later-data/old-result reads, protected writes,
networking and process spawning. Numerical child sandboxes do not restrict every
same-user tool: the agent's other tools were instruction-scoped, and coordinator
custody/ranking/reporting operated outside those children. Historical OOS is not
live execution or a claim that human background knowledge was erased.

## What the clean test showed

| Expense-inclusive pretax result | Annualized return | Maximum drawdown | Volatility | Cash-excess Sharpe |
|---|---:|---:|---:|---:|
| Validation-selected rule | **16.22%** | **40.43%** | 29.15% | **0.585** |
| QQQ buy-and-hold | **21.12%** | **35.12%** | 22.18% | **0.872** |
| Hypothetical 3× buy-and-hold | 42.55% | 81.45% | 66.55% | 0.835 |

**The selected rule underperformed QQQ.** Higher leverage did not establish an
advantage. Its CAGR declined from **20.62% before added costs to 16.22% after
trading/funding/fund assumptions**, and **12.93% under hypothetical taxes**.
After-tax drawdown was **43.98%**. The gross-to-after-tax annualized-return gap is
**7.69 percentage points**, not pure fixed-share-order cost attribution.

The selected rule requires both trailing 252- and 63-session gross 3× compounding
paths to be positive. It then targets a two-thirds nominal investment in the
simulated instrument, otherwise cash. Targets use pre-trade equity and can drift;
this is not a maintained 2× exposure or a volatility match. No old B5/MACD/MA
optimization result was supplied to its development.

## Inspectable evidence

- [Complete study and limitations](reports/research_validation/20261006_blind_v4/README.md)
- [Data-only policy](reports/research_validation/20261006_blind_v4/policy.py)
- [Complete development ledger](reports/research_validation/20261006_blind_v4/development_log.json)
- [Pre-validation candidate proposal](reports/research_validation/20261006_blind_v4/proposal.json)
- [All 12 validation results](reports/research_validation/20261006_blind_v4/validation_metrics.csv)
- [Final test/control/cost metrics](reports/research_validation/20261006_blind_v4/test_metrics.csv)
- [Phase and source receipt](reports/research_validation/20261006_blind_v4/final_run_receipt.json)
- [Independent arithmetic audit](reports/research_validation/20261006_blind_v4/independent_audit.md)
- [Custody CLI](scripts/run_qqq_blind_guard.py), [coordinator CLI](scripts/run_qqq_blind_study.py)
  and [versioned protocol](configs/qqq_blind_v4.yaml)

**379 software tests passed**, including **52 new v4 tests**; all **11 production
kernel probes passed**. Independent calculations reconcile **41,872 pretax daily
states** plus tax structural identities. Financial arithmetic passed, but the
audit is **not entirely green: 716/717 checks passed**. The remaining issue is a
preserved exit-timestamp metadata defect (start time was copied into exit
records). Phase order remains verified through the custody chain and separate
selection/completion timestamps. It is documented, not retroactively rewritten.

Software checks and child-process barriers improve research integrity; they are
not independent market trials, a full investor-tax certification or proof of a
profitable strategy. Raw quotes, private account paths, executable custody packets
and personal resumes are deliberately excluded from Git.

## Run and reproduce

Use the requested Python environment for local work. For public code checks:

```bash
python -m pytest
python -m ruff check .
python scripts/run_qqq_blind_guard.py --help
python scripts/run_qqq_blind_study.py --help
```

Market reproduction requires properly licensed raw daily QQQ/action files,
independent date coverage and FRED raw/prepared rate inputs. The guard supports
`prepare --source-map` with an exact role/path mapping. A **new** study directory,
fresh development context and matching dependencies are required. macOS
sandbox-exec is mandatory; unavailable enforcement fails closed. Existing
research/evaluation attempts cannot silently be overwritten.

## Previous work is preserved separately

- [Previous split-study overview](ARCHIVED_SPLIT_V3_README.md)
- [Earlier full-sample/walk-forward overview](ARCHIVED_RESEARCH_README.md)
- [Earlier research evidence index](reports/research_validation/README_pre_blind_v4.md)
- [Original generic framework](LEGACY_README.md)

The separately frozen [prospective OOS study](docs/research_validation/oos_study_v2.md)
remains unchanged; it is not performance evidence for this different policy.
No broker orders, recurring collection jobs or notifications are created here.
