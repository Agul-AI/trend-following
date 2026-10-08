# Published research evidence

These are selected **aggregate** outputs of the October 2026 research study, not
real-account returns. Raw licensed prices, account paths, private freezes and
personal resume/review files are excluded.

| Output | Scope |
| --- | --- |
| [Full-history cost waterfall](20261002_v1_full_history/cost_waterfall.csv) | Fixed B5 and QQQ, Jan 10, 2002–Jun 1, 2026; end-to-end and fixed-target scenarios |
| [Continuous evaluation](20261002_v1_historical/performance.csv) | Selected/fixed policies and benchmarks, Jan 2, 2009–Jun 1, 2026; comparison-specific overlaps are recorded |
| [All fold rows](20261002_v1_historical/outer_fold_performance.csv) | Six chronological periods; final period partial; continuous account slices without fold liquidation |
| [Training selections](20261002_v1_historical/fold_selections.csv) | Selection occurs before each test period |
| [Inner annual validation](20261002_v1_historical/inner_validation.csv) | Candidate ranking and risk screening use earlier observations |
| [Cost sensitivities](20261002_v1_historical/cost_sensitivity.csv) | Slippage, financing spreads, execution delay and cash assumptions |
| [Training-only risk sizing](20261002_v1_historical/risk_matching.csv) | Scaling is fixed from training, not test volatility |
| [Overlap exclusions](20261002_v1_historical/alignment_exceptions.csv) | Missing common observations are disclosed, not filled |
| [Stress summary](20261002_v1_bootstrap/summary.csv) | 1,000 paths each for 60/126/252-session mean blocks |
| [Additional stress percentiles](20261002_v1_qa/bootstrap_full_percentiles.csv) | Paired QQQ differences and 5th/50th/95th scenario percentiles |

## Main findings

The fixed B5 full-history end-to-end cost scenarios yield:

| Scenario | CAGR | Cash-excess Sharpe | Session-close maximum drawdown |
| --- | ---: | ---: | ---: |
| Gross | 37.52% | 1.009 | 37.07% |
| Trading costs | 37.28% | 1.004 | 37.34% |
| Financing and fund drag included | 30.53% | 0.861 | 38.39% |
| Hypothetical taxes included | 25.78% | 0.769 | 41.70% |

The 11.74-percentage-point gross-to-after-tax CAGR difference is a scenario
comparison, not an exact fixed-trade attribution. These are simulated, known
historical data. The old resume's 28.9%/0.87/41.16% belongs to a different
execution/accounting version; the old Sharpe was zero-risk-free rather than
cash-excess. Do not label the new figures an improved verified edge.

For 2009–2026, the train-selected net-cost pretax account has 38.05% CAGR, 1.018
cash-excess Sharpe and 41.77% drawdown, versus fixed B5 37.93%/1.013/37.74% and
QQQ buy-and-hold 21.19%/0.958/35.12%. The selected policy uses the anchor in five
folds and simple 2x trend in one; B5 is never selected. Anchor/B5 tie on the
specified median training objective; simpler complexity breaks the tie, not a
claim of statistical superiority. The hypothetical taxable B5/selected Sharpe
is below taxable QQQ on this shorter sample. Do not mix periods or make a
universal dominance claim.

| Mean stationary block, sessions | Paths | Median drawdown | Frequency drawdown exceeds 60% | Wilson 95% Monte Carlo interval |
| --- | ---: | ---: | ---: | ---: |
| 60 | 1,000 | 74.86% | 89.5% | 87.4%–91.3% |
| 126 | 1,000 | 70.07% | 79.2% | 76.6%–81.6% |
| 252 | 1,000 | 65.14% | 67.5% | 64.5%–70.3% |

The bootstrap rebuilds underlying prices, leveraged prices, signals and portfolio
returns for fixed B5—not the adaptive selected policy. Stress drawdowns occur
across roughly 24.35 observed-session-year histories, not annual crash periods.
Wilson intervals measure simulation error **conditional on this sampling model**;
they are not future-risk confidence bounds. Resampling disrupts trends/regimes
and does not correct the earlier strategy-discovery bias.

## Provenance and limitations

`publication_provenance.json` binds copied numerical source and selected table
hashes to the original locally reviewed experiment without publishing its private
archive. The original historical manifest SHA is
`3c779b2ac2cd3c118876de4a0165754ffadbc72a88a281021a747b043484cbbf`.
It is a reference identifier, not a claim that the original private manifest or
all its data can be reconstructed from this checkout. The bootstrap's copied
identity/completion records are retained as study metadata, not rerun proof.

CAGR uses observed sessions/252; drawdown uses last-cached daily marks. Rates are
latest-vintage proxies with assumed availability lags, not point-in-time borrowing
quotes. Taxes are ST24/LT15/cash24 scenarios, not personal liabilities. Synthetic
reset/cash/data conventions, partial sessions and actual-vehicle overlap can
materially affect results. See the source modules and protocol. Current tests
validate software logic; they do not establish predictive performance.
