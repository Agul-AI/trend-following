# Trend Following Study — Historical Evaluation

**2016–May 28, 2026; 2,615 daily valuation sessions.** Only **one sealed validation
winner** was evaluated: daily **EMA200 entry, 5.5-hour confirmation wait**, and
daily **EMA100 exit, 2.5-hour wait**. The winner was locked before this stage and
kept unchanged across all four slippage scenarios. No post-evaluation
reselection or gap-flag filtering occurred.

## Lower drawdown, but lower return and Sharpe

Baseline costs are **1 bp commission + 3 bps adverse slippage per order**.
All portfolios start with $1,000 and share corporate-action, cash, 17:00 NY
valuation and terminal conventions. Taxes, financing and extra expenses are zero.

| Portfolio | Net CAGR | Maximum drawdown loss | Cash-excess Sharpe | Annualized volatility | Ending $1,000 |
|---|---:|---:|---:|---:|---:|
| Locked EMA200/EMA100 winner | 13.81% | 23.08% | 0.795 | 14.83% | $3,842.90 |
| QQQ buy-and-hold | 20.96% | 35.12% | 0.868 | 22.19% | $7,240.70 |
| Causal cash proxy | 2.31% | <0.01% | — | 0.17% | $1,268.35 |

The strategy had lower drawdown and volatility than buy-and-hold, but lower
net CAGR and cash-excess Sharpe. **Lower risk does not demonstrate overall
superiority.**

## Trading activity and modeled cost impact

Baseline winner statistics use **10.4035 elapsed 365-day years**, obtained from
`total_turnover / annualized_turnover`, rather than rounding the segment length.

| Statistic | Locked winner |
|---|---:|
| Completed round trips | 41 (3.941 per elapsed year) |
| One-way orders | 82 (7.882 per elapsed year) |
| Mean holding time | 1,718.05 elapsed calendar hours (71.59 days) |
| Invested exposure | 77.29% of elapsed calendar time |
| Annualized turnover | 7.8804 × pretrade equity per elapsed year |
| Direct commissions / adverse slippage | $15.69 / $47.07 per initial $1,000 |
| Gross / net terminal wealth | $3,971.04 / $3,842.90 per initial $1,000 |

Turnover sums each order's absolute reference notional divided by **pretrade
equity**, then divides by the elapsed year count; it is not a round-trip count.
Holding/exposure time includes nights and nontrading days, unlike the trading-hour
confirmation waits. Buy-and-hold has two one-way orders; matched per-order costs
do not mean matched total fees.

Direct modeled costs total **$62.77**. Gross-minus-net terminal wealth is
**$128.14 per initial $1,000**, or **12.8137 percentage points of cumulative
return**. This same-decision gross/net difference includes compounding and cash
allocation effects; it is not the direct fee sum or an annual CAGR loss.

## Same winner, all four modeled costs

Commission remains 1 bp per order; the table varies adverse slippage only.
Costs apply to actual orders, not every observation. No winner was selected
separately for each cost.

| Slippage per order | Winner net CAGR | QQQ net CAGR | Winner cash-excess Sharpe | QQQ cash-excess Sharpe | Winner drawdown loss | QQQ drawdown loss |
|---|---:|---:|---:|---:|---:|---:|
| 1 bp | 13.99% | 20.97% | 0.806 | 0.868 | 22.94% | 35.12% |
| 3 bps (baseline) | 13.81% | 20.96% | 0.795 | 0.868 | 23.08% | 35.12% |
| 10 bps | 13.19% | 20.94% | 0.756 | 0.867 | 23.56% | 35.12% |
| 25 bps | 11.86% | 20.91% | 0.674 | 0.866 | 24.63% | 35.12% |

Buy-and-hold had higher net CAGR and cash-excess Sharpe in every scenario;
the strategy had a shallower maximum drawdown in every scenario. These are
hypothetical execution-cost stresses, not evidence of achievable live fills.

## Interpretation and limitations

**This is retrospective, previously examined history—not untouched OOS or live
evidence.** Chronological seals prevent retuning within this run; they do not
erase prior familiarity with the history or the development search over
56,644 configurations. Validation selected from nine finalists; historical
evaluation tested only its locked winner.

Indicator periods are daily sessions. Confirmation waits are trading hours
after the first qualifying completed half-hour hit, requiring 12 entry hits
and 6 exit hits. Orders fill at the next safe observed bar Open. Daily
cash-excess Sharpe uses 252 annualization and sample standard deviation
(`ddof=1`); cash's excess Sharpe is undefined. Maximum drawdown is measured on
daily 17:00 NY portfolio values, not intraday lows.

Alpha-only intraday prices are reconstructed as-traded units, not native raw
captures or certified point-in-time history. Provider daily closes are assumed
available at 17:00 NY and are not guaranteed official closing or execution
quotes. The causal cash proxy uses latest-vintage FRED DTB3 with modeled release
lags, not a traded Treasury-bill return series. The disclosed source gaps and
regulatory halt occur outside this segment; no post-gap flags occurred here.
That does not prove development results are unaffected by missing observations.

## Inspectable evidence

- [Every winner/BH/cash cost scenario](cost_scenario_comparison.csv).
- [Derived daily curves for all scenarios](daily_portfolio_curves.csv), with no
  raw quote or trade export.
- [Summary and unchanged sealed winner identity](summary.json).
- [Independent selection, metrics and cost custody audit](independent_audit.json):
  verifies the winner-before-OOS chronology, unchanged decisions across costs,
  daily metrics and gross/net cost arithmetic, without strategy reexecution or
  post-evaluation fitting.
- [Independent passive-control audit](independent_control_audit.json): QQQ
  share/dividend replay and causal cash integration were independently
  implemented and matched all 2,615 daily marks at all four costs. This checks
  passive controls, not an independent strategy reimplementation.
- [Public artifact integrity hashes](artifact_hashes.json).

See [validation](../../validation/20261008/README.md) for the nine-candidate
selection. Future market runs still require intact source/data/config seals and
explicit authorization; the same frozen study does not permit post-OOS retuning.
