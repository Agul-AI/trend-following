# Trend Following Study — Validation Results

**2012–2015; 1,006 daily valuation sessions.** Only the **nine sealed development
family finalists** entered validation, not the full 56,644-configuration grid.
The finite daily cash-excess Sharpe selection rule chose **daily EMA200 entry
with a 5.5-hour wait and daily EMA100 exit with a 2.5-hour wait**. The winner was
sealed before historical evaluation; flags did not filter or rank candidates.

## The selected winner lagged buy-and-hold

All portfolios start with $1,000 and share the same corporate-action, cash,
17:00 NY valuation and terminal conventions. Baseline order costs are
**1 bp commission + 3 bps adverse slippage**; taxes, financing and extra expenses
are zero.

| Portfolio | Net CAGR | Maximum drawdown loss | Cash-excess Sharpe | Annualized volatility | Ending $1,000 |
|---|---:|---:|---:|---:|---:|
| Selected EMA200/EMA100 winner | 11.43% | 13.96% | 0.930 | 12.42% | $1,541.21 |
| QQQ buy-and-hold | 19.88% | 13.94% | 1.288 | 14.92% | $2,063.56 |
| Causal cash proxy | 0.06% | <0.01% | — | <0.01% | $1,002.32 |

The strategy had lower volatility but lower net CAGR and cash-excess Sharpe;
its maximum drawdown was slightly deeper. **Validation therefore does not
demonstrate superiority over QQQ buy-and-hold.** It is also selection data, not
an independent evaluation of a previously locked winner.

## Trading activity and modeled cost impact

Baseline winner statistics use **3.9954 elapsed 365-day years**, obtained from
`total_turnover / annualized_turnover`, not four assumed full years.

| Statistic | Selected winner |
|---|---:|
| Completed round trips | 20 (5.006 per elapsed year) |
| One-way orders | 40 (10.012 per elapsed year) |
| Mean holding time | 1,499.05 elapsed calendar hours (62.46 days) |
| Invested exposure | 85.66% of elapsed calendar time |
| Annualized turnover | 10.0096 × pretrade equity per elapsed year |
| Direct commissions / adverse slippage | $4.87 / $14.62 per initial $1,000 |
| Gross / net terminal wealth | $1,566.07 / $1,541.21 per initial $1,000 |

Turnover sums each order's absolute reference notional divided by **pretrade
equity**, then divides by the elapsed year count; it is not a round-trip count.
Holding/exposure time includes nights and nontrading days, unlike the trading-hour
confirmation waits. Buy-and-hold has two one-way orders, so identical per-order
cost assumptions do not imply identical total fees.

Direct modeled costs total **$19.49**. The gross-minus-net terminal difference
is **$24.86 per initial $1,000**, or **2.4858 percentage points of cumulative
return**. This same-decision gross/net comparison includes compounding and cash
allocation effects; it is neither the direct fee sum nor an annual CAGR loss.

## Locked rules and definitions

- Indicator periods are **daily trading sessions**, not half-hours. Every
  intraday preview uses the prior finalized daily state.
- Confirmation waits are trading hours **after the first qualifying hit**:
  5.5 hours requires 12 completed half-hour hits; 2.5 hours requires 6.
  Confirmed orders fill at the next safe observed bar Open.
- Primary selection uses `sqrt(252) * mean(daily return − cash return) /
  sample standard deviation`, with `ddof=1`. Differences within `1e-10` tie on
  stable candidate ID. Cash's cash-excess Sharpe is undefined, not zero.
- Maximum drawdown is loss from the daily 17:00 NY wealth running peak,
  including initial wealth; it is not an intraday-low measure.

Alpha-only intraday prices are reconstructed as-traded units, not native raw
captures or certified point-in-time history. Provider daily closes are assumed
available at 17:00 NY and are not guaranteed official closing or execution
quotes. The cash proxy uses latest-vintage FRED DTB3 with modeled causal release
lags, not a traded Treasury-bill return series. Slippage is a modeled scenario,
not measured execution quality. The source gaps occur outside this segment;
that does not establish insensitivity to gaps in development.

## Inspectable evidence

- [All nine candidate scores and ranking](candidate_ranking.csv).
- [Winner and matched benchmark metrics](winner_benchmark_comparison.csv).
- [Derived daily portfolio curves](daily_portfolio_curves.csv), with no raw
  quotes or trades exported. Curves were produced from the same frozen inputs;
  the publication helper did not evaluate candidates or change selection.
- [Summary and sealed winner identity](summary.json).
- [Independent selection and metrics custody audit](independent_audit.json):
  verifies the primary-only nine-candidate selection, winner seal before OOS
  and curve-derived metrics. It does not reexecute the strategy.
- [Independent passive-control audit](independent_control_audit.json):
  independent QQQ share/dividend replay matched the separate frozen benchmark
  rerun; cash arithmetic and endpoints were checked, not independently
  reimplemented rate integration. This is a control audit, not an independent
  strategy-selection audit.
- [Public artifact integrity hashes](artifact_hashes.json).

See the [historical comparison](../../historical_oos/20261008/README.md) for the
same locked winner evaluated at all four costs. Future runs still require
intact source/data/config seals and explicit stage authorization.
