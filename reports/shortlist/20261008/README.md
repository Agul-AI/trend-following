# Trend Following Study — Diversified Development Shortlist

**45 development-selected options, retained unchanged through both later periods.**
This is an **exploratory retrospective follow-up after prior validation and
historical results were examined**, not a restoration of untouched OOS evidence.
The original nine-finalist / single-winner evaluation remains unchanged.

## What was selected

From the original 56,644 development results (2001–2011), select **five distinct
entry/exit indicator pairs in each of nine family cells**. For each retained pair,
keep its development-selected confirmation waits. Selection uses finite net daily
cash-excess Sharpe only, with the original `1e-10` / stable-ID tie rule. Removing
other wait variants of an already selected pair prevents the shortlist consisting
mostly of near-identical timings. No positivity, drawdown, return or gap-flag
filter was added.

The entire set was sealed before the follow-up validation loaded. All 45 were
then evaluated in validation (2012–2015) and all four historical cost scenarios
(2016–May 28, 2026). **Validation did not eliminate any member; historical results
did not select a new winner.** Each option is a separate $1,000 QQQ/cash simulation,
not one component of a newly constructed ensemble portfolio.

## Baseline comparison: 1 bp commission + 3 bps adverse slippage per order

| Metric | Validation: 2012–2015 | Historical: 2016–May 28, 2026 |
|---|---:|---:|
| Candidate CAGR range | 1.78%–16.25% | 4.00%–17.73% |
| Median candidate CAGR | 5.73% | 14.32% |
| QQQ buy-and-hold CAGR | 19.88% | 20.96% |
| Candidates with higher CAGR than QQQ | 0/45 | 0/45 |
| Candidates with higher cash-excess Sharpe than QQQ | 0/45 | 22/45 |
| Candidates with shallower daily drawdown than QQQ | 13/45 | 45/45 |
| Median candidate drawdown loss | 15.14% | 18.00% |
| QQQ drawdown loss | 13.94% | 35.12% |
| Undefined candidate Sharpe scores retained | 0 | 0 |

**The broader set changes the conclusion about risk-adjusted performance, not raw
return:** 22 of 45 had higher historical cash-excess Sharpe than QQQ, and all 45
had shallower historical drawdowns. None had higher net CAGR. No candidate beat
QQQ on all three measures in either later segment. These descriptive comparisons
are not significance tests, proof of profitability or grounds to retrospectively
promote the best historical row to a validated winner.

Cross-candidate medians/ranges describe 45 separate strategies; they are **not
returns or drawdowns of an equally weighted portfolio**. Maximum drawdown uses
daily 17:00 NY portfolio values, not intraday lows. Comparison counts require an
improvement greater than `1e-10`.

## Same set under all four execution stresses

Commission remains 1 bp; slippage applies to every actual order across the entire
period, without hindsight-based regime labels. These are hypothetical stresses,
not measured QQQ execution costs.

| Historical slippage | Total nominal one-way cost | Median CAGR | CAGR range | Higher CAGR than QQQ | Higher Sharpe than QQQ | Shallower drawdown than QQQ |
|---|---:|---:|---:|---:|---:|---:|
| 1 bps | 2 bps | 14.67% | 4.34%–17.81% | 0/45 | 23/45 | 45/45 |
| 3 bps | 4 bps | 14.32% | 4.00%–17.73% | 0/45 | 22/45 | 45/45 |
| 10 bps | 11 bps | 13.10% | 2.80%–17.43% | 0/45 | 12/45 | 45/45 |
| 25 bps | 26 bps | 10.97% | -0.57%–16.78% | 0/45 | 4/45 | 45/45 |

The set was not replaced at higher costs. Every candidate, including the severe-
stress negative-return member, remains in the complete comparison files.

## All 45 options, in development selection order

Indicators below use **daily sessions**. `Entry / exit wait` gives additional
**trading hours after the first qualifying completed half-hour hit**, not indicator
lookbacks or execution latency. Every fill uses the next safe observed bar Open.
The full-precision files retain all risk, exposure, turnover, holding-time and
cost-drag metrics. The table does not rank options using validation or historical
outcomes.

| Family / dev rank | Entry | Exit | Entry / exit wait (h) | Validation CAGR | Historical CAGR | Historical cash-excess Sharpe | Historical drawdown loss |
|---|---|---|---:|---:|---:|---:|---:|
| SMA/SMA / 1 | SMA100 | SMA50 | 6.5 / 0 | 3.05% | 14.30% | 0.950 | 18.04% |
| SMA/SMA / 2 | SMA10 | SMA20 | 0.5 / 0 | 5.73% | 16.24% | 1.055 | 15.93% |
| SMA/SMA / 3 | SMA20 | SMA50 | 3 / 0 | 5.92% | 17.36% | 1.114 | 12.29% |
| SMA/SMA / 4 | SMA10 | SMA50 | 2 / 0 | 4.84% | 17.01% | 1.077 | 14.86% |
| SMA/SMA / 5 | SMA50 | SMA50 | 6.5 / 0 | 2.66% | 15.98% | 1.021 | 18.04% |
| SMA/EMA / 1 | SMA20 | EMA100 | 0 / 3 | 10.63% | 15.74% | 0.903 | 17.66% |
| SMA/EMA / 2 | SMA10 | EMA100 | 1.5 / 1 | 9.37% | 16.48% | 0.958 | 13.71% |
| SMA/EMA / 3 | SMA250 | EMA100 | 1.5 / 1 | 11.88% | 15.02% | 0.868 | 17.28% |
| SMA/EMA / 4 | SMA50 | EMA250 | 0 / 5.5 | 14.40% | 17.73% | 0.930 | 23.86% |
| SMA/EMA / 5 | SMA20 | EMA250 | 0 / 5.5 | 16.25% | 16.96% | 0.875 | 27.37% |
| SMA/MACD / 1 | SMA250 | MACD(96,208,72) | 1 / 0.5 | 4.32% | 8.80% | 0.502 | 25.18% |
| SMA/MACD / 2 * | SMA250 | MACD(12,26,9) | 0 / 0.5 | 8.12% | 7.30% | 0.494 | 16.28% |
| SMA/MACD / 3 | SMA100 | MACD(96,208,72) | 3.5 / 0 | 4.81% | 9.94% | 0.573 | 25.18% |
| SMA/MACD / 4 | SMA100 | MACD(48,104,36) | 6.5 / 6.5 | 8.03% | 11.74% | 0.789 | 20.65% |
| SMA/MACD / 5 | SMA250 | MACD(6,13,5) | 4.5 / 6.5 | 1.78% | 8.34% | 0.564 | 18.00% |
| EMA/SMA / 1 | EMA100 | SMA50 | 3 / 0 | 5.21% | 14.35% | 0.909 | 19.33% |
| EMA/SMA / 2 | EMA200 | SMA50 | 4.5 / 0 | 4.83% | 14.86% | 0.973 | 17.26% |
| EMA/SMA / 3 | EMA10 | SMA20 | 0.5 / 0 | 5.74% | 16.38% | 1.063 | 14.80% |
| EMA/SMA / 4 | EMA250 | SMA50 | 4.5 / 0 | 4.83% | 14.32% | 0.938 | 21.14% |
| EMA/SMA / 5 | EMA10 | SMA50 | 2.5 / 0 | 5.13% | 16.33% | 1.032 | 14.50% |
| EMA/EMA / 1 | EMA200 | EMA100 | 5.5 / 2.5 | 11.43% | 13.81% | 0.795 | 23.08% |
| EMA/EMA / 2 | EMA20 | EMA100 | 1.5 / 3 | 10.54% | 15.32% | 0.880 | 17.65% |
| EMA/EMA / 3 | EMA10 | EMA100 | 2 / 1.5 | 11.16% | 15.87% | 0.922 | 15.02% |
| EMA/EMA / 4 | EMA50 | EMA100 | 1 / 3 | 11.02% | 15.38% | 0.883 | 17.85% |
| EMA/EMA / 5 | EMA250 | EMA100 | 2 / 1 | 11.65% | 14.51% | 0.836 | 19.43% |
| EMA/MACD / 1 | EMA200 | MACD(96,208,72) | 1 / 0.5 | 4.32% | 9.29% | 0.532 | 25.18% |
| EMA/MACD / 2 | EMA250 | MACD(96,208,72) | 1 / 0.5 | 4.32% | 8.90% | 0.509 | 25.18% |
| EMA/MACD / 3 | EMA250 | MACD(6,13,5) | 5 / 6 | 2.59% | 8.60% | 0.581 | 15.97% |
| EMA/MACD / 4 | EMA250 | MACD(12,26,9) | 2 / 0.5 | 8.56% | 8.44% | 0.593 | 15.64% |
| EMA/MACD / 5 | EMA200 | MACD(6,13,5) | 5.5 / 6 | 1.86% | 7.46% | 0.495 | 17.74% |
| MACD/SMA / 1 | MACD(96,208,72) | SMA50 | 6.5 / 0 | 2.26% | 7.65% | 0.517 | 18.04% |
| MACD/SMA / 2 | MACD(96,208,72) | SMA200 | 1 / 0.5 | 11.21% | 16.47% | 0.930 | 17.92% |
| MACD/SMA / 3 | MACD(24,52,18) | SMA250 | 0 / 1.5 | 16.17% | 16.59% | 0.885 | 23.95% |
| MACD/SMA / 4 | MACD(12,26,9) | SMA50 | 3.5 / 0 | 4.24% | 16.91% | 1.098 | 14.95% |
| MACD/SMA / 5 | MACD(96,208,72) | SMA100 | 4 / 6.5 | 4.87% | 8.06% | 0.460 | 22.50% |
| MACD/EMA / 1 | MACD(96,208,72) | EMA100 | 2 / 1.5 | 4.45% | 8.75% | 0.519 | 18.42% |
| MACD/EMA / 2 | MACD(12,26,9) | EMA100 | 2 / 3.5 | 9.28% | 15.47% | 0.898 | 17.95% |
| MACD/EMA / 3 | MACD(96,208,72) | EMA200 | 1 / 6.5 | 9.03% | 14.24% | 0.786 | 20.37% |
| MACD/EMA / 4 | MACD(6,13,5) | EMA250 | 5 / 5.5 | 15.21% | 17.29% | 0.897 | 26.85% |
| MACD/EMA / 5 | MACD(24,52,18) | EMA100 | 1 / 3 | 7.80% | 13.49% | 0.795 | 17.85% |
| MACD/MACD / 1 | MACD(96,208,72) | MACD(48,104,36) | 0.5 / 6.5 | 6.69% | 9.01% | 0.618 | 17.98% |
| MACD/MACD / 2 | MACD(12,26,9) | MACD(96,208,72) | 0 / 0 | 4.77% | 9.65% | 0.553 | 25.18% |
| MACD/MACD / 3 | MACD(96,208,72) | MACD(12,26,9) | 3.5 / 0.5 | 3.49% | 4.00% | 0.245 | 11.01% |
| MACD/MACD / 4 | MACD(48,104,36) | MACD(96,208,72) | 0 / 0 | 4.98% | 9.18% | 0.522 | 25.18% |
| MACD/MACD / 5 | MACD(96,208,72) | MACD(96,208,72) | 2.5 / 0 | 4.82% | 9.15% | 0.520 | 25.18% |

`*` The development gap flag affects one shortlisted member, with two confirmed
post-gap decisions. That member was retained; flags never filtered selection.
The March 29, 2001 opening half-hour and November 30, 2004 session remain unknown
source gaps treated as disclosed no-trade blackouts, not repaired prices.

## Source and interpretation limits

- Unlevered 100% QQQ or cash; no taxes, financing or separate fund-expense charge.
- Alpha Vantage prices only, ending May 28, 2026. Intraday values are reconstructed
  as-traded units, not native raw captures or certified point-in-time history.
- Provider daily quotes are assumed available at 17:00 NY, not guaranteed official
  closing or executable quotes. Daily state finalizes once; intraday previews use
  yesterday's finalized state.
- Cash uses latest-vintage FRED DTB3 under modeled causal availability, not a traded
  Treasury-bill return series. Cash's cash-excess Sharpe is undefined, not zero.
- All portfolios share starting capital, corporate actions, daily valuation and
  terminal liquidation. Every segment resets positions/orders/streaks to cash.
- The follow-up shortlist rule was designed after the earlier phase outputs were
  seen. Development-only numerical selection and new seals constrain this run;
  they do not erase researcher familiarity or multiple-comparison risk. Fresh
  prospective evidence would require rules frozen before future observations.

## Inspectable evidence and reproducibility

- [All development-selected configurations and descriptive flags](development_shortlist.csv).
- [Every validation option plus QQQ/cash](validation_comparison.csv).
- [Every historical option and control at all four costs](historical_cost_comparison.csv).
- [All 45 baseline results joined across periods](validation_and_historical_baseline.csv).
- [Derived validation daily curves](validation_daily_curves.csv.gz) and
  [derived historical daily curves](historical_oos_daily_curves.csv.gz). No quotes
  or trade ledgers are exported.
- [Summary, fixed-set identity and full distribution statistics](summary.json).
- [Independent integrity and arithmetic audit](independent_audit.json): all 235
  curve groups reconcile; original sources/config/data and sealed reports remain
  unchanged. This is not an independent strategy/accounting reimplementation.
- [Artifact integrity hashes](artifact_hashes.json).
- [Shortlist design and operating rules](../../../docs/research_validation/shortlist_followup.md).

```bash
# Private licensed inputs and a genuine direct-user authorization receipt required.
python scripts/run_qqq_research.py shortlist seal --study-dir PRIVATE_NEW_DIRECTORY \
  --per-family 5 --authorization-receipt PRIVATE_USER_RECEIPT
python scripts/run_qqq_research.py shortlist validation --study-dir PRIVATE_NEW_DIRECTORY
python scripts/run_qqq_research.py shortlist historical-oos --study-dir PRIVATE_NEW_DIRECTORY
```

The original single-winner monitor is unchanged; this follow-up places no orders.
