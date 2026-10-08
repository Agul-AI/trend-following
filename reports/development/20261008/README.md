# Trend Following Study — Development Results

**Development only: January 2, 2001–December 30, 2011; 2,767 daily valuation sessions.**
All 56,644 frozen configurations were evaluated once using Alpha-only QQQ prices,
100% QQQ or cash, and the same **1-bp commission + 3-bp adverse slippage** per
one-way order. Both strategy and benchmark start with $1,000 and use the same
corporate-action, cash, 17:00 NY valuation and terminal-liquidation conventions.
No taxes, financing or separate fund-expense charge is modeled.

## What the development sweep shows

- **47,526/56,644 (83.90%)** had higher net CAGR than QQQ buy-and-hold.
- **35,529 (62.72%)** had higher daily cash-excess Sharpe. The same number beat BH on CAGR, Sharpe and drawdown together.
- **56,550 (99.83%)** had a shallower daily maximum drawdown.

| Portfolio | Net CAGR | Max drawdown loss | Cash-excess Sharpe | Annualized volatility | Ending $1,000 |
|---|---:|---:|---:|---:|---:|
| Development primary-score leader | 8.86% | 14.17% | 0.583 | 12.54% | $2,544.78 |
| QQQ buy-and-hold | -0.04% | 70.40% | 0.078 | 29.45% | $995.52 |
| Causal cash proxy | 2.00% | 0.00% | — | 0.15% | $1,243.89 |

**This does not establish out-of-sample superiority.** The leader was selected
from 56,644 variants on the same development history used to measure these
results. The 0.583 Sharpe is not presented as a strong stand-alone selling point.
Validation (2012–2015) and historical OOS have **not** been evaluated. Nine
finite family finalists are sealed; the leader below is not yet a validation
winner. Counts include all configurations; no gap-flag exclusions were applied.

## Development leader, not a validated trading recommendation

- Entry indicator: standalone daily MACD **(96,208,72)**, histogram >0.
- Entry also requires the exit-side support: provisional daily signal > **SMA50**.
- Entry wait: **6.5 trading hours after the first qualifying close**, i.e. 14
  consecutive completed half-hour hits. EMA/MACD daily states advance only once
  per daily session, not every half-hour.
- Exit: provisional daily signal at/below **SMA50**, with **0 additional hours**
  (the first completed failing close). Fill at the next safe observed bar Open.
- Exposure: **36.47%** of elapsed calendar time; **47 round trips** (4.27/year); 94 one-way orders.
- Average holding duration: **31.15 calendar days**.
- Direct modeled commissions/slippage paid: **$58.41**. Gross-vs-net terminal wealth differs by **$97.51**, including compounding/opportunity effects—not the same measure.

## Missing-data flags and assumptions

The March 29, 2001 opening half-hour and November 30, 2004 full session remain
missing (14 half-hours). Holdings were carried through those gaps; there were
no invented quotes/trades. Orders were cancelled and confirmations restarted.
Daily values still use genuine provider quotes at the assumed 17:00 NY clock.

**2,691 candidates (4.75%)** had **3,012 confirmed decisions** after a gap and through the next trading-session close. All events are retained without truncation. The leader and all nine family finalists have zero such flags, but that was **not** a selection criterion and does not prove gap insensitivity.

Intraday prices are verified reconstructed as-traded Alpha values, not native
raw captures or certified point-in-time vendor history. The cash proxy is
latest-vintage FRED DTB3 with modeled causal release lags; it is not a traded
bill return series. Slippage scenarios are hypothetical, not measured execution
costs. Maximum drawdown is measured on daily portfolio values, not intraday lows.

## Inspectable current-run evidence

- `metrics.csv.gz`: all 56,644 configuration results, including poor results.
- `post_gap_candidate_flags.csv.gz`: one descriptive flag/count row per candidate.
- `post_gap_signal_events.csv.gz` and `gap_diagnostics.json`: all 3,012 decisions
  and exact next-session-close windows.
- `family_finalists.csv`: the nine sealed family selections; flags never enter ranking.
- `comparison.csv` / `summary.json`: exact cost-matched comparison and definitions.
- `daily_portfolio_curves.csv`: derived leader/BH/cash curves; no raw quote export.
- `independent_bh_ledger_audit.json`: an independent passive-share/dividend ledger
  matches all 2,767 BH daily values within 4.55e-13; the near-zero BH CAGR is
  explained by the exact chosen opening-to-terminal horizon, not a price-unit error.
- `independent_audit.json`: independent full-universe/seal/flag/finalist checks
  and curve-derived CAGR/Sharpe/volatility/drawdown reconciliation.
- `artifact_hashes.json`: public artifact integrity hashes.

Only current development outcomes and current price inputs informed this run.
No other study's outcomes or selections were used. Prior repository files were
removed from the current tree; Git history remains as explicitly requested.

## Software and publication checks

The current-only suite passes **203 software tests**. These are separate from
the **56,644 actual development simulations**. Ruff and whitespace checks pass.
All eight frozen implementation files and the immutable current configuration
remain byte-identical; price inputs/authorizations remain private.
