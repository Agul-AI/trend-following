# Trend-Following Real Data — Project Log

This is the living research log for the `trend-following-real-data` project.
Each entry should record the research question, implementation/script, assumptions,
key result, and decision. The goal is to preserve the reasoning path, not just the
final answer.

Last updated: 2026-06-02.

---

## Current best candidate

As of 2026-06-02, the strongest synthetic-TQQQ candidate is:

```text
Traded asset: synthetic TQQQ from QQQ
Daily regime gate: QQQ daily 200MA trend regime, no variance-ratio trend rule
Entry: SMA-MACD day-equivalent histogram > 0
Entry source: QQQ
Exit: 200MA slow exit
Exit source: synthetic TQQQ
Sizing: 100% on entry
Profit lock: +200% unrealized gain -> 75%; +300% unrealized gain -> 50%
Execution: no-lookahead close-to-close convention, max one trade per day
Costs/tax: 5 bps slippage, 1 bp transaction cost, 24% short-term tax approximation
```

Latest metrics:

| Entry source | Exit source | Lock | Ann. return | Sharpe | Max DD | Trades | Exposure | DD episodes >30/>40/>50 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| QQQ | TQQQ | +200/+300 | 22.39% | 0.728 | -49.59% | 184 | 59.75% | 10/4/0 |
| TQQQ | TQQQ | +200/+300 | 22.32% | 0.725 | -49.59% | 188 | 60.02% | 9/3/0 |

Interpretation: QQQ entry is slightly smoother and marginally improves return/Sharpe,
while TQQQ-based exit remains more responsive to synthetic leveraged drawdown behavior.
The TQQQ/TQQQ version has one fewer >30% and >40% drawdown episode, so both remain
active candidates.

Main output:

- `reports/tables/tqqq_mixed_entry_exit_source_comparison_compact.csv`
- `reports/figures/tqqq_mixed_entry_exit_source_comparison_top_equity_drawdown.png`

Low-turnover alternative to keep watching:

```text
Entry source: QQQ
Exit source: QQQ
Lock: none
Ann. return: 22.34%
Sharpe: 0.701
Max DD: -58.86%
Trades: 101
Exposure: 64.92%
DD episodes >30/>40/>50: 13/6/4
```

This has almost half the trades of the leading locked mixed-source candidate and
nearly the same annualized return, but it has worse Sharpe, deeper max drawdown,
and more severe drawdown episodes.

Active three-candidate set for the next research steps:

| Candidate | Entry source | Exit source | Lock | Ann. return | Sharpe | Max DD | Trades | Exposure | DD >30/>40/>50 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| High-Sharpe mixed | QQQ | TQQQ | +200/+300 | 22.39% | 0.728 | -49.59% | 184 | 59.75% | 10/4/0 |
| Robust locked self-exit | TQQQ | TQQQ | +200/+300 | 22.32% | 0.725 | -49.59% | 188 | 60.02% | 9/3/0 |
| Low-turnover discretionary | QQQ | QQQ | none | 22.34% | 0.701 | -58.86% | 101 | 64.92% | 13/6/4 |

Because the user currently trades manually rather than through an automated live
algorithm, the low-turnover QQQ/QQQ full candidate remains important despite its
worse drawdown statistics.

Shared worst drawdown clusters across the three active candidates:

| Rank | Shared peak/start | Shared bottom | Avg DD | QQQ->TQQQ lock DD | TQQQ->TQQQ lock DD | QQQ->QQQ full DD |
|---:|---|---|---:|---:|---:|---:|
| 1 | 2004-01-20 | 2005-06-27 | -52.22% | -49.59% | -49.59% | -58.86% |
| 2 | 2020-02-19 | 2020-04-09 | -49.10% | -48.51% | -48.51% | -50.29% |
| 3 | 2021-11-22 | 2023-03-28 | -43.65% | -38.96% | -38.89% | -53.10% |
| 4 | 2018-10-01 | 2019-08-05 | -42.65% | -36.09% | -36.70% | -55.15% |
| 5 | 2015-07-20 | 2016-06-17 | -41.00% | -37.28% | -37.04% | -48.69% |
| 6 | 2020-09-03 | 2020-09-21 | -36.92% | -36.92% | -36.92% | -36.92% |

Detailed peak-to-bottom periods for each candidate are saved in:

- `reports/tables/tqqq_three_candidates_shared_worst_6_drawdowns_compact.csv`
- `reports/tables/tqqq_three_candidates_shared_worst_6_drawdowns.csv`

---

## Chronological research log

### 2026-05-28 to 2026-05-30 — Alpha Vantage bulk data acquisition

**Question.** Can we download a broad ETF/leveraged ETF universe at intraday
frequencies for research?

**Implementation.**

- Used Alpha Vantage premium API.
- Downloaded ETF data for 15min, 30min, 60min, and 1d.
- Added restart/resume behavior and monitoring.
- Repaired skipped/bad months where possible.

**Important findings.**

- Daily data is adjusted; intraday Alpha Vantage bars are generally unadjusted.
- The pipeline caches parquet files and resumes by skipping completed files.
- Missing/bad month repair can skip unavailable bad months while downloading the rest.

**Decision.** Keep raw Alpha Vantage data cached locally and handle adjusted intraday
processing as a post-processing concern if needed.

Key outputs:

- `reports/tables/alpha_vantage_integrity_summary.csv`
- `reports/tables/alpha_vantage_integrity_problems.csv`
- `reports/tables/alpha_vantage_skipped_months_repair.csv`

---

### 2026-05-30 — Crypto data download

**Question.** Can the same Alpha Vantage API download crypto intraday data?

**Implementation.**

- Added/used crypto download script for 15min, 30min, and 60min intervals.

**Decision.** Crypto support is useful as a future extension, but the active strategy
research stayed focused on QQQ/TQQQ.

Key outputs:

- `reports/tables/alpha_vantage_crypto_download_failures.csv`
- `reports/tables/data_validation_alpha_vantage_crypto_15min.csv`
- `reports/tables/data_validation_alpha_vantage_crypto_30min.csv`
- `reports/tables/data_validation_alpha_vantage_crypto_60min.csv`

---

### 2026-05-30 to 2026-06-01 — Regime module

**Question.** Can we classify QQQ into trend / mean-reversion / risk-off / neutral
without lookahead?

**Implementation.**

- Added daily-first regime module:
  - 200MA level and slope
  - variance ratio
  - realized volatility percentile
  - pullback z-score
- Added `regime_switch` strategy.
- Added no-lookahead regime estimation: today's regime estimate uses data through
  yesterday, then can be compared with today's close-based regime for diagnostics.

**Findings.**

- Variance-ratio trend confirmation was too restrictive for recent QQQ trend periods.
- Removing the VR rule from the trend regime improved activation.
- Regime classifications are useful for visualization and gating, but a pure
  mean-reversion leg did not meaningfully contribute.

**Decision.**

- Keep daily QQQ 200MA trend gate.
- Do not use variance-ratio as a required trend rule for the active strategy.
- Keep mean-reversion diagnostics but do not rely on the mean-reversion leg.

Key outputs:

- `src/trend_following/regime.py`
- `tests/test_regime.py`
- `reports/figures/qqq_price_by_daily_regime.png`
- `reports/tables/qqq_daily_regimes_sma200_no_vr.csv`

---

### 2026-06-01 — Hourly strategy with daily regime gate

**Question.** Can we use hourly data for trend following while keeping the daily
regime reference?

**Implementation.**

- Daily QQQ 200MA regime gate.
- Hourly fast-entry / slow-exit state machine.
- Reference windows expressed in trading-day equivalents on hourly bars.
- Max one trade per day.

**Finding.**

- Daily 200MA gate + hourly fast-entry/slow-exit worked better than the earlier
  pure daily `regime_switch`, which had too little exposure.

**Decision.** Use daily regime for macro trend gating, and hourly bars for entries/exits.

Key outputs:

- `scripts/run_fast_slow_state_machine.py`
- `reports/figures/synthetic_tqqq_fast_slow_qqq_regime_*`

---

### 2026-06-01 — Synthetic TQQQ construction

**Question.** Can we extend leveraged ETF history by constructing synthetic +3x
daily-reset versions from underlying ETFs?

**Implementation.**

- Added synthetic leverage construction from underlying returns.
- Created synthetic +3x variants:
  - `SPY_3X_CALC`
  - `QQQ_3X_CALC`
  - `IWM_3X_CALC`
  - `XLK_3X_CALC`
  - `XLF_3X_CALC`
  - `TLT_3X_CALC`
  - `EEM_3X_CALC`
  - `GLD_3X_CALC`

**Findings.**

- Synthetic QQQ/TQQQ-like exposure was strongest.
- XLK, SPY, and GLD were also promising.
- IWM, XLF, TLT, and EEM were weaker for this trend structure.

**Decision.** Focus active research on synthetic TQQQ / QQQ first.

Key outputs:

- `src/trend_following/synthetic_leverage.py`
- `scripts/create_synthetic_tqqq.py`
- `scripts/run_synthetic_3x_batch_fast_slow.py`
- `reports/tables/synthetic_3x_fast_slow_200_gate_compact_comparison.csv`

---

### 2026-06-01 — VIX and external risk filters

**Question.** Can VIX or external risk data warn earlier before large TQQQ drawdowns?

**Implementation.**

- Downloaded VIX from Yahoo Finance after Alpha Vantage VIX attempts were unavailable.
- Downloaded additional market indicators:
  - VXN
  - VIX3M
  - MOVE
  - TNX
- Tested filters:
  - VIX percentile
  - VIX absolute thresholds
  - VIX term structure
  - breadth proxy
  - HYG/LQD credit stress
  - SMH/QQQ leadership
  - defensive rotation
  - TLT shock
  - MOVE/TNX stress proxies

**Findings.**

- VIX percentile > 90% marginally improved Sharpe but increased trades materially.
- Most external filters reduced compounding or did not meaningfully reduce the worst
  drawdowns.
- None provided a strong enough improvement relative to added complexity.

**Decision.** Do not include VIX/external risk filters in the active strategy.

Key outputs:

- `scripts/download_vix_data.py`
- `scripts/download_market_indicators.py`
- `scripts/run_tqqq_vix_exit_experiments.py`
- `scripts/run_tqqq_external_risk_filter_comparison.py`
- `reports/tables/tqqq_vix_exit_experiments_metrics.csv`
- `reports/tables/tqqq_external_risk_filter_comparison_metrics.csv`

---

### 2026-06-01 — Slippage and tax assumptions

**Question.** How does the strategy behave with more realistic frictions?

**Implementation.**

- Added 5 bps slippage.
- Added 1 bp transaction cost.
- Added approximate 24% short-term tax model.
- Losses carry forward; gains taxed annually; final liquidation included.

**Findings.**

- Costs/taxes materially reduce compounding but strategy remains viable in the
  synthetic TQQQ tests.
- After-tax analysis is sensitive to trade frequency, so reducing unnecessary churn
  became a priority.

**Decision.** Use 5 bps slippage, 1 bp cost, and 24% short-term tax for current
comparisons unless explicitly testing pre-tax behavior.

Key outputs:

- `scripts/run_tqqq_tax_slippage_analysis.py`
- `reports/tables/tqqq_no_vix_tax_slippage_metrics.csv`

---

### 2026-06-01 — Drawdown episode analysis

**Question.** How often does the strategy experience large drawdowns?

**Implementation.**

- Counted drawdown episodes crossing thresholds once per drawdown/recovery cycle.
- Initially used >20%, then moved to >30%, >40%, and >50% thresholds.

**Findings.**

- >20% episodes were frequent enough that the threshold was not very discriminating
  for synthetic TQQQ.
- >30/>40/>50 episode counts are more useful for comparing severe drawdown behavior.

**Decision.** Current comparison tables report `>30/>40/>50` drawdown episodes.

Key outputs:

- `reports/tables/tqqq_no_vix_tax_slippage_drawdown_gt20_summary.csv`
- `reports/tables/tqqq_no_vix_tax_slippage_drawdown_gt20_episodes.csv`

---

### 2026-06-01 — MA derivative / MA health filters

**Question.** Can derivatives of the 200MA or 20/50MA combinations warn before
big drawdowns?

**Implementation.**

- Tested:
  - 200MA slope
  - 200MA slope percentile
  - 20MA and 50MA slope
  - price vs 20/50/200MA
  - trend score
  - fast-health score

**Findings.**

- Fast MA deterioration often warns before drawdowns.
- Hard binary exits based on these filters were too costly and often removed too
  much profitable exposure.

**Decision.** Use MA derivative/health information as diagnostics or possible sizing
inputs, not as hard all-or-nothing exits.

Key outputs:

- `scripts/run_tqqq_ma_derivative_filter_experiments.py`
- `reports/tables/tqqq_ma_derivative_filter_experiments_metrics.csv`

---

### 2026-06-01 — Tiered sizing and profit-lock ideas

**Question.** Should exposure be 0/25/50/75/100 instead of all-or-nothing?

**Implementation.**

- Tested daily MA-tier sizing:
  - trend score tiers
  - fast score tiers
  - slope percentile tiers
  - crash-defense tiers
- Tested trade-level sizing:
  - profit lock after large unrealized gains
  - peak-drawdown tiers

**Findings.**

- Daily MA-tier sizing created too many partial rebalances and tax events.
- Trade-level profit locks were much better because they changed exposure only
  during large winning trades.

**Decision.** Drop daily tiered sizing as the main path; keep trade-level profit lock.

Key outputs:

- `scripts/run_tqqq_tiered_sizing_experiments.py`
- `reports/tables/tqqq_tiered_sizing_experiments_metrics.csv`

---

### 2026-06-01 to 2026-06-02 — MACD entry vs 20MA/10MA entry

**Question.** Is MACD a better entry condition than price crossing 20MA or 10MA?

**Implementation.**

- Compared:
  - 10MA entry
  - 20MA entry
  - EMA-MACD
  - SMA-MACD
  - bar-based MACD
  - day-equivalent MACD
- Retained 100% entry; gradual entry was tested and rejected.

**Findings.**

- 10MA was better than 20MA.
- MACD variants were better than MA entries.
- Gradual entry increased trades and did not materially improve performance.
- Simple `histogram > 0` was sufficient; `hist rising` and `hist > 0 plus MACD > 0`
  added complexity without enough benefit.

**Decision.**

- Drop 10MA/20MA entry from active research.
- Keep only MACD histogram > 0.
- Prefer SMA-MACD day-equivalent histogram > 0 in the current candidate.

Key outputs:

- `scripts/run_tqqq_macd_entry_experiments.py`
- `scripts/run_tqqq_entry_signal_comparison.py`
- `reports/tables/tqqq_entry_signal_comparison_macd_only_compact.csv`

---

### 2026-06-02 — QQQ signal vs TQQQ signal

**Question.** Should MACD/200MA be computed on QQQ or synthetic TQQQ when trading
synthetic TQQQ?

**Implementation.**

- Compared:
  - signal from synthetic TQQQ, trade synthetic TQQQ
  - signal from QQQ, trade synthetic TQQQ

**Findings.**

- QQQ signals trade less and are smoother.
- Full no-lock QQQ signal sometimes improves raw annualized return.
- QQQ signal plus early profit lock can under-participate in long TQQQ rallies.

**Decision.** Test mixed signal sources: QQQ for entry, TQQQ for exit.

Key outputs:

- `scripts/run_tqqq_signal_source_comparison.py`
- `reports/tables/tqqq_signal_source_comparison_compact.csv`

---

### 2026-06-02 — Profit-lock thresholds

**Question.** Should the profit lock start later than +150/+250?

**Implementation.**

- Compared:
  - old lock: +150% -> 75%, +250% -> 50%
  - new lock: +200% -> 75%, +300% -> 50%
  - full no-lock

**Findings.**

- The old lock was too early and cut exposure during profitable trends.
- The new lock generally improved return and Sharpe relative to the old lock.
- New lock also improved large drawdown episode counts versus full no-lock in the
  best TQQQ-signal case.

**Decision.** Use:

```text
+200% unrealized gain -> 75%
+300% unrealized gain -> 50%
```

Key outputs:

- `scripts/run_tqqq_profit_lock_threshold_comparison.py`
- `reports/tables/tqqq_profit_lock_threshold_comparison_compact.csv`
- `reports/tables/tqqq_profit_lock_hist_gt_0_full_new_rows_with_hits_exposure.csv`

---

### 2026-06-02 — Mixed entry/exit signal source

**Question.** Can entry use QQQ MACD while exit uses TQQQ 200MA, or vice versa?

**Implementation.**

- Compared:
  - entry QQQ, exit TQQQ
  - entry TQQQ, exit TQQQ
  - entry QQQ, exit QQQ
  - entry TQQQ, exit QQQ
- Always traded synthetic TQQQ.
- Entry rule: SMA-MACD day-equivalent histogram > 0.
- New profit lock: +200% -> 75%, +300% -> 50%.

**Findings.**

| Entry source | Exit source | Lock | Ann. return | Sharpe | Max DD | Trades | Exposure | DD >30/>40/>50 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| QQQ | TQQQ | new lock | 22.39% | 0.728 | -49.59% | 184 | 59.75% | 10/4/0 |
| TQQQ | TQQQ | new lock | 22.32% | 0.725 | -49.59% | 188 | 60.02% | 9/3/0 |
| QQQ | QQQ | new lock | 21.73% | 0.718 | -58.86% | 109 | 58.68% | 7/5/2 |
| TQQQ | QQQ | new lock | 21.44% | 0.709 | -58.86% | 111 | 59.16% | 7/5/2 |

**Decision.**

- Keep the mixed-source strategy as the leading candidate:
  - QQQ entry
  - TQQQ exit
  - new +200/+300 profit lock
- Keep TQQQ/TQQQ as a close robustness candidate due to fewer >30/>40 drawdown
  episodes.
- Keep QQQ/QQQ full no-lock as a low-turnover alternative: it has similar annualized
  return and only 101 trades, but worse Sharpe and drawdown behavior.

Key outputs:

- `scripts/run_tqqq_mixed_entry_exit_source_comparison.py`
- `reports/tables/tqqq_mixed_entry_exit_source_comparison_compact.csv`
- `reports/tables/tqqq_mixed_entry_exit_source_comparison_ann_gt_22.csv`
- `reports/figures/tqqq_mixed_entry_exit_source_comparison_top_equity_drawdown.png`

---

## Rejected or de-emphasized ideas

| Idea | Reason |
|---|---|
| VIX hard exit | Added trades and complexity; only marginal improvement in selected cases. |
| External risk filters | Most reduced compounding or did not materially improve worst drawdowns. |
| Daily MA-tier sizing | Too much rebalance/tax drag. |
| Gradual entry 25/50/75/100 | More trades with little/no performance improvement. |
| 20MA entry | Underperformed MACD variants. |
| Variance-ratio trend requirement | Too restrictive for trend classification. |
| Mean-reversion leg | Did not contribute enough; trend leg dominates. |
| QQQ exit source | Smoother but exits too slowly for synthetic TQQQ drawdowns. |

---

## No-lookahead conventions to preserve

- Daily regime estimates used by intraday bars are lagged so date `D` only uses
  daily closes through `D-1`.
- Raw intraday signals are close-bar signals and must be shifted by the existing
  execution convention before returns are earned.
- Close-to-close returns cannot be earned by a signal observed at the same close.
- Max one trade per day remains active for intraday strategy variants.
- Profit-lock state uses only the traded asset's price path up to the current bar
  and is still passed through the same executable-position shift.

---

## Template for future entries

```markdown
### YYYY-MM-DD — Short experiment title

**Question.** What did we test?

**Implementation.**

- Script:
- Data:
- Key assumptions:

**Results.**

| Variant | Ann. return | Sharpe | Max DD | Trades | Exposure | DD >30/>40/>50 |
|---|---:|---:|---:|---:|---:|---:|

**Decision.** Keep / reject / needs more testing.

**Outputs.**

- `reports/tables/...`
- `reports/figures/...`
```

---

### 2026-06-02 — Shared worst-10 drawdowns vs Fed hiking cycles

**Question.** How many of the three candidate TQQQ strategy shared worst drawdowns occurred during Fed hiking cycles, and did any hiking cycle avoid a >30% shared drawdown?

**Implementation.**

- Used the three kept candidates:
  - QQQ entry + TQQQ exit + new +200/+300 profit lock
  - TQQQ entry + TQQQ exit + new +200/+300 profit lock
  - QQQ entry + QQQ exit + full no-lock
- Defined shared drawdown as the average of the three candidate drawdown series, then ranked peak-to-bottom episodes by worst average drawdown.
- Classified overlap with effective-date Fed hiking cycles:
  - 1999-2000 partial in this backtest
  - 2004-2006
  - 2015-2018
  - 2022-2023

**Results.**

- 4 of the shared worst 10 overlapped hiking cycles: ranks 1, 3, 4, 5.
- 3 of the shared worst 10 had the bottom inside a hiking cycle: ranks 1, 3, 5.
- Among full hiking cycles covered by the strategy test after 2000, all major hike cycles had at least one overlapping >30% shared drawdown.
- The partial 1999-2000 cycle is not a clean full-cycle test because the synthetic/intraday backtest starts in January 2000.

**Outputs.**

- `reports/tables/tqqq_three_candidates_shared_worst_10_drawdowns.csv`
- `reports/tables/tqqq_three_candidates_shared_worst_10_drawdowns_compact.csv`
- `reports/tables/tqqq_shared_worst_10_rate_hike_cycle_classification.csv`
- `reports/tables/tqqq_hike_cycle_gt30_shared_drawdown_summary.csv`

---

### 2026-06-02 — Shorter exit MA only during Fed hiking cycles

**Question.** If the strategy is in a Fed hiking cycle, does replacing the 200-day exit MA with 150/100/50-day exits help?

**Implementation.**

- Script: `scripts/run_tqqq_hiking_cycle_exit_ma_experiments.py`
- Tested the three kept candidates:
  - QQQ entry + TQQQ exit + new +200/+300 lock
  - TQQQ entry + TQQQ exit + new +200/+300 lock
  - QQQ entry + QQQ exit + full no-lock
- Entry remained MACD histogram > 0.
- Exit used 200-day MA normally, but during a Fed hiking-cycle flag used one of 150/100/50-day MA.
- The hiking-cycle flag is shifted by one observed trading day so intraday decisions on date D use only cycle status known through D-1.

**Results.**

- For the two locked TQQQ-exit candidates, shorter hiking-cycle exits made results worse: lower annualized return, lower Sharpe, more trades, and worse max drawdown.
- For the QQQ-entry/QQQ-exit full no-lock candidate, 150MA slightly reduced the single worst max drawdown, but increased trades and large-drawdown episode counts; 100/50 reduced return more.
- The practical conclusion is to keep the 200MA exit for the current leading locked candidates.

**Outputs.**

- `reports/tables/tqqq_hiking_cycle_exit_ma_experiments_compact.csv`
- `reports/tables/tqqq_hiking_cycle_exit_ma_experiments_deltas.csv`
- `reports/tables/tqqq_hiking_cycle_exit_ma_experiments_metrics.csv`
- `reports/tables/tqqq_hiking_cycle_exit_ma_experiments_after_tax_returns.csv`
- `reports/tables/tqqq_hiking_cycle_exit_ma_experiments_weights.csv`

---

### 2026-06-02 — Sell-timing plots for hiking-cycle shared drawdowns

**Question.** Keep only two candidate strategies and visualize the four shared worst drawdowns that overlapped Fed hiking cycles, with two months of context before/after, marking peak, bottom, and sell timing.

**Implementation.**

- Script: `scripts/plot_tqqq_hiking_dd_sell_timing.py`
- Strategies:
  - QQQ entry + TQQQ exit + new +200/+300 lock
  - QQQ entry + QQQ exit + full no-lock
- Drawdown periods: shared worst ranks 1, 3, 4, and 5 from the hiking-cycle overlap classification.
- Each plot uses synthetic TQQQ price on a regular price scale, strategy equity, a position panel, shared peak/bottom markers, entry/buy markers, and sell/reduce markers from executable weights.
- Window: full overlapping Fed hiking cycle plus two calendar months on each side, while preserving the shared drawdown peak/bottom when the drawdown starts before the hiking cycle.

**Outputs.**

- 8 plots in `reports/figures/tqqq_hiking_dd_sell_timing/`
- `reports/tables/tqqq_hiking_dd_sell_timing_manifest.csv`
- `reports/tables/tqqq_hiking_dd_sell_timing_events.csv`

**2026-06-02 update.** Rebuilt the sell-timing plots with regular-price scale instead of log scale, added entry/buy markers, and expanded each window to include the full overlapping Fed hiking cycle plus two months on both sides while preserving the shared drawdown peak/bottom when the drawdown started before the hiking cycle.

**2026-06-02 update 2.** Added NASDAQ-100 monthly trailing P/E values as a QQQ valuation proxy at the shared peak and bottom markers, added the P/E proxy values to the plot titles/annotations, saved the source values to `reports/tables/tqqq_hiking_dd_sell_timing_pe_proxy_values.csv`, and extended the left edge of each plot to include the last buy before the shared drawdown peak.

**2026-06-02 update 3.** Added Fed hike-signal announcement markers to the sell-timing plots using official FOMC statement dates: 2004-05-04 for measured-pace guidance, 2015-10-28 for next-meeting hike language, and 2022-01-26 for the statement that raising rates would soon be appropriate. Saved those dates and source URLs to `reports/tables/tqqq_hiking_dd_sell_timing_fed_hike_signal_events.csv`.

**2026-06-02 update 4.** Corrected the P/E annotations so “bottom PE” refers to the synthetic TQQQ price bottom inside the shared drawdown window, not the strategy equity bottom. The plots now mark and annotate the synthetic TQQQ price peak-before-trough and the synthetic TQQQ price trough, with QQQ P/E labels at those months.


**2026-06-02 update 5.** Relabeled all sell-timing figure annotations/titles so PE is explicitly shown as QQQ PE: “QQQ PE at TQQQ peak” and “QQQ PE at TQQQ bottom.”

**2026-06-02 update 6.** Reduced the sell-timing figure set to the preferred QQQ-entry/QQQ-exit no-lock candidate only, so there are now four plots. Added QQQ PE annotations at the plot start and every vertical reference line: plot start, Fed hike signal, hike start, hike end, shared peak, and shared bottom. All PE annotations are labeled simply as “QQQ PE”.

---

### 2026-06-02 — Switch synthetic TQQQ to QQQ during Fed hiking-cycle / high-QQQ-PE windows

**Question.** What if the preferred QQQ-entry/QQQ-exit synthetic TQQQ strategy switches into unlevered QQQ during Fed hiking-cycle windows when QQQ PE is above 30?

**Implementation.**

- Script: `scripts/run_tqqq_hiking_cycle_pe_switch_experiment.py`
- Base strategy: QQQ MACD-histogram entry + QQQ 200-day MA exit, no profit lock.
- Trading asset normally: synthetic TQQQ (`QQQ_3X_CALC`).
- Overlay: replace synthetic TQQQ weight with QQQ weight when the overlay flag is active.
- Costs/tax: 1 bp transaction cost, 5 bps slippage, 24% short-term tax approximation, same as recent TQQQ analyses.
- No-lookahead handling:
  - Fed announcement/hike-cycle flags are shifted by one observed trading day.
  - Raw desired weights still pass through the project’s executable-weight shift.
  - Primary dynamic valuation variant uses the last completed month’s QQQ PE proxy.
- QQQ PE proxy: Nasdaq-100 monthly trailing P/E from Trendonify, because QQQ tracks Nasdaq-100.
- Tested two interpretations:
  1. `dynamic_hike_and_pe_gt_threshold`: switch only while both hike-window and monthly QQQ PE > 30 are true.
  2. `whole_announced_cycle_if_entry_pe_gt_threshold`: if QQQ PE at the hike announcement month is > 30, switch to QQQ for that whole announced cycle.

**Results.**

| Variant | Ann. return | Sharpe | Max DD | Trades | DD >30/>40/>50 |
|---|---:|---:|---:|---:|---:|
| Base TQQQ only | 22.34% | 0.701 | -58.86% | 101 | 13/6/4 |
| Dynamic announced-cycle + prior-month PE>30 | 21.68% | 0.693 | -55.15% | 103 | 13/7/4 |
| Whole announced cycle if same-month announcement PE>30 | 23.23% | 0.726 | -55.15% | 101 | 14/7/4 |
| Whole announced cycle if prior-month announcement PE>30 | 21.67% | 0.696 | -55.15% | 102 | 13/7/4 |

**Decision.** The most promising interpretation is the same-month announcement-PE gate: it improved annualized return, Sharpe, and max drawdown without adding trades. However, it depends on having a contemporaneous QQQ PE estimate available at the announcement date; if only month-end backfilled P/E is available, the conservative prior-month variants are less attractive.

**Outputs.**

- `reports/tables/tqqq_hiking_cycle_pe_switch_experiment_compact.csv`
- `reports/tables/tqqq_hiking_cycle_pe_switch_experiment_metrics.csv`
- `reports/tables/tqqq_hiking_cycle_pe_switch_experiment_cycle_entry_pe_decisions.csv`
- `reports/figures/tqqq_hiking_cycle_pe_switch_experiment_equity_drawdown.png`

---

### 2026-06-02 — QQQ PE: daily proxy vs Alpha Vantage holdings snapshot

**Question.** Can we obtain daily QQQ PE up to yesterday by either calculating it ourselves or using Alpha Vantage, and how different are the two methods?

**Implementation.**

- Script: `scripts/compare_qqq_pe_methods.py`
- Option A: fact-sheet anchored daily proxy.
  - Anchor: Invesco QQQ Q1 2026 fact sheet P/E = 36.52 as of 2026-03-31.
  - Use Alpha Vantage QQQ daily close to infer an EPS proxy on 2026-03-31.
  - Daily PE proxy = daily QQQ close / anchor-implied EPS.
- Option B: Alpha Vantage holdings look-through snapshot.
  - `ETF_PROFILE` for QQQ holdings.
  - `OVERVIEW` for each holding's current `PERatio`.
  - Compute weighted harmonic PE across holdings with positive PE.
- No-lookahead note: Option A can be shifted one trading day for backtests. Option B is a point-in-time snapshot only after we save it; it is useful going forward but is not historical without daily snapshots.

**Results as of the latest available QQQ close through 2026-06-01.**

| Method | QQQ PE |
|---|---:|
| Option A fact-sheet anchored daily proxy | 47.00 |
| Option A previous-day-known value for 2026-06-01 decision | 46.72 |
| Option B Alpha Vantage holdings harmonic snapshot | 36.10 |

Alpha Vantage returned 103 QQQ holdings. 92 had positive usable PE values, covering 95.04% of portfolio weight. Excluded weight was 4.94%, mostly negative/no-earnings holdings and non-equity/cash/futures lines.

**Decision.** Option B aligns much better with current public QQQ PE snapshots than the stale Option A anchor. Option A is still useful historically if we have frequent monthly/quarterly PE anchors; a single 2026-03-31 anchor became stale quickly because QQQ price and constituent earnings/weights changed materially after quarter-end.

**Outputs.**

- `scripts/compare_qqq_pe_methods.py`
- `reports/tables/qqq_pe_method_comparison_summary.csv`
- `reports/tables/qqq_pe_method_comparison_option_a_daily_proxy.csv`
- `reports/tables/qqq_pe_method_comparison_option_b_av_holdings.csv`

**2026-06-02 update.** Dropped Option A as the preferred current QQQ PE method. The single fact-sheet anchor approach became stale too quickly and materially overestimated current QQQ PE. Going forward, use Option B-style look-through weighted harmonic PE snapshots for current/live QQQ PE, and only use historical PE anchors for backtests if frequent point-in-time monthly/quarterly anchors are available.

**2026-06-02 update 2.** For current/live QQQ PE, the preferred implementation is now Option B only: a saved daily weighted-harmonic look-through snapshot. Alpha Vantage Premium is convenient but not unique. Candidate free/freemium substitutes to evaluate next: StockAnalysis current QQQ PE/holdings page for a one-page snapshot check, Finnhub ETF holdings/basic-financials endpoints if available on the free tier, issuer holdings from Invesco plus a free constituent PE source, and a longer-term SEC EDGAR-based constituent EPS rebuild. These alternatives should be treated as current snapshots unless we archive them daily ourselves.

---

### 2026-06-02 — Option B QQQ PE daily history snapshot saved

**Question.** Use Option B to calculate QQQ daily P/E and save it.

**Implementation.**

- Script: `scripts/save_qqq_pe_option_b_snapshot.py`
- Method: Alpha Vantage `ETF_PROFILE` QQQ holdings + `OVERVIEW` constituent `PERatio` values.
- Portfolio P/E: weighted harmonic P/E normalized over holdings with positive usable P/E.
- Important no-lookahead rule: this is a point-in-time snapshot. It is saved as an append-only daily history and marked usable from the next business day. The script intentionally does not backfill older dates with current holdings/fundamentals.

**Result.**

| Snapshot date | Usable from | QQQ PE | Holdings | Included holdings | Included weight | Excluded weight |
|---|---|---:|---:|---:|---:|---:|
| 2026-06-02 | 2026-06-03 | 36.10 | 103 | 92 | 95.04% | 4.94% |

**Outputs.**

- `scripts/save_qqq_pe_option_b_snapshot.py`
- `data/processed/valuation/qqq_pe_option_b_daily_history.parquet`
- `reports/tables/qqq_pe_option_b_daily_history.csv`
- `reports/tables/qqq_pe_option_b_latest_summary.csv`
- `reports/tables/qqq_pe_option_b_latest_holdings.csv`
- Raw snapshot cache: `data/raw/valuation/alpha_vantage_option_b_snapshots/2026-06-02/`

---

### 2026-06-02 — Daily QQQ 200-day regime gate ablation

**Question.** Is the daily QQQ 200-day trend-regime gate actually needed if the preferred strategy already has a QQQ 200-day exit rule?

**Implementation.**

- Script: `scripts/run_tqqq_daily_gate_ablation.py`
- Base: preferred QQQ-entry/QQQ-exit no-lock synthetic TQQQ strategy.
- Compared three variants:
  1. Current daily 200-day regime gate.
  2. No daily gate, with only the QQQ hourly 200-day exit rule.
  3. No daily gate, but require QQQ to be above the same hourly 200-day MA for entry as well as using it for exit.
- Same cost/tax assumptions: 1 bp transaction cost, 5 bps slippage, 24% short-term tax approximation.

**Results.**

| Variant | Ann. return | Sharpe | Max DD | Trades | Exposure | DD >30/>40/>50 |
|---|---:|---:|---:|---:|---:|---:|
| Current daily 200d regime gate | 22.34% | 0.701 | -58.86% | 101 | 64.92% | 13/6/4 |
| No daily gate, exit only | 16.62% | 0.554 | -94.26% | 949 | 77.87% | 11/5/3 |
| No daily gate, hourly 200d entry gate | 23.76% | 0.716 | -57.69% | 109 | 68.97% | 17/9/7 |

**Interpretation.**

- The daily gate is not redundant with the exit rule if the exit rule is only used as an exit. Removing the daily gate while allowing MACD entries anywhere led to many whipsaws and a much worse max drawdown.
- If the hourly 200-day MA is also used as an entry gate, the daily gate can be simplified away in this test and annualized return/Sharpe improve slightly, but large drawdown episode counts worsen.
- This is a candidate simplification, not yet a confirmed preferred-rule change.

**Outputs.**

- `scripts/run_tqqq_daily_gate_ablation.py`
- `reports/tables/tqqq_daily_gate_ablation_compact.csv`
- `reports/tables/tqqq_daily_gate_ablation_metrics.csv`
- `reports/figures/tqqq_daily_gate_ablation_equity_drawdown.png`

**2026-06-02 update.** Decomposed why the no-daily-gate / hourly-200d-entry-gate variant had higher return. It is mostly **earlier entries**, not later exits. The hourly-entry-gate variant was long for 1,645 bars when the current daily-gate strategy was cash; 1,111 of those bars were earlier-entry stretches before the daily regime gate turned on. These earlier-entry stretches compounded to roughly +195% gross synthetic-TQQQ return, while later-exit stretches were a drag at roughly -24% gross. Outputs: `reports/tables/tqqq_daily_gate_ablation_state_attribution.csv`, `reports/tables/tqqq_daily_gate_ablation_hourly_only_stretches.csv`, and `reports/tables/tqqq_daily_gate_ablation_hourly_only_category_summary.csv`.

---

### 2026-06-02 — Serious candidate test: remove daily regime gate, use QQQ hourly 200MA entry/exit gate

**Question.** Test the simplified candidate seriously: remove the daily QQQ 200-day trend-regime gate, require QQQ hourly price to be above the QQQ hourly 200-day MA for entry, and exit when QQQ hourly price falls below the same MA.

**Implementation.**

- Script: `scripts/run_tqqq_hourly_200ma_gate_candidate.py`
- Compared:
  - Current preferred: daily QQQ 200-day regime gate + QQQ hourly MACD entry + QQQ hourly 200-day exit.
  - Candidate: no daily regime gate; QQQ hourly MACD histogram > 0 and QQQ hourly close > 200-day hourly MA for entry; QQQ hourly close < 200-day hourly MA for exit.
- Both use QQQ signals, synthetic TQQQ exposure, no profit lock, max one trade per day.
- Same assumptions: 1 bp transaction cost, 5 bps slippage, 24% short-term tax approximation.
- Split: in-sample through 2018-12-31, out-of-sample after 2018-12-31.

**Results.**

| Segment | Variant | Ann. return | Sharpe | Max DD | Trades | Exposure | DD >20/>30/>40/>50 |
|---|---|---:|---:|---:|---:|---:|---:|
| Full | Current preferred daily gate | 22.34% | 0.701 | -58.86% | 101 | 64.92% | 20/13/6/4 |
| Full | Candidate hourly 200MA gate | 23.76% | 0.716 | -57.69% | 109 | 68.97% | 26/17/9/7 |
| In-sample | Current preferred daily gate | 14.51% | 0.556 | -58.86% | 78 | 60.23% | 10/6/3/1 |
| In-sample | Candidate hourly 200MA gate | 14.35% | 0.542 | -57.69% | 88 | 64.73% | 15/10/6/4 |
| Out-of-sample | Current preferred daily gate | 44.92% | 0.994 | -53.10% | 23 | 76.94% | 11/8/2/2 |
| Out-of-sample | Candidate hourly 200MA gate | 51.56% | 1.069 | -52.66% | 21 | 79.83% | 13/8/2/2 |

**Interpretation.**

- The candidate improves full-sample annualized return, Sharpe, and max drawdown slightly.
- The candidate is weaker in-sample on return/Sharpe and has many more large drawdown episodes.
- The candidate is stronger out-of-sample after 2018, with higher return/Sharpe and similar worst drawdown.
- It remains a serious candidate, but the higher number of drawdown episodes means it should not automatically replace the current preferred rule without explicit confirmation.

**Decision.** Keep as a serious candidate, not yet confirmed as the preferred rule.

**Outputs.**

- `scripts/run_tqqq_hourly_200ma_gate_candidate.py`
- `reports/tables/tqqq_hourly_200ma_gate_candidate_compact.csv`
- `reports/tables/tqqq_hourly_200ma_gate_candidate_metrics.csv`
- `reports/tables/tqqq_hourly_200ma_gate_candidate_drawdown_episodes_gt30.csv`
- `reports/tables/tqqq_hourly_200ma_gate_candidate_annual_returns.csv`
- `reports/figures/tqqq_hourly_200ma_gate_candidate_equity_drawdown.png`

---

### 2026-06-02 — Recalculate kept TQQQ candidates with 3% risk-free cash return

**Question.** Assume out-of-market cash earns a 3% annual risk-free rate and redo the calculation for the three previous kept candidates plus the new hourly-200MA-gate candidate.

**Implementation.**

- Script: `scripts/run_tqqq_cash_yield_candidate_comparison.py`
- Candidates:
  1. QQQ entry + synthetic-TQQQ exit + +200/+300 profit lock.
  2. Synthetic-TQQQ entry + synthetic-TQQQ exit + +200/+300 profit lock.
  3. Preferred: QQQ entry + QQQ exit + no profit lock.
  4. New serious candidate: no daily gate; QQQ hourly 200MA entry/exit gate.
- Cash assumption: uninvested cash earns 3% annualized, converted to hourly bars using the configured 1512 bars/year annualization.
- Tax assumption: cash interest is taxed at the same 24% short-term tax rate in the after-tax approximation. This can be rerun with `--cash-interest-tax-rate 0` for pre-tax cash yield.

**Results.**

| Candidate | Ann. return with 3% cash | Zero-cash ann. return | Delta | Sharpe | Max DD | Trades | Avg cash weight | DD >20/>30/>40/>50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| New candidate: no daily gate + QQQ hourly 200MA gate | 24.68% | 23.76% | +0.92% | 0.734 | -56.36% | 109 | 31.03% | 26/17/9/6 |
| QQQ entry + TQQQ exit + +200/+300 lock | 23.57% | 22.39% | +1.18% | 0.753 | -48.83% | 184 | 40.15% | 22/10/4/0 |
| TQQQ entry + TQQQ exit + +200/+300 lock | 23.50% | 22.32% | +1.17% | 0.751 | -48.83% | 188 | 39.87% | 20/9/3/0 |
| Preferred: QQQ entry + QQQ exit + no lock | 23.37% | 22.34% | +1.03% | 0.721 | -58.19% | 101 | 35.08% | 21/14/7/5 |

**Interpretation.**

- Cash yield helps every strategy by roughly 0.9%-1.2% annualized.
- Locked candidates benefit most because they spend more time partially or fully in cash.
- The new hourly-200MA-gate candidate remains the highest annualized-return candidate.
- The QQQ-entry/TQQQ-exit locked candidate has the best Sharpe and much better worst max drawdown among these four.
- The preferred QQQ-entry/QQQ-exit no-lock candidate remains simplest and lowest-turnover among the synthetic-TQQQ strategies, but with 3% cash yield it is no longer the top return/Sharpe candidate.

**Outputs.**

- `scripts/run_tqqq_cash_yield_candidate_comparison.py`
- `reports/tables/tqqq_cash_yield_candidate_comparison_compact.csv`
- `reports/tables/tqqq_cash_yield_candidate_comparison_metrics.csv`
- `reports/tables/tqqq_cash_yield_candidate_comparison_after_tax_returns_with_cash_yield.csv`
- `reports/tables/tqqq_cash_yield_candidate_comparison_after_tax_returns_zero_cash.csv`
- `reports/figures/tqqq_cash_yield_candidate_comparison_equity_drawdown.png`

**2026-06-02 update.** Reduced the 3% cash-yield candidate comparison to only the current preferred QQQ-entry/QQQ-exit no-lock strategy and the new no-daily-gate QQQ hourly-200MA entry/exit candidate. Saved the focused comparison to `reports/tables/tqqq_cash_yield_preferred_vs_hourly_200ma_candidate.csv`. The 3% annualized return on out-of-market cash is now treated as the default evaluation assumption in `reports/preferred_strategy_rules.md`.

---

### 2026-06-02 — Promoted hourly-200MA-gate candidate to preferred and plotted worst 6 drawdowns vs hiking cycles

**Question.** Promote the new candidate to the preferred option, adjust the preferred-rule document, then plot the worst six drawdowns and analyze their relationship with Fed hiking cycles.

**Implementation.**

- Current preferred strategy is now:
  - QQQ signal source.
  - Synthetic TQQQ exposure.
  - No profit lock.
  - No daily regime gate.
  - Entry: QQQ hourly MACD histogram > 0 and QQQ hourly close > QQQ hourly 200-day MA.
  - Exit: QQQ hourly close < QQQ hourly 200-day MA.
  - Max one trade per day.
  - Out-of-market cash earns 3% annualized in evaluation.
- Updated: `reports/preferred_strategy_rules.md`
- Plot/analysis script: `scripts/plot_preferred_worst_drawdowns_hiking.py`
- Drawdowns are based on the after-tax return stream with the 3% cash return assumption.
- Fed hiking-cycle effective-date windows used:
  - 1999-06-30 to 2000-05-16, partial in this backtest.
  - 2004-06-30 to 2006-06-29.
  - 2015-12-17 to 2018-12-20.
  - 2022-03-17 to 2023-07-27.
- Fed announcement-to-cycle-end windows used for comparison:
  - 2004-05-04 to 2006-06-29.
  - 2015-10-28 to 2018-12-20.
  - 2022-01-26 to 2023-07-27.

**Worst 6 drawdowns.**

| Rank | Peak | Trough | Recovery | Max DD | Effective hiking-cycle relationship |
|---:|---|---|---|---:|---|
| 1 | 2007-10-31 | 2009-07-08 | 2009-12-24 | -56.36% | No overlap |
| 2 | 2010-04-23 | 2010-08-11 | 2011-02-14 | -52.15% | No overlap |
| 3 | 2018-10-01 | 2019-08-05 | 2020-02-10 | -52.04% | Overlapped late 2015-2018 hiking cycle; trough after cycle ended |
| 4 | 2021-11-22 | 2023-03-10 | 2023-06-15 | -51.33% | Overlapped 2022-2023 hiking cycle; trough during cycle |
| 5 | 2020-02-19 | 2020-04-07 | 2020-07-06 | -50.86% | No overlap |
| 6 | 2004-01-20 | 2005-10-28 | 2007-07-12 | -50.73% | Overlapped 2004-2006 hiking cycle; trough during cycle |

**Relationship summary.**

- 3 of the worst 6 drawdowns overlapped an effective Fed hiking cycle: ranks 3, 4, and 6.
- 2 of the worst 6 had the trough inside an effective hiking cycle: ranks 4 and 6.
- 3 of the worst 6 also overlapped a Fed announcement-to-cycle-end hiking window: ranks 3, 4, and 6.
- The single worst drawdown, 2007-2009, did not overlap the effective hiking cycle window itself, but it occurred after the 2004-2006 hiking cycle and during the financial-crisis unwind.
- Conclusion: hiking cycles are an important risk context but not a sufficient explanation for the largest strategy drawdowns. Major non-hiking shocks, especially the financial crisis and COVID crash, also dominate the worst drawdown list.

**Outputs.**

- `scripts/plot_preferred_worst_drawdowns_hiking.py`
- `reports/tables/preferred_hourly_200ma_worst6_drawdowns_hiking_analysis.csv`
- `reports/figures/preferred_hourly_200ma_worst6_drawdowns_hiking_analysis.png`
- Individual plots in `reports/figures/preferred_hourly_200ma_worst6_drawdowns/`
- Updated `reports/preferred_strategy_rules.md`
