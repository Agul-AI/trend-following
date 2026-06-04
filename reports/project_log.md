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

---

### 2026-06-02 — Reframed repository presentation around the QQQ/synthetic-TQQQ case study

**Question.** The repository originally read like a general ETF trend-following project, but the current research focus has moved toward the QQQ/synthetic-TQQQ strategy. How should the repo present this without losing the broader pipeline value?

**Implementation.**

- Updated `README.md` to present the project in two layers:
  - a general ETF data/backtesting/research framework, and
  - a focused QQQ-driven synthetic-TQQQ flagship case study.
- Added `docs/qqq_tqqq_case_study.md` with the research question, current preferred rule, no-lookahead convention, latest compact metrics, worst drawdown/hiking-cycle context, reproduction commands, failed experiments, and limitations.
- Kept `reports/preferred_strategy_rules.md` as the source of truth for confirmed preferred strategy rules.
- Kept `reports/project_log.md` as the chronological record of the research process.

**Interpretation.**

This framing is stronger for GitHub/interviews: the project demonstrates generalizable quant-engineering skill while also showing focused research depth on the QQQ/synthetic-TQQQ strategy.

---

### 2026-06-02 — Added current preferred-strategy signal updater

**Question.** Based on the most recent hourly data, should the current preferred QQQ/synthetic-TQQQ strategy be long or cash, and how can the signal be refreshed every trading hour?

**Implementation.**

- Added `scripts/update_preferred_strategy_signal.py`.
- The script downloads the latest Alpha Vantage QQQ 60-minute bars for the current/prior month, refreshes QQQ daily adjusted data, rebuilds `QQQ_3X_CALC`, computes the current preferred hourly-200MA-gated MACD signal, and writes:
  - `reports/tables/preferred_strategy_current_signal.csv`
  - `reports/tables/preferred_strategy_current_signal_history.csv`
- The report separates the latest raw desired signal from the no-lookahead executable position.

**Latest run.**

- Latest available Alpha Vantage hourly QQQ bar at run time: `2026-06-01 15:00:00`.
- Executable target position: long synthetic TQQQ exposure.
- Approximate QQQ hourly 200-day MA exit trigger at that bar: `617.47`, with QQQ close `742.69`.

---

### 2026-06-02 — Applied current and previous preferred rules to actual TQQQ hourly data

**Question.** What happens if the current and previous preferred QQQ-signal rules are applied to actual TQQQ, instead of synthetic TQQQ, over actual TQQQ's available Alpha Vantage 60-minute history?

**Implementation.**

- Refreshed cached actual TQQQ 60-minute data for May/June 2026.
- Ran `scripts/run_tqqq_cash_yield_candidate_comparison.py` with:
  - `--target-ticker TQQQ`
  - `--target-raw-dir data/raw/alpha_vantage_60min`
  - `--benchmark-ticker QQQ`
  - `--output-prefix actual_tqqq_current_previous_preferred_comparison`
- Used the current default evaluation assumptions: 1 bp transaction cost, 5 bps slippage, 24% short-term tax approximation, 3% annualized return on out-of-market cash.
- Actual TQQQ common hourly period: 2010-02-11 10:00 to 2026-06-01 15:00.

**Results.**

| Strategy | Cumulative return | Ann. return | Sharpe | Max DD | Trades | Exposure | DD >20/>30/>40/>50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current preferred: no daily gate + QQQ hourly 200MA gate | 9,115.18% | 32.07% | 0.853 | -53.34% | 61 | 78.85% | 18/13/5/4 |
| Previous preferred: daily QQQ regime gate + QQQ exit | 6,055.07% | 28.83% | 0.793 | -55.69% | 75 | 79.48% | 17/12/5/2 |
| Buy & hold actual TQQQ, pretax | 43,847.91% | 45.39% | 0.932 | -82.08% | 0 | 100% | not counted |
| Buy & hold QQQ, pretax | 1,891.50% | 20.20% | 1.038 | -36.10% | 0 | 100% | not counted |

**Interpretation.**

- Actual TQQQ buy-and-hold delivered much higher return during the available 2010-2026 bull-heavy sample, but with an extreme -82% max drawdown.
- The current preferred rule outperformed the previous preferred rule on actual TQQQ in annualized return, Sharpe, max drawdown, and trade count.
- The current preferred rule still had large drawdowns above 50%, so it reduces but does not remove leveraged-ETF crash risk.

**Outputs.**

- `reports/tables/actual_tqqq_current_previous_preferred_summary.csv`
- `reports/tables/actual_tqqq_current_previous_preferred_comparison_compact.csv`
- `reports/tables/actual_tqqq_current_previous_preferred_comparison_metrics.csv`
- `reports/figures/actual_tqqq_current_previous_preferred_equity_drawdown.png`

---

### 2026-06-02 — Documented hourly update workflow and created Codex updater skill

**Question.** How should a separate thread update QQQ/TQQQ data and rerun the current preferred strategy every trading hour?

**Implementation.**

- Added `docs/hourly_update_workflow.md` with:
  - the one-shot manual update command,
  - interpretation of `preferred_strategy_current_signal.csv`,
  - a local macOS cron schedule for weekday trading-hour updates,
  - operational cautions about no-lookahead interpretation and delayed vendor data.
- Created a reusable Codex skill:
  - `$CODEX_HOME/skills/qqq-tqqq-hourly-updater`
- The skill tells a future thread to run:

```bash
cd /path/to/trend-following
source ~/.venvs/myenv/bin/activate
python scripts/update_preferred_strategy_signal.py --pause-seconds 0.85
cat reports/tables/preferred_strategy_current_signal.csv
```

**Interpretation.**

A separate monitoring thread should use the no-lookahead `executable_position_latest_bar` field as the current long/cash state, not the unshifted raw signal. It should always report the actual `asof_intraday_bar` because Alpha Vantage data can be delayed.

---

### 2026-06-02 — Expanded hourly updater to include actual TQQQ and QQQ P/E history

**Question.** When updating data hourly, also refresh actual TQQQ data and QQQ P/E ratio, and make sure each refreshed dataset is merged into the project's local history.

**Implementation.**

- Updated `scripts/update_preferred_strategy_signal.py` so the normal update now refreshes:
  - QQQ 60-minute bars from Alpha Vantage into `data/raw/alpha_vantage_60min/QQQ.parquet`.
  - Actual TQQQ 60-minute bars from Alpha Vantage into `data/raw/alpha_vantage_60min/TQQQ.parquet`.
  - QQQ daily adjusted OHLCV into `data/raw/alpha_vantage_daily_adjusted/QQQ.parquet`.
  - Synthetic +3x QQQ into `data/raw/synthetic_3x_60min/QQQ_3X_CALC.parquet` and daily synthetic history.
  - QQQ Option-B P/E into `data/processed/valuation/qqq_pe_option_b_daily_history.parquet` and `reports/tables/qqq_pe_option_b_daily_history.csv`.
- Added actual TQQQ and QQQ P/E fields to `reports/tables/preferred_strategy_current_signal.csv` and `reports/tables/preferred_strategy_current_signal_history.csv`.
- Updated `docs/hourly_update_workflow.md` and the local Codex skill `$CODEX_HOME/skills/qqq-tqqq-hourly-updater`.

**Operational detail.**

- QQQ and actual TQQQ hourly prices are refreshed every updater run by downloading recent monthly slices and merging/deduplicating into parquet history.
- QQQ P/E is a point-in-time holdings/fundamentals snapshot, not a true hourly data series. The updater therefore creates/reuses one P/E snapshot per local calendar date by default. Use `--force-pe` only when intentionally refetching the same day's P/E snapshot.

**Latest verification run.**

- Latest QQQ bar: `2026-06-01 15:00:00`.
- Latest actual TQQQ bar: `2026-06-01 15:00:00`.
- Latest QQQ Option-B P/E snapshot date: `2026-06-02`, P/E `36.104342`, usable from `2026-06-03` under the no-lookahead convention.

---

### 2026-06-02 — Switched hourly updater from Alpha Vantage to Yahoo Finance/yfinance

**Question.** The hourly updater should not rely on Alpha Vantage because the subscription may be canceled. Use Yahoo Finance data instead, while still merging the newest data into the project's own local history.

**Implementation.**

- Rewrote `scripts/update_preferred_strategy_signal.py` to use Yahoo Finance through `yfinance` only.
- The updater no longer requires `ALPHA_VANTAGE_API_KEY`.
- New local Yahoo/yfinance histories:
  - QQQ 60-minute bars: `data/raw/yfinance_60min/QQQ.parquet`
  - Actual TQQQ 60-minute bars: `data/raw/yfinance_60min/TQQQ.parquet`
  - QQQ daily bars: `data/raw/yfinance_daily/QQQ.parquet`
  - Synthetic +3x QQQ from Yahoo QQQ: `data/raw/synthetic_yfinance_3x_60min/QQQ_3X_CALC.parquet` and `data/raw/synthetic_yfinance_3x_1d/QQQ_3X_CALC.parquet`
  - Yahoo-reported QQQ trailing P/E snapshot history: `data/processed/valuation/qqq_pe_yfinance_snapshot_history.parquet` and `reports/tables/qqq_pe_yfinance_snapshot_history.csv`
- Updated `docs/hourly_update_workflow.md` and `$CODEX_HOME/skills/qqq-tqqq-hourly-updater`.

**Important inconsistency/inaccuracy notes.**

- Yahoo 60-minute data usually has seven regular-session bars per full day (`09:30`, `10:30`, ..., `15:30`), while the prior Alpha Vantage 60-minute research cache used six bars/day (`10:00`, ..., `15:00`). The updater therefore uses `bars_per_day=7`, so the current monitoring signal can differ from Alpha Vantage-based backtest outputs.
- Yahoo intraday history is limited, usually around 730 days. That is enough for the current 200-trading-day hourly MA monitor, but not enough to reproduce the full long-history research backtest from Yahoo alone.
- Yahoo's QQQ P/E is a vendor-reported ETF `trailingPE` snapshot, not the prior transparent Alpha Vantage holdings-level harmonic P/E calculation. It may differ from the prior Option-B estimate.
- Yahoo/yfinance is an unofficial interface and can have delays, temporary failures, or revised data. The updater always reports the actual `asof_intraday_bar`.

**Latest verification run.**

- Latest Yahoo QQQ hourly bar: `2026-06-02 13:30:00`.
- Latest Yahoo actual TQQQ hourly bar: `2026-06-02 13:30:00`.
- Current executable position: long synthetic TQQQ exposure.
- Yahoo QQQ trailing P/E snapshot: approximately `36.315`.

---

### 2026-06-02 — Added GitHub-facing performance tables for preferred strategy and two retained candidates

**Question.** Include the performance test results for the current preferred strategy and the other two retained candidate strategies in GitHub.

**Implementation.**

- Added curated retained-candidate performance tables:
  - `reports/tables/qqq_tqqq_retained_candidate_performance_synthetic.csv`
  - `reports/tables/qqq_tqqq_retained_candidate_performance_actual_tqqq.csv`
- Updated `docs/qqq_tqqq_case_study.md` with two GitHub-readable tables:
  - Long-history synthetic `QQQ_3X_CALC` test.
  - Actual TQQQ available-history sanity check.
- Included the existing actual-TQQQ equity/drawdown plot as a case-study figure.

**Retained candidate set.**

1. Current preferred: QQQ hourly 200MA gate, no daily gate, no profit lock.
2. Candidate A: QQQ entry + TQQQ/synthetic exit + +200/+300 profit lock.
3. Candidate B: TQQQ/synthetic entry + TQQQ/synthetic exit + +200/+300 profit lock.

**Main interpretation.**

The current preferred strategy has the best return and lowest trade count among the retained candidates, while the two profit-lock alternatives reduce drawdown but require substantially more trades.

### 2026-06-02 — QQQ P/E rolling percentiles at preferred-strategy worst drawdowns

**Question.** For the six worst drawdowns of the current preferred hourly-200MA-gate synthetic-TQQQ strategy, what was QQQ P/E's percentile within its trailing 3-year and 5-year history at the strategy peak and bottom, and can this improve the strategy?

**Method.** Used a complete monthly Nasdaq-100/QQQ P/E history from WorldPERatio as a QQQ valuation proxy because the local Option-B look-through QQQ P/E history only contains current/live snapshots. For no-lookahead alignment, each intraday event date uses the previous completed calendar month's P/E. Rolling percentiles are computed within the previous 36 or 60 monthly observations, including the known lookup month.

**Outputs.**
- Saved monthly P/E history: `reports/tables/qqq_pe_worldperatio_monthly_history.csv`.
- Saved event table: `reports/tables/preferred_hourly_200ma_worst6_qqq_pe_percentiles.csv`.
- Saved simple PE-filter sensitivity table: `reports/tables/preferred_hourly_200ma_pe_filter_sensitivity.csv`.

**Result.** High QQQ P/E percentile was not a reliable standalone warning across the six worst drawdowns. Only 1/6 peaks were above the trailing 3-year 80th percentile; 3/6 were above the trailing 5-year 80th percentile; only 1/6 was above the trailing 5-year 90th percentile. P/E usually fell by the bottom, but 2007-2009 is an exception where the P/E percentile rose into the bottom because earnings fell faster than price.

**Sensitivity check.** Simple rules that cut exposure to cash or half-size when QQQ P/E percentile was high reduced annualized return materially and did not consistently improve max drawdown. The base preferred strategy remained best in annualized return in this quick sensitivity. Conclusion: do not add a hard P/E percentile exit/filter yet. If valuation is used later, treat it as a slow sizing/context variable, possibly combined with macro/liquidity/hiking-cycle information, not as an exclusive sell signal.

### 2026-06-02 — Cross-check free sources for QQQ trailing P/E history and current updates

**Question.** Cross-check options for historical and up-to-date QQQ trailing P/E, try Alpha Vantage for history, and choose the best free option.

**Checks.**
- Alpha Vantage `ETF_PROFILE` for QQQ returned current ETF metrics and holdings, but not historical QQQ P/E.
- Alpha Vantage fund-level `OVERVIEW`, `EARNINGS`, and `INCOME_STATEMENT` for QQQ returned empty objects in the local test; constituent-level data exists, but reconstructing historical QQQ P/E without point-in-time holdings would be lookahead-biased.
- WorldPERatio provides a free QQQ/Nasdaq-100 monthly P/E history and current monthly value.
- StockAnalysis provides a free current QQQ PE value and holdings as-of date.
- Yahoo Finance via yfinance provides a free current trailingPE snapshot suitable for daily local archival, but methodology is vendor black-box and unofficial.

**Decision.** Best free historical source: WorldPERatio monthly QQQ/Nasdaq-100 P/E history. Best free up-to-date automation source: Yahoo/yfinance daily `trailingPE`, cross-checked against StockAnalysis and WorldPERatio. For the most robust workflow, append Yahoo/yfinance current P/E daily into our local snapshot history and periodically cross-check against StockAnalysis/WorldPERatio; do not use Alpha Vantage for ongoing updates after cancellation.

**Output.** `reports/tables/qqq_pe_source_crosscheck_2026_06_02.csv`.

### 2026-06-02 — Benchmark reporting convention

**User preference.** In future strategy comparison tables, always include the QQQ buy-and-hold option.

**Decision.** QQQ BH is now the mandatory unlevered benchmark row for all QQQ/TQQQ candidate comparison tables over the same sample window. TQQQ BH can be optional, but QQQ BH should not be omitted.

**2026-06-02 update — comparison table convention applied.** Updated the P/E filter sensitivity comparison output to include `final_return` (cumulative ending return) and an explicit `QQQ_BH` benchmark row. Output table: `reports/tables/preferred_hourly_200ma_pe_filter_sensitivity.csv`.

### 2026-06-02 — Apply current preferred hourly-200MA strategy to synthetic +3x S&P 500

**Question.** Apply the current preferred QQQ/synthetic-TQQQ rule to 3x S&P 500 over the same historical period and include QQQ buy-and-hold in the comparison table.

**Implementation.** Added `scripts/run_spy_3x_preferred_strategy.py`. The SPY analogue uses SPY hourly data for both entry and exit signals, trades synthetic `SPY_3X_CALC`, uses MACD histogram > 0 plus SPY hourly 200-day MA entry gate, exits when SPY falls below the same hourly 200-day MA, and keeps the same no-lookahead executable-weight shift, max-one-trade-per-day convention, 1 bp transaction cost, 5 bps slippage, 24% short-term tax approximation, and 3% out-of-market cash yield. The table includes QQQ BH per the benchmark convention.

**Output.** `reports/tables/spy_3x_preferred_strategy_comparison.csv`.

**Result.** Over 2000-01-03 10:00 to 2026-05-28 15:00, synthetic `SPY_3X_CALC` with the preferred rule returned 5,835.0% cumulative / 16.78% annualized with Sharpe 0.648 and max drawdown -60.44%. It beat SPY 3x buy-and-hold on return and drawdown, but materially lagged the QQQ 3x preferred-strategy reference.

### 2026-06-02 — Profit-lock variants applied to current preferred strategy

**Question.** Apply possible profit-lock rules to the current preferred strategy.

**Implementation.** Added `scripts/run_preferred_profit_lock_comparison.py`. Base is the current preferred no-daily-gate QQQ hourly 200MA entry/exit strategy trading synthetic TQQQ. Tested a grid of intra-trade profit locks: after synthetic-TQQQ unrealized trade gain reaches the first threshold, reduce exposure to 75%; after the second threshold, reduce exposure to 50%. Threshold grid: first threshold +100%, +150%, +200%, +250%; second threshold +200%, +250%, +300%, +400%, +500%, with second > first. All variants keep the no-lookahead executable-weight shift, max-one-trade-per-day convention, 1 bp transaction cost, 5 bps slippage, 24% short-term tax approximation, and 3% annual cash yield. QQQ BH is included as the required benchmark row.

**Outputs.**
- `reports/tables/preferred_profit_lock_comparison_compact.csv`
- `reports/tables/preferred_profit_lock_comparison_metrics.csv`
- `reports/tables/preferred_profit_lock_comparison_returns.csv`
- `reports/tables/preferred_profit_lock_comparison_weights.csv`
- `reports/figures/preferred_profit_lock_comparison_equity_drawdown.png`

**Result.** The best final-return variant was `lock_250_500_to_75_50` with +24.95% annualized return vs +24.79% for no-lock base, Sharpe 0.755 vs 0.736, and the same -56.36% max drawdown. It reduced DD episode counts from 26/17/9/6 to 24/12/7/4 for >20/>30/>40/>50%. The best Sharpe variant was `lock_100_400_to_75_50` with Sharpe 0.766 but lower annualized return of 24.46%. Interpretation: profit locks can slightly improve Sharpe and reduce the number of large drawdown episodes, but they did not reduce the single worst max drawdown. Treat `lock_250_500_to_75_50` as a candidate enhancement, not a confirmed rule yet.

**2026-06-02 update — +300/+500 profit lock added.** Added `+300% unrealized trade gain -> 75% exposure` and `+500% unrealized trade gain -> 50% exposure` to the preferred profit-lock grid. It became the best final-return variant in this test: final return 39,250.8%, annualized return 25.47%, Sharpe 0.765, max drawdown -56.36%, 114 trades, exposure 65.90%, and DD episode counts 24/13/7/4 for >20/>30/>40/>50%. It improves annualized return and Sharpe versus the no-lock base but still does not improve the single worst max drawdown.

**2026-06-02 update — +300-only, +400-only, and +400/+600 locks tested.** Added single-step `+300% -> 75%`, single-step `+400% -> 75%`, and two-step `+400% -> 75%, +600% -> 50%`. The `+400% -> 75%, +600% -> 50%` result equals `+400% -> 75%` because the +600% threshold was never hit. Among the newly requested cases, `+400% -> 75%` produced final return 37,647.6%, annualized return 25.27%, Sharpe 0.749, max drawdown -56.36%, 111 trades, exposure 68.25%, DD episode counts 25/16/9/5. However, `+400% -> 75%, +500% -> 50%` remains the best final-return variant in the expanded grid: final return 41,145.6%, annualized return 25.69%, Sharpe 0.758, max drawdown -56.36%, 112 trades, exposure 68.04%, DD counts 25/16/8/5.

### 2026-06-02 — Preferred +300/+400 profit lock and stop-exit tests

**Question.** Use the preferred `+300% -> 75%, +400% -> 50%` profit-lock overlay, list the periods when the +300% threshold was hit, and test extra trade-level peak-drawdown exits at -30%, -35%, -40%, -45%, and -50% in addition to the QQQ 200MA exit.

**Implementation.** Added `scripts/run_preferred_profit_lock_stop_exit_comparison.py`. The stop is measured on synthetic TQQQ from each open trade's peak price. A stop trigger forces the raw trade state to cash until the base 200MA/MACD trade resets; executable weights are still shifted, preserving no-lookahead timing. QQQ BH is included as the benchmark row.

**Outputs.**
- `reports/tables/preferred_profit_lock_stop_exit_comparison_compact.csv`
- `reports/tables/preferred_profit_lock_300_hit_periods.csv`
- `reports/tables/preferred_profit_lock_300_hit_periods_with_executable.csv`
- `reports/figures/preferred_profit_lock_stop_exit_comparison_equity_drawdown.png`

**Result.** The +300% threshold hit 4 historical trade periods: 2014-11-28, 2018-03-12, 2021-02-05, and 2024-06-17. The +400% threshold hit in the 2020-2022 and 2023-2025 trades. Stop exits at -40% and -45% gave identical best results in this test: final return 43,107.1%, annualized return 25.92%, Sharpe 0.776, max drawdown -56.36%, 115 trades, and DD episode counts 24/13/7/3. The -50% stop never triggered and matches no-stop; the -30% and -35% stops were too aggressive and reduced return materially. Interpretation: -40%/-45% stop is a candidate overlay, but it still did not improve the single worst max drawdown.

### 2026-06-02 — Entry-loss stops tested for preferred +300/+400 profit-lock strategy

**Question.** Test -30%, -35%, -40%, -45%, and -50% loss stops measured from the current synthetic-TQQQ trade entry price, rather than from the trade peak.

**Implementation.** Added `scripts/run_preferred_profit_lock_loss_stop_comparison.py`. The entry-loss stop exits only if synthetic TQQQ unrealized return from the current trade entry falls below the threshold. It is combined with the preferred `+300% -> 75%, +400% -> 50%` profit-lock overlay and the existing QQQ 200MA exit. Raw stop signals are still shifted through executable weights.

**Outputs.**
- `reports/tables/preferred_profit_lock_loss_stop_comparison_compact.csv`
- `reports/tables/preferred_profit_lock_loss_stop_comparison_stop_events.csv`
- `reports/figures/preferred_profit_lock_loss_stop_comparison_equity_drawdown.png`

**Result.** No entry-loss stop from -30% through -50% triggered in the historical test. Therefore all entry-loss-stop variants produced identical performance to the no-entry-loss-stop +300/+400 profit-lock strategy: final return 38,856.1%, annualized return 25.42%, Sharpe 0.766, max drawdown -56.36%, 115 trades, and DD episode counts 24/13/7/4. Interpretation: losses in this preferred strategy mostly occur as givebacks after trades are already profitable, not as immediate losses below trade entry. Entry-loss stops do not address that pattern; peak-drawdown stops are the relevant form if we want to protect large accumulated gains.

### 2026-06-03 — Single-trade loss stats for preferred +300/+400 profit-lock plus 40% peak stop

**Question.** For the current preferred QQQ-hourly-200MA synthetic-TQQQ strategy, with `+300% -> 75%, +400% -> 50%` profit lock and an additional 40% synthetic-TQQQ trade-peak stop, quantify max single-trade loss.

**Definition used.** A trade is one continuous executable exposure period from weight `0 -> >0` until weight returns to `0`; partial profit-lock reductions stay inside the same trade. Trade P&L is measured on synthetic `QQQ_3X_CALC` with the executable no-lookahead weights. The primary loss statistic is pre-tax sized trade return, including the 1 bp transaction cost + 5 bps slippage approximation, but excluding cash yield and year-end tax effects because those can contaminate individual trade attribution.

**Output.** Detailed trade table saved to `reports/tables/preferred_plus_40pct_peak_stop_trade_stats.csv`.

**Result.** Over the local 60-minute sample through 2026-06-01 15:00, there were 55 round-trip/active exposure periods. The worst entry-to-exit synthetic-TQQQ trade loss was -14.66% after sizing/cost approximation (-14.61% on the underlying synthetic-TQQQ price path). The worst max adverse excursion from entry was -16.72%. The 40% peak stop triggered once, on the 2019-06 to 2020-03 trade, after synthetic TQQQ fell -45.54% from that trade's peak; that trade still exited with a +25.8% trade return because the stop was measured from the trade peak, not from entry. Interpretation: the 40% peak stop can protect large accumulated gains, but it does not guarantee the strategy/equity curve max drawdown stays below 40% because the stop is trade-level, delayed by no-lookahead execution, and separate from account-level drawdown.

### 2026-06-03 — Log-scale round-trip trade return plot and loss-streak totals

**Question.** Replot all 55 round-trip trade returns on a log y-axis and summarize all consecutive loss streaks.

**Implementation.** Because raw percentage returns can be negative and cannot be directly shown on a standard log scale, the plot uses gross return multiple `1 + sized_pre_tax_return` on the y-axis. Values below `1.0x` are losing trades and values above `1.0x` are winning trades. Consecutive loss streaks are defined as chronological runs of trades with negative sized pre-tax returns, reset by any non-negative trade.

**Outputs.**
- `reports/figures/preferred_plus_40pct_peak_stop_round_trip_trade_returns_log_scale.png`
- `reports/tables/preferred_plus_40pct_peak_stop_loss_streaks.csv`

**Result.** There were 13 loss streaks among 55 round-trip exposure periods. The longest loss streak was 6 trades. The worst compounded loss streak was trades 1-5, losing -36.22% compounded. Compounding all losing trades together gives -89.70%, while the arithmetic sum of losing trade returns is -217.11%; this emphasizes that many small/moderate losses are offset by a few large winners in the trend-following profile.

### 2026-06-03 — Locations of >40% and >50% drawdown episodes

**Question.** Locate the 7 drawdown episodes over 40% and 4 episodes over 50% reported for the preferred `+300% -> 75%, +400% -> 50%` profit-lock strategy.

**Definition.** These are strategy equity-curve drawdown episodes, measured from the last equity peak to the trough before recovery. They are not single-trade entry-to-exit losses.

**Outputs.**
- `reports/tables/profit_lock_300_400_no_stop_drawdown_episodes_gt40.csv`
- `reports/tables/profit_lock_300_400_no_stop_drawdown_episodes_gt50.csv`
- `reports/tables/profit_lock_300_400_stop_40pct_drawdown_episodes_gt40.csv`
- `reports/tables/profit_lock_300_400_stop_40pct_drawdown_episodes_gt50.csv`

**Result.** For the no-stop `+300/+400` profit-lock strategy, the seven >40% episodes are: 2001-12 to 2003-03, 2004-01 to 2005-10, 2007-10 to 2009-07, 2010-04 to 2010-08, 2011-05 to 2012-01, 2018-10 to 2019-08, and 2020-02 to 2020-04. Four of those exceeded 50%: 2004-2005, 2007-2009, 2010, and 2020. With the additional 40% peak stop, the 2020 episode is reduced to -45.54%, so the >50% count becomes three while the >40% count remains seven.

### 2026-06-03 — Labeled drawdown episodes on closed-trade return plot

**Question.** Label the strategy-level >40%/>50% drawdown episodes on the closed round-trip trade return plot and explain why these do not match compounded closed-trade loss streaks.

**Implementation.** Added a labeled log-scale plot where each point is a closed round-trip gross return (`1 + sized_pre_tax_return`), red shaded regions are strategy equity drawdown episodes, darker red regions are >50% drawdowns, and gray bottom bands are consecutive losing closed-trade streaks.

**Outputs.**
- `reports/figures/preferred_plus_40pct_peak_stop_trade_returns_log_dd_vs_loss_streaks_labeled.png`
- `reports/tables/preferred_plus_40pct_peak_stop_dd_vs_loss_streak_mapping.csv`

**Result/interpretation.** The large drawdown episodes are strategy equity peak-to-trough events measured on every hourly bar. They do not need to coincide with consecutive losing closed trades. Several drawdown troughs occur inside trades that later close profitably, and a drawdown episode can span profitable trades that still do not recover the prior high-water mark. Loss streaks, by contrast, are only consecutive negative completed round trips. This is why the >40%/>50% drawdown labels do not match the compounded closed-trade loss streak totals.

### 2026-06-03 — Gain-activated exit overlays after +100% unrealized trade gain

**Question.** Instead of the current 40% trade-peak stop, test exits that become active only after synthetic TQQQ has been up more than +100% in the current base trade: exit on QQQ 100-day MA, QQQ 50-day MA, or a 30% stop from synthetic-TQQQ trade peak.

**Implementation.** Added `scripts/run_preferred_gain100_exit_overlay_comparison.py`. The base strategy remains QQQ MACD histogram > 0 entry, QQQ hourly 200-day MA entry/exit gate, synthetic `QQQ_3X_CALC` exposure, no daily regime gate, max one trade per day, 3% cash yield, 1 bp transaction cost, 5 bps slippage, 24% short-term tax approximation, and the `+300% -> 75%, +400% -> 50%` profit lock. The new overlays activate after +100% unrealized synthetic-TQQQ trade gain and then exit on either QQQ 100MA, QQQ 50MA, or synthetic TQQQ -30% from trade peak. Overlay exits are still shifted by the no-lookahead executable-weight conversion. After an overlay exit, the strategy stays in cash until the base 200MA/MACD state resets.

**Outputs.**
- `reports/tables/preferred_gain100_exit_overlay_comparison_compact.csv`
- `reports/tables/preferred_gain100_exit_overlay_comparison_metrics.csv`
- `reports/tables/preferred_gain100_exit_overlay_comparison_returns.csv`
- `reports/tables/preferred_gain100_exit_overlay_comparison_weights.csv`
- `reports/tables/preferred_gain100_exit_overlay_comparison_diagnostics.parquet`
- `reports/figures/preferred_gain100_exit_overlay_comparison_equity_drawdown.png`

**Result.** The current `+300/+400 profit lock + 40% peak stop` remained best: final return 43,107.1%, annualized return 25.92%, Sharpe 0.776, max drawdown -56.36%, and DD episodes 24/13/7/3 for >20/>30/>40/>50%. The +100%-activated exits cut major winners too early: `gain100_peak_stop_30pct` fell to 18.57% annualized, `gain100_exit_qqq_50ma` to 16.46%, and `gain100_exit_qqq_100ma` to 16.04%. The 50MA version reduced >50% DD episodes to 1, but at a large return cost. Interpretation: these +100%-activated exits are too aggressive under the current re-entry convention; do not promote them into the preferred rule.

### 2026-06-03 — 40% trade-peak stop without profit lock

**Question.** Test the preferred hourly-200MA strategy with only a synthetic-TQQQ 40% trade-peak stop and no `+300%/+400%` profit-lock sizing overlay.

**Implementation.** Reused the current preferred QQQ MACD + QQQ hourly 200MA entry/exit base strategy trading synthetic `QQQ_3X_CALC`, with 3% cash yield, 1 bp transaction cost, 5 bps slippage, and 24% short-term tax approximation. The stop is measured from each open synthetic-TQQQ trade peak and is shifted through the no-lookahead executable-weight conversion.

**Outputs.**
- `reports/tables/preferred_stop40_without_profit_lock_comparison.csv`
- `reports/tables/preferred_stop40_without_profit_lock_returns.csv`
- `reports/tables/preferred_stop40_without_profit_lock_weights.csv`
- `reports/tables/preferred_stop40_without_profit_lock_stop_events.csv`

**Result.** The 40% peak stop without profit lock improved the no-lock base from 34,014.8% final return / 24.79% annualized to 37,696.1% final return / 25.28% annualized, and reduced >50% drawdown episodes from 6 to 5. However, it still lagged the `+300/+400 profit lock + 40% peak stop` version at 43,107.1% final return / 25.92% annualized, and it did not improve the single worst max drawdown (-56.36%). The 40% stop triggered once, on 2020-03-09 10:00.

### 2026-06-03 — Gain-activated exit overlays after +200% unrealized trade gain

**Question.** Repeat the gain-activated exit overlay test, but activate only after synthetic TQQQ has been up more than +200% in the current base trade. Test QQQ 100-day MA exit, QQQ 50-day MA exit, and synthetic-TQQQ -30% from trade-peak stop, as alternatives to the current -40% trade-peak stop.

**Implementation.** Reused `scripts/run_preferred_gain100_exit_overlay_comparison.py` with `--activation-gain 2.0 --output-prefix preferred_gain200_exit_overlay_comparison`. The script was adjusted so output names/plot titles reflect the activation threshold dynamically. All no-lookahead timing, max-one-trade-per-day, 3% cash yield, 1 bp transaction cost, 5 bps slippage, 24% short-term tax approximation, QQQ signal source, synthetic `QQQ_3X_CALC` exposure, and `+300% -> 75%, +400% -> 50%` profit lock assumptions remain unchanged.

**Outputs.**
- `reports/tables/preferred_gain200_exit_overlay_comparison_compact.csv`
- `reports/tables/preferred_gain200_exit_overlay_comparison_metrics.csv`
- `reports/tables/preferred_gain200_exit_overlay_comparison_returns.csv`
- `reports/tables/preferred_gain200_exit_overlay_comparison_weights.csv`
- `reports/tables/preferred_gain200_exit_overlay_comparison_diagnostics.parquet`
- `reports/figures/preferred_gain200_exit_overlay_comparison_equity_drawdown.png`

**Result.** The current `+300/+400 profit lock + 40% peak stop` still remained best: final return 43,107.1%, annualized return 25.92%, Sharpe 0.776, max drawdown -56.36%, DD episodes 24/13/7/3 for >20/>30/>40/>50%. The +200%-activated alternatives were materially worse: `gain200_peak_stop_30pct` annualized 20.35%, `gain200_exit_qqq_50ma` annualized 19.15%, and `gain200_exit_qqq_100ma` annualized 17.54%. Interpretation: waiting until +200% is less damaging than +100%, but these exits still cut the large winners too early and did not reduce the worst max drawdown. Do not promote these overlays into the preferred rule.

### 2026-06-03 — Gain-activated exits after +300% unrealized trade gain

**Question.** Replace the current 40% synthetic-TQQQ trade-peak stop with exits that activate only after the current trade has gained more than +300%: QQQ 100-day MA exit, QQQ 50-day MA exit, or synthetic-TQQQ -30% from trade peak.

**Implementation.** Reused and generalized `scripts/run_preferred_gain100_exit_overlay_comparison.py` so the activation threshold is dynamic. Ran it with `--activation-gain 3.0 --output-prefix preferred_gain300_exit_overlay_comparison`. The base remains QQQ MACD histogram > 0 entry, QQQ hourly 200-day MA entry/exit gate, synthetic `QQQ_3X_CALC` exposure, no daily regime gate, max one trade per day, 3% cash yield, 1 bp transaction cost, 5 bps slippage, 24% short-term tax approximation, and the `+300% -> 75%, +400% -> 50%` profit lock. The new overlays activate after +300% unrealized synthetic-TQQQ trade gain and then exit on QQQ 100MA, QQQ 50MA, or synthetic TQQQ -30% from trade peak.

**Outputs.**
- `reports/tables/preferred_gain300_exit_overlay_comparison_compact.csv`
- `reports/tables/preferred_gain300_exit_overlay_comparison_metrics.csv`
- `reports/tables/preferred_gain300_exit_overlay_comparison_returns.csv`
- `reports/tables/preferred_gain300_exit_overlay_comparison_weights.csv`
- `reports/tables/preferred_gain300_exit_overlay_comparison_diagnostics.parquet`
- `reports/figures/preferred_gain300_exit_overlay_comparison_equity_drawdown.png`

**Result.** The current `+300/+400 profit lock + 40% peak stop` remained best: final return 43,107.1%, annualized return 25.92%, Sharpe 0.776, max drawdown -56.36%, and DD episodes 24/13/7/3 for >20/>30/>40/>50%. The +300%-activated overlays materially reduced return without improving max drawdown: 50MA exit returned 24,977.6% / 23.34% annualized, peak-stop-30 returned 18,885.2% / 22.05%, and 100MA exit returned 18,116.9% / 21.85%. All kept max drawdown at -56.36%; their DD episode counts were 21/14/7/4. Interpretation: after +300%, switching to the 50MA/100MA/-30% peak stop exits still cuts the big winners too early and does not reduce the worst strategy-level drawdown. Do not promote this rule.

### 2026-06-03 — Post-winner loss streaks after >100% round-trip winners

**Question.** Among round-trip trades with final return greater than +100%, what percentage were immediately followed by more than three losing round-trip trades?

**Output.** `reports/tables/preferred_plus_40pct_peak_stop_gt100_winners_following_loss_streaks.csv`.

**Result.** There were 4 completed round-trip trades with final return greater than +100%. One of the four was immediately followed by more than three losing round trips: trade 42 was followed by losing trades 43-46. Conditional percentage: 25.0%. As a percentage of all 55 round-trip trades, this pattern occurred once, or 1.82%.

### 2026-06-03 — Next-4-trade risk after trades with >100% unrealized peak

**Question.** Among round-trip trades whose intra-trade synthetic-TQQQ peak return exceeded +100%, calculate how often they were followed by more than 3 consecutive losing trades or by more than a 30% compounded loss over the next 4 trades.

**Definition.** Used the current preferred `+300/+400 profit lock + 40% peak stop` trade table. A qualifying trade has `max_favorable_from_entry > +100%`. For each qualifying trade, inspected the next four completed round trips. The loss-streak condition means the immediate following completed-trade streak has at least 4 losers. The next-4 loss condition means the compounded return of the next four completed trades is <= -30%.

**Output.** `reports/tables/preferred_plus_40pct_peak_stop_peak100_next4_risk_stats.csv`.

**Result.** 8 of 55 trades had an intra-trade peak above +100%. Of those 8, 2 were followed by at least 4 immediate losing trades and 1 had next-four-trade compounded return below -30%. The union of the two conditions was 2 of 8, or 25.0% of the >100%-peak trades; relative to all 55 trades, that is 3.64%.

### 2026-06-03 — +100% peak trades followed by >20% pullback before a new high

**Question.** Among round-trip trades whose synthetic-TQQQ path reached more than +100% unrealized gain, measure how often the trade then suffered a >20% pullback from a running peak before making a new running peak.

**Implementation.** Scanned the synthetic `QQQ_3X_CALC` price path inside each of the 55 round-trip exposure periods from `reports/tables/preferred_plus_40pct_peak_stop_trade_stats.csv`. A trade qualified if its maximum unrealized price return exceeded +100%. A hit occurred if, after the running peak was already above +100% from entry, price fell at least 20% from that running peak before a newer high-water mark.

**Output.** `reports/tables/preferred_plus_40pct_peak_stop_peak100_pullback20_by_trade.csv`.

**Result.** 8 of 55 trades reached more than +100% peak unrealized return. All 8 of those trades later had a >20% pullback from a running peak before making a new peak or before exit. Therefore the hit rate is 100.0% among >100%-peak trades, or 14.5% of all 55 trades. Interpretation: in this leveraged trend strategy, every major winner historically required tolerating at least one >20% intratrade giveback after the trade had already doubled.

### 2026-06-03 — Frequency of >30% peak-loss after trades exceed +100% unrealized gain

**Question.** Among round-trip trades whose synthetic-TQQQ exposure first reaches more than +100% unrealized gain, how often is that followed by more than a 30% loss from a running trade peak before a new peak is made?

**Definition.** Denominator is completed/active round-trip trades with max favorable synthetic-TQQQ return above +100%. The loss event is a running trade-peak drawdown of at least -30% after the first +100% threshold is reached. This is measured on synthetic `QQQ_3X_CALC` price path during the executable trade.

**Output.** `reports/tables/preferred_plus_40pct_peak_stop_gain100_then_30pct_peak_loss_stats.csv`.

**Result.** 8 of 55 round-trip trades reached a >+100% peak. Of those 8, 6 had a >30% drawdown from a running trade peak after crossing +100%, so the conditional rate is 75.0% (6/8). As a share of all round-trip trades, this is 10.9% (6/55). If requiring the trade to later make a new higher peak after the >30% drawdown, only 3 of the 8 qualify: trades 42, 49, and 52.

### 2026-06-03 — Post-big-winner stay-out/cooldown rule

**Question.** After exiting a round-trip trade whose synthetic-TQQQ peak gain exceeded +100%, stay out of market while continuing to run the usual strategy in the background. Resume only after either a blocked/paper trade experiences a -20% synthetic-TQQQ loss or three calendar months pass.

**Implementation.** Added `scripts/run_preferred_post_big_trade_cooldown.py`. The cooldown is applied at the raw-signal level and then passed through the existing no-lookahead executable-weight shift. A completed actual raw trade whose full synthetic-TQQQ peak gain reaches +100% starts the cooldown. During cooldown, desired raw entries are ignored but paper-tracked; the cooldown ends on a -20% paper loss or after three calendar months. Tested both the profit-lock-only baseline and the `+300/+400 profit lock + 40% peak stop` version.

**Outputs.**
- `reports/tables/preferred_post_big_trade_cooldown_compact.csv`
- `reports/tables/preferred_post_big_trade_cooldown_metrics.csv`
- `reports/tables/preferred_post_big_trade_cooldown_returns.csv`
- `reports/tables/preferred_post_big_trade_cooldown_weights.csv`
- `reports/tables/preferred_post_big_trade_cooldown_events.csv`
- `reports/tables/preferred_post_big_trade_cooldown_diagnostics.parquet`
- `reports/figures/preferred_post_big_trade_cooldown_equity_drawdown.png`

**Result.** The cooldown triggered 8 times historically, but every cooldown ended by the three-month time rule; the -20% paper-loss release never triggered. The current `+300/+400 profit lock + 40% peak stop` remained best: final return 43,107.1%, annualized return 25.92%, Sharpe 0.776, max drawdown -56.36%, and DD episodes 24/13/7/3. Adding the post-peak100 cooldown to the stop40 version reduced final return to 29,357.7% / 24.10% annualized and did not improve max drawdown; DD episodes became 26/14/8/2. Interpretation: this cooldown cut some productive re-entries, reduced one >50% DD episode, but did not improve the worst drawdown and reduced return. Do not promote it into the preferred rule.

### 2026-06-03 — Stay-out-after-big-winner rule

**Question.** After exiting a completed round-trip trade whose synthetic TQQQ peak gain exceeded +100%, stay out of the market while continuing to run the usual strategy on paper. Re-entry is allowed only after the imaginary paper strategy experiences another 20% drawdown, or after six calendar months, whichever comes first.

**Implementation.** Added `scripts/run_preferred_stay_out_after_big_winner.py`. The rule is applied to raw close-bar signals before profit-lock sizing and before the existing executable-weight shift, preserving the no-lookahead convention. Tested two versions: (1) replacing the 40% peak stop with this stay-out rule using the no-stop base, and (2) adding the stay-out rule on top of the 40% peak-stop base. The base remains QQQ MACD histogram > 0 entry, QQQ hourly 200-day MA entry/exit gate, synthetic `QQQ_3X_CALC` exposure, no daily regime gate, max one trade per day, 3% cash yield, 1 bp transaction cost, 5 bps slippage, 24% short-term tax approximation, and the `+300% -> 75%, +400% -> 50%` profit lock.

**Outputs.**
- `reports/tables/preferred_stay_out_after_big_winner_compact.csv`
- `reports/tables/preferred_stay_out_after_big_winner_metrics.csv`
- `reports/tables/preferred_stay_out_after_big_winner_returns.csv`
- `reports/tables/preferred_stay_out_after_big_winner_weights.csv`
- `reports/tables/preferred_stay_out_after_big_winner_events.csv`
- `reports/tables/preferred_stay_out_after_big_winner_diagnostics.parquet`
- `reports/figures/preferred_stay_out_after_big_winner_equity_drawdown.png`

**Result.** The stay-out rule triggered 8 times; 3 releases came from the imaginary strategy hitting a 20% drawdown and 5 releases came from the six-month timeout. It reduced trades from 115 to 87, but it materially reduced return and did not improve the worst drawdown. Current `+300/+400 profit lock + 40% peak stop` remained best at 43,107.1% final return / 25.92% annualized / Sharpe 0.776 / max DD -56.36%. The stay-out rule on the 40%-stop base returned 15,156.7% / 21.04% annualized / Sharpe 0.687 / max DD -55.82%. Replacing the 40% stop with stay-out returned 13,821.1% / 20.62% annualized / Sharpe 0.678 / max DD -58.29%. Interpretation: the stay-out rule lowers activity and may suit manual behavior, but it sits out too much post-trend continuation and does not solve the main drawdown problem. Do not promote as a return/improvement rule; consider only if the goal is lower psychological/manual trading burden.

### 2026-06-03 — Post-big-win cooldown / stay-out rule

**Question.** After exiting a round-trip trade whose synthetic-TQQQ peak gain exceeded +100%, stay out of the market while still running the usual strategy as an imaginary account. Re-enable entries after either the imaginary account suffers a 20% drawdown or 12/18 months pass.

**Implementation.** Added `scripts/run_preferred_post_bigwin_cooldown_comparison.py`. The test applies the cooldown to the current `+300/+400 profit lock + 40% trade-peak stop` base. A big winner is an actual raw round trip whose synthetic `QQQ_3X_CALC` max price gain from entry was at least +100%. During cooldown, actual raw exposure is forced to cash; meanwhile an imaginary version of the usual current strategy continues to run using pre-tax returns with trading costs. Cooldown ends when that imaginary equity has a 20% drawdown from its post-exit high-water mark, or after 12/18 calendar months. After that, entries again follow the usual rule. Signals are still passed through the existing executable-position shift and max-one-trade-per-day convention.

**Outputs.**
- `reports/tables/preferred_post_bigwin_cooldown_comparison_compact.csv`
- `reports/tables/preferred_post_bigwin_cooldown_comparison_metrics.csv`
- `reports/tables/preferred_post_bigwin_cooldown_comparison_returns.csv`
- `reports/tables/preferred_post_bigwin_cooldown_comparison_weights.csv`
- `reports/tables/preferred_post_bigwin_cooldown_comparison_diagnostics.parquet`
- `reports/figures/preferred_post_bigwin_cooldown_comparison_equity_drawdown.png`

**Result.** The current `+300/+400 profit lock + 40% peak stop` remained best: final return 43,107.1%, annualized return 25.92%, Sharpe 0.776, max drawdown -56.36%, and DD episodes 24/13/7/3 for >20/>30/>40/>50%. The cooldown overlays started 8 times and reduced exposure/trades, but materially reduced return without improving max drawdown. The 18-month variant returned 20,104.7% / 22.33% annualized with Sharpe 0.720 and DD counts 22/12/8/3. The 12-month variant returned 19,454.6% / 22.18% annualized with Sharpe 0.715 and DD counts 22/12/8/3. Interpretation: the cooldown avoids some ordinary churn but skips too much of the next uptrend and does not reduce the worst drawdown. Do not promote this rule.

### 2026-06-03 — Stay-out rule after exiting a >100% peak-gain trade

**Question.** After exiting a round-trip trade whose unrealized peak gain exceeded +100%, stay out of the market, keep running the strategy on paper, and only re-enter after the imaginary/paper strategy experiences a 20% drawdown; after that, resume normal rules.

**Implementation.** Added `scripts/run_preferred_stay_out_after_big_peak.py`. A qualifying trade is an actual completed raw exposure period whose synthetic `QQQ_3X_CALC` max unrealized gain exceeded +100% before exit. During cooldown, actual raw exposure is forced to cash while a paper strategy continues to follow the unsuppressed base signal. Paper drawdown is measured from the paper equity high-water mark since cooldown start. Once paper drawdown reaches -20%, the cooldown ends and normal raw trading resumes; executable weights are still shifted, preserving no-lookahead timing. Tested the stay-out rule both as a replacement for the 40% peak stop and combined with the 40% peak stop, while keeping the `+300% -> 75%, +400% -> 50%` profit-lock sizing.

**Outputs.**
- `reports/tables/preferred_stay_out_after_big_peak_compact.csv`
- `reports/tables/preferred_stay_out_after_big_peak_metrics.csv`
- `reports/tables/preferred_stay_out_after_big_peak_returns.csv`
- `reports/tables/preferred_stay_out_after_big_peak_weights.csv`
- `reports/tables/preferred_stay_out_after_big_peak_diagnostics.parquet`
- `reports/figures/preferred_stay_out_after_big_peak_equity_drawdown.png`

**Result.** The current `+300/+400 profit lock + 40% peak stop` remained best: final return 43,107.1%, annualized return 25.92%, Sharpe 0.776, max drawdown -56.36%, 115 trades, and DD episodes 24/13/7/3 for >20/>30/>40/>50%. The stay-out rule triggered 8 cooldowns and 8 paper-DD releases. As a replacement for the 40% peak stop, it returned 17,876.6% / 21.79% annualized, Sharpe 0.708, max drawdown -58.07%, 85 trades, and DD episodes 22/12/8/3. Combined with the 40% peak stop, it returned 19,597.7% / 22.22% annualized, Sharpe 0.718, max drawdown -56.40%, 85 trades, and DD episodes 22/12/8/3. Interpretation: the stay-out rule lowers trade count and exposure, but it misses too much upside after large trends and does not improve worst drawdown enough. Do not promote this rule.

### 2026-06-03 — Stay-out overlay after >100% peak winning trades

**Question.** After exiting a round-trip trade whose intratrade synthetic-TQQQ peak gain was more than +100%, stay out of the market. Continue running the base strategy as an imaginary strategy, then re-enter only after the imaginary trade has a 10% drawdown from its imaginary peak. After re-entry, behavior returns to normal.

**Implementation.** Added `scripts/run_preferred_stay_out_after_big_peak.py`. The base is the current preferred QQQ hourly-200MA strategy trading synthetic `QQQ_3X_CALC`, with `+300% -> 75%, +400% -> 50%` profit lock and the -40% synthetic-TQQQ trade-peak stop. The stay-out overlay is applied at raw-signal level, then passed through the same no-lookahead executable-weight shift and max-one-trade-per-day rule. A release is observed at bar close and cannot earn same-bar return.

**Outputs.**
- `reports/tables/preferred_stay_out_after_big_peak_compact.csv`
- `reports/tables/preferred_stay_out_after_big_peak_metrics.csv`
- `reports/tables/preferred_stay_out_after_big_peak_returns.csv`
- `reports/tables/preferred_stay_out_after_big_peak_weights.csv`
- `reports/tables/preferred_stay_out_after_big_peak_diagnostics.parquet`
- `reports/tables/preferred_stay_out_after_big_peak_stay_out_records.csv`
- `reports/tables/preferred_stay_out_after_big_peak_release_summary.csv`
- `reports/figures/preferred_stay_out_after_big_peak_equity_drawdown.png`

**Result.** The stay-out rule activated 8 times and released 8 times after a 10% imaginary drawdown. It reduced trades from 115 to 98 and exposure from 65.4% to 63.0%, but reduced final return from 43,107.1% to 29,845.4% and annualized return from 25.92% to 24.18%. Sharpe fell from 0.776 to 0.746. Max drawdown stayed -56.36%. DD episode counts worsened from 24/13/7/3 to 25/15/9/3 for >20/>30/>40/>50%. Interpretation: this rule reduces trading and forces post-big-win patience, but it misses too much early re-entry return and does not reduce the worst drawdown; do not promote it into the preferred rule in this form.

### 2026-06-03 — Seven worst drawdown plots for current preferred + 40% peak stop

**Question.** For the current preferred strategy with the `+300% -> 75%, +400% -> 50%` profit lock and the -40% synthetic-TQQQ trade-peak stop, make plots for the seven worst drawdowns with two-month extensions and label buy/sell locations, round-trip returns, and profit-lock hits.

**Implementation.** Added `scripts/plot_preferred_worst7_dd_trade_events.py`. Each plot shows synthetic `QQQ_3X_CALC` price in the top panel and strategy equity drawdown in the bottom panel. The drawdown peak-to-trough region and trough-to-recovery region are shaded. Buy markers, sell markers with sized pre-tax round-trip returns, and profit-lock hit stars are annotated. The plot window has at least two months before the drawdown peak and after recovery; when needed, it is extended further to include the full buy/sell markers for trades overlapping the drawdown episode.

**Outputs.**
- Plot folder: `reports/figures/preferred_stop40_worst7_dd_trade_events/`
- Episode table: `reports/tables/preferred_stop40_worst7_dd_episodes_for_plots.csv`
- Plot index: `reports/tables/preferred_stop40_worst7_dd_plot_index.csv`
- Profit-lock events: `reports/tables/preferred_stop40_profit_lock_events.csv`

**Result.** Generated seven separate plots for DD severity ranks 1-7. Profit-lock hit markers appear in DD5, DD6, and DD7; the earlier drawdown plots have no profit-lock hits in the displayed/overlapping trade windows.

### 2026-06-03 — Seven worst drawdown plots for current preferred +40% peak-stop strategy

**Question.** For the current preferred strategy with `+300% -> 75%, +400% -> 50%` profit lock and the -40% synthetic-TQQQ trade-peak stop, create seven plots for the seven worst strategy-level drawdowns. Extend each plot two months before the drawdown peak and two months after recovery, label round-trip entries/exits with trade return, label profit-lock hits, and include the MA line relevant to entry/exit.

**Implementation.** Added `scripts/plot_preferred_peak_stop_worst7_drawdowns.py`. Each plot has three panels: QQQ hourly close with the QQQ hourly 200-day MA (1200 bars) used by the entry gate and base exit, synthetic TQQQ price with executable weight plus entry/exit labels and round-trip returns, and strategy equity/drawdown. Profit-lock hits are marked when present; for these exact seven worst drawdown windows no +300/+400 profit-lock hit falls inside the two-month-extended plot windows, so the plots explicitly note that.

**Outputs.**
- `reports/figures/preferred_peak_stop_worst7_drawdowns/preferred_peak_stop_worst_dd_01.png`
- `reports/figures/preferred_peak_stop_worst7_drawdowns/preferred_peak_stop_worst_dd_02.png`
- `reports/figures/preferred_peak_stop_worst7_drawdowns/preferred_peak_stop_worst_dd_03.png`
- `reports/figures/preferred_peak_stop_worst7_drawdowns/preferred_peak_stop_worst_dd_04.png`
- `reports/figures/preferred_peak_stop_worst7_drawdowns/preferred_peak_stop_worst_dd_05.png`
- `reports/figures/preferred_peak_stop_worst7_drawdowns/preferred_peak_stop_worst_dd_06.png`
- `reports/figures/preferred_peak_stop_worst7_drawdowns/preferred_peak_stop_worst_dd_07.png`
- `reports/tables/preferred_peak_stop_worst7_drawdown_plot_summary.csv`
- `reports/tables/preferred_peak_stop_worst7_drawdown_episodes.csv`

**Result.** The seven plotted drawdowns are -56.36%, -52.15%, -50.73%, -49.77%, -48.77%, -45.54%, and -44.83%. No profit-lock hit occurred inside these two-month-extended windows; profit locks in the full sample occurred on 2014-11-28, 2018-03-12, 2021-02-05, 2021-07-13, 2024-06-17, and 2024-12-06, which are outside the displayed windows for these worst drawdown episodes.

### 2026-06-03 — Add trade DD/RT labels and footer tables to worst-drawdown plots

**Question.** For the seven worst drawdown plots of the current preferred `+300/+400 profit lock + -40% peak stop` strategy, label each entry/exit with the related trade DD in addition to round-trip return, and add a table summary of all trade DD/RT plus compounded DD/RT at the end of each plot.

**Implementation.** Updated `scripts/plot_preferred_peak_stop_worst7_drawdowns.py`. Entry/exit annotations now show `RT` and `DD`; trade DD is defined as that round trip's maximum drawdown from its own synthetic-TQQQ trade peak. Each plot now has a footer table listing all visible overlapping round trips with trade ID, entry date, exit date, trade DD, and RT. The final `Comp` row compounds the visible trade DD proxy and visible round-trip returns using `product(1 + value) - 1`. Per-plot footer tables are also saved next to each PNG as CSV files.

**Outputs.** Updated plots in `reports/figures/preferred_peak_stop_worst7_drawdowns/`, updated `reports/tables/preferred_peak_stop_worst7_drawdown_plot_summary.csv`, and per-plot CSV files such as `reports/figures/preferred_peak_stop_worst7_drawdowns/preferred_peak_stop_worst_dd_01_trade_table.csv`.

**Important interpretation.** The footer's compounded trade-DD value is a trade-level DD proxy, not the same object as strategy equity drawdown. Strategy drawdown is measured continuously from account equity high-water mark to trough; trade DD is measured inside each round trip from that trade's own price peak.

### 2026-06-03 — Combined worst-drawdown and best-winning-trade tables

**Question.** Combine `reports/tables/preferred_peak_stop_worst7_drawdown_plot_summary.csv` with the best 12 round-trip winning-trade table.

**Implementation/outputs.** Created a long-form combined CSV, a crosswalk CSV, a best-winners-with-DD-crossrefs CSV, and a Markdown combined report:
- `reports/tables/preferred_peak_stop_worst7_and_best12_combined.csv`
- `reports/tables/preferred_peak_stop_worst7_best12_crosswalk.csv`
- `reports/tables/preferred_best12_winners_with_worst7_dd_crossrefs.csv`
- `reports/tables/preferred_peak_stop_worst7_and_best12_combined.md`

**Result.** The crosswalk shows which of the top-12 winning round trips appear inside each of the worst-7 drawdown windows. Several worst drawdowns include major winning trades, reinforcing that strategy drawdowns are high-water-mark equity events rather than simply strings of losing round trips.

### 2026-06-03 — Rename worst-DD labels by chronological order

**Question.** Rename the worst-drawdown ranks by time order and print the best 12 round-trip winning trades in chronological order.

**Implementation/outputs.** Created time-ranked versions of the worst-DD summary/crosswalk and a chronological best-winner table:
- `reports/tables/preferred_peak_stop_worst7_drawdown_plot_summary_time_ranked.csv`
- `reports/tables/preferred_peak_stop_worst7_best12_crosswalk_time_ranked.csv`
- `reports/tables/preferred_best12_winning_trades_time_order.csv`
- updated `reports/tables/preferred_peak_stop_worst7_and_best12_combined.md`

**Result.** DD labels are now chronological: DD1 = 2001-2003, DD2 = 2004-2007, DD3 = 2007-2009, DD4 = 2010-2011, DD5 = 2011-2013, DD6 = 2018-2020, DD7 = 2020-2020. The original severity rank is preserved in the time-ranked tables.

### 2026-06-03 — Check whether Top8 winning trades are followed by weak/loss streaks

**Question.** Do all Top8 round-trip winners get followed by at least 3 or 4 loss/weak trades?

**Definition.** Used the current weak/loss definition `round-trip RT < +3%`; checked the immediate next weak/loss streak after each Top8 winner.

**Output.** `reports/tables/preferred_top8_following_weak_streaks_rt_lt_3pct.csv`.

**Result.** No. Among the 8 Top8 winners, 5 were followed immediately by at least 3 weak/loss trades, and 4 were followed by at least 4. Excluding the still-recent/open-ended 2026 winner T55, it is 5/7 and 4/7. The exceptions were T49, followed by only 2 weak trades; T52, followed by only 1; and T55, with no following completed weak streak yet.

### 2026-06-03 — Check weak/loss streaks after Top8 winning round trips

**Question.** Check whether each Top8 winning round trip, excluding the current trade and trades that already experienced more than 30% trade-level peak drawdown, is followed by at least 3-4 weak/loss trades.

**Definition.** A weak/loss trade is a completed round trip with RT < +3%. The post-winner streak is the immediate consecutive sequence of such trades after the Top8 trade exits. A trade is considered to have already contained >30% DD if its trade-level max drawdown from peak was <= -30%.

**Output.** `reports/tables/preferred_top8_following_weak_streak_check.csv`.

**Result.** Yes for the filtered set. After excluding current T55 and Top8 trades whose own trade DD was worse than -30%, the remaining Top8 trades are T26 and T36. T26 was followed by four weak/loss trades (T27-T30), and T36 was followed by five weak/loss trades (T37-T41). Across all non-current Top8 trades, the pattern is not universal because T49 was followed by only two weak/loss trades and T52 by only one; both of those had already experienced >30% trade-level DD.

### 2026-06-03 — Immediate weak/loss streaks after Top8 winning trades

**Question.** Check whether all Top8 winning trades, excluding the current/latest trade, are immediately followed by weak/loss streaks that realize more than 10%, 15%, or 20% loss, and summarize the threshold for Top-N winners (`N <= 8`).

**Definition.** A weak/loss trade is a completed round trip with RT < +3%. The immediate weak/loss streak is the consecutive run of RT < +3% trades immediately after the Top8 winner exits. Realized loss is the compounded RT of that immediate streak.

**Outputs.**
- `reports/tables/preferred_top8_immediate_weak_streak_loss_thresholds.csv`
- `reports/tables/preferred_topN_immediate_weak_streak_threshold_summary.csv`

**Result.** Every non-current Top8 winner was immediately followed by at least one weak/loss trade, but not all had a weak/loss streak deeper than 10%, 15%, or 20%. T49 was followed by only -4.5% and T52 by only -2.3%. Therefore no Top-N set starting from the largest winner satisfies an all-trades >10%, >15%, or >20% immediate realized-loss rule. The maximum loss threshold satisfied by all non-current Top8 winners is only about 2.3%.

### 2026-06-03 — Top8 winning trades with three largest intratrade drawdowns

**Question.** For the Top8 winning round-trip trades, excluding the current/latest trade, replace the single trade-DD column with the biggest three intratrade drawdown episodes.

**Definition.** `Peak from entry` remains the maximum unrealized gain from that round trip's entry price. The three DD columns are the three largest local high-water-mark drawdown episodes within that same round trip, measured on synthetic `QQQ_3X_CALC` price from each intratrade peak to trough. `rec no rec` means the trade exited before that intratrade high was recovered.

**Outputs.**
- `reports/tables/preferred_top8_ex_current_winning_trades_biggest3_dds.csv`
- `reports/tables/preferred_top8_ex_current_winning_trades_biggest3_dds_raw.csv`

### 2026-06-03 — Top8 winners excluding current: top three intratrade drawdowns and QQQ 200MA distance

**Question.** For the Top8 winning round trips except the current/latest trade, replace the single Trade DD column with the biggest three intratrade drawdown percentages, and include QQQ's distance from its hourly 200-day MA at the beginning/peak of each drawdown.

**Definition.** For each completed round trip, intratrade drawdown episodes are computed from synthetic `QQQ_3X_CALC` price using that trade's own running peak. The largest three episodes are ranked by max drawdown. `QQQ dist 200MA @ DD start` is `QQQ / QQQ_hourly_200d_MA - 1` at the drawdown episode's starting peak. Dates for individual DD episodes are saved only in the detail CSV, not in the printed summary.

**Outputs.**
- `reports/tables/preferred_top8_ex_current_top3_trade_dd_q200ma_distance.csv`
- `reports/tables/preferred_top8_ex_current_top3_trade_dd_q200ma_distance_detail.csv`

**Result.** The largest intratrade drawdowns in the biggest winners usually started while QQQ was still well above its hourly 200-day MA, typically by about +9% to +33%, which explains why a 200MA exit alone does not catch those givebacks early.

### 2026-06-03 — Distribution of QQQ distance from hourly 200-day MA during Top8 winners

**Question.** For the Top8 winning round-trip trades, excluding the current/latest trade, summarize the distribution of QQQ's distance from its hourly 200-day moving average across the trades together.

**Definition.** Distance is `QQQ close / QQQ hourly 200-day MA - 1`, with the hourly 200-day MA implemented as 1,200 hourly bars. The sample includes completed Top8 trades T6, T22, T26, T36, T42, T49, and T52; the current/latest T55 is excluded.

**Outputs.**
- `reports/figures/preferred_top8_ex_current_qqq_200ma_distance_distribution.png`
- `reports/tables/preferred_top8_ex_current_qqq_200ma_distance_distribution_summary.csv`
- `reports/tables/preferred_top8_ex_current_qqq_200ma_distance_by_trade_summary.csv`
- `reports/tables/preferred_top8_ex_current_qqq_200ma_distance_bar_observations.csv`

**Result.** Across 17,681 hourly bars, QQQ was typically meaningfully above its 200-day MA during the large winning trades: median distance was +10.7%, mean +11.4%, 75th percentile +14.3%, and 90th percentile +19.4%. Only 0.3% of bars were below the MA. At the start of the largest intratrade DD episodes, QQQ was even more extended: median +16.0% and 95.2% of those DD starts occurred with QQQ more than +10% above the 200-day MA. This supports using QQQ/200MA as a broad trend gate, but suggests it is too slow to protect gains after strongly extended advances.

### 2026-06-03 — Distribution of QQQ distance from hourly 200-day MA during Top8 winners including current

**Question.** Repeat the QQQ distance-from-200MA distribution analysis for the Top8 winning round-trip trades together, this time including the current/latest Top8 trade T55.

**Definition.** Distance is `QQQ close / QQQ hourly 200-day MA - 1`, with the hourly 200-day MA implemented as 1,200 hourly bars. The sample includes Top8 trades T6, T22, T26, T36, T42, T49, T52, and current/latest T55.

**Implementation.** Added `scripts/analyze_top8_qqq_200ma_distance_distribution.py` so the analysis can be regenerated from the cached Alpha Vantage 60min QQQ data and the preferred Top8 trade table.

**Outputs.**
- `reports/figures/preferred_top8_including_current_qqq_200ma_distance_distribution.png`
- `reports/tables/preferred_top8_including_current_qqq_200ma_distance_distribution_summary.csv`
- `reports/tables/preferred_top8_including_current_qqq_200ma_distance_by_trade_summary.csv`
- `reports/tables/preferred_top8_including_current_qqq_200ma_distance_bar_observations.csv`

**Result.** Including T55 barely changes the pooled distribution because T55 is short relative to the older large winners. Across 17,903 hourly bars, mean distance was +11.4%, median +10.7%, p10 +5.0%, and p90 +19.4%. Only 0.3% of bars were below the hourly 200-day MA. Most bars were between +5% and +20% above the MA, while +20% or more above the MA occurred 8.6% of the time.

### 2026-06-03 — Normalized-time plot of QQQ distance from 200MA for all Top8 winners

**Question.** Plot QQQ's distance from the hourly 200-day MA as a function of time for all Top8 winning round-trip trades, renormalizing every trade's x-axis to the same 0-to-1 range. For the current/latest trade, use today as the normalization endpoint. Mark the peak position for the 7 completed trades.

**Definition.** The y-axis is `QQQ close / QQQ hourly 200-day MA - 1`. The x-axis is normalized trade progress: entry equals 0, completed-trade exit equals 1, and current/latest T55 uses today at 15:00 as 1. Peak markers are the synthetic `QQQ_3X_CALC` trade-price peaks for the 7 completed Top8 trades, not the peak QQQ/200MA-distance points.

**Implementation.** Added `scripts/plot_top8_normalized_qqq_200ma_distance.py`.

**Outputs.**
- `reports/figures/preferred_top8_normalized_qqq_200ma_distance.png`
- `reports/tables/preferred_top8_normalized_qqq_200ma_distance_series.csv`
- `reports/tables/preferred_top8_normalized_qqq_200ma_distance_peak_markers.csv`

**Result.** The completed-trade synthetic-3x price peaks mostly occurred late in the normalized trade life, around 0.74 to 0.97 of the trade duration. At those peak points, QQQ was still above its hourly 200-day MA by about +9.1% to +19.7%, reinforcing that the 200MA exit is a slow trend gate rather than a profit-protection signal for extended winners.

### 2026-06-03 — QQQ/200MA-distance stats before first +100% synthetic-3x gain

**Question.** For the Top8 winning trades, check whether QQQ's distance-from-hourly-200MA statistics differ a lot if each trade only uses data before it first reaches +100% synthetic `QQQ_3X_CALC` unrealized gain. If a trade never reaches +100%, keep its available data in the inclusive pre-100 sample and also report a hit-only sample.

**Implementation.** Added `scripts/analyze_top8_pre100_qqq_200ma_distance.py`.

**Outputs.**
- `reports/tables/preferred_top8_pre100_qqq_200ma_distance_comparison.csv`
- `reports/tables/preferred_top8_pre100_qqq_200ma_distance_by_trade_summary.csv`
- `reports/tables/preferred_top8_pre100_qqq_200ma_distance_observations.csv`
- `reports/figures/preferred_top8_pre100_qqq_200ma_distance_comparison.png`

**Result.** The center of the distribution does not change much. Full Top8 mean/median QQQ distance was +11.42%/+10.70%; the pre-first-100 sample was +11.58%/+10.63%. The main difference is in the tails/buckets: pre-100 has no >=30% observations, a slightly higher p90 (+20.56% vs +19.40%), and a larger 20-30% share (11.56% vs 8.50%). Interpretation: before the first +100% synthetic-3x gain, QQQ is already typically extended above its 200MA; this statistic is not materially different enough by itself to define a clean "early big winner" filter.

### 2026-06-03 — Promote +300/+400 profit lock plus 40% peak stop to preferred strategy

**User decision.** Promote the strategy variant `profit_lock_300_400_stop_40pct` to the updated preferred strategy.

**Confirmed preferred rule.** QQQ hourly MACD histogram entry, QQQ hourly 200-day MA entry/exit gate, no daily regime gate, synthetic `QQQ_3X_CALC` exposure, max one trade per day, 3% out-of-market cash assumption, +300% unrealized synthetic-3x trade gain -> 75% exposure, +400% -> 50% exposure, and a 40% synthetic-3x trade-peak stop.

**Reference result.** Final return +43,107.1%, annualized return 25.92%, Sharpe 0.776, max drawdown -56.36%, 115 trades, exposure 65.37%, and drawdown episodes >20/>30/>40/>50% = 24/13/7/3. The 40% peak stop triggered once historically.

**Files updated.** `reports/preferred_strategy_rules.md`, `README.md`, `docs/qqq_tqqq_case_study.md`, `docs/hourly_update_workflow.md`, and `scripts/update_preferred_strategy_signal.py`.

### 2026-06-03 — Drawdown-reduction overlay experiment suite and local website

**Question.** Starting from the updated preferred strategy, test whether mean-reversion overlays and Fed hiking-cycle overlays can reduce max drawdown and large drawdown counts without materially hurting annualized return. Also create a local website to review the backtesting results.

**Implementation.**
- Added `src/trend_following/risk_overlays.py` for reusable profit-lock, peak-stop, extension-trim/rebuy, soft peak-drawdown trim/rebuy, and QQQ mean-reversion feature helpers.
- Added `src/trend_following/fed_cycles.py` for no-lookahead Fed-cycle flags and monthly QQQ P/E alignment.
- Added `configs/fed_hiking_cycles.yaml` with effective, announced, and oracle-pre-announcement hiking windows.
- Added `scripts/run_preferred_drawdown_reduction_experiments.py`.
- Generated a static local website at `reports/site/index.html`.

**Outputs.**
- `reports/tables/preferred_dd_reduction_experiments_compact.csv`
- `reports/tables/preferred_dd_reduction_experiments_metrics.csv`
- `reports/tables/preferred_dd_reduction_experiments_returns.csv`
- `reports/tables/preferred_dd_reduction_experiments_weights.parquet`
- `reports/tables/preferred_dd_reduction_experiments_fed_cycles.csv`
- `reports/tables/preferred_dd_reduction_experiments_fed_and_pe_flags.csv`
- `reports/figures/preferred_dd_reduction_experiments_return_vs_drawdown.png`
- `reports/figures/preferred_dd_reduction_experiments_top_candidates_equity_drawdown.png`
- `reports/figures/preferred_dd_reduction_experiments_worst_dd_comparison.png`
- `reports/site/index.html`

**Result.** The first compact pass did not find an overlay that materially improves the single worst max drawdown while preserving return. The best extension mean-reversion variants improved annualized return and Sharpe, and some reduced the count of >30% or >40% drawdown episodes, but the worst max drawdown stayed around -56%. The best objective-row was `extension_trim_g200_dist22_to50_rema20`: final return +55,001%, annualized return 27.08%, Sharpe 0.801, max drawdown -56.36%, 127 trades, and DD episode counts 25/12/7/3. The baseline remains +43,107%, 25.92% annualized, Sharpe 0.776, max drawdown -56.36%, 115 trades, and DD counts 24/13/7/3. Hiking-cycle overlays reduced some >50% episode counts but generally reduced annualized return and did not improve max drawdown in this pass.

**Interpretation.** The Top-8 distance-to-200MA mean-reversion information is useful for return/Sharpe and drawdown-episode tuning, but it is not yet enough to solve the worst max-drawdown problem. The next serious direction should isolate the specific worst drawdown path and test targeted earlier exits or hedges that activate before those episodes, rather than broad cycle caps.

### 2026-06-03 — +100% split-to-MACD sleeve overlay on preferred +40% stop strategy

**Question.** Starting from the current preferred strategy plus a synthetic-3x 40% trade-peak stop, test this rule: whenever an open trade reaches +100% unrealized return, split capital into two halves. One half keeps following the preferred strategy; the other half switches to a faster MACD exit/re-entry sleeve until the QQQ 200MA/base-trade exit signal sends everything to cash.

**Implementation.** Added `scripts/run_preferred_split100_macd_sleeve.py`. The preferred half keeps the existing `+300% -> 75%, +400% -> 50%` profit-lock sizing within that half. The fast sleeve uses QQQ SMA-MACD histogram: confirmed `hist > 0` to be invested and confirmed `hist <= 0` to be out. The split trigger is based on synthetic `QQQ_3X_CALC` return from the current base-trade entry. Raw split and MACD sleeve decisions are shifted by the existing no-lookahead executable-weight convention and still obey max-one-trade-per-day.

**Variants.**
- `split100_macd_sleeve_global_stop40`: the 40% trade-peak stop forces both sleeves to cash.
- `split100_macd_sleeve_branch_stop40`: the stop applies to the preferred sleeve; the fast sleeve exits on MACD/200MA.
- `split100_macd_sleeve_future_exit_global_stop40`: same as global stop, but the fast sleeve is not allowed to exit on the same bar that first triggers the +100% split.

**Outputs.**
- `reports/tables/preferred_split100_macd_sleeve_compact.csv`
- `reports/tables/preferred_split100_macd_sleeve_metrics.csv`
- `reports/tables/preferred_split100_macd_sleeve_returns.csv`
- `reports/tables/preferred_split100_macd_sleeve_weights.csv`
- `reports/tables/preferred_split100_macd_sleeve_diagnostics.parquet`
- `reports/figures/preferred_split100_macd_sleeve_equity_drawdown.png`

**Result.** No improvement. The current preferred `+300/+400 profit lock + 40% stop` stayed best: final return 43,107.1%, annualized return 25.92%, Sharpe 0.776, max drawdown -56.36%, and 115 trades. The best split variant returned 19,192.0%, annualized 22.12%, Sharpe 0.731, max drawdown still -56.36%, and 263 trades. It reduced large drawdown episode counts to 18/12/6/1 for >20/>30/>40/>50%, but the return cost and extra trading were too large. Interpretation: after a trade is already up +100%, forcing half the capital into a faster MACD sleeve cuts too much exposure during the biggest winners; it improves some drawdown-count statistics but does not improve the worst max drawdown or return quality. Do not promote this rule.

### 2026-06-03 — Normalized-time QQQ distance paths for all Top8 winners

**Question.** For all Top8 winning trades, plot QQQ's distance from its hourly 200-day MA as a function of time, renormalizing each trade's x-axis to the same range. For the current/latest trade, use today as the end time.

**Definition.** For every Top8 trade, x-axis is normalized trade time: `entry = 0`, `exit = 1`. For the current/latest T55 trade, the normalization endpoint is today's market close, `2026-06-03 16:00`. Distance is `QQQ close / QQQ hourly 200-day MA - 1`. The long-history Alpha Vantage 60min QQQ cache is used, with the local Yahoo 60min QQQ cache appended only for newer bars. The current-trade plotted line stops at the latest available local bar rather than forward-filling missing bars.

**Implementation.** Added `scripts/plot_top8_normalized_qqq_200ma_distance.py`.

**Outputs.**
- `reports/figures/preferred_top8_normalized_qqq_200ma_distance.png`
- `reports/tables/preferred_top8_normalized_qqq_200ma_distance_paths.csv`
- `reports/tables/preferred_top8_normalized_qqq_200ma_distance_summary.csv`

**Result.** The current/latest T55 trade is still far above the hourly 200-day MA versus the historical Top8 end conditions. With normalization endpoint `2026-06-03 16:00`, the latest local plotted bar was `2026-06-02 15:30`, at normalized time 0.982 and QQQ distance +20.6%. Most completed Top8 trades ended near or below the 200MA, while the current trade remains extended above it.

### 2026-06-03 — Normalized-time plot of QQQ distance from hourly 200-day MA during completed Top8 winners

**Question.** For all Top8 winning trades except the current/latest one, plot QQQ's distance from its hourly 200-day MA as a function of time, renormalizing all seven trades' x-axes into the same range and overlaying them in one plot.

**Definition.** The x-axis is normalized trading-time progress from 0% at the first hourly bar of each round trip to 100% at the final hourly bar. Distance remains `QQQ close / QQQ hourly 200-day MA - 1`.

**Implementation.** Added `scripts/plot_top8_qqq_200ma_distance_normalized_time.py`. The script interpolates each completed Top8 trade onto a common 0-100% grid, overlays all seven paths, and adds a cross-trade median line plus 25th-75th percentile band.

**Outputs.**
- `reports/figures/preferred_top8_ex_current_qqq_200ma_distance_normalized_time.png`
- `reports/tables/preferred_top8_ex_current_qqq_200ma_distance_normalized_time_summary.csv`
- `reports/tables/preferred_top8_ex_current_qqq_200ma_distance_normalized_time_grid.csv`
- `reports/tables/preferred_top8_ex_current_qqq_200ma_distance_normalized_time_observations.csv`

**Result.** The cross-trade median distance starts near +1.1%, rises to about +15.7% near the normalized midpoint, and ends near -0.7%. The completed Top8 winners generally begin close to the 200MA gate, become strongly extended above it during the middle of the trade, and exit close to or slightly below the 200MA.

### 2026-06-03 — Dynamic pre-+100% QQQ/200MA-distance trim overlay

**Question.** Replace the fixed QQQ distance threshold used by the mean-reversion trim overlay with a dynamic threshold learned from each trade's own data up to the first +100% synthetic-3x unrealized return. Only allow the trim logic after the trade is already up at least +100%, count how many of the 55 baseline trades trigger it, and evaluate the best candidate during hiking-cycle windows defined from one month before hike-announcement guidance to one month before the first cut.

**Implementation.** Added `dynamic_pre100_distance_trim_rebuy_cap` to `src/trend_following/risk_overlays.py` and `scripts/run_preferred_dynamic_trim_experiments.py`. For each baseline trade, the overlay learns the selected quantile of QQQ's hourly distance from its 200-day MA using bars from trade entry through the first +100% synthetic-3x gain. After that point, if QQQ revisits/exceeds the learned threshold, exposure is capped at 50%; full exposure is restored when QQQ pulls back to its 20-day MA. The raw overlay is still shifted through the existing no-lookahead executable-weight convention.

**Outputs.**
- `reports/tables/preferred_dynamic_trim_experiments_compact.csv`
- `reports/tables/preferred_dynamic_trim_experiments_metrics.csv`
- `reports/tables/preferred_dynamic_trim_trade_triggers.csv`
- `reports/tables/preferred_dynamic_trim_hiking_cycle_performance.csv`
- `reports/tables/preferred_dynamic_trim_fed_cycles.csv`

**Result.** The best objective candidate was `dynamic_pre100_q100_to50_rema20`, meaning the threshold is the maximum QQQ/200MA distance observed before the first +100% trade gain. It triggered in 6 of 55 baseline trades with 21 total trim events. Full-sample result: final return +55,241%, annualized return 27.11%, Sharpe 0.805, max drawdown -56.36%, 150 trades, and DD episode counts 25/10/6/4. Lower quantiles trigger more often and slightly reduce max drawdown, but they add many trades and reduce annualized return; for example q50 triggered in 7/55 trades with 890 trim events, annualized return 24.26%, max drawdown -55.40%, and DD counts 18/10/6/2.

**Hiking-cycle result.** For diagnostic windows from one month before hike-announcement guidance to one month before the first cut, the best dynamic candidate matched baseline in 2004-2007, improved 2015-2019 cumulative return from +111.6% to +138.2% and max drawdown from -47.4% to -44.1%, and improved 2021-2024 cumulative return from +166.4% to +184.8% and max drawdown from -37.6% to -32.1%. This is encouraging for cycle-specific behavior, but it still does not solve the full-sample worst max drawdown.

### 2026-06-03 — Promoted dynamic q100 trim into preferred strategy

**Decision.** Promoted `dynamic_pre100_q100_to50_rema20` into the confirmed preferred strategy. The preferred rule is now the QQQ hourly 200MA/MACD synthetic-TQQQ strategy with +300/+400 profit locks, dynamic q100 mean-reversion trim, and 40% synthetic-3x trade-peak stop.

**Rule added.** For each open synthetic-TQQQ trade, wait until unrealized synthetic-3x gain first reaches +100%. Using only bars from trade entry through that first +100% bar, learn `q100 = max(QQQ close / QQQ hourly 200-day MA - 1)`. After +100%, if QQQ revisits/exceeds that learned distance, cap exposure at 50%. Restore full exposure when QQQ pulls back to/touches its hourly 20-day MA. Raw trim/re-entry signals are still shifted through the no-lookahead executable-position convention.

**Updated preferred result.** `dynamic_pre100_q100_to50_rema20`: final return +55,241%, annualized return 27.11%, Sharpe 0.805, max drawdown -56.36%, 150 trades, exposure 63.95%, DD episode counts 25/10/6/4. It triggered in 6 of 55 baseline trades with 21 total trim events.

**Worst max drawdown.** The worst drawdown remains unchanged: peak `2007-10-31 15:00`, trough `2009-07-08 12:00`, recovery `2009-12-24 11:00`, max drawdown -56.36%, over 615 calendar days peak-to-trough. The dynamic q100 overlay did not trigger inside this peak-to-trough window because the trades during this drawdown did not reach +100% before losses/whipsaws occurred. This identifies the next problem as a bear-market whipsaw / re-entry-risk problem, not a late-stage winner-overextension problem.

**Files updated.** `reports/preferred_strategy_rules.md`, `docs/qqq_tqqq_case_study.md`, `README.md`, `scripts/update_preferred_strategy_signal.py`, and `scripts/run_preferred_dynamic_trim_experiments.py`.

### 2026-06-04 — Bear-market whipsaw reduction experiment suite

**Question.** Starting from the updated preferred q100 strategy, test whether targeted bear-market whipsaw overlays can reduce the unchanged worst drawdown, especially the 2007-2009 peak-to-trough episode, without cutting annualized return below about 24% or raising trading frequency above about 8 trades/year.

**Implementation.** Added `src/trend_following/bear_whipsaw.py` and `scripts/run_preferred_bear_whipsaw_reduction_experiments.py`. The experiment keeps the current preferred baseline: QQQ hourly MACD/200MA entry-exit, synthetic `QQQ_3X_CALC`, +300/+400 profit locks, dynamic q100 trim after +100%, 40% synthetic trade-peak stop, max-one-trade-per-day execution, costs/slippage/tax approximation, and 3% cash. Tested bear re-entry filters, failed-breakout cooldowns, crisis-volatility caps, portfolio drawdown circuit breakers, and two-stage bear re-entry. Added unit tests in `tests/test_bear_whipsaw.py`.

**Outputs.**
- `reports/tables/preferred_bear_whipsaw_experiments_compact.csv`
- `reports/tables/preferred_bear_whipsaw_experiments_metrics.csv`
- `reports/tables/preferred_bear_whipsaw_experiments_returns.csv`
- `reports/tables/preferred_bear_whipsaw_experiments_weights.parquet`
- `reports/tables/preferred_bear_whipsaw_experiments_diagnostics.parquet`
- `reports/tables/preferred_bear_whipsaw_worst_drawdown_summary.csv`
- `reports/tables/preferred_bear_whipsaw_2007_2009_trigger_summary.csv`
- `reports/tables/preferred_bear_whipsaw_hiking_cycle_performance.csv`
- `reports/figures/preferred_bear_whipsaw_return_vs_2007_2009_dd.png`
- `reports/figures/preferred_bear_whipsaw_top_equity_drawdown.png`
- `reports/site/bear_whipsaw.html`

**Result.** The best objective candidate was `bear_reentry_buf1_slope20_20gt50`: when QQQ's 200-day MA slope is negative, new entries require QQQ to be at least 1% above its hourly 200-day MA, QQQ 50MA slope over 20 trading days to be positive, and QQQ 20MA > 50MA. Result: final return +77,466%, annualized return 28.75%, Sharpe 0.844, max drawdown -52.15%, 2007-2009 max drawdown -51.01%, 134 trades, 5.09 trades/year, exposure 62.87%, and DD counts 24/8/5/3. Baseline q100 was +55,241%, annualized 27.11%, Sharpe 0.805, max drawdown -56.36%, 2007-2009 max drawdown -56.36%, 150 trades, 5.70 trades/year, and DD counts 25/10/6/4.

**Interpretation.** The bear re-entry filter is the first overlay in this sequence that materially improves the 2007-2009 whipsaw drawdown while also improving return and reducing trade count. Portfolio drawdown circuit breakers reduced the 2007-2009 drawdown more aggressively in some cases, but generally paid for it with lower full-sample return. The next serious candidate for promotion is therefore the `1% buffer + 20-day 50MA slope positive + 20MA>50MA` bear re-entry filter on top of the current q100 preferred strategy.

### 2026-06-04 — Trade-level inspection of best bear re-entry filter

**Question.** Inspect the trade-level behavior of the best bear-whipsaw candidate, `bear_reentry_buf1_slope20_20gt50`, around 2004-2005, 2007-2009, and 2010 versus the q100 preferred baseline.

**Implementation.** Added `scripts/inspect_bear_whipsaw_trade_behavior.py`. The script reads the saved bear-whipsaw experiment returns/weights/diagnostics, extracts executable trade intervals, contiguous blocked-entry intervals, period-local returns/drawdowns, and period plots with QQQ, 20/50/200MA lines, blocked raw entries, executable weights, and period-local drawdowns.

**Outputs.**
- `reports/tables/preferred_bear_whipsaw_trade_behavior_period_summary.csv`
- `reports/tables/preferred_bear_whipsaw_trade_behavior_trades.csv`
- `reports/tables/preferred_bear_whipsaw_trade_behavior_blocked_entries.csv`
- `reports/tables/preferred_bear_whipsaw_trade_behavior_blocked_entry_intervals.csv`
- `reports/figures/preferred_bear_whipsaw_trade_behavior_2004_2005.png`
- `reports/figures/preferred_bear_whipsaw_trade_behavior_2007_2009.png`
- `reports/figures/preferred_bear_whipsaw_trade_behavior_2010.png`

**Findings.** In 2007-2009 the filter blocked three raw-entry spans: 2008-05-01 12:00 to 14:00, 2008-08-08 15:00 to 2008-08-21 11:00, and 2008-08-22 11:00 to 2008-08-25 11:00. This avoided the baseline's two August 2008 failed re-entry trades and reduced position changes from 13 to 9, improving period-local return from +66.7% to +87.2% and max drawdown from -56.36% to -51.01%. In 2004-2005 the filter delayed the 2005-10-26 entry until 2005-11-08, slightly improving local max drawdown but hurting period-local return. In 2010 it made no entry-blocking changes; behavior was effectively identical except for inherited q100/profit-lock sizing effects.

### 2026-06-04 — Bear re-entry filter marked as serious candidate

**Decision.** Marked `bear_reentry_buf1_slope20_20gt50` as a serious candidate, not yet preferred. Added `reports/serious_candidates.md`.

**Trigger years.** The filter blocked raw entries in 2002, 2003, 2005, 2008, 2011, 2012, 2016, 2023, and 2025. The largest blocked-entry periods were 2003-03-17 to 2003-04-04, 2005-10-26 to 2005-11-08, 2008-08-08 to 2008-08-21, 2016-05-24 to 2016-06-10, and 2025-05-12 to 2025-05-16.

**Worst five drawdowns under the candidate.** 2010-04-23 to 2010-08-11 (-52.15%), 2004-01-20 to 2006-10-03 (-51.34%), 2007-10-31 to 2009-07-08 (-51.01%), 2018-10-01 to 2019-08-05 (-45.53%), and 2011-05-02 to 2011-12-30 (-44.91%). These motivate the next variants: combine the bear re-entry filter with post-big-winner protection for 2010, a lighter recovery filter for 2004-2006, additional crisis-vol/circuit-breaker logic for 2007-2009 and 2011, and hiking-cycle-aware/volatility-aware sizing for 2018-2019.

### 2026-06-04 — Bear filter variant suite and dynamic q100 activation sweep

**Question.** Extend the serious bear-filter candidate tests by trying q100 activation at +90%, +80%, +70%, +60%, and +50% instead of only +100%, and by testing variants motivated by the candidate's worst five drawdowns.

**Implementation.** Added `scripts/run_bear_filter_variant_experiments.py`. It evaluates dynamic q100 activation variants, robust bear-filter parameter variants, partial bear-entry variants, post-big-winner protection overlays, bear-regime volatility caps, and bear-regime circuit breakers. Added a local website page `reports/site/bear_filter_variants.html` and aggregate dashboard `reports/site/all_backtests.html`.

**Outputs.**
- `reports/tables/bear_filter_variant_experiments_compact.csv`
- `reports/tables/bear_filter_variant_experiments_metrics.csv`
- `reports/tables/bear_filter_variant_experiments_returns.csv`
- `reports/tables/bear_filter_variant_experiments_weights.parquet`
- `reports/tables/bear_filter_variant_diagnostics_summary.csv`
- `reports/tables/bear_filter_variant_worst5_drawdowns.csv`
- `reports/tables/bear_filter_variant_period_summary.csv`
- `reports/figures/bear_filter_variant_return_vs_dd.png`
- `reports/figures/bear_filter_variant_top_equity_drawdown.png`
- `reports/site/bear_filter_variants.html`
- `reports/site/all_backtests.html`

**Result.** The best objective row was `robust_buf010bp_slope30_20gt50`, a robustness variant requiring QQQ to be at least 1% above the 200MA, 50MA slope positive over 30 trading days, and 20MA > 50MA when the 200MA slope is negative. It improved annualized return to 29.18% and Sharpe to 0.853, with max drawdown unchanged versus the serious candidate at -52.15%. Dynamic q100 activations below +100% did not improve the drawdown problem: +90% was similar to +100%, while +80% to +50% generally added trades and reduced return without improving the 2010 drawdown. Post-big-winner protection improved the 2010 drawdown and full max drawdown in some cases, but usually with too many trades and/or lower annualized return. No variant passed the strict filter of annualized return >= 26%, max DD better than -50%, and trades/year <= 8.

### 2026-06-04 — Promoted q110 + best robustness bear filter to preferred

**Question.** Use q100 activation at `+110%` and the best robustness bear filter as the preferred strategy.

**Decision.** Promoted the best current robustness compromise into the confirmed preferred rule:

- QQQ hourly MACD histogram > 0 entry.
- QQQ hourly close > QQQ hourly 200-day MA entry gate.
- QQQ hourly close < QQQ hourly 200-day MA exit.
- Synthetic `QQQ_3X_CALC` exposure.
- `+300% -> 75%`, `+400% -> 50%` profit lock.
- Dynamic q100 trim activates after `+110%` synthetic-3x trade gain instead of `+100%`.
- Best robustness bear filter: if QQQ hourly 200MA slope is negative, allow a new entry only when QQQ is at least 1% above the hourly 200MA, QQQ 50MA slope over 30 trading days is positive, and QQQ 20MA > QQQ 50MA.
- 40% synthetic trade-peak stop, max one trade/day, 3% cash assumption.

**Main long-history result.** The promoted `robust_slope30_q110` variant returned 88,809.7% cumulative / 29.41% annualized, Sharpe 0.857, max drawdown -52.15%, 128 trades, 4.86 trades/year, exposure 63.09%, and DD episode counts 23/8/5/3 for >20/>30/>40/>50%.

**2010+ q100 activation sweep.** From 2010-01-01 onward, q110 was also best by annualized return for both q100-only and q100 + best robustness bear-filter versions. Saved tables:

- `reports/tables/preferred_q100_activation_sweep_2010plus.csv`
- `reports/tables/preferred_q100_activation_sweep_full_history_100_150.csv`

**Implementation updates.** Updated the preferred rules document, QQQ/TQQQ case study, hourly update workflow, serious-candidates provenance file, local qqq-tqqq updater skill, and `scripts/update_preferred_strategy_signal.py` so current monitoring applies the q110 dynamic trim and best robustness bear filter before no-lookahead executable-position shifting.

### 2026-06-04 — Default requested backtest start changed to 1990-01-01

**Question.** Extend the default backtest starting day to `1990-01-01`.

**Implementation.** Updated the main default date settings:

- `configs/default.yaml`: `data.start_date = "1990-01-01"`
- `configs/regime_hourly_qqq.yaml`: `data.start_date = "1990-01-01"`
- `configs/alpha_vantage_max_history.yaml`: `data.start_date = "1990-01-01"`
- `src/trend_following/config.py`: fallback `DataConfig.start_date = "1990-01-01"`

**Important caveat.** This changes the requested start date, not the guaranteed effective data start. ETF inception dates and vendor history still control the actual first available bar. The current local Alpha Vantage 60-minute QQQ cache starts at `2000-01-03 10:00`, and the latest preferred-strategy table's first executable long entry is `2002-01-10 15:00`. A true 1990 QQQ-like backtest would require adding a documented pre-QQQ proxy such as Nasdaq-100 index data.

### 2026-06-04 — Accepted Nasdaq-100 as documented pre-QQQ proxy

**Decision.** For `1990-01-01` through the start of actual QQQ data, Nasdaq-100 index data may be used as a **pre-QQQ proxy** for QQQ-like research, but it must be labeled as a proxy and not as actual QQQ ETF data.

**Documentation.** Added `docs/pre_qqq_proxy.md` and linked it from the README and QQQ/synthetic-TQQQ case study. The policy requires keeping actual QQQ raw data unchanged, using a separate proxy series such as `QQQ_NDX_PROXY`, and reporting requested start date separately from effective/proxy data start.

**Caveat.** A daily Nasdaq-100 proxy is acceptable for signal warmup and daily sensitivity tests. For the current hourly preferred strategy, pre-QQQ hourly Nasdaq-100 data may not be freely available; do not fabricate hourly bars from daily data for primary performance claims.

### 2026-06-04 — Alpha Vantage Nasdaq-100 hourly proxy probe failed

**Question.** Download the pre-QQQ Nasdaq-100 hourly proxy data from Alpha Vantage.

**Attempt.** Used the local Alpha Vantage API key and probed common Nasdaq-100 index symbols with `TIME_SERIES_INTRADAY` and daily-adjusted endpoints: `NDX`, `^NDX`, `NASDAQ:NDX`, `INDEXNASDAQ:NDX`, and `NDX.X`. Also queried Alpha Vantage symbol search for `NASDAQ 100`, `Nasdaq-100`, `NDX`, and `NASDAQ100`.

**Result.** Alpha Vantage did not return usable Nasdaq-100 index hourly bars for those symbols. Symbol search found QQQ and some Nasdaq-100 mutual funds/ETFs, but not a direct historical intraday Nasdaq-100 index symbol suitable for a 1990-1999 pre-QQQ hourly proxy.

**Outputs.** Saved probe tables:

- `reports/tables/alpha_vantage_ndx_symbol_probe.csv`
- `reports/tables/alpha_vantage_ndx_symbol_search.csv`

**Decision.** Do not claim a 1990-1999 hourly QQQ proxy from Alpha Vantage. Keep the 1990 start as requested, but the current Alpha Vantage hourly QQQ/synthetic-TQQQ preferred-strategy backtest still effectively starts with actual QQQ hourly data in 2000. If extending to 1990 remains important, search for a reliable Nasdaq-100 index source outside Alpha Vantage, likely daily first.

### 2026-06-04 — Start-date and walk-forward parameter CV implemented

**Question.** Vary key parameters and cross-validate the preferred QQQ/synthetic-TQQQ strategy across different starting times.

**Implementation.** Added reusable CV utilities and two experiment drivers:

- `scripts/preferred_cv_utils.py`
- `scripts/run_preferred_start_date_cv.py`
- `scripts/run_preferred_walk_forward_cv.py`

The start-date CV evaluates controlled one-factor variants around the current preferred rule across 13 evaluation starts from `2002-01-10 15:00` through `2022-01-01`, with QQQ buy-and-hold aligned to the same start date. Parameter families include long MA days, entry/exit confirmation bars, MACD fast/slow/signal windows, q100 activation/trim/re-entry settings, bear-filter buffer/slope/20MA>50MA settings, peak-stop thresholds, and profit-lock schemes.

**Outputs.**

- `reports/tables/preferred_start_date_cv_metrics.csv`
- `reports/tables/preferred_start_date_cv_summary.csv`
- `reports/tables/preferred_parameter_robustness_rank.csv`
- `reports/tables/preferred_start_date_cv_diagnostics.csv`
- `reports/tables/preferred_start_date_cv_weights.parquet`
- `reports/figures/preferred_start_date_cv_heatmap.png`
- `reports/figures/preferred_parameter_rank_stability.png`
- `reports/site/start_date_cv.html`
- `reports/tables/preferred_walk_forward_cv.csv`
- `reports/tables/preferred_walk_forward_cv_all_candidates.csv`
- `reports/tables/preferred_walk_forward_cv_returns.csv`
- `reports/figures/preferred_walk_forward_cv_equity_drawdown.png`
- `reports/site/walk_forward_cv.html`

**Start-date CV result.** The current preferred rule ranked 15 of 62 by the robustness score, with median annualized return 42.72%, 10th-percentile annualized return 32.04%, worst max drawdown -52.15%, and median trades/year 5.38 across start dates. The highest-scoring variant was `macd_signal_8d`, with median annualized return 43.81%, 10th-percentile annualized return 32.68%, the same worst max drawdown -52.15%, and median trades/year 5.43.

**Walk-forward result.** Walk-forward parameter selection did not consistently beat the frozen preferred rule. Selected variants underperformed the current preferred in 2007-2010, 2011-2014, and 2019-2022, outperformed in 2015-2018, and was essentially tied in 2023-2026. This argues against promoting a new parameter set based only on start-date CV. `macd_signal_8d` is a candidate for further testing, not a confirmed preferred-rule change.

**Method caveat.** The CV uses the already constructed no-lookahead executable weights and slices the simulated return/turnover paths for speed. It is intended as a robustness screen. Any final candidate promotion should be rerun with exact fold-local tax/state resets if the differences are small.

### 2026-06-04 — Promoted 12/24/9 MACD option, retained MACD robustness set

**Question.** Use `macd_slow_24d` as the preferred MACD option, while keeping the standard 12/26/9 and faster-signal 12/26/8 options visible because all three are very similar and may be overfit.

**Implementation.** Updated the preferred-rule documentation and Yahoo updater defaults so the active preferred MACD option is now SMA-MACD **12/24/9** trading-day windows. Added explicit updater arguments `--macd-fast-days`, `--macd-slow-days`, and `--macd-signal-days` so the retained MACD options can be run without code changes. Updated the shared CV utilities so future robustness runs treat 12/24/9 as the current preferred default and keep 12/26/9, 12/26/8, and 12/24/9 as named MACD options.

**Output.** Added `reports/tables/preferred_macd_options_comparison.csv` summarizing the three MACD choices under the same official-start convention.

**Decision.** `macd_slow_24d` / 12-24-9 is now the active preferred default, but this is treated as a small robustness preference rather than a precise optimized edge. Future reports should continue to show all three MACD options when comparing parameter robustness.

### 2026-06-04 — Annual net-tax accounting audit for preferred-variant CV

**Question.** If realized gains are not taxed immediately but are netted and taxed at year-end, do parameter variants make a difference?

**Implementation.** Confirmed the preferred-strategy simulator uses annual net-tax accounting: realized gains/losses accumulate during each calendar year, net taxable gains are taxed at year-end, losses carry forward, and cash interest is also taxed at year-end. Added explicit `tax_timing = annual_net_eoy` labels to preferred CV outputs, annual tax-payment diagnostics, and unit tests for year-end tax timing, same-year loss offsets, loss carryforward, and cash-interest tax timing.

**Outputs.** Refreshed the existing preferred start-date CV variant set using the current 12/24/9 preferred MACD default and retained 12/26/9 and 12/26/8 MACD options. Saved:

- `reports/tables/preferred_start_date_cv_metrics.csv`
- `reports/tables/preferred_start_date_cv_summary.csv`
- `reports/tables/preferred_parameter_robustness_rank.csv`
- `reports/tables/preferred_start_date_cv_tax_audit.csv`
- `reports/tables/preferred_start_date_cv_annual_tax_payments.csv`
- `reports/tables/preferred_macd_options_comparison.csv`
- Updated `reports/site/start_date_cv.html` with an Annual Net Tax Check section.

**Result.** The tax audit passed for all 64 official-start simulations: positive tax payments occurred only on year-end/final-liquidation bars, with zero non-year-end tax-payment dates. The three retained MACD options remain close under annual net-tax accounting. Official-start results: 12/24/9 returned 95,949% cumulative / 32.62% annualized with -52.79% max DD; 12/26/8 returned 95,351% cumulative / 32.58% annualized with -52.15% max DD; 12/26/9 returned 84,863% cumulative / 31.95% annualized with -52.15% max DD. Across the broader refreshed CV ranking, `macd_fast_14d` ranked highest by robustness score, but this remains a variant-screen result rather than a promoted preferred-rule change.

**Caveat.** The refreshed start-date CV uses the existing fast-slice CV methodology for tractability, while the tax audit and unit tests verify the annual net-tax simulator itself. Exact per-start tax-state-reset simulation was attempted but was too slow for this run.

### 2026-06-04 — Next planned direction: Monte Carlo simulation for preferred strategy

**Question.** After refreshing the annual net-tax CV results, evaluate whether the preferred strategy is robust under resampled market paths rather than only the realized historical path.

**Planned scope.** Add a Monte Carlo / block-bootstrap research module for the preferred QQQ/synthetic-TQQQ strategy. Initial tests should resample return blocks with volatility clustering preserved where possible, rerun the strategy path with the same no-lookahead timing and annual net-tax convention, and summarize distributions for CAGR, final wealth, max drawdown, drawdown episodes above 20/30/40/50%, trades/year, exposure, and time underwater.

**Decision.** Do not change the preferred rule yet. Treat Monte Carlo as the next robustness analysis layer on top of the current preferred rule and retained MACD variants.
