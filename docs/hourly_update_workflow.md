# Hourly Update Workflow for Preferred QQQ / Synthetic-TQQQ Strategy

_Last updated: 2026-06-04_

This document records how to refresh the data and rerun the current preferred strategy signal during each trading hour. In this workflow, "update the data" means pulling the newest vendor data available and merging it into the project's local history files, not just overwriting a one-off snapshot.

This is for research monitoring only, not financial advice.

## Preferred strategy being monitored

Current preferred rule:

- Signal source: QQQ 60-minute bars.
- Exposure target: synthetic +3x QQQ exposure, `QQQ_3X_CALC`.
- Entry: QQQ hourly SMA-MACD histogram > 0, currently using the 12/24/9 trading-day MACD option.
- Entry gate: QQQ hourly close > QQQ hourly 200-day MA.
- Exit: QQQ hourly close < QQQ hourly 200-day MA.
- Profit lock: +300% unrealized synthetic-3x trade gain -> 75%; +400% -> 50%.
- Dynamic q100 trim: after +110% unrealized synthetic-3x trade gain, learn the max QQQ distance above its hourly 200MA up to that point; later trim to 50% if QQQ revisits/exceeds it, and re-add on a QQQ 20MA pullback.
- Best robustness bear filter: if QQQ's hourly 200MA slope is negative, delay new entries until QQQ is at least 1% above the 200MA, QQQ 50MA slope over 30 trading days is positive, and QQQ 20MA > QQQ 50MA.
- Trade-peak stop: exit if synthetic `QQQ_3X_CALC` falls 40% from its current trade peak.
- No daily regime gate.
- Max one trade per day.
- Out-of-market cash earns 3% annualized in evaluation.

## Manual one-shot update

```bash
cd /path/to/trend-following
source ~/.venvs/myenv/bin/activate
python scripts/update_preferred_strategy_signal.py
cat reports/tables/preferred_strategy_current_signal.csv
```

The updater refreshes and appends/merges local history for:

| Data | Source | Local history file | Update cadence |
|---|---|---|---|
| QQQ 60-minute OHLCV | Yahoo Finance via `yfinance.download`, regular-session bars | `data/raw/yfinance_60min/QQQ.parquet` | Every updater run |
| Actual TQQQ 60-minute OHLCV | Yahoo Finance via `yfinance.download`, regular-session bars | `data/raw/yfinance_60min/TQQQ.parquet` | Every updater run |
| QQQ daily adjusted OHLCV | Yahoo Finance via `yfinance.download` daily bars | `data/raw/yfinance_daily/QQQ.parquet` | Every updater run |
| Synthetic +3x QQQ | Calculated locally from QQQ | `data/raw/synthetic_yfinance_3x_60min/QQQ_3X_CALC.parquet` and `data/raw/synthetic_yfinance_3x_1d/QQQ_3X_CALC.parquet` | Rebuilt every updater run |
| QQQ P/E snapshot | Yahoo Finance via `yfinance.Ticker("QQQ").get_info()` | `data/processed/valuation/qqq_pe_yfinance_snapshot_history.parquet` and `reports/tables/qqq_pe_yfinance_snapshot_history.csv` | Every updater run |
| Signal history | Local strategy output | `reports/tables/preferred_strategy_current_signal_history.csv` | Every updater run |

The updater writes the latest signal snapshot to:

```text
reports/tables/preferred_strategy_current_signal.csv
```

QQQ P/E is now a Yahoo vendor-reported ETF `trailingPE` snapshot. It is saved on every updater run into a local snapshot history and also rolled up into a daily latest-per-date history.

## How to interpret the report

Use `executable_position_latest_bar`, not the raw signal, as the actionable no-lookahead state.

| Field | Meaning |
|---|---|
| `asof_intraday_bar` | Latest available completed hourly bar from the data vendor. |
| `base_raw_position_latest_bar` | Base QQQ MACD + QQQ 200MA state before profit lock or peak stop. |
| `stopped_raw_position_latest_bar` | Base raw state after applying the 40% synthetic-3x trade-peak stop. |
| `dynamic_trimmed_position_latest_bar` | Position after profit lock, peak stop, and q110 dynamic q100 trim, before the bear filter and execution shifting. |
| `raw_desired_position_latest_bar` | Final raw target weight after all raw overlays, including the best robustness bear filter, before execution shifting. |
| `executable_position_latest_bar` | No-lookahead executable position after shifting and max-one-trade-per-day logic; can be 0%, 50%, 75%, or 100%. |
| `position_interpretation` | Human-readable long/cash interpretation. |
| `action_if_following_strategy` | Suggested research action implied by the executable state. |
| `qqq_latest_close` | QQQ close on the latest hourly bar. |
| `actual_tqqq_latest_close` | Actual TQQQ close on the latest available TQQQ hourly bar. |
| `qqq_200d_hourly_ma` | Current QQQ 200-trading-day hourly MA reference. |
| `distance_to_exit_trigger_pct` | Distance from QQQ price to the hourly 200MA exit reference. |
| `macd_hist_latest` | Current QQQ MACD histogram. |
| `entry_flag_latest` | Whether entry conditions fired on the latest bar. |
| `exit_flag_latest` | Whether the exit condition fired on the latest bar. |
| `trade_peak_drawdown_raw_latest` | Current raw synthetic-3x drawdown from the open trade peak. |
| `peak_stop_trigger_raw_latest` | Whether the 40% peak stop fired on the latest raw bar. |
| `bear_filter_blocked_entry_raw_latest` | Whether the best robustness bear filter blocked a new raw entry on the latest bar. |
| `bear_filter_distance_to_200ma_latest` | QQQ's latest distance above/below its hourly 200MA. |
| `qqq_pe_yfinance_trailing_pe` | Latest locally recorded Yahoo Finance QQQ trailing P/E snapshot. |
| `qqq_pe_snapshot_date` | Date of the P/E snapshot. |
| `qqq_pe_update_status` | Whether the updater downloaded a new P/E snapshot or reused today's existing one. |

If `raw_desired_position_latest_bar` and `executable_position_latest_bar` differ, report both and emphasize the no-lookahead convention.

## Local cron schedule for every trading hour

Use a buffer after the hour to allow delayed data to appear. On macOS, run:

```bash
crontab -e
```

Add:

```cron
20 10-15 * * 1-5 cd /path/to/trend-following && mkdir -p logs && /bin/zsh -lc 'source ~/.venvs/myenv/bin/activate && python scripts/update_preferred_strategy_signal.py' >> /path/to/trend-following/logs/preferred_signal_update.log 2>&1
```

This runs Monday-Friday at 10:20, 11:20, 12:20, 13:20, 14:20, and 15:20 local machine time.

Manual checks:

```bash
tail -n 80 logs/preferred_signal_update.log
cat reports/tables/preferred_strategy_current_signal.csv
```

## Codex skill for a separate auto-update thread

A reusable Codex skill has been created at:

```text
$CODEX_HOME/skills/qqq-tqqq-hourly-updater
```

In a separate thread, ask Codex to use the `qqq-tqqq-hourly-updater` skill to update/check the current signal or set up a recurring trading-hour monitor.

## Operational cautions

- This updater intentionally does not use Alpha Vantage.
- Yahoo/yfinance data can be delayed, limited, or temporarily unavailable; always report the actual `asof_intraday_bar`.
- Yahoo 60-minute bars generally use 7 bars per full regular-session day, while earlier Alpha Vantage research used 6 bars/day, so signal values may differ from prior Alpha Vantage-based backtests.
- Yahoo intraday history is limited, usually around 730 days, so this updater is for current monitoring rather than full long-history backtests.
- Yahoo ETF P/E is a vendor-reported snapshot, not the prior transparent Alpha Vantage holdings-based harmonic P/E calculation.
- This strategy is research-only and does not model all live execution, tax, settlement, or broker constraints.
- The strategy can still experience large drawdowns even when the signal is long.
