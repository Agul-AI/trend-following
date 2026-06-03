# Preferred Strategy Rules

_Last updated: 2026-06-03_

This file records the **confirmed preferred strategy rules so far**. Experimental ideas are not promoted into the confirmed rule set unless explicitly confirmed.

This is for research/backtesting only, not live trading or financial advice.

## Current preferred strategy

**Name:** QQQ hourly-200MA-gated synthetic-TQQQ trend strategy with profit lock and 40% peak stop  
**Short label:** `preferred_qqq_hourly_200ma_macd_profit_lock_300_400_stop40`

## Summary bullets

- QQQ entry + QQQ exit
- Synthetic TQQQ exposure
- Profit lock: +300% unrealized trade gain -> 75%; +400% -> 50%
- Trade-peak stop: exit if synthetic TQQQ falls 40% from its current trade peak
- Hourly MACD histogram > 0 entry
- QQQ hourly 200-day MA entry gate
- QQQ hourly 200-day MA exit
- No daily regime gate
- Max one trade per day
- Out-of-market cash earns 3% annualized in evaluation

## Detailed explanation of each summary bullet

### QQQ entry + QQQ exit

The strategy uses **QQQ** as the signal source for both buying and selling.

This means:

- Entry indicators are computed from QQQ, not TQQQ.
- Exit indicators are computed from QQQ, not TQQQ.
- TQQQ-like exposure is treated as the traded risk asset, but QQQ is the cleaner reference asset for trend detection.

Reason:

- QQQ is the unlevered underlying ETF.
- QQQ is less path-dependent and less noisy than TQQQ.
- Prior tests showed that QQQ-based signals can stay competitive while being easier to monitor manually.

### Synthetic TQQQ exposure

The strategy backtests exposure using `QQQ_3X_CALC`, a synthetic +3x version of QQQ.

This means:

- The backtest does not rely on actual TQQQ history as the main research asset.
- Synthetic TQQQ is calculated from QQQ returns under the simplified assumption of perfect +3x tracking.
- The goal is to study the behavior of leveraged QQQ exposure using a longer and cleaner underlying history.

Important limitation:

- Synthetic TQQQ is not the same as real TQQQ.
- Real TQQQ can differ because of fees, financing, daily rebalancing, spreads, tracking error, and market microstructure.

### Profit lock: +300% -> 75%, +400% -> 50%

The preferred strategy now includes a simple intra-trade profit-lock overlay on synthetic TQQQ exposure.

Current rule:

```text
If unrealized synthetic-TQQQ trade gain >= +300%:
    reduce exposure to 75%
If unrealized synthetic-TQQQ trade gain >= +400%:
    reduce exposure to 50%
```

This is evaluated within each base trade and resets on the next new entry.

Reason:

- The `+300% -> 75%, +400% -> 50%` overlay improved return and Sharpe versus the no-lock base in the latest test.
- It reduced the number of large drawdown episodes, although it did **not** reduce the single worst max drawdown.
- It triggered only a few times historically, so it preserves relatively low manual trading frequency.

No-lookahead convention:

- The profit-lock condition is detected at the close of the bar where the unrealized gain threshold is observed.
- The executable exposure reduction happens after the signal shift; it cannot earn the same bar's return.

### Trade-peak stop: synthetic TQQQ -40% from current trade peak

The preferred strategy now includes a trade-level peak-drawdown stop on the synthetic TQQQ exposure.

Current rule:

```text
Within each open base trade:
    track the highest synthetic-TQQQ price since entry
    if synthetic TQQQ falls 40% or more from that trade peak:
        exit to cash
        stay out until the base QQQ 200MA/MACD state resets
```

Reason:

- The stop improved the latest preferred-strategy test versus the same profit-lock strategy without the stop.
- It reduced the number of >50% strategy drawdown episodes from 4 to 3 in the historical test.
- It triggered only once historically, so it does not materially increase trading frequency.

Important interpretation:

- This is a **trade-level synthetic-TQQQ peak stop**, not an account-level drawdown stop.
- It does **not** guarantee the portfolio max drawdown will stay above -40%.
- The stop is observed at a completed bar and then shifted through the executable-position conversion, preserving the no-lookahead convention.

### Hourly MACD histogram > 0 entry

The entry trigger uses the QQQ hourly MACD histogram.

Current rule:

```text
Enter only when:
QQQ hourly MACD histogram > 0
and
QQQ hourly close > QQQ hourly 200-day MA
```

Current MACD settings:

```text
MACD source: QQQ hourly price
MACD type: SMA-based MACD
Fast window: 12 trading days
Slow window: 26 trading days
Signal window: 9 trading days
Bar frequency: hourly / 60-minute bars
Bars per day assumption: 6
Entry confirmation: 2 consecutive hourly bars
Entry size: 100%
```

Interpretation:

- MACD histogram > 0 means the MACD line is above its signal line.
- This is a short/intermediate momentum re-acceleration signal.
- The 200-day hourly MA condition prevents MACD entries while QQQ is below the long-term hourly trend reference.

Confirmed choices:

- Use `histogram > 0`.
- Do not require histogram to be rising.
- Do not use gradual entry.
- Enter at 100% target exposure when the confirmed entry condition is met.

### QQQ hourly 200-day MA entry gate

The preferred strategy no longer uses the separate **daily QQQ 200-day trend regime gate**.

Instead, entry is gated directly by the QQQ hourly 200-day moving average:

```text
QQQ hourly close > QQQ 200-day hourly MA
```

Hourly conversion:

```text
200 trading days * 6 hourly bars/day = 1200 hourly bars
```

Reason:

- The serious candidate test showed this simplified gate improved full-sample annualized return and Sharpe versus the prior daily-gate preferred rule.
- It also uses the same long-term trend reference for both entry permission and exit, making the logic more internally consistent.

### QQQ hourly 200-day MA exit

If long, exit to cash when QQQ loses the same long-term hourly trend reference.

Current exit rule:

```text
Exit when QQQ hourly close < QQQ hourly 200-day MA
```

Current exit settings:

```text
Exit source: QQQ hourly price
Exit MA reference: 200 trading days = 1200 hourly bars
Exit confirmation: 3 consecutive hourly bars
```

Interpretation:

- The strategy exits when QQQ falls below the long-term hourly trend reference.
- Since QQQ is the underlying asset, this avoids letting TQQQ's leveraged path noise drive exits.

### No daily regime gate

The prior preferred version required a daily QQQ 200-day trend regime gate:

```text
QQQ daily close > QQQ daily 200-day SMA
and
QQQ daily 200-day SMA slope > 0
```

That rule is **not part of the current preferred strategy**.

Current preferred logic is simpler:

```text
Hourly entry permission: QQQ hourly close > QQQ hourly 200-day MA
Hourly exit trigger:     QQQ hourly close < QQQ hourly 200-day MA
```

Reason:

- Applying a regime gate consistently means it should affect both entries and exits.
- The hourly 200MA gate is now used consistently for both entry permission and exit.

### Max one trade per day

The executable strategy is constrained to at most one position change per calendar day.

This means:

- One buy per day maximum.
- One sell per day maximum.
- Never buy and sell on the same day.
- If multiple intraday signals occur on the same day, only the first accepted position change is used.

Reason:

- This matches the preference for fewer trades and manual monitoring.
- It avoids unrealistic intraday flip-flopping.
- It reduces the chance that the backtest benefits from overly reactive intraday behavior.

### Out-of-market cash earns 3% annualized in evaluation

Evaluation now assumes that cash not invested in synthetic TQQQ earns a 3% annualized risk-free/cash-management return.

This means:

- When the strategy is out of the market, cash earns a small positive return.
- The current backtest approximation taxes cash interest at the same 24% short-term tax rate unless otherwise stated.
- This reflects the intended practical use of Webull Cash Management, a Treasury/government money market fund, or a short-term Treasury ETF as a cash substitute.

## No-lookahead timing convention

Raw signals are timestamped when the information becomes available at the close of a bar.

The executable-position conversion shifts raw signals forward so the strategy cannot earn returns from the same bar that generated the signal.

For QQQ PE snapshots, if used later:

```text
A snapshot saved on date D is usable starting on the next business day, not on D.
```

## Benchmark reporting convention

All future strategy comparison tables should include a **QQQ buy-and-hold (QQQ BH)** benchmark over the same date range and data frequency. For synthetic/actual TQQQ strategy work, QQQ BH is the primary unlevered reference row; TQQQ buy-and-hold can be included only when explicitly useful, but QQQ BH should not be omitted.

## Current evaluation assumptions

Recent comparisons use:

```text
Transaction cost: 1 bp
Slippage: 5 bps
Short-term tax approximation: 24%
Out-of-market cash return: 3% annualized
Annualization: 1512 hourly bars/year
```

## Reference performance

From `reports/tables/preferred_profit_lock_stop_exit_comparison_compact.csv`, row `profit_lock_300_400_stop_40pct`:

```text
Candidate: Preferred hourly 200MA gate +300/+400 profit lock + 40% trade-peak stop
Final return:       43,107.1%
Annualized return:  25.92%
Sharpe ratio:       0.776
Max drawdown:      -56.36%
Number of trades:  115
Exposure:           65.37%
Average cash:       34.63%
DD episodes >20/>30/>40/>50%: 24/13/7/3
40% stop triggers:  1
```

## Confirmed dropped or replaced rules

These were tested but are **not** part of the current preferred strategy:

- Actual TQQQ buy-and-hold benchmark as a core comparison target.
- VIX percentile exit overlay.
- Gradual entry sizing.
- 10MA / 20MA entry variants.
- Option A QQQ PE proxy based on a stale fact-sheet anchor.
- Separate daily QQQ 200-day trend regime gate; replaced by the QQQ hourly 200MA entry/exit gate.
- No-stop version of the +300/+400 profit-lock strategy; replaced by the 40% trade-peak stop version.

## Candidate ideas not yet confirmed

These are research candidates only, not part of the preferred strategy unless explicitly confirmed:

1. **Fed hiking-cycle + high QQQ PE switch**
   - If the Fed announces a hiking cycle and QQQ PE > 30, switch from synthetic TQQQ exposure to QQQ until the cycle ends.
   - Promising in one test, but not yet confirmed as a rule.

2. **Option B QQQ PE data input**
   - Current/live QQQ PE from Alpha Vantage QQQ holdings + constituent P/E values.
   - Saved as a point-in-time daily snapshot history.
   - Data input only; not yet a confirmed trading rule.

## Update policy

When a new rule is proposed:

1. Test it separately.
2. Record results in `reports/project_log.md`.
3. Promote it into this file only after explicit confirmation, e.g. “confirm this rule”, “make this preferred”, or “add this to the preferred strategy.”
