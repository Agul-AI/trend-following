# Preferred Strategy Rules

_Last updated: 2026-06-04_

This file records the **confirmed preferred strategy rules so far**. Experimental ideas are not promoted into the confirmed rule set unless explicitly confirmed.

This is for research/backtesting only, not live trading or financial advice.

## Current preferred strategy

**Name:** QQQ hourly-200MA-gated synthetic-TQQQ trend strategy with 12/24/9 SMA-MACD entry, profit lock, q110 dynamic q100 trim, best robustness bear filter, and 40% peak stop  
**Short label:** `preferred_qqq_hourly_200ma_macd_slow24_profit_lock_300_400_stop40_q110_best_bear_filter`

## Summary bullets

- QQQ entry + QQQ exit
- Synthetic TQQQ exposure
- Profit lock: +300% unrealized trade gain -> 75%; +400% -> 50%
- Dynamic q100 mean-reversion trim: after +110% gain, learn the max QQQ distance above its 200MA up to that point; later trim to 50% if QQQ revisits/exceeds that learned distance; re-add on a QQQ 20MA pullback
- Best robustness bear re-entry filter: if QQQ's hourly 200MA slope is negative, delay new entries until QQQ is at least 1% above the 200MA, the QQQ 50MA slope over 30 trading days is positive, and QQQ 20MA > QQQ 50MA
- Trade-peak stop: exit if synthetic TQQQ falls 40% from its current trade peak
- Hourly SMA-MACD histogram > 0 entry, using 12/24/9 as the preferred MACD option
- Retain 12/26/9 and 12/26/8 as near-equivalent MACD robustness options, because the three MACD choices are very similar and may be overfit
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


### Dynamic q100 mean-reversion trim after +110% trade gain

The preferred strategy now includes a dynamic extension/mean-reversion trim overlay.

Current rule:

```text
For each open synthetic-TQQQ trade:
    wait until unrealized synthetic-TQQQ trade gain first reaches +110%
    using only bars from entry through that first +110% bar:
        learn q100 = max(QQQ distance above its hourly 200-day MA)
    after the +110% point:
        if QQQ revisits/exceeds that learned q100 distance:
            cap exposure at 50%
        if QQQ then pulls back to/touches its hourly 20-day MA:
            restore full exposure subject to the other overlays
```

In formula form, the learned distance is:

```text
q100_threshold = max(QQQ close / QQQ hourly 200-day MA - 1)
                 from trade entry through the first +110% synthetic-3x gain bar
```

Reason:

- A fixed 20%/22% distance threshold was too arbitrary and potentially overfit.
- The q100 rule learns the extension threshold separately for each trade using only information available before or at the first activation gain.
- The +110% activation was the best recent sweep from 2010 onward and modestly improved full-sample return versus +100%, while reducing q100 trim/re-entry churn.
- In the latest start-date CV official-start comparison, the 12/24/9 MACD option produced 32.62% annualized return, 0.90 Sharpe, and -52.79% max drawdown from 2002-01-10, while the prior 12/26/9 baseline produced 31.95% annualized return, 0.89 Sharpe, and -52.15% max drawdown. Treat this as a small robustness preference rather than a strong optimized edge.
- It triggered q100 trims 15 times in the full long-history preferred test.

Important limitation:

- This overlay by itself does **not** solve the single worst max drawdown.
- It only activates after a trade has already reached +110%, so it cannot protect against bear-market whipsaw drawdowns that occur before a large winning trade develops.

No-lookahead convention:

- The learned q100 threshold is computed only from bars available from trade entry through the first +110% bar.
- A trim or re-entry decision is raw close-bar information and is still shifted into an executable position before returns are applied.

### Best robustness bear re-entry filter

The preferred strategy now includes the best robustness variant of the bear re-entry filter.

Current rule:

```text
If QQQ hourly 200-day MA slope is negative:
    allow a new entry only if all are true:
        QQQ close is at least 1% above its hourly 200-day MA
        QQQ hourly 50MA slope over 30 trading days is positive
        QQQ hourly 20MA > QQQ hourly 50MA
```

Reason:

- Bear-market recoveries can create repeated MACD/200MA whipsaws.
- The filter only applies when the long-term QQQ 200MA slope is negative, so it does not slow ordinary entries in positive long-trend environments.
- The 30-day 50MA-slope version slightly improved annualized return and Sharpe versus the earlier 20-day serious candidate, with the same full-sample max drawdown.
- This rule reduced the 2007-2009 drawdown from about -56.36% in the q100-only preferred baseline to about -51.12% in the preferred q110 + best robustness variant.

No-lookahead convention:

- The filter uses only the current completed QQQ bar and moving-average values known at that bar.
- Block/release decisions are raw close-bar decisions and are shifted into executable positions before returns are applied.

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

Current preferred MACD settings:

```text
MACD source: QQQ hourly price
MACD type: SMA-based MACD
Fast window: 12 trading days
Slow window: 24 trading days
Signal window: 9 trading days
Bar frequency: hourly / 60-minute bars
Bars per day assumption: 6 for Alpha Vantage long-history research; Yahoo updater uses 7 by default
Entry confirmation: 2 consecutive hourly bars
Entry size: 100%
```

Retained MACD robustness options:

| Option | MACD windows | Status | Reason to keep |
|---|---:|---|---|
| Preferred | 12 / 24 / 9 | Active default | Highest official-start return and slightly fewer trades in the latest comparison |
| Standard | 12 / 26 / 9 | Retained | Original baseline and easiest to explain |
| Faster signal | 12 / 26 / 8 | Retained | Similar return improvement without worsening worst max DD in the start-date robustness check |

These MACD choices are intentionally treated as a **robustness set**, not as a precise optimized parameter. Their behavior is very similar, and the differences may be overfit. The active default is 12/24/9, but reports should continue showing the other two where useful.

Interpretation:

- MACD histogram > 0 means the MACD line is above its signal line.
- This is a short/intermediate momentum re-acceleration signal.
- The 200-day hourly MA condition prevents MACD entries while QQQ is below the long-term hourly trend reference.

Confirmed choices:

- Use `histogram > 0`.
- Do not require histogram to be rising.
- Do not use gradual entry.
- Enter at 100% target exposure when the confirmed entry condition is met.

MACD parameter decision as of 2026-06-04:

- Promote `macd_slow_24d` / 12-24-9 as the active preferred default.
- Keep 12-26-9 and 12-26-8 in the documented comparison set.
- Do **not** interpret the 12-24-9 choice as a durable optimized optimum; the top MACD variants are close enough that this may be parameter noise.

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

From the latest q100-activation and bear-filter sweep, row `robust_slope30_q110`:

```text
Candidate: Preferred q110 + best robustness bear filter
Final return:       88,809.7%
Annualized return:  29.41%
Sharpe ratio:       0.857
Max drawdown:      -52.15%
Number of trades:  128
Trades/year:        4.86
Exposure:           63.09%
DD episodes >20/>30/>40/>50%: 23/8/5/3
q100 trim triggers: 15
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
- q100 +100% activation; replaced by q100 +110% activation.
- Earlier serious bear filter with 20-day 50MA-slope confirmation; replaced by the best robustness variant with 30-day confirmation.

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
