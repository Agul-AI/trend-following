# Alpha Vantage quote reconciliation and coverage policy

This correction separates genuine source errors from incompatible quote roles.
It retains the 56,644 daily-indicator/half-hour-confirmation configurations,
Alpha-only prices, May 28, 2026 endpoint, costs and chronological splits. Earlier
code and registrations remain preserved; this is a new quality-aware study.

## What was corrected

The earlier gate treated the provider daily close as identical to the last
16:00 intraday trade. That identity is not valid by definition. QQQ previously
traded to 16:15 on AMEX, and its listing transferred to Nasdaq on December 1,
2004. No earlier quote is overwritten to force it to match a different quote.
[Historical broker notice](https://www.fidelity.com/products/stocksbonds/content/qqq.shtml),
[Nasdaq announcement](https://ir.nasdaq.com/news-releases/news-release-details/qqq-list-nasdaq).

The new profile uses distinct, explicit roles:

| Role | Quote and clock |
|---|---|
| Intraday provisional daily indicator | Observed Alpha 30-minute closing trade; previous finalized daily state only |
| Entry/exit fill | Verified tradable next half-hour opening quote, plus slippage/commission |
| Finalized daily feature history | Alpha provider daily raw close, used only on later sessions |
| Daily valuation | Alpha provider daily raw close at an assumed 17:00 New York availability clock |
| Terminal liquidation | Safe observed final-session RTH close; never a missing interval or daily valuation substitute |

The 17:00 clock is a conservative research availability assumption, not a
reconstructed feed timestamp or a claim that the provider's field is Nasdaq's
official closing auction price. All portfolios and benchmarks share this
valuation convention. Terminal cash accrues to the same valuation endpoint.

## Reconstructed physical price units

The source caches remain unchanged. Historical intraday OHLC is restored with
the same Alpha daily raw-close/adjusted-close factor. It is labeled
**derived as-traded units, not native raw capture**. The audit binds the original
daily JSON, daily Parquet and intraday Parquet hashes and verifies:

- All 6,685 native daily observations agree across eight provider fields.
- Adjustment-factor changes occur only on recorded dividends/splits.
- All 317 native months have at least ten independent opening/high/low price
  anchor days at a precision-derived bound; the daily close is not an anchor.
- All 86,170 comparable 15-to-30-minute OHLC windows agree.

The 558 required-window last-intraday/daily close differences remain recorded;
their removal from an inappropriate equality gate is not a tolerance increase.
Structural OHLC, source-unit/vintage, action, calendar, time and provenance
failures still stop preparation. Vendor vintages are retrospective, not certified
point-in-time history. No raw quote values are published in the reports.

## Known trading halt

Nasdaq's August 22, 2013 regulatory halt ended at 15:25 after all listed
securities were halted by 12:23. Vendor rows can exist during a halt without
being executable trades. [Nasdaq notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2013-77).

The frozen coverage table therefore distinguishes observed, unknown-missing
and halt-overlapping intervals. It contains no OHLC values for absent slots.
The 12:00 open precedes the halt and may fill an already-confirmed order, but
its overlapping interval cannot confirm a new signal. The 15:00-labelled
opening quote cannot be executed at 15:00: trading did not resume until 15:25.
The first fully tradable canonical post-halt interval begins at 15:30.
Blackout starts cancel unfilled orders and reset both confirmation streaks.

## Actual missing data and consent

Two genuine source holes remain: the March 29, 2001 opening half-hour and all
13 half-hours on November 30, 2004. Alpha's other cached resolutions/backups do
not recover them. A fresh same-provider request was denied as premium-only.
Daily OHLC cannot determine those missing intraday paths.

The default **strict tradable** profile permits documented market halts but
still blocks these 14 unknown intervals. A separate **observed-only blackout**
profile may be enabled only by explicit recorded human consent, before any
performance runs. It does not repair the data:

- Keep existing QQQ/cash holdings through the source blackout.
- Make no signal decisions or fills in missing intervals.
- Cancel pending orders, reset streaks and require fresh confirmation afterward.
- Apply calendar-opening splits/dividend entitlements even on a wholly missing
  day; keep distribution cash until a safe observed DRIP price exists.
- Preserve every daily date and value holdings using actual provider daily quotes.
- Report the source gaps and changed operating assumption alongside results.

No fallback is enabled merely because the complete-data profile fails. The
authorization receipt is read and hash-bound; the CLI cannot manufacture it.
No funds, subscriptions or broker orders are created by this correction.

## Running the quality-aware entry point

Use the requested environment and the same publication checkout:

```bash
source ~/.venvs/myenv/bin/activate
python scripts/run_qqq_reconciled_v5.py --help
python scripts/run_qqq_reconciled_v5.py prepare
```

The classified Alpha packet is stored privately with immutable hashes, observed
prices, coverage, blackouts, unit proofs and timestamped cash observations.
Strict preparation reports the specific unresolved gaps, rather than the old
blanket raw-price/closing-equality error. No market score is calculated during
preparation. Parameter selection still requires development, sealed finalists,
validation and a sealed winner before historical OOS is opened.

CLI exit code `3` means `data_not_ready`, not a software crash. The default
strict registration cannot later be changed in place to enable the alternative
policy. An explicitly authorized policy needs a fresh configuration freeze and
private study directory; the earlier failed preparation remains recorded.
