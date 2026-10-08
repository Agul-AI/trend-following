# OOS v2 pre-freeze source/accounting audit

**Scope:** Read-only review and independent calculations using the locally acquired Yahoo frames and licensed Alpha Vantage caches. Only this aggregate audit was written. No original data, historical results, production source, or Git state was changed. This is calibration/warmup QA, **not prospective performance evidence**.

## Result

**PASS for the reviewed input package and deterministic fallback.** The normalized panel covers October 7, 2024 through October 5, 2026: **3,485 expected and observed completed-bar timestamps, zero missing and zero extra**, for both QQQ and TQQQ.

- Seven incomplete Yahoo sessions are replaced **in full for both assets**, rather than splicing only absent observations. This replaces 34 endpoints per asset, including 14 missing endpoints per asset; 40 already-present asset/endpoints provide overlap comparisons.
- All 122 required regular-session 30-minute subbars are present across the replacement asset/sessions. All **60 out-of-session subbars** on early-close dates are excluded, not selected as session closes.
- Independently recomputing all **68 replacement asset/endpoints** from the exact final 30-minute subbar and same-provider daily restoration factor agrees with the adapter within **1.14e-13** currency units (floating-point rounding).
- Corporate-action output contains **eight QQQ distributions, eight TQQQ distributions, and one TQQQ split**. Fallback does not add a second copy of those actions.

## Price and timestamp conventions

The Alpha Vantage download code requests `adjusted=true`. The vendor's [official documentation](https://www.alphavantage.co/documentation/#intraday) states that this includes historical split and dividend adjustments. These closes must not be used directly as physical raw prices. The reviewed conversion is:

`raw_intraday = adjusted_intraday * daily_raw_close / daily_adjusted_close`.

This reverses the daily factor once. Yahoo's request-end split-adjusted closes are separately restored to historical physical share units using later effective split ratios. The AV fallback is **not multiplied by that Yahoo restoration factor again**. The physical ledger then applies splits to holdings and distributions to receivables/cash once; the signal index separately uses causal cumulative split normalization and does not reinvest dividends.

The cache's **start-labeled** convention is empirically verified: on seven affected sessions plus January 29 and February 3, 2026, all **117 AV 30-minute closes per asset exactly equal the corresponding AV 15-minute close at start + 15 minutes**. Selecting the AV30 close at desired endpoint minus 30 minutes aligns with Yahoo's observed endpoint quotes; treating the AV stamp as an end label does not. This is evidence for **these cache files**, not a claim that the vendor's current documentation explicitly guarantees start labels for all historical products.

The old naive Yahoo cache contains timezone-stripped UTC labels on the investigated January 30/February 2 rows. These are duplicates of observed New York-time rows, not additional quotes that can fill the absent intervals. No arbitrary timestamp relocation was used as a fallback.

## Observed cross-vendor residuals

Maximum absolute differences in basis points between AV restored closes and overlapping, physically normalized Yahoo closes:

| Session | QQQ | TQQQ |
|---|---:|---:|
| 2024-11-29 | 0.00094 | 0.00871 |
| 2024-12-24 | 0.00077 | 1.34976 |
| 2025-07-03 | 0.00071 | 0.01078 |
| 2025-11-28 | 0.16304 | 0.00400 |
| 2025-12-24 | 0.00062 | 0.00799 |
| 2026-01-30 | 2.88080 | 18.00105 |
| 2026-02-02 | 0.31995 | 0.89626 |

The January 30 second Yahoo interval has OHLC/volume inconsistent with the complete corresponding AV interval and appears truncated. Whole-session replacement avoids retaining that questionable interval. Neighboring complete sessions also have genuine provider differences: the maximum on January 29 is 0.16109 bp for QQQ and 5.34279 bp for TQQQ; on February 3 it is 0.00067 bp and 2.01142 bp. Consequently, a universal 0.2 bp agreement claim would be false. Quotes were **not averaged, fitted, interpolated, or replaced with daily closes** to eliminate residuals.

## Independent economic reconciliation

For each asset, an independent zero-cost, zero-cash-yield buy-and-hold calculation used physical shares, raw closes, split quantities, ex-date receivables, and pay-date cash. Cash distributions were not reinvested. Its equity, cash and outstanding receivables match `simulate_oos_ledger` at **every one of 3,485 timestamps, with zero numerical difference**. The terminal liquidation clone preserves wealth under the zero-cost, zero-tax scenario and leaves the continuous account separate.

Additional artificial checks passed: a missing endpoint in only one asset triggers full-session replacement of both assets; future split units are restored exactly once; after-close provider rows cannot influence replacement closes; a missing intermediate half-hour subbar fails even if the requested endpoint quotes exist.

## Limitations and reviewed version

- Restoration relies on compatible intraday/daily adjustment vintages; local overlap checks support these files, not every possible future acquisition.
- Historical adjusted caches and later retrieval do not establish point-in-time vendor availability. These inputs are retrospective **pre-freeze** calibration/warmup material. Action availability remains the explicitly disclosed public-by-ex-date assumption, not measured contemporaneous availability.
- Distributions with unknown tax character remain provisional hypothetical scenarios, not verified investor tax treatment. The no-tax reconciliation does not certify tax filing compliance.
- The full real-input exercise reviewed `research_oos_data.py` SHA-256 `598ff3e53523ac9d5353f9161b859999d57a205a978aa462cbf7fcccf0abce6c`. Subsequent root-owned changes require their own tests; this hash is an audit checkpoint, not the final study freeze identifier.
- This audit publishes only aggregate counts, residuals and methodology; no licensed raw quotes, private absolute paths or source credentials are included.
