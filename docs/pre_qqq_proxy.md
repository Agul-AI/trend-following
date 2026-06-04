# Pre-QQQ Nasdaq-100 Proxy Policy

_Last updated: 2026-06-04_

This project's default requested start date is `1990-01-01`. QQQ did not exist for the full requested period, so any QQQ-like research before QQQ's inception must use a clearly labeled proxy.

## Confirmed assumption

For **1990-01-01 through the start of actual QQQ history**, the project may use **Nasdaq-100 index data** as a **pre-QQQ proxy**.

This proxy must be documented as:

```text
Pre-QQQ proxy: Nasdaq-100 index data used as a QQQ-like signal proxy before QQQ existed.
This is not actual QQQ ETF data.
```

## What the proxy is allowed to represent

The proxy is acceptable for:

- Long-term QQQ-like trend detection before actual QQQ data exists.
- Historical warmup for moving averages and MACD-style indicators.
- Sensitivity tests that ask how the current QQQ-based strategy may have behaved under a Nasdaq-100-like price path before QQQ launched.

## What the proxy must not imply

The proxy must **not** be presented as actual QQQ ETF performance. It does not include:

- QQQ ETF fees and expense drag.
- ETF creation/redemption effects.
- ETF trading spreads or intraday liquidity.
- Tracking error.
- QQQ distributions or adjusted-close mechanics.
- Any exact tradable QQQ execution before QQQ existed.

## Price vs total-return preference

Preferred source hierarchy:

1. Nasdaq-100 **total return** index, if available and licensed/accessible, especially for return simulation.
2. Nasdaq-100 **price** index, acceptable for signal generation and moving-average warmup.
3. Other Nasdaq-100-linked proxies only if explicitly documented and compared.

For the current hourly preferred strategy, the proxy problem is harder because long-history 60-minute Nasdaq-100 index data before QQQ may not be freely available. Do **not** fabricate hourly bars from daily data unless the output is explicitly labeled as a coarse approximation and excluded from primary performance claims.

## Recommended implementation convention

When implemented, create separate local series rather than overwriting actual QQQ:

- `QQQ_PROXY` or `QQQ_NDX_PROXY` for the stitched signal series.
- Keep actual `QQQ` raw data unchanged.
- Add columns/metadata such as `source_segment = ndx_proxy` or `source_segment = actual_qqq`.
- Save a proxy audit table showing segment start/end dates and source files.

## Reporting convention

Any backtest using this proxy should state both:

- **Requested start date:** `1990-01-01`.
- **Proxy/effective data convention:** Nasdaq-100 index data is used before actual QQQ data; actual QQQ is used after QQQ data begins.

If the proxy is daily while the strategy is hourly, report it as a daily/proxy sensitivity test, not the primary hourly preferred-strategy result.

## Alpha Vantage hourly availability check

A local Alpha Vantage probe was run for common Nasdaq-100 index symbols:

- `NDX`
- `^NDX`
- `NASDAQ:NDX`
- `INDEXNASDAQ:NDX`
- `NDX.X`

Result: Alpha Vantage did **not** return usable Nasdaq-100 index hourly bars for these symbols through `TIME_SERIES_INTRADAY`. Alpha Vantage symbol search also did not reveal a direct Nasdaq-100 index symbol suitable for historical intraday index data.

Probe outputs:

- `reports/tables/alpha_vantage_ndx_symbol_probe.csv`
- `reports/tables/alpha_vantage_ndx_symbol_search.csv`

Therefore, for a true 1990-1999 pre-QQQ hourly proxy, Alpha Vantage is currently not a usable source unless a different entitled index symbol is later identified. The next step would be to evaluate other sources for Nasdaq-100 index history, likely daily first and hourly only if a reliable licensed/free source exists.
