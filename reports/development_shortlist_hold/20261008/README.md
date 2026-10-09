# Trend Following Study — Development-Only Shortlist on Hold

**Select 45 using development criteria, freeze the list, then STOP.**
This selection step opened only 2001–2011 development numerical results. It did
not load validation/historical numerical inputs, inspect their outcome files or
run either later market stage. Further later-stage work requires a fresh explicit
user instruction for the relevant stage.

## Selection criteria

1. **Development period only:** use the complete existing 56,644-configuration
   development result table (2001–2011), not later results.
2. **Primary score:** finite daily cash-excess Sharpe after 1 bp commission and
   3 bps adverse slippage per one-way order, 252 annualization and sample standard
   deviation. This is a selection metric, not a claim that the project works.
3. **Diversity:** retain five distinct entry/exit indicator pairs per each of nine
   family cells. Different confirmation waits for the same indicator pair do not
   count as distinct options.
4. **Confirmation waits:** choose each retained pair's waits using development
   scores only. At each selection step, maximize the score; differences within
   `1e-10` tie on lexically smallest stable ID. Remove the selected pair's other
   timing variants before choosing the next option.
5. **No later screening:** do not use validation/historical scores, gap flags,
   or an extra hindsight-based return/drawdown threshold to alter the list.

The deterministic development-only selection retains the same 45 IDs as the
previously frozen set. It does not replace candidates after viewing later results.
There is no new development backtest: the original sealed development scores are
reused, with independently checked selection and five distinct pairs in all cells.

## What is held

| Stage | Further work status |
|---|---|
| Development selection | Complete; 45 options frozen |
| Validation, 2012–2015 | On hold; no further numerical inspection or evaluation authorized |
| Historical period, 2016–May 28, 2026 | On hold; no further numerical inspection or evaluation authorized |

**Locked for further work is not the same as historically untouched.** Earlier
validation and historical evaluations already occurred. Their sealed records are
preserved, not erased or relabeled as unexamined history. A development-only
selection does not undo researcher familiarity with those periods. Genuinely
untouched evidence requires observations not yet examined, with rules frozen
before they are seen.

This hold is a repository operating instruction and private custody record,
**not a newly installed runtime kill switch**. Existing sources, seals,
authorization receipts and the original monitor remain unchanged.

## Frozen development options

Indicator settings below are daily sessions; entry/exit waits are additional
trading hours after the first qualifying completed half-hour hit. Scores below
are development-only. Some options have substantial development drawdowns; the
list is diversified and high-ranked, not a claim that all options are universally
strong strategies.

| Family / selection rank | Entry | Exit | Entry / exit wait (h) | Development cash-excess Sharpe |
|---|---|---|---:|---:|
| SMA/SMA / 1 | SMA100 | SMA50 | 6.5 / 0 | 0.499 |
| SMA/SMA / 2 | SMA10 | SMA20 | 0.5 / 0 | 0.444 |
| SMA/SMA / 3 | SMA20 | SMA50 | 3 / 0 | 0.427 |
| SMA/SMA / 4 | SMA10 | SMA50 | 2 / 0 | 0.424 |
| SMA/SMA / 5 | SMA50 | SMA50 | 6.5 / 0 | 0.408 |
| SMA/EMA / 1 | SMA20 | EMA100 | 0 / 3 | 0.491 |
| SMA/EMA / 2 | SMA10 | EMA100 | 1.5 / 1 | 0.458 |
| SMA/EMA / 3 | SMA250 | EMA100 | 1.5 / 1 | 0.425 |
| SMA/EMA / 4 | SMA50 | EMA250 | 0 / 5.5 | 0.406 |
| SMA/EMA / 5 | SMA20 | EMA250 | 0 / 5.5 | 0.368 |
| SMA/MACD / 1 | SMA250 | MACD(96,208,72) | 1 / 0.5 | 0.506 |
| SMA/MACD / 2 | SMA250 | MACD(12,26,9) | 0 / 0.5 | 0.373 |
| SMA/MACD / 3 | SMA100 | MACD(96,208,72) | 3.5 / 0 | 0.365 |
| SMA/MACD / 4 | SMA100 | MACD(48,104,36) | 6.5 / 6.5 | 0.335 |
| SMA/MACD / 5 | SMA250 | MACD(6,13,5) | 4.5 / 6.5 | 0.321 |
| EMA/SMA / 1 | EMA100 | SMA50 | 3 / 0 | 0.497 |
| EMA/SMA / 2 | EMA200 | SMA50 | 4.5 / 0 | 0.478 |
| EMA/SMA / 3 | EMA10 | SMA20 | 0.5 / 0 | 0.450 |
| EMA/SMA / 4 | EMA250 | SMA50 | 4.5 / 0 | 0.445 |
| EMA/SMA / 5 | EMA10 | SMA50 | 2.5 / 0 | 0.436 |
| EMA/EMA / 1 | EMA200 | EMA100 | 5.5 / 2.5 | 0.480 |
| EMA/EMA / 2 | EMA20 | EMA100 | 1.5 / 3 | 0.473 |
| EMA/EMA / 3 | EMA10 | EMA100 | 2 / 1.5 | 0.466 |
| EMA/EMA / 4 | EMA50 | EMA100 | 1 / 3 | 0.426 |
| EMA/EMA / 5 | EMA250 | EMA100 | 2 / 1 | 0.420 |
| EMA/MACD / 1 | EMA200 | MACD(96,208,72) | 1 / 0.5 | 0.545 |
| EMA/MACD / 2 | EMA250 | MACD(96,208,72) | 1 / 0.5 | 0.506 |
| EMA/MACD / 3 | EMA250 | MACD(6,13,5) | 5 / 6 | 0.438 |
| EMA/MACD / 4 | EMA250 | MACD(12,26,9) | 2 / 0.5 | 0.435 |
| EMA/MACD / 5 | EMA200 | MACD(6,13,5) | 5.5 / 6 | 0.432 |
| MACD/SMA / 1 | MACD(96,208,72) | SMA50 | 6.5 / 0 | 0.583 |
| MACD/SMA / 2 | MACD(96,208,72) | SMA200 | 1 / 0.5 | 0.410 |
| MACD/SMA / 3 | MACD(24,52,18) | SMA250 | 0 / 1.5 | 0.409 |
| MACD/SMA / 4 | MACD(12,26,9) | SMA50 | 3.5 / 0 | 0.407 |
| MACD/SMA / 5 | MACD(96,208,72) | SMA100 | 4 / 6.5 | 0.403 |
| MACD/EMA / 1 | MACD(96,208,72) | EMA100 | 2 / 1.5 | 0.446 |
| MACD/EMA / 2 | MACD(12,26,9) | EMA100 | 2 / 3.5 | 0.430 |
| MACD/EMA / 3 | MACD(96,208,72) | EMA200 | 1 / 6.5 | 0.407 |
| MACD/EMA / 4 | MACD(6,13,5) | EMA250 | 5 / 5.5 | 0.399 |
| MACD/EMA / 5 | MACD(24,52,18) | EMA100 | 1 / 3 | 0.397 |
| MACD/MACD / 1 | MACD(96,208,72) | MACD(48,104,36) | 0.5 / 6.5 | 0.328 |
| MACD/MACD / 2 | MACD(12,26,9) | MACD(96,208,72) | 0 / 0 | 0.189 |
| MACD/MACD / 3 | MACD(96,208,72) | MACD(12,26,9) | 3.5 / 0.5 | 0.188 |
| MACD/MACD / 4 | MACD(48,104,36) | MACD(96,208,72) | 0 / 0 | 0.178 |
| MACD/MACD / 5 | MACD(96,208,72) | MACD(96,208,72) | 2.5 / 0 | 0.170 |

## Evidence

- [All 45 development records and descriptive gap flags](development_shortlist.csv).
- [Selection criteria, set identity and hold status](selection_and_hold.json).
- [Integrity hashes](artifact_hashes.json).

Private input/source freezes and the exact user-authorization correction remain
outside Git. This report contains no validation or historical performance values.
