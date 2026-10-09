# Trend Following Study — Diversified Shortlist Follow-up

This is an **exploratory, retrospective follow-up** using the same 56,644
configurations, Alpha-only data, periods, costs, accounting and daily-indicator /
half-hour-confirmation logic. The earlier validation and historical results have
already been examined. A new seal cannot make that history untouched again.
The original single-winner study and its frozen artifacts remain unchanged.

## Development-only shortlist

The default is **up to five distinct indicator pairs in each of nine entry/exit
family cells**, or up to **45 configurations**. The number is a follow-up design
choice, not an optimum inferred from validation or historical performance.

Within each family cell:

1. Consider all development configurations with a finite baseline daily
   cash-excess Sharpe. Do not filter by gap flags, validation or historical results.
2. Take the maximum Sharpe; values within `1e-10` tie on lexically smallest stable
   candidate ID, exactly as in the original selection rule.
3. Retain that configuration, including its development-selected confirmation
   waits. Remove other configurations with the same entry/exit indicator pair
   from consideration in this cell. Different waits alone do not create a new pair.
4. Repeat until five distinct pairs are retained, or finite pairs are exhausted.

Indicator periods remain daily sessions; confirmation waits remain trading hours
following the first qualifying completed half-hour close. No new grid search,
indicator, cost model, tax charge or financing assumption is introduced.

## Keep the set, not a replacement winner

- **Validation, 2012–2015:** evaluate every sealed option at 1 bp commission plus
  3 bps adverse slippage. Scores are descriptive; no option is dropped or replaced.
- **Historical comparison, 2016–May 28, 2026:** evaluate the exact same set with
  1 bp commission and slippage of 1, 3, 10 and 25 bps. Do not select a different
  configuration for each cost or call the best historical row a validated winner.
- Compare each option separately with cost-matched QQQ buy-and-hold and cash,
  using equal starting capital and the same terminal-liquidation conventions.
  Cross-option medians/ranges are descriptive distributions, **not returns of an
  equally weighted ensemble portfolio**.
- Report the complete shortlist, all subsequent results, poor performers and
  undefined later scores. Undefined development scores cannot enter selection;
  undefined later scores do not permit replacement or silent attrition.

This broader set provides information about variation across indicator pairs,
not a new independent confirmation of profitability. Any future prospective
study needs rules frozen before new observations become available.

## Separate custody and public commands

A private sidecar registration binds the original configuration, source/data
freeze, full development metric file, chosen count, selection code, candidate
set and direct user authorization. It must be sealed before the new validation
loads; historical evaluation requires the completed validation seal. Source
hashes, candidate identity and full result universes verify before each stage.

```bash
python scripts/run_qqq_research.py shortlist --help
```

The public command routes the follow-up separately. It does not change the
original monitor's winner or place orders. Private prices, consent receipts,
source/data freezes and full runtime packets are never committed.
