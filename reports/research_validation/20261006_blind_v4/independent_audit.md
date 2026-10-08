# Independent v4 post-evaluation audit

**Arithmetic verified; custody metadata qualified.** Of 717 checks, 716 passed. The only failed check is exit-timestamp monotonicity—not an economic reconciliation failure.

## Coverage and numerical agreement

The independent checker reconstructed **16 pretax case/layer paths × 2,617 sessions = 41,872 daily states**, covering the selected method, physical QQQ buy-and-hold, synthetic 3x buy-and-hold and cash across gross, trading, financing and expense layers. Cash, equity, quantities, marks, daily and continuous returns, turnover, terminal liquidation and metrics reconcile. The frozen 12-candidate validation objective selects the recorded winner independently.

Maximum absolute daily discrepancies: equity **3.54e-09**, cash **1.31e-10**, quantity **3.04e-11**, mark **2.52e-11**, return **9.82e-16**, normalized turnover **2.33e-15**. These are below the declared comparison tolerances.

| Expense-inclusive pretax result | Selected | QQQ buy-and-hold |
|---|---:|---:|
| CAGR | 16.2184% | 21.1226% |
| Maximum drawdown magnitude | 40.4258% | 35.1227% |
| Cash-excess Sharpe | 0.584725 | 0.871930 |

Selected-minus-QQQ cash-excess Sharpe is **-0.287204**. Primary benchmark superiority was **not demonstrated**.

## Custody evidence and preserved defect

All **30 audit-event hashes/chain links**, the canonical 12-candidate grid, policy/proposal and selection seals, **162 result-file hashes**, and **four released test-input hashes** verify. Hash-chain phase order and the independent selection receipt verify **selection before test**.

Both held-stage exit records reused their stage-start timestamp, causing two small timestamp regressions. Those values **are not completion times**. The defect remains in the frozen records; neither every timestamp nor every custody-quality check passed.

## Scope and limits

- The checker used independent NumPy/pandas arithmetic, not a project engine, ledger, rate helper or policy execution. Targets are nominal fractions of **pre-cost** marked NAV; transaction costs can lift partial actual weights above nominal caps. Cash uses a zero-quantity QQQ mark placeholder.
- Tax event totals and holdings/account/return/terminal/metric identities verify using the matched realized cash-tax benchmark. **Full tax-lot, wash-sale, holding-period, income/filing and investor-specific legal tax compliance are not certified.** Tax rates are hypothetical research assumptions.
- Rates are latest-vintage with assumed availability, not historical release-time or ALFRED-vintage reconstruction. The leveraged instrument is synthetic; fixed funding, transaction and drag assumptions are not live execution evidence.
- Internal custody hashes establish consistency, not externally anchored signatures or erased human knowledge. Kernel restrictions cover launched child processes, not all unsandboxed same-user agent tools. Engine/kernel hash declarations were cross-checked without reading or importing that source.
- Audit v1-v3 snapshots retain technical contract clarifications. **No frozen strategy, costs, input or primary output was changed.** Historical OOS reconciliation does not establish live returns, income assurance or a profitable edge.
