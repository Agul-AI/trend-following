# Fresh split v3 — independent artificial economic audit

**PASS; no material accounting issue found.** Reviewed `research_split_v3.py` SHA-256 `3a64e35f8d46afdc1bf1d4869db335b66750ca01c2ced23abadfc70ce5a036a1` before selection/test execution. Read-only code review and separately written arithmetic; no historical market returns or strategy results were used.

## Independent checks

An artificial eight-session fixture spans a weekend and year boundary, includes two splits and multiple distributions, positive DFF/cash rates, and deliberately large 1% transaction fees. Independent direct share/cash arithmetic—not the implementation's signal, synthetic or ledger helpers—was compared against all intermediate equities, cash balances and quantities for:

- Physical QQQ with full allocation and same-close distribution reinvestment.
- A synthetic daily-reset 3× instrument with a fixed one-third allocation, allowed to drift.
- Each of gross, trading, financing and expense layers.

All eight paths reconcile, with maximum absolute equity error **2.28e-13**. The fixed-fraction synthetic account has only one initial purchase; the physical control has four purchases because it reinvests dividends. Investor portfolio rebalancing is therefore distinct from the synthetic instrument's internal daily reset.

Additional independent checks passed:

1. Gross physical QQQ wealth equals hand-computed reinvested total return beginning only after its delayed second-session-close entry. A distribution on that purchase close does not create entitlement for the new buyer.
2. Splits change quantities once, distributions are credited once, and same-close dividend cash earns none of the preceding cash-accrual interval.
3. Borrowing is charged exactly twice the stated DFF-plus-spread elapsed-calendar accrual; expense/extra drag is charged once. Physical QQQ does not receive synthetic financing or expense deductions.
4. Costed terminal liquidation matches independent arithmetic. Continuous ledger state remains unchanged by the terminal clone and boundary-factor extraction.
5. Removing first-entry and terminal multipliers, then applying them once to an identity return sequence, reproduces terminal wealth for every path.
6. Two additional hand-calculated hypothetical tax examples reconcile ordinary distribution income, FIFO short-term realization, terminal settlement, and a January funding sale for the preceding year's tax. Both final-equity differences are zero.

## Contract clarification

Root explicitly approved **daily reinvestment for physical QQQ controls only**: these have full 0/1 targets and may reinvest ex-date cash at the same accounting close, subject to costs. Synthetic fractional positions remain target-change-only and are not silently reset to target weight every day. This supersedes any earlier general shorthand suggesting all investor paths use transition-only rebalancing.

These tests validate the declared simplified economics. They do not establish actual TQQQ returns, verified historical dividend payment timing, investor tax compliance, executable live closes, or strategy profitability. Real historical test results remain unopened for this audit.
