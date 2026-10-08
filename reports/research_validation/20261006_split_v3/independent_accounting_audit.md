# Fresh split v3 — independent final accounting audit

**PASS: 210 numerical checks.** Eight pretax account paths were independently reconstructed across all **2,617 test sessions**, January 4, 2016–June 1, 2026. The independent script imports no project evaluator, generic ledger, signal builder or rate-integration helper. Original sources, frozen code and result files were not modified.

## Integrity and chronology

- Protocol SHA-256: `9e8aad4c77c87d4f55d2d79a96f00bd2ac3a2928656ab1e8c0d9869dd985e671`.
- Selection SHA-256: `562c815ec9c06295dfc4f6c92e232e967d79f95d71316527c82e14fe7845f1de`.
- Protocol sealed at **16:45:00 UTC**, selection began **16:45:02 UTC**, and test evaluation began **16:45:58 UTC**, October 6, 2026.
- **103 bound hashes** match their files: source inputs, archived code/configuration/tests and completed outputs. The selected configuration is the 200-session moving average with full allocation to the modeled daily-reset 3× instrument.

This verifies the new run's recorded sequence. It does not erase earlier exposure to the historical years or make them an untouched research holdout.

## Independent reconstruction

Starting from frozen raw daily close, dividends and split coefficients, the independent calculation rebuilt the causal total-return index, 200-session mean and one-session-delayed targets. It separately integrated the timestamp-available DFF and DTB3 step functions over elapsed calendar intervals, without calling project rate code.

For both the selected strategy and QQQ buy-and-hold, it reconstructed gross, trading-only, financing-inclusive and expense-inclusive paths. Daily cash, quantities, marks, equity, returns and turnover match exported paths; terminal liquidation and reported CAGR, drawdown, volatility, both Sharpe definitions, fees, slippage, order counts and average weight also reconcile within declared numerical tolerances. Both first purchases occur **January 5, 2016**, after the first test-session decision, not before the return ending at that fill close.

The reconstruction verifies that physical QQQ reinvests eligible ex-date cash, new same-close buyers do not receive that distribution, and new dividend cash earns no interest for the interval that just ended. Synthetic borrowing and expense are charged once, not charged again to physical QQQ. First-entry and terminal boundary multipliers reproduce exact terminal wealth when removed and reapplied once.

## Recomputed like-period results

| Path | CAGR | Maximum drawdown | Daily cash-excess Sharpe |
|---|---:|---:|---:|
| Selected, gross | 47.7712% | −58.3971% | 0.993511 |
| Selected, trading | 47.3109% | −58.6956% | 0.987223 |
| Selected, financing | 40.6079% | −59.2625% | 0.893464 |
| Selected, expense-inclusive pretax | **38.9484%** | **−59.4409%** | **0.869563** |
| QQQ, gross | 21.1371% | −35.1225% | 0.872468 |
| QQQ, investor-cost net | **21.1226%** | **−35.1227%** | **0.871930** |

QQQ's trading, financing and expense rows coincide because synthetic funding/expense is not imposed on actual QQQ. The selected strategy has higher modeled return but substantially greater risk and a slightly lower cash-excess Sharpe. Its expense-inclusive volatility is approximately **49.67%**, versus **22.18%** for QQQ; mean effective exposure is approximately **2.45×**, not a risk match to QQQ. **Higher return is not evidence of alpha.** No risk-matched comparison or statistical outperformance conclusion is established by this accounting audit.

## Hypothetical-tax reconciliation

For the selected strategy, QQQ buy-and-hold, the QQQ 200-session control and cash, exported event cash flows, tax sums, fee/slippage totals, holdings/equity identities, terminal costs/taxes and compounded-return/terminal-equity identities all reconcile. Separate artificial hand-calculated tax tests previously covered FIFO gains, ordinary distribution income and annual funding sales.

This real-path check is **structural reconciliation**, not a separate reimplementation of every historical tax lot or a certification of investor tax compliance. Ex-date cash timing, ordinary distribution character and synthetic fund-unit realizations remain explicitly hypothetical. The experiment uses a declared daily accounting clock and retrospective source/rate vintages, not live executable fills or actual TQQQ investor returns.

Machine-readable checks and aggregate recomputed metrics are in `final_accounting_audit.json`. No raw market quotes, private absolute paths or personal information are included in these audit deliverables.
