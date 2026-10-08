# Research validation workflow

## Publication boundary

This checkout publishes the reviewed research modules, seven manual CLIs,
artificial-input tests and selected aggregate outputs. It does **not** distribute
the licensed vendor price caches, private exact freezes, daily/hourly account
paths, live paper ledger, warmup or personal resume files.

The October 2 historical outputs were generated before subsequent prospective
recorder work. The numerical research modules are preserved in this publication,
but this reduced checkout is a **new snapshot**, not a byte-identical copy of the
original broader dirty working tree. Do not fabricate a matching original freeze
or claim that a fresh data download reproduces its exact vendor vintages.

## Install and check

Use Python >=3.10 in a virtual environment on macOS/Linux. Run
`python -m pip install -e '.[dev]'`, `python -m pytest` and
`python -m ruff check .`. The publication suite has 193 checks based on artificial
fixtures; they do not prove a market edge. Dependencies are declared in
`pyproject.toml`. The original Markdown tables used `tabulate==0.9.0`.

## Historical inputs and a fresh run

The runner requires the price files listed in
`configs/qqq_research_v1.yaml`: QQQ/TQQQ hourly caches, a separate synthetic
QQQ 3x price path, daily adjusted quotes plus split/distribution fields, and
causally mapped DFF/DTB3 rate caches. `research_data.load_research_price` uses the
existing price-file validator. Use data that you are licensed to obtain and use;
none is included here. Never invent or zero-fill missing financing history.

The rate download is public-source acquisition; market-input acquisition and
synthetic cache preparation remain separate prerequisites. All outputs must use
new directory names. From this repository root, after preparing required inputs:

```bash
python scripts/download_research_rates.py --output-dir data/raw/research_rates/NEW_RUN
python scripts/freeze_qqq_research_protocol.py \
  --protocol configs/qqq_research_v1.yaml --stage protocol_only \
  --output-dir reports/research_validation/NEW_PROTOCOL
python scripts/run_qqq_research_validation.py \
  --manifest reports/research_validation/NEW_PROTOCOL/manifest.json \
  --rate-dir data/raw/research_rates/NEW_RUN \
  --output-dir reports/research_validation/NEW_HISTORICAL
python scripts/run_qqq_research_bootstrap.py \
  --manifest reports/research_validation/NEW_PROTOCOL/manifest.json \
  --rate-dir data/raw/research_rates/NEW_RUN \
  --output-dir reports/research_validation/NEW_STRESS \
  --n-sims 1000 --block-lengths 60 126 252 --workers 8
```

If inputs differ, report the resulting experiment as a new retrospective study.
The original aggregate outputs are in `reports/research_validation`; full numeric
reproduction requires the original exact inputs and private snapshot, not merely
this public checkout. Resume or review claims must preserve that distinction.

## Accounting and metrics

- Close observations, decisions, delayed fills and subsequently earned returns
  are separate. The signal close cannot earn a return ending at the fill.
- One account carries holdings, FIFO tax lots, loss carry and costs across folds;
  it never imports wealth from a candidate that was not held.
- First-bar drawdowns include initial capital. Cash-excess Sharpe uses aligned
  daily cash-proxy returns; zero-risk-free Sharpe is separately identified.
- CAGR annualizes observed sessions at 252/year; drawdown is measured on daily
  last-cached-session-bar equity, not the worst intraday loss.
- DFF financing uses ACT/360 and DTB3-derived cash accrual ACT/365. Latest-vintage
  rates with an assumed availability lag are not point-in-time borrowing quotes.
- Hypothetical taxes use ST24/LT15/cash24, annual settlement, loss carry and
  same-security wash-sale handling. Household-specific taxes, NIIT, state taxes,
  other accounts and investor-specific filing cases are outside the model.
- Synthetic leveraged resets use the last observed session bar, not necessarily
  an actual ETF's 16:00 reset. Actual-fund prices are separate and fund expenses
  are not charged twice. Distributions/splits require explicit accounting.
- Fixed-target attribution holds target allocations, not identical share orders,
  constant. End-to-end cost scenarios can change rule decisions.

## Manual forward module

See `forward_paper_contract.md`. It cannot be started from the omitted private
warmup or manifests. A new trial requires reviewed input hashes, a new contract
version, tests and an actual final-code freeze. No historical bars may be passed
off as future observations. The code creates no broker connection or scheduler.

## Prospective OOS extension

The new [frozen-rule study](oos_study_v2.md) uses a separate input package and
versioned final freeze. It does not reuse or backfill the original manual pilot.
Its `run_qqq_oos_study.py` CLI separates acquisition, readiness, freeze verification
and embargoed 12/24-month scoring. Source/parameter/runtime checks precede any
performance output.
