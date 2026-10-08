# Manual forward paper testing

This is a reviewed **fail-closed manual prototype**, not an unattended production
system or evidence of positive future returns. The supplied
`configs/qqq_forward_paper_v1.yaml` documents the original October 2 pilot's exact
policy declaration and private input hashes. Its actual warmup, freeze and ledger
are not published, so that declaration is **not a turnkey new trial**.

## Scope

- Fixed B5 rule; actual QQQ signals and actual TQQQ paper quotes. No retuning.
- $1,000 hypothetical initial cash and no inherited holdings or warmup returns.
- Seven completed observations on a full New York session; four on an early
  close. Vendor labels are bar starts, and the final interval is 30 minutes.
- One-bar-delayed hypothetical close fills, explicit acknowledgement, 1 bp fee
  and 5 bps adverse slippage. Orders declare target allocation before the later
  observed quote sizes shares within the cash budget.
- Zero cash interest and pretax paper accounting. Actual TQQQ prices already
  reflect fund-level costs; no duplicate synthetic financing charge.
- Policy state is reconstructed and compared with hash-bound checkpoints, not
  accepted from a free-form assertion. The generic recorder alone is not policy
  verification.

## Operating limits

Every expected completed bar needs recording within 15 minutes. Missing/stale
observations, missed eligible fills, unsupported dividends/splits/capital-gain
payments or partial crash batches stop the version. It does not backfill or
silently recover. Scheduled calendar coverage ends October 8, 2027; exceptional
exchange closures require review. New action/recovery support requires a new
reviewed version. No uninterrupted six-/twelve-month readiness is claimed.

The separate `snapshot_qqq_forward_bar.py` collects one immutable source snapshot;
`run_qqq_forward_paper.py init|record|verify` implements policy-checked paper events.
No real orders, notifications or recurring automation are created.

A new trial must acquire reviewed warmup data, write a new versioned contract with
its actual hashes, pass tests, then use `freeze_qqq_research_protocol.py --stage
final_code --attest-final-code-ready` with a verified future exchange schedule.
Its holdout starts at the first complete scheduled session **after actual freeze
completion**. It must not reuse the old October 5 start as a new experiment date.
Descriptive six-/twelve-month reviews are conditional on observed coverage and
are not strong statistical proof of an edge.
