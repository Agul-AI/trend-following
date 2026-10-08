# Repository and publishing instructions

- The user-designated repository for v5 and all subsequent updates is
  `https://github.com/Agul-quant/trend-following`.
- Before committing or pushing, verify the current branch, staged files and
  `git remote get-url --push origin`. The destination must be
  `Agul-quant/trend-following`, including when a local SSH host alias is used.
- Do not push to `Agul-AI/trend-following`; any legacy remote is for read-only
  historical reference. Use non-force pushes. Do not replace existing remote
  history or merge unrelated local legacy work into v5 without an explicit request.
- Preserve frozen v4 evidence, prior registrations, private source snapshots,
  existing monitors and resumes. Never retune or overwrite a sealed study.
- Never commit credentials, downloaded/raw market prices, processed price packets,
  `.cache`, human authorization receipts, private freezes or runtime logs.
  Curated metadata reports must not expose secrets or private machine paths.
- Use `~/.venvs/myenv/bin/python` and `~/.venvs/myenv/bin/pip` for this user's local
  Python work when that environment exists. Otherwise use an isolated environment
  with the project dependencies.
- Before publishing v5 changes, run the software tests, Ruff and
  `git diff --check`. Keep software-test evidence separate from market performance.
- The current v5 entry point is `scripts/run_qqq_gap_flags_v5.py`, with the
  separately frozen gap-held policy and descriptive post-gap signal flags.
  Flags never filter, rank or replace configurations. Previously inspected
  historical OOS is retrospective, not untouched or live evidence.
