# Current repository and publishing instructions

- This repository contains only the current 56,644-configuration QQQ/cash study,
  its transitive code dependencies, current tests and current documentation.
  Do not reintroduce archived code, strategies, returns, figures or study reports
  into the current tree or use them to choose current parameters/results.
- The default publishing target is `https://github.com/Agul-quant/trend-following`.
  Before committing/pushing, verify branch, staged files and the origin push URL.
  Never push to `Agul-AI/trend-following`; use non-force pushes and preserve Git
  history unless the user explicitly requests a separate history operation.
- The public entry point is `scripts/run_qqq_research.py`. Keep its current
  immutable config and eight frozen implementation dependencies byte-identical
  during a running or sealed study; internal versioned filenames/schema IDs are
  technical identifiers, not public version-number branding.
- Current development is 2001–2011 against cost-matched QQQ buy-and-hold. Do not
  run validation or historical OOS until the required frozen selections and user
  authorization permit it. Previously inspected historical OOS is retrospective.
- Hold the last filled position during missing source intervals. Gap flags are
  descriptive only and never filter, rank, or replace configurations.
- Never commit credentials, downloaded/raw prices, processed market packets,
  `.cache`, private authorization receipts, freezes, source backups or runtime
  logs. Publish only current source/tests/docs and reviewed current results.
- Use `~/.venvs/myenv/bin/python` and `~/.venvs/myenv/bin/pip` for this user's
  Python work when available; otherwise use an isolated project environment.
- Before publishing, run current synthetic software tests, Ruff and
  `git diff --check`. Software tests/preparation are not market performance.
