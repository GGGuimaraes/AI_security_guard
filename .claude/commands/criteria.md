---
description: Reviews or adjusts config/risk_criteria.md without breaking the cache
argument-hint: "[what to change]"
---

Requested change: $ARGUMENTS

Edit `config/risk_criteria.md` (and `config/risk_criteria.example.md` if the change applies to every home) while respecting:
- 100% static content (no dates, times, counters or file names).
- Keep the JSON output format required by `classifier.py` (the fields and the `risk` values).
- Keep the size above the provider's caching minimum (`tests/test_config.py` checks more than 4000 characters) — don't shrink it too much.
Then run `uv run pytest -q tests/test_config.py tests/test_classifier.py` and show me the diff.
Remember that any change invalidates the cache on the next call (expected).
