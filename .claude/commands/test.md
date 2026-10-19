---
description: Runs the unit tests and lint, and summarizes failures with their likely cause
---

Run `uv run pytest -q` and `uv run ruff check .`.
If there are failures, for each one: file/test, likely cause and suggested fix.
Don't change code unless I ask, except when the fix is trivial and obvious (typo, import).
