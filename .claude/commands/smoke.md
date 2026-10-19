---
description: Explains and prepares the real smoke test for a component (camera | ai | telegram | pipeline)
argument-hint: "camera | ai | telegram | pipeline"
---

Component: $ARGUMENTS

1. Check whether the matching smoke test exists in `tests/` (marked `@pytest.mark.live`) or `scripts/`.
2. List which `.env` variables it needs (names only, never values — you can't read `.env`).
3. Give me the exact command to run and what I should look for as success.
4. Don't run `live` tests on your own.
