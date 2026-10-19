---
description: Implements the next pending phase of docs/PLAN.md (or the given phase), tests first
argument-hint: "[phase number, optional]"
---

Read `CLAUDE.md` and `docs/PLAN.md` (local, not versioned — if it doesn't exist, say so and stop).

Target phase: $ARGUMENTS (if empty, the first phase with status `[ ]`).

Follow exactly:
1. Re-read the phase's acceptance criteria. If anything is ambiguous or needs a decision from me, **ask and stop**.
2. List in 3–6 lines what you will create/change (files and public interfaces).
3. Write the unit tests first (no network, no camera). Run them and confirm they fail for the right reason.
4. Implement the minimum to make them pass. Add dependencies only with `uv add`.
5. Run `uv run pytest -q` and `uv run ruff check .` until everything is green.
6. If the phase has a `live` smoke test, create it but **don't run it** — tell me the command to run.
7. Mark the phase as `[x]` in PLAN.md and record new decisions in `docs/DECISOES.md`.
8. Stop and report: what was done, the test results, what I should test by hand.
9. For anything that has to be configured outside the code (e.g. access to images, access to an image platform), stop, ask, and give clear instructions on what needs to be done.
Don't move on to the next phase.
