---
name: maintain-cheat-sheet
description: Keep this repository's cheat_sheet.md synchronized with code changes. Use after changing CLI commands, entry points, configuration files, model settings, environment variables, reporting or analysis workflows, or any file linked from the cheat sheet.
---

# Maintain Cheat Sheet

Review [`cheat_sheet.md`](../../../cheat_sheet.md) before finishing relevant
code changes.

1. Inspect the changed implementation rather than copying commands from prior
   conversation context. In particular, check `scripts/`, `pyproject.toml`,
   `src/belief_revision/config.py`, CLI argument definitions, and `sample.env`
   when they are affected.
2. Update only the cheat-sheet sections affected by the change. Add a concise
   entry when a new routine command or user-facing configuration location is
   introduced.
3. Keep commands directly runnable from the repository root. Do not add
   placeholders as if they were working commands.
4. Use repository-relative Markdown links and verify that every local target
   exists by running `scripts/validate_links.py` from this skill directory.
5. Verify changed commands safely. Prefer `--help`, imports, parsing, or a
   local dry run. Do not launch paid model inference, a full experiment, or a
   live write merely to validate documentation unless the user requested it.
6. If a command cannot be exercised without external cost or mutation, verify
   its CLI parsing and underlying entry point and state that limitation in the
   work summary.

Preserve the cheat sheet as a short operational reference. Detailed setup,
architecture, rationale, and migration history belong in the README or
`diff.md`, not here.
