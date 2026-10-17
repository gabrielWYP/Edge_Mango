# Vendored skills

These project skills are copied from an external repository and pinned to one commit.

| Skill | Upstream path |
| --- | --- |
| `statistical-analysis` | `skills/statistical-analysis/` |
| `statistical-power` | `skills/statistical-power/` |
| `hypothesis-generation` | `skills/hypothesis-generation/` |

- **Upstream:** https://github.com/K-Dense-AI/scientific-agent-skills
- **Commit:** `a1043ef544e3a100c28e70a736b6cf32554d0677` (fetched 2026-10-17)
- **License:** MIT, copyright (c) 2025 K-Dense Inc. Each skill directory carries a copy as `LICENSE.md`.

## Local modifications

- Removed the trailing `## Citing Scientific Agent Skills` section from each `SKILL.md`. It told the agent to add the vendor's paper to the user's manuscript references and to fetch arXiv to build that reference. What the thesis cites is the author's decision, not the skill's.

No other file was changed. To update, fetch the same paths at a newer commit, repeat the edit above, and update the commit hash here.

## Review notes

- The bundled scripts were scanned before vendoring: they make no network calls and use no `subprocess`, `eval`, `exec`, or environment variables.
- `statistical-analysis` and `statistical-power` set up their own venvs (`.venv-statistics/`, `.venv-power/`) in the repository root. Both are ignored in `.gitignore`.
- None of these skills covers statistics specific to this project: a lot-level cluster bootstrap for AP, the Nadeau–Bengio corrected resampled t-test or Benavoli's correlated t-test for comparing models across leave-one-lot-out folds, or seed-variance planning. Those need a project skill of their own.
