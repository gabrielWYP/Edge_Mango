"""Shared CLI root resolution for the findings analysis tools (analysis tooling, not part of the training pipeline).

Every tool takes the same optional flags instead of hard-coded absolute paths:

  --repo-root  checkout that holds the code (src/, scripts/, configs/, notebooks/).
               Default: the repository this file lives in.
  --data-root  checkout that holds the untracked data (data/, checkpoints/).
               Default: --repo-root.
  --out-dir    where caches, intermediate files and figures are written.
               Default: <repo-root>/reports/findings-2026-10-18-work.

Unknown arguments are returned untouched, so a tool can keep its own positionals.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# <repo>/docs/findings/2026-10-18/tools/_paths.py -> parents[4] is <repo>.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_roots(
    description: str | None = None, *, repo: bool = True, data: bool = True
) -> tuple[Path, Path, Path, list[str]]:
    """Parse the root flags and return (repo_root, data_root, out_dir, leftover_args)."""
    parser = argparse.ArgumentParser(description=description)
    if repo:
        parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT,
                            help="code checkout (default: this repository)")
    if data:
        parser.add_argument("--data-root", type=Path, default=None,
                            help="checkout with data/ and checkpoints/ (default: --repo-root)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: <repo-root>/reports/findings-2026-10-18-work)")
    ns, rest = parser.parse_known_args()
    repo_root = (getattr(ns, "repo_root", None) or DEFAULT_REPO_ROOT).resolve()
    data_root = (getattr(ns, "data_root", None) or repo_root).resolve()
    out_dir = (ns.out_dir or repo_root / "reports" / "findings-2026-10-18-work").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return repo_root, data_root, out_dir, rest
