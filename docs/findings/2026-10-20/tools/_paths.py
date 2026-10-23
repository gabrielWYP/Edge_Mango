"""Shared CLI root resolution for the 2026-10-20 findings tools (analysis tooling, not part of the training pipeline).

Every tool takes the same optional flags instead of hard-coded absolute paths:

  --repo-root  checkout that holds the code (src/, scripts/, configs/, notebooks/).
               Default: the repository this file lives in.
  --data-root  checkout that holds the untracked data (data/cache/incoming/...).
               Default: the environment variable MANGO_DATA_ROOT, else the primary
               checkout of the repository (so a linked git worktree reads the data of
               the main checkout), else --repo-root.
  --out-dir    where the evidence files are written.
               Default: <repo-root>/docs/findings/2026-10-20/evidence.

The tool scripts run under `python -I` (isolated mode), which does not put the script directory on
sys.path; they insert it explicitly before importing this module.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

# <repo>/docs/findings/2026-10-20/tools/_paths.py -> parents[4] is <repo>.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT_ENV = "MANGO_DATA_ROOT"
OUT_DIR_RELATIVE = Path("docs/findings/2026-10-20/evidence")


def primary_checkout(repo_root: Path) -> Path:
    """Primary checkout of the repository that `repo_root` belongs to.

    In a linked worktree `git rev-parse --git-common-dir` points at `<primary>/.git`; in the primary
    checkout (or when git is unavailable) the answer is `repo_root` itself.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return repo_root
    common = Path(out)
    return common.parent if common.name == ".git" else repo_root


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Argument parser that already carries the three root flags; tools add their own on top."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=None,
                        help="code checkout (default: this repository)")
    parser.add_argument("--data-root", type=Path, default=None,
                        help=f"checkout with the untracked data/ (default: ${DATA_ROOT_ENV}, "
                             "else the primary checkout of this repository)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: <repo-root>/docs/findings/2026-10-20/evidence)")
    return parser


def resolve_roots(ns: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Return (repo_root, data_root, out_dir) from a parsed namespace; creates out_dir."""
    repo_root = (ns.repo_root or DEFAULT_REPO_ROOT).resolve()
    if ns.data_root is not None:
        data_root = ns.data_root.resolve()
    elif os.environ.get(DATA_ROOT_ENV):
        data_root = Path(os.environ[DATA_ROOT_ENV]).resolve()
    else:
        data_root = primary_checkout(repo_root).resolve()
    out_dir = (ns.out_dir or repo_root / OUT_DIR_RELATIVE).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return repo_root, data_root, out_dir
