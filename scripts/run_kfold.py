#!/usr/bin/env python3
"""Sequential, resumable leave-one-lot-out runner (folds x arms x seeds).

For every fold (built by `scripts/make_grouped_folds.py`), seed and arm
(experiment config) it trains one model and scores `best_model.pt` on the
fold's `val` and `test` splits:

    python -m src.training.train --config <arm> --machine <machine> --model master \\
        --override seed=<S> patience=1000 selection_metric=damage_ap50 \\
                   labels_dir=<fold>/labels split_manifest=<fold>/splits.json \\
                   output_dir=<checkpoints-root>/<arm>/fold<k>/seed<S>

    python scripts/evaluate_checkpoint.py --config <arm> --checkpoint <run>/best_model.pt \\
        --split <val|test> --model master --output-dir <reports-root>/<arm>/fold<k>_seed<S>/<split> \\
        --override labels_dir=<fold>/labels split_manifest=<fold>/splits.json ...

The experiment files are used untouched (`experiment_sha256` stays comparable);
`patience=1000` disables early stopping so every arm trains the same number of
epochs, and `selection_metric=damage_ap50` selects the checkpoint on the damage
class for both arms.

Resumable: re-run the exact same command after a crash, a reboot or a Ctrl-C.

  * training is skipped only when `<run>/run_complete.json` exists (written
    after the training subprocess exits 0 and left `best_model.pt`) and the
    checkpoint is still there. `best_model.pt` alone proves nothing: the
    trainer writes it after the first epoch;
  * a run directory without that marker is an interrupted run. If it holds
    weights it is moved aside to `<run>.interrupted-<n>` (a directory with only
    TensorBoard logs is simply removed) and the run is trained from scratch;
    its stale evaluations are dropped;
  * an evaluation is skipped when its `metrics.json` exists.

A failed subprocess stops the runner at once (non-zero exit, clear message);
Ctrl-C / SIGTERM / SIGHUP stop it cleanly (exit 130, the child is killed).
Every command and outcome is appended to `<reports-root>/runner.log`.

Order is fold-major (then seed, then arm), so both arms of a fold are done
before the next fold starts and paired comparisons become available early.

Usage:
    python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml
    python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml --dry-run

    # Then: python scripts/aggregate_kfold.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shlex
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

DEFAULT_ARMS = (
    "configs/experiment/twostream.yaml",
    "configs/experiment/twostream_roi_crop.yaml",
)
DEFAULT_FOLDS_DIR = "data/annotations/yolo/folds_lot"
DEFAULT_CHECKPOINTS_ROOT = "checkpoints/kfold_lot"
DEFAULT_REPORTS_ROOT = "reports/kfold_lot"

EVAL_SPLITS = ("val", "test")
CHECKPOINT = "best_model.pt"
COMPLETE_MARKER = "run_complete.json"
# The protocol shared by every arm: no early stopping, select on damage AP50.
PROTOCOL_OVERRIDES = ("patience=1000", "selection_metric=damage_ap50")

_FOLD_DIR_RE = re.compile(r"fold_(\d+)")


class RunnerError(RuntimeError):
    """A step failed or the run cannot start; the runner stops."""


@dataclass
class RunnerOptions:
    folds_dir: Path
    arms: list[Path]
    folds: list[int] | None
    seeds: list[int]
    machine: Path
    epochs: int | None
    python: str
    checkpoints_root: Path
    reports_root: Path
    dry_run: bool = False
    extra_overrides: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Naming, discovery, paths
# --------------------------------------------------------------------------


def arm_name(config: str | Path) -> str:
    """`configs/experiment/twostream_roi_crop.yaml` -> `twostream_roi_crop`."""
    return Path(config).stem


def parse_int_list(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise ValueError(f"Expected comma-separated integers, got '{text}'.") from None


def discover_folds(folds_dir: Path) -> list[int]:
    """Sorted indices of the `fold_<k>/` dirs (with `splits.json` and `labels/`)."""
    folds_dir = Path(folds_dir)
    indices = []
    for path in folds_dir.glob("fold_*"):
        match = _FOLD_DIR_RE.fullmatch(path.name)
        if match and path.is_dir() and (path / "splits.json").is_file() and (path / "labels").is_dir():
            indices.append(int(match.group(1)))
    if not indices:
        raise FileNotFoundError(
            f"No fold_<k>/ directories (with splits.json and labels/) in {folds_dir}. "
            "Run scripts/make_grouped_folds.py first."
        )
    return sorted(indices)


def run_path(checkpoints_root: Path, arm: str, fold: int, seed: int) -> Path:
    return Path(checkpoints_root) / arm / f"fold{fold}" / f"seed{seed}"


def report_run_dir(reports_root: Path, arm: str, fold: int, seed: int) -> Path:
    return Path(reports_root) / arm / f"fold{fold}_seed{seed}"


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def data_overrides(fold_path: Path) -> list[str]:
    """Point the dataset at one materialized fold (labels copied per split)."""
    return [f"labels_dir={Path(fold_path) / 'labels'}", f"split_manifest={Path(fold_path) / 'splits.json'}"]


def build_train_command(
    python: str,
    config: Path,
    machine: Path,
    fold_path: Path,
    run_path: Path,
    seed: int,
    epochs: int | None,
    extra_overrides: list[str],
) -> list[str]:
    cmd = [
        python, "-m", "src.training.train",
        "--config", str(config), "--machine", str(machine), "--model", "master",
        "--override",
        f"seed={seed}", *PROTOCOL_OVERRIDES, *data_overrides(fold_path), f"output_dir={run_path}",
    ]
    if epochs is not None:
        cmd.append(f"epochs={epochs}")
    return cmd + list(extra_overrides)


def machine_overrides(machine: Path) -> list[str]:
    """The machine profile as `key=value` overrides.

    `evaluate_checkpoint.py` has no `--machine`; without these it would score on
    the TrainingConfig defaults (batch size, workers, device) instead of the
    hardware the runs trained on.
    """
    profile = yaml.safe_load(Path(machine).read_text()) or {}
    return [
        f"{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in profile.items()
    ]


def build_eval_command(
    python: str,
    config: Path,
    checkpoint: Path,
    split: str,
    fold_path: Path,
    scratch_dir: Path,
    output_dir: Path,
    machine_overrides: Sequence[str] = (),
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    """Score `checkpoint` on `split` of one fold.

    `output_dir=<scratch_dir>` redirects the Trainer that `evaluate_checkpoint.py`
    builds (it creates its output dir and a TensorBoard `logs/` on construction)
    away from the experiment file's `checkpoints/<arm>`; `--output-dir` receives
    `metrics.json`.
    """
    return [
        python, "scripts/evaluate_checkpoint.py",
        "--config", str(config), "--checkpoint", str(checkpoint),
        "--split", split, "--model", "master", "--output-dir", str(output_dir),
        "--override",
        *data_overrides(fold_path), f"output_dir={scratch_dir}", *machine_overrides, *extra_overrides,
    ]


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------


def is_train_complete(run_path: Path) -> bool:
    """True only after the training subprocess exited 0 and a checkpoint exists."""
    marker = Path(run_path) / COMPLETE_MARKER
    if not marker.is_file() or not (Path(run_path) / CHECKPOINT).is_file():
        return False
    try:
        return json.loads(marker.read_text()).get("returncode") == 0
    except (OSError, ValueError, AttributeError):
        return False


def mark_train_complete(run_path: Path, command: list[str]) -> None:
    payload = {
        "command": command,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "returncode": 0,
    }
    (Path(run_path) / COMPLETE_MARKER).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def move_aside_incomplete(run_path: Path) -> Path | None:
    """Move an interrupted run to `<run>.interrupted-<n>`; None if nothing was kept.

    A directory holding no `.pt` file (only the TensorBoard logs of an attempt
    that died early) has nothing worth keeping and is removed instead, so
    repeated failed attempts leave no clutter.
    """
    run_path = Path(run_path)
    if not run_path.exists():
        return None
    if not any(run_path.rglob("*.pt")):
        shutil.rmtree(run_path)
        return None
    n = 1
    while run_path.with_name(f"{run_path.name}.interrupted-{n}").exists():
        n += 1
    target = run_path.with_name(f"{run_path.name}.interrupted-{n}")
    run_path.rename(target)
    return target


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


@contextmanager
def _interrupt_on_termination():
    """Treat SIGTERM / SIGHUP like Ctrl-C while the runner is active.

    `subprocess.run` kills its child on KeyboardInterrupt; a bare SIGTERM
    would end the runner and leave the training process running orphaned.
    The previous handlers are restored on exit.
    """
    signals = [s for s in (signal.SIGTERM, getattr(signal, "SIGHUP", None)) if s is not None]

    def raise_interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = {s: signal.signal(s, raise_interrupt) for s in signals}
    try:
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def _execute(cmd: list[str], cwd: Path) -> int:
    return subprocess.run(cmd, cwd=cwd).returncode


class _RunLog:
    """Append-only `runner.log`; a no-op when `path` is None (dry run)."""

    def __init__(self, path: Path | None):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, key: str, text: str = "") -> None:
        if self.path is None:
            return
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with self.path.open("a") as handle:
            handle.write(f"{stamp} {event:<6} {key} {text}".rstrip() + "\n")


def _run_step(key: str, cmd: list[str], options: RunnerOptions, log: _RunLog, execute) -> None:
    line = shlex.join(cmd)
    print(f"[{key}] {line}", flush=True)
    if options.dry_run:
        return
    log.write("START", key, line)
    code = execute(cmd, PROJECT_ROOT)
    if code != 0:
        log.write("FAILED", key, f"exit code {code}")
        raise RunnerError(
            f"{key} failed with exit code {code}. Fix the problem and re-run the same command "
            f"to resume; finished steps are skipped.\n  command: {line}"
        )
    log.write("OK", key, "exit code 0")


def run_kfold(options: RunnerOptions, execute=None) -> dict[str, str]:
    """Run (or resume) every fold x seed x arm. Returns `{step key: status}`.

    Status is `ran`, `skipped` or `dry-run`. Raises `RunnerError` at the first
    failed step.
    """
    execute = execute or _execute
    available = discover_folds(options.folds_dir)
    folds = options.folds if options.folds is not None else available
    for fold in folds:
        if fold not in available:
            raise RunnerError(
                f"Requested fold {fold} is not in {options.folds_dir} (available: {available})."
            )

    log = _RunLog(None if options.dry_run else Path(options.reports_root) / "runner.log")
    eval_machine_overrides = machine_overrides(options.machine)
    summary: dict[str, str] = {}

    def skip(key: str, reason: str) -> None:
        print(f"[{key}] skip: {reason}", flush=True)
        log.write("SKIP", key, reason)
        summary[key] = "skipped"

    for fold in folds:
        fold_path = Path(options.folds_dir) / f"fold_{fold}"
        for seed in options.seeds:
            for config in options.arms:
                arm = arm_name(config)
                base = f"{arm}/fold{fold}/seed{seed}"
                run = run_path(options.checkpoints_root, arm, fold, seed)
                reports = report_run_dir(options.reports_root, arm, fold, seed)

                train_key = f"{base}/train"
                if is_train_complete(run):
                    skip(train_key, f"{COMPLETE_MARKER} found")
                else:
                    cmd = build_train_command(
                        options.python, config, options.machine, fold_path, run, seed,
                        options.epochs, options.extra_overrides,
                    )
                    if not options.dry_run:
                        aside = move_aside_incomplete(run)
                        if aside is not None:
                            logger.warning("%s: interrupted run moved to %s; training from scratch", train_key, aside)
                            log.write("ASIDE", train_key, f"interrupted run moved to {aside}")
                        # Evaluations of a previous checkpoint no longer describe this run.
                        shutil.rmtree(reports, ignore_errors=True)
                        run.parent.mkdir(parents=True, exist_ok=True)
                    _run_step(train_key, cmd, options, log, execute)
                    if not options.dry_run:
                        if not (run / CHECKPOINT).is_file():
                            log.write("FAILED", train_key, f"exit code 0 but no {CHECKPOINT}")
                            raise RunnerError(
                                f"{train_key} exited 0 but wrote no {CHECKPOINT} in {run}; not marking it complete."
                            )
                        mark_train_complete(run, cmd)
                    summary[train_key] = "dry-run" if options.dry_run else "ran"

                for split in EVAL_SPLITS:
                    key = f"{base}/eval_{split}"
                    out = reports / split
                    if (out / "metrics.json").is_file():
                        skip(key, "metrics.json found")
                        continue
                    cmd = build_eval_command(
                        options.python, config, run / CHECKPOINT, split, fold_path,
                        run / f"eval_{split}", out, eval_machine_overrides, options.extra_overrides,
                    )
                    _run_step(key, cmd, options, log, execute)
                    if not options.dry_run and not (out / "metrics.json").is_file():
                        log.write("FAILED", key, "exit code 0 but no metrics.json")
                        raise RunnerError(f"{key} exited 0 but wrote no metrics.json in {out}.")
                    summary[key] = "dry-run" if options.dry_run else "ran"
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _find_file(value: str, what: str) -> Path:
    """Absolute, cwd-relative or repo-relative path to an existing file."""
    path = Path(value)
    for candidate in ([path] if path.is_absolute() else [Path.cwd() / path, PROJECT_ROOT / path]):
        if candidate.is_file():
            return candidate.resolve()
    raise RunnerError(f"{what} not found: {value}")


def _dir(value: str | None, default: str) -> Path:
    """Explicit dirs are cwd-relative; defaults are anchored at the repo root."""
    return (Path(value) if value is not None else PROJECT_ROOT / default).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate every fold x arm x seed, resumably.",
    )
    parser.add_argument("--folds-dir", default=None,
                        help=f"Output of make_grouped_folds.py (default: <repo>/{DEFAULT_FOLDS_DIR}).")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Experiment configs to compare (default: " + " ".join(DEFAULT_ARMS) + ").")
    parser.add_argument("--folds", type=parse_int_list, default=None,
                        help="Comma-separated fold indices to run, e.g. 0,1 (default: every fold found).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds (default: 42).")
    parser.add_argument("--machine", required=True,
                        help="Machine profile, e.g. configs/machines/rtx3080.yaml.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the experiment's epochs (smoke tests only).")
    parser.add_argument("--python", default=sys.executable, help="Interpreter for the subprocesses.")
    parser.add_argument("--checkpoints-root", default=None,
                        help=f"Run dirs go to <root>/<arm>/fold<k>/seed<S> (default: <repo>/{DEFAULT_CHECKPOINTS_ROOT}).")
    parser.add_argument("--reports-root", default=None,
                        help=f"Evaluations go to <root>/<arm>/fold<k>_seed<S>/<split>; also holds runner.log "
                             f"(default: <repo>/{DEFAULT_REPORTS_ROOT}).")
    parser.add_argument("--extra-override", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Extra overrides appended to every train AND eval command (e.g. rgb_dir=... "
                             "nir_dir=...). Passed verbatim; commands run with the repo root as cwd, so "
                             "use absolute paths.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    try:
        arms = (
            [_find_file(a, "Experiment config") for a in args.arms]
            if args.arms is not None else [PROJECT_ROOT / a for a in DEFAULT_ARMS]
        )
        options = RunnerOptions(
            folds_dir=_dir(args.folds_dir, DEFAULT_FOLDS_DIR),
            arms=arms,
            folds=args.folds,
            seeds=args.seeds,
            machine=_find_file(args.machine, "Machine profile"),
            epochs=args.epochs,
            python=args.python,
            checkpoints_root=_dir(args.checkpoints_root, DEFAULT_CHECKPOINTS_ROOT),
            reports_root=_dir(args.reports_root, DEFAULT_REPORTS_ROOT),
            dry_run=args.dry_run,
            extra_overrides=args.extra_override,
        )
        with _interrupt_on_termination():
            summary = run_kfold(options)
    except (RunnerError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume; finished steps are skipped and "
              "the interrupted run is retrained from scratch (its partial files are kept aside).",
              file=sys.stderr)
        return 130
    counts = {status: sum(1 for s in summary.values() if s == status) for status in sorted(set(summary.values()))}
    print(f"Done: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
