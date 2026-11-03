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
  * the marker records an identity: sha256 of the train command, the config and
    machine files and the fold's content. A marker with another identity (other
    `--epochs`, `--extra-override`, machine, config or fold) stops the runner;
    it is never skipped nor overwritten;
  * a run directory without that marker is an interrupted run. If it holds
    weights it is moved aside to `<run>.interrupted-<n>` (a directory with only
    TensorBoard logs is simply removed) and the run is trained from scratch;
    its stale evaluations are dropped;
  * an evaluation is skipped when its `metrics.json` exists and its
    `eval_complete.json` has the current identity (else the runner stops).

A failed subprocess stops the runner at once (non-zero exit, clear message);
Ctrl-C / SIGTERM / SIGHUP stop it cleanly (exit 130, the child is killed).
Every command and outcome is appended to `<reports-root>/runner.log`.

Order is fold-major (then seed, then arm), so both arms of a fold are done
before the next fold starts and paired comparisons become available early.

Arms of plan-experimental E1 and E2: `configs/experiment/v4_<arm>.yaml`, on the
folds of `scripts/make_v4_folds.py` (see `V4_ARMS` for what each arm runs with;
exact commands in `docs/tesis/runbook-v4.md`). A v4 arm differs from the above in:

  * its name is the one in the plan (`maestro_rgb`), for the run and report dirs;
  * the data comes from `--dataset` (default: the parent of `--folds-dir`), with
    `nir_mean` / `nir_std` of the fold from `<folds-dir>/folds_meta.json`, for
    `--nir-variant` and the `frame` or `roi` view of the arm's `roi_crop`;
  * a student trains with `src.training.train --model student` or, with a teacher,
    `src.training.kd_train`, and is evaluated with `--model student`, RGB only;
  * a student distills from the teacher of the same fold and seed. Teachers in
    `--arms` train first whatever the order given. Any other teacher must already be
    complete, under the same settings, or the runner stops before anything runs;
    there is no fallback;
  * each evaluation also writes `predictions.jsonl`, followed by a step that runs
    `scripts/v4_metrics.py` into `metrics_v4.json`, with its own
    `metrics_v4_complete.json` identity marker;
  * the identity also covers `folds_meta.json` and, for a KD run, the bytes of the
    teacher checkpoint (a retrained teacher stops its students) and the evaluation
    config.

Usage:
    python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml
    python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml --dry-run

    # Then: python scripts/aggregate_kfold.py
"""

from __future__ import annotations

import argparse
import hashlib
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
EVAL_MARKER = "eval_complete.json"
# The protocol shared by every arm: no early stopping, select on damage AP50.
PROTOCOL_OVERRIDES = ("patience=1000", "selection_metric=damage_ap50")

FOLDS_META = "folds_meta.json"
PREDICTIONS = "predictions.jsonl"
METRICS_V4 = "metrics_v4.json"
METRICS_V4_MARKER = "metrics_v4_complete.json"
NIR_VARIANTS = ("nir", "nir_clahe")
DEFAULT_NIR_VARIANT = "nir"
V4_CONFIG_PREFIX = "v4_"
# `evaluate_checkpoint.py` loads a TrainingConfig, which refuses the KD keys, so
# every student is scored with the config of the student without a teacher.
V4_STUDENT_EVAL_CONFIG = "v4_est_etiquetas.yaml"
# In the identity of a KD run on a dry run whose plan trains the teacher first.
TEACHER_PENDING = "pending"

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
    dataset: Path | None = None
    nir_variant: str = DEFAULT_NIR_VARIANT


@dataclass(frozen=True)
class V4Arm:
    """What an arm of the plan runs with, besides its config file.

    It is here and not in the config because TrainingConfig and KDConfig refuse
    unknown keys, and `modality_ablation` is an override shared by the steps of the arm.
    """

    trainer: str  # "master", "student" or "kd"
    modality_ablation: str | None = None
    teacher: str | None = None  # the arm this one distills from


V4_ARMS = {
    "maestro_rgbnir": V4Arm("master"),
    "maestro_rgb": V4Arm("master", modality_ablation="nir_const"),
    "maestro_nir": V4Arm("master", modality_ablation="rgb_const"),
    "est_etiquetas": V4Arm("student"),
    # The teacher has to see the constant NIR it was trained on.
    "est_kd_rgb": V4Arm("kd", modality_ablation="nir_const", teacher="maestro_rgb"),
    "est_kd_rgbnir": V4Arm("kd", teacher="maestro_rgbnir"),
}


@dataclass(frozen=True)
class V4Folds:
    """The dataset and per-fold NIR stats of the v4 arms, checked before anything runs."""

    dataset: Path
    nir_variant: str
    nir_stats: dict[int, dict[str, dict[str, float]]]  # fold -> view -> {"mean", "std"}
    meta_sha256: str


# --------------------------------------------------------------------------
# Naming, discovery, paths
# --------------------------------------------------------------------------


def arm_name(config: str | Path) -> str:
    """`configs/experiment/twostream_roi_crop.yaml` -> `twostream_roi_crop`.

    A config of the plan takes the name in the plan: `v4_maestro_rgb.yaml` -> `maestro_rgb`.
    """
    stem = Path(config).stem
    name = stem.removeprefix(V4_CONFIG_PREFIX)
    return name if name in V4_ARMS else stem


def v4_arm(config: str | Path) -> V4Arm | None:
    """The arm of a `v4_*` config, None for any other config.

    An unknown `v4_*` config stops the runner: run as a plain arm, it would train on
    the data and NIR stats of its file, with no teacher and no ablation.
    """
    stem = Path(config).stem
    if not stem.startswith(V4_CONFIG_PREFIX):
        return None
    try:
        return V4_ARMS[stem.removeprefix(V4_CONFIG_PREFIX)]
    except KeyError:
        raise RunnerError(f"{stem} is not an arm of the plan; V4_ARMS has: {', '.join(V4_ARMS)}.") from None


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


def load_v4_folds(options: RunnerOptions, folds: Sequence[int]) -> V4Folds:
    """Read `folds_meta.json` and check that the folds were built from the dataset."""
    folds_dir = Path(options.folds_dir)
    dataset = Path(options.dataset) if options.dataset is not None else folds_dir.parent
    meta_path = folds_dir / FOLDS_META
    hint = "The v4 arms need the folds of scripts/make_v4_folds.py."
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError) as error:
        raise RunnerError(f"Cannot read {meta_path}: {error}. {hint}") from None
    nir_stats = {}
    for fold in folds:
        try:
            views = meta["folds"][str(fold)]["nir_stats"][options.nir_variant]
            nir_stats[fold] = {
                view: {"mean": views[view]["mean"], "std": views[view]["std"]} for view in ("frame", "roi")
            }
        except (KeyError, TypeError):
            raise RunnerError(
                f"{meta_path} has no nir_stats of fold {fold} and variant {options.nir_variant}. {hint}"
            ) from None
    index = dataset / "index.csv"
    if not index.is_file():
        raise RunnerError(f"{index} not found: --dataset must be the output of scripts/build_v4_dataset.py.")
    if meta.get("index_sha256") != _file_sha256(index):
        raise RunnerError(f"{folds_dir} was built from another {index}. Rebuild it with scripts/make_v4_folds.py.")
    return V4Folds(dataset, options.nir_variant, nir_stats, _file_sha256(meta_path))


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def data_overrides(fold_path: Path) -> list[str]:
    """Point the dataset at one materialized fold (labels copied per split)."""
    return [f"labels_dir={Path(fold_path) / 'labels'}", f"split_manifest={Path(fold_path) / 'splits.json'}"]


def crop_view(config: Path, extra_overrides: Sequence[str]) -> str:
    """The `folds_meta.json` view of an arm: `roi` with its `roi_crop` on, else `frame`."""
    roi_crop = (yaml.safe_load(Path(config).read_text()) or {}).get("roi_crop", False)
    for token in extra_overrides:
        if token.startswith("roi_crop="):
            roi_crop = token.split("=", 1)[1].lower() in ("true", "1", "yes")
    return "roi" if roi_crop else "frame"


def v4_overrides(spec: V4Arm, config: Path, fold: int, v4: V4Folds, extra_overrides: Sequence[str]) -> list[str]:
    """Data, NIR normalization and ablation of one arm on one fold, the same on all its steps."""
    stats = v4.nir_stats[fold][crop_view(config, extra_overrides)]
    ablation = [f"modality_ablation={spec.modality_ablation}"] if spec.modality_ablation else []
    return [
        f"rgb_dir={v4.dataset / 'rgb'}", f"nir_dir={v4.dataset / v4.nir_variant}",
        f"nir_homography_path={v4.dataset / 'homographies' / 'identity.npy'}",
        f"nir_mean={stats['mean']}", f"nir_std={stats['std']}", *ablation,
    ]


def build_train_command(
    python: str,
    config: Path,
    machine: Path,
    fold_path: Path,
    run_path: Path,
    seed: int,
    epochs: int | None,
    extra_overrides: list[str],
    model: str = "master",
    arm_overrides: Sequence[str] = (),
) -> list[str]:
    cmd = [
        python, "-m", "src.training.train",
        "--config", str(config), "--machine", str(machine), "--model", model,
        "--override",
        f"seed={seed}", *PROTOCOL_OVERRIDES, *data_overrides(fold_path), f"output_dir={run_path}", *arm_overrides,
    ]
    if epochs is not None:
        cmd.append(f"epochs={epochs}")
    return cmd + list(extra_overrides)


def build_kd_command(
    python: str,
    config: Path,
    machine: Path,
    fold_path: Path,
    run_path: Path,
    teacher_checkpoint: Path,
    seed: int,
    epochs: int | None,
    extra_overrides: Sequence[str],
    arm_overrides: Sequence[str] = (),
) -> list[str]:
    """Distill one student. `kd_train.py` has no `--machine`, so the profile goes as overrides.

    All but `batch_size`: KDTrainer has no gradient accumulation, so here it is the batch
    of each optimizer step and comes from the config, equal to `effective_batch`. With the
    batch of a smaller profile, a KD arm would take more steps than `est_etiquetas`.
    """
    profile = [token for token in machine_overrides(machine) if not token.startswith("batch_size=")]
    cmd = [
        python, "-m", "src.training.kd_train",
        "--config", str(config),
        "--override",
        f"seed={seed}", *PROTOCOL_OVERRIDES, *data_overrides(fold_path), f"output_dir={run_path}",
        f"teacher_checkpoint={teacher_checkpoint}", *profile, *arm_overrides,
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
    model: str = "master",
    save_predictions: bool = False,
) -> list[str]:
    """Score `checkpoint` on `split` of one fold.

    `output_dir=<scratch_dir>` redirects the Trainer that `evaluate_checkpoint.py`
    builds (it creates its output dir and a TensorBoard `logs/` on construction)
    away from the experiment file's `checkpoints/<arm>`; `--output-dir` receives
    `metrics.json` and, with `save_predictions`, `predictions.jsonl`.
    """
    return [
        python, "scripts/evaluate_checkpoint.py",
        "--config", str(config), "--checkpoint", str(checkpoint),
        "--split", split, "--model", model, "--output-dir", str(output_dir),
        *(["--save-predictions"] if save_predictions else []),
        "--override",
        *data_overrides(fold_path), f"output_dir={scratch_dir}", *machine_overrides, *extra_overrides,
    ]


def build_v4_metrics_command(
    python: str, predictions: Path, index: Path, labels_dir: Path, output: Path,
) -> list[str]:
    """Stratified metrics of one evaluation, from its `predictions.jsonl`."""
    return [
        python, "scripts/v4_metrics.py",
        "--predictions", str(predictions), "--index", str(index),
        "--labels-dir", str(labels_dir), "--output", str(output),
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


def _digest(value) -> str:
    """sha256 of the canonical JSON of `value`."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(parts: dict[str, str]) -> dict[str, str]:
    return {**parts, "sha256": _digest(parts)}


def run_identity(
    command: Sequence[str], config: Path, machine: Path, fold_path: Path, extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Digest of each input that determines a training run, plus their overall `sha256`.

    The fold is `splits.json` plus every file under `labels/` (path and bytes), sorted.
    `extra` holds the digests of the inputs only some arms have.
    """
    fold_path = Path(fold_path)
    files = [fold_path / "splits.json", *sorted(p for p in (fold_path / "labels").rglob("*") if p.is_file())]
    return _identity({
        "command": _digest(list(command)),
        "config": _file_sha256(config),
        "machine": _file_sha256(machine),
        "fold": _digest([[p.relative_to(fold_path).as_posix(), _file_sha256(p)] for p in files]),
        **(extra or {}),
    })


def eval_identity(
    run_sha256: str, split: str, command: Sequence[str], extra: dict[str, str] | None = None,
) -> dict[str, str]:
    return _identity({"run": run_sha256, "split": split, "command": _digest(list(command)), **(extra or {})})


def metrics_identity(eval_sha256: str, command: Sequence[str]) -> dict[str, str]:
    return _identity({"eval": eval_sha256, "command": _digest(list(command))})


def require_identity(marker: Path, current: dict[str, str], key: str) -> None:
    """Return only if `marker` records `current`; a missing, legacy or other identity stops the runner."""
    try:
        stored = json.loads(Path(marker).read_text()).get("identity")
    except (OSError, ValueError, AttributeError):
        stored = None
    stored = stored if isinstance(stored, dict) else {}
    if stored.get("sha256") == current["sha256"]:
        return
    detail = (
        "changed: " + ", ".join(k for k in current if k != "sha256" and stored.get(k) != current[k])
        if stored else f"{Path(marker).name} has no identity"
    )
    where = Path(marker).parent
    raise RunnerError(
        f"{key}: {where} comes from other settings ({detail}); it is not reused and not overwritten. "
        f"Use a different --checkpoints-root/--reports-root, or delete {where} and re-run."
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def mark_train_complete(run_path: Path, command: list[str], identity: dict[str, str]) -> None:
    payload = {
        "command": command,
        "commit": _git("rev-parse", "HEAD"),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "identity": identity,
        "returncode": 0,
    }
    (Path(run_path) / COMPLETE_MARKER).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_identity(marker: Path, identity: dict[str, str]) -> None:
    Path(marker).write_text(json.dumps({"identity": identity}, indent=2, sort_keys=True) + "\n")


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


def plan_train(
    config: Path, fold: int, seed: int, options: RunnerOptions, v4: V4Folds | None,
) -> tuple[list[str], dict[str, str]]:
    """The train command of one arm, fold and seed, and the identity a resume must find."""
    fold_path = Path(options.folds_dir) / f"fold_{fold}"
    run = run_path(options.checkpoints_root, arm_name(config), fold, seed)
    spec = v4_arm(config)
    if spec is None:
        cmd = build_train_command(
            options.python, config, options.machine, fold_path, run, seed,
            options.epochs, options.extra_overrides,
        )
        return cmd, run_identity(cmd, config, options.machine, fold_path)

    arm_overrides = v4_overrides(spec, config, fold, v4, options.extra_overrides)
    extra = {"folds_meta": v4.meta_sha256}
    if spec.teacher is None:
        cmd = build_train_command(
            options.python, config, options.machine, fold_path, run, seed,
            options.epochs, options.extra_overrides, model=spec.trainer, arm_overrides=arm_overrides,
        )
    else:
        teacher_run = run_path(options.checkpoints_root, spec.teacher, fold, seed)
        cmd = build_kd_command(
            options.python, config, options.machine, fold_path, run, teacher_run / CHECKPOINT, seed,
            options.epochs, options.extra_overrides, arm_overrides,
        )
        extra["teacher"] = (
            _file_sha256(teacher_run / CHECKPOINT) if is_train_complete(teacher_run) else TEACHER_PENDING
        )
    return cmd, run_identity(cmd, config, options.machine, fold_path, extra)


def plan_eval(
    config: Path, split: str, fold: int, seed: int, options: RunnerOptions, v4: V4Folds | None,
    profile: Sequence[str], run_sha256: str,
) -> tuple[list[str], dict[str, str]]:
    """The evaluation command of one arm, fold, seed and split, and its identity."""
    fold_path = Path(options.folds_dir) / f"fold_{fold}"
    arm = arm_name(config)
    run = run_path(options.checkpoints_root, arm, fold, seed)
    out = report_run_dir(options.reports_root, arm, fold, seed) / split
    spec = v4_arm(config)
    if spec is None:
        cmd = build_eval_command(
            options.python, config, run / CHECKPOINT, split, fold_path,
            run / f"eval_{split}", out, profile, options.extra_overrides,
        )
        return cmd, eval_identity(run_sha256, split, cmd)

    eval_config = Path(config).with_name(V4_STUDENT_EVAL_CONFIG) if spec.teacher else config
    cmd = build_eval_command(
        options.python, eval_config, run / CHECKPOINT, split, fold_path, run / f"eval_{split}", out, profile,
        [*v4_overrides(spec, config, fold, v4, options.extra_overrides), *options.extra_overrides],
        model="master" if spec.trainer == "master" else "student", save_predictions=True,
    )
    extra = {"eval_config": _file_sha256(eval_config)} if spec.teacher else None
    return cmd, eval_identity(run_sha256, split, cmd, extra)


def require_teacher(config: Path, fold: int, seed: int, options: RunnerOptions, v4: V4Folds) -> None:
    """Return only if the teacher of this fold and seed finished, under the current settings."""
    teacher = v4_arm(config).teacher
    key = f"{arm_name(config)}/fold{fold}/seed{seed}/train"
    run = run_path(options.checkpoints_root, teacher, fold, seed)
    if not is_train_complete(run):
        raise RunnerError(
            f"{key}: the teacher checkpoint {run / CHECKPOINT} is missing or its run did not finish "
            f"(no {COMPLETE_MARKER}). Train {teacher} on this fold and seed first, with the same "
            "--folds-dir, --nir-variant and --checkpoints-root, or add it to --arms."
        )
    _, identity = plan_train(Path(config).with_name(f"{V4_CONFIG_PREFIX}{teacher}.yaml"), fold, seed, options, v4)
    require_identity(run / COMPLETE_MARKER, identity, key)


def run_kfold(options: RunnerOptions, execute=None) -> dict[str, str]:
    """Run (or resume) every fold x seed x arm. Returns `{step key: status}`.

    Status is `ran`, `skipped` or `dry-run`. Raises `RunnerError` at the first
    failed step, or before any step when a v4 arm cannot run.
    """
    execute = execute or _execute
    available = discover_folds(options.folds_dir)
    folds = options.folds if options.folds is not None else available
    for fold in folds:
        if fold not in available:
            raise RunnerError(
                f"Requested fold {fold} is not in {options.folds_dir} (available: {available})."
            )

    specs = {config: v4_arm(config) for config in options.arms}
    v4 = load_v4_folds(options, folds) if any(specs.values()) else None
    # Within a fold and seed, distilling arms go after every teacher of this plan.
    arms = sorted(options.arms, key=lambda config: bool(specs[config] and specs[config].teacher))
    planned = {arm_name(config) for config in arms}
    for config, spec in specs.items():
        if spec is not None and spec.teacher is not None and spec.teacher not in planned:
            for fold in folds:
                for seed in options.seeds:
                    require_teacher(config, fold, seed, options, v4)

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
            for config in arms:
                arm = arm_name(config)
                spec = specs[config]
                base = f"{arm}/fold{fold}/seed{seed}"
                run = run_path(options.checkpoints_root, arm, fold, seed)
                reports = report_run_dir(options.reports_root, arm, fold, seed)

                train_key = f"{base}/train"
                if spec is not None and spec.teacher is not None and not options.dry_run:
                    require_teacher(config, fold, seed, options, v4)
                cmd, identity = plan_train(config, fold, seed, options, v4)
                if is_train_complete(run):
                    require_identity(run / COMPLETE_MARKER, identity, train_key)
                    skip(train_key, f"{COMPLETE_MARKER} found")
                else:
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
                        mark_train_complete(run, cmd, identity)
                    summary[train_key] = "dry-run" if options.dry_run else "ran"

                for split in EVAL_SPLITS:
                    key = f"{base}/eval_{split}"
                    out = reports / split
                    cmd, eval_id = plan_eval(
                        config, split, fold, seed, options, v4, eval_machine_overrides, identity["sha256"],
                    )
                    outputs = ("metrics.json", PREDICTIONS) if spec is not None else ("metrics.json",)
                    evaluated = False
                    if (out / "metrics.json").is_file():
                        require_identity(out / EVAL_MARKER, eval_id, key)
                    if all((out / name).is_file() for name in outputs):
                        skip(key, "metrics.json found")
                    else:
                        evaluated = True
                        if not options.dry_run:
                            # The v4 metrics of the predictions about to be replaced.
                            (out / METRICS_V4).unlink(missing_ok=True)
                        _run_step(key, cmd, options, log, execute)
                        if not options.dry_run:
                            for name in outputs:
                                if not (out / name).is_file():
                                    log.write("FAILED", key, f"exit code 0 but no {name}")
                                    raise RunnerError(f"{key} exited 0 but wrote no {name} in {out}.")
                            _write_identity(out / EVAL_MARKER, eval_id)
                        summary[key] = "dry-run" if options.dry_run else "ran"

                    if spec is None:
                        continue
                    key = f"{base}/metrics_{split}"
                    cmd = build_v4_metrics_command(
                        options.python, out / PREDICTIONS, v4.dataset / "index.csv",
                        fold_path / "labels" / split, out / METRICS_V4,
                    )
                    metrics_id = metrics_identity(eval_id["sha256"], cmd)
                    if not evaluated and (out / METRICS_V4).is_file():
                        require_identity(out / METRICS_V4_MARKER, metrics_id, key)
                        skip(key, f"{METRICS_V4} found")
                        continue
                    _run_step(key, cmd, options, log, execute)
                    if not options.dry_run:
                        if not (out / METRICS_V4).is_file():
                            log.write("FAILED", key, f"exit code 0 but no {METRICS_V4}")
                            raise RunnerError(f"{key} exited 0 but wrote no {METRICS_V4} in {out}.")
                        _write_identity(out / METRICS_V4_MARKER, metrics_id)
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
                        help=f"Output of make_grouped_folds.py or, for the v4 arms, make_v4_folds.py "
                             f"(default: <repo>/{DEFAULT_FOLDS_DIR}).")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Experiment configs to compare (default: " + " ".join(DEFAULT_ARMS) + "). "
                             "configs/experiment/v4_*.yaml are the arms of plan E1 and E2; teachers train "
                             "first whatever the order given.")
    parser.add_argument("--dataset", default=None,
                        help="v4 arms only: output of build_v4_dataset.py, with rgb/, nir/, nir_clahe/, "
                             "homographies/ and index.csv (default: the parent of --folds-dir).")
    parser.add_argument("--nir-variant", choices=NIR_VARIANTS, default=DEFAULT_NIR_VARIANT,
                        help="v4 arms only: NIR images and per-fold NIR stats to use (default: %(default)s). "
                             "Give each variant its own --checkpoints-root and --reports-root.")
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
            dataset=Path(args.dataset).resolve() if args.dataset is not None else None,
            nir_variant=args.nir_variant,
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
