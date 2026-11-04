#!/usr/bin/env python3
"""The four arms of the reformulated H1 and H2: the same small student on RGB, on NIR, and distilled.

    est_rgb   labels only, on RGB. Baseline of H1 and H2.
    est_nir   labels only, on NIR. H1: est_nir against est_rgb.
    kd_nir    RGB student distilled from est_nir of its fold and seed. H2: against est_rgb.
    kd_rgb    RGB student distilled from est_rgb of its fold and seed. Control of H2.

Every run is evaluated on val and test, with predictions and metrics_v4.json, under
`<reports-root>/<arm>/fold<k>_seed<S>/<split>/`, the layout of `scripts/aggregate_kfold.py`.
Re-run the same command to resume: finished steps are skipped, and an interrupted training continues from
its last finished epoch (`--no-resume` trains it from scratch). A state from other settings stops the step.

The NIR arm reads `--nir-as-rgb-dir`, made with `--make-nir-as-rgb`: links named like the RGB files.

    python scripts/run_small_arms.py --dataset data/cache/v4_captura_v3_final \\
        --weights data/weights/yolov8n_state.pt --variant wide --epochs 50 --seeds 42 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_kfold as rk  # noqa: E402

STUDENT_CONFIG = "configs/experiment/v4_est_etiquetas.yaml"
KD_CONFIG = "configs/experiment/v4_est_kd_rgbnir.yaml"
NIR_VARIANT = "nir_clahe"
PROTOCOL = ("patience=1000", "selection_metric=damage_ap50")


@dataclass(frozen=True)
class Arm:
    reads_nir: bool = False
    teacher: str | None = None


ARMS = {
    "est_rgb": Arm(),
    "est_nir": Arm(reads_nir=True),
    "kd_nir": Arm(teacher="est_nir"),
    "kd_rgb": Arm(teacher="est_rgb"),
}


# This runner does not hash its settings: the trainer refuses a state from other ones, by name.
IDENTITY = {"sha256": "small-arms"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="built v4 dataset, with folds_lot/")
    parser.add_argument("--nir-as-rgb-dir", type=Path, default=None)
    parser.add_argument("--make-nir-as-rgb", action="store_true", help="create the links, if missing")
    parser.add_argument("--weights", default="", help="YOLOv8n state_dict; empty trains from scratch")
    parser.add_argument("--variant", choices=("wide", "nano"), default="wide")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=list(ARMS))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints/v4_small"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/v4_small"))
    parser.add_argument("--machine", type=Path, default=Path("configs/machines/rtx3080.yaml"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-resume", action="store_true", help="train interrupted runs from scratch")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def make_nir_as_rgb(dataset: Path, target: Path) -> int:
    """Link each NIR image under the name of its RGB image; returns how many links exist."""
    target.mkdir(parents=True, exist_ok=True)
    for nir in (dataset / NIR_VARIANT).glob("mango_nir_*.png"):
        link = target / f"mango_rgb_{nir.stem.removeprefix('mango_nir_')}.jpg"
        if not link.exists():
            link.symlink_to(nir.resolve())
    return sum(1 for _ in target.iterdir())


def plan_arm(args, arm_name: str, fold: int, seed: int) -> list[tuple[str, list[str], Path | None]]:
    """(key, command, marker) of each step. A train step's marker is its run directory."""
    arm = ARMS[arm_name]
    fold_path = args.dataset / "folds_lot" / f"fold_{fold}"
    stats = json.loads((args.dataset / "folds_lot" / "folds_meta.json").read_text())
    roi = stats["folds"][str(fold)]["nir_stats"][NIR_VARIANT]["roi"]
    rgb_dir = args.nir_as_rgb_dir if arm.reads_nir else args.dataset / "rgb"
    data = [
        *rk.data_overrides(fold_path),
        f"rgb_dir={rgb_dir}",
        f"nir_dir={args.dataset / NIR_VARIANT}",
        f"nir_homography_path={args.dataset / 'homographies' / 'identity.npy'}",
        f"nir_mean={roi['mean']}",
        f"nir_std={roi['std']}",
        f"student_variant={args.variant}",
    ]
    if arm.reads_nir:
        # The dataset pads NIR with its mean and RGB with 114. This keeps the pad of
        # the NIR student equal to the one its KD input has as a teacher.
        data.append(f"letterbox_value={int(roi['mean'] * 255)}")
    run = rk.run_path(args.checkpoints_root, arm_name, fold, seed)
    weights = [f"student_pretrained_path={args.weights}"] if args.weights else []
    common = [*data, *PROTOCOL, f"seed={seed}", f"epochs={args.epochs}", f"output_dir={run}"]
    if arm.teacher:
        teacher = rk.run_path(args.checkpoints_root, arm.teacher, fold, seed) / rk.CHECKPOINT
        train = [
            args.python, "-m", "src.training.kd_train", "--config", KD_CONFIG, "--override",
            *common, *weights, *rk.machine_overrides(args.machine), "batch_size=8",
            f"teacher_checkpoint={teacher}", "teacher_kind=student",
            f"teacher_input={'nir' if ARMS[arm.teacher].reads_nir else 'rgb'}",
        ]
    else:
        train = [
            args.python, "-m", "src.training.train", "--config", STUDENT_CONFIG,
            "--machine", str(args.machine), "--model", "student", "--override", *common, *weights,
        ]
    key = f"{arm_name}/fold{fold}/seed{seed}"
    steps: list[tuple[str, list[str], Path | None]] = [(f"{key}/train", train, run)]
    report = rk.report_run_dir(args.reports_root, arm_name, fold, seed)
    for split in ("val", "test"):
        out = report / split
        steps.append((
            f"{key}/eval_{split}",
            rk.build_eval_command(
                args.python, Path(STUDENT_CONFIG), run / rk.CHECKPOINT, split, fold_path,
                run / "eval_scratch", out, rk.machine_overrides(args.machine),
                [x for x in data if not x.startswith(("labels_dir=", "split_manifest="))],
                model="student", save_predictions=True,
            ),
            out / "metrics.json",
        ))
        steps.append((
            f"{key}/metrics_{split}",
            rk.build_v4_metrics_command(
                args.python, out / "predictions.jsonl", args.dataset / "index.csv",
                fold_path / "labels" / split, out / "metrics_v4.json",
            ),
            out / "metrics_v4.json",
        ))
    return steps


def main(argv=None) -> int:
    args = parse_args(argv)
    args.nir_as_rgb_dir = args.nir_as_rgb_dir or args.dataset.parent / "v4_nir_as_rgb_clahe"
    if args.make_nir_as_rgb and not args.dry_run:
        print(f"{make_nir_as_rgb(args.dataset, args.nir_as_rgb_dir)} NIR links in {args.nir_as_rgb_dir}")
    # Teachers first within each fold and seed, whatever order --arms has.
    order = sorted(args.arms, key=lambda name: ARMS[name].teacher is not None)
    done = 0
    for seed in args.seeds:
        for fold in args.folds:
            for arm_name in order:
                for key, command, marker in plan_arm(args, arm_name, fold, seed):
                    is_train = key.endswith("/train")
                    finished = rk.is_train_complete(marker) if is_train else marker.exists()
                    if args.dry_run:
                        print(f"[{'skip' if finished else 'run'}] {key}: {' '.join(map(str, command))}")
                        continue
                    if finished:
                        continue
                    step = command
                    if is_train:
                        state, _ = rk.prepare_training(marker, IDENTITY, resume=not args.no_resume)
                        if state is not None:
                            # Not part of the marker's command: it is that of the run, not of this attempt.
                            step = [*command, "--resume", str(state)]
                    print(f"START {key}{' (continuing)' if step is not command else ''}", flush=True)
                    code = rk._execute(step, PROJECT_ROOT)
                    print(f"END {key} exit {code}", flush=True)
                    if code != 0:
                        return code
                    if is_train:
                        rk.mark_train_complete(marker, command, IDENTITY)
                    done += 1
    print(f"Done: {done} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
