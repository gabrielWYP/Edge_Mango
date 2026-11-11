#!/usr/bin/env python
"""Can RGB features predict where the teacher's decision depends on NIR, on a lot not used for fitting?

For each fold and seed, with the frozen est_rgbnir (teacher) and est_rgb (control student, no KD):

1. NIR dependence per cell: |sigmoid(z_full) - sigmoid(z_no_nir)| on the damage channel of the teacher, with the
   NIR replaced by zeros (its normalized mean), as `scripts/kd_feature_audit.py` ablates it.
2. Label per cell, per image and level P3-P5: inside the fruit-or-damage cells, 1 where the dependence is above
   the median of those cells (the `nir_high` gate of `src/training/kd_crosshead.py`), 0 for the rest.
3. Features: the control student's FPN map and its class-head stem at that level, concatenated per cell.
4. A logistic probe (`src.evaluation.kd_audit.probe_auroc`) fitted on the fold's train lots and scored on its
   val lot, plus the same fit with the train labels shuffled as a sanity control.

An AUROC well above the shuffled one says the RGB features hold what tells where NIR matters; near 0.5 says they
do not. The test split is never read. Descriptive only. Writes `<output>/probe.json` and `<output>/README.md`.

    python scripts/kd_nir_probe.py --dataset data/cache/v4_captura_v3_final \\
        --checkpoints-root checkpoints/v4_small --folds 0 1 2 3 4 --seed 42 --output reports/kd-nir-probe
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_kfold as rk  # noqa: E402
from scripts import run_small_arms as rs  # noqa: E402
from scripts.kd_feature_audit import batches, dataset  # noqa: E402
from src.evaluation.arms import segment_arm_name  # noqa: E402
from src.evaluation.kd_audit import probe_auroc  # noqa: E402
from src.training.kd_config import KDConfig  # noqa: E402
from src.training.kd_crosshead import compute_gate, nir_dependence  # noqa: E402
from src.training.kd_loss import build_region_masks  # noqa: E402
from src.training.kd_train import apply_overrides  # noqa: E402
from src.training.kd_trainer import KDTrainer  # noqa: E402

LEVELS = ("P3", "P4", "P5")
TEACHER_ARM = "est_rgbnir"
CONTROL_ARM = "est_rgb"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="built v4 dataset, with folds_lot/")
    parser.add_argument("--checkpoints-root", type=Path, required=True, help="<arm>/fold<k>/seed<s>/best_model.pt")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("reports/kd-nir-probe"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--machine", type=Path, default=Path("configs/machines/rtx3080.yaml"))
    parser.add_argument("--variant", choices=("wide", "nano"), default="wide")
    parser.add_argument("--cells-per-image", type=int, default=256, help="cap per image and level, for memory")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--segment", action="store_true",
                        help="probe the segment masters (master_rgbnir_seg teacher, master_rgb_seg control); "
                             "their task comes from the checkpoint")
    return parser.parse_args(argv)


def arms_of(args) -> tuple[str, str]:
    """(teacher arm, control arm): the detect `est_*` ones, or the segment masters with `--segment`."""
    return (segment_arm_name(TEACHER_ARM), segment_arm_name(CONTROL_ARM)) if args.segment else (TEACHER_ARM, CONTROL_ARM)


def probe_labels(
    delta: list[torch.Tensor], bboxes: list[torch.Tensor], labels: list[torch.Tensor]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per level, (label, universe) boolean (N, H, W) maps.

    The universe is the fruit or damage cells; the label is 1 where `delta` is strictly above the median of
    the universe in that image, exactly the cells of the `nir_high` gate.
    """
    high = compute_gate("nir_high", nir_delta=delta, bboxes=bboxes, labels=labels)
    out = []
    for level_delta, level_high in zip(delta, high):
        masks = build_region_masks(bboxes, labels, *level_delta.shape[1:], device=level_delta.device)
        out.append((level_high > 0, masks["damage"] | masks["fruit"]))
    return out


def collect_cells(
    feat: torch.Tensor,
    label: torch.Tensor,
    universe: torch.Tensor,
    cap: int,
    generator: torch.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """(cells, label) of the universe cells of a (N, C, H, W) map, at most `cap` random cells per image."""
    xs, ys = [], []
    for i in range(feat.shape[0]):
        index = torch.nonzero(universe[i].flatten()).flatten()
        if index.numel() > cap:
            index = index[torch.randperm(index.numel(), generator=generator)[:cap].to(index.device)]
        xs.append(feat[i].flatten(1)[:, index].T.float().cpu().numpy())
        ys.append(label[i].flatten()[index].float().cpu().numpy())
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys)


def score(
    train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, seed: int
) -> tuple[float, float]:
    """(AUROC, AUROC with shuffled train labels) on val; NaN twice when either split has one class."""
    if len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
        return math.nan, math.nan
    auroc = probe_auroc(train_x, train_y, val_x, val_y)
    shuffled_y = train_y[np.random.RandomState(seed).permutation(len(train_y))]
    return auroc, probe_auroc(train_x, shuffled_y, val_x, val_y)


@torch.no_grad()
def collect_split(teacher, control, split_batches, cap: int, generator: torch.Generator) -> dict:
    """Per level, the stacked cells and labels of every batch."""
    stores = {name: {"x": [], "y": []} for name in LEVELS}
    for rgb, nir, bboxes, labels in split_batches:
        full = teacher(rgb, nir)["cls_preds"]
        ablated = teacher(rgb, torch.zeros_like(nir))["cls_preds"]
        maps = probe_labels(nir_dependence(full, ablated), bboxes, labels)
        student = control(rgb)
        for level, name in enumerate(LEVELS):
            feat = torch.cat([student["distill_fpn"][level], student["distill_head_cls"][level]], dim=1)
            label, universe = maps[level]
            x, y = collect_cells(feat, label, universe, cap, generator)
            stores[name]["x"].append(x)
            stores[name]["y"].append(y)
    return {name: (np.concatenate(s["x"]), np.concatenate(s["y"])) for name, s in stores.items()}


def probe_fold(teacher, control, train_batches, val_batches, seed: int, cap: int) -> dict:
    """{level: {auroc, shuffled_auroc, n_train, n_val, positive_rate_train}} for one fold; NaN becomes None."""
    train = collect_split(teacher, control, train_batches, cap, torch.Generator().manual_seed(seed))
    val = collect_split(teacher, control, val_batches, cap, torch.Generator().manual_seed(seed + 1))
    result = {}
    for name in LEVELS:
        (train_x, train_y), (val_x, val_y) = train[name], val[name]
        auroc, shuffled = score(train_x, train_y, val_x, val_y, seed)
        result[name] = {
            "auroc": None if math.isnan(auroc) else auroc,
            "shuffled_auroc": None if math.isnan(shuffled) else shuffled,
            "n_train": int(len(train_y)),
            "n_val": int(len(val_y)),
            "positive_rate_train": float(train_y.mean()) if len(train_y) else None,
        }
    return result


def mean_over_folds(folds: dict) -> dict:
    """Per level, the mean AUROC and shuffled AUROC over the folds where they are defined."""
    means = {}
    for name in LEVELS:
        means[name] = {}
        for key in ("auroc", "shuffled_auroc"):
            values = [f[name][key] for f in folds.values() if f[name][key] is not None]
            means[name][key] = float(np.mean(values)) if values else None
    return means


def render(folds: dict, meta: dict) -> str:
    def cell(entry: dict) -> str:
        if entry["auroc"] is None:
            return "n/a"
        return f"{entry['auroc']:.3f} / {entry['shuffled_auroc']:.3f}"

    lines = [
        "# KD NIR-dependence probe",
        "",
        "Generated by `scripts/kd_nir_probe.py`. Descriptive: can the RGB control student's features (FPN map and",
        "class-head stem, per cell) tell which fruit or damage cells the est_rgbnir teacher's damage score depends",
        "on NIR for (above the median, per image)? A logistic probe is fitted on the fold's train lots and scored on",
        "its val lot; the test split is never read. Cells are `AUROC / AUROC with shuffled train labels`.",
        "",
        f"- Seed: {meta.get('seed')}, student variant: {meta.get('variant')}",
        f"- Teacher: `{TEACHER_ARM}`, control student: `{CONTROL_ARM}`",
        "",
        "| Fold | " + " | ".join(LEVELS) + " |",
        "|:--|" + "--:|" * len(LEVELS),
    ]
    for fold, levels in folds.items():
        lines.append(f"| {fold} | " + " | ".join(cell(levels[name]) for name in LEVELS) + " |")
    means = mean_over_folds(folds)
    lines.append("| mean | " + " | ".join(cell(means[name]) for name in LEVELS) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(output: Path, folds: dict, meta: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "folds": folds, "mean": mean_over_folds(folds)}
    (output / "probe.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "README.md").write_text(render(folds, meta))


def fold_config(args, fold: int) -> KDConfig:
    """The data settings of a KD run of this fold, as `run_small_arms` plans them."""
    plan_args = rs.parse_args([
        "--dataset", str(args.dataset), "--checkpoints-root", str(args.checkpoints_root),
        "--variant", args.variant, "--machine", str(args.machine),
    ])
    plan_args.nir_as_rgb_dir = args.dataset.parent / "v4_nir_as_rgb_clahe"
    command = [str(x) for x in rs.plan_arm(plan_args, "kd_rgbnir", fold, args.seed)[0][1]]
    config = KDConfig.from_yaml(str(PROJECT_ROOT / rs.KD_CONFIG))
    apply_overrides(config, command[command.index("--override") + 1:])
    return config


def load_models(config: KDConfig, args, fold: int, device: torch.device):
    """Frozen est_rgbnir teacher and est_rgb control of this fold and seed (the segment masters with `--segment`)."""
    models = []
    teacher_arm, control_arm = arms_of(args)
    for arm, reads in ((teacher_arm, "rgbnir"), (control_arm, "rgb")):
        config.teacher_checkpoint = str(rk.run_path(args.checkpoints_root, arm, fold, args.seed) / rk.CHECKPOINT)
        config.teacher_input = reads
        models.append(KDTrainer._load_student_teacher(config).to(device).eval())
    return models


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    folds: dict = {}
    for fold in args.folds:
        config = fold_config(args, fold)
        teacher, control = load_models(config, args, fold, device)
        # Only train and val: the test split is never built here.
        train_ds, val_ds = dataset(config, "train"), dataset(config, "val")
        folds[str(fold)] = probe_fold(
            teacher, control,
            batches(train_ds, device, args.batch_size), batches(val_ds, device, args.batch_size),
            args.seed, args.cells_per_image,
        )
        print(f"fold {fold}: " + ", ".join(
            f"{n} {folds[str(fold)][n]['auroc']}" for n in LEVELS), flush=True)
    meta = {"seed": args.seed, "variant": args.variant, "folds": args.folds, "cells_per_image": args.cells_per_image}
    write_outputs(args.output, folds, meta)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
