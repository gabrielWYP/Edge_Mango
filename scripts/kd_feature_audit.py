#!/usr/bin/env python
"""Measure, per distillation level, what the teacher offers and what drives the KD loss.

For each teacher level (backbone S3-S4, FPN and head stems P3-P5):

1. Share of the imitation MSE by region (damage, fruit, background), with a
   trained KD student and its adapters.
2. Sensitivity to NIR: relative change of the feature when the NIR input is
   replaced by its normalized mean (zeros), by region.
3. Linear probe, damage vs healthy fruit cells: fitted on train, scored on val,
   for the teacher, the teacher without NIR and a student trained without KD.

Descriptive only: it never touches the test split.

    python scripts/kd_feature_audit.py --config configs/kd_training_twostream.yaml \
        --teacher <maestro.pt> --kd-student <kd.pt> --control-student <student.pt> \
        --out reports/kd-feature-audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.kd_audit import collect_probe_cells, probe_auroc, region_sums
from src.models.master.distill_projections import (
    backbone_projections,
    fpn_projections,
    head_projections,
)
from src.models.student.student_model import StudentModel, strip_legacy_kd_adapter_keys
from src.training.dataset import YOLODataset, collate_fn
from src.training.kd_config import KDConfig
from src.training.kd_loss import KD_REGIONS, build_region_masks
from src.training.kd_trainer import STUDENT_STRIDES, KDTrainer, select_by_strides

_NEGATIVES_PER_IMAGE = 40
_LEVEL_NAMES = {
    "backbone": ["S3", "S4"],
    "fpn": ["P3", "P4", "P5"],
    "head_cls": ["P3", "P4", "P5"],
    "head_reg": ["P3", "P4", "P5"],
}


def teacher_features(teacher, rgb, nir) -> dict[str, list[torch.Tensor]]:
    out = teacher(rgb, nir)
    strides = teacher.head_strides
    return {
        "backbone": out["distill_backbone"][2:],
        "fpn": select_by_strides(out["distill_fpn"], strides, STUDENT_STRIDES),
        "head_cls": select_by_strides(out["distill_head_cls"], strides, STUDENT_STRIDES),
        "head_reg": select_by_strides(out["distill_head_reg"], strides, STUDENT_STRIDES),
    }


def student_features(student, rgb) -> dict[str, list[torch.Tensor]]:
    out = student(rgb)
    return {
        "backbone": out["distill_backbone"],
        "fpn": out["distill_fpn"],
        "head_cls": out["distill_head_cls"],
        "head_reg": out["distill_head_reg"],
    }


def load_student(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    student = StudentModel(num_classes=2)
    student.load_state_dict(strip_legacy_kd_adapter_keys(checkpoint["model_state_dict"]))
    return student.to(device).eval(), checkpoint


def load_adapters(checkpoint: dict, device: torch.device) -> torch.nn.ModuleDict:
    adapters = torch.nn.ModuleDict(
        {
            "backbone": backbone_projections(),
            "fpn": fpn_projections(),
            "head_cls": head_projections(),
            "head_reg": head_projections(),
        }
    )
    adapters.load_state_dict(checkpoint["kd_adapters_state_dict"])
    return adapters.to(device).eval()


def dataset(config: KDConfig, split: str) -> YOLODataset:
    return YOLODataset(
        rgb_dir=config.rgb_dir,
        nir_dir=config.nir_dir,
        labels_dir=config.labels_dir,
        split=split,
        image_size=config.image_size,
        nir_mean=config.nir_mean,
        nir_std=config.nir_std,
        letterbox_value=config.letterbox_value,
        manifest_path=config.split_manifest,
        nir_homography_path=config.nir_homography_path,
        roi_crop=config.roi_crop,
        roi_crop_margin=config.roi_crop_margin,
        roi_crop_jitter=config.roi_crop_jitter,
    )


def batches(ds: YOLODataset, device: torch.device, batch_size: int = 2):
    for start in range(0, len(ds), batch_size):
        batch = collate_fn([ds[i] for i in range(start, min(start + batch_size, len(ds)))])
        yield (
            batch["rgb"].to(device),
            batch["nir"].to(device),
            [b.to(device) for b in batch["bboxes"]],
            [l.to(device) for l in batch["labels"]],
        )


def empty_probe_store() -> dict:
    return {"x": [], "y": []}


def to_arrays(store: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate(store["x"]), np.concatenate(store["y"])


@torch.no_grad()
def collect_probes(ds, teacher, control, device, seed) -> dict:
    generator = torch.Generator().manual_seed(seed)
    stores: dict = {}
    for rgb, nir, bboxes, labels in batches(ds, device):
        sources = {
            "teacher": teacher_features(teacher, rgb, nir),
            "teacher_no_nir": teacher_features(teacher, rgb, torch.zeros_like(nir)),
            "control_student": student_features(control, rgb),
        }
        for source, feats in sources.items():
            for level, tensors in feats.items():
                for name, feat in zip(_LEVEL_NAMES[level], tensors):
                    masks = build_region_masks(bboxes, labels, *feat.shape[2:], device=feat.device)
                    x, y = collect_probe_cells(feat, masks, _NEGATIVES_PER_IMAGE, generator)
                    store = stores.setdefault((source, level, name), empty_probe_store())
                    store["x"].append(x.astype(np.float32))
                    store["y"].append(y)
    return stores


@torch.no_grad()
def measure_regions(ds, teacher, kd_student, adapters, weights, device) -> dict:
    results: dict = {}
    for rgb, nir, bboxes, labels in batches(ds, device):
        with_nir = teacher_features(teacher, rgb, nir)
        without_nir = teacher_features(teacher, rgb, torch.zeros_like(nir))
        student = student_features(kd_student, rgb)
        for level in _LEVEL_NAMES:
            adapted = adapters[level](student[level])
            for name, t_feat, t0_feat, s_feat in zip(
                _LEVEL_NAMES[level], with_nir[level], without_nir[level], adapted
            ):
                masks = build_region_masks(bboxes, labels, *t_feat.shape[2:], device=t_feat.device)
                entry = results.setdefault(
                    (level, name),
                    {
                        kind: {r: [0.0, 0] for r in KD_REGIONS}
                        for kind in ("kd_error", "nir_change", "energy")
                    },
                )
                maps = {
                    "kd_error": (s_feat - t_feat).pow(2).mean(dim=1),
                    "nir_change": (t_feat - t0_feat).pow(2).mean(dim=1),
                    "energy": t_feat.pow(2).mean(dim=1),
                }
                for kind, per_cell in maps.items():
                    for region, (total, count) in region_sums(per_cell, masks).items():
                        entry[kind][region][0] += total
                        entry[kind][region][1] += count

    report: dict = {}
    for (level, name), entry in results.items():
        cells = sum(entry["kd_error"][r][1] for r in KD_REGIONS)
        error = sum(entry["kd_error"][r][0] for r in KD_REGIONS)
        report[f"{level}/{name}"] = {
            "level": level,
            "kd_mse": error / cells,
            "kd_error_share": {r: entry["kd_error"][r][0] / error for r in KD_REGIONS},
            "cell_share": {r: entry["kd_error"][r][1] / cells for r in KD_REGIONS},
            "nir_sensitivity": {
                r: entry["nir_change"][r][0] / max(entry["energy"][r][0], 1e-12) for r in KD_REGIONS
            },
        }

    per_group = {level: [] for level in _LEVEL_NAMES}
    for entry in report.values():
        per_group[entry["level"]].append(entry["kd_mse"])
    weighted = {level: weights[level] * float(np.mean(v)) for level, v in per_group.items()}
    total = sum(weighted.values())
    return {
        "levels": report,
        "weighted_kd_share": {level: value / total for level, value in weighted.items()},
    }


def kd_loss_name(checkpoint: dict) -> str:
    config = checkpoint.get("config") or {}
    config = config if isinstance(config, dict) else vars(config)
    return str(config.get("kd_loss_type") or config.get("kd_feature_norm") or "mse")


def render(result: dict, meta: dict) -> str:
    lines = [
        "# KD feature audit",
        "",
        "Generated by `scripts/kd_feature_audit.py`. Descriptive and exploratory: preliminary corpus split",
        "by pair (not by mango), a single seed, and NIR ablated by feeding its normalized mean.",
        "",
        f"- Teacher: `{meta['teacher']}`",
        f"- KD student: `{meta['kd_student']}`",
        f"- Control student: `{meta['control_student']}`",
        f"- Images: {meta['train_images']} train, {meta['val_images']} val",
        f"- KD student's training loss: `{meta['kd_student_loss']}`. The shares below use raw MSE, so with a",
        "  normalized training loss they describe the raw feature scales, not what that run minimized.",
        "",
        "## Weighted share of the KD loss by level group",
        "",
        "| Group | Weight | Share |",
        "|:--|--:|--:|",
    ]
    for level, share in result["regions"]["weighted_kd_share"].items():
        lines.append(f"| {level} | {meta['weights'][level]} | {share:.1%} |")
    lines += [
        "",
        "## Per level: where the MSE comes from and what depends on NIR",
        "",
        "`NIR sens.` is the feature's mean squared change without NIR over its mean squared value.",
        "",
        "| Level | Damage cells | Damage error | Fruit error | Background error | NIR sens. damage | NIR sens. fruit | NIR sens. background |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for key, entry in result["regions"]["levels"].items():
        share, sens = entry["kd_error_share"], entry["nir_sensitivity"]
        lines.append(
            f"| {key} | {entry['cell_share']['damage']:.2%} | {share['damage']:.2%} | "
            f"{share['fruit']:.1%} | {share['background']:.1%} | "
            f"{sens['damage']:.3f} | {sens['fruit']:.3f} | {sens['background']:.3f} |"
        )
    lines += [
        "",
        "## Probe AUROC, damage vs healthy fruit cells (fit on train, score on val)",
        "",
        "| Level | Teacher | Teacher without NIR | NIR gain | Control student | Teacher minus student |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for key, entry in result["probes"].items():
        lines.append(
            f"| {key} | {entry['teacher']:.3f} | {entry['teacher_no_nir']:.3f} | "
            f"{entry['teacher'] - entry['teacher_no_nir']:+.3f} | {entry['control_student']:.3f} | "
            f"{entry['teacher'] - entry['control_student']:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--kd-student", required=True)
    parser.add_argument("--control-student", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = KDConfig.from_yaml(args.config)
    config.teacher_checkpoint = args.teacher

    teacher = KDTrainer._load_teacher(config).to(device).eval()
    kd_student, kd_checkpoint = load_student(args.kd_student, device)
    control, _ = load_student(args.control_student, device)
    adapters = load_adapters(kd_checkpoint, device)

    train_ds, val_ds = dataset(config, "train"), dataset(config, "val")
    regions = measure_regions(val_ds, teacher, kd_student, adapters, config.distill_levels, device)

    train_probes = collect_probes(train_ds, teacher, control, device, args.seed)
    val_probes = collect_probes(val_ds, teacher, control, device, args.seed)
    probes: dict = {}
    for (source, level, name), store in train_probes.items():
        train_x, train_y = to_arrays(store)
        val_x, val_y = to_arrays(val_probes[(source, level, name)])
        probes.setdefault(f"{level}/{name}", {})[source] = probe_auroc(train_x, train_y, val_x, val_y)
        print(f"{source} {level}/{name}: {probes[f'{level}/{name}'][source]:.3f}", flush=True)

    result = {"regions": regions, "probes": probes}
    meta = {
        "teacher": args.teacher,
        "kd_student_loss": kd_loss_name(kd_checkpoint),
        "kd_student": args.kd_student,
        "control_student": args.control_student,
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "weights": config.distill_levels,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"meta": meta, **result}, indent=2))
    (out / "README.md").write_text(render(result, meta))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
