#!/usr/bin/env python3
"""Eval-only entrypoint for an existing checkpoint (D10).

Runs `Trainer.evaluate()` against a chosen split with no training loop and
no `train_loader` dependency, so a checkpoint can be re-scored under a
different decode configuration (NMS on/off, per-class decode on/off,
assigner center radius, ...) without retraining. This is the tool the
damage-map-audit experiment ladder (E0-E3) needs: every inference-only rung
runs through this script.

Also supports `--assigner-stats` (A0/A4): a forward-only pass (no backward,
no optimizer step) over the chosen split that accumulates per-class,
per-level positive-anchor counts via `YOLOv8Loss`'s non-destructive
instrumentation, written to `<output-dir>/assigner_stats.csv`.

Usage:
    # Step A — NMS-corrected baseline re-evaluation (no training).
    python scripts/evaluate_checkpoint.py --config configs/training_student.yaml \\
        --checkpoint checkpoints/student/best_model.pt --split val \\
        --override nms_enabled=true decode_per_class=true

    # Step A0 / E2 — assigner instrumentation on an existing checkpoint.
    python scripts/evaluate_checkpoint.py --config configs/training_student.yaml \\
        --checkpoint checkpoints/student/best_model.pt --split train \\
        --assigner-stats --override assigner_center_radius=0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.master.master_model import MasterModel
from src.models.student.student_model import StudentModel, strip_legacy_kd_adapter_keys
from src.training.config import TrainingConfig
from src.training.dataset import YOLODataset, build_dataloader, collate_fn
from src.training.fusion_modes import (
    FUSION_MODE_CROSS_ATTENTION,
    accepted_arch_versions,
    fusion_mode_for_arch_version,
    fusion_pos_encoding_from_checkpoint,
)
from src.training.loop import Trainer
from src.training.strides import (
    CROSS_ATTENTION_HEAD_STRIDES,
    resolve_active_strides,
    resolve_from_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing checkpoint without retraining."
    )
    parser.add_argument("--config", required=True, type=str, help="Path to YAML training config.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to .pt checkpoint file.")
    parser.add_argument(
        "--model", type=str, choices=["master", "student"], default=None,
        help="Model architecture. Defaults to the config's model_type.",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: val).",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override config values: key=value pairs (same syntax as train.py).",
    )
    parser.add_argument(
        "--assigner-stats", action="store_true",
        help=(
            "Run a forward-only pass collecting per-class, per-level "
            "positive-anchor instrumentation (A0/A4/E2) instead of computing mAP."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports/damage-map-audit/eval",
        help="Directory to write metrics.json / assigner_stats.csv.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cuda or cpu (default: auto-detect).",
    )
    return parser.parse_args()


def _apply_overrides(config: TrainingConfig, overrides: list[str]) -> None:
    """Apply `key=value` overrides in place, matching train.py's coercion rules."""
    for override in overrides:
        key, value = override.split("=", 1)
        if not hasattr(config, key):
            logger.warning("Ignoring unknown override key: %s", key)
            continue
        current = getattr(config, key)
        if isinstance(current, bool):
            value = value.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, list):
            value = json.loads(value) if value.strip().startswith("[") else [
                float(v) for v in value.split(",")
            ]
        setattr(config, key, value)


def _check_arch_version(checkpoint_path: str, arch_version, fusion_mode: str) -> None:
    """Raise unless `arch_version` belongs to `fusion_mode`'s architecture.

    `arch_version` 2 is early fusion, 1 (and the untagged `None` written
    before the tag existed) is the two-stream cross-attention path. Loading
    either as the other is always wrong, and always caught here rather than
    by `load_state_dict`.
    """
    accepted = accepted_arch_versions(fusion_mode)
    if arch_version in accepted:
        return
    checkpoint_mode = fusion_mode_for_arch_version(arch_version)
    origin = (
        f"which is a fusion_mode='{checkpoint_mode}' checkpoint"
        if checkpoint_mode is not None
        else "which belongs to no known architecture"
    )
    raise ValueError(
        f"Checkpoint at {checkpoint_path} has arch_version={arch_version!r}, "
        f"{origin}, but the config selects fusion_mode='{fusion_mode}' "
        f"(accepts arch_version in {sorted(accepted, key=str)}). "
        "Architecture v1 checkpoints (dual-stream cross-attention fusion) "
        "are not loadable by MasterModel v2 (early fusion) or vice versa — "
        "run with the matching config (e.g. configs/experiment/twostream.yaml "
        "for a v1 checkpoint). See openspec/changes/fusion-redesign."
    )


def _resolve_head_strides(checkpoint: dict, fusion_mode: str) -> list[int]:
    """Return the strides to rebuild the checkpoint's MasterModel with.

    `fusion_mode="cross_attention"` is no longer fixed at one pyramid —
    `DualFPN` emits either `CROSS_ATTENTION_HEAD_STRIDES` or the full
    4-level pyramid — so the checkpoint's own recorded strides are read
    first, exactly as on the early path. Only when there are none does it
    fall back to `CROSS_ATTENTION_HEAD_STRIDES`, which is what keeps pre-tag
    checkpoints (written before `config.head_strides` was persisted, and
    3-level by construction) loadable at all.
    `fusion_mode="early"` still resolves strides from the checkpoint, and
    still fails loudly if it recorded none.
    """
    if fusion_mode == FUSION_MODE_CROSS_ATTENTION:
        try:
            return resolve_from_checkpoint(checkpoint)
        except KeyError:
            return list(CROSS_ATTENTION_HEAD_STRIDES)
    return resolve_from_checkpoint(checkpoint)


def _resolve_fusion_pos_encoding(checkpoint: dict, config: TrainingConfig) -> bool:
    """Return the `fusion_pos_encoding` to rebuild the checkpoint's
    MasterModel with.

    The encoding adds no state_dict key, so `strict=True` cannot catch a
    mismatch — the checkpoint's recorded config is the source of truth
    (absent -> False for checkpoints that predate the field), and a
    disagreeing caller config is overridden with a warning rather than
    silently evaluating a different forward pass.
    """
    resolved = fusion_pos_encoding_from_checkpoint(checkpoint)
    if resolved != config.fusion_pos_encoding:
        logger.warning(
            "Checkpoint records fusion_pos_encoding=%s but the config says %s; "
            "using the checkpoint's value.",
            resolved, config.fusion_pos_encoding,
        )
    return resolved


def _load_model(model_type: str, checkpoint_path: str, config: TrainingConfig, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    is_checkpoint_dict = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint

    head_strides = None
    fusion_pos_encoding = config.fusion_pos_encoding
    if is_checkpoint_dict:
        arch_version = checkpoint.get("arch_version")
        if model_type == "master":
            # fusion-redesign D-C: the two fusion modes share no
            # state_dict key upstream of the head, so a checkpoint of one
            # is unloadable into the other.
            # Check `arch_version` against the mode the caller's config
            # selected, BEFORE `load_state_dict`, so the failure is one
            # actionable sentence instead of a 200-line
            # missing/unexpected-key dump — and so a two-stream checkpoint
            # can never be silently evaluated as an early-fusion one.
            _check_arch_version(checkpoint_path, arch_version, config.fusion_mode)
            head_strides = _resolve_head_strides(checkpoint, config.fusion_mode)
            fusion_pos_encoding = _resolve_fusion_pos_encoding(checkpoint, config)
        state_dict = checkpoint["model_state_dict"]
        logger.info(
            "Loaded checkpoint (epoch=%s, best_map50=%s)",
            checkpoint.get("epoch", "?"), checkpoint.get("best_map50", "?"),
        )
    else:
        state_dict = checkpoint
        logger.info("Loaded raw state_dict.")

    if model_type == "master":
        model = MasterModel(
            num_classes=config.num_classes,
            pretrained_backbone=False,
            backbone_variant=config.backbone_variant,
            head_strides=head_strides,
            in_channels=config.in_channels,
            fusion_mode=config.fusion_mode,
            fusion_pos_encoding=fusion_pos_encoding,
        )
    else:
        model = StudentModel(num_classes=config.num_classes)
        # Pre-fix KD checkpoints stored the training-only adapters inside
        # `model_state_dict`; they are not part of the student.
        state_dict = strip_legacy_kd_adapter_keys(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _run_assigner_stats(
    model, config: TrainingConfig, loader, device: torch.device, output_dir: Path,
) -> dict:
    """A0/E2: forward-only pass accumulating per-class, per-level positive
    anchor counts. No backward pass, no optimizer step, no weight update."""
    from src.training.loss import YOLOv8Loss

    criterion = YOLOv8Loss(
        num_classes=config.num_classes,
        box_weight=config.box_weight,
        cls_weight=config.cls_weight,
        class_weights=config.class_weights,
        strides=resolve_active_strides(config),
        assigner_center_radius=config.assigner_center_radius,
        assigner_level_ranges=config.assigner_level_ranges,
        assigner_collect_stats=True,
    ).to(device)

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            nir = batch["nir"].to(device)
            bboxes = [b.to(device) for b in batch["bboxes"]]
            labels = [l.to(device) for l in batch["labels"]]

            if config.model_type == "student":
                output = model(rgb)
            else:
                output = model(rgb, nir)

            targets = {"bboxes": bboxes, "labels": labels}
            criterion(output["preds"], targets)  # forward only; stats accumulate as a side effect

    stats_path = output_dir / "assigner_stats.csv"
    criterion.write_assigner_stats_csv(stats_path)
    stats = criterion.get_assigner_stats()
    logger.info("Assigner stats written to %s", stats_path)

    # Median positive anchors per GT instance, per class (E2 bar: damage <1.0
    # vs mango >=5.0 pre-fix; see proposal.md falsifiable criteria).
    class_counts = _gt_instance_counts(loader.dataset)
    per_class_positives: dict[int, list[int]] = {}
    for (cls_id, _level), count in stats.items():
        per_class_positives.setdefault(cls_id, []).append(count)

    summary = {}
    for cls_id, counts in per_class_positives.items():
        total_positives = sum(counts)
        n_gt = class_counts.get(cls_id, 0)
        summary[cls_id] = {
            "total_positive_anchors": total_positives,
            "gt_instances": n_gt,
            "mean_positives_per_gt": (total_positives / n_gt) if n_gt else None,
        }
        logger.info(
            "class %d: %d positive anchors across %d GT instances (mean %.3f/GT)",
            cls_id, total_positives, n_gt,
            (total_positives / n_gt) if n_gt else float("nan"),
        )

    return {"per_class_level_counts": {f"{k[0]}_{k[1]}": v for k, v in stats.items()}, "summary": summary}


def _gt_instance_counts(dataset: YOLODataset) -> dict[int, int]:
    return dataset.get_class_counts()


def main() -> int:
    args = parse_args()

    config = TrainingConfig.from_yaml(args.config)
    _apply_overrides(config, args.override)

    model_type = args.model or config.model_type
    config.model_type = model_type

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Device: %s", device)
    logger.info(
        "Config: model=%s split=%s conf_threshold=%.3f eval_conf_threshold=%.3f nms_enabled=%s decode_per_class=%s "
        "assigner_center_radius=%.2f",
        model_type, args.split, config.conf_threshold, config.eval_conf_threshold, config.nms_enabled,
        config.decode_per_class, config.assigner_center_radius,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(model_type, args.checkpoint, config, device)
    if model_type == "master":
        # Keep config.head_strides consistent with whatever the checkpoint
        # actually used (fusion-redesign D-C/D-D) — the Trainer's criterion
        # and decode both derive their strides from config, not from the
        # model instance, so a mismatch here would silently misalign
        # predictions against the wrong anchor grid.
        config.head_strides = model.head_strides

    dataset = YOLODataset(
        rgb_dir=config.rgb_dir,
        nir_dir=config.nir_dir,
        labels_dir=config.labels_dir,
        split=args.split,
        image_size=config.image_size,
        nir_mean=config.nir_mean,
        nir_std=config.nir_std,
        letterbox_value=config.letterbox_value,
        manifest_path=config.split_manifest,
        nir_homography_path=config.nir_homography_path,
    )
    logger.info("Split '%s': %d images", args.split, len(dataset))

    loader = build_dataloader(
        dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True,
    )

    if args.assigner_stats:
        result = _run_assigner_stats(model, config, loader, device, output_dir)
        (output_dir / "assigner_stats_summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result["summary"], indent=2))
        return 0

    trainer = Trainer(model=model, config=config, train_loader=None, val_loader=loader)
    metrics = trainer.evaluate()

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    logger.info("Metrics written to %s", metrics_path)

    print(f"mAP@0.5:      {metrics['map50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['map_50_95']:.4f}")
    for cls_id, ap in sorted(metrics["per_class_ap_50"].items()):
        precision = metrics["per_class_precision"].get(cls_id, 0.0)
        recall = metrics["per_class_recall"].get(cls_id, 0.0)
        counts = metrics["per_class_counts"].get(cls_id, {})
        print(
            f"  class {cls_id}: AP50={ap:.4f} precision={precision:.4f} recall={recall:.4f} "
            f"tp={counts.get('tp', 0)} fp={counts.get('fp', 0)} fn={counts.get('fn', 0)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
