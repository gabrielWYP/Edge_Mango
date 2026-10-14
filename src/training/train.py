"""
CLI entry point for MasterModel training.

Usage:
    python -m src.training.train --config configs/training.yaml
    python -m src.training.train --config configs/training.yaml --override batch_size=4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import TrainingConfig
from src.training.dataset import YOLODataset, build_dataloader, collate_fn, build_weighted_sampler
from src.training.loop import Trainer
from src.training.machine import apply_machine_profile, experiment_sha256, load_machine_profile
from src.training.run_artifacts import (
    build_versioned_stage_plan,
    collect_stage_results,
    finalize_stage,
    write_run_summary,
)
from src.models.master.master_model import MasterModel
from src.models.student.student_model import StudentModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MasterModel / StudentModel Training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["master", "student"],
        default=None,
        help="Model type to train: master or student.",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default=None,
        help=(
            "Path to a machine-profile YAML (configs/machines/*.yaml) setting "
            "device/batch_size/num_workers/pin_memory/precision. Applied after "
            "--config, before --override. A profile may set only those five "
            "keys — anything else raises (fusion-redesign D-H)."
        ),
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values: key=value pairs.",
    )
    parser.add_argument(
        "--versioned-run",
        action="store_true",
        help="Write this run using the timestamped final_runs artifact layout.",
    )
    parser.add_argument(
        "--run-root",
        default="checkpoints/final_runs",
        help="Root directory for --versioned-run outputs.",
    )
    parser.add_argument(
        "--run-timestamp",
        default=None,
        help="Optional timestamp/run ID for --versioned-run.",
    )
    parser.add_argument(
        "--stage",
        choices=["maestro", "estudiante"],
        default=None,
        help="Optional stage name for --versioned-run. Defaults from model_type.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Without a root handler, INFO records from library modules are dropped,
    # including the per-sample augmentation bbox-drop summaries
    # (src/training/augmentations.py) that show how many damage boxes the
    # spatial transforms discard. Configured here, not at import time, so
    # importing this module never reconfigures a caller's logging.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load config
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()

    # experiment_sha256 (D-H): hash of the experiment file's raw bytes, so
    # two runs are comparable only if this matches (alongside
    # effective_batch). Computed before --machine/--override touch the
    # in-memory config — it hashes the file, not the resolved config.
    run_experiment_sha256 = experiment_sha256(args.config) if args.config else None

    # Machine profile (D-H): device/batch_size/num_workers/pin_memory/
    # precision only — whitelist-enforced so a profile cannot alter the
    # experiment being measured.
    if args.machine:
        machine_profile = load_machine_profile(args.machine)
        apply_machine_profile(config, machine_profile)

    # Apply overrides
    for override in args.override:
        key, value = override.split("=", 1)
        # Type coercion
        if hasattr(config, key):
            current = getattr(config, key)
            if isinstance(current, bool):
                value = value.lower() in ("true", "1", "yes")
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            elif isinstance(current, list):
                # List coercion (ported from evaluate_checkpoint.py's
                # _apply_overrides so the two entrypoints cannot drift):
                # `--override head_strides=[4,8,16,32]` used to assign the
                # literal string "[4,8,16,32]" (len()==12), silently
                # producing a wrong-but-plausible pyramid instead of the
                # intended 4-level one. `[...]` parses as JSON (preserving
                # int/float as written); comma-separated values fall back
                # to floats.
                value = json.loads(value) if value.strip().startswith("[") else [
                    float(v) for v in value.split(",")
                ]
            setattr(config, key, value)

    # CLI --model flag overrides config model_type
    if args.model is not None:
        config.model_type = args.model

    versioned_plan = None
    if args.versioned_run:
        default_stage = "estudiante" if config.model_type == "student" else "maestro"
        stage_key = args.stage or default_stage
        if stage_key != default_stage:
            raise ValueError(
                f"Stage '{stage_key}' is inconsistent with model_type='{config.model_type}'. "
                f"Use stage '{default_stage}' or change --model."
            )
        versioned_plan = build_versioned_stage_plan(
            output_root=args.run_root,
            stage_key=stage_key,
            run_id=args.run_timestamp,
        )
        if versioned_plan.tmp_dir.exists():
            raise FileExistsError(f"Partial run directory already exists: {versioned_plan.tmp_dir}")
        config.output_dir = str(versioned_plan.tmp_dir)
        print(f"Versioned run: {versioned_plan.run_dir}")
        print(f"  Stage: {versioned_plan.stage_key}")
        print(f"  Temporary output: {versioned_plan.tmp_dir}")

    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print(f"Config: model_type={config.model_type}, backbone={config.backbone_variant}, image_size={config.image_size}")
    if config.model_type == "student":
        print(f"  Single-phase: {config.epochs} epochs, lr={config.lr}")
    else:
        print(f"  Phase 1: {config.epochs_phase1} epochs, lr={config.lr_phase1}")
        print(f"  Phase 2: {config.epochs_phase2} epochs, lr={config.lr_phase2}")
    print(
        f"  Batch size: {config.batch_size}, Precision: {config.precision}, "
        f"Device: {config.device}, effective_batch: {config.effective_batch}"
    )
    if run_experiment_sha256:
        print(f"  experiment_sha256: {run_experiment_sha256}")

    # Create datasets. Do not pass explicit transforms here: YOLODataset's
    # default path keeps spatial augmentations synchronized across RGB, NIR,
    # and bounding boxes.
    train_dataset = YOLODataset(
        rgb_dir=config.rgb_dir,
        nir_dir=config.nir_dir,
        labels_dir=config.labels_dir,
        split="train",
        image_size=config.image_size,
        nir_mean=config.nir_mean,
        nir_std=config.nir_std,
        letterbox_value=config.letterbox_value,
        manifest_path=config.split_manifest,
        nir_homography_path=config.nir_homography_path,
    )

    val_dataset = YOLODataset(
        rgb_dir=config.rgb_dir,
        nir_dir=config.nir_dir,
        labels_dir=config.labels_dir,
        split="val",
        image_size=config.image_size,
        nir_mean=config.nir_mean,
        nir_std=config.nir_std,
        letterbox_value=config.letterbox_value,
        manifest_path=config.split_manifest,
        nir_homography_path=config.nir_homography_path,
    )

    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val images")

    # DataLoaders (with graceful fallback to num_workers=0 if IPC sockets fail)
    train_sampler = build_weighted_sampler(train_dataset)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # Create model
    if config.model_type == "student":
        model = StudentModel(num_classes=config.num_classes)
        params = model.count_parameters()
        print(f"StudentModel parameters: {params['total']:,} total")
    else:
        model = MasterModel(
            num_classes=config.num_classes,
            pretrained_backbone=True,
            backbone_variant=config.backbone_variant,
            head_strides=config.head_strides,
            in_channels=config.in_channels,
            fusion_mode=config.fusion_mode,
            fusion_pos_encoding=config.fusion_pos_encoding,
        )
        params = model.count_parameters()
        print(f"Model parameters: {params['total']:,} total, {params['backbone']:,} backbone")

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    trainer.experiment_sha256 = run_experiment_sha256

    results = trainer.fit()

    final_checkpoint = results["checkpoint"]
    if versioned_plan is not None:
        stage_result = finalize_stage(
            tmp_dir=versioned_plan.tmp_dir,
            final_parent=versioned_plan.final_parent,
            run_id=versioned_plan.run_id,
            command=[sys.executable, *sys.argv],
            stage_key=versioned_plan.stage_key,
            stage_label=versioned_plan.stage_label,
            experiment_sha256=run_experiment_sha256,
            effective_batch=config.effective_batch,
            precision=config.precision,
            device=config.device,
        )
        run_results = collect_stage_results(versioned_plan.run_dir)
        write_run_summary(versioned_plan.run_dir, versioned_plan.run_id, run_results)
        final_checkpoint = stage_result.checkpoint

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best mAP@0.5: {results['best_map50']:.4f}")
    print(f"  Checkpoint: {final_checkpoint}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
