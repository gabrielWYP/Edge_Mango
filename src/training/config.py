"""
Training configuration for MasterModel fine-tuning.

All hyperparameters for the 2-phase training pipeline are defined here.
Supports loading from YAML files for experiment management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .machine import derive_grad_accum_steps
from .precision import validate_bf16_support, validate_precision
from .strides import DEFAULT_HEAD_STRIDES, validate_strides


@dataclass
class TrainingConfig:
    """Configuration for MasterModel 2-phase training.

    Attributes:
        rgb_dir: Path to RGB images directory.
        nir_dir: Path to NIR images directory.
        labels_dir: Path to YOLO label files root (contains train/, val/).
        output_dir: Where to save checkpoints and logs.
        split_manifest: Path to the split manifest (splits.json) used to
            fail-fast if a split's directory contents diverge from it. Set to
            None to skip this guard (e.g., synthetic/unsplit test fixtures).
        backbone_variant: ConvNeXt variant ("tiny" or "small").
        num_classes: Number of detection classes.
        image_size: Input image size (square).
        letterbox_value: Padding pixel value for letterbox.
        batch_size: Training batch size.
        epochs_phase1: Epochs for Phase 1 (frozen backbone).
        epochs_phase2: Epochs for Phase 2 (partial unfreeze).
        lr_phase1: Learning rate for Phase 1.
        lr_phase2: Learning rate for Phase 2.
        weight_decay: AdamW weight decay.
        box_weight: Regression loss weight.
        cls_weight: Classification loss weight.
        class_weights: Per-class BCE weights [mango, danado].
        warmup_epochs: Linear warmup epochs per phase.
        patience: Early stopping patience (epochs without mAP improvement).
            `0` or less disables early stopping and runs the full epoch
            budget. On a small validation split the per-epoch mAP is mostly
            sampling noise, so patience stops on noise rather than on
            convergence; the best-mAP checkpoint is kept either way. See
            `Trainer._should_stop_early`.
        log_interval: Steps between TensorBoard logs.
        save_interval: Epochs between periodic checkpoints.
        precision: One of "fp32" | "fp16" | "bf16". Replaces the old
            `amp: bool = True` flag, which silently enabled fp16 in any
            config omitting the key. Defaults to "fp32". "bf16" requires an
            Ampere-or-newer CUDA device and is validated at load time.
        grad_clip: Gradient clipping max norm.
        num_workers: DataLoader workers.
        nir_mean: NIR channel normalization mean.
        nir_std: NIR channel normalization std.
        seed: Random seed.
        device: "auto" | "cuda" | "cpu" | an explicit `torch.device` string
            (e.g. "cuda:0"). "auto" resolves to CUDA if available, else CPU
            — the same expression the trainer used to hardcode.
        pin_memory: DataLoader `pin_memory`. A machine-profile key (D-H).
        effective_batch: The experimental constant. Physical `batch_size` is
            a memory knob; `effective_batch` is what every run must match
            for results to be comparable (alongside `experiment_sha256`).
            `grad_accum_steps = effective_batch // batch_size` is always
            derived, never itself a config field.
        in_channels: MasterModel backbone stem input channels. 4 (default)
            for the early-fused RGB+NIR input; 3 for the RGB-only H-D
            control arm (fusion-redesign). Ignored for model_type="student".
        head_strides: MasterModel FPN/head pyramid strides, finest-first.
            Default `[4, 8, 16, 32]` includes the reconnected P2 level
            (fusion-redesign D-3). The single source of truth for
            `FPNNeck.emit_levels`, `YOLODetectionHead`'s level count,
            `YOLOv8Loss.strides`, and `decode_detections(strides=...)` — see
            `src/training/strides.py`. Ignored for model_type="student"
            (the student backbone is unaffected by this redesign and always
            uses 3 fixed levels at `[8, 16, 32]`).
        schedule: "two_phase" | "end_to_end". MasterModel training schedule
            (fusion-redesign D-4). "end_to_end" (default) trains every
            backbone stage, the stem, the neck and the head from epoch 0,
            with pretrained stages at `backbone_lr_mult` times the
            neck/head learning rate. "two_phase" preserves the previous
            frozen-then-partially-unfrozen schedule for configs that
            explicitly opt back into it (e.g. `training_mango.yaml`).
            Ignored for model_type="student" (always single-phase).
        backbone_lr_mult: Discriminative LR multiplier applied to pretrained
            backbone-stage parameters under `schedule="end_to_end"`. The
            stem is intentionally NOT in this group — it is 4-channel and
            out of ImageNet distribution by construction, so it trains at
            the full (non-discriminated) rate alongside the neck and head.
    """

    # Paths
    rgb_dir: str = "data/cache/mango/rgb"
    nir_dir: str = "data/cache/mango/nir"
    labels_dir: str = "data/annotations/yolo/labels"
    output_dir: str = "checkpoints/mastermodel"
    split_manifest: str | None = "data/annotations/yolo/splits.json"

    # Model
    backbone_variant: str = "tiny"
    num_classes: int = 2

    # Augmentation
    image_size: int = 640
    letterbox_value: int = 114

    # Training
    batch_size: int = 2
    epochs_phase1: int = 50
    epochs_phase2: int = 30
    lr_phase1: float = 1e-3
    lr_phase2: float = 1e-4
    weight_decay: float = 5e-4

    # Loss
    box_weight: float = 7.5
    cls_weight: float = 0.5
    # [mango, damage]. `[0.5, 1.5]` is the value every recorded checkpoint was
    # trained under; held constant for the duration of the damage-map-audit
    # experiment ladder so a class-1 AP change can be attributed to the
    # assigner fix rather than a reweighting. See
    # openspec/specs/yolo-loss/spec.md and proposal.md Q2 for the full
    # inverse-frequency-vs-experimental-constant rationale.
    class_weights: list[float] = field(default_factory=lambda: [0.5, 1.5])

    # Decode (src/training/decode.py::decode_detections) — see training-loop spec.
    conf_threshold: float = 0.25
    nms_iou_threshold: float = 0.5
    nms_enabled: bool = True
    decode_per_class: bool = True
    max_detections: int = 300

    # Task-Aligned Assigner (src/training/loss.py) — see yolo-loss spec.
    # 0.0 = legacy strict anchor-center-inside-GT containment (default, D9):
    # ships off so the pre-fix assignment behavior is measurable (E2) before
    # any run opts into center-sampling via a nonzero radius (stride units).
    assigner_center_radius: float = 0.0
    # Per-level GT-size admissibility bins. fusion-redesign D-3/D-D default:
    # max(w,h)<32 -> P2/stride-4, <64 -> P3/stride-8, <128 -> P4/stride-16,
    # else -> P5/stride-32 (pixel space at image_size=640). Must always have
    # exactly len(head_strides) - 1 entries — enforced in __post_init__.
    assigner_level_ranges: list[float] = field(default_factory=lambda: [32.0, 64.0, 128.0])
    # Non-destructive per-class/per-level positive-anchor instrumentation
    # (A4). Off by default; must not alter target_classes/bboxes/scores/fg_mask.
    assigner_collect_stats: bool = False

    # Scheduler
    warmup_epochs: int = 3
    patience: int = 15

    # Logging
    log_interval: int = 10
    save_interval: int = 5

    # Mixed precision — "fp32" | "fp16" | "bf16". See `precision` docstring
    # above; `amp` is removed, not deprecated (fusion-redesign D-I).
    precision: str = "fp32"

    # Gradient clipping
    grad_clip: float = 10.0

    # DataLoader
    num_workers: int = 4

    # NIR normalization stats (computed from dataset)
    nir_mean: float = 0.0569
    nir_std: float = 0.0546

    # Model type dispatch
    model_type: str = "master"  # "master" | "student"

    # Single-phase training (student)
    epochs: int = 50
    lr: float = 1e-3

    # Reproducibility
    seed: int = 42

    # Hardware portability (W9, fusion-redesign D-H) — machine-profile
    # fields. Kept identical in meaning to a machine profile's own keys so
    # `machine.apply_machine_profile` can `setattr` them directly.
    device: str = "auto"
    pin_memory: bool = True
    # The experimental constant (Q7, fusion-redesign): must be fixed before
    # a validation ladder starts and held constant across every rung. Not
    # itself read by the model; only `grad_accum_steps` (derived) matters
    # to the training loop.
    effective_batch: int = 8

    # Architecture (W1-W5, fusion-redesign) — MasterModel only; ignored for
    # model_type="student".
    in_channels: int = 4
    head_strides: list[int] = field(default_factory=lambda: list(DEFAULT_HEAD_STRIDES))

    # Training schedule (D-4, fusion-redesign) — MasterModel only.
    schedule: str = "end_to_end"
    backbone_lr_mult: float = 0.1

    def __post_init__(self) -> None:
        valid_types = {"master", "student"}
        if self.model_type not in valid_types:
            raise ValueError(
                f"Invalid model_type '{self.model_type}'. "
                f"Must be one of: {valid_types}"
            )
        validate_precision(self.precision)
        if self.precision == "bf16":
            validate_bf16_support()
        # Fails fast: an effective_batch/batch_size pair that does not
        # divide evenly would mean the recorded effective_batch never
        # matched what actually ran (fusion-redesign D-H). The returned
        # value is intentionally discarded here — grad_accum_steps is
        # derived on demand by the trainer, never stored as a config field.
        derive_grad_accum_steps(self.effective_batch, self.batch_size)
        valid_schedules = {"two_phase", "end_to_end"}
        if self.schedule not in valid_schedules:
            raise ValueError(
                f"Invalid schedule '{self.schedule}'. Must be one of: {valid_schedules}"
            )
        # fusion-redesign D-D: the one genuinely silent failure in this area
        # — a shorter assigner_level_ranges leaves the coarsest pyramid
        # level with zero positive assignments, with no exception. Skipped
        # for model_type="student": the student backbone is unaffected by
        # this redesign and always uses its own fixed 3-level [8, 16, 32]
        # strides, so head_strides/assigner_level_ranges do not apply to it.
        if self.model_type == "master":
            validate_strides(self.head_strides, self.assigner_level_ranges)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            TrainingConfig instance with values from the file.
        """
        path = Path(path)
        with open(path, "r") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                {k: v for k, v in self.__dict__.items()},
                f,
                default_flow_style=False,
            )

    @property
    def total_epochs(self) -> int:
        # model_type="student": always single-phase, `epochs` epochs.
        # model_type="master", schedule="end_to_end" (fusion-redesign D-4
        # default): also single-phase, `epochs` epochs.
        # model_type="master", schedule="two_phase" (legacy, opt-in):
        # epochs_phase1 + epochs_phase2.
        if self.model_type == "student" or self.schedule == "end_to_end":
            return self.epochs
        return self.epochs_phase1 + self.epochs_phase2
