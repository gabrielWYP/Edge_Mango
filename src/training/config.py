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

from .fusion_modes import (
    DEFAULT_FUSION_MODE,
    FUSION_MODE_CROSS_ATTENTION,
    validate_fusion_mode,
    validate_fusion_pos_encoding,
)
from .ema import DEFAULT_EMA_DECAY, DEFAULT_EMA_TAU, validate_ema
from .machine import derive_grad_accum_steps
from .neck_norm import (
    DEFAULT_NECK_GN_GROUPS,
    DEFAULT_NECK_NORM,
    validate_neck_norm,
    validate_neck_norm_for_fusion_mode,
)
from .nir_registration import DEFAULT_NIR_HOMOGRAPHY_PATH
from .precision import validate_bf16_support, validate_precision
from .roi_crop import validate_roi_crop, validate_roi_crop_train_margin_range
from .strides import (
    CROSS_ATTENTION_SUPPORTED_HEAD_STRIDES,
    DEFAULT_HEAD_STRIDES,
    validate_strides,
)

# Class index of the damage class in every label file ([mango, damage]).
DAMAGE_CLASS_ID = 1
# Validation metrics `selection_metric` may name, all on the val split:
# "map50" is the two-class AP@0.5 mean; the "damage_*" metrics are class
# DAMAGE_CLASS_ID alone: "damage_ap50" is AP@0.5, "damage_ap30" is AP at IoU 0.3
# and "damage_ap_center" is AP under the impact-point criterion (the prediction
# box contains the GT box center). The last two tolerate the uncertain extent of
# the induced-impact damage boxes (see src/training/metrics.py).
SELECTION_METRICS = frozenset({"map50", "damage_ap50", "damage_ap30", "damage_ap_center"})


@dataclass
class TrainingConfig:
    """Configuration for MasterModel 2-phase training.

    Attributes:
        rgb_dir: Path to RGB images directory.
        nir_dir: Path to NIR images directory.
        labels_dir: Path to YOLO label files root (contains train/, val/).
        output_dir: Where to save checkpoints and logs.
        nir_homography_path: RGB -> NIR homography (.npy) used to register NIR
            to the RGB frame at load time (`cv2.warpPerspective(nir,
            inv(H), ...)`), before letterbox/augmentation. Labels are in the
            RGB frame, so unregistered NIR is misaligned with them (~120 px).
            The file must exist (loud failure). Set to null ONLY for an
            explicit unregistered-NIR ablation. Runs trained before this
            field existed (2026-07-02 .. 2026-10) used unregistered NIR.
        roi_crop: Crop RGB and registered NIR to the ground-truth mango box
            (class 0) plus a margin before letterboxing, and re-express the
            labels relative to the crop. Opt-in (default False keeps the
            full-frame pipeline byte-identical). Val/test crops use the GT
            mango box, so metrics are an ORACLE-ROI upper bound until a
            stage-1 mango detector supplies the window, and mango (class 0)
            AP is meaningless under this setting.
        roi_crop_margin: Per-side margin as a fraction of the mango box
            width/height. Must be >= 0.
        roi_crop_jitter: Half-range of the per-side margin scale
            (`uniform(1 - j, 1 + j)`), applied to the train split only so the
            model sees varied framing; val/test crops stay deterministic.
            Must be in [0, 1], which guarantees the crop contains the whole
            mango box. 0 disables it.
        roi_crop_train_margin_range: `[min_total, max_total]` or null (off,
            today's behavior). Train split only: per axis, a TOTAL margin
            `t ~ U(min_total, max_total)` (a fraction of the mango box size) is
            split asymmetrically (`left = t * f`, `right = t * (1 - f)`,
            `f ~ U(0, 1)`), so the window still contains the whole mango box
            but the mango, and the damage the impact protocol concentrates at
            its center, lands at varied positions and scales in the input.
            Val/test keep the centered `roi_crop_margin`. Needs `roi_crop` and
            is mutually exclusive with `roi_crop_jitter > 0`.
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
        fusion_mode: MasterModel multimodal fusion architecture — "early"
            (default) or "cross_attention". "early" stacks RGB+NIR into one
            4-channel tensor before a single ConvNeXt stem (fusion-redesign).
            "cross_attention" is the restored two-stream path: two
            modality-specific stems over shared stages, per-stage
            cross-attention, and two FPNs fused per level. The two share no
            state_dict key and are tagged with different checkpoint
            `arch_version`s (2 and 1) — see `src/training/fusion_modes.py`.
            Ignored for model_type="student".
        fusion_pos_encoding: Add a fixed 2D sinusoidal positional encoding
            to the cross-attention query/key (`StageAttentionFusion`).
            Default False keeps the pre-existing behaviour. Adds no
            state_dict key, so loaders read it back from the checkpoint's
            recorded config (absent -> False) rather than trusting the
            caller's config. Only valid with fusion_mode="cross_attention";
            ignored for model_type="student". See
            reports/fusion-gradient-collapse/diagnosis.md.
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
        neck_norm: "batch" (default) | "group". Norm layer of the
            cross-attention neck's fusion convs, the only BatchNorm layers in
            the master model (`src/training/neck_norm.py`). "group" swaps in
            GroupNorm, which has no batch statistics: with a physical batch of
            2 the BatchNorm running statistics are noisy and drift between
            epochs. Changes the checkpoint schema; loaders read it back from
            the checkpoint's recorded config (absent -> "batch").
            `fusion_mode="cross_attention"` only - "group" is rejected on
            "early", whose neck has no norm layer. Ignored for
            model_type="student".
        neck_gn_groups: GroupNorm group-count cap; the largest divisor of the
            channel count <= `neck_gn_groups` is used (256 channels -> 32).
        ema: Keep an exponential moving average of the weights
            (`src/training/ema.py`), updated once per OPTIMIZER step. When
            on, validation, best-checkpoint selection and early stopping use
            the EMA weights, and every checkpoint stores them as
            `model_state_dict` (with `ema: True`), so evaluation tooling
            works unchanged. Off by default so existing runs are unchanged.
        ema_decay: Asymptotic EMA decay, in (0, 1); the average spans about
            1 / (1 - ema_decay) optimizer steps. The default 0.99 (~100
            steps) is about 5-6 epochs on the leave-one-lot-out folds, whose
            117-162 training pairs give 15-21 optimizer steps per epoch at
            `effective_batch=8` (1.5k-2.1k steps over 100 epochs). The
            YOLOv8-style 0.9998 would span 5000 steps - longer than the whole
            run.
        ema_tau: EMA warm-up time constant in optimizer steps (> 0): the
            decay applied by update `n` is
            `ema_decay * (1 - exp(-n / ema_tau))`, so the average tracks the
            live weights early instead of staying anchored to the
            initialization. The default 50 reaches ~0.87 after 5 epochs.
    """

    # Paths
    rgb_dir: str = "data/cache/mango/rgb"
    nir_dir: str = "data/cache/mango/nir"
    labels_dir: str = "data/annotations/yolo/labels"
    output_dir: str = "checkpoints/mastermodel"
    split_manifest: str | None = "data/annotations/yolo/splits.json"
    nir_homography_path: str | None = DEFAULT_NIR_HOMOGRAPHY_PATH

    # Model
    backbone_variant: str = "tiny"
    num_classes: int = 2

    # Augmentation
    image_size: int = 640
    letterbox_value: int = 114
    # Mango ROI crop (src/training/roi_crop.py) — opt-in, so every existing
    # config and checkpoint keeps the full-frame behaviour it has today.
    roi_crop: bool = False
    roi_crop_margin: float = 0.10
    roi_crop_jitter: float = 0.0
    roi_crop_train_margin_range: list[float] | None = None

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
    # `conf_threshold` is the deployment operating point: the visualizer's
    # default and the threshold the reported precision/recall/F1/TP/FP/FN
    # are measured at.
    conf_threshold: float = 0.25
    # Decode threshold for AP evaluation (`Trainer.evaluate`). AP integrates
    # the precision-recall curve over every ranked prediction, so the decode
    # must keep low-score candidates: decoding at `conf_threshold` truncated
    # the curve at the recall reached at 0.25 and scored AP 0.0 for any epoch
    # whose scores all fell below it. 0.001 is the COCO/YOLO validation value.
    eval_conf_threshold: float = 0.001
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

    # Checkpoint selection + early stopping metric (see SELECTION_METRICS).
    # "map50" is the unweighted two-class mean every recorded checkpoint was
    # selected on, kept as the default so existing configs are unchanged.
    # "damage_ap50" selects on the damage-class AP50 alone: under "map50" the
    # mango AP (~0.9) dominates the mean, so the epoch with the best damage AP
    # can go unsaved (twostream seed 42: val damage AP50 0.2077 at epoch 35,
    # never written to best_model.pt). "damage_ap30" / "damage_ap_center" select
    # on the localization-tolerant damage AP instead (IoU 0.3 / impact point).
    selection_metric: str = "map50"

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
    # Fusion architecture (restored two-stream path) — "early" | "cross_attention".
    # "early" is the default so every existing config, checkpoint and test
    # keeps the behaviour it has today.
    fusion_mode: str = DEFAULT_FUSION_MODE
    # Positional encoding on the cross-attention Q/K — opt-in, so every
    # existing config and checkpoint keeps the behaviour it has today.
    fusion_pos_encoding: bool = False

    # Training schedule (D-4, fusion-redesign) — MasterModel only.
    schedule: str = "end_to_end"
    backbone_lr_mult: float = 0.1

    # Neck normalization (src/training/neck_norm.py) - MasterModel only, and
    # only meaningful for fusion_mode="cross_attention". "batch" is the
    # existing checkpoint schema.
    neck_norm: str = DEFAULT_NECK_NORM
    neck_gn_groups: int = DEFAULT_NECK_GN_GROUPS

    # Weight EMA (src/training/ema.py) - opt-in, so every existing config
    # keeps validating and saving the live weights.
    ema: bool = False
    ema_decay: float = DEFAULT_EMA_DECAY
    ema_tau: float = DEFAULT_EMA_TAU

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
        # Loud on an unknown mode, for every model_type: a typo must not fall
        # back to the default and silently train the wrong architecture.
        # Unlike head_strides this needs no student exemption — the default
        # is valid for both model types.
        validate_fusion_mode(self.fusion_mode)
        validate_roi_crop(self.roi_crop_margin, self.roi_crop_jitter)
        validate_roi_crop_train_margin_range(
            self.roi_crop_train_margin_range, self.roi_crop_jitter, self.roi_crop
        )
        validate_neck_norm(self.neck_norm, self.neck_gn_groups)
        validate_ema(self.ema, self.ema_decay, self.ema_tau)
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(
                f"Invalid selection_metric '{self.selection_metric}'. "
                f"Must be one of: {sorted(SELECTION_METRICS)}"
            )
        if not 0.0 <= self.eval_conf_threshold <= self.conf_threshold:
            raise ValueError(
                f"eval_conf_threshold ({self.eval_conf_threshold}) must be in "
                f"[0, conf_threshold={self.conf_threshold}]: the precision/recall "
                "operating point is read off the AP decode, so that decode must "
                "keep every prediction scoring at or above conf_threshold."
            )
        if self.model_type == "master":
            validate_fusion_pos_encoding(self.fusion_pos_encoding, self.fusion_mode)
            validate_neck_norm_for_fusion_mode(self.neck_norm, self.fusion_mode)
            validate_strides(self.head_strides, self.assigner_level_ranges)
            # `DualFPN` builds one fusion conv per emitted level, so the
            # cross-attention path can honour its 3-level default or the
            # full 4-level pyramid — and nothing else. Caught here rather
            # than inside MasterModel so a config that can never build a
            # model fails at load time, before a run starts.
            if (
                self.fusion_mode == FUSION_MODE_CROSS_ATTENTION
                and tuple(self.head_strides) not in CROSS_ATTENTION_SUPPORTED_HEAD_STRIDES
            ):
                supported = [list(s) for s in CROSS_ATTENTION_SUPPORTED_HEAD_STRIDES]
                raise ValueError(
                    f"fusion_mode='{FUSION_MODE_CROSS_ATTENTION}' requires "
                    f"head_strides in {supported} (DualFPN emits one fused "
                    f"level per entry), got {list(self.head_strides)}."
                )

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
