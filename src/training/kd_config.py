"""
Knowledge Distillation training configuration.

Extends TrainingConfig with KD-specific fields: teacher checkpoint path,
distillation weight, temperature, feature-loss type, and per-level loss
weights.

Inherits from_yaml / to_yaml / __post_init__ from TrainingConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import TrainingConfig
from .kd_crosshead import KD_GATES, KD_MODES, NIR_GATES
from .kd_loss import KD_LOSS_TYPES, validate_region_weights, validate_sublevels


@dataclass
class KDConfig(TrainingConfig):
    """Configuration for knowledge distillation training.

    Extends TrainingConfig with teacher checkpoint and KD hyperparameters.
    All training parameters (epochs, patience, precision, batch_size,
    model_type) have KD-specific defaults that override the base
    TrainingConfig defaults.

    Attributes:
        teacher_checkpoint: Path to the teacher checkpoint (.pt file).
        teacher_kind: "master" (default, MasterModel) or "student", a frozen
            StudentModel whose variant comes from its checkpoint.
        teacher_input: What a student teacher reads: "rgb", "nir" or "rgbnir".
            The NIR is given the way a student trained on NIR images read it;
            "rgbnir" needs a teacher trained with `student_in_channels=4`.
        kd_weight: Scalar weight for KD loss in total loss composition.
        kd_temperature: Temperature for KD feature scaling (unused for MSE, reserved).
        kd_loss_type: Per-pair feature loss, "mse" (default, raw-feature MSE)
            or "pearson" (PKD, per-channel standardized MSE). See
            `src/training/kd_loss.py`.
        kd_region_weights: Optional weights for "damage", "fruit" and
            "background". Each region is averaged over its own cells, so a
            small lesion is not lost in the map mean. None (default) keeps the
            plain per-map loss.
        kd_sublevels: Optional map indices to distill per level, finest first,
            for example {"fpn": [0, 1]} for P3 and P4. A level left out keeps
            all its maps; an empty list turns it off. None (default) keeps all.
        kd_mode: "feature" (default, FitNets-style MSE through adapters) or
            "crosshead" (CrossKD: the student's mid head features go through the
            frozen teacher head; no adapters). See `src/training/kd_crosshead.py`.
        kd_gate: Per-cell weight of the crosshead term: "none" (whole map),
            "nir_high" / "nir_low" (cells where the teacher's damage decision
            depends most / least on NIR; needs `teacher_input="rgbnir"`) or
            "teacher_better". Only with `kd_mode="crosshead"`.
        distill_levels: Per-level weights for KD loss (backbone, fpn, head_cls, head_reg).
    """

    # KD-specific fields
    teacher_checkpoint: str = ""
    teacher_kind: str = "master"
    teacher_input: str = "rgb"
    kd_weight: float = 1.0
    kd_temperature: float = 1.0
    kd_loss_type: str = "mse"
    kd_region_weights: dict | None = None
    kd_sublevels: dict | None = None
    kd_mode: str = "feature"
    kd_gate: str = "none"
    distill_levels: dict = field(default_factory=lambda: {
        "backbone": 1.0,
        "fpn": 0.1,
        "head_cls": 0.3,
        "head_reg": 0.5,
    })

    # Overridden defaults for KD training
    epochs: int = 80
    patience: int = 30
    precision: str = "fp32"
    batch_size: int = 4
    model_type: str = "student"

    # KD output directory
    output_dir: str = "checkpoints/kd_student"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.teacher_kind not in ("master", "student"):
            raise ValueError(f"Invalid teacher_kind '{self.teacher_kind}'.")
        if self.teacher_input not in ("rgb", "nir", "rgbnir"):
            raise ValueError(f"Invalid teacher_input '{self.teacher_input}'.")
        # Loud on a typo: an unknown loss type must not silently fall back to
        # MSE and invalidate an mse-vs-pearson ablation arm.
        if self.kd_loss_type not in KD_LOSS_TYPES:
            raise ValueError(
                f"Invalid kd_loss_type '{self.kd_loss_type}'. "
                f"Must be one of: {KD_LOSS_TYPES}"
            )
        validate_region_weights(self.kd_region_weights)
        validate_sublevels(self.kd_sublevels, self.distill_levels)
        self.validate_kd_mode()

    def validate_kd_mode(self) -> None:
        """Refuse a `kd_mode` / `kd_gate` combination that cannot run as named.

        Also called by `KDTrainer`, because `kd_train --override` sets fields
        after `__post_init__` has run.
        """
        if self.kd_mode not in KD_MODES:
            raise ValueError(f"Invalid kd_mode '{self.kd_mode}'. Must be one of: {KD_MODES}")
        if self.kd_gate not in KD_GATES:
            raise ValueError(f"Invalid kd_gate '{self.kd_gate}'. Must be one of: {KD_GATES}")
        if self.kd_mode == "crosshead":
            if self.teacher_kind != "student":
                raise ValueError("kd_mode=crosshead needs teacher_kind=student: it reuses the teacher's head.")
            if self.ignore_classes:
                raise ValueError("kd_mode=crosshead does not support ignore_classes.")
        elif self.kd_gate != "none":
            raise ValueError(f"kd_gate={self.kd_gate} needs kd_mode=crosshead.")
        if self.kd_gate in NIR_GATES and self.teacher_input != "rgbnir":
            raise ValueError(f"kd_gate={self.kd_gate} needs teacher_input=rgbnir.")
