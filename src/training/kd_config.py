"""
Knowledge Distillation training configuration.

Extends TrainingConfig with KD-specific fields: teacher checkpoint path,
distillation weight, temperature, and per-level loss weights.

Inherits from_yaml / to_yaml / __post_init__ from TrainingConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import TrainingConfig
from .kd_loss import KD_FEATURE_NORMS


@dataclass
class KDConfig(TrainingConfig):
    """Configuration for knowledge distillation training.

    Extends TrainingConfig with teacher checkpoint and KD hyperparameters.
    All training parameters (epochs, patience, precision, batch_size,
    model_type) have KD-specific defaults that override the base
    TrainingConfig defaults.

    Attributes:
        teacher_checkpoint: Path to MasterModel checkpoint (.pt file).
        kd_weight: Scalar weight for KD loss in total loss composition.
        kd_temperature: Temperature for KD feature scaling (unused for MSE, reserved).
        distill_levels: Per-level weights for KD loss (backbone, fpn, head_cls, head_reg).
        kd_feature_norm: Feature normalization applied before the per-level
            MSE. "none" (default) keeps the raw MSE. "pkd" z-scores student
            and teacher features per channel over (N, H, W) and halves the
            MSE (PKD, Cao et al. 2022), making the loss invariant to the
            teacher's large constant per-channel offsets.
    """

    # KD-specific fields
    teacher_checkpoint: str = ""
    kd_weight: float = 1.0
    kd_temperature: float = 1.0
    distill_levels: dict = field(default_factory=lambda: {
        "backbone": 1.0,
        "fpn": 0.1,
        "head_cls": 0.3,
        "head_reg": 0.5,
    })
    kd_feature_norm: str = "none"

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
        if self.kd_feature_norm not in KD_FEATURE_NORMS:
            raise ValueError(
                f"Invalid kd_feature_norm '{self.kd_feature_norm}'. "
                f"Must be one of: {sorted(KD_FEATURE_NORMS)}"
            )
