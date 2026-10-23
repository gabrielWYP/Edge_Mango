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
from .kd_loss import KD_LOSS_TYPES


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
        kd_loss_type: Per-pair feature loss, "mse" (default, raw-feature MSE)
            or "pearson" (PKD, per-channel standardized MSE). See
            `src/training/kd_loss.py`.
        distill_levels: Per-level weights for KD loss (backbone, fpn, head_cls, head_reg).
    """

    # KD-specific fields
    teacher_checkpoint: str = ""
    kd_weight: float = 1.0
    kd_temperature: float = 1.0
    kd_loss_type: str = "mse"
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
        # Loud on a typo: an unknown loss type must not silently fall back to
        # MSE and invalidate an mse-vs-pearson ablation arm.
        if self.kd_loss_type not in KD_LOSS_TYPES:
            raise ValueError(
                f"Invalid kd_loss_type '{self.kd_loss_type}'. "
                f"Must be one of: {KD_LOSS_TYPES}"
            )
