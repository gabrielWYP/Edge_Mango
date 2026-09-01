"""
Knowledge Distillation training configuration.

Extends TrainingConfig with KD-specific fields: teacher checkpoint path,
distillation weight, temperature, and per-level loss weights.

Inherits from_yaml / to_yaml / __post_init__ from TrainingConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import TrainingConfig


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

    # Overridden defaults for KD training. epochs/patience mirror
    # configs/training_student.yaml so a KDConfig built without a YAML file
    # still yields a run comparable to the RGB-only baseline. patience is 0
    # (early stopping disabled) for the reason given in that file.
    epochs: int = 50
    patience: int = 0
    precision: str = "fp32"
    batch_size: int = 4
    model_type: str = "student"

    # KD output directory
    output_dir: str = "checkpoints/kd_student"
