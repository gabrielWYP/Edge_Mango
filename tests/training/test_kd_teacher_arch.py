"""The KD teacher must be rebuilt from ITS OWN checkpoint's architecture.

Verified defect, reproduced by the fix-kd-blockers task 4.2 pipeline run
(`20260831T223558592803Z`): the master stage trained under
`configs/training_mango.yaml` (`head_strides: [8, 16, 32]`), and the KD stage
that consumed its checkpoint then built the teacher with
`config.head_strides` — which for `KDConfig` is the post-fusion-redesign
4-level default `[4, 8, 16, 32]`, because `configs/kd_training.yaml` never
sets the key. The stage died in `KDTrainer._load_teacher`:

    RuntimeError: Error(s) in loading state_dict for MasterModel:
        Missing key(s) in state_dict: "head.heads.3.cls_stem.0.weight", ...

`Trainer._save_checkpoint` already stores the full training config, so the
geometry was recorded and simply not read back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.master.master_model import MasterModel
from src.training.kd_config import KDConfig
from src.training.kd_trainer import KDTrainer

THREE_LEVEL_STRIDES = [8, 16, 32]


def _save_teacher(path: Path, head_strides: list[int], config: dict | None) -> None:
    """Write a teacher checkpoint with the shape `Trainer._save_checkpoint` writes."""
    teacher = MasterModel(
        num_classes=2,
        pretrained_backbone=False,
        backbone_variant="tiny",
        head_strides=head_strides,
    )
    payload: dict = {"model_state_dict": teacher.state_dict()}
    if config is not None:
        payload["config"] = config
    torch.save(payload, path)


def test_three_level_teacher_loads_against_the_four_level_kd_default(tmp_path):
    ckpt = tmp_path / "teacher.pt"
    _save_teacher(
        ckpt,
        head_strides=THREE_LEVEL_STRIDES,
        config={"head_strides": THREE_LEVEL_STRIDES, "backbone_variant": "tiny"},
    )

    config = KDConfig(teacher_checkpoint=str(ckpt))
    # The precondition that made this defect reachable: the KD run's own
    # config carries a different pyramid than the checkpoint it consumes.
    assert config.head_strides != THREE_LEVEL_STRIDES

    teacher = KDTrainer._load_teacher(config)

    assert teacher.head_strides == THREE_LEVEL_STRIDES


def test_checkpoint_without_a_config_key_falls_back_to_the_kd_config(tmp_path):
    """Checkpoints written before `config` was stored must still load."""
    ckpt = tmp_path / "legacy_teacher.pt"
    config = KDConfig(teacher_checkpoint=str(ckpt), head_strides=THREE_LEVEL_STRIDES)
    _save_teacher(ckpt, head_strides=THREE_LEVEL_STRIDES, config=None)

    teacher = KDTrainer._load_teacher(config)

    assert teacher.head_strides == THREE_LEVEL_STRIDES


def test_missing_teacher_checkpoint_still_raises(tmp_path):
    config = KDConfig(teacher_checkpoint=str(tmp_path / "absent.pt"))

    with pytest.raises(FileNotFoundError):
        KDTrainer._load_teacher(config)
