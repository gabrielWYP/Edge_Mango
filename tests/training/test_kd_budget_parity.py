"""The distilled stage must receive the same optimizer budget as the baseline.

Task 4.2 of fix-kd-blockers measures the distilled student against the
RGB-only student under a matching budget. `KDTrainer` overrode
`_train_epoch` with its own copy of the loop and, as its docstring stated,
had no gradient accumulation: one optimizer step per batch. `KDConfig`
inherits `effective_batch` from `TrainingConfig`, so the checkpoint recorded
`effective_batch: 8` while the trainer never read it.

Measured in run `20260901T200124175077Z` under
`configs/machines/rtx3080.yaml` (batch_size 4):

    estudiante (Trainer):    accum 2, ceil(13/2) = 7 optimizer steps/epoch
    destilado  (KDTrainer):  no accum,           13 optimizer steps/epoch

Nearly twice the updates at half the effective batch, while both checkpoints
claimed the same `effective_batch`. It also meant the machine profile changed
the KD experiment outright — the D-H violation the profile whitelist exists
to prevent — because with no accumulation `batch_size` *is* the effective
batch.

The fix does not copy accumulation into `KDTrainer`. The two loop bodies were
identical except for the forward pass and the loss, and that duplication is
how they drifted. The base `Trainer._train_epoch` now owns the loop and calls
a `_forward_loss` hook; `KDTrainer` overrides only the hook.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.master.master_model import MasterModel
from src.models.student.student_model import StudentModel
from src.training.config import TrainingConfig
from src.training.kd_config import KDConfig
from src.training.kd_trainer import KDTrainer
from src.training.loop import Trainer
from src.training.machine import apply_machine_profile, load_machine_profile

_IMAGE_SIZE = 64
_NUM_CLASSES = 2
RTX3080_PROFILE = Path("configs/machines/rtx3080.yaml")


def _fake_batch() -> dict:
    return {
        "rgb": torch.randn(1, 3, _IMAGE_SIZE, _IMAGE_SIZE),
        "nir": torch.randn(1, 1, _IMAGE_SIZE, _IMAGE_SIZE),
        "bboxes": [torch.tensor([[0.5, 0.5, 0.3, 0.3]])],
        "labels": [torch.tensor([0])],
    }


class _FakeLoader:
    def __init__(self, n_batches: int):
        self.n_batches = n_batches

    def __iter__(self):
        return (_fake_batch() for _ in range(self.n_batches))

    def __len__(self):
        return self.n_batches


@pytest.fixture(scope="module")
def teacher_checkpoint_path(tmp_path_factory):
    teacher = MasterModel(
        num_classes=_NUM_CLASSES, pretrained_backbone=False, backbone_variant="tiny"
    )
    path = tmp_path_factory.mktemp("kd_parity_ckpt") / "teacher.pt"
    torch.save({"model_state_dict": teacher.state_dict()}, path)
    return path


_SHARED = dict(
    num_classes=_NUM_CLASSES,
    backbone_variant="tiny",
    image_size=_IMAGE_SIZE,
    batch_size=1,
    effective_batch=4,
    precision="fp32",
    num_workers=0,
)


def _student_trainer(n_batches: int, tmp_path: Path) -> Trainer:
    config = TrainingConfig(model_type="student", output_dir=str(tmp_path / "s"), **_SHARED)
    return Trainer(
        model=StudentModel(num_classes=_NUM_CLASSES),
        config=config,
        train_loader=_FakeLoader(n_batches),
        val_loader=_FakeLoader(0),
    )


def _kd_trainer(teacher_checkpoint_path, n_batches: int, tmp_path: Path) -> KDTrainer:
    config = KDConfig(
        teacher_checkpoint=str(teacher_checkpoint_path),
        output_dir=str(tmp_path / "kd"),
        **_SHARED,
    )
    return KDTrainer(
        model=StudentModel(num_classes=_NUM_CLASSES),
        config=config,
        train_loader=_FakeLoader(n_batches),
        val_loader=_FakeLoader(0),
    )


def _one_epoch(trainer: Trainer) -> dict:
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
    return trainer._train_epoch(optimizer, epoch=1, phase=1)


class TestKDStepBudgetMatchesTheBaseline:
    def test_kd_derives_grad_accum_steps_like_the_base_trainer(
        self, teacher_checkpoint_path, tmp_path
    ):
        kd = _kd_trainer(teacher_checkpoint_path, n_batches=1, tmp_path=tmp_path)

        assert kd.grad_accum_steps == 4

    def test_same_batches_yield_the_same_optimizer_steps(
        self, teacher_checkpoint_path, tmp_path
    ):
        """Six micro-batches at accum 4 -> 2 steps (4 + a flushed tail of 2)."""
        n_batches = 6
        student = _one_epoch(_student_trainer(n_batches, tmp_path))
        kd = _one_epoch(_kd_trainer(teacher_checkpoint_path, n_batches, tmp_path))

        expected = math.ceil(n_batches / 4)
        assert student["steps_taken"] == expected
        assert kd["steps_taken"] == expected

    def test_kd_reports_its_loss_components_averaged_over_micro_batches(
        self, teacher_checkpoint_path, tmp_path
    ):
        result = _one_epoch(_kd_trainer(teacher_checkpoint_path, n_batches=6, tmp_path=tmp_path))

        for key in ("cls_loss", "box_loss", "kd_loss", "total_loss"):
            assert key in result
            assert torch.isfinite(torch.tensor(result[key]))


class TestTheLoopHasOneOwner:
    """Two copies of the loop is how they drifted apart in the first place."""

    def test_kd_trainer_does_not_override_train_epoch(self):
        assert "_train_epoch" not in KDTrainer.__dict__

    def test_kd_trainer_overrides_only_the_forward_loss_hook(self):
        assert "_forward_loss" in KDTrainer.__dict__


class TestMachineProfileCannotChangeTheKDExperiment:
    """The real version of the assurance `test_machine_profile_plumbing` gave.

    That file checks the `effective_batch` *field* survives the profile. It
    did, and it meant nothing, because the trainer ignored the field.
    """

    def test_rtx3080_profile_yields_accumulation_not_a_smaller_effective_batch(
        self, teacher_checkpoint_path, tmp_path
    ):
        config = KDConfig(
            teacher_checkpoint=str(teacher_checkpoint_path),
            num_classes=_NUM_CLASSES,
            backbone_variant="tiny",
            image_size=_IMAGE_SIZE,
            output_dir=str(tmp_path),
        )
        apply_machine_profile(config, load_machine_profile(RTX3080_PROFILE))
        config.device = "cpu"

        trainer = KDTrainer(
            model=StudentModel(num_classes=_NUM_CLASSES),
            config=config,
            train_loader=_FakeLoader(0),
            val_loader=_FakeLoader(0),
        )

        assert config.batch_size == 4
        assert config.effective_batch == 8
        assert trainer.grad_accum_steps == 2
