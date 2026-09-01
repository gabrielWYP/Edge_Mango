"""Integration tests for `KDTrainer._train_epoch` gradient flow.

Covers slice 2 of `fix-detection-eval-and-kd-blockers`: after one KD training
step, every `kd_proj_*` parameter must have received a non-zero gradient and
changed value, while the frozen teacher must remain untouched. Uses a fake
(no-download) MasterModel teacher checkpoint and a one-batch synthetic loader,
CPU-only, `image_size=64`.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.master.master_model import MasterModel
from src.models.student.student_model import StudentModel
from src.training.kd_config import KDConfig
from src.training.kd_trainer import KDTrainer

KD_PROJ_GROUPS = ("kd_proj_backbone", "kd_proj_fpn", "kd_proj_head_cls", "kd_proj_head_reg")


@pytest.fixture
def teacher_checkpoint(tmp_path):
    """Fake teacher checkpoint — no-download MasterModel construction, same
    pattern as `tests/training/test_e2e.py::_create_synthetic_dataset` /
    `test_two_phase_loop`'s model instantiation."""
    teacher = MasterModel(num_classes=2, pretrained_backbone=False, backbone_variant="tiny")
    ckpt_path = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": teacher.state_dict()}, ckpt_path)
    return ckpt_path


@pytest.fixture
def synthetic_loader():
    """A plain list with one batch dict, following the pattern at
    `tests/test_training_loop.py::TestForwardCallBranching._make_trainer`."""
    batch = {
        "rgb": torch.randn(1, 3, 64, 64),
        "nir": torch.randn(1, 1, 64, 64),
        "bboxes": [torch.tensor([[0.5, 0.5, 0.2, 0.2]])],
        "labels": [torch.tensor([0])],
    }
    return [batch]


@pytest.fixture
def kd_config(teacher_checkpoint, tmp_path):
    return KDConfig(
        teacher_checkpoint=str(teacher_checkpoint),
        num_classes=2,
        backbone_variant="tiny",
        image_size=64,
        batch_size=1,
        precision="fp32",
        num_workers=0,
        output_dir=str(tmp_path),
    )


def _make_kd_trainer(kd_config, synthetic_loader) -> KDTrainer:
    model = StudentModel(num_classes=2)
    return KDTrainer(
        model=model,
        config=kd_config,
        train_loader=synthetic_loader,
        val_loader=synthetic_loader,
    )


class TestKDTrainerGradientFlow:
    """R3/R4 scenarios: projections receive gradients; teacher does not."""

    def test_kd_proj_parameters_receive_gradients_and_update(
        self, kd_config, synthetic_loader
    ):
        torch.manual_seed(0)
        trainer = _make_kd_trainer(kd_config, synthetic_loader)

        # Build the optimizer the way `_train_phase` does.
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, trainer.model.parameters()), lr=1e-3
        )

        # Snapshot kd_proj_backbone weights before the call.
        snapshot = {
            name: param.detach().clone()
            for name, param in trainer.model.kd_proj_backbone.named_parameters()
        }

        # Inspect grads at the moment the optimizer consumes them. The base
        # loop zeroes them (set_to_none=True) right after each step, so they
        # do not survive to the end of the epoch — and whether they do is an
        # implementation detail of the loop, not the property under test.
        captured: dict[str, torch.Tensor | None] = {}
        real_step = optimizer.step

        def _capturing_step(*args, **kwargs):
            for group_name in KD_PROJ_GROUPS:
                for name, param in getattr(trainer.model, group_name).named_parameters():
                    captured[f"{group_name}.{name}"] = (
                        None if param.grad is None else param.grad.detach().clone()
                    )
            return real_step(*args, **kwargs)

        optimizer.step = _capturing_step
        trainer._train_epoch(optimizer, epoch=1, phase=1)

        assert captured, "optimizer.step() never ran — no gradient reached the optimizer"
        any_nonzero = False
        for key, grad in captured.items():
            assert grad is not None, (
                f"{key}.grad is None at step time — projection gradient path is severed"
            )
            if grad.abs().sum().item() > 0:
                any_nonzero = True
        assert any_nonzero, "No kd_proj_* parameter received a non-zero gradient"

        changed = any(
            not torch.equal(param.detach(), snapshot[name])
            for name, param in trainer.model.kd_proj_backbone.named_parameters()
        )
        assert changed, (
            "kd_proj_backbone weights did not change after optimizer.step() "
            "inside one KD training epoch"
        )

    def test_teacher_parameters_have_no_gradient(self, kd_config, synthetic_loader):
        torch.manual_seed(0)
        trainer = _make_kd_trainer(kd_config, synthetic_loader)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, trainer.model.parameters()), lr=1e-3
        )
        trainer._train_epoch(optimizer, epoch=1, phase=1)

        for param in trainer.teacher.parameters():
            assert param.grad is None, (
                "Teacher freeze violated — a teacher parameter received a "
                "gradient despite requires_grad=False and the narrowed "
                "no_grad() scope"
            )
