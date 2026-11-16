"""A two-stem (`student_fusion=mid`) teacher loads through the KD teacher path and distills like an early one."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW

from src.models.student.student_model import StudentModel
from src.training.kd_config import KDConfig
from src.training.kd_trainer import KDTrainer

SIZE = 64


def _batch() -> dict:
    return {
        "rgb": torch.randn(2, 3, SIZE, SIZE),
        "nir": torch.randn(2, 1, SIZE, SIZE),
        "bboxes": [torch.tensor([[0.5, 0.5, 0.6, 0.6], [0.5, 0.5, 0.2, 0.2]])] * 2,
        "labels": [torch.tensor([0, 1])] * 2,
    }


class _Loader:
    def __init__(self, n: int):
        self.n = n

    def __iter__(self):
        return (_batch() for _ in range(self.n))


def _teacher_file(tmp_path: Path, fusion: str | None) -> Path:
    """A tiny 4-channel segment teacher; `fusion=None` writes a checkpoint without the key (every older one)."""
    path = tmp_path / f"teacher_{fusion}.pt"
    model = StudentModel(num_classes=2, variant="nano", in_channels=4, task="segment", fusion=fusion or "early")
    cfg = {"student_variant": "nano", "student_in_channels": 4, "task": "segment"}
    if fusion is not None:
        cfg["student_fusion"] = fusion
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, path)
    return path


def _config(tmp_path: Path, fusion: str | None) -> KDConfig:
    return KDConfig(
        teacher_checkpoint=str(_teacher_file(tmp_path, fusion)),
        teacher_kind="student", teacher_input="rgbnir", kd_mode="crosshead",
        num_classes=2, image_size=SIZE, batch_size=2, num_workers=0, device="cpu",
        nir_mean=0.3, nir_std=0.1, output_dir=str(tmp_path / "out"),
    )


def _trainer(tmp_path: Path, fusion: str | None) -> KDTrainer:
    return KDTrainer(
        model=StudentModel(num_classes=2, variant="nano", task="segment"),
        config=_config(tmp_path, fusion), train_loader=_Loader(1), val_loader=_Loader(0),
    )


def test_a_two_stem_teacher_checkpoint_loads_as_mid(tmp_path) -> None:
    teacher = KDTrainer._load_student_teacher(_config(tmp_path, "mid"))
    assert teacher.fusion == "mid" and teacher.in_channels == 4
    assert type(teacher.backbone).__name__ == "TwoStemBackbone"
    assert all(not p.requires_grad for p in teacher.parameters()) and not teacher.training


def test_an_early_teacher_still_loads_as_early_with_or_without_the_key(tmp_path) -> None:
    for fusion in ("early", None):
        teacher = KDTrainer._load_student_teacher(_config(tmp_path, fusion))
        assert teacher.fusion == "early" and teacher.in_channels == 4
        assert type(teacher.backbone).__name__ != "TwoStemBackbone"


def test_a_mid_teacher_is_fed_rgb_and_nir_as_separate_tensors(tmp_path) -> None:
    trainer = _trainer(tmp_path, "mid")
    batch = _batch()
    inputs = trainer._teacher_inputs(batch["rgb"], batch["nir"])
    assert [t.shape[1] for t in inputs] == [3, 1]


def test_a_crosshead_step_with_a_mid_teacher_has_finite_loss_and_grads_on_the_student_only(tmp_path) -> None:
    trainer = _trainer(tmp_path, "mid")
    teacher_before = {k: v.clone() for k, v in trainer.teacher.state_dict().items()}
    result = trainer._train_epoch(AdamW(trainer.model.parameters(), lr=1e-3), epoch=1, phase=1)
    assert result["steps_taken"] == 1.0 and torch.isfinite(torch.tensor(result["kd_loss"]))
    assert result["kd_loss"] != 0.0
    assert all(p.grad is None for p in trainer.teacher.parameters())
    assert any(p.grad is not None for p in trainer.model.parameters())
    assert all(torch.equal(v, trainer.teacher.state_dict()[k]) for k, v in teacher_before.items())
