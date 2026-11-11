"""`task=segment` through the trainers: config, one train step, evaluation, and KD teachers.

CPU only, tiny synthetic tensors: nothing here touches a GPU or trains for real.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from src.models.student.student_model import StudentModel
from src.training.config import TrainingConfig
from src.training.kd_config import KDConfig
from src.training.kd_trainer import KDTrainer
from src.training.loop import Trainer

SIZE = 64
PROTO = SIZE // 4


def _batch(with_masks: bool = True) -> dict:
    mango = torch.zeros(PROTO, PROTO, dtype=torch.uint8)
    mango[3:13, 3:13] = 1
    damage = torch.zeros(PROTO, PROTO, dtype=torch.uint8)
    damage[6:10, 6:10] = 1
    batch = {
        "rgb": torch.randn(2, 3, SIZE, SIZE),
        "nir": torch.randn(2, 1, SIZE, SIZE),
        "bboxes": [torch.tensor([[0.5, 0.5, 0.6, 0.6], [0.5, 0.5, 0.25, 0.25]])] * 2,
        "labels": [torch.tensor([0, 1])] * 2,
        "image_ids": [0, 1],
    }
    if with_masks:
        batch["masks"] = [torch.stack([mango, damage])] * 2
        batch["has_mask"] = [torch.tensor([True, True])] * 2
    return batch


class _Loader:
    def __init__(self, n: int, with_masks: bool = True):
        self.n, self.with_masks = n, with_masks

    def __iter__(self):
        return (_batch(self.with_masks) for _ in range(self.n))


def _config(tmp_path: Path, **extra) -> TrainingConfig:
    return TrainingConfig(
        model_type="student", image_size=SIZE, batch_size=2, effective_batch=2, num_workers=0,
        device="cpu", precision="fp32", output_dir=str(tmp_path / "out"), **extra,
    )


# --- config --------------------------------------------------------------------------------------


def test_config_defaults_keep_detection() -> None:
    config = TrainingConfig()
    assert config.task == "detect" and config.mask_weight == 7.5 and config.mask_ratio == 4


@pytest.mark.parametrize("fields", [{"task": "pose"}, {"task": "segment", "model_type": "master"}, {"mask_ratio": 0}])
def test_invalid_task_settings_are_refused(fields) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**fields)


def test_segment_student_config_is_accepted() -> None:
    assert TrainingConfig(model_type="student", task="segment").task == "segment"


# --- training and evaluation ---------------------------------------------------------------------


def test_train_epoch_reports_a_mask_loss_and_updates_the_mask_layers(tmp_path) -> None:
    model = StudentModel(num_classes=2, task="segment")
    before = model.head.proto.cv1.conv.weight.detach().clone()
    trainer = Trainer(model, _config(tmp_path, task="segment"), _Loader(2), _Loader(1))
    optimizer = AdamW(model.parameters(), lr=1e-3)
    metrics = trainer._train_epoch(optimizer, epoch=1, phase=1)
    assert metrics["mask_loss"] > 0 and torch.isfinite(torch.tensor(metrics["total_loss"]))
    assert not torch.equal(before, model.head.proto.cv1.conv.weight)


def test_detect_student_keeps_its_train_metrics(tmp_path) -> None:
    trainer = Trainer(StudentModel(num_classes=2), _config(tmp_path), _Loader(1, with_masks=False), _Loader(1))
    metrics = trainer._train_epoch(AdamW(trainer.model.parameters(), lr=1e-3), epoch=1, phase=1)
    assert "mask_loss" not in metrics


def test_evaluate_adds_mask_metrics_without_touching_the_box_ones(tmp_path) -> None:
    model = StudentModel(num_classes=2, task="segment")
    segment = Trainer(model, _config(tmp_path, task="segment"), None, _Loader(1)).evaluate()
    assert {"mask_map50", "mask_map_50_95", "mask_per_class_ap_50", "mask_per_class_ap_50_95"} <= set(segment)
    detect = Trainer(StudentModel(num_classes=2), _config(tmp_path), None, _Loader(1)).evaluate()
    assert not any(key.startswith("mask_") for key in detect)
    assert {"map50", "map_50_95", "per_class_ap_50"} <= set(segment) & set(detect)


def test_evaluate_of_a_segment_student_on_box_only_batches_has_no_mask_metrics(tmp_path) -> None:
    metrics = Trainer(
        StudentModel(num_classes=2, task="segment"), _config(tmp_path, task="segment"), None, _Loader(1, with_masks=False)
    ).evaluate()
    assert "mask_map50" not in metrics


# --- KD: detect teacher / segment student, and segment / segment ---------------------------------


def _teacher_file(tmp_path: Path, task: str, in_channels: int = 3) -> Path:
    path = tmp_path / f"teacher_{task}_{in_channels}.pt"
    model = StudentModel(num_classes=2, in_channels=in_channels, task=task)
    config = {"student_variant": "wide", "student_in_channels": in_channels}
    if task != "detect":  # a detect checkpoint predates the key, as every recorded one does
        config["task"] = task
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
    return path


def _kd_config(tmp_path: Path, teacher_task: str, task: str, **extra) -> KDConfig:
    return KDConfig(
        teacher_checkpoint=str(_teacher_file(tmp_path, teacher_task)),
        teacher_kind="student", num_classes=2, image_size=SIZE, batch_size=2, num_workers=0,
        device="cpu", nir_mean=0.3, nir_std=0.1, output_dir=str(tmp_path / "out"), task=task, **extra,
    )


@pytest.mark.parametrize("teacher_task", ["detect", "segment"])
@pytest.mark.parametrize("kd_mode", ["crosshead", "feature"])
def test_kd_with_a_segment_student(tmp_path, teacher_task: str, kd_mode: str) -> None:
    config = _kd_config(tmp_path, teacher_task, "segment", kd_mode=kd_mode)
    model = StudentModel(num_classes=2, task="segment")
    trainer = KDTrainer(model, config, _Loader(1), _Loader(0))
    assert trainer.teacher.task == teacher_task
    metrics = trainer._train_epoch(AdamW(model.parameters(), lr=1e-3), epoch=1, phase=1)
    assert metrics["mask_loss"] > 0 and metrics["kd_loss"] >= 0
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))
    # Distillation reaches the student's backbone and the mask layers get the mask loss.
    grads = {n: p.grad for n, p in model.named_parameters()}
    assert any(g is not None for n, g in grads.items() if n.startswith("backbone."))


def test_kd_crosshead_gradient_does_not_depend_on_the_teacher_task(tmp_path) -> None:
    """The CrossKD term reads only cls/reg/mid features, so a segment teacher's extra layers are inert."""
    losses = []
    for teacher_task in ("detect", "segment"):
        torch.manual_seed(0)
        student = StudentModel(num_classes=2, task="segment")
        config = _kd_config(tmp_path, teacher_task, "segment", kd_mode="crosshead")
        trainer = KDTrainer(student, config, _Loader(1), _Loader(0))
        # Same teacher weights for both runs: copy the detection layers across.
        reference = StudentModel(num_classes=2, task="detect")
        torch.manual_seed(1)
        for p in reference.parameters():
            torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else None
        trainer.teacher.load_state_dict(reference.state_dict(), strict=False)
        batch = _batch()
        with torch.no_grad():
            targets = trainer._crosshead_targets(batch["rgb"], batch["nir"])
        s_out = student(batch["rgb"])
        losses.append(trainer._crosshead_kd(s_out, targets, batch["bboxes"], batch["labels"]).item())
    assert losses[0] == pytest.approx(losses[1], rel=1e-5)
