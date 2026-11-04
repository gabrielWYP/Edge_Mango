"""KD from a frozen student: the same architecture as a teacher, on RGB or on NIR."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.models.student.student_model import StudentModel
from src.training.dataset import IMAGENET_MEAN, IMAGENET_STD, nir_as_rgb_tensor
from src.training.kd_config import KDConfig
from src.training.kd_trainer import KDTrainer

SIZE = 64


def _batch() -> dict:
    return {
        "rgb": torch.full((1, 3, SIZE, SIZE), 1.0),
        "nir": torch.full((1, 1, SIZE, SIZE), -1.0),
        "bboxes": [torch.tensor([[0.5, 0.5, 0.3, 0.3]])],
        "labels": [torch.tensor([0])],
    }


class _Loader:
    def __init__(self, n: int):
        self.n = n

    def __iter__(self):
        return (_batch() for _ in range(self.n))


def _teacher_file(tmp_path: Path, variant: str = "wide") -> Path:
    path = tmp_path / f"teacher_{variant}.pt"
    model = StudentModel(num_classes=2, variant=variant)
    torch.save({"model_state_dict": model.state_dict(), "config": {"student_variant": variant}}, path)
    return path


def _trainer(tmp_path: Path, teacher_input: str, variant: str = "wide", n: int = 1) -> KDTrainer:
    config = KDConfig(
        teacher_checkpoint=str(_teacher_file(tmp_path, variant)),
        teacher_kind="student",
        teacher_input=teacher_input,
        num_classes=2,
        image_size=SIZE,
        batch_size=1,
        num_workers=0,
        device="cpu",
        nir_mean=0.3,
        nir_std=0.1,
        student_variant=variant,
        output_dir=str(tmp_path / "out"),
    )
    return KDTrainer(
        model=StudentModel(num_classes=2, variant=variant),
        config=config,
        train_loader=_Loader(n),
        val_loader=_Loader(0),
    )


def test_nir_tensor_matches_what_the_nir_student_read_in_training(tmp_path: Path) -> None:
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8) * 3
    path = tmp_path / "nir.png"
    cv2.imwrite(str(path), gray)
    image = cv2.imread(str(path))  # the way the NIR student loads its "RGB" input
    expected = (image.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN, np.float32)) / np.array(
        IMAGENET_STD, np.float32
    )
    normalized = torch.from_numpy((gray.astype(np.float32) / 255.0 - 0.3) / 0.1)[None, None]

    result = nir_as_rgb_tensor(normalized, 0.3, 0.1)

    assert result.shape == (1, 3, 8, 8)
    assert torch.allclose(result[0].permute(1, 2, 0), torch.from_numpy(expected), atol=1e-5)


@pytest.mark.parametrize("teacher_input, expected_channel_value", [("rgb", 1.0), ("nir", None)])
def test_teacher_reads_the_requested_modality(tmp_path, teacher_input, expected_channel_value) -> None:
    trainer = _trainer(tmp_path, teacher_input)
    seen = {}
    real = trainer.teacher.forward
    trainer.teacher.forward = lambda x: seen.setdefault("x", x) is not None and real(x)
    batch = _batch()

    trainer._teacher_features(batch["rgb"], batch["nir"])

    if teacher_input == "rgb":
        assert (seen["x"] == 1.0).all()
    else:
        assert torch.allclose(seen["x"], nir_as_rgb_tensor(batch["nir"], 0.3, 0.1))


@pytest.mark.parametrize("variant", ["wide", "nano"])
def test_one_epoch_trains_student_and_adapters_and_leaves_the_teacher(tmp_path, variant) -> None:
    trainer = _trainer(tmp_path, "nir", variant)
    teacher_before = {k: v.clone() for k, v in trainer.teacher.state_dict().items()}
    student_before = trainer.model.backbone.stem[0].conv.weight.clone()
    adapter_before = next(trainer.kd_adapters.parameters()).clone()
    optimizer = torch.optim.AdamW(
        [*trainer.model.parameters(), *trainer.kd_adapters.parameters()], lr=0.01
    )

    metrics = trainer._train_epoch(optimizer, epoch=1, phase=1)

    assert metrics["steps_taken"] == 1 and metrics["kd_loss"] > 0
    assert all((teacher_before[k] == v).all() for k, v in trainer.teacher.state_dict().items())
    assert not torch.equal(trainer.model.backbone.stem[0].conv.weight, student_before)
    assert not torch.equal(next(trainer.kd_adapters.parameters()), adapter_before)


def test_a_master_teacher_is_still_the_default() -> None:
    config = KDConfig()
    assert config.teacher_kind == "master" and config.teacher_input == "rgb"


@pytest.mark.parametrize("field, value", [("teacher_kind", "x"), ("teacher_input", "x")])
def test_a_typo_is_refused(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        KDConfig(**{field: value})


def test_the_teacher_variant_comes_from_its_checkpoint(tmp_path: Path) -> None:
    config = KDConfig(
        teacher_checkpoint=str(_teacher_file(tmp_path, "nano")),
        teacher_kind="student",
        teacher_input="nir",
        output_dir=str(tmp_path / "out"),
        device="cpu",
    )
    trainer = KDTrainer._load_student_teacher(config)
    assert trainer.variant == "nano"
