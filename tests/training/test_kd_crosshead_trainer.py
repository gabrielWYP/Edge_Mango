"""`kd_mode=crosshead` in `KDConfig` and `KDTrainer`: no adapters, the CrossKD term, the gates."""

from __future__ import annotations

from pathlib import Path

import pytest
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


def _teacher_file(tmp_path: Path, variant: str = "wide", in_channels: int = 3) -> Path:
    path = tmp_path / f"teacher_{variant}_{in_channels}.pt"
    model = StudentModel(num_classes=2, variant=variant, in_channels=in_channels)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"student_variant": variant, "student_in_channels": in_channels},
        },
        path,
    )
    return path


def _config(tmp_path: Path, teacher_input: str = "rgb", variant: str = "wide", **extra) -> KDConfig:
    return KDConfig(
        teacher_checkpoint=str(_teacher_file(tmp_path, variant, 4 if teacher_input == "rgbnir" else 3)),
        teacher_kind="student",
        teacher_input=teacher_input,
        num_classes=2,
        image_size=SIZE,
        batch_size=2,
        num_workers=0,
        device="cpu",
        nir_mean=0.3,
        nir_std=0.1,
        output_dir=str(tmp_path / "out"),
        **extra,
    )


def _trainer(tmp_path: Path, teacher_input: str = "rgb", student_variant: str = "wide", **extra) -> KDTrainer:
    config = _config(tmp_path, teacher_input, **extra)
    return KDTrainer(
        model=StudentModel(num_classes=2, variant=student_variant),
        config=config,
        train_loader=_Loader(1),
        val_loader=_Loader(0),
    )


# --- config --------------------------------------------------------------------------------------


def test_defaults_keep_the_feature_mode() -> None:
    config = KDConfig()
    assert config.kd_mode == "feature" and config.kd_gate == "none"


@pytest.mark.parametrize(
    "fields",
    [
        {"kd_mode": "bogus"},
        {"kd_gate": "bogus"},
        {"kd_mode": "crosshead", "teacher_kind": "master"},
        {"kd_gate": "teacher_better"},  # gate without crosshead
        {"kd_mode": "crosshead", "teacher_kind": "student", "kd_gate": "nir_high", "teacher_input": "rgb"},
        {"kd_mode": "crosshead", "teacher_kind": "student", "kd_gate": "nir_low", "teacher_input": "nir"},
        {"kd_mode": "crosshead", "teacher_kind": "student", "ignore_classes": [0]},
    ],
)
def test_invalid_combinations_are_refused(fields) -> None:
    with pytest.raises(ValueError):
        KDConfig(**fields)


def test_valid_crosshead_combinations_are_accepted() -> None:
    KDConfig(kd_mode="crosshead", teacher_kind="student")
    KDConfig(kd_mode="crosshead", teacher_kind="student", teacher_input="rgbnir", kd_gate="nir_high")
    KDConfig(kd_mode="crosshead", teacher_kind="student", teacher_input="rgb", kd_gate="teacher_better")


def test_overrides_that_skip_post_init_are_still_refused_by_the_trainer(tmp_path) -> None:
    config = _config(tmp_path, kd_mode="crosshead")
    config.kd_gate = "nir_high"  # what `kd_train --override` does: no __post_init__
    with pytest.raises(ValueError, match="nir_high"):
        KDTrainer(StudentModel(num_classes=2), config, _Loader(1), _Loader(0))


# --- construction --------------------------------------------------------------------------------


def test_crosshead_builds_no_adapters_and_no_empty_optimizer_group(tmp_path) -> None:
    trainer = _trainer(tmp_path, kd_mode="crosshead")
    assert len(trainer.kd_adapters) == 0
    assert trainer._auxiliary_param_groups(1e-3) == []
    optimizer = AdamW([{"params": list(trainer.model.parameters()), "lr": 1e-3}])
    assert all(group["params"] for group in optimizer.param_groups)


def test_feature_mode_still_has_its_adapters_and_group(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    assert len(trainer.kd_adapters) == 4
    assert len(trainer._auxiliary_param_groups(1e-3)) == 1


def test_checkpoint_state_round_trips_with_empty_adapters(tmp_path) -> None:
    trainer = _trainer(tmp_path, kd_mode="crosshead")
    state = trainer._auxiliary_checkpoint_state()
    assert state == {"kd_adapters_state_dict": {}}
    trainer._load_auxiliary_checkpoint_state(state)


def test_different_head_stem_channels_are_refused(tmp_path) -> None:
    config = _config(tmp_path, variant="wide", kd_mode="crosshead")
    with pytest.raises(ValueError, match="stem channels"):
        KDTrainer(StudentModel(num_classes=2, variant="nano"), config, _Loader(1), _Loader(0))


# --- one epoch -----------------------------------------------------------------------------------


def _epoch(trainer: KDTrainer) -> dict:
    optimizer = AdamW(trainer.model.parameters(), lr=1e-3)
    return trainer._train_epoch(optimizer, epoch=1, phase=1)


def test_a_crosshead_epoch_runs_and_logs_kd_loss(tmp_path) -> None:
    trainer = _trainer(tmp_path, kd_mode="crosshead")
    before = trainer.model.head.heads[0].cls_stem[0].conv.weight.detach().clone()
    result = _epoch(trainer)
    assert {"kd_loss", "total_loss", "cls_loss", "box_loss", "steps_taken"} <= set(result)
    assert result["steps_taken"] == 1.0 and torch.isfinite(torch.tensor(result["kd_loss"]))
    assert not torch.equal(before, trainer.model.head.heads[0].cls_stem[0].conv.weight)
    assert all(p.grad is None for p in trainer.teacher.parameters())
    assert not trainer.teacher.training


@pytest.mark.parametrize("gate, forwards", [("none", 1), ("teacher_better", 1), ("nir_high", 2), ("nir_low", 2)])
def test_only_the_nir_gates_run_a_second_teacher_forward_with_zero_nir(tmp_path, gate, forwards) -> None:
    trainer = _trainer(tmp_path, "rgbnir", kd_mode="crosshead", kd_gate=gate)
    seen = []
    real = trainer.teacher.forward

    def spy(x, nir=None):
        seen.append(nir.clone())
        return real(x, nir)

    trainer.teacher.forward = spy
    _epoch(trainer)
    assert len(seen) == forwards
    if forwards == 2:
        assert (seen[1] == 0).all() and not (seen[0] == 0).all()


@pytest.mark.parametrize("gate", ["none", "nir_high", "nir_low", "teacher_better"])
def test_every_gate_trains_without_error(tmp_path, gate) -> None:
    teacher_input = "rgbnir" if gate.startswith("nir") else "rgb"
    result = _epoch(_trainer(tmp_path, teacher_input, kd_mode="crosshead", kd_gate=gate))
    assert result["steps_taken"] == 1.0


def test_the_student_forward_in_crosshead_never_sees_the_nir(tmp_path) -> None:
    trainer = _trainer(tmp_path, "rgbnir", kd_mode="crosshead")
    seen = []
    real = trainer.model.forward
    trainer.model.forward = lambda x, nir=None: (seen.append(nir), real(x, nir))[1]
    _epoch(trainer)
    assert seen == [None]
