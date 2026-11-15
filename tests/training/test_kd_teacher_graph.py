"""KDTrainer runs its frozen teacher through `GraphedForward`; results must not change.

CPU only: a fake graph backend stands in for CUDA, so this checks the routing (which tensors
reach the teacher, replay vs eager, the clone-out of static buffers), not the CUDA capture.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from src.models.student.student_model import StudentModel
from src.training import perf
from src.training.kd_config import KDConfig
from src.training.kd_trainer import KDTrainer
from src.training.perf import GraphedForward

SIZE = 64


class _FakeBackend:
    def __init__(self):
        self.captures = self.replays = 0

    def usable(self, tensors) -> bool:
        return True

    def capture(self, fn, example_inputs):
        self.captures += 1
        static_in = [t.clone() for t in example_inputs]
        static_out = fn(*static_in)

        def replay(inputs):
            self.replays += 1
            for dst, src in zip(static_in, inputs):
                dst.copy_(src)
            leaves_out, _ = perf.tree_flatten(static_out)
            leaves_new, _ = perf.tree_flatten(fn(*static_in))
            for dst, new in zip(leaves_out, leaves_new):
                if isinstance(dst, torch.Tensor):
                    dst.copy_(new)
            return static_out

        return replay


def _batch() -> dict:
    gen = torch.Generator().manual_seed(3)
    return {
        "rgb": torch.randn(2, 3, SIZE, SIZE, generator=gen),
        "nir": torch.randn(2, 1, SIZE, SIZE, generator=gen),
        "bboxes": [torch.tensor([[0.5, 0.5, 0.6, 0.6], [0.5, 0.5, 0.2, 0.2]])] * 2,
        "labels": [torch.tensor([0, 1])] * 2,
    }


class _Loader:
    def __iter__(self):
        return iter([_batch(), _batch()])


def _teacher_file(tmp_path: Path, in_channels: int) -> Path:
    torch.manual_seed(5)
    path = tmp_path / f"teacher_{in_channels}.pt"
    model = StudentModel(num_classes=2, in_channels=in_channels)
    torch.save(
        {"model_state_dict": model.state_dict(),
         "config": {"student_variant": "wide", "student_in_channels": in_channels}},
        path,
    )
    return path


def _trainer(tmp_path: Path, teacher_input: str, **extra) -> KDTrainer:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = KDConfig(
        teacher_checkpoint=str(_teacher_file(tmp_path, 4 if teacher_input == "rgbnir" else 3)),
        teacher_kind="student", teacher_input=teacher_input, num_classes=2, image_size=SIZE,
        batch_size=2, num_workers=0, device="cpu", nir_mean=0.3, nir_std=0.1,
        output_dir=str(tmp_path / "out"), **extra,
    )
    torch.manual_seed(11)
    return KDTrainer(StudentModel(num_classes=2), config, _Loader(), _Loader())


def _epoch(trainer: KDTrainer) -> dict:
    return trainer._train_epoch(AdamW(trainer.model.parameters(), lr=1e-3), epoch=1, phase=1)


def test_the_teacher_graph_is_eager_on_cpu(tmp_path) -> None:
    trainer = _trainer(tmp_path, "rgb")
    assert isinstance(trainer._graphed_teacher, GraphedForward)
    assert trainer._graphed_teacher.enabled is False


@pytest.mark.parametrize("teacher_input, extra", [
    ("rgb", {}),
    ("nir", {}),
    ("rgbnir", {"kd_mode": "crosshead", "kd_gate": "nir_high"}),  # two forwards per step
    ("rgb", {"kd_mode": "crosshead"}),
])
def test_replayed_teacher_gives_the_same_epoch_as_the_eager_teacher(tmp_path, teacher_input, extra) -> None:
    eager = _epoch(_trainer(tmp_path / "eager", teacher_input, **extra))

    trainer = _trainer(tmp_path / "graph", teacher_input, **extra)
    backend = _FakeBackend()
    trainer._graphed_teacher = GraphedForward(trainer._teacher_forward, backend=backend)
    replayed = _epoch(trainer)

    assert backend.captures == 1 and backend.replays >= 2
    for key, value in eager.items():
        assert replayed[key] == pytest.approx(value, rel=1e-6, abs=1e-7), key


def test_the_teacher_sees_only_the_tensors_it_reads(tmp_path) -> None:
    rgb, nir = torch.zeros(1), torch.ones(1)
    assert len(_trainer(tmp_path / "a", "rgb")._teacher_inputs(rgb, nir)) == 1
    assert _trainer(tmp_path / "b", "nir")._teacher_inputs(rgb, nir)[0] is nir
    assert len(_trainer(tmp_path / "c", "rgbnir")._teacher_inputs(rgb, nir)) == 2
