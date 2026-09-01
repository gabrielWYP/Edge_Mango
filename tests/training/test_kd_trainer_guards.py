"""D-G, mirrored into KDTrainer: OOM/NaN batch-skip guards must count and
raise, never skip silently.

Verified defect (maintainer-approved addition to fusion-redesign PR1):
`KDTrainer._train_epoch` (`src/training/kd_trainer.py`) had the identical
shape of defect fixed in `Trainer._train_epoch` (`src/training/loop.py`)
under D-G — NaN/Inf and CUDA-OOM batches were caught, printed, and
`continue`d before the batch counter incremented, and the loss divisor was
floored at `max(n_batches, 1)`. An epoch where every batch failed therefore
completed with finite-looking averaged losses, indistinguishable from a
clean run. `KDTrainer` had zero test coverage before this file.

These tests build a REAL `KDTrainer` (real `StudentModel`, real small
untrained `MasterModel` teacher checkpoint) rather than a fake, because
`backbone_projections()` / `fpn_projections()` / `head_projections()`
(`src/models/master/distill_projections.py`) hardcode channel counts tied to
the real student/teacher architectures — a fake model would have to
precisely reimplement those shapes to avoid a spurious shape-mismatch
failure that has nothing to do with the guard being tested. A fake loader
is still used for batches: no dataset/dataloader machinery is needed since
`_train_epoch` only reads `batch["rgb"|"nir"|"bboxes"|"labels"]`.
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

# Small enough to run fast on CPU; large enough that every FPN/head level
# still has a non-degenerate spatial size (mirrors tests/training/test_e2e.py's
# proven image_size=64 pattern for the sibling MasterModel trainer).
_IMAGE_SIZE = 64
_NUM_CLASSES = 2


def _fake_batch(n: int = 1) -> dict:
    return {
        "rgb": torch.randn(n, 3, _IMAGE_SIZE, _IMAGE_SIZE),
        "nir": torch.randn(n, 1, _IMAGE_SIZE, _IMAGE_SIZE),
        "bboxes": [torch.tensor([[0.5, 0.5, 0.3, 0.3]]) for _ in range(n)],
        "labels": [torch.tensor([0]) for _ in range(n)],
    }


class _FakeLoader:
    """Replayable iterable of `n_batches` fake batches, one sample each."""

    def __init__(self, n_batches: int):
        self.n_batches = n_batches

    def __iter__(self):
        return (_fake_batch(1) for _ in range(self.n_batches))


@pytest.fixture(scope="module")
def teacher_checkpoint_path(tmp_path_factory):
    """A real, untrained MasterModel state_dict saved as a teacher checkpoint."""
    teacher = MasterModel(
        num_classes=_NUM_CLASSES, pretrained_backbone=False, backbone_variant="tiny"
    )
    path = tmp_path_factory.mktemp("kd_guard_ckpt") / "teacher.pt"
    torch.save({"model_state_dict": teacher.state_dict()}, path)
    return path


def _make_kd_trainer(teacher_checkpoint_path, n_batches: int, tmp_path) -> KDTrainer:
    config = KDConfig(
        teacher_checkpoint=str(teacher_checkpoint_path),
        num_classes=_NUM_CLASSES,
        backbone_variant="tiny",
        image_size=_IMAGE_SIZE,
        batch_size=1,
        # No accumulation: these tests are about the OOM/NaN guards, and
        # they count one optimizer step per successful batch. Budget parity
        # under accumulation is covered by test_kd_budget_parity.py.
        effective_batch=1,
        precision="fp32",
        num_workers=0,
        output_dir=str(tmp_path),
    )
    model = StudentModel(num_classes=_NUM_CLASSES)
    trainer = KDTrainer(
        model=model,
        config=config,
        train_loader=_FakeLoader(n_batches),
        val_loader=_FakeLoader(0),
    )
    return trainer


def _optimizer_for(trainer: KDTrainer) -> torch.optim.AdamW:
    return torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)


class TestKDAllBatchesFail:
    """An epoch where every batch fails must raise, not report finite losses."""

    def test_all_oom_epoch1_raises_on_zero_steps(self, teacher_checkpoint_path, tmp_path):
        trainer = _make_kd_trainer(teacher_checkpoint_path, n_batches=3, tmp_path=tmp_path)

        def _raise_oom(rgb):
            raise RuntimeError("CUDA out of memory. Tried to allocate 999.00 GiB")

        trainer.model.forward = _raise_oom
        optimizer = _optimizer_for(trainer)

        with pytest.raises(RuntimeError, match="zero optimizer"):
            trainer._train_epoch(optimizer, epoch=1, phase=1)

    def test_all_nan_raises_and_records_nan_skipped(self, teacher_checkpoint_path, tmp_path):
        trainer = _make_kd_trainer(teacher_checkpoint_path, n_batches=4, tmp_path=tmp_path)
        # Forcing det_loss to NaN forces the summed loss (det + kd_weight*kd)
        # to NaN regardless of the real (finite) kd_loss value.
        trainer.criterion = lambda preds, targets: (
            torch.tensor(float("nan")),
            {"cls_loss": 0.0, "box_loss": 0.0},
        )
        optimizer = _optimizer_for(trainer)

        with pytest.raises(RuntimeError, match="nan_skipped=4") as exc_info:
            trainer._train_epoch(optimizer, epoch=1, phase=1)
        assert "zero optimizer" in str(exc_info.value)

    def test_oom_not_tolerated_from_epoch_2(self, teacher_checkpoint_path, tmp_path):
        trainer = _make_kd_trainer(teacher_checkpoint_path, n_batches=3, tmp_path=tmp_path)

        def _raise_oom(rgb):
            raise RuntimeError("CUDA out of memory. Tried to allocate 999.00 GiB")

        trainer.model.forward = _raise_oom
        optimizer = _optimizer_for(trainer)

        with pytest.raises(RuntimeError, match="epoch 2"):
            trainer._train_epoch(optimizer, epoch=2, phase=1)


class TestKDPartialFailureEpoch:
    """3.x-style: a partial-failure epoch counts correctly and still completes."""

    def test_partial_nan_epoch_completes_and_counts(self, teacher_checkpoint_path, tmp_path):
        trainer = _make_kd_trainer(teacher_checkpoint_path, n_batches=4, tmp_path=tmp_path)

        real_criterion = trainer.criterion
        calls = {"n": 0}

        def _flaky_criterion(preds, targets):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                return torch.tensor(float("nan")), {"cls_loss": 0.0, "box_loss": 0.0}
            return real_criterion(preds, targets)

        trainer.criterion = _flaky_criterion
        optimizer = _optimizer_for(trainer)

        result = trainer._train_epoch(optimizer, epoch=1, phase=1)

        assert result["nan_skipped"] == 2.0
        assert result["oom_skipped"] == 0.0
        assert result["steps_taken"] == 2.0

    def test_partial_oom_epoch1_tolerated_and_counted(self, teacher_checkpoint_path, tmp_path):
        trainer = _make_kd_trainer(teacher_checkpoint_path, n_batches=4, tmp_path=tmp_path)

        real_forward = trainer.model.forward
        calls = {"n": 0}

        def _flaky_forward(rgb):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
            return real_forward(rgb)

        trainer.model.forward = _flaky_forward
        optimizer = _optimizer_for(trainer)

        result = trainer._train_epoch(optimizer, epoch=1, phase=1)

        assert result["oom_skipped"] == 2.0
        assert result["nan_skipped"] == 0.0
        assert result["steps_taken"] == 2.0


class TestKDCountersInReturnedMetrics:
    """The counters appear in the returned metrics on a fully clean epoch too."""

    def test_clean_epoch_reports_zero_skips_and_all_counters_present(
        self, teacher_checkpoint_path, tmp_path
    ):
        trainer = _make_kd_trainer(teacher_checkpoint_path, n_batches=2, tmp_path=tmp_path)
        optimizer = _optimizer_for(trainer)

        result = trainer._train_epoch(optimizer, epoch=1, phase=1)

        assert result["oom_skipped"] == 0.0
        assert result["nan_skipped"] == 0.0
        assert result["steps_taken"] == 2.0
        for key in ("cls_loss", "box_loss", "kd_loss", "total_loss"):
            assert key in result
            assert torch.isfinite(torch.tensor(result[key]))
