"""The trainers consume the collated padded targets and report what the per-image path reported.

CPU only, tiny synthetic tensors.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch.optim import AdamW

from src.models.student.student_model import StudentModel
from src.training.config import TrainingConfig
from src.training.dataset import collate_fn
from src.training.loop import Trainer

SIZE = 64
PROTO = SIZE // 4
PADDED_KEYS = ("bboxes_padded", "labels_padded", "counts", "masks_padded", "has_mask_padded")


def _item(n: int, seed: int, masks: bool) -> dict:
    gen = torch.Generator().manual_seed(seed)
    item = {
        "rgb": torch.randn(3, SIZE, SIZE, generator=gen),
        "nir": torch.randn(1, SIZE, SIZE, generator=gen),
        "bboxes": torch.cat([torch.rand(n, 2, generator=gen) * 0.4 + 0.3,
                             torch.rand(n, 2, generator=gen) * 0.3 + 0.2], dim=1),
        "labels": torch.randint(0, 2, (n,), generator=gen),
        "image_id": str(seed),
    }
    if masks:
        item["masks"] = (torch.rand(n, PROTO, PROTO, generator=gen) > 0.5).to(torch.uint8)
        item["has_mask"] = torch.rand(n, generator=gen) > 0.2
    return item


class _Loader:
    def __init__(self, batches: list[dict]):
        self.batches = batches

    def __iter__(self):
        return iter(copy.deepcopy(self.batches))


def _batches(masks: bool, padded: bool) -> list[dict]:
    out = []
    for step in range(2):
        batch = collate_fn([_item(n, 10 * step + i, masks) for i, n in enumerate((2, 0, 1, 3))])
        if not padded:
            batch = {k: v for k, v in batch.items() if k not in PADDED_KEYS}
        out.append(batch)
    return out


def _config(tmp_path, **extra) -> TrainingConfig:
    return TrainingConfig(
        model_type="student", image_size=SIZE, batch_size=4, effective_batch=4, num_workers=0,
        device="cpu", precision="fp32", output_dir=str(tmp_path / "out"), **extra,
    )


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_epoch_metrics_are_the_same_from_padded_and_list_batches(tmp_path, task) -> None:
    masks = task == "segment"
    metrics = []
    for padded in (True, False):
        torch.manual_seed(0)
        model = StudentModel(num_classes=2, task=task)
        trainer = Trainer(
            model, _config(tmp_path / str(padded), task=task),
            _Loader(_batches(masks, padded)), _Loader(_batches(masks, padded)),
        )
        metrics.append(trainer._train_epoch(AdamW(model.parameters(), lr=1e-3), epoch=1, phase=1))
    padded_metrics, list_metrics = metrics
    assert padded_metrics.keys() == list_metrics.keys()
    for key, value in list_metrics.items():
        assert isinstance(padded_metrics[key], float)
        assert padded_metrics[key] == pytest.approx(value, rel=1e-4, abs=1e-6), key


def test_device_targets_of_a_padded_batch_hand_the_loss_the_padded_tensors(tmp_path) -> None:
    trainer = Trainer(StudentModel(num_classes=2), _config(tmp_path), None, _Loader([]))
    batch = _batches(masks=False, padded=True)[0]
    targets, bboxes, labels = trainer._device_targets(batch)
    assert targets["bboxes_padded"].shape == (4, 3, 4) and targets["counts"].tolist() == [2, 0, 1, 3]
    assert [len(b) for b in bboxes] == [2, 0, 1, 3] and [len(l) for l in labels] == [2, 0, 1, 3]
    assert torch.equal(bboxes[3], batch["bboxes"][3])
    plain, plain_boxes, _ = trainer._device_targets(_batches(masks=False, padded=False)[0])
    assert "bboxes_padded" not in plain and len(plain_boxes) == 4


def test_the_trainer_asks_the_loss_for_device_tensors_not_floats(tmp_path) -> None:
    trainer = Trainer(StudentModel(num_classes=2), _config(tmp_path), None, _Loader([]))
    assert trainer.criterion.log_as_tensors is True
