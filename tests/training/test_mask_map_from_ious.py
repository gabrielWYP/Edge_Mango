"""`compute_mask_map_from_ious` must equal `compute_mask_map` and `Trainer.evaluate` must use it."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.student.student_model import StudentModel
from src.training.config import TrainingConfig
from src.training.loop import Trainer
from src.training.metrics import _mask_iou, compute_mask_map, compute_mask_map_from_ious

SIZE = 64
PROTO = SIZE // 4


def _random_case(seed: int, n_images: int = 8, num_classes: int = 2, side: int = 12):
    g = torch.Generator().manual_seed(seed)
    pred_masks, scores, labels, gt_masks, gt_labels, has_mask = [], [], [], [], [], []
    for i in range(n_images):
        n_pred = 0 if i == 1 else int(torch.randint(1, 9, (1,), generator=g))
        n_gt = 0 if i == 2 else int(torch.randint(1, 5, (1,), generator=g))
        pred_masks.append(torch.rand(n_pred, side, side, generator=g) > 0.5)
        scores.append(torch.rand(n_pred, generator=g))
        labels.append(torch.randint(0, num_classes, (n_pred,), generator=g))
        gt_masks.append(torch.rand(n_gt, side, side, generator=g) > 0.5)
        gt_labels.append(torch.randint(0, num_classes, (n_gt,), generator=g))
        has_mask.append(torch.rand(n_gt, generator=g) > 0.3)
    has_mask[3] = torch.zeros_like(has_mask[3])  # an image whose GT has no polygon at all
    return pred_masks, scores, labels, gt_masks, gt_labels, has_mask


def _ious(pred_masks, gt_masks):
    return [_mask_iou(p.flatten(1), g.flatten(1)) for p, g in zip(pred_masks, gt_masks)]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("use_has_mask", [True, False])
def test_from_ious_equals_compute_mask_map(seed: int, use_has_mask: bool) -> None:
    pm, sc, lb, gm, gl, hm = _random_case(seed)
    has = hm if use_has_mask else None
    expected = compute_mask_map(pm, sc, lb, gm, gl, num_classes=2, gt_has_mask=has)
    got = compute_mask_map_from_ious(_ious(pm, gm), sc, lb, gl, num_classes=2, gt_has_mask=has)
    assert got.keys() == expected.keys()
    assert got["mask_map50"] == pytest.approx(expected["mask_map50"], abs=1e-12)
    assert got["mask_map_50_95"] == pytest.approx(expected["mask_map_50_95"], abs=1e-12)
    for key in ("mask_per_class_ap_50", "mask_per_class_ap_50_95"):
        assert got[key].keys() == expected[key].keys()
        for cls in expected[key]:
            assert got[key][cls] == pytest.approx(expected[key][cls], abs=1e-12)


def test_from_ious_with_nothing_at_all_matches() -> None:
    expected = compute_mask_map([], [], [], [], [], num_classes=2)
    assert compute_mask_map_from_ious([], [], [], [], num_classes=2) == expected


def test_from_ious_perfect_prediction_scores_one() -> None:
    m = torch.zeros(1, 8, 8, dtype=torch.bool)
    m[0, 2:6, 2:6] = True
    out = compute_mask_map_from_ious(
        _ious([m], [m]), [torch.tensor([0.9])], [torch.tensor([0])], [torch.tensor([0])], num_classes=2
    )
    assert out["mask_per_class_ap_50"][0] == pytest.approx(1.0)


# --- Trainer.evaluate ----------------------------------------------------------------------------


def _batch() -> dict:
    mango = torch.zeros(PROTO, PROTO, dtype=torch.uint8)
    mango[3:13, 3:13] = 1
    damage = torch.zeros(PROTO, PROTO, dtype=torch.uint8)
    damage[6:10, 6:10] = 1
    return {
        "rgb": torch.randn(2, 3, SIZE, SIZE),
        "nir": torch.randn(2, 1, SIZE, SIZE),
        "bboxes": [torch.tensor([[0.5, 0.5, 0.6, 0.6], [0.5, 0.5, 0.25, 0.25]])] * 2,
        "labels": [torch.tensor([0, 1])] * 2,
        "image_ids": [0, 1],
        "masks": [torch.stack([mango, damage])] * 2,
        "has_mask": [torch.tensor([True, False]), torch.tensor([True, True])],
    }


class _Loader:
    def __iter__(self):
        torch.manual_seed(0)
        return iter([_batch(), _batch()])


def test_evaluate_mask_metrics_equal_the_cpu_reference(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = StudentModel(num_classes=2, task="segment")
    config = TrainingConfig(
        model_type="student", image_size=SIZE, batch_size=2, effective_batch=2, num_workers=0,
        device="cpu", precision="fp32", output_dir=str(tmp_path / "out"), task="segment",
        eval_conf_threshold=1e-6,
    )
    trainer = Trainer(model, config, None, _Loader())
    seen: dict = {"scores": [], "labels": [], "masks": []}

    def sink(image_id, boxes, scores, labels, masks=None):
        seen["scores"].append(scores)
        seen["labels"].append(labels)
        seen["masks"].append(masks)

    metrics = trainer.evaluate(prediction_sink=sink)
    assert sum(len(s) for s in seen["scores"]) > 0, "fixture must produce detections"
    gt_masks, gt_labels, has = [], [], []
    for batch in _Loader():
        gt_masks += [m.bool() for m in batch["masks"]]
        gt_labels += list(batch["labels"])
        has += list(batch["has_mask"])
    expected = compute_mask_map(
        seen["masks"], seen["scores"], seen["labels"], gt_masks, gt_labels, num_classes=2, gt_has_mask=has
    )
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value, abs=1e-9), key
