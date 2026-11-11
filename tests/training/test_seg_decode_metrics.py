"""Per-detection masks after NMS, and mask AP next to box AP."""

from __future__ import annotations

import pytest
import torch

from src.training.decode import decode_detections, decode_detections_with_coefs, process_masks
from src.training.metrics import compute_map, compute_mask_map

SIZE = 64
STRIDES = (8, 16, 32)
NM = 4
PROTO = SIZE // 4


def _outputs(scores_at: dict[tuple[int, int, int, int], float]) -> tuple[list, list, list]:
    """`{(level, cls, y, x): score}` -> preds, cls_preds, mask_coefs for a batch of 1.

    Every cell predicts a 16x16 px box at its own anchor; coefficients encode the level.
    """
    preds, cls_preds, coefs = [], [], []
    for level, stride in enumerate(STRIDES):
        n = SIZE // stride
        cls = torch.full((1, 2, n, n), -20.0)
        reg = torch.zeros(1, 4, n, n)
        reg[:, 2:] = torch.log(torch.tensor(16.0))
        coef = torch.zeros(1, NM, n, n)
        for (lv, c, y, x), score in scores_at.items():
            if lv == level:
                cls[0, c, y, x] = torch.logit(torch.tensor(score))
                coef[0, level, y, x] = 1.0 + y + x  # tag the cell
        preds.append(torch.cat([cls, reg], dim=1))
        cls_preds.append(cls)
        coefs.append(coef)
    return preds, cls_preds, coefs


def _decode_kwargs() -> dict:
    return dict(batch_idx=0, num_classes=2, image_size=SIZE, strides=STRIDES, conf_threshold=0.1)


def test_decode_with_coefs_matches_plain_decode_and_returns_the_kept_coefficients() -> None:
    preds, cls_preds, coefs = _outputs({(0, 1, 3, 3): 0.9, (0, 1, 3, 4): 0.5, (1, 0, 1, 1): 0.8})
    boxes, scores, labels = decode_detections(preds, cls_preds, **_decode_kwargs())
    boxes_c, scores_c, labels_c, kept = decode_detections_with_coefs(
        preds, cls_preds, coefs, **_decode_kwargs()
    )
    assert torch.equal(boxes, boxes_c) and torch.equal(scores, scores_c) and torch.equal(labels, labels_c)
    assert kept.shape == (len(scores), NM)
    # The coefficient row follows the detection through NMS and sorting: level and cell tag.
    for row, label, score in zip(kept, labels_c, scores_c):
        if abs(score.item() - 0.8) < 1e-4:
            assert row[1].item() == pytest.approx(3.0)  # level 1, cell (1, 1): 1 + 1 + 1
        if abs(score.item() - 0.9) < 1e-4:
            assert row[0].item() == pytest.approx(7.0)  # level 0, cell (3, 3): 1 + 3 + 3


def test_decode_with_coefs_on_an_empty_image() -> None:
    preds, cls_preds, coefs = _outputs({})
    boxes, scores, labels, kept = decode_detections_with_coefs(preds, cls_preds, coefs, **_decode_kwargs())
    assert boxes.shape == (0, 4) and kept.shape == (0, NM)


def test_process_masks_crops_to_the_box_and_thresholds() -> None:
    protos = torch.full((NM, PROTO, PROTO), 8.0)  # sigmoid ~ 1 everywhere, for coefficient 1 on channel 0
    coefs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    box = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # centre half of the image
    masks = process_masks(protos, coefs, box)
    assert masks.shape == (1, PROTO, PROTO) and masks.dtype == torch.bool
    assert masks[0, 4:12, 4:12].all() and masks.sum().item() == 8 * 8


def test_process_masks_upsamples_to_the_requested_size() -> None:
    protos = torch.full((NM, PROTO, PROTO), 8.0)
    masks = process_masks(protos, torch.tensor([[1.0, 0, 0, 0]]), torch.tensor([[0.5, 0.5, 0.5, 0.5]]), out_size=(SIZE, SIZE))
    assert masks.shape == (1, SIZE, SIZE)
    assert masks[0, 20:44, 20:44].all() and not masks[0, :12].any()


def test_process_masks_without_detections() -> None:
    masks = process_masks(torch.zeros(NM, PROTO, PROTO), torch.zeros(0, NM), torch.zeros(0, 4))
    assert masks.shape == (0, PROTO, PROTO)


# --- mask AP -----------------------------------------------------------------------------------


def _square(x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
    mask = torch.zeros(PROTO, PROTO, dtype=torch.bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_perfect_masks_give_ap_one_and_wrong_masks_zero() -> None:
    gt = [torch.stack([_square(2, 2, 10, 10)])]
    labels = [torch.tensor([1])]
    perfect = compute_mask_map(gt, [torch.tensor([0.9])], labels, gt, labels, num_classes=2)
    assert perfect["mask_per_class_ap_50"][1] == pytest.approx(1.0)
    assert perfect["mask_map_50_95"] > 0.49  # only class 1 has GT: mean over two classes
    wrong = compute_mask_map(
        [torch.stack([_square(10, 10, 16, 16)])], [torch.tensor([0.9])], labels, gt, labels, num_classes=2
    )
    assert wrong["mask_per_class_ap_50"][1] == 0.0


def test_mask_ap_uses_mask_iou_not_box_iou() -> None:
    # Two triangles-like masks with identical boxes but IoU well under 0.5.
    gt_mask = torch.zeros(PROTO, PROTO, dtype=torch.bool)
    pred_mask = torch.zeros(PROTO, PROTO, dtype=torch.bool)
    for i in range(8):
        gt_mask[2 + i, 2 : 2 + i + 1] = True  # lower-left triangle of an 8x8 box
        pred_mask[2 + i, 2 + i : 10] = True  # upper-right triangle of the same box
    labels = [torch.tensor([1])]
    result = compute_mask_map(
        [pred_mask[None]], [torch.tensor([0.9])], labels, [gt_mask[None]], labels, num_classes=2
    )
    assert result["mask_per_class_ap_50"][1] == 0.0


def test_gt_without_a_mask_is_left_out() -> None:
    gt = [torch.stack([_square(2, 2, 10, 10), _square(0, 0, PROTO, PROTO)])]
    labels = [torch.tensor([1, 1])]
    has_mask = [torch.tensor([True, False])]
    pred = [torch.stack([_square(2, 2, 10, 10)])]
    result = compute_mask_map(pred, [torch.tensor([0.9])], [torch.tensor([1])], gt, labels,
                              num_classes=2, gt_has_mask=has_mask)
    assert result["mask_per_class_ap_50"][1] == pytest.approx(1.0)


def test_box_metrics_are_untouched() -> None:
    boxes = [torch.tensor([[0.5, 0.5, 0.2, 0.2]])]
    result = compute_map(boxes, [torch.tensor([0.9])], [torch.tensor([1])], boxes, [torch.tensor([1])], num_classes=2)
    assert not any(key.startswith("mask_") for key in result)
