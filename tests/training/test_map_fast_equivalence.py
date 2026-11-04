"""The ranked AP must give exactly the values of the per-prediction loop it replaced."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision.ops import box_iou

from src.training.metrics import _compute_ap_at_iou


def _reference_ap(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold, num_classes):
    """The original implementation: one `box_iou` per prediction, in rank order."""
    per_class_ap = {}
    for cls_id in range(num_classes):
        predictions, class_gts = [], []
        for image_idx, (boxes_i, scores_i, labels_i, gts_i, gt_labels_i) in enumerate(
            zip(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
        ):
            mask = labels_i == cls_id
            for box, score in zip(boxes_i[mask], scores_i[mask]):
                predictions.append((float(score.item()), image_idx, box))
            class_gts.append(gts_i[gt_labels_i == cls_id])
        n_gt = sum(len(g) for g in class_gts)
        if n_gt == 0 or not predictions:
            per_class_ap[cls_id] = 0.0
            continue
        predictions.sort(key=lambda item: item[0], reverse=True)
        matched = set()
        tp, fp = torch.zeros(len(predictions)), torch.zeros(len(predictions))
        for i, (_, image_idx, box) in enumerate(predictions):
            gts = class_gts[image_idx]
            if len(gts) == 0:
                fp[i] = 1
                continue
            best_iou, best_gt = box_iou(box.unsqueeze(0), gts).squeeze(0).max(dim=0)
            key = (image_idx, best_gt.item())
            if best_iou >= iou_threshold and key not in matched:
                tp[i] = 1
                matched.add(key)
            else:
                fp[i] = 1
        cum_tp, cum_fp = tp.cumsum(0), fp.cumsum(0)
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / n_gt
        recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = precision[i].max(precision[i + 1])
        per_class_ap[cls_id] = ((recall[1:] - recall[:-1]) * precision[1:]).sum().item()
    return float(np.mean(list(per_class_ap.values()))), per_class_ap


def _boxes(rng, n):
    xy = rng.random((n, 2)) * 0.7
    wh = rng.random((n, 2)) * 0.3 + 0.01
    return torch.tensor(np.hstack([xy, xy + wh]), dtype=torch.float32)


def _dataset(seed, images=40, tied_scores=False, duplicate_gts=False):
    rng = np.random.default_rng(seed)
    out = ([], [], [], [], [])
    for _ in range(images):
        n_gt = int(rng.integers(0, 4))
        gts = _boxes(rng, n_gt)
        if duplicate_gts and n_gt >= 2:
            gts[1] = gts[0]  # an IoU tie: the first GT wins
        n_pred = int(rng.integers(0, 30))
        boxes = _boxes(rng, n_pred)
        if n_gt:  # some predictions near a GT, so that thresholds matter
            near = rng.random(n_pred) < 0.5
            picks = rng.integers(0, n_gt, n_pred)
            noise = torch.tensor(rng.normal(0, 0.01, (n_pred, 4)), dtype=torch.float32)
            boxes[near] = gts[picks[near]] + noise[near]
        scores = torch.tensor(rng.random(n_pred), dtype=torch.float32)
        if tied_scores:
            scores = (scores * 4).round() / 4  # many equal scores across images
        out[0].append(boxes)
        out[1].append(scores)
        out[2].append(torch.tensor(rng.integers(0, 2, n_pred), dtype=torch.int64))
        out[3].append(gts)
        out[4].append(torch.tensor(rng.integers(0, 2, n_gt), dtype=torch.int64))
    return out


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("threshold", [0.5 + 0.05 * i for i in range(10)])
@pytest.mark.parametrize("tied", [False, True])
def test_values_are_exactly_those_of_the_original_loop(seed, threshold, tied) -> None:
    data = _dataset(seed, tied_scores=tied, duplicate_gts=tied)
    assert _compute_ap_at_iou(*data, threshold, 2) == _reference_ap(*data, threshold, 2)


def test_equal_scores_keep_image_order() -> None:
    gt = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    miss = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
    one = torch.ones(1)
    label = torch.ones(1, dtype=torch.int64)
    no_gt, no_label = torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64)
    data = ([miss, box], [one, one], [label, label], [no_gt, gt], [no_label, label])
    assert _compute_ap_at_iou(*data, 0.5, 2) == _reference_ap(*data, 0.5, 2)
    assert _compute_ap_at_iou(*data, 0.5, 2)[1][1] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "data",
    [
        ([torch.zeros((0, 4))], [torch.zeros(0)], [torch.zeros(0, dtype=torch.int64)],
         [torch.tensor([[0.0, 0.0, 1.0, 1.0]])], [torch.tensor([1])]),
        ([torch.tensor([[0.0, 0.0, 1.0, 1.0]])], [torch.ones(1)], [torch.tensor([1])],
         [torch.zeros((0, 4))], [torch.zeros(0, dtype=torch.int64)]),
    ],
    ids=["no_predictions", "no_ground_truth"],
)
def test_empty_cases_give_zero(data) -> None:
    assert _compute_ap_at_iou(*data, 0.5, 2) == (0.0, {0: 0.0, 1: 0.0})
