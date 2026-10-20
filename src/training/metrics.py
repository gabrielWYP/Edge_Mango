"""
Detection metrics: mAP@0.5, mAP@0.5:0.95, per-class AP.

Custom implementation using torchvision.ops.box_iou — no torchmetrics
or pycocotools dependency required.

Two localization-tolerant per-class metrics sit next to the IoU-based ones,
because the damage labels come from induced impacts (a 66 g steel ball dropped
from 50 cm, labeled from the known impact location) and the GT box extent is
therefore uncertain:

* AP@0.3 (`per_class_ap_30`): the same matching as AP@0.5 at IoU 0.3.
* Impact-point criterion (`per_class_ap_center`, `per_class_center_recall`): a
  prediction matches a GT when the prediction box contains the GT box center.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import numpy as np
from torchvision.ops import box_iou

logger = logging.getLogger(__name__)

# IoU threshold of the localization-tolerant AP (`per_class_ap_30`).
LOOSE_IOU_THRESHOLD = 0.3


@dataclass
class LossHistory:
    """Track per-epoch losses and validation metrics for analysis.

    The trainer owns this object and updates it once per epoch. It is kept
    intentionally dependency-light so the same history can be used by master,
    student, and KD training runs.
    """

    epoch: list[int] = field(default_factory=list)
    phase: list[int] = field(default_factory=list)
    cls_loss: list[float] = field(default_factory=list)
    box_loss: list[float] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    map50: list[float] = field(default_factory=list)
    map_50_95: list[float] = field(default_factory=list)
    per_class_ap_50: list[dict[int, float]] = field(default_factory=list)
    per_class_ap_50_95: list[dict[int, float]] = field(default_factory=list)
    per_class_precision: list[dict[int, float]] = field(default_factory=list)
    per_class_recall: list[dict[int, float]] = field(default_factory=list)
    per_class_f1: list[dict[int, float]] = field(default_factory=list)
    per_class_counts: list[dict[int, dict[str, int]]] = field(default_factory=list)
    # Localization-tolerant metrics (see the module docstring). Empty dicts for
    # callers that do not provide them, in which case the CSV omits the columns.
    per_class_ap_30: list[dict[int, float]] = field(default_factory=list)
    per_class_ap_center: list[dict[int, float]] = field(default_factory=list)
    per_class_center_recall: list[dict[int, float]] = field(default_factory=list)
    extra_losses: dict[str, list[float]] = field(default_factory=dict)

    def update(
        self,
        cls: float,
        box: float,
        total: float,
        *,
        epoch: int | None = None,
        phase: int | None = None,
        map50: float = 0.0,
        map_50_95: float = 0.0,
        per_class_ap_50: dict[int, float] | None = None,
        per_class_ap_50_95: dict[int, float] | None = None,
        per_class_precision: dict[int, float] | None = None,
        per_class_recall: dict[int, float] | None = None,
        per_class_f1: dict[int, float] | None = None,
        per_class_counts: dict[int, dict[str, int]] | None = None,
        per_class_ap_30: dict[int, float] | None = None,
        per_class_ap_center: dict[int, float] | None = None,
        per_class_center_recall: dict[int, float] | None = None,
        extra_losses: dict[str, float] | None = None,
    ) -> None:
        """Append one epoch of train and validation metrics."""
        self.epoch.append(epoch if epoch is not None else len(self.total_loss) + 1)
        self.phase.append(phase if phase is not None else 1)
        self.cls_loss.append(cls)
        self.box_loss.append(box)
        self.total_loss.append(total)
        self.map50.append(map50)
        self.map_50_95.append(map_50_95)
        self.per_class_ap_50.append(per_class_ap_50 or {})
        self.per_class_ap_50_95.append(per_class_ap_50_95 or {})
        self.per_class_precision.append(per_class_precision or {})
        self.per_class_recall.append(per_class_recall or {})
        self.per_class_f1.append(per_class_f1 or {})
        self.per_class_counts.append(per_class_counts or {})
        self.per_class_ap_30.append(per_class_ap_30 or {})
        self.per_class_ap_center.append(per_class_ap_center or {})
        self.per_class_center_recall.append(per_class_center_recall or {})

        extra_losses = extra_losses or {}
        for key in set(self.extra_losses) | set(extra_losses):
            values = self.extra_losses.setdefault(key, [float("nan")] * (len(self.total_loss) - 1))
            values.append(float(extra_losses.get(key, float("nan"))))

    def class_ids(self) -> list[int]:
        """Return sorted class IDs present in tracked AP dictionaries."""
        ids: set[int] = set()
        for per_epoch in (
            self.per_class_ap_50
            + self.per_class_ap_50_95
            + self.per_class_precision
            + self.per_class_recall
            + self.per_class_f1
            + self.per_class_ap_30
            + self.per_class_ap_center
            + self.per_class_center_recall
        ):
            ids.update(int(class_id) for class_id in per_epoch)
        return sorted(ids)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


# A matcher decides whether one prediction matches a GT of its image: it gets the
# prediction box, the (non-empty) GT boxes of the same image and class, and the
# indices of those GTs already matched, and returns the matched GT index or None.
Matcher = Callable[[torch.Tensor, torch.Tensor, set[int]], "int | None"]


def _iou_matcher(iou_threshold: float) -> Matcher:
    """Match the best-IoU GT when its IoU reaches `iou_threshold` and it is still free."""

    def match(pred_box: torch.Tensor, gt_boxes: torch.Tensor, matched: set[int]) -> int | None:
        ious = box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_gt = ious.max(dim=0)
        if best_iou >= iou_threshold and best_gt.item() not in matched:
            return best_gt.item()
        return None

    return match


def _center_inside(pred_box: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """(G,) bool: which GT box centers lie inside `pred_box` (borders included)."""
    cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    return (
        (cx >= pred_box[0]) & (cx <= pred_box[2]) & (cy >= pred_box[1]) & (cy <= pred_box[3])
    )


def _center_match(pred_box: torch.Tensor, gt_boxes: torch.Tensor, matched: set[int]) -> int | None:
    """Impact-point match: a free GT whose center the prediction box contains.

    A prediction matches at most one GT: when it contains several free centers
    it takes the one closest to its own center. A GT is matched at most once,
    so callers rank predictions by score and the first to cover it wins.
    """
    candidates = _center_inside(pred_box, gt_boxes)
    for index in matched:
        candidates[index] = False
    if not candidates.any():
        return None
    cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    distance = (cx - (pred_box[0] + pred_box[2]) / 2) ** 2 + (
        cy - (pred_box[1] + pred_box[3]) / 2
    ) ** 2
    distance[~candidates] = float("inf")
    return int(distance.argmin())


def _compute_ap(
    pred_boxes: list[torch.Tensor],
    pred_scores: list[torch.Tensor],
    pred_labels: list[torch.Tensor],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    matcher: Matcher,
    num_classes: int,
) -> tuple[float, dict[int, float]]:
    """Compute Average Precision under the match criterion `matcher`.

    Uses all-point interpolation (VOC2010+ style). Predictions are ranked by
    score over the whole dataset and matched greedily, each GT at most once and
    only within its own image.

    Args:
        pred_boxes: List of (P_i, 4) predicted boxes per image in xyxy format.
        pred_scores: List of (P_i,) confidence scores per image.
        pred_labels: List of (P_i,) predicted class IDs per image.
        gt_boxes: List of (G_i, 4) ground truth boxes per image in xyxy format.
        gt_labels: List of (G_i,) ground truth class IDs per image.
        matcher: Match criterion (see `Matcher`).
        num_classes: Total number of classes.

    Returns:
        mAP: Mean AP across all classes.
        per_class_ap: Dict mapping class_id → AP.
    """
    per_class_ap = {}

    for cls_id in range(num_classes):
        class_predictions: list[tuple[float, int, torch.Tensor]] = []
        class_gt_boxes: list[torch.Tensor] = []

        for image_idx, (boxes_i, scores_i, labels_i, gt_boxes_i, gt_labels_i) in enumerate(
            zip(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
        ):
            pred_mask = labels_i == cls_id
            for box, score in zip(boxes_i[pred_mask], scores_i[pred_mask]):
                class_predictions.append((float(score.item()), image_idx, box))

            gt_mask = gt_labels_i == cls_id
            class_gt_boxes.append(gt_boxes_i[gt_mask])

        n_gt = sum(len(boxes_i) for boxes_i in class_gt_boxes)
        if n_gt == 0:
            per_class_ap[cls_id] = 0.0
            continue

        if not class_predictions:
            per_class_ap[cls_id] = 0.0
            continue

        class_predictions.sort(key=lambda item: item[0], reverse=True)

        matched_gt: list[set[int]] = [set() for _ in class_gt_boxes]
        tp = torch.zeros(len(class_predictions))
        fp = torch.zeros(len(class_predictions))

        for i, (_, image_idx, pred_box) in enumerate(class_predictions):
            gt_boxes_i = class_gt_boxes[image_idx]
            if len(gt_boxes_i) == 0:
                fp[i] = 1
                continue

            matched = matcher(pred_box, gt_boxes_i, matched_gt[image_idx])
            if matched is not None:
                tp[i] = 1
                matched_gt[image_idx].add(matched)
            else:
                fp[i] = 1

        # Compute precision-recall curve
        cum_tp = tp.cumsum(0)
        cum_fp = fp.cumsum(0)

        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / n_gt

        # All-point interpolation (VOC2010+)
        # Prepend 0 and append 0 for proper integration
        recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

        # Make precision monotonically decreasing
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = precision[i].max(precision[i + 1])

        # Compute AP as area under PR curve
        recall_diff = recall[1:] - recall[:-1]
        ap = (recall_diff * precision[1:]).sum()

        per_class_ap[cls_id] = ap.item()

    # Mean AP across all classes
    mAP = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
    return float(mAP), per_class_ap


def _compute_ap_at_iou(
    pred_boxes: list[torch.Tensor],
    pred_scores: list[torch.Tensor],
    pred_labels: list[torch.Tensor],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    iou_threshold: float,
    num_classes: int,
) -> tuple[float, dict[int, float]]:
    """Compute Average Precision at a single IoU threshold.

    A prediction is a true positive when its best-IoU GT reaches
    `iou_threshold` and has not been matched by a higher-scored prediction.
    See `_compute_ap` for the ranking and the argument layout.
    """
    return _compute_ap(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels,
        _iou_matcher(iou_threshold), num_classes,
    )


def _compute_ap_center(
    pred_boxes: list[torch.Tensor],
    pred_scores: list[torch.Tensor],
    pred_labels: list[torch.Tensor],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    num_classes: int,
) -> tuple[float, dict[int, float]]:
    """Average Precision under the impact-point criterion.

    A prediction is a true positive when its box contains the center of a GT
    that no higher-scored prediction has matched (see `_center_match`).
    """
    return _compute_ap(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, _center_match, num_classes
    )


def _compute_center_recall(
    pred_boxes: list[torch.Tensor],
    pred_scores: list[torch.Tensor],
    pred_labels: list[torch.Tensor],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    score_threshold: float,
    num_classes: int,
) -> dict[int, float]:
    """Classwise fraction of GT centers covered by a prediction scoring >= `score_threshold`.

    Same matching as `_compute_ap_center` (greedy by score, one GT per
    prediction, one prediction per GT) restricted to the operating point.
    A class without GT has recall 0.0.
    """
    recall: dict[int, float] = {}
    for cls_id in range(num_classes):
        n_gt = covered = 0
        for boxes_i, scores_i, labels_i, gt_boxes_i, gt_labels_i in zip(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
        ):
            targets = gt_boxes_i[gt_labels_i == cls_id]
            n_gt += len(targets)
            keep = (labels_i == cls_id) & (scores_i >= score_threshold)
            if len(targets) == 0 or not keep.any():
                continue
            order = torch.sort(scores_i[keep], descending=True, stable=True).indices
            matched: set[int] = set()
            for box in boxes_i[keep][order]:
                index = _center_match(box, targets, matched)
                if index is not None:
                    matched.add(index)
            covered += len(matched)
        recall[cls_id] = covered / n_gt if n_gt else 0.0
    return recall


def _compute_operating_point(
    pred_boxes: list[torch.Tensor],
    pred_scores: list[torch.Tensor],
    pred_labels: list[torch.Tensor],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    iou_threshold: float,
    score_threshold: float,
    num_classes: int,
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, dict[str, int]]]:
    """Compute classwise precision, recall, F1 and TP/FP/FN at one threshold."""
    precision: dict[int, float] = {}
    recall: dict[int, float] = {}
    f1: dict[int, float] = {}
    counts: dict[int, dict[str, int]] = {}

    for cls_id in range(num_classes):
        tp = fp = fn = 0
        for boxes_i, scores_i, labels_i, gt_boxes_i, gt_labels_i in zip(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
        ):
            pred_mask = (labels_i == cls_id) & (scores_i >= score_threshold)
            predictions = list(zip(scores_i[pred_mask], boxes_i[pred_mask]))
            predictions.sort(key=lambda item: float(item[0]), reverse=True)
            targets = gt_boxes_i[gt_labels_i == cls_id]
            matched: set[int] = set()
            for _, box in predictions:
                if len(targets) == 0:
                    fp += 1
                    continue
                ious = box_iou(box.unsqueeze(0), targets).squeeze(0)
                best_iou, best_idx = ious.max(dim=0)
                if best_iou >= iou_threshold and int(best_idx) not in matched:
                    tp += 1
                    matched.add(int(best_idx))
                else:
                    fp += 1
            fn += len(targets) - len(matched)

        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision[cls_id] = p
        recall[cls_id] = r
        f1[cls_id] = 2 * p * r / (p + r) if p + r else 0.0
        counts[cls_id] = {"tp": tp, "fp": fp, "fn": fn}
    return precision, recall, f1, counts


def compute_map(
    pred_boxes: list[torch.Tensor],
    pred_scores: list[torch.Tensor],
    pred_labels: list[torch.Tensor],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    num_classes: int = 2,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> dict:
    """Compute mAP metrics across a dataset.

    AP (`_compute_ap_at_iou`) ranks the predictions it is given by score, so
    it is only as complete as the decode that produced them: predictions
    decoded at a threshold t yield a precision-recall curve that stops at
    the recall reached at t, and AP 0.0 when nothing scores above t.
    `Trainer.evaluate` therefore decodes at `eval_conf_threshold` (0.001)
    for AP. `score_threshold` only gates the classwise
    precision/recall/F1/TP-FP-FN operating point, read off that same list
    (training-loop spec: "Configurable Confidence and NMS Thresholds").

    Args:
        pred_boxes: List of (P_i, 4) predicted boxes per image (cxcywh normalized).
        pred_scores: List of (P_i,) confidence scores per image.
        pred_labels: List of (P_i,) predicted class IDs per image.
        gt_boxes: List of (G_i, 4) GT boxes per image (cxcywh normalized).
        gt_labels: List of (G_i,) GT class IDs per image.
        num_classes: Number of classes.
        iou_threshold: IoU threshold for mAP@IoU computation.
        score_threshold: Confidence threshold for the precision/recall/F1
            operating point (the config's `conf_threshold`). Must be at or
            above the threshold the predictions were decoded at.

    Returns:
        Dict with mAP, AP, and classwise precision/recall/F1 at IoU=0.5 and
        `score_threshold`, followed by the localization-tolerant per-class
        metrics: `per_class_ap_30` (AP at IoU 0.3), `per_class_ap_center` (AP
        under the impact-point criterion: the prediction box contains the GT
        box center) and `per_class_center_recall` (that criterion's recall at
        `score_threshold`).
    """
    pred_boxes_xyxy = [
        _cxcywh_to_xyxy(boxes) if len(boxes) > 0 else boxes
        for boxes in pred_boxes
    ]
    gt_boxes_xyxy = [
        _cxcywh_to_xyxy(boxes) if len(boxes) > 0 else boxes
        for boxes in gt_boxes
    ]

    # mAP@0.5
    map50, per_class_50 = _compute_ap_at_iou(
        pred_boxes_xyxy, pred_scores, pred_labels,
        gt_boxes_xyxy, gt_labels, 0.5, num_classes,
    )

    # mAP@0.5:0.95 (COCO-style, 10 thresholds)
    thresholds = [0.50 + 0.05 * i for i in range(10)]
    maps_per_thresh = []
    per_class_95 = {c: 0.0 for c in range(num_classes)}

    for thresh in thresholds:
        mAP_t, pc_t = _compute_ap_at_iou(
            pred_boxes_xyxy, pred_scores, pred_labels,
            gt_boxes_xyxy, gt_labels, thresh, num_classes,
        )
        maps_per_thresh.append(mAP_t)
        for c in range(num_classes):
            per_class_95[c] += pc_t.get(c, 0.0)

    map_50_95 = float(np.mean(maps_per_thresh))
    for c in per_class_95:
        per_class_95[c] /= len(thresholds)

    precision, recall, f1, counts = _compute_operating_point(
        pred_boxes_xyxy,
        pred_scores,
        pred_labels,
        gt_boxes_xyxy,
        gt_labels,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        num_classes=num_classes,
    )

    # GT-count accounting integrity (training-metrics spec "GT-Count Accounting
    # Integrity"): tp + fn must equal the number of GT instances for that
    # class. This holds algebraically by construction in
    # `_compute_operating_point` (fn += len(targets) - len(matched) per
    # image), so a mismatch here means a discrepancy was introduced upstream
    # (e.g. a corrupted split) — log it explicitly instead of silently
    # trusting the metric.
    for cls_id, count_dict in counts.items():
        gt_total = sum(int((labels_i == cls_id).sum()) for labels_i in gt_labels)
        observed = count_dict["tp"] + count_dict["fn"]
        if observed != gt_total:
            logger.warning(
                "GT-count mismatch for class %d: tp+fn=%d but %d GT instances "
                "exist in this split. This is not expected by construction — "
                "investigate before trusting this run's metrics for this class.",
                cls_id, observed, gt_total,
            )

    _, per_class_30 = _compute_ap_at_iou(
        pred_boxes_xyxy, pred_scores, pred_labels,
        gt_boxes_xyxy, gt_labels, LOOSE_IOU_THRESHOLD, num_classes,
    )
    _, per_class_center = _compute_ap_center(
        pred_boxes_xyxy, pred_scores, pred_labels, gt_boxes_xyxy, gt_labels, num_classes,
    )
    center_recall = _compute_center_recall(
        pred_boxes_xyxy, pred_scores, pred_labels, gt_boxes_xyxy, gt_labels,
        score_threshold, num_classes,
    )

    return {
        "map50": map50,
        "map_50_95": map_50_95,
        "per_class_ap_50": per_class_50,
        "per_class_ap_50_95": per_class_95,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "per_class_counts": counts,
        "per_class_ap_30": per_class_30,
        "per_class_ap_center": per_class_center,
        "per_class_center_recall": center_recall,
    }


def _write_metrics_csv(history: LossHistory, output_dir: Path) -> None:
    """Persist epoch metrics as a CSV file for notebooks and thesis tables.

    The localization-tolerant columns (`ap30_class_k`, `apcenter_class_k`,
    `center_recall_class_k`) come last and only when the history carries them,
    so a history built without them keeps the columns it always had.
    """
    class_ids = history.class_ids()
    has_localization = any(
        history.per_class_ap_30 + history.per_class_ap_center + history.per_class_center_recall
    )
    fieldnames = [
        "epoch",
        "phase",
        "cls_loss",
        "box_loss",
        "total_loss",
        *history.extra_losses.keys(),
        "map50",
        "map_50_95",
        *[f"ap50_class_{class_id}" for class_id in class_ids],
        *[f"ap50_95_class_{class_id}" for class_id in class_ids],
        *[f"precision_class_{class_id}" for class_id in class_ids],
        *[f"recall_class_{class_id}" for class_id in class_ids],
        *[f"f1_class_{class_id}" for class_id in class_ids],
        *[f"tp_class_{class_id}" for class_id in class_ids],
        *[f"fp_class_{class_id}" for class_id in class_ids],
        *[f"fn_class_{class_id}" for class_id in class_ids],
        *(
            [
                *[f"ap30_class_{class_id}" for class_id in class_ids],
                *[f"apcenter_class_{class_id}" for class_id in class_ids],
                *[f"center_recall_class_{class_id}" for class_id in class_ids],
            ]
            if has_localization
            else []
        ),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, epoch in enumerate(history.epoch):
            row = {
                "epoch": epoch,
                "phase": history.phase[idx],
                "cls_loss": history.cls_loss[idx],
                "box_loss": history.box_loss[idx],
                "total_loss": history.total_loss[idx],
                "map50": history.map50[idx],
                "map_50_95": history.map_50_95[idx],
            }
            for name, values in history.extra_losses.items():
                row[name] = values[idx]
            for class_id in class_ids:
                row[f"ap50_class_{class_id}"] = history.per_class_ap_50[idx].get(class_id, 0.0)
                row[f"ap50_95_class_{class_id}"] = history.per_class_ap_50_95[idx].get(class_id, 0.0)
                row[f"precision_class_{class_id}"] = history.per_class_precision[idx].get(class_id, 0.0)
                row[f"recall_class_{class_id}"] = history.per_class_recall[idx].get(class_id, 0.0)
                row[f"f1_class_{class_id}"] = history.per_class_f1[idx].get(class_id, 0.0)
                counts = history.per_class_counts[idx].get(class_id, {})
                row[f"tp_class_{class_id}"] = counts.get("tp", 0)
                row[f"fp_class_{class_id}"] = counts.get("fp", 0)
                row[f"fn_class_{class_id}"] = counts.get("fn", 0)
                if has_localization:
                    row[f"ap30_class_{class_id}"] = history.per_class_ap_30[idx].get(class_id, 0.0)
                    row[f"apcenter_class_{class_id}"] = history.per_class_ap_center[idx].get(
                        class_id, 0.0
                    )
                    row[f"center_recall_class_{class_id}"] = history.per_class_center_recall[
                        idx
                    ].get(class_id, 0.0)
            writer.writerow(row)


def _plot_losses(history: LossHistory, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = history.epoch
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history.total_loss, label="total_loss", linewidth=2)
    ax.plot(epochs, history.cls_loss, label="cls_loss", linewidth=2)
    ax.plot(epochs, history.box_loss, label="box_loss", linewidth=2)
    for name, values in history.extra_losses.items():
        ax.plot(epochs, values, label=name, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training losses")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_map(history: LossHistory, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = history.epoch
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history.map50, label="mAP@0.5", linewidth=2)
    ax.plot(epochs, history.map_50_95, label="mAP@0.5:0.95", linewidth=2)
    if history.map50:
        best_idx = int(np.argmax(history.map50))
        ax.scatter(
            [epochs[best_idx]],
            [history.map50[best_idx]],
            zorder=3,
            label=f"best mAP@0.5 epoch {epochs[best_idx]}: {history.map50[best_idx]:.4f}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Validation mAP")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "map_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_class_ap(history: LossHistory, output_dir: Path, key: str) -> None:
    import matplotlib.pyplot as plt

    class_ids = history.class_ids()
    if not class_ids:
        return

    source = history.per_class_ap_50 if key == "ap50" else history.per_class_ap_50_95
    title = "Per-class AP@0.5" if key == "ap50" else "Per-class AP@0.5:0.95"
    filename = "per_class_ap50.png" if key == "ap50" else "per_class_ap50_95.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    for class_id in class_ids:
        values = [per_epoch.get(class_id, 0.0) for per_epoch in source]
        ax.plot(history.epoch, values, label=f"class_{class_id}", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_detection_quality(history: LossHistory, output_dir: Path) -> None:
    """Plot operating-point quality for each class at IoU=0.5."""
    import matplotlib.pyplot as plt

    class_ids = history.class_ids()
    if not class_ids:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for class_id in class_ids:
        axes[0].plot(history.epoch, [x.get(class_id, 0.0) for x in history.per_class_precision], label=f"class_{class_id}")
        axes[1].plot(history.epoch, [x.get(class_id, 0.0) for x in history.per_class_recall], label=f"class_{class_id}")
        axes[2].plot(history.epoch, [x.get(class_id, 0.0) for x in history.per_class_f1], label=f"class_{class_id}")
    for axis, title in zip(axes, ("Precision@0.5", "Recall@0.5", "F1@0.5")):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(output_dir / "detection_quality_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_training_curves(history: LossHistory, output_dir: Path) -> None:
    """Generate CSV and PNG training reports from LossHistory.

    Outputs:
        - metrics_history.csv
        - training_curves.png (backward-compatible summary)
        - loss_curves.png
        - map_curves.png
        - per_class_ap50.png
        - per_class_ap50_95.png
        - detection_quality_curves.png

    Args:
        history: LossHistory with per-epoch loss and validation data.
        output_dir: Directory to save generated artifacts.
    """
    if not history.total_loss:
        logger.warning("No metric data to plot - skipping training curves")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable, writing CSV only")
        _write_metrics_csv(history, Path(output_dir))
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_csv(history, output_dir)

    try:
        _plot_losses(history, output_dir)
        _plot_map(history, output_dir)
        _plot_per_class_ap(history, output_dir, "ap50")
        _plot_per_class_ap(history, output_dir, "ap50_95")
        _plot_detection_quality(history, output_dir)

        epochs = history.epoch
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(epochs, history.total_loss, label="total_loss", linewidth=2)
        axes[0, 0].plot(epochs, history.cls_loss, label="cls_loss", linewidth=2)
        axes[0, 0].plot(epochs, history.box_loss, label="box_loss", linewidth=2)
        for name, values in history.extra_losses.items():
            axes[0, 0].plot(epochs, values, label=name, linewidth=2)
        axes[0, 0].set_title("Training losses")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, history.map50, label="mAP@0.5", linewidth=2)
        axes[0, 1].plot(epochs, history.map_50_95, label="mAP@0.5:0.95", linewidth=2)
        axes[0, 1].set_title("Validation mAP")
        axes[0, 1].set_ylabel("mAP")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        class_ids = history.class_ids()
        for class_id in class_ids:
            axes[1, 0].plot(
                epochs,
                [per_epoch.get(class_id, 0.0) for per_epoch in history.per_class_ap_50],
                label=f"class_{class_id}",
                linewidth=2,
            )
            axes[1, 1].plot(
                epochs,
                [per_epoch.get(class_id, 0.0) for per_epoch in history.per_class_ap_50_95],
                label=f"class_{class_id}",
                linewidth=2,
            )
        axes[1, 0].set_title("Per-class AP@0.5")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("AP")
        axes[1, 0].grid(True, alpha=0.3)
        if class_ids:
            axes[1, 0].legend()

        axes[1, 1].set_title("Per-class AP@0.5:0.95")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("AP")
        axes[1, 1].grid(True, alpha=0.3)
        if class_ids:
            axes[1, 1].legend()

        plt.tight_layout()
        fig.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Training metrics saved to %s", output_dir)

    except (RuntimeError, ValueError) as e:
        logger.warning("Could not generate training curves: %s", e)
