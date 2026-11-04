"""Shared detection decode: raw anchor-free YOLO outputs -> filtered boxes.

This module is the single source of truth for turning raw model outputs
(`preds`, `cls_preds` per FPN level) into detections. Both
`Trainer._decode_predictions` (src/training/loop.py) and
`scripts/visualize_damage_predictions.py` delegate to `decode_detections`
so they cannot silently drift apart (design.md D3; training-loop spec
"Decode Consistency Across Consumers").

Implements:
    - Per-class threshold emission (D5/E3): one candidate per (class, anchor)
      pair whose score exceeds `conf_threshold`, instead of only the argmax
      class per anchor.
    - Per-class NMS (D4/E1) via `torchvision.ops.batched_nms`, so a
      high-confidence box of one class cannot suppress an overlapping box of
      a different class (e.g. damage fully nested inside a mango box).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torchvision.ops import batched_nms


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def decode_detections(
    preds: list[torch.Tensor],
    cls_preds: list[torch.Tensor],
    *,
    batch_idx: int,
    num_classes: int,
    image_size: int,
    strides: Sequence[int],
    conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    nms_enabled: bool = True,
    per_class_candidates: bool = True,
    max_detections: int = 300,
    normalize: bool = True,
    ignore_classes: Sequence[int] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode raw anchor-free YOLO outputs into detections for one image.

    Args:
        preds: list of (B, nc+4, H_i, W_i) raw predictions per FPN level.
            The first `num_classes` channels are unused here (class logits
            live in `cls_preds`); the last 4 are bbox regression deltas.
        cls_preds: list of (B, nc, H_i, W_i) classification logits per level.
        batch_idx: which image in the batch to decode.
        num_classes: number of detection classes.
        image_size: input image size (square), used to normalize output boxes.
        strides: FPN level strides, aligned with `preds`/`cls_preds` order.
            Required, no default (fusion-redesign D-D) — a forgotten
            argument is a `TypeError` instead of a silent `[8, 16, 32]`
            default that misaligns a 4-level checkpoint's predictions.
        conf_threshold: minimum per-class sigmoid score to keep a candidate.
            Default 0.25 matches the legacy hardcoded threshold.
        nms_iou_threshold: IoU threshold for per-class NMS suppression.
        nms_enabled: whether to run NMS at all. Set False to reproduce the
            legacy (no-NMS) decode for A/B comparison (E1).
        per_class_candidates: if True (default), emit one candidate per
            qualifying class per anchor (D5/E3 — repairs argmax suppression
            of nested damage boxes). If False, emit only the argmax class per
            anchor (legacy `scores.max(dim=0)` behavior, E3 comparison arm).
        max_detections: hard cap on returned detections after NMS/sorting.
        normalize: if True (default), boxes are cx/cy/w/h normalized to
            [0, 1] by `image_size`; if False, boxes are pixel coordinates.

    Returns:
        boxes: (P, 4) cxcywh detections.
        scores: (P,) confidence scores.
        labels: (P,) integer class IDs.
    """
    if not (len(preds) == len(cls_preds) == len(strides)):
        raise ValueError(
            f"preds/cls_preds/strides must have matching lengths: got "
            f"{len(preds)} pred levels, {len(cls_preds)} cls levels, "
            f"{len(strides)} strides."
        )

    all_boxes: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for pred, cls_pred, stride in zip(preds, cls_preds, strides):
        p = pred[batch_idx]  # (nc+4, H, W)
        c = cls_pred[batch_idx]  # (nc, H, W)
        reg = p[num_classes:]  # (4, H, W)

        scores = c.sigmoid()  # (nc, H, W)
        if len(ignore_classes) > 0:
            # An untrained channel must not take detections from the capped list.
            scores = scores.clone()
            scores[list(ignore_classes)] = 0.0

        if per_class_candidates:
            mask = scores > conf_threshold  # (nc, H, W)
            if not mask.any():
                continue
            cls_idx, ys, xs = mask.nonzero(as_tuple=True)
            cand_scores = scores[cls_idx, ys, xs]
            cand_labels = cls_idx
        else:
            max_scores, max_labels = scores.max(dim=0)  # (H, W)
            mask = max_scores > conf_threshold
            if not mask.any():
                continue
            ys, xs = mask.nonzero(as_tuple=True)
            cand_scores = max_scores[ys, xs]
            cand_labels = max_labels[ys, xs]

        dx = reg[0, ys, xs]
        dy = reg[1, ys, xs]
        w = reg[2, ys, xs].exp()
        h = reg[3, ys, xs].exp()

        anchor_x = (xs.float() + 0.5) * stride
        anchor_y = (ys.float() + 0.5) * stride

        cx = anchor_x + dx
        cy = anchor_y + dy

        boxes = torch.stack([cx, cy, w, h], dim=1)  # (K, 4) pixel cxcywh

        all_boxes.append(boxes)
        all_scores.append(cand_scores)
        all_labels.append(cand_labels.long())

    if not all_boxes:
        return (
            torch.zeros((0, 4)),
            torch.zeros((0,)),
            torch.zeros((0,), dtype=torch.long),
        )

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    if nms_enabled and boxes.shape[0] > 0:
        boxes_xyxy = _cxcywh_to_xyxy(boxes)
        keep = batched_nms(boxes_xyxy, scores, labels, nms_iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    if boxes.shape[0] > max_detections:
        top_scores, top_idx = scores.topk(max_detections)
        boxes = boxes[top_idx]
        scores = top_scores
        labels = labels[top_idx]

    if normalize:
        boxes = boxes / image_size

    return boxes, scores, labels
