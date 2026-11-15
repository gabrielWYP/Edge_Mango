"""
YOLOv8-style detection loss with Task-Aligned Assigner (TAL).

Implements:
    - TaskAlignedAssigner: matches predictions to GT using alignment metric
    - YOLOv8Loss: class-weighted BCE + CIoU regression loss

References:
    - TOOD: Task-aligned One-stage Object Detection (ICCV 2021)
    - YOLOv8: https://github.com/ultralytics/ultralytics
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchvision.ops import complete_box_iou_loss


class TaskAlignedAssigner:
    """Task-Aligned Assigner for YOLOv8-style detection.

    Computes alignment metric = cls_score^alpha * iou^beta, selects top-k
    positive anchors per ground truth box.

    Spatial admissibility (D6/A1) is applied as a mask on the alignment
    metric BEFORE top-k selection: an anchor is admissible if its center is
    inside the GT box, OR within `center_radius * stride` of the GT center
    (center_radius=0.0, the default, reduces to strict containment — D9
    legacy behavior). An optional per-level size-bin mask (D8/A3) can further
    restrict candidates to the FPN level(s) appropriate for the GT's size.
    Masking before top-k (rather than filtering the already-selected top-k
    afterward, the pre-fix behavior) means top-k always returns the best
    *usable* candidates instead of wasting budget on inadmissible ones.

    The "keep at least 1 anchor per GT" fallback (D7/A2) runs AFTER these
    masks, ranked by anchor centers (not predicted-box centers, the pre-fix
    coordinate-source bug), so a GT is never silently dropped and the
    fallback's anchors can never be subsequently discarded by the filter that
    used to run after it.

    Args:
        topk: Number of candidate anchors per GT.
        alpha: Exponent for classification score in alignment metric.
        beta: Exponent for IoU in alignment metric.
        center_radius: Center-sampling tolerance in stride units. 0.0 (default)
            = legacy strict anchor-center-inside-GT containment (D9).
        level_ranges: Open-ended FCOS-style size bins in pixels, e.g.
            `[64, 128]` -> max(w,h)<64 admits level 0 only, <128 admits level
            1 only, else level 2 only. `None` disables level restriction.
        collect_stats: Non-destructive instrumentation (A4) — when True,
            accumulates per-class, per-level positive-anchor counts in
            `self.last_stats` without altering any assignment output. Off by
            default. Call `reset_stats()` before a fresh instrumentation pass.
    """

    def __init__(
        self,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        center_radius: float = 0.0,
        level_ranges: list[float] | None = None,
        collect_stats: bool = False,
    ):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.center_radius = center_radius
        self.level_ranges = level_ranges
        self.collect_stats = collect_stats
        self.last_stats: dict[tuple[int, int], int] = {}
        # (B, num_anchors) index of the GT each anchor was assigned to in the last call, -1 for
        # negatives. The mask loss reads it to pick every positive anchor's instance mask.
        self.last_gt_idx: torch.Tensor | None = None

    def reset_stats(self) -> None:
        """Clear accumulated instrumentation counts (A4)."""
        self.last_stats = {}

    @torch.no_grad()
    def __call__(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        gt_labels: list[torch.Tensor] | torch.Tensor,
        gt_bboxes: list[torch.Tensor] | torch.Tensor,
        anchors: torch.Tensor,
        strides: list[int],
        anchor_strides: torch.Tensor | None = None,
        num_per_level: list[int] | None = None,
        gt_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign predictions to ground truth targets, for the whole batch at once.

        Every per-image / per-GT quantity is a batched tensor op over padded GT: there is no
        Python loop over images or GTs and no host synchronisation (`.item()`, `.any()`,
        `nonzero`) on the hot path. The result is the one the per-image loop produced
        (`src/training/loss_reference.py`): same candidates, same fallback, same conflict
        winner (highest alignment score, the lowest GT index on a tie).

        Args:
            pred_scores: (B, num_anchors, num_classes) predicted class scores (sigmoid).
            pred_bboxes: (B, num_anchors, 4) predicted bboxes in cxcywh format.
            gt_labels: List of (N_i,) GT class labels per image, or one padded (B, N) tensor
                together with `gt_counts`.
            gt_bboxes: List of (N_i, 4) GT bboxes per image in cxcywh format, or one padded
                (B, N, 4) tensor together with `gt_counts`.
            anchors: (num_anchors, 2) anchor center points in pixel coords.
            strides: List of stride values per FPN level (retained for
                backward-compat call signatures; `anchor_strides` is the
                per-anchor source of truth when provided).
            anchor_strides: (num_anchors,) stride value per anchor, from
                `_generate_anchors`. If omitted, every anchor is assumed to
                use `strides[0]` (single-level backward-compat fallback).
            num_per_level: Anchor count per FPN level, from `_generate_anchors`.
                Required for the per-level mask (D8) and instrumentation (A4);
                omit to skip both (e.g. single-level synthetic test inputs).
            gt_counts: (B,) number of real GTs per image; required with padded tensors.

        Returns:
            target_classes: (B, num_anchors) assigned class IDs (-1 for negative).
            target_bboxes: (B, num_anchors, 4) assigned GT bboxes.
            target_scores: (B, num_anchors) alignment scores for positive samples.
            fg_mask: (B, num_anchors) boolean mask of foreground (positive) anchors.
        """
        B, num_anchors, num_classes = pred_scores.shape
        device = pred_scores.device

        if isinstance(gt_labels, torch.Tensor):
            if gt_counts is None:
                raise ValueError("padded gt_labels/gt_bboxes need gt_counts")
            labels, boxes = gt_labels.to(device), gt_bboxes.to(device)
            counts = gt_counts.to(device)
        else:
            boxes, labels, counts, _, _ = pad_targets(
                {"bboxes": gt_bboxes, "labels": gt_labels}, device=device
            )
        N = labels.shape[1]
        gt_valid = torch.arange(N, device=device).unsqueeze(0) < counts.unsqueeze(1)  # (B, N)

        target_classes = torch.full((B, num_anchors), -1, dtype=torch.long, device=device)
        target_bboxes = torch.zeros((B, num_anchors, 4), device=device)
        target_scores = torch.zeros((B, num_anchors), device=device)
        fg_mask = torch.zeros((B, num_anchors), dtype=torch.bool, device=device)
        target_gt_idx = torch.full((B, num_anchors), -1, dtype=torch.long, device=device)
        self.last_gt_idx = target_gt_idx
        if N == 0:
            return target_classes, target_bboxes, target_scores, fg_mask

        if anchor_strides is None:
            anchor_strides = torch.full((num_anchors,), float(strides[0]), device=device)

        level_ids = (
            self._anchor_level_ids(num_per_level, num_anchors, device)
            if num_per_level is not None
            else None
        )

        # Alignment metric score^alpha * iou^beta for every (image, anchor, GT).
        safe_labels = labels.clamp(0, num_classes - 1)
        gt_cls_scores = pred_scores.gather(
            2, safe_labels.unsqueeze(1).expand(B, num_anchors, N)
        )  # (B, A, N)
        ious = self._batched_box_iou(
            self._cxcywh_to_xyxy(pred_bboxes), self._cxcywh_to_xyxy(boxes)
        )  # (B, A, N)
        alignment = (gt_cls_scores ** self.alpha) * (ious ** self.beta)

        # D6: spatial admissibility mask, applied BEFORE top-k.
        admissible = self._spatial_admissibility_batched(
            anchors, anchor_strides, boxes, self.center_radius
        )  # (B, A, N)
        level_mask = None
        if level_ids is not None and self.level_ranges:
            target_level = self._target_levels(boxes, self.level_ranges)  # (B, N)
            level_mask = level_ids.view(1, -1, 1) == target_level.unsqueeze(1)  # (B, A, N)
            admissible = admissible & level_mask
        admissible = admissible & gt_valid.unsqueeze(1)
        alignment = alignment.masked_fill(~admissible, 0.0)

        topk = min(self.topk, num_anchors)
        topk_vals, topk_idx = alignment.transpose(1, 2).topk(topk, dim=-1)  # (B, N, k)
        valid = topk_vals > 0

        # D7: a GT with no usable candidate falls back to the anchors closest to its center,
        # AFTER the spatial/level filter, ranked by anchor centers (never predicted boxes).
        need_fallback = gt_valid & ~valid.any(dim=-1)  # (B, N)
        dist = (
            (anchors[:, 0].view(1, 1, -1) - boxes[:, :, 0:1]) ** 2
            + (anchors[:, 1].view(1, 1, -1) - boxes[:, :, 1:2]) ** 2
        )  # (B, N, A)
        if level_mask is not None:
            lm = level_mask.transpose(1, 2)  # (B, N, A)
            restrict = lm.any(dim=-1, keepdim=True) & ~lm
            dist = dist.masked_fill(restrict, float("inf"))
        n_fallback = min(3, num_anchors)
        _, fallback_idx = dist.topk(n_fallback, dim=-1, largest=False)  # (B, N, f)

        width = max(topk, n_fallback)
        pad = width - n_fallback
        fb = need_fallback.unsqueeze(-1)
        main_idx = F.pad(topk_idx, (0, width - topk))
        main_score = F.pad(topk_vals, (0, width - topk))
        main_valid = F.pad(valid, (0, width - topk)) & ~fb
        fb_idx = F.pad(fallback_idx, (0, pad))
        fb_valid = F.pad(torch.ones_like(fallback_idx, dtype=torch.bool), (0, pad)) & fb
        cand_idx = torch.where(fb, fb_idx, main_idx)
        cand_valid = main_valid | fb_valid
        cand_score = torch.where(fb, torch.full_like(main_score, 0.01), main_score)

        # Conflict resolution: an anchor taken by several GTs keeps the one with the highest
        # alignment score; the earliest GT index wins an exact tie (what the sequential
        # `target_scores < pos_scores` update did). Invalid slots scatter into a spare column.
        dense = torch.zeros((B, N, num_anchors + 1), device=device)
        slot = torch.where(cand_valid, cand_idx, torch.full_like(cand_idx, num_anchors))
        dense.scatter_(2, slot, torch.where(cand_valid, cand_score, torch.zeros_like(cand_score)))
        best_score, best_gt = dense[:, :, :num_anchors].max(dim=1)  # (B, A)
        fg_mask = best_score > 0
        target_gt_idx = torch.where(fg_mask, best_gt, torch.full_like(best_gt, -1))
        self.last_gt_idx = target_gt_idx
        target_classes = torch.where(
            fg_mask, labels.gather(1, best_gt), torch.full_like(best_gt, -1)
        )
        target_bboxes = torch.where(
            fg_mask.unsqueeze(-1),
            boxes.gather(1, best_gt.unsqueeze(-1).expand(B, num_anchors, 4)),
            torch.zeros((), device=device),
        )
        target_scores = torch.where(fg_mask, best_score, torch.zeros_like(best_score))

        if self.collect_stats and level_ids is not None:
            self._accumulate_stats(labels, cand_idx, cand_valid, level_ids)

        return target_classes, target_bboxes, target_scores, fg_mask

    def _accumulate_stats(
        self,
        labels: torch.Tensor,
        cand_idx: torch.Tensor,
        cand_valid: torch.Tensor,
        level_ids: torch.Tensor,
    ) -> None:
        """A4: count every candidate (also those that lost a conflict) per (class, level).

        Instrumentation only, so one host transfer per call is fine.
        """
        n_levels = int(level_ids.max().item()) + 1
        n_classes = int(labels.max().item()) + 1
        levels = level_ids[cand_idx]  # (B, N, W)
        flat = (labels.unsqueeze(-1) * n_levels + levels)[cand_valid]
        counts = torch.bincount(flat, minlength=n_classes * n_levels).tolist()
        for key, count in enumerate(counts):
            if count:
                pair = (key // n_levels, key % n_levels)
                self.last_stats[pair] = self.last_stats.get(pair, 0) + count

    @staticmethod
    def _batched_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """IoU of (B, A, 4) against (B, N, 4) xyxy boxes -> (B, A, N); torchvision's formula."""
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # (B, A)
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # (B, N)
        lt = torch.max(boxes1[:, :, None, :2], boxes2[:, None, :, :2])
        rb = torch.min(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area1[:, :, None] + area2[:, None, :] - inter
        return inter / union

    @staticmethod
    def _spatial_admissibility_batched(
        anchors: torch.Tensor,
        anchor_strides: torch.Tensor,
        gt_boxes: torch.Tensor,
        center_radius: float,
    ) -> torch.Tensor:
        """D6 for every GT of every image: (B, A, N) bool. See `_spatial_admissibility`."""
        gt_cxcy = gt_boxes[..., :2]  # (B, N, 2)
        gt_wh = gt_boxes[..., 2:]
        gt_min = (gt_cxcy - gt_wh / 2).unsqueeze(1)  # (B, 1, N, 2)
        gt_max = (gt_cxcy + gt_wh / 2).unsqueeze(1)
        ax = anchors[:, 0].view(1, -1, 1)
        ay = anchors[:, 1].view(1, -1, 1)
        inside = (
            (ax >= gt_min[..., 0]) & (ax <= gt_max[..., 0])
            & (ay >= gt_min[..., 1]) & (ay <= gt_max[..., 1])
        )
        if center_radius <= 0:
            return inside
        radius_px = center_radius * anchor_strides.view(1, -1, 1)
        dist = (
            (ax - gt_cxcy[..., 0].unsqueeze(1)) ** 2 + (ay - gt_cxcy[..., 1].unsqueeze(1)) ** 2
        ).sqrt()
        return inside | (dist <= radius_px)

    @staticmethod
    def _target_levels(gt_boxes: torch.Tensor, level_ranges: list[float]) -> torch.Tensor:
        """D8 on device: (B, N) FPN level each GT is restricted to (see `_level_admissibility`)."""
        size = torch.maximum(gt_boxes[..., 2], gt_boxes[..., 3])  # (B, N)
        bounds = torch.tensor(level_ranges, dtype=size.dtype, device=size.device)
        below = size.unsqueeze(-1) < bounds  # (B, N, K)
        first = below.int().argmax(dim=-1)  # first bound the size is under
        return torch.where(below.any(dim=-1), first, torch.full_like(first, len(level_ranges)))

    @staticmethod
    def _spatial_admissibility(
        anchors: torch.Tensor,
        anchor_strides: torch.Tensor,
        gt_box: torch.Tensor,
        center_radius: float,
    ) -> torch.Tensor:
        """D6: anchor admissible if inside the GT box OR within
        `center_radius * stride` of the GT center. `center_radius<=0`
        reduces to strict containment (D9 legacy default)."""
        gt_cxcy = gt_box[:2]
        gt_wh = gt_box[2:]
        gt_min = gt_cxcy - gt_wh / 2
        gt_max = gt_cxcy + gt_wh / 2

        inside = (
            (anchors[:, 0] >= gt_min[0]) &
            (anchors[:, 0] <= gt_max[0]) &
            (anchors[:, 1] >= gt_min[1]) &
            (anchors[:, 1] <= gt_max[1])
        )

        if center_radius <= 0:
            return inside

        radius_px = center_radius * anchor_strides
        dist = ((anchors[:, 0] - gt_cxcy[0]) ** 2 + (anchors[:, 1] - gt_cxcy[1]) ** 2).sqrt()
        return inside | (dist <= radius_px)

    @staticmethod
    def _level_admissibility(
        level_ids: torch.Tensor,
        gt_w: float,
        gt_h: float,
        level_ranges: list[float],
    ) -> torch.Tensor:
        """D8: open-ended FCOS-style size bins.

        `level_ranges=[64, 128]`: max(w,h)<64 -> level 0 (P3/stride8) only,
        <128 -> level 1 (P4/stride16) only, else -> level 2 (P5/stride32) only.
        """
        size = max(gt_w, gt_h)
        target_level = len(level_ranges)
        for i, bound in enumerate(level_ranges):
            if size < bound:
                target_level = i
                break
        return level_ids == target_level

    @staticmethod
    def _anchor_level_ids(
        num_per_level: list[int],
        num_anchors: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Map each anchor index to its FPN level index (0=P3, 1=P4, 2=P5, ...)."""
        ids = torch.zeros(num_anchors, dtype=torch.long, device=device)
        offset = 0
        for level_idx, count in enumerate(num_per_level):
            ids[offset:offset + count] = level_idx
            offset += count
        return ids

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)


def pad_targets(
    targets: dict, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Per-image target lists -> padded tensors, with no host synchronisation.

    Args:
        targets: dict with `bboxes` (list of (N_i, 4)) and `labels` (list of (N_i,)), plus the
            optional `masks` (list of (N_i, h, w)) and `has_mask` (list of (N_i,)).
        device: Where to build the padded tensors; defaults to where the lists live.

    Returns:
        boxes (B, N, 4) zero padded, labels (B, N) long (0 padded), counts (B,) long, masks
        (B, N, h, w) or None, has_mask (B, N) bool (False padded) or None. N is the largest
        per-image count, read from the list shapes (host ints).
    """
    bboxes, labels = list(targets["bboxes"]), list(targets["labels"])
    if device is not None:
        bboxes = [b.to(device) for b in bboxes]
        labels = [l.to(device) for l in labels]
    dev = bboxes[0].device
    counts = torch.tensor([len(b) for b in bboxes], dtype=torch.long, device=dev)
    n_max = max(len(b) for b in bboxes)
    boxes = pad_sequence(bboxes, batch_first=True) if n_max else bboxes[0].new_zeros(len(bboxes), 0, 4)
    lab = (
        pad_sequence([l.long() for l in labels], batch_first=True)
        if n_max
        else torch.zeros(len(bboxes), 0, dtype=torch.long, device=dev)
    )
    masks = has_mask = None
    if "masks" in targets:
        masks = pad_sequence([m.to(dev) for m in targets["masks"]], batch_first=True)
        if "has_mask" in targets:
            has_mask = pad_sequence([h.to(dev) for h in targets["has_mask"]], batch_first=True)
        else:
            has_mask = torch.arange(n_max, device=dev).unsqueeze(0) < counts.unsqueeze(1)
    return boxes, lab, counts, masks, has_mask


def _generate_anchors(
    feat_sizes: list[tuple[int, int]],
    strides: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Generate anchor center points (and per-anchor strides) for all FPN levels.

    Args:
        feat_sizes: List of (H_i, W_i) per FPN level.
        strides: List of stride values per level.
        device: Target device.

    Returns:
        anchors: (num_anchors, 2) center points in pixel coords.
        anchor_strides: (num_anchors,) stride value for each anchor — needed
            by the assigner's center-sampling radius (D6/A1), which is
            expressed in stride units per anchor.
        num_per_level: Number of anchors per level.
    """
    all_anchors = []
    all_strides = []
    num_per_level = []

    for (h, w), stride in zip(feat_sizes, strides):
        y = (torch.arange(h, device=device).float() + 0.5) * stride
        x = (torch.arange(w, device=device).float() + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        all_anchors.append(centers)
        all_strides.append(torch.full((h * w,), float(stride), device=device))
        num_per_level.append(h * w)

    anchors = torch.cat(all_anchors, dim=0)
    anchor_strides = torch.cat(all_strides, dim=0)
    return anchors, anchor_strides, num_per_level


class YOLOv8Loss(nn.Module):
    """YOLOv8-style detection loss with TAL assignment.

    Combines Focal classification loss with CIoU regression loss.

    Args:
        num_classes: Number of detection classes.
        box_weight: Weight for regression (CIoU) loss.
        cls_weight: Weight for classification (Focal) loss.
        class_weights: Per-class weights for classification loss.
        focal_gamma: Focal loss gamma (0 = standard BCE, 2.0 default).
        strides: FPN level strides. Required, keyword-only (fusion-redesign
            D-D) — a forgotten argument is a `TypeError` at the call site
            instead of a silent, wrong-but-plausible `[8, 16, 32]` default.
        assigner_center_radius: Center-sampling tolerance in stride units
            passed to `TaskAlignedAssigner` (0.0 = legacy strict containment, D9).
        assigner_level_ranges: Per-level GT-size admissibility bins (D8).
        assigner_collect_stats: Enable non-destructive per-class/per-level
            positive-anchor instrumentation (A4).
        log_as_tensors: Return the `loss_dict` terms as detached 0-dim tensors instead of
            floats, so a trainer can accumulate them on the device and convert once per
            epoch instead of synchronising with the GPU three times per step.
        mask_weight: Gain of the YOLOv8-seg mask loss. Only used when `forward` receives
            `mask_outputs`, so a detect run never sees it. 7.5 is the value ultralytics
            multiplies its segmentation loss by (its box gain).
    """

    def __init__(
        self,
        num_classes: int = 2,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        class_weights: list[float] | None = None,
        focal_gamma: float = 2.0,
        *,
        strides: list[int],
        assigner_center_radius: float = 0.0,
        assigner_level_ranges: list[float] | None = None,
        assigner_collect_stats: bool = False,
        ignore_classes: list[int] | None = None,
        mask_weight: float = 7.5,
        log_as_tensors: bool = False,
    ):
        super().__init__()
        self.log_as_tensors = log_as_tensors
        self.mask_weight = mask_weight
        self.ignore_classes = list(ignore_classes or [])
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.focal_gamma = focal_gamma
        self.strides = list(strides)

        if class_weights is None:
            class_weights = [1.0] * num_classes
        class_weights = [
            0.0 if cls_id in self.ignore_classes else weight
            for cls_id, weight in enumerate(class_weights)
        ]
        self.register_buffer(
            "class_weights", torch.tensor(class_weights, dtype=torch.float32)
        )

        self.assigner = TaskAlignedAssigner(
            topk=13,
            alpha=1.0,
            beta=6.0,
            center_radius=assigner_center_radius,
            level_ranges=assigner_level_ranges,
            collect_stats=assigner_collect_stats,
        )

    def get_assigner_stats(self) -> dict[tuple[int, int], int]:
        """Return accumulated per-(class, level) positive-anchor counts (A4)."""
        return dict(self.assigner.last_stats)

    def reset_assigner_stats(self) -> None:
        """Clear accumulated assigner instrumentation counts (A4)."""
        self.assigner.reset_stats()

    def write_assigner_stats_csv(self, path: str | Path) -> None:
        """Write accumulated per-class, per-level positive-anchor counts to CSV (A4).

        Used by `scripts/evaluate_checkpoint.py --assigner-stats` (E2/A0).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "level", "positive_anchor_count"])
            for (cls_id, level), count in sorted(self.assigner.last_stats.items()):
                writer.writerow([cls_id, level, count])

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: dict,
        mask_outputs: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute detection loss, plus the mask loss of a segmentation student.

        The whole batch is processed with padded tensor ops: no Python loop over images and no
        host synchronisation on the hot path (the one exception is the mask loss, which reads
        the largest positive count of the batch once to size its padded prototype product).

        Args:
            predictions: List of (B, nc+4, H_i, W_i) per FPN level.
                First nc channels are class logits, last 4 are bbox deltas.
            targets: Dict, in either of two forms (identical results):
                per-image lists
                    bboxes: List of (N_i, 4) GT bboxes per image (cxcywh, normalized 0-1).
                    labels: List of (N_i,) GT class labels per image.
                    masks: Optional list of (N_i, h, w) instance masks per image, aligned with
                        `bboxes`. Used only together with `mask_outputs`.
                    has_mask: Optional list of (N_i,) bool; instances without a polygon are
                        False and take no part in the mask loss. All True when absent.
                or the padded form the collate function also provides
                    bboxes_padded (B, N, 4), labels_padded (B, N), counts (B,), and the
                    optional masks_padded (B, N, h, w) / has_mask_padded (B, N).
            mask_outputs: Optional `{"mask_coefs": [(B, nm, H_i, W_i), ...], "protos":
                (B, nm, h, w)}` from a segment student. None (default) is the plain detection
                loss; so is a batch that carries no `masks`.

        Returns:
            total_loss: Scalar loss for backward().
            loss_dict: {"box_loss": float, "cls_loss": float}, plus "mask_loss" when the
                mask term ran. Floats by default; detached 0-dim tensors with
                `log_as_tensors=True`, which spares the host synchronisation of `.item()`.
        """
        predictions = [pred.float() for pred in predictions]
        device = predictions[0].device
        B = predictions[0].shape[0]
        gt_boxes, gt_labels, gt_counts, gt_masks, gt_has_mask = self._padded_targets(targets, device)
        gt_valid = torch.arange(gt_labels.shape[1], device=device).unsqueeze(0) < gt_counts.unsqueeze(1)
        if self.ignore_classes:
            # Ignored GTs stay in their padded slot but become padding, so slots (and the GT
            # index the mask loss reads) keep addressing the matching mask.
            ignored = torch.tensor(self.ignore_classes, device=device)
            gt_valid = gt_valid & ~torch.isin(gt_labels, ignored)
            gt_counts = gt_valid.sum(dim=1)
            gt_boxes, gt_labels, gt_masks, gt_has_mask = self._compact(
                gt_valid, gt_boxes, gt_labels, gt_masks, gt_has_mask
            )

        # Split predictions into class scores and bbox deltas
        all_cls = []
        all_reg = []
        feat_sizes = []

        for pred in predictions:
            # pred: (B, nc+4, H, W)
            cls_pred = pred[:, :self.num_classes, :, :]  # (B, nc, H, W)
            reg_pred = pred[:, self.num_classes:, :, :]  # (B, 4, H, W)

            H, W = pred.shape[2], pred.shape[3]
            feat_sizes.append((H, W))

            # Flatten spatial: (B, nc, H*W) → (B, H*W, nc)
            cls_flat = cls_pred.flatten(2).permute(0, 2, 1)
            reg_flat = reg_pred.flatten(2).permute(0, 2, 1)

            all_cls.append(cls_flat)
            all_reg.append(reg_flat)

        # Concatenate all levels: (B, total_anchors, nc) and (B, total_anchors, 4)
        pred_cls = torch.cat(all_cls, dim=1)  # (B, A, nc)
        pred_reg = torch.cat(all_reg, dim=1)  # (B, A, 4)

        # Generate anchors
        anchors, anchor_strides, num_per_level = _generate_anchors(feat_sizes, self.strides, device)

        # Convert reg predictions from deltas to absolute bboxes
        # pred_reg is (cx, cy, w, h) in pixel space relative to anchors
        pred_bboxes = self._decode_bboxes(pred_reg, anchors)  # (B, A, 4) cxcywh pixel

        # The image size follows from the first feature map and its stride.
        img_h = feat_sizes[0][0] * self.strides[0]
        img_w = feat_sizes[0][1] * self.strides[0]

        # GT boxes: normalized -> pixel coordinates (padding stays zero).
        scale = gt_boxes.new_tensor([img_w, img_h, img_w, img_h])
        gt_boxes_pixel = gt_boxes * scale

        # TAL assignment
        pred_scores = pred_cls.sigmoid()
        target_classes, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores, pred_bboxes, gt_labels, gt_boxes_pixel, anchors, self.strides,
            anchor_strides=anchor_strides, num_per_level=num_per_level, gt_counts=gt_counts,
        )

        # Build target_cls: one-hot for positives (binary), zero for negatives
        pos_classes = target_classes.clamp(0, self.num_classes - 1)
        target_cls = F.one_hot(pos_classes, self.num_classes).to(pred_cls.dtype)
        target_cls = target_cls * fg_mask.unsqueeze(-1)

        # --- Classification loss: Focal BCE ---
        # BCE with logits (element-wise)
        bce = F.binary_cross_entropy_with_logits(
            pred_cls, target_cls, reduction="none"
        )  # (B, A, nc)

        # Focal weight: (1 - p_t)^gamma
        p_t = torch.exp(-bce)
        focal_weight = (1.0 - p_t) ** self.focal_gamma

        # Class weights
        cls_weights = self.class_weights.to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, nc)

        # Sum over classes, then over anchors
        weighted = (focal_weight * cls_weights * bce).sum(dim=-1)  # (B, A)

        # Normalize by number of positive anchors (all on device: no host synchronisation)
        n_fg = fg_mask.sum()
        num_pos = n_fg.clamp(min=1).to(weighted.dtype)
        if self.ignore_classes:
            # A batch of healthy fruit alone has no positives once the mango box is gone.
            # Dividing by 1 would turn its summed negatives into a spike.
            num_pos = torch.where(
                n_fg == 0, torch.full_like(num_pos, float(B * self.assigner.topk)), num_pos
            )
        cls_loss = weighted.sum() / num_pos

        # --- Regression loss (CIoU on positives only) ---
        # Dense over every anchor and masked by `fg_mask`, rather than gathering the positives
        # (`pred[fg_mask]` would synchronise). Negatives are compared with their own detached
        # prediction (CIoU loss 0, gradient 0), so no degenerate target reaches the loss.
        safe_target = torch.where(fg_mask.unsqueeze(-1), target_bboxes, pred_bboxes.detach())
        ciou = complete_box_iou_loss(
            TaskAlignedAssigner._cxcywh_to_xyxy(pred_bboxes).reshape(-1, 4),
            TaskAlignedAssigner._cxcywh_to_xyxy(safe_target).reshape(-1, 4),
            reduction="none",
        ).view_as(fg_mask)
        box_loss = (ciou * fg_mask).sum() / num_pos

        # --- Total loss ---
        total_loss = self.box_weight * box_loss + self.cls_weight * cls_loss

        loss_dict = {"box_loss": self._log(box_loss), "cls_loss": self._log(cls_loss)}

        # --- Mask loss (YOLOv8-seg), on the anchors the TAL assignment made positive ---
        if mask_outputs is not None and gt_masks is not None:
            mask_loss = self._mask_loss(
                mask_outputs, gt_boxes, gt_masks, gt_has_mask, fg_mask, self.assigner.last_gt_idx
            )
            total_loss = total_loss + self.mask_weight * mask_loss
            loss_dict["mask_loss"] = self._log(mask_loss)

        return total_loss, loss_dict

    def _log(self, value: torch.Tensor):
        """A loss term as it is logged: a float, or (log_as_tensors) a detached tensor."""
        return value.detach() if self.log_as_tensors else value.item()

    @staticmethod
    def _padded_targets(targets: dict, device: torch.device):
        """The targets of a batch as padded tensors, whichever form they arrive in."""
        if "bboxes_padded" in targets:
            masks = targets.get("masks_padded")
            has_mask = targets.get("has_mask_padded")
            counts = targets["counts"].to(device)
            if masks is not None and has_mask is None:
                n = targets["labels_padded"].shape[1]
                has_mask = torch.arange(n, device=device).unsqueeze(0) < counts.unsqueeze(1)
            return (
                targets["bboxes_padded"].to(device),
                targets["labels_padded"].to(device),
                counts,
                masks.to(device) if masks is not None else None,
                has_mask.to(device) if has_mask is not None else None,
            )
        return pad_targets(targets, device=device)

    @staticmethod
    def _compact(valid, boxes, labels, masks, has_mask):
        """Left-pack the valid GTs of every image, order kept (ignored GTs drop out).

        The same GT order and indices the list form gets from filtering the per-image lists.
        """
        order = torch.argsort((~valid).to(torch.int8), dim=1, stable=True)  # valid first
        boxes = boxes.gather(1, order.unsqueeze(-1).expand(-1, -1, 4))
        labels = labels.gather(1, order)
        if masks is not None:
            masks = masks[torch.arange(masks.shape[0], device=masks.device).unsqueeze(1), order]
            has_mask = has_mask.gather(1, order)
        return boxes, labels, masks, has_mask

    @staticmethod
    def _mask_loss(
        mask_outputs: dict,
        gt_boxes: torch.Tensor,
        gt_masks: torch.Tensor,
        gt_has_mask: torch.Tensor,
        fg_mask: torch.Tensor,
        gt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Mean, over positive anchors with a GT mask, of the BCE inside the GT box.

        Per positive anchor the predicted mask is `sigmoid(coef . protos)`; it is compared
        with its GT instance mask at prototype resolution, only inside the GT box, and the
        BCE is divided by the box area in prototype pixels (the YOLOv8-seg normalisation).

        Batched: each image's positives are packed into a (B, K, nm) coefficient block and
        multiplied with its prototypes in one `bmm`; empty slots carry weight 0. K, the
        largest positive count of the batch, is read once from the device.

        Args:
            gt_boxes: (B, N, 4) normalized cxcywh, padded.
            gt_masks: (B, N, h, w) instance masks, padded.
            gt_has_mask: (B, N) bool.
            fg_mask / gt_idx: (B, A) positives and the GT index each one was assigned to.
        """
        protos = mask_outputs["protos"].float()  # (B, nm, h, w)
        coefs = torch.cat(
            [c.float().flatten(2).permute(0, 2, 1) for c in mask_outputs["mask_coefs"]], dim=1
        )  # (B, A, nm)
        B, nm, h, w = protos.shape
        device = protos.device
        if gt_masks.shape[1] == 0:  # no GT in the whole batch (static shape, no sync)
            return protos.sum() * 0.0

        gt_safe = gt_idx.clamp(min=0)
        take = fg_mask & gt_has_mask.gather(1, gt_safe)  # (B, A) positives that own a mask
        n_per_image = take.sum(dim=1)
        k_max = int(n_per_image.max().item())  # the one host read of the mask loss
        if k_max == 0:
            return protos.sum() * 0.0  # keeps the graph, adds nothing

        # Slot of every taken anchor inside its image's block, in anchor order.
        rank = take.long().cumsum(dim=1) - 1  # (B, A)
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(take)
        slot_b, slot_k = b_idx[take], rank[take]
        coef_block = coefs.new_zeros(B, k_max, nm)
        coef_block[slot_b, slot_k] = coefs[take]
        gt_block = gt_idx.new_zeros(B, k_max)
        gt_block[slot_b, slot_k] = gt_safe[take]
        weight = (torch.arange(k_max, device=device).unsqueeze(0) < n_per_image.unsqueeze(1))  # (B, K)

        logits = torch.bmm(coef_block, protos.flatten(2)).view(B, k_max, h, w)

        gt = gt_masks[b_idx[:, :1].expand(B, k_max), gt_block].to(device).float()  # (B, K, h, w)
        if gt.shape[-2:] != (h, w):
            gt = (
                F.interpolate(gt.flatten(0, 1)[:, None], size=(h, w), mode="area")[:, 0] > 0.5
            ).float().view(B, k_max, h, w)

        box = gt_boxes.gather(1, gt_block.unsqueeze(-1).expand(-1, -1, 4))  # (B, K, 4)
        x1 = ((box[..., 0] - box[..., 2] / 2) * w).unsqueeze(-1).unsqueeze(-1)
        x2 = ((box[..., 0] + box[..., 2] / 2) * w).unsqueeze(-1).unsqueeze(-1)
        y1 = ((box[..., 1] - box[..., 3] / 2) * h).unsqueeze(-1).unsqueeze(-1)
        y2 = ((box[..., 1] + box[..., 3] / 2) * h).unsqueeze(-1).unsqueeze(-1)
        cols = torch.arange(w, device=device, dtype=logits.dtype).view(1, 1, 1, w)
        rows = torch.arange(h, device=device, dtype=logits.dtype).view(1, 1, h, 1)
        crop = ((cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2)).to(logits.dtype)

        bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
        per_anchor = (bce * crop).sum(dim=(2, 3)) / crop.sum(dim=(2, 3)).clamp(min=1.0)  # (B, K)
        return (per_anchor * weight).sum() / weight.sum()

    @staticmethod
    def _decode_bboxes(
        reg_preds: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Decode bbox regression predictions to absolute cxcywh boxes.

        reg_preds are (cx_offset, cy_offset, w, h) where offsets are relative
        to anchor centers.

        Args:
            reg_preds: (B, A, 4) raw regression outputs.
            anchors: (A, 2) anchor center points.

        Returns:
            (B, A, 4) decoded bboxes in cxcywh pixel coords.
        """
        # reg_preds: (B, A, 4) — (dx, dy, w, h)
        # Anchors: (A, 2) — (cx, cy)
        anchor_cx = anchors[:, 0].unsqueeze(0)  # (1, A)
        anchor_cy = anchors[:, 1].unsqueeze(0)  # (1, A)

        # Decode: cx = anchor_cx + dx, cy = anchor_cy + dy
        # w and h are exp-predicted, clamp raw inputs to avoid exp→∞
        dx = reg_preds[:, :, 0]
        dy = reg_preds[:, :, 1]
        w = reg_preds[:, :, 2].clamp(-9.0, 9.0).exp()   # exp(9)≈8103, safe
        h = reg_preds[:, :, 3].clamp(-9.0, 9.0).exp()

        cx = anchor_cx + dx
        cy = anchor_cy + dy

        return torch.stack([cx, cy, w, h], dim=-1)
