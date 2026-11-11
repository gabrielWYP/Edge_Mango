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
from torchvision.ops import complete_box_iou_loss, box_iou


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
        gt_labels: list[torch.Tensor],
        gt_bboxes: list[torch.Tensor],
        anchors: torch.Tensor,
        strides: list[int],
        anchor_strides: torch.Tensor | None = None,
        num_per_level: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign predictions to ground truth targets.

        Args:
            pred_scores: (B, num_anchors, num_classes) predicted class scores (sigmoid).
            pred_bboxes: (B, num_anchors, 4) predicted bboxes in cxcywh format.
            gt_labels: List of (N_i,) GT class labels per image.
            gt_bboxes: List of (N_i, 4) GT bboxes per image in cxcywh format.
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

        Returns:
            target_classes: (B, num_anchors) assigned class IDs (-1 for negative).
            target_bboxes: (B, num_anchors, 4) assigned GT bboxes.
            target_scores: (B, num_anchors) alignment scores for positive samples.
            fg_mask: (B, num_anchors) boolean mask of foreground (positive) anchors.
        """
        B, num_anchors, num_classes = pred_scores.shape
        device = pred_scores.device

        target_classes = torch.full((B, num_anchors), -1, dtype=torch.long, device=device)
        target_bboxes = torch.zeros((B, num_anchors, 4), device=device)
        target_scores = torch.zeros((B, num_anchors), device=device)
        fg_mask = torch.zeros((B, num_anchors), dtype=torch.bool, device=device)
        target_gt_idx = torch.full((B, num_anchors), -1, dtype=torch.long, device=device)
        self.last_gt_idx = target_gt_idx

        if anchor_strides is None:
            anchor_strides = torch.full((num_anchors,), float(strides[0]), device=device)

        level_ids = (
            self._anchor_level_ids(num_per_level, num_anchors, device)
            if num_per_level is not None
            else None
        )

        for b in range(B):
            n_gt = len(gt_labels[b])
            if n_gt == 0:
                continue

            gt_cls = gt_labels[b]  # (N_gt,)
            gt_box = gt_bboxes[b]  # (N_gt, 4) cxcywh

            # Compute alignment metric for each GT-anchor pair
            # pred_scores[b]: (num_anchors, num_classes)
            # For each GT, get the predicted score for that GT's class
            gt_cls_scores = pred_scores[b, :, gt_cls]  # (num_anchors, N_gt)

            # Compute IoU between predicted bboxes and GT bboxes
            # Convert cxcywh to xyxy for IoU computation
            pred_xyxy = self._cxcywh_to_xyxy(pred_bboxes[b])  # (num_anchors, 4)
            gt_xyxy = self._cxcywh_to_xyxy(gt_box)  # (N_gt, 4)
            ious = box_iou(pred_xyxy, gt_xyxy)  # (num_anchors, N_gt)

            # Alignment metric: score^alpha * iou^beta
            alignment = (gt_cls_scores ** self.alpha) * (ious ** self.beta)  # (num_anchors, N_gt)

            for gt_i in range(n_gt):
                # D6: spatial admissibility mask, applied BEFORE top-k.
                admissible = self._spatial_admissibility(
                    anchors, anchor_strides, gt_box[gt_i], self.center_radius
                )

                level_mask = None
                if level_ids is not None and self.level_ranges:
                    gt_w, gt_h = gt_box[gt_i, 2].item(), gt_box[gt_i, 3].item()
                    level_mask = self._level_admissibility(level_ids, gt_w, gt_h, self.level_ranges)
                    admissible = admissible & level_mask

                gt_alignment = alignment[:, gt_i].clone()
                gt_alignment[~admissible] = 0.0  # mask inadmissible anchors before top-k

                topk = min(self.topk, num_anchors)
                topk_vals, topk_idx = gt_alignment.topk(topk)

                valid = topk_vals > 0
                pos_idx = topk_idx[valid]
                pos_scores = topk_vals[valid]

                # D7: fallback runs AFTER the spatial/level filter (not before,
                # the pre-fix ordering that let the filter erase it), and ranks
                # by anchor centers — not predicted-box centers, the pre-fix
                # coordinate-source bug (predicted boxes move during training;
                # the filter above tests fixed anchor grid centers).
                if len(pos_idx) == 0:
                    gt_cxcy = gt_box[gt_i, :2]
                    dist = (anchors[:, 0] - gt_cxcy[0]) ** 2 + (anchors[:, 1] - gt_cxcy[1]) ** 2
                    if level_mask is not None and level_mask.any():
                        dist = dist.clone()
                        dist[~level_mask] = float("inf")
                    _, fallback_idx = dist.topk(min(3, num_anchors), largest=False)
                    pos_idx = fallback_idx
                    pos_scores = torch.ones(len(fallback_idx), device=device) * 0.01

                if len(pos_idx) == 0:
                    continue

                # Conflict resolution: if an anchor is assigned to multiple
                # GTs, keep the one with the highest alignment score. The top-k
                # indices are distinct, so this needs no per-anchor loop, whose
                # comparisons would each wait for the GPU.
                wins = (target_scores[b, pos_idx] < pos_scores) | (target_classes[b, pos_idx] == -1)
                winners = pos_idx[wins]
                target_classes[b, winners] = gt_cls[gt_i]
                target_bboxes[b, winners] = gt_box[gt_i]
                target_scores[b, winners] = pos_scores[wins]
                target_gt_idx[b, winners] = gt_i
                fg_mask[b, winners] = True

                if self.collect_stats and level_ids is not None:
                    cls_id = int(gt_cls[gt_i].item())
                    for idx in pos_idx.tolist():
                        key = (cls_id, int(level_ids[idx].item()))
                        self.last_stats[key] = self.last_stats.get(key, 0) + 1

        return target_classes, target_bboxes, target_scores, fg_mask

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
    ):
        super().__init__()
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

        Args:
            predictions: List of (B, nc+4, H_i, W_i) per FPN level.
                First nc channels are class logits, last 4 are bbox deltas.
            targets: Dict with keys:
                bboxes: List of (N_i, 4) GT bboxes per image (cxcywh, normalized 0-1).
                labels: List of (N_i,) GT class labels per image.
                masks: Optional list of (N_i, h, w) instance masks per image, aligned with
                    `bboxes`. Used only together with `mask_outputs`.
                has_mask: Optional list of (N_i,) bool; instances without a polygon are False
                    and take no part in the mask loss. All True when absent.
            mask_outputs: Optional `{"mask_coefs": [(B, nm, H_i, W_i), ...], "protos":
                (B, nm, h, w)}` from a segment student. None (default) is the plain detection
                loss; so is a batch that carries no `masks`.

        Returns:
            total_loss: Scalar loss for backward().
            loss_dict: {"box_loss": float, "cls_loss": float}, plus "mask_loss" when the
                mask term ran.
        """
        predictions = [pred.float() for pred in predictions]
        device = predictions[0].device
        B = predictions[0].shape[0]
        if self.ignore_classes:
            targets = self._without_ignored_classes(targets, device)

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

        # Convert GT bboxes from normalized to pixel coords
        # We need the image size — assume all same size from the batch
        img_size = predictions[0].shape[2] * self.strides[0]  # approximate
        # Better: use the actual feature sizes to compute
        img_h = feat_sizes[0][0] * self.strides[0]
        img_w = feat_sizes[0][1] * self.strides[0]

        gt_bboxes_pixel = []
        gt_labels_list = targets["labels"]
        for b in range(B):
            gt_b = targets["bboxes"][b]  # (N, 4) normalized cxcywh
            if len(gt_b) > 0:
                gt_pixel = gt_b.clone()
                gt_pixel[:, 0] *= img_w  # cx
                gt_pixel[:, 1] *= img_h  # cy
                gt_pixel[:, 2] *= img_w  # w
                gt_pixel[:, 3] *= img_h  # h
                gt_bboxes_pixel.append(gt_pixel)
            else:
                gt_bboxes_pixel.append(gt_b)

        # TAL assignment
        pred_scores = pred_cls.sigmoid()
        target_classes, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores, pred_bboxes, gt_labels_list, gt_bboxes_pixel, anchors, self.strides,
            anchor_strides=anchor_strides, num_per_level=num_per_level,
        )

        # Build target_cls: one-hot for positives (binary), zero for negatives
        target_cls = torch.zeros_like(pred_cls)  # (B, A, nc)
        for b in range(B):
            pos = fg_mask[b]
            if pos.any():
                pos_classes = target_classes[b, pos]  # (K,)
                pos_classes = pos_classes.clamp(0, self.num_classes - 1)
                target_cls[b, pos] = F.one_hot(pos_classes, self.num_classes).float()

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

        # Normalize by number of positive anchors
        num_pos = fg_mask.sum().clamp(min=1)
        if self.ignore_classes and not fg_mask.any():
            # A batch of healthy fruit alone has no positives once the mango box is gone.
            # Dividing by 1 would turn its summed negatives into a spike.
            num_pos = torch.tensor(float(B * self.assigner.topk), device=device)
        cls_loss = weighted.sum() / num_pos

        # --- Regression loss (CIoU on positives only) ---
        box_loss = torch.tensor(0.0, device=device)
        if fg_mask.any():
            for b in range(B):
                pos = fg_mask[b]
                if not pos.any():
                    continue
                pred_pos = pred_bboxes[b, pos]  # (K, 4) cxcywh
                target_pos = target_bboxes[b, pos]  # (K, 4) cxcywh

                # Convert to xyxy for CIoU
                pred_xyxy = TaskAlignedAssigner._cxcywh_to_xyxy(pred_pos)
                target_xyxy = TaskAlignedAssigner._cxcywh_to_xyxy(target_pos)

                ciou = complete_box_iou_loss(pred_xyxy, target_xyxy, reduction="sum")
                box_loss = box_loss + ciou

            box_loss = box_loss / num_pos

        # --- Total loss ---
        total_loss = self.box_weight * box_loss + self.cls_weight * cls_loss

        loss_dict = {
            "box_loss": box_loss.item(),
            "cls_loss": cls_loss.item(),
        }

        # --- Mask loss (YOLOv8-seg), on the anchors the TAL assignment made positive ---
        if mask_outputs is not None and "masks" in targets:
            mask_loss = self._mask_loss(
                mask_outputs, targets, fg_mask, self.assigner.last_gt_idx, img_h, img_w
            )
            total_loss = total_loss + self.mask_weight * mask_loss
            loss_dict["mask_loss"] = mask_loss.item()

        return total_loss, loss_dict

    @staticmethod
    def _mask_loss(
        mask_outputs: dict,
        targets: dict,
        fg_mask: torch.Tensor,
        gt_idx: torch.Tensor,
        img_h: float,
        img_w: float,
    ) -> torch.Tensor:
        """Mean, over positive anchors with a GT mask, of the BCE inside the GT box.

        Per positive anchor the predicted mask is `sigmoid(coef . protos)`; it is compared
        with its GT instance mask at prototype resolution, only inside the GT box, and the
        BCE is divided by the box area in prototype pixels (the YOLOv8-seg normalisation).
        """
        protos = mask_outputs["protos"].float()  # (B, nm, h, w)
        coefs = torch.cat(
            [c.float().flatten(2).permute(0, 2, 1) for c in mask_outputs["mask_coefs"]], dim=1
        )  # (B, A, nm)
        B, nm, h, w = protos.shape
        has_mask_all = targets.get("has_mask")
        total = protos.new_zeros(())
        count = 0
        for b in range(B):
            if len(targets["bboxes"][b]) == 0:
                continue
            idx = gt_idx[b][fg_mask[b]]  # (K,) GT index of each positive anchor
            if idx.numel() == 0:
                continue
            masks_b = targets["masks"][b]
            has = (
                has_mask_all[b].to(idx.device)[idx]
                if has_mask_all is not None
                else torch.ones_like(idx, dtype=torch.bool)
            )
            if not has.any():
                continue
            idx = idx[has]
            coef = coefs[b][fg_mask[b]][has]  # (K, nm)
            logits = (coef @ protos[b].flatten(1)).view(-1, h, w)

            gt = masks_b[idx].to(logits.device).float()
            if gt.shape[-2:] != (h, w):
                gt = (F.interpolate(gt[:, None], size=(h, w), mode="area")[:, 0] > 0.5).float()

            box = targets["bboxes"][b][idx]  # (K, 4) normalized cxcywh
            x1 = (box[:, 0] - box[:, 2] / 2) * w
            x2 = (box[:, 0] + box[:, 2] / 2) * w
            y1 = (box[:, 1] - box[:, 3] / 2) * h
            y2 = (box[:, 1] + box[:, 3] / 2) * h
            cols = torch.arange(w, device=logits.device, dtype=logits.dtype).view(1, 1, w)
            rows = torch.arange(h, device=logits.device, dtype=logits.dtype).view(1, h, 1)
            crop = (
                (cols >= x1.view(-1, 1, 1)) & (cols < x2.view(-1, 1, 1))
                & (rows >= y1.view(-1, 1, 1)) & (rows < y2.view(-1, 1, 1))
            ).to(logits.dtype)

            bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
            per_anchor = (bce * crop).sum(dim=(1, 2)) / crop.sum(dim=(1, 2)).clamp(min=1.0)
            total = total + per_anchor.sum()
            count += per_anchor.numel()
        if count == 0:
            return protos.sum() * 0.0  # keeps the graph, adds nothing
        return total / count

    def _without_ignored_classes(self, targets: dict, device: torch.device) -> dict:
        ignored = torch.tensor(self.ignore_classes, device=device)
        keep = [~torch.isin(labels, ignored) for labels in targets["labels"]]
        filtered = {
            **targets,
            "bboxes": [boxes[mask] for boxes, mask in zip(targets["bboxes"], keep)],
            "labels": [labels[mask] for labels, mask in zip(targets["labels"], keep)],
        }
        # Instance masks travel with their boxes.
        for key in ("masks", "has_mask"):
            if key in targets:
                filtered[key] = [values[mask] for values, mask in zip(targets[key], keep)]
        return filtered

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
