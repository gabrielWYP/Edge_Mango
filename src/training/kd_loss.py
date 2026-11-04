"""
Knowledge Distillation loss — per-level weighted feature imitation.

Computes a feature-imitation loss between adapted student features
(student → teacher channels, FitNets-style) and the frozen teacher features
at four distillation levels: backbone, fpn, head_cls, head_reg. The teacher
side is the regression target and is always detached, so gradients flow only
into the student and its adapters.

Each level contains a list of tensors (one per sub-level / scale).
The loss averages the per-pair loss across all sub-levels within each group,
then computes a weighted sum across groups.

Two per-pair losses are available (``loss_type``):

- ``"mse"`` (default): plain MSE between raw feature maps.
- ``"pearson"``: PKD (Cao et al., NeurIPS 2022, arXiv 2207.02039). Each map
  is standardized per channel to zero mean / unit variance, then
  ``0.5 * MSE`` is taken between the standardized maps — equivalent to
  ``1 - r`` with ``r`` the Pearson correlation per channel. This makes the
  loss invariant to a per-channel positive scale and shift of either side,
  so the student imitates the teacher's relational pattern rather than its
  feature magnitudes. Matches mmrazor's ``PKDLoss`` exactly (see
  ``_pkd_normalize``).

Loss formula:
    kd_loss = Σ(level_weight × mean(pair loss per sub-level))

Region balancing (``region_weights``, off by default): a plain per-map mean lets
the background dominate, since a 30 px lesion covers 0.22 % of a 640x640 map.
With region weights, the labels split each map into damage, fruit and background
cells, and the pair loss becomes ``Σ(region_weight × mean(error over its cells))``.
A batch without a region adds nothing for it. Weights that sum to 1 keep the loss
on the scale of the plain mean when the error is uniform.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DAMAGE_CLASS_ID
from .roi_crop import MANGO_CLASS_ID

# Valid values for `KDLoss(loss_type=...)` and `KDConfig.kd_loss_type`.
KD_LOSS_TYPES = ("mse", "pearson")

# Required keys of `KDLoss(region_weights=...)` and `KDConfig.kd_region_weights`.
KD_REGIONS = ("damage", "fruit", "background")

# Added to the std (not the variance) before dividing, as in mmrazor PKDLoss.
_PKD_EPS = 1e-6


def _pkd_normalize(feat: torch.Tensor) -> torch.Tensor:
    """Standardize a (N, C, H, W) map per channel, as mmrazor ``PKDLoss.norm``.

    Statistics are taken per channel over (N, H, W) — the batch is pooled,
    not normalized per sample — with the unbiased std, and the result is
    ``(x - mean) / (std + 1e-6)``.
    """
    assert feat.dim() == 4, f"PKD expects (N, C, H, W) features, got {tuple(feat.shape)}"
    n, c, h, w = feat.shape
    flat = feat.transpose(0, 1).reshape(c, -1)
    mean = flat.mean(dim=-1, keepdim=True)
    std = flat.std(dim=-1, keepdim=True)
    flat = (flat - mean) / (std + _PKD_EPS)
    return flat.reshape(c, n, h, w).transpose(0, 1)


def validate_region_weights(region_weights: dict[str, float] | None) -> None:
    """Fail loud on a typo: a dropped region would silently change the loss."""
    if region_weights is None:
        return
    if set(region_weights) != set(KD_REGIONS):
        raise ValueError(
            f"Invalid region weights {region_weights}. Must set exactly: {KD_REGIONS}"
        )
    values = list(region_weights.values())
    if any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError(
            f"Invalid region weights {region_weights}. "
            "They must be non-negative with a positive sum."
        )


def validate_sublevels(sublevels: dict[str, list[int]] | None, levels) -> None:
    """Fail loud on a group that does not exist or an index that cannot be one."""
    if sublevels is None:
        return
    for level, indices in sublevels.items():
        valid = all(isinstance(i, int) and i >= 0 for i in indices)
        if level not in levels or not valid:
            raise ValueError(
                f"Invalid sublevels {sublevels}. Keys must be in {list(levels)} "
                "and values lists of non-negative map indices."
            )


def _fill_boxes(mask: torch.Tensor, boxes: torch.Tensor) -> None:
    """Mark every cell a normalized cxcywh box touches, so none rounds to empty."""
    h, w = mask.shape
    for cx, cy, bw, bh in boxes.tolist():
        x1 = max(math.floor((cx - bw / 2) * w), 0)
        y1 = max(math.floor((cy - bh / 2) * h), 0)
        x2 = min(math.ceil((cx + bw / 2) * w), w)
        y2 = min(math.ceil((cy + bh / 2) * h), h)
        mask[y1:y2, x1:x2] = True


def build_region_masks(
    bboxes: list[torch.Tensor],
    labels: list[torch.Tensor],
    height: int,
    width: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Split a (N, H, W) grid into disjoint damage, fruit and background cells.

    Args:
        bboxes: Per-image (N_i, 4) boxes, cxcywh normalized to the letterboxed image.
        labels: Per-image (N_i,) class ids.
        height: Grid height.
        width: Grid width.
        device: Device of the returned masks.

    Returns:
        Dict mapping each of `KD_REGIONS` to a boolean (N, H, W) mask. A cell in
        both a damage and a fruit box counts as damage.
    """
    n = len(bboxes)
    damage = torch.zeros(n, height, width, dtype=torch.bool, device=device)
    fruit = torch.zeros(n, height, width, dtype=torch.bool, device=device)
    for i, (boxes, classes) in enumerate(zip(bboxes, labels)):
        _fill_boxes(damage[i], boxes[classes == DAMAGE_CLASS_ID])
        _fill_boxes(fruit[i], boxes[classes == MANGO_CLASS_ID])
    fruit = fruit & ~damage
    return {"damage": damage, "fruit": fruit, "background": ~(damage | fruit)}


class KDLoss(nn.Module):
    """Per-level weighted feature distillation loss (MSE or PKD Pearson).

    Args:
        level_weights: Dict mapping level name to scalar weight.
            Expected keys: backbone, fpn, head_cls, head_reg.
        temperature: Temperature scaling factor (reserved for future use).
        loss_type: Per-pair loss, one of ``KD_LOSS_TYPES``: ``"mse"``
            (default, raw-feature MSE) or ``"pearson"`` (PKD).
        region_weights: Optional weight per `KD_REGIONS` entry. ``None``
            (default) keeps the plain per-map mean.
        sublevels: Optional map indices to keep per level, finest first, such
            as ``{"fpn": [0, 1]}``. A level left out keeps all its maps; an
            empty list turns it off.

    Raises:
        ValueError: If ``loss_type`` is not one of ``KD_LOSS_TYPES``, or
            ``region_weights`` or ``sublevels`` are invalid.
    """

    def __init__(
        self,
        level_weights: dict[str, float],
        temperature: float = 1.0,
        loss_type: str = "mse",
        region_weights: dict[str, float] | None = None,
        sublevels: dict[str, list[int]] | None = None,
    ):
        super().__init__()
        if loss_type not in KD_LOSS_TYPES:
            raise ValueError(
                f"Invalid loss_type '{loss_type}'. Must be one of: {KD_LOSS_TYPES}"
            )
        validate_region_weights(region_weights)
        validate_sublevels(sublevels, level_weights)
        self.level_weights = level_weights
        self.temperature = temperature
        self.loss_type = loss_type
        self.region_weights = region_weights
        self.sublevels = sublevels or {}

    def forward(
        self,
        teacher_feats: dict[str, list[torch.Tensor]],
        student_feats: dict[str, list[torch.Tensor]],
        bboxes: list[torch.Tensor] | None = None,
        labels: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the weighted distillation loss across all levels.

        Args:
            teacher_feats: Dict with keys {backbone, fpn, head_cls, head_reg},
                each mapping to a list of frozen teacher tensors (the
                regression targets; detached here).
            student_feats: Dict with same keys, each mapping to a list of
                adapted student tensors (already in teacher channel space).
            bboxes: Per-image normalized cxcywh boxes. Required with
                ``region_weights``; ignored otherwise.
            labels: Per-image class ids, paired with ``bboxes``.

        Returns:
            total_loss: Scalar weighted sum of per-level losses.
            per_level: Dict mapping level name to its loss value (float).

        Raises:
            ValueError: If ``region_weights`` is set and boxes are missing.
        """
        if self.region_weights is not None and (bboxes is None or labels is None):
            raise ValueError("region_weights require bboxes and labels")

        total_loss = torch.tensor(0.0, device=self._get_device(teacher_feats))
        per_level: dict[str, float] = {}

        for level_name, weight in self.level_weights.items():
            t_list = teacher_feats[level_name]
            s_list = student_feats[level_name]

            assert len(t_list) == len(s_list), (
                f"Level '{level_name}': teacher has {len(t_list)} sub-levels, "
                f"student has {len(s_list)}"
            )

            if level_name in self.sublevels:
                t_list = [t_list[i] for i in self.sublevels[level_name]]
                s_list = [s_list[i] for i in self.sublevels[level_name]]

            level_loss = torch.tensor(0.0, device=total_loss.device)
            for t_feat, s_feat in zip(t_list, s_list):
                if self.region_weights is None:
                    pair_loss = self._pair_loss(s_feat, t_feat.detach())
                else:
                    pair_loss = self._region_pair_loss(s_feat, t_feat.detach(), bboxes, labels)
                level_loss = level_loss + pair_loss

            # Average across sub-levels
            if len(t_list) > 0:
                level_loss = level_loss / len(t_list)

            per_level[level_name] = level_loss.item()
            total_loss = total_loss + weight * level_loss

        return total_loss, per_level

    def _pair_loss(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """Loss for one (student, detached teacher) feature pair."""
        if self.loss_type == "mse":
            return F.mse_loss(s_feat, t_feat)
        # PKD: shapes already match after the 1x1 adapters, so a mismatch is
        # a wiring bug — fail loudly instead of mmrazor's silent interpolate.
        assert s_feat.shape == t_feat.shape, (
            f"PKD needs matching shapes, got student {tuple(s_feat.shape)} "
            f"vs teacher {tuple(t_feat.shape)}"
        )
        return 0.5 * F.mse_loss(_pkd_normalize(s_feat), _pkd_normalize(t_feat))

    def _region_pair_loss(
        self,
        s_feat: torch.Tensor,
        t_feat: torch.Tensor,
        bboxes: list[torch.Tensor],
        labels: list[torch.Tensor],
    ) -> torch.Tensor:
        """Weighted sum of the mean per-cell error of each region."""
        assert s_feat.shape == t_feat.shape, (
            f"Region balancing needs matching shapes, got student {tuple(s_feat.shape)} "
            f"vs teacher {tuple(t_feat.shape)}"
        )
        scale = 1.0
        if self.loss_type == "pearson":
            s_feat, t_feat = _pkd_normalize(s_feat), _pkd_normalize(t_feat)
            scale = 0.5
        error = (s_feat - t_feat).pow(2).mean(dim=1)
        masks = build_region_masks(bboxes, labels, *error.shape[1:], device=error.device)
        loss = error.new_zeros(())
        for region, weight in self.region_weights.items():
            if masks[region].any():
                loss = loss + weight * error[masks[region]].mean()
        return scale * loss

    @staticmethod
    def _get_device(feats: dict[str, list[torch.Tensor]]) -> torch.device:
        """Extract device from the first available tensor."""
        for tensors in feats.values():
            if tensors:
                return tensors[0].device
        return torch.device("cpu")
