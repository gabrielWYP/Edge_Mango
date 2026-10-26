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
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Valid values for `KDLoss(loss_type=...)` and `KDConfig.kd_loss_type`.
KD_LOSS_TYPES = ("mse", "pearson")

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


class KDLoss(nn.Module):
    """Per-level weighted feature distillation loss (MSE or PKD Pearson).

    Args:
        level_weights: Dict mapping level name to scalar weight.
            Expected keys: backbone, fpn, head_cls, head_reg.
        temperature: Temperature scaling factor (reserved for future use).
        loss_type: Per-pair loss, one of ``KD_LOSS_TYPES``: ``"mse"``
            (default, raw-feature MSE) or ``"pearson"`` (PKD).

    Raises:
        ValueError: If ``loss_type`` is not one of ``KD_LOSS_TYPES``.
    """

    def __init__(
        self,
        level_weights: dict[str, float],
        temperature: float = 1.0,
        loss_type: str = "mse",
    ):
        super().__init__()
        if loss_type not in KD_LOSS_TYPES:
            raise ValueError(
                f"Invalid loss_type '{loss_type}'. Must be one of: {KD_LOSS_TYPES}"
            )
        self.level_weights = level_weights
        self.temperature = temperature
        self.loss_type = loss_type

    def forward(
        self,
        teacher_feats: dict[str, list[torch.Tensor]],
        student_feats: dict[str, list[torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the weighted distillation loss across all levels.

        Args:
            teacher_feats: Dict with keys {backbone, fpn, head_cls, head_reg},
                each mapping to a list of frozen teacher tensors (the
                regression targets; detached here).
            student_feats: Dict with same keys, each mapping to a list of
                adapted student tensors (already in teacher channel space).

        Returns:
            total_loss: Scalar weighted sum of per-level losses.
            per_level: Dict mapping level name to its loss value (float).
        """
        total_loss = torch.tensor(0.0, device=self._get_device(teacher_feats))
        per_level: dict[str, float] = {}

        for level_name, weight in self.level_weights.items():
            t_list = teacher_feats[level_name]
            s_list = student_feats[level_name]

            assert len(t_list) == len(s_list), (
                f"Level '{level_name}': teacher has {len(t_list)} sub-levels, "
                f"student has {len(s_list)}"
            )

            level_loss = torch.tensor(0.0, device=total_loss.device)
            for t_feat, s_feat in zip(t_list, s_list):
                level_loss = level_loss + self._pair_loss(s_feat, t_feat.detach())

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

    @staticmethod
    def _get_device(feats: dict[str, list[torch.Tensor]]) -> torch.device:
        """Extract device from the first available tensor."""
        for tensors in feats.values():
            if tensors:
                return tensors[0].device
        return torch.device("cpu")
