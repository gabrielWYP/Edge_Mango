"""
Knowledge Distillation loss — per-level weighted MSE.

Computes MSE between adapted student features (student → teacher channels,
FitNets-style) and the frozen teacher features at four distillation levels:
backbone, fpn, head_cls, head_reg. The teacher side is the regression
target and is always detached, so gradients flow only into the student and
its adapters.

Each level contains a list of tensors (one per sub-level / scale).
The loss averages MSE across all sub-levels within each group,
then computes a weighted sum across groups.

Loss formula:
    kd_loss = Σ(level_weight × mean(MSE per sub-level))

With feature_norm="pkd" each per-sub-level MSE is replaced by the PKD loss
(Cao et al. 2022, arXiv 2207.02039): both features are z-scored per channel
over (N, H, W) and the MSE is halved. With feature_norm="none" (default) the
raw MSE above is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allowed values of `feature_norm` (and of KDConfig.kd_feature_norm).
KD_FEATURE_NORMS = frozenset({"none", "pkd"})
# Matches mmrazor's PKDLoss.norm.
_PKD_EPS = 1e-6


def _pkd_normalize(feat: torch.Tensor) -> torch.Tensor:
    """Z-score each channel of an (N, C, H, W) tensor over (N, H, W)."""
    n, c, h, w = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(c, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    feat = (feat - mean) / (std + _PKD_EPS)
    return feat.reshape(c, n, h, w).permute(1, 0, 2, 3)


def pkd_loss(s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
    """PKD feature loss: MSE between per-channel z-scored features, halved.

    Normalizing makes the loss invariant to per-channel offset and (positive)
    scale of either side, so large constant channel offsets in the teacher's
    fused features no longer dominate it; it is bounded by 2.
    """
    return F.mse_loss(_pkd_normalize(s_feat), _pkd_normalize(t_feat.detach())) / 2


class KDLoss(nn.Module):
    """Per-level weighted MSE distillation loss.

    Args:
        level_weights: Dict mapping level name to scalar weight.
            Expected keys: backbone, fpn, head_cls, head_reg.
        temperature: Temperature scaling factor (reserved for future use).
        feature_norm: "none" (raw MSE, default) or "pkd" (per-channel
            z-score of both sides before the MSE, see `pkd_loss`).
    """

    def __init__(
        self,
        level_weights: dict[str, float],
        temperature: float = 1.0,
        feature_norm: str = "none",
    ):
        super().__init__()
        if feature_norm not in KD_FEATURE_NORMS:
            raise ValueError(
                f"Invalid feature_norm '{feature_norm}'. "
                f"Must be one of: {sorted(KD_FEATURE_NORMS)}"
            )
        self.level_weights = level_weights
        self.temperature = temperature
        self.feature_norm = feature_norm

    def forward(
        self,
        teacher_feats: dict[str, list[torch.Tensor]],
        student_feats: dict[str, list[torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted MSE distillation loss across all levels.

        Args:
            teacher_feats: Dict with keys {backbone, fpn, head_cls, head_reg},
                each mapping to a list of frozen teacher tensors (the
                regression targets; detached here).
            student_feats: Dict with same keys, each mapping to a list of
                adapted student tensors (already in teacher channel space).

        Returns:
            total_loss: Scalar weighted sum of per-level MSE losses.
            per_level: Dict mapping level name to its MSE loss value (float).
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

            level_mse = torch.tensor(0.0, device=total_loss.device)
            for t_feat, s_feat in zip(t_list, s_list):
                if self.feature_norm == "pkd":
                    level_mse = level_mse + pkd_loss(s_feat, t_feat)
                else:
                    level_mse = level_mse + F.mse_loss(s_feat, t_feat.detach())

            # Average across sub-levels
            if len(t_list) > 0:
                level_mse = level_mse / len(t_list)

            per_level[level_name] = level_mse.item()
            total_loss = total_loss + weight * level_mse

        return total_loss, per_level

    @staticmethod
    def _get_device(feats: dict[str, list[torch.Tensor]]) -> torch.device:
        """Extract device from the first available tensor."""
        for tensors in feats.values():
            if tensors:
                return tensors[0].device
        return torch.device("cpu")
