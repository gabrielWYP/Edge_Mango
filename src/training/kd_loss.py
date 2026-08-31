"""
Knowledge Distillation loss — per-level weighted MSE.

Computes MSE between projected teacher features and student features
at four distillation levels: backbone, fpn, head_cls, head_reg.

Each level contains a list of tensors (one per sub-level / scale).
The loss averages MSE across all sub-levels within each group,
then computes a weighted sum across groups.

Loss formula:
    kd_loss = Σ(level_weight × mean(MSE per sub-level))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """Per-level weighted MSE distillation loss.

    Args:
        level_weights: Dict mapping level name to scalar weight.
            Expected keys: backbone, fpn, head_cls, head_reg.
        temperature: Temperature scaling factor (reserved for future use).
    """

    def __init__(
        self,
        level_weights: dict[str, float],
        temperature: float = 1.0,
    ):
        super().__init__()
        self.level_weights = level_weights
        self.temperature = temperature

    def forward(
        self,
        teacher_feats: dict[str, list[torch.Tensor]],
        student_feats: dict[str, list[torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted MSE distillation loss across all levels.

        Args:
            teacher_feats: Dict with keys {backbone, fpn, head_cls, head_reg},
                each mapping to a list of tensors from the projected teacher.
            student_feats: Dict with same keys, each mapping to a list of
                tensors from the student model.

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
                # No detach() on t_feat: `teacher_feats` are the outputs of the
                # student-owned kd_proj_* modules, whose only gradient source is
                # this loss. The raw teacher forward is already detached inside
                # KDTrainer._train_epoch, so nothing propagates into the teacher.
                level_mse = level_mse + F.mse_loss(s_feat, t_feat)

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
