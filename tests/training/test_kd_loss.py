"""Tests for `KDLoss` gradient flow into the teacher-side (projection) tensor.

Covers slice 2 of `fix-detection-eval-and-kd-blockers`: `KDLoss.forward` must
NOT detach `t_feat` before computing MSE, or gradients never reach the
`kd_proj_*` projection parameters. Model-free — a single `nn.Conv2d` stands
in for a projection layer, so no `KDTrainer`/`MasterModel` is needed.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.kd_loss import KDLoss


class TestKDLossGradientFlow:
    """RED test for the `t_feat.detach()` removal (kd_loss.py:71)."""

    def test_teacher_side_projection_receives_gradient(self):
        torch.manual_seed(0)

        # Stand-in for a kd_proj_* projection: a single learnable 1x1 conv.
        conv = nn.Conv2d(4, 4, 1)
        x = torch.randn(1, 4, 8, 8)
        t_feat = conv(x)  # projection output — carries grad_fn from conv.weight

        s_feat = torch.randn(1, 4, 8, 8, requires_grad=True)

        kd_loss = KDLoss(level_weights={"backbone": 1.0})
        total_loss, per_level = kd_loss(
            teacher_feats={"backbone": [t_feat]},
            student_feats={"backbone": [s_feat]},
        )
        total_loss.backward()

        assert conv.weight.grad is not None, (
            "conv.weight.grad is None — the teacher-side tensor is still "
            "detached before the MSE computation, severing the projection "
            "gradient path"
        )
        assert conv.weight.grad.abs().sum().item() > 0, (
            "conv.weight.grad is all-zero — the projection received no "
            "meaningful gradient signal from the KD loss"
        )

    def test_student_side_still_receives_gradient(self):
        """Sanity: the previously-working student-side gradient path is unaffected."""
        torch.manual_seed(0)

        conv = nn.Conv2d(4, 4, 1)
        x = torch.randn(1, 4, 8, 8)
        t_feat = conv(x)
        s_feat = torch.randn(1, 4, 8, 8, requires_grad=True)

        kd_loss = KDLoss(level_weights={"backbone": 1.0})
        total_loss, _ = kd_loss(
            teacher_feats={"backbone": [t_feat]},
            student_feats={"backbone": [s_feat]},
        )
        total_loss.backward()

        assert s_feat.grad is not None
        assert s_feat.grad.abs().sum().item() > 0

    def test_per_level_dict_is_reporting_only_and_does_not_break_backward(self):
        """`per_level[level_name] = level_mse.item()` must stay a plain float
        read that does not interfere with the loss graph."""
        torch.manual_seed(0)

        conv_backbone = nn.Conv2d(4, 4, 1)
        conv_fpn = nn.Conv2d(2, 2, 1)
        x_backbone = torch.randn(1, 4, 4, 4)
        x_fpn = torch.randn(1, 2, 4, 4)

        t_backbone = conv_backbone(x_backbone)
        t_fpn = conv_fpn(x_fpn)
        s_backbone = torch.randn(1, 4, 4, 4, requires_grad=True)
        s_fpn = torch.randn(1, 2, 4, 4, requires_grad=True)

        kd_loss = KDLoss(level_weights={"backbone": 1.0, "fpn": 0.1})
        total_loss, per_level = kd_loss(
            teacher_feats={"backbone": [t_backbone], "fpn": [t_fpn]},
            student_feats={"backbone": [s_backbone], "fpn": [s_fpn]},
        )

        assert isinstance(per_level["backbone"], float)
        assert isinstance(per_level["fpn"], float)

        total_loss.backward()

        assert conv_backbone.weight.grad is not None
        assert conv_backbone.weight.grad.abs().sum().item() > 0
        assert conv_fpn.weight.grad is not None
        assert conv_fpn.weight.grad.abs().sum().item() > 0
