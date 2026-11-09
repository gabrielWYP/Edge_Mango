"""
Decoupled detection head for YOLOv8-Nano student model.

Per FPN level, separate cls and reg stems with intermediate features
exposed for knowledge distillation (KD).

Stem channels per level: [64, 128, 256] matching head_projections
student_channels contract.

Architecture per level:
    cls_stem: Conv_BN_SiLU(3x3, c_in->c_stem) -> Conv_BN_SiLU(3x3, c_stem->c_stem)
    cls_pred: Conv2d(1x1, c_stem->num_classes)
    reg_stem: Conv_BN_SiLU(3x3, c_in->c_stem) -> Conv_BN_SiLU(3x3, c_stem->c_stem)
    reg_pred: Conv2d(1x1, c_stem->4)

    distill_cls = cls_stem output (after 2nd conv, before cls_pred)
    distill_reg = reg_stem output (after 2nd conv, before reg_pred)

Channel contract with ProjectionLayers:
    head_projections student_channels = [64, 128, 256] for (P3, P4, P5).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.student.backbone import Conv_BN_SiLU

NUM_CLASSES = 2  # sano, danado


class DecoupledHead(nn.Module):
    """
    Decoupled head for a single FPN level.

    Two 3x3 conv stem branches (cls + reg), each followed by a 1x1
    prediction conv. Intermediate stem outputs are exposed as distill_cls
    and distill_reg for feature-level knowledge distillation.

    Args:
        c_in: Input channels from FPN level.
        c_stem: Stem hidden channels (also the distillation feature channels).
        num_classes: Number of detection classes (default 2).
    """

    def __init__(self, c_in: int, c_stem: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Classification branch: 2 x Conv_BN_SiLU (3x3)
        self.cls_stem = nn.Sequential(
            Conv_BN_SiLU(c_in, c_stem, 3, 1),
            Conv_BN_SiLU(c_stem, c_stem, 3, 1),
        )
        self.cls_pred = nn.Conv2d(c_stem, num_classes, 1, 1, 0)

        # Regression branch: 2 x Conv_BN_SiLU (3x3)
        self.reg_stem = nn.Sequential(
            Conv_BN_SiLU(c_in, c_stem, 3, 1),
            Conv_BN_SiLU(c_stem, c_stem, 3, 1),
        )
        self.reg_pred = nn.Conv2d(c_stem, 4, 1, 1, 0)

        self._init_weights()

    def _init_weights(self) -> None:
        """YOLOv8-style initialization: normal(0.01) for convs, bias init for cls."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Prior probability init for classification bias
        prior_prob = 0.01
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    @staticmethod
    def _before_activation(block: Conv_BN_SiLU, x: Tensor) -> Tensor:
        """Output of a Conv_BN_SiLU block without its SiLU, which is in-place."""
        return block.bn(block.conv(x))

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Args:
            x: (N, c_in, H, W) — feature map from one FPN level.

        Returns:
            dict with:
                cls_logits: (N, num_classes, H, W)
                reg_logits: (N, 4, H, W)
                distill_cls: (N, c_stem, H, W) — intermediate cls features for KD
                distill_reg: (N, c_stem, H, W) — intermediate reg features for KD
                distill_cls_mid: (N, c_stem, H, W) — first cls stem conv, before its SiLU
                distill_reg_mid: (N, c_stem, H, W) — first reg stem conv, before its SiLU
                (the two `*_mid` outputs feed cross-head distillation)
        """
        # Same two convs as `self.cls_stem(x)`, split to expose the first output
        # before its activation, the point where CrossKD aligns the features.
        cls_mid = self._before_activation(self.cls_stem[0], x)
        reg_mid = self._before_activation(self.reg_stem[0], x)
        cls_feat = self.cls_stem[1](F.silu(cls_mid))
        reg_feat = self.reg_stem[1](F.silu(reg_mid))
        cls_logits = self.cls_pred(cls_feat)
        reg_logits = self.reg_pred(reg_feat)

        return {
            "cls_logits": cls_logits,
            "reg_logits": reg_logits,
            "distill_cls": cls_feat,
            "distill_reg": reg_feat,
            "distill_cls_mid": cls_mid,
            "distill_reg_mid": reg_mid,
        }


class YOLOStudentHead(nn.Module):
    """
    Complete detection head for all 3 FPN levels.

    Instantiates one DecoupledHead per level with:
        fpn_channels:  [128, 256, 256] (from PANet output)
        stem_channels: [64, 128, 256]   (matching head_projections contract)

    Forward returns a dict with 5 keys:
        preds:            [(N,6,H,W), ...]  — cls+reg concatenated
        cls_preds:        [(N,2,H,W), ...]  — class logits
        reg_preds:        [(N,4,H,W), ...]  — bbox regression
        distill_head_cls: [(N,64,H,W), (N,128,H,W), (N,256,H,W)]
        distill_head_reg: [(N,64,H,W), (N,128,H,W), (N,256,H,W)]
    """

    FPN_CHANNELS = [128, 256, 256]
    STEM_CHANNELS = [64, 128, 256]

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        fpn_channels: list[int] | None = None,
        stem_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.heads = nn.ModuleList([
            DecoupledHead(c_in=fpn_ch, c_stem=stem_ch, num_classes=num_classes)
            for fpn_ch, stem_ch in zip(
                fpn_channels or self.FPN_CHANNELS, stem_channels or self.STEM_CHANNELS
            )
        ])

    def forward(self, fpn_features: dict[str, Tensor]) -> dict[str, list]:
        """
        Args:
            fpn_features: dict with 'p3', 'p4', 'p5' from PANet.

        Returns:
            dict with keys:
                preds:            list of (N, num_classes+4, H_i, W_i)
                cls_preds:        list of (N, num_classes, H_i, W_i)
                reg_preds:        list of (N, 4, H_i, W_i)
                distill_head_cls: list of (N, stem_ch_i, H_i, W_i)
                distill_head_reg: list of (N, stem_ch_i, H_i, W_i)
                distill_head_cls_mid / distill_head_reg_mid: list of
                    (N, stem_ch_i, H_i, W_i), output of the first stem conv
        """
        levels = [fpn_features["p3"], fpn_features["p4"], fpn_features["p5"]]

        all_preds = []
        all_cls = []
        all_reg = []
        distill_cls = []
        distill_reg = []
        distill_cls_mid = []
        distill_reg_mid = []

        for head, feat in zip(self.heads, levels):
            out = head(feat)
            pred = torch.cat([out["cls_logits"], out["reg_logits"]], dim=1)
            all_preds.append(pred)
            all_cls.append(out["cls_logits"])
            all_reg.append(out["reg_logits"])
            distill_cls.append(out["distill_cls"])
            distill_reg.append(out["distill_reg"])
            distill_cls_mid.append(out["distill_cls_mid"])
            distill_reg_mid.append(out["distill_reg_mid"])

        return {
            "preds": all_preds,
            "cls_preds": all_cls,
            "reg_preds": all_reg,
            "distill_head_cls": distill_cls,
            "distill_head_reg": distill_reg,
            "distill_head_cls_mid": distill_cls_mid,
            "distill_head_reg_mid": distill_reg_mid,
        }
