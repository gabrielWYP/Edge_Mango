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
NUM_MASKS = 32  # YOLOv8-seg: prototypes, and coefficients per anchor
PROTO_CHANNELS = 64  # hidden width of the Proto module (yolov8n-seg's `npr`)
TASKS = ("detect", "segment")


class Proto(nn.Module):
    """YOLOv8-seg prototype branch: P3 features -> `num_masks` prototypes at twice the P3 size.

    On a 640 input P3 is 80x80 (stride 8), so the prototypes are 160x160, 1/4 of the input.
    Layer names (`cv1`, `upsample`, `cv2`, `cv3`) follow the official `Proto`, so the
    yolov8n-seg weights map onto them one to one (`pretrained.official_key`).

    Args:
        c_in: Channels of the P3 feature map.
        c_hidden: Hidden channels.
        num_masks: Number of prototypes.
    """

    def __init__(self, c_in: int, c_hidden: int = PROTO_CHANNELS, num_masks: int = NUM_MASKS) -> None:
        super().__init__()
        self.cv1 = Conv_BN_SiLU(c_in, c_hidden, 3, 1)
        self.upsample = nn.ConvTranspose2d(c_hidden, c_hidden, 2, 2, 0, bias=True)
        self.cv2 = Conv_BN_SiLU(c_hidden, c_hidden, 3, 1)
        self.cv3 = Conv_BN_SiLU(c_hidden, num_masks, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


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
        num_masks: Mask coefficients per anchor; 0 (default) builds no mask branch, which is the
            plain detection head. Otherwise `mask_stem` (2 x Conv_BN_SiLU 3x3) and a 1x1
            `mask_pred` produce them, as the `cv4` branch of the official Segment head.
    """

    def __init__(
        self, c_in: int, c_stem: int, num_classes: int = NUM_CLASSES, num_masks: int = 0
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_masks = num_masks

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

        if num_masks > 0:
            # Built after `_init_weights` so the detection layers draw the same init as in a
            # detect head, and the mask layers keep the PyTorch default init (as in YOLOv8-seg).
            c_mask = max(c_in // 4, num_masks)
            self.mask_stem = nn.Sequential(
                Conv_BN_SiLU(c_in, c_mask, 3, 1),
                Conv_BN_SiLU(c_mask, c_mask, 3, 1),
            )
            self.mask_pred = nn.Conv2d(c_mask, num_masks, 1, 1, 0)

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
                mask_coefs: (N, num_masks, H, W) — only when the head has a mask branch
        """
        # Same two convs as `self.cls_stem(x)`, split to expose the first output
        # before its activation, the point where CrossKD aligns the features.
        cls_mid = self._before_activation(self.cls_stem[0], x)
        reg_mid = self._before_activation(self.reg_stem[0], x)
        cls_feat = self.cls_stem[1](F.silu(cls_mid))
        reg_feat = self.reg_stem[1](F.silu(reg_mid))
        cls_logits = self.cls_pred(cls_feat)
        reg_logits = self.reg_pred(reg_feat)

        out = {
            "cls_logits": cls_logits,
            "reg_logits": reg_logits,
            "distill_cls": cls_feat,
            "distill_reg": reg_feat,
            "distill_cls_mid": cls_mid,
            "distill_reg_mid": reg_mid,
        }
        if self.num_masks > 0:
            out["mask_coefs"] = self.mask_pred(self.mask_stem(x))
        return out


class YOLOStudentHead(nn.Module):
    """
    Complete detection head for all 3 FPN levels.

    Instantiates one DecoupledHead per level with:
        fpn_channels:  [128, 256, 256] (from PANet output)
        stem_channels: [64, 128, 256]   (matching head_projections contract)

    With `task="segment"` every level also predicts `num_masks` mask coefficients and a
    `Proto` module on P3 (the first level) yields the prototypes: the YOLOv8-seg design. The
    detection outputs are untouched, so KD and decoding read them as before.

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
        task: str = "detect",
        num_masks: int = NUM_MASKS,
    ) -> None:
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}: {TASKS}")
        self.num_classes = num_classes
        self.task = task
        fpn_channels = fpn_channels or self.FPN_CHANNELS
        mask_channels = num_masks if task == "segment" else 0

        self.heads = nn.ModuleList([
            DecoupledHead(
                c_in=fpn_ch, c_stem=stem_ch, num_classes=num_classes, num_masks=mask_channels
            )
            for fpn_ch, stem_ch in zip(fpn_channels, stem_channels or self.STEM_CHANNELS)
        ])
        if task == "segment":
            self.proto = Proto(fpn_channels[0], PROTO_CHANNELS, num_masks)

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
                mask_coefs: list of (N, num_masks, H_i, W_i), segment task only
                protos: (N, num_masks, 2*H_0, 2*W_0), segment task only
        """
        levels = [fpn_features["p3"], fpn_features["p4"], fpn_features["p5"]]

        all_preds = []
        all_cls = []
        all_reg = []
        distill_cls = []
        distill_reg = []
        distill_cls_mid = []
        distill_reg_mid = []
        mask_coefs = []

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
            if "mask_coefs" in out:
                mask_coefs.append(out["mask_coefs"])

        result = {
            "preds": all_preds,
            "cls_preds": all_cls,
            "reg_preds": all_reg,
            "distill_head_cls": distill_cls,
            "distill_head_reg": distill_reg,
            "distill_head_cls_mid": distill_cls_mid,
            "distill_head_reg_mid": distill_reg_mid,
        }
        if self.task == "segment":
            result["mask_coefs"] = mask_coefs
            result["protos"] = self.proto(levels[0])
        return result
