"""
StudentModel — YOLOv8-Nano integration for knowledge distillation.

Composes CSPDarknet-Nano backbone → PANet neck → DecoupledHead into a
single forward pipeline that produces a 7-key output dict matching the
teacher MasterModel's distillation contract.

Input:  RGB only (N, 3, 640, 640)
Output: 7-key dict with detection predictions + intermediate features for KD

Pipeline:
    x (N,3,640,640)
        ↓
    CSPDarknetNano → stages [S1..S4], distill (S3, S4)
        ↓
    PANet → {p3, p4, p5}
        ↓
    YOLOStudentHead → preds, cls_preds, reg_preds, distill_head_cls, distill_head_reg
        ↓
    7-key output dict

Channel contract (all dims locked to the KD adapter presets' input side;
the adapters themselves are owned by `KDTrainer`, not by this model):
    distill_backbone:  [128, 256]       → backbone_projections student_channels
    distill_fpn:       [128, 256, 256]  → fpn_projections student_channels
    distill_head_cls:  [64, 128, 256]   → head_projections student_channels
    distill_head_reg:  [64, 128, 256]   → head_projections student_channels
"""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.models.student.backbone import CSPDarknetNano
from src.models.student.neck import NanoPANet, PANet
from src.models.student.head import YOLOStudentHead, NUM_CLASSES, TASKS

# Prefix of the training-only KD adapters that `KDTrainer` used to attach to
# the student as submodules (`kd_proj_backbone`, `kd_proj_fpn`,
# `kd_proj_head_cls`, `kd_proj_head_reg`). KD checkpoints written before the
# adapters moved to the trainer carry them inside `model_state_dict`; they
# are not part of the student architecture and never needed for inference.
LEGACY_KD_ADAPTER_PREFIX = "kd_proj_"


def strip_legacy_kd_adapter_keys(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return `state_dict` without legacy `kd_proj_*` KD adapter entries.

    Lets an old KD checkpoint load into a bare `StudentModel` with
    `strict=True` — every remaining key is still checked. A no-op for any
    checkpoint written since the adapters became trainer-owned.
    """
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(LEGACY_KD_ADAPTER_PREFIX)
    }


# "wide" is the student of the v4 plan; "nano" has the neck and head channels of YOLOv8n.
STUDENT_VARIANTS = ("wide", "nano")
NANO_STEM_CHANNELS = [64, 64, 64]
# 3: RGB, the student of the plan. 4: RGB and NIR stacked at the input, for `est_rgbnir`, the arm of H1 and the teacher of `kd_rgbnir`.
STUDENT_IN_CHANNELS = (3, 4)


class StudentModel(nn.Module):
    """
    YOLOv8-Nano student model for knowledge distillation.

    Single-modality RGB input, produces 7-key output dict with detection
    predictions and intermediate features at backbone, FPN, and head levels.

    Args:
        num_classes: Number of detection classes (default 2: sano, danado).
        in_channels: 3 reads `x` alone. 4 stacks `nir` after `x`, and then needs both at inference.
        task: "detect" (default, boxes only) or "segment" (YOLOv8-seg: the head also returns
            `mask_coefs` and `protos`). "detect" builds exactly the layers it always had.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        variant: str = "wide",
        in_channels: int = 3,
        task: str = "detect",
    ) -> None:
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"Unknown student task {task!r}: {TASKS}")
        self.task = task
        if variant not in STUDENT_VARIANTS:
            raise ValueError(f"Unknown student variant {variant!r}: {sorted(STUDENT_VARIANTS)}")
        if in_channels not in STUDENT_IN_CHANNELS:
            raise ValueError(f"Unknown student in_channels {in_channels}: {STUDENT_IN_CHANNELS}")
        self.num_classes = num_classes
        self.variant = variant
        self.in_channels = in_channels

        self.backbone = CSPDarknetNano(in_channels)
        if variant == "nano":
            self.neck = NanoPANet()
            self.head = YOLOStudentHead(
                num_classes=num_classes,
                fpn_channels=NanoPANet.OUTPUT_CHANNELS,
                stem_channels=NANO_STEM_CHANNELS,
                task=task,
            )
        else:
            self.neck = PANet()
            self.head = YOLOStudentHead(num_classes=num_classes, task=task)

    def forward(self, x: Tensor, nir: Tensor | None = None) -> dict[str, Any]:
        """
        Forward pass: backbone → neck → head → 7-key output dict.

        Args:
            x: (N, 3, 640, 640) RGB input tensor.
            nir: (N, 1, 640, 640) NIR tensor, required with `in_channels=4` and unused otherwise.

        Returns:
            dict with exactly 7 keys:
                'preds':            [(N,6,80,80), (N,6,40,40), (N,6,20,20)]
                'cls_preds':        [(N,2,80,80), (N,2,40,40), (N,2,20,20)]
                'reg_preds':        [(N,4,80,80), (N,4,40,40), (N,4,20,20)]
                'distill_backbone': [(N,128,40,40), (N,256,20,20)]
                'distill_fpn':      [(N,128,80,80), (N,256,40,40), (N,256,20,20)]
                'distill_head_cls': [(N,64,80,80), (N,128,40,40), (N,256,20,20)]
                'distill_head_reg': [(N,64,80,80), (N,128,40,40), (N,256,20,20)]
            plus 'distill_head_cls_mid' / 'distill_head_reg_mid', same shapes:
            the output of the first stem conv (9 keys in total). With `task="segment"` two more:
                'mask_coefs':       [(N,32,80,80), (N,32,40,40), (N,32,20,20)]
                'protos':           (N,32,160,160)
        """
        if self.in_channels == 4:
            if nir is None:
                raise ValueError("A student with in_channels=4 needs the NIR tensor.")
            x = torch.cat([x, nir], dim=1)

        # Backbone: extract multi-scale features
        stages, (s3, s4) = self.backbone(x)

        # Neck: bidirectional feature fusion (FPN + PAN)
        fpn = self.neck(stages)

        # Head: decoupled detection predictions
        head_out = self.head(fpn)

        output = {
            # Detection outputs
            "preds": head_out["preds"],
            "cls_preds": head_out["cls_preds"],
            "reg_preds": head_out["reg_preds"],
            # Distillation features
            "distill_backbone": [s3, s4],
            "distill_fpn": [fpn["p3"], fpn["p4"], fpn["p5"]],
            "distill_head_cls": head_out["distill_head_cls"],
            "distill_head_reg": head_out["distill_head_reg"],
            # First stem conv of each head branch, for cross-head KD.
            "distill_head_cls_mid": head_out["distill_head_cls_mid"],
            "distill_head_reg_mid": head_out["distill_head_reg_mid"],
        }
        if self.task == "segment":
            # Mask coefficients per level and the P3 prototypes; the keys above are unchanged.
            output["mask_coefs"] = head_out["mask_coefs"]
            output["protos"] = head_out["protos"]
        return output

    def freeze_backbone(self, freeze_stages: int = 0) -> None:
        """No-op for single-phase student training (API compatibility).

        This method exists so that Trainer can call freeze_backbone()
        polymorphically on both MasterModel and StudentModel. Since student
        training is single-phase with all parameters trainable from epoch 1,
        no freezing is performed.
        """
        pass

    def unfreeze_backbone_stages(
        self,
        unfreeze_stages: list[int],
        unfreeze_stem: bool = False,
    ) -> None:
        """No-op for single-phase student training (API compatibility).

        This method exists so that Trainer can call unfreeze_backbone_stages()
        polymorphically on both MasterModel and StudentModel. Since student
        training is single-phase with all parameters trainable from epoch 1,
        no unfreezing is performed. `unfreeze_stem` mirrors MasterModel's
        signature (renamed from `unfreeze_rgb_stem`, fusion-redesign D-4 —
        there is only one stem now) and is accepted but ignored — the
        student has no separate stem to unfreeze here.
        """
        pass

    def count_parameters(self) -> dict[str, int]:
        """Returns parameter counts per module and total."""
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "backbone": count(self.backbone),
            "neck": count(self.neck),
            "head": count(self.head),
            "total": count(self),
        }
