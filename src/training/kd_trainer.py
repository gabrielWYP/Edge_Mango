"""
Knowledge Distillation Trainer — subclass of Trainer with frozen teacher.

Extends the base Trainer with:
    - Frozen MasterModel teacher loaded from checkpoint
    - 4 ProjectionLayers groups attached to StudentModel as submodules
    - Overridden _forward_loss with the KD forward pass (teacher → project → MSE)
    - Inherited _train_epoch (gradient accumulation, OOM/NaN guards, counters),
      _validate (student-only), fit, checkpoint, early stopping

Teacher is stored as a plain attribute (NOT nn.Module submodule) so it is
naturally excluded from optimizer.param_groups and model.state_dict().
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .kd_config import KDConfig
from .kd_loss import KDLoss
from .loop import Trainer
from .strides import STUDENT_STRIDES, select_by_strides
from src.models.master.master_model import MasterModel
from src.models.master.distill_projections import (
    backbone_projections,
    fpn_projections,
    head_projections,
)

# The teacher (MasterModel) may emit more levels than the student's fixed
# `STUDENT_STRIDES` (default [4, 8, 16, 32], fusion-redesign D-3); KD must
# select the matching 3 by stride rather than by position — see
# `select_by_strides` and design.md §5's ProjectionLayers hazard.


class KDTrainer(Trainer):
    """Knowledge distillation trainer with frozen teacher.

    Attaches 4 projection groups to the student model as registered submodules,
    loads a frozen MasterModel teacher, and overrides _train_epoch to compute
    both detection loss and per-level MSE distillation loss.

    Args:
        model: StudentModel instance (projections attached before super().__init__).
        config: KDConfig with teacher_checkpoint and KD hyperparameters.
        train_loader: Training data loader.
        val_loader: Validation data loader.
    """

    def __init__(
        self,
        model: nn.Module,
        config: KDConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        # Attach projection layers BEFORE super().__init__() so they are
        # moved to device by self.model.to(device) in Trainer.__init__.
        model.kd_proj_backbone = backbone_projections()
        model.kd_proj_fpn = fpn_projections()
        model.kd_proj_head_cls = head_projections()
        model.kd_proj_head_reg = head_projections()

        super().__init__(model, config, train_loader, val_loader)

        # Load frozen teacher — stored as plain attribute, NOT a submodule.
        # This ensures teacher params are excluded from:
        #   - optimizer.param_groups (built in _train_phase via self.model.parameters())
        #   - model.state_dict() (used in inherited _save_checkpoint)
        self.teacher = self._load_teacher(config)
        self.teacher.to(self.device)

        # KD loss
        self.kd_criterion = KDLoss(
            level_weights=config.distill_levels,
            temperature=config.kd_temperature,
        ).to(self.device)

    @staticmethod
    def _load_teacher(config: KDConfig) -> MasterModel:
        """Load a frozen MasterModel from checkpoint.

        Args:
            config: KDConfig with teacher_checkpoint path and num_classes.

        Returns:
            MasterModel in eval mode with all params frozen.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        ckpt_path = Path(config.teacher_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {ckpt_path}"
            )

        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        # The teacher's geometry is a property of ITS checkpoint, not of the KD
        # run's config. `Trainer._save_checkpoint` stores the full training
        # config, so read the architecture back from there and fall back to the
        # KD config only for a checkpoint written before that key existed.
        #
        # Without this, a master trained under configs/training_mango.yaml
        # (`head_strides: [8, 16, 32]`) cannot be loaded by a KD run carrying
        # the post-fusion-redesign 4-level default: `load_state_dict` fails on
        # the missing `head.heads.3.*` weights. `select_by_strides` in
        # `_train_epoch` already reads `self.teacher.head_strides`, so a teacher
        # with fewer levels than the default stays correctly aligned.
        teacher_config = checkpoint.get("config") or {}
        teacher = MasterModel(
            num_classes=config.num_classes,
            pretrained_backbone=False,
            backbone_variant=teacher_config.get(
                "backbone_variant", config.backbone_variant
            ),
            head_strides=teacher_config.get("head_strides", config.head_strides),
            in_channels=teacher_config.get("in_channels", config.in_channels),
        )
        teacher.load_state_dict(checkpoint["model_state_dict"])

        # Freeze all teacher parameters
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        print(f"[KDTrainer] Teacher loaded from {ckpt_path} "
              f"({sum(p.numel() for p in teacher.parameters()):,} params, frozen, "
              f"head_strides={teacher.head_strides})")

        return teacher

    def _forward_loss(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
        bboxes: list[torch.Tensor],
        labels: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Teacher forward (no grad) -> projections -> student forward ->
        det_loss + kd_weight * kd_loss.

        Only the forward pass and the loss live here. The base
        `Trainer._train_epoch` owns the loop — gradient accumulation against
        `effective_batch`, the D-G OOM/NaN guards, the counters — so the
        distilled stage receives exactly the optimizer budget the baseline
        does. This trainer used to carry its own copy of that loop, and the
        copy never gained accumulation: under `batch_size: 4` it took 13
        steps per epoch against the baseline's 7 (run
        `20260901T200124175077Z`).
        """
        # The no_grad scope covers the teacher forward and the stride
        # selection ONLY. The kd_proj_* modules below are student-owned
        # trainable parameters, so running them here would detach them from
        # the graph and leave them frozen at their random init for the whole
        # run. The teacher is set to eval() once in `_load_teacher`; it is a
        # plain attribute, not a submodule, so `self.model.train()` cannot
        # flip it back.
        with torch.no_grad():
            t_out = self.teacher(rgb, nir)
            # fusion-redesign D-3/§5: the teacher may emit more pyramid/head
            # levels than the student's fixed 3 (default
            # head_strides=[4,8,16,32] vs the student's [8,16,32]). Select
            # the matching levels by stride, not by position — a positional
            # zip/truncation would silently distill P2/P3/P4 into student
            # P3/P4/P5.
            teacher_strides = self.teacher.head_strides
            fpn_for_student = select_by_strides(
                t_out["distill_fpn"], teacher_strides, STUDENT_STRIDES
            )
            head_cls_for_student = select_by_strides(
                t_out["distill_head_cls"], teacher_strides, STUDENT_STRIDES
            )
            head_reg_for_student = select_by_strides(
                t_out["distill_head_reg"], teacher_strides, STUDENT_STRIDES
            )

        # Teacher-to-student projections (MUST carry grad). Inputs are
        # already detached by the no_grad block above, so no teacher gradient
        # is produced; the graph built here ends at the kd_proj_* parameters,
        # which is exactly what the KD loss needs in order to train them.
        # `distill_backbone` (renamed from `distill_backbone_rgb`,
        # fusion-redesign W7/W4): the teacher's backbone is now a single
        # early-fused stream, so this is no longer an "RGB-only" feature set
        # — it carries NIR information too (design.md D-5). [2:] still
        # selects S3, S4; channels [384, 768] are unchanged.
        proj_backbone = self.model.kd_proj_backbone(t_out["distill_backbone"][2:])
        proj_fpn = self.model.kd_proj_fpn(fpn_for_student)
        proj_head_cls = self.model.kd_proj_head_cls(head_cls_for_student)
        proj_head_reg = self.model.kd_proj_head_reg(head_reg_for_student)

        # Student forward (RGB only) and detection loss.
        s_out = self.model(rgb)
        targets = {"bboxes": bboxes, "labels": labels}
        det_loss, det_dict = self.criterion(s_out["preds"], targets)

        kd_teacher = {
            "backbone": proj_backbone,
            "fpn": proj_fpn,
            "head_cls": proj_head_cls,
            "head_reg": proj_head_reg,
        }
        kd_student = {
            "backbone": s_out["distill_backbone"],
            "fpn": s_out["distill_fpn"],
            "head_cls": s_out["distill_head_cls"],
            "head_reg": s_out["distill_head_reg"],
        }
        kd_loss, _ = self.kd_criterion(kd_teacher, kd_student)

        loss = det_loss + self.config.kd_weight * kd_loss
        return loss, {
            "cls_loss": det_dict["cls_loss"],
            "box_loss": det_dict["box_loss"],
            "kd_loss": kd_loss.item(),
        }
