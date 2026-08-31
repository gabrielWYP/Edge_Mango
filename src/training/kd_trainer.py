"""
Knowledge Distillation Trainer — subclass of Trainer with frozen teacher.

Extends the base Trainer with:
    - Frozen MasterModel teacher loaded from checkpoint
    - 4 ProjectionLayers groups attached to StudentModel as submodules
    - Overridden _train_epoch with KD forward pass (teacher → project → MSE)
    - Inherited _validate (student-only), fit, checkpoint, early stopping

Teacher is stored as a plain attribute (NOT nn.Module submodule) so it is
naturally excluded from optimizer.param_groups and model.state_dict().
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .kd_config import KDConfig
from .kd_loss import KDLoss
from .loop import Trainer
from .precision import autocast_ctx
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

        teacher = MasterModel(
            num_classes=config.num_classes,
            pretrained_backbone=False,
            backbone_variant=config.backbone_variant,
            head_strides=config.head_strides,
        )

        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        teacher.load_state_dict(checkpoint["model_state_dict"])

        # Freeze all teacher parameters
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        print(f"[KDTrainer] Teacher loaded from {ckpt_path} "
              f"({sum(p.numel() for p in teacher.parameters()):,} params, frozen)")

        return teacher

    def _train_epoch(self, optimizer: AdamW, epoch: int, phase: int) -> dict:
        """Run one KD training epoch.

        Teacher forward under no_grad → project features → student forward →
        det_loss + kd_weight * kd_loss → backward → grad clip → step.

        D-G (fusion-redesign), mirrored from `Trainer._train_epoch`: this
        method used to catch OOM and NaN/Inf, print a warning, and
        `continue` — before the batch counter incremented — then floor the
        loss divisor at `max(n_batches, 1)`. An epoch where every batch
        failed therefore completed with finite-looking averaged losses,
        indistinguishable from a clean run, with zero coverage in `tests/`
        to catch it. Same policy as `loop.py`, so a reader does not have to
        learn two sets of rules:

        - `oom_skipped`: tolerated on epoch 1 only (allocator warm-up);
          raises immediately from epoch 2 onward.
        - `nan_skipped`: always counted, never itself fatal.
        - `steps_taken`: optimizer steps actually executed; zero in any
          epoch is always fatal, regardless of cause.

        Divergence from `loop.py`, stated explicitly: `KDTrainer` has no
        gradient accumulation (`grad_accum_steps` is not part of its
        structure and adding it is out of scope for this fix), so
        `steps_taken` here is simply "successful micro-batches" — one
        optimizer step per successful batch, as it always was for this
        trainer. There is no separate micro-batch/effective-step split to
        preserve, unlike the base `Trainer`.

        Returns:
            Dict with avg cls_loss, box_loss, kd_loss, total_loss, and the
            oom_skipped/nan_skipped/steps_taken counters.
        """
        self.model.train()
        self.teacher.eval()  # Ensure teacher stays in eval mode

        total_cls = 0.0
        total_box = 0.0
        total_kd = 0.0
        total_loss = 0.0
        steps_taken = 0
        oom_skipped = 0
        nan_skipped = 0

        for batch in self.train_loader:
            rgb = batch["rgb"].to(self.device)
            nir = batch["nir"].to(self.device)
            bboxes = [b.to(self.device) for b in batch["bboxes"]]
            labels = [l.to(self.device) for l in batch["labels"]]

            optimizer.zero_grad(set_to_none=True)

            try:
                with autocast_ctx(self.device.type, self.config.precision):
                    # --- Teacher forward (no grad) ---
                    # The no_grad scope covers the teacher forward and the
                    # stride selection ONLY. The kd_proj_* modules below are
                    # student-owned trainable parameters, so running them here
                    # would detach them from the graph and leave them frozen at
                    # their random init for the whole run.
                    with torch.no_grad():
                        t_out = self.teacher(rgb, nir)
                        # fusion-redesign D-3/§5: the teacher may emit more
                        # pyramid/head levels than the student's fixed 3
                        # (default head_strides=[4,8,16,32] vs the student's
                        # [8,16,32]). Select the matching levels by stride,
                        # not by position — a positional zip/truncation would
                        # silently distill P2/P3/P4 into student P3/P4/P5.
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

                    # --- Teacher-to-student projections (MUST carry grad) ---
                    # Inputs are already detached by the no_grad block above, so
                    # no teacher gradient is produced; the graph built here ends
                    # at the kd_proj_* parameters, which is exactly what the KD
                    # loss needs in order to train them.
                    # `distill_backbone` (renamed from `distill_backbone_rgb`,
                    # fusion-redesign W7/W4): the teacher's backbone is now
                    # a single early-fused stream, so this is no longer an
                    # "RGB-only" feature set — it carries NIR information
                    # too (design.md D-5). [2:] still selects S3, S4;
                    # channels [384, 768] are unchanged.
                    proj_backbone = self.model.kd_proj_backbone(
                        t_out["distill_backbone"][2:]
                    )
                    proj_fpn = self.model.kd_proj_fpn(fpn_for_student)
                    proj_head_cls = self.model.kd_proj_head_cls(head_cls_for_student)
                    proj_head_reg = self.model.kd_proj_head_reg(head_reg_for_student)

                    # --- Student forward (RGB only) ---
                    s_out = self.model(rgb)

                    # --- Detection loss ---
                    targets = {"bboxes": bboxes, "labels": labels}
                    det_loss, det_dict = self.criterion(s_out["preds"], targets)

                    # --- KD loss ---
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
                    kd_loss, kd_per_level = self.kd_criterion(kd_teacher, kd_student)

                    # --- Total loss ---
                    loss = det_loss + self.config.kd_weight * kd_loss

                # NaN / Inf guard: skip batch if loss explodes. Always
                # counted — see D-G docstring above for why this must not
                # also be silent.
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_skipped += 1
                    print(f"  [NaN] Skipping batch (loss={loss.item():.2f}), "
                          f"nan_skipped={nan_skipped}")
                    continue

                if self.config.precision == "fp16":
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                else:
                    loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                if self.config.precision == "fp16":
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_skipped += 1
                    torch.cuda.empty_cache()
                    if epoch >= 2:
                        raise RuntimeError(
                            f"[OOM] Batch skipped on epoch {epoch} (phase {phase}); "
                            "OOM tolerance is limited to epoch 1 (allocator "
                            "warm-up). A run that keeps OOM-ing past epoch 1 is "
                            "not a valid training run — reduce batch_size "
                            "instead of retrying. "
                            f"oom_skipped={oom_skipped}, nan_skipped={nan_skipped}, "
                            f"steps_taken={steps_taken}."
                        ) from e
                    print(
                        f"  [OOM] Skipping batch (epoch 1 warm-up tolerance), "
                        f"clearing CUDA cache. oom_skipped={oom_skipped}"
                    )
                    continue
                raise

            total_cls += det_dict["cls_loss"]
            total_box += det_dict["box_loss"]
            total_kd += kd_loss.item()
            total_loss += loss.item()
            steps_taken += 1
            self.global_step += 1

        if steps_taken == 0:
            raise RuntimeError(
                f"KD training epoch {epoch} (phase {phase}) took zero optimizer "
                f"steps (oom_skipped={oom_skipped}, nan_skipped={nan_skipped}). "
                "A run with zero steps produced no gradient update and must "
                "not be reported as a completed epoch — see "
                "openspec/changes/fusion-redesign/design.md D-G."
            )

        return {
            "cls_loss": total_cls / steps_taken,
            "box_loss": total_box / steps_taken,
            "kd_loss": total_kd / steps_taken,
            "total_loss": total_loss / steps_taken,
            "oom_skipped": float(oom_skipped),
            "nan_skipped": float(nan_skipped),
            "steps_taken": float(steps_taken),
        }
