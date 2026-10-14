"""
Training loop for MasterModel/StudentModel fine-tuning.

MasterModel (fusion-redesign D-4): default schedule is `end_to_end` —
every backbone stage, the stem, the neck and the head train from epoch 0,
with pretrained backbone stages at `backbone_lr_mult` times the neck/head
learning rate (discriminative LR). The previous `two_phase` schedule
(Phase 1: frozen backbone; Phase 2: partial unfreeze) is preserved and
selectable via `config.schedule = "two_phase"` for configs that explicitly
opt back into it (e.g. `training_mango.yaml`).

StudentModel: unchanged single-phase schedule.

Features:
    - Explicit fp32 | fp16 | bf16 precision selection (D-I)
    - Gradient accumulation with a fixed effective batch (D-H)
    - Fatal (counted, not silently skipped) OOM/NaN batch-skip guards (D-G)
    - Gradient clipping
    - CosineAnnealingWarmRestarts with linear warmup
    - Early stopping on val mAP@0.5
    - Best checkpoint + periodic checkpointing
    - TensorBoard logging, including per-module gradient norms (D-F)
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader

from .config import DAMAGE_CLASS_ID, TrainingConfig
from .decode import decode_detections
from .loss import YOLOv8Loss
from .fusion_modes import DEFAULT_FUSION_MODE, arch_version_for_mode
from .machine import derive_grad_accum_steps
from .metrics import compute_map, generate_training_curves, LossHistory
from .nir_registration import homography_sha256
from .precision import autocast_ctx, make_scaler
from .strides import resolve_active_strides

# Checkpoint schema version, per fusion mode (fusion-redesign D-C). The two
# `MasterModel` architectures share no state_dict key upstream of the
# detection head (`backbone.rgb_stem.*` / `backbone.nir_stem.*` /
# `backbone.shared_stages.*` / `fusion.*` / `neck.fpn_rgb.*` /
# `neck.fpn_nir.*` / `neck.fusion_convs.*` for
# `fusion_mode="cross_attention"` vs `backbone.stem.*` /
# `backbone.stages.*` / `neck.fpn.*` for `fusion_mode="early"`), so a
# checkpoint of one mode is unloadable into the other. `arch_version`
# converts a 200-line missing/unexpected-key dump into one actionable
# sentence — see `scripts/evaluate_checkpoint.py` and
# `scripts/visualize_damage_predictions.py`.
#
# `CHECKPOINT_ARCH_VERSION` remains the early-fusion (default-mode) version,
# unchanged at 2, so every existing importer keeps its meaning; the
# per-mode mapping lives in `src/training/fusion_modes.py`.
CHECKPOINT_ARCH_VERSION = arch_version_for_mode(DEFAULT_FUSION_MODE)


class Trainer:
    """Two-phase training loop for MasterModel.

    Args:
        model: MasterModel instance.
        config: TrainingConfig with all hyperparameters.
        train_loader: Training data loader. Optional — only required for
            `fit()`; `evaluate()` (D10) has no dependency on it, so eval-only
            entrypoints (`scripts/evaluate_checkpoint.py`) can omit it.
        val_loader: Validation data loader.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader | None,
        val_loader: DataLoader,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device selection (W9, fusion-redesign D-H): "auto" preserves the
        # previous hardcoded expression; any other value (e.g. "cuda:0",
        # "cpu") is passed through explicitly, closing the gap with
        # scripts/evaluate_checkpoint.py's existing --device flag.
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        self.model.to(self.device)

        # Gradient accumulation (D-H): effective_batch is the experimental
        # constant; batch_size is a memory knob. Derived, never read from a
        # machine profile directly — see src/training/machine.py.
        self.grad_accum_steps = derive_grad_accum_steps(
            config.effective_batch, config.batch_size
        )

        # Loss. `strides` is resolved from config rather than hardcoded
        # (fusion-redesign D-D) — `[8, 16, 32]` for model_type="student"
        # (out of scope, unchanged), `config.head_strides` for "master"
        # (default `[4, 8, 16, 32]`, includes the reconnected P2 level).
        self.criterion = YOLOv8Loss(
            num_classes=config.num_classes,
            box_weight=config.box_weight,
            cls_weight=config.cls_weight,
            class_weights=config.class_weights,
            focal_gamma=getattr(config, 'focal_gamma', 2.0),
            strides=resolve_active_strides(config),
            assigner_center_radius=getattr(config, "assigner_center_radius", 0.0),
            assigner_level_ranges=getattr(config, "assigner_level_ranges", None),
            assigner_collect_stats=getattr(config, "assigner_collect_stats", False),
        ).to(self.device)

        # Mixed precision: fp32 | fp16 | bf16 (src/training/precision.py).
        # Only fp16 needs loss scaling; the scaler is a no-op otherwise.
        self.scaler = make_scaler(config.precision)

        # Output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.loss_history = LossHistory()
        self.best_map50 = 0.0
        # Best value of `config.selection_metric` so far. -inf so the first
        # epoch always saves, even at score 0.0; ties keep the earlier epoch.
        self.best_score = float("-inf")
        self.patience_counter = 0
        self.global_step = 0
        self.history_epoch = 0
        # D-H: SHA-256 of the experiment config file's raw bytes. Set
        # externally (e.g. by train.py, after resolving --config) since the
        # Trainer itself has no notion of "which file was --config". None
        # for callers that do not opt into the experiment/machine split.
        self.experiment_sha256: str | None = None

        # TensorBoard
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        except ImportError:
            print("[Trainer] TensorBoard not available, skipping logging.")

    def fit(self) -> dict:
        """Run training.

        StudentModel: single-phase, unchanged.
        MasterModel, `schedule="end_to_end"` (fusion-redesign D-4, default):
            one phase, `freeze_stages=0` from epoch 0, discriminative LR
            (pretrained backbone stages at `backbone_lr_mult` times the
            neck/head rate).
        MasterModel, `schedule="two_phase"` (legacy, opt-in): Phase 1 frozen
            backbone, Phase 2 partial unfreeze — preserved for configs that
            explicitly request it (e.g. `training_mango.yaml`).

        Returns:
            Dict with final metrics and checkpoint paths.
        """
        model_label = "StudentModel" if self.config.model_type == "student" else "MasterModel"
        is_end_to_end_master = (
            self.config.model_type == "master" and self.config.schedule == "end_to_end"
        )

        print(f"\n{'='*60}")
        print(f"Training {model_label} on {self.device}")

        if self.config.model_type == "student":
            print(f"Single-phase: {self.config.epochs} epochs, lr={self.config.lr}")
        elif is_end_to_end_master:
            print(
                f"End-to-end: {self.config.epochs} epochs, lr={self.config.lr}, "
                f"backbone_lr_mult={self.config.backbone_lr_mult}"
            )
        else:
            print(f"Phase 1: {self.config.epochs_phase1} epochs (frozen backbone)")
            print(f"Phase 2: {self.config.epochs_phase2} epochs (unfreeze stages 3-4)")

        print(f"{'='*60}\n")

        if self.config.model_type == "student":
            # Single-phase student training
            self._train_phase(
                phase=1,
                epochs=self.config.epochs,
                lr=self.config.lr,
                freeze_stages=0,
            )
        elif is_end_to_end_master:
            # End-to-end MasterModel training (D-4): freeze_stages=0 from
            # epoch 0, discriminative LR via a two-group optimizer.
            self._train_phase(
                phase=1,
                epochs=self.config.epochs,
                lr=self.config.lr,
                freeze_stages=0,
                discriminative=True,
            )
        else:
            # Legacy two-phase master training (opt-in via schedule="two_phase")
            if self.config.epochs_phase1 > 0:
                self._train_phase(
                    phase=1,
                    epochs=self.config.epochs_phase1,
                    lr=self.config.lr_phase1,
                    freeze_stages=4,
                )

            if self.config.epochs_phase2 > 0:
                self._train_phase(
                    phase=2,
                    epochs=self.config.epochs_phase2,
                    lr=self.config.lr_phase2,
                    freeze_stages=2,
                    unfreeze_stages=[2, 3],
                )

        # Generate training curves after training completes
        generate_training_curves(self.loss_history, self.output_dir)

        if self.writer:
            self.writer.close()

        return {
            "best_map50": self.best_map50,
            "loss_history": {
                "epoch": self.loss_history.epoch,
                "phase": self.loss_history.phase,
                "cls_loss": self.loss_history.cls_loss,
                "box_loss": self.loss_history.box_loss,
                "total_loss": self.loss_history.total_loss,
                "map50": self.loss_history.map50,
                "map_50_95": self.loss_history.map_50_95,
                "extra_losses": self.loss_history.extra_losses,
            },
            "checkpoint": str(self.output_dir / "best_model.pt"),
        }

    def _train_phase(
        self,
        phase: int,
        epochs: int,
        lr: float,
        freeze_stages: int = 0,
        unfreeze_stages: list[int] | None = None,
        discriminative: bool = False,
    ):
        """Run training for one phase.

        Args:
            phase: Phase number (1 or 2).
            epochs: Number of epochs.
            lr: Learning rate for this phase (base rate — see
                `discriminative`).
            freeze_stages: Number of backbone stages to freeze.
            unfreeze_stages: Specific stages to unfreeze (legacy two-phase
                schedule's Phase 2).
            discriminative: If True, build a two-group AdamW (D-4/D-F):
                pretrained backbone-stage parameters at
                `config.backbone_lr_mult * lr`; stem, neck and head at `lr`.
                Used by the end-to-end MasterModel schedule. Ignored (single
                group at `lr`) otherwise.
        """
        print(f"\n{'─'*50}")
        print(f"Phase {phase}: {epochs} epochs | LR={lr}")
        print(f"{'─'*50}")

        # Freeze/unfreeze backbone. `unfreeze_stem=True` (renamed from
        # `unfreeze_rgb_stem`, fusion-redesign D-4 — there is only one stem
        # now) is the E6/Q10 production fix: `freeze_backbone()`
        # unconditionally freezes the stem whenever freeze_stages > 0, so
        # without this flag a Phase 2 "unfreeze" call never actually lets it
        # adapt (design.md Out of Scope; proposal.md Round 3 Q10).
        self.model.freeze_backbone(freeze_stages=freeze_stages)
        if unfreeze_stages is not None:
            self.model.unfreeze_backbone_stages(unfreeze_stages, unfreeze_stem=True)

        # Count trainable params
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")

        # Optimizer
        if discriminative:
            param_groups = self._discriminative_param_groups(lr)
        else:
            param_groups = [
                {"params": [p for p in self.model.parameters() if p.requires_grad], "lr": lr}
            ]
        param_groups += self._auxiliary_param_groups(lr)
        optimizer = AdamW(
            param_groups,
            lr=lr,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler: warmup + cosine annealing
        warmup_epochs = self.config.warmup_epochs
        total_epochs = epochs

        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, total_epochs - warmup_epochs), T_mult=2
        )

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train
            train_metrics = self._train_epoch(optimizer, epoch, phase)

            # Validate
            val_metrics = self._validate(epoch, phase)

            elapsed = time.time() - t0

            # Update scheduler
            if epoch <= warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

            # Log
            map50 = val_metrics.get("map50", 0.0)
            map_50_95 = val_metrics.get("map_50_95", 0.0)
            per_class_ap_50 = val_metrics.get("per_class_ap_50", {})
            per_class_ap_50_95 = val_metrics.get("per_class_ap_50_95", {})
            per_class_precision = val_metrics.get("per_class_precision", {})
            per_class_recall = val_metrics.get("per_class_recall", {})
            per_class_f1 = val_metrics.get("per_class_f1", {})
            per_class_counts = val_metrics.get("per_class_counts", {})
            ap50_text = ", ".join(
                f"AP50[c{cls_id}]={ap:.4f}"
                for cls_id, ap in sorted(per_class_ap_50.items())
            )
            extra_loss_text = "".join(
                f", {name}={value:.4f}"
                for name, value in sorted(train_metrics.items())
                if name.endswith("_loss") and name not in {"cls_loss", "box_loss", "total_loss"}
            )
            print(
                f"  [P{phase}] Epoch {epoch:02d}/{epochs} | "
                f"loss={train_metrics['total_loss']:.4f} "
                f"(cls={train_metrics['cls_loss']:.4f}, box={train_metrics['box_loss']:.4f}{extra_loss_text}) | "
                f"mAP@0.5={map50:.4f} | mAP@0.5:0.95={map_50_95:.4f} | "
                f"{ap50_text} | {elapsed:.1f}s"
            )

            # Record epoch history for CSV + plots.
            self.history_epoch += 1
            self.loss_history.update(
                cls=train_metrics["cls_loss"],
                box=train_metrics["box_loss"],
                total=train_metrics["total_loss"],
                epoch=self.history_epoch,
                phase=phase,
                map50=map50,
                map_50_95=map_50_95,
                per_class_ap_50=per_class_ap_50,
                per_class_ap_50_95=per_class_ap_50_95,
                per_class_precision=per_class_precision,
                per_class_recall=per_class_recall,
                per_class_f1=per_class_f1,
                per_class_counts=per_class_counts,
                extra_losses={
                    **{
                        name: value
                        for name, value in train_metrics.items()
                        if name.endswith("_loss") and name not in {"cls_loss", "box_loss", "total_loss"}
                    },
                    # D-G counters: always recorded, regardless of the
                    # "_loss" naming filter above, so a clean run is
                    # distinguishable from a run where nothing trained.
                    "oom_skipped": train_metrics["oom_skipped"],
                    "nan_skipped": train_metrics["nan_skipped"],
                    "steps_taken": train_metrics["steps_taken"],
                    # D-F per-module gradient norms (generalised from the
                    # previous single-module rgb_stem_grad_norm).
                    **{
                        name: value
                        for name, value in train_metrics.items()
                        if name.startswith("grad_norm_")
                    },
                },
            )

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar(f"Phase{phase}/train/total_loss", train_metrics["total_loss"], epoch)
                self.writer.add_scalar(f"Phase{phase}/train/cls_loss", train_metrics["cls_loss"], epoch)
                self.writer.add_scalar(f"Phase{phase}/train/box_loss", train_metrics["box_loss"], epoch)
                self.writer.add_scalar(f"Phase{phase}/val/map50", map50, epoch)
                self.writer.add_scalar(f"Phase{phase}/val/map_50_95", val_metrics.get("map_50_95", 0.0), epoch)
                self.writer.add_scalar(f"Phase{phase}/lr", optimizer.param_groups[0]["lr"], epoch)

                # D-G counters (see _train_epoch docstring).
                self.writer.add_scalar(f"Phase{phase}/train/oom_skipped", train_metrics["oom_skipped"], epoch)
                self.writer.add_scalar(f"Phase{phase}/train/nan_skipped", train_metrics["nan_skipped"], epoch)
                self.writer.add_scalar(f"Phase{phase}/train/steps_taken", train_metrics["steps_taken"], epoch)

                # Per-class AP
                for cls_id, ap in val_metrics.get("per_class_ap_50", {}).items():
                    self.writer.add_scalar(f"Phase{phase}/val/ap50_class{cls_id}", ap, epoch)

                # D-F instrumentation: verify the schedule actually reaches
                # every module, independent of its effect on mAP — the same
                # bit-exact-LayerNorm failure mode (D2/D3) that motivated
                # H-F's init-distance check, made visible during training
                # rather than only discoverable after it.
                if self.config.model_type == "master":
                    stem_trainable = any(
                        p.requires_grad
                        for stem in self.model.backbone.stem_modules.values()
                        for p in stem.parameters()
                    )
                    self.writer.add_scalar(
                        f"Phase{phase}/backbone_stem_requires_grad", float(stem_trainable), epoch
                    )
                    for name, value in train_metrics.items():
                        if name.startswith("grad_norm_"):
                            self.writer.add_scalar(f"Phase{phase}/{name}", value, epoch)

            # Checkpoint + early stopping on `config.selection_metric`.
            # `best_map50` stays the mAP@0.5 of the saved checkpoint, whatever
            # metric selected it, since run artifacts are named after it.
            score = self._selection_score(val_metrics)
            if score > self.best_score:
                self.best_score = score
                self.best_map50 = map50
                self.patience_counter = 0
                self._save_checkpoint(epoch, phase, val_metrics, "best_model.pt", train_metrics=train_metrics)
                print(f"    ✓ New best {self.config.selection_metric}: {score:.4f}")
            else:
                self.patience_counter += 1

            # Periodic checkpoint
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(
                    epoch, phase, val_metrics, f"checkpoint_epoch{epoch}.pt", train_metrics=train_metrics
                )

            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"  Early stopping at epoch {epoch} (patience={self.config.patience})")
                break

    def _auxiliary_param_groups(self, lr: float) -> list[dict]:
        """Extra optimizer parameter groups for trainable modules that are
        NOT part of `self.model` (and therefore not of its state_dict).

        Empty for the base trainer. `KDTrainer` overrides it to add its
        training-only distillation adapters, which must be optimized
        alongside the student but must never be saved into (or required
        by) the student's `model_state_dict`.
        """
        return []

    def _auxiliary_checkpoint_state(self) -> dict:
        """Extra top-level checkpoint entries for modules outside
        `self.model` (see `_auxiliary_param_groups`). Empty by default."""
        return {}

    def _selection_score(self, val_metrics: dict) -> float:
        """Value of `config.selection_metric` in one epoch's val metrics."""
        if self.config.selection_metric == "damage_ap50":
            return val_metrics.get("per_class_ap_50", {}).get(DAMAGE_CLASS_ID, 0.0)
        return val_metrics.get("map50", 0.0)

    def _discriminative_param_groups(self, lr: float) -> list[dict]:
        """Two-group AdamW parameter groups for end-to-end MasterModel
        training (fusion-redesign D-4/D-F).

        Pretrained backbone-stage parameters (`backbone.stages.*`, aliasing
        `backbone.shared_stages.*` under `fusion_mode="cross_attention"`)
        train at `backbone_lr_mult * lr`. Everything else that is new or
        must adapt — the stem(s), the cross-attention fusion module when
        present, the neck, and the head — trains at the full `lr`. The
        stem is deliberately in the "new" group, not the pretrained one: it
        is 4-channel and out of ImageNet distribution by construction (the
        same D-1 argument applied consistently), even though 3 of its 4
        channels start from pretrained weights.

        The second group is defined by subtraction (every trainable
        parameter that is not a backbone stage) rather than by naming
        modules: with two fusion modes, an explicit module list is one
        `fusion_mode` away from silently dropping a whole submodule out of
        the optimizer. Parameter order is unchanged for the early path —
        `model.parameters()` yields stem, stages, neck, head in registration
        order, so removing the stages leaves exactly the previous
        stem/neck/head sequence.

        Only `requires_grad=True` parameters are included, so this is safe
        to call under `freeze_stages=0` (every parameter trainable, the
        end-to-end default) or any other freeze configuration.
        """
        stage_param_ids = {id(p) for p in self.model.backbone.stages.parameters()}
        backbone_stage_params = [
            p for p in self.model.backbone.stages.parameters() if p.requires_grad
        ]
        new_params = [
            p
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in stage_param_ids
        ]
        return [
            {"params": backbone_stage_params, "lr": lr * self.config.backbone_lr_mult},
            {"params": new_params, "lr": lr},
        ]

    def _module_grad_norms(self) -> dict[str, float]:
        """L2 gradient norm per trainable module (fusion-redesign D-F).

        Generalises the previous `_rgb_stem_grad_norm` (single-module,
        dual-stream-specific instrumentation) into per-module coverage of
        every backbone stage, the stem, the neck, and each head level — so
        an end-to-end run that silently fails to train one of them (D2's
        failure mode: a schedule that looks like it trains a module but
        never actually reaches it) is visible in TensorBoard/`extra_losses`
        rather than discovered 68 epochs later by a bit-exact LayerNorm.

        Returns 0.0 for a module with no gradients yet (frozen, or before
        the first backward pass) rather than raising.
        """
        def _norm(module: nn.Module) -> float:
            total_sq = 0.0
            for param in module.parameters():
                if param.grad is not None:
                    total_sq += param.grad.detach().float().norm(2).item() ** 2
            return total_sq ** 0.5

        norms = {
            f"backbone_{name}": _norm(stem)
            for name, stem in self.model.backbone.stem_modules.items()
        }
        for i, stage in enumerate(self.model.backbone.stages):
            norms[f"backbone_stage{i}"] = _norm(stage)
        # Present only under fusion_mode="cross_attention" — the early path
        # has no `fusion` submodule, and a constant 0.0 would read as "this
        # module never gets gradients" rather than "it does not exist".
        fusion = getattr(self.model, "fusion", None)
        if fusion is not None:
            norms["fusion"] = _norm(fusion)
        norms["neck"] = _norm(self.model.neck)
        for i, head in enumerate(self.model.head.heads):
            norms[f"head_level{i}"] = _norm(head)
        return norms

    def _train_epoch(self, optimizer: AdamW, epoch: int, phase: int) -> dict:
        """Run one training epoch with gradient accumulation.

        D-G (fusion-redesign): OOM and NaN/Inf batch faults are counted and,
        past a bounded tolerance, fatal. A run in which every batch fails
        used to complete "successfully" with `total_loss=0.0` and a saved
        checkpoint from a model that never received a gradient — see
        `openspec/changes/fusion-redesign/design.md` D-G. Silent skipping is
        no longer an option:

        - `oom_skipped`: CUDA OOM is tolerated on epoch 1 only (allocator
          warm-up). From epoch 2 onward, any OOM skip raises immediately.
        - `nan_skipped`: always counted; never itself the trigger for a raise
          (see `steps_taken` below), but recorded so a future bf16 rung
          cannot convert a REFUTE into a false CONFIRM by silently discarding
          the batches that produced non-finite loss.
        - `steps_taken`: **optimizer** steps actually executed (i.e. effective
          steps, after accumulation — see D-H below). Zero in any epoch is
          always fatal, regardless of which guard caused it.

        D-H (fusion-redesign): gradient accumulation keeps `effective_batch`
        constant while `batch_size` (a memory knob) varies across hardware.
        Each micro-batch's loss is scaled by `1/grad_accum_steps` before
        `backward()`; `backward()` runs every micro-batch, but
        `clip_grad_norm_` + `optimizer.step()` + `zero_grad()` run only once
        per `grad_accum_steps` micro-batches, with any leftover
        micro-batches flushed as a final (smaller) step at epoch end.
        `grad_clip` is therefore applied once per **effective** step, never
        per micro-batch — clipping a partial gradient would change the
        optimisation accumulation exists to hold constant. With
        `grad_accum_steps == 1` (no accumulation), this degrades exactly to
        the pre-accumulation per-micro-batch step behaviour.

        Returns:
            Dict with avg cls_loss, box_loss, total_loss (averaged over
            successful *micro-batches*), and the
            oom_skipped/nan_skipped/steps_taken (optimizer steps) counters.
        """
        self.model.train()
        accum = self.grad_accum_steps
        total_cls = 0.0
        total_box = 0.0
        total_loss = 0.0
        micro_batches = 0   # successful forward+backward passes
        steps_taken = 0     # optimizer.step() calls (effective steps)
        oom_skipped = 0
        nan_skipped = 0
        module_grad_norms: dict[str, list[float]] = {}
        micro_step = 0      # micro-batches accumulated since the last flush
        optimizer.zero_grad(set_to_none=True)

        def _flush_step() -> None:
            nonlocal steps_taken
            if self.config.precision == "fp16":
                self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            if self.config.precision == "fp16":
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps_taken += 1

        for batch in self.train_loader:
            rgb = batch["rgb"].to(self.device)
            nir = batch["nir"].to(self.device)
            bboxes = [b.to(self.device) for b in batch["bboxes"]]
            labels = [l.to(self.device) for l in batch["labels"]]

            try:
                with autocast_ctx(self.device.type, self.config.precision):
                    if self.config.model_type == "student":
                        output = self.model(rgb)
                    else:
                        output = self.model(rgb, nir)
                    predictions = output["preds"]

                    targets = {"bboxes": bboxes, "labels": labels}
                    loss, loss_dict = self.criterion(predictions, targets)

                # NaN / Inf guard: skip batch if loss explodes. Always
                # counted — see D-G docstring above for why this must not
                # also be silent. Checked on the unscaled loss so the
                # reported/counted value is independent of grad_accum_steps.
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_skipped += 1
                    print(f"  [NaN] Skipping batch (loss={loss.item():.2f}), "
                          f"nan_skipped={nan_skipped}")
                    continue

                scaled_loss = loss / accum
                if self.config.precision == "fp16":
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # D-F instrumentation: capture per-module gradient norms
                # right after backward(), before clipping/step touch them —
                # generalises the previous single-module (RGB-stem-only) E6
                # instrumentation to every backbone stage, the stem, the
                # neck, and each head level, so an end-to-end run that
                # silently fails to reach one of them is visible rather than
                # discovered by a bit-exact LayerNorm after the fact (D2/H-F).
                # No-op cost for the student model.
                if self.config.model_type == "master":
                    for name, value in self._module_grad_norms().items():
                        module_grad_norms.setdefault(name, []).append(value)

                micro_step += 1
                if micro_step % accum == 0:
                    _flush_step()
                    micro_step = 0

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_skipped += 1
                    torch.cuda.empty_cache()
                    if epoch >= 2:
                        raise RuntimeError(
                            f"[OOM] Batch skipped on epoch {epoch} (phase {phase}); "
                            "OOM tolerance is limited to epoch 1 (allocator "
                            "warm-up). A run that keeps OOM-ing past epoch 1 is "
                            "not a valid training run — reduce batch_size or "
                            "increase grad_accum_steps instead of retrying. "
                            f"oom_skipped={oom_skipped}, nan_skipped={nan_skipped}, "
                            f"steps_taken={steps_taken}."
                        ) from e
                    print(
                        f"  [OOM] Skipping batch (epoch 1 warm-up tolerance), "
                        f"clearing CUDA cache. oom_skipped={oom_skipped}"
                    )
                    continue
                raise

            total_cls += loss_dict["cls_loss"]
            total_box += loss_dict["box_loss"]
            total_loss += loss.item()
            micro_batches += 1
            self.global_step += 1

        # Flush a partial accumulation window at epoch end so the last
        # (possibly incomplete) group of micro-batches is not silently
        # dropped from the optimizer update.
        if micro_step != 0:
            _flush_step()

        if steps_taken == 0:
            raise RuntimeError(
                f"Training epoch {epoch} (phase {phase}) took zero optimizer "
                f"steps (oom_skipped={oom_skipped}, nan_skipped={nan_skipped}). "
                "A run with zero steps produced no gradient update and must "
                "not be reported as a completed epoch — see "
                "openspec/changes/fusion-redesign/design.md D-G."
            )

        result = {
            "cls_loss": total_cls / micro_batches,
            "box_loss": total_box / micro_batches,
            "total_loss": total_loss / micro_batches,
            "oom_skipped": float(oom_skipped),
            "nan_skipped": float(nan_skipped),
            "steps_taken": float(steps_taken),
        }
        for name, values in module_grad_norms.items():
            result[f"grad_norm_{name}"] = sum(values) / len(values)
        return result

    @torch.no_grad()
    def _validate(self, epoch: int, phase: int) -> dict:
        """Run validation and compute mAP metrics.

        Thin wrapper over `evaluate()` — `epoch`/`phase` are accepted for the
        training-loop call site's logging context but are not needed by
        evaluation itself (D10).

        Returns:
            Dict with map50, map_50_95, per_class_ap_50, per_class_ap_50_95.
        """
        return self.evaluate()

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Run evaluation over `val_loader` and compute mAP metrics.

        Public, eval-only entrypoint with no dependency on a training loop or
        `train_loader` (D10). Used by both the training loop's periodic
        validation and `scripts/evaluate_checkpoint.py`.

        Returns:
            Dict with map50, map_50_95, per_class_ap_50, per_class_ap_50_95,
            per_class precision/recall/f1, and per_class_counts.
        """
        self.model.eval()

        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_gt_boxes = []
        all_gt_labels = []

        for batch in self.val_loader:
            rgb = batch["rgb"].to(self.device)
            nir = batch["nir"].to(self.device)

            with autocast_ctx(self.device.type, self.config.precision):
                if self.config.model_type == "student":
                    output = self.model(rgb)
                else:
                    output = self.model(rgb, nir)

            # Decode predictions for mAP computation
            preds = output["preds"]  # list of (B, nc+4, H, W)
            cls_preds = output["cls_preds"]  # list of (B, nc, H, W)

            B = rgb.shape[0]
            for b in range(B):
                # AP needs the full ranked list, not just the operating point.
                pred_boxes_b, pred_scores_b, pred_labels_b = self._decode_predictions(
                    preds, cls_preds, b, conf_threshold=self.config.eval_conf_threshold
                )
                all_pred_boxes.append(pred_boxes_b)
                all_pred_scores.append(pred_scores_b)
                all_pred_labels.append(pred_labels_b)

                # GT
                gt_bboxes = batch["bboxes"][b]
                gt_labels_batch = batch["labels"][b]
                all_gt_boxes.append(gt_bboxes)
                all_gt_labels.append(gt_labels_batch)

        # Compute mAP
        metrics = compute_map(
            pred_boxes=all_pred_boxes,
            pred_scores=all_pred_scores,
            pred_labels=all_pred_labels,
            gt_boxes=all_gt_boxes,
            gt_labels=all_gt_labels,
            num_classes=self.config.num_classes,
            score_threshold=self.config.conf_threshold,
        )

        return metrics

    @staticmethod
    def _decode_predictions_static(
        preds: list[torch.Tensor],
        cls_preds: list[torch.Tensor],
        batch_idx: int,
        config: TrainingConfig,
        conf_threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Config-driven decode, usable without a full `Trainer` instance.

        `conf_threshold` defaults to `config.conf_threshold` (the operating
        point); `evaluate()` passes `config.eval_conf_threshold` instead so AP
        sees the full ranked prediction list.

        Thin wrapper over `decode.decode_detections` (D3) — the single
        source of truth for decode semantics shared with
        `scripts/visualize_damage_predictions.py`. `strides` is resolved
        from `config` (fusion-redesign D-D), not hardcoded, so a 4-level
        checkpoint decodes with its own 4 strides instead of a stale
        `(8, 16, 32)` that would misalign predictions and anchor grids.
        """
        boxes, scores, labels = decode_detections(
            preds,
            cls_preds,
            batch_idx=batch_idx,
            num_classes=config.num_classes,
            image_size=config.image_size,
            strides=tuple(resolve_active_strides(config)),
            conf_threshold=config.conf_threshold if conf_threshold is None else conf_threshold,
            nms_iou_threshold=config.nms_iou_threshold,
            nms_enabled=config.nms_enabled,
            per_class_candidates=config.decode_per_class,
            max_detections=config.max_detections,
            normalize=True,
        )
        return boxes.cpu(), scores.cpu(), labels.cpu()

    def _decode_predictions(
        self,
        preds: list[torch.Tensor],
        cls_preds: list[torch.Tensor],
        batch_idx: int,
        conf_threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode model predictions for a single image in the batch.

        Returns:
            pred_boxes: (P, 4) cxcywh normalized.
            pred_scores: (P,) confidence scores.
            pred_labels: (P,) class IDs.
        """
        return self._decode_predictions_static(
            preds, cls_preds, batch_idx, self.config, conf_threshold=conf_threshold
        )

    def _save_checkpoint(
        self,
        epoch: int,
        phase: int,
        metrics: dict,
        filename: str,
        train_metrics: dict | None = None,
    ):
        """Save model checkpoint.

        `train_metrics`, when provided, carries the D-G fault counters
        (`oom_skipped`, `nan_skipped`, `steps_taken`) from the epoch that
        produced this checkpoint, so a checkpoint can be audited after the
        fact without re-reading the training log. `effective_batch`,
        `precision`, and `device` are already part of `self.config.__dict__`
        below; `experiment_sha256` (D-H) is recorded alongside separately
        since it is not itself a config field — two runs are comparable
        only if `experiment_sha256` AND `effective_batch` both match.

        `arch_version` (fusion-redesign D-C) tags the checkpoint's state
        dict schema. Every key changes under the redesign, so a v1
        (dual-stream fusion) checkpoint is unloadable into the v2 model;
        `arch_version` converts that into one actionable error message
        (`scripts/evaluate_checkpoint.py`,
        `scripts/visualize_damage_predictions.py`) instead of a 200-line
        missing/unexpected-key dump.
        """
        path = self.output_dir / filename
        checkpoint = {
            "epoch": epoch,
            "phase": phase,
            "arch_version": arch_version_for_mode(self.config.fusion_mode),
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "best_map50": self.best_map50,
            "selection_metric": self.config.selection_metric,
            "best_score": self.best_score,
            "config": self.config.__dict__,
            "experiment_sha256": self.experiment_sha256,
            # NIR registration provenance: checkpoints trained before the
            # 2026-10 fix lack this key and used unregistered NIR.
            "nir_registration": {
                "homography_path": self.config.nir_homography_path,
                "homography_sha256": homography_sha256(self.config.nir_homography_path),
            },
        }
        checkpoint.update(self._auxiliary_checkpoint_state())
        if train_metrics is not None:
            checkpoint["oom_skipped"] = train_metrics.get("oom_skipped")
            checkpoint["nan_skipped"] = train_metrics.get("nan_skipped")
            checkpoint["steps_taken"] = train_metrics.get("steps_taken")
        torch.save(checkpoint, path)
