# Proposal: Fix KD gradient flow, budget parity, and per-phase early stopping

Change id: `fix-kd-blockers` · Store: hybrid · Supersedes `fix-detection-eval-and-kd-blockers`.

## Intent

Three defects make the "distilled" milestone unmeasurable. They are independent in code but
share one re-validation run, because each one invalidates the others' baseline.

1. **The KD projections never train.** `KDTrainer._train_epoch` ran all four `kd_proj_*`
   calls inside the teacher's `torch.no_grad()` block, and `KDLoss.forward` detached the
   teacher-side tensor again before the MSE. The `kd_proj_*` modules are student-owned
   trainable parameters whose only gradient source is the KD loss, so both severings left
   them frozen at random init for the entire run. "Distilled" did not mean distillation.
   Either fix alone is insufficient — the `no_grad` scope and the `detach()` each sever the
   same path independently, so they ship together.

2. **The distilled and baseline runs had different budgets.** `configs/kd_training.yaml`
   ran 80 epochs / patience 30 / warmup 3 / workers 4 against
   `configs/training_student.yaml` at 50 / 15 / 5 / 2. Any measured difference between the
   two runs confounds distillation with training length.

3. **Early stopping leaked across phases.** `Trainer.patience_counter` was initialized once
   in `__init__` and never reset on entry to `_train_phase`. A phase entered with the
   counter already at or near `config.patience` early-stops after a single epoch regardless
   of its own configuration. `loop.py` now calls `_train_phase` from four sites (the master
   two-phase schedule and the end-to-end schedule), so every multi-phase run is affected.
   This is the mechanism behind the master phase 2 that ran exactly one epoch.

## Relationship to prior work

The original change `fix-detection-eval-and-kd-blockers` carried a fourth slice: NMS in
`Trainer._decode_predictions`. That slice is **dropped as already delivered**. `damage-map-audit`
(PR #11) shipped a better implementation — a shared `src/training/decode.py` module with
per-class `torchvision.ops.batched_nms` and configurable `nms_enabled` / `nms_iou_threshold`
in `TrainingConfig` — so re-applying the original slice would duplicate and conflict with it.

The original branch was cut from a `main` that is now 21 commits stale; this change re-applies
its three surviving slices against current `main`.

## Scope

### In Scope

1. **`src/training/kd_trainer.py`** — narrow the `torch.no_grad()` block to the teacher
   forward and the stride selection only; move the four `kd_proj_*` calls out of it. The
   projections' inputs are already detached by that block, so no teacher gradient is created.
2. **`src/training/kd_loss.py`** — drop `t_feat.detach()` in `KDLoss.forward`.
3. **`configs/kd_training.yaml`** — set `epochs: 50`, `patience: 15`, `warmup_epochs: 5`,
   `num_workers: 2` to match `configs/training_student.yaml`.
   **`src/training/kd_config.py`** — align the `KDConfig` dataclass defaults for
   `epochs` / `patience` so a config built without a YAML file is comparable too.
4. **`src/training/loop.py`** — `self.patience_counter = 0` at the top of `_train_phase`.
   `best_map50` is deliberately NOT reset: the best checkpoint is a run-level artifact, and
   resetting it would let a worse later phase overwrite a better earlier one.

### Out of Scope

- Broken warmup (`LambdaLR` overwritten by `CosineAnnealingWarmRestarts`) — deferred to its
  own change by user decision.
- `kd_proj_*` keys polluting the student `state_dict` — pre-existing, unchanged here, no
  current consumer.
- Staging paths recorded into `stage_summary.json` / `run_summary.json` before
  `publish_atomic_run` renames the directory — cosmetic, no metric impact.
- Everything already owned by the open `damage-map-audit` and `fusion-redesign` changes.

## Validation

The three fixes share one re-validation run: a full three-stage pipeline execution whose
distilled stage is now comparable to its baseline. Until that run exists, no claim about
distillation's effect on mAP is supportable — including the previously recorded result where
the RGB-only student beat the distilled student.
