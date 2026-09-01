# Tasks: Fix KD gradient flow, budget parity, and per-phase early stopping

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~230 (source ~60, tests ~170) |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Delivery strategy | single PR |

## Phase 1 — KD gradient flow (slices 1+2, ship together)

- [x] 1.1 Narrow the `torch.no_grad()` block in `KDTrainer._train_epoch` to the teacher
      forward and the `select_by_strides` calls; move `kd_proj_backbone`, `kd_proj_fpn`,
      `kd_proj_head_cls`, and `kd_proj_head_reg` out of it.
      (`src/training/kd_trainer.py:184`, projections at `:203`)
- [x] 1.2 Remove `t_feat.detach()` from `KDLoss.forward`. (`src/training/kd_loss.py:75`)
- [x] 1.3 Add `tests/training/test_kd_loss.py` — model-free gradient-flow tests over a
      stand-in `nn.Conv2d` projection.
- [x] 1.4 Add `tests/training/test_kd_trainer.py` — one-step integration test asserting every
      `kd_proj_*` parameter receives a gradient and changes value, and the teacher does not.
- [x] 1.5 Confirm both tests are RED against the unfixed code, independently: reverting only
      the `detach()` and reverting only the `no_grad` scope each reproduce a failure.

## Phase 2 — Training budget parity (slice 3)

- [x] 2.1 Set `epochs: 50`, `patience: 15`, `warmup_epochs: 5`, `num_workers: 2` in
      `configs/kd_training.yaml`, with a comment binding them to
      `configs/training_student.yaml`.
- [x] 2.2 Align the `KDConfig` dataclass defaults (`epochs`, `patience`) to the same values.
      (`src/training/kd_config.py:45-46`)
- [x] 2.3 Add `test_kd_config_training_budget_matches_student_baseline` to
      `tests/test_training_loop.py`.

## Phase 3 — Per-phase early stopping (slice 4)

- [x] 3.1 Add `self.patience_counter = 0` at the top of `Trainer._train_phase`, with a comment
      recording why `best_map50` is deliberately not reset. (`src/training/loop.py:269`)
- [x] 3.2 Add `TestPatiencePhaseReset` to `tests/test_training_loop.py`; confirm it is RED
      without the reset.

## Phase 3b — Machine-profile plumbing (added 2026-08-31)

Found while sizing the 4.2 re-run for the local RTX 3080: a machine profile could not
reach a pipeline run at all. `--machine` existed only on `src/training/train.py`, the
pipeline runner never forwarded it, and `src/training/kd_train.py` had no such flag — so
the only way to change the physical batch size was to edit the experiment YAML, which is
exactly what D-H exists to prevent.

- [x] 3b.1 Add `--machine` to `src/training/kd_train.py`, applied after `--config` and
      before `--override` (mirrors `src/training/train.py`).
- [x] 3b.2 Add `--machine` to `scripts/run_final_training_pipeline.py` and forward it to
      all three stages through `run_training_stage`.
- [x] 3b.3 Add `tests/training/test_machine_profile_plumbing.py`; confirmed RED without
      3b.1/3b.2 (4 failures), green with them.

## Phase 4 — Verification

- [x] 4.1 Full suite green: `./myLinuxVenv/bin/python -m pytest tests/ -q` → 282 passed.
- [ ] 4.2 Re-run the three-stage final pipeline so the distilled stage is measured against a
      baseline with a matching budget and with the projections actually training. Until this
      lands, no distillation claim (including "the RGB-only student beat the distilled one")
      is supportable.

      Sized against measured peak VRAM on the local RTX 3080 (10 GB total, ~2.0 GB held by
      the desktop). MasterModel, fp32, 640, `head_strides: [8, 16, 32]`: 2.05 GB @ batch 2,
      3.95 GB @ batch 4, 7.75 GB @ batch 8 (does not fit). KD stage (teacher in `no_grad`
      plus the now-trainable `kd_proj_*`): 3.29 GB @ batch 8. So the run uses
      `--machine configs/machines/rtx3080.yaml` (batch_size 4 → `grad_accum_steps` 2 against
      the unchanged `effective_batch: 8`).

      First attempt (`20260831T223558592803Z`) reached the KD stage and died there — see 4.2a.
      Its master stage scored best mAP@0.5 = 0.1673 against 0.4438 in the pre-merge run
      `20260826T224117154204Z`, which looked like a quality regression. It was not: the run
      hit an isolated 0.1673 at epoch 7, failed to beat it for `patience: 20` epochs, and was
      early-stopped at 27. The batch-2 run `20260831T224505981163Z` reached its best of 0.4806
      at **epoch 62** — 35 epochs past where the other was killed. Re-running batch 4 with
      `num_workers: 2` completed all 80 phase-1 epochs and scored **0.5078**, the best master
      result to date. Batch 4 is sound; the run had simply been truncated. Both earlier
      hypotheses were wrong and are recorded so they are not retried: the frozen single stem
      (both runs share it) and the gradient-accumulation path (batch 4 trains fine).

      Blocked on 4.2b: the budget those runs received was not in fact equal.

- [x] 4.2a Fix `KDTrainer._load_teacher`: rebuild the teacher from the architecture recorded
      in its own checkpoint (`head_strides` / `backbone_variant` / `in_channels`), falling
      back to the KD config only for a checkpoint written before `config` was stored. The
      master trains under `configs/training_mango.yaml` (`head_strides: [8, 16, 32]`) while
      `configs/kd_training.yaml` never sets the key, so `KDConfig`'s post-fusion-redesign
      4-level default made `load_state_dict` fail on the missing `head.heads.3.*` weights.
      Regression test: `tests/training/test_kd_teacher_arch.py` (RED reproduces the exact
      production traceback).
- [x] 4.2b Make a run's budget independent of the machine, so 4.2 measures what it claims to.
      D-H declares two runs comparable when `experiment_sha256` and `effective_batch` match,
      but three defects let `batch_size` — a machine knob — change the experiment anyway:

      1. Both entry points built the train loader with `drop_last=True`. On the 50-image
         training set that means batch 2 sees all 50 images in `ceil(25/4)` = 7 optimizer
         steps while batch 4 sees only 48 in `ceil(12/2)` = 6. Because early stopping counts
         epochs, phase 1 delivered 560 updates at batch 2 against 162 at batch 4 — a 3.5x
         budget gap under an identical `effective_batch`. Restored `build_dataloader`'s
         `drop_last=False` default; the accumulation loop already flushes the leftover
         micro-batch as a final smaller step, so no gradient is discarded. Verified in real
         runs: `steps_taken` is now 7 per epoch at both batch 2 and batch 4.
      2. `scripts/run_final_training_pipeline.py` called `finalize_stage` without the four
         D-H fields, so every pipeline `run_summary.json` recorded `null` for
         `experiment_sha256`, `effective_batch`, `precision`, and `device` — the comparability
         rule had no data to run on. `finalize_stage` now reads them from the checkpoint it
         already opens (ground truth from the run, not a second derivation that can drift),
         with an explicit argument still winning for `train.py --versioned-run`.
      3. `LossHistory.update` seeded its key order by iterating `set(a) | set(b)`. String
         hashing is randomised per process, so `metrics_history.csv` came out with a different
         column order on every run — `steps_taken` is column 16 in one published run and
         column 6 in another. Any positional cross-run comparison silently read the wrong
         column. Now ordered first-seen via `dict.fromkeys`.

      Tests: `tests/training/test_budget_comparability.py`,
      `tests/training/test_metrics_csv_determinism.py`,
      `tests/training/test_stage_provenance.py`. All confirmed RED first.

      Note: this changes training behaviour, so runs published before it are not reproducible
      under the new code. That is the point — their budgets were the defect.

- [ ] 4.3 Open the PR against `main`.

## Follow-ups (not blockers of this change)

- Early stopping measures noise, not convergence. The validation split is 9 images, so
  mAP@0.5 swings hard between consecutive epochs (0.0100 → 0.2944 → 0.0617 → 0.1409 → 0.2622
  in `20260831T224505981163Z`). That run's longest gap between improvements was 12 epochs,
  clearing `patience: 20` by only 8 — the batch-4 attempt did not clear it and died at epoch
  27 with 53 epochs of useful training left. `patience` on this split is a coin flip.
- Class 1 (damage) remains unlearned: the 0.5078 run ends at `AP50[c0]=0.9889` against
  `AP50[c1]=0.0267`. Whole-model mAP is carried by one class.

## Dropped from the original change

- ~~NMS in `Trainer._decode_predictions`~~ — delivered by `damage-map-audit` (PR #11) as the
  shared `src/training/decode.py` module with per-class `batched_nms` and configurable
  `nms_enabled` / `nms_iou_threshold`. Re-applying it would duplicate and conflict.
