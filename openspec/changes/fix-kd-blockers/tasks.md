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

- [x] 4.2c Disable early stopping, so a stage cannot be truncated by validation noise before
      4.2 measures it. `patience <= 0` now disables the mechanism
      (`Trainer._should_stop_early`), and all three pipeline configs set `patience: 0`.

      Evidence. The validation split is 9 images, so one detection moving in or out shifts
      mAP@0.5 by roughly ten points and the per-epoch metric early stopping watches is mostly
      sampling noise. Across the two runs that trained to completion, the longest gap between
      *genuine* improvements was 12 and 13 epochs, against a shipped `patience` of 20 (master)
      and 15 (student, KD) — a margin of 7 and 2 epochs. It has already misfired once:
      `20260831T223558592803Z` spiked to 0.1673 at epoch 7, could not beat that bar, and was
      killed at epoch 27 with 53 epochs left, producing a number that read as a quality
      regression and was not one. The best run was still improving at epoch 76 of 80.

      The mechanism buys nothing here: a full 90-epoch run is 3.8 minutes at 2.5s/epoch, and
      `_save_checkpoint` keeps the best-mAP checkpoint however long training continues, so
      stopping early prevents no overfitting the best-checkpoint rule does not already handle.
      The student and KD configs matter most — 4.2 compares those two stages directly, so one
      stopping on noise while the other runs on would invalidate exactly that comparison.

      Verified in a real run: 30/30 phase-1 epochs at batch 4, zero early-stopping events,
      in the epoch range where the earlier attempt died.

      This changes `experiment_sha256` for all three configs. That is correct — it is a
      different experiment, and the comparability machinery now says so.

      Tests: `tests/training/test_early_stopping_budget.py` (confirmed RED first).
      `tests/test_training_loop.py::test_kd_config_training_budget_matches_student_baseline`
      hardcoded `patience == 15`; it now asserts the two budgets match each other, which is
      the contract its own docstring states.

- [x] 4.2d Give the distilled stage the same optimizer budget as the baseline. The first
      complete run (`20260901T200124175077Z`: maestro 0.5073, estudiante 0.5303, destilado
      0.5352) looked like the distilled stage had finally beaten the baseline. It had not been
      measured against one. `KDTrainer` overrode `_train_epoch` with its own copy of the loop,
      and that copy — as its docstring said — had no gradient accumulation: one optimizer step
      per batch. Under `batch_size: 4` the student took `ceil(13/2)` = 7 steps/epoch at
      effective batch 8; the distilled stage took 13 at effective batch 4, while `KDConfig`
      (inheriting `effective_batch` from `TrainingConfig`) recorded 8 in its checkpoint. Nearly
      twice the updates at half the batch, under an identical claimed budget — and the machine
      profile changed the KD experiment outright, since with no accumulation `batch_size` *is*
      the effective batch.

      Not fixed by copying accumulation into `KDTrainer`: the two loop bodies were identical
      except for the forward pass and the loss, and that duplication is how they drifted.
      `Trainer._train_epoch` now owns the loop and calls a `_forward_loss` hook returning the
      loss and its scalar components; `KDTrainer` overrides only the hook, and its
      `_train_epoch` is gone. `kd_train.py` now computes `experiment_sha256` and passes the
      four D-H fields to `finalize_stage` like `train.py` does — the distilled stage used to
      publish `experiment_sha256: null`. `KDConfig.patience` defaults to 0 to keep mirroring
      the student config.

      Tests: `tests/training/test_kd_budget_parity.py` (same batches → same optimizer steps
      for `Trainer` and `KDTrainer`; `KDTrainer` defines no `_train_epoch`; the rtx3080
      profile yields `grad_accum_steps == 2` on a real `KDTrainer` — the honest version of the
      assurance `test_machine_profile_plumbing` gave by checking only the field). Confirmed
      RED first. `test_kd_trainer_guards` pins `effective_batch=1` (those tests count one
      step per batch and are about OOM/NaN, not accumulation).
      `test_kd_proj_parameters_receive_gradients_and_update` read `param.grad` after the
      epoch, relying on the old loop never zeroing after a step; it now captures grads at
      `optimizer.step()` time, which is the property it actually tests.

- [x] 4.2e Record stage paths relative to the run root, so publishing cannot kill them. The
      pipeline stages a run under `.partial/<run_id>-<uuid>/` and publishes by renaming that
      directory; `finalize_stage` had already written `stage_summary.json` — and the pipeline
      its `run_summary.json` — with `output_dir`/`checkpoint` naming the staging root. After
      the rename every one was dead: `Path(result.checkpoint).exists()` was False for all
      three stages of `20260901T200124175077Z` and of `20260831T224505981163Z`.
      `train.py --versioned-run` was unaffected (it stages *inside* `run_dir`, which never
      moves). `StageResult.output_dir`/`checkpoint` are now run-relative
      (`maestro/best_model_<run_id>_mAP0p5073/best_model.pt`); `run_summary.json` keeps
      `run_dir` as the anchor; the three in-repo consumers join against the root they hold.
      By construction — not a rewrite step a future mover could forget.

      Tests: `tests/training/test_stage_paths_survive_publish.py`, including a full
      `pipeline.main()` with fake subprocesses that plant checkpoints, then following the
      published `run_summary.json` to files that exist. Confirmed RED first, reproducing the
      production message verbatim.

- [x] 4.2 (re-run) `20260901T234436331584Z`, with 4.2b–4.2e in place and
      `--machine configs/machines/rtx3080.yaml`. The first run in which the three stages are
      actually comparable — verified from the published artifact, not from the log:

      | stage | best mAP@0.5 | @ epoch | epochs | steps/epoch | AP50 c0 | AP50 c1 |
      |---|---|---|---|---|---|---|
      | maestro | 0.5087 | 86 | 90/90 | 7 | 0.9889 | 0.0285 |
      | estudiante | 0.5209 | 46 | 50/50 | 7 | 1.0000 | 0.0419 |
      | destilado | **0.5490** | 29 | 50/50 | **7** | 1.0000 | **0.0980** |

      All three: `effective_batch: 8`, fp32, cuda, `experiment_sha256` populated (the
      distilled stage's for the first time), zero early-stopping events, every `checkpoint`
      in `run_summary.json` resolving to a file that exists, and the student/distilled
      `metrics_history.csv` headers identical apart from `kd_loss`.

      What it supports: the distilled student no longer loses to the RGB-only baseline
      (0.5490 vs 0.5209, +2.8 points), and it reaches its best 17 epochs earlier. Its damage
      AP50 (0.0980) is 2.3x the baseline's (0.0419) — the only stage where class 1 registers
      above noise, and still tiny in absolute terms. The RGB-only student still beats the
      RGB+NIR master (0.5209 vs 0.5087), so the "distilling from a worse teacher" observation
      stands.

      What it does not support: a claim that distillation helps. One seed, and a 9-image
      validation split where one detection moves mAP@0.5 by roughly ten points — a 2.8-point
      gap is inside that noise. It is the first *valid* measurement, not a conclusive one.
      A multi-seed run, or a larger split, is what would turn it into a claim; both are out of
      scope for this change and are listed under follow-ups.

      The earlier `20260901T200124175077Z` remains on disk and must not be cited: its distilled
      stage ran 13 steps/epoch at effective batch 4 (4.2d).
- [ ] 4.3 Open the PR against `main`.

## Follow-ups (not blockers of this change)

- Class 1 (damage) remains unlearned: the 0.5078 run ends at `AP50[c0]=0.9889` against
  `AP50[c1]=0.0267`. Whole-model mAP is carried by one class.
- `epochs_phase1: 80` may be short. The best master run was still setting new bests at epochs
  72, 75 and 76, so the epoch budget — not convergence — is what ended it.
- The 9-image validation split is the root cause behind both the early-stopping misfire and
  the metric's volatility. Re-splitting would invalidate every prior comparison, so it is out
  of scope here, but it caps how much any of these numbers can be trusted.

## Dropped from the original change

- ~~NMS in `Trainer._decode_predictions`~~ — delivered by `damage-map-audit` (PR #11) as the
  shared `src/training/decode.py` module with per-class `batched_nms` and configurable
  `nms_enabled` / `nms_iou_threshold`. Re-applying it would duplicate and conflict.
