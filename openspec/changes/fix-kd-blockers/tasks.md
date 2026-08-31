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

## Phase 4 — Verification

- [x] 4.1 Full suite green: `./myLinuxVenv/bin/python -m pytest tests/ -q` → 274 passed.
- [ ] 4.2 Re-run the three-stage final pipeline so the distilled stage is measured against a
      baseline with a matching budget and with the projections actually training. Until this
      lands, no distillation claim (including "the RGB-only student beat the distilled one")
      is supportable.
- [ ] 4.3 Open the PR against `main`.

## Dropped from the original change

- ~~NMS in `Trainer._decode_predictions`~~ — delivered by `damage-map-audit` (PR #11) as the
  shared `src/training/decode.py` module with per-class `batched_nms` and configurable
  `nms_enabled` / `nms_iou_threshold`. Re-applying it would duplicate and conflict.
