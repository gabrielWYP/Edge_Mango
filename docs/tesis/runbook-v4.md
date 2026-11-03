# Runbook: E1 and E2 on the v4 folds

The exact commands for [`plan-experimental.md`](plan-experimental.md), sections E1, E2, 6 and 8, from the
dataset to the results. Run them from the repository root. Every `run_kfold.py` command is resumable: after a
crash or a Ctrl-C, re-run the same command and the finished steps are skipped.

Use the same flags in E1 and E2. A student only distills from a teacher trained on the same fold and seed, with
the same machine profile, `--nir-variant`, `--epochs` and `--extra-override`; anything else stops the runner.

## 1. Regenerate the dataset and the folds

```bash
python scripts/build_v4_dataset.py --output data/cache/v4_captura_v3 --workers 8
python scripts/make_v4_folds.py --dataset data/cache/v4_captura_v3
```

The second command writes `data/cache/v4_captura_v3/folds_lot/`, with the per-fold NIR stats in `folds_meta.json`.
`--output` must be a new or empty folder. Regenerating changes the identity of every run, so later commands
stop on old runs instead of reusing them: use new roots, or delete the old ones.

## 2. E1: the three teachers

Check the plan, then train the first seed.

```bash
python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml \
    --folds-dir data/cache/v4_captura_v3/folds_lot --nir-variant nir \
    --checkpoints-root checkpoints/v4/nir --reports-root reports/v4/nir \
    --arms configs/experiment/v4_maestro_rgbnir.yaml configs/experiment/v4_maestro_rgb.yaml \
           configs/experiment/v4_maestro_nir.yaml \
    --seeds 42 --dry-run

python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml \
    --folds-dir data/cache/v4_captura_v3/folds_lot --nir-variant nir \
    --checkpoints-root checkpoints/v4/nir --reports-root reports/v4/nir \
    --arms configs/experiment/v4_maestro_rgbnir.yaml configs/experiment/v4_maestro_rgb.yaml \
           configs/experiment/v4_maestro_nir.yaml \
    --seeds 42
```

Then the remaining seeds of the pre-registration (section 8.1). The seed 42 steps are skipped:

```bash
python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml \
    --folds-dir data/cache/v4_captura_v3/folds_lot --nir-variant nir \
    --checkpoints-root checkpoints/v4/nir --reports-root reports/v4/nir \
    --arms configs/experiment/v4_maestro_rgbnir.yaml configs/experiment/v4_maestro_rgb.yaml \
           configs/experiment/v4_maestro_nir.yaml \
    --seeds 42 1337 2024
```

## 3. E2: the three students

```bash
python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml \
    --folds-dir data/cache/v4_captura_v3/folds_lot --nir-variant nir \
    --checkpoints-root checkpoints/v4/nir --reports-root reports/v4/nir \
    --arms configs/experiment/v4_est_etiquetas.yaml configs/experiment/v4_est_kd_rgb.yaml \
           configs/experiment/v4_est_kd_rgbnir.yaml \
    --seeds 42 1337 2024
```

Before anything runs, this checks that `maestro_rgb` and `maestro_rgbnir` finished on every fold and seed, and
stops with the path of the first missing teacher if not. Add `--dry-run` to run only that check and see the plan.

Passing the six arms in one command also works, with the teachers of each fold and seed trained first. On the
five folds and three seeds it plans 90 trainings and 450 steps: each training is followed by the evaluation on
`val` and `test`, and each evaluation by its `metrics_v4.json`.

## 4. Results

```bash
python scripts/aggregate_kfold.py --reports-root reports/v4/nir \
    --fold-spec data/cache/v4_captura_v3/folds_lot/folds_meta.json \
    --arms maestro_rgb maestro_rgbnir --baseline maestro_rgb --treatment maestro_rgbnir \
    --seeds 42 1337 2024 --rope 0.02
python scripts/plan_seeds.py --results reports/v4/nir/kfold_results.json \
    --fold-spec data/cache/v4_captura_v3/folds_lot/folds_meta.json
```

The aggregator overwrites `kfold_results.json` under `--reports-root`, so copy it before aggregating H2 the same
way: `--arms est_kd_rgb est_kd_rgbnir --baseline est_kd_rgb --treatment est_kd_rgbnir`.

`plan_seeds.py --results` needs at least 2 seeds in the file. Section 8.2 reads it after the first seed of E1, when
it cannot run yet, so the first estimate of the number of seeds comes after the second seed.

## The raw vs CLAHE choice

`--nir-variant` picks the NIR images and the per-fold stats together. The default is `nir`, and the choice is
still open. For CLAHE, run sections 2 to 4 with its own roots, because the two variants would otherwise collide
in the same run folders:

```bash
--nir-variant nir_clahe --checkpoints-root checkpoints/v4/nir_clahe --reports-root reports/v4/nir_clahe
```

## What the configs fix

Changing any of these changes the identity of the runs that used it.

| Setting | Value | Where |
| :-- | :-- | :-- |
| Mango ROI crop | on, margin 0.10, jitter 0.5, in all six arms; it selects the `roi` NIR stats | `configs/experiment/v4_*.yaml` |
| Teacher budget | 100 epochs, no early stopping | `v4_maestro_*.yaml`, `run_kfold.py` |
| Student budget | 80 epochs, warmup 3, no early stopping, in all three student arms | `v4_est_*.yaml`, `run_kfold.py` |
| KD batch | 8 per optimizer step, whatever the machine profile says, because `KDTrainer` has no gradient accumulation | `v4_est_kd_*.yaml` |

If a KD run does not fit in memory, `--extra-override batch_size=4` lowers its batch, along with the batch of every
other command of that run. The KD arms then take twice the optimizer steps of `est_etiquetas`, which affects the
exploratory comparisons against that arm. Use it in both KD arms so that H2 stays paired.
