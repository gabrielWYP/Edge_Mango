# Runbook: E1 and E2 on the v4 folds

> **Since 2026-11-03 the plan runs the four small arms** of [`scripts/run_small_arms.py`](../../scripts/run_small_arms.py), described in section 1 of the plan. The sections below describe the original six arms, which remain usable.
>
> ```bash
> python scripts/run_small_arms.py --dataset data/cache/v4_captura_v3_final --make-nir-as-rgb \
>     --weights <yolov8n state_dict> --variant <wide|nano> --epochs <n> --seeds 42
> python scripts/aggregate_kfold.py --seeds 42 --reports-root reports/v4_small --rope 0.02 \
>     --fold-spec data/cache/v4_captura_v3_final/folds_lot/folds_meta.json \
>     --arms est_rgb est_nir --baseline est_rgb --treatment est_nir
> ```
>
> Re-run the same command after a crash: an interrupted training continues from its last finished epoch, or stops if
> the settings changed (`--no-resume` trains it from scratch instead).

The exact commands for [`plan-experimental.md`](plan-experimental.md), sections E1, E2, 6 and 8, from the
dataset to the results. Run them from the repository root. Every `run_kfold.py` command is resumable: after a
crash or a Ctrl-C, re-run the same command: the finished steps are skipped and an interrupted training continues from
its last finished epoch (`--no-resume` trains it from scratch instead).

Use the same flags in E1 and E2. A student only distills from a teacher trained on the same fold and seed, with
the same machine profile, `--nir-variant`, `--epochs` and `--extra-override`; anything else stops the runner.

`python` below is the interpreter of the training environment: `myLinuxVenv/bin/python` on the lab machine.

## 1. Regenerate the dataset and the folds

First update the labels file the build reads: export the Label Studio project with all its tasks and replace only the
tasks that changed, matching by `timestamp`
([`captura-v3-label-studio.md`](../annotation/captura-v3-label-studio.md)). Nothing warns about stale labels: a build on
the old file exits 0 and lists the pending tasks as `unlabeled` in its report.

```bash
python scripts/build_v4_dataset.py --labels data/labels/captura_v3_lots1-5_12h_0h.json \
    --output data/cache/v4_captura_v3_final --workers 8
python scripts/make_v4_folds.py --dataset data/cache/v4_captura_v3_final --fold-spec configs/folds/v4_lot_5fold.json
```

Build one dataset with one Python environment: the NIR PNGs and the residual shift differ slightly between OpenCV
versions. The build takes about 14 GB and 15 minutes on 8 workers; keep `--output` out of `/tmp`, which is a 16 GB
tmpfs on this machine. `--residual-shift fruit` is the default, see
[`docs/findings/2026-11-01-rgb-nir-residual`](../findings/2026-11-01-rgb-nir-residual/README.md).

The second command writes `data/cache/v4_captura_v3_final/folds_lot/`, with the per-fold NIR stats in `folds_meta.json`,
and `configs/folds/v4_lot_5fold.json`, the fold spec of the pre-registration. Commit that file with the final labels.
`--output` must be a new or empty folder. Regenerating changes the identity of every run, so later commands
stop on old runs instead of reusing them: use new roots, or delete the old ones.

## 2. E1: the three teachers

Check the plan, then train the first seed.

```bash
python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml \
    --folds-dir data/cache/v4_captura_v3_final/folds_lot --nir-variant nir_clahe \
    --checkpoints-root checkpoints/v4/nir_clahe --reports-root reports/v4/nir_clahe \
    --arms configs/experiment/v4_maestro_rgbnir.yaml configs/experiment/v4_maestro_rgb.yaml \
           configs/experiment/v4_maestro_nir.yaml \
    --seeds 42 --dry-run

python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml \
    --folds-dir data/cache/v4_captura_v3_final/folds_lot --nir-variant nir_clahe \
    --checkpoints-root checkpoints/v4/nir_clahe --reports-root reports/v4/nir_clahe \
    --arms configs/experiment/v4_maestro_rgbnir.yaml configs/experiment/v4_maestro_rgb.yaml \
           configs/experiment/v4_maestro_nir.yaml \
    --seeds 42
```

Then the remaining seeds of the pre-registration (section 8.1). The seed 42 steps are skipped:

```bash
python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml \
    --folds-dir data/cache/v4_captura_v3_final/folds_lot --nir-variant nir_clahe \
    --checkpoints-root checkpoints/v4/nir_clahe --reports-root reports/v4/nir_clahe \
    --arms configs/experiment/v4_maestro_rgbnir.yaml configs/experiment/v4_maestro_rgb.yaml \
           configs/experiment/v4_maestro_nir.yaml \
    --seeds 42 1337 2024
```

## 3. E2: the three students

```bash
python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml \
    --folds-dir data/cache/v4_captura_v3_final/folds_lot --nir-variant nir_clahe \
    --checkpoints-root checkpoints/v4/nir_clahe --reports-root reports/v4/nir_clahe \
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
python scripts/aggregate_kfold.py --reports-root reports/v4/nir_clahe \
    --fold-spec data/cache/v4_captura_v3_final/folds_lot/folds_meta.json \
    --arms maestro_rgb maestro_rgbnir --baseline maestro_rgb --treatment maestro_rgbnir \
    --seeds 42 1337 2024 --rope 0.02
python scripts/plan_seeds.py --metric latent_ap50 --results reports/v4/nir_clahe/kfold_results.json \
    --fold-spec data/cache/v4_captura_v3_final/folds_lot/folds_meta.json
```

The aggregator overwrites `kfold_results.json` under `--reports-root`, so copy it before aggregating H2 the same
way: `--arms est_kd_rgb est_kd_rgbnir --baseline est_kd_rgb --treatment est_kd_rgbnir`.

`plan_seeds.py --results` needs at least 2 seeds in the file. Section 8.2 reads it after the first seed of E1, when
it cannot run yet, so the first estimate of the number of seeds comes after the second seed.

## The NIR variant

The plan pre-registers `nir_clahe` for E1 and E2 (section 8.1), so the commands above use it. `--nir-variant` picks
the NIR images and the per-fold stats together. A run on the raw NIR is exploratory and needs its own roots, because
the two variants would otherwise collide in the same run folders:

```bash
--nir-variant nir --checkpoints-root checkpoints/v4/nir --reports-root reports/v4/nir
```

E0 measures intensity, so it always reads the raw NIR.

## What the configs fix

Changing any of these changes the identity of the runs that used it.

| Setting | Value | Where |
| :-- | :-- | :-- |
| Mango ROI crop | on, margin 0.10, jitter 0.5, in all six arms; it selects the `roi` NIR stats | `configs/experiment/v4_*.yaml` |
| Teacher budget | 100 epochs, no early stopping | `v4_maestro_*.yaml`, `run_kfold.py` |
| Student budget | 80 epochs, warmup 3, no early stopping, in all three student arms | `v4_est_*.yaml`, `run_kfold.py` |
| KD batch | 8 per optimizer step, whatever the machine profile says, because `KDTrainer` has no gradient accumulation | `v4_est_kd_*.yaml` |

If a KD run does not fit in memory, lower the batch of every arm with the same `--extra-override` in E1 and E2
(`batch_size=4`). The teacher identity includes that flag, so adding it only for the students stops the runner: a
change at that point means training the teachers again.


## Segment arms (opt-in)

Instance-segmentation arms: the same students with a YOLOv8-seg head (`Proto` on P3 plus 32 mask coefficients per
anchor), trained on polygon labels. Nothing above changes: the detect arms never mention the task.

**Prerequisite.** Build the dataset from the polygon export (both views already in the RGB frame; the export replaces
`--labels`; the mango polygon is the fruit-mask contour), then the folds. The fold label files keep the polygon lines.

```bash
python scripts/build_v4_dataset.py --polygons data/labels/<export>.json \
    --output data/cache/v4_captura_v3_poly --workers 8
python scripts/make_v4_folds.py --dataset data/cache/v4_captura_v3_poly --fold-spec configs/folds/v4_lot_5fold_poly.json
```

**Arms.** Every arm of `run_small_arms.py` has a `_seg` twin, chosen by name in `--arms` (no default list holds them,
and a name keeps each arm in its own checkpoint and report folder, which a global flag would not):

| arm | student | teacher |
|---|---|---|
| `est_rgb_seg`, `est_rgbnir_seg`, `est_nir_seg` | segment | none |
| `kd_rgb_seg`, `kd_rgbnir_seg`, `crosskd_*_seg` | segment | segment teacher `est_*_seg` of its fold and seed |
| `kd_rgb_segdt`, `kd_rgbnir_segdt`, `crosskd_*_segdt` | segment | detect teacher `est_*` (train it first, on its own labels) |

```bash
python scripts/run_small_arms.py --dataset data/cache/v4_captura_v3_poly --make-nir-as-rgb \
    --weights data/weights/yolov8n_state.pt --variant nano --epochs 80 --seeds 42 \
    --arms est_rgb_seg est_rgbnir_seg kd_rgbnir_seg --checkpoints-root checkpoints/v4_seg \
    --reports-root reports/v4_seg --val-only --dry-run      # drop --dry-run to launch
```

`--dry-run` prints every command. A segment arm adds `task=segment mask_ratio=4 mask_weight=7.5` (the `--mask-ratio` and
`--mask-weight` flags) to its training and evaluation overrides; `task=segment` is what makes the dataset load masks. For
`run_kfold.py` arms use `--extra-override task=segment`. The KD teacher loader reads `task` from the teacher
checkpoint, so a detect or a segment teacher both work, and KD acts on cls and box only. `--weights` with
a yolov8n-seg state dict also initialises the Proto and mask layers of the `nano` variant.

**Results.** `evaluate_checkpoint.py` takes `task` from the checkpoint (`--override task=...` still wins), and
`metrics.json` gets `mask_map50`, `mask_map_50_95`, `mask_per_class_ap_50` and `mask_per_class_ap_50_95` next to the box
metrics, measured at prototype resolution (160x160 on a 640 input). `aggregate_kfold.py` adds a `mask` block and a
"Mask AP" section when a run has them, and `export_run_results.py` carries the `mask_*` columns and reads the `_seg` and
`_segdt` folders. Box metrics, the latent AP and every detect result keep their format. Not covered: ONNX/TFLite export and
`visualize_damage_predictions.py` are box-only.
