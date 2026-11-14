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

**Arms.** Every arm of `run_small_arms.py` has a segment twin, trained from scratch on polygon labels, teachers included.
Undistilled arms are "masters" and distilled ones keep their name. The twins are chosen by name or by preset (no default
list holds them, and a name keeps each arm in its own checkpoint and report folder, which a global flag would not). The
detect arms (`est_*`, `kd_*`, `crosskd_*`) keep their names and results:

| detect arm | segment twin | teacher |
|---|---|---|
| `est_rgb`, `est_rgbnir`, `est_nir` | `master_rgb_seg`, `master_rgbnir_seg`, `master_nir_seg` | none |
| `crosskd_rgb`, `crosskd_rgbnir` | `crosskd_rgb_seg`, `crosskd_rgbnir_seg` | `master_rgb_seg`, `master_rgbnir_seg` of its fold and seed |
| `crosskd_rgbnir_high`, `_low`, `_better` | `crosskd_rgbnir_high_seg`, `_low_seg`, `_better_seg` | `master_rgbnir_seg` |
| `kd_rgb`, `kd_rgbnir` (MSE feature KD) | `kd_rgb_seg`, `kd_rgbnir_seg` | `master_rgb_seg`, `master_rgbnir_seg` (in no preset) |

`--arms-preset seg` runs master_rgb_seg, master_rgbnir_seg, master_nir_seg, crosskd_rgb_seg and crosskd_rgbnir_seg;
`seg-full` adds the three gated CrossKD arms. Masters come first in both. `thesis` is `seg` plus `master_rgb_det`, a detect RGB student with the exact training of `est_rgb` (same
command, other output folder) that learns from the tight boxes of the polygons: the final plan's six arms, run on the
rental machine ([`runbook-rental.md`](runbook-rental.md)). A segment student can still load a detect
teacher (the loader reads `task` from the checkpoint), but no arm name does it.

```bash
python scripts/run_small_arms.py --dataset data/cache/v4_captura_v3_poly --make-nir-as-rgb \
    --weights data/weights/yolov8n_state.pt --variant nano --epochs 80 --seeds 42 \
    --arms-preset seg --checkpoints-root checkpoints/v4_seg \
    --reports-root reports/v4_seg --val-only --dry-run      # drop --dry-run to launch
```

`--dry-run` prints every command. A segment arm adds `task=segment mask_ratio=4 mask_weight=7.5` (the `--mask-ratio` and
`--mask-weight` flags) to its training and evaluation overrides; `task=segment` is what makes the dataset load masks. KD
acts on cls and box only. `--weights` with a yolov8n-seg state dict also initialises the Proto and mask layers of the
`nano` variant. `scripts/kd_nir_probe.py --segment` probes the segment masters (`master_rgbnir_seg` against
`master_rgb_seg`).

**`run_kfold.py` arms.** The plan's student arms have segment twins in `V4_ARMS`, so `--extra-override task=segment` is not
needed: `master_etiquetas_seg`, the undistilled student. The KD students distill from a MasterModel, which cannot
segment, and the plan's masters are MasterModels, so they have no twin; the segment plan with CrossKD is the preset
above. A twin has no config file of its own: `--arms configs/experiment/v4_master_etiquetas_seg.yaml` reads
`v4_est_etiquetas.yaml`. `--mask-weight` and `--mask-ratio`
(7.5 and 4) apply. Detect arms never get `task` or `mask_*`.

```bash
python scripts/run_kfold.py --folds-dir data/cache/v4_captura_v3_poly/folds_lot --machine configs/machines/rtx3080.yaml \
    --arms configs/experiment/v4_master_etiquetas_seg.yaml --checkpoints-root checkpoints/v4_seg \
    --reports-root reports/v4_seg --dry-run      # drop --dry-run to launch
```

**Results.** `evaluate_checkpoint.py` takes `task` from the checkpoint (`--override task=...` still wins), and
`metrics.json` gets `mask_map50`, `mask_map_50_95`, `mask_per_class_ap_50` and `mask_per_class_ap_50_95` next to the box
metrics, measured at prototype resolution (160x160 on a 640 input). With `--save-predictions` each damage detection of a
segment model also carries a `mask` in `predictions.jsonl`: the polygon of its largest mask contour, normalized to the
source RGB frame like `xyxy`. `scripts/v4_metrics.py` then adds mask-IoU twins of the latent metrics to
`metrics_v4.json`, with the ground-truth polygons read from the label files (box-only lines leave the mask ground truth;
`mask_gt_box_only` counts them): `latent_mask_ap50` (primary set, IoU 0.50), `latent_mask_ap30`, `mask_ap50_by_hour` and
`developed_mask_ap50`. Face AUROC and the false positives per `sano` photo are score-based and stay as they are.
`aggregate_kfold.py` adds the two latent mask metrics to the Latent-window tables and their paired inference, plus a
"Mask AP" section from `metrics.json`. `export_run_results.py` carries the `mask_*` columns and reads the segment arm
folders; `differences.csv` has the segment comparisons H1_seg (master_rgbnir_seg - master_rgb_seg), H2_seg
(crosskd_rgbnir_seg - crosskd_rgb_seg), secondary_seg (crosskd_rgbnir_seg - master_rgb_seg), exploratory_seg
(crosskd_rgb_seg - master_rgb_seg), mask_supervision (master_rgb_seg - master_rgb_det, box metrics only: the detect
arm has no masks) and the gate_high_vs_low_seg and gate_better_vs_none_seg ones, each for
`latent_ap50`, `latent_ap30`, `latent_mask_ap50` and `latent_mask_ap30` when both arms of a pair were exported.
`val_operating_points.py`, `plot_run_curves.py` and `zone_analysis.py` read the twins too (box AP, per-photo
classification); the mask metrics are in `metrics_v4.json` only. Box metrics, the latent AP and every detect result keep
their format.

**Export.** `scripts/export_student.py` reads `task` and `student_variant` from the checkpoint; an RGB+NIR student cannot
be exported. A segment student exports four outputs, `boxes_raw`, `cls_logits`, `mask_coefs` and `protos`, with the
assembly left outside the graph (see `docs/deployment/mobile-export.md`, section 7):

```bash
python scripts/export_student.py --format onnx --checkpoint checkpoints/v4_seg/master_rgb_seg/fold0/seed42/best_model.pt \
    --out-dir exports --name master_rgb_seg --verify   # needs onnx and onnxruntime; --verify runs on random inputs
```

**Visualization.** For a segment checkpoint `visualize_damage_predictions.py` draws the ground-truth polygons (left) and the
predicted masks (right) as filled contours, with the boxes as before; a detect checkpoint gets boxes only:

```bash
python scripts/visualize_damage_predictions.py --checkpoint checkpoints/v4_seg/master_rgb_seg/fold0/seed42/best_model.pt \
    --config configs/experiment/v4_est_etiquetas.yaml --rgb-dir data/cache/v4_captura_v3_poly/rgb \
    --labels-dir data/cache/v4_captura_v3_poly/folds_lot/fold_0/labels --model-type student --split test --num-samples 20
```

## Top-1 localization (CorLoc and LROC)

In operation a photo shows at most one damage and the protocol gives every damaged photo one damage box, so these
metrics look only at the single highest-score damage box of each photo (any confidence, damage class). A photo with no
damage prediction has score `-inf`: never above a threshold, never correct. `scripts/top1_localization.py` discovers the
evaluations like `export_run_results.py` and reads their `predictions.jsonl`, so it runs on the training machine:

```bash
python scripts/top1_localization.py --runs-root ~/edge_mango/runs/final \
    --dataset ~/edge_mango/data/v4_captura_v3_final --split val --output reports/top1-localization --plot
# test only for the pre-registered sets, with the semantics of export_run_results.py:
#   --split test --test-folds 0 1 --test-seeds 42 [--test-arms est_rgb est_rgbnir]
```

- **CorLoc@IoU** (0.5 and 0.3): over damaged photos, the fraction whose top-1 box has IoU >= the threshold with the
  ground-truth box. Reported on the primary window (damaged photos of `LATENT_HOURS`), the developed hours and all
  hours, with the photo selection of `v4_metrics`. For segment runs, `mask_corloc_*` uses the IoU of the top-1 mask
  against the ground-truth polygon.
- **LROC** (Swensson 1996), on the primary-window damaged photos plus `sano`: sweeping a threshold t over the top-1
  scores, FPF(t) is the fraction of `sano` photos with top-1 score >= t and TPF_loc(t) the fraction of damaged photos
  with top-1 score >= t and a correct top-1 box (IoU 0.5; also 0.3). **AUC-LROC** is the trapezoidal area over
  FPF in [0, 1], the curve ending at FPF = 1 at the lowest threshold. Its maximum is CorLoc: **AUC-LROC <= CorLoc**,
  with equality when every `sano` photo scores below every correct damaged photo.
- **Top-1 AUROC:** photo-level AUROC of the top-1 score (damaged primary-window photos against `sano`), ignoring
  localization. It is the `face_auroc` of `metrics_v4.json`, the maximum damage confidence of the photo, with `-inf`
  instead of 0 for a photo with no prediction; the ranking is the same because confidences are positive.
- **Operating points** at confidence 0.25 and 0.5: `fpf_at_<c>`, `tpf_loc_at_<c>_iou50` (and `_iou30`) and
  `tpf_at_<c>` (detected, with no localization requirement).

Outputs under `--output`: `top1_runs.csv` (one row per evaluation), `top1_lroc_curves.csv` (long format, to redraw the
curves), `top1_differences.csv` (per fold and seed, the `export_run_results.DIFFERENCES` pairs, including the `_seg`
ones, on `corloc_primary_iou50` and `lroc_auc_iou50`, plus the mask CorLoc for segment arms) and `top1_inference.csv`
(the same differences averaged over seeds per fold, with the corrected t and the correlated Bayesian ROPE of
`aggregate_kfold.py`, `--rope` 0.02 by default). `--plot` adds `top1_lroc_curves_<split>.png`.
