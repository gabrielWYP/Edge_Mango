# Full chain on registered NIR — teacher, student, distilled (seed 42)

This is the first full chain (teacher → student → distilled student) trained after NIR was
registered to the RGB frame (#24). The two-stream teacher reaches **test damage AP50 = 0.398**,
about double the three-seed mean measured on unregistered NIR. Distillation does **not** help:
the distilled student (0.084) scores below the student trained alone (0.111).

Single seed, on a split with session leakage (see Caveats). Read the teacher number as a
strong signal to confirm with three seeds, not a final figure.

## Result

Damage = class 1. AP50 is computed at `eval_conf_threshold` (0.001), and P/R at the 0.25 operating point.

| Model | Best epoch / ran | Val AP50 | **Test AP50** | Test AP50-95 | Test P / R | Test TP / FP / FN |
|-------|------------------|----------|---------------|--------------|------------|-------------------|
| Teacher (two-stream) | 89 / 100 | 0.737 | **0.398** | 0.140 | 0.165 / 0.595 | 22 / 111 / 15 |
| Student (YOLOv8-nano) | 29 / 50 | 0.106 | **0.111** | 0.021 | 0.020 / 0.622 | 23 / 1147 / 14 |
| Distilled student | 35 / 80 | 0.102 | **0.084** | 0.025 | 0.020 / 0.568 | 21 / 1030 / 16 |

All three models reach mango AP50 = 1.0, so mean mAP50 hides the damage class.

## How it was run

1. **Code:** `main` at `237b30c`, which includes #24 (NIR registered with
   `notebooks/matriz_homografia_aruco.npy` at load time).
2. **Data:** `data/mango_dataset_v3.zip` extracted with overwrite, so `data/annotations` and
   `data/cache` are byte-identical to the zip. The split is the zip's own `splits.json`
   (188 / 23 / 25).
3. **No early stopping.** The training loop has no off switch: it stops when
   `patience_counter >= patience` (`src/training/loop.py`), so `patience=0` would stop after
   epoch 1. `patience=1000` makes every stage run its full epoch budget.
4. **Selection on damage AP:** `selection_metric=damage_ap50` in all three stages.
5. **KD config for a two-stream teacher:** `configs/kd_training_twostream.yaml` (added in this PR).
   `KDTrainer._load_teacher` builds the teacher from the *KD config's* `fusion_mode` and
   `head_strides`, not from the checkpoint. With the default `kd_training.yaml` (early fusion,
   strides [4, 8, 16, 32]) a cross-attention checkpoint fails strict loading. An `--override
   head_strides=[8,16,32]` does not work because the KD CLI does not parse lists.
6. **Smoke test first:** a 1-epoch run of each stage confirmed loading, registration, and the teacher handoff.
7. **Training** (RTX 3080, profile `configs/machines/rtx3080.yaml`, about 83 min in total):

   ```bash
   PY=myLinuxVenv/bin/python; RID=20261014T223110Z
   $PY -m src.training.train --config configs/experiment/twostream.yaml \
     --machine configs/machines/rtx3080.yaml --model master --versioned-run --run-timestamp $RID \
     --override seed=42 patience=1000 selection_metric=damage_ap50
   $PY -m src.training.train --config configs/training_student.yaml \
     --machine configs/machines/rtx3080.yaml --model student --versioned-run --run-timestamp $RID \
     --override seed=42 patience=1000 selection_metric=damage_ap50
   $PY -m src.training.kd_train --config configs/kd_training_twostream.yaml \
     --versioned-run --run-timestamp $RID \
     --override teacher_checkpoint=<maestro best_model.pt> seed=42 patience=1000 selection_metric=damage_ap50
   ```

   | Stage | Epochs | s / epoch | Time |
   |-------|--------|-----------|------|
   | Teacher | 100 | 34.3 | 57 min |
   | Student | 50 | 9.5 | 8 min |
   | Distilled | 80 | 13.6 | 18 min |

8. **Evaluation** with `scripts/evaluate_checkpoint.py` on val and test for each `best_model.pt`.
   The distilled model is evaluated with `configs/training_student.yaml`, which has the same
   architecture. The KD yaml is not a valid `TrainingConfig` (`teacher_checkpoint` is rejected).

## Teacher: stable, and the jump is consistent with registration

- **Not a lone spike:** the best is 0.737 at epoch 89, the second best 0.735, and the neighbors
  0.730 / 0.718. The last 10 epochs average 0.673 ± 0.050.
- **Test improves from 0.177 ± 0.102 to 0.398**, and false positives on test drop from about
  450–950 to 111. The previous measurement (`reports/twostream-v3-3seeds/`) used code from
  before #24. There, NIR was about 168 px off the RGB-frame labels, so the NIR stream saw the
  wrong place for every lesion.
- **Several things changed at once** (registration, and seed 42 against a three-seed mean), so
  this is not a controlled comparison.
- **The val–test gap (0.74 against 0.40) is large.** With 43 and 37 damage boxes, most of that
  gap is sampling noise.

## Distillation: KD loss dominates, detection learns late

The distilled student's per-epoch history (`metrics_history.csv`):

| Epoch | kd_loss | cls_loss | Val damage AP50 |
|-------|---------|----------|-----------------|
| 1  | 327.9 | 1.37 | 0.000 |
| 5  | 215.3 | 0.78 | 0.000 |
| 10 | 95.4  | 0.78 | 0.000 |
| 20 | 15.5  | 0.74 | 0.019 |
| 35 | 4.2   | 0.71 | 0.102 (best) |
| 80 | 1.4   | 0.67 | 0.061 |

- **KD is about 120–280× the classification loss for the first ten epochs,** and damage AP stays at
  zero while it does. The student spends its early budget imitating teacher features. Damage
  AP only rises once the KD term has collapsed.
- **The distilled model ends below the plain student** on test (0.084 against 0.111). Both
  students also produce over 1000 false positives on test against the teacher's 111.
- This points first to a **loss-scale problem** (unnormalized feature MSE with `kd_weight=1.0`).
  It may also point to a representation gap between a fused RGB+NIR ConvNeXt teacher and a nano
  student. Separating the two is the next experiment.

## Caveats

- **Single seed (42).** No σ is available for any of the three models.
- **Session leakage in the split.** Images taken seconds apart land in different splits. Of the
  val images, 7/23 are within 60 s of a train image (18/23 within 300 s); of the test images,
  13/25 (22/25 within 300 s). All 50 pairs added in v3 come from one session (2026-09-27),
  split 40 / 5 / 5. Absolute AP is likely inflated for all three models. Comparisons between
  them on the same split remain fair.
- **Class mix:** the 50 v3 pairs have no damage-free negatives and about twice the damage per
  image of the v2 pairs.

## Artifacts

| Path | Content |
|------|---------|
| `checkpoints/final_runs/20261014T223110Z/{maestro,estudiante,destilado}/` | `metrics_history.csv`, curves, and TensorBoard logs. The `.pt` files are git-ignored. |
| `logs/final_run_20261014T223110Z/*.log` | Full stdout of each stage |
| `reports/final-run-20261014/{stage}_{val,test}/metrics.json` | Evaluation of each `best_model.pt` |

## Next steps

1. **Fix the KD loss scale before anything else:** use a normalized feature loss or a rebalanced
   / warmed-up `kd_weight`, then rerun the distilled stage against the same teacher.
2. **Confirm the teacher with three seeds** on registered NIR.
3. **Use a session-grouped split** (`scripts/make_grouped_folds.py`, currently only on `feat/kfold-stability`)
   for numbers that go into the thesis.
