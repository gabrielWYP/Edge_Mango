# Two-stream on dataset v3 — three seeds, selected on damage AP

The two-stream arm (`configs/experiment/twostream.yaml`) retrained on dataset v3 with the
checkpoint selected on damage AP50 reaches **test damage AP50 = 0.177 ± 0.102** over seeds
42 / 1337 / 2024, up from 0.101 ± 0.084 in the v2 run of 2026-09-06. Two of the three seeds
converge late and stay stable. The third was cut short by early stopping, and it accounts for
most of the spread.

## Result

Damage = class 1. AP50 is computed at `eval_conf_threshold` (0.001), and P/R at the 0.25
operating point.

| Seed | Best epoch / ran | Val AP50 | Test AP50 | Test AP50-95 | Test P / R |
|------|------------------|----------|-----------|--------------|------------|
| 42   | 78 / 98          | 0.251    | 0.126     | 0.034        | 0.042 / 0.541 |
| 1337 | 73 / 93          | 0.317    | **0.294** | 0.076        | 0.039 / 0.568 |
| 2024 | 35 / 55          | 0.142    | 0.110     | 0.022        | 0.023 / 0.595 |
| **Mean ± σ** (ddof=1) | | **0.237 ± 0.089** | **0.177 ± 0.102** | | |

The val values that `evaluate_checkpoint.py` reports match the `New best damage_ap50` lines in
the training logs exactly, so the saved `best_model.pt` is the epoch that was selected.

## How we got here

1. **Code:** `origin/main` at `c745bc9`, unmodified. No branch-only code was used.
2. **Data:** `data/mango_dataset_v3.zip` (from #22), extracted into `data/`. It holds 236
   labeled RGB+NIR pairs: 188 train / 23 val / 25 test.
3. **Manifest fix (local, git-ignored).** The zip ships a stale **v2** `splits.json` with 186
   ids (148/18/20). The label directories hold all 236 pairs. `YOLODataset._verify_manifest_alignment`
   raises on that mismatch, so training cannot start. The manifest was rebuilt from the label
   directories, keeping the 186 original assignments and appending the 50 new stems to the split
   whose directory holds them. Every one of the 50 has a non-empty label file and an RGB/NIR
   image pair. The old file is kept locally as `splits.v2_186.json.bak`.

   ```python
   import json, pathlib
   root = pathlib.Path("data/annotations/yolo")
   old = json.load(open(root / "splits.json"))
   new = {sp: sorted(old[sp]) + sorted({p.stem for p in (root / "labels" / sp).glob("*.txt")} - set(old[sp]))
          for sp in ["train", "val", "test"]}
   json.dump(new, open(root / "splits.json", "w"), indent=2)
   ```

   `scripts/prepare_yolo_splits.py --reconcile` does **not** fix this. It reports unmanifested
   stems but does not add them.
4. **One override that matters:** `selection_metric=damage_ap50`. The config default is
   `map50`, the two-class mean. Class 0 (mango) sits near 1.0, so that metric barely sees damage
   and picks checkpoints far from the damage peak. That is why the v2 run's checkpoints missed
   its 0.18–0.21 val peaks.
5. **Smoke test:** a 1-epoch run confirmed 188/23 images load and selection runs on `damage_ap50`.
6. **Training,** sequentially on an RTX 3080 (fp32, batch 4 × accum 2 = effective 8), about
   32–37 s/epoch and 2 h 20 min for all three seeds:

   ```bash
   for S in 42 1337 2024; do
     myLinuxVenv/bin/python -m src.training.train \
       --config configs/experiment/twostream.yaml --model master \
       --machine configs/machines/rtx3080.yaml \
       --override seed=$S selection_metric=damage_ap50 output_dir=checkpoints/twostream_v3/seed$S \
       > logs/twostream_v3/seed$S.log 2>&1
   done
   ```

   Everything else is the config as committed: 100 epochs, patience 20, `end_to_end` schedule,
   lr 1e-3, backbone_lr_mult 0.1, warmup 5, `fusion_mode: cross_attention`, strides [8, 16, 32].
7. **Evaluation** of each `best_model.pt` on val and test:

   ```bash
   myLinuxVenv/bin/python scripts/evaluate_checkpoint.py \
     --config configs/experiment/twostream.yaml --model master \
     --checkpoint checkpoints/twostream_v3/seed$S/best_model.pt --split test \
     --override seed=$S selection_metric=damage_ap50 \
     --output-dir reports/twostream-v3-3seeds/test_seed$S
   ```

## Training stability

Statistics of the per-epoch val damage AP50 come from `checkpoints/twostream_v3/seed*/metrics_history.csv`.

| Seed | Best | 2nd best | Neighbors of best | Final | Last-10 mean ± σ | Median / max epoch jump |
|------|------|----------|-------------------|-------|------------------|-------------------------|
| 42   | 0.251 | 0.248 | 0.159 / 0.179 | 0.215 | 0.211 ± 0.012 | 0.031 / 0.181 |
| 1337 | 0.317 | 0.300 | 0.205 / 0.262 | 0.200 | 0.217 ± 0.030 | 0.036 / 0.146 |
| 2024 | 0.142 | 0.125 | **0.021 / 0.055** | 0.125 | 0.070 ± 0.024 | 0.020 / 0.121 |

- **Seeds 42 and 1337 are stable late.** Their best epoch is not a lone spike: the
  second-best value is within 0.003 and 0.017 of it. Their last 10 epochs hold around 0.21.
- **Seed 2024 is a premature stop, not a different model.** Its best (epoch 35) is a spike
  between two epochs at 0.02–0.06. Patience 20 then ended the run at epoch 55. The other two
  seeds needed 73–78 epochs to reach their best, so this run was cut before it could converge.
- Compared with the v2 run (median epoch jump 0.04–0.05, max 0.15–0.16), epoch-to-epoch noise
  is lower, but single jumps of 0.12–0.18 still happen.

## Comparison with the v2 run (2026-09-06)

| Run | Data (train/val/test) | Selected on | Val AP50 | Test AP50 |
|-----|-----------------------|-------------|----------|-----------|
| `reports/twostream-3seeds/` | v2: 148 / 18 / 20 | two-class mean (`>=`) | 0.163 ± 0.025 | 0.101 ± 0.084 |
| this report | v3: 188 / 23 / 25 | `damage_ap50` (`>`) | 0.237 ± 0.089 | 0.177 ± 0.102 |

**Not a controlled comparison.** Three things changed at once: the data (+40 train images), the
selection metric, and the val/test sets. v3 val and test are supersets of v2's (+5 images each);
val went from 27 to 43 damage boxes and test from 26 to 37. Read this as "the arm on current
data and current main", not as the effect of any single change.

## Caveats

- **Small eval sets.** Val has 43 damage boxes and test 37, so one lesion moves AP by a few
  hundredths. σ over three seeds is itself a rough estimate.
- **Precision at the 0.25 operating point is 0.02–0.08**, with 454–953 false positives on test.
  AP is a ranking metric and does not penalize this, but the detector is not usable at this
  threshold yet.
- **Test was evaluated once per seed and not used for any choice.** Selection used val only.

## Artifacts

| Path | Content |
|------|---------|
| `checkpoints/twostream_v3/seed*/` | `metrics_history.csv`, curves (PNG), TensorBoard logs. The `.pt` files are git-ignored. |
| `logs/twostream_v3/seed*.log` | Full training stdout |
| `reports/twostream-v3-3seeds/{val,test}_seed*/metrics.json` | Per-split evaluation of each `best_model.pt` |

## Next step

Rerun the three seeds with `patience=40` (one `--override`, about 3 h). That isolates whether the
seed-2024 outlier is purely the early-stopping cut. If σ drops, patience is the fix. If not, the
next candidate is weight EMA (`feat/kfold-stability`).
