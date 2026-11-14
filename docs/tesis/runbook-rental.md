# Runbook: the final thesis runs on a rented GPU machine

The final plan: **6 arms x 5 folds x 3 seeds (42, 1337, 2024) = 90 trainings**, from scratch, on the polygon v4
dataset. The arms are the `thesis` preset of [`run_small_arms.py`](../../scripts/run_small_arms.py): `master_rgb_seg`,
`master_rgbnir_seg`, `master_nir_seg`, `master_rgb_det`, `crosskd_rgb_seg`, `crosskd_rgbnir_seg` (masters first; each
CrossKD student distills from the master of its fold and seed). `master_rgb_det` is `est_rgb` trained on the
polygon-derived boxes; `master_rgb_seg - master_rgb_det` is the mask-supervision comparison. The plan row
([plan-experimental.md](plan-experimental.md), section 11, 2026-11-13) is a DRAFT: the author confirms it before the
first run. Segment details are in [runbook-v4.md](runbook-v4.md), "Segment arms (opt-in)".

Written for Ubuntu with an NVIDIA driver and Python 3.12. Nothing below needs the image or GPU to be known in advance.
`$WORK` is a folder on the rented machine with about 60 GB free (dataset 10 GB, pixel cache, checkpoints, logs).

## 0. By hand, once

- On the rental machine: `gh auth login -h github.com -p https -w` (device flow). No token goes in any script or file.
- Starting weights: all six arms start from the same yolov8n-seg state dict (plan section 11, 2026-11-13 row). Build it
  once, locally, from the official checkpoint (no ultralytics package needed):

  ```bash
  curl -L -o /tmp/yolov8n-seg.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
  sha256sum /tmp/yolov8n-seg.pt   # a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c
  python scripts/convert_ultralytics_weights.py /tmp/yolov8n-seg.pt data/weights/yolov8n-seg_state.pt
  ```

  Detect arms take its backbone, neck and box/class head; segment arms also take the Proto and mask-coefficient layers.

## 1. Package the data (local machine)

Build the polygon v4 dataset and its folds as in runbook-v4 (`build_v4_dataset.py --polygons ...`, then
`make_v4_folds.py`). The folds need the raw NIR, so they are built here and shipped inside the archive; the raw NIR stays home.

```bash
scripts/remote/package_v4.sh data/cache/v4_captura_v3_poly --dry-run     # contents and estimated size, writes nothing
scripts/remote/package_v4.sh data/cache/v4_captura_v3_poly --out /tmp/v4_poly.tar.zst
```

Contents: `rgb/` (symlinks dereferenced), `nir_clahe/` (every arm reads this variant), `labels/`, `homographies/`,
`index.csv`, `report.json`, `folds_lot/` and a `MANIFEST.sha256`. Left out: `nir_clahe_view/` and the raw `nir/`.
PNG and JPEG do not compress: expect about 10 GB (rgb 1.1 GB + nir_clahe 9.1 GB for the current 3966-photo box dataset).
`--with-raw-nir` adds 7.6 GB; it is needed only if `folds_lot/` is absent and the remote machine must build it.
The script writes `<archive>.sha256` next to the archive.

## 2. Upload

The host and user come from the rental provider.

```bash
rsync -avP --partial /tmp/v4_poly.tar.zst /tmp/v4_poly.tar.zst.sha256 <user>@<host>:<WORK>/upload/
rsync -avP data/weights/yolov8n-seg_state.pt <user>@<host>:<WORK>/upload/
# or: scp -r /tmp/v4_poly.tar.zst* <user>@<host>:<WORK>/upload/
```

`rsync --partial` resumes an interrupted upload. The repository itself comes from GitHub, not from here.

## 3. Set up and smoke-test (remote)

Get the script first (it is in the repo; for the first run fetch just that file, then it clones the rest):

```bash
mkdir -p $WORK/upload && cd $WORK
gh repo clone gabrielWYP/Edge_Mango edge_mango -- --branch <branch>      # or: git clone, once gh is logged in
edge_mango/scripts/remote/setup_remote.sh --data $WORK/upload/v4_poly.tar.zst --workdir $WORK \
    --branch <branch> --weights $WORK/upload/yolov8n-seg_state.pt --smoke
```

The script is safe to re-run. It prints `nvidia-smi` and the Python version, updates the repo, makes `$WORK/.venv`,
installs torch from the PyTorch index that matches the driver's CUDA version (cu128, cu126, cu124, cu121 or cu118;
override with `--torch-index cu126` or `TORCH_INDEX=https://download.pytorch.org/whl/cu130`) and
`requirements-remote.txt`, unpacks the archive, verifies the manifest, builds the pixel cache (and the folds only if
the archive had none), runs a quick pytest subset and writes `$WORK/env.sh`.

`--smoke` trains 2 epochs of `master_rgb_seg` on fold 0 into `$WORK/runs/smoke` (not a result) and prints the mean
**seconds per epoch**. Keep that number for the cost estimate. `--machine configs/machines/<x>.yaml` picks the profile
(default `rtx_a5000`, batch 8, for 20 GB or more; `rtx3080` below that).

From now on, in every shell: `source $WORK/env.sh && cd $REPO` (defines `PY`, `REPO`, `DATASET`, `PIXEL`, `RUNS`,
`WEIGHTS`, `MACHINE`).

## 4. Launch

Review the plan first: 450 lines (90 runs x 5 steps), masters before the CrossKD runs of the same fold and seed.

```bash
python scripts/run_small_arms.py --dataset $DATASET --weights $WEIGHTS --variant nano --epochs 80 \
    --arms-preset thesis --folds 0 1 2 3 4 --seeds 42 1337 2024 --machine $MACHINE \
    --pixel-cache-root $PIXEL --checkpoints-root $RUNS --reports-root $RUNS/reports --jobs 3 --dry-run | head -12
```

Drop `--dry-run` to launch, inside `tmux new -s thesis` (or `nohup ... &`). Seeds are the outer loop, so seed 42 finishes
for all arms and folds first. Each run is evaluated on val and on test; test results are written but **not read** until
the reported set is complete (plan, section 8.1). Add `--val-only` to skip the test evaluations altogether.

**`--jobs`.** Three concurrent runs was the throughput sweet spot on one GPU earlier. It depends on the GPU, so
re-measure: run the smoke or the first masters with `--jobs 2`, `3` and `4`, compare `jobs / seconds-per-epoch`
(epoch lines of `$RUNS/reports/logs/*.log` end with `| <seconds>s`), and watch `nvidia-smi` for memory and the CPU
count (`nproc`): each run uses `num_workers` data workers (4 in `rtx_a5000`, 8 in `rtx6000ada`), so `jobs x num_workers`
should not exceed the vCPUs. Logs: `$RUNS/reports/logs/<arm>_fold<k>_seed<s>.log`.

**Resume.** After a crash, a preemption or a reboot: `source $WORK/env.sh`, re-run `setup_remote.sh` only if the
machine was reimaged, then re-run the exact same launch command. Finished steps are skipped and an interrupted training
continues from its last finished epoch. A change of any setting (epochs, variant, machine, overrides, preset) stops the
step: use the same flags, or new roots.

**Cost.** wall hours = runs x epochs x s_per_epoch(k) / k / 3600, with k = `--jobs` and `s_per_epoch(k)` measured at
that concurrency (the smoke gives k = 1, so it is an upper bound for the per-run time and a lower bound for the
total; CrossKD runs also forward a teacher, so they are slower than masters). Add the evaluations (val and test of every run, a few
minutes in total per run). Example, only to show the arithmetic: 90 x 80 x 40 s / 3 = 96,000 s = 26.7 h; multiply by the
hourly price. Measure instead of trusting the example.

## 5. Push progress to git (remote)

```bash
scripts/remote/push_progress.sh --runs-root $RUNS --dataset $DATASET --every-min 30     # loop; Ctrl-C to stop
scripts/remote/push_progress.sh --runs-root $RUNS --dataset $DATASET                      # once (cron friendly)
```

It runs `export_run_results.py` (val only; test results need `--test-folds ... --test-seeds ... [--test-arms ...]`,
which are passed through unchanged, so use them only for the sets the plan registers), then commits only the `.csv`,
`.json` and `.md` files of `reports/thesis-export/` to the branch `runs/thesis-<YYYYMMDD>` in a separate git worktree
(`$WORK/.progress_wt`, so the code under the running jobs never changes) and pushes. Checkpoints, logs, data,
`predictions.jsonl` and plots are never added. With nothing new it prints "nothing new" and exits 0. `--dry-run` shows
the export command and the branch.

## 6. Pull the results back (local)

```bash
git fetch origin runs/thesis-<YYYYMMDD>                       # the small report files
mkdir -p runs_thesis
rsync -avP --include='*/' --include='best_model.pt' --include='run_complete.json' --exclude='*' \
    <user>@<host>:<WORK>/runs/thesis/ runs_thesis/            # checkpoints (90 of them, small)
rsync -avP --include='*/' --include='predictions.jsonl' --include='metrics*.json' --exclude='*' \
    <user>@<host>:<WORK>/runs/thesis/reports/ runs_thesis/reports/    # for plots and top-1 localization; several GB
```

Do this before the instance is destroyed. The analyses (`export_run_results.py`, `top1_localization.py`,
`val_operating_points.py`, `plot_run_curves.py`, `zone_analysis.py`) read `master_rgb_det` and the `mask_supervision`
comparison without further flags.

## 7. Label effect: the old `est_rgb` on the new labels (descriptive only)

Numbers evaluated against different ground truth are never compared. To see what the labels did, re-evaluate the OLD
`est_rgb` checkpoints (box labels, `~/edge_mango/runs/final/est_rgb/fold<k>/seed<s>/best_model.pt` on the training machine)
against the NEW polygon v4 labels, and compare that with `master_rgb_det`, which trained and is evaluated on them.

Copy the old checkpoints to a scratch root (the evaluation writes `eval_scratch/` next to them, so never evaluate in place):

```bash
rsync -avP --include='*/' --include='best_model.pt' --exclude='*' \
    <training-machine>:~/edge_mango/runs/final/est_rgb/ $WORK/runs/old_est_rgb/est_rgb/
```

Then, with the new dataset, take the evaluation commands `run_small_arms.py` would run for `est_rgb` and execute only those
(never the training lines: the old checkpoints have no completion marker here, so the runner would train them again):

```bash
LE=$WORK/runs/label_effect
python scripts/run_small_arms.py --dataset $DATASET --variant nano --epochs 80 --arms est_rgb \
    --folds 0 1 2 3 4 --seeds 42 1337 2024 --machine $MACHINE \
    --checkpoints-root $WORK/runs/old_est_rgb --reports-root $LE/reports --val-only --dry-run \
  | grep -E '^\[run\] [^:]*/(eval|metrics)_(val|test):' | sed -E 's/^\[run\] [^:]*: //' \
  | while IFS= read -r cmd; do echo ">> $cmd"; bash -c "$cmd" || break; done
```

Each command is `evaluate_checkpoint.py --checkpoint <old best_model.pt> --split <split> --override <new fold labels> ...`
followed by `v4_metrics.py` on its predictions. `--val-only` keeps the test split unread; drop it only for the look the
plan registers. Then export it like any run set and put it beside the new arms:

```bash
python scripts/export_run_results.py --runs-root $LE --dataset $DATASET --output reports/label-effect-export
```

`est_rgb` (old checkpoints, new labels) against `master_rgb_det` (new training, new labels) is the label effect, reported as
a description; the plan registers no test on it. The old `est_rgb` results on the old labels stay in their own tables.
