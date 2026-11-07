#!/usr/bin/env python3
"""Validation AP50 and AP30 of the damage class at every saved epoch of the finished runs (only validation).

Training logs only the validation AP50 of each epoch. This evaluates the `checkpoint_epoch<E>.pt` that the runs keep
every 5 epochs on the validation lot, with the same evaluation command as the main analysis (`eval_last_epoch.py`
rewrites it to read that checkpoint), and computes from its `predictions.jsonl` the damage AP at IoU 0.5 and 0.3 on
all validation photos (the curve of the training logs) and on the latent window plus sano (the plan's metric).
Never runs a test evaluation. Finished evaluations are skipped, so the command can be repeated as more runs finish.

    python scripts/eval_epoch_curve.py --curve-root ~/edge_mango/runs/final/reports_epochs \\
        --output reports/final-runs-progress/epoch_curve_val.csv --curve-jobs 2 \\
        <the arguments of run_small_arms.py: --dataset ... --arms ... --folds ... --seeds 42 ...>
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_kfold as rk  # noqa: E402
from scripts import run_small_arms as sas  # noqa: E402
from scripts.eval_last_epoch import epoch_steps  # noqa: E402
from src.evaluation import v4_metrics  # noqa: E402

FIELDS = ["arm", "fold", "seed", "epoch", "ap50", "ap30", "latent_ap50", "latent_ap30", "n_gt", "n_images"]


def val_step(args, arm: str, fold: int, seed: int, epoch: int):
    """The validation evaluation of one checkpoint, as (key, command, marker); the marker is written last."""
    for key, command, marker in epoch_steps(args, arm, fold, seed, epoch):
        if key.split("@")[0].endswith("/eval_val"):
            return key, command, marker
    raise RuntimeError(f"no validation step for {arm}/fold{fold}/seed{seed}")


def curve_row(arm, fold, seed, epoch, predictions: Path, dataset: Path, index: dict) -> dict:
    labels = v4_metrics.read_damage_labels(dataset / "folds_lot" / f"fold_{fold}" / "labels" / "val")
    photos = v4_metrics.join_photos(v4_metrics.read_predictions(predictions), index, labels)
    latent = [p for p in photos if p.hour is None or p.hour in v4_metrics.LATENT_HOURS]
    every = v4_metrics.damage_ap(photos, v4_metrics.AP50_IOU)
    return {
        "arm": arm, "fold": fold, "seed": seed, "epoch": epoch,
        "ap50": every["ap"], "ap30": v4_metrics.damage_ap(photos, v4_metrics.AP30_IOU)["ap"],
        "latent_ap50": v4_metrics.damage_ap(latent, v4_metrics.AP50_IOU)["ap"],
        "latent_ap30": v4_metrics.damage_ap(latent, v4_metrics.AP30_IOU)["ap"],
        "n_gt": every["n_gt"], "n_images": every["n_images"],
    }


def main(argv=None) -> int:
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--curve-root", type=Path, required=True, help="evaluations land in <root>/epoch<E>/")
    extra.add_argument("--output", type=Path, required=True, help="CSV, one row per run and epoch")
    extra.add_argument("--every", type=int, default=5, help="the checkpoint interval of the runs")
    extra.add_argument("--curve-jobs", type=int, default=2)
    own, rest = extra.parse_known_args(argv)
    args = sas.parse_args(rest)
    env = dict(os.environ)

    todo, done = [], []
    for seed in args.seeds:
        for fold in args.folds:
            for arm in args.arms:
                run = rk.run_path(args.checkpoints_root, arm, fold, seed)
                if not rk.is_train_complete(run):
                    print(f"[skip] {arm}/fold{fold}/seed{seed}: training not finished")
                    continue
                for epoch in range(own.every, args.epochs + 1, own.every):
                    if not (run / f"checkpoint_epoch{epoch}.pt").exists():
                        continue
                    args.reports_root = own.curve_root / f"epoch{epoch}"
                    key, command, marker = val_step(args, arm, fold, seed, epoch)
                    item = (arm, fold, seed, epoch, marker.parent / "predictions.jsonl")
                    (done if marker.exists() else todo).append((key, command, item))

    def run_one(job):
        key, command, item = job
        print(f"[run] {key}", flush=True)
        code = sas._run_logged(command, None, env)
        if code:
            print(f"[fail] {key}: exit {code}", flush=True)
        return item if code == 0 else None

    with ThreadPoolExecutor(max_workers=own.curve_jobs) as pool:
        finished = [item for item in pool.map(run_one, todo) if item]
    items = [job[2] for job in done] + finished

    index = v4_metrics.read_index(args.dataset / "index.csv")
    rows = sorted((curve_row(*item, args.dataset, index) for item in items),
                  key=lambda r: (r["arm"], r["fold"], r["seed"], r["epoch"]))
    own.output.parent.mkdir(parents=True, exist_ok=True)
    with open(own.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {own.output}")
    return 0 if len(finished) == len(todo) else 1


if __name__ == "__main__":
    raise SystemExit(main())
