#!/usr/bin/env python3
"""Evaluate the checkpoint of a fixed epoch of each finished run, for the selection sensitivity (plan, section 8.6).

Takes the same arguments as `scripts/run_small_arms.py` and runs its evaluation steps unchanged, except that they
read `checkpoint_epoch<E>.pt` instead of `best_model.pt`, write under `--epoch-reports-root` and use their own
scratch directory, so they never touch the files of the main analysis. Training steps are never run: a run whose
training has not finished, or that has no checkpoint of that epoch, is skipped. Finished steps are skipped too, so
the command can be repeated as more runs finish.

    python scripts/eval_last_epoch.py --epoch 80 --epoch-reports-root ~/edge_mango/runs/final/reports_epoch80 \\
        --dataset ... --nir-as-rgb-dir ... --variant nano --epochs 80 --folds 0 1 --seeds 42 \\
        --arms est_rgb est_rgbnir kd_rgb kd_rgbnir --machine configs/machines/rtx6000ada.yaml \\
        --checkpoints-root ~/edge_mango/runs/final --reports-root ~/edge_mango/runs/final/reports
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_kfold as rk  # noqa: E402
from scripts import run_small_arms as sas  # noqa: E402


def epoch_steps(args, arm: str, fold: int, seed: int, epoch: int) -> list[tuple[str, list[str], Path]]:
    """The evaluation steps of one run, rewritten to read the checkpoint of `epoch`."""
    run = rk.run_path(args.checkpoints_root, arm, fold, seed)
    best, chosen = str(run / rk.CHECKPOINT), str(run / f"checkpoint_epoch{epoch}.pt")
    scratch, own_scratch = str(run / "eval_scratch"), str(run / f"eval_scratch_epoch{epoch}")
    steps = []
    for key, command, marker in sas.plan_arm(args, arm, fold, seed):
        if key.endswith("/train"):
            continue
        command = [str(c).replace(best, chosen).replace(scratch, own_scratch) for c in command]
        steps.append((f"{key}@epoch{epoch}", command, marker))
    return steps


def main(argv=None) -> int:
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--epoch", type=int, default=80)
    extra.add_argument("--epoch-reports-root", type=Path, required=True)
    own, rest = extra.parse_known_args(argv)
    args = sas.parse_args(rest)
    args.reports_root = own.epoch_reports_root  # every report path of plan_arm now lands here
    env = dict(os.environ)
    failures = 0
    for seed in args.seeds:
        for fold in args.folds:
            for arm in args.arms:
                run = rk.run_path(args.checkpoints_root, arm, fold, seed)
                if not rk.is_train_complete(run) or not (run / f"checkpoint_epoch{own.epoch}.pt").exists():
                    print(f"[skip] {arm}/fold{fold}/seed{seed}: training not finished or no epoch {own.epoch}")
                    continue
                for key, command, marker in epoch_steps(args, arm, fold, seed, own.epoch):
                    if marker.exists():
                        continue
                    print(f"[run] {key}", flush=True)
                    if args.dry_run:
                        print(" ".join(command))
                        continue
                    code = sas._run_logged(command, None, env)
                    if code:
                        print(f"[fail] {key}: exit {code}", flush=True)
                        failures += 1
                        break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
