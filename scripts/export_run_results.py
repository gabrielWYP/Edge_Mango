#!/usr/bin/env python3
"""Export everything needed to report the definitive runs into small CSV and JSON files that can live in git.

The raw artefacts (checkpoints, TensorBoard logs, `predictions.jsonl` of 20+ MB per evaluation) stay on the training
machine. This script keeps what the thesis tables and figures are made from:

    runs.csv              one row per run, split and checkpoint: every scalar of metrics.json and metrics_v4.json
    metrics_full.json     the same evaluations with their nested content, unchanged
    differences.csv       per fold, split, checkpoint and metric: the H1, H2, secondary and exploratory differences
    training_history.csv  per epoch of every finished run: training losses and validation metrics (metrics_history.csv)
    photos.csv            one row per evaluated photo: lot, hour, zone, damage labels, highest damage score, and the
                          damage boxes and true positives (IoU 0.5) at confidence 0.25 and 0.5
    pr_grid.csv           precision-recall of the damage boxes on the photos of the primary metric (latent window +
                          sano) on a grid of confidence thresholds, enough to redraw every PR curve
    folds.csv             validation, test and training lots of each fold

Test results are exported only for the folds and seeds given with `--test-folds` and `--test-seeds` (none by
default): the plan forbids reading the others before the reported seed set is complete (sections 8.1 and 8.6).

    python scripts/export_run_results.py --runs-root ~/edge_mango/runs/final \\
        --dataset ~/edge_mango/data/v4_captura_v3_final --test-folds 0 1 --test-seeds 42 \\
        --output reports/final-runs-export
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.val_operating_points import ARMS, _iou  # noqa: E402
from src.data_pipeline.v4_selection import LATENT_HOURS  # noqa: E402
from src.evaluation.v4_metrics import join_photos, read_damage_labels, read_index, read_predictions  # noqa: E402

DIFFERENCES = {
    "H1": ("est_rgbnir", "est_rgb"),
    "H2": ("kd_rgbnir", "kd_rgb"),
    "secondary": ("kd_rgbnir", "est_rgb"),
    "exploratory": ("kd_rgb", "est_rgb"),
    # CrossKD arms, outside the plan: a comparison whose arms were not exported is skipped.
    "crosskd_H2": ("crosskd_rgbnir", "crosskd_rgb"),
    "crosskd_vs_mse": ("crosskd_rgbnir", "kd_rgbnir"),
    "crosskd_vs_est": ("crosskd_rgbnir", "est_rgb"),
    "gate_high_vs_low": ("crosskd_rgbnir_high", "crosskd_rgbnir_low"),
    "gate_better_vs_none": ("crosskd_rgbnir_better", "crosskd_rgbnir"),
}
# Segment twins of every arm (scripts/run_small_arms.py) come with val_operating_points.ARMS: `est_*` -> `master_*_seg`, others `_seg`.
EXPORT_ARMS = list(ARMS)  # val_operating_points.ARMS already holds the twins, detect arms first
# The segment plan's comparisons (`H1_seg`, ...), on the box metrics and, because segment runs also carry them, on the
# mask ones (`latent_mask_ap50`, `latent_mask_ap30`). The segment masters play the part of the detect `est_*` arms and
# CrossKD is the distillation, so H2 and the secondary and exploratory comparisons use it. The gate comparisons mirror
# the detect `gate_*` ones.
DIFFERENCES.update({
    "H1_seg": ("master_rgbnir_seg", "master_rgb_seg"),
    "H2_seg": ("crosskd_rgbnir_seg", "crosskd_rgb_seg"),
    "secondary_seg": ("crosskd_rgbnir_seg", "master_rgb_seg"),
    "exploratory_seg": ("crosskd_rgb_seg", "master_rgb_seg"),
    "gate_high_vs_low_seg": ("crosskd_rgbnir_high_seg", "crosskd_rgbnir_low_seg"),
    "gate_better_vs_none_seg": ("crosskd_rgbnir_better_seg", "crosskd_rgbnir_seg"),
})
AP_METRICS = ("latent_ap50", "latent_ap30")
MASK_AP_METRICS = ("latent_mask_ap50", "latent_mask_ap30")  # only in the metrics_v4.json of segment runs
METRIC_PREFIXES = ("map", "per_class", "mask_")  # metrics.json scalars exported; mask_* only exist for segment runs
CHECKPOINTS = {"best": "reports", "epoch80": "reports_epoch80"}  # checkpoint label: reports directory under runs root
GRID = [0.001] + [round(0.01 * i, 2) for i in range(1, 100)]
RUN_DIR = re.compile(r"fold(\d+)_seed(\d+)")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, required=True, help="<arm>/fold<k>/seed<s>/ and reports*/")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--test-folds", type=int, nargs="*", default=[])
    parser.add_argument("--test-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--test-arms", nargs="*", default=None,
                        help="arms whose test results may be read (default: every arm); a test look the plan "
                             "registers for some arms only must name them")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluations(args, folds: dict):
    """(arm, fold, seed, split, checkpoint, directory) of every finished evaluation that may be read."""
    for checkpoint, reports in CHECKPOINTS.items():
        for arm in EXPORT_ARMS:
            for run in sorted((args.runs_root / reports / arm).glob("fold*_seed*")):
                match = RUN_DIR.fullmatch(run.name)
                if not match:
                    continue
                fold, seed = int(match[1]), int(match[2])
                for split in ("val", "test"):
                    if split == "test" and (fold not in args.test_folds or seed not in args.test_seeds
                                            or (args.test_arms is not None and arm not in args.test_arms)):
                        continue
                    if (run / split / "metrics_v4.json").exists():
                        yield arm, fold, seed, split, checkpoint, run / split


def photo_rows(directory: Path, labels_dir: Path, index: dict) -> tuple[list[dict], list[tuple[float, bool]], int]:
    """Per-photo rows, and the ranked damage boxes and label count of the primary-metric photos."""
    photos = join_photos(read_predictions(directory / "predictions.jsonl"), index, read_damage_labels(labels_dir))
    rows, ranked, n_gt = [], [], 0
    for photo in photos:
        hits = []
        matched: set[int] = set()
        order = sorted(range(len(photo.pred_scores)), key=lambda i: -float(photo.pred_scores[i]))
        for i in order:
            hit = False
            if len(photo.gt_boxes):
                best_iou, best = _iou(photo.pred_boxes[i], photo.gt_boxes).max(dim=0)
                if best_iou >= 0.5 and int(best) not in matched:
                    hit = True
                    matched.add(int(best))
            hits.append((float(photo.pred_scores[i]), hit))
        primary = photo.hour is None or photo.hour in LATENT_HOURS
        if primary:
            ranked += hits
            n_gt += len(photo.gt_boxes)
        row = {
            "stem": photo.stem, "lot": photo.face[0], "mango_id": photo.face[1], "cara": photo.face[2],
            "hour": "sano" if photo.hour is None else photo.hour, "zona": photo.zona,
            "provenance": photo.provenance, "primary_metric_photo": int(primary), "n_gt": len(photo.gt_boxes),
            "max_damage_score": round(photo.face_score, 5),
        }
        for t in (0.25, 0.5):
            row[f"boxes_at_{t}"] = sum(s >= t for s, _ in hits)
            row[f"tp_at_{t}"] = sum(h for s, h in hits if s >= t)
        rows.append(row)
    return rows, ranked, n_gt


def ap_entries(v4: dict) -> dict:
    """`{metric: ap}` of the latent AP metrics a `metrics_v4.json` holds: box ones always, mask ones if segment."""
    return {metric: v4[metric]["ap"] for metric in (*AP_METRICS, *MASK_AP_METRICS) if metric in v4}


def compute_differences(ap: dict) -> list[dict]:
    """One row per run, split, checkpoint and metric for every `DIFFERENCES` pair whose two arms were exported.

    `ap` maps (fold, seed, split, checkpoint, metric, arm) to the AP.
    """
    differences = []
    for (fold, seed, split, checkpoint, metric, arm) in sorted({k[:5] + ("",) for k in ap}):
        for name, (treatment, baseline) in DIFFERENCES.items():
            t = ap.get((fold, seed, split, checkpoint, metric, treatment))
            b = ap.get((fold, seed, split, checkpoint, metric, baseline))
            if t is not None and b is not None:
                differences.append({"fold": fold, "seed": seed, "split": split, "checkpoint": checkpoint,
                                    "metric": metric, "comparison": name, "treatment": treatment,
                                    "baseline": baseline, "treatment_ap": t, "baseline_ap": b, "delta": t - b})
    return differences


def main(argv=None) -> int:
    args = parse_args(argv)
    meta = json.loads((args.dataset / "folds_lot" / "folds_meta.json").read_text())["folds"]
    lot = lambda name: name.removeprefix("lot")  # noqa: E731
    folds = {
        int(k): {
            "val_lot": lot(v["val"]), "test_lot": lot(v["test"]),
            "train_lots": " ".join(lot(x) for x in v["train"]),
        }
        for k, v in meta.items()
    }
    index = read_index(args.dataset / "index.csv")
    runs, full, photos, grid, ap = [], {}, [], [], {}
    for arm, fold, seed, split, checkpoint, directory in evaluations(args, folds):
        key = {"arm": arm, "fold": fold, "seed": seed, "split": split, "checkpoint": checkpoint,
               "eval_lot": folds[fold][f"{split}_lot"]}
        metrics = json.loads((directory / "metrics.json").read_text())
        v4 = json.loads((directory / "metrics_v4.json").read_text())
        full[f"{arm}/fold{fold}/seed{seed}/{split}/{checkpoint}"] = {"metrics": metrics, "metrics_v4": v4}
        row = dict(key)
        flatten("v4", v4, row)
        flatten("", {k: v for k, v in metrics.items() if k.startswith(METRIC_PREFIXES)}, row)
        runs.append(row)
        for metric, value in ap_entries(v4).items():
            ap[(fold, seed, split, checkpoint, metric, arm)] = value
        labels = args.dataset / "folds_lot" / f"fold_{fold}" / "labels" / split
        rows, ranked, n_gt = photo_rows(directory, labels, index)
        photos += [{**key, **r} for r in rows]
        for t in GRID:
            kept = [h for s, h in ranked if s >= t]
            tp = sum(kept)
            grid.append({**key, "threshold": t, "tp": tp, "fp": len(kept) - tp, "fn": n_gt - tp,
                         "precision": round(tp / len(kept), 5) if kept else None,
                         "recall": round(tp / n_gt, 5) if n_gt else None})
        print(f"{arm} fold{fold} seed{seed} {split} {checkpoint}: {len(rows)} photos", flush=True)
    differences = compute_differences(ap)
    history = []
    for arm in EXPORT_ARMS:
        for path in sorted(args.runs_root.glob(f"{arm}/fold*/seed*/metrics_history.csv")):
            run = path.parent
            if not (run / "run_complete.json").exists():
                continue
            fold, seed = int(run.parent.name.removeprefix("fold")), int(run.name.removeprefix("seed"))
            history += [{"arm": arm, "fold": fold, "seed": seed, **r} for r in csv.DictReader(path.open())]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "runs.csv", runs)
    write_csv(args.output / "differences.csv", differences)
    write_csv(args.output / "training_history.csv", history)
    write_csv(args.output / "photos.csv", photos)
    write_csv(args.output / "pr_grid.csv", grid)
    write_csv(args.output / "folds.csv", [{"fold": k, **v} for k, v in sorted(folds.items())])
    (args.output / "metrics_full.json").write_text(json.dumps(full, indent=1, sort_keys=True) + "\n")
    print(f"{len(runs)} evaluations, {len(history)} epochs, {len(photos)} photo rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
