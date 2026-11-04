#!/usr/bin/env python3
"""Stratified damage metrics of one evaluation (plan-experimental section 7).

Reads the `predictions.jsonl` of `scripts/evaluate_checkpoint.py --save-predictions`, the `index.csv`
of the v4 dataset and the label files of the evaluated split, and writes `metrics_v4.json`: latent
AP50 (primary), latent AP30, per-face AUROC, and AP50 per hour, on developed, per impact zone and per
box provenance, plus false positives per `sano` image. The definitions are in
`src/evaluation/v4_metrics.py`. `scripts/aggregate_kfold.py` reads the file next to `metrics.json`.

A stratum with no ground truth is null, never 0. The predictions and the label files must hold the
same photos, else this exits with 2 and writes nothing.

Usage:
    python scripts/v4_metrics.py --predictions <p>/predictions.jsonl --index <dataset>/index.csv \\
        --labels-dir <fold>/labels/<split> --output <p>/metrics_v4.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import v4_metrics  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratified damage metrics of one evaluation.")
    parser.add_argument("--predictions", required=True,
                        help="predictions.jsonl written by evaluate_checkpoint.py --save-predictions.")
    parser.add_argument("--index", required=True, help="index.csv of the v4 dataset.")
    parser.add_argument("--labels-dir", required=True, help="Label files of the evaluated split: <fold>/labels/<split>.")
    parser.add_argument("--output", required=True, help="Where to write metrics_v4.json.")
    parser.add_argument("--conf-threshold", type=float, default=v4_metrics.DEFAULT_CONF_THRESHOLD,
                        help="Operating point of the false positives per sano image "
                             f"(default: {v4_metrics.DEFAULT_CONF_THRESHOLD}, the training config's conf_threshold).")
    return parser.parse_args(argv)


def _fmt(value: float | None) -> str:
    return "null" if value is None else f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (math.isfinite(args.conf_threshold) and 0.0 <= args.conf_threshold <= 1.0):
        print(f"ERROR: --conf-threshold must be in [0, 1], got {args.conf_threshold}", file=sys.stderr)
        return 2
    try:
        index = v4_metrics.read_index(args.index)
        photos = v4_metrics.join_photos(
            v4_metrics.read_predictions(args.predictions), index, v4_metrics.read_damage_labels(args.labels_dir)
        )
        metrics = v4_metrics.compute_metrics(photos, index, args.conf_threshold)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    for label, key in (("latent AP50", "latent_ap50"), ("latent AP30", "latent_ap30")):
        block = metrics[key]
        print(f"{label}: {_fmt(block['ap'])} (n_gt={block['n_gt']}, n_images={block['n_images']})")
    auc = metrics["face_auroc"]
    print(f"face AUROC: {_fmt(auc['auroc'])} (n_positive={auc['n_positive']}, n_negative={auc['n_negative']})")
    fp = metrics["sano_fp_per_image"]
    print(f"FP per sano image at {fp['conf_threshold']:g}: {_fmt(fp['fp_per_image'])} (n_images={fp['n_images']})")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
