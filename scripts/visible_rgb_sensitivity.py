#!/usr/bin/env python3
"""Sensitivity analysis of plan section 8.1: the same comparisons without the latent photos visible in RGB.

The latent photos whose annotation says the mark was already visible in RGB (`visible_rgb = yes` in the dataset
index, 146 of 1987) are the ones where the NIR has the least to add. This recomputes the latent AP50 and AP30 of each
evaluation without them, from the same `predictions.jsonl` and label files as `metrics_v4.json` (the sano photos stay,
so the false positives are charged as in the primary metric), and runs the paired inference of
`scripts/aggregate_kfold.py` on the fold-level differences of H1, H2, the secondary and the exploratory comparison.

Test folds and seeds must be given explicitly, so that results the plan does not allow to read yet stay closed.

    python scripts/visible_rgb_sensitivity.py --reports-root ~/edge_mango/runs/final/reports \\
        --dataset ~/edge_mango/data/v4_captura_v3_final --split test --folds 0 1 2 3 4 --seeds 42 \\
        --output reports/interim-test-seed42/visible-rgb-sensitivity
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aggregate_kfold import load_fold_spec, paired_inference  # noqa: E402
from src.evaluation import v4_metrics  # noqa: E402

COMPARISONS = {
    "H1": ("est_rgb", "est_rgbnir"),
    "H2": ("kd_rgb", "kd_rgbnir"),
    "secondary": ("est_rgb", "kd_rgbnir"),
    "exploratory": ("est_rgb", "kd_rgb"),
}
METRICS = {"latent_ap50": v4_metrics.AP50_IOU, "latent_ap30": v4_metrics.AP30_IOU}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reports-root", type=Path, required=True, help="<arm>/fold<k>_seed<s>/<split>/")
    parser.add_argument("--dataset", type=Path, required=True, help="v4 dataset with index.csv and folds_lot/")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--rope", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def latent_sets(photos, index) -> tuple[list, list]:
    """The primary-metric photos (latent window plus sano) and the same set without the latent ones visible in RGB."""
    latent = [p for p in photos if p.hour is None or p.hour in v4_metrics.LATENT_HOURS]
    kept = [p for p in latent if p.hour is None or index[p.stem]["visible_rgb"] != "yes"]
    return latent, kept


def evaluate(run_dir: Path, labels_dir: Path, index: dict) -> dict:
    """Latent AP50/AP30 of one evaluation, with and without the latent photos visible in RGB."""
    photos = v4_metrics.join_photos(
        v4_metrics.read_predictions(run_dir / "predictions.jsonl"), index, v4_metrics.read_damage_labels(labels_dir)
    )
    latent, kept = latent_sets(photos, index)
    row = {"n_images": len(latent), "n_images_kept": len(kept), "n_removed": len(latent) - len(kept)}
    for name, iou in METRICS.items():
        row[f"{name}_all"] = v4_metrics.damage_ap(latent, iou)["ap"]
        row[f"{name}_no_visible_rgb"] = v4_metrics.damage_ap(kept, iou)["ap"]
    return row


def main(argv=None) -> int:
    args = parse_args(argv)
    index = v4_metrics.read_index(args.dataset / "index.csv")
    spec = {row["fold"]: row for row in load_fold_spec(args.dataset / "folds_lot" / "folds_meta.json")}
    arms = sorted({arm for pair in COMPARISONS.values() for arm in pair})

    runs = []
    for arm in arms:
        for fold in args.folds:
            for seed in args.seeds:
                run_dir = args.reports_root / arm / f"fold{fold}_seed{seed}" / args.split
                if not (run_dir / "predictions.jsonl").is_file():
                    print(f"missing: {run_dir}", file=sys.stderr)
                    continue
                labels_dir = args.dataset / "folds_lot" / f"fold_{fold}" / "labels" / args.split
                runs.append({"arm": arm, "fold": fold, "seed": seed, **evaluate(run_dir, labels_dir, index)})

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / f"runs_{args.split}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)

    by_key = {(r["arm"], r["fold"], r["seed"]): r for r in runs}
    results, lines = {}, []
    for comparison, (baseline, treatment) in COMPARISONS.items():
        results[comparison] = {"baseline": baseline, "treatment": treatment}
        for metric in METRICS:
            for variant in ("all", "no_visible_rgb"):
                column, diffs, folds = f"{metric}_{variant}", [], []
                for fold in args.folds:
                    seeds = [s for s in args.seeds if (baseline, fold, s) in by_key and (treatment, fold, s) in by_key]
                    values = [by_key[(treatment, fold, s)][column] - by_key[(baseline, fold, s)][column]
                              for s in seeds
                              if by_key[(treatment, fold, s)][column] is not None
                              and by_key[(baseline, fold, s)][column] is not None]
                    if values:
                        diffs.append(statistics.mean(values))
                        folds.append(fold)
                inference = paired_inference(diffs, [spec[f] for f in folds], rope=args.rope)
                results[comparison][column] = {"folds": folds, "diffs": diffs, "inference": inference}
                if metric == "latent_ap50":
                    lines.append((comparison, variant, folds, diffs, inference))

    (args.output / f"sensitivity_{args.split}.json").write_text(json.dumps(results, indent=2) + "\n")

    md = [f"# Sensibilidad sin las fotos latentes visibles en RGB ({args.split})", "",
          f"Semillas {args.seeds}, folds {args.folds}. AP50 latente; `todas` = métrica primaria, "
          f"`sin visibles` = sin las fotos latentes con `visible_rgb = yes`. ROPE ±{args.rope}.", "",
          "| Comparación | Fotos | " + " | ".join(f"Δ fold {f}" for f in args.folds)
          + " | Media | IC 95% corregido | P(left) | P(rope) | P(right) | k/5 |",
          "| :-- | :-- | " + " | ".join("--:" for _ in args.folds) + " | --: | :-- | --: | --: | --: | --: |"]
    for comparison, variant, folds, diffs, inf in lines:
        cells = {f: f"{d:+.3f}" for f, d in zip(folds, diffs)}
        if "error" in inf:
            tail = f"{statistics.mean(diffs):+.3f} | {inf['error']} | | | | "
        else:
            ct, b = inf["corrected_t"], inf["bayes"]
            tail = (f"{ct['mean']:+.3f} | [{ct['ci_low']:+.3f}, {ct['ci_high']:+.3f}] | {b['p_left']:.3f} | "
                    f"{b['p_rope']:.3f} | {b['p_right']:.3f} | {inf['n_positive']}/{inf['n_folds']}")
        label = "todas" if variant == "all" else "sin visibles"
        md.append(f"| {comparison} | {label} | " + " | ".join(cells.get(f, "—") for f in args.folds) + f" | {tail} |")
    (args.output / f"sensitivity_{args.split}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
