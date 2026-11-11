#!/usr/bin/env python3
"""Top-1 localization (CorLoc and LROC) of every evaluation under a runs root, from the saved `predictions.jsonl`.

In operation a photo shows at most one damage, so these metrics look only at the highest-confidence damage box of each
photo (definitions in `src/evaluation/top1_localization.py`): CorLoc@IoU 0.5 and 0.3 on the primary window, the developed
hours and all hours; the LROC curve and its area (AUC-LROC <= CorLoc) on the primary window plus `sano`; the top-1 AUROC;
and FPF, TPF_loc and TPF at confidence 0.25 and 0.5. Segment evaluations also get the mask CorLoc.

Evaluations are discovered exactly as `scripts/export_run_results.py` does (same `--runs-root`, `--dataset`, arms, folds,
seeds and the `best` / `epoch80` checkpoints). Only `--split val` is read by default; `--split test` needs `--test-folds`
and `--test-seeds` (and `--test-arms` to restrict the arms), with the semantics of the export: the plan forbids reading
test outside the pre-registered sets (sections 8.1 and 8.6).

Writes under `--output`:

    top1_runs.csv           one row per evaluation: counts, CorLoc, mask CorLoc, LROC areas, top-1 AUROC, operating points
    top1_lroc_curves.csv    the LROC curves in long format (one point per threshold), enough to redraw them
    top1_differences.csv    per fold and seed, the `export_run_results.DIFFERENCES` pairs on CorLoc@0.5 (primary window),
                            AUC-LROC@0.5 and, for segment arms, the mask CorLoc@0.5
    top1_inference.csv      the same differences averaged over seeds per fold, with the corrected t and the correlated
                            Bayesian ROPE of `scripts/aggregate_kfold.py` (lot-level inference)
    top1_lroc_curves_<split>.png   with `--plot`: LROC@0.5 per arm, mean over folds and seeds

    python scripts/top1_localization.py --runs-root ~/edge_mango/runs/final \\
        --dataset ~/edge_mango/data/v4_captura_v3_final --split val --output reports/top1-localization
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aggregate_kfold import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    check_folds_in_spec,
    load_fold_spec,
    paired_inference,
)
from scripts.export_run_results import (  # noqa: E402
    compute_differences,
    evaluations,
    flatten,
    write_csv,
)
from scripts.val_operating_points import ARMS  # noqa: E402
from src.evaluation import top1_localization as t1  # noqa: E402
from src.evaluation.v4_metrics import (  # noqa: E402
    join_photos,
    read_damage_labels,
    read_damage_polygons,
    read_index,
    read_prediction_masks,
    read_predictions,
)

DIFFERENCE_METRICS = ("corloc_primary_iou50", "lroc_auc_iou50", "mask_corloc_primary_iou50")
DEFAULT_ROPE = 0.02  # as in the AP50 inference of the README; CorLoc and AUC-LROC are on the same 0-1 scale
FPF_GRID = np.linspace(0.0, 1.0, 101)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, required=True, help="<arm>/fold<k>/seed<s>/ and reports*/")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="split to read (default: val); test needs --test-folds and --test-seeds")
    parser.add_argument("--test-folds", type=int, nargs="*", default=[])
    parser.add_argument("--test-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--test-arms", nargs="*", default=None,
                        help="arms whose test results may be read (default: every arm); a test look the plan "
                             "registers for some arms only must name them")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-spec", type=Path, default=None,
                        help="fold assignment file for the inference (default: the dataset's folds_lot/folds_meta.json)")
    parser.add_argument("--rope", type=float, default=DEFAULT_ROPE, help=f"ROPE half-width (default: {DEFAULT_ROPE})")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--plot", action="store_true", help="also draw the LROC curves per arm")
    return parser.parse_args(argv)


def load_photos(directory: Path, labels_dir: Path, index: dict):
    """The evaluated photos of one evaluation; segment ones (masks in the predictions) carry their polygons."""
    predictions = directory / "predictions.jsonl"
    masks = read_prediction_masks(predictions)  # None for a detect evaluation
    return join_photos(
        read_predictions(predictions), index, read_damage_labels(labels_dir),
        pred_masks=masks, gt_polygons=None if masks is None else read_damage_polygons(labels_dir),
    )


def curve_rows(key: dict, curves: dict) -> list[dict]:
    return [
        {**key, "iou": iou, "threshold": float(t), "fpf": float(x), "tpf_loc": float(y)}
        for iou, curve in curves.items()
        for t, x, y in zip(curve.thresholds, curve.fpf, curve.tpf)
    ]


def mean_curve(curves: list[t1.LrocCurve], grid: np.ndarray = FPF_GRID) -> np.ndarray:
    """Mean TPF_loc of several LROC curves on a common FPF grid (linear between points, as the trapezoidal area)."""
    values = []
    for curve in curves:
        fpf = np.unique(curve.fpf)
        tpf = np.array([curve.tpf[curve.fpf == x].max() for x in fpf])  # a vertical rise keeps its top
        values.append(np.interp(grid, fpf, tpf))
    return np.mean(values, axis=0)


def difference_rows(metrics: dict) -> list[dict]:
    """`compute_differences` of the export on the top-1 metrics; `metrics` maps (fold, seed, split, checkpoint, arm) to a row."""
    values = {
        (*key[:4], metric, key[4]): row[metric]
        for key, row in metrics.items() for metric in DIFFERENCE_METRICS if row.get(metric) is not None
    }
    renamed = []
    for row in compute_differences(values):
        row = {("treatment_value" if k == "treatment_ap" else "baseline_value" if k == "baseline_ap" else k): v
               for k, v in row.items()}
        renamed.append(row)
    return renamed


def inference_rows(differences: list[dict], spec: list[dict], rope: float, confidence: float) -> list[dict]:
    """Lot-level inference on the per-fold differences, averaged over the seeds of each fold."""
    groups: dict[tuple, dict[int, list[float]]] = {}
    for row in differences:
        key = (row["split"], row["checkpoint"], row["comparison"], row["metric"], row["treatment"], row["baseline"])
        groups.setdefault(key, {}).setdefault(row["fold"], []).append(row["delta"])
    by_id = {int(entry["fold"]): entry for entry in spec}
    rows = []
    for key, per_fold in sorted(groups.items()):
        folds = sorted(f for f in per_fold if f in by_id)
        deltas = [float(np.mean(per_fold[f])) for f in folds]
        row = dict(zip(("split", "checkpoint", "comparison", "metric", "treatment", "baseline"), key))
        row.update(n_folds=len(folds), mean_delta=float(np.mean(deltas)) if deltas else None)
        flatten("", paired_inference(deltas, [by_id[f] for f in folds], rope, confidence) if deltas else
                {"error": "no fold of the fold spec"}, row)
        rows.append(row)
    return rows


def plot(curves: dict, path: Path, split: str) -> None:
    """LROC@0.5 per arm, mean over folds and seeds, one panel per checkpoint (style of `val_operating_points.plot`)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    checkpoints = sorted({checkpoint for _, checkpoint in curves})
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(4.6 * len(checkpoints), 4.6), squeeze=False, facecolor=surface)
    for ax, checkpoint in zip(axes[0], checkpoints):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for (arm, ckpt), arm_curves in sorted(curves.items()):
            if ckpt == checkpoint:
                color, ls, _ = ARMS.get(arm, (ink2, "-", arm))
                ax.plot(FPF_GRID, mean_curve(arm_curves), color=color, ls=ls, lw=2)
        ax.set_title(f"Checkpoint {checkpoint}", color=ink, fontsize=12, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("FPF (fotos sanas con top-1 ≥ umbral)")
    axes[0][0].set_ylabel("TPF de localización (top-1 correcto, IoU 0.5)")
    present = [a for a in dict.fromkeys(arm for arm, _ in curves)]
    handles = [Line2D([], [], color=ARMS.get(a, (ink2, "-", a))[0], ls=ARMS.get(a, (ink2, "-", a))[1], lw=2,
                      label=ARMS.get(a, (ink2, "-", a))[2]) for a in present]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, labelcolor=ink, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, f"Curvas LROC en {split}, ventana latente + fotos sanas, media de folds y semillas. "
             "El área bajo cada curva (AUC-LROC) nunca supera CorLoc.", color=ink2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.85))
    fig.savefig(path, dpi=200, facecolor=surface)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.split == "test" and not (args.test_folds and args.test_seeds):
        print("ERROR: --split test reads test results: name the pre-registered sets with --test-folds and "
              "--test-seeds (and --test-arms).", file=sys.stderr)
        return 2
    if not (math.isfinite(args.rope) and args.rope >= 0 and 0.0 < args.confidence < 1.0):
        print("ERROR: --rope must be finite and >= 0 and --confidence in (0, 1).", file=sys.stderr)
        return 2
    index = read_index(args.dataset / "index.csv")
    runs, curves_long, metrics, plotted = [], [], {}, {}
    for arm, fold, seed, split, checkpoint, directory in evaluations(args, {}):
        if split != args.split:
            continue
        key = {"arm": arm, "fold": fold, "seed": seed, "split": split, "checkpoint": checkpoint}
        photos = load_photos(directory, args.dataset / "folds_lot" / f"fold_{fold}" / "labels" / split, index)
        row, curves = t1.evaluate(photos)
        runs.append({**key, **row})
        metrics[(fold, seed, split, checkpoint, arm)] = row
        curves_long += curve_rows(key, curves)
        plotted.setdefault((arm, checkpoint), []).append(curves[0.5])
        print(f"{arm} fold{fold} seed{seed} {split} {checkpoint}: {len(photos)} photos", flush=True)
    if not runs:
        print(f"No finished {args.split} evaluation under {args.runs_root}", file=sys.stderr)
        return 1
    differences = difference_rows(metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "top1_runs.csv", runs)
    write_csv(args.output / "top1_lroc_curves.csv", curves_long)
    write_csv(args.output / "top1_differences.csv", differences)
    spec_path = args.fold_spec or args.dataset / "folds_lot" / "folds_meta.json"
    try:
        spec = load_fold_spec(spec_path)
        check_folds_in_spec(spec, sorted({r["fold"] for r in differences}))
        inference = inference_rows(differences, spec, args.rope, args.confidence)
        write_csv(args.output / "top1_inference.csv", inference)
    except ValueError as error:
        print(f"WARNING: no lot-level inference ({error}); the per-fold differences are in top1_differences.csv.",
              file=sys.stderr)
    if args.plot:
        plot(plotted, args.output / f"top1_lroc_curves_{args.split}.png", args.split)
    print(f"{len(runs)} evaluations, {len(differences)} difference rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
