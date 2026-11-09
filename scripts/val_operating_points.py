#!/usr/bin/env python3
"""Precision-recall curves and operating points of the damage class, from the saved predictions of each evaluation.

Reads the `predictions.jsonl` that `scripts/run_small_arms.py` writes next to each `metrics_v4.json` (boxes decoded at
`eval_conf_threshold`, 0.001), so any confidence threshold can be applied afterwards without retraining. The photos
are those of the primary metric: the latent window (ronda_h 0-6) plus the `sano` photos, whose damage boxes are
false positives (`src/evaluation/v4_metrics.py`).

Two levels:

- **box** (detection): a damage box is a true positive when it overlaps an unmatched damage label with IoU >= 0.5,
  matched greedily by confidence inside each photo, the rule of `_compute_operating_point` in
  `src/training/metrics.py`. The precision-recall curve sweeps the threshold; its all-point AP is checked against
  `latent_ap50` of `metrics_v4.json`.
- **photo** (classification, "does this photo show damage?"): positive = a latent window photo, negative = a `sano`
  photo, score = the highest damage confidence on the photo (0 with none), as in `face_auroc`. A photo is called
  damaged when its score reaches the threshold. Positives outnumber negatives about 4 to 1, so accuracy is
  reported next to balanced accuracy and specificity.

Writes `operating_points_<split>.csv` (one row per run, level and threshold), `operating_points_<split>.md`
(seed-mean table) and `pr_curves_<split>.png` under `--output`.

The plan forbids reading test results before the reported seed set is complete (section 8.1): use `--split val`
until then.

    python scripts/val_operating_points.py --reports-root ~/edge_mango/runs/final/reports \\
        --dataset ~/edge_mango/data/v4_captura_v3_final --split val --output reports/final-runs-progress
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.v4_selection import LATENT_HOURS  # noqa: E402
from src.evaluation.v4_metrics import (  # noqa: E402
    AP50_IOU,
    EvaluatedPhoto,
    join_photos,
    read_damage_labels,
    read_index,
    read_predictions,
)

ARMS = {  # arm: (colour, line style, label). One solid colour per arm, as in plot_run_curves.py.
    "est_rgb": ("#2a78d6", "-", "est_rgb (RGB)"),
    "est_rgbnir": ("#eb6834", "-", "est_rgbnir (RGB+NIR)"),
    "kd_rgb": ("#1baf7a", "-", "kd_rgb (RGB, maestro RGB)"),
    "kd_rgbnir": ("#e1c300", "-", "kd_rgbnir (RGB, maestro RGB+NIR)"),
    "est_nir": ("#3d3c39", "-", "est_nir (NIR, referencia)"),
    # CrossKD arms, outside the plan; dashed so they never pass for a planned arm.
    "crosskd_rgb": ("#1baf7a", "--", "crosskd_rgb (RGB, maestro RGB)"),
    "crosskd_rgbnir": ("#e1c300", "--", "crosskd_rgbnir (RGB, maestro RGB+NIR)"),
    "crosskd_rgbnir_high": ("#8a3ffc", "--", "crosskd_rgbnir_high (celdas con alta dependencia de NIR)"),
    "crosskd_rgbnir_low": ("#a56eff", ":", "crosskd_rgbnir_low (celdas con baja dependencia de NIR)"),
    "crosskd_rgbnir_better": ("#d12771", "--", "crosskd_rgbnir_better (celdas donde el maestro supera al alumno)"),
}
RUN_DIR = re.compile(r"fold(\d+)_seed(\d+)")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports-root", type=Path, required=True, help="<arm>/fold<k>_seed<s>/<split>/")
    parser.add_argument("--dataset", type=Path, required=True, help="v4 dataset with index.csv and folds_lot/")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="only these folds (default: all found)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="only these seeds (default: all found)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=1)
    area = (box[2:] - box[:2]).prod()
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1)
    return inter / (area + areas - inter)


def ranked_boxes(photos: list[EvaluatedPhoto], iou_threshold: float = AP50_IOU) -> tuple[list[tuple[float, bool]], int]:
    """(score, is true positive) of every damage box, best score first, and the number of damage labels."""
    ranked, n_gt = [], 0
    for photo in photos:
        n_gt += len(photo.gt_boxes)
        matched: set[int] = set()
        for i in torch.argsort(photo.pred_scores, descending=True).tolist():
            hit = False
            if len(photo.gt_boxes):
                ious = _iou(photo.pred_boxes[i], photo.gt_boxes)
                best_iou, best = ious.max(dim=0)
                if best_iou >= iou_threshold and int(best) not in matched:
                    hit = True
                    matched.add(int(best))
            ranked.append((float(photo.pred_scores[i]), hit))
    ranked.sort(key=lambda item: -item[0])
    return ranked, n_gt


def pr_curve(ranked: list[tuple[float, bool]], n_gt: int) -> tuple[list[float], list[float], list[float]]:
    """Scores, precision and recall after each box of `ranked`."""
    scores, precision, recall, tp = [], [], [], 0
    for k, (score, hit) in enumerate(ranked, 1):
        tp += hit
        scores.append(score)
        precision.append(tp / k)
        recall.append(tp / n_gt if n_gt else 0.0)
    return scores, precision, recall


def all_point_ap(precision: list[float], recall: list[float]) -> float:
    """Area under the interpolated precision envelope (VOC2010+), as `_compute_ap_at_iou`."""
    ap, prev_recall = 0.0, 0.0
    envelope = precision[:]
    for i in range(len(envelope) - 2, -1, -1):
        envelope[i] = max(envelope[i], envelope[i + 1])
    for p, r in zip(envelope, recall):
        ap += (r - prev_recall) * p
        prev_recall = r
    return ap


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def box_point(ranked: list[tuple[float, bool]], n_gt: int, n_photos: int, threshold: float) -> dict:
    kept = [hit for score, hit in ranked if score >= threshold]
    tp = sum(kept)
    fp = len(kept) - tp
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, n_gt)
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"tp": tp, "fp": fp, "fn": n_gt - tp, "precision": precision, "recall": recall, "f1": f1,
            "fp_per_photo": fp / n_photos}


def photo_point(photos: list[EvaluatedPhoto], threshold: float) -> dict:
    positive = [p.face_score >= threshold for p in photos if p.hour in LATENT_HOURS]
    negative = [p.face_score >= threshold for p in photos if p.hour is None]
    tp, fn = sum(positive), len(positive) - sum(positive)
    fp, tn = sum(negative), len(negative) - sum(negative)
    precision, recall, specificity = _ratio(tp, tp + fp), _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1,
            "specificity": specificity, "accuracy": _ratio(tp + tn, tp + tn + fp + fn), "balanced_accuracy": balanced}


def latent_photos(run_split: Path, dataset: Path, fold: int, split: str, index: dict) -> list[EvaluatedPhoto]:
    photos = join_photos(
        read_predictions(run_split / "predictions.jsonl"),
        index,
        read_damage_labels(dataset / "folds_lot" / f"fold_{fold}" / "labels" / split),
    )
    return [p for p in photos if p.hour is None or p.hour in LATENT_HOURS]


def main(argv=None) -> int:
    args = parse_args(argv)
    index = read_index(args.dataset / "index.csv")
    rows, curves = [], {}
    for arm in ARMS:
        for run in sorted((args.reports_root / arm).glob("fold*_seed*")):
            split_dir = run / args.split
            match = RUN_DIR.fullmatch(run.name)
            if not match or not (split_dir / "metrics_v4.json").exists():
                continue  # evaluation not finished
            fold, seed = int(match[1]), int(match[2])
            if (args.folds and fold not in args.folds) or (args.seeds and seed not in args.seeds):
                continue
            photos = latent_photos(split_dir, args.dataset, fold, args.split, index)
            ranked, n_gt = ranked_boxes(photos)
            scores, precision, recall = pr_curve(ranked, n_gt)
            curves[(arm, fold, seed)] = (scores, precision, recall)
            base = {"arm": arm, "fold": fold, "seed": seed, "ap50_check": round(all_point_ap(precision, recall), 4)}
            for t in args.thresholds:
                rows.append({**base, "level": "box", "threshold": t, **box_point(ranked, n_gt, len(photos), t)})
                rows.append({**base, "level": "photo", "threshold": t, **photo_point(photos, t)})
    if not rows:
        print(f"No finished {args.split} evaluation under {args.reports_root}", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (args.output / f"operating_points_{args.split}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    write_table(rows, args.output / f"operating_points_{args.split}.md", args.split)
    plot(curves, args.thresholds, args.output / f"pr_curves_{args.split}.png", args.split)
    print((args.output / f"operating_points_{args.split}.md").read_text())
    return 0


def write_table(rows: list[dict], path: Path, split: str) -> None:
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    lines = [f"# Puntos de operación del daño ({split})", ""]
    groups = {}
    for r in rows:
        groups.setdefault((r["level"], r["fold"], r["arm"], r["threshold"]), []).append(r)
    order = lambda kv: (kv[0][1], list(ARMS).index(kv[0][2]), kv[0][3])  # noqa: E731
    mean = lambda rs, k: statistics.mean(r[k] for r in rs) if all(r[k] is not None for r in rs) else None  # noqa: E731
    lines += [
        "## Cajas (detección, IoU 0.5)", "",
        "| Fold | Brazo | Umbral | Semillas | Precisión | Recall | F1 | FP por foto |",
        "| :-- | :-- | --: | --: | --: | --: | --: | --: |",
    ]
    for (level, fold, arm, t), rs in sorted(groups.items(), key=order):
        if level == "box":
            lines.append(f"| {fold} | {arm} | {t:g} | {len(rs)} | {fmt(mean(rs, 'precision'))} | "
                         f"{fmt(mean(rs, 'recall'))} | {fmt(mean(rs, 'f1'))} | {fmt(mean(rs, 'fp_per_photo'))} |")
    lines += [
        "", "## Fotos (clasificación: ¿la foto muestra daño?)", "",
        "| Fold | Brazo | Umbral | Semillas | Recall (sensib.) | Especificidad | Precisión | F1 | Exactitud | "
        "Exactitud balanceada |",
        "| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for (level, fold, arm, t), rs in sorted(groups.items(), key=order):
        if level == "photo":
            lines.append(
                f"| {fold} | {arm} | {t:g} | {len(rs)} | {fmt(mean(rs, 'recall'))} | {fmt(mean(rs, 'specificity'))} | "
                f"{fmt(mean(rs, 'precision'))} | {fmt(mean(rs, 'f1'))} | {fmt(mean(rs, 'accuracy'))} | "
                f"{fmt(mean(rs, 'balanced_accuracy'))} |"
            )
    path.write_text("\n".join(lines) + "\n")


def plot(curves: dict, thresholds: list[float], path: Path, split: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    markers = ["o", "s", "^", "D"]
    folds = sorted({fold for _, fold, _ in curves})
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, len(folds), figsize=(4.6 * len(folds), 4.6), squeeze=False, facecolor=surface)
    for ax, fold in zip(axes[0], folds):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for (arm, f, seed), (scores, precision, recall) in sorted(curves.items()):
            if f != fold:
                continue
            color, ls, _ = ARMS[arm]
            ax.plot(recall, precision, color=color, ls=ls, lw=2)
            for t, marker in zip(thresholds, markers):
                k = sum(s >= t for s in scores)
                if k:
                    ax.plot([recall[k - 1]], [precision[k - 1]], marker, ms=8, color=color, mec=surface, mew=2)
        ax.set_title(f"Fold {fold}", color=ink, fontsize=12, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall (cajas de daño, IoU 0.5)")
    axes[0][0].set_ylabel("Precisión")
    present = [a for a in ARMS if any(arm == a for arm, _, _ in curves)]
    handles = [Line2D([], [], color=ARMS[a][0], ls=ARMS[a][1], lw=2, label=ARMS[a][2]) for a in present]
    handles += [Line2D([], [], color=ink2, marker=m, ls="", ms=7, label=f"conf ≥ {t:g}")
                for t, m in zip(thresholds, markers)]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, labelcolor=ink, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, f"Curvas precisión-recall del daño en {split}, ventana latente + fotos sanas, una curva por "
             "corrida. Los marcadores son los umbrales de confianza.", color=ink2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.85))
    fig.savefig(path, dpi=200, facecolor=surface)


if __name__ == "__main__":
    raise SystemExit(main())
