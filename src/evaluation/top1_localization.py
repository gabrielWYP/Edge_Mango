"""Top-1 localization of the damage detector: CorLoc and LROC.

In operation a photo shows at most one damage, and the annotation protocol gives every damaged photo exactly one
damage box. AP looks at every box; these metrics look only at the model's highest-confidence damage box of each photo,
whatever its confidence (damage class only, IoU against the photo's ground-truth damage box).

    top-1 box        the box with the highest score of the photo. A photo with no damage prediction has score
                     `-inf`: it is never above a threshold and never correct (IoU 0).
    CorLoc@IoU       over damaged photos, the fraction whose top-1 box has IoU >= the threshold with the ground truth.
                     Computed on three photo sets: `primary` (damaged photos of `LATENT_HOURS`, the latent ones that
                     enter the primary metric), `developed` (the other post-impact hours) and `all`. A photo without
                     a ground-truth box (never the case under the protocol) leaves every set.
    LROC             localization ROC of the single-lesion task (Swensson 1996) on the primary photos plus `sano`.
                     Sweeping a threshold t over the top-1 scores, FPF(t) is the fraction of `sano` photos whose top-1
                     score is >= t, and TPF_loc(t) the fraction of latent damaged photos whose top-1 score is >= t
                     AND whose top-1 box is correct. AUC-LROC is the trapezoidal area over FPF in [0, 1]; the curve
                     starts at (0, 0) and ends at FPF = 1 at the lowest threshold, where TPF_loc is CorLoc. So
                     AUC-LROC <= CorLoc, with equality when every `sano` photo scores below every correct photo.
    top-1 AUROC      photo-level AUROC from the top-1 score, positives the latent damaged photos and negatives the
                     `sano` ones. It ignores localization. It is the `face_auroc` of `v4_metrics` (the maximum
                     damage confidence of the photo) up to the score of a photo with no prediction: `-inf` here, 0
                     there, which ranks the same because confidences are positive.
    operating point  at a confidence c: FPF, TPF_loc and TPF (the same without the localization requirement).

Segment evaluations (`pred_polygons` set) also get the mask variant of CorLoc: the IoU of the top-1 mask against the
ground-truth polygon, rasterized as in `v4_metrics`. A photo whose ground truth has no polygon is left out of it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.ops import box_iou

from src.data_pipeline.v4_selection import LATENT_HOURS
from src.evaluation.detectability import auroc
from src.evaluation.v4_metrics import EvaluatedPhoto, _rasterize
from src.training.metrics import _mask_iou

IOU_THRESHOLDS = (0.5, 0.3)
CONFIDENCES = (0.25, 0.5)
PHOTO_SETS = ("primary", "developed", "all")
NO_DETECTION = -math.inf


@dataclass(frozen=True)
class Top1:
    """The top-1 damage detection of one photo.

    `iou` is None for a photo without a ground-truth damage box (`sano`); `mask_iou` is None unless the evaluation is
    a segment one and the photo has a ground-truth polygon.
    """

    hour: int | None
    score: float
    iou: float | None
    mask_iou: float | None = None


@dataclass(frozen=True)
class LrocCurve:
    """Points of an LROC curve, from threshold `inf` (0, 0) to `-inf` (1, CorLoc)."""

    thresholds: np.ndarray
    fpf: np.ndarray
    tpf: np.ndarray


# --------------------------------------------------------------------------
# Top-1 extraction and CorLoc
# --------------------------------------------------------------------------


def top1_record(photo: EvaluatedPhoto) -> Top1:
    """The top-1 damage detection of `photo` and its overlap with the ground truth."""
    count = len(photo.pred_scores)
    top = int(torch.argmax(photo.pred_scores)) if count else None  # first one on a tie
    score = float(photo.pred_scores[top]) if count else NO_DETECTION
    iou = None
    if len(photo.gt_boxes):
        iou = float(box_iou(photo.pred_boxes[top : top + 1], photo.gt_boxes).max()) if count else 0.0
    mask_iou = None
    if photo.pred_polygons is not None and photo.gt_polygons:
        mask_iou = 0.0
        if count and top < len(photo.pred_polygons) and len(photo.pred_polygons[top]) >= 3:
            overlap = _mask_iou(_rasterize([photo.pred_polygons[top]]), _rasterize(photo.gt_polygons))
            mask_iou = float(overlap.max())
    return Top1(hour=photo.hour, score=score, iou=iou, mask_iou=mask_iou)


def photo_sets(records: Sequence[Top1]) -> dict[str, list[Top1]]:
    """The damaged photos (with a ground-truth box) of the primary window, of the developed hours and of all hours."""
    damaged = [r for r in records if r.hour is not None and r.iou is not None]
    return {
        "primary": [r for r in damaged if r.hour in LATENT_HOURS],
        "developed": [r for r in damaged if r.hour not in LATENT_HOURS],
        "all": damaged,
    }


def corloc(records: Sequence[Top1], iou_threshold: float, mask: bool = False) -> float | None:
    """Fraction of `records` whose top-1 box (or top-1 mask, `mask=True`) has IoU >= `iou_threshold`; None if empty."""
    values = [r.mask_iou if mask else r.iou for r in records]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(v >= iou_threshold for v in values) / len(values)


# --------------------------------------------------------------------------
# LROC
# --------------------------------------------------------------------------


def lroc_curve(positives: Sequence[Top1], negatives: Sequence[Top1], iou_threshold: float) -> LrocCurve:
    """LROC of the latent damaged photos (`positives`) against the `sano` ones (`negatives`).

    One point per distinct top-1 score, plus the start (threshold `inf`) and the end (`-inf`, FPF = 1). With either
    group empty the curve is undefined and `auc_lroc` gives None.
    """
    pos = np.array([r.score for r in positives], dtype=float)
    neg = np.array([r.score for r in negatives], dtype=float)
    correct = np.array([r.iou is not None and r.iou >= iou_threshold for r in positives], dtype=bool)
    scores = np.unique(np.concatenate([pos, neg, [NO_DETECTION]]))[::-1]
    thresholds = np.concatenate([[math.inf], scores])
    fpf = np.array([(neg >= t).mean() if neg.size else np.nan for t in thresholds])
    tpf = np.array([((pos >= t) & correct).mean() if pos.size else np.nan for t in thresholds])
    return LrocCurve(thresholds=thresholds, fpf=fpf, tpf=tpf)


def auc_lroc(curve: LrocCurve) -> float | None:
    """Trapezoidal area under the LROC curve over FPF in [0, 1]; None when either group was empty."""
    if np.isnan(curve.fpf).any() or np.isnan(curve.tpf).any():
        return None
    return float(np.sum(np.diff(curve.fpf) * (curve.tpf[1:] + curve.tpf[:-1]) / 2))


def top1_auroc(records: Sequence[Top1]) -> float | None:
    """Photo-level AUROC of the top-1 score, latent damaged photos against `sano` ones; None if a group is empty."""
    positives = [r.score for r in records if r.hour in LATENT_HOURS and r.iou is not None]
    negatives = [r.score for r in records if r.hour is None]
    return auroc(positives, negatives)


def operating_point(positives: Sequence[Top1], negatives: Sequence[Top1], confidence: float, iou_threshold: float) -> dict:
    """FPF, TPF_loc and TPF at `confidence`: the top-1 boxes with score >= `confidence` count as detections."""
    fpf = sum(r.score >= confidence for r in negatives) / len(negatives) if negatives else None
    if not positives:
        return {"fpf": fpf, "tpf_loc": None, "tpf": None}
    found = [r for r in positives if r.score >= confidence]
    return {
        "fpf": fpf,
        "tpf_loc": sum(r.iou is not None and r.iou >= iou_threshold for r in found) / len(positives),
        "tpf": len(found) / len(positives),
    }


# --------------------------------------------------------------------------
# One evaluation
# --------------------------------------------------------------------------


def _label(threshold: float) -> str:
    return f"iou{round(threshold * 100):02d}"


def evaluate(photos: Sequence[EvaluatedPhoto]) -> tuple[dict, dict[float, LrocCurve]]:
    """All the top-1 metrics of one evaluation, as a flat row, and its LROC curve per IoU threshold."""
    records = [top1_record(photo) for photo in photos]
    sets = photo_sets(records)
    positives = sets["primary"]
    negatives = [r for r in records if r.hour is None]
    segment = bool(photos) and all(photo.pred_polygons is not None for photo in photos)
    row: dict = {f"n_{name}": len(sets[name]) for name in PHOTO_SETS}
    row["n_sano"] = len(negatives)
    row["n_damaged_without_gt"] = sum(1 for r in records if r.hour is not None and r.iou is None)
    for name in PHOTO_SETS:
        for thr in IOU_THRESHOLDS:
            row[f"corloc_{name}_{_label(thr)}"] = corloc(sets[name], thr)
    if segment:
        for name in PHOTO_SETS:
            for thr in IOU_THRESHOLDS:
                row[f"mask_corloc_{name}_{_label(thr)}"] = corloc(sets[name], thr, mask=True)
    curves = {thr: lroc_curve(positives, negatives, thr) for thr in IOU_THRESHOLDS}
    for thr in IOU_THRESHOLDS:
        row[f"lroc_auc_{_label(thr)}"] = auc_lroc(curves[thr])
    row["auroc_top1"] = top1_auroc(records)
    for conf in CONFIDENCES:
        for thr in IOU_THRESHOLDS:
            point = operating_point(positives, negatives, conf, thr)
            row[f"fpf_at_{conf}"] = point["fpf"]
            row[f"tpf_loc_at_{conf}_{_label(thr)}"] = point["tpf_loc"]
            row[f"tpf_at_{conf}"] = point["tpf"]
    return row, curves
