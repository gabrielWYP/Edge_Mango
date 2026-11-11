"""Stratified damage metrics of one evaluation (docs/tesis/plan-experimental.md, section 7).

Joins the per-image predictions of `scripts/evaluate_checkpoint.py --save-predictions`, the YOLO
label files of the evaluated split and the `index.csv` of the v4 dataset, and computes:

    latent_ap50         primary: damage AP50 (`ap50_class_1`) on the latent window photos plus the
                        `sano` photos, whose damage predictions are false positives.
    latent_ap30         secondary 1: the same photos at IoU 0.30. "En latente" is the same set as the
                        primary (plan 8.1), so the two differ only in the IoU.
    face_auroc          secondary 2: "does this face have damage?", see below.
    ap50_by_hour        descriptive: photos of each post-impact hour in the index.
    ap50_developed      descriptive: photos of the post-impact hours outside `LATENT_HOURS`.
    ap50_by_zone        descriptive: photos of each impact zone in the index.
    ap50_by_provenance  descriptive: photos of each box provenance in the index.
    sano_fp_per_image   descriptive: damage boxes per `sano` photo at the `conf_threshold` operating
                        point (at the evaluation threshold nearly every box would count).

Mask twins (segment checkpoints only; `predictions.jsonl` then carries a `mask` polygon per damage
detection and the label files hold polygon lines). Same photo sets and ranking as the box metrics,
IoU between masks instead of boxes, rasterized on a `MASK_CANVAS` square of the normalized frame
(IoU is invariant to the axis scaling). Box metrics are never touched by them:

    latent_mask_ap50    the primary's set, mask IoU 0.50.
    latent_mask_ap30    the secondary's, mask IoU 0.30.
    mask_ap50_by_hour   mask AP50 per post-impact hour of the index.
    developed_mask_ap50 mask AP50 on the hours outside `LATENT_HOURS`.
    mask_gt_box_only    damage ground-truth lines without a polygon; they leave the mask ground
                        truth (as in training), so a prediction on one is a false positive.

Face AUROC and the false positives per `sano` photo depend on scores alone and stay as they are.

A `sano` photo has no hour, zone or provenance, so it enters none of the descriptive AP strata:
only the two latent metrics, by the plan's definition, charge its false positives.

A stratum with no ground truth has `"ap": null`, and an AUROC with an empty group `"auroc": null`,
never 0: `_compute_ap_at_iou` would say 0.0 for "no ground truth", which reads as a detector that
found nothing. The strata listed are those of the whole index, not of the evaluated lot, so every
fold reports the same keys.

Per-face AUROC. One photo is one look at one face at one moment: the positives are the latent
window photos, the negatives the `sano` photos, and the score of a photo is the maximum damage
confidence on it, 0 when it has no damage prediction. Pooling a face's latent photos into their
maximum is deliberately not done: a face has up to four latent photos and one `sano` photo, so a
detector with no signal would score P(max of 3 > 1) = 0.75 instead of 0.5. The design is paired
because the two groups come from the same faces; `n_paired_faces` counts those holding both.

AP comes from `src.training.metrics._compute_ap_at_iou`, the function behind `metrics.json`, and
the AUROC from `src.evaluation.detectability.auroc`. Predictions and labels are both normalized to
the source RGB frame (see `scripts/evaluate_checkpoint.py`), and IoU does not depend on the scale.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data_pipeline.v4_selection import LATENT_HOURS, hour_label
from src.evaluation.detectability import auroc
from src.training.config import DAMAGE_CLASS_ID, TrainingConfig
from src.training.metrics import _ap_from_ranked, _compute_ap_at_iou, _cxcywh_to_xyxy, _mask_iou, _rank_class
from src.training.seg_labels import parse_label_line, rasterize_polygons, read_label_boxes

AP50_IOU = 0.50
AP30_IOU = 0.30
DEFAULT_CONF_THRESHOLD = TrainingConfig.conf_threshold
SANO = hour_label(None)
MASK_CANVAS = 512  # side, in pixels, of the square the masks of a photo are rasterized on
REQUIRED_COLUMNS = ("stem", "lot", "mango_id", "cara", "ronda_h", "zona", "provenance")


@dataclass(frozen=True)
class EvaluatedPhoto:
    """One photo of the evaluated split. Boxes are damage boxes, xyxy in the source RGB frame."""

    stem: str
    hour: int | None  # None for a `sano` photo
    zona: str
    provenance: str
    face: tuple[str, str, str]  # (lot, mango_id, cara)
    gt_boxes: torch.Tensor
    pred_boxes: torch.Tensor
    pred_scores: torch.Tensor
    # Segment only: normalized (n, 2) polygons of the source frame, aligned with `pred_boxes` /
    # `gt_polygons`. `pred_polygons` is None for a detect evaluation.
    pred_polygons: tuple[np.ndarray, ...] | None = None
    gt_polygons: tuple[np.ndarray, ...] = ()
    n_gt_box_only: int = 0

    @property
    def face_score(self) -> float:
        """Maximum damage confidence on the photo; 0 when it has no damage prediction."""
        return float(self.pred_scores.max()) if len(self.pred_scores) else 0.0


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _preview(stems: Sequence[str], limit: int = 5) -> str:
    return ", ".join(stems[:limit]) + (f" and {len(stems) - limit} more" if len(stems) > limit else "")


def _boxes_tensor(boxes: Sequence[Sequence[float]]) -> torch.Tensor:
    return torch.tensor(boxes, dtype=torch.float64).reshape(-1, 4)


def read_predictions(path: str | Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """`{stem: (damage boxes, damage scores)}` of a predictions.jsonl, in file order.

    Raises ValueError, naming the file and the line, for a malformed line or a repeated stem.
    """
    path = Path(path)
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"Cannot read the predictions {path}: {error}") from None
    predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            stem = record["stem"]
            parsed = [(int(b["class"]), float(b["confidence"]), [float(v) for v in b["xyxy"]]) for b in record["boxes"]]
            if any(len(xyxy) != 4 for _, _, xyxy in parsed):
                raise ValueError("xyxy needs 4 values")
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}:{number}: malformed prediction line ({type(error).__name__}: {error}).") from None
        if stem in predictions:
            raise ValueError(f"{path}:{number}: stem {stem} appears twice.")
        damage = [(score, xyxy) for cls, score, xyxy in parsed if cls == DAMAGE_CLASS_ID]
        predictions[stem] = (
            _boxes_tensor([xyxy for _, xyxy in damage]),
            torch.tensor([score for score, _ in damage], dtype=torch.float64),
        )
    return predictions


def read_prediction_masks(path: str | Path) -> dict[str, list[np.ndarray]] | None:
    """`{stem: damage masks}` of a segment predictions.jsonl, None for a detect one.

    A mask is the `mask` field of a damage box, a flat normalized `[x1, y1, x2, y2, ...]` polygon of
    the source frame (empty when the mask was); the list is aligned with `read_predictions`'s damage
    boxes. A file is a segment one when any box has a `mask`; then every damage box must have it.
    Raises ValueError, naming the file and the line, otherwise.
    """
    path = Path(path)
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"Cannot read the predictions {path}: {error}") from None
    records = []
    for number, line in enumerate(lines, start=1):
        if line.strip():
            try:
                records.append((number, json.loads(line)))
            except ValueError as error:
                raise ValueError(f"{path}:{number}: malformed prediction line ({error}).") from None
    if not any("mask" in box for _, record in records for box in record.get("boxes", ())):
        return None
    masks: dict[str, list[np.ndarray]] = {}
    for number, record in records:
        polygons = []
        for box in record["boxes"]:
            if int(box["class"]) != DAMAGE_CLASS_ID:
                continue
            flat = box.get("mask")
            if flat is None or len(flat) % 2:
                raise ValueError(f"{path}:{number}: a damage box has no valid mask (flat x, y pairs).")
            polygons.append(np.asarray(flat, dtype=np.float64).reshape(-1, 2))
        masks[record["stem"]] = polygons
    return masks


def read_index(path: str | Path) -> dict[str, dict[str, str]]:
    """The rows of a v4 `index.csv` by stem; raises ValueError for a missing column or a repeated stem."""
    path = Path(path)
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"The index {path} has no column {', '.join(missing)}.")
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"Cannot read the index {path}: {error}") from None
    index = {row["stem"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"The index {path} repeats a stem.")
    return index


def read_damage_labels(labels_dir: str | Path) -> dict[str, torch.Tensor]:
    """`{stem: damage boxes}` of a YOLO label directory, as xyxy in the frame the labels use."""
    labels_dir = Path(labels_dir)
    files = sorted(labels_dir.glob("*.txt"))
    if not files:
        raise ValueError(f"No label files in {labels_dir}.")
    labels = {}
    for path in files:
        all_boxes, classes = read_label_boxes(path)  # box and polygon lines alike
        boxes = all_boxes[classes == DAMAGE_CLASS_ID].astype("float64").tolist()
        labels[path.stem] = _cxcywh_to_xyxy(_boxes_tensor(boxes))
    return labels


def read_damage_polygons(labels_dir: str | Path) -> dict[str, tuple[list[np.ndarray], int]]:
    """`{stem: (damage polygons, number of damage lines without one)}` of a YOLO-seg label directory."""
    polygons: dict[str, tuple[list[np.ndarray], int]] = {}
    for path in sorted(Path(labels_dir).glob("*.txt")):
        found, box_only = [], 0
        for line in path.read_text().splitlines():
            parsed = parse_label_line(line.split())
            if parsed is None or parsed[0] != DAMAGE_CLASS_ID:
                continue
            if parsed[2] is None:
                box_only += 1
            else:
                found.append(parsed[2])
        polygons[path.stem] = (found, box_only)
    return polygons


def _hour(row: Mapping[str, str]) -> int | None:
    """The `ronda_h` of an index row: None for `sano`, else the post-impact hour."""
    value = row["ronda_h"]
    if value == SANO:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"The index row of {row['stem']} has ronda_h {value!r}.") from None


def join_photos(
    predictions: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    index: Mapping[str, Mapping[str, str]],
    labels: Mapping[str, torch.Tensor],
    pred_masks: Mapping[str, Sequence[np.ndarray]] | None = None,
    gt_polygons: Mapping[str, tuple[Sequence[np.ndarray], int]] | None = None,
) -> list[EvaluatedPhoto]:
    """The evaluated photos, in prediction order.

    `pred_masks` and `gt_polygons` (segment evaluations) fill the photos' mask fields; a photo
    without an entry has none.

    The predictions and the label files must hold the same stems: a photo silently missing from
    one side would change the metric. Index rows outside them are the other lots of the dataset.
    Raises ValueError naming the stems that break this.
    """
    no_label = [s for s in predictions if s not in labels]
    if no_label:
        raise ValueError(f"Predictions without a label file: {_preview(no_label)}.")
    no_prediction = [s for s in labels if s not in predictions]
    if no_prediction:
        raise ValueError(f"Label files without a prediction line: {_preview(no_prediction)}.")
    not_indexed = [s for s in predictions if s not in index]
    if not_indexed:
        raise ValueError(f"Photos missing from the index: {_preview(not_indexed)}.")
    photos = []
    for stem, (boxes, scores) in predictions.items():
        row = index[stem]
        photos.append(
            EvaluatedPhoto(
                stem=stem,
                hour=_hour(row),
                zona=row["zona"],
                provenance=row["provenance"],
                face=(row["lot"], row["mango_id"], row["cara"]),
                gt_boxes=labels[stem],
                pred_boxes=boxes,
                pred_scores=scores,
                pred_polygons=None if pred_masks is None else tuple(pred_masks.get(stem, ())),
                gt_polygons=tuple((gt_polygons or {}).get(stem, ((), 0))[0]),
                n_gt_box_only=(gt_polygons or {}).get(stem, ((), 0))[1],
            )
        )
    return photos


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def damage_ap(photos: Sequence[EvaluatedPhoto], iou_threshold: float) -> dict:
    """`{"ap", "n_gt", "n_images"}` of the damage class on `photos`; `ap` is None with no ground truth."""
    n_gt = sum(len(photo.gt_boxes) for photo in photos)
    ap = None
    if n_gt:
        _, per_class = _compute_ap_at_iou(
            [photo.pred_boxes for photo in photos],
            [photo.pred_scores for photo in photos],
            [torch.full((len(photo.pred_scores),), DAMAGE_CLASS_ID) for photo in photos],
            [photo.gt_boxes for photo in photos],
            [torch.full((len(photo.gt_boxes),), DAMAGE_CLASS_ID) for photo in photos],
            iou_threshold,
            DAMAGE_CLASS_ID + 1,
        )
        ap = per_class[DAMAGE_CLASS_ID]
    return {"ap": ap, "n_gt": n_gt, "n_images": len(photos)}


class _MaskOverlap:
    """Mask IoU of the photos, rasterized once and shared by every stratum and threshold.

    `_rank_class` hands its IoU function the per-image "boxes" it was given, so each mask is
    represented by a `[photo, instance]` row and the IoU matrix of a photo is computed on first use.
    """

    def __init__(self, photos: Sequence[EvaluatedPhoto]):
        self._position = {photo.stem: position for position, photo in enumerate(photos)}
        self._photos = list(photos)
        self._cache: dict[int, torch.Tensor] = {}

    def handles(self, stem: str, count: int) -> torch.Tensor:
        position = self._position[stem]
        return torch.stack([torch.full((count,), position), torch.arange(count)], dim=1).double()

    def iou(self, preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
        position = int(preds[0, 0])
        if position not in self._cache:
            photo = self._photos[position]
            self._cache[position] = _mask_iou(_rasterize(photo.pred_polygons), _rasterize(photo.gt_polygons))
        return self._cache[position][preds[:, 1].long()][:, gts[:, 1].long()]


def _rasterize(polygons: Sequence[np.ndarray]) -> torch.Tensor:
    """(n, MASK_CANVAS ** 2) flat binary masks of normalized polygons."""
    pixels = [polygon * MASK_CANVAS - 0.5 for polygon in polygons]
    return torch.from_numpy(rasterize_polygons(pixels, (MASK_CANVAS, MASK_CANVAS))).reshape(len(pixels), -1)


def damage_mask_ap(photos: Sequence[EvaluatedPhoto], iou_threshold: float, overlap: _MaskOverlap) -> dict:
    """`damage_ap` with mask IoU over the polygon ground truth; `n_gt` counts polygons only."""
    n_gt = sum(len(photo.gt_polygons) for photo in photos)
    ap = None
    if n_gt:
        damage = torch.tensor(DAMAGE_CLASS_ID)
        ranked = {DAMAGE_CLASS_ID: _rank_class(
            [overlap.handles(p.stem, len(p.pred_polygons)) for p in photos],
            [p.pred_scores for p in photos],
            [damage.repeat(len(p.pred_scores)) for p in photos],
            [overlap.handles(p.stem, len(p.gt_polygons)) for p in photos],
            [damage.repeat(len(p.gt_polygons)) for p in photos],
            DAMAGE_CLASS_ID,
            overlap.iou,
        )}
        ap = _ap_from_ranked(ranked, iou_threshold)[1][DAMAGE_CLASS_ID]
    return {"ap": ap, "n_gt": n_gt, "n_images": len(photos)}


def face_auroc(photos: Sequence[EvaluatedPhoto]) -> dict:
    """Per-face AUROC (module docstring). `auroc` is None when either group is empty."""
    positives = [photo for photo in photos if photo.hour in LATENT_HOURS]
    negatives = [photo for photo in photos if photo.hour is None]
    return {
        "auroc": auroc([p.face_score for p in positives], [p.face_score for p in negatives]),
        "n_negative": len(negatives),
        "n_negative_no_detection": sum(1 for p in negatives if not len(p.pred_scores)),
        "n_paired_faces": len({p.face for p in positives} & {p.face for p in negatives}),
        "n_positive": len(positives),
        "n_positive_no_detection": sum(1 for p in positives if not len(p.pred_scores)),
    }


def sano_fp_per_image(photos: Sequence[EvaluatedPhoto], conf_threshold: float) -> dict:
    """Damage boxes at `conf_threshold` per `sano` photo; `fp_per_image` is None with no such photo."""
    sano = [photo for photo in photos if photo.hour is None]
    n_fp = sum(int((photo.pred_scores >= conf_threshold).sum()) for photo in sano)
    return {
        "conf_threshold": conf_threshold,
        "fp_per_image": n_fp / len(sano) if sano else None,
        "n_fp": n_fp,
        "n_images": len(sano),
    }


def compute_metrics(
    photos: Sequence[EvaluatedPhoto],
    index: Mapping[str, Mapping[str, str]],
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> dict:
    """The content of `metrics_v4.json` (module docstring)."""
    rows = list(index.values())
    hours = sorted({h for h in map(_hour, rows) if h is not None} | set(LATENT_HOURS))
    developed = [hour for hour in hours if hour not in LATENT_HOURS]
    zones = sorted({row["zona"] for row in rows if row["zona"]})
    provenances = sorted({row["provenance"] for row in rows if row["provenance"]})
    latent_set = [photo for photo in photos if photo.hour is None or photo.hour in LATENT_HOURS]
    mask_metrics = {}
    if photos and all(photo.pred_polygons is not None for photo in photos):
        overlap = _MaskOverlap(photos)
        mask_metrics = {
            "developed_mask_ap50": damage_mask_ap([p for p in photos if p.hour in developed], AP50_IOU, overlap),
            "latent_mask_ap30": damage_mask_ap(latent_set, AP30_IOU, overlap),
            "latent_mask_ap50": damage_mask_ap(latent_set, AP50_IOU, overlap),
            "mask_ap50_by_hour": {
                str(hour): damage_mask_ap([p for p in photos if p.hour == hour], AP50_IOU, overlap) for hour in hours
            },
            "mask_gt_box_only": sum(photo.n_gt_box_only for photo in photos),
        }
    return {
        **mask_metrics,
        "ap50_by_hour": {
            str(hour): damage_ap([p for p in photos if p.hour == hour], AP50_IOU) for hour in hours
        },
        "ap50_by_provenance": {
            value: damage_ap([p for p in photos if p.provenance == value], AP50_IOU) for value in provenances
        },
        "ap50_by_zone": {
            value: damage_ap([p for p in photos if p.zona == value], AP50_IOU) for value in zones
        },
        "ap50_developed": damage_ap([p for p in photos if p.hour in developed], AP50_IOU),
        "damage_class_id": DAMAGE_CLASS_ID,
        "developed_hours": developed,
        "face_auroc": face_auroc(photos),
        "latent_ap30": damage_ap(latent_set, AP30_IOU),
        "latent_ap50": damage_ap(latent_set, AP50_IOU),
        "latent_hours": sorted(LATENT_HOURS),
        "num_images": len(photos),
        "sano_fp_per_image": sano_fp_per_image(photos, conf_threshold),
    }
