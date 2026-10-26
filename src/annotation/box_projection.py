"""Project hand-drawn +12 h damage boxes backwards onto the earlier photos of the same fruit face.

The bruise grows, so a box drawn at +12 h is an upper bound of the damage at earlier moments.
For every face with a +12 h box and every earlier moment, the +12 h photo is registered to the
earlier one by independent methods (see ``mango_registration``), the box corners are carried
across, and each projection gets a confidence tier:

* **A**: two valid methods whose projected boxes agree (IoU >= 0.8 or corner displacement
  < 10% of the box side).
* **B**: exactly one valid method.
* **C**: no valid method (or the methods contradict each other, or the box leaves the frame):
  no projection.

Boxes drawn on NIR are registered on NIR; for those, the RGB registration of the same photos is
optionally carried over through a fixed RGB-to-NIR map (``Bridge``) as two more estimates,
because skin texture is far more visible in RGB.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .captura_v3_tasks import (
    Capture,
    Dropped,
    Moment,
    Slot,
    describe_slot,
    load_captures,
    moment_label,
    resolve_slots,
)
from .mango_registration import Similarity

Box = tuple[float, float, float, float]

MOMENTS: tuple[Moment, ...] = (None, 0, 2, 4, 6, 8, 10, 12)
EARLIER: tuple[Moment, ...] = (10, 8, 6, 4, 2, 0, None)
REFERENCE_HOUR = 12
VALIDATION_HOUR = 10

AGREE_IOU = 0.8
AGREE_DISPLACEMENT = 0.10
MIN_VISIBLE = 0.5
SCORE = {"A": 0.9, "B": 0.5}

# modality -> (Label Studio control name, image tag name)
CONTROLS = {"nir": ("damage_nir", "nir"), "rgb": ("damage_rgb", "rgb")}
LABEL = "damage"
FROM_NAME_MODALITY = {name: modality for modality, (name, _) in CONTROLS.items()}


class Face(NamedTuple):
    jornada: str
    mango_id: int
    cara: int


def moment_key(moment: Moment) -> int:
    return -1 if moment is None else moment


def moment_text(moment: Moment) -> str:
    return "sano" if moment is None else str(moment)


# ---------------------------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------------------------


def percent_to_box(value: dict, width: int, height: int) -> Box:
    """Label Studio percent rectangle -> ``(x, y, w, h)`` in pixels of the original image."""
    return (
        value["x"] / 100 * width,
        value["y"] / 100 * height,
        value["width"] / 100 * width,
        value["height"] / 100 * height,
    )


def box_to_percent(box: Box, width: int, height: int) -> dict:
    x, y, w, h = box
    return {
        "x": x / width * 100,
        "y": y / height * 100,
        "width": w / width * 100,
        "height": h / height * 100,
    }


def box_corners(box: Box) -> np.ndarray:
    x, y, w, h = box
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class Projection:
    quad: np.ndarray
    box: Box | None
    visible: float


def project_box(box: Box, transform, width: int, height: int) -> Projection:
    """Carry the corners across and take the axis-aligned bounds, clipped to the frame."""
    quad = transform.apply(box_corners(box))
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    area = max(x1 - x0, 1e-9) * max(y1 - y0, 1e-9)
    cx0, cy0, cx1, cy1 = max(x0, 0.0), max(y0, 0.0), min(x1, float(width)), min(y1, float(height))
    if cx1 <= cx0 or cy1 <= cy0:
        return Projection(quad, None, 0.0)
    return Projection(
        quad, (cx0, cy0, cx1 - cx0, cy1 - cy0), float((cx1 - cx0) * (cy1 - cy0) / area)
    )


@dataclass(frozen=True)
class Agreement:
    iou: float
    displacement: float
    agree: bool


def agreement(box: Box, a, b, width: int, height: int) -> Agreement:
    """Do two transforms put the box in the same place? Displacement is a fraction of the box side."""
    pa, pb = project_box(box, a, width, height), project_box(box, b, width, height)
    iou = box_iou(pa.box, pb.box) if pa.box and pb.box else 0.0
    side = math.sqrt(box[2] * box[3])
    displacement = float(np.linalg.norm(pa.quad - pb.quad, axis=1).max() / side)
    return Agreement(iou, displacement, iou >= AGREE_IOU or displacement < AGREE_DISPLACEMENT)


# ---------------------------------------------------------------------------------------------
# Estimates and decisions
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """One method's transform of the box frame ``+12 h -> earlier moment``."""

    method: str
    sim: object | None
    valid: bool
    metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    tier: str
    box: Box | None
    methods: tuple[str, ...]
    agreement_iou: float | None
    agreement_displacement: float | None
    reason: str


def _mean_box(boxes: list[Box]) -> Box:
    return tuple(float(v) for v in np.mean(boxes, axis=0))  # type: ignore[return-value]


def decide(estimates: list[Estimate], box: Box, width: int, height: int) -> Decision:
    """Tier A/B/C of one projection from the estimates of every method, in order of precedence."""
    valid = [e for e in estimates if e.valid and e.sim is not None]
    if not valid:
        return Decision("C", None, (), None, None, "no_valid_method")
    projections = {e.method: project_box(box, e.sim, width, height) for e in valid}
    usable = [e for e in valid if projections[e.method].visible >= MIN_VISIBLE]
    if not usable:
        return Decision("C", None, (), None, None, "out_of_frame")
    if len(usable) == 1:
        only = usable[0]
        return Decision("B", projections[only.method].box, (only.method,), None, None, "")
    links = {
        (a.method, b.method): agreement(box, a.sim, b.sim, width, height)
        for a, b in combinations(usable, 2)
    }

    def agrees(a: Estimate, b: Estimate) -> bool:
        key = (a.method, b.method) if (a.method, b.method) in links else (b.method, a.method)
        return links[key].agree

    groups = [[a] + [b for b in usable if b is not a and agrees(a, b)] for a in usable]
    best = max(groups, key=len)
    if len(best) < 2:
        return Decision("C", None, (), None, None, "methods_disagree")
    members = [e for e in usable if e in best]
    first, second = members[0], members[1]
    key = (
        (first.method, second.method)
        if (first.method, second.method) in links
        else (second.method, first.method)
    )
    pair = links[key]
    box_out = _mean_box([projections[e.method].box for e in members])
    return Decision("A", box_out, tuple(e.method for e in members), pair.iou, pair.displacement, "")


# ---------------------------------------------------------------------------------------------
# RGB <-> NIR bridge
# ---------------------------------------------------------------------------------------------


class Bridge:
    """A fixed map from the RGB frame to the NIR frame (homography plus a lot shift).

    The map only matters to first order for the box centre: a motion measured in RGB is carried
    into the NIR frame by conjugating it with the map, so a map error of ``d`` pixels moves the
    projected box by about ``d`` times the rotation (in radians) of the fruit, not by ``d``.
    """

    def __init__(self, homography: np.ndarray, shift: tuple[float, float] = (0.0, 0.0)) -> None:
        self.h = np.asarray(homography, dtype=float)
        self.h_inv = np.linalg.inv(self.h)
        self.shift = np.asarray(shift, dtype=float)

    @staticmethod
    def _project(h: np.ndarray, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        homogeneous = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
        return homogeneous[:, :2] / homogeneous[:, 2:3]

    def to_nir(self, points: np.ndarray) -> np.ndarray:
        return self._project(self.h, points) + self.shift

    def from_nir(self, points: np.ndarray) -> np.ndarray:
        return self._project(self.h_inv, np.asarray(points, dtype=float) - self.shift)

    def carry(self, sim: Similarity) -> CarriedTransform:
        return CarriedTransform(self, sim)


@dataclass(frozen=True)
class CarriedTransform:
    """An RGB-frame similarity expressed in the NIR frame."""

    bridge: Bridge
    sim: Similarity

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.bridge.to_nir(self.sim.apply(self.bridge.from_nir(points)))


# ---------------------------------------------------------------------------------------------
# Hand boxes from the Label Studio export
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HandBox:
    face: Face
    ronda_h: Moment
    timestamp: int
    modality: str
    box: Box
    width: int
    height: int


@dataclass
class HandBoxes:
    boxes: list[HandBox]
    without_box: list[Face]
    unannotated: list[Face]
    seen: set[Face] = field(default_factory=set)


def _face_of(data: dict) -> Face:
    return Face(str(data["jornada"]), int(data["mango_id"]), int(data["cara"]))


def load_hand_boxes(export_path: Path, hours: set[Moment] | None = None) -> HandBoxes:
    """Read the boxes of a Label Studio JSON export, keeping only tasks of ``hours`` (all if None).

    A task with several annotations counts through its latest non-cancelled one.
    """
    tasks = json.loads(Path(export_path).read_text(encoding="utf-8"))
    boxes: list[HandBox] = []
    without_box: list[Face] = []
    unannotated: list[Face] = []
    seen: set[Face] = set()
    for task in tasks:
        data = task["data"]
        if hours is not None and data.get("ronda_h") not in hours:
            continue
        face = _face_of(data)
        seen.add(face)
        live = [a for a in task.get("annotations") or [] if not a.get("was_cancelled")]
        if not live:
            unannotated.append(face)
            continue
        annotation = max(live, key=lambda a: a.get("updated_at") or "")
        found = [
            r
            for r in annotation["result"]
            if r.get("type") == "rectanglelabels" and r.get("from_name") in FROM_NAME_MODALITY
        ]
        if not found:
            without_box.append(face)
        for result in found:
            width, height = int(result["original_width"]), int(result["original_height"])
            boxes.append(
                HandBox(
                    face,
                    data.get("ronda_h"),
                    int(data["timestamp"]),
                    FROM_NAME_MODALITY[result["from_name"]],
                    percent_to_box(result["value"], width, height),
                    width,
                    height,
                )
            )
    return HandBoxes(boxes, without_box, unannotated, seen)


# ---------------------------------------------------------------------------------------------
# Which photos belong to which face
# ---------------------------------------------------------------------------------------------


@dataclass
class FacePlan:
    face: Face
    hands: list[HandBox]
    captures: dict[Moment, Capture]


@dataclass
class PlannedFaces:
    plans: list[FacePlan]
    dropped: list[Dropped]
    missing: list[Slot]


def plan_faces(
    data_root: Path, hand_boxes: list[HandBox], moments: tuple[Moment, ...] = MOMENTS
) -> PlannedFaces:
    """One capture per (face, moment): the duplicate rule of the task builder, missing slots reported."""
    captures = [c for c in load_captures(Path(data_root)) if c.ronda_h in moments]
    kept, dropped = resolve_slots(captures)
    by_slot = {c.slot: c for c in kept}
    faces: dict[Face, list[HandBox]] = {}
    for hand in hand_boxes:
        faces.setdefault(hand.face, []).append(hand)

    plans: list[FacePlan] = []
    missing: list[Slot] = []
    wanted: set[Slot] = set()
    for face, hands in faces.items():
        slots = {m: Slot(face.jornada, face.mango_id, face.cara, m) for m in moments}
        wanted.update(slots.values())
        if slots[REFERENCE_HOUR] not in by_slot:
            raise ValueError(
                f"no {moment_label(REFERENCE_HOUR)} photo found for {describe_slot(slots[REFERENCE_HOUR])}"
            )
        found = {m: by_slot[s] for m, s in slots.items() if s in by_slot}
        missing += [s for m, s in slots.items() if m not in found]
        plans.append(FacePlan(face, hands, found))
    return PlannedFaces(plans, [d for d in dropped if d.slot in wanted], missing)


# ---------------------------------------------------------------------------------------------
# Records, reports and Label Studio predictions
# ---------------------------------------------------------------------------------------------


@dataclass
class ProjectionRecord:
    face: Face
    moment: Moment
    hand: HandBox
    decision: Decision
    estimates: list[Estimate]


def prediction_filename(moment: Moment) -> str:
    """+10 h is the blind validation set: its projections never go to an importable name."""
    if moment == VALIDATION_HOUR:
        return "projection_10h_for_validation.json"
    return f"predictions_{moment_text(moment)}{'' if moment is None else 'h'}.json"


def _prediction(record: ProjectionRecord) -> dict:
    control, target = CONTROLS[record.hand.modality]
    value = box_to_percent(record.decision.box, record.hand.width, record.hand.height)
    value.update({"rotation": 0, "rectanglelabels": [LABEL]})
    return {
        "model_version": f"proj-{REFERENCE_HOUR}h-{record.decision.tier}",
        "score": SCORE[record.decision.tier],
        "result": [
            {
                "from_name": control,
                "to_name": target,
                "type": "rectanglelabels",
                "original_width": record.hand.width,
                "original_height": record.hand.height,
                "image_rotation": 0,
                "origin": "prediction",
                "value": value,
            }
        ],
    }


def attach_predictions(
    tasks: list[dict], records: list[ProjectionRecord], moment: Moment
) -> list[dict]:
    """Copy ``tasks`` adding a ``predictions`` entry per projected box (tiers A and B only)."""
    by_face: dict[Face, list[ProjectionRecord]] = {}
    for record in records:
        if record.moment == moment and record.decision.box is not None:
            by_face.setdefault(record.face, []).append(record)
    out = []
    for task in tasks:
        copy = dict(task)
        found = by_face.get(_face_of(task["data"]))
        if found:
            copy["predictions"] = [_prediction(r) for r in found]
        out.append(copy)
    return out


_PARAMS = ("scale", "angle_deg", "tx", "ty")
_METHOD_COLUMNS = {
    "sift": ("inliers", "rmse_px"),
    "dense": ("score",),
    "bridge_sift": ("inliers", "rmse_px"),
    "bridge_dense": ("score",),
}
_METRIC_KEY = {"rmse_px": "rmse"}
REPORT_COLUMNS = (
    ["jornada", "mango_id", "cara", "ronda_h", "modality", "tier", "reason", "methods"]
    + ["agreement_iou", "agreement_disp_frac", "box_x", "box_y", "box_w", "box_h"]
    + [f"{m}_{c}" for m, cols in _METHOD_COLUMNS.items() for c in ("valid", *cols, *_PARAMS)]
)


def _sim_params(sim) -> dict:
    if isinstance(sim, Similarity):
        return {p: getattr(sim, p) for p in _PARAMS}
    if isinstance(sim, CarriedTransform):
        return {p: getattr(sim.sim, p) for p in _PARAMS}
    return {}


def report_rows(records: list[ProjectionRecord]) -> list[dict]:
    rows = []
    for rec in records:
        d = rec.decision
        row: dict = {
            "jornada": rec.face.jornada,
            "mango_id": rec.face.mango_id,
            "cara": rec.face.cara,
            "ronda_h": "sano" if rec.moment is None else rec.moment,
            "modality": rec.hand.modality,
            "tier": d.tier,
            "reason": d.reason,
            "methods": "+".join(d.methods),
            "agreement_iou": d.agreement_iou,
            "agreement_disp_frac": d.agreement_displacement,
        }
        if d.box is not None:
            row.update(zip(("box_x", "box_y", "box_w", "box_h"), d.box))
        for est in rec.estimates:
            if est.method not in _METHOD_COLUMNS:
                continue
            row[f"{est.method}_valid"] = est.valid
            for column in _METHOD_COLUMNS[est.method]:
                row[f"{est.method}_{column}"] = est.metrics.get(_METRIC_KEY.get(column, column))
            for name, value in _sim_params(est.sim).items():
                row[f"{est.method}_{name}"] = value
        rows.append(row)
    return rows


def write_report(records: list[ProjectionRecord], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in report_rows(records):
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


# ---------------------------------------------------------------------------------------------
# Blind validation against later hand boxes
# ---------------------------------------------------------------------------------------------


@dataclass
class TierStats:
    n: int
    median_iou: float | None
    share_at_least_05: float | None
    share_at_least_07: float | None


@dataclass
class CenterError:
    n: int
    median_px: float | None
    median_fraction: float | None  # of the side sqrt(w * h) of the hand box


@dataclass
class AreaRatio:
    n: int
    median: float | None
    share_smaller: float | None


@dataclass
class ValidationReport:
    moment: Moment
    tiers: dict[str, TierStats]
    cross_modality: int
    no_hand_box: int
    unprojected: int
    sources: dict[str, TierStats] = field(default_factory=dict)
    projected: TierStats = field(default_factory=lambda: TierStats(0, None, None, None))
    by_lot: dict[str, dict[str, TierStats]] = field(default_factory=dict)
    by_modality: dict[str, dict[str, TierStats]] = field(default_factory=dict)
    copy_baseline: dict[str, TierStats] = field(default_factory=dict)
    area_ratio: AreaRatio = field(default_factory=lambda: AreaRatio(0, None, None))
    center_error: dict[str, CenterError] = field(default_factory=dict)
    center_bias: dict[str, tuple[float, float, int]] = field(default_factory=dict)
    size_ceiling: TierStats = field(default_factory=lambda: TierStats(0, None, None, None))
    cross_modality_projected: int = 0
    absent_faces: list[Face] = field(default_factory=list)
    no_box_faces: list[Face] = field(default_factory=list)
    unannotated_faces: list[Face] = field(default_factory=list)


def _tier_of(task: dict) -> str:
    predictions = task.get("predictions") or []
    return predictions[0]["model_version"].rsplit("-", 1)[-1] if predictions else "C"


def method_source(methods: str) -> str:
    """Where a projection's registration came from: the box's own modality, the RGB bridge, or both."""
    names = [m for m in methods.split("+") if m]
    bridged = [n.startswith("bridge_") for n in names]
    return "bridge" if all(bridged) else "native" if not any(bridged) else "mixed"


def _stats(values: list[float]) -> TierStats:
    if not values:
        return TierStats(0, None, None, None)
    arr = np.array(values)
    return TierStats(
        len(arr), float(np.median(arr)), float((arr >= 0.5).mean()), float((arr >= 0.7).mean())
    )


def _tier_groups(rows: list[tuple[str, float]]) -> dict[str, TierStats]:
    groups = {t: _stats([iou for tier, iou in rows if tier == t]) for t in ("A", "B")}
    groups["A+B"] = _stats([iou for _, iou in rows])
    return {k: v for k, v in groups.items() if v.n}


def validate_projections(
    hand_export: Path,
    projections: Path,
    report_csv: Path | None = None,
    source_export: Path | None = None,
) -> ValidationReport:
    """Score projected boxes against later hand boxes of the same modality and moment.

    ``hand_export`` holds the hand boxes of the projected moment (other moments are ignored); the
    +12 h source boxes come from ``source_export`` (default: the same file) and give the baseline
    of copying the +12 h box unchanged. With the run's ``report_csv`` the tiers are also split by
    where the registration came from.
    """
    tasks = json.loads(Path(projections).read_text(encoding="utf-8"))
    hours = {t["data"]["ronda_h"] for t in tasks}
    if len(hours) != 1:
        raise ValueError(
            f"projections must be for a single moment, found {sorted(map(str, hours))}"
        )
    moment = hours.pop()
    later = load_hand_boxes(hand_export, hours={moment})
    source = load_hand_boxes(source_export or hand_export, hours={REFERENCE_HOUR})
    hand = {(h.face, h.modality): h.box for h in later.boxes}
    hand_faces = {face for face, _ in hand}
    origin = {(h.face, h.modality): h.box for h in source.boxes}

    methods_of: dict[tuple[Face, str], str] = {}
    if report_csv is not None:
        with Path(report_csv).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (Face(row["jornada"], int(row["mango_id"]), int(row["cara"])), row["ronda_h"])
                methods_of[key] = row["methods"]

    projected_faces = [_face_of(t["data"]) for t in tasks]
    tier_of_face = {_face_of(t["data"]): _tier_of(t) for t in tasks}
    projected_on: set[tuple[Face, str]] = set()
    scored: list[tuple[str, str, str, float, float, float]] = (
        []
    )  # tier, lot, modality, IoU, centre error px, in sides
    by_source: dict[str, list[float]] = defaultdict(list)
    cross_projected = no_hand = unprojected = 0
    for task in tasks:
        face = _face_of(task["data"])
        predictions = task.get("predictions") or []
        if not predictions:
            unprojected += face in hand_faces
            continue
        tier = _tier_of(task)
        for prediction in predictions:
            result = prediction["result"][0]
            modality = FROM_NAME_MODALITY[result["from_name"]]
            projected_on.add((face, modality))
            projected = percent_to_box(
                result["value"], result["original_width"], result["original_height"]
            )
            if face not in hand_faces:
                no_hand += 1
            elif (face, modality) not in hand:
                cross_projected += 1
            else:
                iou = box_iou(projected, hand[(face, modality)])
                error = _center_distance(projected, hand[(face, modality)])
                side = math.sqrt(hand[(face, modality)][2] * hand[(face, modality)][3])
                offset = (
                    hand[(face, modality)][0]
                    + hand[(face, modality)][2] / 2
                    - projected[0]
                    - projected[2] / 2,
                    hand[(face, modality)][1]
                    + hand[(face, modality)][3] / 2
                    - projected[1]
                    - projected[3] / 2,
                )
                scored.append((tier, face.jornada, modality, iou, error, error / side, offset))
                methods = methods_of.get((face, moment_text(moment)))
                if methods is not None:
                    by_source[f"{tier}:{method_source(methods)}"].append(iou)

    def grouped(key: int) -> dict[str, dict[str, TierStats]]:
        keys = sorted({row[key] for row in scored})
        return {
            k: _tier_groups([(row[0], row[3]) for row in scored if row[key] == k]) for k in keys
        }

    copy_rows: dict[str, list[float]] = defaultdict(list)
    ratios: list[float] = []
    centred: list[float] = []
    cross_all = 0
    modality_12: dict[Face, set[str]] = defaultdict(set)
    for face, modality in origin:
        modality_12[face].add(modality)
    for face, modality in projected_on:  # without a source export, the projection tells
        if face not in modality_12:
            modality_12[face].add(modality)
    for face in sorted(modality_12):
        modalities_12 = modality_12[face]
        modalities_10 = {m for f, m in hand if f == face}
        cross_all += bool(modalities_10) and not (modalities_12 & modalities_10)
        for modality in sorted(modalities_12 & modalities_10):
            if (face, modality) not in origin:
                continue
            earlier, later_box = origin[(face, modality)], hand[(face, modality)]
            ratios.append(later_box[2] * later_box[3] / (earlier[2] * earlier[3]))
            cx, cy = later_box[0] + later_box[2] / 2, later_box[1] + later_box[3] / 2
            centred.append(
                box_iou(
                    (cx - earlier[2] / 2, cy - earlier[3] / 2, earlier[2], earlier[3]), later_box
                )
            )
            if face in tier_of_face:
                iou = box_iou(earlier, later_box)
                copy_rows[tier_of_face[face]].append(iou)
                copy_rows["all"].append(iou)
    copy_rows["A+B"] = copy_rows.get("A", []) + copy_rows.get("B", [])
    copy_baseline = {
        k: _stats(copy_rows[k]) for k in ("A", "B", "A+B", "C", "all") if copy_rows.get(k)
    }

    center_errors = {}
    for name, tiers in (("A", {"A"}), ("B", {"B"}), ("A+B", {"A", "B"})):
        rows = [r for r in scored if r[0] in tiers]
        if rows:
            center_errors[name] = CenterError(
                len(rows),
                float(np.median([r[4] for r in rows])),
                float(np.median([r[5] for r in rows])),
            )
    center_bias = {}
    for lot in sorted({r[1] for r in scored}):
        offsets = np.array([r[6] for r in scored if r[1] == lot])
        center_bias[lot] = (
            float(np.median(offsets[:, 0])),
            float(np.median(offsets[:, 1])),
            len(offsets),
        )
    in_projections = set(projected_faces)
    return ValidationReport(
        moment=moment,
        tiers={t: _stats([r[3] for r in scored if r[0] == t]) for t in ("A", "B")},
        cross_modality=cross_all,
        no_hand_box=no_hand,
        unprojected=unprojected,
        sources={k: _stats(v) for k, v in sorted(by_source.items())},
        projected=_stats([r[3] for r in scored]),
        by_lot=grouped(1),
        by_modality=grouped(2),
        copy_baseline=copy_baseline,
        area_ratio=AreaRatio(
            len(ratios),
            float(np.median(ratios)) if ratios else None,
            float(np.mean(np.array(ratios) < 1)) if ratios else None,
        ),
        center_error=center_errors,
        center_bias=center_bias,
        size_ceiling=_stats(centred),
        cross_modality_projected=cross_projected,
        absent_faces=sorted(f for f in in_projections if f not in later.seen),
        no_box_faces=sorted(set(later.without_box) & in_projections),
        unannotated_faces=sorted(set(later.unannotated) & in_projections),
    )


def _center_distance(a: Box, b: Box) -> float:
    return math.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2, a[1] + a[3] / 2 - b[1] - b[3] / 2)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _stats_line(name: str, s: TierStats) -> str:
    median = "-" if s.median_iou is None else f"{s.median_iou:.2f}"
    return f"{name:<14}{s.n:>4}  {median:>10}  {_pct(s.share_at_least_05):>8}  {_pct(s.share_at_least_07):>8}"


def format_validation(report: ValidationReport) -> str:
    header = "                  n  median IoU  IoU>=0.5  IoU>=0.7"
    lines = [f"Validation at {moment_label(report.moment)}: projected boxes vs hand boxes", header]
    for tier, s in report.tiers.items():
        lines.append(_stats_line(f"tier {tier}", s))
    lines.append(_stats_line("tier A+B", report.projected))
    for name, s in report.sources.items():
        lines.append(_stats_line(f"  {name}", s))
    lines.append("centre error of the projection, median (px / box sides):")
    for name, c in report.center_error.items():
        lines.append(
            f"  {name:<5}{c.n:>4}  {c.median_px:>7.0f} px  {c.median_fraction:>5.2f} sides"
        )
    lines.append("median centre offset, hand box minus projection (px), by lot:")
    for lot, (dx, dy, n) in report.center_bias.items():
        lines.append(f"  {lot[:5]}{n:>4}  dx {dx:>+6.0f}  dy {dy:>+6.0f}")
    lines.append("size ceiling (IoU of the +12 h box centred exactly on the hand box):")
    lines.append(_stats_line("  all", report.size_ceiling))
    lines.append("copy of the +12 h box, same faces (no registration):")
    for name, s in report.copy_baseline.items():
        lines.append(_stats_line(f"  {name}", s))
    lines.append("by lot:")
    for lot, groups in report.by_lot.items():
        for name, s in groups.items():
            lines.append(_stats_line(f"  {lot[:5]} {name}", s))
    lines.append("by modality (the +12 h and the hand box on the same modality):")
    for modality, groups in report.by_modality.items():
        for name, s in groups.items():
            lines.append(_stats_line(f"  {modality.upper()} {name}", s))
    a = report.area_ratio
    median = "-" if a.median is None else f"{a.median:.2f}"
    lines.append(
        f"hand box area ratio, {moment_label(report.moment)} / +12 h: median {median} "
        f"({_pct(a.share_smaller)} smaller than at +12 h, n={a.n})"
    )
    lines.append(
        f"skipped, cross-modality (+12 h and later box on different modalities): "
        f"{report.cross_modality} faces, {report.cross_modality_projected} of them projected"
    )
    lines.append(f"projected but no hand box on that face: {report.no_hand_box}")
    for label, faces in (
        ("absent from the hand export", report.absent_faces),
        ("annotated with no box", report.no_box_faces),
        ("not annotated", report.unannotated_faces),
    ):
        if faces:
            lines.append(
                f"  {label}: "
                + "; ".join(f"{f.jornada[:5]} m{f.mango_id} f{f.cara}" for f in faces)
            )
    lines.append(f"hand box on a face that got no projection (Tier C): {report.unprojected}")
    return "\n".join(lines)


def validation_to_dict(report: ValidationReport) -> dict:
    """JSON-friendly form of the report."""
    from dataclasses import asdict

    out = asdict(report)
    out["center_bias"] = {k: list(v) for k, v in report.center_bias.items()}
    for key in ("absent_faces", "no_box_faces", "unannotated_faces"):
        out[key] = [list(f) for f in out[key]]
    out["moment"] = moment_text(report.moment)
    return out
