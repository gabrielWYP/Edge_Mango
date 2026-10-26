"""Run the box projection over a Label Studio export: photos, registrations, outputs.

``box_projection`` holds the decisions; this module reads the photos (half-size decodes), runs the
registrations face by face (optionally in worker processes), and writes the Label Studio
predictions, the CSV report and the contact sheet.
"""

from __future__ import annotations

import json
import multiprocessing
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np

from .box_projection import (
    EARLIER,
    MOMENTS,
    REFERENCE_HOUR,
    VALIDATION_HOUR,
    Bridge,
    Estimate,
    Face,
    FacePlan,
    HandBox,
    HandBoxes,
    PlannedFaces,
    ProjectionRecord,
    agreement,
    attach_predictions,
    box_corners,
    decide,
    load_hand_boxes,
    moment_key,
    plan_faces,
    prediction_filename,
)
from .captura_v3_tasks import (
    Capture,
    Moment,
    build_tasks,
    load_captures,
    load_wb_timestamps,
    moment_label,
    resolve_slots,
    write_tasks,
)
from .mango_registration import (
    MIN_INLIERS,
    DenseResult,
    Prepared,
    SiftResult,
    nir_mango_mask,
    prepare,
    register_dense,
    register_sift,
    rgb_mango_mask,
)
from .nir_display import enhance_nir, nir_vis_relpath

HALF = 0.5
# Minimum masked-ZNCC peak for the dense method to count as registered, per modality: just above
# the maximum seen over 300 random unrelated-fruit pairs of the same lot (rgb 0.772, nir 0.803;
# `scripts/project_boxes.py --calibrate 300`). SIFT needs no such tuning: unrelated pairs never
# gave more than 6 inliers against the 15 required.
DENSE_MIN_SCORE = {"rgb": 0.78, "nir": 0.81}
MODALITIES = ("rgb", "nir")
_SPAWN = multiprocessing.get_context("spawn")
LOT_SHIFT_MIN_PAIRS = 50
SHEET_MOMENTS: tuple[Moment, ...] = (12, *EARLIER)
JPEG_LIMIT = 400_000


def _odd(value: float) -> int:
    n = max(3, int(round(value)))
    return n if n % 2 else n + 1


def _grown(mask_small: np.ndarray, shape: tuple[int, int], grow: float) -> np.ndarray:
    big = cv2.resize(mask_small, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    big = (big > 127).astype(np.uint8) * 255
    if grow > 0:
        big = cv2.dilate(
            big, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(grow * shape[1]),) * 2)
        )
    return big


class PhotoLoader:
    """Reads captura_v3 photos at half size and derives the fruit mask of each."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root)
        self._wb = load_wb_timestamps(self.data_root)
        self._empty: dict[str, np.ndarray] = {}
        self._index: dict[tuple[Face, Moment], Capture] | None = None

    def rgb_file(self, capture: Capture) -> Path:
        folder = "rgb_wb" if capture.timestamp in self._wb else "rgb"
        return self.data_root / "news" / folder / f"mango_rgb_{capture.timestamp}.jpg"

    def nir_file(self, capture: Capture) -> Path:
        return self.data_root / "news" / "nir" / f"mango_nir_{capture.timestamp}.jpg"

    def empty_tray_file(self, jornada: str) -> Path:
        """The latest empty-tray NIR of the jornada (lighting drifts, so the closest in time)."""
        found: dict[int, Path] = {}
        for meta in (self.data_root / "calib").glob("calib-vacia-meta-*.json"):
            info = json.loads(meta.read_text(encoding="utf-8"))
            if (info.get("jornada") or {}).get("id") == jornada:
                found[int(info["timestamp"])] = meta.with_name(
                    f"calib-vacia-nir-{info['timestamp']}.jpg"
                )
        if not found:
            raise FileNotFoundError(
                f"no empty-tray NIR calibration for {jornada} in {self.data_root / 'calib'}"
            )
        return found[max(found)]

    def capture(self, face: Face, moment: Moment) -> Capture | None:
        if self._index is None:
            kept, _ = resolve_slots(load_captures(self.data_root))
            self._index = {(Face(c.jornada, c.mango_id, c.cara), c.ronda_h): c for c in kept}
        return self._index.get((face, moment))

    def _empty_gray(self, jornada: str) -> np.ndarray:
        if jornada not in self._empty:
            path = self.empty_tray_file(jornada)
            image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
            if image is None:
                raise FileNotFoundError(f"empty-tray image not readable: {path}")
            self._empty[jornada] = _nir_gray(image)
        return self._empty[jornada]

    def photo(self, capture: Capture, modality: str) -> tuple[np.ndarray, np.ndarray]:
        """Half-size gray image and fruit mask (0/255) of a photo in ``modality``."""
        if modality == "rgb":
            return self._rgb(capture)
        if modality == "nir":
            return self._nir(capture)
        raise ValueError(f"unknown modality {modality!r}")

    def prepare(self, capture: Capture, modality: str) -> Prepared:
        gray, mask = self.photo(capture, modality)
        return prepare(gray, mask, HALF)

    def _rgb(self, capture: Capture) -> tuple[np.ndarray, np.ndarray]:
        path = self.rgb_file(capture)
        bgr = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
        if bgr is None:
            raise FileNotFoundError(f"RGB photo not readable: {path}")
        small = cv2.resize(bgr, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        mask = _grown(rgb_mango_mask(small), bgr.shape[:2], grow=0.012)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), mask

    def _nir(self, capture: Capture) -> tuple[np.ndarray, np.ndarray]:
        raw_path = self.nir_file(capture)
        vis_path = self.data_root / nir_vis_relpath(capture.timestamp)
        gray = (
            cv2.imread(str(vis_path), cv2.IMREAD_REDUCED_GRAYSCALE_2)
            if vis_path.is_file()
            else None
        )
        if gray is None:
            full = cv2.imread(str(raw_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if full is None:
                raise FileNotFoundError(f"NIR photo not readable: {raw_path}")
            gray = cv2.resize(
                enhance_nir(full), None, fx=HALF, fy=HALF, interpolation=cv2.INTER_AREA
            )
        small = cv2.imread(str(raw_path), cv2.IMREAD_REDUCED_COLOR_8)
        if small is None:
            raise FileNotFoundError(f"NIR photo not readable: {raw_path}")
        mask = nir_mango_mask(_nir_gray(small), self._empty_gray(capture.jornada))
        return gray, _grown(mask, gray.shape, grow=0.0)

    def thumbnail(self, capture: Capture, modality: str, width: int) -> np.ndarray:
        """A small BGR picture of the photo in ``modality`` (the enhanced NIR when it exists)."""
        if modality == "rgb":
            image = cv2.imread(str(self.rgb_file(capture)), cv2.IMREAD_REDUCED_COLOR_8)
        else:
            vis = self.data_root / nir_vis_relpath(capture.timestamp)
            image = cv2.imread(
                str(vis if vis.is_file() else self.nir_file(capture)), cv2.IMREAD_REDUCED_COLOR_8
            )
        if image is None:
            raise FileNotFoundError(f"photo not readable for {capture.timestamp} ({modality})")
        return cv2.resize(
            image,
            (width, round(image.shape[0] * width / image.shape[1])),
            interpolation=cv2.INTER_AREA,
        )


def _nir_gray(bgr: np.ndarray) -> np.ndarray:
    return (bgr[..., 0].astype(np.float32) + bgr[..., 2].astype(np.float32)) / 2


# ---------------------------------------------------------------------------------------------
# Registrations per face
# ---------------------------------------------------------------------------------------------

PairKey = tuple[str, int, Moment]  # (modality, reference hour, moving moment)


@dataclass
class FaceResult:
    face: Face
    pairs: dict[PairKey, tuple[SiftResult, DenseResult]] = field(default_factory=dict)
    centroids: dict[tuple[str, Moment], tuple[float, float]] = field(default_factory=dict)


def _centroid(prepared: Prepared) -> tuple[float, float] | None:
    ys, xs = np.nonzero(prepared.mask > 127)
    if len(xs) == 0:
        return None
    return (float(xs.mean() + 0.5) / prepared.scale, float(ys.mean() + 0.5) / prepared.scale)


def register_face(
    plan: FacePlan,
    loader: PhotoLoader,
    modalities: Iterable[str],
    with_cycles: bool,
    moments: tuple[Moment, ...],
) -> FaceResult:
    """Register the +12 h photo (and the +10 h one, for the cycle check) to the earlier ones."""
    result = FaceResult(plan.face)
    cache: dict[tuple[str, Moment], Prepared] = {}

    def photo(modality: str, moment: Moment) -> Prepared:
        if (modality, moment) not in cache:
            cache[(modality, moment)] = loader.prepare(plan.captures[moment], modality)
            centre = _centroid(cache[(modality, moment)])
            if centre is not None:
                result.centroids[(modality, moment)] = centre
        return cache[(modality, moment)]

    references = [REFERENCE_HOUR]
    if with_cycles and VALIDATION_HOUR in plan.captures:
        references.append(VALIDATION_HOUR)
    for modality in modalities:
        for reference in references:
            ref = photo(modality, reference)
            for moment in EARLIER:
                if (
                    moment not in moments
                    or moment not in plan.captures
                    or moment_key(moment) >= reference
                ):
                    continue
                mov = photo(modality, moment)
                result.pairs[(modality, reference, moment)] = (
                    register_sift(ref, mov),
                    register_dense(ref, mov),
                )
    return result


def _register_job(args: tuple) -> FaceResult:
    cv2.setNumThreads(1)
    plan, data_root, modalities, with_cycles, moments = args
    return register_face(plan, PhotoLoader(data_root), modalities, with_cycles, moments)


# ---------------------------------------------------------------------------------------------
# RGB -> NIR bridge per lot
# ---------------------------------------------------------------------------------------------


def estimate_lot_shifts(
    pairs_by_lot: Mapping[str, list[tuple[np.ndarray, np.ndarray]]],
    homography: np.ndarray,
    reference: str,
    min_pairs: int = LOT_SHIFT_MIN_PAIRS,
) -> dict[str, tuple[float, float]]:
    """Shift of each lot's RGB-to-NIR map relative to the lot the homography was measured on.

    ``pairs_by_lot`` holds (RGB fruit centroid, NIR fruit-mask centroid) per photo. The NIR mask
    includes the shadow, which biases every lot the same way, so the difference between a lot's
    median residual and the reference lot's is the rig shift.
    """
    bridge = Bridge(homography)

    def median_residual(pairs) -> np.ndarray | None:
        if len(pairs) < min_pairs:
            return None
        rgb = np.array([p[0] for p in pairs], dtype=float)
        nir = np.array([p[1] for p in pairs], dtype=float)
        return np.median(nir - bridge.to_nir(rgb), axis=0)

    base = median_residual(pairs_by_lot.get(reference, []))
    shifts: dict[str, tuple[float, float]] = {}
    for lot, pairs in pairs_by_lot.items():
        here = median_residual(pairs)
        if lot == reference or here is None or base is None:
            shifts[lot] = (0.0, 0.0)
        else:
            shifts[lot] = (float(here[0] - base[0]), float(here[1] - base[1]))
    return shifts


# ---------------------------------------------------------------------------------------------
# From registrations to records
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    data_root: Path
    export: Path
    moments: tuple[Moment, ...] = MOMENTS
    workers: int = 1
    with_cycles: bool = True
    limit_faces: int | None = None
    homography: Path | None = None
    homography_jornada: str | None = None
    dense_min: Mapping[str, float] = field(default_factory=lambda: dict(DENSE_MIN_SCORE))
    min_inliers: int = MIN_INLIERS


@dataclass
class ProjectionRun:
    records: list[ProjectionRecord]
    cycles: list[dict]
    planned: PlannedFaces
    hand: HandBoxes
    results: list[FaceResult]
    shifts: dict[str, tuple[float, float]]


def _carry(bridge: Bridge | None):
    return (lambda sim: sim) if bridge is None else bridge.carry


def _estimates(
    sift: SiftResult, dense: DenseResult, prefix: str, dense_min: float, min_inliers: int, carry
) -> list[Estimate]:
    return [
        Estimate(
            f"{prefix}sift",
            None if sift.sim is None else carry(sift.sim),
            sift.valid and sift.inliers >= min_inliers,
            {"inliers": sift.inliers, "rmse": sift.rmse},
        ),
        Estimate(
            f"{prefix}dense",
            None if dense.sim is None else carry(dense.sim),
            dense.sim is not None and dense.score >= dense_min,
            {"score": dense.score},
        ),
    ]


def _record_estimates(
    result: FaceResult, hand: HandBox, moment: Moment, bridge: Bridge | None, config: RunConfig
) -> list[Estimate]:
    native = result.pairs.get((hand.modality, REFERENCE_HOUR, moment))
    if native is None:
        return []
    estimates = _estimates(
        *native, "", config.dense_min[hand.modality], config.min_inliers, lambda s: s
    )
    rgb = result.pairs.get(("rgb", REFERENCE_HOUR, moment))
    if hand.modality == "nir" and bridge is not None and rgb is not None:
        estimates += _estimates(
            *rgb, "bridge_", config.dense_min["rgb"], config.min_inliers, bridge.carry
        )
    return estimates


def _cycle_rows(
    plan: FacePlan, hand: HandBox, result: FaceResult, bridge: Bridge | None, config: RunConfig
) -> list[dict]:
    if VALIDATION_HOUR not in plan.captures:
        return []
    rows = []
    for moment in EARLIER:
        if moment == VALIDATION_HOUR or moment not in plan.captures or moment not in config.moments:
            continue
        sources = [("", hand.modality, lambda s: s)]
        if hand.modality == "nir" and bridge is not None:
            sources.append(("bridge_", "rgb", bridge.carry))
        for prefix, modality, carry in sources:
            via = result.pairs.get((modality, REFERENCE_HOUR, VALIDATION_HOUR))
            direct = result.pairs.get((modality, REFERENCE_HOUR, moment))
            onward = result.pairs.get((modality, VALIDATION_HOUR, moment))
            if not (via and direct and onward):
                continue
            for index, name in enumerate(("sift", "dense")):
                legs = [via[index], direct[index], onward[index]]
                if not all(_leg_valid(leg, name, modality, config) for leg in legs):
                    continue
                first, second, third = (carry(leg.sim) for leg in (legs[0], legs[2], legs[1]))
                gap = _cycle_gap(hand.box, first, second, third)
                rows.append(
                    {
                        "face": plan.face,
                        "moment": moment,
                        "method": f"{prefix}{name}",
                        "displacement_px": gap,
                        "displacement_frac": gap / float(np.sqrt(hand.box[2] * hand.box[3])),
                    }
                )
    return rows


def _cycle_gap(box, via, onward, direct) -> float:
    """Largest corner distance between going +12 h -> +10 h -> moment and going straight there."""
    corners = box_corners(box)
    return float(
        np.linalg.norm(onward.apply(via.apply(corners)) - direct.apply(corners), axis=1).max()
    )


def _leg_valid(leg, name: str, modality: str, config: RunConfig) -> bool:
    if name == "sift":
        return leg.sim is not None and leg.valid and leg.inliers >= config.min_inliers
    return leg.sim is not None and leg.score >= config.dense_min[modality]


def run_projection(config: RunConfig) -> ProjectionRun:
    """Project every +12 h hand box onto the earlier moments of its face."""
    hand = load_hand_boxes(config.export, hours={REFERENCE_HOUR})
    boxes = hand.boxes
    planned = plan_faces(config.data_root, boxes, config.moments)
    plans = sorted(planned.plans, key=lambda p: p.face)[: config.limit_faces]
    homography = None if config.homography is None else np.load(config.homography)

    jobs = []
    for plan in plans:
        modalities = {h.modality for h in plan.hands}
        if homography is not None and "nir" in modalities:
            modalities.add("rgb")
        jobs.append(
            (
                plan,
                config.data_root,
                tuple(m for m in MODALITIES if m in modalities),
                config.with_cycles,
                config.moments,
            )
        )
    if config.workers > 1:
        with ProcessPoolExecutor(config.workers, mp_context=_SPAWN) as pool:
            results = list(pool.map(_register_job, jobs, chunksize=1))
    else:
        results = [_register_job(job) for job in jobs]

    shifts: dict[str, tuple[float, float]] = {}
    bridges: dict[str, Bridge | None] = {plan.face.jornada: None for plan in plans}
    if homography is not None:
        pairs: dict[str, list] = defaultdict(list)
        for plan, result in zip(plans, results):
            if not any(h.modality == "nir" for h in plan.hands):
                continue
            for moment in plan.captures:
                rgb, nir = result.centroids.get(("rgb", moment)), result.centroids.get(
                    ("nir", moment)
                )
                if rgb and nir:
                    pairs[plan.face.jornada].append((np.array(rgb), np.array(nir)))
        reference = config.homography_jornada or sorted(pairs)[0] if pairs else ""
        shifts = estimate_lot_shifts(pairs, homography, reference)
        bridges = {lot: Bridge(homography, shifts.get(lot, (0.0, 0.0))) for lot in bridges}

    records: list[ProjectionRecord] = []
    cycles: list[dict] = []
    for plan, result in zip(plans, results):
        bridge = bridges[plan.face.jornada]
        for hand_box in plan.hands:
            for moment in EARLIER:
                if moment not in plan.captures or moment not in config.moments:
                    continue
                estimates = _record_estimates(result, hand_box, moment, bridge, config)
                if not estimates:
                    continue
                decision = decide(estimates, hand_box.box, hand_box.width, hand_box.height)
                records.append(ProjectionRecord(plan.face, moment, hand_box, decision, estimates))
            cycles += _cycle_rows(plan, hand_box, result, bridge, config)
    return ProjectionRun(records, cycles, planned, hand, results, shifts)


# ---------------------------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------------------------


def tier_table(records: list[ProjectionRecord]) -> dict[tuple[str, Moment], Counter]:
    table: dict[tuple[str, Moment], Counter] = defaultdict(Counter)
    for record in records:
        table[(record.face.jornada, record.moment)][record.decision.tier] += 1
    return dict(sorted(table.items(), key=lambda kv: (kv[0][0], -moment_key(kv[0][1]))))


def format_tier_table(table: dict[tuple[str, Moment], Counter]) -> str:
    lines = [f"{'jornada':<16}{'moment':<9}{'n':>5}{'A':>6}{'B':>6}{'C':>6}  A+B"]
    for (jornada, moment), counts in table.items():
        n = sum(counts.values())
        lines.append(
            f"{jornada:<16}{moment_label(moment):<9}{n:>5}{counts['A']:>6}{counts['B']:>6}{counts['C']:>6}"
            f"  {(counts['A'] + counts['B']) / n * 100:.0f}%"
        )
    return "\n".join(lines)


def validity_table(records: list[ProjectionRecord]) -> dict[tuple[str, Moment], dict[str, float]]:
    """Share of valid registrations per method, by box modality and moment."""
    seen: dict[tuple[str, Moment], list[ProjectionRecord]] = defaultdict(list)
    for record in records:
        seen[(record.hand.modality, record.moment)].append(record)
    table = {}
    for key, group in sorted(seen.items(), key=lambda kv: (kv[0][0], -moment_key(kv[0][1]))):
        row: dict[str, float] = {"n": float(len(group))}
        for method in ("sift", "dense", "bridge_sift", "bridge_dense"):
            flags = [e.valid for r in group for e in r.estimates if e.method == method]
            if flags:
                row[method] = sum(flags) / len(flags)
        table[key] = row
    return table


def consistency_summary(records: list[ProjectionRecord], cycles: list[dict]) -> dict:
    """Cycle error 12->t->10 vs 12->t, and native SIFT vs dense agreement."""
    out: dict = {"cycle": {}, "sift_vs_dense": {}}
    for method in sorted({c["method"] for c in cycles}):
        rows = [c for c in cycles if c["method"] == method]
        px = np.array([c["displacement_px"] for c in rows])
        frac = np.array([c["displacement_frac"] for c in rows])
        out["cycle"][method] = {
            "n": len(rows),
            "median_px": float(np.median(px)),
            "p90_px": float(np.percentile(px, 90)),
            "share_under_10pct_side": float((frac < 0.10).mean()),
        }
    for prefix in ("", "bridge_"):
        both = []
        for record in records:
            by = {e.method: e for e in record.estimates}
            a, b = by.get(f"{prefix}sift"), by.get(f"{prefix}dense")
            if a and b and a.valid and b.valid:
                both.append(
                    agreement(record.hand.box, a.sim, b.sim, record.hand.width, record.hand.height)
                )
        if both:
            disp = np.array([x.displacement for x in both])
            out["sift_vs_dense"][prefix or "native"] = {
                "n": len(both),
                "share_agree": float(np.mean([x.agree for x in both])),
                "share_under_10pct_side": float((disp < 0.10).mean()),
                "median_displacement_frac": float(np.median(disp)),
            }
    return out


# ---------------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------------


def write_predictions(
    records: list[ProjectionRecord],
    data_root: Path,
    document_root: Path,
    out_dir: Path,
    *,
    nir_display: str = "enhanced",
) -> list[Path]:
    """One Label Studio task file per projected moment, with predictions on tiers A and B."""
    out_dir = Path(out_dir)
    written = []
    for moment in sorted({r.moment for r in records}, key=moment_key, reverse=True):
        built = build_tasks(
            Path(data_root), Path(document_root), (moment,), nir_display=nir_display
        )
        path = out_dir / prediction_filename(moment)
        write_tasks(attach_predictions(built.tasks, records, moment), path)
        written.append(path)
    return written


TIER_COLOUR = {"A": (60, 200, 60), "B": (0, 165, 255), "C": (40, 40, 220)}


def pick_sheet_faces(records: list[ProjectionRecord], count: int) -> list[tuple[Face, str]]:
    """Faces for the sheet: round-robin over (modality, jornada) with NIR first, most varied tiers first."""
    tiers: dict[tuple[Face, str], Counter] = defaultdict(Counter)
    for record in records:
        tiers[(record.face, record.hand.modality)][record.decision.tier] += 1
    queues: dict[tuple[str, str], list] = defaultdict(list)
    for (face, modality), counts in tiers.items():
        queues[(modality, face.jornada)].append(
            ((-len(counts), -counts["A"], face), face, modality)
        )
    ordered = [sorted(queues[key]) for key in sorted(queues, key=lambda k: (k[0] != "nir", k[1]))]
    picked: list[tuple[Face, str]] = []
    while len(picked) < count and any(ordered):
        for queue in ordered:
            if queue and len(picked) < count:
                _, face, modality = queue.pop(0)
                picked.append((face, modality))
    return picked


def _save_jpeg_under(path: Path, image: np.ndarray, limit: int = JPEG_LIMIT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for quality in (80, 70, 60, 50, 40, 30):
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(buffer) < limit:
            break
    path.write_bytes(buffer.tobytes())


def render_contact_sheet(
    records: list[ProjectionRecord],
    loader: PhotoLoader,
    out: Path,
    *,
    faces: int = 6,
    thumb_width: int = 200,
) -> None:
    """Faces by moments, the +12 h hand box and each projection drawn tier-coloured."""
    by_key = {(r.face, r.hand.modality, r.moment): r for r in records}
    picked = pick_sheet_faces(records, faces)
    rows = []
    for face, modality in picked:
        cells = []
        for moment in SHEET_MOMENTS:
            capture = loader.capture(face, moment)
            record = by_key.get((face, modality, moment))
            hand = next(r.hand for r in records if r.face == face and r.hand.modality == modality)
            thumb = (
                loader.thumbnail(capture, modality, thumb_width)
                if capture
                else np.full((thumb_width * 3 // 4, thumb_width, 3), 235, np.uint8)
            )
            ratio = thumb.shape[1] / hand.width
            label = "hand" if moment == REFERENCE_HOUR else record.decision.tier if record else "-"
            colour = (
                (255, 120, 0)
                if moment == REFERENCE_HOUR
                else TIER_COLOUR.get(label, (150, 150, 150))
            )
            box = (
                hand.box if moment == REFERENCE_HOUR else (record.decision.box if record else None)
            )
            if box is not None:
                x, y, w, h = (v * ratio for v in box)
                cv2.rectangle(thumb, (round(x), round(y)), (round(x + w), round(y + h)), colour, 2)
            text = f"{moment_label(moment)} {label}"
            cv2.putText(
                thumb, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                thumb,
                text,
                (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                colour if label != "-" else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cells.append(thumb)
        height = min(c.shape[0] for c in cells)
        rows.append(np.hstack([c[:height] for c in cells]))
    width = min(r.shape[1] for r in rows)
    _save_jpeg_under(Path(out), np.vstack([r[:, :width] for r in rows]))


def render_mask_sheet(
    loader: PhotoLoader,
    items: list[tuple[Capture, str]],
    out: Path,
    *,
    thumb_width: int = 240,
    columns: int = 4,
) -> None:
    """The fruit mask outline (red) on each photo, to check the masks by eye."""
    cells = []
    for capture, modality in items:
        gray, mask = loader.photo(capture, modality)
        scale = thumb_width / gray.shape[1]
        thumb = cv2.cvtColor(
            cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
            cv2.COLOR_GRAY2BGR,
        )
        small = cv2.resize(mask, (thumb.shape[1], thumb.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(
            (small > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(thumb, contours, -1, (0, 0, 255), 2)
        label = f"{capture.jornada[:5]} m{capture.mango_id}/{capture.cara} {moment_label(capture.ronda_h)} {modality}"
        cv2.putText(
            thumb, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            thumb, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA
        )
        cells.append(thumb)
    height = min(c.shape[0] for c in cells)
    cells = [c[:height] for c in cells]
    while len(cells) % columns:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i : i + columns]) for i in range(0, len(cells), columns)]
    _save_jpeg_under(Path(out), np.vstack(rows))


# ---------------------------------------------------------------------------------------------
# Null calibration
# ---------------------------------------------------------------------------------------------


def _null_job(args: tuple) -> tuple[float, float]:
    cv2.setNumThreads(1)
    data_root, modality, ref_capture, mov_capture = args
    loader = PhotoLoader(data_root)
    ref, mov = loader.prepare(ref_capture, modality), loader.prepare(mov_capture, modality)
    return float(register_sift(ref, mov).inliers), float(register_dense(ref, mov).score)


def calibrate_nulls(
    loader: PhotoLoader,
    plans: list[FacePlan],
    *,
    modality: str,
    pairs: int,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, list[float]]:
    """Registration scores of unrelated fruits (same jornada), to set the validity thresholds."""
    rng = random.Random(seed)
    by_lot: dict[str, list[FacePlan]] = defaultdict(list)
    for plan in plans:
        by_lot[plan.face.jornada].append(plan)
    lots = [lot for lot, group in sorted(by_lot.items()) if len(group) > 1]
    if not lots:
        raise ValueError("need at least two faces in a jornada to draw unrelated pairs")
    jobs = []
    for _ in range(pairs):
        ref_plan, other = rng.sample(by_lot[rng.choice(lots)], 2)
        moment = rng.choice([m for m in EARLIER if m in other.captures])
        jobs.append(
            (loader.data_root, modality, ref_plan.captures[REFERENCE_HOUR], other.captures[moment])
        )
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=_SPAWN) as pool:
            scores = list(pool.map(_null_job, jobs, chunksize=1))
    else:
        scores = [_null_job(job) for job in jobs]
    return {"sift_inliers": [s[0] for s in scores], "dense_score": [s[1] for s in scores]}
