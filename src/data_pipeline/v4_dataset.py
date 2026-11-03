"""captura_v3 photos + Label Studio damage boxes -> the v4 training dataset.

The training loader registers the NIR onto the RGB frame with ONE homography, but every
captura_v3 lot has its own. As in ``scripts/export_aligned_corpus.py``, the registration is
baked into the files, so training runs with ``nir_homography_path=<out>/homographies/identity.npy``:

    <out>/rgb/mango_rgb_<ts>.jpg     symlink to news/rgb_wb/ (white balance corrected) or news/rgb/
    <out>/nir/mango_nir_<ts>.png     gray NIR (B + R) / 2, warped into the RGB frame with inv(H)
                                     plus the residual shift of the photo
    <out>/nir_clahe/mango_nir_<ts>.png  the CLAHE annotation view (``enhance_nir``), same warp
    <out>/labels/mango_rgb_<ts>.txt  YOLO labels in the RGB frame: 0 mango, 1 damage
    <out>/homographies/identity.npy
    <out>/index.csv  exclusions.csv  nir_lines.csv  report.json

The photos come from ``v4_selection.select_photos``. The mango box (class 0) does not exist in
Label Studio: it is the bounding box of ``rgb_mango_mask`` on the 1/4-scale RGB; a photo where
no plausible mango is found is excluded as ``mango_mask_failed``.

NIR line artifacts (``nir_lines``) are detected on the full NIR frame. Those whose pixels,
mapped into the RGB frame, overlap the mango box are interpolated before both NIR variants are
made; the rest are left. Every line is listed in ``nir_lines.csv``.

The homography does not align the fruit, so ``residual_shift="fruit"`` adds one translation per
photo on top of it (``v4_residual``). It moves both NIR variants, the boxes drawn on the NIR and
the position where the line rule above looks; boxes drawn on the RGB and the mango box stay.
``index.csv`` holds the shift of each photo and ``report.json`` the summary per lot and moment.
``residual_shift="none"`` leaves the homography alone and skips the measurement.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.annotation.captura_v3_tasks import load_wb_timestamps
from src.annotation.mango_registration import rgb_mango_mask
from src.annotation.nir_display import enhance_nir, method_params, nir_gray
from src.data_pipeline.nir_lines import (
    HALO,
    MAX_RATIO,
    MIN_DROP,
    MIN_RUN,
    ROW_GAP,
    SMOOTH_PX,
    detect_lines,
    interpolate_lines,
    line_corners,
)
from src.data_pipeline.v4_residual import (
    NO_SHIFT,
    FruitMatch,
    ResidualShift,
    measure_fruit_match,
    residual_params,
    resolve_shifts,
)
from src.data_pipeline.v4_selection import (
    LATENT_HOURS,
    RGB_SIZE,
    VIEW_SIZES,
    Exclusion,
    Photo,
    box_in_rgb_frame,
    hour_label,
    load_homography_rules,
    lot_number,
    parse_tasks,
    select_photos,
    sin_bruise_faces,
)
from src.training.nir_registration import load_homography

MANGO_CLASS = 0
DAMAGE_CLASS = 1
MASK_DOWNSCALE = 4
MANGO_AREA_RANGE = (0.01, 0.5)  # plausible fraction of the frame covered by the mango box
NIR_BORDER_VALUE = 0  # registered NIR pixels with no source
PNG_COMPRESSION = 6
RESIDUAL_MODES = ("fruit", "none")
INDEX_FIELDS = (
    "stem",
    "timestamp",
    "lot",
    "jornada",
    "mango_id",
    "cara",
    "ronda_h",
    "zona",
    "horas_desde_impacto",
    "wb_corrected",
    "homography",
    "n_damage",
    "provenance",
    "ls_origin",
    "view",
    "rotated",
    "visible_rgb",
    "visible_nir",
    "annotation_created_at",
    "mango_x0",
    "mango_y0",
    "mango_x1",
    "mango_y1",
    "damage_x0",
    "damage_y0",
    "damage_x1",
    "damage_y1",
    "damage_in_mango",
    "nir_lines",
    "nir_lines_interpolated",
    "residual_dx",
    "residual_dy",
    "residual_corr",
    "residual_source",
)
EXCLUSION_FIELDS = (
    "timestamp",
    "lot",
    "jornada",
    "mango_id",
    "cara",
    "ronda_h",
    "reason",
    "detail",
)
NIR_LINE_FIELDS = (
    "stem",
    "timestamp",
    "lot",
    "mango_id",
    "cara",
    "ronda_h",
    "row",
    "x0",
    "x1",
    "crosses_mango",
)


def yolo_line(cls: int, box: np.ndarray) -> str:
    width, height = RGB_SIZE
    x0, y0, x1, y1 = box
    return (
        f"{cls} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
        f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}\n"
    )


def fruit_mask(rgb_path: Path) -> np.ndarray | None:
    """The mango mask of the 1/4-scale RGB (0/255), or None if implausible."""
    small = cv2.imread(str(rgb_path), cv2.IMREAD_REDUCED_COLOR_4)
    if small is None or small.shape[:2] != (RGB_SIZE[1] // 4, RGB_SIZE[0] // 4):
        raise ValueError(f"{rgb_path}: unreadable or not {RGB_SIZE[0]}x{RGB_SIZE[1]}")
    mask = rgb_mango_mask(small)
    if not mask.any():
        return None
    _, _, w, h = cv2.boundingRect(mask)
    area = w * h / (small.shape[0] * small.shape[1])
    if not MANGO_AREA_RANGE[0] <= area <= MANGO_AREA_RANGE[1]:
        return None
    return mask


def mango_box(rgb_path: Path) -> np.ndarray | None:
    """Bounding box of the mango mask in full-resolution pixels, or None if implausible."""
    mask = fruit_mask(rgb_path)
    if mask is None:
        return None
    x, y, w, h = cv2.boundingRect(mask)
    return np.array([x, y, x + w, y + h], dtype=np.float64) * MASK_DOWNSCALE


def read_nir(nir_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(nir_path), cv2.IMREAD_COLOR)
    if bgr is None or (bgr.shape[1], bgr.shape[0]) != VIEW_SIZES["nir"]:
        raise ValueError(f"{nir_path}: unreadable or not a {VIEW_SIZES['nir']} NIR image")
    return bgr


def raw_nir(bgr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(nir_gray(bgr)), 0, 255).astype(np.uint8)


def registered_nir(
    gray: np.ndarray, H_rgb_to_nir: np.ndarray, shift: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Warp a full NIR-frame gray image into the RGB frame, then move it by ``shift``.

    Both go in one matrix, so the image is resampled once.
    """
    translation = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]], dtype=np.float64)
    return cv2.warpPerspective(
        gray,
        translation @ np.linalg.inv(H_rgb_to_nir),
        RGB_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=NIR_BORDER_VALUE,
    )


def overlaps(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two [x0, y0, x1, y1] boxes share some area."""
    return bool(a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


def shifted_box(box: np.ndarray, shift: ResidualShift) -> np.ndarray:
    """Move a [x0, y0, x1, y1] box of the RGB frame by the residual shift, clipped to the image."""
    moved = box + np.array([shift.dx, shift.dy, shift.dx, shift.dy])
    return np.clip(moved, 0, np.tile(RGB_SIZE, 2))


def _rgb_source(photo: Photo, data_root: Path) -> Path:
    sub = "rgb_wb" if photo.wb_corrected else "rgb"
    return data_root / "news" / sub / f"mango_rgb_{photo.capture.timestamp}.jpg"


def measure_photo(job: tuple[Photo, Path, Path]) -> FruitMatch | None:
    """Match the fruit of one photo with its NIR warped by the homography; None if no mango."""
    photo, data_root, repo_root = job
    rgb_src = _rgb_source(photo, data_root)
    mask = fruit_mask(rgb_src)
    if mask is None:
        return None
    H = load_homography(repo_root / photo.homography)
    bgr = read_nir(data_root / "news" / "nir" / f"mango_nir_{photo.capture.timestamp}.jpg")
    rgb_gray = cv2.imread(str(rgb_src), cv2.IMREAD_GRAYSCALE)
    return measure_fruit_match(rgb_gray, registered_nir(raw_nir(bgr), H), mask)


def bake_photo(
    job: tuple[Photo, Path, Path, Path, ResidualShift],
) -> tuple[dict, list[dict]] | None:
    """Write the RGB link, registered NIRs and labels of one photo.

    Returns the index row and one row per NIR line artifact, or None if no mango is found.
    """
    photo, data_root, repo_root, out_dir, shift = job
    c, label = photo.capture, photo.label
    stem = f"mango_rgb_{c.timestamp}"
    rgb_src = _rgb_source(photo, data_root)
    H = load_homography(repo_root / photo.homography)
    mango = mango_box(rgb_src)
    if mango is None:
        return None
    lines = [yolo_line(MANGO_CLASS, mango)]
    row = {
        "stem": stem,
        "timestamp": c.timestamp,
        "lot": lot_number(c.jornada),
        "jornada": c.jornada,
        "mango_id": c.mango_id,
        "cara": c.cara,
        "ronda_h": hour_label(c.ronda_h),
        "zona": c.zona or "",
        "horas_desde_impacto": c.horas_desde_impacto,
        "wb_corrected": int(photo.wb_corrected),
        "homography": photo.homography,
        "n_damage": 0,
        "provenance": photo.provenance,
        **dict(zip(("mango_x0", "mango_y0", "mango_x1", "mango_y1"), np.round(mango, 1))),
        "residual_dx": shift.dx,
        "residual_dy": shift.dy,
        "residual_corr": shift.corr,
        "residual_source": shift.source,
    }
    if label is not None:
        boxes = [box_in_rgb_frame(corners, label.view, H) for corners in label.corners]
        if label.view == "nir":
            boxes = [shifted_box(box, shift) for box in boxes]
        if any(b[2] <= b[0] or b[3] <= b[1] for b in boxes):
            raise ValueError(f"capture {c.timestamp}: a damage box falls outside the RGB frame")
        lines += [yolo_line(DAMAGE_CLASS, b) for b in boxes]
        union = np.array([*np.min(boxes, axis=0)[:2], *np.max(boxes, axis=0)[2:]])
        inter = np.clip(
            np.minimum(union[2:], mango[2:]) - np.maximum(union[:2], mango[:2]), 0, None
        )
        row |= {
            "n_damage": len(boxes),
            "ls_origin": "|".join(sorted(set(label.origins))),
            "view": label.view,
            "rotated": int(label.rotated),
            "visible_rgb": label.visible_rgb,
            "visible_nir": label.visible_nir,
            "annotation_created_at": label.created_at,
            **dict(
                zip(
                    ("damage_x0", "damage_y0", "damage_x1", "damage_y1"),
                    np.round(union, 1),
                )
            ),
            "damage_in_mango": round(float(np.prod(inter) / np.prod(union[2:] - union[:2])), 3),
        }
    (out_dir / "rgb" / f"{stem}.jpg").symlink_to(rgb_src.resolve())
    bgr = read_nir(data_root / "news" / "nir" / f"mango_nir_{c.timestamp}.jpg")
    nir_lines = detect_lines(nir_gray(bgr))
    crossing = [
        line
        for line in nir_lines
        if overlaps(shifted_box(box_in_rgb_frame(line_corners(line), "nir", H), shift), mango)
    ]
    bgr = interpolate_lines(bgr, crossing)
    row |= {"nir_lines": len(nir_lines), "nir_lines_interpolated": len(crossing)}
    line_rows = [
        {
            **{k: row[k] for k in ("stem", "timestamp", "lot", "mango_id", "cara", "ronda_h")},
            "row": line.row,
            "x0": line.x0,
            "x1": line.x1,
            "crosses_mango": int(line in crossing),
        }
        for line in nir_lines
    ]
    # CLAHE runs on the whole NIR frame, as in the annotation view: after the warp the
    # zero border would leak into its tiles.
    for sub, gray in (("nir", raw_nir(bgr)), ("nir_clahe", enhance_nir(bgr))):
        nir_out = out_dir / sub / f"mango_nir_{c.timestamp}.png"
        nir = registered_nir(gray, H, (shift.dx, shift.dy))
        if not cv2.imwrite(str(nir_out), nir, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
            raise OSError(f"cannot write {nir_out}")
    (out_dir / "labels" / f"{stem}.txt").write_text("".join(lines), encoding="utf-8")
    return row, line_rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows({k: ("" if v is None else v) for k, v in row.items()} for row in rows)


def _map_jobs(function, jobs: list, workers: int) -> list:
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            return list(pool.map(function, jobs, chunksize=4))
    return [function(job) for job in jobs]


def _residual_shifts(
    photos: list[Photo], data_root: Path, repo_root: Path, workers: int
) -> tuple[list[ResidualShift | None], dict]:
    """The shift of every photo (None if no mango) and the report block, per lot and moment.

    Every photo is measured before any is baked: a photo that falls back needs the others.
    """
    jobs = [(photo, data_root, repo_root) for photo in photos]
    matches = _map_jobs(measure_photo, jobs, workers)
    keys = [(lot_number(p.capture.jornada), hour_label(p.capture.ronda_h)) for p in photos]
    shifts, summary = resolve_shifts(keys, matches)
    report: dict[str, dict] = {}
    for (lot, moment), values in sorted(summary.items()):
        report.setdefault(f"lot{lot}", {})[moment] = values
    return shifts, {lot: dict(sorted(group.items())) for lot, group in report.items()}


def build_dataset(
    data_root: Path,
    labels_path: Path,
    rules_path: Path,
    identity_path: Path,
    out_dir: Path,
    repo_root: Path,
    workers: int = 1,
    tolerance_path: Path | None = None,
    residual_shift: str = "fruit",
) -> dict:
    """Build the dataset under ``out_dir`` (must be new or empty) and return the report."""
    if residual_shift not in RESIDUAL_MODES:
        raise ValueError(f"residual_shift must be one of {RESIDUAL_MODES}, got {residual_shift!r}")
    data_root, out_dir = Path(data_root), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; pick a new output folder")
    metas = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((data_root / "news" / "meta").glob("mango_meta_*.json"))
    ]
    labels, conflicts = parse_tasks(json.loads(Path(labels_path).read_text(encoding="utf-8")))
    entries = yaml.safe_load(Path(identity_path).read_text(encoding="utf-8"))["exclusions"] or []
    identity = {int(e["timestamp"]): str(e["reason"]) for e in entries}
    rules = load_homography_rules(rules_path)
    tolerance_exceptions = []
    if tolerance_path is not None:
        tolerance_config = yaml.safe_load(Path(tolerance_path).read_text(encoding="utf-8"))
        tolerance_exceptions = tolerance_config["exceptions"] or []
    photos, excluded = select_photos(
        metas, labels, identity, load_wb_timestamps(data_root), rules, tolerance_exceptions
    )

    for sub in ("rgb", "nir", "nir_clahe", "labels", "homographies"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "homographies" / "identity.npy", np.eye(3))
    shifts: list[ResidualShift | None] = [NO_SHIFT] * len(photos)
    residual: dict[str, dict] = {}
    if residual_shift == "fruit":
        shifts, residual = _residual_shifts(photos, data_root, Path(repo_root), workers)
    jobs = [
        (photo, data_root, Path(repo_root), out_dir, shift or NO_SHIFT)
        for photo, shift in zip(photos, shifts)
    ]
    results = _map_jobs(bake_photo, jobs, workers)
    rows, line_rows = [], []
    for photo, result in zip(photos, results):
        if result is None:
            excluded.append(Exclusion(photo.capture, "mango_mask_failed"))
        else:
            rows.append(result[0])
            line_rows += result[1]
    excluded.sort(key=lambda e: e.capture.timestamp)

    _write_csv(out_dir / "index.csv", INDEX_FIELDS, rows)
    _write_csv(out_dir / "exclusions.csv", EXCLUSION_FIELDS, [e.row() for e in excluded])
    _write_csv(out_dir / "nir_lines.csv", NIR_LINE_FIELDS, line_rows)
    nir_line_counts = {f"lot{r['lot']}": Counter(detected=0, interpolated=0) for r in rows}
    for line_row in line_rows:
        nir_line_counts[f"lot{line_row['lot']}"]["detected"] += 1
        nir_line_counts[f"lot{line_row['lot']}"]["interpolated"] += line_row["crosses_mango"]
    included: dict[str, Counter] = {}
    for row in rows:
        included.setdefault(f"lot{row['lot']}", Counter())[row["ronda_h"]] += 1
    excluded_counts: dict[str, Counter] = {}
    for exclusion in excluded:
        lot = f"lot{lot_number(exclusion.capture.jornada)}"
        excluded_counts.setdefault(lot, Counter())[exclusion.reason] += 1
    matrices = sorted({photo.homography for photo in photos})
    report = {
        "inputs": {
            "labels": {"path": str(labels_path), "sha256": _sha256(labels_path)},
            "homography_rules": {
                "path": str(rules_path),
                "sha256": _sha256(rules_path),
            },
            "identity_exclusions": {
                "path": str(identity_path),
                "sha256": _sha256(identity_path),
            },
            "tolerance_exceptions": tolerance_exceptions,
            "matrices": {m: _sha256(Path(repo_root) / m) for m in matrices},
            "num_metas": len(metas),
        },
        "params": {
            "nir": "gray = (B + R) / 2 of the NIR JPEG, rounded to uint8, warped with inv(H) "
            f"(bilinear, border {NIR_BORDER_VALUE}), PNG",
            "nir_clahe": {
                "method": "enhance_nir (annotation view) of the full NIR frame, then the same "
                "warp as nir",
                **{k: v for k, v in method_params().items() if k != "jpeg_quality"},
            },
            "nir_lines": {
                "method": f"per pixel after a 1x{SMOOTH_PX} box filter: at least min_drop below "
                "the darker of the rows row_gap above and below, and at most max_ratio of it; "
                "runs of min_run px or more. Lines overlapping the mango box in the RGB frame "
                "get rows +- halo linearly interpolated before both NIR variants",
                "row_gap": ROW_GAP,
                "min_drop": MIN_DROP,
                "max_ratio": MAX_RATIO,
                "min_run": MIN_RUN,
                "halo": HALO,
            },
            "mango_box": f"rgb_mango_mask on the 1/{MASK_DOWNSCALE} RGB, area fraction in "
            f"{list(MANGO_AREA_RANGE)}",
            "latent_hours": list(LATENT_HOURS),
            "residual_shift": {
                "mode": residual_shift,
                **(residual_params() if residual_shift == "fruit" else {}),
            },
        },
        "num_included": len(rows),
        "num_excluded": len(excluded),
        "included": {lot: dict(sorted(c.items())) for lot, c in sorted(included.items())},
        "excluded": {lot: dict(sorted(c.items())) for lot, c in sorted(excluded_counts.items())},
        "nir_lines": {lot: dict(sorted(c.items())) for lot, c in sorted(nir_line_counts.items())},
        "residual": residual,
        "sin_bruise_faces": sorted(map(list, sin_bruise_faces(labels))),
        "annotation_conflicts": conflicts,
        "damage_mostly_outside_mango": sorted(
            r["stem"] for r in rows if r["n_damage"] and r["damage_in_mango"] < 0.5
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
