"""captura_v3 photos + Label Studio damage boxes -> the v4 training dataset.

The training loader registers the NIR onto the RGB frame with ONE homography, but every
captura_v3 lot has its own. As in ``scripts/export_aligned_corpus.py``, the registration is
baked into the files, so training runs with ``nir_homography_path=<out>/homographies/identity.npy``:

    <out>/rgb/mango_rgb_<ts>.jpg     symlink to news/rgb_wb/ (white balance corrected) or news/rgb/
    <out>/nir/mango_nir_<ts>.png     gray NIR (B + R) / 2, warped into the RGB frame with inv(H)
    <out>/nir_clahe/mango_nir_<ts>.png  the CLAHE annotation view (``enhance_nir``), same warp
    <out>/labels/mango_rgb_<ts>.txt  YOLO labels in the RGB frame: 0 mango, 1 damage
    <out>/homographies/identity.npy
    <out>/index.csv  exclusions.csv  report.json

The photos come from ``v4_selection.select_photos``. The mango box (class 0) does not exist in
Label Studio: it is the bounding box of ``rgb_mango_mask`` on the 1/4-scale RGB; a photo where
no plausible mango is found is excluded as ``mango_mask_failed``.
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
from src.data_pipeline.v4_selection import (
    DRAWN_FROM_HOUR,
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
    "task_id",
    "annotation_id",
    "mango_x0",
    "mango_y0",
    "mango_x1",
    "mango_y1",
    "damage_x0",
    "damage_y0",
    "damage_x1",
    "damage_y1",
    "damage_in_mango",
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


def yolo_line(cls: int, box: np.ndarray) -> str:
    width, height = RGB_SIZE
    x0, y0, x1, y1 = box
    return (
        f"{cls} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
        f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}\n"
    )


def mango_box(rgb_path: Path) -> np.ndarray | None:
    """Bounding box of the mango mask in full-resolution pixels, or None if implausible."""
    small = cv2.imread(str(rgb_path), cv2.IMREAD_REDUCED_COLOR_4)
    if small is None or small.shape[:2] != (RGB_SIZE[1] // 4, RGB_SIZE[0] // 4):
        raise ValueError(f"{rgb_path}: unreadable or not {RGB_SIZE[0]}x{RGB_SIZE[1]}")
    mask = rgb_mango_mask(small)
    if not mask.any():
        return None
    x, y, w, h = cv2.boundingRect(mask)
    area = w * h / (small.shape[0] * small.shape[1])
    if not MANGO_AREA_RANGE[0] <= area <= MANGO_AREA_RANGE[1]:
        return None
    return np.array([x, y, x + w, y + h], dtype=np.float64) * MASK_DOWNSCALE


def read_nir(nir_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(nir_path), cv2.IMREAD_COLOR)
    if bgr is None or (bgr.shape[1], bgr.shape[0]) != VIEW_SIZES["nir"]:
        raise ValueError(f"{nir_path}: unreadable or not a {VIEW_SIZES['nir']} NIR image")
    return bgr


def raw_nir(bgr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(nir_gray(bgr)), 0, 255).astype(np.uint8)


def registered_nir(gray: np.ndarray, H_rgb_to_nir: np.ndarray) -> np.ndarray:
    """Warp a full NIR-frame gray image into the RGB frame."""
    return cv2.warpPerspective(
        gray,
        np.linalg.inv(H_rgb_to_nir),
        RGB_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=NIR_BORDER_VALUE,
    )


def bake_photo(job: tuple[Photo, Path, Path, Path]) -> dict | None:
    """Write the RGB link, registered NIRs and labels of one photo; None if no mango is found."""
    photo, data_root, repo_root, out_dir = job
    c, label = photo.capture, photo.label
    stem = f"mango_rgb_{c.timestamp}"
    rgb_src = data_root / "news" / ("rgb_wb" if photo.wb_corrected else "rgb") / f"{stem}.jpg"
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
    }
    if label is not None:
        boxes = [box_in_rgb_frame(corners, label.view, H) for corners in label.corners]
        if any(b[2] <= b[0] or b[3] <= b[1] for b in boxes):
            raise ValueError(f"task {label.task_id}: a damage box falls outside the RGB frame")
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
            "task_id": label.task_id,
            "annotation_id": label.annotation_id,
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
    # CLAHE runs on the whole NIR frame, as in the annotation view: after the warp the
    # zero border would leak into its tiles.
    for sub, gray in (("nir", raw_nir(bgr)), ("nir_clahe", enhance_nir(bgr))):
        nir_out = out_dir / sub / f"mango_nir_{c.timestamp}.png"
        nir = registered_nir(gray, H)
        if not cv2.imwrite(str(nir_out), nir, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
            raise OSError(f"cannot write {nir_out}")
    (out_dir / "labels" / f"{stem}.txt").write_text("".join(lines), encoding="utf-8")
    return row


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows({k: ("" if v is None else v) for k, v in row.items()} for row in rows)


def build_dataset(
    data_root: Path,
    labels_path: Path,
    rules_path: Path,
    identity_path: Path,
    out_dir: Path,
    repo_root: Path,
    workers: int = 1,
) -> dict:
    """Build the dataset under ``out_dir`` (must be new or empty) and return the report."""
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
    photos, excluded = select_photos(metas, labels, identity, load_wb_timestamps(data_root), rules)

    for sub in ("rgb", "nir", "nir_clahe", "labels", "homographies"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "homographies" / "identity.npy", np.eye(3))
    jobs = [(photo, data_root, Path(repo_root), out_dir) for photo in photos]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(bake_photo, jobs, chunksize=4))
    else:
        results = [bake_photo(job) for job in jobs]
    rows = []
    for photo, row in zip(photos, results):
        if row is None:
            excluded.append(Exclusion(photo.capture, "mango_mask_failed"))
        else:
            rows.append(row)
    excluded.sort(key=lambda e: e.capture.timestamp)

    _write_csv(out_dir / "index.csv", INDEX_FIELDS, rows)
    _write_csv(out_dir / "exclusions.csv", EXCLUSION_FIELDS, [e.row() for e in excluded])
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
            "mango_box": f"rgb_mango_mask on the 1/{MASK_DOWNSCALE} RGB, area fraction in "
            f"{list(MANGO_AREA_RANGE)}",
            "latent_hours": list(LATENT_HOURS),
            "drawn_from_hour": DRAWN_FROM_HOUR,
        },
        "num_included": len(rows),
        "num_excluded": len(excluded),
        "included": {lot: dict(sorted(c.items())) for lot, c in sorted(included.items())},
        "excluded": {lot: dict(sorted(c.items())) for lot, c in sorted(excluded_counts.items())},
        "sin_bruise_faces": sorted(map(list, sin_bruise_faces(labels))),
        "annotation_conflicts": conflicts,
        "damage_mostly_outside_mango": sorted(
            r["stem"] for r in rows if r["n_damage"] and r["damage_in_mango"] < 0.5
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
