#!/usr/bin/env python3
"""Export the paired RGB+NIR dataset as ONE zip with the NIR already registered.

The training loader registers the NIR onto the RGB frame on the fly with a
single homography; external tools cannot do that, and lot 6 (2026-10-18, rebuilt
rig) needs its own matrix. This tool bakes the registration into the files:

    mango_corpus_v4/
        full_aligned/   images/{rgb,nir}/<split>/..., labels/{rgb,nir}/<split>/...  (native size)
        roi640/         the same layout as 640x640 mango crops + crop_windows.csv
        homographies/   aruco, lot 6 and identity matrices
        splits.json  manifest.csv  data_{rgb,nir}[_roi640].yaml  README.md

`full_aligned` RGB and labels are byte copies. Labels are written once per modality
(`labels/rgb/...` and `labels/nir/...`, identical content) so that Ultralytics, which finds a label by swapping
the last `/images/` of an image path for `/labels/`, works on either modality with no copy step.
Both views ship one dataset yaml per modality. The NIR is read exactly as
`YOLODataset.__getitem__` reads it (`cv2.IMREAD_GRAYSCALE`), registered with
`register_nir_to_rgb` using the matrix of its capture day and the loader's
border value (`int(nir_mean * 255)`), and saved as lossless 8-bit PNG.

`roi640` cuts the `crop_to_roi` window (margin 0.10, no jitter), grows it to a
square, shifts it back inside the frame and resizes it to 640x640. Boxes are
re-normalized to the crop and dropped when less than 50 % stays visible; the
mango box always survives.

The zip is deterministic (sorted iteration, fixed entry order and timestamps):
re-running on the same inputs gives a byte-identical file. Every input is
checked for existence before anything is written, and the zip is published
atomically, so a failure never leaves a partial archive.

Typical use (a linked worktree reads the untracked data of the primary checkout):

    ./.venv/bin/python scripts/export_aligned_corpus.py
    ./.venv/bin/python scripts/export_aligned_corpus.py --only-labelled --output /tmp/subset.zip
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_grouped_folds import (  # noqa: E402
    DEFAULT_TZ,
    SPLITS,
    Lot,
    group_into_lots,
    local_date,
    stem_timestamp,
)
from src.training.nir_registration import load_homography, register_nir_to_rgb  # noqa: E402
from src.training.roi_crop import MANGO_CLASS_ID, crop_to_roi  # noqa: E402

CORPUS_NAME = "mango_corpus_v4"
OUT_SIZE = 640
ROI_MARGIN = 0.10
MIN_VISIBLE = 0.5
JPEG_QUALITY = 95
PNG_COMPRESSION = 6
ZIP_COMPRESSLEVEL = 6
# Mirrors TrainingConfig.nir_mean / YOLODataset's default; a test pins both.
NIR_MEAN = 0.0569
ZIP_DATE_TIME = (2026, 10, 21, 0, 0, 0)
CLASS_NAMES = {0: "mango", 1: "damage"}

ARUCO_HOMOGRAPHY = "matriz_homografia_aruco.npy"
LOT6_HOMOGRAPHY = "matriz_homografia_lot6_20261018.npy"
IDENTITY_HOMOGRAPHY = "identity.npy"
LOT6_DATE = date(2026, 10, 18)  # first capture day on the rebuilt rig (America/Lima)

DEFAULT_MANIFEST = "data/annotations/yolo/splits.json"
DEFAULT_LABELS_DIR = "data/annotations/yolo/labels"
DEFAULT_RGB_DIR = "data/cache/mango/rgb"
DEFAULT_NIR_DIR = "data/cache/mango/nir"
DEFAULT_EXPORT_DIR = "data/exports"

Window = tuple[int, int, int, int]  # x0, y0, x1, y1 in native pixels, x1/y1 exclusive

_STORED_SUFFIXES = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------
# Loader parity and homography selection
# --------------------------------------------------------------------------


def nir_border_value(nir_mean: float = NIR_MEAN) -> int:
    """Fill for pixels with no NIR source: what the training loader passes."""
    return int(nir_mean * 255)


def homography_for_stem(stem: str) -> str:
    """File name of the RGB -> NIR matrix that registers this capture.

    Chosen by the capture day in America/Lima, not UTC. Lot 6 (2026-10-18) was
    shot on a rebuilt rig and has its own matrix; everything earlier uses the
    ArUco matrix. A later day has no calibrated matrix yet, which is an error
    rather than a silent fall back to the wrong one.
    """
    day = local_date(stem_timestamp(stem), DEFAULT_TZ)
    if day == LOT6_DATE:
        return LOT6_HOMOGRAPHY
    if day < LOT6_DATE:
        return ARUCO_HOMOGRAPHY
    raise ValueError(
        f"no homography is registered for {stem} (capture day {day}); calibrate the rig and "
        "add its matrix to export_aligned_corpus.homography_for_stem before exporting it."
    )


def check_lot_homography_consistency(
    lot_stems: Mapping[str, Sequence[str]], homography_of: Mapping[str, str]
) -> None:
    """Every stem of a lot must be registered with the same matrix."""
    for lot_id, stems in lot_stems.items():
        used = sorted({homography_of[s] for s in stems})
        if len(used) > 1:
            raise ValueError(
                f"{lot_id} mixes captures that need different homographies: {used}. "
                "A lot must come from one rig; check the lot grouping parameters."
            )


def assign_lots(stems: Sequence[str]) -> dict[str, Lot]:
    """Stem -> its capture lot, grouped over the WHOLE manifest (same rules as the CV folds)."""
    return {stem: lot for lot in group_into_lots(sorted(set(stems))) for stem in lot.stems}


# --------------------------------------------------------------------------
# Windows, boxes, resize
# --------------------------------------------------------------------------


def roi_window(
    boxes: np.ndarray,
    classes: np.ndarray,
    width: int,
    height: int,
    margin: float = ROI_MARGIN,
) -> Window:
    """The exact window `crop_to_roi` cuts for these labels (margin, no jitter).

    The window is read back from `crop_to_roi` itself, on two position-coded
    planes, so the margin, the integer snapping and the clipping are never
    re-implemented here and cannot drift from the training loader.
    """
    classes = np.asarray(classes)
    if not (classes == MANGO_CLASS_ID).any():
        raise ValueError(f"no class {MANGO_CLASS_ID} (mango) box: cannot define a crop window.")
    xs = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    ys = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], (height, width))
    cut_y, cut_x, _, _ = crop_to_roi(
        ys, xs, np.asarray(boxes), classes.astype(np.int64), roi_class=MANGO_CLASS_ID, margin=margin
    )
    if cut_y is ys:  # crop_to_roi gives the inputs back when the mango box is outside the frame
        raise ValueError("the mango box lies outside the image: cannot define a crop window.")
    x0, y0 = int(cut_x[0, 0]), int(cut_y[0, 0])
    return (x0, y0, x0 + cut_x.shape[1], y0 + cut_y.shape[0])


def square_window(window: Window, img_w: int, img_h: int) -> Window:
    """Grow `window` to a square of side max(w, h), centred, then shift it inside the image.

    The side is clamped to the shorter image dimension. Centring uses floor
    division on integer pixels, so the result is deterministic.
    """
    x0, y0, x1, y1 = (int(v) for v in window)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        raise ValueError(f"empty crop window {window}.")
    side = min(max(w, h), img_w, img_h)
    left = min(max((x0 + x1 - side) // 2, 0), img_w - side)
    top = min(max((y0 + y1 - side) // 2, 0), img_h - side)
    return (left, top, left + side, top + side)


@dataclass(frozen=True)
class BoxTransform:
    """Result of mapping full-frame boxes into a crop window."""

    boxes: np.ndarray  # (K, 4) kept boxes, YOLO cx,cy,w,h normalized to the window
    classes: np.ndarray  # (K,) their class ids
    kept: np.ndarray  # (N,) bool, one per input box
    visible: np.ndarray  # (N,) fraction of each input box's area inside the window


def transform_boxes(
    boxes: np.ndarray,
    classes: np.ndarray,
    img_w: int,
    img_h: int,
    window: Window,
    min_visible: float = MIN_VISIBLE,
) -> BoxTransform:
    """Clip YOLO boxes to `window` and re-normalize them to it.

    A box with less than `min_visible` of its area inside the window is dropped.
    The largest class-0 box (the one that defines the window) is never dropped.
    Because the window is resized uniformly, normalized coordinates are the same
    before and after the resize.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    classes = np.asarray(classes, dtype=np.int64).reshape(-1)
    x0, y0, x1, y1 = window
    crop_w, crop_h = x1 - x0, y1 - y0

    bx1 = (boxes[:, 0] - boxes[:, 2] / 2) * img_w - x0
    bx2 = (boxes[:, 0] + boxes[:, 2] / 2) * img_w - x0
    by1 = (boxes[:, 1] - boxes[:, 3] / 2) * img_h - y0
    by2 = (boxes[:, 1] + boxes[:, 3] / 2) * img_h - y0
    original_area = (bx2 - bx1) * (by2 - by1)

    cx1, cx2 = np.clip(bx1, 0.0, crop_w), np.clip(bx2, 0.0, crop_w)
    cy1, cy2 = np.clip(by1, 0.0, crop_h), np.clip(by2, 0.0, crop_h)
    clipped_area = (cx2 - cx1) * (cy2 - cy1)

    visible = np.divide(clipped_area, original_area, out=np.zeros_like(clipped_area), where=original_area > 0)
    kept = (clipped_area > 0.0) & (visible >= min_visible)
    mango = np.flatnonzero(classes == MANGO_CLASS_ID)
    if mango.size:
        kept[mango[np.argmax(boxes[mango, 2] * boxes[mango, 3])]] = True

    normalized = np.stack(
        [(cx1 + cx2) / 2 / crop_w, (cy1 + cy2) / 2 / crop_h, (cx2 - cx1) / crop_w, (cy2 - cy1) / crop_h],
        axis=1,
    )
    return BoxTransform(normalized[kept], classes[kept], kept, visible)


def resize_to_square(img: np.ndarray, size: int = OUT_SIZE) -> np.ndarray:
    """Resize a square crop to `size`: Lanczos4 when enlarging, area averaging when shrinking."""
    h, w = img.shape[:2]
    if h != w:
        raise ValueError(f"crop must be square, got {w}x{h}.")
    if w == size:
        return img
    interpolation = cv2.INTER_LANCZOS4 if w < size else cv2.INTER_AREA
    return cv2.resize(img, (size, size), interpolation=interpolation)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def parse_label_text(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse YOLO label text into ((N, 4) float64 boxes, (N,) int64 classes). Strict."""
    boxes: list[list[float]] = []
    classes: list[int] = []
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            if len(parts) != 5:
                raise ValueError(f"expected 5 fields, got {len(parts)}")
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
            if not all(np.isfinite([cx, cy, w, h])) or w <= 0 or h <= 0:
                raise ValueError("box must be finite with positive width and height")
        except ValueError as exc:
            raise ValueError(f"bad label at line {number} ({line!r}): {exc}") from None
        classes.append(cls)
        boxes.append([cx, cy, w, h])
    return np.array(boxes, dtype=np.float64).reshape(-1, 4), np.array(classes, dtype=np.int64)


def format_label_text(boxes: np.ndarray, classes: np.ndarray) -> str:
    """Serialize YOLO labels with six decimals, one box per line."""
    return "".join(
        f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for (cx, cy, w, h), cls in zip(boxes, classes)
    )


# --------------------------------------------------------------------------
# Deterministic zip
# --------------------------------------------------------------------------


class DeterministicZip:
    """Zip writer whose bytes depend only on the entries added, in the order added.

    Fixed timestamp, permissions and creator system; images are stored (they are
    already compressed) and everything else is deflated. Written to `<path>.part`
    and renamed on success, so an aborted export never leaves a partial zip.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._partial = self.path.with_name(self.path.name + ".part")
        self._zip: zipfile.ZipFile | None = None
        self._names: set[str] = set()

    def __enter__(self) -> "DeterministicZip":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._zip = zipfile.ZipFile(self._partial, "w", allowZip64=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._zip is not None
        self._zip.close()
        if exc_type is None:
            os.replace(self._partial, self.path)
        else:
            self._partial.unlink(missing_ok=True)

    def add_bytes(self, arcname: str, data: bytes) -> None:
        assert self._zip is not None
        if arcname in self._names:
            raise ValueError(f"duplicate zip entry: {arcname}")
        self._names.add(arcname)
        info = zipfile.ZipInfo(arcname, date_time=ZIP_DATE_TIME)
        info.compress_type = (
            zipfile.ZIP_STORED if arcname.lower().endswith(_STORED_SUFFIXES) else zipfile.ZIP_DEFLATED
        )
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        info.create_system = 3  # Unix, whatever platform runs the export
        self._zip.writestr(info, data, compresslevel=ZIP_COMPRESSLEVEL)

    def add_text(self, arcname: str, text: str) -> None:
        self.add_bytes(arcname, text.encode("utf-8"))

    def add_file(self, arcname: str, path: Path) -> None:
        self.add_bytes(arcname, Path(path).read_bytes())


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


class MissingInputsError(FileNotFoundError):
    """Input files the export needs do not exist. Raised before anything is written."""

    def __init__(self, missing: Sequence[Path]) -> None:
        self.missing = list(missing)
        listing = "\n".join(f"  {path}" for path in self.missing)
        super().__init__(f"{len(self.missing)} input file(s) are missing; nothing was written:\n{listing}")


@dataclass(frozen=True)
class Pair:
    stem: str
    split: str
    rgb_path: Path
    nir_path: Path
    label_path: Path

    @property
    def nir_name(self) -> str:
        return self.nir_path.stem


def primary_checkout_root(repo_root: Path | str) -> Path:
    """Root of the primary checkout, where the untracked `data/` lives.

    In a linked worktree `git rev-parse --git-common-dir` points at
    `<primary>/.git`; in the primary checkout (or without git) it is `repo_root`.
    """
    repo_root = Path(repo_root).resolve()
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return repo_root
    common = Path(out).resolve()
    return common.parent if common.name == ".git" else repo_root


def load_manifest(path: Path | str) -> dict[str, list[str]]:
    """Read `splits.json`: exactly train/val/test, each stem in exactly one split once."""
    manifest = json.loads(Path(path).read_text())
    unknown = sorted(set(manifest) - set(SPLITS))
    if unknown or set(manifest) != set(SPLITS):
        raise ValueError(f"{path} must have exactly the splits {list(SPLITS)}; unexpected keys: {unknown}.")
    seen: dict[str, str] = {}
    for split in SPLITS:
        for stem in manifest[split]:
            if stem in seen:
                raise ValueError(f"stem {stem} appears twice in {path} ({seen[stem]} and {split}).")
            if "_rgb_" not in stem:
                raise ValueError(f"stem {stem!r} in {path} is not a mango_rgb_<timestamp> stem.")
            seen[stem] = split
    return {split: list(manifest[split]) for split in SPLITS}


def resolve_pairs(
    manifest: Mapping[str, Sequence[str]],
    *,
    rgb_dir: Path,
    nir_dir: Path,
    labels_dir: Path,
    only_labelled: bool = False,
) -> tuple[list[Pair], list[str]]:
    """Locate every pair's files, ordered by (split, stem), failing on any missing file.

    Returns (pairs, skipped_stems). With `only_labelled`, a stem whose label file
    is absent is skipped (and returned in `skipped_stems`); a missing image is
    always an error.
    """
    pairs: list[Pair] = []
    skipped: list[str] = []
    missing: list[Path] = []
    for split in SPLITS:
        for stem in sorted(manifest[split]):
            rgb = Path(rgb_dir) / f"{stem}.jpg"
            nir = Path(nir_dir) / f"{stem.replace('_rgb_', '_nir_')}.jpg"
            label = Path(labels_dir) / split / f"{stem}.txt"
            absent = [p for p in (rgb, nir) if not p.is_file()]
            if not label.is_file():
                if only_labelled:
                    skipped.append(stem)
                else:
                    absent.append(label)
            missing.extend(absent)
            if label.is_file() or not only_labelled:
                pairs.append(Pair(stem, split, rgb, nir, label))
    if missing:
        raise MissingInputsError(missing)
    return pairs, sorted(skipped)


def default_output_path(data_root: Path, n_pairs: int) -> Path:
    return Path(data_root) / DEFAULT_EXPORT_DIR / f"{CORPUS_NAME}_{n_pairs}pairs.zip"


# --------------------------------------------------------------------------
# Per-pair processing
# --------------------------------------------------------------------------


@dataclass
class PairRecord:
    stem: str
    split: str
    lot: str
    capture_date: str
    homography: str
    full_counts: dict[int, int]
    roi_counts: dict[int, int]
    window: Window
    dropped: list[tuple[int, float]] = field(default_factory=list)  # (class id, visible fraction)

    @property
    def side(self) -> int:
        return self.window[2] - self.window[0]


def _encode(ext: str, image: np.ndarray, params: list[int]) -> bytes:
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"could not encode a {ext} image")
    return buffer.tobytes()


def _png(image: np.ndarray) -> bytes:
    return _encode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])


def _jpeg(image: np.ndarray) -> bytes:
    return _encode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def _count(classes: np.ndarray) -> dict[int, int]:
    return {cls: int((classes == cls).sum()) for cls in CLASS_NAMES}


def export_pair(
    pair: Pair,
    lot: Lot,
    homography_name: str,
    homography: np.ndarray,
    archive: DeterministicZip,
) -> PairRecord:
    """Register one pair, write its full_aligned and roi640 entries, return its bookkeeping."""
    rgb_bytes = pair.rgb_path.read_bytes()
    label_text = pair.label_path.read_text()
    # Same reads as YOLODataset.__getitem__ (default colour read; NIR as grayscale).
    rgb = cv2.imread(str(pair.rgb_path))
    nir = cv2.imread(str(pair.nir_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        raise ValueError(f"{pair.stem}: cannot decode RGB image {pair.rgb_path}")
    if nir is None:
        raise ValueError(f"{pair.stem}: cannot decode NIR image {pair.nir_path}")
    height, width = rgb.shape[:2]
    if nir.shape != (height, width):
        raise ValueError(
            f"{pair.stem}: NIR is {nir.shape[1]}x{nir.shape[0]} but RGB is {width}x{height}; "
            "the homography relates frames of the same size."
        )
    boxes, classes = parse_label_text(label_text)
    if not (classes == MANGO_CLASS_ID).any():
        raise ValueError(f"{pair.stem}: label file {pair.label_path} has no class 0 (mango) box.")

    registered = register_nir_to_rgb(nir, homography, (width, height), nir_border_value())

    window = square_window(roi_window(boxes, classes, width, height), width, height)
    x0, y0, x1, y1 = window
    moved = transform_boxes(boxes, classes, width, height, window)
    roi_rgb = resize_to_square(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
    roi_nir = resize_to_square(np.ascontiguousarray(registered[y0:y1, x0:x1]))

    nir_name = pair.nir_name
    split = pair.split
    archive.add_bytes(f"{CORPUS_NAME}/full_aligned/images/rgb/{split}/{pair.stem}.jpg", rgb_bytes)
    archive.add_bytes(f"{CORPUS_NAME}/full_aligned/images/nir/{split}/{nir_name}.png", _png(registered))
    # The NIR is registered to the RGB frame, so one label text serves both modalities.
    archive.add_text(f"{CORPUS_NAME}/full_aligned/labels/rgb/{split}/{pair.stem}.txt", label_text)
    archive.add_text(f"{CORPUS_NAME}/full_aligned/labels/nir/{split}/{nir_name}.txt", label_text)
    archive.add_bytes(f"{CORPUS_NAME}/roi640/images/rgb/{split}/{pair.stem}.jpg", _jpeg(roi_rgb))
    archive.add_bytes(f"{CORPUS_NAME}/roi640/images/nir/{split}/{nir_name}.png", _png(roi_nir))
    roi_label_text = format_label_text(moved.boxes, moved.classes)
    archive.add_text(f"{CORPUS_NAME}/roi640/labels/rgb/{split}/{pair.stem}.txt", roi_label_text)
    archive.add_text(f"{CORPUS_NAME}/roi640/labels/nir/{split}/{nir_name}.txt", roi_label_text)

    return PairRecord(
        stem=pair.stem,
        split=split,
        lot=lot.lot_id,
        capture_date=local_date(stem_timestamp(pair.stem), DEFAULT_TZ).isoformat(),
        homography=homography_name,
        full_counts=_count(classes),
        roi_counts=_count(moved.classes),
        window=window,
        dropped=[(int(classes[i]), float(moved.visible[i])) for i in np.flatnonzero(~moved.kept)],
    )


# --------------------------------------------------------------------------
# Shared files
# --------------------------------------------------------------------------


def _csv_text(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def manifest_csv(records: Sequence[PairRecord]) -> str:
    return _csv_text(
        ["stem", "split", "lot", "capture_local_date", "homography_file", "n_mango", "n_damage"],
        [
            [r.stem, r.split, r.lot, r.capture_date, f"homographies/{r.homography}",
             r.full_counts[0], r.full_counts[1]]
            for r in records
        ],
    )


def crop_windows_csv(records: Sequence[PairRecord]) -> str:
    return _csv_text(
        ["stem", "split", "lot", "x0", "y0", "x1", "y1", "side_px", "scale_to_640"],
        [
            [r.stem, r.split, r.lot, *r.window, r.side, f"{OUT_SIZE / r.side:.6f}"]
            for r in records
        ],
    )


def dataset_yaml(folder: str, modality: str) -> str:
    """Ultralytics dataset file for one view (`folder`) and one modality (`rgb` or `nir`).

    `path` is omitted so Ultralytics resolves the entries relative to the yaml. The labels
    sit at `<folder>/labels/<modality>/<split>/`, which is where Ultralytics looks for the
    images in `<folder>/images/<modality>/<split>/`.
    """
    return (
        f"# Ultralytics dataset definition: {modality.upper()} images of {folder}.\n"
        "# `path` is deliberately omitted so the entries resolve relative to this file.\n"
        f"# Labels are read from {folder}/labels/{modality}/<split>/ (images -> labels substitution).\n"
        f"train: {folder}/images/{modality}/train\n"
        f"val: {folder}/images/{modality}/val\n"
        f"test: {folder}/images/{modality}/test\n"
        "names:\n"
        + "".join(f"  {cls}: {name}\n" for cls, name in CLASS_NAMES.items())
    )


def identity_npy_bytes() -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.eye(3, dtype=np.float64))
    return buffer.getvalue()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _totals(records: Sequence[PairRecord]) -> dict[str, dict[int, int]]:
    out = {"full": {c: 0 for c in CLASS_NAMES}, "roi": {c: 0 for c in CLASS_NAMES}}
    for r in records:
        for cls in CLASS_NAMES:
            out["full"][cls] += r.full_counts[cls]
            out["roi"][cls] += r.roi_counts[cls]
    return out


def render_readme(
    records: Sequence[PairRecord],
    lots: Mapping[str, Lot],
    hashes: Mapping[str, str],
    skipped: Sequence[str] = (),
) -> str:
    n = len(records)
    by_split = {s: sum(1 for r in records if r.split == s) for s in SPLITS}
    totals = _totals(records)
    dropped = [(r.stem, cls, vis) for r in records for cls, vis in r.dropped]
    lot_ids = sorted({r.lot for r in records}, key=lambda x: int(x[3:]))
    lot_rows = []
    for lot_id in lot_ids:
        members = [r for r in records if r.lot == lot_id]
        counts = " | ".join(str(sum(1 for r in members if r.split == s)) for s in SPLITS)
        days = ", ".join(lots[members[0].stem].days)
        lot_rows.append(f"| {lot_id} | {days} | `{members[0].homography}` | {len(members)} | {counts} |")
    windows = [r for r in records]
    sides = [r.side for r in windows]
    scales = [OUT_SIZE / s for s in sides]
    dropped_text = (
        "none"
        if not dropped
        else "; ".join(f"{stem} class {cls} ({vis:.0%} visible)" for stem, cls, vis in dropped)
    )
    border = nir_border_value()
    subset_note = (
        f"\n> **Subset.** Built with `--only-labelled`: {len(skipped)} stem(s) without a label file were left out "
        f"({', '.join(skipped)}), and `splits.json` lists only what is in this zip.\n"
        if skipped
        else ""
    )

    return f"""# {CORPUS_NAME}

Paired RGB + NIR captures of mangoes with bounding boxes for the mango (class 0) and its
bruises (class 1, "damage"). **The NIR images are ALREADY registered onto the RGB frame.**
{subset_note}
- Pairs: **{n}** (train {by_split['train']}, val {by_split['val']}, test {by_split['test']}).
- Capture lots: {len(lot_ids)} (table below). Every lot used new mangos; split by lot, not by image, when you
  need an honest estimate.
- Two views of the same pairs: `full_aligned/` (native resolution) and `roi640/` (mango crops, 640x640).
- Produced by `scripts/export_aligned_corpus.py` of the project repository. The zip is deterministic:
  re-running the tool on the same inputs gives a byte-identical file.

## Contents

```
{CORPUS_NAME}/
  full_aligned/
    images/rgb/<split>/mango_rgb_<ts>.jpg   byte copies of the original RGB captures
    images/nir/<split>/mango_nir_<ts>.png   NIR registered onto the RGB frame (8-bit, 1 channel, lossless)
    labels/rgb/<split>/mango_rgb_<ts>.txt   YOLO labels in the RGB frame (byte copies of the originals)
    labels/nir/<split>/mango_nir_<ts>.txt   the same bytes: the NIR is registered, so the labels are valid for both
  roi640/
    images/rgb/<split>/mango_rgb_<ts>.jpg   640x640 crop around the mango (JPEG quality 95)
    images/nir/<split>/mango_nir_<ts>.png   the same window of the registered NIR (8-bit, 1 channel)
    labels/rgb/<split>/mango_rgb_<ts>.txt   labels re-normalized to the crop
    labels/nir/<split>/mango_nir_<ts>.txt   the same bytes (same window for both modalities)
    crop_windows.csv                        per image: x0,y0,x1,y1 (native px), side_px, scale_to_640
  homographies/                             matriz_homografia_aruco.npy, matriz_homografia_lot6_20261018.npy, identity.npy
  splits.json                               train/val/test stems (copy of the project manifest)
  manifest.csv                              stem, split, lot, capture_local_date, homography_file, n_mango, n_damage
  data_rgb.yaml, data_nir.yaml              Ultralytics dataset files for full_aligned (one per modality)
  data_rgb_roi640.yaml, data_nir_roi640.yaml   the same for roi640
  README.md
```

Classes: `0 = mango`, `1 = damage`. Labels are YOLO `class cx cy w h`, normalized to the image they sit next to.
Every image has its label at the path Ultralytics derives by replacing the last `/images/` of the image path with
`/labels/` (`images/rgb/<split>/x.jpg` -> `labels/rgb/<split>/x.txt`, `images/nir/<split>/y.png` -> `labels/nir/<split>/y.txt`),
so the zip can be used by Ultralytics as it is.

### Counts

| | full_aligned | roi640 |
|---|---|---|
| mango boxes (class 0) | {totals['full'][0]} | {totals['roi'][0]} |
| damage boxes (class 1) | {totals['full'][1]} | {totals['roi'][1]} |

Boxes dropped by the 50 % visibility rule in `roi640`: {dropped_text}.

### Lots

| lot | capture days (America/Lima) | homography | pairs | train | val | test |
|---|---|---|---|---|---|---|
{chr(10).join(lot_rows)}

Lots are consecutive capture days (gap of at most 1 day, lots under 5 pairs merged into their nearest
neighbour), numbered chronologically; they are the lots the project's leave-one-lot-out folds are built from
(`scripts/make_grouped_folds.py`).

## NIR registration

The two sensors sit side by side, so a raw NIR frame is offset from the RGB frame by about (-120, +110) px.
Labels are expressed in the RGB frame, so the NIR has to be brought into it before anything else.

- **Convention.** Each matrix `H` is a 3x3 homography that maps RGB pixels to NIR pixels
  (`cv2.findHomography(pts_rgb, pts_nir)`). A NIR image is resampled into the RGB frame with
  `cv2.warpPerspective(nir, inv(H), (rgb_width, rgb_height), flags=cv2.INTER_LINEAR,
  borderMode=cv2.BORDER_CONSTANT, borderValue={border})`. This is `register_nir_to_rgb` in the repository.
- **Already applied.** The files in this zip are the result of that warp. Do not warp them again.
  Pixels that had no NIR source are filled with {border} (the training loader's NIR pad value, `int(0.0569 * 255)`);
  no validity mask is shipped, and a real dark pixel can also be {border}.
- **Which matrix.** Lots 1-5 use `matriz_homografia_aruco.npy` (estimated from ArUco markers).
  Lot 6 (2026-10-18, America/Lima) uses `matriz_homografia_lot6_20261018.npy`: the rig was rebuilt, and on it
  the registered NIR lands about 4 px right and 9 px below the RGB content with the ArUco matrix. The lot-6 matrix is `H_aruco` composed with a single
  integer translation of (+4, +9) px, estimated from the 16 lot-6 captures; its damage labels were projected to the
  RGB with the same matrix. Details and the evidence are in `docs/findings/2026-10-20/lot6-intake.md` of the
  repository. It is one translation per lot, so a residual of a few pixels remains. Lot 5 showed a residual of
  about 10 px with the ArUco matrix; it is not corrected here. The matrix used by every pair is in `manifest.csv`.
- **Hashes (SHA-256 of the shipped files).**
{chr(10).join(f"  - `{name}`: `{digest}`" for name, digest in hashes.items())}

### NIR image format

The source NIR captures are 3-channel JPEGs. The corpus NIR is what the training loader sees:
`cv2.imread(path, cv2.IMREAD_GRAYSCALE)` (OpenCV's luma conversion), then registered, then written as a lossless
**8-bit single-channel PNG**. The JPEG loss of the original capture is not undone, and the bilinear resampling
softens the image slightly; both are inherent to what the model was trained on.

Ultralytics reads images with OpenCV in colour mode, so it loads these 1-channel PNGs as 3 channels (the grey value
repeated). Train an NIR model with `data_nir*.yaml` as a 3-channel input, or set up the channel handling yourself if you
need a true 1-channel input.

## roi640

Built from the label file of each image:

1. Start from the window `crop_to_roi` cuts (`src/training/roi_crop.py`): the largest class-0 box plus 10 % of its
   width/height on each side, snapped outward to integer pixels and clipped to the frame. No jitter.
2. Expand it to a **square** of side `max(w, h)`, centred on that window's centre (integer floor division).
3. Shift it to stay inside the image. If the side exceeds the shorter image dimension, clamp the side to it
   (`side_px` equal to the shorter dimension in `crop_windows.csv` marks a clamped window).
4. Cut the same window from the RGB and from the registered NIR.
5. Resize to 640x640: `cv2.INTER_LANCZOS4` when enlarging (side < 640), `cv2.INTER_AREA` when shrinking, none at 640.
6. Boxes are clipped to the window and re-normalized to it; a box with less than 50 % of its area inside is dropped
   (`min_visible=0.5`, as in `crop_to_roi`). The largest mango box always survives, even if clipped.

Scale: windows have a side of {min(sides)}-{max(sides)} px, so `scale_to_640` runs from {min(scales):.2f} to {max(scales):.2f}
(see `crop_windows.csv`). The mango gets the same scale as the loader's `crop_to_roi` followed by its letterbox
(`640 / max(w, h)`), but the strip the loader would fill with grey padding (114 in RGB, {border} in NIR) is real image
context here. The two differ only when the side had to be clamped.

## Which folder to use

- **`roi640/`** is for external YOLO tools (Ultralytics and the like): the mango is already cropped to a fixed
  640x640 window, with no online margin jitter, so the training crop does not vary between epochs.
- **`full_aligned/`** is for pipelines that crop online, like this repository with `roi_crop: true` (margin jitter on
  the train split).
- **Never feed `roi640/` to a loader that also crops**: it would crop twice.

## Using this corpus

**With Ultralytics (no copy step).** Point the trainer at one of the four yamls, which sit next to `full_aligned/` and
`roi640/`:

| yaml | images | labels |
|---|---|---|
| `data_rgb.yaml` | `full_aligned/images/rgb/<split>` | `full_aligned/labels/rgb/<split>` |
| `data_nir.yaml` | `full_aligned/images/nir/<split>` | `full_aligned/labels/nir/<split>` |
| `data_rgb_roi640.yaml` | `roi640/images/rgb/<split>` | `roi640/labels/rgb/<split>` |
| `data_nir_roi640.yaml` | `roi640/images/nir/<split>` | `roi640/labels/nir/<split>` |

`path` is omitted on purpose, so the entries resolve relative to the yaml; keep the yamls where they are. Class names are
`{{0: mango, 1: damage}}`. The NIR PNGs are single-channel and Ultralytics loads them as 3 channels (see above).

**With this repository's loader (`YOLODataset`).**

- Set `nir_homography_path` to `homographies/identity.npy`. The loader warps every NIR by `inv(H)` whenever a
  homography path is set, so pointing it at an ArUco or lot-6 matrix would warp the already registered NIR a second time.
- The loader crops online (`roi_crop: true`, margin 0.10, with per-side margin jitter on the train split), so it should
  read `full_aligned/`, never `roi640/`.
- Directories per split: `rgb_dir=full_aligned/images/rgb/<split>`, `nir_dir=full_aligned/images/nir/<split>`,
  `labels_dir=full_aligned/labels/rgb` (the loader appends the split).
- **Known gap.** At the commit that produced this corpus, `YOLODataset._load_pairs` looks for the NIR file under the RGB
  file name with the `.jpg` extension and skips (with a printed warning) every pair whose NIR it does not find. This corpus
  stores the NIR as PNG, so the loader needs a small extension (look for `.png`) before it can read these files.
"""


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


@dataclass
class ExportReport:
    output: Path
    records: list[PairRecord]
    skipped: list[str]
    lots: dict[str, Lot]
    size_bytes: int
    sha256: str


def export_corpus(
    *,
    manifest_path: Path,
    rgb_dir: Path,
    nir_dir: Path,
    labels_dir: Path,
    homography_dir: Path,
    output: Path | None,
    data_root: Path,
    only_labelled: bool = False,
    progress=None,
) -> ExportReport:
    """Build the corpus zip. Every input is verified before the first byte is written."""
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    all_stems = [s for split in SPLITS for s in manifest[split]]
    lots = assign_lots(all_stems)
    homography_of = {s: homography_for_stem(s) for s in all_stems}
    lot_members: dict[str, list[str]] = {}
    for stem, lot in lots.items():
        lot_members.setdefault(lot.lot_id, []).append(stem)
    check_lot_homography_consistency(lot_members, homography_of)

    pairs, skipped = resolve_pairs(
        manifest, rgb_dir=rgb_dir, nir_dir=nir_dir, labels_dir=labels_dir, only_labelled=only_labelled
    )
    shipped = {name: Path(homography_dir) / name for name in (ARUCO_HOMOGRAPHY, LOT6_HOMOGRAPHY)}
    absent = [p for p in shipped.values() if not p.is_file()]
    if absent:
        raise MissingInputsError(absent)
    matrices = {name: load_homography(path) for name, path in shipped.items()}
    if not pairs:
        raise ValueError("no pairs to export.")

    output = Path(output) if output is not None else default_output_path(data_root, len(pairs))
    records: list[PairRecord] = []
    identity = identity_npy_bytes()
    hashes = {
        **{f"homographies/{name}": _sha256(path.read_bytes()) for name, path in shipped.items()},
        f"homographies/{IDENTITY_HOMOGRAPHY}": _sha256(identity),
    }

    with DeterministicZip(output) as archive:
        for index, pair in enumerate(pairs, start=1):
            name = homography_of[pair.stem]
            records.append(export_pair(pair, lots[pair.stem], name, matrices[name], archive))
            if progress is not None:
                progress(index, len(pairs), pair.stem)

        root = CORPUS_NAME
        archive.add_text(f"{root}/README.md", render_readme(records, lots, hashes, skipped))
        archive.add_text(f"{root}/manifest.csv", manifest_csv(records))
        archive.add_text(f"{root}/roi640/crop_windows.csv", crop_windows_csv(records))
        if skipped:
            exported = {s: [r.stem for r in records if r.split == s] for s in SPLITS}
            archive.add_text(f"{root}/splits.json", json.dumps(exported, indent=2) + "\n")
        else:
            archive.add_bytes(f"{root}/splits.json", manifest_path.read_bytes())
        for modality in ("rgb", "nir"):
            archive.add_text(f"{root}/data_{modality}.yaml", dataset_yaml("full_aligned", modality))
            archive.add_text(f"{root}/data_{modality}_roi640.yaml", dataset_yaml("roi640", modality))
        for name, path in shipped.items():
            archive.add_file(f"{root}/homographies/{name}", path)
        archive.add_bytes(f"{root}/homographies/{IDENTITY_HOMOGRAPHY}", identity)

    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return ExportReport(output, records, skipped, lots, output.stat().st_size, digest.hexdigest())


# --------------------------------------------------------------------------
# Summary and CLI
# --------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    cells = [[str(c) for c in row] for row in [header, *rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    return [
        "  " + "  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths)))
        for row in cells
    ]


def format_summary(report: ExportReport) -> str:
    records = report.records
    header = ["", "pairs", "full:mango", "full:damage", "roi:mango", "roi:damage", "dropped"]

    def row(label: str, members: Sequence[PairRecord]) -> list[object]:
        t = _totals(members)
        return [
            label, len(members), t["full"][0], t["full"][1], t["roi"][0], t["roi"][1],
            sum(len(r.dropped) for r in members),
        ]

    lines = [
        f"Wrote {report.output}",
        f"  {len(records)} pairs, {report.size_bytes / 2**20:.1f} MiB, sha256 {report.sha256}",
    ]
    if report.skipped:
        lines.append(f"  skipped {len(report.skipped)} stem(s) without a label file (--only-labelled):")
        lines.extend(f"    {stem}" for stem in report.skipped)
    lines += ["", "Per split:"]
    split_rows = [row(s, [r for r in records if r.split == s]) for s in SPLITS]
    lines += _table(header, [*split_rows, row("total", records)])
    lot_rows = []
    for lot_id in sorted({r.lot for r in records}, key=lambda x: int(x[3:])):
        members = [r for r in records if r.lot == lot_id]
        lot_rows.append([*row(lot_id, members), ",".join(report.lots[members[0].stem].days)])
    lines += ["", "Per lot:"]
    lines += _table([*header, "days"], lot_rows)
    dropped = [(r.stem, cls, vis) for r in records for cls, vis in r.dropped]
    lines += ["", f"Boxes dropped in roi640 (under {MIN_VISIBLE:.0%} visible): {len(dropped)}"]
    lines.extend(f"  {stem}  class {cls}  {vis:.1%} visible" for stem, cls, vis in dropped)
    sides = [r.side for r in records]
    scales = f"{OUT_SIZE / max(sides):.2f}-{OUT_SIZE / min(sides):.2f}"
    lines += ["", f"Crop side: {min(sides)}-{max(sides)} px (scale to 640: {scales})"]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the RGB+NIR dataset as one deterministic zip with the NIR registered onto the RGB frame.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="checkout that holds the untracked data/ directory (default: the primary checkout of this repository)",
    )
    for flag, default, what in (
        ("--manifest", DEFAULT_MANIFEST, "splits.json"),
        ("--rgb-dir", DEFAULT_RGB_DIR, "RGB images"),
        ("--nir-dir", DEFAULT_NIR_DIR, "raw NIR images"),
        ("--labels-dir", DEFAULT_LABELS_DIR, "YOLO labels, <split>/<stem>.txt"),
    ):
        parser.add_argument(flag, type=Path, default=None, help=f"{what} (default: <data-root>/{default})")
    parser.add_argument(
        "--homography-dir", type=Path, default=PROJECT_ROOT / "notebooks",
        help="folder with the aruco and lot-6 matrices (default: this repository's notebooks/)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"zip to write (default: <data-root>/{DEFAULT_EXPORT_DIR}/{CORPUS_NAME}_<N>pairs.zip, N = pairs exported)",
    )
    parser.add_argument(
        "--only-labelled", action="store_true",
        help="skip stems whose label file does not exist yet (listed in the summary) instead of failing; "
        "lots and homographies are still derived from the whole manifest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = (args.data_root or primary_checkout_root(PROJECT_ROOT)).resolve()

    def progress(done: int, total: int, stem: str) -> None:
        if done % 25 == 0 or done == total:
            print(f"[{done}/{total}] {stem}", file=sys.stderr)

    try:
        report = export_corpus(
            manifest_path=args.manifest or data_root / DEFAULT_MANIFEST,
            rgb_dir=args.rgb_dir or data_root / DEFAULT_RGB_DIR,
            nir_dir=args.nir_dir or data_root / DEFAULT_NIR_DIR,
            labels_dir=args.labels_dir or data_root / DEFAULT_LABELS_DIR,
            homography_dir=args.homography_dir,
            output=args.output,
            data_root=data_root,
            only_labelled=args.only_labelled,
            progress=progress,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_summary(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
