"""Read-only +12 h reference images for the Label Studio tasks of the earlier rounds.

The tasks of the +10 h and earlier rounds show, next to the photo to annotate, the same fruit face at +12 h
with the hand-drawn damage box on it (``data.ref``). Tasks annotated without one (the lot 3 +12 h and +10 h
rounds) point at a gray "No reference" placeholder.

Everything is rebuilt from a Label Studio JSON export: the reference of a face is the image its +12 h task
displays (``data.nir`` or ``data.rgb``, whichever modality the single box was drawn on) with that box
outlined, saved as a JPEG with the same width and height. Only
``<document-root>/cache/captura_v3/news/ref_12h`` is ever written.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

import cv2
import numpy as np

from .box_projection import (
    FROM_NAME_MODALITY,
    REFERENCE_HOUR,
    Box,
    Face,
    percent_to_box,
)

URL_PREFIX = "/data/local-files/?d="
REF_DIR = Path("cache/captura_v3/news/ref_12h")
NO_REFERENCE_NAME = "no_reference.jpg"
NO_REFERENCE_TEXT = "No reference"
REF_NAME = re.compile(r"mango_(nir|rgb)_(\d+)\.jpg")

PLACEHOLDER_SIZE = (3280, 2464)  # width, height of a capture
PLACEHOLDER_GRAY = 128
TEXT_GRAY = 235
BOX_COLOR = (0, 0, 255)  # BGR red
BOX_WIDTH_FRACTION = 0.004  # outline thickness as a fraction of the image width
JPEG_QUALITY = 95


@dataclass(frozen=True)
class ReferenceJob:
    """One reference file: where it goes, what it is made from, and the box to draw.

    Paths are relative to the document root. ``source`` and ``box`` are ``None`` for the placeholder.
    ``box`` is ``(x, y, w, h)`` in pixels of an image of ``box_size`` (width, height), the size the box was
    drawn on; it is scaled when the source image has another size.
    """

    ref: Path
    source: Path | None
    box: Box | None
    box_size: tuple[int, int] | None


@dataclass
class ReferenceReport:
    written: list[Path]
    skipped: list[Path]
    out_dir: Path


# ---------------------------------------------------------------------------------------------
# Jobs from the export
# ---------------------------------------------------------------------------------------------


def _face_of(data: dict) -> Face:
    return Face(str(data["jornada"]), int(data["mango_id"]), int(data["cara"]))


def _face_text(face: Face) -> str:
    return f"{face.jornada} mango {face.mango_id} face {face.cara}"


def _url_relpath(url: str) -> PurePosixPath | None:
    """Path relative to the document root named by a Label Studio local-files URL, else ``None``.

    Absolute paths and ``..`` parts are refused so a source can never leave the document root.
    """
    if not url.startswith(URL_PREFIX):
        return None
    path = PurePosixPath(unquote(url[len(URL_PREFIX) :]))
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _box_of(face: Face, task: dict) -> tuple[str, Box, tuple[int, int]]:
    """Modality, pixel box and its image size from the last non-cancelled annotation of ``task``."""
    live = [a for a in task.get("annotations") or [] if not a.get("was_cancelled")]
    if not live:
        raise ValueError(f"{_face_text(face)}: the +12 h task has no annotation")
    annotation = max(live, key=lambda a: a.get("updated_at") or "")
    found = [
        r
        for r in annotation["result"]
        if r.get("type") == "rectanglelabels" and r.get("from_name") in FROM_NAME_MODALITY
    ]
    if not found:
        raise ValueError(f"{_face_text(face)}: the +12 h task has no box")
    if len(found) > 1:
        raise ValueError(f"{_face_text(face)}: the +12 h task has {len(found)} boxes, expected one")
    result = found[0]
    size = (int(result["original_width"]), int(result["original_height"]))
    return FROM_NAME_MODALITY[result["from_name"]], percent_to_box(result["value"], *size), size


def _reference_job(face: Face, ref: PurePosixPath, reference_tasks: dict[Face, dict]) -> ReferenceJob:
    if ref.name == NO_REFERENCE_NAME:
        return ReferenceJob(Path(ref), None, None, None)
    match = REF_NAME.fullmatch(ref.name)
    assert match is not None  # checked by the caller
    prefix, timestamp = match.group(1), int(match.group(2))
    task = reference_tasks.get(face)
    if task is None:
        raise ValueError(f"{_face_text(face)}: no +12 h task for reference {ref.name}")
    modality, box, size = _box_of(face, task)
    if prefix != modality:
        raise ValueError(
            f"{_face_text(face)}: reference {ref.name} names modality {prefix} "
            f"but the +12 h box was drawn on {modality}"
        )
    if timestamp != int(task["data"]["timestamp"]):
        raise ValueError(
            f"{_face_text(face)}: reference {ref.name} does not carry the timestamp "
            f"{task['data']['timestamp']} of the +12 h task"
        )
    url = task["data"].get(modality)
    source = _url_relpath(url) if isinstance(url, str) else None
    if source is None:
        raise ValueError(
            f"{_face_text(face)}: the +12 h {modality} image {url!r} is not a local-files URL"
        )
    return ReferenceJob(Path(ref), Path(source), box, size)


def collect_reference_jobs(tasks: list[dict]) -> list[ReferenceJob]:
    """One job per distinct ``data.ref`` of the export, sorted by file name.

    Tasks without ``ref`` are ignored. Every reference must be a file of ``ref_12h`` and, unless it is the
    placeholder, match the +12 h task of the same face: same timestamp, and the prefix names the modality of
    that task's box. The first inconsistency raises a ``ValueError`` naming the face.
    """
    reference_tasks = {
        _face_of(t["data"]): t for t in tasks if t["data"].get("ronda_h") == REFERENCE_HOUR
    }
    jobs: dict[Path, ReferenceJob] = {}
    for task in tasks:
        url = task["data"].get("ref")
        if url is None:
            continue
        face = _face_of(task["data"])
        ref = _url_relpath(url)
        valid = ref is not None and PurePosixPath(*ref.parts[:-1]) == PurePosixPath(REF_DIR)
        if not valid or not (ref.name == NO_REFERENCE_NAME or REF_NAME.fullmatch(ref.name)):
            raise ValueError(f"{_face_text(face)}: ref {url!r} is not a file of {REF_DIR}")
        job = _reference_job(face, ref, reference_tasks)
        jobs[job.ref] = job
    return [jobs[ref] for ref in sorted(jobs)]


def load_reference_jobs(labels_path: Path) -> list[ReferenceJob]:
    """Read a Label Studio JSON export and collect its reference jobs."""
    tasks = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    return collect_reference_jobs(tasks)


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------


def box_thickness(width: int) -> int:
    """Outline thickness in pixels for an image of ``width``: clearly visible at any scale."""
    return max(2, round(BOX_WIDTH_FRACTION * width))


def render_reference(image: np.ndarray, box: Box, box_size: tuple[int, int]) -> np.ndarray:
    """Return a 3-channel copy of ``image`` with ``box`` (pixels of ``box_size``) outlined."""
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    height, width = bgr.shape[:2]
    scale_x, scale_y = width / box_size[0], height / box_size[1]
    x, y, w, h = box
    left, right = round(x * scale_x), round((x + w) * scale_x)
    top, bottom = round(y * scale_y), round((y + h) * scale_y)
    cv2.rectangle(
        bgr,
        (max(0, left), max(0, top)),
        (min(width - 1, right), min(height - 1, bottom)),
        BOX_COLOR,
        thickness=box_thickness(width),
    )
    return bgr


def render_placeholder(size: tuple[int, int] = PLACEHOLDER_SIZE) -> np.ndarray:
    """Neutral gray image of ``size`` (width, height) with the centered text "No reference"."""
    width, height = size
    image = np.full((height, width, 3), PLACEHOLDER_GRAY, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = width / 640
    thickness = max(1, round(2 * scale))
    (text_w, text_h), _ = cv2.getTextSize(NO_REFERENCE_TEXT, font, scale, thickness)
    origin = ((width - text_w) // 2, (height + text_h) // 2)
    cv2.putText(image, NO_REFERENCE_TEXT, origin, font, scale, (TEXT_GRAY,) * 3, thickness, cv2.LINE_AA)
    return image


# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------


def _write_jpeg(image: np.ndarray, dst: Path) -> None:
    """Write ``image`` atomically as a JPEG."""
    part = dst.with_name(f"{dst.stem}.part{dst.suffix}")
    if not cv2.imwrite(str(part), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise ValueError(f"{dst.name}: could not be written")
    os.replace(part, dst)


def _render(document_root: Path, job: ReferenceJob) -> np.ndarray:
    if job.source is None:
        return render_placeholder()
    src = document_root / job.source
    image = cv2.imread(str(src), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"{src.name}: not a readable image")
    return render_reference(image, job.box, job.box_size)


def build_reference_images(
    document_root: Path, jobs: list[ReferenceJob], *, overwrite: bool = False
) -> ReferenceReport:
    """Write every reference into ``<document_root>/cache/captura_v3/news/ref_12h``.

    Existing images are skipped unless ``overwrite``. Every output must be a direct child of that folder,
    which must not resolve outside the document root, and every source image must exist: all of this is
    checked before anything is written.
    """
    document_root = Path(document_root)
    out_dir = document_root / REF_DIR
    for job in jobs:
        if job.ref.parent != REF_DIR:
            raise ValueError(f"{job.ref}: not directly inside {REF_DIR}")
        if job.source is not None and not (document_root / job.source).is_file():
            raise FileNotFoundError(f"source image not found: {document_root / job.source}")
    if jobs and not Path(os.path.realpath(out_dir)).is_relative_to(Path(os.path.realpath(document_root))):
        raise ValueError(f"{out_dir} resolves outside the document root")

    written: list[Path] = []
    skipped: list[Path] = []
    for job in jobs:
        dst = document_root / job.ref
        if dst.exists() and not overwrite:
            skipped.append(job.ref)
            continue
        image = _render(document_root, job)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jpeg(image, dst)
        written.append(job.ref)
    return ReferenceReport(written=written, skipped=skipped, out_dir=out_dir)
