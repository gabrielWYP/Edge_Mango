"""Contrast-enhanced grayscale NIR images for annotation.

Bruises sit in the shadowed parts of the curved fruit, so the raw NIR is hard to read. The enhanced image
keeps the broad low-frequency darkening that makes the bruise visible (no illumination flattening) and has
exactly the original NIR width and height, so a box drawn on it maps 1:1 onto the original NIR.

The NoIR camera (awbgains 1,1) puts the NIR signal in R and B, with G about half as strong, so the gray
image is (B + R) / 2.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

GAUSSIAN_SIGMA = 1.0
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (16, 16)
JPEG_QUALITY = 95

NIR_VIS_DIR = "nir_vis"
MANIFEST_NAME = "manifest.json"
PROTECTED_DIRS = ("rgb", "nir", "meta", "rgb_wb")
METHOD = (
    "gray = (B + R) / 2 of the 3-channel NIR JPEG as float32 -> Gaussian blur -> clip to uint8 -> CLAHE; "
    "saved as a single-channel JPEG with the original NIR width and height"
)


def method_params() -> dict:
    return {
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "clahe_clip_limit": CLAHE_CLIP_LIMIT,
        "clahe_tile_grid": list(CLAHE_TILE_GRID),
        "jpeg_quality": JPEG_QUALITY,
    }


def nir_vis_relpath(timestamp: int) -> Path:
    """Path of the enhanced image relative to the captura_v3 data root."""
    return Path("news") / NIR_VIS_DIR / f"mango_nir_{timestamp}.jpg"


def nir_gray(bgr: np.ndarray) -> np.ndarray:
    """Float32 mean of the blue and red channels of a BGR NIR image."""
    return (bgr[..., 0].astype(np.float32) + bgr[..., 2].astype(np.float32)) / 2


def enhance_nir(bgr: np.ndarray) -> np.ndarray:
    """Return the single-channel uint8 enhanced image, same height and width as ``bgr``."""
    blurred = cv2.GaussianBlur(nir_gray(bgr), (0, 0), GAUSSIAN_SIGMA)
    as_u8 = np.clip(blurred, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(as_u8)


def enhance_file(src: Path, dst: Path) -> None:
    """Enhance one NIR JPEG and write the result atomically as a single-channel JPEG."""
    if not src.is_file():
        raise FileNotFoundError(f"NIR image not found: {src}")
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise ValueError(f"{src.name}: not a readable image")
    part = dst.with_name(f"{dst.stem}.part{dst.suffix}")
    if not cv2.imwrite(str(part), enhance_nir(bgr), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise ValueError(f"{dst.name}: could not be written")
    os.replace(part, dst)


def check_output_dir(data_root: Path, out_dir: Path) -> None:
    """Refuse an output folder that is, or resolves into, a capture folder."""
    target = Path(os.path.realpath(out_dir))
    for name in PROTECTED_DIRS:
        protected = Path(os.path.realpath(data_root / "news" / name))
        if target == protected or target.is_relative_to(protected):
            raise ValueError(f"{out_dir} is inside the protected capture folder news/{name}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_entries(manifest: Path, overwrite: bool) -> dict[int, str]:
    if not manifest.is_file():
        return {}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        params = data["params"]
        entries = {int(f["timestamp"]): str(f["source_sha256"]) for f in data["files"]}
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{manifest}: invalid manifest ({exc})") from exc
    if params == method_params():
        return entries
    if overwrite:
        return {}
    raise ValueError(
        f"{manifest.parent} holds images made with different parameters; "
        "rerun with --overwrite to regenerate them"
    )


def _write_manifest(manifest: Path, entries: dict[int, str]) -> None:
    payload = {
        "method": METHOD,
        "params": method_params(),
        "files": [
            {"timestamp": ts, "source_sha256": entries[ts]} for ts in sorted(entries)
        ],
    }
    part = manifest.with_name(manifest.name + ".part")
    part.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(part, manifest)


@dataclass
class DisplayReport:
    written: list[int]
    skipped: list[int]
    out_dir: Path


def build_nir_display(
    data_root: Path,
    timestamps: list[int],
    *,
    overwrite: bool = False,
    out_dir: Path | None = None,
) -> DisplayReport:
    """Write ``mango_nir_<ts>.jpg`` for each timestamp into ``news/nir_vis`` plus a manifest.

    Existing images are skipped unless ``overwrite``. The manifest records the method, its
    parameters and the sha256 of every source NIR.
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir) if out_dir is not None else data_root / "news" / NIR_VIS_DIR
    check_output_dir(data_root, out_dir)

    sources = {}
    for ts in sorted(set(timestamps)):
        src = data_root / "news" / "nir" / f"mango_nir_{ts}.jpg"
        if not src.is_file():
            raise FileNotFoundError(f"NIR image not found: {src}")
        sources[ts] = src

    manifest = out_dir / MANIFEST_NAME
    entries = _load_entries(manifest, overwrite)
    written: list[int] = []
    skipped: list[int] = []
    try:
        for ts, src in sources.items():
            dst = out_dir / src.name
            if dst.exists() and not overwrite:
                skipped.append(ts)
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                enhance_file(src, dst)
                written.append(ts)
            entries[ts] = _sha256(src)
    finally:
        if entries and out_dir.is_dir():
            _write_manifest(manifest, entries)
    return DisplayReport(written=written, skipped=skipped, out_dir=out_dir)
