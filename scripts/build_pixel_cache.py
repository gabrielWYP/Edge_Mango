#!/usr/bin/env python3
"""Store the mango window of every pair, uncompressed, so that training does not decode images.

Most of the time of a data worker goes to decoding a 3264x2448 photo of which it uses the mango.
This keeps those pixels exactly: `YOLODataset(pixel_cache_dir=...)` gives the same output as
decoding, with the same crop window and the same random draws. It is only about speed, and
needs one folder per RGB directory, because the NIR arm reads other images in its RGB slot.

    python scripts/build_pixel_cache.py --dataset data/cache/v4_captura_v3_final \\
        --rgb-dir data/cache/v4_captura_v3_final/rgb --output data/cache/pixels/rgb
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.nir_registration import register_nir_to_rgb  # noqa: E402
from src.training.roi_crop import MANGO_CLASS_ID  # noqa: E402

# The widest window of the v4 configs: margin 0.10 with jitter 0.5.
DEFAULT_MAX_MARGIN = 0.15


def stored_window(mango: np.ndarray, width: int, height: int, max_margin: float) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) that holds every crop of a YOLO mango box up to `max_margin`."""
    w, h = mango[2] * width, mango[3] * height
    left, top = mango[0] * width - w / 2, mango[1] * height - h / 2
    x0 = max(math.floor(left - max_margin * w) - 1, 0)
    y0 = max(math.floor(top - max_margin * h) - 1, 0)
    x1 = min(math.ceil(left + w + max_margin * w) + 1, width)
    y1 = min(math.ceil(top + h + max_margin * h) + 1, height)
    return x0, y0, x1, y1


def store_pair(job: tuple) -> str:
    """Write one pair; reads it the way `YOLODataset.__getitem__` does."""
    rgb_path, nir_path, label_path, output, homography, max_margin = job
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    nir = cv2.imread(str(nir_path), cv2.IMREAD_GRAYSCALE)
    height, width = rgb.shape[:2]
    # The fill value never shows: the v4 NIR is registered already, and this homography is the identity.
    nir = register_nir_to_rgb(nir, np.load(homography), (width, height), 0)
    rows = np.loadtxt(label_path, ndmin=2, dtype=np.float32)
    mangos = rows[rows[:, 0] == MANGO_CLASS_ID]
    if len(mangos) == 0:
        raise ValueError(f"{label_path}: no mango box, so no window to store")
    # The largest mango box, with the first one on ties, as in `crop_to_roi`.
    boxes = mangos[:, 1:].astype(np.float64)
    x0, y0, x1, y1 = stored_window(boxes[int(np.argmax(boxes[:, 2] * boxes[:, 3]))], width, height, max_margin)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp.npz")
    np.savez(
        temporary,
        rgb=rgb[y0:y1, x0:x1],
        nir=nir[y0:y1, x0:x1],
        origin=np.array([x0, y0]),
        frame_size=np.array([height, width]),
    )
    temporary.replace(output)
    return output.stem


def plan_jobs(dataset: Path, rgb_dir: Path, nir_variant: str, output: Path, max_margin: float) -> list[tuple]:
    """One job per labelled pair of fold 0, whose three splits hold every photo."""
    homography = dataset / "homographies" / "identity.npy"
    jobs = []
    for label_path in sorted((dataset / "folds_lot" / "fold_0" / "labels").glob("*/*.txt")):
        stem = label_path.stem
        target = output / f"{stem}.npz"
        if target.exists():
            continue
        nir_path = dataset / nir_variant / f"{stem.replace('mango_rgb_', 'mango_nir_')}.png"
        jobs.append((rgb_dir / f"{stem}.jpg", nir_path, label_path, target, homography, max_margin))
    return jobs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nir-variant", default="nir_clahe")
    parser.add_argument("--max-margin", type=float, default=DEFAULT_MAX_MARGIN)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(args.dataset, args.rgb_dir, args.nir_variant, args.output, args.max_margin)
    with ProcessPoolExecutor(args.workers) as pool:
        done = sum(1 for _ in pool.map(store_pair, jobs, chunksize=16))
    print(f"{done} pairs stored in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
