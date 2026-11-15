#!/usr/bin/env python3
"""Generate the JVM test fixtures of the Android app from the Python reference.

Run from the repository root with the project venv (CPU only):

    CUDA_VISIBLE_DEVICES="" .venv/bin/python mobile/android/tools/make_fixtures.py \
        --rgb-dir data/cache/v4_captura_v3_poly/rgb

Outputs go to mobile/android/app/src/test/resources/fixtures/:

* roi/*.rgb.gz + roi/expected.json   Kotlin `rgb_mango_mask` port vs `src.annotation.mango_registration.rgb_mango_mask`
                                  (images downscaled to 320 px wide, the width the app runs it at), the
                                  `roi_crop` window rule, and the letterbox + normalization.
* decode/*                        Raw model outputs (random, quantized so float sums are exact) and the detections
                                  `src.export.reference.decode_exported_segmentation` produces from them.
* ../assets/sample_crop.png       Bundled benchmark input (an ROI crop of one dataset frame).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.annotation.mango_registration import rgb_mango_mask  # noqa: E402
from src.export.reference import decode_exported_segmentation, unletterbox_boxes, LetterboxInfo  # noqa: E402
from src.training.dataset import letterbox  # noqa: E402
from src.training.roi_crop import crop_to_roi  # noqa: E402

APP = ROOT / "mobile" / "android" / "app" / "src"
FIX = APP / "test" / "resources" / "fixtures"
ASSETS = APP / "main" / "assets"


def write_rgb_gz(path: Path, bgr: np.ndarray) -> None:
    """Raw fixture format read by the JVM tests (no ImageIO on the Android unit-test classpath):
    gzip of int32-LE width, int32-LE height, then RGB bytes row-major."""
    import gzip
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(np.array([w, h], "<i4").tobytes() + rgb.tobytes())


def bbox_of(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def roi_fixtures(rgb_dir: Path, indices: list[int]) -> dict:
    files = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    out_dir = FIX / "roi"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    first_small = None
    first_full = None
    for n, idx in enumerate(indices):
        full = cv2.imread(str(files[idx]))
        h, w = full.shape[:2]
        small = cv2.resize(full, (320, round(320 * h / w)), interpolation=cv2.INTER_AREA)
        name = f"frame{n}_320.rgb.gz"
        write_rgb_gz(out_dir / name, small)
        mask = rgb_mango_mask(small)
        full_mask = rgb_mango_mask(full)
        cases.append({
            "file": name,
            "source": files[idx].name,
            "width": int(small.shape[1]), "height": int(small.shape[0]),
            "bbox": bbox_of(mask),
            "mask_area": int((mask > 0).sum()),
            "full_frame": [int(w), int(h)],
            "full_res_bbox": bbox_of(full_mask),
        })
        if first_small is None:
            first_small, first_full = small, full
    return {"cases": cases, "first_small": first_small, "first_full": first_full}


def crop_window_cases() -> list[dict]:
    """The roi_crop window (margin 0.10) for a pixel mango box in a frame."""
    cases = []
    for frame, box in [
        ((1280, 720), (500, 200, 780, 480)),
        ((1280, 720), (10, 5, 300, 200)),       # clipped on the left/top
        ((1280, 720), (1000, 400, 1275, 718)),  # clipped on the right/bottom
        ((320, 180), (100, 40, 180, 130)),
        ((1280, 720), (600, 300, 601, 301)),    # degenerate 1 px box
    ]:
        w, h = frame
        x1, y1, x2, y2 = box
        bboxes = np.array([[(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]], np.float32)
        rgb = np.zeros((h, w, 3), np.uint8)
        nir = np.zeros((h, w), np.uint8)
        rgb_c, _, _, _ = crop_to_roi(rgb, nir, bboxes, np.array([0]), margin=0.10)
        # Recover the window from the cropped size by re-running the same rule on the box.
        import math
        bw, bh = x2 - x1, y2 - y1
        eps = 1e-3
        wx1 = min(max(math.floor(x1 - 0.1 * bw + eps), 0), w)
        wy1 = min(max(math.floor(y1 - 0.1 * bh + eps), 0), h)
        wx2 = min(max(math.ceil(x2 + 0.1 * bw - eps), 0), w)
        wy2 = min(max(math.ceil(y2 + 0.1 * bh - eps), 0), h)
        assert (wx2 - wx1, wy2 - wy1) == (rgb_c.shape[1], rgb_c.shape[0]), "window mismatch vs crop_to_roi"
        cases.append({"frame": [w, h], "box": [x1, y1, x2, y2], "window": [wx1, wy1, wx2, wy2]})
    return cases


def letterbox_fixtures(small: np.ndarray) -> dict:
    out_dir = FIX / "roi"
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    cases = []
    for name, window in [("full", None), ("crop", (40, 20, 200, 140))]:
        img = rgb if window is None else rgb[window[1]:window[3], window[0]:window[2]]
        padded, scale, pad_x, pad_y = letterbox(np.ascontiguousarray(img), 640, 114)
        write_rgb_gz(out_dir / f"letterbox_{name}.rgb.gz", cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))
        cases.append({
            "name": name, "window": list(window) if window else None,
            "scale": float(scale), "pad_x": int(pad_x), "pad_y": int(pad_y),
            "new_w": int(int(img.shape[1] * scale)), "new_h": int(int(img.shape[0] * scale)),
            "file": f"letterbox_{name}.rgb.gz",
        })
    return {"cases": cases, "source": "frame0_320.rgb.gz"}


def pack_mask(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(np.uint8).ravel()).tobytes()).decode()


def decode_fixtures(seed: int = 7) -> dict:
    out_dir = FIX / "decode"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n, nm, ps = 8400, 32, 160
    boxes = np.empty((n, 4), np.float32)
    boxes[:, 0:2] = rng.normal(0.0, 6.0, (n, 2))
    boxes[:, 2:4] = rng.normal(np.log(70.0), 0.5, (n, 2))
    cls = rng.normal(-3.8, 1.6, (n, 2)).astype(np.float32)
    coefs_q = rng.integers(-16, 17, (n, nm), dtype=np.int8)       # value = q / 16
    protos_q = rng.integers(-24, 25, (nm, ps, ps), dtype=np.int8)  # value = q / 8
    coefs = (coefs_q.astype(np.float32) / 16.0)
    protos = (protos_q.astype(np.float32) / 8.0)

    boxes.astype("<f4").tofile(out_dir / "boxes_raw.bin")
    cls.astype("<f4").tofile(out_dir / "cls_logits.bin")
    coefs_q.tofile(out_dir / "mask_coefs_q16.bin")
    protos_q.tofile(out_dir / "protos_q8.bin")

    conf, iou, max_det = 0.25, 0.5, 40
    b, s, lab, masks = decode_exported_segmentation(
        torch.from_numpy(boxes)[None], torch.from_numpy(cls)[None],
        torch.from_numpy(coefs)[None], torch.from_numpy(protos)[None],
        image_size=640, strides=(8, 16, 32),
        conf_threshold=conf, nms_iou_threshold=iou, max_detections=max_det,
    )
    # Letterbox geometry of a 480x360 ROI crop that sat at (300, 120) of a 1280x720 frame.
    crop_x, crop_y, crop_w, crop_h = 300, 120, 480, 360
    scale = 640 / max(crop_w, crop_h)
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    info = LetterboxInfo(orig_w=crop_w, orig_h=crop_h, scale=scale,
                         pad_x=(640 - new_w) // 2, pad_y=(640 - new_h) // 2)
    xyxy = unletterbox_boxes(b.clone().double(), info)
    dets = []
    for i in range(len(s)):
        m = masks[i].numpy()
        dets.append({
            "cls": int(lab[i]),
            "score": float(s[i]),
            "box_cxcywh": [float(v) for v in b[i]],
            "box_crop_xyxy": [float(v) for v in xyxy[i]],
            "box_frame_xyxy": [float(xyxy[i][0]) + crop_x, float(xyxy[i][1]) + crop_y,
                               float(xyxy[i][2]) + crop_x, float(xyxy[i][3]) + crop_y],
            "mask_true": int(m.sum()),
            "mask_bits": pack_mask(m),
        })
    return {
        "conf": conf, "iou": iou, "max_det": max_det,
        "crop": [crop_x, crop_y, crop_w, crop_h],
        "scale": scale, "pad_x": info.pad_x, "pad_y": info.pad_y,
        "n_candidates_over_conf": int((1 / (1 + np.exp(-cls)) > conf).sum()),
        "detections": dets,
    }


def sample_crop(first_full: np.ndarray, mask_bbox_full: list[int]) -> None:
    x1, y1, x2, y2 = mask_bbox_full
    bw, bh = x2 - x1, y2 - y1
    wx1, wy1 = max(int(x1 - 0.1 * bw), 0), max(int(y1 - 0.1 * bh), 0)
    wx2, wy2 = min(int(x2 + 0.1 * bw) + 1, first_full.shape[1]), min(int(y2 + 0.1 * bh) + 1, first_full.shape[0])
    crop = first_full[wy1:wy2, wx1:wx2]
    ASSETS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ASSETS / "sample_crop.png"), crop)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-dir", required=True, type=Path)
    ap.add_argument("--indices", type=int, nargs=3, default=[100, 1500, 3000])
    args = ap.parse_args()

    roi = roi_fixtures(args.rgb_dir, args.indices)
    expected = {
        "cases": roi["cases"],
        "crop_windows": crop_window_cases(),
        "letterbox": letterbox_fixtures(roi["first_small"]),
    }
    (FIX / "roi" / "expected.json").write_text(json.dumps(expected, indent=1))
    sample_crop(roi["first_full"], roi["cases"][0]["full_res_bbox"])
    dec = decode_fixtures()
    (FIX / "decode" / "expected.json").write_text(json.dumps(dec))
    print("ROI bboxes:", [(c["bbox"], c["full_res_bbox"]) for c in roi["cases"]])
    print("decode detections:", len(dec["detections"]), "candidates:", dec["n_candidates_over_conf"])


if __name__ == "__main__":
    main()
