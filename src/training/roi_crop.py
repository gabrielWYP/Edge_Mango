"""Mango ROI crop: cut RGB, registered NIR and labels to the mango box.

Frames are 1280x720 and the mango covers only ~8% of them, so a plain
letterbox to 640 halves the resolution of every damage box (class 1, ~26 px
median short side). Cropping to the ground-truth mango box (class 0) plus a
margin BEFORE the letterbox puts the same boxes at ~92 px.

The crop window lives in the RGB frame, so NIR must already be registered to
RGB (`nir_registration.register_nir_to_rgb`); the same window is then valid for
both modalities.

At evaluation time the window comes from the GT mango box, so metrics measured
this way are an ORACLE-ROI upper bound until a detector supplies the window.
"""

from __future__ import annotations

import math

import numpy as np

# Class id of the mango box in every label file ([mango, damage]).
MANGO_CLASS_ID = 0

# Slack (pixels) when snapping the window to integers. Normalized float32 labels
# carry ~1e-4 px of noise on a 1280 px frame, which would otherwise turn an
# exact 128.0 into 129 under ceil() and an exact 32.0 into 31 under floor().
_PIXEL_EPS = 1e-3


def validate_roi_crop(margin: float, jitter: float) -> None:
    """Validate the ROI crop margin and jitter.

    Args:
        margin: Per-side margin as a fraction of the ROI box width/height.
        jitter: Half-range of the per-side margin scale factor, in [0, 1].

    Raises:
        ValueError: `margin` is negative or not finite, or `jitter` is outside
            [0, 1] (NaN is rejected too). A jitter of at most 1 guarantees the
            window always contains the whole ROI box.
    """
    if not (math.isfinite(margin) and margin >= 0.0):
        raise ValueError(f"roi_crop_margin must be a finite value >= 0, got {margin}.")
    if not 0.0 <= jitter <= 1.0:
        raise ValueError(
            f"roi_crop_jitter must be in [0, 1] so the crop always contains the ROI box, "
            f"got {jitter}."
        )


def crop_to_roi(
    rgb: np.ndarray,
    nir: np.ndarray,
    bboxes: np.ndarray,
    labels: np.ndarray,
    roi_class: int = MANGO_CLASS_ID,
    margin: float = 0.10,
    margin_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
    min_visible: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crop both modalities to the largest `roi_class` box plus a margin.

    Each of the four sides gets a margin of `margin * (box width or height)`.
    With `margin_jitter > 0` every side's margin is independently scaled by
    `uniform(1 - margin_jitter, 1 + margin_jitter)`; since the factor is never
    negative the window always contains the whole ROI box. The window is
    snapped outward to integer pixels and clipped to the image.

    Args:
        rgb: (H, W, 3) RGB image.
        nir: (H, W) NIR image, already registered to the RGB frame.
        bboxes: (N, 4) YOLO `[cx, cy, w, h]` normalized to the full frame.
        labels: (N,) integer class ids aligned with `bboxes`.
        roi_class: Class whose largest box defines the window.
        margin: Per-side margin as a fraction of the ROI box size.
        margin_jitter: Half-range of the per-side margin scale, in [0, 1].
        rng: Generator for the jitter. Required when `margin_jitter > 0`; the
            global numpy RNG is never used.
        min_visible: A box whose area inside the window is below this
            fraction of its original area is dropped together with its label.
            Boxes with no visible area are always dropped.

    Returns:
        (rgb_c, nir_c, bboxes_c, labels_c). Boxes are float32 `(N, 4)` YOLO
        coordinates normalized to the CROPPED image, labels int64 `(N,)`, in
        the input order. When there is no `roi_class` box (or the window would
        be empty) the inputs come back unchanged, i.e. the full frame.

    Raises:
        ValueError: `nir` and `rgb` differ in height/width (NIR must be
            registered to RGB first), `margin`/`margin_jitter` are out of
            range, or `margin_jitter > 0` without an `rng`.
    """
    if nir.shape[:2] != rgb.shape[:2]:
        raise ValueError(
            f"NIR shape {nir.shape[:2]} differs from RGB shape {rgb.shape[:2]}: the crop "
            "window is computed in the RGB frame, so NIR must be registered to RGB first."
        )
    validate_roi_crop(margin, margin_jitter)
    if margin_jitter > 0.0 and rng is None:
        raise ValueError("margin_jitter > 0 requires an explicit rng (np.random.Generator).")

    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    roi_candidates = np.flatnonzero(labels == roi_class)
    if roi_candidates.size == 0:
        return rgb, nir, bboxes, labels

    height, width = rgb.shape[:2]
    boxes = bboxes.astype(np.float64)

    # ROI in full-frame pixels (largest box of the class; ties -> first).
    roi_idx = int(roi_candidates[np.argmax(boxes[roi_candidates, 2] * boxes[roi_candidates, 3])])
    roi_w = boxes[roi_idx, 2] * width
    roi_h = boxes[roi_idx, 3] * height
    roi_x1 = boxes[roi_idx, 0] * width - roi_w / 2
    roi_y1 = boxes[roi_idx, 1] * height - roi_h / 2

    # Per-side margin scale: (left, top, right, bottom).
    if margin_jitter > 0.0:
        scale = rng.uniform(1.0 - margin_jitter, 1.0 + margin_jitter, size=4)
    else:
        scale = np.ones(4)
    left = roi_x1 - margin * roi_w * scale[0]
    top = roi_y1 - margin * roi_h * scale[1]
    right = roi_x1 + roi_w + margin * roi_w * scale[2]
    bottom = roi_y1 + roi_h + margin * roi_h * scale[3]

    x_min = min(max(math.floor(left + _PIXEL_EPS), 0), width)
    y_min = min(max(math.floor(top + _PIXEL_EPS), 0), height)
    x_max = min(max(math.ceil(right - _PIXEL_EPS), 0), width)
    y_max = min(max(math.ceil(bottom - _PIXEL_EPS), 0), height)
    if x_max <= x_min or y_max <= y_min:
        # ROI entirely outside the image: nothing sensible to crop to.
        return rgb, nir, bboxes, labels

    crop_w, crop_h = x_max - x_min, y_max - y_min

    # All boxes -> pixel xyxy relative to the window origin.
    bx1 = (boxes[:, 0] - boxes[:, 2] / 2) * width - x_min
    by1 = (boxes[:, 1] - boxes[:, 3] / 2) * height - y_min
    bx2 = (boxes[:, 0] + boxes[:, 2] / 2) * width - x_min
    by2 = (boxes[:, 1] + boxes[:, 3] / 2) * height - y_min
    original_area = (bx2 - bx1) * (by2 - by1)

    cx1, cx2 = np.clip(bx1, 0.0, crop_w), np.clip(bx2, 0.0, crop_w)
    cy1, cy2 = np.clip(by1, 0.0, crop_h), np.clip(by2, 0.0, crop_h)
    clipped_area = (cx2 - cx1) * (cy2 - cy1)

    keep = (clipped_area > 0.0) & (clipped_area >= min_visible * original_area)
    keep[roi_idx] = True  # the ROI defines the window; never lose it to rounding

    cropped = np.stack(
        [
            (cx1 + cx2) / 2 / crop_w,
            (cy1 + cy2) / 2 / crop_h,
            (cx2 - cx1) / crop_w,
            (cy2 - cy1) / crop_h,
        ],
        axis=1,
    )
    rgb_c = np.ascontiguousarray(rgb[y_min:y_max, x_min:x_max])
    nir_c = np.ascontiguousarray(nir[y_min:y_max, x_min:x_max])
    return rgb_c, nir_c, cropped[keep].astype(np.float32), labels[keep]
