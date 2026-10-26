"""Mango masks and two independent registrations of the same fruit photographed twice.

The fruit is re-placed by hand every round, so the same face appears shifted, turned and
slightly rolled. Both methods estimate a similarity (scale, rotation, shift) of the fruit
interior from a reference photo to a moving photo, inside a fruit mask so the static sheet
cannot pull the result toward identity:

* ``register_sift``: SIFT keypoints (low contrast threshold, CLAHE first) + Lowe ratio test +
  RANSAC similarity.
* ``register_dense``: exhaustive masked zero-mean cross-correlation of the band-passed
  interior over rotation, scale and shift. It shares no code path with SIFT.

All transforms are in edge-based pixel coordinates ``u`` (pixel ``i`` covers ``[i, i+1]``) of
the full-resolution frame, which is how Label Studio percent boxes convert to pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

SIFT_CONTRAST = 0.008
SIFT_RATIO = 0.75
SIFT_MAX_POINTS = 6000
RANSAC_PX_FULL = 12.0
MIN_INLIERS = 15
SIFT_SCALE_RANGE = (0.7, 1.4)
SIFT_MAX_ANGLE_DEG = 60.0

DENSE_DIAMETER = 110
DENSE_BAND = (0.012, 0.04)
DENSE_ERODE = 0.18
DENSE_ANGLES_DEG = tuple(range(-36, 37, 3))
DENSE_SCALES = (0.94, 1.0, 1.06)
DENSE_MIN_SCORE = 0.45
DENSE_RUNNER_UP_RADIUS = 0.25


@dataclass(frozen=True)
class Similarity:
    """``u' = scale * R(angle) @ u + (tx, ty)`` with image axes (x right, y down)."""

    scale: float = 1.0
    angle_deg: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Similarity:
        m = np.asarray(matrix, dtype=float)
        return cls(
            scale=float(math.hypot(m[0, 0], m[1, 0])),
            angle_deg=float(math.degrees(math.atan2(m[1, 0], m[0, 0]))),
            tx=float(m[0, 2]),
            ty=float(m[1, 2]),
        )

    def matrix(self) -> np.ndarray:
        a = math.radians(self.angle_deg)
        c, s = self.scale * math.cos(a), self.scale * math.sin(a)
        return np.array([[c, -s, self.tx], [s, c, self.ty]])

    def apply(self, points: np.ndarray) -> np.ndarray:
        m = self.matrix()
        return np.asarray(points, dtype=float) @ m[:, :2].T + m[:, 2]

    def inverse(self) -> Similarity:
        m = self.matrix()
        linear_inv = np.linalg.inv(m[:, :2])
        return Similarity.from_matrix(np.hstack([linear_inv, (-linear_inv @ m[:, 2])[:, None]]))

    def then(self, other: Similarity) -> Similarity:
        """The transform that applies ``self`` first and ``other`` second."""
        a, b = self.matrix(), other.matrix()
        return Similarity.from_matrix(
            np.hstack([b[:, :2] @ a[:, :2], (b[:, :2] @ a[:, 2] + b[:, 2])[:, None]])
        )

    def rescaled(self, factor: float) -> Similarity:
        """The same motion expressed in frames resized by ``factor`` (``u' = factor * u``)."""
        return Similarity(self.scale, self.angle_deg, self.tx * factor, self.ty * factor)

    def cv_matrix(self) -> np.ndarray:
        """Affine for ``cv2.warpAffine``, whose coordinates are pixel centres (index = u - 0.5)."""
        m = self.matrix()
        m[:, 2] += (m[:, :2] - np.eye(2)) @ np.array([0.5, 0.5])
        return m


def _odd(value: float, minimum: int = 3) -> int:
    n = max(minimum, int(round(value)))
    return n if n % 2 else n + 1


def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return mask
    return ((labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))) * 255).astype(np.uint8)


def _hull(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if contours:
        cv2.fillConvexPoly(out, cv2.convexHull(np.vstack(contours)), 255)
    return out


def rgb_mango_mask(bgr: np.ndarray) -> np.ndarray:
    """Fruit mask of an RGB photo: yellow skin has little blue next to the cyan/white sheet."""
    height, width = bgr.shape[:2]
    blurred = cv2.GaussianBlur(bgr.astype(np.float32), (0, 0), max(1.0, 0.005 * width))
    blue, green, red = blurred[..., 0], blurred[..., 1], blurred[..., 2]
    ratio8 = np.clip(blue / (green + red + 1) * 256, 0, 255).astype(np.uint8)
    lit = (green + red) > 40
    if not lit.any():
        return np.zeros((height, width), np.uint8)
    level, _ = cv2.threshold(
        ratio8[lit].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask = ((ratio8 < level) & lit).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(_odd(0.022 * width)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(_odd(0.037 * width)))
    return _largest_component(mask)


def _log_ratio_to_empty_tray(gray: np.ndarray, empty: np.ndarray, degree: int = 3):
    """Log ratio to the empty tray after fitting out the illumination change with a polynomial."""
    height, width = gray.shape
    valid = (gray > 20) & (empty > 20)
    log_ratio = np.log((gray + 1) / (empty + 1)).astype(np.float32)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x, y = xx / width * 2 - 1, yy / height * 2 - 1
    design = np.stack([x**i * y**j for i in range(degree + 1) for j in range(degree + 1 - i)], -1)
    design = design.reshape(-1, design.shape[-1])
    flat = log_ratio.ravel()
    if not valid.any():
        return np.zeros_like(log_ratio), valid
    residual = log_ratio - np.median(log_ratio[valid])
    use = (valid & (np.abs(residual) < 0.15)).ravel()
    for _ in range(6):
        rows = np.flatnonzero(use)[::7]
        if len(rows) < design.shape[1]:
            break
        coef, *_ = np.linalg.lstsq(design[rows], flat[rows], rcond=None)
        residual = log_ratio - (design @ coef).reshape(height, width)
        use = (valid & (np.abs(residual) < 0.10)).ravel()
    return np.where(valid, residual, 0.0), valid


def nir_mango_mask(nir_gray: np.ndarray, empty_gray: np.ndarray) -> np.ndarray:
    """Fruit (plus its shadow) in a NIR photo: what the empty tray does not explain.

    Both inputs are float gray images of the same shape. The empty tray is not aligned to the
    photo (the sheet moves), so pixels that are dark in either image are ignored, a smooth
    illumination change is fitted out, and the convex hull of the largest darker region is
    grown a little to take in the lit side of the fruit.
    """
    width = nir_gray.shape[1]
    sigma = max(1.0, 0.0037 * width)
    gray = cv2.GaussianBlur(nir_gray.astype(np.float32), (0, 0), sigma)
    empty = cv2.GaussianBlur(empty_gray.astype(np.float32), (0, 0), sigma)
    residual, valid = _log_ratio_to_empty_tray(gray, empty)
    darker = ((residual < -0.06) & valid).astype(np.uint8) * 255
    darker = cv2.morphologyEx(darker, cv2.MORPH_OPEN, _ellipse(_odd(0.017 * width)))
    body = _hull(_largest_component(darker))
    if not body.any():
        return body
    lit = ((residual > 0.04) & valid).astype(np.uint8) * 255
    lit = cv2.morphologyEx(lit, cv2.MORPH_OPEN, _ellipse(_odd(0.012 * width)))
    lit &= cv2.dilate(body, _ellipse(_odd(0.06 * width)))
    return cv2.dilate(_hull(body | lit), _ellipse(_odd(0.06 * width)))


@dataclass
class Prepared:
    """One photo ready for registration, at ``scale`` times the full-resolution size."""

    gray: np.ndarray
    mask: np.ndarray
    scale: float
    points: np.ndarray
    descriptors: np.ndarray | None


def prepare(gray: np.ndarray, mask: np.ndarray, scale: float) -> Prepared:
    """Detect SIFT keypoints inside ``mask`` on a CLAHE-enhanced copy of ``gray``."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    sift = cv2.SIFT_create(contrastThreshold=SIFT_CONTRAST)
    keypoints, descriptors = sift.detectAndCompute(clahe.apply(gray), mask)
    if keypoints and len(keypoints) > SIFT_MAX_POINTS:
        order = np.argsort([-k.response for k in keypoints])[:SIFT_MAX_POINTS]
        keypoints = [keypoints[i] for i in order]
        descriptors = descriptors[order]
    points = np.array([k.pt for k in keypoints], np.float32).reshape(-1, 2) + 0.5
    return Prepared(gray, mask, scale, points, descriptors)


@dataclass(frozen=True)
class SiftResult:
    sim: Similarity | None
    inliers: int
    rmse: float | None
    matches: int
    valid: bool


def register_sift(ref: Prepared, mov: Prepared, *, min_inliers: int = MIN_INLIERS) -> SiftResult:
    """Similarity ``ref -> mov`` from SIFT matches, in full-resolution coordinates."""
    none = SiftResult(None, 0, None, 0, False)
    if (
        ref.descriptors is None
        or mov.descriptors is None
        or len(ref.points) < 4
        or len(mov.points) < 4
    ):
        return none
    neighbours = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ref.descriptors, mov.descriptors, k=2)
    good = [
        m
        for m, n in (pair for pair in neighbours if len(pair) == 2)
        if m.distance < SIFT_RATIO * n.distance
    ]
    if len(good) < 4:
        return SiftResult(None, 0, None, len(good), False)
    src = ref.points[[m.queryIdx for m in good]]
    dst = mov.points[[m.trainIdx for m in good]]
    matrix, inlier_flags = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_PX_FULL * ref.scale,
        maxIters=5000,
        confidence=0.999,
        refineIters=10,
    )
    if matrix is None:
        return SiftResult(None, 0, None, len(good), False)
    inliers = inlier_flags.ravel().astype(bool)
    residual = np.linalg.norm(src[inliers] @ matrix[:, :2].T + matrix[:, 2] - dst[inliers], axis=1)
    rmse = float(np.sqrt((residual**2).mean())) / ref.scale
    sim = Similarity.from_matrix(matrix).rescaled(1 / ref.scale)
    plausible = (
        SIFT_SCALE_RANGE[0] <= sim.scale <= SIFT_SCALE_RANGE[1]
        and abs(sim.angle_deg) <= SIFT_MAX_ANGLE_DEG
    )
    count = int(inliers.sum())
    return SiftResult(sim, count, rmse, len(good), plausible and count >= min_inliers)


@dataclass(frozen=True)
class DenseResult:
    sim: Similarity | None
    score: float
    valid: bool
    runner_up: float = 0.0


def _band_pass(gray: np.ndarray, factor: float) -> np.ndarray:
    small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    fine, coarse = (max(0.8, s * DENSE_DIAMETER) for s in DENSE_BAND)
    return cv2.GaussianBlur(small, (0, 0), fine) - cv2.GaussianBlur(small, (0, 0), coarse)


def _binary_at(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return (resized > 127).astype(np.float32)


def _correlate(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    return cv2.matchTemplate(image, template, cv2.TM_CCORR)


def _masked_zncc(image, image_mask, template, template_mask) -> np.ndarray:
    """Zero-mean normalised cross-correlation over the support of ``template_mask``."""
    n = float(template_mask.sum())
    centred = (template - (template * template_mask).sum() / n) * template_mask
    energy = float((centred * centred).sum())
    numerator = _correlate(image, centred)
    local_sum = _correlate(image, template_mask)
    local_sq = _correlate(image * image, template_mask)
    variance = local_sq - local_sum * local_sum / n
    floor = 0.05 * n * float((image * image).sum() / max(float(image_mask.sum()), 1.0))
    return np.clip(
        numerator / np.maximum(np.sqrt(np.maximum(variance, floor) * energy), 1e-9), -1, 1
    )


def _template(
    band: np.ndarray,
    mask: np.ndarray,
    centre: np.ndarray,
    angle_rad: float,
    scale: float,
    size: int,
):
    """Warp the reference band-pass and mask so ``centre`` lands in the middle of a square patch."""
    turn = scale * np.array(
        [[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]]
    )
    shift = np.array([size / 2, size / 2]) - turn @ centre
    matrix = Similarity.from_matrix(np.hstack([turn, shift[:, None]])).cv_matrix()
    patch = cv2.warpAffine(band, matrix, (size, size), flags=cv2.INTER_LINEAR)
    support = cv2.warpAffine(mask, matrix, (size, size), flags=cv2.INTER_LINEAR) > 0.5
    return patch, support.astype(np.float32), turn, shift


def _parabolic(values: np.ndarray, x: int, y: int) -> tuple[float, float]:
    def offset(lo: float, mid: float, hi: float) -> float:
        denom = lo - 2 * mid + hi
        return 0.0 if abs(denom) < 1e-9 else float(np.clip(0.5 * (lo - hi) / denom, -0.5, 0.5))

    dx = (
        offset(values[y, x - 1], values[y, x], values[y, x + 1])
        if 0 < x < values.shape[1] - 1
        else 0.0
    )
    dy = (
        offset(values[y - 1, x], values[y, x], values[y + 1, x])
        if 0 < y < values.shape[0] - 1
        else 0.0
    )
    return dx, dy


def register_dense(
    ref: Prepared, mov: Prepared, *, min_score: float = DENSE_MIN_SCORE
) -> DenseResult:
    """Similarity ``ref -> mov`` maximising masked ZNCC of the band-passed fruit interior."""
    area = float((ref.mask > 0).sum())
    if area < 100:
        return DenseResult(None, 0.0, False)
    factor = DENSE_DIAMETER / math.sqrt(4 * area / math.pi)
    ref_band = _band_pass(ref.gray, factor)
    mov_band = _band_pass(mov.gray, factor)
    ref_mask = _binary_at(ref.mask, ref_band.shape)
    mov_mask = cv2.dilate(
        _binary_at(mov.mask, mov_band.shape), _ellipse(_odd(0.1 * DENSE_DIAMETER))
    )
    interior = cv2.erode(ref_mask, _ellipse(_odd(2 * DENSE_ERODE * DENSE_DIAMETER)))
    if interior.sum() < 100:
        interior = ref_mask
    ys, xs = np.nonzero(interior > 0.5)
    centre = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
    size = int(math.ceil(math.hypot(np.ptp(xs) + 1, np.ptp(ys) + 1) * 1.15)) + 4
    pad = size // 2
    padded = cv2.copyMakeBorder(
        mov_band * mov_mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
    )
    padded_mask = cv2.copyMakeBorder(mov_mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    def search(angle_deg: float, scale: float):
        patch, support, turn, shift = _template(
            ref_band, interior, centre, math.radians(angle_deg), scale, size
        )
        if support.sum() < 50:
            return None
        score = _masked_zncc(padded, padded_mask, patch * support, support)
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        return float(score[y, x]), angle_deg, scale, x, y, score

    coarse = [r for a in DENSE_ANGLES_DEG for s in DENSE_SCALES if (r := search(a, s)) is not None]
    if not coarse:
        return DenseResult(None, 0.0, False)
    _, angle0, scale0, *_ = max(coarse, key=lambda r: r[0])
    fine = [
        r
        for a in (angle0 - 1.0, angle0, angle0 + 1.0)
        for s in (scale0 - 0.03, scale0 - 0.015, scale0, scale0 + 0.015, scale0 + 0.03)
        if (r := search(a, s)) is not None
    ]
    best, angle, scale, x, y, score = max(fine + coarse, key=lambda r: r[0])
    dx, dy = _parabolic(score, x, y)
    target = np.array([x + dx + size / 2 - pad, y + dy + size / 2 - pad])
    turn = scale * np.array(
        [
            [math.cos(math.radians(angle)), -math.sin(math.radians(angle))],
            [math.sin(math.radians(angle)), math.cos(math.radians(angle))],
        ]
    )
    working = Similarity.from_matrix(np.hstack([turn, (target - turn @ centre)[:, None]]))
    sim = working.rescaled(1 / factor).rescaled(1 / ref.scale)
    rows, cols = np.mgrid[0 : score.shape[0], 0 : score.shape[1]]
    far = (rows - y) ** 2 + (cols - x) ** 2 > (DENSE_RUNNER_UP_RADIUS * DENSE_DIAMETER) ** 2
    runner_up = max((float(r[5][far].max()) for r in coarse if far.any()), default=0.0)
    return DenseResult(sim, best, best >= min_score, runner_up)
