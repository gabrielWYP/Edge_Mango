"""Per-photo translation that takes the NIR from the homography to the fruit.

A per-lot homography is right on its calibration board, but the fruit stands above that plane,
and the cameras sit side by side. What is left after the warp behaves like parallax, plus whatever
moved between the two exposures (``docs/findings/2026-11-01-rgb-nir-residual``). The outline of the
fruit and the skin facing the cameras need different shifts, 50-72 px apart along x on captura_v3,
and one translation cannot close both. Bruises are on the skin, so the skin decides.

Two matches per photo, of the RGB gradients against the gradients of the warped NIR at 1/2 scale.
The template comes from the RGB because that is where the fruit mask is exact; the sheet and the
shadow stay out of it, and with them their votes.

- Outline match: over a ring around the fruit outline, within the search range. It is sharp, it
  follows the fruit photo by photo, also when it moved between the exposures, and a strong mark on
  the skin cannot pull it because the ring leaves the skin out.
- Surface match: over the fruit mask inside the ring, within a window around the outline match.
  The skin has too little in common between both modalities to trust one photo, but the median of
  a lot and moment gives the parallax from the outline to the skin.

The shift of a photo is its outline match plus the parallax of its lot and moment. A photo whose
outline match is unreliable takes the median outline match of its lot and moment instead.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

SCALE = 0.5  # of the full-resolution frame; the shift has a precision of 2 px
GRADIENT_SIGMA = 1.5  # px at SCALE
SEARCH_PX = 160  # the outline match searches +- this many full-resolution px
OUTLINE_GROW_PX = 32  # the ring reaches this far outside the fruit mask, so the outline votes
SURFACE_ERODE = 0.2  # and this fraction of the equivalent radius inside; the rest is the skin
SURFACE_WINDOW_PX = (-20, 140, 16)  # dx low, dx high, +- dy around the outline match
MIN_CORR = 0.30  # outline correlation below which a match is not trusted

SOURCE_FRUIT = "fruit"  # the outline match of the photo itself
SOURCE_MEDIAN = "median"  # fell back to the median of its lot and moment
SOURCE_UNMEASURED = "unmeasured"  # nothing to fall back on: no accepted photo, zero shift
SOURCE_NONE = "none"  # the builder ran with the homography alone


@dataclass(frozen=True)
class FruitMatch:
    """What one photo measures, in full-resolution px of the RGB frame.

    Moving the warped NIR by (``dx``, ``dy``) closes its fruit outline on the RGB one. ``surface``
    is the skin shift relative to that, or None when its best shift sits on the window edge.
    """

    dx: float
    dy: float
    corr: float
    corr_zero: float
    at_limit: bool
    surface: tuple[float, float] | None = None

    @property
    def reliable(self) -> bool:
        return self.corr >= MIN_CORR and not self.at_limit


@dataclass(frozen=True)
class ResidualShift:
    """The translation applied to one photo; ``corr`` is that of its own best outline shift."""

    dx: float = 0.0
    dy: float = 0.0
    corr: float | None = None
    source: str = SOURCE_NONE


NO_SHIFT = ResidualShift()


def _gradient(image: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(image, (0, 0), GRADIENT_SIGMA)
    return cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))


def _ellipse(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _search(
    rgb_grad: np.ndarray,
    nir_grad: np.ndarray,
    mask: np.ndarray,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
) -> tuple[int, int, float, float | None, bool]:
    """Best NIR shift within both inclusive ranges, in px at SCALE.

    Returns dx, dy, the correlation there and at zero shift (None if out of range), and whether
    the best shift is on the edge of the ranges.
    """
    x, y, w, h = cv2.boundingRect(mask)
    pad = max(abs(limit) for limit in (*x_range, *y_range))
    target = cv2.copyMakeBorder(nir_grad, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    scores = cv2.matchTemplate(
        target[y + pad - y_range[1] : y + h + pad - y_range[0],
               x + pad - x_range[1] : x + w + pad - x_range[0]],
        rgb_grad[y : y + h, x : x + w],
        cv2.TM_CCOEFF_NORMED,
        mask=mask[y : y + h, x : x + w],
    )
    # A flat window divides by zero, which can also land as a huge value: neither is a match.
    scores = np.where(np.isfinite(scores) & (np.abs(scores) <= 1.0 + 1e-3), scores, -1.0)
    _, best, _, (ix, iy) = cv2.minMaxLoc(scores)
    zero = None
    if x_range[0] <= 0 <= x_range[1] and y_range[0] <= 0 <= y_range[1]:
        zero = float(scores[y_range[1], x_range[1]])
    edge = ix in (0, scores.shape[1] - 1) or iy in (0, scores.shape[0] - 1)
    return x_range[1] - ix, y_range[1] - iy, float(best), zero, bool(edge)


def measure_fruit_match(
    rgb_gray: np.ndarray, nir_gray: np.ndarray, fruit_mask: np.ndarray
) -> FruitMatch:
    """Match one photo. ``nir_gray`` is already warped into the frame of ``rgb_gray``.

    ``fruit_mask`` (0/255) covers the frame at any scale.
    """
    rgb_small = cv2.resize(rgb_gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    nir_small = cv2.resize(nir_gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    height, width = rgb_small.shape
    mask = cv2.resize(fruit_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    rgb_grad, nir_grad = _gradient(rgb_small), _gradient(nir_small)

    radius = math.sqrt(np.count_nonzero(mask) / math.pi)
    inner = cv2.erode(mask, _ellipse(round(SURFACE_ERODE * radius)))
    ring = cv2.subtract(cv2.dilate(mask, _ellipse(round(OUTLINE_GROW_PX * SCALE))), inner)
    reach = round(SEARCH_PX * SCALE)
    dx, dy, corr, corr_zero, at_limit = _search(
        rgb_grad, nir_grad, ring, (-reach, reach), (-reach, reach)
    )

    surface = None
    if inner.any():
        low, high, side = (round(v * SCALE) for v in SURFACE_WINDOW_PX)
        skin_dx, skin_dy, _, _, on_edge = _search(
            rgb_grad, nir_grad, inner, (dx + low, dx + high), (dy - side, dy + side)
        )
        if not on_edge:
            surface = ((skin_dx - dx) / SCALE, (skin_dy - dy) / SCALE)
    return FruitMatch(dx / SCALE, dy / SCALE, corr, float(corr_zero), at_limit, surface)


def _median(values: list[float]) -> float:
    return round(float(statistics.median(values)), 1) if values else 0.0


def _summarize(group: list[FruitMatch]) -> dict:
    accepted = [match for match in group if match.reliable]
    surfaces = [match.surface for match in accepted if match.surface is not None]
    parallax_dx = _median([surface[0] for surface in surfaces])
    parallax_dy = _median([surface[1] for surface in surfaces])
    return {
        "median_dx": round(_median([m.dx for m in accepted]) + parallax_dx, 1) if accepted else 0.0,
        "median_dy": round(_median([m.dy for m in accepted]) + parallax_dy, 1) if accepted else 0.0,
        "parallax_dx": parallax_dx,
        "parallax_dy": parallax_dy,
        "accepted": len(accepted),
        "fallback": len(group) - len(accepted) if accepted else 0,
        "unmeasured": 0 if accepted else len(group),
        "median_corr": round(_median([m.corr for m in accepted]), 3) if accepted else None,
        "median_corr_zero": round(statistics.median(m.corr_zero for m in group), 3),
    }


def resolve_shifts(
    keys: Sequence[Hashable], matches: Sequence[FruitMatch | None]
) -> tuple[list[ResidualShift | None], dict[Hashable, dict]]:
    """Turn the matches into shifts; ``keys`` name the group (lot, moment) of each photo.

    Returns one shift per photo (None for a photo with no match) and a summary per group. The
    result depends on the matches alone, not on their order.
    """
    groups: dict[Hashable, list[FruitMatch]] = defaultdict(list)
    for key, match in zip(keys, matches):
        if match is not None:
            groups[key].append(match)
    summary = {key: _summarize(group) for key, group in groups.items()}
    shifts: list[ResidualShift | None] = []
    for key, match in zip(keys, matches):
        if match is None:
            shifts.append(None)
            continue
        values, corr = summary[key], round(match.corr, 3)
        if match.reliable:
            shifts.append(
                ResidualShift(
                    round(match.dx + values["parallax_dx"], 1),
                    round(match.dy + values["parallax_dy"], 1),
                    corr,
                    SOURCE_FRUIT,
                )
            )
        elif values["accepted"]:
            shifts.append(
                ResidualShift(values["median_dx"], values["median_dy"], corr, SOURCE_MEDIAN)
            )
        else:
            shifts.append(ResidualShift(0.0, 0.0, corr, SOURCE_UNMEASURED))
    return shifts, summary


def residual_params() -> dict:
    """The method and its constants, for the dataset report."""
    return {
        "method": "RGB vs warped NIR gradient correlation over the RGB fruit mask. Per photo: "
        "outline match over a ring around the mask outline. Per lot and moment: parallax from the "
        "outline to the skin, the median over accepted photos of a match over the mask inside the "
        "ring. Shift = outline + parallax; an unreliable photo takes the median outline of its "
        "lot and moment + parallax",
        "scale": SCALE,
        "gradient_sigma": GRADIENT_SIGMA,
        "search_px": SEARCH_PX,
        "outline_grow_px": OUTLINE_GROW_PX,
        "surface_erode": SURFACE_ERODE,
        "surface_window_px": list(SURFACE_WINDOW_PX),
        "min_corr": MIN_CORR,
    }
