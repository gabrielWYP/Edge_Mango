"""Horizontal line artifacts of the NIR camera (plan-experimental 3.1 rule 4).

A marginal CSI contact sometimes drops a stretch of one sensor row: the NIR shows a dark line
one pixel high (plus a few rows of JPEG ringing) whose length comes in packets of about 205 px
(3280 / 16). The line can sit on the bright sheet, on the shadow or on the fruit, so a pixel
counts as part of a line when it is much darker, in absolute and relative terms, than BOTH the
pixels ``ROW_GAP`` rows above and below it. A horizontal edge (dark on one side only) or a dark
region (dark on both sides) does not qualify.

Lines that cross the fruit are rebuilt by linear interpolation between the clean rows around
them; lines on the background only are left as they are.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ROW_GAP = 3  # reference rows, outside the JPEG halo of the line
SMOOTH_PX = 9  # horizontal box filter against JPEG noise
MIN_DROP = 12.0  # gray levels below the darker reference row
MAX_RATIO = 0.5  # and at most this fraction of it
MIN_RUN = 50  # px; the shortest real lines on captura_v3 are about 72 px
HALO = 4  # rows rebuilt on each side of the line: the JPEG ringing reaches about 4 rows
X_MARGIN = SMOOTH_PX  # px rebuilt beyond the run ends, which the smoothing blurs


@dataclass(frozen=True, order=True)
class NirLine:
    """A dark line on ``row`` covering columns ``[x0, x1)`` of the NIR frame."""

    row: int
    x0: int
    x1: int


def detect_lines(gray: np.ndarray) -> list[NirLine]:
    """Line artifacts of a single-channel NIR image, sorted by row and column."""
    g = cv2.blur(np.asarray(gray, dtype=np.float32), (SMOOTH_PX, 1))
    mid = g[ROW_GAP:-ROW_GAP]
    ref = np.minimum(g[: -2 * ROW_GAP], g[2 * ROW_GAP :])
    mask = (ref - mid >= MIN_DROP) & (mid <= MAX_RATIO * ref)
    edges = np.diff(np.pad(mask, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    runs = []
    for y in np.flatnonzero(mask.sum(axis=1) >= MIN_RUN):
        starts, ends = np.flatnonzero(edges[y] == 1), np.flatnonzero(edges[y] == -1)
        runs += [
            (int(y) + ROW_GAP, int(a), int(b)) for a, b in zip(starts, ends) if b - a >= MIN_RUN
        ]
    return _merge(runs)


def _merge(runs: list[tuple[int, int, int]]) -> list[NirLine]:
    """One line per group of overlapping runs on rows at most ``HALO`` apart."""
    groups: list[list[int]] = []  # [first row, last row, x0, x1]
    for row, x0, x1 in sorted(runs):
        for group in groups:
            if row - group[1] <= HALO and x0 < group[3] and group[2] < x1:
                group[1], group[2], group[3] = row, min(group[2], x0), max(group[3], x1)
                break
        else:
            groups.append([row, row, x0, x1])
    return sorted(NirLine(round((first + last) / 2), x0, x1) for first, last, x0, x1 in groups)


def interpolate_lines(image: np.ndarray, lines: list[NirLine]) -> np.ndarray:
    """Rebuild rows ``row +- HALO`` of each line between the rows just outside them."""
    out = image.astype(np.float32)
    height, width = image.shape[:2]
    for line in lines:
        top, bottom = max(line.row - HALO - 1, 0), min(line.row + HALO + 1, height - 1)
        xs = slice(max(line.x0 - X_MARGIN, 0), min(line.x1 + X_MARGIN, width))
        for y in range(top + 1, bottom):
            t = (y - top) / (bottom - top)
            out[y, xs] = (1 - t) * out[top, xs] + t * out[bottom, xs]
    return np.clip(np.rint(out), 0, 255).astype(image.dtype)


def line_corners(line: NirLine) -> np.ndarray:
    """Edge-based corners of the pixels of ``line``, as for a box drawn on the NIR view."""
    return np.array(
        [
            [line.x0, line.row],
            [line.x1, line.row],
            [line.x1, line.row + 1],
            [line.x0, line.row + 1],
        ],
        dtype=np.float64,
    )
