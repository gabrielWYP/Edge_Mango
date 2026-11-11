"""Polygon labels (YOLO-seg lines) and the masks rasterized from them.

A label line is either a box, `cls cx cy w h`, or a polygon, `cls x1 y1 x2 y2 ... xn yn`
(n >= 3, every coordinate normalized to the frame). A polygon also defines a box, its tight
bounding box, so a polygon file trains a detect model as well as a segment one.

Everything here is pure numpy so the dataset, the ROI crop and the v4 builder share it.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_POLYGON_POINTS = 3


def polygon_bbox_cxcywh(polygon: np.ndarray) -> np.ndarray:
    """Tight bounding box of a (n, 2) polygon, as cxcywh."""
    x1, y1 = polygon.min(axis=0)
    x2, y2 = polygon.max(axis=0)
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float32)


def parse_label_line(parts: list[str]) -> tuple[int, np.ndarray, np.ndarray | None] | None:
    """One label line -> `(class_id, box_cxcywh, polygon_or_None)`, or None when unusable.

    Five fields are a box (a line with a sixth or later field that is not a whole polygon is
    still read as a box from its first five, as before). An odd number of fields from seven
    on is a polygon, clipped to the frame; its box is its tight bounding box. Zero-size
    boxes and polygons are skipped.
    """
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    if len(parts) >= 1 + 2 * MIN_POLYGON_POINTS and len(parts) % 2 == 1:
        polygon = np.clip(np.array(parts[1:], dtype=np.float64).reshape(-1, 2), 0.0, 1.0)
        box = polygon_bbox_cxcywh(polygon)
        if box[2] <= 0 or box[3] <= 0:
            return None
        return cls_id, box, polygon
    cx, cy, w, h = (float(v) for v in parts[1:5])
    if w <= 0 or h <= 0:
        return None
    return cls_id, np.array([cx, cy, w, h], dtype=np.float32), None


def read_label_boxes(path) -> tuple[np.ndarray, np.ndarray]:
    """Boxes (N, 4 cxcywh, float32) and class ids (N,) of a label file, box or polygon lines.

    The same parsing as `YOLODataset`: degenerate lines are skipped and a polygon gives its
    tight box. Raises FileNotFoundError for a missing file.
    """
    boxes, classes = [], []
    with open(path) as handle:
        for line in handle:
            parsed = parse_label_line(line.split())
            if parsed is not None:
                classes.append(parsed[0])
                boxes.append(parsed[1])
    return (
        np.array(boxes, dtype=np.float32).reshape(-1, 4),
        np.array(classes, dtype=np.int64),
    )


def format_polygon_line(cls_id: int, polygon: np.ndarray) -> str:
    """A YOLO-seg line from a (n, 2) normalized polygon."""
    coords = " ".join(f"{v:.6f}" for v in np.asarray(polygon, dtype=np.float64).reshape(-1))
    return f"{int(cls_id)} {coords}"


def transform_polygons_letterbox(
    polygons: list[np.ndarray | None],
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    target_size: int,
) -> list[np.ndarray | None]:
    """Normalized polygons of the original image -> pixel polygons of the letterboxed one.

    The same transform as `YOLODataset._rescale_bboxes`, kept in pixels and unclipped: the
    rasterizer clips, so a polygon leaving the canvas loses no shape on the way.
    """
    _ = target_size
    out: list[np.ndarray | None] = []
    for polygon in polygons:
        if polygon is None:
            out.append(None)
            continue
        px = polygon * np.array([orig_w, orig_h], dtype=np.float64) * scale
        out.append(px + np.array([pad_x, pad_y], dtype=np.float64))
    return out


def rasterize_polygons(polygons_px: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """Pixel polygons -> (n, H, W) uint8 masks (1 inside), `size` = (H, W)."""
    height, width = size
    masks = np.zeros((len(polygons_px), height, width), dtype=np.uint8)
    for i, polygon in enumerate(polygons_px):
        cv2.fillPoly(masks[i], [np.round(polygon).astype(np.int32).reshape(-1, 1, 2)], 1)
    return masks


def mask_bbox_cxcywh(mask: np.ndarray) -> np.ndarray | None:
    """Tight box of a (H, W) mask as cxcywh normalized to its size, None when it is empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    height, width = mask.shape
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return np.array(
        [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height],
        dtype=np.float32,
    )


def downsample_masks(masks: np.ndarray, ratio: int) -> np.ndarray:
    """(n, H, W) binary masks -> (n, H // ratio, W // ratio), a cell is on when half of it was."""
    n, height, width = masks.shape
    if ratio == 1:
        return masks.astype(np.uint8)
    out = np.zeros((n, height // ratio, width // ratio), dtype=np.uint8)
    for i in range(n):
        small = cv2.resize(
            masks[i].astype(np.float32), (width // ratio, height // ratio), interpolation=cv2.INTER_AREA
        )
        out[i] = small >= 0.5
    return out
