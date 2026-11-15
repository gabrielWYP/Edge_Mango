"""Python reference for app-side pre/post-processing of the exported student.

This is the executable spec the mobile app mirrors:

1. `preprocess_bgr` — the exact eval-time preprocessing of
   `YOLODataset.__getitem__` for the val/test splits (BGR->RGB, centered
   letterbox with pad 114, /255, ImageNet mean/std). It calls the dataset's
   own `letterbox` and `normalize_rgb`, so it cannot drift from training.
2. `decode_exported_outputs` — turns the RAW exported tensors
   (`boxes_raw`, `cls_logits`, see `src/export/layout.py`) into detections.
   It only re-shapes them back to per-level maps and delegates to
   `src.training.decode.decode_detections`, the decode used by training
   evaluation, so the math (sigmoid, per-class candidates, exp sizes,
   per-class NMS, max-detections cap) exists in exactly one place.
3. `unletterbox_boxes` — maps detections from the 640x640 letterboxed frame
   back to original image pixels.
4. `preprocess_roi_bgr` / `roi_letterbox_to_source` — the same preprocessing for a model trained with
   `roi_crop` (crop to the mango box + margin, then letterbox) and the inverse map of a point of the
   letterboxed input to the normalized source frame, as `scripts/evaluate_checkpoint.py` does it.
5. `decode_exported_segmentation` — the same decode for a segment export
   (`mask_coefs`, `protos` besides the two raw tensors): the boxes are those of
   (2), and each detection's mask is `sigmoid(coefs . protos)` cropped to its
   box and thresholded, by `src.training.decode.process_masks`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from src.export.layout import unflatten_levels
from src.training.dataset import letterbox, normalize_rgb
from src.training.decode import decode_detections, decode_detections_with_coefs, process_masks
from src.training.roi_crop import MANGO_CLASS_ID, crop_to_roi
from src.training.seg_labels import read_label_boxes


@dataclass(frozen=True)
class LetterboxInfo:
    """Geometry needed to map boxes back to the original image."""

    orig_w: int
    orig_h: int
    scale: float
    pad_x: int
    pad_y: int


def preprocess_bgr(
    image_bgr: np.ndarray,
    image_size: int = 640,
    pad_value: int = 114,
) -> tuple[torch.Tensor, LetterboxInfo]:
    """Eval-time preprocessing of one `cv2.imread` (BGR uint8) image.

    Returns:
        (1, 3, image_size, image_size) float32 NCHW tensor, and the
        letterbox geometry. Use `.permute(0, 2, 3, 1)` for the NHWC model.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("preprocess_bgr expects an (H, W, 3) BGR uint8 image.")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    padded, scale, pad_x, pad_y = letterbox(rgb, image_size, pad_value)
    tensor = normalize_rgb(padded).unsqueeze(0)
    info = LetterboxInfo(
        orig_w=int(rgb.shape[1]), orig_h=int(rgb.shape[0]),
        scale=float(scale), pad_x=int(pad_x), pad_y=int(pad_y),
    )
    return tensor, info


@dataclass(frozen=True)
class RoiLetterboxInfo:
    """Crop window and letterbox geometry of one image: source pixel -> model input pixel.

    `crop_*` is the window cut from the source frame (the whole frame when there is no ROI
    crop); `scale`, `pad_x`, `pad_y` are the letterbox of that window.
    """

    orig_w: int
    orig_h: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    scale: float
    pad_x: int
    pad_y: int


def preprocess_roi_bgr(
    image_bgr: np.ndarray,
    label_path: str | Path | None,
    image_size: int = 640,
    pad_value: int = 114,
    roi_margin: float | None = 0.10,
) -> tuple[torch.Tensor, RoiLetterboxInfo]:
    """Eval-time preprocessing of a model trained with `roi_crop` (`YOLODataset` without jitter).

    The window is the ground-truth mango box (class 0, the largest one) of `label_path` plus
    `roi_margin` per side, cut by `crop_to_roi` exactly as in training; an image without a mango
    box, `label_path=None` or `roi_margin=None` stays full-frame. Then BGR->RGB, `letterbox`
    and `normalize_rgb`, so the tensor equals the dataset's val/test item.

    Returns:
        (1, 3, image_size, image_size) float32 NCHW tensor and the geometry to map detections
        back to the source frame (`roi_letterbox_to_source`).
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("preprocess_roi_bgr expects an (H, W, 3) BGR uint8 image.")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    crop_x, crop_y = 0, 0
    if roi_margin is not None and label_path is not None:
        boxes, labels = read_label_boxes(label_path)
        # `crop_to_roi` does not return its window: crop images that hold their own coordinates.
        xs = np.broadcast_to(np.arange(width, dtype=np.int32), (height, width))
        ys = np.broadcast_to(np.arange(height, dtype=np.int32)[:, None], (height, width))
        rgb_c, xs_c = crop_to_roi(rgb, xs, boxes, labels, roi_class=MANGO_CLASS_ID, margin=roi_margin)[:2]
        _, ys_c = crop_to_roi(rgb, ys, boxes, labels, roi_class=MANGO_CLASS_ID, margin=roi_margin)[:2]
        crop_x, crop_y, rgb = int(xs_c[0, 0]), int(ys_c[0, 0]), rgb_c
    padded, scale, pad_x, pad_y = letterbox(rgb, image_size, pad_value)
    info = RoiLetterboxInfo(
        orig_w=width, orig_h=height, crop_x=crop_x, crop_y=crop_y,
        crop_w=int(rgb.shape[1]), crop_h=int(rgb.shape[0]),
        scale=float(scale), pad_x=int(pad_x), pad_y=int(pad_y),
    )
    return normalize_rgb(padded).unsqueeze(0), info


def roi_letterbox_to_source(points_xy: torch.Tensor, info: RoiLetterboxInfo) -> torch.Tensor:
    """Points `(..., 2)` in letterboxed-input pixels -> the source frame, normalized to [0, 1].

    The map `scripts/evaluate_checkpoint.py::_to_source_frame` applies to boxes and mask
    polygons: undo the letterbox, add the crop window, divide by the source size. No clipping.
    """
    points = points_xy.double()
    x = (points[..., 0] - info.pad_x) / info.scale + info.crop_x
    y = (points[..., 1] - info.pad_y) / info.scale + info.crop_y
    return torch.stack([x / info.orig_w, y / info.orig_h], dim=-1)


def decode_exported_outputs(
    boxes_raw: torch.Tensor | np.ndarray,
    cls_logits: torch.Tensor | np.ndarray,
    *,
    image_size: int,
    strides: Sequence[int],
    conf_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    per_class_candidates: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode one image's RAW exported outputs into detections.

    Args:
        boxes_raw: (1, N, 4) [dx, dy, log_w, log_h].
        cls_logits: (1, N, num_classes) raw logits.

    Returns:
        boxes: (P, 4) cxcywh in letterboxed-input pixels.
        scores: (P,) sigmoid scores.
        labels: (P,) int64 class ids.
    """
    boxes_raw = torch.as_tensor(np.asarray(boxes_raw), dtype=torch.float32)
    cls_logits = torch.as_tensor(np.asarray(cls_logits), dtype=torch.float32)
    if boxes_raw.shape[0] != 1 or cls_logits.shape[0] != 1:
        raise ValueError("decode_exported_outputs decodes a single image (batch size 1).")

    num_classes = cls_logits.shape[-1]
    reg_levels = unflatten_levels(boxes_raw, image_size, strides)
    cls_levels = unflatten_levels(cls_logits, image_size, strides)
    # `decode_detections` reads regressions from `preds[:, num_classes:]`,
    # i.e. the head's `cat([cls, reg])` layout.
    preds = [torch.cat([c, r], dim=1) for c, r in zip(cls_levels, reg_levels)]
    return decode_detections(
        preds,
        cls_levels,
        batch_idx=0,
        num_classes=num_classes,
        image_size=image_size,
        strides=tuple(strides),
        conf_threshold=conf_threshold,
        nms_iou_threshold=nms_iou_threshold,
        nms_enabled=True,
        per_class_candidates=per_class_candidates,
        max_detections=max_detections,
        normalize=False,
    )


def decode_exported_segmentation(
    boxes_raw: torch.Tensor | np.ndarray,
    cls_logits: torch.Tensor | np.ndarray,
    mask_coefs: torch.Tensor | np.ndarray,
    protos: torch.Tensor | np.ndarray,
    *,
    image_size: int,
    strides: Sequence[int],
    conf_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    per_class_candidates: bool = True,
    mask_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode one image's RAW segment outputs into detections and their masks.

    Args:
        boxes_raw, cls_logits: as in `decode_exported_outputs`.
        mask_coefs: (1, N, nm) mask coefficients per anchor, the anchor order of the boxes.
        protos: (1, nm, h, w) prototype masks, h = w = image_size / 4.
        mask_threshold: probability above which a prototype pixel belongs to the mask.

    Returns:
        boxes (P, 4) cxcywh in letterboxed-input pixels, scores (P,), labels (P,) and
        masks (P, h, w) bool at prototype resolution, cropped to their boxes. Upsample
        the masks (bilinear on the probabilities) to the input size only if a consumer needs it.
    """
    mask_coefs = torch.as_tensor(np.asarray(mask_coefs), dtype=torch.float32)
    protos = torch.as_tensor(np.asarray(protos), dtype=torch.float32)
    if mask_coefs.shape[0] != 1 or protos.shape[0] != 1:
        raise ValueError("decode_exported_segmentation decodes a single image (batch size 1).")
    boxes_raw = torch.as_tensor(np.asarray(boxes_raw), dtype=torch.float32)
    cls_logits = torch.as_tensor(np.asarray(cls_logits), dtype=torch.float32)

    num_classes = cls_logits.shape[-1]
    cls_levels = unflatten_levels(cls_logits, image_size, strides)
    preds = [torch.cat([c, r], dim=1) for c, r in zip(cls_levels, unflatten_levels(boxes_raw, image_size, strides))]
    boxes, scores, labels, coefs = decode_detections_with_coefs(
        preds,
        cls_levels,
        unflatten_levels(mask_coefs, image_size, strides),
        batch_idx=0,
        num_classes=num_classes,
        image_size=image_size,
        strides=tuple(strides),
        conf_threshold=conf_threshold,
        nms_iou_threshold=nms_iou_threshold,
        nms_enabled=True,
        per_class_candidates=per_class_candidates,
        max_detections=max_detections,
        normalize=False,
    )
    masks = process_masks(protos[0], coefs, boxes / image_size, threshold=mask_threshold)
    return boxes, scores, labels, masks


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def unletterbox_boxes(boxes_cxcywh: torch.Tensor, info: LetterboxInfo) -> torch.Tensor:
    """Letterboxed-input cxcywh pixels -> original-image xyxy pixels, clipped."""
    xyxy = cxcywh_to_xyxy(boxes_cxcywh)
    xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - info.pad_x) / info.scale).clamp(0, info.orig_w)
    xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - info.pad_y) / info.scale).clamp(0, info.orig_h)
    return xyxy
