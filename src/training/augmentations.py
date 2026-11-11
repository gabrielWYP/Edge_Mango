"""Bbox-aware augmentation pipelines for YOLO-style detection training.

The master model consumes aligned RGB/NIR pairs. Spatial transforms must be
applied to RGB, NIR, and bounding boxes with the same sampled parameters;
photometric transforms are applied only to RGB so NIR intensity statistics remain
physically meaningful.
"""

from __future__ import annotations

import logging
from collections import Counter

import albumentations as A

logger = logging.getLogger(__name__)


def _bbox_params(min_visibility: float, min_area: float) -> A.BboxParams:
    """Build YOLO-format bbox params shared by train and validation transforms."""
    return A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=min_visibility,
        min_area=min_area,
    )


def get_train_spatial_transforms(image_size: int = 640) -> A.Compose:
    """Build geometry-only transforms for aligned RGB/NIR pairs.

    The `nir` additional target is treated as an image so interpolation is
    continuous for rotations and perspective transforms. Every sampled spatial
    transform is shared by RGB, NIR, and YOLO bounding boxes.

    Args:
        image_size: Target image size. Letterbox already handles resizing, so
            this value is accepted for API consistency.

    Returns:
        Albumentations Compose configured for `image`, `nir`, bboxes, and labels.
    """
    _ = image_size
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=20, p=0.4, border_mode=0),
            A.Perspective(scale=(0.02, 0.05), keep_size=True, p=0.2),
            A.Transpose(p=0.1),
        ],
        bbox_params=_bbox_params(min_visibility=0.15, min_area=2.0),
        additional_targets={"nir": "image"},
    )


def get_rgb_photometric_transforms(image_size: int = 640) -> A.Compose:
    """Build RGB-only photometric transforms.

    These transforms do not modify geometry, so bounding boxes and NIR alignment
    remain valid. NIR is intentionally excluded to preserve measured reflectance
    distributions used by the multimodal teacher.

    Args:
        image_size: Target image size. Kept for API consistency.

    Returns:
        Albumentations Compose operating only on the RGB image.
    """
    _ = image_size
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.4
            ),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05,
                p=0.3,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
            A.ISONoise(p=0.15),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=114,
                p=0.2,
            ),
        ]
    )


def get_train_transforms(image_size: int = 640) -> A.Compose:
    """Build the legacy RGB-only training augmentation pipeline.

    New multimodal training code should use `get_train_spatial_transforms` plus
    `get_rgb_photometric_transforms` so RGB/NIR alignment is preserved. This
    function remains for backward-compatible tests or external callers that only
    augment RGB images.
    """
    _ = image_size
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=20, p=0.4, border_mode=0),
            A.Perspective(scale=(0.02, 0.05), keep_size=True, p=0.2),
            A.Transpose(p=0.1),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.4
            ),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05,
                p=0.3,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
            A.ISONoise(p=0.15),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=114,
                p=0.2,
            ),
        ],
        bbox_params=_bbox_params(min_visibility=0.15, min_area=2.0),
    )


def get_val_transforms(image_size: int = 640) -> A.Compose:
    """Build validation transform pipeline with no augmentation."""
    _ = image_size
    return A.Compose(
        [A.NoOp()],
        bbox_params=_bbox_params(min_visibility=0.1, min_area=1.0),
    )


def apply_transforms_with_bbox_tracking(
    transform: A.Compose,
    image,
    bboxes: list[list[float]],
    class_labels: list[int],
    *,
    nir=None,
    masks=None,
    log_level: int = logging.INFO,
) -> dict:
    """Apply an Albumentations transform and log per-class bbox drop rates.

    Spatial transforms with ``min_visibility`` / ``min_area`` filtering can
    silently drop bounding boxes — especially small damage bboxes. This wrapper
    counts bboxes per class before and after the transform call and logs the
    difference so training runs surface augmentation-induced data loss.

    Args:
        transform: An Albumentations Compose (typically from
            ``get_train_spatial_transforms`` or ``get_train_transforms``).
        image: The input image (numpy array).
        bboxes: YOLO-format bounding boxes ``[[x, y, w, h], ...]``.
        class_labels: Integer class labels aligned with *bboxes*.
        nir: Optional NIR image for multimodal transforms.
        masks: Optional list of (H, W) instance masks, transformed with the same sampled
            geometry (nearest interpolation). They are not filtered here, and the drop
            counts below only concern `bboxes`.
        log_level: Logging level for the drop summary. Defaults to INFO.

    Returns:
        The dict returned by ``transform(...)`` (keys: ``image``, ``bboxes``,
        ``class_labels``, and optionally ``nir`` and ``masks``).
    """
    # Count bboxes per class BEFORE transform
    before_counts: Counter[int] = Counter(class_labels)

    # Build transform input kwargs
    kwargs: dict = {
        "image": image,
        "bboxes": bboxes,
        "class_labels": class_labels,
    }
    if nir is not None:
        kwargs["nir"] = nir
    if masks is not None:
        kwargs["masks"] = masks

    result = transform(**kwargs)

    # Count bboxes per class AFTER transform
    after_counts: Counter[int] = Counter(result["class_labels"])

    # Log drop summary
    mango_before = before_counts.get(0, 0)
    damage_before = before_counts.get(1, 0)
    mango_after = after_counts.get(0, 0)
    damage_after = after_counts.get(1, 0)

    mango_dropped = mango_before - mango_after
    damage_dropped = damage_before - damage_after

    if mango_dropped > 0 or damage_dropped > 0:
        logger.log(
            log_level,
            "Augmentation bbox drops — before: mango=%d, damage=%d; "
            "after: mango=%d, damage=%d (dropped: mango=%d, damage=%d)",
            mango_before,
            damage_before,
            mango_after,
            damage_after,
            mango_dropped,
            damage_dropped,
        )

    return result
