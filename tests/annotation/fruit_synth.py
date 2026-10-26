"""Synthetic fruit-on-a-sheet scenes for the box projection tests."""

from __future__ import annotations

import cv2
import numpy as np

from src.annotation.mango_registration import Similarity

SIZE = (480, 640)  # height, width
AXES = (105, 80)
CENTER = (330.0, 240.0)


def _noise(
    rng: np.random.Generator, shape: tuple[int, int], sigma: float, std: float
) -> np.ndarray:
    field = cv2.GaussianBlur(rng.normal(size=shape).astype(np.float32), (0, 0), sigma)
    return field / field.std() * std


def fruit_mask(shape: tuple[int, int] = SIZE, center=CENTER, axes=AXES) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(
        mask, (int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), 20, 0, 360, 255, -1
    )
    return mask


def fruit_layer(
    seed: int, shape: tuple[int, int] = SIZE, center=CENTER, axes=AXES
) -> tuple[np.ndarray, np.ndarray]:
    """Return (gray texture, mask) of a speckled fruit."""
    rng = np.random.default_rng(seed)
    texture = 135 + _noise(rng, shape, 6, 10) + _noise(rng, shape, 2, 7) + _noise(rng, shape, 1, 4)
    mask = fruit_mask(shape, center, axes)
    ys, xs = np.nonzero(mask)
    zoom = shape[1] / SIZE[1]
    for _ in range(90):
        i = rng.integers(len(ys))
        radius = int(rng.integers(2, 5) * zoom)
        cv2.circle(texture, (int(xs[i]), int(ys[i])), radius, float(rng.integers(40, 80)), -1)
    texture = cv2.GaussianBlur(texture, (0, 0), 1.2)
    return np.clip(texture, 0, 255).astype(np.uint8), mask


def sheet(seed: int, shape: tuple[int, int] = SIZE) -> np.ndarray:
    """A static textured background with creases (strong, repeatable features)."""
    rng = np.random.default_rng(seed)
    base = 200 + _noise(rng, shape, 8, 14) + _noise(rng, shape, 2, 8)
    for _ in range(40):
        p = rng.integers(0, [shape[1], shape[0]], size=2)
        q = rng.integers(0, [shape[1], shape[0]], size=2)
        cv2.line(
            base,
            tuple(int(v) for v in p),
            tuple(int(v) for v in q),
            float(rng.integers(120, 255)),
            2,
        )
    return np.clip(cv2.GaussianBlur(base, (0, 0), 1.0), 0, 255).astype(np.uint8)


def scene(
    background: np.ndarray, texture: np.ndarray, mask: np.ndarray, motion: Similarity | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Composite the fruit over the background, moved by ``motion``; return (image, fruit mask)."""
    if motion is not None:
        matrix = motion.cv_matrix()
        size = (mask.shape[1], mask.shape[0])
        texture = cv2.warpAffine(texture, matrix, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST)
    alpha = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 1.0)
    image = background.astype(np.float32) * (1 - alpha) + texture.astype(np.float32) * alpha
    return np.clip(image, 0, 255).astype(np.uint8), mask


def motion_about_center(
    scale: float, angle_deg: float, shift: tuple[float, float], center=CENTER
) -> Similarity:
    """Rotate/scale about the fruit centre, then move by ``shift`` (edge-based pixel coordinates)."""
    turn = Similarity(scale, angle_deg)
    centre = np.array(center, dtype=float)
    moved = turn.apply(centre[None])[0]
    return Similarity(scale, angle_deg, *(centre - moved + np.array(shift)))


MOTION = motion_about_center(1.03, 8.0, (40.0, -25.0))


def color_scene(
    background: np.ndarray, texture: np.ndarray, mask: np.ndarray, motion: Similarity | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """BGR version of ``scene``: yellow fruit on a cyan sheet."""
    if motion is not None:
        matrix = motion.cv_matrix()
        size = (mask.shape[1], mask.shape[0])
        texture = cv2.warpAffine(texture, matrix, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST)
    alpha = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 1.0)[..., None]
    fruit = texture.astype(np.float32)[..., None] * np.array([0.25, 1.0, 1.1], np.float32)
    paper = background.astype(np.float32)[..., None] * np.array([1.0, 1.0, 0.4], np.float32)
    return np.clip(paper * (1 - alpha) + fruit * alpha, 0, 255).astype(np.uint8), mask
