"""NIR -> RGB spatial registration with the ArUco homography.

The two MAPIR sensors are mounted side by side, so a raw NIR frame is offset
from the RGB frame by roughly (-120, +110) px. Labels are expressed in the RGB
frame (class 1 is drawn on NIR and projected to RGB with H^-1 by
`scripts/convert_nir_labels.py`), so the NIR image must be brought into the
RGB frame before it is letterboxed, augmented or fed to a model.

Direction convention: `notebooks/matriz_homografia_aruco.npy` was estimated as
`cv2.findHomography(pts_rgb, pts_nir, ...)` (notebooks/homografia_script.py),
i.e. `H` maps RGB -> NIR. Resampling a NIR image into the RGB frame therefore
needs `cv2.warpPerspective(nir, inv(H), (w_rgb, h_rgb))`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

# Default homography shipped with the repo (RGB -> NIR, estimated from ArUco markers).
DEFAULT_NIR_HOMOGRAPHY_PATH = "notebooks/matriz_homografia_aruco.npy"


def load_homography(path: str | Path) -> np.ndarray:
    """Load and validate the RGB -> NIR homography.

    Raises:
        FileNotFoundError: `path` does not exist. There is deliberately no
            fallback: training on unregistered NIR is a silent accuracy bug.
        ValueError: the array is not a finite, invertible 3x3 matrix.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"NIR homography file not found: {path}. NIR must be registered to "
            "RGB at load time; set `nir_homography_path` to a valid .npy file "
            "(or to null to explicitly train on unregistered NIR)."
        )
    H = np.load(path)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise ValueError(f"Homography at {path} must be a finite 3x3 matrix, got shape {H.shape}.")
    H = H.astype(np.float64)
    if abs(np.linalg.det(H)) < 1e-12:
        raise ValueError(f"Homography at {path} is singular and cannot be inverted.")
    return H


def homography_sha256(path: str | Path | None) -> str | None:
    """SHA-256 of the homography file's raw bytes (None when registration is off)."""
    if path is None:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register_nir_to_rgb(
    nir: np.ndarray,
    H_rgb_to_nir: np.ndarray,
    rgb_size: tuple[int, int],
    border_value: int = 0,
) -> np.ndarray:
    """Warp a NIR image into the RGB frame.

    Args:
        nir: (H, W) or (H, W, C) uint8 NIR image in the NIR sensor frame.
        H_rgb_to_nir: 3x3 homography mapping RGB pixels to NIR pixels.
        rgb_size: (width, height) of the RGB image (output size).
        border_value: Fill for pixels with no NIR source (the dataset passes
            its NIR letterbox pad value so the fill matches the padding).

    Returns:
        NIR image of size (height, width) in the RGB frame.
    """
    return cv2.warpPerspective(
        nir,
        np.linalg.inv(H_rgb_to_nir),
        rgb_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
