"""White-balance correction for captura_v3 RGB photos shot with the camera's automatic WB.

The correction is a diagonal von Kries transform in linear light: each channel is scaled by
``reference / observed`` where both are the mean of the background sheet in the capture QC border
frame (``capture.calidad.region_borde``). Working in linear space keeps the gains physical (a camera
white-balance gain multiplies linear sensor values); doing it on gamma-encoded values would not.
"""

import cv2
import numpy as np

from capture.calidad import region_borde

class WhiteBalanceError(ValueError):
    """A white-balance pre-condition failed (e.g. a border channel with no signal)."""


BORDER_FRAC = 0.10
_CLIP_EPS = 1e-9
_LEVELS = np.arange(256, dtype=np.float64) / 255.0
_LINEAR_LUT = np.where(_LEVELS <= 0.04045, _LEVELS / 12.92, ((_LEVELS + 0.055) / 1.055) ** 2.4)

# Mango mask: pixels yellower (b*) than the sheet, brighter than the dark tray frame / shadow cores.
MASK_MIN_YELLOW = 12.0
MASK_MIN_L = 12.0
MASK_SHEET_MIN_L = 40.0
MASK_MIN_AREA_FRAC = 0.005


def srgb_to_linear(img):
    """uint8 sRGB values -> linear light in [0, 1] (standard sRGB EOTF)."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        raise ValueError("srgb_to_linear expects uint8 input")
    return _LINEAR_LUT[arr]


def linear_to_srgb(lin):
    """Linear light -> uint8 sRGB (out-of-range values are clipped)."""
    x = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    enc = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.rint(enc * 255.0).astype(np.uint8)


def border_means_linear(img, frac=BORDER_FRAC):
    """Per-channel mean of linear light over the QC border frame of an HxWx3 uint8 image."""
    mask = region_borde(img.shape, frac)
    return srgb_to_linear(img[mask]).mean(axis=0)


def chroma(means):
    """(R/G, B/G) of a per-channel mean."""
    r, g, b = (float(v) for v in means)
    if not g > 0:
        raise WhiteBalanceError(f"cannot take chroma: the green border mean is zero or invalid ({g})")
    return r / g, b / g


def von_kries_gains(reference_means, observed_means):
    """Per-channel gains that map the observed border means onto the reference ones."""
    ref = np.asarray(reference_means, dtype=np.float64)
    obs = np.asarray(observed_means, dtype=np.float64)
    if ref.shape != (3,) or obs.shape != (3,):
        raise ValueError("expected three per-channel means")
    if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(obs))):
        raise WhiteBalanceError(f"border means must be finite (reference {ref.tolist()}, observed {obs.tolist()})")
    if not (np.all(ref > 0) and np.all(obs > 0)):
        raise WhiteBalanceError(
            f"a border channel mean is zero or negative (reference {ref.tolist()}, observed {obs.tolist()})")
    return ref / obs


def apply_gains(img, gains):
    """Scale each channel in linear light and re-encode to uint8 sRGB.

    Returns ``(image, clipped_fraction)``; a pixel counts as clipped when any channel exceeds 1.0
    after the gain. Diagonal gains act per channel, so the transform is a 256-entry LUT per channel.
    """
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected an HxWx3 uint8 image")
    g = np.asarray(gains, dtype=np.float64)
    if g.shape != (3,) or not np.all(np.isfinite(g)) or np.any(g <= 0):
        raise ValueError("gains must be three positive finite numbers")
    out = np.empty_like(img)
    clipped = np.zeros(img.shape[:2], dtype=bool)
    for c in range(3):
        scaled = _LINEAR_LUT * g[c]
        out[..., c] = linear_to_srgb(scaled)[img[..., c]]
        clipped |= (scaled > 1.0 + _CLIP_EPS)[img[..., c]]
    return out, float(clipped.mean())


def srgb_to_lab(img):
    """uint8 sRGB (..., 3) -> CIELAB (D65), float32 (..., 3)."""
    arr = np.asarray(img)
    flat = arr.reshape(-1, 1, 3).astype(np.float32) / 255.0
    return cv2.cvtColor(flat, cv2.COLOR_RGB2Lab).reshape(arr.shape)


def delta_e76(lab_a, lab_b):
    """CIE76 colour difference: Euclidean distance in CIELAB."""
    return float(np.linalg.norm(np.asarray(lab_a, dtype=np.float64) - np.asarray(lab_b, dtype=np.float64)))


def mango_mask(img, frac=BORDER_FRAC):
    """Boolean mask of the fruit on a (downscaled) RGB photo of a mango on the cyan sheet.

    Fruit = pixels yellower (higher b*) than the sheet, which is estimated from the bright pixels of
    the border frame; the largest connected component is kept, holes are filled and the edge is eroded.
    The cast shadow and the dark tray frame share the sheet's hue or are too dark, so they drop out.
    Returns an all-False mask when no fruit is found.
    """
    h, w = img.shape[:2]
    empty = np.zeros((h, w), dtype=bool)
    lab = srgb_to_lab(img)
    lightness, b_star = lab[..., 0], lab[..., 2]
    sheet = region_borde(img.shape, frac) & (lightness > MASK_SHEET_MIN_L)
    if sheet.sum() < 50:
        return empty
    candidate = (b_star - np.median(b_star[sheet]) > MASK_MIN_YELLOW) & (lightness > MASK_MIN_L)
    k = max(3, 2 * round(min(h, w) * 0.004) + 1)
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    if n < 2:
        return empty
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < MASK_MIN_AREA_FRAC * h * w:
        return empty
    blob = (labels == best).astype(np.uint8)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(blob)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    erode = max(1, round(min(h, w) * 0.01))
    eroded = cv2.erode(filled, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    return eroded.astype(bool)
