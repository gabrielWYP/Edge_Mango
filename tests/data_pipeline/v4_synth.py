"""Synthetic Label Studio tasks and homography rules for the v4 dataset tests."""

from __future__ import annotations

import cv2
import numpy as np

from src.data_pipeline.v4_selection import RGB_SIZE, VIEW_SIZES
from tests.annotation.synthetic import JORNADA_1

JORNADA_3 = "lote3-20261027"
RULES = {
    JORNADA_1: [{"matrix": "h1.npy"}],
    JORNADA_3: [
        {"matrix": "morning.npy", "max_timestamp": 100},
        {"matrix": "closing.npy", "min_timestamp": 200},
    ],
}
SHIFT = (8.0, 6.0)  # RGB -> NIR translation of the synthetic homography
H_SHIFT = np.array([[1, 0, SHIFT[0]], [0, 1, SHIFT[1]], [0, 0, 1]], dtype=np.float64)


def make_task(
    meta: dict,
    box: dict | None,
    view: str = "nir",
    created_at: str = "2026-10-28T20:40:40Z",
    visible: tuple[str, str] = ("yes", "yes"),
) -> dict:
    """A Label Studio task with one annotation; ``box`` holds pixel x, y, w, h (and rotation)."""
    result = [
        {
            "type": "choices",
            "from_name": "visible_rgb",
            "value": {"choices": [visible[0]]},
        },
        {
            "type": "choices",
            "from_name": "visible_nir",
            "value": {"choices": [visible[1]]},
        },
    ]
    if box is not None:
        width, height = (3280, 2464) if view == "nir" else RGB_SIZE
        result.append(
            {
                "type": "rectanglelabels",
                "from_name": f"damage_{view}",
                "to_name": view,
                "original_width": width,
                "original_height": height,
                "origin": "manual",
                "value": {
                    "x": box["x"] / width * 100,
                    "y": box["y"] / height * 100,
                    "width": box["w"] / width * 100,
                    "height": box["h"] / height * 100,
                    "rotation": box.get("rotation", 0),
                    "rectanglelabels": ["damage"],
                },
            }
        )
    data = {k: meta[k] for k in ("timestamp", "mango_id", "cara", "ronda_h")}
    data["jornada"] = meta["jornada"]["id"]
    return {
        "data": data,
        "annotations": [{"created_at": created_at, "result": result}],
    }


FRUIT_CENTER = (1600, 1200)
FRUIT_AXES = (500, 400)  # bounding box [1100, 800, 2100, 1600]
SHEET_BGR = (230, 220, 150)
SHEET_SHADE = 0.5


def fruit_photos(
    shift: tuple[int, int] = (0, 0),
    surface: tuple[int, int] = (0, 0),
    seed: int = 1,
    sheet_clutter: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """An RGB photo (BGR) and its NIR gray image, already in the RGB frame.

    Moving the NIR fruit by ``shift`` closes its outline on the RGB one. ``surface`` moves the
    marks on the skin further than the outline, as parallax does on the visible surface.
    ``sheet_clutter`` adds sharp grid lines and a shadow edge that do not move with the fruit.
    The grid and the shadow keep the hue of the sheet, so they stay out of the RGB fruit mask.
    """
    width, height = RGB_SIZE
    rgb = np.full((height, width, 3), SHEET_BGR, dtype=np.uint8)
    nir = np.full((height, width), 190, dtype=np.uint8)
    if sheet_clutter:
        dark = tuple(int(c * SHEET_SHADE) for c in SHEET_BGR)
        # Uneven, or the grid would match itself at its own period.
        for pos in np.cumsum(np.random.default_rng(5).integers(80, 200, 30)).tolist():
            for image, color in ((rgb, dark), (nir, 60)):
                cv2.line(image, (pos, 0), (pos, height), color, 10)
                cv2.line(image, (0, pos), (width, pos), color, 10)
        # The grid stops short of the fruit; the shadow touches it, as a real one does.
        clear = (FRUIT_AXES[0] + 60, FRUIT_AXES[1] + 60)
        cv2.ellipse(rgb, FRUIT_CENTER, clear, 0, 0, 360, SHEET_BGR, -1)
        cv2.ellipse(nir, FRUIT_CENTER, clear, 0, 0, 360, 190, -1)
        for image, color in ((rgb, dark), (nir, 60)):
            cv2.ellipse(image, (1250, 1250), (520, 330), 0, 0, 360, color, -1)
    moved = (FRUIT_CENTER[0] - shift[0], FRUIT_CENTER[1] - shift[1])
    cv2.ellipse(rgb, FRUIT_CENTER, FRUIT_AXES, 0, 0, 360, (30, 200, 220), -1)
    cv2.ellipse(nir, moved, FRUIT_AXES, 0, 0, 360, 100, -1)
    rng = np.random.default_rng(seed)
    for _ in range(70):
        angle, reach = rng.uniform(0, 2 * np.pi), 0.75 * np.sqrt(rng.uniform())
        x = round(FRUIT_CENTER[0] + reach * FRUIT_AXES[0] * np.cos(angle))
        y = round(FRUIT_CENTER[1] + reach * FRUIT_AXES[1] * np.sin(angle))
        radius = int(rng.integers(8, 24))
        cv2.circle(rgb, (x, y), radius, (20, 120, 140), -1)
        cv2.circle(nir, (x - shift[0] - surface[0], y - shift[1] - surface[1]), radius, 40, -1)
    return rgb, cv2.GaussianBlur(nir, (0, 0), 1.0)


def nir_sensor_photo(nir_in_rgb_frame: np.ndarray) -> np.ndarray:
    """The NIR JPEG content (3 channels, sensor size) that H_SHIFT warps to ``nir_in_rgb_frame``."""
    moved = cv2.warpAffine(
        nir_in_rgb_frame,
        np.float64([[1, 0, SHIFT[0]], [0, 1, SHIFT[1]]]),
        VIEW_SIZES["nir"],
        borderMode=cv2.BORDER_REPLICATE,
    )
    return cv2.cvtColor(moved, cv2.COLOR_GRAY2BGR)
