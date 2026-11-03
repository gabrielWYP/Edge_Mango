"""Synthetic Label Studio tasks and homography rules for the v4 dataset tests."""

from __future__ import annotations

import numpy as np

from src.data_pipeline.v4_selection import RGB_SIZE
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
