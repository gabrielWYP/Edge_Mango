"""Synthetic Label Studio export tasks for the box projection tests."""

from __future__ import annotations

import json
from pathlib import Path

from tests.annotation.synthetic import JORNADA_1

W, H = 3280, 2464


def task(
    jornada: str = JORNADA_1,
    mango: int = 1,
    cara: int = 1,
    hour: int | None = 12,
    ts: int = 100,
    box: tuple[str, float, float, float, float] | None = ("damage_nir", 10.0, 20.0, 8.0, 6.0),
    size: tuple[int, int] = (W, H),
    cancelled: bool = False,
    annotated: bool = True,
) -> dict:
    result = [
        {
            "type": "choices",
            "from_name": "visible_nir",
            "to_name": "nir",
            "value": {"choices": ["yes"]},
        }
    ]
    if box:
        name, x, y, w, h = box
        result.append(
            {
                "type": "rectanglelabels",
                "from_name": name,
                "to_name": "nir" if name == "damage_nir" else "rgb",
                "original_width": size[0],
                "original_height": size[1],
                "image_rotation": 0,
                "value": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "rotation": 0,
                    "rectanglelabels": ["damage"],
                },
            }
        )
    annotations = [
        {"was_cancelled": cancelled, "updated_at": "2026-10-26T12:00:00Z", "result": result}
    ]
    return {
        "id": ts,
        "annotations": annotations if annotated else [],
        "data": {
            "jornada": jornada,
            "mango_id": mango,
            "cara": cara,
            "ronda_h": hour,
            "timestamp": ts,
        },
    }


def write_export(tmp_path: Path, tasks: list[dict], name: str = "export.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(tasks))
    return path
