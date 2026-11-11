"""Label Studio damage polygons of the v4 dataset (opt-in, ``build_v4_dataset.py --polygons``).

The polygon export replaces the box export of ``v4_selection.parse_tasks``. A task's ``data``
carries ``rgb``, ``nir``, ``timestamp`` and ``stem`` (``mango_rgb_<timestamp>``); each damage
region is a ``polygonlabels`` result with ``from_name`` ``damage_rgb`` or ``damage_nir`` and
``value.points`` as ``[x%, y%]`` of the image. Both views are ALREADY in the RGB frame, so no
homography applies: percent -> RGB pixels is the whole conversion. The ``visible_rgb`` and
``visible_nir`` choices are read as before.

The result is a ``PolygonTaskLabel``, a ``TaskLabel`` whose ``corners`` are the (n, 2) polygons
in RGB pixels, so ``v4_selection.select_photos`` applies the same inclusion rules to it.
The slot of a capture comes from its meta, not from the task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.annotation.captura_v3_tasks import Slot
from src.data_pipeline.v4_selection import RGB_SIZE, TaskLabel

DAMAGE_FROM_NAMES = ("damage_rgb", "damage_nir")
MIN_POLYGON_POINTS = 3


@dataclass
class PolygonTaskLabel(TaskLabel):
    """A `TaskLabel` whose ``corners`` are damage polygons in RGB pixels, not box corners."""


def _timestamp(data: dict) -> int:
    if "timestamp" in data:
        return int(data["timestamp"])
    return int(str(data["stem"]).rsplit("_", 1)[-1])


def parse_polygon_tasks(
    tasks: list[dict], slots: dict[int, Slot]
) -> tuple[dict[int, PolygonTaskLabel], list[dict]]:
    """Index polygon tasks by capture timestamp; the newest annotation wins and is a conflict.

    Args:
        tasks: A Label Studio JSON export.
        slots: ``timestamp -> Slot`` of every capture meta.

    Raises:
        ValueError: A task of a capture with no meta, a polygon on an image that is not the
            RGB frame, a polygon with fewer than three points, or two tasks of one capture.
    """
    labels: dict[int, PolygonTaskLabel] = {}
    conflicts = []
    for task in tasks:
        timestamp = _timestamp(task["data"])
        if timestamp not in slots:
            raise ValueError(f"Label Studio tasks for captures with no meta: [{timestamp}]")
        done = [a for a in task["annotations"] if not a.get("was_cancelled")]
        if not done:
            continue
        annotation = max(done, key=lambda a: a["created_at"])
        if len(done) > 1:
            conflicts.append(
                {
                    "timestamp": timestamp,
                    "created_at": sorted(a["created_at"] for a in done),
                    "used": annotation["created_at"],
                }
            )
        choices, polygons, views, origins = {}, [], set(), []
        for result in annotation["result"]:
            if result["type"] == "choices":
                choices[result["from_name"]] = result["value"]["choices"][0]
            elif result["type"] == "polygonlabels" and result["from_name"] in DAMAGE_FROM_NAMES:
                size = (result.get("original_width"), result.get("original_height"))
                if None not in size and size != RGB_SIZE:
                    raise ValueError(
                        f"capture {timestamp}: polygon on a {size} image, the RGB frame is {RGB_SIZE}"
                    )
                points = np.asarray(result["value"]["points"], dtype=np.float64)
                if points.ndim != 2 or len(points) < MIN_POLYGON_POINTS:
                    raise ValueError(
                        f"capture {timestamp}: a damage polygon needs at least "
                        f"{MIN_POLYGON_POINTS} points, got {len(points)}"
                    )
                polygons.append(points / 100.0 * np.array(RGB_SIZE, dtype=np.float64))
                views.add(result["from_name"].removeprefix("damage_"))
                origins.append(result.get("origin") or "manual")
        if timestamp in labels:
            raise ValueError(f"two Label Studio tasks for capture {timestamp}")
        labels[timestamp] = PolygonTaskLabel(
            created_at=annotation["created_at"],
            slot=slots[timestamp],
            view="|".join(sorted(views)) if views else None,
            corners=polygons,
            origins=origins,
            visible_rgb=choices.get("visible_rgb"),
            visible_nir=choices.get("visible_nir"),
            rotated=False,
            n_annotations=len(done),
        )
    return labels, conflicts
