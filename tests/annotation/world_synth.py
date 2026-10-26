"""A tiny captura_v3 tree of synthetic fruit photos for the projection run tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.annotation.box_projection import box_to_percent
from tests.annotation.export_synth import task, write_export
from tests.annotation.fruit_synth import color_scene, fruit_layer, motion_about_center, sheet
from tests.annotation.synthetic import make_meta, write_dataset

SHAPE = (720, 960)
HEIGHT, WIDTH = SHAPE
CENTRE = (480.0, 360.0)
AXES = (160, 120)
HAND = (420.0, 315.0, 135.0, 100.0)
HAND_2 = (390.0, 300.0, 150.0, 110.0)
MOTIONS = {
    10: motion_about_center(1.0, 6.0, (40.0, -22.0), CENTRE),
    8: motion_about_center(1.02, -9.0, (-30.0, 38.0), CENTRE),
}


def write_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])


def metas_for(first_ts: int, cara: int, mango: int = 1) -> list[dict]:
    return [
        make_meta(first_ts, mango_id=mango, cara=cara, ronda_h=12),
        make_meta(first_ts + 1, mango_id=mango, cara=cara, ronda_h=10),
        make_meta(first_ts + 2, mango_id=mango, cara=cara, ronda_h=8),
    ]


def write_rgb_world(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Two faces (different fruits) over the same sheet, photographed at +12, +10 and +8 h.

    Returns (data_root, export path, document root).
    """
    data_root = tmp_path / "data" / "cache" / "captura_v3"
    write_dataset(data_root, metas_for(1, 1) + metas_for(11, 2))
    background = sheet(seed=2, shape=SHAPE)
    for first_ts, seed in ((1, 1), (11, 5)):
        texture, mask = fruit_layer(seed, SHAPE, CENTRE, AXES)
        for offset, hour in ((0, 12), (1, 10), (2, 8)):
            image, _ = color_scene(background, texture, mask, MOTIONS.get(hour))
            write_jpeg(data_root / "news" / "rgb" / f"mango_rgb_{first_ts + offset}.jpg", image)
    boxes = {1: HAND, 2: HAND_2}
    export = write_export(
        tmp_path,
        [
            task(
                cara=cara,
                ts=ts,
                box=("damage_rgb", *box_to_percent(boxes[cara], WIDTH, HEIGHT).values()),
                size=(WIDTH, HEIGHT),
            )
            for cara, ts in ((1, 1), (2, 11))
        ],
    )
    return data_root, export, tmp_path / "data"
