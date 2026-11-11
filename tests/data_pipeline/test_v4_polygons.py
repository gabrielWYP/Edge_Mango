"""v4 builder, opt-in polygon path: Label Studio polygons + fruit-mask contour -> YOLO-seg labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import build_v4_dataset as build_cli
from src.annotation.captura_v3_tasks import Slot
from src.data_pipeline.v4_dataset import INDEX_FIELDS, build_dataset, mango_polygon
from src.data_pipeline.v4_polygons import PolygonTaskLabel, parse_polygon_tasks
from src.data_pipeline.v4_selection import RGB_SIZE
from src.training.seg_labels import parse_label_line
from tests.annotation.synthetic import JORNADA_1, make_meta, write_dataset
from tests.data_pipeline.test_v4_dataset import _write_mango_rgb
from tests.data_pipeline.v4_synth import H_SHIFT, RULES

# Damage triangle in the RGB frame, in pixels, and its Label Studio form (percent).
TRIANGLE_PX = [(1500.0, 1200.0), (1700.0, 1200.0), (1600.0, 1400.0)]


def _percent(points: list[tuple[float, float]]) -> list[list[float]]:
    return [[x / RGB_SIZE[0] * 100, y / RGB_SIZE[1] * 100] for x, y in points]


def make_polygon_task(
    meta: dict,
    polygons: list[list[tuple[float, float]]],
    from_name: str = "damage_rgb",
    created_at: str = "2026-11-02T10:00:00Z",
    visible: tuple[str, str] = ("yes", "yes"),
    extra_data: dict | None = None,
) -> dict:
    result = [
        {"type": "choices", "from_name": "visible_rgb", "value": {"choices": [visible[0]]}},
        {"type": "choices", "from_name": "visible_nir", "value": {"choices": [visible[1]]}},
    ]
    for polygon in polygons:
        result.append(
            {
                "type": "polygonlabels",
                "from_name": from_name,
                "to_name": from_name.removeprefix("damage_"),
                "original_width": RGB_SIZE[0],
                "original_height": RGB_SIZE[1],
                "origin": "manual",
                "value": {"points": _percent(polygon), "polygonlabels": ["damage"]},
            }
        )
    ts = meta["timestamp"]
    data = {
        "rgb": f"/data/mango_rgb_{ts}.jpg",
        "nir": f"/data/mango_nir_{ts}.jpg",
        "timestamp": ts,
        "stem": f"mango_rgb_{ts}",
        **(extra_data or {}),
    }
    return {"data": data, "annotations": [{"created_at": created_at, "result": result}]}


SLOTS = {2: Slot(JORNADA_1, 1, 1, 12)}


class TestParsePolygonTasks:
    def test_points_become_pixel_polygons_in_the_rgb_frame(self) -> None:
        task = make_polygon_task(make_meta(2, ronda_h=12), [TRIANGLE_PX])
        labels, conflicts = parse_polygon_tasks([task], {2: Slot(JORNADA_1, 1, 1, 12)})
        label = labels[2]
        assert isinstance(label, PolygonTaskLabel) and conflicts == []
        assert label.view == "rgb" and label.slot == Slot(JORNADA_1, 1, 1, 12)
        assert len(label.corners) == 1
        assert label.corners[0] == pytest.approx(np.array(TRIANGLE_PX), abs=1e-6)
        assert (label.visible_rgb, label.visible_nir) == ("yes", "yes")
        assert label.origins == ["manual"]

    def test_both_views_are_accepted_in_one_task(self) -> None:
        meta = make_meta(2, ronda_h=12)
        task = make_polygon_task(meta, [TRIANGLE_PX])
        nir_task = make_polygon_task(meta, [TRIANGLE_PX], from_name="damage_nir")
        task["annotations"][0]["result"] += nir_task["annotations"][0]["result"][2:]
        labels, _ = parse_polygon_tasks([task], {2: Slot(JORNADA_1, 1, 1, 12)})
        assert labels[2].view == "nir|rgb" and len(labels[2].corners) == 2

    def test_newest_annotation_wins_and_cancelled_ones_are_ignored(self) -> None:
        meta = make_meta(2, ronda_h=12)
        old = make_polygon_task(meta, [TRIANGLE_PX], created_at="2026-11-01T00:00:00Z")
        new = make_polygon_task(meta, [TRIANGLE_PX[:3], TRIANGLE_PX[:3]], created_at="2026-11-03T00:00:00Z")
        cancelled = make_polygon_task(meta, [], created_at="2026-11-09T00:00:00Z")
        cancelled["annotations"][0]["was_cancelled"] = True
        task = {
            "data": old["data"],
            "annotations": old["annotations"] + new["annotations"] + cancelled["annotations"],
        }
        labels, conflicts = parse_polygon_tasks([task], {2: Slot(JORNADA_1, 1, 1, 12)})
        assert len(labels[2].corners) == 2 and labels[2].n_annotations == 2
        assert conflicts[0]["used"] == "2026-11-03T00:00:00Z"

    def test_timestamp_falls_back_to_the_stem(self) -> None:
        task = make_polygon_task(make_meta(2, ronda_h=12), [TRIANGLE_PX])
        del task["data"]["timestamp"]
        labels, _ = parse_polygon_tasks([task], {2: Slot(JORNADA_1, 1, 1, 12)})
        assert list(labels) == [2]

    def test_polygon_on_a_wrong_sized_image_is_refused(self) -> None:
        task = make_polygon_task(make_meta(2, ronda_h=12), [TRIANGLE_PX])
        task["annotations"][0]["result"][2]["original_width"] = 3280
        with pytest.raises(ValueError, match="3280"):
            parse_polygon_tasks([task], {2: Slot(JORNADA_1, 1, 1, 12)})

    def test_a_polygon_with_fewer_than_three_points_is_refused(self) -> None:
        task = make_polygon_task(make_meta(2, ronda_h=12), [TRIANGLE_PX[:2]])
        with pytest.raises(ValueError, match="points"):
            parse_polygon_tasks([task], {2: Slot(JORNADA_1, 1, 1, 12)})

    def test_a_task_of_an_unknown_capture_is_refused(self) -> None:
        task = make_polygon_task(make_meta(2, ronda_h=12), [TRIANGLE_PX])
        with pytest.raises(ValueError, match="no meta"):
            parse_polygon_tasks([task], {})

    def test_a_task_without_polygons_has_no_corners(self) -> None:
        task = make_polygon_task(make_meta(2, ronda_h=12), [], visible=("no", "no"))
        labels, _ = parse_polygon_tasks([task], {2: Slot(JORNADA_1, 1, 1, 12)})
        assert labels[2].corners == [] and labels[2].view is None


class TestMangoPolygon:
    def test_largest_external_contour_in_full_resolution_pixels(self) -> None:
        mask = np.zeros((RGB_SIZE[1] // 4, RGB_SIZE[0] // 4), np.uint8)
        cv2.ellipse(mask, (400, 300), (125, 100), 0, 0, 360, 255, -1)
        cv2.circle(mask, (50, 50), 8, 255, -1)  # a smaller blob that must not win
        polygon = mango_polygon(mask)
        assert polygon.shape[1] == 2 and len(polygon) >= 8
        x0, y0 = polygon.min(axis=0)
        x1, y1 = polygon.max(axis=0)
        assert (x0, y0, x1, y1) == pytest.approx((1100, 800, 2100, 1600), abs=8)

    def test_the_hole_of_a_ring_is_not_part_of_the_outline(self) -> None:
        mask = np.zeros((RGB_SIZE[1] // 4, RGB_SIZE[0] // 4), np.uint8)
        cv2.circle(mask, (400, 300), 100, 255, -1)
        cv2.circle(mask, (400, 300), 40, 0, -1)
        polygon = mango_polygon(mask)
        assert polygon.min(axis=0)[0] == pytest.approx(1200, abs=8)


@pytest.fixture
def setup(tmp_path: Path) -> dict:
    data_root = tmp_path / "cache" / "captura_v3"
    metas = [make_meta(1, ronda_h=None), make_meta(2, ronda_h=12), make_meta(3, ronda_h=6)]
    write_dataset(data_root, metas, nir_shape=(2464, 3280))
    for meta in metas:
        _write_mango_rgb(data_root / "news" / "rgb" / f"mango_rgb_{meta['timestamp']}.jpg")
    np.save(tmp_path / "h1.npy", H_SHIFT)
    (tmp_path / "rules.yaml").write_text(json.dumps({"lots": {JORNADA_1: RULES[JORNADA_1]}}))
    (tmp_path / "identity.yaml").write_text("exclusions: []\n")
    (tmp_path / "labels.json").write_text(json.dumps([]))
    task = make_polygon_task(metas[1], [TRIANGLE_PX, [(1650.0, 1250.0), (1750.0, 1250.0), (1700.0, 1350.0)]])
    (tmp_path / "polygons.json").write_text(json.dumps([task]))
    return {
        "data_root": data_root,
        "labels_path": tmp_path / "labels.json",
        "rules_path": tmp_path / "rules.yaml",
        "identity_path": tmp_path / "identity.yaml",
        "out_dir": tmp_path / "out",
        "repo_root": tmp_path,
        "polygons_path": tmp_path / "polygons.json",
    }


class TestBuildWithPolygons:
    def test_writes_yolo_seg_lines_for_mango_and_damage(self, setup: dict) -> None:
        report = build_dataset(**setup)
        out = setup["out_dir"]
        assert report["num_included"] == 2 and report["excluded"] == {"lot1": {"unlabeled": 1}}

        sano = (out / "labels" / "mango_rgb_1.txt").read_text().splitlines()
        assert len(sano) == 1
        cls_id, box, polygon = parse_label_line(sano[0].split())
        assert cls_id == 0 and polygon is not None and len(polygon) >= 8
        assert box * np.array([*RGB_SIZE, *RGB_SIZE]) == pytest.approx(
            [1600, 1200, 1000, 800], abs=10
        )

        lines = (out / "labels" / "mango_rgb_2.txt").read_text().splitlines()
        assert [line.split()[0] for line in lines] == ["0", "1", "1"]
        _, _, triangle = parse_label_line(lines[1].split())
        assert triangle * np.array(RGB_SIZE) == pytest.approx(np.array(TRIANGLE_PX), abs=0.01)

    def test_index_describes_the_polygon_annotation(self, setup: dict) -> None:
        build_dataset(**setup)
        with (setup["out_dir"] / "index.csv").open() as handle:
            row = {r["stem"]: r for r in csv.DictReader(handle)}["mango_rgb_2"]
        assert row["n_damage"] == "2" and row["view"] == "rgb" and row["ls_origin"] == "manual"
        assert row["damage_in_mango"] == "1.0"
        assert [float(row[k]) for k in ("damage_x0", "damage_y0", "damage_x1", "damage_y1")] == [1500, 1200, 1750, 1400]

    def test_the_nir_is_still_registered_and_baked(self, setup: dict) -> None:
        build_dataset(**setup)
        nir = cv2.imread(str(setup["out_dir"] / "nir" / "mango_nir_2.png"), cv2.IMREAD_UNCHANGED)
        assert nir.shape == (RGB_SIZE[1], RGB_SIZE[0])

    def test_polygons_replace_the_box_export_as_the_inclusion_source(self, setup: dict) -> None:
        # With --polygons the box export is not read at all, so even an unreadable one is fine.
        setup["labels_path"].write_text("not json")
        assert build_dataset(**setup)["num_included"] == 2

    def test_a_damage_polygon_outside_the_frame_is_refused(self, setup: dict) -> None:
        meta = make_meta(2, ronda_h=12)
        degenerate = [(100.0, 100.0), (100.0, 100.0), (100.0, 100.0)]
        setup["polygons_path"].write_text(json.dumps([make_polygon_task(meta, [degenerate])]))
        with pytest.raises(ValueError, match="polygon"):
            build_dataset(**setup)


class TestDefaultStaysUnchanged:
    def test_default_report_and_index_have_no_polygon_trace(self, setup: dict, tmp_path: Path) -> None:
        box_task = {
            "data": {"timestamp": 2, "mango_id": 1, "cara": 1, "ronda_h": 12, "jornada": JORNADA_1},
            "annotations": [{
                "created_at": "2026-10-28T20:40:40Z",
                "result": [
                    {"type": "choices", "from_name": "visible_rgb", "value": {"choices": ["yes"]}},
                    {"type": "choices", "from_name": "visible_nir", "value": {"choices": ["yes"]}},
                    {"type": "rectanglelabels", "from_name": "damage_nir", "to_name": "nir",
                     "original_width": 3280, "original_height": 2464, "origin": "manual",
                     "value": {"x": 1508 / 3280 * 100, "y": 1206 / 2464 * 100, "width": 200 / 3280 * 100,
                               "height": 100 / 2464 * 100, "rotation": 0, "rectanglelabels": ["damage"]}},
                ],
            }],
        }
        setup["labels_path"].write_text(json.dumps([box_task]))
        del setup["polygons_path"]
        report = build_dataset(**setup)
        assert "polygons" not in json.dumps(report["inputs"]).lower()
        lines = (setup["out_dir"] / "labels" / "mango_rgb_2.txt").read_text().splitlines()
        assert [len(line.split()) for line in lines] == [5, 5]  # box lines, as always
        assert "polygon" not in ",".join(INDEX_FIELDS)

    def test_cli_exposes_the_opt_in_flag(self) -> None:
        assert build_cli.parse_args([]).polygons is None
        assert build_cli.parse_args(["--polygons", "export.json"]).polygons == Path("export.json")
