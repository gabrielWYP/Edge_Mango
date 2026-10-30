"""Tests for the v4 dataset inclusion rules and Label Studio box conversion."""

from __future__ import annotations

import numpy as np
import pytest

from src.data_pipeline.v4_selection import (
    box_in_rgb_frame,
    homography_for,
    parse_tasks,
    rect_corners,
    select_photos,
)
from tests.annotation.synthetic import make_meta
from tests.data_pipeline.v4_synth import H_SHIFT, JORNADA_3, RULES, make_task


class TestGeometry:
    def test_rotation_is_clockwise_about_the_top_left_corner(self) -> None:
        value = {"x": 10, "y": 20, "width": 10, "height": 5, "rotation": 90}
        corners = rect_corners(value, width=100, height=100)
        # (10, 20) stays; the 10 px top edge now points down (+y in image coordinates).
        assert corners == pytest.approx(np.array([[10, 20], [10, 30], [5, 30], [5, 20]]))

    def test_nir_box_is_mapped_back_with_the_inverse_homography(self) -> None:
        corners = np.array([[108, 206], [208, 206], [208, 306], [108, 306]], dtype=float)
        box = box_in_rgb_frame(corners, "nir", H_SHIFT)
        assert box == pytest.approx([100, 200, 200, 300])

    def test_rgb_box_is_only_clipped_to_the_frame(self) -> None:
        corners = np.array([[-5, 10], [50, 10], [50, 40], [-5, 40]], dtype=float)
        assert box_in_rgb_frame(corners, "rgb", H_SHIFT) == pytest.approx([0, 10, 50, 40])


class TestHomographyFor:
    def test_lot_3_switches_matrix_at_the_re_aim(self) -> None:
        assert homography_for(RULES, JORNADA_3, 100) == "morning.npy"
        assert homography_for(RULES, JORNADA_3, 200) == "closing.npy"

    def test_photo_between_the_two_pairs_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="0 homography rules"):
            homography_for(RULES, JORNADA_3, 150)


class TestParseTasks:
    def test_newest_annotation_wins_and_is_reported(self) -> None:
        meta = make_meta(1)
        task = make_task(meta, {"x": 1, "y": 1, "w": 10, "h": 10}, annotation_id=5)
        task["annotations"].append(make_task(meta, None, annotation_id=7)["annotations"][0])
        labels, conflicts = parse_tasks([task])
        assert labels[1].annotation_id == 7 and labels[1].corners == []
        assert conflicts == [{"task_id": 1, "timestamp": 1, "annotation_ids": [5, 7], "used": 7}]

    def test_boxes_on_both_views_are_rejected(self) -> None:
        meta = make_meta(1)
        task = make_task(meta, {"x": 1, "y": 1, "w": 10, "h": 10}, view="nir")
        task["annotations"][0]["result"] += make_task(
            meta, {"x": 1, "y": 1, "w": 5, "h": 5}, view="rgb"
        )["annotations"][0]["result"][2:]
        with pytest.raises(ValueError, match="both the RGB and the NIR"):
            parse_tasks([task])


class TestSelectPhotos:
    def test_inclusion_rules(self) -> None:
        box = {"x": 1, "y": 1, "w": 10, "h": 10}
        metas = [
            make_meta(1, ronda_h=None),  # sano: negative
            make_meta(2, ronda_h=12),  # labelled: kept
            make_meta(3, ronda_h=8, en_tolerancia=False),  # late retake
            make_meta(4, ronda_h=6),  # no task
            make_meta(5, ronda_h=2, cara=2),  # latent, sin_bruise face
            make_meta(6, ronda_h=12, cara=2),  # no/no at +12 h
            {**make_meta(7, ronda_h=4), "apartada": {"motivo": "x"}},  # set aside
            make_meta(8, ronda_h=4),  # its slot is kept
            make_meta(9, ronda_h=10),  # SIFT not confirmed
        ]
        tasks = [
            make_task(m, box, task_id=m["timestamp"])
            for m in metas
            if m["timestamp"] in (2, 3, 5, 8, 9)
        ]
        tasks.append(make_task(metas[5], box, task_id=6, visible=("no", "no")))
        labels, _ = parse_tasks(tasks)
        photos, excluded = select_photos(metas, labels, {9: "sift"}, {2}, RULES)

        assert [p.capture.timestamp for p in photos] == [1, 2, 6, 8]
        assert photos[0].label is None and photos[1].wb_corrected
        assert [p.provenance for p in photos] == [
            "",
            "dibujada",
            "dibujada",
            "proyectada",
        ]
        assert {e.capture.timestamp: e.reason for e in excluded} == {
            3: "out_of_tolerance",
            4: "unlabeled",
            5: "sin_bruise",
            7: "apartada",
            9: "identity_unconfirmed",
        }

    def test_task_slot_must_match_the_meta(self) -> None:
        meta = make_meta(1, ronda_h=12)
        task = make_task(meta, {"x": 1, "y": 1, "w": 10, "h": 10})
        task["data"]["cara"] = 3
        labels, _ = parse_tasks([task])
        with pytest.raises(ValueError, match="slot"):
            select_photos([meta], labels, {}, set(), RULES)
