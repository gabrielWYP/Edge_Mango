"""Tests for the v4 dataset inclusion rules and Label Studio box conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data_pipeline.v4_selection import (
    LATENT_HOURS,
    box_in_rgb_frame,
    homography_for,
    load_homography_rules,
    parse_tasks,
    rect_corners,
    select_photos,
)
from tests.annotation.synthetic import JORNADA_1, make_meta
from tests.data_pipeline.v4_synth import H_SHIFT, JORNADA_3, RULES, make_task

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        task = make_task(meta, {"x": 1, "y": 1, "w": 10, "h": 10}, created_at="2026-10-28T20:40:40Z")
        task["annotations"].append(
            make_task(meta, None, created_at="2026-10-28T20:41:00Z")["annotations"][0]
        )
        labels, conflicts = parse_tasks([task])
        assert labels[1].created_at == "2026-10-28T20:41:00Z" and labels[1].corners == []
        assert conflicts == [
            {
                "timestamp": 1,
                "created_at": ["2026-10-28T20:40:40Z", "2026-10-28T20:41:00Z"],
                "used": "2026-10-28T20:41:00Z",
            }
        ]

    def test_tasks_carry_no_database_ids(self) -> None:
        """The canonical labels file holds no Label Studio ids, because they repeat across projects."""
        task = make_task(make_meta(1), {"x": 1, "y": 1, "w": 10, "h": 10})
        assert "id" not in task and "id" not in task["annotations"][0]
        labels, conflicts = parse_tasks([task])
        assert list(labels) == [1] and conflicts == []

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
            make_task(m, box)
            for m in metas
            if m["timestamp"] in (2, 3, 5, 8, 9)
        ]
        tasks.append(make_task(metas[5], box, visible=("no", "no")))
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

    def test_tolerance_exception_keeps_a_late_round_and_is_scoped_to_it(self) -> None:
        box = {"x": 1, "y": 1, "w": 10, "h": 10}
        metas = [
            make_meta(1, ronda_h=4, en_tolerancia=False),
            make_meta(2, ronda_h=8, en_tolerancia=False),
            make_meta(3, ronda_h=4, en_tolerancia=False, mango_id=9),
        ]
        labels, _ = parse_tasks([make_task(m, box) for m in metas])
        exceptions = [{"jornada": JORNADA_1, "ronda_h": 4, "mangos": [metas[0]["mango_id"]]}]
        photos, excluded = select_photos(
            metas, labels, {}, set(), RULES, tolerance_exceptions=exceptions
        )
        assert [p.capture.timestamp for p in photos] == [1]
        assert {e.capture.timestamp: e.reason for e in excluded} == {
            2: "out_of_tolerance",
            3: "out_of_tolerance",
        }

    def test_late_photos_are_excluded_without_an_exception(self) -> None:
        meta = make_meta(1, ronda_h=4, en_tolerancia=False)
        labels, _ = parse_tasks([make_task(meta, {"x": 1, "y": 1, "w": 10, "h": 10})])
        photos, excluded = select_photos([meta], labels, {}, set(), RULES)
        assert photos == [] and excluded[0].reason == "out_of_tolerance"

    def test_task_slot_must_match_the_meta(self) -> None:
        meta = make_meta(1, ronda_h=12)
        task = make_task(meta, {"x": 1, "y": 1, "w": 10, "h": 10})
        task["data"]["cara"] = 3
        labels, _ = parse_tasks([task])
        with pytest.raises(ValueError, match="slot"):
            select_photos([meta], labels, {}, set(), RULES)


class TestRepositoryConfigs:
    """The committed configs cover the five captura_v3 lots."""

    def test_every_lot_has_a_homography_rule_with_its_matrix_on_disk(self) -> None:
        rules = load_homography_rules(REPO_ROOT / "configs/data/v4_homographies.yaml")
        assert sorted(rules) == [
            "lote1-20261024",
            "lote2-20261025",
            "lote3-20261027",
            "lote4-20261031",
            "lote5-20261101",
        ]
        for lot_rules in rules.values():
            for rule in lot_rules:
                assert np.load(REPO_ROOT / rule["matrix"]).shape == (3, 3)

    def test_the_wrong_face_of_lot_4_is_excluded_by_identity(self) -> None:
        config = yaml.safe_load(
            (REPO_ROOT / "configs/data/v4_identity_exclusions.yaml").read_text(encoding="utf-8")
        )
        assert 1793488780 in {entry["timestamp"] for entry in config["exclusions"]}

    def test_tolerance_exceptions_are_the_rounds_of_the_plan(self) -> None:
        config = yaml.safe_load(
            (REPO_ROOT / "configs/data/v4_tolerance_exceptions.yaml").read_text(encoding="utf-8")
        )
        assert config["exceptions"] == [
            {"jornada": "lote4-20261031", "ronda_h": 4},
            {"jornada": "lote5-20261101", "ronda_h": 4, "mangos": [123, 124, 125]},
        ]

    def test_the_latent_window_is_the_one_of_the_plan(self) -> None:
        assert LATENT_HOURS == (0, 2, 4, 6)
