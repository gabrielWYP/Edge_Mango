"""Tests for projecting +12 h damage boxes onto earlier photos of the same mango face."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.annotation.box_projection import (
    Bridge,
    Decision,
    Estimate,
    Face,
    HandBox,
    ProjectionRecord,
    agreement,
    attach_predictions,
    box_iou,
    box_to_percent,
    decide,
    format_validation,
    load_hand_boxes,
    percent_to_box,
    plan_faces,
    prediction_filename,
    project_box,
    report_rows,
    validate_projections,
    write_report,
)
from src.annotation.mango_registration import Similarity
from tests.annotation.export_synth import task, write_export
from tests.annotation.synthetic import JORNADA_1, make_meta, write_dataset

W, H = 3280, 2464
BOX = (1000.0, 800.0, 260.0, 200.0)


def shifted(dx: float, dy: float = 0.0) -> Similarity:
    return Similarity(1.0, 0.0, dx, dy)


class TestPercentConversion:
    def test_percent_to_pixels_uses_the_original_size(self) -> None:
        value = {"x": 25.0, "y": 50.0, "width": 10.0, "height": 5.0}

        assert percent_to_box(value, 3280, 2464) == pytest.approx((820.0, 1232.0, 328.0, 123.2))

    def test_round_trip(self) -> None:
        box = (1000.0, 800.0, 260.0, 200.0)

        value = box_to_percent(box, W, H)

        assert percent_to_box(value, W, H) == pytest.approx(box)
        assert value["x"] == pytest.approx(1000 / W * 100)


class TestProjectBox:
    def test_translation_moves_the_box(self) -> None:
        projection = project_box(BOX, shifted(30, -20), W, H)

        assert projection.box == pytest.approx((1030.0, 780.0, 260.0, 200.0))
        assert projection.visible == pytest.approx(1.0)

    def test_rotation_takes_the_axis_aligned_bounds_of_the_corners(self) -> None:
        rotate = Similarity(1.0, 90.0, 500.0, 0.0)

        projection = project_box((100.0, 50.0, 40.0, 10.0), rotate, W, H)

        assert projection.box == pytest.approx((440.0, 100.0, 10.0, 40.0))

    def test_partly_outside_is_clipped_and_reports_what_is_visible(self) -> None:
        projection = project_box((0.0, 0.0, 100.0, 100.0), shifted(-50, 0), W, H)

        assert projection.box == pytest.approx((0.0, 0.0, 50.0, 100.0))
        assert projection.visible == pytest.approx(0.5)

    def test_fully_outside_has_no_box(self) -> None:
        projection = project_box(BOX, shifted(-5000, 0), W, H)

        assert projection.box is None and projection.visible == 0.0

    def test_quad_keeps_the_unclipped_corners(self) -> None:
        projection = project_box(BOX, shifted(10, 10), W, H)

        np.testing.assert_allclose(projection.quad[0], [1010.0, 810.0])
        assert projection.quad.shape == (4, 2)


class TestIou:
    def test_identical_and_disjoint(self) -> None:
        assert box_iou(BOX, BOX) == pytest.approx(1.0)
        assert box_iou(BOX, (0.0, 0.0, 10.0, 10.0)) == 0.0

    def test_half_overlap(self) -> None:
        assert box_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


class TestAgreement:
    def test_small_displacement_agrees(self) -> None:
        result = agreement(BOX, shifted(0), shifted(5), W, H)

        assert result.agree
        assert result.iou > 0.95
        assert result.displacement == pytest.approx(5 / np.sqrt(260 * 200))

    def test_large_displacement_disagrees(self) -> None:
        result = agreement(BOX, shifted(0), shifted(60), W, H)

        assert not result.agree
        assert result.iou < 0.8

    def test_a_displacement_under_ten_percent_of_the_side_agrees_even_below_iou_08(self) -> None:
        result = agreement((0.0, 0.0, 40.0, 40.0), shifted(0), shifted(3.9), W, H)

        assert result.iou < 0.9
        assert result.displacement < 0.10 and result.agree


def estimate(method: str, dx: float | None, **metrics) -> Estimate:
    return Estimate(method, None if dx is None else shifted(dx), dx is not None, metrics)


class TestDecide:
    def test_tier_a_when_two_valid_methods_agree(self) -> None:
        decision = decide([estimate("sift", 40), estimate("dense", 44)], BOX, W, H)

        assert decision.tier == "A"
        assert decision.methods == ("sift", "dense")
        assert decision.box == pytest.approx((1042.0, 800.0, 260.0, 200.0))
        assert decision.agreement_iou > 0.9

    def test_tier_b_when_exactly_one_method_is_valid(self) -> None:
        decision = decide([estimate("sift", None), estimate("dense", 30)], BOX, W, H)

        assert decision.tier == "B"
        assert decision.methods == ("dense",)
        assert decision.box == pytest.approx((1030.0, 800.0, 260.0, 200.0))
        assert decision.agreement_iou is None

    def test_tier_c_when_none_is_valid(self) -> None:
        decision = decide([estimate("sift", None), estimate("dense", None)], BOX, W, H)

        assert decision.tier == "C" and decision.box is None
        assert decision.reason == "no_valid_method"

    def test_two_valid_methods_that_disagree_give_no_projection(self) -> None:
        decision = decide([estimate("sift", 0), estimate("dense", 150)], BOX, W, H)

        assert decision.tier == "C" and decision.box is None
        assert decision.reason == "methods_disagree"

    def test_an_outlier_among_three_valid_methods_is_ignored(self) -> None:
        decision = decide(
            [estimate("sift", 10), estimate("dense", 400), estimate("bridge_sift", 12)], BOX, W, H
        )

        assert decision.tier == "A"
        assert decision.methods == ("sift", "bridge_sift")

    def test_a_box_that_leaves_the_frame_is_not_projected(self) -> None:
        decision = decide([estimate("sift", -1200), estimate("dense", -1200)], BOX, W, H)

        assert decision.tier == "C" and decision.reason == "out_of_frame"

    def test_a_single_method_whose_box_leaves_the_frame_is_not_projected(self) -> None:
        decision = decide([estimate("sift", None), estimate("dense", -1200)], BOX, W, H)

        assert decision.tier == "C" and decision.reason == "out_of_frame"


class TestBridge:
    def test_translation_only_map_carries_a_shift_unchanged(self) -> None:
        offset = np.array([[1, 0, -600.0], [0, 1, 380.0], [0, 0, 1]])
        bridge = Bridge(offset)

        moved = bridge.carry(shifted(25, -10))

        np.testing.assert_allclose(
            moved.apply(np.array([[1000.0, 900.0]])), [[1025.0, 890.0]], atol=1e-6
        )

    def test_a_rotation_is_taken_about_the_mapped_point(self) -> None:
        offset = np.array([[1, 0, 200.0], [0, 1, 0.0], [0, 0, 1]])
        bridge = Bridge(offset)
        turn = Similarity(1.0, 90.0, 0.0, 0.0)  # in the RGB frame, about the RGB origin

        moved = bridge.carry(turn)

        # NIR point (200, 0) is RGB (0, 0), the fixed point of the rotation.
        np.testing.assert_allclose(moved.apply(np.array([[200.0, 0.0]])), [[200.0, 0.0]], atol=1e-6)

    def test_shift_of_the_map_changes_the_result_only_through_the_rotation(self) -> None:
        turn = Similarity(1.0, 5.0, 60.0, 20.0)
        base = Bridge(np.eye(3)).carry(turn)
        moved = Bridge(np.array([[1, 0, 80.0], [0, 1, 0.0], [0, 0, 1]])).carry(turn)
        point = np.array([[1500.0, 1000.0]])

        gap = np.linalg.norm(base.apply(point) - moved.apply(point))

        assert gap < 80 * np.radians(5.0) * 1.1

    def test_extra_shift_is_added_to_the_map(self) -> None:
        plain = Bridge(np.eye(3))
        shifted_bridge = Bridge(np.eye(3), shift=(80.0, -7.0))

        assert shifted_bridge.to_nir(np.array([[10.0, 10.0]]))[0] == pytest.approx([90.0, 3.0])
        assert plain.to_nir(np.array([[10.0, 10.0]]))[0] == pytest.approx([10.0, 10.0])


class TestLoadHandBoxes:
    def test_reads_nir_and_rgb_boxes_in_pixels(self, tmp_path: Path) -> None:
        export = write_export(
            tmp_path,
            [
                task(cara=1, ts=1),
                task(cara=2, ts=2, box=("damage_rgb", 50.0, 50.0, 10.0, 10.0), size=(3264, 2448)),
            ],
        )

        loaded = load_hand_boxes(export)

        nir, rgb = loaded.boxes
        assert (nir.modality, nir.width, nir.height) == ("nir", W, H)
        assert nir.box == pytest.approx((328.0, 492.8, 262.4, 147.84))
        assert (rgb.modality, rgb.width, rgb.height) == ("rgb", 3264, 2448)
        assert nir.face == Face(JORNADA_1, 1, 1) and nir.timestamp == 1 and nir.ronda_h == 12

    def test_a_task_without_a_box_is_reported_and_skipped(self, tmp_path: Path) -> None:
        export = write_export(tmp_path, [task(cara=1, ts=1), task(mango=8, cara=1, ts=2, box=None)])

        loaded = load_hand_boxes(export)

        assert [b.timestamp for b in loaded.boxes] == [1]
        assert loaded.without_box == [Face(JORNADA_1, 8, 1)]

    def test_cancelled_and_missing_annotations_are_not_boxes(self, tmp_path: Path) -> None:
        export = write_export(
            tmp_path,
            [task(ts=1, cancelled=True), task(cara=2, ts=2, annotated=False), task(cara=3, ts=3)],
        )

        loaded = load_hand_boxes(export)

        assert [b.timestamp for b in loaded.boxes] == [3]
        assert loaded.unannotated == [Face(JORNADA_1, 1, 1), Face(JORNADA_1, 1, 2)]


class TestLoadHandBoxesByHour:
    def export(self, tmp_path: Path) -> Path:
        return write_export(
            tmp_path,
            [
                task(cara=1, hour=12, ts=1),
                task(
                    cara=1, hour=10, ts=2, box=("damage_rgb", 1.0, 1.0, 2.0, 2.0), size=(3264, 2448)
                ),
                task(cara=2, hour=10, ts=3, box=None),
                task(cara=3, hour=10, ts=4, annotated=False),
            ],
        )

    def test_hours_filter_keeps_only_those_moments(self, tmp_path: Path) -> None:
        loaded = load_hand_boxes(self.export(tmp_path), hours={12})

        assert [b.timestamp for b in loaded.boxes] == [1]
        assert loaded.without_box == [] and loaded.unannotated == []
        assert loaded.seen == {Face(JORNADA_1, 1, 1)}

    def test_without_a_filter_every_moment_is_read(self, tmp_path: Path) -> None:
        loaded = load_hand_boxes(self.export(tmp_path))

        assert [b.timestamp for b in loaded.boxes] == [1, 2]
        assert loaded.without_box == [Face(JORNADA_1, 1, 2)]
        assert loaded.unannotated == [Face(JORNADA_1, 1, 3)]

    def test_the_latest_live_annotation_wins(self, tmp_path: Path) -> None:
        old = task(cara=1, ts=1, box=("damage_nir", 10.0, 10.0, 5.0, 5.0))
        newer = task(cara=1, ts=1, box=("damage_nir", 40.0, 40.0, 5.0, 5.0))
        newer["annotations"][0]["updated_at"] = "2026-10-26T15:00:00Z"
        cancelled = task(cara=1, ts=1, box=("damage_nir", 70.0, 70.0, 5.0, 5.0), cancelled=True)
        cancelled["annotations"][0]["updated_at"] = "2026-10-26T16:00:00Z"
        merged = dict(old)
        merged["annotations"] = old["annotations"] + newer["annotations"] + cancelled["annotations"]

        loaded = load_hand_boxes(write_export(tmp_path, [merged]))

        (box,) = loaded.boxes
        assert box.box[0] == pytest.approx(0.40 * W)


class TestPlanFaces:
    def metas(self) -> list[dict]:
        return [
            make_meta(1, ronda_h=12),
            make_meta(2, ronda_h=10),
            make_meta(3, ronda_h=8),
            make_meta(4, ronda_h=6, en_tolerancia=False),  # duplicate, off time
            make_meta(5, ronda_h=6),
            make_meta(7, ronda_h=None),  # healthy; +4 h is missing
        ]

    def hand(self, ts: int = 1) -> HandBox:
        return HandBox(Face(JORNADA_1, 1, 1), 12, ts, "nir", BOX, W, H)

    def test_earlier_moments_follow_the_duplicate_rule_and_skip_missing_slots(
        self, tmp_path: Path
    ) -> None:
        write_dataset(tmp_path, self.metas())

        planned = plan_faces(tmp_path, [self.hand()])

        (plan,) = planned.plans
        assert {m: c.timestamp for m, c in plan.captures.items()} == {
            12: 1,
            10: 2,
            8: 3,
            6: 5,
            None: 7,
        }
        assert [d.timestamp for d in planned.dropped] == [4]
        assert {(s.ronda_h) for s in planned.missing} == {4, 2, 0}

    def test_a_face_without_its_12h_photo_in_the_dataset_is_an_error(self, tmp_path: Path) -> None:
        write_dataset(tmp_path, [make_meta(2, ronda_h=10)])

        with pytest.raises(ValueError, match="12 h"):
            plan_faces(tmp_path, [self.hand()])


def record(
    tier: str,
    modality: str = "nir",
    box: tuple[float, float, float, float] = (1030.0, 800.0, 260.0, 200.0),
    moment: int | None = 8,
    face: Face = Face(JORNADA_1, 1, 1),
    methods: tuple[str, ...] | None = None,
) -> ProjectionRecord:
    width, height = (W, H) if modality == "nir" else (3264, 2448)
    hand = HandBox(face, 12, 100, modality, BOX, width, height)
    projected = tier != "C"
    decision = Decision(
        tier=tier,
        box=box if projected else None,
        methods=methods or (("sift", "dense") if tier == "A" else ("sift",) if tier == "B" else ()),
        agreement_iou=0.9 if tier == "A" else None,
        agreement_displacement=0.02 if tier == "A" else None,
        reason="" if projected else "no_valid_method",
    )
    return ProjectionRecord(
        face, moment, hand, decision, [estimate("sift", 30), estimate("dense", 30)]
    )


class TestPredictions:
    def tasks(self) -> list[dict]:
        return [
            {
                "data": {
                    "jornada": JORNADA_1,
                    "mango_id": 1,
                    "cara": 1,
                    "ronda_h": 8,
                    "timestamp": 11,
                }
            },
            {
                "data": {
                    "jornada": JORNADA_1,
                    "mango_id": 1,
                    "cara": 2,
                    "ronda_h": 8,
                    "timestamp": 12,
                }
            },
            {
                "data": {
                    "jornada": JORNADA_1,
                    "mango_id": 1,
                    "cara": 3,
                    "ronda_h": 8,
                    "timestamp": 13,
                }
            },
        ]

    def test_prediction_has_the_label_studio_result_shape(self) -> None:
        tasks = attach_predictions(self.tasks(), [record("A")], 8)

        (prediction,) = tasks[0]["predictions"]
        assert prediction["model_version"] == "proj-12h-A"
        assert prediction["score"] == pytest.approx(0.9)
        (result,) = prediction["result"]
        assert result["from_name"] == "damage_nir" and result["to_name"] == "nir"
        assert result["type"] == "rectanglelabels"
        assert (result["original_width"], result["original_height"]) == (W, H)
        value = result["value"]
        assert value["rectanglelabels"] == ["damage"] and value["rotation"] == 0
        assert value["x"] == pytest.approx(1030 / W * 100)
        assert value["width"] == pytest.approx(260 / W * 100)

    def test_rgb_boxes_go_on_the_rgb_control(self) -> None:
        tasks = attach_predictions(self.tasks(), [record("B", modality="rgb")], 8)

        (result,) = tasks[0]["predictions"][0]["result"]
        assert (result["from_name"], result["to_name"]) == ("damage_rgb", "rgb")
        assert (result["original_width"], result["original_height"]) == (3264, 2448)
        assert tasks[0]["predictions"][0]["model_version"] == "proj-12h-B"
        assert tasks[0]["predictions"][0]["score"] == pytest.approx(0.5)

    def test_tier_c_and_unmatched_tasks_get_no_prediction(self) -> None:
        tasks = attach_predictions(self.tasks(), [record("C", face=Face(JORNADA_1, 1, 2))], 8)

        assert all("predictions" not in t for t in tasks)

    def test_only_the_requested_moment_is_used(self) -> None:
        tasks = attach_predictions(self.tasks(), [record("A", moment=6)], 8)

        assert "predictions" not in tasks[0]

    def test_tasks_are_not_modified_in_place(self) -> None:
        original = self.tasks()

        attach_predictions(original, [record("A")], 8)

        assert "predictions" not in original[0]

    def test_filenames_keep_the_10h_set_out_of_the_import_names(self) -> None:
        assert prediction_filename(8) == "predictions_8h.json"
        assert prediction_filename(0) == "predictions_0h.json"
        assert prediction_filename(None) == "predictions_sano.json"
        assert prediction_filename(10) == "projection_10h_for_validation.json"


class TestValidation:
    def projections(self, tmp_path: Path) -> Path:
        def projected(
            cara: int, tier: str, modality: str, box: tuple[float, float, float, float]
        ) -> dict:
            return attach_predictions(
                [
                    {
                        "data": {
                            "jornada": JORNADA_1,
                            "mango_id": 1,
                            "cara": cara,
                            "ronda_h": 10,
                            "timestamp": cara,
                        }
                    }
                ],
                [record(tier, modality, box, 10, Face(JORNADA_1, 1, cara))],
                10,
            )[0]

        tasks = [
            projected(1, "A", "nir", (1000.0, 800.0, 260.0, 200.0)),  # hand box identical
            projected(2, "A", "nir", (1000.0, 800.0, 260.0, 200.0)),  # hand box shifted a lot
            projected(3, "B", "nir", (1000.0, 800.0, 260.0, 200.0)),  # hand box slightly off
            projected(
                4, "B", "rgb", (1000.0, 800.0, 260.0, 200.0)
            ),  # hand box drawn on the other modality
            projected(5, "A", "nir", (1000.0, 800.0, 260.0, 200.0)),  # no hand box at all
            {
                "data": {
                    "jornada": JORNADA_1,
                    "mango_id": 1,
                    "cara": 6,
                    "ronda_h": 10,
                    "timestamp": 6,
                }
            },
        ]
        path = tmp_path / "projection_10h_for_validation.json"
        path.write_text(json.dumps(tasks))
        return path

    def hand_export(self, tmp_path: Path) -> Path:
        def pct(box: tuple[float, float, float, float], width: int = W, height: int = H) -> tuple:
            v = box_to_percent(box, width, height)
            return v["x"], v["y"], v["width"], v["height"]

        tasks = [
            task(cara=1, hour=10, ts=1, box=("damage_nir", *pct((1000.0, 800.0, 260.0, 200.0)))),
            task(cara=2, hour=10, ts=2, box=("damage_nir", *pct((1200.0, 800.0, 260.0, 200.0)))),
            task(cara=3, hour=10, ts=3, box=("damage_nir", *pct((1030.0, 800.0, 260.0, 200.0)))),
            task(cara=4, hour=10, ts=4, box=("damage_nir", *pct((1000.0, 800.0, 260.0, 200.0)))),
            task(cara=6, hour=10, ts=6, box=("damage_nir", *pct((900.0, 800.0, 260.0, 200.0)))),
        ]
        return write_export(tmp_path, tasks, "hand_10h.json")

    def test_per_tier_iou_statistics(self, tmp_path: Path) -> None:
        report = validate_projections(self.hand_export(tmp_path), self.projections(tmp_path))

        a, b = report.tiers["A"], report.tiers["B"]
        assert a.n == 2  # face 5 has no hand box
        assert a.median_iou == pytest.approx(
            (1.0 + box_iou((1000, 800, 260, 200), (1200, 800, 260, 200))) / 2
        )
        assert a.share_at_least_05 == pytest.approx(0.5)
        assert a.share_at_least_07 == pytest.approx(0.5)
        assert b.n == 1
        assert b.median_iou == pytest.approx(box_iou((1000, 800, 260, 200), (1030, 800, 260, 200)))
        assert b.share_at_least_05 == pytest.approx(1.0) and b.share_at_least_07 == pytest.approx(
            1.0
        )

    def test_cross_modality_pairs_are_skipped_and_counted(self, tmp_path: Path) -> None:
        report = validate_projections(self.hand_export(tmp_path), self.projections(tmp_path))

        assert report.cross_modality == 1

    def test_missing_hand_boxes_and_unprojected_faces_are_counted(self, tmp_path: Path) -> None:
        report = validate_projections(self.hand_export(tmp_path), self.projections(tmp_path))

        assert report.no_hand_box == 1
        assert report.unprojected == 1  # face 6 is Tier C, with a hand box we cannot score

    def test_only_the_moment_of_the_projections_is_compared(self, tmp_path: Path) -> None:
        extra = task(cara=1, hour=12, ts=50, box=("damage_nir", 1.0, 1.0, 5.0, 5.0))
        hand = write_export(
            tmp_path, json.loads(self.hand_export(tmp_path).read_text()) + [extra], "h.json"
        )

        report = validate_projections(hand, self.projections(tmp_path))

        assert report.tiers["A"].n == 2


def _pct(box: tuple[float, float, float, float], width: int = W, height: int = H) -> tuple:
    v = box_to_percent(box, width, height)
    return v["x"], v["y"], v["width"], v["height"]


LOT_2 = "lote2-20261025"
P10 = (1000.0, 800.0, 260.0, 200.0)


class TestValidationBreakdown:
    """Faces 1-7 at +10 h: projections, the +12 h boxes they came from, and later hand boxes."""

    def projections(self, tmp_path: Path) -> Path:
        def task_for(face: Face, tier: str, modality: str) -> dict:
            base = {
                "data": {
                    "jornada": face.jornada,
                    "mango_id": face.mango_id,
                    "cara": face.cara,
                    "ronda_h": 10,
                    "timestamp": face.cara,
                }
            }
            if tier == "C":
                return base
            return attach_predictions([base], [record(tier, modality, P10, 10, face)], 10)[0]

        tasks = [
            task_for(Face(JORNADA_1, 1, 1), "A", "nir"),
            task_for(Face(JORNADA_1, 1, 2), "A", "nir"),
            task_for(Face(JORNADA_1, 1, 3), "B", "nir"),
            task_for(Face(JORNADA_1, 1, 4), "B", "rgb"),
            task_for(Face(JORNADA_1, 1, 5), "A", "nir"),
            task_for(Face(JORNADA_1, 1, 6), "C", "nir"),
            task_for(Face(LOT_2, 1, 1), "B", "nir"),
            task_for(Face(JORNADA_1, 1, 8), "A", "nir"),
        ]
        path = tmp_path / "projection_10h_for_validation.json"
        path.write_text(json.dumps(tasks))
        return path

    def export(self, tmp_path: Path) -> Path:
        rgb = ("damage_rgb", *_pct(BOX, 3264, 2448))
        nir = lambda b: ("damage_nir", *_pct(b))  # noqa: E731
        tasks = [
            # +12 h source boxes
            *[task(cara=c, hour=12, ts=100 + c, box=nir(BOX)) for c in (1, 2, 3, 5, 6, 8)],
            task(cara=4, hour=12, ts=104, box=rgb, size=(3264, 2448)),
            task(jornada=LOT_2, cara=1, hour=12, ts=201, box=nir(BOX)),
            # +10 h hand boxes
            task(cara=1, hour=10, ts=1, box=nir(P10)),
            task(cara=2, hour=10, ts=2, box=nir((1200.0, 800.0, 260.0, 200.0))),
            task(cara=3, hour=10, ts=3, box=nir((1030.0, 800.0, 260.0, 200.0))),
            task(cara=4, hour=10, ts=4, box=nir(P10)),  # +12 on RGB, +10 on NIR: cross-modality
            task(cara=6, hour=10, ts=6, box=nir((900.0, 800.0, 260.0, 200.0))),
            task(jornada=LOT_2, cara=1, hour=10, ts=7, box=nir((1050.0, 820.0, 130.0, 100.0))),
            task(cara=8, hour=10, ts=8, box=None),  # annotated, no box drawn
        ]
        return write_export(tmp_path, tasks, "hand_12h_10h.json")

    def report(self, tmp_path: Path):
        return validate_projections(self.export(tmp_path), self.projections(tmp_path))

    def test_pooled_projected_boxes_and_the_tier_rows(self, tmp_path: Path) -> None:
        report = self.report(tmp_path)

        assert report.tiers["A"].n == 2 and report.tiers["B"].n == 2
        assert report.projected.n == 4
        assert report.projected.median_iou == pytest.approx(
            float(
                np.median(
                    [
                        1.0,
                        box_iou(P10, (1200, 800, 260, 200)),
                        box_iou(P10, (1030, 800, 260, 200)),
                        0.25,
                    ]
                )
            )
        )

    def test_breakdown_by_lot(self, tmp_path: Path) -> None:
        report = self.report(tmp_path)

        assert report.by_lot[JORNADA_1]["A"].n == 2
        assert report.by_lot[JORNADA_1]["B"].n == 1
        assert report.by_lot[JORNADA_1]["A+B"].n == 3
        assert report.by_lot[LOT_2]["B"].n == 1 and report.by_lot[LOT_2][
            "B"
        ].median_iou == pytest.approx(0.25)

    def test_breakdown_by_modality(self, tmp_path: Path) -> None:
        report = self.report(tmp_path)

        assert set(report.by_modality) == {"nir"}  # the one RGB projection is cross-modality
        assert report.by_modality["nir"]["A+B"].n == 4

    def test_cross_modality_pairs_are_counted_over_all_faces_and_over_projected_ones(
        self, tmp_path: Path
    ) -> None:
        report = self.report(tmp_path)

        assert report.cross_modality == 1  # face 4: +12 on RGB, +10 on NIR
        assert report.cross_modality_projected == 1

    def test_copying_the_12h_box_is_the_baseline_on_the_same_faces(self, tmp_path: Path) -> None:
        report = self.report(tmp_path)

        copy = report.copy_baseline
        assert copy["A"].n == 2 and copy["A"].median_iou == pytest.approx(
            float(np.median([box_iou(BOX, P10), box_iou(BOX, (1200, 800, 260, 200))]))
        )
        assert copy["B"].n == 2
        assert copy["C"].n == 1 and copy["C"].median_iou == pytest.approx(
            box_iou(BOX, (900, 800, 260, 200))
        )
        assert copy["all"].n == 5  # faces 1, 2, 3, 6 and lot 2 face 1 (face 8 has no hand box)

    def test_hand_boxes_shrink_towards_earlier_moments(self, tmp_path: Path) -> None:
        report = self.report(tmp_path)

        assert report.area_ratio.n == 5
        assert report.area_ratio.median == pytest.approx(1.0)
        assert report.area_ratio.share_smaller == pytest.approx(1 / 5)

    def test_faces_without_a_hand_box_are_listed_by_why(self, tmp_path: Path) -> None:
        report = self.report(tmp_path)

        assert report.absent_faces == [Face(JORNADA_1, 1, 5)]
        assert report.no_box_faces == [Face(JORNADA_1, 1, 8)]
        assert report.unannotated_faces == []
        assert report.no_hand_box == 2

    def test_center_error_of_the_projections_in_pixels_and_box_sides(self, tmp_path: Path) -> None:
        report = self.report(tmp_path)

        a, both = report.center_error["A"], report.center_error["A+B"]
        assert a.n == 2 and a.median_px == pytest.approx(100.0)  # 0 px and 200 px
        assert both.median_px == pytest.approx((30.0 + np.hypot(15.0, 30.0)) / 2)
        assert both.median_fraction == pytest.approx(
            (30 / np.sqrt(260 * 200) + np.hypot(15, 30) / np.sqrt(130 * 100)) / 2
        )

    def test_centre_bias_by_lot_is_the_median_offset_from_projection_to_hand_box(
        self, tmp_path: Path
    ) -> None:
        report = self.report(tmp_path)

        dx, dy, n = report.center_bias[JORNADA_1]
        assert (n, dx, dy) == (3, pytest.approx(30.0), pytest.approx(0.0))  # offsets 0, 200, 30
        dx2, dy2, n2 = report.center_bias[LOT_2]
        assert (n2, dx2, dy2) == (1, pytest.approx(-15.0), pytest.approx(-30.0))

    def test_size_ceiling_is_the_iou_of_the_12h_box_centred_on_the_hand_box(
        self, tmp_path: Path
    ) -> None:
        report = self.report(tmp_path)

        ceiling = report.size_ceiling
        assert ceiling.n == 5
        assert ceiling.median_iou == pytest.approx(1.0)  # same-size boxes, one 0.25 case
        assert ceiling.share_at_least_07 == pytest.approx(4 / 5)

    def test_text_report_has_the_breakdowns(self, tmp_path: Path) -> None:
        text = format_validation(self.report(tmp_path))

        for needle in (
            "centre error",
            "size ceiling",
            "by lot",
            "by modality",
            "copy of the +12 h box",
            "area ratio",
            "cross-modality",
        ):
            assert needle in text


class TestValidationBySource:
    def test_report_csv_splits_tiers_by_where_the_registration_came_from(
        self, tmp_path: Path
    ) -> None:
        validation = TestValidation()
        projections = validation.projections(tmp_path)
        hand = validation.hand_export(tmp_path)
        bridge = record("B", methods=("bridge_sift",), moment=10, face=Face(JORNADA_1, 1, 3))
        native = record("A", methods=("sift", "dense"), moment=10, face=Face(JORNADA_1, 1, 1))
        mixed = record("A", methods=("sift", "bridge_dense"), moment=10, face=Face(JORNADA_1, 1, 2))
        write_report([native, mixed, bridge], tmp_path / "report.csv")

        report = validate_projections(hand, projections, report_csv=tmp_path / "report.csv")

        assert report.sources["A:native"].n == 1
        assert report.sources["A:mixed"].n == 1
        assert report.sources["B:bridge"].n == 1
        assert report.sources["A:native"].median_iou == pytest.approx(1.0)

    def test_without_a_report_there_is_no_split(self, tmp_path: Path) -> None:
        validation = TestValidation()

        report = validate_projections(
            validation.hand_export(tmp_path), validation.projections(tmp_path)
        )

        assert report.sources == {}


class TestReport:
    def test_rows_hold_tier_metrics_and_transform_parameters(self) -> None:
        decision = decide(
            [
                Estimate("sift", shifted(40), True, {"inliers": 31, "rmse": 2.5}),
                Estimate("dense", shifted(44), True, {"score": 0.71}),
            ],
            BOX,
            W,
            H,
        )
        rec = ProjectionRecord(
            Face(JORNADA_1, 3, 2),
            8,
            HandBox(Face(JORNADA_1, 3, 2), 12, 100, "nir", BOX, W, H),
            decision,
            [
                Estimate("sift", shifted(40), True, {"inliers": 31, "rmse": 2.5}),
                Estimate("dense", shifted(44), True, {"score": 0.71}),
            ],
        )

        (row,) = report_rows([rec])

        assert row["jornada"] == JORNADA_1 and row["mango_id"] == 3 and row["cara"] == 2
        assert row["ronda_h"] == 8 and row["modality"] == "nir" and row["tier"] == "A"
        assert row["sift_inliers"] == 31 and row["sift_rmse_px"] == pytest.approx(2.5)
        assert row["dense_score"] == pytest.approx(0.71)
        assert row["agreement_iou"] == pytest.approx(decision.agreement_iou)
        assert row["sift_tx"] == pytest.approx(40.0) and row["dense_tx"] == pytest.approx(44.0)
        assert row["methods"] == "sift+dense"

    def test_a_healthy_moment_is_written_as_sano(self) -> None:
        (row,) = report_rows([record("A", moment=None)])

        assert row["ronda_h"] == "sano"

    def test_write_report_makes_a_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "report.csv"

        write_report([record("A"), record("C", moment=6)], out)

        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert [r["tier"] for r in rows] == ["A", "C"]
        assert "agreement_iou" in rows[0]
