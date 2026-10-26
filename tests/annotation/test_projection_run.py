"""Tests for loading photos, running the projection over faces and writing the outputs."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.annotation.box_projection import (
    Bridge,
    Decision,
    Face,
    HandBox,
    ProjectionRecord,
    box_iou,
    box_to_percent,
    load_hand_boxes,
    project_box,
)
from src.annotation.captura_v3_tasks import load_captures
from src.annotation.projection_run import (
    PhotoLoader,
    RunConfig,
    calibrate_nulls,
    consistency_summary,
    estimate_lot_shifts,
    format_tier_table,
    pick_sheet_faces,
    render_contact_sheet,
    render_mask_sheet,
    run_projection,
    tier_table,
    validity_table,
    write_predictions,
)
from tests.annotation.export_synth import task, write_export
from tests.annotation.fruit_synth import color_scene, fruit_layer, scene, sheet
from tests.annotation.synthetic import JORNADA_1, make_meta, write_dataset
from tests.annotation.world_synth import (
    AXES,
    CENTRE,
    HAND,
    HEIGHT,
    MOTIONS,
    SHAPE,
    WIDTH,
    metas_for,
    write_jpeg,
    write_rgb_world,
)


@pytest.fixture(scope="module")
def rgb_world(tmp_path_factory):
    return write_rgb_world(tmp_path_factory.mktemp("rgb_world"))


class TestPhotoLoader:
    def test_rgb_prefers_the_white_balance_corrected_photo(self, tmp_path: Path) -> None:
        data_root = tmp_path / "captura_v3"
        write_dataset(
            data_root, [make_meta(5, ronda_h=12), make_meta(6, ronda_h=10)], wb_timestamps=(6,)
        )
        loader = PhotoLoader(data_root)
        captures = {c.timestamp: c for c in load_captures(data_root)}

        assert loader.rgb_file(captures[5]).parent.name == "rgb"
        assert loader.rgb_file(captures[6]).parent.name == "rgb_wb"

    def test_empty_tray_is_the_latest_one_of_the_jornada(self, tmp_path: Path) -> None:
        calib = tmp_path / "calib"
        calib.mkdir()
        for ts, jornada in ((100, JORNADA_1), (200, JORNADA_1), (300, "lote2-20261025")):
            (calib / f"calib-vacia-meta-{ts}.json").write_text(
                json.dumps({"timestamp": ts, "jornada": {"id": jornada}})
            )
            (calib / f"calib-vacia-nir-{ts}.jpg").write_bytes(b"x")

        loader = PhotoLoader(tmp_path)

        assert loader.empty_tray_file(JORNADA_1).name == "calib-vacia-nir-200.jpg"
        assert loader.empty_tray_file("lote2-20261025").name == "calib-vacia-nir-300.jpg"
        with pytest.raises(FileNotFoundError):
            loader.empty_tray_file("lote9-x")

    def test_rgb_photo_is_prepared_at_half_size_with_the_fruit_masked(self, rgb_world) -> None:
        data_root, export, _ = rgb_world
        capture = next(c for c in load_captures(data_root) if c.timestamp == 1)

        prepared = PhotoLoader(data_root).prepare(capture, "rgb")

        assert prepared.scale == 0.5 and prepared.gray.shape == (360, 480)
        assert prepared.mask[180, 240] == 255  # the fruit centre
        assert prepared.mask[20, 20] == 0
        assert len(prepared.points) > 50


@pytest.fixture(scope="module")
def rgb_run(rgb_world):
    data_root, export, _ = rgb_world
    return run_projection(RunConfig(data_root=data_root, export=export))


class TestRunOnRgb:
    def run(self, rgb_world, **kwargs):
        data_root, export, _ = rgb_world
        return run_projection(RunConfig(data_root=data_root, export=export, **kwargs))

    def test_projected_boxes_land_where_the_fruit_moved(self, rgb_run) -> None:
        run = rgb_run

        for record in run.records:
            if record.face.cara != 1:
                continue
            truth = project_box(HAND, MOTIONS[record.moment], WIDTH, HEIGHT).box
            assert record.decision.tier in {"A", "B"}
            assert box_iou(record.decision.box, truth) > 0.85

    def test_both_methods_agree_on_clean_images(self, rgb_run) -> None:
        run = rgb_run

        tiers = {(r.face.cara, r.moment): r.decision.tier for r in run.records}

        assert tiers[(1, 10)] == "A" and tiers[(1, 8)] == "A"

    def test_only_existing_earlier_moments_are_projected(self, rgb_run) -> None:
        run = rgb_run

        assert sorted((r.face.cara, r.moment) for r in run.records) == [
            (1, 8),
            (1, 10),
            (2, 8),
            (2, 10),
        ]
        assert {s.ronda_h for s in run.planned.missing} == {6, 4, 2, 0, None}

    def test_cycle_check_composes_the_registrations_through_10h(self, rgb_run) -> None:
        run = rgb_run

        rows = [r for r in run.cycles if r["face"].cara == 1 and r["moment"] == 8]

        assert {r["method"] for r in rows} == {"sift", "dense"}
        assert all(r["displacement_px"] < 12 for r in rows)

    def test_cycles_can_be_switched_off(self, rgb_world) -> None:
        run = self.run(rgb_world, with_cycles=False)

        assert run.cycles == []

    def test_limit_faces_keeps_the_first_faces(self, rgb_world) -> None:
        run = self.run(rgb_world, limit_faces=1)

        assert {r.face.cara for r in run.records} == {1}

    def test_runs_in_worker_processes_too(self, rgb_world, rgb_run) -> None:
        serial = rgb_run
        parallel = self.run(rgb_world, workers=2)

        assert [(r.face, r.moment, r.decision.tier) for r in serial.records] == [
            (r.face, r.moment, r.decision.tier) for r in parallel.records
        ]

    def test_tier_table_counts_per_lot_and_moment(self, rgb_run) -> None:
        run = rgb_run

        table = tier_table(run.records)

        assert table[(JORNADA_1, 10)]["A"] == 2
        assert sum(table[(JORNADA_1, 8)].values()) == 2
        text = format_tier_table(table)
        assert JORNADA_1 in text and "+10 h" in text

    def test_validity_table_gives_the_share_of_valid_methods(self, rgb_run) -> None:
        table = validity_table(rgb_run.records)

        assert table[("rgb", 10)] == {"n": 2.0, "sift": 1.0, "dense": 1.0}

    def test_consistency_summary_has_cycle_and_method_agreement_numbers(self, rgb_run) -> None:
        summary = consistency_summary(rgb_run.records, rgb_run.cycles)

        assert summary["cycle"]["sift"]["n"] == 2
        assert summary["cycle"]["sift"]["median_px"] < 12
        assert summary["cycle"]["dense"]["share_under_10pct_side"] == 1.0
        assert summary["sift_vs_dense"]["native"]["n"] == 4
        assert summary["sift_vs_dense"]["native"]["share_agree"] == 1.0

    def test_predictions_files_are_written_with_the_validation_set_apart(
        self, rgb_world, rgb_run, tmp_path: Path
    ) -> None:
        data_root, export, doc_root = rgb_world
        run = rgb_run
        out = tmp_path / "ls"

        paths = write_predictions(run.records, data_root, doc_root, out, nir_display="original")

        assert sorted(p.name for p in paths) == [
            "predictions_8h.json",
            "projection_10h_for_validation.json",
        ]
        tasks = json.loads((out / "predictions_8h.json").read_text())
        assert {t["data"]["ronda_h"] for t in tasks} == {8}
        assert all("predictions" in t for t in tasks)
        assert tasks[0]["predictions"][0]["model_version"].startswith("proj-12h-")
        assert not any(out.glob("predictions_10h.json"))

    def test_contact_sheet_is_a_small_jpeg(self, rgb_world, rgb_run, tmp_path: Path) -> None:
        data_root, _, _ = rgb_world
        run = rgb_run
        out = tmp_path / "sheet.jpg"

        render_contact_sheet(run.records, PhotoLoader(data_root), out, faces=2, thumb_width=160)

        image = cv2.imread(str(out))
        assert image is not None and out.stat().st_size < 400_000
        assert image.shape[1] >= 160 * 4

    def test_mask_sheet_outlines_the_fruit_mask_on_each_photo(
        self, rgb_world, tmp_path: Path
    ) -> None:
        data_root, _, _ = rgb_world
        loader = PhotoLoader(data_root)
        captures = [
            loader.capture(Face(JORNADA_1, 1, 1), 12),
            loader.capture(Face(JORNADA_1, 1, 2), 10),
        ]
        out = tmp_path / "masks.jpg"

        render_mask_sheet(loader, [(c, "rgb") for c in captures], out, thumb_width=240, columns=2)

        sheet_image = cv2.imread(str(out))
        assert sheet_image.shape[1] == 480 and out.stat().st_size < 400_000
        red = (sheet_image[..., 2] > 200) & (sheet_image[..., 1] < 80) & (sheet_image[..., 0] < 80)
        assert red.sum() > 200  # the mask outline is drawn

    def test_null_pairs_of_unrelated_faces_do_not_register(self, rgb_world, rgb_run) -> None:
        data_root, export, _ = rgb_world
        run = rgb_run

        nulls = calibrate_nulls(
            PhotoLoader(data_root), run.planned.plans, modality="rgb", pairs=3, seed=0
        )

        assert len(nulls["sift_inliers"]) == 3 and max(nulls["sift_inliers"]) < 15
        assert max(nulls["dense_score"]) < 0.45


class TestRunOnNir:
    @pytest.fixture(scope="class")
    def nir_world(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("nir_world")
        data_root = tmp_path / "data" / "cache" / "captura_v3"
        write_dataset(data_root, metas_for(1, 1))
        background = sheet(seed=2, shape=SHAPE)
        texture, mask = fruit_layer(1, SHAPE, CENTRE, AXES)
        for ts, hour in ((1, 12), (2, 10), (3, 8)):
            gray, _ = scene(background, texture, mask, MOTIONS.get(hour))
            write_jpeg(
                data_root / "news" / "nir" / f"mango_nir_{ts}.jpg", cv2.merge([gray, gray, gray])
            )
        calib = data_root / "calib"
        calib.mkdir()
        (calib / "calib-vacia-meta-9.json").write_text(
            json.dumps({"timestamp": 9, "jornada": {"id": JORNADA_1}})
        )
        write_jpeg(calib / "calib-vacia-nir-9.jpg", cv2.merge([background] * 3))
        export = write_export(
            tmp_path,
            [
                task(
                    ts=1,
                    box=("damage_nir", *box_to_percent(HAND, WIDTH, HEIGHT).values()),
                    size=(WIDTH, HEIGHT),
                )
            ],
        )
        return data_root, export

    def test_nir_photos_are_registered_on_nir(self, nir_world) -> None:
        data_root, export = nir_world

        run = run_projection(RunConfig(data_root=data_root, export=export))

        assert {r.hand.modality for r in run.records} == {"nir"}
        for record in run.records:
            truth = project_box(HAND, MOTIONS[record.moment], WIDTH, HEIGHT).box
            assert record.decision.tier in {"A", "B"}
            assert box_iou(record.decision.box, truth) > 0.8

    def test_nir_mask_leaves_the_static_sheet_out(self, nir_world) -> None:
        data_root, _ = nir_world
        captures = load_captures(data_root)

        prepared = PhotoLoader(data_root).prepare(
            next(c for c in captures if c.timestamp == 1), "nir"
        )

        assert prepared.mask[180, 240] == 255
        assert prepared.mask[:30, :30].sum() == 0 and prepared.mask[-30:, -30:].sum() == 0

    def test_bridge_adds_the_rgb_registration_as_two_more_estimates(
        self, nir_world, tmp_path: Path
    ) -> None:
        data_root, export = nir_world
        background = sheet(seed=2, shape=SHAPE)
        texture, mask = fruit_layer(1, SHAPE, CENTRE, AXES)
        for ts, hour in ((1, 12), (2, 10), (3, 8)):
            image, _ = color_scene(background, texture, mask, MOTIONS.get(hour))
            write_jpeg(data_root / "news" / "rgb" / f"mango_rgb_{ts}.jpg", image)
        homography = tmp_path / "h.npy"
        np.save(homography, np.array([[1, 0, 25.0], [0, 1, -10.0], [0, 0, 1]]))

        run = run_projection(RunConfig(data_root=data_root, export=export, homography=homography))

        for record in run.records:
            methods = {e.method for e in record.estimates}
            assert methods == {"sift", "dense", "bridge_sift", "bridge_dense"}
            assert record.decision.tier == "A"
            truth = project_box(HAND, MOTIONS[record.moment], WIDTH, HEIGHT).box
            assert box_iou(record.decision.box, truth) > 0.8


class TestSheetFaces:
    def records(self) -> list:
        def rec(jornada: str, mango: int, modality: str, tier: str):
            face = Face(jornada, mango, 1)
            hand = HandBox(face, 12, 1, modality, HAND, 10, 10)
            decision = Decision(tier, None if tier == "C" else HAND, (), None, None, "")
            return ProjectionRecord(face, 8, hand, decision, [])

        return [
            rec("lote1", 1, "rgb", "A"),
            rec("lote1", 2, "rgb", "C"),
            rec("lote2", 3, "rgb", "A"),
            rec("lote1", 4, "nir", "A"),
            rec("lote1", 5, "nir", "B"),
            rec("lote1", 5, "nir", "C"),
            rec("lote2", 6, "nir", "A"),
            rec("lote2", 7, "nir", "A"),
        ]

    def test_nir_comes_first_and_lots_alternate(self) -> None:
        picked = pick_sheet_faces(self.records(), 4)

        assert [(f.jornada, m) for f, m in picked] == [
            ("lote1", "nir"),
            ("lote2", "nir"),
            ("lote1", "rgb"),
            ("lote2", "rgb"),
        ]

    def test_within_a_group_the_face_with_most_varied_tiers_comes_first(self) -> None:
        picked = pick_sheet_faces(self.records(), 5)

        assert picked[0][0] == Face("lote1", 5, 1)  # tiers B and C beat a lone A

    def test_asking_for_more_than_exist_returns_all(self) -> None:
        assert len(pick_sheet_faces(self.records(), 50)) == 7  # one face has two records


class TestLotShifts:
    def test_second_lot_shift_is_measured_against_the_reference_lot(self) -> None:
        rng = np.random.default_rng(0)
        homography = np.array([[1.0, 0.02, -600.0], [-0.03, 0.95, 380.0], [0, 0, 1]])
        bridge = Bridge(homography)

        def pairs(extra: tuple[float, float]) -> list[tuple[np.ndarray, np.ndarray]]:
            rgb = rng.uniform([600, 500], [2400, 1700], size=(120, 2))
            nir = (
                bridge.to_nir(rgb)
                + np.array([-220.0, 70.0])
                + np.array(extra)
                + rng.normal(0, 15, (120, 2))
            )
            return list(zip(rgb, nir))

        shifts = estimate_lot_shifts(
            {"lote1": pairs((0, 0)), "lote2": pairs((-80.0, -7.0))}, homography, reference="lote1"
        )

        assert shifts["lote1"] == (0.0, 0.0)
        assert shifts["lote2"] == pytest.approx((-80.0, -7.0), abs=5.0)

    def test_too_few_pairs_leave_the_shift_at_zero(self) -> None:
        shifts = estimate_lot_shifts(
            {"lote1": [(np.zeros(2), np.zeros(2))] * 200, "lote2": [(np.zeros(2), np.ones(2))] * 5},
            np.eye(3),
            reference="lote1",
        )

        assert shifts["lote2"] == (0.0, 0.0)


def test_hand_boxes_fixture_is_readable(rgb_world) -> None:
    _, export, _ = rgb_world

    boxes = load_hand_boxes(export).boxes

    assert [b.face for b in boxes] == [Face(JORNADA_1, 1, 1), Face(JORNADA_1, 1, 2)]
    assert boxes[0].box == pytest.approx(HAND)
