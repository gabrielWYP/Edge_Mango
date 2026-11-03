"""Tests for the v4 dataset builder (captura_v3 + Label Studio boxes, NIR registered per lot)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import build_v4_dataset as build_cli
from src.annotation.nir_display import enhance_nir
from src.data_pipeline.v4_dataset import INDEX_FIELDS, build_dataset, raw_nir, registered_nir
from src.data_pipeline.v4_selection import RGB_SIZE
from tests.annotation.synthetic import JORNADA_1, make_meta, synthetic_nir, write_dataset
from tests.data_pipeline.v4_synth import (
    H_SHIFT,
    RULES,
    fruit_photos,
    make_task,
    nir_sensor_photo,
)


def _write_mango_rgb(path: Path) -> None:
    """Cyan-white sheet with a yellow ellipse whose bounding box is [1100, 800, 2100, 1600]."""
    rgb = np.full((RGB_SIZE[1], RGB_SIZE[0], 3), (230, 220, 150), dtype=np.uint8)
    cv2.ellipse(rgb, (1600, 1200), (500, 400), 0, 0, 360, (30, 200, 220), -1)
    cv2.imwrite(str(path), rgb)


class TestBuildDataset:
    @pytest.fixture
    def setup(self, tmp_path: Path) -> dict:
        data_root = tmp_path / "cache" / "captura_v3"
        metas = [
            make_meta(1, ronda_h=None),
            make_meta(2, ronda_h=12),
            make_meta(3, ronda_h=6),
        ]
        write_dataset(data_root, metas, nir_shape=(2464, 3280))
        for meta in metas:
            _write_mango_rgb(data_root / "news" / "rgb" / f"mango_rgb_{meta['timestamp']}.jpg")
        np.save(tmp_path / "h1.npy", H_SHIFT)
        (tmp_path / "rules.yaml").write_text(json.dumps({"lots": {JORNADA_1: RULES[JORNADA_1]}}))
        (tmp_path / "identity.yaml").write_text("exclusions: []\n")
        # NIR box at [1508, 1206, 1708, 1306] -> RGB [1500, 1200, 1700, 1300].
        task = make_task(metas[1], {"x": 1508, "y": 1206, "w": 200, "h": 100})
        (tmp_path / "labels.json").write_text(json.dumps([task]))
        return {
            "data_root": data_root,
            "labels_path": tmp_path / "labels.json",
            "rules_path": tmp_path / "rules.yaml",
            "identity_path": tmp_path / "identity.yaml",
            "out_dir": tmp_path / "out",
            "repo_root": tmp_path,
        }

    def test_writes_registered_pairs_labels_and_reports(self, setup: dict) -> None:
        report = build_dataset(**setup)
        out = setup["out_dir"]
        assert report["num_included"] == 2 and report["excluded"] == {"lot1": {"unlabeled": 1}}
        assert sorted(p.name for p in (out / "nir").iterdir()) == [
            "mango_nir_1.png",
            "mango_nir_2.png",
        ]
        assert (out / "rgb" / "mango_rgb_2.jpg").resolve() == (
            setup["data_root"] / "news" / "rgb" / "mango_rgb_2.jpg"
        ).resolve()
        nir = cv2.imread(str(out / "nir" / "mango_nir_2.png"), cv2.IMREAD_UNCHANGED)
        assert nir.shape == (RGB_SIZE[1], RGB_SIZE[0]) and nir.dtype == np.uint8
        assert np.array_equal(np.load(out / "homographies" / "identity.npy"), np.eye(3))

        sano = (out / "labels" / "mango_rgb_1.txt").read_text().splitlines()
        lines = (out / "labels" / "mango_rgb_2.txt").read_text().splitlines()
        assert len(sano) == 1 and sano[0].startswith("0 ")
        cx, cy, w, h = map(float, lines[0].split()[1:])
        assert (cx * RGB_SIZE[0], cy * RGB_SIZE[1]) == pytest.approx((1600, 1200), abs=20)
        assert (w * RGB_SIZE[0], h * RGB_SIZE[1]) == pytest.approx((1000, 800), abs=40)
        damage = np.array(lines[1].split()[1:], dtype=float) * np.tile(RGB_SIZE, 2)
        assert lines[1].startswith("1 ") and damage == pytest.approx(
            [1600, 1250, 200, 100], abs=0.01
        )

        with (out / "index.csv").open() as handle:
            rows = {row["stem"]: row for row in csv.DictReader(handle)}
        assert rows["mango_rgb_2"]["provenance"] == "dibujada"
        assert rows["mango_rgb_2"]["damage_in_mango"] == "1.0"
        assert (out / "exclusions.csv").read_text().count("unlabeled") == 1

    def test_writes_the_clahe_view_warped_like_the_raw_nir(self, setup: dict) -> None:
        build_dataset(**setup)
        out = setup["out_dir"]
        bgr = cv2.imread(str(setup["data_root"] / "news" / "nir" / "mango_nir_2.jpg"))
        H = np.load(setup["repo_root"] / "h1.npy")
        clahe = cv2.imread(str(out / "nir_clahe" / "mango_nir_2.png"), cv2.IMREAD_UNCHANGED)
        raw = cv2.imread(str(out / "nir" / "mango_nir_2.png"), cv2.IMREAD_UNCHANGED)
        assert clahe.shape == (RGB_SIZE[1], RGB_SIZE[0]) and clahe.dtype == np.uint8
        assert np.array_equal(clahe, registered_nir(enhance_nir(bgr), H))
        assert np.array_equal(raw, registered_nir(raw_nir(bgr), H))
        assert sorted(p.name for p in (out / "nir_clahe").iterdir()) == [
            "mango_nir_1.png",
            "mango_nir_2.png",
        ]

    def test_interpolates_only_the_nir_lines_that_cross_the_mango(self, setup: dict) -> None:
        nir_path = setup["data_root"] / "news" / "nir" / "mango_nir_2.jpg"
        nir = synthetic_nir(2, 2464, 3280)
        nir[1000, 1400:1700] = 5  # NIR mango box ~[1108, 806, 2108, 1606] (H_SHIFT)
        nir[300, 200:500] = 5  # background
        cv2.imwrite(str(nir_path), nir)
        report = build_dataset(**setup)
        out = setup["out_dir"]

        with (out / "nir_lines.csv").open() as handle:
            lines = list(csv.DictReader(handle))
        assert [(line["stem"], line["row"], line["crosses_mango"]) for line in lines] == [
            ("mango_rgb_2", "300", "0"),
            ("mango_rgb_2", "1000", "1"),
        ]
        assert report["nir_lines"] == {"lot1": {"detected": 2, "interpolated": 1}}
        with (out / "index.csv").open() as handle:
            row = {r["stem"]: r for r in csv.DictReader(handle)}["mango_rgb_2"]
        assert (row["nir_lines"], row["nir_lines_interpolated"]) == ("2", "1")
        # In the RGB frame the lines sit 6 rows and 8 columns earlier.
        for sub in ("nir", "nir_clahe"):
            baked = cv2.imread(str(out / sub / "mango_nir_2.png"), cv2.IMREAD_UNCHANGED)
            assert np.median(baked[994, 1420:1660]) > 60
            assert np.median(baked[294, 220:460]) < 30

    def test_refuses_a_non_empty_output(self, setup: dict) -> None:
        setup["out_dir"].mkdir()
        (setup["out_dir"] / "stale.txt").write_text("x")
        with pytest.raises(FileExistsError):
            build_dataset(**setup)


# NIR box as drawn -> RGB frame with the homography alone: [1500, 1200, 1700, 1300].
NIR_BOX = {"x": 1508, "y": 1206, "w": 200, "h": 100}
RGB_BOX = {"x": 1500, "y": 1200, "w": 200, "h": 100}
SHIFT_SANO = (84, 20)
SHIFT_12H = (90, 24)
SHIFT_TOLERANCE_PX = 2.0
RESIDUAL_FIELDS = ("residual_dx", "residual_dy", "residual_corr", "residual_source")


def _write_fruit_dataset(root: Path, line_on_nir_fruit: bool = False) -> dict:
    """Four photos of one lot: sano; +12 h with a NIR box, an RGB box, and an unrelated NIR."""
    data_root = root / "cache" / "captura_v3"
    metas = [
        make_meta(1, ronda_h=None),
        make_meta(2, ronda_h=12),
        make_meta(3, ronda_h=12, mango_id=2),
        make_meta(4, ronda_h=12, mango_id=3),
    ]
    write_dataset(data_root, metas, nir_shape=(2464, 3280))
    for meta, shift in zip(metas, (SHIFT_SANO, SHIFT_12H, SHIFT_12H, SHIFT_12H)):
        ts = meta["timestamp"]
        rgb, nir = fruit_photos(shift, seed=ts)
        cv2.imwrite(str(data_root / "news" / "rgb" / f"mango_rgb_{ts}.jpg"), rgb)
        sensor = nir_sensor_photo(nir)
        if ts == 2 and line_on_nir_fruit:
            # H alone puts it at x 1020-1080, left of the mango box; the shift takes it inside.
            sensor[1206, 1028:1088] = 5
        if ts == 4:
            sensor = synthetic_nir(ts, 2464, 3280)
        cv2.imwrite(str(data_root / "news" / "nir" / f"mango_nir_{ts}.jpg"), sensor)
    np.save(root / "h1.npy", H_SHIFT)
    (root / "rules.yaml").write_text(json.dumps({"lots": {JORNADA_1: RULES[JORNADA_1]}}))
    (root / "identity.yaml").write_text("exclusions: []\n")
    tasks = [
        make_task(metas[1], NIR_BOX),
        make_task(metas[2], RGB_BOX, view="rgb"),
        make_task(metas[3], NIR_BOX),
    ]
    (root / "labels.json").write_text(json.dumps(tasks))
    return {
        "data_root": data_root,
        "labels_path": root / "labels.json",
        "rules_path": root / "rules.yaml",
        "identity_path": root / "identity.yaml",
        "out_dir": root / "out",
        "repo_root": root,
    }


def _index(out: Path) -> dict[str, dict]:
    with (out / "index.csv").open() as handle:
        return {row["stem"]: row for row in csv.DictReader(handle)}


def _damage_box(out: Path, ts: int) -> np.ndarray:
    """[x0, y0, x1, y1] in RGB pixels of the single damage line of a photo."""
    lines = (out / "labels" / f"mango_rgb_{ts}.txt").read_text().splitlines()
    damage = [line for line in lines if line.startswith("1 ")]
    assert len(damage) == 1
    cx, cy, w, h = np.array(damage[0].split()[1:], dtype=float) * np.tile(RGB_SIZE, 2)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _tree_bytes(out: Path) -> dict[str, bytes]:
    files = (p for p in sorted(out.rglob("*")) if p.is_file() and not p.is_symlink())
    return {
        str(p.relative_to(out)): p.read_bytes() for p in files if p.name != "report.json"
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict]:
    setup = _write_fruit_dataset(tmp_path_factory.mktemp("fruit"))
    return setup, build_dataset(**setup)


class TestResidualShift:
    def test_is_on_by_default_and_recorded_in_the_report(self, built: tuple[dict, dict]) -> None:
        _, report = built
        assert report["params"]["residual_shift"]["mode"] == "fruit"

    def test_index_records_the_measured_shift_and_its_source(
        self, built: tuple[dict, dict]
    ) -> None:
        setup, _ = built
        rows = _index(setup["out_dir"])
        assert set(RESIDUAL_FIELDS) <= set(INDEX_FIELDS)
        row = rows["mango_rgb_2"]
        assert (float(row["residual_dx"]), float(row["residual_dy"])) == pytest.approx(
            SHIFT_12H, abs=SHIFT_TOLERANCE_PX
        )
        assert float(row["residual_corr"]) > 0.5
        assert row["residual_source"] == "fruit"

    def test_report_gives_median_shift_and_counts_per_lot_and_moment(
        self, built: tuple[dict, dict]
    ) -> None:
        _, report = built
        late = report["residual"]["lot1"]["12"]
        assert (late["median_dx"], late["median_dy"]) == pytest.approx(
            SHIFT_12H, abs=SHIFT_TOLERANCE_PX
        )
        assert (late["accepted"], late["fallback"]) == (2, 1)
        sano = report["residual"]["lot1"]["sano"]
        assert (sano["median_dx"], sano["median_dy"]) == pytest.approx(
            SHIFT_SANO, abs=SHIFT_TOLERANCE_PX
        )
        assert (sano["accepted"], sano["fallback"]) == (1, 0)

    def test_unreliable_match_falls_back_to_the_median_of_its_lot_and_moment(
        self, built: tuple[dict, dict]
    ) -> None:
        setup, report = built
        row = _index(setup["out_dir"])["mango_rgb_4"]
        late = report["residual"]["lot1"]["12"]
        assert row["residual_source"] == "median"
        assert float(row["residual_dx"]) == late["median_dx"]
        assert float(row["residual_dy"]) == late["median_dy"]
        assert float(row["residual_corr"]) < 0.3

    def test_box_drawn_on_the_nir_moves_with_the_shift(self, built: tuple[dict, dict]) -> None:
        setup, _ = built
        out = setup["out_dir"]
        for ts in (2, 4):
            row = _index(out)[f"mango_rgb_{ts}"]
            dx, dy = float(row["residual_dx"]), float(row["residual_dy"])
            expected = [1500 + dx, 1200 + dy, 1700 + dx, 1300 + dy]
            assert _damage_box(out, ts) == pytest.approx(expected, abs=0.05)
            assert float(row["damage_x0"]) == pytest.approx(expected[0], abs=0.05)

    def test_box_drawn_on_the_rgb_does_not_move(self, built: tuple[dict, dict]) -> None:
        setup, _ = built
        out = setup["out_dir"]
        row = _index(out)["mango_rgb_3"]
        assert abs(float(row["residual_dx"])) > 50  # the photo itself was shifted
        assert _damage_box(out, 3) == pytest.approx([1500, 1200, 1700, 1300], abs=0.05)
        assert float(row["damage_x0"]) == 1500.0

    def test_mango_box_stays_in_the_rgb_frame(self, built: tuple[dict, dict]) -> None:
        setup, _ = built
        for ts in (1, 2, 3, 4):
            row = _index(setup["out_dir"])[f"mango_rgb_{ts}"]
            box = [float(row[k]) for k in ("mango_x0", "mango_y0", "mango_x1", "mango_y1")]
            assert box == pytest.approx([1100, 800, 2100, 1600], abs=40)

    def test_both_nir_variants_share_the_shift(self, built: tuple[dict, dict]) -> None:
        setup, _ = built
        out = setup["out_dir"]
        H = np.load(setup["repo_root"] / "h1.npy")
        for ts in (1, 2, 4):
            row = _index(out)[f"mango_rgb_{ts}"]
            shift = float(row["residual_dx"]), float(row["residual_dy"])
            bgr = cv2.imread(str(setup["data_root"] / "news" / "nir" / f"mango_nir_{ts}.jpg"))
            raw = cv2.imread(str(out / "nir" / f"mango_nir_{ts}.png"), cv2.IMREAD_UNCHANGED)
            clahe = cv2.imread(str(out / "nir_clahe" / f"mango_nir_{ts}.png"), cv2.IMREAD_UNCHANGED)
            assert np.array_equal(raw, registered_nir(raw_nir(bgr), H, shift))
            assert np.array_equal(clahe, registered_nir(enhance_nir(bgr), H, shift))

    def test_the_shift_closes_the_nir_fruit_on_the_rgb_one(self, built: tuple[dict, dict]) -> None:
        setup, _ = built
        raw = cv2.imread(
            str(setup["out_dir"] / "nir" / "mango_nir_2.png"), cv2.IMREAD_UNCHANGED
        )
        # Fruit gray 100, sheet 190: the row through the centre changes at the RGB outline.
        row = raw[1200].astype(float)
        assert row[1090] > 150 and row[1120] < 150
        assert row[2110] > 150 and row[2080] < 150

    def test_none_reproduces_the_unshifted_output(self, tmp_path: Path) -> None:
        setup = _write_fruit_dataset(tmp_path)
        report = build_dataset(**setup, residual_shift="none")
        out = setup["out_dir"]
        H = np.load(tmp_path / "h1.npy")
        assert report["params"]["residual_shift"] == {"mode": "none"}
        assert report["residual"] == {}
        for ts in (1, 2, 3, 4):
            bgr = cv2.imread(str(setup["data_root"] / "news" / "nir" / f"mango_nir_{ts}.jpg"))
            raw = cv2.imread(str(out / "nir" / f"mango_nir_{ts}.png"), cv2.IMREAD_UNCHANGED)
            clahe = cv2.imread(str(out / "nir_clahe" / f"mango_nir_{ts}.png"), cv2.IMREAD_UNCHANGED)
            assert np.array_equal(raw, registered_nir(raw_nir(bgr), H))
            assert np.array_equal(clahe, registered_nir(enhance_nir(bgr), H))
            row = _index(out)[f"mango_rgb_{ts}"]
            assert [row[f] for f in RESIDUAL_FIELDS] == ["0.0", "0.0", "", "none"]
        for ts in (2, 3, 4):
            assert _damage_box(out, ts) == pytest.approx([1500, 1200, 1700, 1300], abs=0.05)

    def test_a_line_on_the_nir_fruit_is_interpolated_where_the_shift_puts_it(
        self, tmp_path: Path
    ) -> None:
        setup = _write_fruit_dataset(tmp_path / "fruit", line_on_nir_fruit=True)
        build_dataset(**setup)
        with (setup["out_dir"] / "nir_lines.csv").open() as handle:
            assert [row["crosses_mango"] for row in csv.DictReader(handle)] == ["1"]

    def test_none_keeps_the_line_rule_of_the_homography_alone(self, tmp_path: Path) -> None:
        setup = _write_fruit_dataset(tmp_path / "fruit", line_on_nir_fruit=True)
        build_dataset(**setup, residual_shift="none")
        with (setup["out_dir"] / "nir_lines.csv").open() as handle:
            assert [row["crosses_mango"] for row in csv.DictReader(handle)] == ["0"]

    def test_workers_do_not_change_the_output(self, tmp_path: Path) -> None:
        setup = _write_fruit_dataset(tmp_path)
        report_serial = build_dataset(**setup)
        serial = _tree_bytes(setup["out_dir"])
        report_parallel = build_dataset(**{**setup, "out_dir": tmp_path / "other"}, workers=2)
        assert _tree_bytes(tmp_path / "other") == serial
        assert report_parallel["residual"] == report_serial["residual"]

    def test_refuses_an_unknown_mode(self, tmp_path: Path) -> None:
        setup = _write_fruit_dataset(tmp_path)
        with pytest.raises(ValueError, match="residual_shift"):
            build_dataset(**setup, residual_shift="outline")


class TestBuildCli:
    def test_residual_shift_defaults_to_fruit(self) -> None:
        assert build_cli.parse_args([]).residual_shift == "fruit"

    def test_residual_shift_accepts_none_and_rejects_the_rest(self) -> None:
        assert build_cli.parse_args(["--residual-shift", "none"]).residual_shift == "none"
        with pytest.raises(SystemExit):
            build_cli.parse_args(["--residual-shift", "outline"])

    def test_existing_options_keep_their_defaults(self) -> None:
        args = build_cli.parse_args(["--workers", "3"])
        assert args.workers == 3
        assert args.data_root is None and args.output is None
        assert args.labels == build_cli.DEFAULT_LABELS
        assert args.homographies == build_cli.DEFAULT_RULES
        assert args.identity_exclusions == build_cli.DEFAULT_IDENTITY
        assert args.tolerance_exceptions == build_cli.DEFAULT_TOLERANCE

    def test_main_hands_the_mode_to_the_builder_and_prints_the_residual(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        seen = {}

        def fake_build(**kwargs) -> dict:
            seen.update(kwargs)
            return {
                "num_included": 3,
                "num_excluded": 1,
                "included": {},
                "excluded": {},
                "damage_mostly_outside_mango": [],
                "residual": {
                    "lot1": {
                        "12": {
                            "median_dx": 90.0,
                            "median_dy": 24.0,
                            "accepted": 2,
                            "fallback": 1,
                            "unmeasured": 0,
                        }
                    }
                },
            }

        monkeypatch.setattr(build_cli, "build_dataset", fake_build)
        code = build_cli.main(["--output", str(tmp_path / "out"), "--residual-shift", "none"])
        printed = capsys.readouterr().out
        assert code == 0
        assert seen["residual_shift"] == "none"
        assert "included 3, excluded 1" in printed
        assert "lot1 +12 h residual" in printed
