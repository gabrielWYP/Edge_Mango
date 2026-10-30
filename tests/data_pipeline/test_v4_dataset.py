"""Tests for the v4 dataset builder (captura_v3 + Label Studio boxes, NIR registered per lot)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.annotation.nir_display import enhance_nir
from src.data_pipeline.v4_dataset import build_dataset, raw_nir, registered_nir
from src.data_pipeline.v4_selection import RGB_SIZE
from tests.annotation.synthetic import JORNADA_1, make_meta, synthetic_nir, write_dataset
from tests.data_pipeline.v4_synth import H_SHIFT, RULES, make_task


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
