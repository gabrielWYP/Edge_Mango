"""Tests for the contrast-enhanced grayscale NIR display images."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.annotation.nir_display import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    GAUSSIAN_SIGMA,
    JPEG_QUALITY,
    build_nir_display,
    enhance_file,
    enhance_nir,
    nir_gray,
    nir_vis_relpath,
)
from tests.annotation.synthetic import make_meta, synthetic_nir, write_dataset


def reference(bgr: np.ndarray, sigma=1.0, clip=3.0, tile=16) -> np.ndarray:
    gray = (bgr[..., 0].astype(np.float32) + bgr[..., 2].astype(np.float32)) / 2
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    as_u8 = np.clip(blurred, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(as_u8)


def low_contrast_nir(height: int = 256, width: int = 256) -> np.ndarray:
    """A shadowed, nearly flat image: CLAHE's clip limit visibly changes its output."""
    rng = np.random.default_rng(7)
    return np.clip(120 + rng.normal(0, 2, size=(height, width, 3)), 0, 255).astype(np.uint8)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data" / "cache" / "captura_v3"


@pytest.fixture
def dataset(data_root: Path) -> Path:
    metas = [make_meta(100 + i, cara=i) for i in (1, 2, 3)]
    return write_dataset(data_root, metas, nir_shape=(61, 97))


class TestParameters:
    def test_method_constants_are_the_chosen_ones(self) -> None:
        assert GAUSSIAN_SIGMA == 1.0
        assert CLAHE_CLIP_LIMIT == 3.0
        assert CLAHE_TILE_GRID == (16, 16)
        assert JPEG_QUALITY == 95


class TestGray:
    def test_gray_is_the_mean_of_blue_and_red(self) -> None:
        bgr = np.array([[[10, 200, 30], [255, 0, 254]]], dtype=np.uint8)

        gray = nir_gray(bgr)

        assert gray.dtype == np.float32
        assert gray.tolist() == [[20.0, 254.5]]

    def test_green_channel_is_ignored(self) -> None:
        bgr = synthetic_nir(1, 40, 50)
        other = bgr.copy()
        other[..., 1] = 255 - other[..., 1]

        assert np.array_equal(nir_gray(bgr), nir_gray(other))
        assert np.array_equal(enhance_nir(bgr), enhance_nir(other))


class TestEnhance:
    def test_output_is_single_channel_uint8_with_the_input_size(self) -> None:
        bgr = synthetic_nir(1, 61, 97)

        out = enhance_nir(bgr)

        assert out.dtype == np.uint8
        assert out.shape == (61, 97)

    def test_is_deterministic(self) -> None:
        bgr = synthetic_nir(2, 64, 96)

        assert np.array_equal(enhance_nir(bgr), enhance_nir(bgr.copy()))

    def test_does_not_modify_its_input(self) -> None:
        bgr = synthetic_nir(3, 64, 96)
        before = bgr.copy()

        enhance_nir(bgr)

        assert np.array_equal(bgr, before)

    @pytest.mark.parametrize(
        "bgr", [synthetic_nir(4, 96, 160), low_contrast_nir()], ids=["textured", "flat"]
    )
    def test_matches_the_cv2_reference_pipeline(self, bgr: np.ndarray) -> None:
        assert np.array_equal(enhance_nir(bgr), reference(bgr))

    @pytest.mark.parametrize(
        "variant",
        [{"sigma": 2.0}, {"clip": 2.0}, {"tile": 8}],
        ids=["sigma", "clip-limit", "tile-grid"],
    )
    def test_reference_is_sensitive_to_each_parameter(self, variant: dict) -> None:
        bgr = low_contrast_nir()

        assert not np.array_equal(enhance_nir(bgr), reference(bgr, **variant))


class TestEnhanceFile:
    def test_writes_a_single_channel_jpeg_with_the_original_size(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "src.jpg"
        dst = tmp_path / "dst.jpg"
        cv2.imwrite(str(src), synthetic_nir(5, 61, 97))

        enhance_file(src, dst)

        with Image.open(dst) as written, Image.open(src) as original:
            assert written.mode == "L"
            assert written.format == "JPEG"
            assert written.size == original.size == (97, 61)

    def test_bytes_are_jpeg_quality_95_of_the_enhanced_image(self, tmp_path: Path) -> None:
        src = tmp_path / "src.jpg"
        dst = tmp_path / "dst.jpg"
        cv2.imwrite(str(src), synthetic_nir(6, 64, 96))

        enhance_file(src, dst)

        expected = cv2.imencode(
            ".jpg", enhance_nir(cv2.imread(str(src))), [cv2.IMWRITE_JPEG_QUALITY, 95]
        )[1].tobytes()
        assert dst.read_bytes() == expected

    def test_unreadable_source_names_the_file(self, tmp_path: Path) -> None:
        src = tmp_path / "broken.jpg"
        src.write_bytes(b"not a jpeg")

        with pytest.raises(ValueError, match="broken.jpg"):
            enhance_file(src, tmp_path / "dst.jpg")


class TestBuildNirDisplay:
    def test_writes_one_image_per_timestamp_under_nir_vis(self, dataset: Path) -> None:
        report = build_nir_display(dataset, [101, 102])

        assert report.written == [101, 102]
        assert report.skipped == []
        for ts in (101, 102):
            out = dataset / nir_vis_relpath(ts)
            assert out.parent == dataset / "news" / "nir_vis"
            assert out.name == f"mango_nir_{ts}.jpg"
            with Image.open(out) as image:
                assert (image.mode, image.size) == ("L", (97, 61))
        assert not (dataset / nir_vis_relpath(103)).exists()

    def test_existing_files_are_skipped_unless_overwrite(self, dataset: Path) -> None:
        out = dataset / nir_vis_relpath(101)
        out.parent.mkdir(parents=True)
        out.write_bytes(b"sentinel")

        skipped = build_nir_display(dataset, [101])
        assert (skipped.written, skipped.skipped) == ([], [101])
        assert out.read_bytes() == b"sentinel"

        replaced = build_nir_display(dataset, [101], overwrite=True)
        assert (replaced.written, replaced.skipped) == ([101], [])
        assert out.read_bytes() != b"sentinel"

    def test_rerun_is_byte_identical(self, dataset: Path) -> None:
        build_nir_display(dataset, [101])
        first = (dataset / nir_vis_relpath(101)).read_bytes()

        build_nir_display(dataset, [101], overwrite=True)

        assert (dataset / nir_vis_relpath(101)).read_bytes() == first

    def test_no_partial_files_are_left_behind(self, dataset: Path) -> None:
        build_nir_display(dataset, [101, 102])

        names = sorted(p.name for p in (dataset / "news" / "nir_vis").iterdir())
        assert names == sorted(["manifest.json", "mango_nir_101.jpg", "mango_nir_102.jpg"])

    def test_missing_source_fails(self, dataset: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_nir_display(dataset, [999])

    def test_manifest_records_method_params_and_source_hashes(self, dataset: Path) -> None:
        build_nir_display(dataset, [102, 101])

        manifest = json.loads((dataset / "news" / "nir_vis" / "manifest.json").read_text())
        assert "(B + R) / 2" in manifest["method"]
        assert manifest["params"] == {
            "gaussian_sigma": 1.0,
            "clahe_clip_limit": 3.0,
            "clahe_tile_grid": [16, 16],
            "jpeg_quality": 95,
        }
        assert manifest["files"] == [
            {"timestamp": ts, "source_sha256": sha256(dataset / "news" / "nir" / f"mango_nir_{ts}.jpg")}
            for ts in (101, 102)
        ]

    def test_manifest_accumulates_across_runs(self, dataset: Path) -> None:
        build_nir_display(dataset, [101])
        build_nir_display(dataset, [102, 101])

        manifest = json.loads((dataset / "news" / "nir_vis" / "manifest.json").read_text())
        assert [f["timestamp"] for f in manifest["files"]] == [101, 102]

    def test_existing_outputs_made_with_other_params_are_not_reused(
        self, dataset: Path
    ) -> None:
        build_nir_display(dataset, [101])
        manifest_path = dataset / "news" / "nir_vis" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["params"]["clahe_clip_limit"] = 2.0
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="--overwrite"):
            build_nir_display(dataset, [102])
        build_nir_display(dataset, [102], overwrite=True)


class TestProtectedDirectories:
    @pytest.mark.parametrize(
        "relative",
        ["rgb", "nir", "meta", "rgb_wb", "nir/sub", "nir_vis/../nir"],
    )
    def test_output_inside_capture_folders_is_refused(
        self, dataset: Path, relative: str
    ) -> None:
        news = dataset / "news"
        (news / "rgb_wb").mkdir(exist_ok=True)
        before = {p: p.read_bytes() for p in news.rglob("*") if p.is_file()}

        with pytest.raises(ValueError, match="protected"):
            build_nir_display(dataset, [101], out_dir=news / relative)

        after = {p: p.read_bytes() for p in news.rglob("*") if p.is_file()}
        assert after == before

    def test_symlink_into_a_capture_folder_is_refused(self, dataset: Path) -> None:
        news = dataset / "news"
        (news / "nir_vis").symlink_to(news / "nir")
        before = (news / "nir" / "mango_nir_101.jpg").read_bytes()

        with pytest.raises(ValueError, match="protected"):
            build_nir_display(dataset, [101])

        assert (news / "nir" / "mango_nir_101.jpg").read_bytes() == before
