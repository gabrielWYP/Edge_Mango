"""Mango ROI crop: pure crop function, config validation and dataset wiring.

The crop cuts both modalities to the ground-truth mango box (class 0) plus a
margin BEFORE letterboxing, so the damage boxes (class 1) keep far more
pixels at the model's 640 input. Labels are re-expressed relative to the crop.
"""

from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.training.augmentations import get_val_transforms
from src.training.config import TrainingConfig
from src.training.dataset import YOLODataset
from src.training.roi_crop import crop_to_roi, validate_roi_crop

REPO_ROOT = Path(__file__).resolve().parents[2]

# Small non-square frame so x/y normalization mistakes cannot cancel out.
W, H = 160, 80
# Mango box in pixels (x1, y1, x2, y2): 80x40, centered.
ROI_PX = (40, 20, 120, 60)


def _yolo(box_px: tuple[float, float, float, float], w: int = W, h: int = H) -> list[float]:
    """Pixel xyxy -> normalized [cx, cy, w, h] relative to a (w, h) frame."""
    x1, y1, x2, y2 = box_px
    return [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]


def _boxes(*boxes_px: tuple[float, float, float, float], w: int = W, h: int = H) -> np.ndarray:
    return np.array([_yolo(b, w, h) for b in boxes_px], dtype=np.float32).reshape(-1, 4)


def _frame() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=np.uint8)


def _labels(*ids: int) -> np.ndarray:
    return np.array(ids, dtype=np.int64)


class TestCropGeometry:
    # Margin 0.10 of the 80x40 mango box is 8 px (x) and 4 px (y), so the crop
    # window is x 32..128, y 16..64 -> a 96x48 image.
    def test_known_roi_and_damage_box_map_to_expected_crop_coordinates(self) -> None:
        rgb, nir = _frame()
        damage_px = (60, 30, 70, 40)
        bboxes = _boxes(ROI_PX, damage_px)

        rgb_c, nir_c, bboxes_c, labels_c = crop_to_roi(
            rgb, nir, bboxes, _labels(0, 1), margin=0.10
        )

        assert rgb_c.shape == (48, 96, 3)
        assert nir_c.shape == (48, 96)
        expected = np.array(
            [
                _yolo((8, 4, 88, 44), 96, 48),  # mango, 8/4 px inside the window
                _yolo((28, 14, 38, 24), 96, 48),  # damage
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(bboxes_c, expected, atol=1e-6)
        assert labels_c.tolist() == [0, 1]

    def test_output_dtypes_and_shapes(self) -> None:
        rgb, nir = _frame()
        _, _, bboxes_c, labels_c = crop_to_roi(
            rgb, nir, _boxes(ROI_PX, (60, 30, 70, 40)), _labels(0, 1)
        )
        assert bboxes_c.dtype == np.float32 and bboxes_c.shape == (2, 4)
        assert labels_c.dtype == np.int64 and labels_c.shape == (2,)

    def test_input_order_is_preserved(self) -> None:
        rgb, nir = _frame()
        bboxes = _boxes((60, 30, 70, 40), ROI_PX)  # damage first, mango second
        _, _, bboxes_c, labels_c = crop_to_roi(rgb, nir, bboxes, _labels(1, 0))
        assert labels_c.tolist() == [1, 0]
        np.testing.assert_allclose(bboxes_c[0], _yolo((28, 14, 38, 24), 96, 48), atol=1e-6)
        np.testing.assert_allclose(bboxes_c[1], _yolo((8, 4, 88, 44), 96, 48), atol=1e-6)

    def test_rgb_and_nir_receive_the_identical_crop(self) -> None:
        rgb, nir = _frame()
        rgb[40, 70] = 200
        nir[40, 70] = 250

        rgb_c, nir_c, _, _ = crop_to_roi(rgb, nir, _boxes(ROI_PX), _labels(0), margin=0.10)

        # Window origin is (x=32, y=16): the marker moves to row 24, col 38 in both.
        assert np.argwhere(rgb_c[..., 0] == 200).tolist() == [[24, 38]]
        assert np.argwhere(nir_c == 250).tolist() == [[24, 38]]

    def test_zero_margin_crops_exactly_to_the_roi(self) -> None:
        rgb, nir = _frame()
        rgb_c, nir_c, bboxes_c, _ = crop_to_roi(
            rgb, nir, _boxes(ROI_PX), _labels(0), margin=0.0
        )
        assert rgb_c.shape == (40, 80, 3)
        assert nir_c.shape == (40, 80)
        np.testing.assert_allclose(bboxes_c, [[0.5, 0.5, 1.0, 1.0]], atol=1e-6)

    def test_pixel_window_floors_the_min_corner_and_ceils_the_max_corner(self) -> None:
        rgb, nir = _frame()
        roi = (40.5, 20.5, 119.5, 59.5)
        rgb_c, _, _, _ = crop_to_roi(rgb, nir, _boxes(roi), _labels(0), margin=0.0)
        assert rgb_c.shape == (40, 80, 3)  # x 40..120, y 20..60

    def test_largest_class0_box_is_the_roi(self) -> None:
        rgb, nir = _frame()
        small_mango = (130, 10, 150, 30)
        for order in ((ROI_PX, small_mango), (small_mango, ROI_PX)):
            rgb_c, _, _, labels_c = crop_to_roi(
                rgb, nir, _boxes(*order), _labels(0, 0), margin=0.10
            )
            # The big box defines the 96x48 window; the small one lies outside it.
            assert rgb_c.shape == (48, 96, 3)
            assert labels_c.tolist() == [0]

    def test_roi_class_is_configurable(self) -> None:
        rgb, nir = _frame()
        rgb_c, _, _, labels_c = crop_to_roi(
            rgb, nir, _boxes(ROI_PX), _labels(1), roi_class=1, margin=0.0
        )
        assert rgb_c.shape == (40, 80, 3)
        assert labels_c.tolist() == [1]


class TestBorderAndVisibility:
    def test_roi_is_clipped_at_the_image_border(self) -> None:
        rgb, nir = _frame()
        roi = (0, 0, 80, 40)  # margin would reach x -8 and y -4
        rgb_c, _, bboxes_c, _ = crop_to_roi(rgb, nir, _boxes(roi), _labels(0), margin=0.10)
        # Window x 0..88, y 0..44.
        assert rgb_c.shape == (44, 88, 3)
        np.testing.assert_allclose(bboxes_c, [_yolo((0, 0, 80, 40), 88, 44)], atol=1e-6)

    def test_roi_covering_the_whole_frame_is_a_noop_crop(self) -> None:
        rgb, nir = _frame()
        bboxes = _boxes((0, 0, W, H), (60, 30, 70, 40))
        rgb_c, nir_c, bboxes_c, labels_c = crop_to_roi(rgb, nir, bboxes, _labels(0, 1))
        assert rgb_c.shape == rgb.shape and nir_c.shape == nir.shape
        np.testing.assert_allclose(bboxes_c, bboxes, atol=1e-6)
        assert labels_c.tolist() == [0, 1]

    def test_box_below_min_visible_is_dropped_together_with_its_label(self) -> None:
        rgb, nir = _frame()
        # Window x 32..128, y 16..64.
        mostly_in = (26, 30, 42, 46)  # 10/16 of its width is visible -> 0.625
        mostly_out = (20, 30, 36, 46)  # 4/16 visible -> 0.25
        outside = (0, 0, 10, 10)
        bboxes = _boxes(ROI_PX, mostly_in, mostly_out, outside)

        _, _, bboxes_c, labels_c = crop_to_roi(
            rgb, nir, bboxes, _labels(0, 1, 1, 1), margin=0.10, min_visible=0.5
        )

        assert labels_c.tolist() == [0, 1]
        # The kept box is clipped to the window: x 32..42 -> 0..10, y 30..46 -> 14..30.
        np.testing.assert_allclose(bboxes_c[1], _yolo((0, 14, 10, 30), 96, 48), atol=1e-6)

    def test_lower_min_visible_keeps_the_partially_visible_box(self) -> None:
        rgb, nir = _frame()
        mostly_out = (20, 30, 36, 46)  # 0.25 visible
        bboxes = _boxes(ROI_PX, mostly_out)
        _, _, _, labels_c = crop_to_roi(
            rgb, nir, bboxes, _labels(0, 1), margin=0.10, min_visible=0.2
        )
        assert labels_c.tolist() == [0, 1]

    def test_fully_outside_box_is_dropped_even_with_min_visible_zero(self) -> None:
        rgb, nir = _frame()
        bboxes = _boxes(ROI_PX, (0, 0, 10, 10))
        _, _, bboxes_c, labels_c = crop_to_roi(
            rgb, nir, bboxes, _labels(0, 1), margin=0.10, min_visible=0.0
        )
        assert labels_c.tolist() == [0]
        assert bboxes_c.shape == (1, 4)  # no zero-area box survives

    def test_boxes_stay_inside_the_unit_square(self) -> None:
        rgb, nir = _frame()
        bboxes = _boxes(ROI_PX, (26, 30, 42, 46), (118, 55, 140, 70))
        _, _, bboxes_c, _ = crop_to_roi(
            rgb, nir, bboxes, _labels(0, 1, 1), margin=0.10, min_visible=0.1
        )
        x1 = bboxes_c[:, 0] - bboxes_c[:, 2] / 2
        x2 = bboxes_c[:, 0] + bboxes_c[:, 2] / 2
        y1 = bboxes_c[:, 1] - bboxes_c[:, 3] / 2
        y2 = bboxes_c[:, 1] + bboxes_c[:, 3] / 2
        assert (x1 >= -1e-7).all() and (x2 <= 1 + 1e-7).all()
        assert (y1 >= -1e-7).all() and (y2 <= 1 + 1e-7).all()


class TestJitter:
    def _crop(self, seed: int, jitter: float = 0.5):
        rgb, nir = _frame()
        return crop_to_roi(
            rgb,
            nir,
            _boxes(ROI_PX, (60, 30, 70, 40)),
            _labels(0, 1),
            margin=0.10,
            margin_jitter=jitter,
            rng=np.random.default_rng(seed),
        )

    @pytest.mark.parametrize("jitter", [0.5, 1.0])
    def test_jittered_crop_always_contains_the_full_roi(self, jitter: float) -> None:
        for seed in range(200):
            rgb_c, _, bboxes_c, labels_c = self._crop(seed, jitter)
            crop_h, crop_w = rgb_c.shape[:2]
            roi = bboxes_c[labels_c.tolist().index(0)]
            x1, x2 = (roi[0] - roi[2] / 2) * crop_w, (roi[0] + roi[2] / 2) * crop_w
            y1, y2 = (roi[1] - roi[3] / 2) * crop_h, (roi[1] + roi[3] / 2) * crop_h
            # Not clipped: the ROI keeps its full 80x40 px and lies inside the window.
            assert x2 - x1 == pytest.approx(80, abs=2e-3), seed
            assert y2 - y1 == pytest.approx(40, abs=2e-3), seed
            assert x1 >= -2e-3 and x2 <= crop_w + 2e-3, seed
            assert y1 >= -2e-3 and y2 <= crop_h + 2e-3, seed

    def test_jitter_varies_the_crop(self) -> None:
        shapes = {self._crop(seed)[0].shape for seed in range(30)}
        assert len(shapes) > 1

    def test_same_seed_gives_the_same_crop(self) -> None:
        first, second = self._crop(3), self._crop(3)
        assert first[0].shape == second[0].shape
        np.testing.assert_array_equal(first[2], second[2])

    def test_different_rngs_give_different_crops(self) -> None:
        first, second = self._crop(0), self._crop(1)
        assert not (
            first[0].shape == second[0].shape and np.array_equal(first[2], second[2])
        )

    def test_jitter_without_an_rng_is_rejected(self) -> None:
        rgb, nir = _frame()
        with pytest.raises(ValueError, match="rng"):
            crop_to_roi(
                rgb, nir, _boxes(ROI_PX), _labels(0), margin_jitter=0.5, rng=None
            )

    def test_zero_jitter_does_not_consume_the_rng(self) -> None:
        rgb, nir = _frame()
        rng = np.random.default_rng(0)
        before = rng.bit_generator.state
        crop_to_roi(rgb, nir, _boxes(ROI_PX), _labels(0), margin_jitter=0.0, rng=rng)
        assert rng.bit_generator.state == before

    def test_global_numpy_rng_is_never_touched(self) -> None:
        np.random.seed(123)
        before = np.random.get_state()[1].copy()
        self._crop(0)
        np.testing.assert_array_equal(np.random.get_state()[1], before)


class TestPassthroughAndErrors:
    def test_no_roi_box_returns_the_inputs_unchanged(self) -> None:
        rgb, nir = _frame()
        bboxes = _boxes((60, 30, 70, 40), (10, 10, 20, 20))
        labels = _labels(1, 1)

        rgb_c, nir_c, bboxes_c, labels_c = crop_to_roi(rgb, nir, bboxes, labels)

        assert rgb_c is rgb and nir_c is nir
        np.testing.assert_array_equal(bboxes_c, bboxes)
        np.testing.assert_array_equal(labels_c, labels)

    @pytest.mark.parametrize(
        ("box_dtype", "label_dtype"),
        [(np.float32, np.int64), (np.float64, np.int32)],
    )
    def test_empty_boxes_keep_canonical_shapes_and_dtypes(self, box_dtype, label_dtype) -> None:
        rgb, nir = _frame()
        rgb_c, nir_c, bboxes_c, labels_c = crop_to_roi(
            rgb,
            nir,
            np.zeros((0, 4), dtype=box_dtype),
            np.zeros((0,), dtype=label_dtype),
        )
        assert rgb_c is rgb and nir_c is nir
        assert bboxes_c.shape == (0, 4) and bboxes_c.dtype == np.float32
        assert labels_c.shape == (0,) and labels_c.dtype == np.int64

    def test_nir_rgb_shape_mismatch_raises(self) -> None:
        rgb, _ = _frame()
        nir = np.zeros((H - 1, W), dtype=np.uint8)
        with pytest.raises(ValueError, match="registered"):
            crop_to_roi(rgb, nir, _boxes(ROI_PX), _labels(0))

    def test_shape_mismatch_raises_even_without_a_roi(self) -> None:
        rgb, _ = _frame()
        nir = np.zeros((H, W + 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="registered"):
            crop_to_roi(rgb, nir, np.zeros((0, 4), np.float32), np.zeros((0,), np.int64))


class TestValidation:
    @pytest.mark.parametrize(
        ("margin", "jitter"),
        [(-0.01, 0.0), (0.1, -0.1), (0.1, 1.01), (float("nan"), 0.0), (0.1, float("nan"))],
    )
    def test_invalid_values_are_rejected(self, margin: float, jitter: float) -> None:
        with pytest.raises(ValueError):
            validate_roi_crop(margin, jitter)

    @pytest.mark.parametrize(("margin", "jitter"), [(0.0, 0.0), (0.1, 0.5), (0.5, 1.0)])
    def test_valid_values_pass(self, margin: float, jitter: float) -> None:
        validate_roi_crop(margin, jitter)


class TestConfig:
    def test_defaults_keep_the_crop_off(self) -> None:
        config = TrainingConfig()
        assert config.roi_crop is False
        assert config.roi_crop_margin == pytest.approx(0.10)
        assert config.roi_crop_jitter == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"roi_crop_margin": -0.1},
            {"roi_crop_jitter": -0.1},
            {"roi_crop_jitter": 1.5},
        ],
    )
    def test_invalid_values_are_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            TrainingConfig(**kwargs)

    def test_yaml_round_trip_keeps_the_keys(self, tmp_path: Path) -> None:
        config = TrainingConfig(roi_crop=True, roi_crop_margin=0.2, roi_crop_jitter=0.3)
        config.to_yaml(tmp_path / "c.yaml")
        loaded = TrainingConfig.from_yaml(tmp_path / "c.yaml")
        assert (loaded.roi_crop, loaded.roi_crop_margin, loaded.roi_crop_jitter) == (
            True,
            0.2,
            0.3,
        )

    def test_existing_experiment_configs_stay_full_frame(self) -> None:
        paths = sorted((REPO_ROOT / "configs" / "experiment").glob("*.yaml"))
        assert paths
        # The `_stable` variant is the crop experiment plus GroupNorm + EMA
        # (pinned in tests/training/test_neck_norm.py), so it crops too.
        crop_experiments = {"twostream_roi_crop.yaml", "twostream_roi_crop_stable.yaml"}
        for path in paths:
            if path.name in crop_experiments:
                continue
            assert TrainingConfig.from_yaml(path).roi_crop is False, path.name

    def test_roi_crop_experiment_only_differs_from_twostream_by_the_crop_keys(self) -> None:
        import yaml

        base_path = REPO_ROOT / "configs" / "experiment" / "twostream.yaml"
        crop_path = REPO_ROOT / "configs" / "experiment" / "twostream_roi_crop.yaml"
        base = yaml.safe_load(base_path.read_text())
        crop = yaml.safe_load(crop_path.read_text())

        changed = {k for k in base.keys() | crop.keys() if base.get(k) != crop.get(k)}
        assert changed == {
            "output_dir",
            "roi_crop",
            "roi_crop_margin",
            "roi_crop_jitter",
            "selection_metric",
        }
        config = TrainingConfig.from_yaml(crop_path)
        assert config.output_dir == "checkpoints/twostream_roi_crop"
        assert config.roi_crop is True
        assert config.roi_crop_margin == pytest.approx(0.10)
        assert config.roi_crop_jitter == pytest.approx(0.5)
        assert config.selection_metric == "damage_ap50"


class TestEntryPointsForwardTheCropKeys:
    """Every YOLODataset built from a TrainingConfig must forward the crop keys."""

    @pytest.mark.parametrize(
        "relative",
        [
            "src/training/train.py",
            "src/training/kd_train.py",
            "scripts/evaluate_checkpoint.py",
        ],
    )
    def test_calls_pass_roi_crop_keywords(self, relative: str) -> None:
        tree = ast.parse((REPO_ROOT / relative).read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "YOLODataset"
        ]
        assert calls, f"{relative} no longer builds a YOLODataset"
        for call in calls:
            passed = {kw.arg for kw in call.keywords}
            assert {"roi_crop", "roi_crop_margin", "roi_crop_jitter"} <= passed, (
                f"{relative}:{call.lineno} drops the ROI crop keys"
            )


# --------------------------------------------------------------------------- #
# Dataset integration
# --------------------------------------------------------------------------- #

FULL_W, FULL_H = 1280, 720
# Mango 320x288 px centered; two damage boxes fully inside it.
MANGO = "0 0.5 0.5 0.25 0.4"
DAMAGE_A = "1 0.5 0.5 0.05 0.08"
DAMAGE_B = "1 0.45 0.6 0.03 0.04"
MARKER_PX = (640, 360)  # mango center


def _write_dataset(root: Path, label_lines: list[str], *, split: str = "val") -> None:
    for sub in ("rgb", "nir", f"labels/{split}"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rgb = np.full((FULL_H, FULL_W, 3), 60, dtype=np.uint8)
    nir = np.full((FULL_H, FULL_W), 20, dtype=np.uint8)
    x, y = MARKER_PX
    rgb[y - 6 : y + 6, x - 6 : x + 6] = 255
    nir[y - 6 : y + 6, x - 6 : x + 6] = 255
    cv2.imwrite(str(root / "rgb" / "mango_rgb_00001.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(root / "nir" / "mango_nir_00001.jpg"), nir)
    (root / "labels" / split / "mango_rgb_00001.txt").write_text("\n".join(label_lines) + "\n")


def _dataset(root: Path, split: str = "val", **kwargs) -> YOLODataset:
    return YOLODataset(
        rgb_dir=root / "rgb",
        nir_dir=root / "nir",
        labels_dir=root / "labels",
        split=split,
        image_size=640,
        nir_homography_path=None,
        **kwargs,
    )


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    assert xs.size > 0, "marker vanished"
    return float(xs.mean()), float(ys.mean())


class TestYOLODatasetRoiCrop:
    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        _write_dataset(tmp_path, [MANGO, DAMAGE_A, DAMAGE_B])
        return tmp_path

    def test_default_and_explicit_off_are_identical(self, root: Path) -> None:
        default = _dataset(root)[0]
        off = _dataset(root, roi_crop=False)[0]
        for key in ("rgb", "nir", "bboxes", "labels"):
            assert torch.equal(default[key], off[key]), key

    def test_crop_off_matches_the_plain_letterbox_formula(self, root: Path) -> None:
        sample = _dataset(root, roi_crop=False)[0]
        # 1280x720 -> scale 0.5, 360 px tall, pad_y 140.
        expected = torch.tensor(
            [
                [0.5, (0.5 * 720 * 0.5 + 140) / 640, 0.25, 0.4 * 720 * 0.5 / 640],
                [0.5, (0.5 * 720 * 0.5 + 140) / 640, 0.05, 0.08 * 720 * 0.5 / 640],
                [0.45, (0.6 * 720 * 0.5 + 140) / 640, 0.03, 0.04 * 720 * 0.5 / 640],
            ]
        )
        torch.testing.assert_close(sample["bboxes"], expected, atol=1e-5, rtol=0)
        assert sample["labels"].tolist() == [0, 1, 1]

    def test_crop_on_enlarges_the_damage_boxes_and_stays_normalized(self, root: Path) -> None:
        off = _dataset(root, roi_crop=False)[0]
        on = _dataset(root, roi_crop=True, roi_crop_margin=0.10)[0]

        assert on["rgb"].shape == (3, 640, 640)
        assert on["nir"].shape == (1, 640, 640)
        assert on["labels"].tolist() == [0, 1, 1]
        bboxes = on["bboxes"]
        assert ((bboxes[:, :2] - bboxes[:, 2:] / 2) >= -1e-6).all()
        assert ((bboxes[:, :2] + bboxes[:, 2:] / 2) <= 1 + 1e-6).all()

        # Window: x 448..832 (384 px), y 187..533 (346 px) -> scale 640/384.
        scale = 640 / 384
        assert bboxes[1, 2].item() == pytest.approx(0.05 * FULL_W * scale / 640, abs=2e-3)
        assert bboxes[1, 3].item() == pytest.approx(0.08 * FULL_H * scale / 640, abs=2e-3)
        # Damage got at least 2.5x larger than without the crop.
        assert bboxes[1, 2] > 2.5 * off["bboxes"][1, 2]
        assert bboxes[1, 3] > 2.5 * off["bboxes"][1, 3]

    def test_rgb_and_nir_stay_aligned_after_the_crop(self, root: Path) -> None:
        sample = _dataset(root, roi_crop=True, roi_crop_margin=0.10)[0]
        rgb_xy = _centroid(sample["rgb"][0].numpy() > 1.5)  # white marker, ImageNet-normalized
        nir_xy = _centroid(sample["nir"][0].numpy() > 5.0)
        # The marker sits on the mango center, which is the window center.
        assert rgb_xy == pytest.approx((320, 320), abs=2.0)
        assert nir_xy == pytest.approx(rgb_xy, abs=1.5)

    def test_zero_margin_still_yields_valid_boxes(self, root: Path) -> None:
        sample = _dataset(root, roi_crop=True, roi_crop_margin=0.0)[0]
        assert sample["labels"].tolist() == [0, 1, 1]
        assert sample["bboxes"].min() >= 0.0 and sample["bboxes"].max() <= 1.0

    def test_image_without_a_mango_box_falls_back_to_the_full_frame(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, [DAMAGE_A])
        off = _dataset(tmp_path, roi_crop=False)[0]
        on = _dataset(tmp_path, roi_crop=True)[0]
        for key in ("rgb", "nir", "bboxes", "labels"):
            assert torch.equal(on[key], off[key]), key

    def test_jitter_is_ignored_outside_the_train_split(self, root: Path) -> None:
        plain = _dataset(root, roi_crop=True, roi_crop_jitter=0.0)[0]
        torch.manual_seed(0)
        state = torch.get_rng_state()
        jittered = _dataset(root, roi_crop=True, roi_crop_jitter=0.5)
        first, second = jittered[0], jittered[0]
        assert torch.equal(torch.get_rng_state(), state)  # val never draws randomness
        for sample in (first, second):
            assert torch.equal(sample["bboxes"], plain["bboxes"])
            assert torch.equal(sample["rgb"], plain["rgb"])

    def test_train_jitter_varies_per_sample_and_follows_torch_seed(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, [MANGO, DAMAGE_A, DAMAGE_B], split="train")
        # A custom no-op transform keeps the albumentations pipeline out of the picture.
        ds = _dataset(
            tmp_path,
            split="train",
            transform=get_val_transforms(640),
            roi_crop=True,
            roi_crop_jitter=0.5,
        )

        torch.manual_seed(7)
        a, b = ds[0], ds[0]
        torch.manual_seed(7)
        a2, b2 = ds[0], ds[0]

        assert not torch.equal(a["bboxes"], b["bboxes"])  # jitter changes the window
        assert torch.equal(a["bboxes"], a2["bboxes"])  # ... reproducibly
        assert torch.equal(b["bboxes"], b2["bboxes"])
        for sample in (a, b):
            assert sample["labels"].tolist() == [0, 1, 1]

    def test_train_jitter_is_reproducible_and_distinct_across_dataloader_workers(
        self, tmp_path: Path
    ) -> None:
        from torch.utils.data import DataLoader

        from src.training.dataset import collate_fn

        _write_dataset(tmp_path, [MANGO, DAMAGE_A, DAMAGE_B], split="train")
        for i in range(2, 7):  # identical pairs: any difference between them is jitter
            for sub, name in (("rgb", "mango_rgb_%05d.jpg"), ("nir", "mango_nir_%05d.jpg")):
                (tmp_path / sub / (name % i)).write_bytes((tmp_path / sub / (name % 1)).read_bytes())
            (tmp_path / "labels/train" / ("mango_rgb_%05d.txt" % i)).write_text(
                (tmp_path / "labels/train/mango_rgb_00001.txt").read_text()
            )
        ds = _dataset(
            tmp_path,
            split="train",
            transform=get_val_transforms(640),
            roi_crop=True,
            roi_crop_jitter=0.5,
        )

        def windows() -> list[tuple[float, ...]]:
            # The label coordinates identify the crop window of each sample.
            torch.manual_seed(0)
            loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=collate_fn)
            return [tuple(batch["bboxes"][0].flatten().round(decimals=5).tolist()) for batch in loader]

        try:
            first, second = windows(), windows()
        except (OSError, RuntimeError) as exc:  # no IPC sockets (some WSL/FUSE setups)
            pytest.skip(f"DataLoader workers unavailable here: {exc!r}")

        assert first == second  # reproducible under torch.manual_seed
        assert len(set(first)) == len(first)  # workers do not replay one jitter stream

    def test_train_split_with_default_augmentation_runs(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, [MANGO, DAMAGE_A, DAMAGE_B], split="train")
        ds = _dataset(tmp_path, split="train", roi_crop=True, roi_crop_jitter=0.5)
        for _ in range(8):
            sample = ds[0]
            assert sample["rgb"].shape == (3, 640, 640)
            assert sample["nir"].shape == (1, 640, 640)
            assert sample["bboxes"].shape[1] == 4

    @pytest.mark.parametrize(
        "kwargs",
        [{"roi_crop_margin": -0.1}, {"roi_crop_jitter": -0.1}, {"roi_crop_jitter": 1.5}],
    )
    def test_invalid_arguments_are_rejected_at_construction(self, root: Path, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            _dataset(root, roi_crop=True, **kwargs)
