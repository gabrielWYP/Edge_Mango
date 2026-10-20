"""scripts/position_prior_check.py: where do damage boxes land in the model input?

CPU-only evidence for the positional prior (no model): iterate the train split of
a fold through `YOLODataset` under two crop configurations and report the
distribution of the damage-box centers in input pixel coordinates.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import position_prior_check as ppc

W, H = 1280, 720
MANGO = "0 0.5 0.5 0.25 0.4"
DAMAGE_A = "1 0.5 0.5 0.05 0.08"
DAMAGE_B = "1 0.45 0.6 0.03 0.04"


def _write_pair(root: Path, index: int, label_lines: list[str]) -> None:
    for sub in ("rgb", "nir", "labels/train"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rgb = np.full((H, W, 3), 60, dtype=np.uint8)
    nir = np.full((H, W), 20, dtype=np.uint8)
    name = f"mango_rgb_{index:05d}"
    cv2.imwrite(str(root / "rgb" / f"{name}.jpg"), rgb)
    cv2.imwrite(str(root / "nir" / f"{name.replace('_rgb', '_nir')}.jpg"), nir)
    (root / "labels" / "train" / f"{name}.txt").write_text("\n".join(label_lines) + "\n")


@pytest.fixture
def fold(tmp_path: Path) -> Path:
    _write_pair(tmp_path, 1, [MANGO, DAMAGE_A, DAMAGE_B])
    _write_pair(tmp_path, 2, [MANGO, DAMAGE_A])
    _write_pair(tmp_path, 3, [MANGO])  # a mango without damage contributes nothing
    return tmp_path


def _args(fold: Path, *extra: str) -> list[str]:
    return [
        "--rgb-dir", str(fold / "rgb"),
        "--nir-dir", str(fold / "nir"),
        "--labels-dir", str(fold / "labels"),
        "--homography", "none",
        *extra,
    ]


class TestSummarize:
    def test_known_distribution(self) -> None:
        # Input center is (320, 320); the window is +-64 px on BOTH axes.
        centers = np.array([[320.0, 320.0], [384.0, 320.0], [320.0, 420.0], [100.0, 100.0]])
        stats = ppc.summarize_centers(centers, image_size=640, window=64)
        assert stats["n"] == 4
        assert stats["mean_x"] == pytest.approx(281.0)
        assert stats["mean_y"] == pytest.approx(290.0)
        assert stats["sd_x"] == pytest.approx(np.std([320, 384, 320, 100]))
        assert stats["sd_y"] == pytest.approx(np.std([320, 320, 420, 100]))
        assert stats["frac_central"] == pytest.approx(2 / 4)   # (320,320) and the border (384,320)

    def test_empty_input_does_not_crash(self) -> None:
        stats = ppc.summarize_centers(np.zeros((0, 2)), image_size=640, window=64)
        assert stats["n"] == 0
        assert np.isnan(stats["mean_x"]) and np.isnan(stats["frac_central"])


class TestCollect:
    def _dataset(self, fold: Path, **crop):
        return ppc.build_dataset(
            rgb_dir=fold / "rgb", nir_dir=fold / "nir", labels_dir=fold / "labels",
            manifest=None, homography=None, split="train", image_size=640,
            augment=False, **crop,
        )

    def test_one_row_per_damage_box_per_pass(self, fold: Path) -> None:
        centers = ppc.collect_damage_centers(self._dataset(fold, **ppc.CURRENT_CROP), passes=3, seed=0)
        assert centers.shape == (3 * 3, 2)     # 3 damage boxes in the fold, 3 passes

    def test_same_seed_same_centers(self, fold: Path) -> None:
        shift = ppc.SHIFT_CROP
        first = ppc.collect_damage_centers(self._dataset(fold, **shift), passes=2, seed=5)
        second = ppc.collect_damage_centers(self._dataset(fold, **shift), passes=2, seed=5)
        other = ppc.collect_damage_centers(self._dataset(fold, **shift), passes=2, seed=6)
        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, other)

    def test_the_shift_config_spreads_the_centers_far_more_than_the_current_one(
        self, tmp_path: Path
    ) -> None:
        # One damage box at the mango center per image: any spread is the crop's doing.
        for index in (1, 2):
            _write_pair(tmp_path, index, [MANGO, DAMAGE_A])
        current = ppc.collect_damage_centers(
            self._dataset(tmp_path, **ppc.CURRENT_CROP), passes=20, seed=0
        )
        shifted = ppc.collect_damage_centers(
            self._dataset(tmp_path, **ppc.SHIFT_CROP), passes=20, seed=0
        )
        assert shifted[:, 0].std() > 3 * current[:, 0].std()
        assert shifted[:, 1].std() > 3 * current[:, 1].std()

    def test_the_two_configs_are_the_documented_ones(self) -> None:
        assert ppc.CURRENT_CROP == {
            "roi_crop_margin": 0.10, "roi_crop_jitter": 0.5, "roi_crop_train_margin_range": None,
        }
        assert ppc.SHIFT_CROP == {
            "roi_crop_margin": 0.10, "roi_crop_jitter": 0.0, "roi_crop_train_margin_range": (0.2, 1.0),
        }

    def test_with_augmentation_on_the_pipeline_still_runs(self, fold: Path) -> None:
        dataset = ppc.build_dataset(
            rgb_dir=fold / "rgb", nir_dir=fold / "nir", labels_dir=fold / "labels",
            manifest=None, homography=None, split="train", image_size=640,
            augment=True, **ppc.CURRENT_CROP,
        )
        centers = ppc.collect_damage_centers(dataset, passes=2, seed=0)
        assert centers.shape[1] == 2 and 0 < len(centers) <= 2 * 3
        assert ((centers >= 0) & (centers <= 640)).all()


class TestMain:
    def test_prints_a_table_with_both_configs_and_returns_zero(self, fold: Path, capsys) -> None:
        assert ppc.main(_args(fold, "--passes", "2")) == 0
        out = capsys.readouterr().out
        assert "current" in out and "shift" in out
        assert "mean x" in out and "sd x" in out and "±64" in out
        assert "2 passes" in out

    def test_missing_labels_fail_loudly(self, fold: Path) -> None:
        with pytest.raises(SystemExit):
            ppc.main(["--rgb-dir", str(fold / "rgb"), "--nir-dir", str(fold / "nir")])

    def test_an_empty_split_is_an_error_not_a_table_of_nans(self, tmp_path: Path, capsys) -> None:
        (tmp_path / "rgb").mkdir()
        (tmp_path / "nir").mkdir()
        (tmp_path / "labels" / "train").mkdir(parents=True)
        assert ppc.main(_args(tmp_path)) == 1
        assert "no damage boxes" in capsys.readouterr().err.lower()
