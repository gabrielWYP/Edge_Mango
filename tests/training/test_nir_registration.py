"""NIR -> RGB registration: warp direction, dataset wiring, loud failures.

`H` maps RGB -> NIR (estimated as `cv2.findHomography(pts_rgb, pts_nir)`), so
NIR must be resampled into the RGB frame with `inv(H)`. Using `H` itself (or
skipping the warp) misaligns NIR against the RGB-frame labels.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.training.config import TrainingConfig
from src.training.dataset import YOLODataset
from src.training.nir_registration import (
    DEFAULT_NIR_HOMOGRAPHY_PATH,
    homography_sha256,
    load_homography,
    register_nir_to_rgb,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# RGB -> NIR: scale 0.9 plus a non-symmetric translation, so H and inv(H)
# send any point to clearly different places.
H_RGB_TO_NIR = np.array(
    [[0.9, 0.0, -30.0], [0.0, 0.9, 20.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
RGB_W, RGB_H = 160, 120


def _apply(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = H @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def _nir_with_block_at_rgb_point(x_rgb: float, y_rgb: float) -> tuple[np.ndarray, tuple[int, int]]:
    """NIR frame image with a bright 5x5 block where the RGB point lands in NIR."""
    x_nir, y_nir = (int(round(v)) for v in _apply(H_RGB_TO_NIR, x_rgb, y_rgb))
    nir = np.zeros((RGB_H, RGB_W), dtype=np.uint8)
    nir[y_nir - 2 : y_nir + 3, x_nir - 2 : x_nir + 3] = 255
    return nir, (x_nir, y_nir)


def _centroid(image: np.ndarray, threshold: int = 128) -> tuple[float, float]:
    ys, xs = np.nonzero(image > threshold)
    assert xs.size > 0, "bright block vanished"
    return float(xs.mean()), float(ys.mean())


class TestRegisterNirToRgb:
    def test_rgb_point_lands_on_expected_nir_pixel_and_back(self) -> None:
        x_rgb, y_rgb = 100, 60
        nir, (x_nir, y_nir) = _nir_with_block_at_rgb_point(x_rgb, y_rgb)
        # Sanity: the block really is displaced in the raw NIR frame.
        assert (x_nir, y_nir) == (60, 74)
        assert _centroid(nir) == pytest.approx((x_nir, y_nir), abs=0.1)

        registered = register_nir_to_rgb(nir, H_RGB_TO_NIR, (RGB_W, RGB_H))

        # inv(H): the NIR pixel H(p_rgb) is mapped back onto p_rgb.
        assert _centroid(registered) == pytest.approx((x_rgb, y_rgb), abs=1.0)

    def test_forward_h_direction_would_be_wrong(self) -> None:
        x_rgb, y_rgb = 100, 60
        nir, _ = _nir_with_block_at_rgb_point(x_rgb, y_rgb)
        wrong = cv2.warpPerspective(nir, H_RGB_TO_NIR, (RGB_W, RGB_H))
        cx, cy = _centroid(wrong)
        assert abs(cx - x_rgb) > 20 or abs(cy - y_rgb) > 20

    def test_output_has_rgb_size_and_constant_border_fill(self) -> None:
        nir = np.full((90, 100), 200, dtype=np.uint8)  # smaller than the RGB frame
        out = register_nir_to_rgb(nir, H_RGB_TO_NIR, (RGB_W, RGB_H), border_value=14)
        assert out.shape == (RGB_H, RGB_W)
        # Right/bottom of the RGB frame has no NIR source (NIR image is only 100x90).
        assert out[RGB_H - 1, RGB_W - 1] == 14
        assert 200 in out


class TestLoadHomography:
    def test_missing_file_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="nir_homography_path"):
            load_homography(tmp_path / "nope.npy")

    @pytest.mark.parametrize(
        "bad", [np.eye(2), np.zeros((3, 3)), np.full((3, 3), np.nan)], ids=["shape", "singular", "nan"]
    )
    def test_invalid_matrix_rejected(self, tmp_path: Path, bad: np.ndarray) -> None:
        path = tmp_path / "bad.npy"
        np.save(path, bad)
        with pytest.raises(ValueError):
            load_homography(path)

    def test_repo_default_is_a_valid_rgb_to_nir_homography(self) -> None:
        H = load_homography(REPO_ROOT / DEFAULT_NIR_HOMOGRAPHY_PATH)
        assert H.shape == (3, 3)
        assert TrainingConfig().nir_homography_path == DEFAULT_NIR_HOMOGRAPHY_PATH

    def test_sha256_is_none_when_registration_is_off(self, tmp_path: Path) -> None:
        assert homography_sha256(None) is None
        path = tmp_path / "h.npy"
        np.save(path, H_RGB_TO_NIR)
        digest = homography_sha256(path)
        assert digest is not None and len(digest) == 64


def _write_pair(root: Path, nir: np.ndarray) -> None:
    for sub in ("rgb", "nir", "labels/val"):
        (root / sub).mkdir(parents=True)
    rgb = np.full((RGB_H, RGB_W, 3), 90, dtype=np.uint8)
    cv2.imwrite(str(root / "rgb" / "mango_rgb_00001.jpg"), rgb)
    cv2.imwrite(str(root / "nir" / "mango_nir_00001.jpg"), nir)
    (root / "labels" / "val" / "mango_rgb_00001.txt").write_text("0 0.5 0.5 0.2 0.2\n")


class TestDatasetRegistersNir:
    IMAGE_SIZE = RGB_W  # 160x120 RGB -> scale 1.0, vertical pad of 20 px

    def _dataset(self, root: Path, homography: Path | None) -> YOLODataset:
        return YOLODataset(
            rgb_dir=root / "rgb",
            nir_dir=root / "nir",
            labels_dir=root / "labels",
            split="val",
            image_size=self.IMAGE_SIZE,
            nir_homography_path=homography,
        )

    def _nir_centroid(self, sample: dict, ds: YOLODataset) -> tuple[float, float]:
        raw = sample["nir"][0].numpy() * ds.nir_std + ds.nir_mean  # back to [0, 1]
        return _centroid((raw * 255).astype(np.uint8))

    def test_nir_is_registered_before_letterbox(self, tmp_path: Path) -> None:
        x_rgb, y_rgb = 100, 60
        nir, _ = _nir_with_block_at_rgb_point(x_rgb, y_rgb)
        _write_pair(tmp_path, nir)
        h_path = tmp_path / "H.npy"
        np.save(h_path, H_RGB_TO_NIR)

        ds = self._dataset(tmp_path, h_path)
        cx, cy = self._nir_centroid(ds[0], ds)
        pad_y = (self.IMAGE_SIZE - RGB_H) // 2
        assert (cx, cy - pad_y) == pytest.approx((x_rgb, y_rgb), abs=1.5)

    def test_registered_border_matches_letterbox_pad_value(self, tmp_path: Path) -> None:
        nir = np.full((RGB_H, RGB_W), 200, dtype=np.uint8)
        _write_pair(tmp_path, nir)
        h_path = tmp_path / "H.npy"
        np.save(h_path, H_RGB_TO_NIR)

        ds = self._dataset(tmp_path, h_path)
        out = ds[0]["nir"][0].numpy()
        pad = int(ds.nir_mean * 255) / 255.0
        expected_pad = (pad - ds.nir_mean) / ds.nir_std
        # Letterbox rows and the warp's uncovered region both use the pad value.
        assert out[0, 0] == pytest.approx(expected_pad, abs=1e-4)
        assert out[self.IMAGE_SIZE // 2, 0] == pytest.approx(expected_pad, abs=1e-4)

    def test_registration_can_be_disabled_explicitly(self, tmp_path: Path) -> None:
        x_rgb, y_rgb = 100, 60
        nir, (x_nir, y_nir) = _nir_with_block_at_rgb_point(x_rgb, y_rgb)
        _write_pair(tmp_path, nir)

        ds = self._dataset(tmp_path, None)
        cx, cy = self._nir_centroid(ds[0], ds)
        pad_y = (self.IMAGE_SIZE - RGB_H) // 2
        assert (cx, cy - pad_y) == pytest.approx((x_nir, y_nir), abs=1.5)

    def test_kwarg_is_required(self, tmp_path: Path) -> None:
        nir, _ = _nir_with_block_at_rgb_point(100, 60)
        _write_pair(tmp_path, nir)
        with pytest.raises(TypeError, match="nir_homography_path"):
            YOLODataset(
                rgb_dir=tmp_path / "rgb",
                nir_dir=tmp_path / "nir",
                labels_dir=tmp_path / "labels",
                split="val",
                image_size=self.IMAGE_SIZE,
            )

    def test_missing_homography_raises_at_construction(self, tmp_path: Path) -> None:
        nir, _ = _nir_with_block_at_rgb_point(100, 60)
        _write_pair(tmp_path, nir)
        with pytest.raises(FileNotFoundError):
            self._dataset(tmp_path, tmp_path / "missing.npy")


class TestCheckpointRecordsRegistration:
    def _save(self, tmp_path: Path, homography_path: str | None) -> dict:
        import torch

        from src.training.loop import Trainer

        trainer = Trainer.__new__(Trainer)
        trainer.output_dir = tmp_path
        trainer.config = TrainingConfig(
            output_dir=str(tmp_path), nir_homography_path=homography_path
        )
        trainer.model = torch.nn.Linear(1, 1)
        trainer.best_map50 = 0.0
        trainer.best_score = 0.0
        trainer.experiment_sha256 = None
        trainer._save_checkpoint(0, 1, {}, "ckpt.pt")
        return torch.load(tmp_path / "ckpt.pt", weights_only=False)

    def test_registered_run_records_path_and_hash(self, tmp_path: Path) -> None:
        h_path = tmp_path / "H.npy"
        np.save(h_path, H_RGB_TO_NIR)
        ckpt = self._save(tmp_path, str(h_path))
        assert ckpt["nir_registration"] == {
            "homography_path": str(h_path),
            "homography_sha256": homography_sha256(h_path),
        }
        assert ckpt["config"]["nir_homography_path"] == str(h_path)

    def test_unregistered_run_is_distinguishable(self, tmp_path: Path) -> None:
        ckpt = self._save(tmp_path, None)
        assert ckpt["nir_registration"] == {"homography_path": None, "homography_sha256": None}
