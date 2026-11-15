"""ROI-crop preprocessing of the export tooling must equal the training dataset's eval path."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.export.reference import preprocess_roi_bgr, roi_letterbox_to_source
from src.training.dataset import YOLODataset

W, H = 640, 360
MANGO = "0 0.5 0.5 0.3 0.5"
DAMAGE = "1 0.55 0.55 0.05 0.08"


def _make_dataset_dir(root: Path) -> tuple[Path, Path, Path]:
    rgb_dir, nir_dir = root / "rgb", root / "nir"
    labels = root / "labels" / "val"
    for d in (rgb_dir, nir_dir, labels):
        d.mkdir(parents=True)
    rng = np.random.default_rng(1)
    cases = {
        "mango_rgb_1": f"{MANGO}\n{DAMAGE}\n",
        "mango_rgb_2": f"{DAMAGE}\n",  # no mango box: the crop falls back to the full frame
    }
    for stem, label in cases.items():
        cv2.imwrite(str(rgb_dir / f"{stem}.jpg"), rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
        cv2.imwrite(str(nir_dir / f"{stem.replace('_rgb', '_nir')}.png"), rng.integers(0, 255, (H, W), dtype=np.uint8))
        (labels / f"{stem}.txt").write_text(label)
    return rgb_dir, nir_dir, root / "labels"


@pytest.mark.parametrize("margin", [0.10, 0.25])
def test_preprocess_roi_matches_the_training_dataset(tmp_path, margin) -> None:
    rgb_dir, nir_dir, labels_dir = _make_dataset_dir(tmp_path)
    dataset = YOLODataset(
        rgb_dir=rgb_dir, nir_dir=nir_dir, labels_dir=labels_dir, split="val", image_size=160,
        nir_mean=0.05, nir_std=0.05, nir_homography_path=None, roi_crop=True, roi_crop_margin=margin,
    )
    assert len(dataset) == 2
    for idx, pair in enumerate(dataset.pairs):
        expected = dataset[idx]["rgb"]
        image = cv2.imread(str(pair["rgb_path"]))
        tensor, _ = preprocess_roi_bgr(image, pair["label_path"], image_size=160, pad_value=114, roi_margin=margin)
        assert tensor.shape == (1, 3, 160, 160)
        torch.testing.assert_close(tensor[0], expected)


def test_without_roi_crop_it_is_the_plain_letterbox(tmp_path) -> None:
    rgb_dir, _, labels_dir = _make_dataset_dir(tmp_path)
    image = cv2.imread(str(rgb_dir / "mango_rgb_1.jpg"))
    full, info = preprocess_roi_bgr(image, labels_dir / "val" / "mango_rgb_1.txt", image_size=160, roi_margin=None)
    assert (info.crop_x, info.crop_y, info.crop_w, info.crop_h) == (0, 0, W, H)


def test_the_crop_window_and_the_inverse_map_recover_source_coordinates(tmp_path) -> None:
    rgb_dir, _, labels_dir = _make_dataset_dir(tmp_path)
    image = cv2.imread(str(rgb_dir / "mango_rgb_1.jpg"))
    _, info = preprocess_roi_bgr(image, labels_dir / "val" / "mango_rgb_1.txt", image_size=160, roi_margin=0.10)
    # Mango box: x 0.35..0.65 of 640 = 224..416, y 0.25..0.75 of 360 = 90..270; margin 10% per side.
    assert (info.crop_x, info.crop_y) == (int(np.floor(224 - 0.1 * 192 + 1e-3)), int(np.floor(90 - 0.1 * 180 + 1e-3)))
    assert info.orig_w == W and info.orig_h == H
    # The centre of the letterboxed crop is the centre of the crop window in the source frame.
    centre = torch.tensor([[80.0, 80.0]])
    source = roi_letterbox_to_source(centre, info)
    expected_x = (info.crop_x + info.crop_w / 2) / W
    expected_y = (info.crop_y + info.crop_h / 2) / H
    assert source[0].tolist() == pytest.approx([expected_x, expected_y], abs=1.5 / W)


def test_inverse_map_equals_evaluate_checkpoint_to_source_frame(tmp_path) -> None:
    """Predictions written by eval_tflite.py land in the same source frame as evaluate_checkpoint.py's."""
    evaluate_checkpoint = pytest.importorskip("scripts.evaluate_checkpoint")
    rgb_dir, nir_dir, labels_dir = _make_dataset_dir(tmp_path)
    dataset = YOLODataset(
        rgb_dir=rgb_dir, nir_dir=nir_dir, labels_dir=labels_dir, split="val", image_size=160,
        nir_mean=0.05, nir_std=0.05, nir_homography_path=None, roi_crop=True, roi_crop_margin=0.10,
    )
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.3], [0.3, 0.6, 0.1, 0.2]])  # cxcywh normalized to the model input
    for pair in dataset.pairs:
        expected = evaluate_checkpoint._to_source_frame(boxes, dataset, pair)
        _, info = preprocess_roi_bgr(cv2.imread(str(pair["rgb_path"])), pair["label_path"], 160, 114, 0.10)
        cx, cy, w, h = (boxes * 160).unbind(-1)
        corners = torch.stack([torch.stack([cx - w / 2, cy - h / 2], -1), torch.stack([cx + w / 2, cy + h / 2], -1)], 1)
        source = roi_letterbox_to_source(corners, info).reshape(-1, 4)
        torch.testing.assert_close(source, expected, atol=1e-6, rtol=0)
