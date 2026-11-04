"""Training on damage alone, and the pixel cache: same pixels, window and random draws."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from scripts.build_pixel_cache import store_pair
from src.training.decode import decode_detections
from src.training.loss import YOLOv8Loss
from src.training.roi_crop import crop_to_roi


def _loss(**kwargs) -> YOLOv8Loss:
    return YOLOv8Loss(num_classes=2, class_weights=[0.5, 1.5], strides=[8, 16, 32],
                      assigner_level_ranges=[64, 128], **kwargs)


def _predictions(requires_grad=False) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(2, 6, s, s, requires_grad=requires_grad) for s in (80, 40, 20)]


MANGO = [0.5, 0.5, 0.83, 0.83]
DAMAGE = [0.4, 0.4, 0.2, 0.2]


def _targets(*per_image) -> dict:
    return {
        "bboxes": [torch.tensor([box for _, box in rows], dtype=torch.float32).reshape(-1, 4) for rows in per_image],
        "labels": [torch.tensor([c for c, _ in rows], dtype=torch.int64) for rows in per_image],
    }


def test_without_ignored_classes_the_loss_does_not_change() -> None:
    targets = _targets([(0, MANGO), (1, DAMAGE)], [(0, MANGO)])
    a, _ = _loss()(_predictions(), targets)
    b, _ = _loss(ignore_classes=[])(_predictions(), targets)
    assert a.item() == b.item()


def test_ignoring_the_mango_is_like_having_no_mango_box_except_for_its_channel() -> None:
    with_box = _targets([(0, MANGO), (1, DAMAGE)], [(0, MANGO)])
    without_box = _targets([(1, DAMAGE)], [])
    a, _ = _loss(ignore_classes=[0])(_predictions(), with_box)
    b, _ = _loss(ignore_classes=[0])(_predictions(), without_box)
    assert a.item() == b.item()


def test_the_ignored_channel_gets_no_gradient() -> None:
    predictions = _predictions(requires_grad=True)
    loss, _ = _loss(ignore_classes=[0])(predictions, _targets([(0, MANGO), (1, DAMAGE)], [(0, MANGO)]))
    loss.backward()
    for prediction in predictions:
        assert prediction.grad[:, 0].abs().sum() == 0
        assert prediction.grad[:, 1].abs().sum() > 0


def test_a_batch_of_healthy_fruit_does_not_spike() -> None:
    healthy = _targets([(0, MANGO)], [(0, MANGO)])
    damaged = _targets([(0, MANGO), (1, DAMAGE)], [(0, MANGO), (1, DAMAGE)])
    spike, _ = _loss(ignore_classes=[0])(_predictions(), healthy)
    usual, _ = _loss(ignore_classes=[0])(_predictions(), damaged)
    assert torch.isfinite(spike) and spike.item() < usual.item()


def test_the_decode_gives_no_detection_of_an_ignored_class() -> None:
    preds = _predictions()
    cls_preds = [p[:, :2] + 5.0 for p in preds]
    kwargs = dict(batch_idx=0, num_classes=2, image_size=640, strides=(8, 16, 32), max_detections=300)
    _, _, labels = decode_detections(preds, cls_preds, **kwargs)
    assert (labels == 0).any()
    _, _, labels = decode_detections(preds, cls_preds, ignore_classes=[0], **kwargs)
    assert len(labels) > 0 and (labels == 1).all()


@pytest.mark.parametrize("jitter", [0.0, 0.5])
def test_a_stored_window_crops_exactly_like_the_full_frame(tmp_path: Path, jitter: float) -> None:
    rng = np.random.default_rng(1)
    frame = (420, 640)
    rgb = rng.integers(0, 255, (*frame, 3), dtype=np.uint8)
    nir = rng.integers(0, 255, frame, dtype=np.uint8)
    (tmp_path / "s.jpg").write_bytes(b"")
    png_rgb, png_nir = tmp_path / "rgb.png", tmp_path / "nir.png"
    cv2.imwrite(str(png_rgb), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(png_nir), nir)
    homography = tmp_path / "h.npy"
    np.save(homography, np.eye(3))
    boxes = np.array([[0.5, 0.45, 0.4, 0.5], [0.45, 0.4, 0.05, 0.05]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    label_path = tmp_path / "l.txt"
    label_path.write_text("\n".join(f"{c} {' '.join(map(str, b))}" for c, b in zip(labels, boxes)))
    output = tmp_path / "out.npz"
    store_pair((png_rgb, png_nir, label_path, output, homography, 0.15))
    stored = np.load(output)
    assert stored["rgb"].shape[0] < frame[0]

    for seed in range(30):
        full = crop_to_roi(rgb, nir, boxes, labels, margin=0.1, margin_jitter=jitter,
                           rng=np.random.default_rng(seed))
        part = crop_to_roi(stored["rgb"], stored["nir"], boxes, labels, margin=0.1, margin_jitter=jitter,
                           rng=np.random.default_rng(seed), frame_size=frame, origin=tuple(stored["origin"]))
        for a, b in zip(full, part):
            assert np.array_equal(a, b)


def test_a_window_wider_than_the_stored_one_is_an_error() -> None:
    rgb, nir = np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), np.uint8)
    boxes, labels = np.array([[0.5, 0.5, 0.2, 0.2]], np.float32), np.array([0])
    with pytest.raises(ValueError, match="pixel cache"):
        crop_to_roi(rgb, nir, boxes, labels, margin=0.5, frame_size=(100, 100), origin=(45, 45))
