"""`scripts/visualize_damage_predictions.py`: polygon labels and predicted masks as filled contours.

CPU only, a random-init `nano` student at image size 64; detect behaviour is checked to stay as it was.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from scripts import visualize_damage_predictions as viz
from src.models.student.student_model import StudentModel
from src.training.config import TrainingConfig

SIZE = 64
RED = np.array(viz.CLASS_COLORS[1])


def _checkpoint(tmp_path: Path, task: str) -> Path:
    torch.manual_seed(0)
    model = StudentModel(num_classes=2, variant="nano", task=task)
    path = tmp_path / f"{task}.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "config": {"task": task, "student_variant": "nano", "num_classes": 2}}, path)
    return path


def _config() -> TrainingConfig:
    return TrainingConfig(model_type="student", image_size=SIZE, student_variant="nano")


# --- labels --------------------------------------------------------------------------------------


def test_a_polygon_label_gives_its_tight_box_and_its_polygon(tmp_path) -> None:
    label = tmp_path / "a.txt"
    label.write_text("1 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6\n0 0.5 0.5 0.4 0.4\n")
    boxes, labels, polygons = viz.load_yolo_labels_with_polygons(label)
    assert labels.tolist() == [1, 0]
    np.testing.assert_allclose(boxes, [[0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]], atol=1e-6)
    assert polygons[0].shape == (4, 2) and polygons[1] is None
    # the box-only reader is unchanged for box lines and no longer misreads polygon lines
    plain_boxes, plain_labels = viz.load_yolo_labels(label)
    np.testing.assert_allclose(plain_boxes, boxes, atol=1e-6)
    assert plain_labels.tolist() == [1, 0]


# --- drawing -------------------------------------------------------------------------------------


def test_draw_masks_fills_the_polygon_and_leaves_the_rest() -> None:
    image = np.zeros((SIZE, SIZE, 3), np.uint8)
    square = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float64)
    out = viz.draw_masks(image, [square, None], np.array([1, 0]))
    assert out is image
    assert image[20, 20].tolist() != [0, 0, 0] and image[20, 20, 2] > image[20, 20, 0]  # red-ish fill
    assert image[50, 50].tolist() == [0, 0, 0]


def test_the_side_by_side_without_polygons_is_the_box_only_image() -> None:
    rgb = np.full((SIZE, SIZE, 3), 90, np.uint8)
    gt_boxes, gt_labels = np.array([[30.0, 30.0, 20.0, 20.0]]), np.array([1])
    pred_boxes, pred_labels, scores = np.array([[32.0, 32.0, 20.0, 20.0]]), np.array([1]), np.array([0.9])
    got = viz.create_side_by_side(rgb, gt_boxes, gt_labels, pred_boxes, pred_labels, scores)
    explicit = viz.create_side_by_side(rgb, gt_boxes, gt_labels, pred_boxes, pred_labels, scores,
                                       gt_polygons=None, pred_polygons=None)
    assert got.shape == (SIZE, 2 * SIZE, 3) and np.array_equal(got, explicit)
    assert abs(int(got[30, 30, 2]) - int(got[30, 30, 0])) < 20  # the box interior stays unfilled


def test_the_side_by_side_draws_gt_masks_left_and_predicted_masks_right() -> None:
    rgb = np.zeros((SIZE, SIZE, 3), np.uint8)
    gt_poly = [np.array([[8, 34], [24, 34], [24, 44], [8, 44]], dtype=np.float64)]
    pred_poly = [np.array([[36, 34], [56, 34], [56, 44], [36, 44]], dtype=np.float64)]
    empty = np.zeros((0, 4))
    got = viz.create_side_by_side(rgb, empty, np.array([], dtype=int), empty, np.array([], dtype=int), np.array([]),
                                  gt_polygons=gt_poly, pred_polygons=pred_poly, gt_mask_labels=np.array([1]),
                                  pred_mask_labels=np.array([1]))
    left, right = got[:, :SIZE], got[:, SIZE:]
    assert left[39, 16].tolist() != [0, 0, 0] and right[39, 16].tolist() == [0, 0, 0]
    assert right[39, 46].tolist() != [0, 0, 0] and left[39, 46].tolist() == [0, 0, 0]


# --- model and decode ----------------------------------------------------------------------------


def test_a_segment_nano_checkpoint_loads_for_visualization(tmp_path) -> None:
    model, strides = viz.load_model("student", str(_checkpoint(tmp_path, "segment")), _config(), torch.device("cpu"))
    assert model.task == "segment" and model.variant == "nano" and strides == [8, 16, 32]


def test_a_detect_checkpoint_still_loads_as_detect(tmp_path) -> None:
    model, _ = viz.load_model("student", str(_checkpoint(tmp_path, "detect")), _config(), torch.device("cpu"))
    assert model.task == "detect"


def test_the_segmentation_decode_returns_masks_for_its_detections(tmp_path) -> None:
    model, strides = viz.load_model("student", str(_checkpoint(tmp_path, "segment")), _config(), torch.device("cpu"))
    with torch.no_grad():
        output = model(torch.randn(1, 3, SIZE, SIZE))
    boxes, scores, labels, masks = viz.decode_segmentation_predictions(
        output, 2, SIZE, 0.005, strides=tuple(strides), max_detections=20)
    assert len(boxes) > 0 and masks.shape == (len(boxes), SIZE // 4, SIZE // 4)
    plain = viz.decode_predictions(output, 2, SIZE, 0.005, strides=tuple(strides), max_detections=20)
    np.testing.assert_allclose(boxes, plain[0])
    np.testing.assert_array_equal(labels, plain[2])


def test_mask_polygons_scale_the_prototype_grid_to_the_letterboxed_image() -> None:
    mask = torch.zeros(2, SIZE // 4, SIZE // 4, dtype=torch.bool)
    mask[0, 4:8, 4:12] = True  # the second mask is empty
    polygons = viz.mask_polygons_px(mask, SIZE)
    assert polygons[1] is None
    assert polygons[0][:, 0].min() == 16 and polygons[0][:, 0].max() == 48  # 4 * 4 .. 12 * 4
    assert polygons[0][:, 1].min() == 16 and polygons[0][:, 1].max() == 32


# --- end to end ----------------------------------------------------------------------------------


def _run_main(tmp_path: Path, monkeypatch, task: str, label: str) -> np.ndarray:
    for sub in ("rgb", "labels/test"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp_path / "rgb" / "mango_rgb_1.jpg"), np.full((48, 96, 3), 120, np.uint8))
    (tmp_path / "labels" / "test" / "mango_rgb_1.txt").write_text(label)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"model_type": "student", "image_size": SIZE, "student_variant": "nano",
                                      "num_classes": 2}))
    out = tmp_path / f"out_{task}"
    monkeypatch.setattr(sys, "argv", [
        "visualize_damage_predictions.py", "--checkpoint", str(_checkpoint(tmp_path, task)),
        "--config", str(config), "--rgb-dir", str(tmp_path / "rgb"), "--labels-dir", str(tmp_path / "labels"),
        "--model-type", "student", "--split", "test", "--num-samples", "1", "--conf-threshold", "0.005",
        "--output-dir", str(out), "--device", "cpu",
    ])
    assert viz.main() == 0
    image = cv2.imread(str(out / "mango_rgb_1_gt_vs_pred.png"))
    assert image is not None and image.shape == (SIZE, 2 * SIZE, 3)
    return image


def test_main_draws_the_ground_truth_polygon_for_a_segment_checkpoint(tmp_path, monkeypatch) -> None:
    # A right triangle, so its filled interior is not its box. The 96 x 48 image letterboxes to 64 with scale 2/3
    # and pad_y 16: the triangle has its right angle at (6.4, 32), runs to (57.6, 32) and to (6.4, 48). The
    # title and count text sit above y = 32 and below y = 46, so rows 34-45 are clean.
    image = _run_main(tmp_path, monkeypatch, "segment", "1 0.1 0.5 0.9 0.5 0.1 1.0\n")
    left = image[:, :SIZE].astype(int)
    inside = left[38, 12]
    assert inside[2] - inside[0] > 40  # red fill over the grey image
    outside = left[44, 50]  # inside the triangle's box, below its hypotenuse
    assert abs(outside[2] - outside[0]) < 20


def test_main_on_a_detect_checkpoint_keeps_drawing_boxes_only(tmp_path, monkeypatch) -> None:
    label = "1 0.5 0.5 0.4 0.4\n"
    image = _run_main(tmp_path, monkeypatch, "detect", label)
    left = image[:, :SIZE].astype(int)
    assert abs(int(left[32, 32][2]) - int(left[32, 32][0])) < 20  # box interior is not filled
