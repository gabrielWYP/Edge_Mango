"""Polygon labels in the dataset: parsing, ROI crop, letterbox, augmentation, collate."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.training.dataset import YOLODataset, collate_fn
from src.training.roi_crop import crop_to_roi
from src.training import seg_labels

FULL_W, FULL_H = 1280, 720
# Mango rectangle-ish polygon, damage triangle fully inside it, and a box-only damage.
MANGO_POLY = "0 0.375 0.3 0.625 0.3 0.625 0.7 0.375 0.7"
DAMAGE_POLY = "1 0.45 0.45 0.55 0.45 0.5 0.6"
DAMAGE_BOX = "1 0.55 0.6 0.04 0.06"


# --- label parsing -----------------------------------------------------------------------------


def test_parse_box_line_has_no_polygon() -> None:
    cls_id, box, polygon = seg_labels.parse_label_line("1 0.5 0.5 0.2 0.4".split())
    assert cls_id == 1 and polygon is None
    assert box.tolist() == pytest.approx([0.5, 0.5, 0.2, 0.4])


def test_parse_polygon_line_derives_the_tight_box() -> None:
    cls_id, box, polygon = seg_labels.parse_label_line(DAMAGE_POLY.split())
    assert cls_id == 1 and polygon.shape == (3, 2)
    assert box.tolist() == pytest.approx([0.5, 0.525, 0.1, 0.15])


@pytest.mark.parametrize("line", ["", "1 0.5 0.5", "1 0.5 0.5 0.0 0.2", "1 0.1 0.1 0.1 0.2 0.1 0.3"])
def test_degenerate_lines_are_skipped(line: str) -> None:
    # Too short, zero-width box, zero-width polygon.
    assert seg_labels.parse_label_line(line.split()) is None


def test_format_polygon_line_round_trips() -> None:
    polygon = np.array([[0.1, 0.2], [0.3, 0.2], [0.2, 0.4]])
    line = seg_labels.format_polygon_line(1, polygon)
    _, _, parsed = seg_labels.parse_label_line(line.split())
    assert parsed == pytest.approx(polygon, abs=1e-6)


def test_rasterize_and_tight_box_agree() -> None:
    masks = seg_labels.rasterize_polygons([np.array([[10.0, 20.0], [50.0, 20.0], [50.0, 60.0], [10.0, 60.0]])], (100, 100))
    assert masks.shape == (1, 100, 100) and masks.dtype == np.uint8
    box = seg_labels.mask_bbox_cxcywh(masks[0])
    assert box == pytest.approx([30.5 / 100, 40.5 / 100, 41 / 100, 41 / 100], abs=1e-6)
    assert seg_labels.mask_bbox_cxcywh(np.zeros((10, 10), np.uint8)) is None


# --- ROI crop ----------------------------------------------------------------------------------


def test_crop_to_roi_moves_polygons_with_the_boxes() -> None:
    rgb = np.zeros((720, 1280, 3), np.uint8)
    nir = np.zeros((720, 1280), np.uint8)
    boxes = np.array([[0.5, 0.5, 0.25, 0.4], [0.5, 0.525, 0.1, 0.15]], np.float32)
    labels = np.array([0, 1])
    polygons = [None, np.array([[0.45, 0.45], [0.55, 0.45], [0.5, 0.6]])]
    *_, boxes_c, labels_c, polygons_c = crop_to_roi(rgb, nir, boxes, labels, margin=0.0, polygons=polygons)
    assert polygons_c[0] is None
    # With no margin the window is the mango box: x 0.375..0.625, y 0.3..0.7 of the frame.
    expected = np.array([[(0.45 - 0.375) / 0.25, (0.45 - 0.3) / 0.4],
                         [(0.55 - 0.375) / 0.25, (0.45 - 0.3) / 0.4],
                         [(0.5 - 0.375) / 0.25, (0.6 - 0.3) / 0.4]])
    assert polygons_c[1] == pytest.approx(expected, abs=2e-3)
    # The polygon's tight box and the cropped box agree.
    _, tight, _ = seg_labels.parse_label_line(seg_labels.format_polygon_line(1, polygons_c[1]).split())
    assert tight == pytest.approx(boxes_c[1], abs=3e-3)


def test_crop_to_roi_without_polygons_keeps_its_four_results() -> None:
    rgb, nir = np.zeros((720, 1280, 3), np.uint8), np.zeros((720, 1280), np.uint8)
    result = crop_to_roi(rgb, nir, np.array([[0.5, 0.5, 0.25, 0.4]], np.float32), np.array([0]))
    assert len(result) == 4


def test_crop_to_roi_polygons_stay_aligned_when_a_box_is_dropped() -> None:
    rgb, nir = np.zeros((720, 1280, 3), np.uint8), np.zeros((720, 1280), np.uint8)
    boxes = np.array([[0.5, 0.5, 0.25, 0.4], [0.05, 0.05, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05]], np.float32)
    polygons = [None, np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]]),
                np.array([[0.48, 0.48], [0.52, 0.48], [0.5, 0.52]])]
    *_, labels_c, polygons_c = crop_to_roi(rgb, nir, boxes, np.array([0, 1, 1]), polygons=polygons)
    assert labels_c.tolist() == [0, 1] and len(polygons_c) == 2 and polygons_c[1].shape == (3, 2)


# --- dataset -----------------------------------------------------------------------------------


def _write_dataset(root: Path, lines: list[str], split: str = "val") -> None:
    for sub in ("rgb", "nir", f"labels/{split}"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rgb = np.full((FULL_H, FULL_W, 3), 60, dtype=np.uint8)
    nir = np.full((FULL_H, FULL_W), 20, dtype=np.uint8)
    cv2.imwrite(str(root / "rgb" / "mango_rgb_00001.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(root / "nir" / "mango_nir_00001.jpg"), nir)
    (root / "labels" / split / "mango_rgb_00001.txt").write_text("\n".join(lines) + "\n")


def _dataset(root: Path, split: str = "val", **kwargs) -> YOLODataset:
    return YOLODataset(
        rgb_dir=root / "rgb", nir_dir=root / "nir", labels_dir=root / "labels",
        split=split, image_size=640, nir_homography_path=None, **kwargs,
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _write_dataset(tmp_path, [MANGO_POLY, DAMAGE_POLY, DAMAGE_BOX])
    return tmp_path


def test_polygon_labels_train_a_detect_student_as_boxes(root: Path) -> None:
    sample = _dataset(root)[0]
    assert set(sample) == {"rgb", "nir", "bboxes", "labels", "image_id"}
    assert sample["labels"].tolist() == [0, 1, 1]
    # The mango polygon is the 0.25 x 0.4 rectangle: its tight box.
    assert sample["bboxes"][0, 2].item() == pytest.approx(0.25, abs=1e-5)


def test_load_masks_adds_masks_and_has_mask(root: Path) -> None:
    sample = _dataset(root, load_masks=True)[0]
    assert sample["masks"].shape == (3, 160, 160) and sample["masks"].dtype == torch.uint8
    assert sample["has_mask"].tolist() == [True, True, False]
    assert sample["masks"][2].sum() == 0  # box-only instance: no mask
    # Mango polygon area: 0.25 * 0.4 of the frame = 320 x 288 px -> x0.5 on 640 -> /4 on 160.
    assert sample["masks"][0].sum().item() == pytest.approx(160 * 144 / 16, rel=0.05)
    # Damage triangle: base 0.1 * 1280, height 0.15 * 720 -> half the box, same scaling.
    assert sample["masks"][1].sum().item() == pytest.approx(108, rel=0.15)


def test_boxes_come_from_the_masks_and_match_the_polygon(root: Path) -> None:
    sample = _dataset(root, load_masks=True)[0]
    plain = _dataset(root)[0]
    torch.testing.assert_close(sample["bboxes"][:2], plain["bboxes"][:2], atol=3e-3, rtol=0)
    assert sample["bboxes"][2].tolist() == pytest.approx(plain["bboxes"][2].tolist())  # box-only untouched


def test_roi_crop_masks_follow_the_crop(root: Path) -> None:
    sample = _dataset(root, load_masks=True, roi_crop=True, roi_crop_margin=0.10)[0]
    mango = sample["masks"][0].numpy()
    ys, xs = np.nonzero(mango)
    # 10 % margin on each side of the mango box, scaled to fill the 640 canvas: the mango
    # spans ~ 1 / 1.2 of the width.
    assert (xs.max() - xs.min() + 1) / 160 == pytest.approx(1 / 1.2, abs=0.03)
    box = sample["bboxes"][0]
    assert xs.min() / 160 == pytest.approx((box[0] - box[2] / 2).item(), abs=0.02)


def test_no_mask_for_box_only_files(tmp_path: Path) -> None:
    _write_dataset(tmp_path, ["0 0.5 0.5 0.25 0.4", "1 0.5 0.5 0.05 0.08"])
    sample = _dataset(tmp_path, load_masks=True)[0]
    assert sample["has_mask"].tolist() == [False, False]
    assert sample["masks"].sum() == 0 and sample["bboxes"].shape == (2, 4)


def test_empty_label_file_gives_empty_masks(tmp_path: Path) -> None:
    _write_dataset(tmp_path, [""])
    sample = _dataset(tmp_path, load_masks=True)[0]
    assert sample["masks"].shape == (0, 160, 160) and sample["has_mask"].shape == (0,)


def test_train_augmentation_keeps_masks_boxes_and_pixels_consistent(tmp_path: Path) -> None:
    _write_dataset(tmp_path, [MANGO_POLY, DAMAGE_POLY], split="train")
    dataset = _dataset(tmp_path, split="train", load_masks=True)
    torch.manual_seed(0)
    np.random.seed(0)
    seen = 0
    for _ in range(40):
        sample = dataset[0]
        for box, mask, has in zip(sample["bboxes"], sample["masks"], sample["has_mask"]):
            assert has
            ys, xs = np.nonzero(mask.numpy())
            assert xs.size > 0
            x0, x1 = xs.min() / 160, (xs.max() + 1) / 160
            y0, y1 = ys.min() / 160, (ys.max() + 1) / 160
            assert box[0].item() == pytest.approx((x0 + x1) / 2, abs=0.02)
            assert box[1].item() == pytest.approx((y0 + y1) / 2, abs=0.02)
            assert box[2].item() == pytest.approx(x1 - x0, abs=0.03)
            assert box[3].item() == pytest.approx(y1 - y0, abs=0.03)
            seen += 1
    assert seen >= 40


def test_collate_keeps_masks_as_lists(root: Path) -> None:
    dataset = _dataset(root, load_masks=True)
    batch = collate_fn([dataset[0], dataset[0]])
    assert len(batch["masks"]) == 2 and batch["masks"][0].shape == (3, 160, 160)
    assert batch["has_mask"][1].tolist() == [True, True, False]
    plain = collate_fn([_dataset(root)[0]])
    assert "masks" not in plain and "has_mask" not in plain


def test_custom_transform_path_carries_masks(root: Path) -> None:
    import albumentations as A

    flip = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0, min_area=0.0),
    )
    plain = _dataset(root, load_masks=True)[0]
    flipped = _dataset(root, load_masks=True, transform=flip)[0]
    assert torch.equal(flipped["masks"][0], plain["masks"][0].flip(-1))
