"""Calibration inputs, complexity counting and the INT8 guard: no litert needed (the conversions are in
test_quantize_tflite.py)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.export.quantize import calibration_inputs, select_calibration_samples
from src.export.reference import preprocess_roi_bgr

SIZE = 64
SEED = 3


def _make_dataset(root: Path, n_train: int = 6, n_val: int = 2) -> tuple[Path, Path]:
    """RGB images plus `<labels>/{train,val}/<stem>.txt` (a mango box and a damage box each)."""
    rgb_dir, labels_dir = root / "rgb", root / "labels"
    rng = np.random.default_rng(0)
    for split, count in (("train", n_train), ("val", n_val)):
        (labels_dir / split).mkdir(parents=True)
        rgb_dir.mkdir(exist_ok=True)
        for i in range(count):
            stem = f"mango_rgb_{split}{i:02d}"
            cv2.imwrite(str(rgb_dir / f"{stem}.jpg"), rng.integers(0, 255, (72, 128, 3), dtype=np.uint8))
            (labels_dir / split / f"{stem}.txt").write_text("0 0.5 0.5 0.4 0.6\n1 0.52 0.5 0.1 0.1\n")
    return rgb_dir, labels_dir


# --- calibration inputs (no litert needed) -----------------------------------------------------------


def test_calibration_samples_come_from_the_requested_split_and_are_deterministic(tmp_path) -> None:
    rgb_dir, labels_dir = _make_dataset(tmp_path)
    first = select_calibration_samples(rgb_dir, labels_dir, split="train", n=4, seed=1)
    again = select_calibration_samples(rgb_dir, labels_dir, split="train", n=4, seed=1)
    assert first == again and len(first) == 4 and len(set(first)) == 4
    assert all("train" in label.parent.name and image.exists() for image, label in first)
    # Asking for more than the split holds returns the whole split, never val images.
    everything = select_calibration_samples(rgb_dir, labels_dir, split="train", n=200)
    assert len(everything) == 6 and not any("val" in image.name for image, _ in everything)


def test_calibration_inputs_are_built_like_training_inputs(tmp_path) -> None:
    rgb_dir, labels_dir = _make_dataset(tmp_path)
    samples = select_calibration_samples(rgb_dir, labels_dir, split="train", n=2)
    batches = list(calibration_inputs(samples, image_size=SIZE, pad_value=114, roi_margin=0.10))
    assert len(batches) == 2
    for array, (image_path, label_path) in zip(batches, samples):
        assert array.shape == (1, SIZE, SIZE, 3) and array.dtype == np.float32
        expected, _ = preprocess_roi_bgr(cv2.imread(str(image_path)), label_path, SIZE, 114, 0.10)
        np.testing.assert_array_equal(array, expected.permute(0, 2, 3, 1).numpy())


def test_an_empty_split_is_an_error(tmp_path) -> None:
    rgb_dir, labels_dir = _make_dataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        select_calibration_samples(rgb_dir, labels_dir, split="test", n=4)


# --- complexity (no litert needed) -------------------------------------------------------------------


def test_complexity_counts_parameters_and_flops() -> None:
    from src.export.quantize import model_complexity
    from src.export.student_export import StudentExportWrapper
    from src.models.student.student_model import StudentModel

    student = StudentModel(variant="nano").eval()
    small = model_complexity(StudentExportWrapper(student), image_size=64)
    big = model_complexity(StudentExportWrapper(student), image_size=128)
    assert small["parameters"] == sum(p.numel() for p in student.parameters())
    assert small["flops"] > 1e6 and big["flops"] == pytest.approx(4 * small["flops"], rel=0.05)
    assert small["flops_tool"].startswith("torch.utils.flop_counter")


def test_int8_without_calibration_data_is_rejected() -> None:
    from src.export.student_export import StudentExportWrapper, export_tflite
    from src.models.student.student_model import StudentModel

    wrapper = StudentExportWrapper(StudentModel(variant="nano").eval())
    with pytest.raises(ValueError, match="calibration_inputs"):
        export_tflite(wrapper, "unused.tflite", image_size=SIZE, precision="int8")
