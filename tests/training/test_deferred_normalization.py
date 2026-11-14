"""Deferred (on-device) normalization, thread caps, and trainer wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.models.student.student_model import StudentModel
from src.training.config import TrainingConfig
from src.training.dataset import YOLODataset, build_dataloader, collate_fn, normalize_on_device
from src.training.loop import Trainer
from src.training.threads import cap_main_threads, limit_worker_threads

SIZE = 64
LABELS = "0 0.5 0.5 0.5 0.5\n1 0.5 0.5 0.1 0.1\n"
STEMS = {"train": ["00000", "00001", "00002"], "val": ["00003", "00004"]}
NIR_MEAN, NIR_STD = 0.3, 0.1


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("deferred_ds")
    rng = np.random.default_rng(0)
    (root / "rgb").mkdir()
    (root / "nir").mkdir()
    for split, stems in STEMS.items():
        (root / "labels" / split).mkdir(parents=True)
        for stem in stems:
            assert cv2.imwrite(str(root / "rgb" / f"mango_rgb_{stem}.jpg"), rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
            assert cv2.imwrite(str(root / "nir" / f"mango_nir_{stem}.jpg"), rng.integers(0, 256, (48, 64), dtype=np.uint8))
            (root / "labels" / split / f"mango_rgb_{stem}.txt").write_text(LABELS)
    return root


def _ds(root: Path, split: str, **kw) -> YOLODataset:
    ds = YOLODataset(
        rgb_dir=root / "rgb", nir_dir=root / "nir", labels_dir=root / "labels", split=split,
        image_size=SIZE, nir_mean=NIR_MEAN, nir_std=NIR_STD, nir_homography_path=None, **kw,
    )
    if split == "train":
        ds.spatial_transform.set_random_seed(0)
        ds.rgb_photometric_transform.set_random_seed(0)
    return ds


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("ablation", ["none", "nir_const", "rgb_const"])
def test_deferred_item_normalizes_to_the_eager_item(root, split, ablation) -> None:
    eager = _ds(root, split, modality_ablation=ablation)[0]
    item = _ds(root, split, modality_ablation=ablation, defer_normalization=True)[0]
    assert item["rgb"].shape == (3, SIZE, SIZE) and item["nir"].shape == (1, SIZE, SIZE)
    assert item["rgb"].dtype == (torch.float32 if ablation == "rgb_const" else torch.uint8)
    assert item["nir"].dtype == (torch.float32 if ablation == "nir_const" else torch.uint8)
    rgb, nir = normalize_on_device(
        item["rgb"].unsqueeze(0), item["nir"].unsqueeze(0), NIR_MEAN, NIR_STD
    )
    assert rgb.dtype == nir.dtype == torch.float32
    torch.testing.assert_close(rgb[0], eager["rgb"], atol=1e-5, rtol=0)
    torch.testing.assert_close(nir[0], eager["nir"], atol=1e-5, rtol=0)
    assert torch.equal(item["bboxes"], eager["bboxes"]) and torch.equal(item["labels"], eager["labels"])


def test_default_dataset_is_unchanged(root) -> None:
    assert _ds(root, "val")[0]["rgb"].dtype == torch.float32


def test_collated_deferred_batch_is_uint8(root) -> None:
    ds = _ds(root, "val", defer_normalization=True)
    batch = collate_fn([ds[0], ds[1]])
    assert batch["rgb"].dtype == torch.uint8 and batch["nir"].dtype == torch.uint8


# --- Trainer wiring ------------------------------------------------------------------------------


def _config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        model_type="student", image_size=SIZE, batch_size=2, effective_batch=2, num_workers=0,
        device="cpu", precision="fp32", output_dir=str(tmp_path / "out"),
        nir_mean=NIR_MEAN, nir_std=NIR_STD, student_in_channels=3,
    )


def _evaluate(root, tmp_path, defer: bool, ablation: str = "none") -> dict:
    torch.manual_seed(0)
    model = StudentModel(num_classes=2)
    loader = build_dataloader(
        _ds(root, "val", defer_normalization=defer, modality_ablation=ablation), batch_size=2,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )
    return Trainer(model, _config(tmp_path), None, loader).evaluate()


@pytest.mark.parametrize("ablation", ["none", "rgb_const"])
def test_evaluate_is_identical_with_and_without_deferral(root, tmp_path, ablation) -> None:
    a = _evaluate(root, tmp_path, defer=False, ablation=ablation)
    b = _evaluate(root, tmp_path, defer=True, ablation=ablation)
    assert a["map50"] == pytest.approx(b["map50"], abs=1e-6)
    assert a["map_50_95"] == pytest.approx(b["map_50_95"], abs=1e-6)
    assert a["per_class_counts"] == b["per_class_counts"]


def test_train_epoch_accepts_deferred_batches(root, tmp_path) -> None:
    torch.manual_seed(0)
    model = StudentModel(num_classes=2)
    loader = build_dataloader(
        _ds(root, "train", defer_normalization=True), batch_size=2, collate_fn=collate_fn,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    trainer = Trainer(model, _config(tmp_path), loader, loader)
    metrics = trainer._train_epoch(torch.optim.AdamW(model.parameters(), lr=1e-3), epoch=1, phase=1)
    assert np.isfinite(metrics["total_loss"])


# --- thread caps ---------------------------------------------------------------------------------


def test_limit_worker_threads_caps_torch_and_cv2() -> None:
    before_t, before_c = torch.get_num_threads(), cv2.getNumThreads()
    try:
        torch.set_num_threads(max(before_t, 2))
        cv2.setNumThreads(2)
        limit_worker_threads(0)
        assert torch.get_num_threads() == 1 and cv2.getNumThreads() == 1
    finally:
        torch.set_num_threads(before_t)
        cv2.setNumThreads(before_c)


def test_cap_main_threads_only_lowers() -> None:
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        cap_main_threads(2)
        assert torch.get_num_threads() == 2
        cap_main_threads(8)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)


def test_build_dataloader_composes_worker_init_fn(root) -> None:
    calls = []
    loader = build_dataloader(
        _ds(root, "val"), batch_size=1, collate_fn=collate_fn, num_workers=0, pin_memory=False,
        limit_threads=True, worker_init_fn=calls.append,
    )
    init = loader.worker_init_fn
    init(3)
    assert calls == [3]
