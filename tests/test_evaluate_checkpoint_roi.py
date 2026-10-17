"""Evaluation must run on the crop the checkpoint was trained with, deterministically.

`scripts/evaluate_checkpoint.py` used to take the ROI crop settings from the
`--config` YAML alone, so scoring a crop-trained checkpoint with a full-frame
config (or vice versa) silently produced metrics for the wrong input
distribution. It also let `YOLODataset` jitter the crop margin on
`--split train`. These tests pin the fix:

* the checkpoint's recorded `roi_crop` / `roi_crop_margin` must agree with the
  effective eval config, otherwise evaluation fails loudly;
* the eval dataset is always built with `roi_crop_jitter=0.0`;
* the effective crop settings are written next to the metrics.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import evaluate_checkpoint as ec
from src.training.config import TrainingConfig
from src.training.dataset import YOLODataset


def _config(**overrides) -> TrainingConfig:
    return TrainingConfig(model_type="student", **overrides)


class TestCheckRoiCropBinding:
    def test_full_frame_checkpoint_and_full_frame_config_pass(self):
        ec._check_roi_crop_binding(
            "ckpt.pt", {"roi_crop": False, "roi_crop_margin": 0.10}, _config(roi_crop=False)
        )

    def test_crop_checkpoint_and_crop_config_with_same_margin_pass(self):
        ec._check_roi_crop_binding(
            "ckpt.pt",
            {"roi_crop": True, "roi_crop_margin": 0.15},
            _config(roi_crop=True, roi_crop_margin=0.15),
        )

    def test_crop_checkpoint_with_full_frame_config_fails(self):
        with pytest.raises(ValueError, match=r"roi_crop=True.*roi_crop=False") as excinfo:
            ec._check_roi_crop_binding(
                "ckpt.pt",
                {"roi_crop": True, "roi_crop_margin": 0.10},
                _config(roi_crop=False),
            )
        message = str(excinfo.value)
        assert "ckpt.pt" in message
        assert "twostream_roi_crop.yaml" in message
        assert "roi_crop_margin=0.1" in message

    def test_full_frame_checkpoint_with_crop_config_fails(self):
        with pytest.raises(ValueError, match=r"roi_crop=False.*roi_crop=True") as excinfo:
            ec._check_roi_crop_binding(
                "ckpt.pt", {"roi_crop": False}, _config(roi_crop=True)
            )
        message = str(excinfo.value)
        assert "ckpt.pt" in message
        assert "twostream.yaml" in message

    def test_margin_mismatch_fails_when_the_crop_is_on(self):
        with pytest.raises(ValueError, match="roi_crop_margin") as excinfo:
            ec._check_roi_crop_binding(
                "ckpt.pt",
                {"roi_crop": True, "roi_crop_margin": 0.10},
                _config(roi_crop=True, roi_crop_margin=0.25),
            )
        message = str(excinfo.value)
        assert "0.1" in message and "0.25" in message

    def test_margin_is_ignored_when_both_sides_are_full_frame(self):
        ec._check_roi_crop_binding(
            "ckpt.pt",
            {"roi_crop": False, "roi_crop_margin": 0.30},
            _config(roi_crop=False, roi_crop_margin=0.05),
        )

    def test_missing_margin_key_counts_as_the_default_margin(self):
        ec._check_roi_crop_binding(
            "ckpt.pt", {"roi_crop": True}, _config(roi_crop=True, roi_crop_margin=0.10)
        )
        with pytest.raises(ValueError, match="roi_crop_margin"):
            ec._check_roi_crop_binding(
                "ckpt.pt", {"roi_crop": True}, _config(roi_crop=True, roi_crop_margin=0.15)
            )

    def test_config_without_roi_keys_counts_as_full_frame(self):
        legacy = {"head_strides": [8, 16, 32], "fusion_mode": "cross_attention"}
        ec._check_roi_crop_binding("old.pt", legacy, _config(roi_crop=False))
        with pytest.raises(ValueError, match=r"roi_crop=False.*roi_crop=True"):
            ec._check_roi_crop_binding("old.pt", legacy, _config(roi_crop=True))

    @pytest.mark.parametrize("saved", [None, {}])
    def test_checkpoint_without_any_config_warns_and_proceeds(self, saved, caplog):
        with caplog.at_level(logging.WARNING, logger=ec.logger.name):
            ec._check_roi_crop_binding("raw.pt", saved, _config(roi_crop=True))
        assert any(
            "raw.pt" in record.getMessage() and "roi_crop" in record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_matching_checkpoint_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger=ec.logger.name):
            ec._check_roi_crop_binding("ckpt.pt", {"roi_crop": False}, _config(roi_crop=False))
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestLoadModelEnforcesTheBinding:
    """`_load_model` already holds the checkpoint, so it runs the check before
    any model is built or any weight is loaded."""

    def _save(self, path: Path, saved_config: dict | None) -> Path:
        payload: dict = {"model_state_dict": {"bogus.weight": torch.zeros(1)}}
        if saved_config is not None:
            payload["config"] = saved_config
        torch.save(payload, path)
        return path

    def test_mismatch_raises_before_load_state_dict(self, tmp_path):
        path = self._save(tmp_path / "crop.pt", {"roi_crop": True, "roi_crop_margin": 0.10})
        with pytest.raises(ValueError, match="roi_crop"):
            ec._load_model("student", str(path), _config(roi_crop=False), torch.device("cpu"))

    def test_matching_settings_get_past_the_check(self, tmp_path):
        path = self._save(tmp_path / "crop.pt", {"roi_crop": True, "roi_crop_margin": 0.10})
        # The bogus state_dict then fails strict loading: proof the roi check passed.
        with pytest.raises(RuntimeError):
            ec._load_model("student", str(path), _config(roi_crop=True), torch.device("cpu"))

    def test_checkpoint_without_config_gets_past_the_check(self, tmp_path):
        path = self._save(tmp_path / "raw.pt", None)
        with pytest.raises(RuntimeError):
            ec._load_model("student", str(path), _config(roi_crop=True), torch.device("cpu"))


class _FakeDataset:
    """Records the kwargs `YOLODataset` was built with."""

    instances: list["_FakeDataset"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).instances.append(self)

    def __len__(self) -> int:
        return 1


@pytest.fixture
def fake_dataset(monkeypatch):
    _FakeDataset.instances = []
    monkeypatch.setattr(ec, "YOLODataset", _FakeDataset)
    return _FakeDataset


class TestEvalDatasetIsDeterministic:
    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_jitter_is_forced_to_zero_for_every_split(self, fake_dataset, split):
        config = _config(roi_crop=True, roi_crop_margin=0.2, roi_crop_jitter=0.5)
        ec._build_eval_dataset(config, split)

        (dataset,) = fake_dataset.instances
        assert dataset.kwargs["roi_crop_jitter"] == 0.0
        assert dataset.kwargs["roi_crop"] is True
        assert dataset.kwargs["roi_crop_margin"] == pytest.approx(0.2)
        assert dataset.kwargs["split"] == split

    def test_the_real_train_split_dataset_draws_no_jitter(self, tmp_path):
        for sub in ("rgb", "nir", "labels/train"):
            (tmp_path / sub).mkdir(parents=True)
        cv2.imwrite(str(tmp_path / "rgb" / "m_rgb_00001.jpg"), np.full((48, 64, 3), 60, np.uint8))
        cv2.imwrite(str(tmp_path / "nir" / "m_nir_00001.jpg"), np.full((48, 64), 20, np.uint8))
        (tmp_path / "labels" / "train" / "m_rgb_00001.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n1 0.5 0.5 0.1 0.1\n"
        )
        config = _config(
            roi_crop=True,
            roi_crop_jitter=0.5,
            rgb_dir=str(tmp_path / "rgb"),
            nir_dir=str(tmp_path / "nir"),
            labels_dir=str(tmp_path / "labels"),
            split_manifest=None,
            nir_homography_path=None,
            image_size=64,
        )

        dataset = ec._build_eval_dataset(config, "train")

        assert isinstance(dataset, YOLODataset)
        assert dataset.roi_crop is True
        assert dataset.roi_crop_jitter == 0.0
        assert dataset._roi_crop_jitter_range() == 0.0


class TestRoiCropProvenance:
    def test_records_the_effective_settings(self):
        config = _config(roi_crop=True, roi_crop_margin=0.2, roi_crop_jitter=0.5)
        assert ec._roi_crop_provenance(config) == {"roi_crop": True, "roi_crop_margin": 0.2}

    def test_full_frame_config_is_recorded_too(self):
        assert ec._roi_crop_provenance(_config()) == {"roi_crop": False, "roi_crop_margin": 0.10}


class _FakeTrainer:
    def __init__(self, **kwargs):
        pass

    def evaluate(self) -> dict:
        return {
            "map50": 0.5,
            "map_50_95": 0.25,
            "per_class_ap_50": {0: 0.9, 1: 0.1},
            "per_class_precision": {},
            "per_class_recall": {},
            "per_class_counts": {},
        }


class TestMainWiring:
    @pytest.fixture
    def patched(self, monkeypatch, tmp_path, fake_dataset):
        config = _config(roi_crop=True, roi_crop_margin=0.2, roi_crop_jitter=0.5)
        monkeypatch.setattr(
            TrainingConfig, "from_yaml", classmethod(lambda cls, path: config)
        )
        monkeypatch.setattr(ec, "_load_model", lambda *a, **k: SimpleNamespace(head_strides=[8]))
        monkeypatch.setattr(ec, "build_dataloader", lambda *a, **k: [])
        monkeypatch.setattr(ec, "Trainer", _FakeTrainer)
        return tmp_path

    def _argv(self, monkeypatch, out: Path, *extra: str):
        monkeypatch.setattr(
            sys,
            "argv",
            ["evaluate_checkpoint.py", "--config", "x.yaml", "--checkpoint", "c.pt",
             "--split", "train", "--device", "cpu", "--output-dir", str(out), *extra],
        )

    def test_metrics_json_records_crop_settings_and_jitter_is_zero(
        self, patched, monkeypatch, fake_dataset
    ):
        self._argv(monkeypatch, patched)
        assert ec.main() == 0

        metrics = json.loads((patched / "metrics.json").read_text())
        assert metrics["roi_crop"] is True
        assert metrics["roi_crop_margin"] == pytest.approx(0.2)
        assert metrics["map50"] == pytest.approx(0.5)
        (dataset,) = fake_dataset.instances
        assert dataset.kwargs["roi_crop_jitter"] == 0.0

    def test_assigner_stats_summary_records_crop_settings(self, patched, monkeypatch):
        monkeypatch.setattr(
            ec, "_run_assigner_stats",
            lambda *a, **k: {"per_class_level_counts": {}, "summary": {}},
        )
        self._argv(monkeypatch, patched, "--assigner-stats")
        assert ec.main() == 0

        summary = json.loads((patched / "assigner_stats_summary.json").read_text())
        assert summary["roi_crop"] is True
        assert summary["roi_crop_margin"] == pytest.approx(0.2)

    def test_binding_mismatch_aborts_before_the_dataset_is_built(
        self, monkeypatch, tmp_path, fake_dataset
    ):
        monkeypatch.setattr(
            TrainingConfig, "from_yaml", classmethod(lambda cls, path: _config(roi_crop=False))
        )
        checkpoint = tmp_path / "crop.pt"
        torch.save(
            {
                "model_state_dict": {"bogus.weight": torch.zeros(1)},
                "config": {"roi_crop": True, "roi_crop_margin": 0.10},
            },
            checkpoint,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["evaluate_checkpoint.py", "--config", "x.yaml", "--checkpoint", str(checkpoint),
             "--device", "cpu", "--output-dir", str(tmp_path / "out")],
        )

        with pytest.raises(ValueError, match="roi_crop"):
            ec.main()
        assert fake_dataset.instances == []
