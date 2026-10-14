"""E2E test: 2-phase training loop on synthetic data."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.master.master_model import MasterModel
from src.training.config import TrainingConfig
from src.training.dataset import YOLODataset, collate_fn
from src.training.loop import Trainer


def _create_synthetic_dataset(tmp_dir: Path, split: str, n_images: int = 2):
    """Create a minimal synthetic dataset for testing.

    Creates RGB and NIR images + YOLO label files.
    """
    rgb_dir = tmp_dir / "rgb"
    nir_dir = tmp_dir / "nir"
    labels_dir = tmp_dir / "labels" / split

    rgb_dir.mkdir(parents=True, exist_ok=True)
    nir_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_images):
        # Create small synthetic images (64x64 for speed)
        rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        nir = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

        import cv2
        cv2.imwrite(str(rgb_dir / f"mango_rgb_{i:04d}.jpg"), rgb)
        cv2.imwrite(str(nir_dir / f"mango_nir_{i:04d}.jpg"), nir)

        # Create YOLO label file
        with open(labels_dir / f"mango_rgb_{i:04d}.txt", "w") as f:
            f.write(f"0 0.5 0.5 0.3 0.3\n")
            f.write(f"1 0.2 0.2 0.1 0.1\n")


class TestE2ETraining:
    """End-to-end training test on synthetic data."""

    def test_two_phase_loop(self, tmp_path):
        """2-phase loop on 2 synthetic images, 2 epochs per phase.

        Verifies:
        - Training completes without crashing
        - Checkpoint is saved
        - Loss values are finite
        """
        # Create synthetic data
        _create_synthetic_dataset(tmp_path, "train", n_images=2)
        _create_synthetic_dataset(tmp_path, "val", n_images=2)

        output_dir = tmp_path / "output"

        config = TrainingConfig(
            rgb_dir=str(tmp_path / "rgb"),
            nir_dir=str(tmp_path / "nir"),
            labels_dir=str(tmp_path / "labels"),
            output_dir=str(output_dir),
            backbone_variant="tiny",
            num_classes=2,
            image_size=64,  # small for speed
            batch_size=2,
            effective_batch=2,
            # This test specifically exercises the legacy two-phase
            # schedule; fusion-redesign D-4 makes "end_to_end" the default
            # for model_type="master", so this must opt back in explicitly.
            schedule="two_phase",
            epochs_phase1=2,
            epochs_phase2=2,
            lr_phase1=1e-3,
            lr_phase2=1e-4,
            warmup_epochs=1,
            patience=10,
            precision="fp32",  # CPU test
            num_workers=0,
        )

        # Create datasets
        train_dataset = YOLODataset(
            rgb_dir=config.rgb_dir,
            nir_dir=config.nir_dir,
            labels_dir=config.labels_dir,
            split="train",
            image_size=config.image_size,
            nir_homography_path=None,
        )
        val_dataset = YOLODataset(
            rgb_dir=config.rgb_dir,
            nir_dir=config.nir_dir,
            labels_dir=config.labels_dir,
            split="val",
            image_size=config.image_size,
            nir_homography_path=None,
        )

        assert len(train_dataset) == 2
        assert len(val_dataset) == 2

        # DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Model
        model = MasterModel(
            num_classes=2,
            pretrained_backbone=False,  # no download for tests
            backbone_variant="tiny",
        )

        # Train
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        results = trainer.fit()

        # Assertions
        assert "best_map50" in results
        assert "loss_history" in results
        assert len(results["loss_history"]["total_loss"]) > 0

        # All losses should be finite
        for loss_val in results["loss_history"]["total_loss"]:
            assert np.isfinite(loss_val), f"Non-finite loss: {loss_val}"

        # Checkpoint should be saved
        best_ckpt = output_dir / "best_model.pt"
        assert best_ckpt.exists(), "Best checkpoint should be saved"

    def test_config_yaml_roundtrip(self, tmp_path):
        """Config should survive YAML save/load roundtrip."""
        config = TrainingConfig(
            backbone_variant="tiny",
            batch_size=4,
            epochs_phase1=10,
        )

        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        loaded = TrainingConfig.from_yaml(yaml_path)
        assert loaded.backbone_variant == "tiny"
        assert loaded.batch_size == 4
        assert loaded.epochs_phase1 == 10
