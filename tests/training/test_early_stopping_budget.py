"""Early stopping must be disableable, because on this split it measures noise.

The validation split is 9 images. A single detection moving in or out
swings mAP@0.5 by roughly ten points, so the per-epoch metric early stopping
watches is dominated by sampling noise rather than convergence. Measured on
the two runs that trained to completion:

    20260831T224505981163Z (batch 2)   best 0.4806 @ epoch 62, max gap 12
    batch 4 / num_workers 2            best 0.4695 @ epoch 76, max gap 13

`patience: 20` therefore cleared the observed worst case by 7 epochs, and
the student and KD configs shipped `patience: 15`, clearing it by 2. It has
already failed once: run `20260831T223558592803Z` spiked to 0.1673 at epoch
7, could not beat that bar for 20 epochs, and was killed at epoch 27 with 53
epochs of useful training left — producing a number that read as a quality
regression and was not one.

The mechanism buys nothing here. A full 90-epoch run takes 3.8 minutes at
2.5s/epoch, and `_save_checkpoint` already keeps the best-mAP checkpoint
regardless of how long training continues, so stopping early protects
against no overfitting the best-checkpoint rule does not already handle.

`patience <= 0` therefore disables early stopping outright. Task 4.2 compares
the distilled stage against the student baseline, so a stage that stops early
while its counterpart runs on would invalidate exactly that comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.loop import Trainer

# Every config the three-stage final pipeline runs.
PIPELINE_CONFIGS = [
    Path("configs/training_mango.yaml"),
    Path("configs/training_student.yaml"),
    Path("configs/kd_training.yaml"),
]


def _trainer(*, patience: int, counter: int) -> SimpleNamespace:
    """`_should_stop_early` reads only these two values."""
    return SimpleNamespace(
        config=SimpleNamespace(patience=patience),
        patience_counter=counter,
    )


class TestEarlyStoppingDecision:
    def test_patience_zero_never_stops(self):
        assert not Trainer._should_stop_early(_trainer(patience=0, counter=999))

    def test_negative_patience_never_stops(self):
        assert not Trainer._should_stop_early(_trainer(patience=-1, counter=999))

    def test_positive_patience_still_stops_at_the_threshold(self):
        assert Trainer._should_stop_early(_trainer(patience=20, counter=20))

    def test_positive_patience_does_not_stop_below_the_threshold(self):
        assert not Trainer._should_stop_early(_trainer(patience=20, counter=19))


class TestPipelineConfigsRunTheFullBudget:
    """A stage that stops early while its counterpart runs on breaks 4.2."""

    @pytest.mark.parametrize("config_path", PIPELINE_CONFIGS, ids=lambda p: p.name)
    def test_early_stopping_is_disabled(self, config_path):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert config.get("patience", 15) <= 0, (
            f"{config_path} enables early stopping (patience="
            f"{config.get('patience')}). On a 9-image validation split that "
            "stops on noise; the observed worst-case gap between genuine "
            "improvements was 13 epochs."
        )
