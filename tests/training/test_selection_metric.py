"""Checkpoint selection and early stopping follow `config.selection_metric`.

Verified defect: `best_model.pt` was always selected on the unweighted
two-class mAP@0.5, where the mango AP (~0.9) dominates. The epoch with the
best damage AP could therefore go unsaved (twostream seed 42: val damage
AP50 0.2077 at epoch 35 was never written). `selection_metric="damage_ap50"`
selects on the damage-class AP alone; "map50" keeps the old behavior.

`_train_epoch` and `_validate` are replaced with scripted stubs, so these
tests drive the real `_train_phase` selection logic without data or a GPU.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import DAMAGE_CLASS_ID, TrainingConfig
from src.training.loop import Trainer


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))

    def freeze_backbone(self, freeze_stages: int) -> None:
        pass


def _val(
    map50: float,
    damage_ap50: float,
    damage_ap30: float | None = None,
    damage_ap_center: float | None = None,
) -> dict:
    metrics = {
        "map50": map50,
        "map_50_95": 0.0,
        "per_class_ap_50": {0: 2 * map50 - damage_ap50, DAMAGE_CLASS_ID: damage_ap50},
    }
    if damage_ap30 is not None:
        metrics["per_class_ap_30"] = {0: 0.9, DAMAGE_CLASS_ID: damage_ap30}
    if damage_ap_center is not None:
        metrics["per_class_ap_center"] = {0: 0.9, DAMAGE_CLASS_ID: damage_ap_center}
    return metrics


_TRAIN_METRICS = {
    "total_loss": 1.0,
    "cls_loss": 0.5,
    "box_loss": 0.5,
    "oom_skipped": 0,
    "nan_skipped": 0,
    "steps_taken": 1,
}


def _run(tmp_path, selection_metric: str, val_sequence: list[dict], patience: int = 50):
    """Run one phase over `val_sequence`; return (trainer, epochs saved as best)."""
    config = TrainingConfig(
        model_type="student",
        batch_size=1,
        effective_batch=1,
        num_workers=0,
        precision="fp32",
        warmup_epochs=1,
        patience=patience,
        save_interval=1000,
        selection_metric=selection_metric,
        output_dir=str(tmp_path),
    )
    trainer = Trainer(model=_StubModel(), config=config, train_loader=[], val_loader=[])
    trainer.writer = None

    scripted = iter(val_sequence)
    saved: list[int] = []
    trainer._train_epoch = lambda optimizer, epoch, phase: dict(_TRAIN_METRICS)
    trainer._validate = lambda epoch, phase: next(scripted)
    trainer._save_checkpoint = (
        lambda epoch, phase, metrics, filename, train_metrics=None:
        saved.append(epoch) if filename == "best_model.pt" else None
    )

    trainer._train_phase(phase=1, epochs=len(val_sequence), lr=1e-3)
    return trainer, saved


# Epoch 2 has the best mAP; epoch 3 has the best damage AP.
_SEQUENCE = [_val(0.40, 0.05), _val(0.55, 0.08), _val(0.50, 0.20), _val(0.45, 0.10)]


class TestSelectionMetric:
    def test_map50_selects_best_mean(self, tmp_path):
        trainer, saved = _run(tmp_path, "map50", _SEQUENCE)
        assert saved == [1, 2]
        assert trainer.best_score == pytest.approx(0.55)
        assert trainer.best_map50 == pytest.approx(0.55)

    def test_damage_ap50_selects_best_damage_epoch(self, tmp_path):
        trainer, saved = _run(tmp_path, "damage_ap50", _SEQUENCE)
        assert saved == [1, 2, 3]
        assert trainer.best_score == pytest.approx(0.20)
        # best_map50 reports the mAP of the checkpoint that was saved.
        assert trainer.best_map50 == pytest.approx(0.50)

    def test_first_epoch_saves_at_zero_and_ties_keep_earlier(self, tmp_path):
        _, saved = _run(tmp_path, "damage_ap50", [_val(0.3, 0.0), _val(0.4, 0.0)])
        assert saved == [1]

    def test_early_stopping_counts_the_selection_metric(self, tmp_path):
        # mAP keeps improving but damage AP never does after epoch 1.
        sequence = [_val(0.30 + 0.05 * i, 0.10 if i == 0 else 0.05) for i in range(5)]
        _, saved = _run(tmp_path, "damage_ap50", sequence, patience=2)
        assert saved == [1]


# Damage AP50 peaks at epoch 2, the localization-tolerant metrics at epoch 3 / 4.
_LOCALIZATION_SEQUENCE = [
    _val(0.40, 0.05, damage_ap30=0.10, damage_ap_center=0.20),
    _val(0.45, 0.20, damage_ap30=0.30, damage_ap_center=0.35),
    _val(0.42, 0.10, damage_ap30=0.50, damage_ap_center=0.40),
    _val(0.41, 0.08, damage_ap30=0.45, damage_ap_center=0.70),
]


class TestLocalizationTolerantSelection:
    def test_damage_ap30_selects_the_best_ap30_epoch(self, tmp_path):
        trainer, saved = _run(tmp_path, "damage_ap30", _LOCALIZATION_SEQUENCE)
        assert saved == [1, 2, 3]
        assert trainer.best_score == pytest.approx(0.50)
        # best_map50 still reports the mAP of the checkpoint that was saved.
        assert trainer.best_map50 == pytest.approx(0.42)

    def test_damage_ap_center_selects_the_best_impact_point_epoch(self, tmp_path):
        trainer, saved = _run(tmp_path, "damage_ap_center", _LOCALIZATION_SEQUENCE)
        assert saved == [1, 2, 3, 4]
        assert trainer.best_score == pytest.approx(0.70)
        assert trainer.best_map50 == pytest.approx(0.41)

    def test_damage_ap50_ignores_the_new_keys(self, tmp_path):
        trainer, saved = _run(tmp_path, "damage_ap50", _LOCALIZATION_SEQUENCE)
        assert saved == [1, 2]
        assert trainer.best_score == pytest.approx(0.20)

    @pytest.mark.parametrize("metric", ["damage_ap30", "damage_ap_center"])
    def test_validation_metrics_without_the_key_fail_loudly(self, tmp_path, metric):
        # A silent 0.0 would save epoch 1 forever and never early-stop correctly.
        with pytest.raises(KeyError, match=metric):
            _run(tmp_path, metric, [_val(0.4, 0.1)])


class TestHistoryRecordsLocalizationMetrics:
    def test_the_training_loop_forwards_them_to_the_history(self, tmp_path):
        sequence = [
            {
                **_val(0.4, 0.1, damage_ap30=0.3, damage_ap_center=0.5),
                "per_class_center_recall": {0: 1.0, DAMAGE_CLASS_ID: 0.6},
            },
            _val(0.5, 0.2, damage_ap30=0.4, damage_ap_center=0.6),
        ]
        trainer, _ = _run(tmp_path, "map50", sequence)
        history = trainer.loss_history
        assert [h[DAMAGE_CLASS_ID] for h in history.per_class_ap_30] == [0.3, 0.4]
        assert [h[DAMAGE_CLASS_ID] for h in history.per_class_ap_center] == [0.5, 0.6]
        assert history.per_class_center_recall == [{0: 1.0, DAMAGE_CLASS_ID: 0.6}, {}]

    def test_validation_metrics_without_them_leave_the_history_empty(self, tmp_path):
        trainer, _ = _run(tmp_path, "map50", [_val(0.4, 0.1)])
        assert trainer.loss_history.per_class_ap_30 == [{}]
        assert trainer.loss_history.per_class_ap_center == [{}]


class TestSelectionMetricConfig:
    def test_default_is_map50(self):
        assert TrainingConfig().selection_metric == "map50"

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="selection_metric"):
            TrainingConfig(selection_metric="damage_ap")

    @pytest.mark.parametrize("metric", ["map50", "damage_ap50", "damage_ap30", "damage_ap_center"])
    def test_known_metrics_are_accepted(self, metric):
        assert TrainingConfig(selection_metric=metric).selection_metric == metric

    @pytest.mark.parametrize("metric", ["damage_ap_30", "damage_center", "ap30", "", "DAMAGE_AP30"])
    def test_close_but_invalid_metrics_are_still_rejected(self, metric):
        with pytest.raises(ValueError, match="selection_metric"):
            TrainingConfig(selection_metric=metric)
