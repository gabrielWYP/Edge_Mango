"""AP is computed over the full ranked prediction list, not the operating point.

Verified defect: `Trainer.evaluate` decoded at `conf_threshold` (0.25) before
computing AP, so the precision-recall curve stopped at the recall reached at
0.25 and any epoch whose scores all fell below it scored AP 0.0 for every
class. `evaluate` now decodes at `eval_conf_threshold` (0.001) for AP and
still reports precision/recall/F1 at `conf_threshold`.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import TrainingConfig
from src.training.loop import Trainer

DAMAGE = 1
IMAGE_SIZE = 64
LEVEL_SIDES = (8, 4, 2)  # student strides (8, 16, 32) at 64 px


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _raw_outputs(damage_score: float) -> dict:
    """Head outputs where exactly one anchor predicts damage at `damage_score`
    and every other class/anchor scores 1e-4 (below any eval threshold)."""
    preds, cls_preds = [], []
    for side in LEVEL_SIDES:
        cls = torch.full((1, 2, side, side), _logit(1e-4))
        reg = torch.zeros((1, 4, side, side))
        if side == LEVEL_SIDES[0]:
            cls[0, DAMAGE, 3, 3] = _logit(damage_score)
            reg[0, 2:, 3, 3] = math.log(16.0)
        cls_preds.append(cls)
        preds.append(torch.cat([cls, reg], dim=1))
    return {"preds": preds, "cls_preds": cls_preds}


class _FixedOutputModel(nn.Module):
    def __init__(self, output: dict):
        super().__init__()
        self.output = output

    def forward(self, rgb):
        return self.output


def _evaluate(damage_score: float, **config_kwargs) -> dict:
    config = TrainingConfig(
        model_type="student",
        num_classes=2,
        image_size=IMAGE_SIZE,
        precision="fp32",
        batch_size=1,
        effective_batch=1,
        **config_kwargs,
    )
    output = _raw_outputs(damage_score)

    # Ground truth = the predicted box itself, decoded with no threshold, so a
    # prediction that reaches the metric is a perfect true positive.
    gt_boxes, _, gt_labels = Trainer._decode_predictions_static(
        output["preds"], output["cls_preds"], 0, config, conf_threshold=0.0
    )
    keep = gt_labels == DAMAGE
    batch = {
        "rgb": torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "nir": torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE),
        "bboxes": [gt_boxes[keep][:1]],
        "labels": [gt_labels[keep][:1]],
    }

    trainer = Trainer.__new__(Trainer)
    trainer.model = _FixedOutputModel(output)
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer.val_loader = [batch]
    return trainer.evaluate()


class TestEvaluateUsesEvalThreshold:
    def test_low_score_true_positive_counts_toward_ap(self):
        metrics = _evaluate(damage_score=0.2)
        assert metrics["per_class_ap_50"][DAMAGE] == pytest.approx(1.0)

    def test_operating_point_still_uses_conf_threshold(self):
        # 0.2 < conf_threshold 0.25: the lesion is ranked (AP 1.0) but not
        # detected at the operating point.
        metrics = _evaluate(damage_score=0.2)
        assert metrics["per_class_recall"][DAMAGE] == pytest.approx(0.0)

    def test_old_behaviour_reproduced_when_thresholds_match(self):
        metrics = _evaluate(damage_score=0.2, eval_conf_threshold=0.25)
        assert metrics["per_class_ap_50"][DAMAGE] == pytest.approx(0.0)

    def test_above_operating_point_unchanged(self):
        metrics = _evaluate(damage_score=0.9)
        assert metrics["per_class_ap_50"][DAMAGE] == pytest.approx(1.0)
        assert metrics["per_class_recall"][DAMAGE] == pytest.approx(1.0)


class TestEvaluateReportsLocalizationMetrics:
    """`Trainer.evaluate` carries AP@0.3 and the impact-point metrics, with the same
    split between the full ranked list (AP) and the operating point (recall)."""

    def test_low_score_true_positive_counts_toward_both_aps(self):
        metrics = _evaluate(damage_score=0.2)
        assert metrics["per_class_ap_30"][DAMAGE] == pytest.approx(1.0)
        assert metrics["per_class_ap_center"][DAMAGE] == pytest.approx(1.0)

    def test_center_recall_uses_conf_threshold_like_the_other_operating_point_metrics(self):
        assert _evaluate(damage_score=0.2)["per_class_center_recall"][DAMAGE] == pytest.approx(0.0)
        assert _evaluate(damage_score=0.9)["per_class_center_recall"][DAMAGE] == pytest.approx(1.0)

    def test_every_class_is_reported(self):
        metrics = _evaluate(damage_score=0.9)
        for key in ("per_class_ap_30", "per_class_ap_center", "per_class_center_recall"):
            assert set(metrics[key]) == {0, 1}, key


class TestEvalConfThresholdConfig:
    def test_default(self):
        assert TrainingConfig().eval_conf_threshold == pytest.approx(0.001)

    def test_must_not_exceed_conf_threshold(self):
        with pytest.raises(ValueError, match="eval_conf_threshold"):
            TrainingConfig(eval_conf_threshold=0.5)

    def test_must_not_be_negative(self):
        with pytest.raises(ValueError, match="eval_conf_threshold"):
            TrainingConfig(eval_conf_threshold=-0.1)
