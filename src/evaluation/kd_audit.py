"""Per-level measures of what a KD pair transfers, split by image region.

Pure helpers for `scripts/kd_feature_audit.py`: which regions drive the
imitation error, how much a teacher feature depends on the NIR input, and how
well a linear probe separates damage cells from healthy fruit cells.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def region_sums(
    per_cell: torch.Tensor, masks: dict[str, torch.Tensor]
) -> dict[str, tuple[float, int]]:
    """Return (sum of `per_cell`, cell count) for each region of a (N, H, W) map."""
    return {
        region: (per_cell[mask].sum().item(), int(mask.sum().item()))
        for region, mask in masks.items()
    }


def collect_probe_cells(
    feat: torch.Tensor,
    masks: dict[str, torch.Tensor],
    max_negatives_per_image: int,
    generator: torch.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cells, is_damage) from a (N, C, H, W) map.

    Keeps every damage cell and samples healthy fruit cells per image, since the
    question is whether a level tells damage from the skin around it.
    """
    positives, negatives = [], []
    for i in range(feat.shape[0]):
        cells = feat[i].permute(1, 2, 0)
        positives.append(cells[masks["damage"][i]])
        fruit = cells[masks["fruit"][i]]
        keep = torch.randperm(fruit.shape[0], generator=generator)[:max_negatives_per_image]
        negatives.append(fruit[keep.to(fruit.device)])
    pos = torch.cat(positives).float().cpu().numpy()
    neg = torch.cat(negatives).float().cpu().numpy()
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return np.concatenate([pos, neg]), y


def probe_auroc(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray
) -> float:
    """AUROC on the test cells of a logistic probe fitted on the train cells."""
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.01, max_iter=500, class_weight="balanced"),
    )
    probe.fit(train_x, train_y)
    return float(roc_auc_score(test_y, probe.decision_function(test_x)))
