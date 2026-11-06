"""Operating points of the damage class: box matching, the PR curve and photo classification."""

from __future__ import annotations

import torch

from scripts import val_operating_points as vop
from src.evaluation.v4_metrics import EvaluatedPhoto


def _photo(hour, gt, boxes, scores) -> EvaluatedPhoto:
    return EvaluatedPhoto(
        stem="s", hour=hour, zona="", provenance="", face=("1", "1", "a"),
        gt_boxes=torch.tensor(gt, dtype=torch.float64).reshape(-1, 4),
        pred_boxes=torch.tensor(boxes, dtype=torch.float64).reshape(-1, 4),
        pred_scores=torch.tensor(scores, dtype=torch.float64),
    )


def test_a_label_matches_one_box_and_a_sano_box_is_a_false_positive() -> None:
    damaged = _photo(0, [[0, 0, 10, 10]], [[0, 0, 10, 10], [0, 0, 10, 9]], [0.9, 0.6])
    sano = _photo(None, [], [[0, 0, 5, 5]], [0.7])
    ranked, n_gt = vop.ranked_boxes([damaged, sano])
    assert n_gt == 1 and ranked == [(0.9, True), (0.7, False), (0.6, False)]
    _, precision, recall = vop.pr_curve(ranked, n_gt)
    assert vop.all_point_ap(precision, recall) == 1.0
    point = vop.box_point(ranked, n_gt, 2, 0.65)
    assert (point["tp"], point["fp"], point["fn"], point["fp_per_photo"]) == (1, 1, 0, 0.5)


def test_a_photo_is_called_damaged_by_its_highest_damage_score() -> None:
    photos = [
        _photo(0, [[0, 0, 1, 1]], [[0, 0, 1, 1]], [0.8]),
        _photo(2, [[0, 0, 1, 1]], [], []),
        _photo(None, [], [[0, 0, 1, 1]], [0.3]),
        _photo(None, [], [], []),
    ]
    point = vop.photo_point(photos, 0.25)
    assert (point["tp"], point["fn"], point["fp"], point["tn"]) == (1, 1, 1, 1)
    assert point["balanced_accuracy"] == 0.5
    assert vop.photo_point(photos, 0.5)["specificity"] == 1.0
