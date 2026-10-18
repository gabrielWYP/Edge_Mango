"""Localization-tolerant detection metrics: AP@0.3 and the impact-point criterion.

Damage labels come from induced impacts (a 66 g steel ball dropped from 50 cm,
labeled from the known impact location), so the GT box extent is uncertain:
damage AP at IoU 0.3 is 2.3-5.3x AP@0.5. Two extra per-class metrics separate
"found the damage" from "drew the same box":

* `per_class_ap_30`: the existing AP, same matching code, IoU threshold 0.3.
* `per_class_ap_center` / `per_class_center_recall`: a prediction matches a GT
  when the prediction box CONTAINS the GT box center (the impact point).
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from src.training.metrics import LossHistory, compute_map, generate_training_curves

DAMAGE = 1


def _t(rows):
    """(N, 4) float boxes; `_t([])` is the empty (0, 4) tensor."""
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 4)


def _one_image(pred_rows, pred_scores, pred_labels, gt_rows, gt_labels):
    """Wrap a single image into the list-per-image arguments of `compute_map`."""
    return dict(
        pred_boxes=[_t(pred_rows)],
        pred_scores=[torch.tensor(pred_scores, dtype=torch.float32)],
        pred_labels=[torch.tensor(pred_labels, dtype=torch.long)],
        gt_boxes=[_t(gt_rows)],
        gt_labels=[torch.tensor(gt_labels, dtype=torch.long)],
    )


def _xyxy_to_cxcywh(x1, y1, x2, y2):
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


# --------------------------------------------------------------------------- #
# AP@0.3
# --------------------------------------------------------------------------- #


class TestApAt30:
    def test_equals_ap50_when_every_match_has_iou_at_least_half(self) -> None:
        gt = [[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]
        # Slightly shifted predictions: IoU well above 0.5.
        pred = [[0.31, 0.3, 0.2, 0.2], [0.7, 0.71, 0.2, 0.2]]
        result = compute_map(
            **_one_image(pred, [0.9, 0.8], [DAMAGE, DAMAGE], gt, [DAMAGE, DAMAGE]), num_classes=2
        )
        assert result["per_class_ap_50"][DAMAGE] == pytest.approx(1.0)
        assert result["per_class_ap_30"][DAMAGE] == pytest.approx(result["per_class_ap_50"][DAMAGE])

    def test_an_iou_of_0_4_counts_at_0_3_but_not_at_0_5(self) -> None:
        gt = [_xyxy_to_cxcywh(0.0, 0.0, 0.2, 0.2)]
        # Two 0.2 x 0.2 boxes, the prediction shifted right by dx: the intersection is
        # (0.2 - dx) * 0.2 and IoU = inter / (0.08 - inter). Solve for IoU = 0.4.
        inter = 0.4 * 0.08 / 1.4
        dx = 0.2 - inter / 0.2
        pred = [_xyxy_to_cxcywh(dx, 0.0, dx + 0.2, 0.2)]
        result = compute_map(
            **_one_image(pred, [0.9], [DAMAGE], gt, [DAMAGE]), num_classes=2
        )
        assert result["per_class_ap_50"][DAMAGE] == pytest.approx(0.0)
        assert result["per_class_ap_30"][DAMAGE] == pytest.approx(1.0)

    def test_an_iou_below_0_3_counts_nowhere(self) -> None:
        gt = [_xyxy_to_cxcywh(0.0, 0.0, 0.2, 0.2)]
        pred = [_xyxy_to_cxcywh(0.15, 0.0, 0.35, 0.2)]  # IoU = 0.05/0.35 = 1/7
        result = compute_map(**_one_image(pred, [0.9], [DAMAGE], gt, [DAMAGE]), num_classes=2)
        assert result["per_class_ap_30"][DAMAGE] == pytest.approx(0.0)

    def test_reports_every_class(self) -> None:
        result = compute_map(
            **_one_image([[0.5, 0.5, 0.2, 0.2]], [0.9], [0], [[0.5, 0.5, 0.2, 0.2]], [0]),
            num_classes=2,
        )
        assert set(result["per_class_ap_30"]) == {0, 1}
        assert result["per_class_ap_30"][0] == pytest.approx(1.0)
        assert result["per_class_ap_30"][1] == pytest.approx(0.0)  # no GT


# --------------------------------------------------------------------------- #
# Impact-point (center containment) criterion
# --------------------------------------------------------------------------- #


class TestCenterCriterion:
    def test_a_box_containing_the_gt_center_matches_whatever_its_iou(self) -> None:
        gt = [_xyxy_to_cxcywh(0.40, 0.40, 0.50, 0.50)]  # small GT, center (0.45, 0.45)
        pred = [_xyxy_to_cxcywh(0.10, 0.10, 0.90, 0.90)]  # huge box: IoU 0.0156
        result = compute_map(**_one_image(pred, [0.9], [DAMAGE], gt, [DAMAGE]), num_classes=2)
        assert result["per_class_ap_50"][DAMAGE] == pytest.approx(0.0)
        assert result["per_class_ap_center"][DAMAGE] == pytest.approx(1.0)
        assert result["per_class_center_recall"][DAMAGE] == pytest.approx(1.0)

    def test_a_box_missing_the_gt_center_does_not_match_even_when_it_overlaps(self) -> None:
        gt = [_xyxy_to_cxcywh(0.40, 0.40, 0.60, 0.60)]  # center (0.5, 0.5)
        pred = [_xyxy_to_cxcywh(0.30, 0.30, 0.49, 0.49)]  # overlaps, but ends before the center
        result = compute_map(**_one_image(pred, [0.9], [DAMAGE], gt, [DAMAGE]), num_classes=2)
        assert result["per_class_ap_center"][DAMAGE] == pytest.approx(0.0)
        assert result["per_class_center_recall"][DAMAGE] == pytest.approx(0.0)

    def test_the_box_border_counts_as_containing(self) -> None:
        gt = [_xyxy_to_cxcywh(0.40, 0.40, 0.60, 0.60)]  # center (0.5, 0.5)
        pred = [_xyxy_to_cxcywh(0.20, 0.20, 0.50, 0.50)]  # center exactly on the corner
        result = compute_map(**_one_image(pred, [0.9], [DAMAGE], gt, [DAMAGE]), num_classes=2)
        assert result["per_class_center_recall"][DAMAGE] == pytest.approx(1.0)

    def test_a_big_box_over_two_gt_centers_matches_only_one(self) -> None:
        gt = [_xyxy_to_cxcywh(0.30, 0.40, 0.40, 0.50), _xyxy_to_cxcywh(0.60, 0.40, 0.70, 0.50)]
        pred = [_xyxy_to_cxcywh(0.10, 0.10, 0.90, 0.90)]  # contains both centers
        result = compute_map(
            **_one_image(pred, [0.9], [DAMAGE], gt, [DAMAGE, DAMAGE]), num_classes=2
        )
        assert result["per_class_center_recall"][DAMAGE] == pytest.approx(0.5)
        # One TP out of two GT: the PR curve stops at recall 0.5 with precision 1.
        assert result["per_class_ap_center"][DAMAGE] == pytest.approx(0.5)

    def test_two_overlapping_big_boxes_can_cover_two_gt(self) -> None:
        gt = [_xyxy_to_cxcywh(0.30, 0.40, 0.40, 0.50), _xyxy_to_cxcywh(0.60, 0.40, 0.70, 0.50)]
        big = _xyxy_to_cxcywh(0.10, 0.10, 0.90, 0.90)
        pred = [big, big]
        result = compute_map(
            **_one_image(pred, [0.9, 0.8], [DAMAGE, DAMAGE], gt, [DAMAGE, DAMAGE]), num_classes=2
        )
        # Each GT is matched at most once and each prediction at most once.
        assert result["per_class_center_recall"][DAMAGE] == pytest.approx(1.0)
        assert result["per_class_ap_center"][DAMAGE] == pytest.approx(1.0)

    def test_duplicate_prediction_on_one_gt_is_a_false_positive(self) -> None:
        gt = [_xyxy_to_cxcywh(0.40, 0.40, 0.60, 0.60)]
        box = _xyxy_to_cxcywh(0.30, 0.30, 0.70, 0.70)
        result = compute_map(
            **_one_image([box, box], [0.9, 0.8], [DAMAGE, DAMAGE], gt, [DAMAGE]), num_classes=2
        )
        assert result["per_class_center_recall"][DAMAGE] == pytest.approx(1.0)
        # The curve reaches recall 1 at precision 1 before the duplicate lowers it.
        assert result["per_class_ap_center"][DAMAGE] == pytest.approx(1.0)

    def test_matching_is_greedy_by_score(self) -> None:
        # One box covers the GT center, the other is far away: ranking the miss first
        # puts a false positive ahead of the only true positive.
        gt = [_xyxy_to_cxcywh(0.40, 0.40, 0.60, 0.60)]
        hit = _xyxy_to_cxcywh(0.30, 0.30, 0.70, 0.70)
        miss = _xyxy_to_cxcywh(0.00, 0.00, 0.10, 0.10)
        ranked_hit_first = compute_map(
            **_one_image([hit, miss], [0.9, 0.5], [DAMAGE, DAMAGE], gt, [DAMAGE]), num_classes=2
        )
        ranked_miss_first = compute_map(
            **_one_image([hit, miss], [0.5, 0.9], [DAMAGE, DAMAGE], gt, [DAMAGE]), num_classes=2
        )
        assert ranked_hit_first["per_class_ap_center"][DAMAGE] == pytest.approx(1.0)
        # FP first: precision at recall 1 is 1/2.
        assert ranked_miss_first["per_class_ap_center"][DAMAGE] == pytest.approx(0.5)

    def test_the_higher_scored_box_takes_the_gt_both_boxes_cover(self) -> None:
        # Both boxes contain the single GT center. The higher-scored one takes the match
        # (ranked first: AP 1.0); the other is the false positive that follows it.
        # Were the lower-scored box to take it, the curve would open with a false
        # positive and AP would be 0.5.
        gt = [_xyxy_to_cxcywh(0.40, 0.40, 0.60, 0.60)]
        a = _xyxy_to_cxcywh(0.30, 0.30, 0.70, 0.70)
        b = _xyxy_to_cxcywh(0.20, 0.20, 0.80, 0.80)
        result = compute_map(
            **_one_image([a, b], [0.4, 0.8], [DAMAGE, DAMAGE], gt, [DAMAGE]), num_classes=2
        )
        assert result["per_class_ap_center"][DAMAGE] == pytest.approx(1.0)

    def test_gt_of_another_image_never_matches(self) -> None:
        result = compute_map(
            pred_boxes=[_t([_xyxy_to_cxcywh(0.1, 0.1, 0.9, 0.9)]), _t([])],
            pred_scores=[torch.tensor([0.9]), torch.zeros(0)],
            pred_labels=[torch.tensor([DAMAGE]), torch.zeros(0, dtype=torch.long)],
            gt_boxes=[_t([]), _t([_xyxy_to_cxcywh(0.4, 0.4, 0.6, 0.6)])],
            gt_labels=[torch.zeros(0, dtype=torch.long), torch.tensor([DAMAGE])],
            num_classes=2,
        )
        assert result["per_class_ap_center"][DAMAGE] == pytest.approx(0.0)
        assert result["per_class_center_recall"][DAMAGE] == pytest.approx(0.0)

    def test_class_mismatch_never_matches(self) -> None:
        gt = [_xyxy_to_cxcywh(0.40, 0.40, 0.60, 0.60)]
        pred = [_xyxy_to_cxcywh(0.30, 0.30, 0.70, 0.70)]
        result = compute_map(**_one_image(pred, [0.9], [0], gt, [DAMAGE]), num_classes=2)
        assert result["per_class_ap_center"][DAMAGE] == pytest.approx(0.0)
        assert result["per_class_center_recall"][DAMAGE] == pytest.approx(0.0)

    def test_no_gt_and_no_predictions_are_zero(self) -> None:
        result = compute_map(
            pred_boxes=[_t([])],
            pred_scores=[torch.zeros(0)],
            pred_labels=[torch.zeros(0, dtype=torch.long)],
            gt_boxes=[_t([])],
            gt_labels=[torch.zeros(0, dtype=torch.long)],
            num_classes=2,
        )
        for key in ("per_class_ap_30", "per_class_ap_center", "per_class_center_recall"):
            assert result[key] == {0: 0.0, 1: 0.0}, key


class TestCenterRecallAtTheOperatingPoint:
    def test_only_predictions_at_or_above_the_threshold_count(self) -> None:
        gt = [_xyxy_to_cxcywh(0.10, 0.10, 0.20, 0.20), _xyxy_to_cxcywh(0.60, 0.60, 0.70, 0.70)]
        pred = [_xyxy_to_cxcywh(0.05, 0.05, 0.25, 0.25), _xyxy_to_cxcywh(0.55, 0.55, 0.75, 0.75)]
        kwargs = _one_image(pred, [0.60, 0.10], [DAMAGE, DAMAGE], gt, [DAMAGE, DAMAGE])

        loose = compute_map(**kwargs, num_classes=2, score_threshold=0.05)
        strict = compute_map(**kwargs, num_classes=2, score_threshold=0.25)

        assert loose["per_class_center_recall"][DAMAGE] == pytest.approx(1.0)
        assert strict["per_class_center_recall"][DAMAGE] == pytest.approx(0.5)
        # AP integrates the whole ranked list, so the threshold does not move it.
        assert strict["per_class_ap_center"][DAMAGE] == loose["per_class_ap_center"][DAMAGE]

    def test_threshold_is_inclusive(self) -> None:
        gt = [_xyxy_to_cxcywh(0.40, 0.40, 0.60, 0.60)]
        pred = [_xyxy_to_cxcywh(0.30, 0.30, 0.70, 0.70)]
        result = compute_map(
            **_one_image(pred, [0.25], [DAMAGE], gt, [DAMAGE]), num_classes=2, score_threshold=0.25
        )
        assert result["per_class_center_recall"][DAMAGE] == pytest.approx(1.0)

    def test_center_recall_is_at_least_iou_recall_for_boxes_that_contain_their_gt(self) -> None:
        gt = [_xyxy_to_cxcywh(0.40, 0.40, 0.60, 0.60)]
        pred = [_xyxy_to_cxcywh(0.10, 0.10, 0.90, 0.90)]
        result = compute_map(**_one_image(pred, [0.9], [DAMAGE], gt, [DAMAGE]), num_classes=2)
        assert result["per_class_recall"][DAMAGE] == pytest.approx(0.0)  # IoU 0.25 < 0.5
        assert result["per_class_center_recall"][DAMAGE] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Existing outputs are unchanged
# --------------------------------------------------------------------------- #


def _regression_fixture():
    """A seeded multi-image fixture (numpy's PCG64 stream is stable across versions)."""
    rng = np.random.default_rng(20261018)
    pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels = [], [], [], [], []
    for _ in range(14):
        n_gt = int(rng.integers(0, 4))
        g = np.stack(
            [
                rng.uniform(0.2, 0.8, n_gt),
                rng.uniform(0.2, 0.8, n_gt),
                rng.uniform(0.05, 0.3, n_gt),
                rng.uniform(0.05, 0.3, n_gt),
            ],
            axis=1,
        ).reshape(-1, 4)
        g_lab = rng.integers(0, 2, n_gt)
        # Jittered copies of the GT plus unrelated boxes.
        jitter = rng.normal(0.0, 0.03, g.shape)
        p_true = np.clip(g + jitter, 0.01, 1.0)
        n_noise = int(rng.integers(0, 5))
        p_noise = np.stack(
            [
                rng.uniform(0.1, 0.9, n_noise),
                rng.uniform(0.1, 0.9, n_noise),
                rng.uniform(0.05, 0.3, n_noise),
                rng.uniform(0.05, 0.3, n_noise),
            ],
            axis=1,
        ).reshape(-1, 4)
        p = np.concatenate([p_true, p_noise], axis=0)
        p_lab = np.concatenate([g_lab, rng.integers(0, 2, n_noise)])
        p_score = rng.uniform(0.001, 1.0, len(p))
        pred_boxes.append(torch.from_numpy(p).float())
        pred_scores.append(torch.from_numpy(p_score).float())
        pred_labels.append(torch.from_numpy(p_lab).long())
        gt_boxes.append(torch.from_numpy(g).float())
        gt_labels.append(torch.from_numpy(g_lab).long())
    return pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels


# Values produced by `compute_map` BEFORE the localization metrics existed, on the
# fixture above with score_threshold=0.25 (captured from the unmodified code).
_GOLDEN = {
    "map50": 0.17517413944005966,
    "map_50_95": 0.03483697534538806,
    "per_class_ap_50": {0: 0.13254289329051971, 1: 0.2178053855895996},
    "per_class_ap_50_95": {0: 0.027166473399847745, 1: 0.04250747729092837},
    "per_class_precision": {0: 0.1875, 1: 0.3333333333333333},
    "per_class_recall": {0: 0.375, 1: 0.4666666666666667},
    "per_class_f1": {0: 0.25, 1: 0.3888888888888889},
    "per_class_counts": {
        0: {"tp": 3, "fp": 13, "fn": 5},
        1: {"tp": 7, "fp": 14, "fn": 8},
    },
}


class TestExistingMetricsAreUnchanged:
    def test_golden_values_of_the_existing_keys(self) -> None:
        pb, ps, pl, gb, gl = _regression_fixture()
        result = compute_map(pb, ps, pl, gb, gl, num_classes=2, score_threshold=0.25)
        assert result["map50"] == pytest.approx(_GOLDEN["map50"], abs=1e-9)
        assert result["map_50_95"] == pytest.approx(_GOLDEN["map_50_95"], abs=1e-9)
        for key in (
            "per_class_ap_50",
            "per_class_ap_50_95",
            "per_class_precision",
            "per_class_recall",
            "per_class_f1",
        ):
            for cls_id in (0, 1):
                assert result[key][cls_id] == pytest.approx(_GOLDEN[key][cls_id], abs=1e-9), key
        assert result["per_class_counts"] == _GOLDEN["per_class_counts"]

    def test_the_existing_keys_come_first_and_in_the_same_order(self) -> None:
        pb, ps, pl, gb, gl = _regression_fixture()
        result = compute_map(pb, ps, pl, gb, gl, num_classes=2)
        assert list(result)[:8] == [
            "map50",
            "map_50_95",
            "per_class_ap_50",
            "per_class_ap_50_95",
            "per_class_precision",
            "per_class_recall",
            "per_class_f1",
            "per_class_counts",
        ]
        assert list(result)[8:] == [
            "per_class_ap_30",
            "per_class_ap_center",
            "per_class_center_recall",
        ]

    def test_ap30_is_never_below_ap50_and_center_is_in_unit_range(self) -> None:
        pb, ps, pl, gb, gl = _regression_fixture()
        result = compute_map(pb, ps, pl, gb, gl, num_classes=2)
        for cls_id in (0, 1):
            assert result["per_class_ap_30"][cls_id] >= result["per_class_ap_50"][cls_id] - 1e-9
            assert 0.0 <= result["per_class_ap_center"][cls_id] <= 1.0
            assert 0.0 <= result["per_class_center_recall"][cls_id] <= 1.0


# --------------------------------------------------------------------------- #
# History / CSV
# --------------------------------------------------------------------------- #


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestMetricsHistoryCsv:
    def test_new_columns_are_written_after_the_existing_ones(self, tmp_path) -> None:
        history = LossHistory()
        history.update(
            cls=0.5,
            box=0.5,
            total=1.0,
            map50=0.4,
            per_class_ap_50={0: 0.9, 1: 0.1},
            per_class_ap_30={0: 0.95, 1: 0.45},
            per_class_ap_center={0: 0.97, 1: 0.6},
            per_class_center_recall={0: 1.0, 1: 0.7},
        )
        generate_training_curves(history, tmp_path)

        with (tmp_path / "metrics_history.csv").open(newline="") as f:
            header = next(csv.reader(f))
        rows = _read_csv(tmp_path / "metrics_history.csv")

        assert rows[0]["ap30_class_1"] == "0.45"
        assert rows[0]["apcenter_class_1"] == "0.6"
        assert rows[0]["center_recall_class_1"] == "0.7"
        assert rows[0]["ap30_class_0"] == "0.95"
        # Appended: every pre-existing column keeps its position.
        assert header.index("fn_class_1") < header.index("ap30_class_0")
        assert header[-6:] == [
            "ap30_class_0",
            "ap30_class_1",
            "apcenter_class_0",
            "apcenter_class_1",
            "center_recall_class_0",
            "center_recall_class_1",
        ]

    def test_history_without_the_new_metrics_writes_the_old_header(self, tmp_path) -> None:
        history = LossHistory()
        history.update(cls=0.5, box=0.5, total=1.0, per_class_ap_50={0: 0.9, 1: 0.1})
        generate_training_curves(history, tmp_path)
        with (tmp_path / "metrics_history.csv").open(newline="") as f:
            header = next(csv.reader(f))
        assert not any(c.startswith(("ap30", "apcenter", "center_recall")) for c in header)
        assert header[-2:] == ["fn_class_0", "fn_class_1"]

    def test_update_accepts_the_new_dicts_and_defaults_them_to_empty(self) -> None:
        history = LossHistory()
        history.update(cls=1.0, box=1.0, total=2.0)
        assert history.per_class_ap_30 == [{}]
        assert history.per_class_ap_center == [{}]
        assert history.per_class_center_recall == [{}]
