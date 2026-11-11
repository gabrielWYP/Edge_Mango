"""Top-1 localization metrics: CorLoc and LROC (src/evaluation/top1_localization.py).

Predictions are hand-made so every expected value is computed by hand in the test that asserts it. Hours come from
`LATENT_HOURS` so the tests keep their meaning if the latent window constant changes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.data_pipeline.v4_selection import LATENT_HOURS
from src.evaluation import top1_localization as t1
from src.evaluation.v4_metrics import EvaluatedPhoto, face_auroc

LATENT = sorted(LATENT_HOURS)[0]
DEVELOPED = max(LATENT_HOURS) + 2
GT = [0.25, 0.25, 0.75, 0.75]  # the one damage box of every damaged photo (binary fractions: exact IoUs)
HIT = [0.25, 0.25, 0.75, 0.75]  # IoU 1 with GT
PARTIAL = [0.25, 0.25, 0.75, 0.5]  # IoU 0.5 with GT (half the area)
LOOSE = [0.25, 0.25, 0.75, 0.4375]  # IoU 0.375 with GT: right at 0.3, wrong at 0.5
MISS = [0.8, 0.8, 0.95, 0.95]  # IoU 0 with GT


def photo(stem, hour, preds=(), gt=True, polygons=None, gt_polygons=()):
    """`preds` is [(score, xyxy)]; `hour` None is a sano photo (no ground truth)."""
    boxes = torch.tensor([b for _, b in preds], dtype=torch.float64).reshape(-1, 4)
    scores = torch.tensor([s for s, _ in preds], dtype=torch.float64)
    gt_boxes = torch.tensor([GT], dtype=torch.float64) if gt and hour is not None else torch.zeros((0, 4), dtype=torch.float64)
    return EvaluatedPhoto(
        stem=stem, hour=hour, zona="x", provenance="p", face=("1", stem, "a"),
        gt_boxes=gt_boxes, pred_boxes=boxes, pred_scores=scores,
        pred_polygons=polygons, gt_polygons=tuple(gt_polygons),
    )


def square(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def mixed_photos():
    """Latent: .9 right, .8 wrong, .4 right, no detection. Sano: .85, .5, no detection, no detection."""
    return [
        photo("p1", LATENT, [(0.9, HIT)]),
        photo("p2", LATENT, [(0.8, MISS)]),
        photo("p3", LATENT, [(0.4, HIT)]),
        photo("p4", LATENT),
        photo("s1", None, [(0.85, MISS)]),
        photo("s2", None, [(0.5, MISS)]),
        photo("s3", None),
        photo("s4", None),
    ]


# --- top-1 extraction ---------------------------------------------------------------------------------------------


def test_top1_is_the_highest_score_box_whatever_its_confidence() -> None:
    record = t1.top1_record(photo("a", LATENT, [(0.01, HIT), (0.005, MISS)]))
    assert record.score == 0.01 and record.iou == pytest.approx(1.0)
    record = t1.top1_record(photo("a", LATENT, [(0.6, MISS), (0.7, HIT), (0.2, HIT)]))
    assert record.score == 0.7 and record.iou == pytest.approx(1.0)  # the 0.7 box, not the first nor the best-IoU one


def test_a_photo_without_damage_prediction_scores_minus_infinity_and_is_never_correct() -> None:
    record = t1.top1_record(photo("a", LATENT))
    assert record.score == -math.inf and record.iou == 0.0
    assert t1.corloc([record], 0.3) == 0.0


def test_the_iou_is_against_the_ground_truth_box() -> None:
    assert t1.top1_record(photo("a", LATENT, [(0.5, PARTIAL)])).iou == pytest.approx(0.5)
    assert t1.top1_record(photo("a", LATENT, [(0.5, LOOSE)])).iou == pytest.approx(0.375)
    assert t1.top1_record(photo("a", LATENT, [(0.5, MISS)])).iou == 0.0


def test_a_sano_or_unlabelled_photo_has_no_iou() -> None:
    assert t1.top1_record(photo("s", None, [(0.5, MISS)])).iou is None
    assert t1.top1_record(photo("a", LATENT, [(0.5, HIT)], gt=False)).iou is None


# --- CorLoc -------------------------------------------------------------------------------------------------------


def test_corloc_counts_the_fraction_of_damaged_photos_with_a_correct_top1() -> None:
    records = [t1.top1_record(p) for p in mixed_photos() if p.hour is not None]
    assert t1.corloc(records, 0.5) == 0.5  # p1 and p3 of four


def test_corloc_depends_on_the_iou_threshold() -> None:
    records = [t1.top1_record(photo(s, LATENT, [(0.5, box)])) for s, box in (("a", HIT), ("b", PARTIAL), ("c", LOOSE), ("d", MISS))]
    assert t1.corloc(records, 0.5) == 0.5  # IoU >= 0.5 is correct: HIT and PARTIAL
    assert t1.corloc(records, 0.3) == 0.75


def test_corloc_is_none_without_photos() -> None:
    assert t1.corloc([], 0.5) is None


def test_the_photo_sets_follow_the_v4_metrics_selection() -> None:
    photos = [
        photo("l", LATENT, [(0.5, HIT)]), photo("d", DEVELOPED, [(0.5, HIT)]), photo("n", LATENT, [(0.5, HIT)], gt=False),
        photo("s", None, [(0.5, MISS)]),
    ]
    sets = t1.photo_sets([t1.top1_record(p) for p in photos])
    assert [r.hour for r in sets["primary"]] == [LATENT]
    assert [r.hour for r in sets["developed"]] == [DEVELOPED]
    assert [r.hour for r in sets["all"]] == [LATENT, DEVELOPED]  # sano and unlabelled photos leave every set


# --- LROC ---------------------------------------------------------------------------------------------------------


def lroc(photos, iou=0.5):
    records = [t1.top1_record(p) for p in photos]
    return t1.lroc_curve(
        [r for r in records if r.hour in LATENT_HOURS and r.iou is not None], [r for r in records if r.hour is None], iou
    )


def test_a_perfect_model_has_corloc_and_auc_lroc_of_one() -> None:
    photos = [photo(f"p{i}", LATENT, [(0.9, HIT)]) for i in range(3)] + [photo(f"s{i}", None) for i in range(3)]
    curve = lroc(photos)
    assert t1.auc_lroc(curve) == pytest.approx(1.0)
    assert t1.corloc([t1.top1_record(p) for p in photos[:3]], 0.5) == 1.0


def test_right_ranking_with_wrong_boxes_gives_high_auroc_and_zero_lroc() -> None:
    photos = [photo(f"p{i}", LATENT, [(0.9, MISS)]) for i in range(3)] + [photo(f"s{i}", None, [(0.1, MISS)]) for i in range(3)]
    records = [t1.top1_record(p) for p in photos]
    assert t1.top1_auroc(records) == 1.0
    assert t1.auc_lroc(lroc(photos)) == 0.0
    assert t1.corloc(records[:3], 0.5) == 0.0


def test_auc_lroc_and_auroc_of_a_hand_made_model() -> None:
    # Thresholds .9 .85 .8 .5 .4 -inf: (FPF, TPF_loc) = (0,.25) (.25,.25) (.25,.25) (.5,.25) (.5,.5) (1,.5)
    curve = lroc(mixed_photos())
    assert curve.fpf[0] == 0.0 and curve.tpf[0] == 0.0 and curve.thresholds[0] == math.inf
    assert (curve.fpf[-1], curve.tpf[-1], curve.thresholds[-1]) == (1.0, 0.5, -math.inf)
    assert t1.auc_lroc(curve) == pytest.approx(0.0625 + 0.0625 + 0.25)
    # positives [.9 .8 .4 -inf] against negatives [.85 .5 -inf -inf]: (4 + 3 + 2 + 2 * 0.5) / 16
    assert t1.top1_auroc([t1.top1_record(p) for p in mixed_photos()]) == pytest.approx(10 / 16)


def test_auc_lroc_never_exceeds_corloc_and_reaches_it_when_every_sano_scores_below() -> None:
    records = [t1.top1_record(p) for p in mixed_photos()]
    corloc = t1.corloc([r for r in records if r.hour is not None], 0.5)
    assert t1.auc_lroc(lroc(mixed_photos())) <= corloc
    below = [p for p in mixed_photos() if p.hour is not None] + [photo(f"s{i}", None) for i in range(4)]
    assert t1.auc_lroc(lroc(below)) == pytest.approx(corloc)  # every sano below every positive: the area is CorLoc


def test_the_lroc_looser_iou_counts_more_correct_boxes() -> None:
    photos = [photo("p", LATENT, [(0.9, LOOSE)]), photo("s", None)]
    assert t1.auc_lroc(lroc(photos, 0.5)) == 0.0
    assert t1.auc_lroc(lroc(photos, 0.3)) == 1.0


def test_tied_scores_move_the_curve_in_one_step() -> None:
    photos = [photo("p", LATENT, [(0.5, HIT)]), photo("s", None, [(0.5, MISS)])]
    curve = lroc(photos)
    assert list(curve.fpf) == [0.0, 1.0, 1.0] and list(curve.tpf) == [0.0, 1.0, 1.0]
    assert t1.auc_lroc(curve) == pytest.approx(0.5)  # the diagonal of a tie


def test_lroc_needs_both_groups() -> None:
    assert t1.auc_lroc(lroc([photo("p", LATENT, [(0.5, HIT)])])) is None
    assert t1.auc_lroc(lroc([photo("s", None)])) is None
    assert t1.top1_auroc([t1.top1_record(photo("p", LATENT, [(0.5, HIT)]))]) is None


def test_the_top1_auroc_matches_the_face_auroc_of_v4_metrics() -> None:
    photos = mixed_photos()
    assert t1.top1_auroc([t1.top1_record(p) for p in photos]) == face_auroc(photos)["auroc"]


# --- operating points ---------------------------------------------------------------------------------------------


def test_operating_points_at_a_confidence() -> None:
    records = [t1.top1_record(p) for p in mixed_photos()]
    positives = [r for r in records if r.hour in LATENT_HOURS]
    negatives = [r for r in records if r.hour is None]
    point = t1.operating_point(positives, negatives, 0.5, 0.5)
    assert point == {"fpf": 0.5, "tpf_loc": 0.25, "tpf": 0.5}  # sano .85 and .5; p1 right; p1 and p2 at >= 0.5
    point = t1.operating_point(positives, negatives, 0.88, 0.5)
    assert point == {"fpf": 0.0, "tpf_loc": 0.25, "tpf": 0.25}


# --- mask CorLoc --------------------------------------------------------------------------------------------------


def test_mask_corloc_uses_the_top1_mask_against_the_ground_truth_polygon() -> None:
    gt = [square(*GT)]
    right = photo("r", LATENT, [(0.9, HIT), (0.1, MISS)], polygons=(square(*GT), square(*MISS)),
                  gt_polygons=gt)
    # right box but a mask that misses: the box CorLoc is 1 and the mask one 0
    wrong_mask = photo("w", LATENT, [(0.9, HIT)], polygons=(square(*MISS),), gt_polygons=gt)
    records = [t1.top1_record(right), t1.top1_record(wrong_mask)]
    assert records[0].mask_iou == pytest.approx(1.0) and records[1].mask_iou == 0.0
    assert t1.corloc(records, 0.5) == 1.0
    assert t1.corloc(records, 0.5, mask=True) == 0.5


def test_mask_corloc_skips_detect_photos_and_box_only_ground_truth() -> None:
    detect = t1.top1_record(photo("d", LATENT, [(0.9, HIT)]))
    box_only = t1.top1_record(photo("b", LATENT, [(0.9, HIT)], polygons=(square(*GT),)))
    assert detect.mask_iou is None and box_only.mask_iou is None
    assert t1.corloc([detect, box_only], 0.5, mask=True) is None


def test_a_missing_or_empty_top1_mask_has_zero_mask_iou() -> None:
    gt = [square(*GT)]
    none = t1.top1_record(photo("n", LATENT, polygons=(), gt_polygons=gt))
    empty = t1.top1_record(photo("e", LATENT, [(0.9, HIT)], polygons=(np.zeros((0, 2)),), gt_polygons=gt))
    assert none.mask_iou == 0.0 and empty.mask_iou == 0.0


# --- the metrics of one evaluation --------------------------------------------------------------------------------


def test_evaluate_gathers_every_metric_of_one_evaluation() -> None:
    photos = mixed_photos() + [photo("d1", DEVELOPED, [(0.7, HIT)]), photo("d2", DEVELOPED, [(0.3, MISS)])]
    row, curves = t1.evaluate(photos)
    assert row["n_primary"] == 4 and row["n_developed"] == 2 and row["n_all"] == 6 and row["n_sano"] == 4
    assert row["corloc_primary_iou50"] == 0.5 and row["corloc_developed_iou50"] == 0.5
    assert row["corloc_all_iou50"] == pytest.approx(3 / 6)
    assert row["lroc_auc_iou50"] == pytest.approx(0.375) and row["lroc_auc_iou30"] == pytest.approx(0.375)
    assert row["auroc_top1"] == pytest.approx(10 / 16)
    assert row["fpf_at_0.5"] == 0.5 and row["tpf_loc_at_0.5_iou50"] == 0.25 and row["tpf_at_0.5"] == 0.5
    assert row["fpf_at_0.25"] == 0.5 and row["tpf_loc_at_0.25_iou50"] == 0.5 and row["tpf_at_0.25"] == 0.75
    assert set(curves) == {0.5, 0.3}
    assert not any(key.startswith("mask_") for key in row)  # detect evaluation


def test_evaluate_adds_the_mask_variant_for_a_segment_evaluation() -> None:
    gt = [square(*GT)]
    photos = [
        photo("p", LATENT, [(0.9, HIT)], polygons=(square(*GT),), gt_polygons=gt),
        photo("s", None, [(0.1, MISS)], polygons=(square(*MISS),)),
    ]
    row, _ = t1.evaluate(photos)
    assert row["mask_corloc_primary_iou50"] == 1.0 and row["mask_corloc_all_iou30"] == 1.0
    assert row["mask_corloc_developed_iou50"] is None


def test_evaluate_reports_none_for_an_empty_set() -> None:
    row, _ = t1.evaluate([photo("p", LATENT, [(0.9, HIT)])])
    assert row["corloc_developed_iou50"] is None and row["lroc_auc_iou50"] is None and row["auroc_top1"] is None
    assert row["fpf_at_0.5"] is None and row["tpf_at_0.5"] == 1.0
