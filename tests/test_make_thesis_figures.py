"""Pure-function tests for scripts/make_thesis_figures.py (CPU, pandas only)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from scripts import make_thesis_figures as mtf


# ---------------------------------------------------------------- arms and styles
def test_discover_arms_follows_fixed_order_and_appends_unknown_last():
    arms = mtf.discover_arms(["crosskd_nir_seg", "zzz_new_arm", "master_rgb_seg", "master_nir_seg", "aaa_new_arm"])
    assert arms == ["master_rgb_seg", "master_nir_seg", "crosskd_nir_seg", "aaa_new_arm", "zzz_new_arm"]


def test_arm_styles_are_unique_and_reserve_twostem():
    colors = [s["color"] for s in mtf.ARM_STYLE.values()]
    assert len(colors) == len(set(colors))
    assert "master_twostem_rgbnir_seg" in mtf.ARM_STYLE


def test_style_of_unknown_arm_is_stable_and_does_not_repaint_known_arms():
    before = {a: mtf.style_of(a)["color"] for a in mtf.ARM_STYLE}
    first = mtf.style_of("brand_new_arm")
    assert mtf.style_of("brand_new_arm") == first
    assert {a: mtf.style_of(a)["color"] for a in mtf.ARM_STYLE} == before
    assert first["color"] not in before.values()


# ---------------------------------------------------------------- fold aggregation
def _long(values: dict[str, dict[int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"arm": a, "fold": f, "v": v} for a, per in values.items() for f, v in per.items()]
    )


def test_fold_table_mean_sd_and_per_fold_values():
    df = _long({"a": {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.4, 4: 0.4}})
    t = mtf.fold_table(df, "v", arms=["a"], folds=[0, 1, 2, 3, 4])
    row = t.iloc[0]
    assert row["n"] == 5
    assert row["mean"] == pytest.approx(0.4)
    assert row["sd"] == pytest.approx(np.std([0.2, 0.4, 0.6, 0.4, 0.4], ddof=1))
    assert [row[f"fold{k}"] for k in range(5)] == pytest.approx([0.2, 0.4, 0.6, 0.4, 0.4])


def test_fold_table_missing_fold_is_nan_not_zero():
    df = _long({"a": {0: 0.2, 1: 0.4, 2: 0.6, 4: 0.8}})  # fold 3 absent
    row = mtf.fold_table(df, "v", arms=["a"], folds=[0, 1, 2, 3, 4]).iloc[0]
    assert math.isnan(row["fold3"])
    assert row["n"] == 4
    assert row["mean"] == pytest.approx(0.5)  # mean of the 4 present folds, a 0 would give 0.4


def test_fold_table_missing_arm_is_listed_with_no_data():
    df = _long({"a": {0: 0.5}})
    t = mtf.fold_table(df, "v", arms=["a", "ghost"], folds=[0, 1])
    ghost = t[t["arm"] == "ghost"].iloc[0]
    assert ghost["n"] == 0 and math.isnan(ghost["mean"]) and math.isnan(ghost["sd"])


def test_fold_table_single_fold_has_no_sd():
    row = mtf.fold_table(_long({"a": {2: 0.7}}), "v", arms=["a"], folds=[0, 1, 2]).iloc[0]
    assert row["n"] == 1 and math.isnan(row["sd"]) and row["mean"] == pytest.approx(0.7)


def test_curve_mean_sd_across_folds_per_epoch():
    hist = pd.DataFrame(
        {
            "arm": ["a"] * 6,
            "fold": [0, 0, 0, 1, 1, 1],
            "epoch": [1, 2, 3, 1, 2, 3],
            "m": [0.0, 1.0, 2.0, 2.0, 3.0, 4.0],
        }
    )
    c = mtf.curve_mean_sd(hist, "a", "m")
    assert list(c["epoch"]) == [1, 2, 3]
    assert list(c["mean"]) == pytest.approx([1.0, 2.0, 3.0])
    assert list(c["sd"]) == pytest.approx([math.sqrt(2)] * 3)
    assert list(c["n"]) == [2, 2, 2]


def test_curve_mean_sd_of_absent_arm_is_empty():
    hist = pd.DataFrame({"arm": ["a"], "fold": [0], "epoch": [1], "m": [1.0]})
    assert mtf.curve_mean_sd(hist, "ghost", "m").empty


# ---------------------------------------------------------------- operating points
def _grid(rows):
    return pd.DataFrame(rows, columns=["threshold", "tp", "fp", "fn"])


def test_prf_from_counts_handles_zero_denominators():
    p, r, f = mtf.prf(0, 0, 5)
    assert math.isnan(p) and r == 0.0 and f == 0.0
    p, r, f = mtf.prf(8, 2, 8)
    assert (p, r) == pytest.approx((0.8, 0.5))
    assert f == pytest.approx(2 * 0.8 * 0.5 / 1.3)


def test_pick_threshold_maximises_f1_on_val_and_breaks_ties_upwards():
    val = _grid(
        [
            (0.1, 10, 90, 0),  # P .1  R 1   F1 .18
            (0.3, 8, 2, 2),  # P .8  R .8  F1 .8
            (0.5, 8, 2, 2),  # identical F1: tie, keep the higher threshold
            (0.9, 1, 0, 9),
        ]
    )
    assert mtf.pick_threshold(val) == 0.5


def test_operating_point_is_chosen_on_val_and_applied_to_test_without_peeking():
    val = _grid([(0.2, 9, 1, 1), (0.6, 5, 0, 5)])  # best on val: 0.2
    test = _grid([(0.2, 1, 50, 9), (0.6, 9, 1, 1)])  # test would favour 0.6; must not be used
    thr, (p, r, f) = mtf.operating_point(val, test)
    assert thr == 0.2
    assert (p, r) == pytest.approx((1 / 51, 0.1))


def test_apply_threshold_missing_in_test_grid_raises():
    with pytest.raises(KeyError):
        mtf.apply_threshold(_grid([(0.5, 1, 1, 1)]), 0.2)


def test_photo_level_classification_at_threshold():
    scores = np.array([0.9, 0.6, 0.2, 0.7, 0.1])
    is_pos = np.array([True, True, True, False, False])
    out = mtf.photo_metrics(scores, is_pos, 0.5)
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["specificity"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(2 / 3)


def test_photo_operating_point_val_only():
    val_s = np.array([0.9, 0.8, 0.3, 0.2])
    val_p = np.array([True, True, False, False])
    test_s = np.array([0.4, 0.45, 0.6, 0.55])
    test_p = np.array([True, True, False, False])
    thr, out = mtf.photo_operating_point(val_s, val_p, test_s, test_p, grid=[0.1, 0.5, 0.85])
    assert thr == 0.5  # val F1 = 1 at 0.5 (first/highest of the ties among 0.5 only); 0.85 loses recall
    assert out["recall"] == 0.0 and out["specificity"] == pytest.approx(0.0)


def test_operating_point_table_never_uses_test_to_pick_and_flags_missing_folds():
    def grid(split, shift):
        return pd.DataFrame(
            [
                {"arm": "a", "fold": f, "split": split, "threshold": t, "tp": tp, "fp": fp, "fn": fn}
                for f in (0, 1)
                for t, tp, fp, fn in [(0.2, 9, 1 + shift, 1), (0.6, 5, 0, 5)]
            ]
        )

    pr = pd.concat([grid("val", 0), grid("test", 0)])
    pr = pr[~((pr.fold == 1) & (pr.split == "test"))]  # fold 1 has no test
    t = mtf.box_operating_table(pr)
    assert set(t["fold"]) == {0}  # absent, not zero
    assert (t["box_threshold"] == 0.2).all()


# ---------------------------------------------------------------- PR curve averaging
def test_mean_pr_curve_averages_folds_per_threshold_and_skips_missing_folds():
    pr = pd.DataFrame(
        [
            {"arm": "a", "fold": 0, "split": "test", "threshold": 0.5, "tp": 5, "fp": 5, "fn": 5},
            {"arm": "a", "fold": 1, "split": "test", "threshold": 0.5, "tp": 9, "fp": 1, "fn": 1},
            {"arm": "a", "fold": 0, "split": "val", "threshold": 0.5, "tp": 0, "fp": 99, "fn": 0},
        ]
    )
    c = mtf.mean_pr_curve(pr, "a", split="test")
    assert len(c) == 1
    assert c.iloc[0]["precision"] == pytest.approx((0.5 + 0.9) / 2)
    assert c.iloc[0]["recall"] == pytest.approx((0.5 + 0.9) / 2)
    assert mtf.mean_pr_curve(pr, "ghost", split="test").empty


# ---------------------------------------------------------------- checkpoint selection
def test_selected_epoch_is_first_argmax_like_the_trainer():
    hist = pd.DataFrame(
        {
            "arm": ["a"] * 4 + ["a"] * 3,
            "fold": [0] * 4 + [1] * 3,
            "epoch": [1, 2, 3, 4, 1, 2, 3],
            "ap50_class_1": [0.1, 0.5, 0.5, 0.3, 0.2, 0.1, 0.4],
        }
    )
    sel = mtf.selected_epochs(hist, "ap50_class_1")
    got = {(r.arm, r.fold): (r.epoch, r.value) for r in sel.itertuples()}
    assert got == {("a", 0): (2, 0.5), ("a", 1): (3, 0.4)}


# ---------------------------------------------------------------- formatting
def test_format_mean_sd():
    assert mtf.format_mean_sd(0.38812, 0.08349) == "0.388 ± 0.083"
    assert mtf.format_mean_sd(float("nan"), float("nan")) == "n/d"
    assert mtf.format_mean_sd(0.5, float("nan")) == "0.500 ± n/d"
    assert mtf.format_mean_sd(41.0, 3.2, digits=1) == "41.0 ± 3.2"


def test_format_value_nan_is_nd_never_zero():
    assert mtf.format_value(float("nan")) == "n/d"
    assert mtf.format_value(0.0) == "0.000"


def test_markdown_table_pipes_header_and_row_count():
    df = pd.DataFrame({"Brazo": ["a", "b"], "Valor": ["1", "2"]})
    md = mtf.to_markdown_table(df)
    lines = md.strip().splitlines()
    assert lines[0] == "| Brazo | Valor |"
    assert set(lines[1].replace("|", "").replace(":", "").replace("-", "").strip()) <= {" "}
    assert len(lines) == 4


def test_build_summary_table_spanish_headers_and_missing_fold_marked():
    df = _long({"master_rgb_seg": {0: 0.2, 1: 0.4, 2: 0.6, 4: 0.8}})
    tbl = mtf.build_summary_table(
        df.rename(columns={"v": "m"}),
        metrics=[("m", "AP50 latente", 3)],
        arms=["master_rgb_seg"],
        folds=[0, 1, 2, 3, 4],
        fold_labels={0: "Lote 1", 1: "Lote 2", 2: "Lote 3", 3: "Lote 4", 4: "Lote 5"},
    )
    assert list(tbl.columns) == [
        "Brazo", "Métrica", "n folds", "Media ± sd", "Lote 1", "Lote 2", "Lote 3", "Lote 4", "Lote 5",
    ]
    row = tbl.iloc[0]
    assert row["Lote 4"] == "n/d" and row["n folds"] == "4"
    assert row["Media ± sd"] == "0.500 ± 0.258"


# ---------------------------------------------------------------- recall-oriented operating points
def test_photo_metrics_reports_discard_rate():
    out = mtf.photo_metrics([0.9, 0.1, 0.8, 0.2], [True, True, False, False], 0.5)
    assert out["discard_rate"] == pytest.approx(1 - out["specificity"])
    assert out["discard_rate"] == pytest.approx(0.5)


def test_pick_threshold_for_recall_takes_highest_threshold_reaching_target():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.3])
    pos = np.array([True] * 5)
    thr, reached = mtf.pick_threshold_for_recall(scores, pos, 0.8, grid=[0.1, 0.5, 0.65, 0.95])
    assert reached and thr == 0.5  # recall: 1.0 at 0.1, 0.8 at 0.5, 0.6 at 0.65


def test_pick_threshold_for_recall_unreachable_uses_lowest_and_flags():
    scores = np.array([0.0, 0.0, 0.9])  # photos with score 0 can never be recalled at thr > 0
    pos = np.array([True, True, True])
    thr, reached = mtf.pick_threshold_for_recall(scores, pos, 0.95, grid=[0.05, 0.5])
    assert thr == 0.05 and reached is False


def test_pick_threshold_fbeta_favours_recall_over_f1():
    scores = np.array([0.9, 0.5, 0.4, 0.8, 0.45])
    pos = np.array([True, True, True, False, False])
    t_f1, _ = mtf.photo_operating_point(scores, pos, scores, pos, grid=[0.3, 0.6, 0.85])
    t_f2 = mtf.pick_threshold_fbeta(scores, pos, beta=2.0, grid=[0.3, 0.6, 0.85])
    assert t_f2 <= t_f1


def test_all_positive_baseline():
    b = mtf.all_positive_baseline(n_pos=8, n_neg=2)
    assert b["recall"] == 1.0 and b["specificity"] == 0.0
    assert b["precision"] == pytest.approx(0.8)
    assert b["f1"] == pytest.approx(2 * 0.8 / 1.8)


def _photos(arm="a", fold=0):
    rows = []
    for split, shift in (("val", 0.0), ("test", -0.5)):
        for i, (h, s) in enumerate([("0", 0.9), ("2", 0.8), ("4", 0.6), ("6", 0.4), ("sano", 0.7), ("sano", 0.2), ("sano", 0.1)]):
            rows.append(dict(arm=arm, fold=fold, split=split, hour=h, primary_metric_photo=1,
                             max_damage_score=max(s + (shift if split == "test" else 0.0), 0.0)))
    return pd.DataFrame(rows)


def test_photo_rules_table_picks_on_val_and_applies_to_test():
    t = mtf.photo_rules_table(_photos())
    r = t.iloc[0]
    assert r["r95_threshold"] <= 0.4  # val recall 1.0 needs thr <= 0.4 (all four damaged photos)
    assert r["r95_reached"] == 1
    # test scores are lower by 0.5: recall at that val threshold must reflect test, not val
    assert r["r95_recall"] < 1.0
    assert r["r95_discard_rate"] == pytest.approx(1 - r["r95_specificity"])
    assert r["fix_recall"] == pytest.approx(0.5)  # test positives at >= 0.25: 0.4 and 0.3 of 4
    assert r["base_f1"] == pytest.approx(2 * (4 / 7) / (1 + 4 / 7))


def test_photo_rules_table_skips_fold_without_test():
    p = _photos()
    p = p[p["split"] == "val"]
    assert mtf.photo_rules_table(p).empty


def test_operating_curve_mean_sd_across_folds():
    p = pd.concat([_photos(fold=0), _photos(fold=1)])
    c = mtf.operating_curve(p, "a", grid=[0.05, 0.5])
    assert list(c["threshold"]) == [0.05, 0.5]
    assert c["n"].tolist() == [2, 2]
    assert c.loc[0, "recall_sd"] == pytest.approx(0.0)  # identical folds
    assert c.loc[0, "recall"] >= c.loc[1, "recall"]  # lower threshold -> more recall
    assert mtf.operating_curve(p, "ghost", grid=[0.5]).empty


def test_hour_recall_uses_val_threshold_and_reports_present_hours_only():
    t = mtf.hour_recall_table(_photos(), target=0.95)
    assert set(t.columns) >= {"arm", "fold", "hour_0", "hour_2", "hour_4", "hour_6"}
    assert "hour_8" not in t.columns  # hour absent from the data: not reported, never 0


# ---------------------------------------------------------------- smoothing, detection loss, main/appendix split
from src.evaluation.smoothing import ema_smooth  # noqa: E402


def test_ema_smooth_bias_corrected_and_identity_without_smoothing():
    assert ema_smooth([1.0, 2.0, 3.0], 0.0) == [1.0, 2.0, 3.0]
    out = ema_smooth([5.0, 5.0, 5.0, 5.0], 0.8)
    assert out == pytest.approx([5.0] * 4)  # bias correction: a constant stays constant
    assert ema_smooth([1.0, 3.0], 0.8)[0] == pytest.approx(1.0)


def test_smoothed_history_smooths_each_fold_independently():
    hist = pd.DataFrame(
        {"arm": ["a"] * 6, "fold": [0, 0, 0, 1, 1, 1], "epoch": [3, 1, 2, 1, 2, 3], "m": [9.0, 1.0, 3.0, 5.0, 5.0, 5.0]}
    )
    out = mtf.smoothed_history(hist, ["m"], w=0.8)
    f0 = out[out["fold"] == 0].sort_values("epoch")["m_smooth"].tolist()
    assert f0 == pytest.approx(ema_smooth([1.0, 3.0, 9.0], 0.8))  # sorted by epoch before smoothing
    assert out[out["fold"] == 1]["m_smooth"].tolist() == pytest.approx([5.0] * 3)
    assert "m" in out.columns  # the raw values are kept


def test_add_detection_loss_is_box_plus_cls_and_falls_back_to_total():
    h = pd.DataFrame({"box_loss": [1.0], "cls_loss": [0.5], "total_loss": [9.0]})
    out, col, fallback = mtf.add_detection_loss(h)
    assert out[col].tolist() == [1.5] and fallback is False
    out, col, fallback = mtf.add_detection_loss(h.drop(columns="cls_loss"))
    assert col == "total_loss" and fallback is True


def test_select_arms_main_keeps_preregistered_arms_in_fixed_order():
    arms = ["master_rgb_seg", "master_nir_seg", "master_rgbnir_seg", "master_rgb_det", "crosskd_rgb_seg",
            "crosskd_rgbnir_seg", "crosskd_nir_seg", "master_twostem_rgbnir_seg"]
    assert mtf.select_arms(arms, "main") == ["master_rgb_seg", "master_nir_seg", "master_rgbnir_seg",
                                             "crosskd_rgb_seg", "crosskd_rgbnir_seg"]
    assert mtf.select_arms(arms, "appendix") == arms
    assert mtf.select_arms(["master_rgb_det"], "main") == []


def test_training_panels_skip_ap30_when_history_has_no_such_column():
    h = pd.DataFrame({"ap50_class_1": [0.1]})
    assert [p[1] for p in mtf.training_panels(h)] == ["det_loss", "ap50_class_1"]
    h["ap30_class_1"] = 0.2
    assert [p[1] for p in mtf.training_panels(h)][-1] == "ap30_class_1"
