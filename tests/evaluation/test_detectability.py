"""Tests for src/evaluation/detectability.py (E0 of docs/tesis/plan-experimental.md)."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import detectability as d


def test_disk_and_ring_do_not_overlap():
    disk, ring = d.disk_and_ring(256)
    assert disk.any() and ring.any()
    assert not (disk & ring).any()


def test_contrast_is_zero_on_a_flat_image_and_ignores_global_gain():
    disk, ring = d.disk_and_ring(64)
    flat = np.full((64, 64), 80.0)
    assert d.contrast(flat, disk, ring) == pytest.approx(0.0)
    img = flat.copy()
    img[disk] = 60.0
    assert d.contrast(img, disk, ring) == pytest.approx(-0.25)
    assert d.contrast(img * 2, disk, ring) == pytest.approx(-0.25)


def test_auroc_extremes_ties_and_empty():
    assert d.auroc([3, 4], [1, 2]) == 1.0
    assert d.auroc([1, 2], [3, 4]) == 0.0
    assert d.auroc([1, 1], [1, 1]) == 0.5
    assert d.auroc([2, 1], [1, 0]) == pytest.approx(0.875)  # pairs: 1, 1, 1/2, 1 over 4
    assert d.auroc([], [1]) is None


def test_cluster_bootstrap_brackets_a_clear_separation_and_keeps_strata():
    rng = np.random.default_rng(0)
    n = 40
    clusters = np.repeat(np.arange(n), 4)
    strata = np.where(clusters < 20, 1, 2)
    labels = np.tile([True, False, False, False], n)
    values = np.where(labels, 1.0, 0.0) + rng.normal(0, 0.1, labels.size)
    lo, hi = d.cluster_bootstrap(d.auroc, values, labels, clusters, strata, 200, rng)
    assert lo > 0.9 and hi <= 1.0


def test_cluster_bootstrap_returns_none_without_both_groups():
    rng = np.random.default_rng(0)
    out = d.cluster_bootstrap(d.auroc, np.zeros(4), np.zeros(4, bool), np.arange(4), np.ones(4), 50, rng)
    assert out == (None, None)


def test_grouped_oof_scores_separate_a_learnable_signal():
    rng = np.random.default_rng(1)
    y = np.tile([1, 0, 0, 0], 50).astype(float)
    X = np.column_stack([y * 2 + rng.normal(0, 0.5, y.size), rng.normal(0, 1, y.size)])
    groups = np.repeat(np.arange(10), 20)
    scores = d.grouped_oof_scores(X, y, groups)
    assert not np.isnan(scores).any()
    assert d.auroc(scores[y == 1], scores[y == 0]) > 0.95


def _null_oof_auroc(rng):
    groups = np.repeat(np.arange(25), 4)  # 25 mangos x 4 faces, 17 random positives: the lot-1 layout
    y = np.zeros(100)
    y[rng.choice(100, 17, replace=False)] = 1
    X = rng.normal(0, 1, (100, 3))
    return d.auroc(*(lambda s: (s[y == 1], s[y == 0]))(d.grouped_oof_scores(X, y, groups))), X, y, groups


def test_grouped_oof_scores_bias_is_small_without_signal():
    # Fold intercepts and fold-wise standardization, pooled, give ~0.39 here; without them ~0.45.
    rng = np.random.default_rng(2)
    assert np.mean([_null_oof_auroc(rng)[0] for _ in range(40)]) > 0.42


def test_permutation_test_calibrates_the_leave_one_out_bias():
    rng = np.random.default_rng(3)
    observed, X, y, groups = _null_oof_auroc(rng)

    def stat(labels):
        s = d.grouped_oof_scores(X, labels.astype(float), groups)
        return d.auroc(s[labels], s[~labels])

    null_mean, p = d.permutation_test(stat, y.astype(bool), np.zeros(100), observed, 200, rng)
    assert null_mean < 0.5
    assert p > 0.05


def test_permutation_test_detects_a_real_signal():
    rng = np.random.default_rng(4)
    y = np.tile([True, False, False, False], 25)
    v = np.where(y, 1.0, 0.0) + rng.normal(0, 0.3, y.size)
    observed = d.auroc(v[y], v[~y])
    _, p = d.permutation_test(lambda lab: d.auroc(v[lab], v[~lab]), y, np.zeros(100), observed, 300, rng)
    assert p < 0.01


def test_grouped_oof_scores_skip_a_fold_whose_training_part_has_one_class():
    y = np.array([1.0, 0.0, 0.0, 0.0])
    groups = np.array([0, 1, 1, 1])
    scores = d.grouped_oof_scores(np.arange(4.0)[:, None], y, groups)
    assert np.isnan(scores[0]) and np.isnan(scores[1:]).all()


def test_excludes_chance():
    assert d.excludes_chance(0.55, 0.8) and d.excludes_chance(0.1, 0.4)
    assert not d.excludes_chance(0.4, 0.6) and not d.excludes_chance(None, None)


def test_onset_hour_requires_the_detection_to_persist():
    hours = [0, 2, 4, 6]
    assert d.onset_hour(hours, [False, True, False, True]) == 6.0
    assert d.onset_hour(hours, [False, True, True, True]) == 2.0
    assert d.onset_hour(hours, [False] * 4) is None


def _reference_ranks(x):
    """The ranking `auroc` has always had: stable sort, ties averaged, each NaN alone at the end."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    ranks[order] = np.arange(1, x.size + 1)
    for value in np.unique(x):
        tied = x == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


@pytest.mark.parametrize("nan_share", [0.0, 0.2])
def test_auroc_keeps_its_ranking_with_ties_and_nan(nan_share):
    rng = np.random.default_rng(5)
    x = rng.integers(0, 12, 300).astype(float)
    x[rng.random(300) < nan_share] = np.nan
    pos, neg = x[:100], x[100:]
    ranks = _reference_ranks(np.concatenate([pos, neg]))
    u = ranks[:100].sum() - 100 * 101 / 2
    assert d.auroc(pos, neg) == pytest.approx(u / (100 * 200), abs=1e-12)


def test_difference_in_differences_subtracts_the_control_change_and_scales_by_the_sd():
    # Impact falls 30, the control falls 10 (drift): the impact effect is -20, in units of sd 4.
    assert d.difference_in_differences(70.0, 100.0, 80.0, 90.0, 4.0) == pytest.approx(-5.0)
    # A change shared by both regions cancels.
    assert d.difference_in_differences(75.0, 100.0, 65.0, 90.0, 4.0) == pytest.approx(0.0)


def test_map_box_returns_the_box_around_the_moved_corners():
    translation = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -2.0]])
    assert d.map_box((10, 20, 30, 40), translation) == pytest.approx([15, 18, 35, 38])
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert d.map_box((10, 20, 30, 40), quarter_turn) == pytest.approx([-40, 10, -20, 30])


def test_invert_warp_undoes_a_euclidean_warp():
    angle = np.radians(7.0)
    warp = np.array([[np.cos(angle), -np.sin(angle), 12.0], [np.sin(angle), np.cos(angle), -3.0]])
    back = d.map_box(d.map_box((50, 60, 50, 60), warp), d.invert_warp(warp))
    assert back == pytest.approx([50, 60, 50, 60])


def _fruit_mask(side=256, axes=(100, 75)):
    yy, xx = np.mgrid[0:side, 0:side]
    return ((xx - side / 2) / axes[0]) ** 2 + ((yy - side / 2) / axes[1]) ** 2 <= 1


def _middle(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@pytest.mark.parametrize(
    "zone, box, side",
    [
        ("superior", (113, 165, 143, 190), "top"),  # superior boxes sit low in the photos
        ("superior", (113, 66, 143, 91), "bottom"),  # the side follows the box, not the name
        ("inferior", (113, 66, 143, 91), "bottom"),
        ("derecha", (50, 115, 80, 140), "right"),
        ("izquierda", (176, 115, 206, 140), "left"),
    ],
)
def test_control_zone_is_the_opposite_side_of_the_box_along_the_axis_of_the_zone(zone, box, side):
    mask = _fruit_mask()
    regions = d.control_regions(zone, box, mask)
    assert regions[0][0] == side
    control = regions[0][1]
    assert control[2] - control[0] == pytest.approx(box[2] - box[0])
    assert control[3] - control[1] == pytest.approx(box[3] - box[1])
    assert not _overlap(control, box)
    assert d.fruit_share(mask, control) >= d.MIN_CONTROL_FRUIT
    # Mirrored: the same distance from the middle of the fruit, on the other side.
    axis = 1 if zone in ("superior", "inferior") else 0
    assert abs(_middle(control)[axis] - 128) == pytest.approx(abs(_middle(box)[axis] - 128), abs=4)


def test_control_zone_of_centro_is_the_farthest_one_inside_the_fruit():
    mask = _fruit_mask(axes=(100, 60))  # wider than tall: left and right reach farther
    box = (118, 118, 138, 138)
    regions = d.control_regions("centro", box, mask)
    first = regions[0]
    assert first[0] in ("left", "right")
    assert d.fruit_share(mask, first[1]) >= d.MIN_CONTROL_FRUIT
    # Inside the fruit, as far as it goes along that side.
    assert abs(_middle(first[1])[0] - 128) > 70


def test_second_control_is_a_different_region_clear_of_the_impact_and_the_first_control():
    mask = _fruit_mask()
    for zone, box in (
        ("superior", (113, 165, 143, 190)),
        ("izquierda", (176, 115, 206, 140)),
        ("centro", (113, 115, 143, 140)),
    ):
        regions = d.control_regions(zone, box, mask)
        assert len(regions) == 2
        assert regions[0][0] != regions[1][0]
        assert not _overlap(regions[1][1], box)
        assert not _overlap(regions[1][1], regions[0][1])
        assert d.fruit_share(mask, regions[1][1]) >= d.MIN_CONTROL_FRUIT


def test_no_control_zone_when_the_box_does_not_fit_on_the_other_side():
    assert d.control_regions("superior", (40, 70, 216, 186), _fruit_mask()) == []
    assert d.control_regions("centro", (113, 115, 143, 140), np.zeros((256, 256), bool)) == []


def test_control_zone_refuses_an_unknown_impact_zone():
    with pytest.raises(ValueError, match="unknown impact zone"):
        d.control_regions("arriba", (113, 115, 143, 140), _fruit_mask())


def test_patch_tiles_cover_the_box_without_overlap_and_a_small_box_is_one_patch():
    tiles = d.patch_tiles((10, 20, 58, 52), side=16)
    assert len(tiles) == 3 * 2
    assert tiles[0] == (10, 20, 26, 36)
    assert d.patch_tiles((10, 20, 20, 30), side=16) == [(10, 20, 20, 30)]


def test_patch_features_are_the_change_of_mean_and_local_std_in_fruit_sd_units():
    side = 32
    before = {ch: np.full((side, side), 100.0, np.float32) for ch in d.PATCH_CHANNELS}
    now = {ch: plane.copy() for ch, plane in before.items()}
    now["NIR"] -= 8.0  # darker
    now["R"][:, ::2] += 4.0  # same size, more texture: half the pixels +4 -> mean +2, std 2
    sd = {ch: 2.0 for ch in d.PATCH_CHANNELS}
    rows = d.patch_features(now, before, (0, 0, side, side), np.ones((side, side), bool), sd)
    assert len(rows) == 4
    assert rows[0]["NIR_mean"] == pytest.approx(-4.0)
    assert rows[0]["NIR_tex"] == pytest.approx(0.0)
    assert rows[0]["R_mean"] == pytest.approx(1.0)
    assert rows[0]["R_tex"] == pytest.approx(1.0)
    assert rows[0]["G_mean"] == pytest.approx(0.0)


def test_patch_features_skip_a_patch_mostly_off_the_fruit():
    side = 16
    plane = {ch: np.full((side, side), 100.0, np.float32) for ch in d.PATCH_CHANNELS}
    valid = np.zeros((side, side), bool)
    valid[:, :4] = True
    sd = {ch: 1.0 for ch in d.PATCH_CHANNELS}
    assert d.patch_features(plane, plane, (0, 0, side, side), valid, sd) == []


def test_median_per_mango_averages_the_faces_and_reports_the_range_between_lots():
    # Lot 1: mangos 1, 2 and 3 with means 1, 2 and 6 (median 2). Lot 2: one mango, mean 10.
    values = [0.0, 2.0, 2.0, 6.0, 10.0]
    mangos = ["1:1", "1:1", "1:2", "1:3", "2:9"]
    lots = [1, 1, 1, 1, 2]
    out = d.median_per_mango(values, mangos, lots)
    assert out["median_per_mango"] == pytest.approx(4.0)  # median of 1, 2, 6, 10
    assert out["lot_range"] == pytest.approx([2.0, 10.0])
    assert out["per_lot"] == {1: pytest.approx(2.0), 2: pytest.approx(10.0)}
    assert out["n_mangos"] == 4


def _patches(n_mangos, lots, effect, seed=0, hours=4):
    """Patch rows where impact patches of each lot get `effect[lot]` on the mean of every channel."""
    rng = np.random.default_rng(seed)
    rows = []
    for lot in lots:
        for mango in range(n_mangos):
            for region in ("impact", "control", "control2"):
                for _ in range(6):
                    shift = effect[lot] if region == "impact" else 0.0
                    row = {"lot": lot, "mango": lot * 100 + mango, "face": 1, "hours": hours, "region": region}
                    for ch in d.PATCH_CHANNELS:
                        row[f"{ch}_mean"] = shift + rng.normal(0, 0.5)
                        row[f"{ch}_tex"] = rng.normal(0, 0.5)
                    rows.append(row)
    return rows


def _face_row(lot, mango, hours, delta=0.0):
    row = {"lot": lot, "mango": mango, "face": 1, "zone": "superior", "moment": str(hours), "hours": hours,
           "status": "ok"}
    for ch in d.BOX_CHANNELS:
        row |= {f"{ch}_impact_t": 100.0 + delta, f"{ch}_impact_sano": 100.0, f"{ch}_control_t": 90.0,
                f"{ch}_control_sano": 90.0, f"{ch}_sd": 1.0, f"{ch}_delta": delta}
    return row


def test_summarize_boxes_trains_leave_one_lot_out():
    # In lots 1 to 3 the impact raises the features; in lot 4 it lowers them. A model trained on the
    # other lots ranks lot 4 backwards; one that had seen lot 4 could not.
    lots = (1, 2, 3, 4)
    patches = _patches(6, lots, {1: 3.0, 2: 3.0, 3: 3.0, 4: -3.0})
    rows = [_face_row(lot, lot * 100 + m, 4) for lot in lots for m in range(6)]
    out = d.summarize_boxes(rows, patches, 20, 0, (0, 2, 4))
    rgb = out["per_hour"]["4"]["auroc"]["RGB"]
    assert rgb["oof_grouping"] == "lot"
    assert all(rgb["per_lot"][lot] > 0.9 for lot in (1, 2, 3))
    assert rgb["per_lot"][4] < 0.1


def test_summarize_boxes_does_not_learn_from_the_lot_it_scores():
    # Each lot carries the signal on a channel of its own, so only a model that saw the lot can use it:
    # leaving one mango out gives about 1.0 here, and a held-out lot stays in the noise around 0.5.
    lots = (1, 2, 3)
    patches = _patches(6, lots, {lot: 0.0 for lot in lots})
    channel = {1: "R", 2: "G", 3: "B"}
    for p in patches:
        if p["region"] == "impact":
            p[f"{channel[p['lot']]}_mean"] += 3.0
    rows = [_face_row(lot, lot * 100 + m, 4) for lot in lots for m in range(6)]
    per_lot = d.summarize_boxes(rows, patches, 10, 0, (0, 2, 4))["per_hour"]["4"]["auroc"]["RGB"]["per_lot"]
    assert all(per_lot[lot] < 0.85 for lot in lots)


def test_summarize_boxes_negative_control_is_about_chance_and_uses_the_two_control_regions():
    patches = _patches(6, (1, 2, 3), {1: 3.0, 2: 3.0, 3: 3.0})
    rows = [_face_row(lot, lot * 100 + m, 4) for lot in (1, 2, 3) for m in range(6)]
    out = d.summarize_boxes(rows, patches, 50, 0, (0, 2, 4))
    hour = out["per_hour"]["4"]
    for modality in d.MODALITIES:
        assert hour["auroc"][modality]["auroc"] > 0.95
        assert hour["negative_control"][modality]["auroc"] == pytest.approx(0.5, abs=0.12)
        assert hour["negative_control"][modality]["n_pos"] == hour["negative_control"][modality]["n_neg"] == 18 * 6


def test_summarize_boxes_states_the_quantities_of_the_reading_table_for_the_latent_hours():
    effect = {1: 2.0, 2: 2.0, 3: 2.0}
    patches = _patches(6, (1, 2, 3), effect) + _patches(6, (1, 2, 3), effect, seed=1, hours=8)
    for p in patches:  # NIR sees it, RGB does not: NIR - RGB is positive by construction
        if p["region"] == "impact":
            for ch in ("R", "G", "B"):
                p[f"{ch}_mean"] -= 2.0
    rows = [_face_row(lot, lot * 100 + m, h) for lot in (1, 2, 3) for m in range(6) for h in (4, 8)]
    out = d.summarize_boxes(rows, patches, 50, 0, (0, 2, 4))
    assert out["latent_hours"] == [0, 2, 4]
    assert sorted(out["reading"]) == ["4"]
    reading = out["reading"]["4"]
    nir_rgb = reading["NIR-RGB"]
    assert nir_rgb["gap"] > 0.2 and nir_rgb["ci"][0] > 0.0
    assert nir_rgb["lots_positive"] == 3 and nir_rgb["n_lots"] == 3
    assert set(nir_rgb["per_lot"]) == {1, 2, 3}
    rgb_neg = reading["RGB-negative_control"]
    assert rgb_neg["gap"] == pytest.approx(0.0, abs=0.15)
    assert rgb_neg["ci"][0] <= rgb_neg["gap"] <= rgb_neg["ci"][1]
    assert reading["negative_control_covers_chance"]["RGB"] is True


def test_summarize_boxes_reports_delta_with_interval_median_and_lot_range():
    rows = [_face_row(1, 100 + m, 4, delta=-2.0) for m in range(4)] + [_face_row(2, 200 + m, 4, delta=-4.0) for m in range(4)]
    patches = _patches(4, (1, 2), {1: 1.0, 2: 1.0})
    out = d.summarize_boxes(rows, patches, 50, 0, (0, 2, 4))
    delta = out["per_hour"]["4"]["delta"]["NIR"]
    assert delta["mean"] == pytest.approx(-3.0)
    assert delta["ci"][0] <= -3.0 <= delta["ci"][1]
    assert delta["median_per_mango"] == pytest.approx(-3.0)
    assert delta["lot_range"] == pytest.approx([-4.0, -2.0])
    assert delta["per_lot"] == {1: pytest.approx(-2.0), 2: pytest.approx(-4.0)}


def test_summarize_boxes_counts_every_exclusion_and_refuses_when_nothing_was_measured():
    excluded = [{**_face_row(1, 100, 4), "status": "no_damage_box"}, {**_face_row(1, 101, 4), "status": "not_aligned"}]
    with pytest.raises(ValueError, match="no moment could be measured"):
        d.summarize_boxes(excluded, [], 10, 0, (0, 2, 4))
    rows = excluded + [_face_row(1, 100 + m, 2) for m in range(4)]
    out = d.summarize_boxes(rows, _patches(4, (1,), {1: 1.0}, hours=2), 10, 0, (0, 2, 4))
    assert out["exclusions"] == {"no_damage_box": 1, "not_aligned": 1}


def test_summarize_boxes_counts_the_moments_without_a_second_control_and_keeps_them_for_the_main_auroc():
    lots = (1, 2)
    patches = [p for p in _patches(4, lots, {1: 3.0, 2: 3.0}) if not (p["mango"] == 100 and p["region"] == "control2")]
    rows = [{**_face_row(lot, lot * 100 + m, 4), "control2_side": "left"} for lot in lots for m in range(4)]
    rows[0]["control2_side"] = ""
    out = d.summarize_boxes(rows, patches, 10, 0, (0, 2, 4))
    assert out["no_second_control"] == 1
    hour = out["per_hour"]["4"]
    assert hour["auroc"]["RGB"]["n_pos"] == 8 * 6  # the mango without a second control still counts here
    assert hour["negative_control"]["RGB"]["n_neg"] == 7 * 6  # and not here


def test_summarize_boxes_says_at_every_hour_whether_the_negative_control_covers_chance():
    patches = _patches(6, (1, 2, 3), {1: 1.0, 2: 1.0, 3: 1.0}, hours=8)
    rows = [_face_row(lot, lot * 100 + m, 8) for lot in (1, 2, 3) for m in range(6)]
    out = d.summarize_boxes(rows, patches, 50, 0, (0, 2, 4))  # +8 h is not latent
    assert out["per_hour"]["8"]["negative_control"]["NIR"]["covers_chance"] is True


def test_summarize_boxes_falls_back_to_leave_one_mango_out_with_a_single_lot_and_says_so():
    rows = [_face_row(1, 100 + m, 2) for m in range(6)]
    out = d.summarize_boxes(rows, _patches(6, (1,), {1: 3.0}, hours=2), 10, 0, (0, 2, 4))
    rgb = out["per_hour"]["2"]["auroc"]["RGB"]
    assert rgb["oof_grouping"] == "mango"
    assert rgb["auroc"] > 0.9


def test_auroc_gap_of_identical_scores_is_zero_and_a_better_model_gives_a_positive_gap():
    rng = np.random.default_rng(6)
    n = 400
    y = np.tile([1.0, 0.0], n // 2)
    mangos = np.repeat(np.arange(n // 8).astype(str), 8)
    lots = np.where(np.arange(n) < n / 2, 1, 2)
    good = d.Scored(y * 2 + rng.normal(0, 0.5, n), y, mangos, lots)
    weak = d.Scored(rng.normal(0, 0.5, n), y, mangos, lots)
    same = d.auroc_gap(good, good, 50, np.random.default_rng(7))
    assert same["gap"] == pytest.approx(0.0)
    assert same["ci"] == pytest.approx([0.0, 0.0])
    better = d.auroc_gap(good, weak, 50, np.random.default_rng(7))
    assert better["gap"] > 0.3 and better["ci"][0] > 0.2
    assert better["lots_positive"] == 2
