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
