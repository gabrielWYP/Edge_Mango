"""Tests for src/evaluation/seed_planning.py: variance components and the seed budget.

Model: d_{k,s} = mu + u_k + e_{k,s}, Var(u_k) = sigma2_fold (lot-by-arm interaction),
Var(e_{k,s}) = sigma2_seed (seed noise), k = fold, s = seed.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

from src.evaluation import seed_planning as sp

# t quantiles with 4 degrees of freedom (5 folds), from standard tables.
T_975_DF4 = 2.7764451
T_80_DF4 = 0.9409646

HAND = [[0.05, 0.15], [0.15, 0.25], [0.25, 0.35]]


# --------------------------------------------------------------------------
# estimate_variance_components
# --------------------------------------------------------------------------


def test_hand_case_recovers_both_components() -> None:
    # fold means 0.1, 0.2, 0.3; within-fold deviations are +/-0.05 -> pooled variance 0.015 / 3 = 0.005;
    # var(fold means) = 0.01 -> sigma2_fold = 0.01 - 0.005 / 2 = 0.0075.
    vc = sp.estimate_variance_components(HAND)
    assert vc.sigma2_seed == pytest.approx(0.005)
    assert vc.sigma2_fold == pytest.approx(0.0075)
    assert vc.truncated is False
    assert (vc.n_folds, vc.n_seeds) == (3, 2)
    assert vc.fold_means == pytest.approx((0.1, 0.2, 0.3))
    assert isinstance(vc.fold_means, tuple)


def test_negative_fold_variance_is_truncated_to_zero() -> None:
    # identical fold means (var 0) with positive seed noise -> 0 - 0.02 / 2 < 0
    vc = sp.estimate_variance_components([[0.0, 0.2], [0.0, 0.2], [0.0, 0.2]])
    assert vc.sigma2_seed == pytest.approx(0.02)
    assert vc.sigma2_fold == 0.0
    assert vc.truncated is True


def test_accepts_numpy_input_and_does_not_mutate_it() -> None:
    values = np.array(HAND)
    before = values.copy()
    assert sp.estimate_variance_components(values) == sp.estimate_variance_components(HAND)
    assert np.array_equal(values, before)


def test_estimator_is_approximately_unbiased_on_simulated_data() -> None:
    rng = np.random.default_rng(7)
    k, s = 4000, 3
    fold_sd, seed_sd = 0.10, 0.05
    values = rng.normal(0, fold_sd, size=(k, 1)) + rng.normal(0, seed_sd, size=(k, s)) + 0.2
    vc = sp.estimate_variance_components(values)
    assert vc.sigma2_seed == pytest.approx(seed_sd ** 2, rel=0.05)
    assert vc.sigma2_fold == pytest.approx(fold_sd ** 2, rel=0.10)
    assert vc.truncated is False


def test_explicit_sigma2_seed_allows_a_single_seed() -> None:
    vc = sp.estimate_variance_components([[0.1], [0.2], [0.3]], sigma2_seed=0.004)
    assert vc.n_seeds == 1
    assert vc.sigma2_seed == 0.004                       # taken as given
    assert vc.sigma2_fold == pytest.approx(0.01 - 0.004)  # var(fold means) - sigma2_seed / 1
    assert vc.truncated is False


def test_explicit_sigma2_seed_overrides_the_pooled_estimate() -> None:
    vc = sp.estimate_variance_components(HAND, sigma2_seed=0.002)
    assert vc.sigma2_seed == 0.002
    assert vc.sigma2_fold == pytest.approx(0.01 - 0.002 / 2)


def test_explicit_sigma2_seed_larger_than_the_data_truncates() -> None:
    vc = sp.estimate_variance_components([[0.1], [0.2], [0.3]], sigma2_seed=0.5)
    assert vc.sigma2_fold == 0.0 and vc.truncated is True


def test_result_is_a_frozen_dataclass() -> None:
    vc = sp.estimate_variance_components(HAND)
    assert dataclasses.is_dataclass(vc)
    with pytest.raises(dataclasses.FrozenInstanceError):
        vc.sigma2_seed = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "values",
    [
        [[0.1], [0.2], [0.3]],                  # one seed and no sigma2_seed
        [[0.1, 0.2]],                           # one fold
        [[0.1, 0.2], [0.3]],                    # ragged
        [[0.1, 0.2], [0.3, float("nan")]],      # non-finite
        [0.1, 0.2, 0.3],                        # not 2-D
        [],
    ],
)
def test_invalid_shapes_are_rejected(values) -> None:
    with pytest.raises(ValueError):
        sp.estimate_variance_components(values)


def test_negative_or_non_finite_explicit_sigma2_seed_is_rejected() -> None:
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            sp.estimate_variance_components(HAND, sigma2_seed=bad)


# --------------------------------------------------------------------------
# plateau_se and project_seed_budget
# --------------------------------------------------------------------------


def test_plateau_se_formula() -> None:
    assert sp.plateau_se(0.0075, 5, 1 / 3) == pytest.approx(math.sqrt(0.0075 * (1 / 5 + 1 / 3)))
    assert sp.plateau_se(0.0, 5, 1 / 3) == 0.0


def test_budget_se_is_non_increasing_and_never_below_the_plateau() -> None:
    rows = sp.project_seed_budget(0.0075, 0.005, 5, 1 / 3, seeds=range(1, 11))
    plateau = sp.plateau_se(0.0075, 5, 1 / 3)
    ses = [r.se for r in rows]
    assert [r.n_seeds for r in rows] == list(range(1, 11))
    assert all(a >= b for a, b in zip(ses, ses[1:]))
    assert all(se >= plateau for se in ses)
    assert ses[-1] - plateau < ses[0] - plateau            # moving towards the floor
    mdes = [r.mde for r in rows]
    assert all(a >= b for a, b in zip(mdes, mdes[1:]))


def test_budget_se_formula_and_mde_against_a_hand_computation() -> None:
    sigma2_fold, sigma2_seed, folds, ratio = 0.0075, 0.005, 5, 1 / 3
    (row,) = sp.project_seed_budget(sigma2_fold, sigma2_seed, folds, ratio, seeds=[3])
    se = math.sqrt((sigma2_fold + sigma2_seed / 3) * (1 / 5 + 1 / 3))
    assert row.n_seeds == 3
    assert row.se == pytest.approx(se)
    assert row.mde == pytest.approx((T_975_DF4 + T_80_DF4) * se, rel=1e-6)
    # the same quantiles straight from scipy
    assert row.mde == pytest.approx((stats.t.ppf(0.975, 4) + stats.t.ppf(0.8, 4)) * se)


def test_budget_alpha_and_power_move_the_mde() -> None:
    base = sp.project_seed_budget(0.01, 0.01, 5, 0.3, seeds=[3])[0]
    strict = sp.project_seed_budget(0.01, 0.01, 5, 0.3, seeds=[3], alpha=0.01)[0]
    powerful = sp.project_seed_budget(0.01, 0.01, 5, 0.3, seeds=[3], power=0.9)[0]
    assert strict.mde > base.mde and powerful.mde > base.mde
    assert strict.se == base.se == powerful.se


def test_without_fold_variance_seeds_keep_helping_as_one_over_sqrt_s() -> None:
    rows = sp.project_seed_budget(0.0, 0.01, 5, 0.3, seeds=[1, 4])
    assert rows[1].se == pytest.approx(rows[0].se / 2)


def test_more_folds_lower_the_plateau() -> None:
    assert sp.plateau_se(0.01, 10, 0.3) < sp.plateau_se(0.01, 5, 0.3)


def test_budget_keeps_the_requested_seed_order_and_accepts_any_iterable() -> None:
    rows = sp.project_seed_budget(0.01, 0.01, 5, 0.3, seeds=(5, 2, 3))
    assert [r.n_seeds for r in rows] == [5, 2, 3]
    assert all(isinstance(r, sp.SeedBudgetRow) for r in rows)
    assert dataclasses.is_dataclass(rows[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sigma2_fold=-0.01),
        dict(sigma2_seed=-0.01),
        dict(sigma2_fold=float("nan")),
        dict(sigma2_seed=float("inf")),
        dict(n_folds=1),
        dict(test_train_ratio=-0.1),
        dict(seeds=[]),
        dict(seeds=[0, 1]),
        dict(alpha=0.0),
        dict(alpha=1.0),
        dict(power=1.0),
        dict(power=0.0),
    ],
)
def test_budget_rejects_invalid_input(kwargs: dict) -> None:
    params = dict(sigma2_fold=0.01, sigma2_seed=0.01, n_folds=5, test_train_ratio=0.3, seeds=[1, 2])
    params.update(kwargs)
    with pytest.raises(ValueError):
        sp.project_seed_budget(**params)


def test_plateau_rejects_invalid_input() -> None:
    for args in ((-0.1, 5, 0.3), (0.1, 1, 0.3), (0.1, 5, -0.3), (float("nan"), 5, 0.3)):
        with pytest.raises(ValueError):
            sp.plateau_se(*args)
