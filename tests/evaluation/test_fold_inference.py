"""Tests for src/evaluation/fold_inference.py: corrected resampled t, correlated Bayes, sign-flip.

Reference: Benavoli, Corani, Demsar & Zaffalon (2017), JMLR 18(77), eqs. (1)-(3) p.4 and (6) p.12;
Nadeau & Bengio (2003).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

from src.evaluation import fold_inference as fi

# Benavoli et al. (2017), p.4 / p.12: 10x10 cross-validation (n = 100, rho = 1/10 so n_te/n_tr = 1/9).
PAPER_MEAN, PAPER_SD, PAPER_N, PAPER_RATIO = -0.0194, 0.01583, 100, 1 / 9


def _folds(*sizes: tuple[int, int, int]) -> list[dict]:
    """`(num_test, num_train, num_val)` triples in the shape of configs/folds/lot_5fold.json."""
    return [{"fold": i, "num_test": te, "num_train": tr, "num_val": va} for i, (te, tr, va) in enumerate(sizes)]


def _diffs_with_summary(mean: float, sd: float, n: int, seed: int = 0) -> np.ndarray:
    """A sample whose mean and ddof=1 standard deviation are exactly `mean` and `sd`."""
    z = np.random.default_rng(seed).standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)
    return mean + sd * z


ARBITRARY = np.array([0.31, -0.12, 0.45, 0.08, 0.27, 0.19, -0.03])


# --------------------------------------------------------------------------
# test_train_ratio
# --------------------------------------------------------------------------


def test_ratio_is_total_test_over_total_train_and_ignores_val() -> None:
    folds = _folds((24, 161, 51), (51, 143, 42), (42, 125, 69), (69, 117, 50), (50, 162, 24))
    assert fi.test_train_ratio(folds) == pytest.approx(236 / 708)
    # val sizes must not move the ratio
    other_val = [{**f, "num_val": f["num_val"] * 10} for f in folds]
    assert fi.test_train_ratio(other_val) == pytest.approx(236 / 708)


def test_ratio_of_a_single_fold() -> None:
    assert fi.test_train_ratio(_folds((10, 40, 5))) == pytest.approx(0.25)


@pytest.mark.parametrize("bad", [(0, 10, 1), (5, 0, 1), (-1, 10, 1), (5, -10, 1)])
def test_ratio_rejects_non_positive_sizes(bad: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        fi.test_train_ratio(_folds((10, 40, 5), bad))


def test_ratio_rejects_empty_input_and_missing_keys() -> None:
    with pytest.raises(ValueError):
        fi.test_train_ratio([])
    with pytest.raises(ValueError, match="num_train"):
        fi.test_train_ratio([{"fold": 0, "num_test": 5}])


# --------------------------------------------------------------------------
# Corrected resampled t
# --------------------------------------------------------------------------


def test_corrected_t_reproduces_the_benavoli_worked_example() -> None:
    r = fi.corrected_t_from_summary(PAPER_MEAN, PAPER_SD, PAPER_N, PAPER_RATIO)
    assert r.t == pytest.approx(-3.52, abs=0.01)
    assert r.p_two_sided == pytest.approx(0.00065, rel=0.05)
    assert r.df == 99 and r.n == 100


def test_corrected_t_fields_follow_the_formulas() -> None:
    r = fi.corrected_t_from_summary(0.1, 0.2, 5, 1 / 3)
    assert r.variance_factor == pytest.approx(1 / 5 + 1 / 3)
    assert r.se == pytest.approx(0.2 * math.sqrt(1 / 5 + 1 / 3))
    assert r.t == pytest.approx(0.1 / r.se)
    assert r.df == 4
    assert r.p_two_sided == pytest.approx(2 * stats.t.sf(abs(r.t), 4))
    assert r.p_greater == pytest.approx(stats.t.sf(r.t, 4))
    half = stats.t.ppf(0.975, 4) * r.se
    assert (r.ci_low, r.ci_high) == (pytest.approx(0.1 - half), pytest.approx(0.1 + half))
    assert r.confidence == 0.95 and r.test_train_ratio == pytest.approx(1 / 3)
    assert (r.mean, r.sd) == (0.1, 0.2)


def test_confidence_changes_the_interval_width_only() -> None:
    wide = fi.corrected_t_from_summary(0.1, 0.2, 5, 1 / 3, confidence=0.99)
    narrow = fi.corrected_t_from_summary(0.1, 0.2, 5, 1 / 3, confidence=0.90)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)
    assert wide.p_two_sided == pytest.approx(narrow.p_two_sided)
    assert wide.confidence == 0.99


def test_ratio_zero_reproduces_scipy_one_sample_t_test() -> None:
    r = fi.corrected_resampled_t(ARBITRARY, 0.0)
    ref = stats.ttest_1samp(ARBITRARY, 0.0)
    ref_greater = stats.ttest_1samp(ARBITRARY, 0.0, alternative="greater")
    assert r.t == pytest.approx(ref.statistic)
    assert r.p_two_sided == pytest.approx(ref.pvalue)
    assert r.p_greater == pytest.approx(ref_greater.pvalue)
    assert r.df == len(ARBITRARY) - 1


def test_corrected_resampled_t_equals_the_summary_version() -> None:
    from_diffs = fi.corrected_resampled_t(list(ARBITRARY), 0.25)
    from_summary = fi.corrected_t_from_summary(float(ARBITRARY.mean()), float(ARBITRARY.std(ddof=1)),
                                               len(ARBITRARY), 0.25)
    assert from_diffs == from_summary


def test_a_larger_ratio_widens_the_interval_and_raises_p() -> None:
    plain = fi.corrected_resampled_t(ARBITRARY, 0.0)
    corrected = fi.corrected_resampled_t(ARBITRARY, 1.0)
    assert corrected.se > plain.se
    assert corrected.p_two_sided > plain.p_two_sided
    assert (corrected.ci_high - corrected.ci_low) > (plain.ci_high - plain.ci_low)


def test_p_values_are_consistent_on_the_negative_side() -> None:
    r = fi.corrected_resampled_t(-ARBITRARY, 0.3)
    assert r.t < 0
    assert r.p_greater > 0.5
    assert r.p_two_sided == pytest.approx(2 * (1 - r.p_greater))


def test_results_are_frozen_dataclasses() -> None:
    r = fi.corrected_resampled_t(ARBITRARY, 0.3)
    assert dataclasses.is_dataclass(r)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.t = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mean=0.1, sd=0.2, n=1, test_train_ratio=0.3),                      # n < 2
        dict(mean=0.1, sd=0.0, n=5, test_train_ratio=0.3),                      # zero variance
        dict(mean=0.1, sd=-0.2, n=5, test_train_ratio=0.3),                     # negative sd
        dict(mean=0.1, sd=0.2, n=5, test_train_ratio=-0.1),                     # negative ratio
        dict(mean=float("nan"), sd=0.2, n=5, test_train_ratio=0.3),
        dict(mean=0.1, sd=float("inf"), n=5, test_train_ratio=0.3),
        dict(mean=0.1, sd=0.2, n=5, test_train_ratio=float("nan")),
        dict(mean=0.1, sd=0.2, n=5, test_train_ratio=0.3, confidence=1.0),
        dict(mean=0.1, sd=0.2, n=5, test_train_ratio=0.3, confidence=0.0),
    ],
)
def test_corrected_t_from_summary_rejects_invalid_input(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        fi.corrected_t_from_summary(**kwargs)


@pytest.mark.parametrize(
    "diffs",
    [
        [0.2],                                      # n < 2
        [0.2, 0.2, 0.2, 0.2],                       # zero variance
        [0.3, 0.30000000000000004, 0.29999999999999993, 0.3],   # floating-point noise is not variance
        [0.1, float("nan"), 0.3],
        [0.1, float("inf"), 0.3],
        [],
    ],
)
def test_corrected_resampled_t_rejects_invalid_diffs(diffs: list[float]) -> None:
    with pytest.raises(ValueError):
        fi.corrected_resampled_t(diffs, 0.3)


# --------------------------------------------------------------------------
# Correlated Bayesian t
# --------------------------------------------------------------------------


def test_bayes_reproduces_the_benavoli_posterior_scale() -> None:
    diffs = _diffs_with_summary(PAPER_MEAN, PAPER_SD, PAPER_N)
    r = fi.correlated_bayesian_t(diffs, PAPER_RATIO)
    assert r.scale ** 2 == pytest.approx(0.000030, abs=1e-6)
    assert r.df == 99 and r.loc == pytest.approx(PAPER_MEAN)


def test_bayes_posterior_parameters() -> None:
    r = fi.correlated_bayesian_t(ARBITRARY, 0.25)
    sd = ARBITRARY.std(ddof=1)
    assert r.df == len(ARBITRARY) - 1
    assert r.loc == pytest.approx(ARBITRARY.mean())
    assert r.scale ** 2 == pytest.approx((1 / len(ARBITRARY) + 0.25) * sd ** 2)
    assert r.scale == pytest.approx(fi.corrected_resampled_t(ARBITRARY, 0.25).se)


def test_bayes_with_zero_rope_matches_the_one_sided_frequentist_p() -> None:
    for diffs in (ARBITRARY, -ARBITRARY, ARBITRARY - 0.2):
        bayes = fi.correlated_bayesian_t(diffs, 1 / 3, rope=0.0)
        freq = fi.corrected_resampled_t(diffs, 1 / 3)
        assert bayes.p_left == pytest.approx(freq.p_greater, abs=1e-12)     # P(mu < 0) == p-value of H1: mu > 0
        assert bayes.p_rope == 0.0
        assert bayes.p_right == pytest.approx(1 - freq.p_greater, abs=1e-12)


@pytest.mark.parametrize("rope", [0.0, 0.02, 0.1, 0.5])
def test_bayes_probabilities_sum_to_one(rope: float) -> None:
    r = fi.correlated_bayesian_t(ARBITRARY, 1 / 3, rope=rope)
    assert r.p_left + r.p_rope + r.p_right == pytest.approx(1.0, abs=1e-12)
    assert min(r.p_left, r.p_rope, r.p_right) >= 0.0
    assert r.rope == rope


def test_bayes_rope_mass_grows_with_the_rope() -> None:
    masses = [fi.correlated_bayesian_t(ARBITRARY, 1 / 3, rope=rope).p_rope for rope in (0.0, 0.01, 0.05, 0.2)]
    assert masses == sorted(masses) and masses[0] == 0.0 and masses[-1] > masses[1] > 0.0


def test_bayes_direction_follows_the_sign_of_the_mean() -> None:
    better = fi.correlated_bayesian_t(ARBITRARY, 1 / 3, rope=0.02)
    worse = fi.correlated_bayesian_t(-ARBITRARY, 1 / 3, rope=0.02)
    assert better.p_right > better.p_left
    assert worse.p_left > worse.p_right
    assert worse.p_left == pytest.approx(better.p_right)


def test_bayes_credible_interval_is_the_central_interval_of_the_posterior() -> None:
    r = fi.correlated_bayesian_t(ARBITRARY, 1 / 3, credibility=0.9)
    freq = fi.corrected_resampled_t(ARBITRARY, 1 / 3, confidence=0.9)
    assert (r.hdi_low, r.hdi_high) == (pytest.approx(freq.ci_low), pytest.approx(freq.ci_high))
    assert r.credibility == 0.9
    assert r.loc - r.hdi_low == pytest.approx(r.hdi_high - r.loc)


@pytest.mark.parametrize(
    "diffs, kwargs",
    [
        ([0.2], {}),
        ([0.2, 0.2, 0.2], {}),
        ([0.1, float("nan"), 0.3], {}),
        ([0.1, 0.2, 0.4], {"rope": -0.01}),
        ([0.1, 0.2, 0.4], {"rope": float("nan")}),
        ([0.1, 0.2, 0.4], {"credibility": 1.0}),
    ],
)
def test_bayes_rejects_invalid_input(diffs: list[float], kwargs: dict) -> None:
    with pytest.raises(ValueError):
        fi.correlated_bayesian_t(diffs, 0.3, **kwargs)


def test_bayes_rejects_a_negative_ratio() -> None:
    with pytest.raises(ValueError):
        fi.correlated_bayesian_t([0.1, 0.2, 0.4], -0.5)


# --------------------------------------------------------------------------
# Sign-flip randomization test
# --------------------------------------------------------------------------


def test_sign_flip_exact_all_positive_five_folds_cannot_beat_0_0625() -> None:
    r = fi.sign_flip_test([1, 2, 3, 4, 5])
    assert r.exact is True and r.n_nonzero == 5 and r.n_permutations == 32
    assert r.p_two_sided == pytest.approx(0.0625)
    assert r.p_greater == pytest.approx(1 / 32)
    assert r.min_attainable_p_two_sided == pytest.approx(0.0625)
    assert r.observed_mean == pytest.approx(3.0)


def test_sign_flip_exact_hand_enumeration_mixed_signs() -> None:
    # sums over the 8 sign vectors: 6, 4, 2, 0, 0, -2, -4, -6; observed sum is 4.
    r = fi.sign_flip_test([3, -1, 2])
    assert r.p_two_sided == pytest.approx(0.5)
    assert r.p_greater == pytest.approx(0.25)
    assert r.n_permutations == 8


def test_sign_flip_drops_zero_differences() -> None:
    r = fi.sign_flip_test([0, 1, 2])
    assert r.n_nonzero == 2 and r.n_permutations == 4
    assert r.min_attainable_p_two_sided == pytest.approx(0.5)
    assert r.p_two_sided == pytest.approx(0.5)            # sums 3, 1, -1, -3; observed 3
    assert r.p_greater == pytest.approx(0.25)
    assert r.observed_mean == pytest.approx(1.0)          # mean of all three differences


def test_sign_flip_all_zero_is_uninformative() -> None:
    r = fi.sign_flip_test([0.0, 0.0, 0.0])
    assert r.n_nonzero == 0 and r.exact is True
    assert r.p_two_sided == 1.0 and r.p_greater == 1.0
    assert r.min_attainable_p_two_sided == 1.0


def test_sign_flip_negative_effect_has_a_large_one_sided_p() -> None:
    r = fi.sign_flip_test([-1, -2, -3, -4, -5])
    assert r.p_two_sided == pytest.approx(0.0625)
    assert r.p_greater == pytest.approx(1.0)


def test_sign_flip_ties_are_compared_with_a_floating_point_tolerance() -> None:
    # Mathematically the observed sum is 0.1 and so are two of the flipped sums, but in floating point
    # 0.1 + 0.2 - 0.2 is 0.10000000000000003: without a tolerance those ties would be lost (p = 0.75).
    r = fi.sign_flip_test([0.1, 0.2, -0.2])
    assert r.p_two_sided == pytest.approx(1.0)
    assert r.p_greater == pytest.approx(0.5)


def test_sign_flip_min_attainable_p_shrinks_with_more_folds() -> None:
    assert fi.sign_flip_test(list(range(1, 11))).min_attainable_p_two_sided == pytest.approx(2 / 1024)


TWELVE = [0.3, 0.1, 0.25, -0.05, 0.2, 0.15, 0.4, -0.1, 0.05, 0.22, 0.12, 0.18]


def test_sign_flip_monte_carlo_is_reproducible_and_flagged() -> None:
    a = fi.sign_flip_test(TWELVE, max_exact_n=5, n_resamples=20_000, seed=3)
    b = fi.sign_flip_test(TWELVE, max_exact_n=5, n_resamples=20_000, seed=3)
    c = fi.sign_flip_test(TWELVE, max_exact_n=5, n_resamples=20_000, seed=4)
    assert a == b
    assert a.exact is False and a.n_permutations == 20_000
    assert a.p_two_sided != c.p_two_sided
    assert a.p_two_sided >= 1 / 20_001
    assert a.min_attainable_p_two_sided >= 1 / 20_001


def test_sign_flip_monte_carlo_is_close_to_the_exact_p() -> None:
    exact = fi.sign_flip_test(TWELVE)
    assert exact.exact is True and exact.n_permutations == 2 ** 12
    mc = fi.sign_flip_test(TWELVE, max_exact_n=5, n_resamples=60_000, seed=0)
    assert mc.p_two_sided == pytest.approx(exact.p_two_sided, abs=0.01)
    assert mc.p_greater == pytest.approx(exact.p_greater, abs=0.01)


@pytest.mark.parametrize(
    "diffs, kwargs",
    [
        ([], {}),
        ([1.0, float("nan")], {}),
        ([1.0, float("inf")], {}),
        ([1.0, 2.0], {"n_resamples": 0}),
        ([1.0, 2.0], {"max_exact_n": -1}),
    ],
)
def test_sign_flip_rejects_invalid_input(diffs: list[float], kwargs: dict) -> None:
    with pytest.raises(ValueError):
        fi.sign_flip_test(diffs, **kwargs)
