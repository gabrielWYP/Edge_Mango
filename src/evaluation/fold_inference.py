"""Inference on paired fold-level differences (leave-one-lot-out).

The unit of analysis is the fold: with five lots there are five folds, so the
paired difference `treatment - baseline` gives n = 5 numbers. Those numbers are
NOT independent -- every pair of folds shares training lots -- so a plain
one-sample t-test is anti-conservative. This module provides three views of the
same differences:

  * `corrected_resampled_t`: the Nadeau-Bengio (2003) corrected resampled
    t-test. The sample variance is inflated from `sd^2 / n` to
    `(1/n + n_test/n_train) * sd^2`, which accounts for the overlap between
    training sets. Benavoli et al. (2017), eqs. (1)-(3), p.4.
  * `correlated_bayesian_t`: the posterior of the mean difference under the
    matching prior, a Student t with `n - 1` degrees of freedom, location
    `mean` and squared scale `(1/n + n_test/n_train) * sd^2`. Benavoli et al.
    (2017), eq. (6), p.12. It yields P(left), P(rope), P(right).
  * `sign_flip_test`: an exact paired randomization test of the mean. It makes
    no distributional assumption, but with n = 5 the smallest attainable
    two-sided p is 2/2^5 = 0.0625, so it can never reach p < 0.05; it is
    reported next to the other two as a distribution-free sanity check, not as
    a significance test.

Everything here is a pure function of numbers (numpy + scipy.stats).

Caveat: Nadeau-Bengio was derived for random train/test resampling and is
applied to k-fold cross-validation by convention (Benavoli et al. 2017,
footnote 2). With n = 5 folds the t distribution has 4 degrees of freedom, so
intervals are wide by construction.

Naming: `test_train_ratio` starts with `test_`; import this module as a module
(`from src.evaluation import fold_inference`) rather than importing that
function by name into a test file, or pytest will collect it as a test.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# A spread below this fraction of the data's magnitude is floating-point noise
# (e.g. 0.7 - 0.4 vs 0.5 - 0.2), not variance.
_ZERO_SPREAD_RTOL = 1e-12
# Sign-flip sums that differ by less than this fraction of sum(|d|) are ties.
_TIE_RTOL = 1e-12
# Upper bound on the (resamples x n) sign matrix held in memory at once.
_MC_CHUNK_ELEMENTS = 2_000_000


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _as_diffs(diffs: Sequence[float] | np.ndarray, min_n: int = 1) -> np.ndarray:
    """A finite 1-D float array of at least `min_n` values."""
    try:
        values = np.asarray(diffs, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"diffs must be a sequence of numbers: {error}") from None
    if values.ndim != 1:
        raise ValueError(f"diffs must be one-dimensional, got shape {values.shape}.")
    if values.size < min_n:
        raise ValueError(f"Need at least {min_n} paired difference(s), got {values.size}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("diffs contains non-finite values (nan or inf).")
    return values


def _check_ratio(ratio: float) -> float:
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError(f"test_train_ratio must be finite and >= 0, got {ratio}.")
    return float(ratio)


def _check_level(value: float, name: str) -> float:
    if not (math.isfinite(value) and 0.0 < value < 1.0):
        raise ValueError(f"{name} must be in the open interval (0, 1), got {value}.")
    return float(value)


def _summarize(diffs: Sequence[float] | np.ndarray) -> tuple[int, float, float]:
    """`(n, mean, sd)` with ddof=1; rejects n < 2 and zero variance."""
    values = _as_diffs(diffs, min_n=2)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if sd <= _ZERO_SPREAD_RTOL * float(np.abs(values).max()):
        raise ValueError(
            "The paired differences have zero variance, so the t statistic and the posterior are "
            "undefined (every fold gave the same difference)."
        )
    return int(values.size), mean, sd


# --------------------------------------------------------------------------
# Test/train ratio
# --------------------------------------------------------------------------


def test_train_ratio(folds: Sequence[Mapping]) -> float:
    """Pooled `n_test / n_train` over the folds: `sum(num_test) / sum(num_train)`.

    This is the Nadeau-Bengio ratio. Only the `train` split counts in the
    denominator; `val` is excluded because those images select a checkpoint
    but do not fit the model. Excluding it makes the denominator smaller and
    the ratio larger, which inflates the variance and widens the intervals --
    the conservative choice. Fold sizes differ (24 to 69 test images), so the
    ratio is pooled (sum over sum) rather than averaged per fold.

    Raises ValueError for an empty fold list, a missing size, or a size <= 0.
    """
    if len(folds) == 0:
        raise ValueError("Need at least one fold to compute the test/train ratio.")
    total_test = total_train = 0
    for fold in folds:
        for key in ("num_test", "num_train"):
            if key not in fold:
                raise ValueError(f"Fold {fold.get('fold', '?')} has no '{key}'.")
            if fold[key] <= 0:
                raise ValueError(f"Fold {fold.get('fold', '?')} has non-positive {key}={fold[key]}.")
        total_test += int(fold["num_test"])
        total_train += int(fold["num_train"])
    return total_test / total_train


# --------------------------------------------------------------------------
# Corrected resampled t (Nadeau & Bengio 2003)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectedTResult:
    """Corrected resampled t-test of H0: mu = 0 on n paired differences."""

    n: int
    mean: float
    sd: float                     # sample standard deviation (ddof=1)
    test_train_ratio: float
    variance_factor: float        # 1/n + n_test/n_train
    se: float                     # sd * sqrt(variance_factor)
    t: float
    df: int                       # n - 1
    p_two_sided: float            # 2 * (1 - T_df(|t|))
    p_greater: float              # H1: mu > 0, 1 - T_df(t)
    confidence: float
    ci_low: float
    ci_high: float


def corrected_t_from_summary(
    mean: float,
    sd: float,
    n: int,
    test_train_ratio: float,
    confidence: float = 0.95,
) -> CorrectedTResult:
    """Corrected resampled t-test from the mean and sd (ddof=1) of n differences.

    Raises ValueError for n < 2, sd <= 0, a negative ratio, a confidence outside
    (0, 1) or non-finite input.
    """
    if n < 2:
        raise ValueError(f"Need at least 2 paired differences, got n={n}.")
    if not (math.isfinite(mean) and math.isfinite(sd)):
        raise ValueError(f"mean and sd must be finite, got mean={mean}, sd={sd}.")
    if sd <= 0:
        raise ValueError(f"sd must be > 0 (zero variance makes t undefined), got {sd}.")
    ratio = _check_ratio(test_train_ratio)
    confidence = _check_level(confidence, "confidence")

    n = int(n)
    df = n - 1
    variance_factor = 1.0 / n + ratio
    se = sd * math.sqrt(variance_factor)
    t_stat = mean / se
    half_width = float(stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df)) * se
    return CorrectedTResult(
        n=n,
        mean=float(mean),
        sd=float(sd),
        test_train_ratio=ratio,
        variance_factor=variance_factor,
        se=se,
        t=t_stat,
        df=df,
        p_two_sided=float(2.0 * stats.t.sf(abs(t_stat), df)),
        p_greater=float(stats.t.sf(t_stat, df)),
        confidence=confidence,
        ci_low=float(mean - half_width),
        ci_high=float(mean + half_width),
    )


def corrected_resampled_t(
    diffs: Sequence[float] | np.ndarray,
    test_train_ratio: float,
    confidence: float = 0.95,
) -> CorrectedTResult:
    """Corrected resampled t-test on the paired fold-level differences."""
    n, mean, sd = _summarize(diffs)
    return corrected_t_from_summary(mean, sd, n, test_train_ratio, confidence)


# --------------------------------------------------------------------------
# Correlated Bayesian t (Benavoli et al. 2017)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelatedBayesResult:
    """Posterior of the mean difference: St(mu; df, loc, scale^2), and its ROPE masses."""

    df: int
    loc: float                    # posterior location = sample mean
    scale: float                  # sqrt((1/n + n_test/n_train) * sd^2)
    rope: float
    p_left: float                 # P(mu < -rope): the baseline is practically better
    p_rope: float                 # P(-rope <= mu <= rope): practically equivalent
    p_right: float                # P(mu > rope): the treatment is practically better
    credibility: float
    hdi_low: float                # central interval of the symmetric posterior == its HDI
    hdi_high: float


def correlated_bayesian_t(
    diffs: Sequence[float] | np.ndarray,
    test_train_ratio: float,
    rope: float = 0.0,
    credibility: float = 0.95,
) -> CorrelatedBayesResult:
    """Posterior of the mean difference under the matching prior (eq. 6, p.12).

    With `rope = 0` the region of practical equivalence is empty (`p_rope = 0`)
    and `p_left` equals the one-sided p-value of the corrected t-test for
    H1: mu > 0 -- the numerical equivalence Benavoli et al. (2017) state.

    Raises ValueError for n < 2, zero variance, a negative `rope`, a negative
    ratio, a credibility outside (0, 1) or non-finite input.
    """
    n, mean, sd = _summarize(diffs)
    ratio = _check_ratio(test_train_ratio)
    credibility = _check_level(credibility, "credibility")
    if not math.isfinite(rope) or rope < 0:
        raise ValueError(f"rope must be finite and >= 0, got {rope}.")

    df = n - 1
    scale = sd * math.sqrt(1.0 / n + ratio)
    posterior = stats.t(df, loc=mean, scale=scale)
    low, high = posterior.interval(credibility)
    return CorrelatedBayesResult(
        df=df,
        loc=mean,
        scale=scale,
        rope=float(rope),
        p_left=float(posterior.cdf(-rope)),
        p_rope=float(posterior.cdf(rope) - posterior.cdf(-rope)),
        p_right=float(posterior.sf(rope)),
        credibility=credibility,
        hdi_low=float(low),
        hdi_high=float(high),
    )


# --------------------------------------------------------------------------
# Sign-flip randomization test
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SignFlipResult:
    """Paired randomization (sign-flip) test of the mean difference."""

    n_nonzero: int
    observed_mean: float          # mean of ALL input differences, zeros included
    p_two_sided: float
    p_greater: float              # H1: mean > 0
    exact: bool                   # False when the p-values are Monte Carlo estimates
    n_permutations: int           # 2^n_nonzero when exact, else the number of resamples
    min_attainable_p_two_sided: float


def _all_signed_sums(values: np.ndarray) -> np.ndarray:
    """`sum(s_i * v_i)` over all 2^n sign vectors, built by doubling (n floats at a time)."""
    sums = np.zeros(1)
    for value in values:
        sums = np.concatenate((sums + value, sums - value))
    return sums


def sign_flip_test(
    diffs: Sequence[float] | np.ndarray,
    max_exact_n: int = 20,
    n_resamples: int = 100_000,
    seed: int = 0,
) -> SignFlipResult:
    """Paired randomization test of the mean using random sign flips.

    Under H0 each difference is equally likely to be positive or negative, so
    the observed mean is compared with the means of all sign-flipped samples.
    Zero differences are dropped: they do not change the statistic under flips.
    With `n_nonzero <= max_exact_n` all 2^n sign vectors are enumerated (the
    identity included), `p = #{|mean_perm| >= |observed|} / 2^n`; otherwise
    `n_resamples` random sign vectors are drawn with `numpy.random.default_rng(seed)`
    and `p = (count + 1) / (n_resamples + 1)`. Ties are compared with a small
    relative tolerance so that floating-point noise cannot break them.

    `min_attainable_p_two_sided` is the smallest two-sided p the test can
    return for this many non-zero differences (2/2^n, 1.0 when none): with five
    folds it is 0.0625, so this test cannot support "significant at 0.05".
    """
    values = _as_diffs(diffs, min_n=1)
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}.")
    if max_exact_n < 0:
        raise ValueError(f"max_exact_n must be >= 0, got {max_exact_n}.")

    nonzero = values[values != 0.0]
    n = int(nonzero.size)
    observed_sum = float(nonzero.sum())
    tolerance = _TIE_RTOL * float(np.abs(nonzero).sum())

    if n <= max_exact_n:
        sums = _all_signed_sums(nonzero)
        n_permutations = int(sums.size)
        p_two = np.count_nonzero(np.abs(sums) >= abs(observed_sum) - tolerance) / n_permutations
        p_greater = np.count_nonzero(sums >= observed_sum - tolerance) / n_permutations
        exact = True
        min_p = min(1.0, 2.0 / n_permutations)
    else:
        rng = np.random.default_rng(seed)
        chunk = max(1, _MC_CHUNK_ELEMENTS // n)
        count_two = count_greater = done = 0
        while done < n_resamples:
            size = min(chunk, n_resamples - done)
            signs = rng.integers(0, 2, size=(size, n)).astype(float) * 2.0 - 1.0
            sums = signs @ nonzero
            count_two += int(np.count_nonzero(np.abs(sums) >= abs(observed_sum) - tolerance))
            count_greater += int(np.count_nonzero(sums >= observed_sum - tolerance))
            done += size
        p_two = (count_two + 1) / (n_resamples + 1)
        p_greater = (count_greater + 1) / (n_resamples + 1)
        exact = False
        n_permutations = n_resamples
        min_p = max(math.ldexp(1.0, 1 - n), 1.0 / (n_resamples + 1))
        logger.debug("sign_flip_test: %d non-zero differences, Monte Carlo with %d resamples.", n, n_resamples)

    return SignFlipResult(
        n_nonzero=n,
        observed_mean=float(values.mean()),
        p_two_sided=float(p_two),
        p_greater=float(p_greater),
        exact=exact,
        n_permutations=n_permutations,
        min_attainable_p_two_sided=float(min_p),
    )
