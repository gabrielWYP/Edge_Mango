"""How many seeds are worth running? Variance components and the seed budget.

Model for the paired difference of fold `k` and seed `s`:

    d_{k,s} = mu + u_k + e_{k,s},   Var(u_k) = sigma2_fold,   Var(e_{k,s}) = sigma2_seed

`u_k` is the lot-by-arm interaction (the treatment helps more on some lots than
on others) and `e_{k,s}` is seed noise. The fold-level difference the inference
works on is the seed average `d_k = mean_s d_{k,s}`, whose variance is
`sigma2_fold + sigma2_seed / S`. More seeds shrink only the second term; the
first is a floor that only more lots can lower (`plateau_se`).

With the Nadeau-Bengio correction the standard error of the mean difference
over `K` folds with `S` seeds each is

    se(S) = sqrt((sigma2_fold + sigma2_seed / S) * (1/K + n_test/n_train))

and the minimum detectable effect is `(t_{1-alpha/2, K-1} + t_{power, K-1}) * se`.
That is the usual normal-approximation power formula with t quantiles in place
of normal ones -- not an exact non-central-t power calculation -- so read it
as a planning figure, not as a guarantee.

Everything here is a pure function of numbers (numpy + scipy.stats).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _check_variance(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value}.")
    return float(value)


def _check_design(n_folds: int, test_train_ratio: float) -> None:
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
    if not math.isfinite(test_train_ratio) or test_train_ratio < 0:
        raise ValueError(f"test_train_ratio must be finite and >= 0, got {test_train_ratio}.")


def _check_level(value: float, name: str) -> float:
    if not (math.isfinite(value) and 0.0 < value < 1.0):
        raise ValueError(f"{name} must be in the open interval (0, 1), got {value}.")
    return float(value)


# --------------------------------------------------------------------------
# Variance components
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VarianceComponents:
    """Estimated variance components of the paired differences."""

    n_folds: int
    n_seeds: int
    sigma2_seed: float
    sigma2_fold: float
    truncated: bool               # True when the fold component was clipped at 0
    fold_means: tuple[float, ...]


def estimate_variance_components(
    values: Sequence[Sequence[float]] | np.ndarray,
    sigma2_seed: float | None = None,
) -> VarianceComponents:
    """Method-of-moments estimate from a balanced K x S matrix of paired differences.

    `values[k][s]` is the difference of fold `k` and seed `s`. Then

        sigma2_seed = sum_k sum_s (d_ks - mean_s d_ks)^2 / (K (S - 1))      (pooled within-fold variance)
        sigma2_fold = max(0, var_ddof1(fold means) - sigma2_seed / S)

    `truncated` is True when the second expression was negative and was clipped
    to 0 (common with few folds: the data then show no lot-by-arm variation
    beyond what seed noise explains, which is weak evidence of absence).

    Passing `sigma2_seed` replaces the pooled estimate with a prior or a value
    measured elsewhere; it is then the only way to analyse a single seed
    (S = 1), where the seed variance cannot be estimated from the data.

    Raises ValueError for ragged or non-finite input, fewer than 2 folds, or
    fewer than 2 seeds without `sigma2_seed`.
    """
    try:
        data = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"values must be a rectangular K x S matrix of numbers: {error}") from None
    if data.ndim != 2:
        raise ValueError(f"values must be a 2-D (folds x seeds) matrix, got shape {data.shape}.")
    n_folds, n_seeds = data.shape
    if n_folds < 2:
        raise ValueError(f"Need at least 2 folds, got {n_folds}.")
    if not np.all(np.isfinite(data)):
        raise ValueError("values contains non-finite entries (nan or inf).")

    if sigma2_seed is None:
        if n_seeds < 2:
            raise ValueError(
                "Need at least 2 seeds to estimate the seed variance; pass sigma2_seed to analyse a single seed."
            )
        within = data - data.mean(axis=1, keepdims=True)
        seed_variance = float((within ** 2).sum() / (n_folds * (n_seeds - 1)))
    else:
        seed_variance = _check_variance(sigma2_seed, "sigma2_seed")

    fold_means = data.mean(axis=1)
    fold_variance = float(fold_means.var(ddof=1)) - seed_variance / n_seeds
    truncated = fold_variance < 0.0
    if truncated:
        logger.debug("sigma2_fold truncated to 0 (raw estimate %.3g).", fold_variance)
    return VarianceComponents(
        n_folds=n_folds,
        n_seeds=n_seeds,
        sigma2_seed=seed_variance,
        sigma2_fold=max(0.0, fold_variance),
        truncated=truncated,
        fold_means=tuple(float(m) for m in fold_means),
    )


# --------------------------------------------------------------------------
# Seed budget
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedBudgetRow:
    """Projected precision with `n_seeds` seeds per fold."""

    n_seeds: int
    se: float                     # standard error of the mean fold-level difference
    mde: float                    # minimum detectable effect at the requested alpha and power


def plateau_se(sigma2_fold: float, n_folds: int, test_train_ratio: float) -> float:
    """Standard error with infinitely many seeds: `sqrt(sigma2_fold * (1/K + n_test/n_train))`.

    Seeds cannot push the standard error below this floor; only more lots
    (a larger `n_folds`, a smaller test/train ratio) can.
    """
    _check_variance(sigma2_fold, "sigma2_fold")
    _check_design(n_folds, test_train_ratio)
    return math.sqrt(sigma2_fold * (1.0 / n_folds + test_train_ratio))


def project_seed_budget(
    sigma2_fold: float,
    sigma2_seed: float,
    n_folds: int,
    test_train_ratio: float,
    seeds: Iterable[int],
    alpha: float = 0.05,
    power: float = 0.8,
) -> list[SeedBudgetRow]:
    """Standard error and minimum detectable effect for each candidate seed count.

    `se(S) = sqrt((sigma2_fold + sigma2_seed / S) * (1/n_folds + test_train_ratio))` and
    `mde(S) = (t_{1 - alpha/2, n_folds - 1} + t_{power, n_folds - 1}) * se(S)`, the usual
    normal-approximation formula with t quantiles, NOT an exact non-central-t
    power calculation. Rows follow the order of `seeds`.
    """
    _check_variance(sigma2_fold, "sigma2_fold")
    _check_variance(sigma2_seed, "sigma2_seed")
    _check_design(n_folds, test_train_ratio)
    alpha = _check_level(alpha, "alpha")
    power = _check_level(power, "power")
    counts = list(seeds)
    if not counts:
        raise ValueError("seeds must contain at least one seed count.")
    if any(int(s) != s or s < 1 for s in counts):
        raise ValueError(f"Seed counts must be integers >= 1, got {counts}.")

    df = n_folds - 1
    multiplier = float(stats.t.ppf(1.0 - alpha / 2.0, df) + stats.t.ppf(power, df))
    factor = 1.0 / n_folds + test_train_ratio
    rows = []
    for count in counts:
        se = math.sqrt((sigma2_fold + sigma2_seed / count) * factor)
        rows.append(SeedBudgetRow(n_seeds=int(count), se=se, mde=multiplier * se))
    return rows
