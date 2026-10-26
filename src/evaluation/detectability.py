"""E0 of docs/tesis/plan-experimental.md: how detectable the impact is, hour by hour, without deep learning.

Every face is measured with a disk-vs-ring contrast at the center of the fruit crop,

    contrast = mean(disk) / mean(ring) - 1

so a uniform brightness change of the whole fruit (light, exposure) cancels. The change against the
face's own pre-impact photo, `delta_t = contrast_t - contrast_sano`, removes what the face already had
(natural spots, curvature). Faces hit in the `centro` zone are the positives; faces hit elsewhere are
the controls, because their center was not hit. The control deltas measure the capture drift
(placement, time of day), and the impact effect is the difference between both groups.

Everything here is a pure function of numbers (numpy only), so the script that reads the images
(`scripts/e0_detectability.py`) stays thin and these functions are unit-tested.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DISK_RADIUS = 0.12          # fraction of the crop side
RING_RADII = (0.18, 0.32)


def disk_and_ring(side: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of the central disk and the ring around it on a `side` x `side` crop."""
    yy, xx = np.mgrid[0:side, 0:side]
    r = np.hypot(xx - side / 2, yy - side / 2)
    return r < DISK_RADIUS * side, (r >= RING_RADII[0] * side) & (r < RING_RADII[1] * side)


def contrast(channel: np.ndarray, disk: np.ndarray, ring: np.ndarray) -> float:
    """`mean(disk) / mean(ring) - 1` of one channel; the ring mean is floored at 1 to avoid dividing by 0."""
    c = channel.astype(np.float64)
    return float(c[disk].mean() / max(c[ring].mean(), 1.0) - 1.0)


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float | None:
    """Mann-Whitney AUROC, P(score_pos > score_neg) with ties counted 1/2. None if a group is empty.

    Not sign-agnostic: 0.5 is chance, and values below 0.5 mean the positives score lower (for a
    contrast, that the impact darkens the center).
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return None
    ranks = _average_ranks(np.concatenate([pos, neg]))
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    ranks[order] = np.arange(1, x.size + 1)
    for value in np.unique(x):
        tied = x == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def cluster_bootstrap(
    stat,
    values: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    strata: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> tuple[float | None, float | None]:
    """Percentile interval of `stat(values[labels], values[~labels])`, resampling whole clusters.

    The cluster is the mango: its 4 faces are not independent. Mangos are resampled with replacement
    inside each stratum (the lot), so every replicate keeps the lot composition. Replicates where a
    group ends up empty are dropped. Returns (None, None) when fewer than half the replicates are valid.

    The interval is descriptive (docs/tesis/plan-experimental.md section 8.4): confirmatory inference
    runs at lot level.
    """
    values, labels = np.asarray(values, float), np.asarray(labels, bool)
    clusters, strata = np.asarray(clusters), np.asarray(strata)
    members = {}
    for s in np.unique(strata):
        ids = np.unique(clusters[strata == s])
        members[s] = [(c, np.flatnonzero((clusters == c) & (strata == s))) for c in ids]
    out = []
    for _ in range(n_boot):
        idx = []
        for s, groups in members.items():
            for k in rng.integers(0, len(groups), len(groups)):
                idx.append(groups[k][1])
        idx = np.concatenate(idx)
        v, lab = values[idx], labels[idx]
        if lab.any() and (~lab).any():
            out.append(stat(v[lab], v[~lab]))
    if len(out) < n_boot / 2:
        return None, None
    a = (1 - level) / 2
    lo, hi = np.quantile(out, [a, 1 - a])
    return float(lo), float(hi)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """L2-penalized logistic regression by Newton's method. Returns weights with the bias last.

    The bias is not penalized. X should be standardized by the caller.
    """
    X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.zeros(X1.shape[1])
    penalty = np.full(X1.shape[1], l2)
    penalty[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        grad = X1.T @ (p - y) + penalty * w
        hess = (X1 * (p * (1 - p))[:, None]).T @ X1 + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def grouped_oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Out-of-fold logistic scores, one fold per group (leave-one-group-out), pooled for one AUROC.

    The score is `w . z` without the intercept, and `z` is standardized with the mean and sd of all
    rows. Both choices remove a known bias of pooling leave-one-out scores: holding out a positive
    lowers the training share of positives, so the fold's intercept (and a fold-wise standardization
    mean) shifts against exactly the held-out rows, and the pooled AUROC drops well below 0.5 on data
    with no signal. The standardization uses no labels, so it does not leak them. A fold whose
    training part lacks a class gets NaN scores.
    """
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    scores = np.full(y.size, np.nan)
    for g in np.unique(groups):
        test = groups == g
        train = ~test
        if len(np.unique(y[train])) < 2:
            continue
        w = fit_logistic(Z[train], y[train], l2)
        scores[test] = Z[test] @ w[:-1]
    return scores


def permutation_test(
    stat, labels: np.ndarray, strata: np.ndarray, observed: float, n_perm: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Null mean and two-sided p of `stat(labels)` with the labels permuted within each stratum.

    This is randomization inference: the impact zone is drawn per face by the capture protocol, so
    shuffling the zone labels inside a lot reproduces the design's own randomness. It also calibrates
    statistics whose null is not 0.5, such as the pooled AUROC of leave-one-out scores (see
    `grouped_oof_scores`). The p is `(1 + #|null - m| >= |observed - m|) / (n_perm + 1)` around the
    null mean m.
    """
    labels, strata = np.asarray(labels, bool), np.asarray(strata)
    null = []
    for _ in range(n_perm):
        perm = labels.copy()
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            perm[idx] = rng.permutation(labels[idx])
        value = stat(perm)
        if value is not None:
            null.append(value)
    null = np.asarray(null)
    m = float(null.mean())
    p = (1 + int((np.abs(null - m) >= abs(observed - m)).sum())) / (null.size + 1)
    return m, float(p)


def excludes_chance(lo: float | None, hi: float | None) -> bool:
    return lo is not None and (lo > 0.5 or hi < 0.5)


def onset_hour(hours: Sequence[float], detected: Sequence[bool]) -> float | None:
    """First hour detected and still detected at every later hour; None if never.

    Requiring it to persist keeps one noisy hour from being read as the onset.
    """
    for i, h in enumerate(hours):
        if all(detected[i:]):
            return float(h)
    return None
