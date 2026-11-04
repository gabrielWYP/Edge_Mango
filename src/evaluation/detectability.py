"""E0 of docs/tesis/plan-experimental.md: how detectable the impact is, hour by hour, without deep learning.

Every face is measured with a disk-vs-ring contrast at the center of the fruit crop,

    contrast = mean(disk) / mean(ring) - 1

so a uniform brightness change of the whole fruit (light, exposure) cancels. The change against the
face's own pre-impact photo, `delta_t = contrast_t - contrast_sano`, removes what the face already had
(natural spots, curvature). Faces hit in the `centro` zone are the positives; faces hit elsewhere are
the controls, because their center was not hit. The control deltas measure the capture drift
(placement, time of day), and the impact effect is the difference between both groups.

Once the damage boxes exist (plan 3.2) the impact region is the box and the control is the control
zone of plan section 2, on the same face and photo. For every moment t, in the frame of the face's
`sano` photo:

    delta_t = [impact_t - impact_sano] - [control_t - control_sano], divided by the sd of the fruit in `sano`

Both regions share the capture of photo t, so a drift of the whole photo cancels. A box darker than
`sano` gives a negative delta_t. Detectability per patch is the AUROC of a logistic regression, impact
patches vs control patches, trained leave-one-lot-out. Its features are the change against `sano`, at
the same pixels, of the mean and the local standard deviation of each channel in a patch: raw values
would let the model learn where on the fruit a patch lies (a `centro` patch faces the camera), which
has nothing to do with the damage. The negative control is the same AUROC between two regions that
were not hit, and its reading is in `summarize_boxes`.

Everything here is a pure function of numbers (numpy only), so the script that reads the images
(`scripts/e0_detectability.py`) stays thin and these functions are unit-tested.
"""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple, Sequence

import numpy as np

DISK_RADIUS = 0.12          # fraction of the crop side
RING_RADII = (0.18, 0.32)

BOX_CHANNELS = ("R", "G", "B", "NIR", "NIR/R")
MODALITIES = {"RGB": ("R", "G", "B"), "NIR": ("NIR",), "RGB+NIR": ("R", "G", "B", "NIR")}
PATCH_CHANNELS = ("R", "G", "B", "NIR")
PATCH_FEATURES = ("mean", "tex")
PATCH_SIDE = 16             # pixels of the crop
MIN_PATCH_FRUIT = 0.5       # share of a patch that must be usable fruit
MIN_CONTROL_FRUIT = 0.9     # share of a control region that must be usable fruit
CENTRO = "centro"
ZONE_AXIS = {"superior": 1, "inferior": 1, "izquierda": 0, "derecha": 0}  # 0: x, 1: y
SIDES = {"left": (0, -1), "right": (0, 1), "top": (1, -1), "bottom": (1, 1)}  # side of the photo: axis, sign
CONTROL_STEPS = np.linspace(0.0, 1.0, 41)  # positions tried along a side, as a fraction of the fruit half-size


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
    ordered = x[order]
    # NaN != NaN, so each NaN stays alone at the end, in input order.
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], x.size]
    ranks = np.empty(x.size)
    ranks[order] = np.repeat((starts + 1 + ends) / 2, ends - starts)
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


def difference_in_differences(
    impact_t: float, impact_sano: float, control_t: float, control_sano: float, sd: float
) -> float:
    """`[impact_t - impact_sano] - [control_t - control_sano]` in units of the fruit's sd in `sano`."""
    return float(((impact_t - impact_sano) - (control_t - control_sano)) / sd)


def map_box(box: Sequence[float], matrix: np.ndarray) -> np.ndarray:
    """Axis-aligned box around the corners of `box` moved by a 2x3 or 3x3 matrix."""
    m = np.asarray(matrix, float)
    if m.shape == (2, 3):
        m = np.vstack([m, [0.0, 0.0, 1.0]])
    x0, y0, x1, y1 = box
    moved = m @ np.array([[x0, x1, x1, x0], [y0, y0, y1, y1], [1.0, 1.0, 1.0, 1.0]])
    moved = moved[:2] / moved[2]
    return np.array([moved[0].min(), moved[1].min(), moved[0].max(), moved[1].max()])


def invert_warp(warp: np.ndarray) -> np.ndarray:
    """Inverse of a 2x3 rigid or affine warp, as another 2x3 warp."""
    linear = np.linalg.inv(np.asarray(warp, float)[:, :2])
    return np.hstack([linear, (-linear @ np.asarray(warp, float)[:, 2])[:, None]])


def _pixel_box(box: Sequence[float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def box_slices(box: Sequence[float], shape: tuple[int, ...]) -> tuple[slice, slice]:
    """Row and column slices of a box, clipped to an image of this shape."""
    x0, y0, x1, y1 = _pixel_box(box)
    height, width = shape[:2]
    return slice(np.clip(y0, 0, height), np.clip(y1, 0, height)), slice(np.clip(x0, 0, width), np.clip(x1, 0, width))


def fruit_share(mask: np.ndarray, box: Sequence[float]) -> float:
    """Share of the whole box that lies on the mask; what falls outside the image counts as off the mask."""
    x0, y0, x1, y1 = _pixel_box(box)
    area = max(x1 - x0, 0) * max(y1 - y0, 0)
    return float(mask[box_slices(box, mask.shape)].sum() / area) if area else 0.0


def _overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return bool(a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


def _middle(box: np.ndarray) -> np.ndarray:
    return (box[:2] + box[2:]) / 2


def control_regions(zone: str, damage_box: Sequence[float], fruit_mask: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Regions of the fruit far from the impact, each with the size of the box: `[(side, box), ...]`.

    The first is the control zone of plan section 2 and the second, when it fits, the other region the
    negative control compares it with. `[]` means there is no control zone inside the fruit.

    - `superior`, `inferior`, `izquierda`, `derecha`: the opposite zone. The box is mirrored through the
      middle of the fruit along the axis of its zone, onto the fruit's center line. The second control
      keeps that distance from the middle on the other axis, on its farther side.
    - `centro`: the side where the box can go farthest from the impact and stay inside the fruit; the
      second control is the next one.

    The side comes from where the box lies, not from the zone name, because the names are the operator's
    view: in the photos `superior` boxes lie at the bottom of the frame and `derecha` boxes on its left,
    in every lot. A region must not touch the impact or the other control, and must have
    `MIN_CONTROL_FRUIT` of its area on the mask; the spot closest to the mirrored one is used.
    """
    if zone != CENTRO and zone not in ZONE_AXIS:
        raise ValueError(f"unknown impact zone {zone!r}")
    box = np.asarray(damage_box, float)
    mask = np.asarray(fruit_mask, bool)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []
    low, high = np.array([xs.min(), ys.min()], float), np.array([xs.max() + 1, ys.max() + 1], float)
    middle, half = (low + high) / 2, (high - low) / 2
    size = box[2:] - box[:2]

    def place(side: str, target: float, taken: list[np.ndarray]) -> np.ndarray | None:
        axis, sign = SIDES[side]
        for fraction in sorted(CONTROL_STEPS, key=lambda step: abs(step - target)):
            spot = middle.copy()
            spot[axis] += sign * fraction * half[axis]
            candidate = np.r_[spot - size / 2, spot + size / 2]
            if not any(_overlap(candidate, other) for other in taken) and fruit_share(mask, candidate) >= MIN_CONTROL_FRUIT:
                return candidate
        return None

    def farthest(sides: list[str], target: float, taken: list[np.ndarray]) -> tuple[str, np.ndarray] | None:
        placed = [(side, place(side, target, taken)) for side in sides]
        return max(
            ((side, spot) for side, spot in placed if spot is not None),
            key=lambda item: float(np.hypot(*(_middle(item[1]) - _middle(box)))),
            default=None,
        )

    if zone == CENTRO:
        first = farthest(list(SIDES), 1.0, [box])
        if first is None:
            return []
        second = farthest([side for side in SIDES if side != first[0]], 1.0, [box, first[1]])
    else:
        axis = ZONE_AXIS[zone]
        offset = float((_middle(box)[axis] - middle[axis]) / half[axis])
        sign = -1 if offset > 0 else 1
        side = next(name for name, value in SIDES.items() if value == (axis, sign))
        spot = place(side, abs(offset), [box])
        if spot is None:
            return []
        first = (side, spot)
        second = farthest([name for name, value in SIDES.items() if value[0] != axis], abs(offset), [box, spot])
    return [first] if second is None else [first, second]


def patch_tiles(box: Sequence[float], side: int = PATCH_SIDE) -> list[tuple[int, int, int, int]]:
    """Non-overlapping `side`-pixel tiles centered in the box; a box smaller than a tile is one patch."""
    x0, y0, x1, y1 = _pixel_box(box)
    nx, ny = (x1 - x0) // side, (y1 - y0) // side
    if nx == 0 or ny == 0:
        return [(x0, y0, x1, y1)]
    left, top = x0 + ((x1 - x0) - nx * side) // 2, y0 + ((y1 - y0) - ny * side) // 2
    return [
        (left + i * side, top + j * side, left + (i + 1) * side, top + (j + 1) * side)
        for j in range(ny)
        for i in range(nx)
    ]


def region_mean(plane: np.ndarray, box: Sequence[float], valid: np.ndarray) -> float:
    """Mean of one channel over the usable pixels of a box."""
    slices = box_slices(box, plane.shape)
    return float(plane[slices][valid[slices]].mean())


def patch_features(
    planes_t: dict[str, np.ndarray],
    planes_sano: dict[str, np.ndarray],
    box: Sequence[float],
    valid: np.ndarray,
    sd: dict[str, float],
) -> list[dict[str, float]]:
    """One row per patch of a region: change against `sano` of its mean and local sd, in fruit sd units.

    Both photos are measured at the same usable pixels. Patches with less than `MIN_PATCH_FRUIT` of
    usable pixels are left out.
    """
    rows = []
    for tile in patch_tiles(box):
        slices = box_slices(tile, valid.shape)
        usable = valid[slices]
        if usable.size == 0 or usable.mean() < MIN_PATCH_FRUIT:
            continue
        row = {}
        for ch in PATCH_CHANNELS:
            now, before = planes_t[ch][slices][usable], planes_sano[ch][slices][usable]
            row[f"{ch}_mean"] = float((now.mean() - before.mean()) / sd[ch])
            row[f"{ch}_tex"] = float((now.std() - before.std()) / sd[ch])
        rows.append(row)
    return rows


def median_per_mango(values: Sequence[float], mangos: Sequence, lots: Sequence) -> dict:
    """Median over mangos of their mean, plus each lot's own median and the range between lots.

    Faces of a mango are repeated measures of it, so they are averaged first. The mango id must not
    repeat between lots.
    """
    values, mangos, lots = np.asarray(values, float), np.asarray(mangos), np.asarray(lots)
    if values.size == 0:
        return {"median_per_mango": None, "lot_range": [None, None], "per_lot": {}, "n_mangos": 0}
    ids = np.unique(mangos)
    means = np.array([values[mangos == m].mean() for m in ids])
    mango_lot = np.array([lots[mangos == m][0] for m in ids])
    per_lot = {lot.item(): float(np.median(means[mango_lot == lot])) for lot in np.unique(mango_lot)}
    return {
        "median_per_mango": float(np.median(means)),
        "lot_range": [min(per_lot.values()), max(per_lot.values())],
        "per_lot": per_lot,
        "n_mangos": int(ids.size),
    }


class Scored(NamedTuple):
    """Out-of-fold patch scores with what a mango-cluster bootstrap and the per-lot reading need."""

    scores: np.ndarray
    y: np.ndarray
    mangos: np.ndarray
    lots: np.ndarray

    def auroc(self) -> float | None:
        positive = self.y.astype(bool)
        return auroc(self.scores[positive], self.scores[~positive])

    def where(self, keep: np.ndarray) -> "Scored":
        return Scored(self.scores[keep], self.y[keep], self.mangos[keep], self.lots[keep])


def _nan_to_none(value: float) -> float | None:
    return None if value is None or np.isnan(value) else float(value)


def _rows_auroc(rows: np.ndarray) -> float:
    value = auroc(rows[rows[:, 0] == 1, 1], rows[rows[:, 0] == 0, 1])
    return np.nan if value is None else value


def auroc_gap(a: Scored, b: Scored, n_boot: int, rng: np.random.Generator) -> dict:
    """`AUROC(a) - AUROC(b)` with a percentile interval and its value per lot.

    The same mangos are resampled in both, inside each lot, so the interval keeps the pairing. As in
    `cluster_bootstrap` it is descriptive: inference on the lots is their per-lot values.
    """
    first, second = a.auroc(), b.auroc()
    if first is None or second is None:
        return {"gap": None, "ci": [None, None], "per_lot": {}, "lots_positive": 0, "n_lots": 0}
    values = np.r_[np.column_stack([a.y, a.scores]), np.column_stack([b.y, b.scores])]
    from_a = np.r_[np.ones(a.y.size, bool), np.zeros(b.y.size, bool)]
    lo, hi = cluster_bootstrap(
        lambda p, n: _rows_auroc(p) - _rows_auroc(n),
        values,
        from_a,
        np.r_[a.mangos, b.mangos],
        np.r_[a.lots, b.lots],
        n_boot,
        rng,
    )
    per_lot = {}
    for lot in np.unique(np.r_[a.lots, b.lots]):
        here_a, here_b = a.where(a.lots == lot).auroc(), b.where(b.lots == lot).auroc()
        if here_a is not None and here_b is not None:
            per_lot[lot.item()] = here_a - here_b
    return {
        "gap": first - second,
        "ci": [_nan_to_none(lo), _nan_to_none(hi)],
        "per_lot": per_lot,
        "lots_positive": sum(gap > 0 for gap in per_lot.values()),
        "n_lots": len(per_lot),
    }


def _patch_curve(
    patches: list[dict], positive: str, negative: str, channels: Sequence[str], n_boot: int, rng: np.random.Generator
) -> tuple[dict, Scored]:
    """AUROC of one modality at one hour, `positive` vs `negative` patches, trained leave-one-lot-out.

    With a single lot at this hour there is nothing to leave out, so it falls back to leave-one-mango-out
    as the centro version does, and says so in `oof_grouping`.
    """
    chosen = [p for p in patches if p["region"] in (positive, negative)]
    y = np.array([p["region"] == positive for p in chosen], float)
    lots = np.array([p["lot"] for p in chosen], dtype=int)
    mangos = np.array([f"{p['lot']}:{p['mango']}" for p in chosen], dtype=str)
    grouping = "lot" if np.unique(lots).size >= 2 else "mango"
    if len(chosen):
        columns = [f"{ch}_{feature}" for ch in channels for feature in PATCH_FEATURES]
        X = np.array([[p[c] for c in columns] for p in chosen], float)
        scores = grouped_oof_scores(X, y, lots if grouping == "lot" else mangos)
    else:
        scores = np.empty(0)
    scored = Scored(scores, y, mangos, lots).where(~np.isnan(scores))
    positive_rows = scored.y.astype(bool)
    value = scored.auroc()
    lo, hi = (None, None)
    if value is not None:
        lo, hi = cluster_bootstrap(auroc, scored.scores, positive_rows, scored.mangos, scored.lots, n_boot, rng)
    per_mango = [
        (mango, scored.lots[scored.mangos == mango][0], scored.where(scored.mangos == mango).auroc())
        for mango in np.unique(scored.mangos)
    ]
    per_mango = [item for item in per_mango if item[2] is not None]
    medians = median_per_mango([v for _, _, v in per_mango], [m for m, _, _ in per_mango], [lot for _, lot, _ in per_mango])
    per_lot = {
        lot.item(): scored.where(scored.lots == lot).auroc() for lot in np.unique(scored.lots)
    }
    return {
        "auroc": value,
        "ci": [lo, hi],
        "n_pos": int(positive_rows.sum()),
        "n_neg": int((~positive_rows).sum()),
        "oof_grouping": grouping,
        "per_lot": {lot: v for lot, v in per_lot.items() if v is not None},
        "median_per_mango": medians["median_per_mango"],
        "lot_range": medians["lot_range"],
        "per_lot_median_per_mango": medians["per_lot"],
        "n_mangos": medians["n_mangos"],
    }, scored


def _delta_at_hour(rows: list[dict], channel: str, n_boot: int, rng: np.random.Generator) -> dict:
    """delta_t of one channel: mean with a mango-cluster bootstrap interval, median per mango, per lot."""
    impact = np.array([(r[f"{channel}_impact_t"] - r[f"{channel}_impact_sano"]) / r[f"{channel}_sd"] for r in rows])
    control = np.array([(r[f"{channel}_control_t"] - r[f"{channel}_control_sano"]) / r[f"{channel}_sd"] for r in rows])
    lots = np.array([r["lot"] for r in rows], dtype=int)
    mangos = np.array([f"{r['lot']}:{r['mango']}" for r in rows], dtype=str)
    labels = np.r_[np.ones(impact.size, bool), np.zeros(control.size, bool)]
    lo, hi = cluster_bootstrap(
        lambda a, b: float(a.mean() - b.mean()), np.r_[impact, control], labels, np.r_[mangos, mangos], np.r_[lots, lots], n_boot, rng
    )
    return {"mean": float((impact - control).mean()), "ci": [lo, hi], "n_faces": len(rows),
            **median_per_mango(impact - control, mangos, lots)}


def summarize_boxes(
    rows: list[dict], patches: list[dict], n_boot: int, seed: int, latent_hours: Sequence[int]
) -> dict:
    """E0 with boxes: delta_t and AUROC per hour, with the quantities the reading table of the plan needs.

    `rows` holds one row per face and moment with its `status`; `patches` one row per patch of the `ok`
    ones, with `region` being `impact`, `control` or `control2`. For every hour:

    - `delta[channel]`: delta_t, with its interval, the median per mango and the range between lots.
    - `auroc[modality]` and `negative_control[modality]`: impact vs control and control vs the other
      control. `per_lot` is the AUROC of the fold that held the lot out. The negative control also says
      whether its interval `covers_chance`. A moment with no room for a second control counts only in
      `auroc` and is counted in `no_second_control`.
    - `gaps`: `NIR-RGB` and `RGB-negative_control`.

    `reading` repeats, for the hours in `latent_hours`, what the table compares: both gaps and whether
    the interval of the negative control covers 0.5. It chooses no sentence of the table. Intervals are
    descriptive (plan section 8.4) and resample mangos inside each lot, never lots.
    """
    rng = np.random.default_rng(seed)
    measured = [r for r in rows if r["status"] == "ok" and r["hours"] is not None]
    exclusions = dict(sorted(Counter(r["status"] for r in rows if r["status"] != "ok").items()))
    if not measured:
        raise ValueError(f"no moment could be measured; exclusions by reason: {exclusions}")
    hours = sorted({r["hours"] for r in measured})
    correlations = [r["reg_corr"] for r in measured if r.get("reg_corr") is not None]
    out = {
        "lots": sorted({r["lot"] for r in measured}),
        "hours": hours,
        "latent_hours": list(latent_hours),
        "n_boot": n_boot,
        "seed": seed,
        "n_faces": len({(r["lot"], r["mango"], r["face"]) for r in rows}),
        "n_moments": len(measured),
        "exclusions": exclusions,
        "no_second_control": sum(r.get("control2_side") == "" for r in measured),
        "registration": None if not correlations else {
            "corr_min": float(np.min(correlations)),
            "corr_p05": float(np.quantile(correlations, 0.05)),
            "corr_median": float(np.median(correlations)),
        },
        "per_hour": {},
        "reading": {},
        "onset_hour": {},
    }
    for h in hours:
        at_hour = [p for p in patches if p["hours"] == h]
        row_at_hour = [r for r in measured if r["hours"] == h]
        hour = {"delta": {ch: _delta_at_hour(row_at_hour, ch, n_boot, rng) for ch in BOX_CHANNELS},
                "auroc": {}, "negative_control": {}}
        scored = {}
        for modality, channels in MODALITIES.items():
            hour["auroc"][modality], scored["auroc", modality] = _patch_curve(
                at_hour, "impact", "control", channels, n_boot, rng)
            hour["negative_control"][modality], scored["negative_control", modality] = _patch_curve(
                at_hour, "control", "control2", channels, n_boot, rng)
            ci = hour["negative_control"][modality]["ci"]
            hour["negative_control"][modality]["covers_chance"] = None if ci[0] is None else not excludes_chance(*ci)
        hour["gaps"] = {
            "NIR-RGB": auroc_gap(scored["auroc", "NIR"], scored["auroc", "RGB"], n_boot, rng),
            "RGB-negative_control": auroc_gap(scored["auroc", "RGB"], scored["negative_control", "RGB"], n_boot, rng),
        }
        out["per_hour"][str(h)] = hour
        if h in latent_hours:
            out["reading"][str(h)] = {
                **hour["gaps"],
                "negative_control_covers_chance": {
                    modality: entry["covers_chance"] for modality, entry in hour["negative_control"].items()
                },
            }
    for modality in MODALITIES:
        lower = [out["per_hour"][str(h)]["auroc"][modality]["ci"][0] for h in hours]
        out["onset_hour"][modality] = onset_hour(hours, [lo is not None and lo > 0.5 for lo in lower])
    return out


def summary_table(summary: dict) -> list[dict]:
    """The table of the plan: per hour, the median per mango and the range between lots of every quantity."""
    table = []
    for h in summary["hours"]:
        hour = summary["per_hour"][str(h)]
        for metric in ("delta", "auroc", "negative_control"):
            for name, entry in hour[metric].items():
                by_lot = entry["per_lot"] if metric == "delta" else entry["per_lot_median_per_mango"]
                table.append({
                    "metric": metric,
                    "name": name,
                    "hours": h,
                    "median_per_mango": entry["median_per_mango"],
                    "lot_min": entry["lot_range"][0],
                    "lot_max": entry["lot_range"][1],
                    "n_mangos": entry["n_mangos"],
                    "n_lots": len(by_lot),
                })
    return table
