# RGB/NIR residual on the fruit after the ArUco homography (captura_v3, lots 1-4)

| | |
| --- | --- |
| Date | 2026-11-01 |
| Data | 3200 captura_v3 pairs, lots 1-4 (without the one photo marked `apartada`) |
| Result | The per-lot homography does not align the fruit: the median residual is about 20-98 px, depending on the lot |
| Output | [`registration_residuals.csv`](registration_residuals.csv) (one row per photo), [`overlays.jpg`](overlays.jpg), lot 4 matrix in [`calibracion/captura_v3/lote4-20261031/`](../../../calibracion/captura_v3/lote4-20261031/) |

## Why it matters

Plan section 4 requires the residual to be measured per lot before training, and an unregistered NIR already cost a
lot on v3 (0.177 to 0.398 once registered). The boxes also depend on it: 1995 of the 2098 hand boxes of lots 1-3 are
drawn on the NIR, so they reach the RGB image through this registration.

| Lot | Median box (px) | Median IoU of a NIR box in RGB, homography only | Boxes under IoU 0.5 |
| --- | --- | --- | --- |
| 1 | 256 x 211 | 0.60 | 29% |
| 2 | 287 x 223 | 0.84 | 1% |
| 3 | 228 x 208 | 0.52 | 44% |

These IoUs apply the measured shift to each box as a rigid translation, so they estimate the loss rather than measure
it. Lot 4 has no boxes yet, and its residual is the largest.

## Residual per lot and moment

Medians per moment. `corr` is the normalized correlation of the RGB and warped NIR gradients over the central area,
at zero shift (`H only`) and at the best shift. `dx`/`dy` are in RGB pixels (3264 x 2448), and `at edge` counts photos
whose best shift hit the +-160 px search limit, which makes that shift unreliable.

| Lot | Moment | corr H only | corr best | dx | dy | at edge |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | sano | 0.17 | 0.35 | -52 | +2 | 0 |
| 1 | +0 | 0.14 | 0.34 | -52 | +4 | 0 |
| 1 | +2 | 0.12 | 0.35 | -64 | +10 | 1 |
| 1 | +4 | 0.14 | 0.37 | -58 | +4 | 0 |
| 1 | +6 | 0.14 | 0.40 | -60 | +4 | 0 |
| 1 | +8 | 0.14 | 0.39 | -58 | +4 | 1 |
| 1 | +10 | 0.15 | 0.41 | -54 | +18 | 2 |
| 1 | +12 | 0.02 | 0.51 | -54 | +16 | 5 |
| 2 | sano | 0.50 | 0.67 | -20 | +2 | 1 |
| 2 | +0 | 0.52 | 0.67 | -20 | +2 | 1 |
| 2 | +2 | 0.43 | 0.67 | -19 | +4 | 1 |
| 2 | +4 | 0.43 | 0.67 | -19 | +4 | 0 |
| 2 | +6 | 0.46 | 0.65 | -19 | +4 | 0 |
| 2 | +8 | 0.46 | 0.65 | -22 | +4 | 0 |
| 2 | +10 | 0.44 | 0.64 | -22 | +4 | 0 |
| 2 | +12 | 0.43 | 0.64 | -24 | +2 | 0 |
| 3 | sano | 0.55 | 0.61 | +66 | +6 | 0 |
| 3 | +0 | 0.56 | 0.62 | +71 | +6 | 0 |
| 3 | +2 | 0.56 | 0.63 | +62 | +6 | 0 |
| 3 | +4 | 0.53 | 0.61 | +80 | +8 | 0 |
| 3 | +6 | 0.55 | 0.61 | +71 | +6 | 2 |
| 3 | +8 | 0.11 | 0.58 | +55 | -12 | 0 |
| 3 | +10 | 0.26 | 0.65 | +66 | -5 | 0 |
| 3 | +12 | 0.42 | 0.68 | +66 | -2 | 0 |
| 4 | sano | 0.32 | 0.68 | +84 | +10 | 0 |
| 4 | +0 | 0.18 | 0.69 | +90 | +14 | 0 |
| 4 | +2 | 0.17 | 0.68 | +94 | +16 | 0 |
| 4 | +4 | 0.15 | 0.66 | +98 | +18 | 1 |
| 4 | +6 | 0.07 | 0.67 | +98 | +24 | 0 |
| 4 | +8 | 0.06 | 0.67 | +98 | +24 | 0 |
| 4 | +10 | 0.07 | 0.68 | +92 | +22 | 0 |
| 4 | +12 | 0.04 | 0.71 | +84 | +22 | 0 |

[`overlays.jpg`](overlays.jpg) shows it by eye (green = RGB, magenta = NIR): the fruit outline doubles with the
homography only and closes with the shift, most clearly on lot 4.

## Reading

**Measured**

- The residual is mostly horizontal and stable within a lot, and its sign and size change between lots.
- In lot 4, `dy` grows from +10 to +24 px between `sano` and +6 h.
- In lot 3, `dy` steps from +6 to -12 px after the 15:25 re-aim, then recovers. This agrees with
  [`calibracion/captura_v3/README.md`](../../../calibracion/captura_v3/README.md).
- The matrices themselves are fine on the board: 1.98 px median error for lot 4.

**Inferred, not measured**

- The horizontal part looks like parallax. The two views are offset mostly along x (H translates x by about 550 px on lot 4), and the
  fruit surface is not on the board plane. A sign change between lots would mean the board sat at a different height
  relative to the fruit. Neither was checked.
- The `dy` drift looks like the mount settling during the day.

## Limits

- One rigid shift per photo. The fruit is curved, so the residual varies across its surface.
- The search area is the central image rectangle, not a fruit mask, so the sheet and the shadow also vote. This may be
  why lot 3 morning has a high `corr H only` next to a large best shift. Treat those lot 3 shifts with more care.
- Lot 1 has the lowest `corr best` (0.34-0.51), and 9 of its photos hit the search limit.

## Next step

Registration for training and for the box export should add a per-photo translation on top of the homography, using a
fruit mask. The v4 dataset builder now does this (next section). `src/training/dataset.py` still loads one global
matrix, so the shift only reaches training through the baked v4 dataset.

## Applied in the v4 dataset builder (2026-11-02, lots 1-5)

`scripts/build_v4_dataset.py --residual-shift fruit` is the default; `none` reproduces the homography alone. The code
is `src/data_pipeline/v4_residual.py`. All 3417 selected photos, current labels.

**The outline and the skin need different shifts.** Matching only the fruit outline leaves the box about 50 px away
from the bruise, in the same direction on every lot. The gap is along x, dy stays within a few px, and it shrinks
toward the rim of the fruit. This supports the parallax reading above, where the skin facing the cameras stands higher
than the silhouette, although the fruit height was not measured.

**Method.** Each photo gets one match of RGB against warped-NIR gradients over a ring around the RGB fruit outline
(+-160 px search). This follows the fruit photo by photo, also when it moved between the two exposures, and neither
the sheet, the shadow nor a strong skin mark votes in it. A second match over the skin inside the ring is too noisy to
trust alone (correlation about 0.3), but its median per lot and moment gives the parallax from outline to skin.

    shift = outline match of the photo + parallax of its lot and moment

A photo with correlation under 0.30, or with its best shift on the search limit, takes the median outline match of its
lot and moment plus the same parallax. The shift moves both NIR variants, the boxes drawn on the NIR and the position
the line rule looks at. Boxes drawn on the RGB and the mango box stay.

| Lot | Photos | Median shift dx, dy | Outline alone dx, dy | Parallax dx | Fell back | Fruit correlation, H only -> shifted | Skin correlation, H only -> shifted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 791 | +2, +2 | -56, +4 | +60 | 0 | 0.22 -> 0.23 | 0.14 -> 0.16 |
| 2 | 792 | +34, 0 | -20, 0 | +56 | 0 | 0.40 -> 0.26 | 0.12 -> 0.21 |
| 3 | 797 | +120, -6 | +66, -4 | +50 | 0 | 0.17 -> 0.26 | 0.07 -> 0.26 |
| 4 | 699 | +150, +18 | +90, +20 | +60 | 0 | 0.10 -> 0.27 | 0.10 -> 0.27 |
| 5 | 338 | +172, +11 | +100, +13 | +72 | 0 | 0.12 -> 0.25 | 0.10 -> 0.22 |

Correlations are medians at zero displacement over the grown fruit mask and over the skin. Lot 2 loses on the whole
fruit because its outline was nearly closed by the homography; the shift gives that up for the skin.

**Box against the bruise.** Reference: for the NIR-drawn boxes at +10 h and +12 h, the shift that best matches the NIR
box patch to the RGB after a band-pass at bruise size (difference of Gaussians, sigma 3 and 24 at 1/4 scale),
correlation at least 0.8, not on the edge of its window. It uses intensity at bruise size, not the fine gradients of
the matches above. Median distance from the box to that position, and the IoU this gives a box of its size:

| Lot | n | H only | Outline alone | Applied shift |
| --- | --- | --- | --- | --- |
| 1 | 106 | 27 px, 0.75 | 52 px, 0.60 | 21 px, 0.81 |
| 2 | 116 | 27 px, 0.77 | 44 px, 0.70 | 24 px, 0.80 |
| 3 | 106 | 118 px, 0.29 | 55 px, 0.58 | 22 px, 0.80 |
| 4 | 95 | 148 px, 0.20 | 62 px, 0.56 | 19 px, 0.82 |
| 5 | 107 | 163 px, 0.21 | 64 px, 0.56 | 22 px, 0.80 |

This reference also chose the method, so the table is not an independent test: the candidates were compared on lots
1-3 and lots 4-5 separately, seen together, and ranked the same on both. The ring was added afterwards, after a synthetic
test and the overlays showed the match locking onto strong skin marks. These numbers are from the final code.

**Thresholds.** The search limit never fires on real photos: their largest outline match is 156 px in x and 140 px in y.
The real correlations run from 0.30 upward (1st percentile 0.45, median 0.60). Pairing an RGB with the NIR of another
mango of the same lot gives median 0.29, 95th percentile 0.43, and 17% hit the limit. The rule rejects 57% of those wrong
pairs and no real photo. A stricter threshold would not help: of the 30 real photos at 0.45 or below, 29 look right by
eye, among them real fruit motions at 0.30 and 0.35 that a fallback would break, and one looks wrong (1792876278).

**Limits.**

- The outline does not close: it stays doubled by the parallax, 50-72 px along x. One translation cannot close both.
- Boxes near the rim move too far. Reference error by box position, from the centre of the mango box (0) to its edge
  (1): under 0.3, 17 px and no bias; 0.5-0.7, 18 px; over 0.7 (23% of boxes), 38 px with a 34 px overshoot.
  1-5% of the NIR-drawn boxes per lot end with under 75% of their area on the RGB fruit (homography alone 0-8%,
  outline alone 0-2%).
- The fallback did not fire on the current selection, so on real data it is untested beyond the wrong-pair check.
- The shift changes by one 2 px step between OpenCV versions on a few photos, so build a dataset with one environment.

## Lot 4 matrix

The script behind the lots 1-3 matrices is not in the repository, so
[`tools/fit_lot_homography.py`](tools/fit_lot_homography.py) rebuilds the estimator and compares it with them on
ChArUco corners that the fit does not use:

| Pair | Rebuilt: median / p95 / max (px) | Published: median / p95 / max (px) |
| --- | --- | --- |
| Lot 1 | 3.25 / 9.90 / 10.87 | 2.90 / 13.33 / 16.53 |
| Lot 2 | 2.63 / 10.17 / 26.68 | 2.75 / 8.29 / 23.41 |
| Lot 3 morning | 1.78 / 4.08 / 10.26 | 1.89 / 7.30 / 9.41 |
| Lot 3 closing | 2.27 / 5.07 / 7.69 | 2.28 / 5.18 / 7.59 |
| **Lot 4** | **1.98 / 4.57 / 11.85** | none |

The lot-3 closing matrix misses the lot 4 board by 50 px (median), so it cannot be reused.

## Reproduce

```bash
.venv/bin/python docs/findings/2026-11-01-rgb-nir-residual/tools/fit_lot_homography.py \
    --calib-root calibracion/captura_v3
.venv/bin/python docs/findings/2026-11-01-rgb-nir-residual/tools/registration_residuals.py \
    --data-root data/cache/captura_v3 \
    --calib-root calibracion/captura_v3 \
    --out-dir docs/findings/2026-11-01-rgb-nir-residual
```

The numbers of "Applied in the v4 dataset builder" came from the exploratory scripts in
[`tools/shift_analysis/`](tools/shift_analysis/). They are kept as a record, not maintained: each reads the paths in its
first lines and takes its output paths as arguments.
