# Illumination correction test

Exploratory, on the final v4 dataset. It reads E0 and the validation predictions of `maestro_rgbnir` fold 0 seed 42; it reads no test metric.

The question: does correcting the light per photo recover the signal in the zones the first teacher detects poorly (derecha, inferior, superior)? Two corrections, inside the fruit only: `poly2` (degree 2 surface fitted to the log brightness) and `gauss25` (division by a blur of 0.25 fruit widths). Each went through E0 in boxes mode, compared by zone with the original images.

## Result

Neither correction is adopted. Latent AUROC, all zones, falls with both: RGB+NIR 0.589 on the original images, 0.568 with `poly2`, 0.575 with `gauss25`. The only interval above zero in a hard zone is `poly2` RGB in derecha (+0.037). The left zone loses up to 0.086. The full table is in `results.txt`.

## Other findings

- Fruit center against a ring around it: darker at `sano` already, and it darkens further with time when the hit was elsewhere too. Change at 8-12 h for faces hit elsewhere: -0.023 RGB, -0.025 NIR; a hit at the center adds -0.010, with intervals [-0.026, +0.004] and [-0.021, -0.000]. So part of what E0 measures at the center may not come from the hit.
- A fixed central box on validation lot 2 reaches IoU 0.5 in 22.5% of centro and 0% of izquierda photos; the teacher's top detection reaches 70.6% and 63.3%. Those two zones are not just a position prior.

## Limits

One fold for the detector. E0 is a linear model pooled over zones, not a detector, so this does not rule out a gain in a CNN. `zone_eval.py` prints "2000 reps" but draws 300. `center2.py` draws 1000.
