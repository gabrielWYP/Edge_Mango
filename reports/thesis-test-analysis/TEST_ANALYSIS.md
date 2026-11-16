# Test-set analysis, thesis set (seed 42, 5 lots, 640 px)

Generated 2026-11-16 by running the repo's own functions (`scripts/export_run_results.py` for the export, `scripts/aggregate_kfold.paired_inference` -> `src/evaluation/fold_inference.py` for corrected t, correlated Bayes and sign-flip). Code at origin/main 8dfe733. Primary metric: TEST latent-window damage box AP50 (t in {0,2,4,6} + sano). ROPE +/-0.02, threshold 0.95. Test/train ratio from `folds_meta.json`, pooled sum(test)/sum(train), val excluded: 0.3333.

## 1. What was registered (quoted from docs/tesis/plan-experimental.md)

- Section 11, 2026-11-13: "Seis brazos, cinco folds por lote y tres semillas (42, 1337 y 2024): `master_rgb_seg`, `master_rgbnir_seg`, `master_nir_seg`, `crosskd_rgb_seg`, `crosskd_rgbnir_seg` y `master_rgb_det`. Hipótesis: H1 = `master_rgbnir_seg` − `master_rgb_seg`; H2 = `crosskd_rgbnir_seg` − `crosskd_rgb_seg`; secundaria = `crosskd_rgbnir_seg` − `master_rgb_seg`; exploratoria = `crosskd_rgb_seg` − `master_rgb_seg`." Plus `master_rgb_seg` − `master_rgb_det` on box metrics (AP50, AP30 latent). "La métrica primaria sigue siendo el AP50 latente de caja (t ∈ {0, 2, 4, 6} más sano) ... el AP50 latente de máscara, CorLoc y AUC-LROC son secundarias. Sin cambios: ROPE ±0.02 con P ≥ 0.95, folds por lote."
- Section 11, 2026-11-14 (amendment): "una sola semilla (42) en lugar de tres, y entrada de 640 × 640 ... la variabilidad entre semillas no se estima y queda como limitación declarada."
- Section 11, 2026-11-14 (exploratory arm): `crosskd_nir_seg` (RGB student distilled from `master_nir_seg`); exploratory comparisons `crosskd_nir_seg` − `master_rgb_seg` and `crosskd_nir_seg` − `crosskd_rgbnir_seg`, "descriptiva ... sin carácter confirmatorio". Section 11, 2026-11-16: same queue, same configuration.
- Section 8.1: "ROPE: ±0.02 AP50; Threshold: P(right) >= 0.95". Section 8.3: "H1 se prueba primero al umbral pre-registrado. Solo si H1 se acepta, H2 se prueba como confirmatoria al mismo umbral (secuencia fija). Si H1 no se acepta, H2 se reporta como exploratoria. Las demás comparaciones (secundarias, E3-E6) son exploratorias y sus p-valores corrected-t llevan corrección de Holm dentro de cada experimento." Section 8.4: per comparison the 5 deltas, mean with corrected 95% CI and p, P(left)/P(rope)/P(right), k/5, sign-flip p with floor 0.0625; "Nunca 'significativo' por el sign-flip con n = 5."

Verification against what was run: arms, metric, ROPE, threshold, seed (42 only), 5 folds, 640 px, 80 epochs: all match the 2026-11-13/14 rows. Discrepancies and caveats:
1. Section 8.1's pre-registration block still lists the OLD arm names (est_rgb, kd_rgb, ...) and 3 seeds; it was superseded by the 2026-11-13/14 rows of section 11 (not edited in 8.1). Section 8.3 likewise uses old names. The analysis follows section 11.
2. The plan defines the confirmatory logic as a **fixed sequence** (H1, then H2 only if H1 accepted), not a Holm family. Holm is defined only for the exploratory comparisons "within each experiment". Holm over {H1, H2, secondary} below is therefore **post hoc** (requested); Holm over the exploratory comparisons is reported separately.
3. The three-commit issue: the 35 runs were trained from three code commits (71bef02: master fold0 of rgb/rgbnir/nir; b37f824: 10 runs of folds 0-2; 937bece, the merge of the PR #105 trainer perf changes: 22 runs, all of folds 3-4, all crosskd_nir_seg, and some of folds 2). The skill checklist asks for identical commits across folds; this does not hold. Fold 0 and 1 of crosskd/master_rgb_det came mostly from older code, folds 3-4 from the perf code. Arms within a fold are not all from the same commit either (e.g. fold 2: master_rgb_seg b37f824 vs master_rgbnir_seg 937bece).
4. The 2026-11-13 row mentions CorLoc; no CorLoc exists in the metrics; the analysis uses face AUROC (the registered 'AUC-LROC').
5. Test of `crosskd_nir_seg` is exploratory; it is not part of the registered six-arm set.
6. Seeds: only 42 exists; 'seeds behind each delta' = 42 for all.

## 2. Per-arm TEST results (seed 42), mean +/- sd across 5 folds (sd ddof=1); folds = lots 1..5 as test (fold k tests on the lot of folds_meta)

Folds order: fold0 | fold1 | fold2 | fold3 | fold4. `val` = checkpoint-selection data (optimistic); `val mean` is the mean latent AP50 on val.

| arm | latent AP50 per fold | AP50 mean +/- sd | latent AP30 per fold | AP30 mean +/- sd | val mean AP50 | test - val |
|---|---|---|---|---|---|---|
| master_rgb_det | 0.217 | 0.461 | 0.335 | 0.236 | 0.285 | 0.307 +/- 0.098 | 0.389 | 0.605 | 0.500 | 0.390 | 0.482 | 0.473 +/- 0.090 | 0.319 | -0.012 |
| master_rgb_seg | 0.239 | 0.398 | 0.327 | 0.228 | 0.340 | 0.306 +/- 0.072 | 0.379 | 0.549 | 0.472 | 0.334 | 0.489 | 0.445 +/- 0.087 | 0.375 | -0.069 |
| master_rgbnir_seg | 0.315 | 0.525 | 0.356 | 0.339 | 0.405 | 0.388 +/- 0.083 | 0.480 | 0.670 | 0.512 | 0.447 | 0.569 | 0.535 +/- 0.087 | 0.427 | -0.040 |
| master_nir_seg | 0.353 | 0.558 | 0.448 | 0.374 | 0.439 | 0.434 +/- 0.080 | 0.477 | 0.693 | 0.561 | 0.467 | 0.602 | 0.560 +/- 0.094 | 0.418 | +0.017 |
| crosskd_rgb_seg | 0.303 | 0.388 | 0.353 | 0.219 | 0.325 | 0.318 +/- 0.063 | 0.497 | 0.528 | 0.505 | 0.362 | 0.489 | 0.476 +/- 0.066 | 0.381 | -0.063 |
| crosskd_rgbnir_seg | 0.336 | 0.490 | 0.400 | 0.229 | 0.316 | 0.354 +/- 0.097 | 0.500 | 0.621 | 0.548 | 0.359 | 0.478 | 0.501 +/- 0.097 | 0.368 | -0.014 |
| crosskd_nir_seg | 0.215 | 0.474 | 0.351 | 0.232 | 0.305 | 0.316 +/- 0.104 | 0.383 | 0.659 | 0.507 | 0.351 | 0.460 | 0.472 +/- 0.121 | 0.365 | -0.049 |

Other per-arm metrics (test, mean +/- sd): latent mask AP50, mask AP30, face AUROC, sano FP per image (conf 0.25).

| arm | mask AP50 | mask AP30 | face AUROC | sano FP/image |
|---|---|---|---|---|
| master_rgb_det | n/a (detect) | n/a (detect) | 0.605 +/- 0.036 | 6.942 +/- 0.384 |
| master_rgb_seg | 0.295 +/- 0.078 | 0.440 +/- 0.089 | 0.599 +/- 0.046 | 5.758 +/- 1.050 |
| master_rgbnir_seg | 0.391 +/- 0.090 | 0.531 +/- 0.089 | 0.627 +/- 0.040 | 6.600 +/- 1.263 |
| master_nir_seg | 0.429 +/- 0.097 | 0.556 +/- 0.098 | 0.642 +/- 0.060 | 4.668 +/- 1.538 |
| crosskd_rgb_seg | 0.308 +/- 0.067 | 0.467 +/- 0.072 | 0.613 +/- 0.042 | 4.598 +/- 0.828 |
| crosskd_rgbnir_seg | 0.334 +/- 0.102 | 0.492 +/- 0.098 | 0.626 +/- 0.039 | 4.336 +/- 0.565 |
| crosskd_nir_seg | 0.312 +/- 0.100 | 0.468 +/- 0.118 | 0.607 +/- 0.046 | 6.046 +/- 2.266 |

Val-vs-test: val is the checkpoint-selection split, so val AP50 is optimistic; test-minus-val above is the per-arm gap (mixed sign across arms; the lots differ, val and test are different lots, so the gap is not a pure optimism estimate).

## 3. Comparisons (all on TEST; delta = treatment - baseline; per-fold order fold0..fold4)

Corrected t: Nadeau-Bengio, df=4, two-sided p. Bayes with ROPE 0.02. `k/5` = folds with delta > 0 (for FP/image, delta > 0 means MORE false positives, i.e. worse). Sign-flip p is exact, floor 0.0625; it is a sanity check, never evidence of significance.

### H1: master_rgbnir_seg - master_rgb_seg  [registered confirmatory]

**latent AP50 (box) -- PRIMARY**

- per-fold deltas (seed 42): +0.076 | +0.127 | +0.029 | +0.111 | +0.065
- mean delta +0.0816; corrected 95% CI [+0.0033, +0.1599]; corrected p (two-sided) 0.0445 (t=2.892, df=4)
- Bayes (ROPE 0.02): P(left)=0.011, P(rope)=0.036, P(right)=0.953
- k/5 folds with delta>0: 5/5; sign-flip exact p 0.0625 (floor 0.0625)
- Licensed sentence (section 4): Practically better: on held-out lots the treatment raised test damage AP50 by more than the ROPE (posterior probability 0.953, mean delta +0.082, corrected 95% CI [+0.003, +0.160]).

**latent AP30 (box)**

- per-fold deltas (seed 42): +0.101 | +0.121 | +0.040 | +0.113 | +0.079
- mean delta +0.0909; corrected 95% CI [+0.0255, +0.1562]; corrected p (two-sided) 0.0181 (t=3.862, df=4)
- Bayes (ROPE 0.02): P(left)=0.005, P(rope)=0.015, P(right)=0.980
- k/5 folds with delta>0: 5/5; sign-flip exact p 0.0625 (floor 0.0625)
- Secondary metric: section-4 wording would give: Practically better: on held-out lots the treatment raised test damage AP50 by more than the ROPE (posterior probability 0.980, mean delta +0.091, corrected 95% CI [+0.026, +0.156]).

**latent mask AP50**

- per-fold deltas (seed 42): +0.096 | +0.135 | +0.036 | +0.115 | +0.102
- mean delta +0.0966; corrected 95% CI [+0.0212, +0.1720]; corrected p (two-sided) 0.0237 (t=3.557, df=4)
- Bayes (ROPE 0.02): P(left)=0.006, P(rope)=0.018, P(right)=0.976
- k/5 folds with delta>0: 5/5; sign-flip exact p 0.0625 (floor 0.0625)
- Secondary metric: section-4 wording would give: Practically better: on held-out lots the treatment raised test damage AP50 by more than the ROPE (posterior probability 0.976, mean delta +0.097, corrected 95% CI [+0.021, +0.172]).

**latent mask AP30**

- per-fold deltas (seed 42): +0.104 | +0.129 | +0.021 | +0.124 | +0.077
- mean delta +0.0908; corrected 95% CI [+0.0009, +0.1808]; corrected p (two-sided) 0.0486 (t=2.804, df=4)
- Bayes (ROPE 0.02): P(left)=0.013, P(rope)=0.034, P(right)=0.953
- k/5 folds with delta>0: 5/5; sign-flip exact p 0.0625 (floor 0.0625)
- Secondary metric: section-4 wording would give: Practically better: on held-out lots the treatment raised test damage AP50 by more than the ROPE (posterior probability 0.953, mean delta +0.091, corrected 95% CI [+0.001, +0.181]).

**face AUROC**

- per-fold deltas (seed 42): +0.031 | +0.049 | -0.013 | +0.061 | +0.011
- mean delta +0.0280; corrected 95% CI [-0.0321, +0.0881]; corrected p (two-sided) 0.2653 (t=1.294, df=4)
- Bayes (ROPE 0.02): P(left)=0.045, P(rope)=0.320, P(right)=0.635
- k/5 folds with delta>0: 4/5; sign-flip exact p 0.1875 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.045, P(rope)=0.320, P(right)=0.635; mean delta +0.028, corrected 95% CI [-0.032, +0.088]); more lots would be needed to change the verdict.

**sano FP per image (lower is better; ROPE 0.02 NOT meaningful, descriptive only)**

- per-fold deltas (seed 42): +2.060 | +3.680 | -0.310 | -1.490 | +0.270
- mean delta +0.8420; corrected 95% CI [-3.2924, +4.9764]; corrected p (two-sided) 0.6020 (t=0.565, df=4)
- Bayes (ROPE 0.02): P(left)=0.297, P(rope)=0.008, P(right)=0.695
- k/5 folds with delta>0: 3/5; sign-flip exact p 0.4375 (floor 0.0625)
- Descriptive only: lower is better; the Bayes probabilities refer to a +/-0.02 FP/image ROPE that has no meaning at this scale (deltas are of order 1 FP/image). No decision sentence licensed.

### H2: crosskd_rgbnir_seg - crosskd_rgb_seg  [registered confirmatory (fixed sequence after H1)]

**latent AP50 (box) -- PRIMARY**

- per-fold deltas (seed 42): +0.033 | +0.102 | +0.047 | +0.009 | -0.009
- mean delta +0.0365; corrected 95% CI [-0.0494, +0.1224]; corrected p (two-sided) 0.3039 (t=1.179, df=4)
- Bayes (ROPE 0.02): P(left)=0.071, P(rope)=0.240, P(right)=0.689
- k/5 folds with delta>0: 4/5; sign-flip exact p 0.1250 (floor 0.0625)
- Licensed sentence (section 4): Inconclusive on five lots (P(left)=0.071, P(rope)=0.240, P(right)=0.689; mean delta +0.036, corrected 95% CI [-0.049, +0.122]); more lots would be needed to change the verdict.

**latent AP30 (box)**

- per-fold deltas (seed 42): +0.003 | +0.094 | +0.043 | -0.003 | -0.011
- mean delta +0.0249; corrected 95% CI [-0.0636, +0.1135]; corrected p (two-sided) 0.4778 (t=0.782, df=4)
- Bayes (ROPE 0.02): P(left)=0.116, P(rope)=0.326, P(right)=0.558
- k/5 folds with delta>0: 3/5; sign-flip exact p 0.4375 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.116, P(rope)=0.326, P(right)=0.558; mean delta +0.025, corrected 95% CI [-0.064, +0.114]); more lots would be needed to change the verdict.

**latent mask AP50**

- per-fold deltas (seed 42): +0.027 | +0.075 | +0.061 | +0.006 | -0.038
- mean delta +0.0264; corrected 95% CI [-0.0651, +0.1178]; corrected p (two-sided) 0.4683 (t=0.800, df=4)
- Bayes (ROPE 0.02): P(left)=0.116, P(rope)=0.312, P(right)=0.572
- k/5 folds with delta>0: 4/5; sign-flip exact p 0.3125 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.116, P(rope)=0.312, P(right)=0.572; mean delta +0.026, corrected 95% CI [-0.065, +0.118]); more lots would be needed to change the verdict.

**latent mask AP30**

- per-fold deltas (seed 42): +0.008 | +0.086 | +0.036 | +0.007 | -0.008
- mean delta +0.0255; corrected 95% CI [-0.0498, +0.1008]; corrected p (two-sided) 0.3997 (t=0.942, df=4)
- Bayes (ROPE 0.02): P(left)=0.084, P(rope)=0.340, P(right)=0.576
- k/5 folds with delta>0: 4/5; sign-flip exact p 0.2500 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.084, P(rope)=0.340, P(right)=0.576; mean delta +0.026, corrected 95% CI [-0.050, +0.101]); more lots would be needed to change the verdict.

**face AUROC**

- per-fold deltas (seed 42): -0.046 | +0.061 | +0.067 | +0.008 | -0.022
- mean delta +0.0138; corrected 95% CI [-0.0873, +0.1149]; corrected p (two-sided) 0.7235 (t=0.380, df=4)
- Bayes (ROPE 0.02): P(left)=0.203, P(rope)=0.360, P(right)=0.437
- k/5 folds with delta>0: 3/5; sign-flip exact p 0.5625 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.203, P(rope)=0.360, P(right)=0.437; mean delta +0.014, corrected 95% CI [-0.087, +0.115]); more lots would be needed to change the verdict.

**sano FP per image (lower is better; ROPE 0.02 NOT meaningful, descriptive only)**

- per-fold deltas (seed 42): +0.120 | -0.900 | -0.740 | -0.250 | +0.460
- mean delta -0.2620; corrected 95% CI [-1.4192, +0.8952]; corrected p (two-sided) 0.5637 (t=-0.629, df=4)
- Bayes (ROPE 0.02): P(left)=0.704, P(rope)=0.028, P(right)=0.268
- k/5 folds with delta>0: 2/5; sign-flip exact p 0.3750 (floor 0.0625)
- Descriptive only: lower is better; the Bayes probabilities refer to a +/-0.02 FP/image ROPE that has no meaning at this scale (deltas are of order 1 FP/image). No decision sentence licensed.

### secondary: crosskd_rgbnir_seg - master_rgb_seg  [registered secondary (exploratory status per plan 8.3)]

**latent AP50 (box) -- PRIMARY**

- per-fold deltas (seed 42): +0.097 | +0.092 | +0.073 | +0.001 | -0.024
- mean delta +0.0477; corrected 95% CI [-0.0648, +0.1603]; corrected p (two-sided) 0.3042 (t=1.178, df=4)
- Bayes (ROPE 0.02): P(left)=0.085, P(rope)=0.181, P(right)=0.734
- k/5 folds with delta>0: 4/5; sign-flip exact p 0.1875 (floor 0.0625)
- Licensed sentence (section 4): Inconclusive on five lots (P(left)=0.085, P(rope)=0.181, P(right)=0.734; mean delta +0.048, corrected 95% CI [-0.065, +0.160]); more lots would be needed to change the verdict.

**latent AP30 (box)**

- per-fold deltas (seed 42): +0.121 | +0.073 | +0.076 | +0.024 | -0.011
- mean delta +0.0566; corrected 95% CI [-0.0466, +0.1599]; corrected p (two-sided) 0.2024 (t=1.523, df=4)
- Bayes (ROPE 0.02): P(left)=0.054, P(rope)=0.136, P(right)=0.810
- k/5 folds with delta>0: 4/5; sign-flip exact p 0.1250 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.054, P(rope)=0.136, P(right)=0.810; mean delta +0.057, corrected 95% CI [-0.047, +0.160]); more lots would be needed to change the verdict.

**latent mask AP50**

- per-fold deltas (seed 42): +0.085 | +0.062 | +0.079 | -0.000 | -0.028
- mean delta +0.0396; corrected 95% CI [-0.0635, +0.1427]; corrected p (two-sided) 0.3462 (t=1.067, df=4)
- Bayes (ROPE 0.02): P(left)=0.092, P(rope)=0.221, P(right)=0.687
- k/5 folds with delta>0: 3/5; sign-flip exact p 0.2500 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.092, P(rope)=0.221, P(right)=0.687; mean delta +0.040, corrected 95% CI [-0.063, +0.143]); more lots would be needed to change the verdict.

**latent mask AP30**

- per-fold deltas (seed 42): +0.111 | +0.076 | +0.068 | +0.025 | -0.019
- mean delta +0.0523; corrected 95% CI [-0.0494, +0.1540]; corrected p (two-sided) 0.2266 (t=1.428, df=4)
- Bayes (ROPE 0.02): P(left)=0.060, P(rope)=0.154, P(right)=0.786
- k/5 folds with delta>0: 4/5; sign-flip exact p 0.1250 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.060, P(rope)=0.154, P(right)=0.786; mean delta +0.052, corrected 95% CI [-0.049, +0.154]); more lots would be needed to change the verdict.

**face AUROC**

- per-fold deltas (seed 42): +0.044 | +0.054 | +0.044 | +0.029 | -0.034
- mean delta +0.0273; corrected 95% CI [-0.0447, +0.0992]; corrected p (two-sided) 0.3519 (t=1.053, df=4)
- Bayes (ROPE 0.02): P(left)=0.071, P(rope)=0.325, P(right)=0.604
- k/5 folds with delta>0: 4/5; sign-flip exact p 0.1875 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.071, P(rope)=0.325, P(right)=0.604; mean delta +0.027, corrected 95% CI [-0.045, +0.099]); more lots would be needed to change the verdict.

**sano FP per image (lower is better; ROPE 0.02 NOT meaningful, descriptive only)**

- per-fold deltas (seed 42): -0.280 | -1.290 | -1.560 | -2.310 | -1.670
- mean delta -1.4220; corrected 95% CI [-2.9225, +0.0785]; corrected p (two-sided) 0.0581 (t=-2.631, df=4)
- Bayes (ROPE 0.02): P(left)=0.970, P(rope)=0.002, P(right)=0.028
- k/5 folds with delta>0: 0/5; sign-flip exact p 0.0625 (floor 0.0625)
- Descriptive only: lower is better; the Bayes probabilities refer to a +/-0.02 FP/image ROPE that has no meaning at this scale (deltas are of order 1 FP/image). No decision sentence licensed.

### exploratory: crosskd_rgb_seg - master_rgb_seg  [registered exploratory]

**latent AP50 (box) -- PRIMARY**

- per-fold deltas (seed 42): +0.064 | -0.010 | +0.025 | -0.009 | -0.015
- mean delta +0.0113; corrected 95% CI [-0.0572, +0.0797]; corrected p (two-sided) 0.6712 (t=0.457, df=4)
- Bayes (ROPE 0.02): P(left)=0.137, P(rope)=0.493, P(right)=0.371
- k/5 folds with delta>0: 2/5; sign-flip exact p 0.5625 (floor 0.0625)
- Licensed sentence (section 4): Inconclusive on five lots (P(left)=0.137, P(rope)=0.493, P(right)=0.371; mean delta +0.011, corrected 95% CI [-0.057, +0.080]); more lots would be needed to change the verdict.

**latent AP30 (box)**

- per-fold deltas (seed 42): +0.118 | -0.021 | +0.034 | +0.028 | +0.000
- mean delta +0.0317; corrected 95% CI [-0.0759, +0.1393]; corrected p (two-sided) 0.4595 (t=0.818, df=4)
- Bayes (ROPE 0.02): P(left)=0.127, P(rope)=0.262, P(right)=0.611
- k/5 folds with delta>0: 4/5; sign-flip exact p 0.1875 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.127, P(rope)=0.262, P(right)=0.611; mean delta +0.032, corrected 95% CI [-0.076, +0.139]); more lots would be needed to change the verdict.

**latent mask AP50**

- per-fold deltas (seed 42): +0.058 | -0.014 | +0.018 | -0.006 | +0.009
- mean delta +0.0132; corrected 95% CI [-0.0438, +0.0703]; corrected p (two-sided) 0.5544 (t=0.644, df=4)
- Bayes (ROPE 0.02): P(left)=0.091, P(rope)=0.530, P(right)=0.380
- k/5 folds with delta>0: 3/5; sign-flip exact p 0.4375 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.091, P(rope)=0.530, P(right)=0.380; mean delta +0.013, corrected 95% CI [-0.044, +0.070]); more lots would be needed to change the verdict.

**latent mask AP30**

- per-fold deltas (seed 42): +0.103 | -0.009 | +0.032 | +0.018 | -0.011
- mean delta +0.0268; corrected 95% CI [-0.0674, +0.1209]; corrected p (two-sided) 0.4742 (t=0.789, df=4)
- Bayes (ROPE 0.02): P(left)=0.120, P(rope)=0.306, P(right)=0.574
- k/5 folds with delta>0: 3/5; sign-flip exact p 0.3125 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.120, P(rope)=0.306, P(right)=0.574; mean delta +0.027, corrected 95% CI [-0.067, +0.121]); more lots would be needed to change the verdict.

**face AUROC**

- per-fold deltas (seed 42): +0.089 | -0.008 | -0.023 | +0.021 | -0.012
- mean delta +0.0134; corrected 95% CI [-0.0787, +0.1056]; corrected p (two-sided) 0.7062 (t=0.405, df=4)
- Bayes (ROPE 0.02): P(left)=0.185, P(rope)=0.388, P(right)=0.427
- k/5 folds with delta>0: 2/5; sign-flip exact p 0.7500 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.185, P(rope)=0.388, P(right)=0.427; mean delta +0.013, corrected 95% CI [-0.079, +0.106]); more lots would be needed to change the verdict.

**sano FP per image (lower is better; ROPE 0.02 NOT meaningful, descriptive only)**

- per-fold deltas (seed 42): -0.400 | -0.390 | -0.820 | -2.060 | -2.130
- mean delta -1.1600; corrected 95% CI [-2.9268, +0.6068]; corrected p (two-sided) 0.1424 (t=-1.823, df=4)
- Bayes (ROPE 0.02): P(left)=0.926, P(rope)=0.005, P(right)=0.069
- k/5 folds with delta>0: 0/5; sign-flip exact p 0.0625 (floor 0.0625)
- Descriptive only: lower is better; the Bayes probabilities refer to a +/-0.02 FP/image ROPE that has no meaning at this scale (deltas are of order 1 FP/image). No decision sentence licensed.

### mask_supervision: master_rgb_seg - master_rgb_det  [registered comparison, box metrics only]

**latent AP50 (box) -- PRIMARY**

- per-fold deltas (seed 42): +0.022 | -0.063 | -0.008 | -0.008 | +0.055
- mean delta -0.0006; corrected 95% CI [-0.0890, +0.0878]; corrected p (two-sided) 0.9858 (t=-0.019, df=4)
- Bayes (ROPE 0.02): P(left)=0.288, P(rope)=0.436, P(right)=0.276
- k/5 folds with delta>0: 2/5; sign-flip exact p 1.0000 (floor 0.0625)
- Licensed sentence (section 4): Inconclusive on five lots (P(left)=0.288, P(rope)=0.436, P(right)=0.276; mean delta -0.001, corrected 95% CI [-0.089, +0.088]); more lots would be needed to change the verdict.

**latent AP30 (box)**

- per-fold deltas (seed 42): -0.010 | -0.056 | -0.028 | -0.056 | +0.007
- mean delta -0.0286; corrected 95% CI [-0.0849, +0.0277]; corrected p (two-sided) 0.2307 (t=-1.412, df=4)
- Bayes (ROPE 0.02): P(left)=0.654, P(rope)=0.309, P(right)=0.037
- k/5 folds with delta>0: 1/5; sign-flip exact p 0.1250 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.654, P(rope)=0.309, P(right)=0.037; mean delta -0.029, corrected 95% CI [-0.085, +0.028]); more lots would be needed to change the verdict.

**face AUROC**

- per-fold deltas (seed 42): +0.007 | -0.037 | +0.020 | -0.057 | +0.040
- mean delta -0.0054; corrected 95% CI [-0.0874, +0.0766]; corrected p (two-sided) 0.8640 (t=-0.183, df=4)
- Bayes (ROPE 0.02): P(left)=0.323, P(rope)=0.457, P(right)=0.219
- k/5 folds with delta>0: 3/5; sign-flip exact p 0.8750 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.323, P(rope)=0.457, P(right)=0.219; mean delta -0.005, corrected 95% CI [-0.087, +0.077]); more lots would be needed to change the verdict.

**sano FP per image (lower is better; ROPE 0.02 NOT meaningful, descriptive only)**

- per-fold deltas (seed 42): -3.150 | -1.870 | +0.300 | -0.340 | -0.860
- mean delta -1.1840; corrected 95% CI [-3.9335, +1.5655]; corrected p (two-sided) 0.2979 (t=-1.196, df=4)
- Bayes (ROPE 0.02): P(left)=0.847, P(rope)=0.007, P(right)=0.145
- k/5 folds with delta>0: 1/5; sign-flip exact p 0.1250 (floor 0.0625)
- Descriptive only: lower is better; the Bayes probabilities refer to a +/-0.02 FP/image ROPE that has no meaning at this scale (deltas are of order 1 FP/image). No decision sentence licensed.

### crosskd_nir: crosskd_nir_seg - master_rgb_seg  [exploratory (2026-11-14)]

**latent AP50 (box) -- PRIMARY**

- per-fold deltas (seed 42): -0.024 | +0.076 | +0.024 | +0.004 | -0.034
- mean delta +0.0092; corrected 95% CI [-0.0792, +0.0976]; corrected p (two-sided) 0.7868 (t=0.289, df=4)
- Bayes (ROPE 0.02): P(left)=0.205, P(rope)=0.419, P(right)=0.376
- k/5 folds with delta>0: 3/5; sign-flip exact p 0.6875 (floor 0.0625)
- Licensed sentence (section 4): Inconclusive on five lots (P(left)=0.205, P(rope)=0.419, P(right)=0.376; mean delta +0.009, corrected 95% CI [-0.079, +0.098]); more lots would be needed to change the verdict.

**latent AP30 (box)**

- per-fold deltas (seed 42): +0.004 | +0.110 | +0.036 | +0.017 | -0.029
- mean delta +0.0276; corrected 95% CI [-0.0772, +0.1324]; corrected p (two-sided) 0.5050 (t=0.732, df=4)
- Bayes (ROPE 0.02): P(left)=0.138, P(rope)=0.287, P(right)=0.575
- k/5 folds with delta>0: 4/5; sign-flip exact p 0.3125 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.138, P(rope)=0.287, P(right)=0.575; mean delta +0.028, corrected 95% CI [-0.077, +0.132]); more lots would be needed to change the verdict.

**latent mask AP50**

- per-fold deltas (seed 42): -0.003 | +0.056 | +0.027 | +0.013 | -0.010
- mean delta +0.0166; corrected 95% CI [-0.0370, +0.0703]; corrected p (two-sided) 0.4379 (t=0.861, df=4)
- Bayes (ROPE 0.02): P(left)=0.065, P(rope)=0.499, P(right)=0.435
- k/5 folds with delta>0: 3/5; sign-flip exact p 0.3125 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.065, P(rope)=0.499, P(right)=0.435; mean delta +0.017, corrected 95% CI [-0.037, +0.070]); more lots would be needed to change the verdict.

**latent mask AP30**

- per-fold deltas (seed 42): +0.005 | +0.108 | +0.035 | +0.028 | -0.036
- mean delta +0.0280; corrected 95% CI [-0.0782, +0.1341]; corrected p (two-sided) 0.5049 (t=0.732, df=4)
- Bayes (ROPE 0.02): P(left)=0.139, P(rope)=0.284, P(right)=0.578
- k/5 folds with delta>0: 4/5; sign-flip exact p 0.3750 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.139, P(rope)=0.284, P(right)=0.578; mean delta +0.028, corrected 95% CI [-0.078, +0.134]); more lots would be needed to change the verdict.

**face AUROC**

- per-fold deltas (seed 42): +0.059 | +0.041 | -0.009 | -0.002 | -0.047
- mean delta +0.0083; corrected 95% CI [-0.0775, +0.0942]; corrected p (two-sided) 0.8012 (t=0.269, df=4)
- Bayes (ROPE 0.02): P(left)=0.206, P(rope)=0.432, P(right)=0.362
- k/5 folds with delta>0: 2/5; sign-flip exact p 0.7500 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.206, P(rope)=0.432, P(right)=0.362; mean delta +0.008, corrected 95% CI [-0.078, +0.094]); more lots would be needed to change the verdict.

**sano FP per image (lower is better; ROPE 0.02 NOT meaningful, descriptive only)**

- per-fold deltas (seed 42): +5.290 | +0.560 | -1.190 | -0.640 | -2.580
- mean delta +0.2880; corrected 95% CI [-5.8257, +6.4017]; corrected p (two-sided) 0.9023 (t=0.131, df=4)
- Bayes (ROPE 0.02): P(left)=0.448, P(rope)=0.007, P(right)=0.546
- k/5 folds with delta>0: 2/5; sign-flip exact p 0.9375 (floor 0.0625)
- Descriptive only: lower is better; the Bayes probabilities refer to a +/-0.02 FP/image ROPE that has no meaning at this scale (deltas are of order 1 FP/image). No decision sentence licensed.

### crosskd_nir_vs_rgbnir: crosskd_nir_seg - crosskd_rgbnir_seg  [exploratory (2026-11-14)]

**latent AP50 (box) -- PRIMARY**

- per-fold deltas (seed 42): -0.121 | -0.016 | -0.049 | +0.004 | -0.010
- mean delta -0.0385; corrected 95% CI [-0.1395, +0.0625]; corrected p (two-sided) 0.3492 (t=-1.059, df=4)
- Bayes (ROPE 0.02): P(left)=0.681, P(rope)=0.227, P(right)=0.091
- k/5 folds with delta>0: 1/5; sign-flip exact p 0.1250 (floor 0.0625)
- Licensed sentence (section 4): Inconclusive on five lots (P(left)=0.681, P(rope)=0.227, P(right)=0.091; mean delta -0.039, corrected 95% CI [-0.140, +0.062]); more lots would be needed to change the verdict.

**latent AP30 (box)**

- per-fold deltas (seed 42): -0.117 | +0.037 | -0.041 | -0.007 | -0.018
- mean delta -0.0290; corrected 95% CI [-0.1437, +0.0856]; corrected p (two-sided) 0.5207 (t=-0.703, df=4)
- Bayes (ROPE 0.02): P(left)=0.581, P(rope)=0.268, P(right)=0.150
- k/5 folds with delta>0: 1/5; sign-flip exact p 0.3125 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.581, P(rope)=0.268, P(right)=0.150; mean delta -0.029, corrected 95% CI [-0.144, +0.086]); more lots would be needed to change the verdict.

**latent mask AP50**

- per-fold deltas (seed 42): -0.088 | -0.005 | -0.052 | +0.013 | +0.018
- mean delta -0.0230; corrected 95% CI [-0.1156, +0.0697]; corrected p (two-sided) 0.5293 (t=-0.688, df=4)
- Bayes (ROPE 0.02): P(left)=0.533, P(rope)=0.333, P(right)=0.134
- k/5 folds with delta>0: 2/5; sign-flip exact p 0.4375 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.533, P(rope)=0.333, P(right)=0.134; mean delta -0.023, corrected 95% CI [-0.116, +0.070]); more lots would be needed to change the verdict.

**latent mask AP30**

- per-fold deltas (seed 42): -0.106 | +0.031 | -0.033 | +0.003 | -0.017
- mean delta -0.0243; corrected 95% CI [-0.1288, +0.0801]; corrected p (two-sided) 0.5531 (t=-0.647, df=4)
- Bayes (ROPE 0.02): P(left)=0.543, P(rope)=0.305, P(right)=0.152
- k/5 folds with delta>0: 2/5; sign-flip exact p 0.4375 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.543, P(rope)=0.305, P(right)=0.152; mean delta -0.024, corrected 95% CI [-0.129, +0.080]); more lots would be needed to change the verdict.

**face AUROC**

- per-fold deltas (seed 42): +0.016 | -0.013 | -0.053 | -0.031 | -0.013
- mean delta -0.0189; corrected 95% CI [-0.0704, +0.0325]; corrected p (two-sided) 0.3645 (t=-1.022, df=4)
- Bayes (ROPE 0.02): P(left)=0.479, P(rope)=0.470, P(right)=0.052
- k/5 folds with delta>0: 1/5; sign-flip exact p 0.2500 (floor 0.0625)
- Secondary metric: section-4 wording would give: Inconclusive on five lots (P(left)=0.479, P(rope)=0.470, P(right)=0.052; mean delta -0.019, corrected 95% CI [-0.070, +0.033]); more lots would be needed to change the verdict.

**sano FP per image (lower is better; ROPE 0.02 NOT meaningful, descriptive only)**

- per-fold deltas (seed 42): +5.570 | +1.850 | +0.370 | +1.670 | -0.910
- mean delta +1.7100; corrected 95% CI [-3.2131, +6.6331]; corrected p (two-sided) 0.3895 (t=0.964, df=4)
- Bayes (ROPE 0.02): P(left)=0.192, P(rope)=0.005, P(right)=0.803
- k/5 folds with delta>0: 4/5; sign-flip exact p 0.1875 (floor 0.0625)
- Descriptive only: lower is better; the Bayes probabilities refer to a +/-0.02 FP/image ROPE that has no meaning at this scale (deltas are of order 1 FP/image). No decision sentence licensed.

## 4. Multiple comparisons

Registered procedure (plan 8.3): fixed sequence. H1 is tested first; H1 crossed P(right) >= 0.95 on latent AP50 (P(right)=0.953, a narrow margin over 0.95), so H1 is accepted at the registered threshold and H2 is tested as confirmatory at the same threshold: H2 P(right)=0.689, inconclusive.

Post hoc Holm over {H1, H2, secondary} on the corrected-t two-sided p of latent AP50 (the plan defines no such family):

| comparison | corrected p | Holm alpha/(m-i) | rejected |
|---|---|---|---|
| H1 | 0.0445 | 0.0167 | no |
| H2 | 0.3039 | 0.0250 | no |
| secondary | 0.3042 | 0.0500 | no |

Stated plainly as both facts: H1 passes the registered Bayesian threshold (P(right)=0.953) and its corrected p is 0.044, but it does NOT survive Holm over three comparisons (0.044 > 0.0167); its corrected CI lower bound (+0.003) is practically at zero. Holm does not adjust the posterior probabilities (skill section 7).

Holm over the exploratory comparisons (exploratory, mask_supervision, crosskd_nir, crosskd_nir_vs_rgbnir; latent AP50; post hoc grouping, the plan says 'within each experiment'):

| comparison | corrected p | Holm alpha/(m-i) | rejected |
|---|---|---|---|
| crosskd_nir_vs_rgbnir | 0.3492 | 0.0125 | no |
| exploratory | 0.6712 | 0.0167 | no |
| crosskd_nir | 0.7868 | 0.0250 | no |
| mask_supervision | 0.9858 | 0.0500 | no |

## 5. Reporting checklist (skill section 8)

- [ ] Pre-registration committed before first test read: the rows of 2026-11-13/14/16 in section 11 precede these runs; section 8.1's block was NOT updated to the new arm names (see discrepancy 1).
- [x] All 5 folds present for all 7 arms (35 `run_complete.json`, 35 test `metrics_v4.json`). Seeds: 42 only, by the 2026-11-14 amendment, not 3.
- [x] Per-fold deltas, mean, corrected 95% CI and p: section 3.
- [x] P(left)/P(rope)/P(right) with ROPE 0.02 and licensed sentence: section 3.
- [x] k/5 and exact sign-flip p with floor 0.0625; 'significant' is not used on its account.
- [x] val labelled as checkpoint-selection data (section 2).
- [ ] The standard 188/23/25 split: not used here (LOLO on v4 lots; the 236-pair v3 split is not part of this analysis).
- [x] Limits: below.
- [ ] Seed budget: `plan_seeds.py` not run: only one seed exists (sigma_seed not estimable); declared limitation.
- [ ] Arms, configs and commit hash identical across folds: NOT satisfied (three commits, discrepancy 3). Config values (image_size 640, epochs 80, seed 42) are identical in all 35 runs.

## 6. Limits

- Five lots, so n = 5 paired differences, df = 4 and wide intervals; per-fold deltas vary a lot (e.g. H1 from +0.029 to +0.127), so lot effects dominate the spread.
- Single seed (42): seed-to-seed variability is not estimated and is not in the intervals; the deltas confound arm effect with seed noise.
- Nadeau-Bengio was derived for random resampling and is applied to k-fold by convention (Benavoli et al. 2017, fn 2); ratio 1/3 pooled, val excluded.
- Lots differ in capture conditions; folds are not exchangeable draws.
- Training commits differ across runs (perf refactor in 937bece vs earlier code).
- Sano FP/image: 100 healthy photos per test lot at confidence 0.25; ROPE 0.02 is not meaningful.
- No cluster bootstrap (invalid with 5 clusters). Val numbers are not evidence.

## 7. Provenance

The export (`export/`, 70 evaluations: 7 arms x 5 folds x val+test, seed 42) was produced by `scripts/export_run_results.py`; its `differences.csv` latent AP50/AP30/mask deltas match the deltas used here (130 values, 0 mismatches). `export_run_results.py` writes deltas only (no corrected t or Bayes); inference comes from `scripts/aggregate_kfold.paired_inference` over `folds_meta.json`. Face AUROC and sano FP/image were read from `metrics_v4.json` (not in the exporter DIFFERENCES). Machine-readable numbers: `analysis.json`.
