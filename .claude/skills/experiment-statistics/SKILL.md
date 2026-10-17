---
name: experiment-statistics
description: "Use when comparing training arms or ablations, reading k-fold / leave-one-lot-out or multi-seed results, deciding whether an AP delta is real, choosing how many seeds to run, or writing experiment results for the thesis. Lot-level paired inference for the mango damage detector: pre-registration, scripts/aggregate_kfold.py, scripts/plan_seeds.py, and the wording each outcome licenses."
---

# Experiment statistics (mango damage detector)

Use this before claiming that one arm beats another. The code lives in
`scripts/run_kfold.py` (train + evaluate every fold x arm x seed),
`scripts/aggregate_kfold.py` (tables + paired inference),
`scripts/plan_seeds.py` (seed budget), `src/evaluation/fold_inference.py` and
`src/evaluation/seed_planning.py` (the formulas), and
`configs/folds/lot_5fold.json` (fold sizes).

## 1. The unit of analysis is the lot

- The dataset has **5 fruit lots**. Leave-one-lot-out (LOLO) gives 5 folds: fold k
  tests on one lot, validates on another and trains on the other three. The
  comparison has **n = 5** paired differences, not hundreds of images.
- Images inside a lot are not independent: the same mango is photographed
  repeatedly and mango identity cannot be recovered from the data.
- The standard random 188/23/25 split puts photographs of the same mango on both
  sides of a split. Its numbers are a **development signal**, not evidence of
  generalization to new lots. Claims go through LOLO.
- `val` is the checkpoint-selection split (`selection_metric=damage_ap50`), so
  `val` deltas are optimistic. Claims rest on `test`.

## 2. Before touching test results

Write these down (a short committed note) before the first test-lot evaluation. If
a `hypothesis-generation` skill is available, use its preregistration scaffold;
otherwise the six lines below are enough.

```
Arms:        baseline = twostream, treatment = <arm>
Metric:      test damage AP50, mean over seeds per fold; delta = treatment - baseline
ROPE:        +/-0.02 AP50 (example; choose and justify your own)
Threshold:   posterior probability >= 0.95 (example; chosen by the author, not a citation)
Seeds:       42, 1337, 2024
Folds:       all five of configs/folds/lot_5fold.json (none dropped)
```

- Checkpoint selection uses the **val lot only**. Never tune thresholds,
  epochs or augmentations on the test lot, and never re-run a fold because its
  test number looked bad.
- Do not change the ROPE, the threshold or the seed set after seeing results; a
  different choice is reported as exploratory.
- Training runs on the 3080 machine and is launched by the user. The analysis
  below needs only the `metrics.json` files that the runner writes.

## 3. Primary analysis

```bash
python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml --seeds 42 1337 2024   # resumable
python scripts/aggregate_kfold.py --seeds 42 1337 2024 --rope 0.02
```

Outputs `reports/kfold_lot/kfold_results.json` and `.md`. Use `--fold-spec`
(default `configs/folds/lot_5fold.json`) for the test/train ratio,
`--confidence` for the interval level, `--reports-root` / `--arms` /
`--baseline` / `--treatment` as usual.

On the seed-averaged fold-level differences, `paired.inference.test.ap50` holds:

- `corrected_t`: Nadeau-Bengio corrected resampled t. The variance of the mean is
  `sd^2 * (1/n + n_test/n_train)`, df = n - 1. Gives the corrected CI and p.
- `bayes`: correlated Bayesian posterior of the mean difference (Corani &
  Benavoli 2015; Benavoli et al. 2017): `P(left)` = Delta < -ROPE (baseline
  practically better), `P(rope)` = inside the ROPE, `P(right)` = Delta > ROPE
  (treatment practically better).
- `sign_flip`: exact paired randomization test; `min_attainable_p_two_sided` is
  its floor.
- A metric whose differences cannot be analysed (zero variance, fewer than 2
  folds) holds `{"error": ...}` instead of numbers. Say so; do not fill it in.

**Report, always:**

1. All 5 per-fold deltas (not just the mean), with the seeds behind each.
2. Mean delta with the corrected 95% CI and corrected p.
3. `P(left)`, `P(rope)`, `P(right)` and the ROPE they refer to.
4. The sign count k/5 (folds with delta > 0).
5. The exact sign-flip p **together with its 0.0625 floor** (2/2^5).

## 4. Decision language

Applied with the pre-registered ROPE and threshold (example: 0.95).

| Outcome | Sentence it licenses |
|---|---|
| `P(right)` >= threshold | "Practically better: on held-out lots the treatment raised test damage AP50 by more than the ROPE (posterior probability P, mean delta d, 95% CI [a, b])." |
| `P(rope)` >= threshold | "Practically equivalent: the difference is within +/-ROPE with posterior probability P." |
| `P(left)` >= threshold | "Practically worse than the baseline" (same format). |
| none of the three | "Inconclusive on five lots": report the three probabilities and the per-fold deltas, and say what would change the verdict (more lots). |

- **Never write "significant" on the strength of the sign-flip test at n = 5.**
  Its smallest attainable two-sided p is 0.0625 > 0.05, so it cannot reject at 5%.
  It is a distribution-free sanity check: say "k of 5 folds improved".
- The CI and p of the corrected t are reported as measured; a wide CI that
  includes 0 next to a high `P(right)` is stated plainly as both facts.
- A threshold crossed with k < 4 of 5 folds improving deserves a sentence on
  which folds disagree and why (lot effects), not a stronger claim.
- Plain paired t-tests on resampled or cross-validated scores are
  anti-conservative (Dietterich 1998; Nadeau & Bengio 2003): never report the
  uncorrected p.

## 5. Known limits

- Nadeau-Bengio was derived for random train/test resampling and is applied to
  k-fold by convention (Benavoli et al. 2017, footnote 2). The ratio used is
  pooled `sum(num_test) / sum(num_train)`, with `val` excluded from the
  denominator (conservative: a larger ratio widens the interval).
- With n = 5 the t distribution has 4 degrees of freedom, so intervals are wide
  by construction. A narrow interval at n = 5 should be double-checked.
- A lot-level cluster bootstrap is **invalid with 5 clusters** (Cameron, Gelbach
  & Miller 2008 report over-rejection with 5-30 clusters). Do not add one.
- An image-level bootstrap inside one lot is anti-conservative: it treats
  correlated photographs of the same mangos as independent.
- No per-image predictions are persisted today (by design), so image-level
  analyses cannot be run after the fact.
- A significance verdict on cross-validated scores can flip when the same
  experiment is repeated with different random partitions (Bouckaert & Frank
  2004 measure this replicability): another reason to fix the seed set up front.

## 6. Seeds

```bash
# after a run: estimate sigma_fold and sigma_seed from kfold_results.json
python scripts/plan_seeds.py --results reports/kfold_lot/kfold_results.json --split test --metric ap50
# before any run: what-if from priors
python scripts/plan_seeds.py --sigma-fold 0.05 --sigma-seed 0.144 --folds 5 --ratio 0.333 --max-seeds 8
```

- Model: `d_ks = mu + u_k + e_ks`. `u_k` is the lot-by-arm interaction
  (`sigma_fold`), `e_ks` is seed noise (`sigma_seed`). More seeds shrink only the
  `sigma_seed^2 / S` term of `se(S)`.
- The **plateau SE** `sqrt(sigma2_fold * (1/K + ratio))` is the floor seeds cannot
  beat. When SE approaches it, the study is **lot-limited** and only more lots
  help; stop adding seeds.
- Results mode needs 2+ seeds present in every fold (else pass `--sigma-seed`).
  With 5 folds `sigma_fold` is poorly identified and is often truncated to 0:
  that is weak evidence of "no interaction", not proof; the plateau is then
  optimistic.
- Known prior: the 3-seed sigma of test damage AP50 on the standard split was
  **0.102** for a single arm (`reports/twostream-v3-3seeds`, trained before the
  NIR registration fix, so treat it as a rough prior). The sigma of a difference of two independent arms
  is about sqrt(2) times larger (0.144, the value in the example). `sigma_fold`
  has no prior; 0.05 is a placeholder.
- The MDE printed is the normal-approximation formula with t quantiles, not an
  exact non-central-t power calculation: a planning figure.
- Seed variance is real (Bouthillier et al. 2021): never compare single runs.
  Do not extend the seed set after looking at the results (optional stopping).

## 7. Multiple arms

Compare each arm against the same baseline and say how many comparisons exist.
Apply a Holm step-down correction to the corrected-t p-values across the m
comparisons (smallest p vs alpha/m, next vs alpha/(m-1), ... stop at the first
failure); the `statistical-analysis` skill covers it if available. Holm does not
adjust the posterior probabilities, so pre-register **one** primary comparison
and label the others exploratory.

## 8. Reporting checklist

- [ ] Pre-registration note (arms, metric, ROPE, threshold, seeds) committed
      before the first test-lot evaluation.
- [ ] All 5 folds and all 3 seeds present; `missing` in the JSON is empty, or
      each gap is explained.
- [ ] Per-fold deltas (test), mean delta, corrected 95% CI and p.
- [ ] `P(left)` / `P(rope)` / `P(right)` with the ROPE, and the sentence from
      section 4 that they license.
- [ ] Sign count k/5 and the exact sign-flip p with its 0.0625 floor; the word
      "significant" appears nowhere on its account.
- [ ] `val` results labelled as checkpoint-selection data, not evidence.
- [ ] The standard 188/23/25 split described as a development signal.
- [ ] Limits stated: 5 lots, df = 4, Nadeau-Bengio by convention.
- [ ] Seed budget: `plan_seeds.py` output or the reason it was skipped.
- [ ] Arms, configs and commit hash identical across folds and seeds.

## 9. References

- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. Machine Learning, 52(3), 239–281. https://doi.org/10.1023/A:1024068626366
- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 10(7), 1895–1923. https://doi.org/10.1162/089976698300017197
- Bouckaert, R. R., & Frank, E. (2004). Evaluating the replicability of significance tests for comparing learning algorithms. In PAKDD 2004, Lecture Notes in Computer Science, pp. 3–12. https://doi.org/10.1007/978-3-540-24775-3_3
- Corani, G., & Benavoli, A. (2015). A Bayesian approach for comparing cross-validated algorithms on multiple data sets. Machine Learning, 100(2–3), 285–304. https://doi.org/10.1007/s10994-015-5486-z
- Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). Time for a change: A tutorial for comparing multiple classifiers through Bayesian analysis. Journal of Machine Learning Research, 18(77), 1–36. https://jmlr.org/papers/v18/16-305.html
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. Review of Economics and Statistics, 90(3), 414–427. https://doi.org/10.1162/rest.90.3.414
- Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. Proceedings of Machine Learning and Systems 3 (MLSys 2021). arXiv:2103.03098.
