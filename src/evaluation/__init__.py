"""Statistics for comparing arms across leave-one-lot-out folds.

Modules:
    fold_inference: Nadeau-Bengio corrected resampled t, correlated Bayesian
                    posterior (Benavoli et al. 2017) and an exact sign-flip
                    test, all on the fold-level paired differences.
    seed_planning:  Variance components (lot-by-arm vs. seed noise) and the
                    standard error / minimum detectable effect as a function
                    of the number of seeds.
    detectability:  E0 of docs/tesis/plan-experimental.md: disk-vs-ring
                    contrast, AUROC, mango-cluster bootstrap and grouped
                    out-of-fold logistic scores.

These modules are pure functions of numbers (numpy + scipy.stats); reading
metrics, images and writing reports stays in `scripts/aggregate_kfold.py`,
`scripts/plan_seeds.py` and `scripts/e0_detectability.py`.
"""
