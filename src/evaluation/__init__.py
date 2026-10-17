"""Statistics for comparing arms across leave-one-lot-out folds.

Modules:
    fold_inference: Nadeau-Bengio corrected resampled t, correlated Bayesian
                    posterior (Benavoli et al. 2017) and an exact sign-flip
                    test, all on the fold-level paired differences.
    seed_planning:  Variance components (lot-by-arm vs. seed noise) and the
                    standard error / minimum detectable effect as a function
                    of the number of seeds.

Both modules are pure functions of numbers (numpy + scipy.stats); reading
metrics and writing reports stays in `scripts/aggregate_kfold.py` and
`scripts/plan_seeds.py`.
"""
