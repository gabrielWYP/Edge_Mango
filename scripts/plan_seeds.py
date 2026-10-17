#!/usr/bin/env python3
"""How many seeds per fold are worth running? Standard error and MDE per seed count.

The paired fold-level difference is `d_{k,s} = mu + u_k + e_{k,s}` with a
lot-by-arm interaction `u_k` (variance sigma2_fold) and seed noise `e_{k,s}`
(variance sigma2_seed). Averaging over S seeds shrinks only the seed term, so
the standard error of the mean difference over K folds (Nadeau-Bengio
corrected) is

    se(S) = sqrt((sigma2_fold + sigma2_seed / S) * (1/K + n_test/n_train))

and it never drops below the plateau `sqrt(sigma2_fold * (1/K + n_test/n_train))`:
when `se` approaches it the study is lot-limited and only more lots help. The
MDE is `(t_{1-alpha/2, K-1} + t_{power, K-1}) * se` (normal-approximation
formula with t quantiles, not an exact non-central-t power calculation).
See `src/evaluation/seed_planning.py`.

Two modes:

  * results mode: estimate both variance components from the per-seed paired
    differences that `scripts/aggregate_kfold.py` stored in
    `kfold_results.json` (`paired.per_fold_seed`). Needs at least 2 seeds
    present in every fold, or `--sigma-seed` to supply the seed noise.
    K and n_test/n_train come from the folds in the file and `--fold-spec`.
  * priors mode (what-if, before any run): give `--sigma-fold` and
    `--sigma-seed` (standard deviations, in metric units), `--folds` and
    `--ratio`.

Prints a markdown table (seeds, SE, MDE) with the plateau and the variance
components; `--json-out` also writes them as JSON.

Usage:
    python scripts/plan_seeds.py --results reports/kfold_lot/kfold_results.json --split test --metric ap50
    python scripts/plan_seeds.py --sigma-fold 0.05 --sigma-seed 0.144 --folds 5 --ratio 0.333 --max-seeds 8
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.aggregate_kfold import (  # noqa: E402
    DEFAULT_FOLD_SPEC,
    METRICS,
    SPLITS,
    check_folds_in_spec,
    load_fold_spec,
)
from src.evaluation import fold_inference, seed_planning  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

DEFAULT_PRIOR_FOLDS = 5
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
DEFAULT_MAX_SEEDS = 10


# --------------------------------------------------------------------------
# Reading the aggregator output
# --------------------------------------------------------------------------


def load_seed_matrix(results_path: Path, split: str, metric: str) -> tuple[list[int], list[int], np.ndarray, list[int]]:
    """`(folds, seeds, K x S matrix, dropped seeds)` of one split and metric.

    Only the seeds present in every fold enter the (balanced) matrix; the others
    are returned as `dropped`. Raises ValueError for an unreadable file, results
    written before `paired.per_fold_seed` existed, or a split / metric / fold
    count that cannot be analysed.
    """
    results_path = Path(results_path)
    try:
        data = json.loads(results_path.read_text())
    except OSError as error:
        raise ValueError(f"Cannot read the results file {results_path}: {error}") from None
    except ValueError as error:
        raise ValueError(f"The results file {results_path} is not valid JSON: {error}") from None
    try:
        per_seed = data["paired"]["per_fold_seed"]
    except (KeyError, TypeError):
        raise ValueError(
            f"{results_path} has no paired.per_fold_seed (written by an older aggregate_kfold.py); "
            "re-run scripts/aggregate_kfold.py."
        ) from None
    if split not in per_seed:
        raise ValueError(f"The results have no '{split}' split (found: {', '.join(sorted(per_seed))}).")
    if metric not in per_seed[split]:
        raise ValueError(f"The results have no '{metric}' metric for '{split}' (found: {', '.join(sorted(per_seed[split]))}).")

    by_fold = per_seed[split][metric]
    folds = sorted(int(fold) for fold in by_fold)
    if len(folds) < 2:
        raise ValueError(f"Need paired values for at least 2 folds, the results have {len(folds)} ({split}, {metric}).")
    seed_sets = [set(by_fold[str(fold)]) for fold in folds]
    common = sorted(set.intersection(*seed_sets), key=int)
    dropped = sorted(set.union(*seed_sets) - set(common), key=int)
    if not common:
        raise ValueError("No seed is present in every fold, so there is nothing to compare across folds.")
    matrix = np.array([[by_fold[str(fold)][seed] for seed in common] for fold in folds], dtype=float)
    return folds, [int(s) for s in common], matrix, [int(s) for s in dropped]


# --------------------------------------------------------------------------
# Planning and rendering
# --------------------------------------------------------------------------


def build_plan(
    *,
    mode: str,
    source: dict | None,
    n_folds: int,
    n_seeds_observed: int | None,
    ratio: float,
    sigma2_fold: float,
    sigma2_seed: float,
    truncated: bool,
    alpha: float,
    power: float,
    max_seeds: int,
    extra: dict | None = None,
) -> dict:
    """The JSON-able plan: inputs, variance components, plateau and one row per seed count."""
    rows = seed_planning.project_seed_budget(sigma2_fold, sigma2_seed, n_folds, ratio,
                                             seeds=range(1, max_seeds + 1), alpha=alpha, power=power)
    return {
        **(extra or {}),
        "alpha": alpha,
        "mode": mode,
        "n_folds": n_folds,
        "n_seeds_observed": n_seeds_observed,
        "plateau_se": seed_planning.plateau_se(sigma2_fold, n_folds, ratio),
        "power": power,
        "rows": [{"mde": r.mde, "n_seeds": r.n_seeds, "se": r.se} for r in rows],
        "sigma2_fold": sigma2_fold,
        "sigma2_seed": sigma2_seed,
        "source": source,
        "test_train_ratio": ratio,
        "truncated": truncated,
    }


def render_plan(plan: dict) -> str:
    """Markdown: design, variance components, the (seeds, SE, MDE) table and the plateau."""
    sigma_fold, sigma_seed = math.sqrt(plan["sigma2_fold"]), math.sqrt(plan["sigma2_seed"])
    if plan["mode"] == "results":
        src = plan["source"]
        origin = f"{src['results']} ({src['split']}, {src['metric']})"
        components = (f"Variance components estimated from {plan['n_folds']} folds × "
                      f"{plan['n_seeds_observed']} seed(s):")
    else:
        origin = "priors given on the command line"
        components = "Variance components (priors, not estimated):"
    lines = [
        "# Seed budget",
        "",
        f"Source: {origin}.",
        f"Design: K = {plan['n_folds']} folds, n_test/n_train = {plan['test_train_ratio']:.4f}, "
        f"alpha = {plan['alpha']:g}, power = {plan['power']:g}.",
        "",
        components,
        "",
        f"- σ_seed = {sigma_seed:.4f} (σ² = {plan['sigma2_seed']:.3e}): noise between seeds within a fold",
        f"- σ_fold = {sigma_fold:.4f} (σ² = {plan['sigma2_fold']:.3e}): lot-by-arm interaction",
    ]
    if plan["truncated"]:
        lines += [
            "",
            "σ_fold was truncated at 0: the fold means vary no more than seed noise explains. With so few "
            "folds this is weak evidence that the interaction is absent, so the plateau below may be too optimistic.",
        ]
    lines += ["", "| seeds | SE | MDE |", "|---|---|---|"]
    lines += [f"| {r['n_seeds']} | {r['se']:.4f} | {r['mde']:.4f} |" for r in plan["rows"]]
    plateau, last = plan["plateau_se"], plan["rows"][-1]
    lines += ["", f"Plateau SE (σ_fold only, infinitely many seeds): {plateau:.4f}."]
    if plateau > 0:
        lines.append(
            f"With {last['n_seeds']} seeds SE is {100 * (last['se'] / plateau - 1):.1f}% above the plateau. "
            "The nearer SE is to it, the more the study is lot-limited: extra seeds then buy little, "
            "and only more lots lower the plateau."
        )
    lines += [
        "",
        "MDE = (t_{1−α/2, K−1} + t_{power, K−1}) · SE, the usual normal-approximation formula with t "
        "quantiles; it is not an exact non-central-t power calculation.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard error and MDE as a function of the number of seeds.")
    parser.add_argument("--results", default=None,
                        help="kfold_results.json written by scripts/aggregate_kfold.py (results mode).")
    parser.add_argument("--split", choices=SPLITS, default="test", help="Split to read in results mode (default: test).")
    parser.add_argument("--metric", choices=METRICS, default="ap50", help="Metric to read in results mode (default: ap50).")
    parser.add_argument("--sigma-fold", type=float, default=None,
                        help="Priors mode: sd of the lot-by-arm interaction, in metric units.")
    parser.add_argument("--sigma-seed", type=float, default=None,
                        help="Priors mode: sd of the seed noise. Results mode: replaces the estimate "
                             "(required when only one seed is common to every fold).")
    parser.add_argument("--folds", type=int, default=None,
                        help=f"Priors mode: number of folds K (default: {DEFAULT_PRIOR_FOLDS}).")
    parser.add_argument("--ratio", type=float, default=None,
                        help="Priors mode: n_test/n_train (default: pooled ratio of --fold-spec).")
    parser.add_argument("--fold-spec", default=None,
                        help=f"Fold assignment file giving the per-fold sizes (default: <repo>/{DEFAULT_FOLD_SPEC}).")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help=f"Two-sided level (default: {DEFAULT_ALPHA}).")
    parser.add_argument("--power", type=float, default=DEFAULT_POWER, help=f"Target power (default: {DEFAULT_POWER}).")
    parser.add_argument("--max-seeds", type=int, default=DEFAULT_MAX_SEEDS,
                        help=f"Largest seed count in the table (default: {DEFAULT_MAX_SEEDS}).")
    parser.add_argument("--json-out", default=None, help="Also write the plan to this JSON file.")
    return parser.parse_args(argv)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _plan_from_results(args: argparse.Namespace, fold_spec_path: Path) -> dict:
    for flag, value in (("--sigma-fold", args.sigma_fold), ("--folds", args.folds), ("--ratio", args.ratio)):
        _require(value is None, f"{flag} is for priors mode and cannot be combined with --results "
                                "(K, the ratio and sigma_fold come from the results; use --sigma-seed to replace "
                                "the seed noise).")
    folds, seeds, matrix, dropped = load_seed_matrix(Path(args.results), args.split, args.metric)
    spec_by_id = check_folds_in_spec(load_fold_spec(fold_spec_path), folds)
    ratio = fold_inference.test_train_ratio([spec_by_id[fold] for fold in folds])
    if args.sigma_seed is None:
        _require(len(seeds) >= 2,
                 f"Only one seed ({seeds[0]}) is present in every fold, so the seed noise cannot be estimated: "
                 "pass --sigma-seed (e.g. 0.144 for a single arm's 3-seed sd on the standard split).")
        sigma2_seed = None
    else:
        sigma2_seed = args.sigma_seed ** 2
    if dropped:
        print(f"NOTE: seed(s) {', '.join(map(str, dropped))} are missing in some fold and were left out; "
              f"using seeds {', '.join(map(str, seeds))}.", file=sys.stderr)
    components = seed_planning.estimate_variance_components(matrix, sigma2_seed=sigma2_seed)
    return build_plan(
        mode="results",
        source={"metric": args.metric, "results": str(args.results), "split": args.split},
        n_folds=components.n_folds, n_seeds_observed=components.n_seeds, ratio=ratio,
        sigma2_fold=components.sigma2_fold, sigma2_seed=components.sigma2_seed, truncated=components.truncated,
        alpha=args.alpha, power=args.power, max_seeds=args.max_seeds,
        extra={"folds_used": folds, "seeds_used": seeds, "fold_means": list(components.fold_means)},
    )


def _plan_from_priors(args: argparse.Namespace, fold_spec_path: Path) -> dict:
    missing = [flag for flag, value in (("--sigma-fold", args.sigma_fold), ("--sigma-seed", args.sigma_seed))
               if value is None]
    _require(not missing, f"Priors mode needs {' and '.join(missing)} (or pass --results to estimate "
                          "the variance components from a kfold_results.json).")
    _require(math.isfinite(args.sigma_fold) and args.sigma_fold >= 0, f"--sigma-fold must be >= 0, got {args.sigma_fold}.")
    _require(math.isfinite(args.sigma_seed) and args.sigma_seed >= 0, f"--sigma-seed must be >= 0, got {args.sigma_seed}.")
    n_folds = DEFAULT_PRIOR_FOLDS if args.folds is None else args.folds
    _require(n_folds >= 2, f"--folds must be >= 2, got {n_folds}.")
    if args.ratio is None:
        ratio = fold_inference.test_train_ratio(load_fold_spec(fold_spec_path))
    else:
        ratio = args.ratio
        _require(math.isfinite(ratio) and ratio >= 0, f"--ratio must be >= 0, got {ratio}.")
    return build_plan(
        mode="priors", source=None, n_folds=n_folds, n_seeds_observed=None, ratio=ratio,
        sigma2_fold=args.sigma_fold ** 2, sigma2_seed=args.sigma_seed ** 2, truncated=False,
        alpha=args.alpha, power=args.power, max_seeds=args.max_seeds,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fold_spec_path = Path(args.fold_spec) if args.fold_spec is not None else PROJECT_ROOT / DEFAULT_FOLD_SPEC
    try:
        _require(args.max_seeds >= 1, f"--max-seeds must be >= 1, got {args.max_seeds}.")
        for flag, value in (("--alpha", args.alpha), ("--power", args.power)):
            _require(math.isfinite(value) and 0.0 < value < 1.0, f"{flag} must be in (0, 1), got {value}.")
        plan = (_plan_from_results(args, fold_spec_path) if args.results is not None
                else _plan_from_priors(args, fold_spec_path))
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(render_plan(plan))
    if args.json_out is not None:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
