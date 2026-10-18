#!/usr/bin/env python3
"""Aggregate the leave-one-lot-out evaluations written by `scripts/run_kfold.py`.

Reads `<reports-root>/<arm>/fold<k>_seed<S>/<split>/metrics.json` (the file
`scripts/evaluate_checkpoint.py` writes) and produces, for the damage class
(id 1) on `val` and `test`:

  * a per-fold table per arm (AP50, AP50-95, precision, recall, number of
    damage GT boxes) and the mean +/- sigma across folds (sigma is the sample
    standard deviation, ddof=1);
  * the PAIRED difference `treatment - baseline` per fold (default
    twostream_roi_crop - twostream) with its mean +/- sigma across folds.

With several seeds, a fold's value is the mean over the seeds present, and a
fold's paired difference averages `treatment - baseline` over the seeds both
arms have. Precision and recall are read at the `conf_threshold` operating
point; AP50 / AP50-95 integrate the whole ranked list.

Runs that are absent or unreadable are listed as missing in both outputs and
never enter a statistic.

Paired inference. The paired fold differences are not independent (the folds
share training lots), so a plain t-test is anti-conservative. Next to the
descriptive mean +/- sigma, `paired.inference[split][metric]` holds, on the
seed-averaged fold-level differences (see `src/evaluation/fold_inference.py`):
the Nadeau-Bengio corrected resampled t (CI and p), the correlated Bayesian
posterior (P(left) / P(rope) / P(right) for a ROPE set by `--rope`) and an
exact sign-flip test. The test/train ratio comes from the per-fold sizes in
`--fold-spec` (default `configs/folds/lot_5fold.json`). A metric whose
differences cannot be analysed (zero variance, fewer than 2 folds) gets
`{"error": ...}` instead of numbers. `paired.per_fold_seed` keeps the
per-seed differences for `scripts/plan_seeds.py`.

Outputs: `<reports-root>/kfold_results.json` and `<reports-root>/kfold_results.md`
(deterministic: the same metrics give byte-identical files).

Usage:
    python scripts/aggregate_kfold.py
    python scripts/aggregate_kfold.py --reports-root reports/kfold_lot --folds 0,1,2,3,4
    python scripts/aggregate_kfold.py --seeds 42 1337 2024 --rope 0.02
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_kfold import DEFAULT_ARMS, DEFAULT_REPORTS_ROOT, arm_name, parse_int_list  # noqa: E402
from src.evaluation import fold_inference  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

DAMAGE_CLASS_ID = 1
SPLITS = ("val", "test")
METRICS = ("ap50", "ap50_95", "precision", "recall")
METRIC_LABELS = {"ap50": "AP50", "ap50_95": "AP50-95", "precision": "Precision", "recall": "Recall"}
DEFAULT_FOLDS = "0,1,2,3,4"
DEFAULT_FOLD_SPEC = "configs/folds/lot_5fold.json"
DEFAULT_CONFIDENCE = 0.95


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def _class_value(per_class: dict, cls_id: int = DAMAGE_CLASS_ID):
    """`metrics.json` keys are strings; in-memory metrics use ints."""
    if str(cls_id) in per_class:
        return per_class[str(cls_id)]
    return per_class[cls_id]


def damage_summary(metrics: dict) -> dict:
    """AP50, AP50-95, precision, recall and GT box count of the damage class."""
    counts = _class_value(metrics["per_class_counts"])
    return {
        "ap50": float(_class_value(metrics["per_class_ap_50"])),
        "ap50_95": float(_class_value(metrics["per_class_ap_50_95"])),
        "num_gt": int(counts["tp"]) + int(counts["fn"]),     # tp + fn == GT boxes
        "precision": float(_class_value(metrics["per_class_precision"])),
        "recall": float(_class_value(metrics["per_class_recall"])),
    }


def mean_std(values: list[float]) -> dict:
    """Mean and sample standard deviation (ddof=1; None when fewer than 2 values)."""
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
    }


def load_run(reports_root: Path, arm: str, fold: int, seed: int, split: str) -> dict:
    """One evaluation: `{"status": "ok", **summary}`, or a missing/invalid record."""
    path = Path(reports_root) / arm / f"fold{fold}_seed{seed}" / split / "metrics.json"
    record = {"arm": arm, "fold": fold, "seed": seed, "split": split}
    if not path.is_file():
        return {**record, "status": "missing", "reason": f"{path} not found"}
    try:
        return {**record, "status": "ok", **damage_summary(json.loads(path.read_text()))}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {**record, "status": "invalid", "reason": f"{path}: {type(error).__name__}: {error}"}


# --------------------------------------------------------------------------
# Fold spec and paired inference
# --------------------------------------------------------------------------


def load_fold_spec(path: Path) -> list[dict]:
    """The `folds` list of a fold assignment file (`configs/folds/lot_5fold.json`).

    Every entry needs `fold`, `num_test` and `num_train`. Raises ValueError, with
    the path in the message, for an unreadable or malformed file.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except OSError as error:
        raise ValueError(f"Cannot read the fold spec {path}: {error}") from None
    except ValueError as error:                      # json.JSONDecodeError and UnicodeDecodeError
        raise ValueError(f"The fold spec {path} is not valid JSON: {error}") from None
    folds = data.get("folds") if isinstance(data, dict) else None
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"The fold spec {path} has no non-empty 'folds' list.")
    seen: set = set()
    for entry in folds:
        for key in ("fold", "num_test", "num_train"):
            if not isinstance(entry, dict) or key not in entry:
                raise ValueError(f"The fold spec {path} has a fold entry without '{key}': {entry!r}")
        if entry["fold"] in seen:
            raise ValueError(f"The fold spec {path} lists fold {entry['fold']} twice.")
        seen.add(entry["fold"])
    return folds


def check_folds_in_spec(fold_spec: Sequence[Mapping], folds: Sequence[int]) -> dict[int, Mapping]:
    """The spec entries by fold id; raises ValueError naming the folds the spec does not know."""
    by_id = {int(entry["fold"]): entry for entry in fold_spec}
    unknown = sorted(set(folds) - set(by_id))
    if unknown:
        raise ValueError(
            f"fold(s) {', '.join(map(str, unknown))} are not in the fold spec "
            f"(known folds: {', '.join(map(str, sorted(by_id)))})."
        )
    return by_id


def paired_inference(
    diffs: Sequence[float],
    spec_rows: Sequence[Mapping],
    rope: float | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict:
    """Corrected t, correlated Bayes and sign-flip on one metric's fold-level differences.

    `spec_rows` are the fold-spec entries of the folds behind `diffs` (they give
    the test/train ratio). Returns `{"error": message}` when the statistics are
    undefined (fewer than 2 folds, zero variance) instead of raising.
    """
    try:
        ratio = fold_inference.test_train_ratio(spec_rows)
        corrected = fold_inference.corrected_resampled_t(diffs, ratio, confidence)
        bayes = fold_inference.correlated_bayesian_t(diffs, ratio, rope=rope or 0.0, credibility=confidence)
        sign_flip = fold_inference.sign_flip_test(diffs)
    except ValueError as error:
        logger.info("Paired inference unavailable: %s", error)
        return {"error": str(error)}
    return {
        "n_folds": len(diffs),
        "n_positive": sum(1 for d in diffs if d > 0),
        "test_train_ratio": ratio,
        "corrected_t": dataclasses.asdict(corrected),
        "bayes": dataclasses.asdict(bayes),
        "sign_flip": dataclasses.asdict(sign_flip),
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate(
    reports_root: Path,
    arms: list[str],
    folds: list[int],
    seeds: list[int],
    baseline: str,
    treatment: str,
    fold_spec: Sequence[Mapping] | None = None,
    rope: float | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict:
    """Load every expected evaluation and compute the per-fold, summary and paired results.

    With a `fold_spec` (the `folds` list of the fold assignment file) the paired
    block also gets `inference` (see the module docstring); every requested fold
    must exist in it, else ValueError.
    """
    spec_by_id = check_folds_in_spec(fold_spec, folds) if fold_spec is not None else None
    runs = {
        (arm, fold, seed, split): load_run(reports_root, arm, fold, seed, split)
        for arm in arms for fold in folds for seed in seeds for split in SPLITS
    }
    missing = [
        {k: run[k] for k in ("arm", "fold", "seed", "split", "status", "reason")}
        for run in runs.values() if run["status"] != "ok"
    ]

    per_fold: dict = {}
    summary: dict = {}
    for arm in arms:
        per_fold[arm], summary[arm] = {}, {}
        for split in SPLITS:
            rows = {}
            for fold in folds:
                ok = [runs[(arm, fold, seed, split)] for seed in sorted(seeds)
                      if runs[(arm, fold, seed, split)]["status"] == "ok"]
                if ok:
                    rows[str(fold)] = {
                        **{m: statistics.fmean(r[m] for r in ok) for m in METRICS},
                        "num_gt": ok[0]["num_gt"],
                        "seeds": [r["seed"] for r in ok],
                    }
            per_fold[arm][split] = rows
            summary[arm][split] = {m: mean_std([row[m] for row in rows.values()]) for m in METRICS}

    paired_rows: dict = {}
    paired_summary: dict = {}
    paired_by_seed: dict = {}
    inference: dict = {}
    for split in SPLITS:
        rows = {}
        by_seed: dict = {m: {} for m in METRICS}
        for fold in folds:
            common = [
                seed for seed in sorted(seeds)
                if runs[(baseline, fold, seed, split)]["status"] == "ok"
                and runs[(treatment, fold, seed, split)]["status"] == "ok"
            ]
            if common:
                deltas = {
                    m: [runs[(treatment, fold, s, split)][m] - runs[(baseline, fold, s, split)][m] for s in common]
                    for m in METRICS
                }
                rows[str(fold)] = {**{m: statistics.fmean(deltas[m]) for m in METRICS}, "seeds": common}
                for m in METRICS:
                    by_seed[m][str(fold)] = {str(s): d for s, d in zip(common, deltas[m])}
        paired_rows[split] = rows
        paired_by_seed[split] = by_seed
        paired_summary[split] = {m: mean_std([row[m] for row in rows.values()]) for m in METRICS}
        if spec_by_id is not None:
            present = [fold for fold in folds if str(fold) in rows]
            inference[split] = {
                m: paired_inference([rows[str(fold)][m] for fold in present],
                                    [spec_by_id[fold] for fold in present], rope, confidence)
                for m in METRICS
            }

    paired: dict = {
        "baseline": baseline,
        "per_fold": paired_rows,
        "per_fold_seed": paired_by_seed,
        "summary": paired_summary,
        "treatment": treatment,
    }
    if spec_by_id is not None:
        paired["inference"] = inference
        paired["inference_settings"] = {"confidence": confidence, "rope": rope}

    return {
        "arms": list(arms),
        "damage_class_id": DAMAGE_CLASS_ID,
        "folds": list(folds),
        "metrics": list(METRICS),
        "missing": missing,
        "paired": paired,
        "per_fold": per_fold,
        "reports_root": str(reports_root),
        "seeds": list(seeds),
        "splits": list(SPLITS),
        "summary": summary,
    }


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _fmt_stat(stat: dict, signed: bool = False) -> str:
    if stat["mean"] is None:
        return "—"
    sigma = "n/a" if stat["std"] is None else f"{stat['std']:.4f}"
    return f"{_fmt(stat['mean'], signed)} ± {sigma}"


def _fmt_p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _render_inference(results: dict) -> list[str]:
    """The 'Paired inference' section: one AP50 row per split."""
    paired = results["paired"]
    confidence, rope = paired["inference_settings"]["confidence"], paired["inference_settings"]["rope"]
    entries = {split: paired["inference"][split]["ap50"] for split in results["splits"]}
    ratios = sorted({f"{e['test_train_ratio']:.4f}" for e in entries.values() if "error" not in e})
    rope_text = ("ROPE ±" + f"{rope:g}") if rope else "no ROPE set (P(rope) = 0)"
    pct = f"{confidence * 100:g}%"
    header = ["split", "n", "mean Δ", f"{pct} CI", "p (two-sided)", "P(left)", "P(rope)", "P(right)",
              "sign k/n", "sign-flip p", "min p"]
    lines = [
        "", "## Paired inference", "",
        f"AP50, paired difference {paired['treatment']} − {paired['baseline']}; n is the number of folds "
        "with a pair (a fold's value is the mean over its seeds). The folds share training lots, so a "
        f"plain paired t-test is anti-conservative: the {pct} CI and p come from the Nadeau–Bengio "
        "corrected resampled t (variance × (1/n + n_test/n_train)"
        + (f", n_test/n_train = {', '.join(ratios)}" if ratios else "") + ", df = n − 1). "
        f"P(left) / P(rope) / P(right) are the posterior mass of Δ < −ROPE / inside the ROPE / Δ > ROPE "
        f"(correlated Bayesian t, Benavoli et al. 2017; {rope_text}). sign k/n counts folds with Δ > 0. "
        "The sign-flip p is an exact randomization test and min p is the smallest value it can reach with "
        "n folds (2/2^n): with 5 folds it cannot support a significance claim. "
        "val is the checkpoint-selection split, so its rows are optimistic; claims rest on test. "
        "Every metric is in `kfold_results.json` under `paired.inference`.",
        "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header),
    ]
    for split in results["splits"]:
        entry = entries[split]
        if "error" in entry:
            cells = [split, "—", f"error: {entry['error']}"] + ["—"] * (len(header) - 3)
        else:
            ct, bayes, flip = entry["corrected_t"], entry["bayes"], entry["sign_flip"]
            cells = [
                split, str(entry["n_folds"]), _fmt(ct["mean"], signed=True),
                f"[{_fmt(ct['ci_low'], signed=True)}, {_fmt(ct['ci_high'], signed=True)}]",
                _fmt_p(ct["p_two_sided"]),
                f"{bayes['p_left']:.3f}", f"{bayes['p_rope']:.3f}", f"{bayes['p_right']:.3f}",
                f"{entry['n_positive']}/{entry['n_folds']}",
                _fmt_p(flip["p_two_sided"]) + ("" if flip["exact"] else " (MC)"),
                _fmt_p(flip["min_attainable_p_two_sided"]),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_markdown(results: dict) -> str:
    """Per-fold tables, mean ± σ, paired difference, paired inference and the missing-run list."""
    folds, seeds = results["folds"], results["seeds"]
    base, treat = results["paired"]["baseline"], results["paired"]["treatment"]
    lines = [
        "# Leave-one-lot-out results",
        "",
        f"Damage class (id {results['damage_class_id']}). Arms: {', '.join(results['arms'])}. "
        f"Folds: {', '.join(map(str, folds))}. Seeds: {', '.join(map(str, seeds))}.",
        "",
        "AP50 and AP50-95 integrate the ranked detections; precision and recall are read at the "
        "`conf_threshold` operating point; #GT is the number of damage GT boxes in the evaluated split. "
        "**mean ± σ** is across folds, σ the sample standard deviation (ddof=1, n/a for a single fold); "
        "a fold with several seeds contributes the mean over its seeds.",
    ]
    labels = [METRIC_LABELS[m] for m in METRICS]
    for split in results["splits"]:
        lines += ["", f"## {split}"]
        for arm in results["arms"]:
            lines += ["", f"### {arm}", "", "| fold | seeds | " + " | ".join(labels) + " | #GT |",
                      "|---|---|" + "---|" * (len(METRICS) + 1)]
            rows = results["per_fold"][arm][split]
            for fold in folds:
                row = rows.get(str(fold))
                if row is None:
                    lines.append(f"| {fold} | missing | " + " | ".join("—" for _ in range(len(METRICS) + 1)) + " |")
                else:
                    lines.append(
                        f"| {fold} | {', '.join(map(str, row['seeds']))} | "
                        + " | ".join(_fmt(row[m]) for m in METRICS) + f" | {row['num_gt']} |"
                    )
            stats = results["summary"][arm][split]
            n = stats["ap50"]["n"]
            lines.append(
                f"| **mean ± σ** | n={n} of {len(folds)} | "
                + " | ".join(_fmt_stat(stats[m]) for m in METRICS) + " | |"
            )
        lines += ["", f"### Paired difference {treat} − {base} ({split})", "",
                  "| fold | seeds | " + " | ".join(f"Δ{label}" for label in labels) + " |",
                  "|---|---|" + "---|" * len(METRICS)]
        rows = results["paired"]["per_fold"][split]
        for fold in folds:
            row = rows.get(str(fold))
            if row is None:
                lines.append(f"| {fold} | missing | " + " | ".join("—" for _ in METRICS) + " |")
            else:
                lines.append(
                    f"| {fold} | {', '.join(map(str, row['seeds']))} | "
                    + " | ".join(_fmt(row[m], signed=True) for m in METRICS) + " |"
                )
        stats = results["paired"]["summary"][split]
        lines.append(
            f"| **mean ± σ** | n={stats['ap50']['n']} of {len(folds)} | "
            + " | ".join(_fmt_stat(stats[m], signed=True) for m in METRICS) + " |"
        )

    if "inference" in results["paired"]:
        lines += _render_inference(results)

    lines += ["", "## Missing runs", ""]
    if not results["missing"]:
        lines.append("None: every expected evaluation was found.")
    else:
        lines.append(f"{len(results['missing'])} expected evaluation(s) are not in the statistics above:")
        lines.append("")
        for m in results["missing"]:
            lines.append(f"- {m['arm']} fold {m['fold']} seed {m['seed']} {m['split']}: {m['status']}"
                         + (f" ({m['reason']})" if m["status"] == "invalid" else ""))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate leave-one-lot-out evaluations.")
    parser.add_argument("--reports-root", default=None,
                        help=f"Where run_kfold.py wrote the evaluations (default: <repo>/{DEFAULT_REPORTS_ROOT}).")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Arm names, i.e. experiment config stems (default: "
                             + ", ".join(arm_name(a) for a in DEFAULT_ARMS) + ").")
    parser.add_argument("--folds", type=parse_int_list, default=parse_int_list(DEFAULT_FOLDS),
                        help=f"Comma-separated expected folds (default: {DEFAULT_FOLDS}).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Expected seeds (default: 42).")
    parser.add_argument("--baseline", default=None, help="Baseline arm of the paired difference (default: first arm).")
    parser.add_argument("--treatment", default=None, help="Treatment arm of the paired difference (default: last arm).")
    parser.add_argument("--fold-spec", default=None,
                        help="Fold assignment file with the per-fold num_test / num_train used for the test/train "
                             f"ratio of the paired inference (default: <repo>/{DEFAULT_FOLD_SPEC}).")
    parser.add_argument("--rope", type=float, default=None,
                        help="Half-width of the region of practical equivalence, in metric units "
                             "(e.g. 0.02 = 2 AP points); applied to every metric. Default: no ROPE (P(rope) = 0).")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                        help=f"Confidence / credibility level of the intervals (default: {DEFAULT_CONFIDENCE}).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports_root = (Path(args.reports_root) if args.reports_root is not None
                    else PROJECT_ROOT / DEFAULT_REPORTS_ROOT)
    arms = args.arms if args.arms is not None else [arm_name(a) for a in DEFAULT_ARMS]
    baseline, treatment = args.baseline or arms[0], args.treatment or arms[-1]
    for name in (baseline, treatment):
        if name not in arms:
            print(f"ERROR: arm '{name}' is not in --arms {arms}", file=sys.stderr)
            return 2
    if args.rope is not None and not (math.isfinite(args.rope) and args.rope >= 0):
        print(f"ERROR: --rope must be finite and >= 0, got {args.rope}", file=sys.stderr)
        return 2
    if not (math.isfinite(args.confidence) and 0.0 < args.confidence < 1.0):
        print(f"ERROR: --confidence must be in (0, 1), got {args.confidence}", file=sys.stderr)
        return 2

    fold_spec_path = Path(args.fold_spec) if args.fold_spec is not None else PROJECT_ROOT / DEFAULT_FOLD_SPEC
    try:
        fold_spec = load_fold_spec(fold_spec_path)
        check_folds_in_spec(fold_spec, args.folds)
    except ValueError as error:
        print(f"ERROR: {error} (fold spec: {fold_spec_path})", file=sys.stderr)
        return 2

    results = aggregate(reports_root, arms, args.folds, args.seeds, baseline, treatment,
                        fold_spec=fold_spec, rope=args.rope, confidence=args.confidence)
    if len(results["missing"]) == len(arms) * len(args.folds) * len(args.seeds) * len(SPLITS):
        print(f"ERROR: No evaluation results found under {reports_root} "
              f"(expected <arm>/fold<k>_seed<S>/<split>/metrics.json).", file=sys.stderr)
        return 1
    results["fold_spec"] = str(fold_spec_path)

    markdown = render_markdown(results)
    (reports_root / "kfold_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (reports_root / "kfold_results.md").write_text(markdown)
    print(markdown)
    print(f"Wrote {reports_root / 'kfold_results.json'} and {reports_root / 'kfold_results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
