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

Outputs: `<reports-root>/kfold_results.json` and `<reports-root>/kfold_results.md`
(deterministic: the same metrics give byte-identical files).

Usage:
    python scripts/aggregate_kfold.py
    python scripts/aggregate_kfold.py --reports-root reports/kfold_lot --folds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_kfold import DEFAULT_ARMS, DEFAULT_REPORTS_ROOT, arm_name, parse_int_list  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DAMAGE_CLASS_ID = 1
SPLITS = ("val", "test")
METRICS = ("ap50", "ap50_95", "precision", "recall")
METRIC_LABELS = {"ap50": "AP50", "ap50_95": "AP50-95", "precision": "Precision", "recall": "Recall"}
DEFAULT_FOLDS = "0,1,2,3,4"


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
# Aggregation
# --------------------------------------------------------------------------


def aggregate(
    reports_root: Path,
    arms: list[str],
    folds: list[int],
    seeds: list[int],
    baseline: str,
    treatment: str,
) -> dict:
    """Load every expected evaluation and compute the per-fold, summary and paired results."""
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
    for split in SPLITS:
        rows = {}
        for fold in folds:
            common = [
                seed for seed in sorted(seeds)
                if runs[(baseline, fold, seed, split)]["status"] == "ok"
                and runs[(treatment, fold, seed, split)]["status"] == "ok"
            ]
            if common:
                rows[str(fold)] = {
                    **{
                        m: statistics.fmean(
                            runs[(treatment, fold, s, split)][m] - runs[(baseline, fold, s, split)][m]
                            for s in common
                        )
                        for m in METRICS
                    },
                    "seeds": common,
                }
        paired_rows[split] = rows
        paired_summary[split] = {m: mean_std([row[m] for row in rows.values()]) for m in METRICS}

    return {
        "arms": list(arms),
        "damage_class_id": DAMAGE_CLASS_ID,
        "folds": list(folds),
        "metrics": list(METRICS),
        "missing": missing,
        "paired": {
            "baseline": baseline,
            "per_fold": paired_rows,
            "summary": paired_summary,
            "treatment": treatment,
        },
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


def render_markdown(results: dict) -> str:
    """Per-fold tables, mean ± σ, paired difference and the missing-run list."""
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

    results = aggregate(reports_root, arms, args.folds, args.seeds, baseline, treatment)
    if len(results["missing"]) == len(arms) * len(args.folds) * len(args.seeds) * len(SPLITS):
        print(f"ERROR: No evaluation results found under {reports_root} "
              f"(expected <arm>/fold<k>_seed<S>/<split>/metrics.json).", file=sys.stderr)
        return 1

    markdown = render_markdown(results)
    (reports_root / "kfold_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (reports_root / "kfold_results.md").write_text(markdown)
    print(markdown)
    print(f"Wrote {reports_root / 'kfold_results.json'} and {reports_root / 'kfold_results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
