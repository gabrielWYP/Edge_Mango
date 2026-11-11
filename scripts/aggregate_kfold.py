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

Latent-window metrics. A run may hold a `metrics_v4.json` next to its
`metrics.json` (`scripts/v4_metrics.py`, plan-experimental section 7). When any
run has one, `latent_ap50` (primary), `latent_ap30` and `face_auroc` join the
other metrics in `per_fold`, `summary` and `paired`, including the paired
inference, with their seeds under `v4_seeds`; `v4` lists the splits covered and
the runs left out. A null value (no ground truth or an empty group in the test
lot) never enters a statistic, so that fold has no pair for that metric. Only a
run whose `metrics.json` is readable counts. With no `metrics_v4.json` the output
is exactly what it is without this feature.

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
V4_METRICS = ("latent_ap50", "latent_ap30", "face_auroc")
MASK_METRICS = ("mask_ap50", "mask_ap50_95")  # segment runs only
MASK_METRIC_LABELS = {"mask_ap50": "Mask AP50", "mask_ap50_95": "Mask AP50-95"}
# Mask-IoU twins of the first two, in `metrics_v4.json` of segment runs only (src/evaluation/v4_metrics.py).
V4_MASK_METRICS = ("latent_mask_ap50", "latent_mask_ap30")
V4_METRIC_LABELS = {
    "latent_ap50": "Latent AP50", "latent_ap30": "Latent AP30", "face_auroc": "Face AUROC",
    "latent_mask_ap50": "Latent mask AP50", "latent_mask_ap30": "Latent mask AP30",
}
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
    summary = {
        "ap50": float(_class_value(metrics["per_class_ap_50"])),
        "ap50_95": float(_class_value(metrics["per_class_ap_50_95"])),
        "num_gt": int(counts["tp"]) + int(counts["fn"]),     # tp + fn == GT boxes
        "precision": float(_class_value(metrics["per_class_precision"])),
        "recall": float(_class_value(metrics["per_class_recall"])),
    }
    if "mask_per_class_ap_50" in metrics:  # a segment run: the damage mask AP next to the box AP
        summary["mask_ap50"] = float(_class_value(metrics["mask_per_class_ap_50"]))
        summary["mask_ap50_95"] = float(_class_value(metrics["mask_per_class_ap_50_95"]))
    return summary


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


def v4_summary(metrics: dict) -> dict:
    """The inferential metrics of one `metrics_v4.json`; None for an empty stratum."""
    values = {
        "latent_ap50": metrics["latent_ap50"]["ap"],
        "latent_ap30": metrics["latent_ap30"]["ap"],
        "face_auroc": metrics["face_auroc"]["auroc"],
    }
    for metric in V4_MASK_METRICS:  # a segment run: the mask-IoU twins; absent from detect runs
        if metric in metrics:
            values[metric] = metrics[metric]["ap"]
    return {metric: None if value is None else float(value) for metric, value in values.items()}


def load_v4_run(reports_root: Path, arm: str, fold: int, seed: int, split: str) -> dict:
    """The `metrics_v4.json` of one run: `{"status": "ok", **summary}`, or a missing/invalid record."""
    path = Path(reports_root) / arm / f"fold{fold}_seed{seed}" / split / "metrics_v4.json"
    record = {"arm": arm, "fold": fold, "seed": seed, "split": split}
    if not path.is_file():
        return {**record, "status": "missing", "reason": f"{path} not found"}
    try:
        return {**record, "status": "ok", **v4_summary(json.loads(path.read_text()))}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {**record, "status": "invalid", "reason": f"{path}: {type(error).__name__}: {error}"}


# --------------------------------------------------------------------------
# Fold spec and paired inference
# --------------------------------------------------------------------------


def _keyed_folds_to_rows(folds: Mapping) -> list[dict] | None:
    """The fold rows of a `folds_meta.json` of `scripts/make_v4_folds.py`, None if it is not one.

    That file keys the folds by id and holds the sizes under `splits.<split>.images`.
    """
    rows = []
    for fold_id, entry in folds.items():
        splits = entry.get("splits") if isinstance(entry, dict) else None
        if not str(fold_id).isdigit() or not isinstance(splits, dict):
            return None
        sizes = {split: splits[split].get("images") for split in splits if isinstance(splits[split], dict)}
        if sizes.get("test") is None or sizes.get("train") is None:
            return None
        rows.append({
            "fold": int(fold_id),
            "num_test": sizes["test"],
            "num_train": sizes["train"],
            "num_val": sizes.get("val"),
            **{key: entry[key] for key in ("test", "train", "val") if key in entry},
        })
    return rows


def load_fold_spec(path: Path) -> list[dict]:
    """The fold rows of a fold assignment file.

    Either a `folds` list (`configs/folds/lot_5fold.json`), whose entries need
    `fold`, `num_test` and `num_train`, or the `folds` block of the `folds_meta.json`
    written by `scripts/make_v4_folds.py`, which is converted to those rows. Raises
    ValueError, with the path in the message, for an unreadable or malformed file.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except OSError as error:
        raise ValueError(f"Cannot read the fold spec {path}: {error}") from None
    except ValueError as error:                      # json.JSONDecodeError and UnicodeDecodeError
        raise ValueError(f"The fold spec {path} is not valid JSON: {error}") from None
    folds = data.get("folds") if isinstance(data, dict) else None
    if isinstance(folds, dict):
        folds = _keyed_folds_to_rows(folds)
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

    results = {
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
    _merge_v4_metrics(results, runs, spec_by_id, rope, confidence)
    _merge_mask_metrics(results, runs)
    return results


def _merge_mask_metrics(results: dict, runs: Mapping[tuple, Mapping]) -> None:
    """Add `results["mask"]`, the damage mask AP of segment runs; nothing when no run has any."""
    if not any(MASK_METRICS[0] in run for run in runs.values() if run["status"] == "ok"):
        return
    per_fold: dict = {}
    summary: dict = {}
    for arm in results["arms"]:
        per_fold[arm], summary[arm] = {}, {}
        for split in results["splits"]:
            rows = {}
            for fold in results["folds"]:
                ok = [runs[(arm, fold, seed, split)] for seed in sorted(results["seeds"])
                      if runs[(arm, fold, seed, split)]["status"] == "ok"
                      and MASK_METRICS[0] in runs[(arm, fold, seed, split)]]
                if ok:
                    rows[str(fold)] = {
                        **{m: statistics.fmean(r[m] for r in ok) for m in MASK_METRICS},
                        "seeds": [r["seed"] for r in ok],
                    }
            per_fold[arm][split] = rows
            summary[arm][split] = {m: mean_std([row[m] for row in rows.values()]) for m in MASK_METRICS}
    results["mask"] = {"metrics": list(MASK_METRICS), "per_fold": per_fold, "summary": summary}


def _render_mask(results: dict) -> list[str]:
    block = results["mask"]
    labels = [MASK_METRIC_LABELS[m] for m in MASK_METRICS]
    lines = ["", "## Mask AP (segment runs)", "",
             "Damage-class mask AP from `metrics.json`, measured at prototype resolution; mean ± σ is across folds."]
    for split in results["splits"]:
        lines += ["", f"### {split}", "", "| arm | folds | " + " | ".join(labels) + " |",
                  "|---|---|" + "---|" * len(MASK_METRICS)]
        for arm in results["arms"]:
            stats = block["summary"][arm][split]
            n = stats[MASK_METRICS[0]]["n"]
            if n:
                lines.append(f"| {arm} | {n} of {len(results['folds'])} | "
                             + " | ".join(_fmt_stat(stats[m]) for m in MASK_METRICS) + " |")
    return lines


def _merge_v4_metrics(
    results: dict,
    runs: Mapping[tuple, Mapping],
    spec_by_id: Mapping[int, Mapping] | None,
    rope: float | None,
    confidence: float,
) -> None:
    """Add the `metrics_v4.json` metrics to `results`, in place; nothing when no run has that file."""
    v4_runs = {
        key: load_v4_run(Path(results["reports_root"]), *key) for key, run in runs.items() if run["status"] == "ok"
    }
    splits = [s for s in SPLITS if any(key[3] == s and run["status"] != "missing" for key, run in v4_runs.items())]
    if not splits:
        return
    v4_metrics = list(V4_METRICS) + [
        m for m in V4_MASK_METRICS if any(m in run for run in v4_runs.values() if run["status"] == "ok")
    ]

    def value(arm: str, fold_key: str, seed: int, split: str, metric: str) -> float | None:
        """None for a run with no such value: no file, an unreadable one, or an empty stratum."""
        return v4_runs.get((arm, int(fold_key), seed, split), {}).get(metric)

    paired = results["paired"]
    baseline, treatment = paired["baseline"], paired["treatment"]
    for split in splits:
        for arm in results["arms"]:
            rows = results["per_fold"][arm][split]
            for fold_key, row in rows.items():
                row["v4_seeds"] = {}
                for metric in v4_metrics:
                    present = [s for s in row["seeds"] if value(arm, fold_key, s, split, metric) is not None]
                    if present:
                        row[metric] = statistics.fmean(value(arm, fold_key, s, split, metric) for s in present)
                        row["v4_seeds"][metric] = present
            results["summary"][arm][split].update(
                {m: mean_std([row[m] for row in rows.values() if m in row]) for m in v4_metrics}
            )

        rows = paired["per_fold"][split]
        for metric in v4_metrics:
            paired["per_fold_seed"][split][metric] = {}
        for fold_key, row in rows.items():
            row["v4_seeds"] = {}
            for metric in v4_metrics:
                deltas = {
                    s: value(treatment, fold_key, s, split, metric) - value(baseline, fold_key, s, split, metric)
                    for s in row["seeds"]
                    if value(treatment, fold_key, s, split, metric) is not None
                    and value(baseline, fold_key, s, split, metric) is not None
                }
                if deltas:
                    row[metric] = statistics.fmean(deltas.values())
                    row["v4_seeds"][metric] = list(deltas)
                    paired["per_fold_seed"][split][metric][fold_key] = {str(s): d for s, d in deltas.items()}
        paired["summary"][split].update(
            {m: mean_std([row[m] for row in rows.values() if m in row]) for m in v4_metrics}
        )
        if spec_by_id is not None:
            for metric in v4_metrics:
                present = [fold for fold in results["folds"] if metric in rows.get(str(fold), {})]
                paired["inference"][split][metric] = paired_inference(
                    [rows[str(fold)][metric] for fold in present],
                    [spec_by_id[fold] for fold in present], rope, confidence,
                )

    left_out = []
    for run in v4_runs.values():
        if run["split"] not in splits:
            continue
        record = {k: run[k] for k in ("arm", "fold", "seed", "split")}
        if run["status"] != "ok":
            left_out.append({**record, "status": run["status"], "reason": run["reason"]})
            continue
        left_out += [
            {**record, "status": "null", "reason": f"{metric} is null: no ground truth or an empty group in its stratum"}
            for metric in v4_metrics if metric in run and run[metric] is None
        ]
    results["metrics"] += list(v4_metrics)
    results["v4"] = {"metrics": list(v4_metrics), "missing": left_out, "splits": splits}


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


def _inference_header(confidence: float) -> list[str]:
    return ["n", "mean Δ", f"{confidence * 100:g}% CI", "p (two-sided)", "P(left)", "P(rope)", "P(right)",
            "sign k/n", "sign-flip p", "min p"]


def _inference_cells(entry: dict) -> list[str]:
    """The cells of one `paired_inference` result, aligned with `_inference_header`."""
    if "error" in entry:
        return ["—", f"error: {entry['error']}"] + ["—"] * 8
    ct, bayes, flip = entry["corrected_t"], entry["bayes"], entry["sign_flip"]
    return [
        str(entry["n_folds"]), _fmt(ct["mean"], signed=True),
        f"[{_fmt(ct['ci_low'], signed=True)}, {_fmt(ct['ci_high'], signed=True)}]",
        _fmt_p(ct["p_two_sided"]),
        f"{bayes['p_left']:.3f}", f"{bayes['p_rope']:.3f}", f"{bayes['p_right']:.3f}",
        f"{entry['n_positive']}/{entry['n_folds']}",
        _fmt_p(flip["p_two_sided"]) + ("" if flip["exact"] else " (MC)"),
        _fmt_p(flip["min_attainable_p_two_sided"]),
    ]


def _render_inference(results: dict) -> list[str]:
    """The 'Paired inference' section: one AP50 row per split."""
    paired = results["paired"]
    confidence, rope = paired["inference_settings"]["confidence"], paired["inference_settings"]["rope"]
    entries = {split: paired["inference"][split]["ap50"] for split in results["splits"]}
    ratios = sorted({f"{e['test_train_ratio']:.4f}" for e in entries.values() if "error" not in e})
    rope_text = ("ROPE ±" + f"{rope:g}") if rope else "no ROPE set (P(rope) = 0)"
    pct = f"{confidence * 100:g}%"
    header = ["split", *_inference_header(confidence)]
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
        lines.append("| " + " | ".join([split, *_inference_cells(entries[split])]) + " |")
    return lines


def _v4_seeds_cell(row: dict) -> str:
    seeds = sorted({seed for present in row["v4_seeds"].values() for seed in present})
    return ", ".join(map(str, seeds)) if seeds else "no value"


def _render_v4(results: dict) -> list[str]:
    """The 'Latent-window metrics' section: per-arm tables, paired differences and their inference."""
    v4, paired, folds = results["v4"], results["paired"], results["folds"]
    v4_metrics = v4["metrics"]
    labels = [V4_METRIC_LABELS[m] for m in v4_metrics]
    table_head = ["| fold | seeds | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(v4_metrics)]
    lines = [
        "", "## Latent-window metrics", "",
        "From `metrics_v4.json` (`scripts/v4_metrics.py`, plan-experimental section 7). **Latent AP50**, the "
        "primary metric, is the damage AP50 on the latent window photos plus the `sano` photos of the lot; "
        "**Latent AP30** is the same photos at IoU 0.30; **Face AUROC** ranks the latent photos against the "
        "`sano` photos by their maximum damage confidence. A cell with — has no value: its stratum had no "
        "ground truth or an empty group, or the file is missing, and that fold enters no statistic of the metric.",
    ]
    for split in v4["splits"]:
        lines += ["", f"### {split}"]
        for arm in results["arms"]:
            lines += ["", f"#### {arm}", "", *table_head]
            rows = results["per_fold"][arm][split]
            for fold in folds:
                row = rows.get(str(fold))
                if row is None:
                    lines.append(f"| {fold} | missing | " + " | ".join("—" for _ in v4_metrics) + " |")
                else:
                    lines.append(f"| {fold} | {_v4_seeds_cell(row)} | "
                                 + " | ".join(_fmt(row.get(m)) for m in v4_metrics) + " |")
            stats = results["summary"][arm][split]
            lines.append("| **mean ± σ** | | "
                         + " | ".join(f"{_fmt_stat(stats[m])} (n={stats[m]['n']})" for m in v4_metrics) + " |")
        lines += ["", f"#### Paired difference {paired['treatment']} − {paired['baseline']} ({split})", "", *table_head]
        rows = paired["per_fold"][split]
        for fold in folds:
            row = rows.get(str(fold))
            if row is None:
                lines.append(f"| {fold} | missing | " + " | ".join("—" for _ in v4_metrics) + " |")
            else:
                lines.append(f"| {fold} | {_v4_seeds_cell(row)} | "
                             + " | ".join(_fmt(row.get(m), signed=True) for m in v4_metrics) + " |")
        stats = paired["summary"][split]
        lines.append("| **mean ± σ** | | "
                     + " | ".join(f"{_fmt_stat(stats[m], signed=True)} (n={stats[m]['n']})" for m in v4_metrics) + " |")

    if "inference" in paired:
        header = ["split", "metric", *_inference_header(paired["inference_settings"]["confidence"])]
        lines += [
            "", "### Paired inference of these metrics", "",
            "Same statistics as the AP50 rows of the section above, on the folds with a pair; the ROPE is in "
            "units of each metric. Their limits are those stated there.",
            "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header),
        ]
        for split in v4["splits"]:
            for metric in v4_metrics:
                cells = [split, V4_METRIC_LABELS[metric], *_inference_cells(paired["inference"][split][metric])]
                lines.append("| " + " | ".join(cells) + " |")

    if v4["missing"]:
        lines += ["", f"{len(v4['missing'])} entries are not in these statistics:", ""]
        for m in v4["missing"]:
            lines.append(f"- {m['arm']} fold {m['fold']} seed {m['seed']} {m['split']}: {m['status']}"
                         + (f" ({m['reason']})" if m["status"] != "missing" else ""))
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

    if "v4" in results:
        lines += _render_v4(results)

    if "mask" in results:
        lines += _render_mask(results)

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
                             f"ratio of the paired inference (default: <repo>/{DEFAULT_FOLD_SPEC}); the "
                             "folds_meta.json of make_v4_folds.py is also accepted.")
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
