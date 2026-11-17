#!/usr/bin/env python3
"""Add exploratory arms to `reports/thesis-test-analysis/analysis.json` and write their TEST_ANALYSIS.md sections.

The registered analysis (`analysis.json`) was produced with `scripts/aggregate_kfold.paired_inference` over the fold-level
values of `metrics_v4.json`. This script repeats that computation for the exploratory comparisons of the plan (section 11,
2026-11-16) and MERGES them: entries already in `analysis.json` are recomputed only to be checked (any difference is an
error) and never overwritten, so the registered H1/H2 numbers cannot move.

    python scripts/add_exploratory_analysis.py --runs-root <runs>/reports --folds-meta <folds_meta.json> \\
        --analysis reports/thesis-test-analysis/analysis.json --markdown-out <scratch>/new_sections.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.aggregate_kfold import load_fold_spec, paired_inference  # noqa: E402

ROPE = 0.02
SEED = 42
METRICS = {  # analysis.json key: (v4 json path, label in the markdown)
    "latent_ap50": (("latent_ap50", "ap"), "latent AP50 (box) -- PRIMARY"),
    "latent_ap30": (("latent_ap30", "ap"), "latent AP30 (box)"),
    "latent_mask_ap50": (("latent_mask_ap50", "ap"), "latent mask AP50"),
    "latent_mask_ap30": (("latent_mask_ap30", "ap"), "latent mask AP30"),
    "face_auroc": (("face_auroc", "auroc"), "face AUROC"),
    "sano_fp_per_image": (("sano_fp_per_image", "fp_per_image"), "sano FP per image (lower is better; ROPE 0.02 NOT meaningful, descriptive only)"),
}
# New comparisons: key -> (treatment, baseline, role). Registered in plan-experimental.md, 2026-11-16 row 2.
NEW_COMPARISONS = {
    "twostem_vs_rgbnir": ("master_twostem_rgbnir_seg", "master_rgbnir_seg", "exploratory (2026-11-16)"),
    "twostem_vs_nir": ("master_twostem_rgbnir_seg", "master_nir_seg", "exploratory (2026-11-16)"),
}


def fold_value(runs_root: Path, arm: str, fold: int, split: str, path: tuple[str, str]):
    f = runs_root / arm / f"fold{fold}_seed{SEED}" / split / "metrics_v4.json"
    value = json.loads(f.read_text(encoding="utf-8")).get(path[0], {}).get(path[1])
    return None if value is None else float(value)


def per_arm_entry(runs_root: Path, arm: str, folds: list[int]) -> dict:
    out = {}
    for key, (path, _) in METRICS.items():
        vals = [fold_value(runs_root, arm, k, "test", path) for k in folds]
        val = [fold_value(runs_root, arm, k, "val", path) for k in folds]
        if any(v is None for v in vals):
            continue
        out[key] = {"folds": vals, "mean": statistics.mean(vals), "sd": statistics.stdev(vals),
                    "val_folds": val, "val_mean": statistics.mean(val)}
    return out


def comparison_entry(per_arm: dict, spec_rows, treatment: str, baseline: str, role: str) -> dict:
    metrics = {}
    for key in METRICS:
        t, b = per_arm[treatment].get(key), per_arm[baseline].get(key)
        if t is None or b is None:
            continue
        deltas = [x - y for x, y in zip(t["folds"], b["folds"])]
        metrics[key] = {"deltas": deltas, "mean": statistics.mean(deltas),
                        "inference": paired_inference(deltas, spec_rows, rope=ROPE)}
    return {"treatment": treatment, "baseline": baseline, "role": role, "metrics": metrics}


def sentence(inf: dict, mean: float, threshold: float = 0.95) -> str:
    b, c = inf["bayes"], inf["corrected_t"]
    p = f"P(left)={b['p_left']:.3f}, P(rope)={b['p_rope']:.3f}, P(right)={b['p_right']:.3f}"
    ci = f"mean delta {mean:+.3f}, corrected 95% CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"
    if b["p_right"] >= threshold:
        return f"Practically better: raised test latent AP50 by more than the ROPE ({p}; {ci})."
    if b["p_rope"] >= threshold:
        return f"Practically equivalent: the difference is within +/-ROPE ({p}; {ci})."
    if b["p_left"] >= threshold:
        return f"Practically worse than the baseline ({p}; {ci})."
    return f"Inconclusive on five lots ({p}; {ci}); more lots would be needed to change the verdict."


def render_comparison(key: str, comp: dict) -> list[str]:
    lines = [f"### {key}: {comp['treatment']} - {comp['baseline']}  [{comp['role']}]", ""]
    for mkey, (_, label) in METRICS.items():
        m = comp["metrics"].get(mkey)
        if m is None:
            continue
        inf = m["inference"]
        c, b, s = inf["corrected_t"], inf["bayes"], inf["sign_flip"]
        lines += [f"**{label}**", "",
                  "- per-fold deltas (seed 42): " + " | ".join(f"{d:+.3f}" for d in m["deltas"]),
                  f"- mean delta {m['mean']:+.4f}; corrected 95% CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]; "
                  f"corrected p (two-sided) {c['p_two_sided']:.4f} (t={c['t']:.3f}, df={c['df']})",
                  f"- Bayes (ROPE {ROPE}): P(left)={b['p_left']:.3f}, P(rope)={b['p_rope']:.3f}, P(right)={b['p_right']:.3f}",
                  f"- k/5 folds with delta>0: {inf['n_positive']}/{inf['n_folds']}; sign-flip exact p "
                  f"{s['p_two_sided']:.4f} (floor {s['min_attainable_p_two_sided']:.4f})"]
        if mkey == "latent_ap50":
            lines.append("- Licensed sentence (section 4): " + sentence(inf, m["mean"]))
        elif mkey == "sano_fp_per_image":
            lines.append("- Descriptive only: lower is better; the Bayes probabilities refer to a +/-0.02 FP/image ROPE that "
                         "has no meaning at this scale. No decision sentence licensed.")
        else:
            lines.append("- Secondary metric: section-4 wording would give: " + sentence(inf, m["mean"]))
        lines.append("")
    return lines


def render_arm_rows(per_arm: dict, arms: list[str]) -> list[str]:
    rows = []
    for arm in arms:
        a50, a30 = per_arm[arm]["latent_ap50"], per_arm[arm]["latent_ap30"]
        f50 = " | ".join(f"{v:.3f}" for v in a50["folds"])
        f30 = " | ".join(f"{v:.3f}" for v in a30["folds"])
        rows.append(f"| {arm} | {f50} | {a50['mean']:.3f} +/- {a50['sd']:.3f} | {f30} | {a30['mean']:.3f} +/- {a30['sd']:.3f} | "
                    f"{a50['val_mean']:.3f} | {a50['mean'] - a50['val_mean']:+.3f} |")
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs-root", type=Path, required=True, help="the `reports` directory of the runs root")
    p.add_argument("--folds-meta", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--markdown-out", type=Path, required=True)
    args = p.parse_args(argv)

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    spec = load_fold_spec(args.folds_meta)
    folds = [int(r["fold"]) for r in spec]
    assert folds == [0, 1, 2, 3, 4], folds
    per_arm = analysis["per_arm"]

    # Check: recompute every registered arm and comparison; they must equal what is stored.
    changed = []
    for arm, stored in per_arm.items():
        if per_arm_entry(args.runs_root, arm, folds) != stored:
            changed.append(f"per_arm.{arm}")
    for key, stored in analysis["comps"].items():
        recomputed = comparison_entry(per_arm, spec, stored["treatment"], stored["baseline"], stored["role"])
        if recomputed != stored:
            changed.append(f"comps.{key}")
    print("existing entries recomputed; differing from stored:", changed or "none")
    if changed:
        return 1

    new_arms = sorted({a for t, b, _ in NEW_COMPARISONS.values() for a in (t, b)} - set(per_arm))
    for arm in new_arms:
        per_arm[arm] = per_arm_entry(args.runs_root, arm, folds)
    sections = []
    for key, (t, b, role) in NEW_COMPARISONS.items():
        analysis["comps"][key] = comparison_entry(per_arm, spec, t, b, role)
        sections += render_comparison(key, analysis["comps"][key])
    holm_rows = sorted(((analysis["comps"][k]["metrics"]["latent_ap50"]["inference"]["corrected_t"]["p_two_sided"], k)
                        for k in NEW_COMPARISONS))
    analysis["exploratory_2026_11_16_outside_holm"] = {k: {"p": pv, "note": "outside H1, H2 and the Holm families"}
                                                       for pv, k in holm_rows}
    args.analysis.write_text(json.dumps(analysis, indent=1, ensure_ascii=False), encoding="utf-8")
    args.markdown_out.write_text("\n".join(
        ["<!-- per-arm rows -->", *render_arm_rows(per_arm, sorted({t for t, _, _ in NEW_COMPARISONS.values()})), "", "<!-- comparisons -->", *sections]), encoding="utf-8")
    print("added arms:", new_arms, "comparisons:", list(NEW_COMPARISONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
