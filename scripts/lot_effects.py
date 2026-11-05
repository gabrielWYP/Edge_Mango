#!/usr/bin/env python3
"""Exploratory per-lot analysis (plan, section 8.5): each fold's arm differences beside the capture of its lot.

For every fold it takes the latent AP50 of each arm (mean over the seeds present in every arm of that fold), the
paired differences of H1 (est_rgbnir - est_rgb) and H2 (kd_rgbnir - kd_rgb), and the descriptors of the lot that
fold evaluates on, from `scripts/lot_descriptors.py`, as the value of that lot and as its distance to the mean of
the three training lots of the fold.

With five lots this is descriptive: it writes a table and one plot per pre-registered descriptor, and computes no
correlation, interval or p-value. The plan fixes `--split test`; any other split is for checking the script.

    python scripts/lot_effects.py --reports-root runs/final/reports \\
        --fold-spec data/cache/v4_captura_v3_final/folds_lot/folds_meta.json \\
        --descriptors reports/lot-descriptors/lot_descriptors.csv --split test --output reports/lot-effects
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ARMS = ("est_rgb", "est_rgbnir", "kd_rgb", "kd_rgbnir", "est_nir")
DIFFERENCES = {"H1": ("est_rgbnir", "est_rgb"), "H2": ("kd_rgbnir", "kd_rgb"), "secondary": ("kd_rgbnir", "est_rgb")}
# Pre-registered on 2026-11-05 (plan, section 8.5), before any test result of the current arms was read.
PRIMARY = (
    "rgb_fruit_r_over_b",
    "rgb_fruit_luma",
    "rgb_border_spread",
    "latent_contrast_ratio",
    "residual_shift_px",
    "latent_visible_rgb_yes_share",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--fold-spec", type=Path, required=True, help="folds_meta.json of make_v4_folds.py")
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _lot_number(name: str) -> str:
    return name.removeprefix("lot")


def load_descriptors(path: Path) -> dict[str, dict[str, float]]:
    lots = {}
    for row in csv.DictReader(path.open()):
        values = {k: float(v) for k, v in row.items() if k != "lot" and v not in ("", None)}
        if values.get("latent_contrast_rgb"):
            values["latent_contrast_ratio"] = values["latent_contrast_nir"] / values["latent_contrast_rgb"]
        lots[row["lot"]] = values
    return lots


def latent_ap(reports_root: Path, arm: str, fold: int, seed: int, split: str) -> float | None:
    path = reports_root / arm / f"fold{fold}_seed{seed}" / split / "metrics_v4.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["latent_ap50"]["ap"]


def fold_rows(args, folds: dict, lots: dict) -> list[dict]:
    rows = []
    for fold_key, spec in sorted(folds.items(), key=lambda kv: int(kv[0])):
        fold = int(fold_key)
        test_lot = _lot_number(spec[args.split])
        train_lots = [_lot_number(x) for x in spec["train"]]
        per_seed = {
            arm: {s: latent_ap(args.reports_root, arm, fold, s, args.split) for s in args.seeds} for arm in ARMS
        }
        row = {"fold": fold, "lot": test_lot}
        for name, (treatment, baseline) in DIFFERENCES.items():
            # A seed counts only when both arms of the difference have it: the difference stays paired.
            seeds = [s for s in args.seeds if per_seed[treatment][s] is not None and per_seed[baseline][s] is not None]
            row[f"seeds_{name}"] = len(seeds)
            row[f"delta_{name}"] = (
                statistics.mean(per_seed[treatment][s] - per_seed[baseline][s] for s in seeds) if seeds else None
            )
        for arm in ARMS:
            present = [v for v in per_seed[arm].values() if v is not None]
            row[f"ap_{arm}"] = statistics.mean(present) if present else None
            row[f"seeds_{arm}"] = len(present)
        for name in PRIMARY:
            value = lots.get(test_lot, {}).get(name)
            train = [lots[t][name] for t in train_lots if name in lots.get(t, {})]
            row[name] = value
            row[f"{name}_vs_train"] = value - statistics.mean(train) if value is not None and train else None
        rows.append(row)
    return rows


def write_outputs(rows: list[dict], output: Path, split: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"lot_effects_{split}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fmt = lambda v: "—" if v is None else f"{v:+.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = [
        f"# Per-lot effects ({split})",
        "",
        "Descriptive only: five lots, no correlation, interval or p-value (plan, section 8.5).",
        "",
        "| Fold | Lot | Δ H1 | Δ H2 | Δ secondary | " + " | ".join(f"{n} (vs train)" for n in PRIMARY) + " |",
        "| :-- | :-- | :-- | :-- | :-- | " + " | ".join(":--" for _ in PRIMARY) + " |",
    ]
    for r in rows:
        cells = [str(r["fold"]), r["lot"], fmt(r["delta_H1"]), fmt(r["delta_H2"]), fmt(r["delta_secondary"])]
        plain = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
        cells += [f"{plain(r[n])} ({fmt(r[f'{n}_vs_train'])})" for n in PRIMARY]
        lines.append("| " + " | ".join(cells) + " |")
    (output / f"lot_effects_{split}.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, name in zip(axes.flat, PRIMARY):
        for delta, marker, label in (("delta_H1", "o", "H1"), ("delta_H2", "s", "H2")):
            pts = [
                (r[f"{name}_vs_train"], r[delta], r["lot"])
                for r in rows
                if r[delta] is not None and r[f"{name}_vs_train"] is not None
            ]
            if not pts:
                continue
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=marker, label=label)
            for x, y, lot in pts:
                ax.annotate(f"L{lot}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(f"{name}, test lot - mean of train lots")
        ax.set_ylabel(f"Δ latent AP50 ({split})")
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(output / f"lot_effects_{split}.png", dpi=150)


def main(argv=None) -> int:
    args = parse_args(argv)
    folds = json.loads(args.fold_spec.read_text())["folds"]
    rows = fold_rows(args, folds, load_descriptors(args.descriptors))
    write_outputs(rows, args.output, args.split)
    print((args.output / f"lot_effects_{args.split}.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
