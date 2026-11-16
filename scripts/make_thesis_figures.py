#!/usr/bin/env python3
"""Thesis figures and tables from the test-analysis export (CPU only: pandas + matplotlib).

Reads `reports/thesis-test-analysis/export/` (`runs.csv`, `training_history.csv`, `pr_grid.csv`, `photos.csv`,
`folds.csv`) and writes, under `--out`, the figures (PNG at 300 dpi and SVG) and the tables (CSV, Markdown and a
docx-friendly Markdown) cited by `reports/thesis-figures/README.md`. Labels and captions are in Spanish; code and
identifiers are in English.

Arms are discovered from the data: a new arm (for example `master_twostem_rgbnir_seg`) is picked up without edits,
with the colour reserved for it in `ARM_STYLE`. A fold that is absent for an arm is reported as absent (`n/d`),
never as 0, and statistics use only the folds present (the `n folds` column says how many).

Operating point (table T3): the threshold is chosen on each fold's VAL split only (maximum F1, ties broken towards
the higher threshold) and applied to that fold's TEST split. Nothing is tuned on test. `scripts/val_operating_points.py`
uses fixed thresholds (0.25 and 0.5) instead; here the threshold is selected on val, never on test.

    python scripts/make_thesis_figures.py                       # real export, writes reports/thesis-figures/
    python scripts/make_thesis_figures.py --export-dir X --out Y
"""

from __future__ import annotations

import argparse
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = REPO_ROOT / "reports" / "thesis-test-analysis" / "export"
DEFAULT_OUT = REPO_ROOT / "reports" / "thesis-figures"

THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.01, 1.0, 0.01)]  # same grid as pr_grid.csv (minus 0.001)
INK = "#222222"
GRAY = "#6b6b6b"
GRID = "#dddddd"

# One entry per arm, in a fixed categorical order: a colour is never reassigned when the number of series changes.
# Okabe-Ito based (colourblind safe); the neutral black/gray slots of the original set are replaced by purple/brown
# because the palette validator requires chroma and lightness inside a band. Masters are solid, students dashed
# (secondary encoding), each arm has its own marker.
ARM_STYLE: dict[str, dict] = {
    "master_rgb_seg": dict(color="#0072B2", linestyle="-", marker="o", label="Maestro RGB", exploratory=False),
    "master_nir_seg": dict(color="#D55E00", linestyle="-", marker="s", label="Maestro NIR", exploratory=False),
    "master_rgbnir_seg": dict(color="#009E73", linestyle="-", marker="^", label="Maestro RGB+NIR", exploratory=False),
    # Reserved slot: trained after the thesis set, appears when its folds are exported.
    "master_twostem_rgbnir_seg": dict(
        color="#E69F00", linestyle="-", marker="D", label="Maestro RGB+NIR, dos troncos (exploratorio)", exploratory=True
    ),
    "master_rgb_det": dict(color="#6A3D9A", linestyle="--", marker="v", label="Alumno RGB, detección", exploratory=False),
    "crosskd_rgb_seg": dict(
        color="#56B4E9", linestyle="--", marker="P", label="Alumno RGB, CrossKD desde RGB", exploratory=False
    ),
    "crosskd_rgbnir_seg": dict(
        color="#CC79A7", linestyle="--", marker="X", label="Alumno RGB, CrossKD desde RGB+NIR", exploratory=False
    ),
    "crosskd_nir_seg": dict(
        color="#8C6D1F", linestyle="--", marker="*", label="Alumno RGB, CrossKD desde NIR (exploratorio)", exploratory=True
    ),
}
EXTRA_COLORS = ["#117733", "#882255", "#332288", "#AA4499", "#44AA99", "#999933"]  # arms unknown to ARM_STYLE

# Panels shared by F1 (curves), the loss figure and F2: at most 5 series each.
PANELS = [
    ("a) Maestros: RGB, NIR y RGB+NIR", ["master_rgb_seg", "master_nir_seg", "master_rgbnir_seg", "master_twostem_rgbnir_seg"]),
    (
        "b) Maestro RGB frente a alumnos RGB",
        ["master_rgb_seg", "master_rgb_det", "crosskd_rgb_seg", "crosskd_rgbnir_seg", "crosskd_nir_seg"],
    ),
]
SELECTION_COL = "ap50_class_1"  # damage-class AP50 on val, the `selection_metric=damage_ap50` of the trainer
LOSS_COL = "total_loss"
LATENT_AP50, LATENT_MASK_AP50 = "v4.latent_ap50.ap", "v4.latent_mask_ap50.ap"
LATENT_AP30, LATENT_MASK_AP30 = "v4.latent_ap30.ap", "v4.latent_mask_ap30.ap"


# ---------------------------------------------------------------- arms and styles
def discover_arms(arms) -> list[str]:
    """Arms present in the data: known ones in `ARM_STYLE` order, then unknown ones alphabetically."""
    present = set(arms)
    known = [a for a in ARM_STYLE if a in present]
    return known + sorted(present - set(ARM_STYLE))


def style_of(arm: str) -> dict:
    """Style of an arm; an arm unknown to `ARM_STYLE` gets a stable extra colour that no known arm uses."""
    if arm in ARM_STYLE:
        return ARM_STYLE[arm]
    color = EXTRA_COLORS[zlib.crc32(arm.encode()) % len(EXTRA_COLORS)]
    return dict(color=color, linestyle=":", marker="h", label=arm, exploratory=True)


def arm_label(arm: str) -> str:
    return style_of(arm)["label"]


# ---------------------------------------------------------------- fold aggregation
def fold_table(df: pd.DataFrame, value_col: str, arms, folds, arm_col: str = "arm", fold_col: str = "fold") -> pd.DataFrame:
    """One row per arm: n folds present, mean, sd (ddof=1) and the value of every fold (NaN when absent)."""
    rows = []
    for arm in arms:
        sub = df[df[arm_col] == arm]
        per_fold = sub.groupby(fold_col)[value_col].mean() if len(sub) else pd.Series(dtype=float)
        row: dict = {"arm": arm}
        values = np.array([per_fold.get(f, np.nan) for f in folds], dtype=float)
        present = values[~np.isnan(values)]
        row["n"] = int(len(present))
        row["mean"] = float(present.mean()) if len(present) else math.nan
        row["sd"] = float(present.std(ddof=1)) if len(present) > 1 else math.nan
        for f, v in zip(folds, values):
            row[f"fold{f}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def curve_mean_sd(hist: pd.DataFrame, arm: str, col: str) -> pd.DataFrame:
    """Per-epoch mean and sd (ddof=1) of `col` across the folds of one arm."""
    sub = hist[hist["arm"] == arm]
    if sub.empty:
        return pd.DataFrame(columns=["epoch", "mean", "sd", "n"])
    g = sub.groupby("epoch")[col]
    out = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.count()}).reset_index()
    return out


def selected_epochs(hist: pd.DataFrame, col: str) -> pd.DataFrame:
    """Epoch of the best `col` per (arm, fold): the first maximum, as the trainer keeps a checkpoint only on `>`."""
    rows = []
    for (arm, fold), sub in hist.groupby(["arm", "fold"]):
        sub = sub.sort_values("epoch").reset_index(drop=True)
        i = int(sub[col].idxmax())
        rows.append({"arm": arm, "fold": fold, "epoch": int(sub.loc[i, "epoch"]), "value": float(sub.loc[i, col])})
    return pd.DataFrame(rows, columns=["arm", "fold", "epoch", "value"])


# ---------------------------------------------------------------- operating points
def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Precision, recall and F1 from counts; precision is NaN with no prediction, recall NaN with no positive."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else math.nan
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _f1_by_row(grid: pd.DataFrame) -> list[float]:
    return [prf(r.tp, r.fp, r.fn)[2] for r in grid.itertuples()]


def pick_threshold(val_grid: pd.DataFrame) -> float:
    """Threshold with the highest F1 on a val grid (columns threshold, tp, fp, fn); ties go to the higher threshold."""
    f1 = np.array(_f1_by_row(val_grid))
    thr = val_grid["threshold"].to_numpy(dtype=float)
    best = f1.max()
    return float(thr[f1 >= best - 1e-12].max())


def apply_threshold(grid: pd.DataFrame, threshold: float) -> tuple[float, float, float]:
    row = grid[np.isclose(grid["threshold"].to_numpy(dtype=float), threshold)]
    if row.empty:
        raise KeyError(f"threshold {threshold} is not in the grid")
    r = row.iloc[0]
    return prf(r["tp"], r["fp"], r["fn"])


def operating_point(val_grid: pd.DataFrame, test_grid: pd.DataFrame) -> tuple[float, tuple[float, float, float]]:
    """Threshold chosen on val only, and the precision/recall/F1 it gives on test."""
    thr = pick_threshold(val_grid)
    return thr, apply_threshold(test_grid, thr)


def photo_metrics(scores, is_pos, threshold: float) -> dict:
    """Photo-level classification: a photo is called damaged when its highest damage score reaches `threshold`."""
    scores = np.asarray(scores, dtype=float)
    is_pos = np.asarray(is_pos, dtype=bool)
    called = scores >= threshold
    tp = int((called & is_pos).sum())
    fp = int((called & ~is_pos).sum())
    fn = int((~called & is_pos).sum())
    tn = int((~called & ~is_pos).sum())
    precision, recall, f1 = prf(tp, fp, fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else math.nan
    return {"precision": precision, "recall": recall, "f1": f1, "specificity": specificity}


def photo_operating_point(val_scores, val_pos, test_scores, test_pos, grid=THRESHOLD_GRID) -> tuple[float, dict]:
    """Photo-level threshold chosen on val only (max F1, ties to the higher threshold), applied to test."""
    f1 = np.array([photo_metrics(val_scores, val_pos, t)["f1"] for t in grid])
    thr = float(np.array(grid, dtype=float)[f1 >= f1.max() - 1e-12].max())
    return thr, photo_metrics(test_scores, test_pos, thr)


def box_operating_table(pr_grid: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, fold): val-chosen threshold and test precision/recall/F1 of latent boxes (IoU 0.5).

    A (arm, fold) without both splits is skipped: it is absent from the table, not zero.
    """
    rows = []
    for (arm, fold), sub in pr_grid.groupby(["arm", "fold"]):
        val, test = sub[sub["split"] == "val"], sub[sub["split"] == "test"]
        if val.empty or test.empty:
            continue
        thr, (p, r, f) = operating_point(val, test)
        rows.append({"arm": arm, "fold": fold, "box_threshold": thr, "box_precision": p, "box_recall": r, "box_f1": f})
    return pd.DataFrame(rows, columns=["arm", "fold", "box_threshold", "box_precision", "box_recall", "box_f1"])


def photo_operating_table(photos: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, fold): photo-level classification on the primary-metric photos (latent window vs `sano`)."""
    prim = photos[photos["primary_metric_photo"] == 1]
    rows = []
    for (arm, fold), sub in prim.groupby(["arm", "fold"]):
        val, test = sub[sub["split"] == "val"], sub[sub["split"] == "test"]
        if val.empty or test.empty:
            continue
        pos = lambda d: (d["hour"].astype(str) != "sano").to_numpy()  # noqa: E731
        thr, m = photo_operating_point(val["max_damage_score"], pos(val), test["max_damage_score"], pos(test))
        rows.append(
            {
                "arm": arm, "fold": fold, "photo_threshold": thr, "photo_precision": m["precision"],
                "photo_recall": m["recall"], "photo_f1": m["f1"], "photo_specificity": m["specificity"],
            }
        )
    return pd.DataFrame(
        rows, columns=["arm", "fold", "photo_threshold", "photo_precision", "photo_recall", "photo_f1", "photo_specificity"]
    )


def mean_pr_curve(pr_grid: pd.DataFrame, arm: str, split: str = "test") -> pd.DataFrame:
    """Threshold-averaged PR curve: at each threshold, the mean precision and recall over the folds that have it."""
    sub = pr_grid[(pr_grid["arm"] == arm) & (pr_grid["split"] == split)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "n"])
    prs = [prf(r.tp, r.fp, r.fn) for r in sub.itertuples()]
    sub["precision"] = [p for p, _, _ in prs]
    sub["recall"] = [r for _, r, _ in prs]
    g = sub.groupby("threshold")
    out = pd.DataFrame({"precision": g["precision"].mean(), "recall": g["recall"].mean(), "n": g["fold"].nunique()})
    return out.reset_index().dropna(subset=["precision", "recall"])


# ---------------------------------------------------------------- formatting
def format_value(x: float, digits: int = 3) -> str:
    return "n/d" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def format_mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    if mean is None or math.isnan(mean):
        return "n/d"
    return f"{mean:.{digits}f} ± {format_value(sd, digits)}"


def to_markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def summary_numeric(df, metrics, arms, folds, fold_labels) -> pd.DataFrame:
    """Numeric long table: one row per (arm, metric) with n, mean, sd and the per-fold values."""
    rows = []
    for col, label, digits in metrics:
        if col not in df.columns:
            continue
        t = fold_table(df, col, arms, folds)
        for r in t.itertuples(index=False):
            d = r._asdict()
            row = {"arm": d["arm"], "metric": label, "n_folds": d["n"], "mean": d["mean"], "sd": d["sd"], "_digits": digits}
            for f in folds:
                row[fold_labels.get(f, f"Fold {f}")] = d[f"fold{f}"]
            rows.append(row)
    return pd.DataFrame(rows)


def build_summary_table(df, metrics, arms, folds, fold_labels) -> pd.DataFrame:
    """Display table with Spanish headers: mean ± sd plus the per-fold values (`n/d` where a fold is absent)."""
    num = summary_numeric(df, metrics, arms, folds, fold_labels)
    labels = [fold_labels.get(f, f"Fold {f}") for f in folds]
    rows = []
    for r in num.to_dict("records"):
        d = r["_digits"]
        row = {
            "Brazo": arm_label(r["arm"]), "Métrica": r["metric"], "n folds": str(r["n_folds"]),
            "Media ± sd": format_mean_sd(r["mean"], r["sd"], d),
        }
        for lab in labels:
            row[lab] = format_value(r[lab], d)
        rows.append(row)
    return pd.DataFrame(rows, columns=["Brazo", "Métrica", "n folds", "Media ± sd", *labels])


# ---------------------------------------------------------------- loading
def load_export(export_dir: Path) -> dict:
    runs = pd.read_csv(export_dir / "runs.csv")
    out = {
        "runs": runs,
        "hist": pd.read_csv(export_dir / "training_history.csv"),
        "pr": pd.read_csv(export_dir / "pr_grid.csv"),
        "photos": pd.read_csv(export_dir / "photos.csv", low_memory=False),
    }
    folds_path = export_dir / "folds.csv"
    out["folds_meta"] = pd.read_csv(folds_path) if folds_path.exists() else None
    return out


def folds_and_labels(data: dict) -> tuple[list[int], dict[int, str]]:
    meta = data["folds_meta"]
    present = sorted(set(data["runs"]["fold"].unique()))
    if meta is not None and len(meta):
        folds = sorted(set(meta["fold"].astype(int)) | set(present))
        labels = {int(r.fold): f"Lote {int(r.test_lot)}" for r in meta.itertuples()}
    else:
        folds, labels = present, {}
    return [int(f) for f in folds], {f: labels.get(f, f"Fold {f}") for f in folds}


def build_test_frame(data: dict) -> pd.DataFrame:
    """Test-split metrics per (arm, fold) joined with the val-selected operating points."""
    runs = data["runs"]
    test = runs[runs["split"] == "test"].copy()
    test = test.merge(box_operating_table(data["pr"]), on=["arm", "fold"], how="left")
    test = test.merge(photo_operating_table(data["photos"]), on=["arm", "fold"], how="left")
    return test


def build_val_frame(data: dict) -> pd.DataFrame:
    """Per (arm, fold): checkpoint-selection epoch and val metrics at that epoch."""
    sel = selected_epochs(data["hist"], SELECTION_COL).rename(columns={"epoch": "selected_epoch", "value": "val_ap50_class"})
    val = data["runs"][data["runs"]["split"] == "val"][["arm", "fold", LATENT_AP50]].rename(columns={LATENT_AP50: "val_latent_ap50"})
    return sel.merge(val, on=["arm", "fold"], how="outer")


# ---------------------------------------------------------------- tables
T1_METRICS = [(LATENT_AP50, "AP50 latente (cajas)", 3), (LATENT_MASK_AP50, "AP50 latente (máscaras)", 3)]
T2_METRICS = [(LATENT_AP30, "AP30 latente (cajas)", 3), (LATENT_MASK_AP30, "AP30 latente (máscaras)", 3)]
T3_METRICS = [
    ("box_threshold", "Umbral elegido en val (cajas)", 2),
    ("box_precision", "Precisión (cajas latentes, IoU 0.5)", 3),
    ("box_recall", "Exhaustividad (cajas latentes, IoU 0.5)", 3),
    ("box_f1", "F1 (cajas latentes, IoU 0.5)", 3),
    ("photo_threshold", "Umbral elegido en val (foto)", 2),
    ("photo_precision", "Precisión (foto)", 3),
    ("photo_recall", "Exhaustividad (foto)", 3),
    ("photo_specificity", "Especificidad (foto)", 3),
    ("photo_f1", "F1 (foto)", 3),
    ("v4.face_auroc.auroc", "AUROC de cara (sin umbral)", 3),
    ("v4.sano_fp_per_image.fp_per_image", "FP en sano por imagen (umbral fijo 0.25)", 2),
]
T4_METRICS = [
    ("selected_epoch", "Época seleccionada", 1),
    ("val_ap50_class", "AP50 val, clase daño, en esa época", 3),
    ("val_latent_ap50", "AP50 latente en val (checkpoint elegido)", 3),
]
TABLES = [
    ("T1_ap50_test", "Tabla T1. Detección y segmentación en prueba, IoU 0.5: AP50 latente de cajas y de máscaras. "
     "Media ± sd entre lotes y valor de cada lote de prueba.", "test", T1_METRICS),
    ("T2_ap30_test", "Tabla T2. Detección y segmentación en prueba, IoU 0.3: AP30 latente de cajas y de máscaras. "
     "Media ± sd entre lotes y valor de cada lote de prueba.", "test", T2_METRICS),
    ("T3_clasificacion_test", "Tabla T3. Clasificación en prueba con el punto de operación elegido solo en validación "
     "(umbral de máximo F1 en val de cada lote, aplicado al test de ese lote), AUROC de cara y falsos positivos en sano.",
     "test", T3_METRICS),
    ("T4_seleccion_checkpoint", "Tabla T4. Datos de selección del checkpoint (validación): época elegida y AP50 en "
     "validación. No constituyen evidencia de generalización.", "val", T4_METRICS),
]


def write_tables(data: dict, out: Path, arms_test, arms_val, folds, labels) -> list[Path]:
    frames = {"test": build_test_frame(data), "val": build_val_frame(data)}
    written = []
    for name, caption, kind, metrics in TABLES:
        arms = arms_test if kind == "test" else arms_val
        df = frames[kind]
        num = summary_numeric(df, metrics, arms, folds, labels).drop(columns="_digits")
        num.insert(1, "arm_label", num["arm"].map(arm_label))
        shown = build_summary_table(df, metrics, arms, folds, labels)
        absent = [(a, labels[f]) for a in arms for f in folds if df[(df["arm"] == a) & (df["fold"] == f)].empty]
        note = ""
        if absent:
            note = "\nFolds ausentes (no se cuentan como 0): " + ", ".join(f"{arm_label(a)} / {l}" for a, l in absent) + ".\n"
        num.to_csv(out / f"{name}.csv", index=False, float_format="%.6f")
        (out / f"{name}.md").write_text(f"{caption}\n\n{to_markdown_table(shown)}{note}", encoding="utf-8")
        (out / f"{name}_docx.md").write_text(to_markdown_table(shown), encoding="utf-8")
        written += [out / f"{name}.csv", out / f"{name}.md", out / f"{name}_docx.md"]
    return written


# ---------------------------------------------------------------- figures
def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)


def _save(fig, out: Path, stem: str) -> list[Path]:
    paths = []
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, facecolor="white", bbox_inches="tight", **kw)
        paths.append(p)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def _panels(arms_present):
    return [(title, [a for a in group if a in arms_present]) for title, group in PANELS]


def _extra_panel(arms_present):
    """Arms the fixed panels do not cover (new arms) go to a third panel so none is silently dropped."""
    covered = {a for _, g in PANELS for a in g}
    extra = [a for a in arms_present if a not in covered]
    return [("c) Brazos adicionales", extra)] if extra else []


def plot_curves(hist, arms_present, col, ylabel, title, out, stem, ylim=None) -> list[Path]:
    import matplotlib.pyplot as plt

    panels = [p for p in _panels(arms_present) + _extra_panel(arms_present) if p[1]]
    if not panels:
        return []
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.4), sharey=True, squeeze=False)
    for ax, (ptitle, arms) in zip(axes[0], panels):
        for arm in arms:
            s, c = style_of(arm), curve_mean_sd(hist, arm, col)
            if c.empty:
                continue
            ax.fill_between(c["epoch"], c["mean"] - c["sd"].fillna(0), c["mean"] + c["sd"].fillna(0),
                            color=s["color"], alpha=0.10, linewidth=0)
            ax.plot(c["epoch"], c["mean"], color=s["color"], linestyle=s["linestyle"], marker=s["marker"],
                    markevery=10, markersize=6, linewidth=2, label=s["label"])
        ax.set_title(ptitle, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("Época", color=INK)
        _style_axes(ax)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK, loc="best")
    axes[0][0].set_ylabel(ylabel, color=INK)
    fig.suptitle(title, fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()
    return _save(fig, out, stem)


def plot_pr(pr, arms_present, out, stem) -> list[Path]:
    import matplotlib.pyplot as plt

    panels = [p for p in _panels(arms_present) + _extra_panel(arms_present) if p[1]]
    if not panels:
        return []
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6), sharey=True, squeeze=False)
    for ax, (ptitle, arms) in zip(axes[0], panels):
        for arm in arms:
            s, c = style_of(arm), mean_pr_curve(pr, arm, "test").query("recall > 0").sort_values("recall")
            if c.empty:
                continue
            ax.plot(c["recall"], c["precision"], color=s["color"], linestyle=s["linestyle"], marker=s["marker"],
                    markevery=12, markersize=6, linewidth=2, label=s["label"])
        ax.set_title(ptitle, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("Exhaustividad (cajas latentes, IoU 0.5)", color=INK)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        _style_axes(ax)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK, loc="upper right")
    axes[0][0].set_ylabel("Precisión", color=INK)
    fig.suptitle("Curvas precisión-exhaustividad en prueba, promedio de los lotes por umbral", fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()
    return _save(fig, out, stem)


def plot_strip(test, arms, folds, labels, out, stem) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t = fold_table(test, LATENT_AP50, arms, folds)
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(arms) + 1.8))
    for i, row in enumerate(t.itertuples(index=False)):
        d = row._asdict()
        s = style_of(d["arm"])
        y = len(arms) - 1 - i
        for k, f in enumerate(folds):
            v = d[f"fold{f}"]
            if math.isnan(v):
                continue
            ax.plot(v, y + (k - (len(folds) - 1) / 2) * 0.07, marker=s["marker"], markersize=8, color=s["color"],
                    markeredgecolor="white", markeredgewidth=0.6, linestyle="none", alpha=0.95)
        if not math.isnan(d["mean"]):
            ax.plot([d["mean"], d["mean"]], [y - 0.3, y + 0.3], color=INK, linewidth=2)
            ax.annotate(format_mean_sd(d["mean"], d["sd"]), (1.0, y), xycoords=("axes fraction", "data"),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=GRAY)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([arm_label(a) for a in reversed(arms)], fontsize=9, color=INK)
    ax.set_ylim(-0.6, len(arms) - 0.4)
    ax.set_xlim(0, max(0.6, float(np.nanmax(t[[f"fold{f}" for f in folds]].to_numpy())) + 0.05))
    ax.set_xlabel("AP50 latente de cajas en prueba", color=INK)
    ax.set_title("AP50 latente en prueba por lote y brazo", fontsize=11.5, color=INK, loc="left")
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], color=INK, linewidth=2, label="Media de los lotes"),
                       Line2D([], [], color=GRAY, marker="o", linestyle="none", markersize=8,
                              label="Un punto por lote de prueba")],
              fontsize=8.5, frameon=False, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    return _save(fig, out, stem)


# ---------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_export(args.export_dir)
    folds, labels = folds_and_labels(data)
    test_arms = discover_arms(data["runs"].loc[data["runs"]["split"] == "test", "arm"].unique())
    all_arms = discover_arms(data["runs"]["arm"].unique())
    print(f"arms: {all_arms}\nfolds: {folds}")
    for arm in all_arms:
        missing = [f for f in folds if data["runs"][(data["runs"]["arm"] == arm) & (data["runs"]["fold"] == f)
                   & (data["runs"]["split"] == "test")].empty]
        if missing:
            print(f"WARNING {arm}: no test run for folds {missing} (reported as absent)", file=sys.stderr)

    written = write_tables(data, args.out, test_arms, all_arms, folds, labels)
    hist_arms = discover_arms(data["hist"]["arm"].unique())
    written += plot_curves(data["hist"], hist_arms, SELECTION_COL, "AP50 en validación, clase daño",
                           "Curvas de validación: métrica de selección del checkpoint (media ± 1 sd entre lotes)",
                           args.out, "F1a_curvas_validacion", ylim=(0, None))
    written += plot_curves(data["hist"], hist_arms, LOSS_COL, "Pérdida total de entrenamiento",
                           "Pérdida de entrenamiento (media ± 1 sd entre lotes)", args.out, "F1b_curvas_perdida")
    written += plot_pr(data["pr"], discover_arms(data["pr"]["arm"].unique()), args.out, "F2_curvas_pr_test")
    written += plot_strip(build_test_frame(data), test_arms, folds, labels, args.out, "F3_ap50_por_lote")
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
