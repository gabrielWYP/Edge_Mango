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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.smoothing import ema_smooth  # noqa: E402

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
    "master_rgb_det": dict(color="#6A3D9A", linestyle="-", marker="v", label="Maestro RGB, solo cajas", exploratory=False),
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
# Main text: the pre-registered arms (H1: master_rgb_seg vs master_rgbnir_seg; H2: crosskd_rgb_seg vs crosskd_rgbnir_seg)
# and the descriptive NIR reference. The appendix shows every arm found in the data.
MAIN_ARMS = ["master_rgb_seg", "master_nir_seg", "master_rgbnir_seg", "crosskd_rgb_seg", "crosskd_rgbnir_seg"]
SMOOTH_W = 0.8  # same exponential smoothing as scripts/plot_run_curves.py
AP30_COL = "ap30_class_1"  # per-epoch val AP30 of the damage class, absent from the current history
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


def select_arms(arms, scope: str) -> list[str]:
    """`main`: the pre-registered arms plus the NIR reference, in fixed order; `appendix`: every arm."""
    arms = list(arms)
    return [a for a in MAIN_ARMS if a in arms] if scope == "main" else arms


SCOPES = [("main", ""), ("appendix", "_apendice")]  # (scope, file suffix)
SPLIT_TABLES = ("T1", "T2", "T3", "T5", "T6")  # tables written once per scope


# ---------------------------------------------------------------- smoothing and detection loss
def smoothed_history(hist: pd.DataFrame, cols, w: float = SMOOTH_W) -> pd.DataFrame:
    """Adds `<col>_smooth`: the bias-corrected EMA of each (arm, fold) curve, taken in epoch order and per fold."""
    parts = []
    for _, sub in hist.groupby(["arm", "fold"], sort=False):
        sub = sub.sort_values("epoch").copy()
        for c in cols:
            sub[f"{c}_smooth"] = ema_smooth(sub[c].tolist(), w)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else hist.assign(**{f"{c}_smooth": [] for c in cols})


def add_detection_loss(hist: pd.DataFrame) -> tuple[pd.DataFrame, str, bool]:
    """Adds `det_loss` = box + cls loss (comparable across segment and CrossKD arms); falls back to total_loss."""
    if {"box_loss", "cls_loss"} <= set(hist.columns):
        return hist.assign(det_loss=hist["box_loss"] + hist["cls_loss"]), "det_loss", False
    return hist, LOSS_COL, True


def training_panels(hist: pd.DataFrame, loss_col: str = "det_loss") -> list[tuple[str, str, str]]:
    """(title, column, y label) of the panels: train loss, val AP50 and, only if the history has it, val AP30."""
    panels = [("Pérdida de detección (entrenamiento)" if loss_col == "det_loss" else "Pérdida total (entrenamiento)",
               loss_col, "Caja + clase" if loss_col == "det_loss" else "Pérdida total"),
              ("AP50 clase daño (validación)", SELECTION_COL, "AP50")]
    if AP30_COL in hist.columns:
        panels.append(("AP30 clase daño (validación)", AP30_COL, "AP30"))
    return panels


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
    return {"precision": precision, "recall": recall, "f1": f1, "specificity": specificity,
            "discard_rate": 1 - specificity}


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


def pick_threshold_for_recall(scores, is_pos, target: float, grid=THRESHOLD_GRID) -> tuple[float, bool]:
    """Highest threshold whose photo-level recall reaches `target`; (lowest grid threshold, False) if none does."""
    grid = sorted(float(t) for t in grid)
    for t in reversed(grid):
        if photo_metrics(scores, is_pos, t)["recall"] >= target - 1e-12:
            return t, True
    return grid[0], False


def pick_threshold_fbeta(scores, is_pos, beta: float = 2.0, grid=THRESHOLD_GRID) -> float:
    """Threshold maximising F-beta of the photo-level classification (ties to the higher threshold)."""
    vals = []
    for t in grid:
        m = photo_metrics(scores, is_pos, t)
        p, r = m["precision"], m["recall"]
        ok = not (math.isnan(p) or math.isnan(r)) and (beta**2 * p + r) > 0
        vals.append((1 + beta**2) * p * r / (beta**2 * p + r) if ok else 0.0)
    vals = np.array(vals)
    return float(np.array(grid, dtype=float)[vals >= vals.max() - 1e-12].max())


def all_positive_baseline(n_pos: int, n_neg: int) -> dict:
    """Trivial classifier that calls every photo damaged: recall 1, specificity 0, precision = prevalence."""
    precision, recall, f1 = prf(n_pos, n_neg, 0)
    return {"precision": precision, "recall": recall, "f1": f1, "specificity": 0.0}


RULES = {  # prefix -> (label, how the threshold is chosen on val)
    "r90": ("recall >= 0.90 en val", lambda s, p: pick_threshold_for_recall(s, p, 0.90)),
    "r95": ("recall >= 0.95 en val", lambda s, p: pick_threshold_for_recall(s, p, 0.95)),
    "f2": ("F2 máximo en val", lambda s, p: (pick_threshold_fbeta(s, p, 2.0), True)),
}
FIXED_THRESHOLD = 0.25  # the eval confidence threshold


def _is_positive(d: pd.DataFrame) -> np.ndarray:
    return (d["hour"].astype(str) != "sano").to_numpy()


def photo_rules_table(photos: pd.DataFrame, fixed: float = FIXED_THRESHOLD) -> pd.DataFrame:
    """Per (arm, fold): photo-level metrics at a fixed threshold and at val-chosen recall/F2 rules, plus baseline.

    Columns are `<prefix>_<metric>`: `fix` (fixed threshold), `r90`, `r95`, `f2` (threshold chosen on val only and
    applied to test; `_reached` is 1 when the val target was met). A fold lacking val or test is skipped.
    """
    prim = photos[photos["primary_metric_photo"] == 1]
    rows = []
    for (arm, fold), sub in prim.groupby(["arm", "fold"]):
        val, test = sub[sub["split"] == "val"], sub[sub["split"] == "test"]
        if val.empty or test.empty:
            continue
        vs, vp = val["max_damage_score"].to_numpy(float), _is_positive(val)
        ts, tp_ = test["max_damage_score"].to_numpy(float), _is_positive(test)
        row: dict = {"arm": arm, "fold": fold}

        def put(prefix, thr, m):
            row.update({f"{prefix}_threshold": thr, f"{prefix}_recall": m["recall"], f"{prefix}_specificity": m["specificity"],
                        f"{prefix}_discard_rate": 1 - m["specificity"], f"{prefix}_precision": m["precision"],
                        f"{prefix}_f1": m["f1"]})

        put("fix", fixed, photo_metrics(ts, tp_, fixed))
        for prefix, (_, choose) in RULES.items():
            thr, reached = choose(vs, vp)
            put(prefix, thr, photo_metrics(ts, tp_, thr))
            row[f"{prefix}_reached"] = int(reached)
        base = all_positive_baseline(int(tp_.sum()), int((~tp_).sum()))
        row.update({"base_precision": base["precision"], "base_f1": base["f1"]})
        rows.append(row)
    return pd.DataFrame(rows)


def box_fixed_table(pr_grid: pd.DataFrame, threshold: float = FIXED_THRESHOLD) -> pd.DataFrame:
    """Per (arm, fold): test box precision/recall (IoU 0.5) at a fixed threshold."""
    rows = []
    for (arm, fold), sub in pr_grid[pr_grid["split"] == "test"].groupby(["arm", "fold"]):
        p, r, _ = apply_threshold(sub, threshold)
        rows.append({"arm": arm, "fold": fold, "boxfix_precision": p, "boxfix_recall": r})
    return pd.DataFrame(rows, columns=["arm", "fold", "boxfix_precision", "boxfix_recall"])


def operating_curve(photos: pd.DataFrame, arm: str, grid=THRESHOLD_GRID, split: str = "test") -> pd.DataFrame:
    """Photo-level recall against healthy discard rate per threshold, mean and sd across folds."""
    prim = photos[(photos["primary_metric_photo"] == 1) & (photos["arm"] == arm) & (photos["split"] == split)]
    if prim.empty:
        return pd.DataFrame(columns=["threshold", "recall", "recall_sd", "discard", "discard_sd", "n"])
    rows = []
    for t in grid:
        recs, dis = [], []
        for _, sub in prim.groupby("fold"):
            m = photo_metrics(sub["max_damage_score"].to_numpy(float), _is_positive(sub), t)
            recs.append(m["recall"])
            dis.append(1 - m["specificity"])
        rows.append({"threshold": t, "recall": np.nanmean(recs), "recall_sd": np.nanstd(recs, ddof=1) if len(recs) > 1 else 0.0,
                     "discard": np.nanmean(dis), "discard_sd": np.nanstd(dis, ddof=1) if len(dis) > 1 else 0.0, "n": len(recs)})
    return pd.DataFrame(rows)


def hour_recall_table(photos: pd.DataFrame, target: float = 0.90) -> pd.DataFrame:
    """Per (arm, fold): test photo recall at each capture hour, at the threshold that reaches `target` on val.

    Hours absent from the data get no column (never 0). Healthy (`sano`) photos have no hour and are skipped.
    """
    prim = photos[photos["primary_metric_photo"] == 1]
    rows = []
    for (arm, fold), sub in prim.groupby(["arm", "fold"]):
        val = sub[sub["split"] == "val"]
        test_all = photos[(photos["arm"] == arm) & (photos["fold"] == fold) & (photos["split"] == "test")]
        if val.empty or test_all.empty:
            continue
        thr, _ = pick_threshold_for_recall(val["max_damage_score"].to_numpy(float), _is_positive(val), target)
        row: dict = {"arm": arm, "fold": fold, "threshold": thr}
        hours = sorted({int(h) for h in test_all["hour"].astype(str) if h.lstrip("-").isdigit()})
        for h in hours:
            s = test_all.loc[test_all["hour"].astype(str) == str(h), "max_damage_score"].to_numpy(float)
            row[f"hour_{h}"] = float((s >= thr).mean())
        rows.append(row)
    return pd.DataFrame(rows)


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
    test = test.merge(photo_rules_table(data["photos"]), on=["arm", "fold"], how="left")
    test = test.merge(box_fixed_table(data["pr"]), on=["arm", "fold"], how="left")
    return test


def build_hour_frame(data: dict, target: float = 0.90) -> pd.DataFrame:
    return hour_recall_table(data["photos"], target)


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
    ("base_f1", "Línea base «todo positivo»: F1 (foto)", 3),
    ("v4.face_auroc.auroc", "AUROC de cara (sin umbral)", 3),
    ("v4.sano_fp_per_image.fp_per_image", "FP en sano por imagen (umbral fijo 0.25)", 2),
]
T4_METRICS = [
    ("selected_epoch", "Época seleccionada", 1),
    ("val_ap50_class", "AP50 val, clase daño, en esa época", 3),
    ("val_latent_ap50", "AP50 latente en val (checkpoint elegido)", 3),
]
T5_METRICS = [
    ("fix_recall", "Exhaustividad por foto (sensibilidad en fotos dañadas)", 3),
    ("fix_specificity", "Especificidad por foto (fotos sanas conservadas)", 3),
    ("fix_precision", "Precisión por foto", 3),
    ("fix_discard_rate", "Tasa de descarte de sanas (1 - especificidad)", 3),
    ("base_f1", "Línea base «todo positivo»: F1 (foto)", 3),
    ("boxfix_precision", "Precisión de cajas (IoU 0.5)", 3),
    ("boxfix_recall", "Exhaustividad de cajas (IoU 0.5)", 3),
]
T6_METRICS = [
    (f"{pre}_{col}", f"{label}: {name}", d)
    for pre, (label, _) in RULES.items()
    for col, name, d in [
        ("threshold", "umbral elegido en val", 2),
        ("reached", "objetivo alcanzado en val (1 = sí)", 2),
        ("recall", "exhaustividad en prueba", 3),
        ("specificity", "especificidad en prueba", 3),
        ("discard_rate", "tasa de descarte de sanas en prueba", 3),
        ("precision", "precisión en prueba", 3),
    ]
]
TABLES = [
    ("T1_ap50_test", "Tabla T1. Detección y segmentación en prueba, IoU 0.5: AP50 latente de cajas y de máscaras. "
     "Media ± sd entre lotes y valor de cada lote de prueba.", "test", T1_METRICS),
    ("T2_ap30_test", "Tabla T2. Detección y segmentación en prueba, IoU 0.3: AP30 latente de cajas y de máscaras. "
     "Media ± sd entre lotes y valor de cada lote de prueba.", "test", T2_METRICS),
    ("T3_clasificacion_test", "Tabla T3. Clasificación en prueba con el punto de operación elegido solo en validación "
     "(umbral de máximo F1 en val de cada lote, aplicado al test de ese lote), AUROC de cara y falsos positivos en sano. "
     "La línea base «todo positivo» (exhaustividad 1, especificidad 0) alcanza el F1 de foto de la fila indicada: "
     "el F1 de foto de los brazos debe leerse frente a ese valor trivial.",
     "test", T3_METRICS),
    ("T4_seleccion_checkpoint", "Tabla T4. Datos de selección del checkpoint (validación): época elegida y AP50 en "
     "validación. No constituyen evidencia de generalización.", "val", T4_METRICS),
    ("T5_umbral_fijo_025", "Tabla T5. Unidad de decisión = foto (daño si alguna detección alcanza el umbral): umbral fijo 0.25 "
     "(umbral de evaluación) en prueba. Exhaustividad, especificidad, precisión y tasa de descarte de fotos sanas, "
     "más precisión y exhaustividad de cajas. Reglas definidas a posteriori; descriptivas.", "test", T5_METRICS),
    ("T6_exhaustividad_objetivo", "Tabla T6. Puntos de operación orientados a exhaustividad, con umbral elegido solo en validación "
     "de cada lote (el mayor umbral con exhaustividad por foto en val >= 0.90 o 0.95; o F2 máximo) y aplicado al prueba de ese lote. "
     "Si ningún umbral alcanza el objetivo se usa el menor de la rejilla y se marca (fila «objetivo alcanzado»). "
     "Reglas definidas a posteriori; descriptivas, fuera de H1/H2.", "test", T6_METRICS),
]


def write_tables(data: dict, out: Path, arms_test, arms_val, folds, labels) -> list[Path]:
    frames = {"test": build_test_frame(data), "val": build_val_frame(data), "hour": build_hour_frame(data)}
    hour_cols = sorted((c for c in frames["hour"].columns if c.startswith("hour_")), key=lambda c: int(c[5:]))
    written = []
    tables = list(TABLES)
    if hour_cols:
        tables.append(("T7_exhaustividad_por_hora", "Tabla T7. Exhaustividad por foto según la hora de captura (0-12 h) con la regla "
                       "«exhaustividad >= 0.90 en val», solo maestros. Las horas 8-12 quedan fuera de la ventana latente de la métrica primaria. "
                       "Regla definida a posteriori; descriptiva.", "hour",
                       [(c, f"Exhaustividad a las {c[5:]} h", 3) for c in hour_cols]))
    expanded = []
    for name, caption, kind, metrics in tables:
        if name.split("_")[0] in SPLIT_TABLES:
            expanded += [(name + suffix, caption, kind, metrics, scope) for scope, suffix in SCOPES]
        else:
            expanded.append((name, caption, kind, metrics, "appendix"))
    for name, caption, kind, metrics, scope in expanded:
        arms = {"test": arms_test, "val": arms_val, "hour": [a for a in arms_test if a.startswith("master_")]}[kind]
        arms = select_arms(arms, scope)
        if name.split("_")[0] in SPLIT_TABLES:
            caption += (" Brazos principales (pre-registrados y referencia NIR)." if scope == "main"
                        else " Apéndice: todos los brazos, incluidos los exploratorios.")
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


def plot_training_curves(hist, arms, out, stem) -> list[Path]:
    """F1: one row (train detection loss | val AP50 [| val AP30]), one shared legend on top.

    Each fold's curve is smoothed first (EMA, w = 0.8); the band is the sd across the smoothed folds.
    """
    import matplotlib.pyplot as plt

    arms = [a for a in arms if a in set(hist["arm"])]
    if not arms:
        return []
    hist, loss_col, fallback = add_detection_loss(hist)
    panels = training_panels(hist, loss_col)
    hist = smoothed_history(hist, [c for _, c, _ in panels])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 4.0), squeeze=False)
    handles = {}
    for ax, (ptitle, col, ylabel) in zip(axes[0], panels):
        for arm in arms:
            st, c = style_of(arm), curve_mean_sd(hist, arm, f"{col}_smooth")
            if c.empty:
                continue
            sd = c["sd"].fillna(0)
            ax.fill_between(c["epoch"], c["mean"] - sd, c["mean"] + sd, color=st["color"], alpha=0.12, linewidth=0)
            (line,) = ax.plot(c["epoch"], c["mean"], color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                              markevery=10, markersize=6, linewidth=2)
            handles[arm] = line
        ax.set_title(ptitle, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("Época", color=INK)
        ax.set_ylabel(ylabel, color=INK)
        _style_axes(ax)
        if col != loss_col:
            ax.set_ylim(bottom=0)
    ncol = min(len(handles), 3)
    fig.legend([handles[a] for a in arms if a in handles], [arm_label(a) for a in arms if a in handles], loc="lower center",
               ncol=ncol, fontsize=8.5, frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    note = "Curvas suavizadas por lote con media móvil exponencial (w = 0.8); media ± 1 sd entre lotes."
    if fallback:
        note += " Pérdida total: faltan los componentes caja y clase."
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, color=GRAY)
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


def plot_operating_curves(photos, arms_present, out, stem) -> list[Path]:
    """F4: test photo-level recall against healthy discard rate, mean +- sd across folds, with the val-chosen points."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = [p for p in _panels(arms_present) + _extra_panel(arms_present) if p[1]]
    if not panels:
        return []
    rules = photo_rules_table(photos)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.8), sharey=True, sharex=True, squeeze=False)
    for ax, (ptitle, arms) in zip(axes[0], panels):
        for arm in arms:
            s, c = style_of(arm), operating_curve(photos, arm).sort_values("discard")
            if c.empty:
                continue
            ax.fill_between(c["discard"], (c["recall"] - c["recall_sd"]).clip(0, 1), (c["recall"] + c["recall_sd"]).clip(0, 1),
                            color=s["color"], alpha=0.10, linewidth=0)
            ax.plot(c["discard"], c["recall"], color=s["color"], linestyle=s["linestyle"], linewidth=2, label=s["label"])
            r = rules[rules["arm"] == arm]
            for pre, mk in (("r90", "o"), ("r95", "D")):
                if len(r):
                    ax.plot(r[f"{pre}_discard_rate"].mean(), r[f"{pre}_recall"].mean(), marker=mk, markersize=9,
                            markerfacecolor=s["color"], markeredgecolor=INK, markeredgewidth=1.3, linestyle="none")
        ax.set_title(ptitle, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("Tasa de descarte de fotos sanas (1 - especificidad)", color=INK)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        _style_axes(ax)
        handles, labels = ax.get_legend_handles_labels()
        handles += [Line2D([], [], marker="o", color="w", markerfacecolor="#cccccc", markeredgecolor=INK, markersize=9),
                    Line2D([], [], marker="D", color="w", markerfacecolor="#cccccc", markeredgecolor=INK, markersize=9)]
        labels += ["Regla val: exhaustividad >= 0.90", "Regla val: exhaustividad >= 0.95"]
        ax.legend(handles, labels, fontsize=7.5, frameon=False, labelcolor=INK, loc="lower right")
    axes[0][0].set_ylabel("Exhaustividad por foto (fotos dañadas detectadas)", color=INK)
    fig.suptitle("Curva de operación por foto en prueba (media ± 1 sd entre lotes); valores de los puntos en la Tabla T6",
                 fontsize=11.5, color=INK, y=1.02)
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
        if s["exploratory"]:
            ax.axhspan(y - 0.5, y + 0.5, color="#efefef", zorder=0, linewidth=0)
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
    from matplotlib.patches import Patch

    ax.legend(handles=[Line2D([], [], color=INK, linewidth=2, label="Media de los lotes"),
                       Patch(facecolor="#efefef", label="Brazo exploratorio (descriptivo)"),
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
    pr_arms = discover_arms(data["pr"]["arm"].unique())
    photo_arms = discover_arms(data["photos"]["arm"].unique())
    for scope, suffix in SCOPES:
        written += plot_training_curves(data["hist"], select_arms(hist_arms, scope), args.out, f"F1_curvas_entrenamiento{suffix}")
        written += plot_pr(data["pr"], select_arms(pr_arms, scope), args.out, f"F2_curvas_pr_test{suffix}")
        written += plot_operating_curves(data["photos"], select_arms(photo_arms, scope), args.out,
                                         f"F4_curva_operacion_foto{suffix}")
    written += plot_strip(build_test_frame(data), test_arms, folds, labels, args.out, "F3_ap50_por_lote")
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
