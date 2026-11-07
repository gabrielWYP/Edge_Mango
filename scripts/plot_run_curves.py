#!/usr/bin/env python3
"""Validation damage AP50 and AP30 and training detection loss per epoch of the definitive runs, one column per fold.

Validation only, never test. AP50 and the loss come from the TensorBoard logs (every epoch). AP30 is not logged: it
comes from the CSV of `scripts/eval_epoch_curve.py` (the checkpoints kept every 5 epochs, on all validation photos,
like the logged AP50); without that CSV its row is left out. One solid colour per arm. The detection loss is box +
cls, the same in every arm: the total loss of a KD run also holds its distillation term.

The AP curves are drawn with an exponential smoothing with bias correction (the TensorBoard one),
smoothed_t = (1 - w) * sum_i w^(t-i) * ap_i / (1 - w^t). The weight w is the optional third argument (default 0.8 per
epoch, about 5 epochs; 0 draws the raw values); the AP30 points, 5 epochs apart, use w^5 so both rows keep the same
time constant. Smoothing changes only the drawing: the dot is the raw AP50 of the best epoch, the checkpoint the run
selected, so it sits above the smoothed line.

    python scripts/plot_run_curves.py runs/final reports/final-runs-progress/curves_seed42.png [0.8] [epoch_curve_val.csv]
"""
import csv, glob, os, sys, datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT, OUT = sys.argv[1], sys.argv[2]  # runs root with <arm>/fold<k>/seed42, output PNG
SMOOTH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
CURVE_CSV = sys.argv[4] if len(sys.argv) > 4 else None
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# One hue per arm. With est_nir in dark grey, the five clear a CVD Delta E of 9.2 and a normal-vision Delta E of 22
# over all pairs (OKLab x100, Machado 2009), so every pair stays apart in every panel.
ARMS = [  # (arm, colour, label)
    ("est_rgb", "#2a78d6", "est_rgb (RGB)"),
    ("est_rgbnir", "#eb6834", "est_rgbnir (RGB+NIR)"),
    ("kd_rgb", "#1baf7a", "kd_rgb (RGB, maestro RGB)"),
    ("kd_rgbnir", "#e1c300", "kd_rgbnir (RGB, maestro RGB+NIR)"),
    ("est_nir", "#3d3c39", "est_nir (NIR, referencia)"),
]

def read_curve(path):
    """{(arm, fold): {epoch: ap30}} from the CSV of eval_epoch_curve.py, seed 42."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    for row in csv.DictReader(open(path)):
        if row["seed"] == "42" and row["ap30"]:
            out.setdefault((row["arm"], int(row["fold"])), {})[int(row["epoch"])] = float(row["ap30"])
    return out

def smooth(values, w):
    """Exponential moving average with bias correction, so the first epochs are not pulled towards 0."""
    out, acc = [], 0.0
    for t, v in enumerate(values, 1):
        acc = w * acc + (1 - w) * v
        out.append(acc / (1 - w ** t) if w else v)
    return out

def scalars(run):
    out = {}
    for f in sorted(glob.glob(run + "/logs/events*")):
        ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
        for tag in ea.Tags()["scalars"]:
            for s in ea.Scalars(tag):
                out.setdefault(tag, {})[s.step] = s.value
    return out

folds = sorted({int(p.split("fold")[-1]) for p in glob.glob(f"{ROOT}/*/fold*")})
curve = read_curve(CURVE_CSV)
rows = ["ap50", "ap30", "loss"] if curve else ["ap50", "loss"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(len(rows), len(folds), figsize=(4.6 * len(folds), 3.6 * len(rows)), sharex=True,
                         squeeze=False, facecolor=SURFACE)
ax_of = {name: axes[i] for i, name in enumerate(rows)}
for col, fold in enumerate(folds):
    for row in range(len(rows)):
        ax = axes[row][col]; ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"): ax.spines[side].set_visible(False)
    loss_top = 0.0
    for arm, color, label in ARMS:
        runs = glob.glob(f"{ROOT}/{arm}/fold{fold}/seed42")
        if not runs: continue
        s = scalars(runs[0])
        ap = s.get("Phase1/val/ap50_class1", {})
        box, cls = s.get("Phase1/train/box_loss", {}), s.get("Phase1/train/cls_loss", {})
        if not ap: continue
        ep = sorted(ap)
        ax_of["ap50"][col].plot(ep, smooth([ap[e] for e in ep], SMOOTH), color=color, lw=2, label=label)
        best = max(ep, key=ap.get)
        ax_of["ap50"][col].plot([best], [ap[best]], "o", ms=8, color=color, mec=SURFACE, mew=2)
        ap30 = curve.get((arm, fold), {})
        if "ap30" in ax_of and ap30:
            e30 = sorted(ap30)
            ax_of["ap30"][col].plot(e30, smooth([ap30[e] for e in e30], SMOOTH ** 5), color=color, lw=2, label=label)
        le = sorted(set(box) & set(cls))
        ax_of["loss"][col].plot(le, [box[e] + cls[e] for e in le], color=color, lw=2, label=label)
        loss_top = max([loss_top] + [box[e] + cls[e] for e in le if e >= 3])
    if loss_top:  # the first two epochs (loss near 2.8) would flatten the rest; they run off the top
        ax_of["loss"][col].set_ylim(top=loss_top * 1.08)
    axes[0][col].set_title(f"Fold {fold}", color=INK, fontsize=12, loc="left")
    ax_of["ap50"][col].set_ylim(bottom=0)
    if "ap30" in ax_of:
        ax_of["ap30"][col].set_ylim(bottom=0)
    axes[-1][col].set_xlabel("Época")
    axes[-1][col].set_xlim(0, 81)
suffix = f", suavizado w={SMOOTH:g}" if SMOOTH else ""
ax_of["ap50"][0].set_ylabel(f"AP50 de daño (val){suffix}\n● = mejor época, sin suavizar")
if "ap30" in ax_of:
    ax_of["ap30"][0].set_ylabel("AP30 de daño (val), mismo suavizado\ncheckpoints cada 5 épocas")
ax_of["loss"][0].set_ylabel("Pérdida de detección (train)\ncaja + clase; épocas 1-2 fuera")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.995), labelcolor=INK
)
stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
note = (
    f"Corridas definitivas, semilla 42, solo validación (elige el checkpoint; no es evidencia). Corte {stamp}. "
    "Todas las horas de validación. Las corridas de est_nir sin terminar se dibujan hasta su última época."
)
fig.text(0.01, 0.005, note, color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print(OUT)
