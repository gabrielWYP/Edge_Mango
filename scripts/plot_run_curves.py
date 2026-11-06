#!/usr/bin/env python3
"""Validation damage AP50 and training detection loss per epoch of the definitive runs, one column per fold.

Reads the TensorBoard logs only (validation, never test). Colour is what the model or its teacher reads (blue RGB,
orange RGB+NIR); a dashed line is a distilled student. The detection loss is box + cls, the same in every arm: the
total loss of a KD run also holds its distillation term.

The AP curve is drawn with an exponential smoothing with bias correction (the TensorBoard one),
smoothed_t = (1 - w) * sum_i w^(t-i) * ap_i / (1 - w^t). The weight w is the optional third argument (default 0.8,
about 5 epochs; 0 draws the raw values). Smoothing changes only the drawing: the dot is the raw AP of the best epoch,
the checkpoint the run selected, so it sits above the smoothed line.

    python scripts/plot_run_curves.py runs/final reports/final-runs-progress/curves_seed42.png [0.8]
"""
import glob, sys, datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT, OUT = sys.argv[1], sys.argv[2]  # runs root with <arm>/fold<k>/seed42, output PNG
SMOOTH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
ARMS = [  # (arm, color by input of the model or its teacher, line by training, label)
    ("est_rgb", BLUE, "-", "est_rgb (RGB)"),
    ("est_rgbnir", ORANGE, "-", "est_rgbnir (RGB+NIR)"),
    ("kd_rgb", BLUE, (0, (4, 2)), "kd_rgb (RGB, maestro RGB)"),
    ("kd_rgbnir", ORANGE, (0, (4, 2)), "kd_rgbnir (RGB, maestro RGB+NIR)"),
]

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
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(2, len(folds), figsize=(4.6 * len(folds), 7.2), sharex=True, squeeze=False,
                         facecolor=SURFACE)
for col, fold in enumerate(folds):
    for row in range(2):
        ax = axes[row][col]; ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"): ax.spines[side].set_visible(False)
    for arm, color, ls, label in ARMS:
        runs = glob.glob(f"{ROOT}/{arm}/fold{fold}/seed42")
        if not runs: continue
        s = scalars(runs[0])
        ap = s.get("Phase1/val/ap50_class1", {})
        box, cls = s.get("Phase1/train/box_loss", {}), s.get("Phase1/train/cls_loss", {})
        if not ap: continue
        ep = sorted(ap)
        axes[0][col].plot(ep, smooth([ap[e] for e in ep], SMOOTH), color=color, ls=ls, lw=2, label=label)
        best = max(ep, key=ap.get)
        axes[0][col].plot([best], [ap[best]], "o", ms=8, color=color, mec=SURFACE, mew=2)
        le = sorted(set(box) & set(cls))
        axes[1][col].plot(le, [box[e] + cls[e] for e in le], color=color, ls=ls, lw=2, label=label)
    axes[0][col].set_title(f"Fold {fold}", color=INK, fontsize=12, loc="left")
    axes[0][col].set_ylim(bottom=0)
    axes[1][col].set_xlabel("Época")
    axes[1][col].set_xlim(0, 81)
axes[0][0].set_ylabel("AP50 de daño en validación\n(todas las horas; ● = mejor época)")
if SMOOTH:
    axes[0][0].set_ylabel(
        f"AP50 de daño en validación, suavizado w={SMOOTH:g}\n(● = AP sin suavizar de la mejor época)"
    )
axes[1][0].set_ylabel("Pérdida de detección en train\n(caja + clase)")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995), labelcolor=INK
)
stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
note = (
    f"Corridas definitivas, semilla 42, solo validación (elige el checkpoint; no es evidencia). Corte {stamp}. "
    "Color = entrada del modelo o de su maestro; línea punteada = destilado."
)
fig.text(0.01, 0.005, note, color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print(OUT)
