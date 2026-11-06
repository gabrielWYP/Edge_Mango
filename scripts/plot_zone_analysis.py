#!/usr/bin/env python3
"""Figure of reports/zone-analysis: damage AP per zone against the position and the light of the damage.

    python scripts/plot_zone_analysis.py reports/zone-analysis
"""
import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(sys.argv[1])
ZONES = ("centro", "izquierda", "derecha", "superior", "inferior")
SURFACE, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
ap = list(csv.DictReader((ROOT / "zone_ap.csv").open()))
cond = {r["zone"]: r for r in csv.DictReader((ROOT / "zone_conditions.csv").open()) if r["subset"] == "latent"}
within = {r["zone"]: r for r in csv.DictReader((ROOT / "zone_within.csv").open())}
mean_ap = {
    (z, s): st.mean(float(r["ap50_latent"]) for r in ap if r["zone"] == z and r["split"] == s and r["ap50_latent"])
    for z in ZONES for s in ("val", "test")
}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
PANELS = (
    (axes[0], "dist_center", "Distancia del daño al centro de la fruta en la imagen\n(0 = centro, 1 = borde)"),
    (axes[1], "luma_ratio", "Brillo RGB en la caja de daño / brillo medio de la fruta\n(1 = igual que la fruta)"),
)
for ax, key, label in PANELS:
    for split, color, marker in (("val", BLUE, "o"), ("test", ORANGE, "s")):
        xs = [float(cond[z][key]) for z in ZONES]
        ys = [mean_ap[(z, split)] for z in ZONES]
        ax.plot(xs, ys, marker, ms=9, color=color, mec=SURFACE, mew=2, ls="", label=f"{split}")
    for z in ZONES:
        ax.annotate(z, (float(cond[z][key]), mean_ap[(z, "val")]), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color=INK)
    ax.set_xlabel(label)
    ax.set_ylim(0, 0.8)
axes[0].set_ylabel("AP50 de daño, ventana latente\n(media de las evaluaciones)")
axes[0].legend(frameon=False, title="split", labelcolor=INK)
x = range(len(ZONES))
w = 0.38
axes[2].bar([i - w / 2 for i in x], [float(within[z]["hit_near_center"]) for z in ZONES], w, color=BLUE,
            label="mitad más cerca del centro")
axes[2].bar([i + w / 2 for i in x], [float(within[z]["hit_far_center"]) for z in ZONES], w, color="#9ec5ef",
            label="mitad más lejos del centro")
axes[2].set_xticks(list(x), ZONES)
axes[2].set_ylim(0, 1)
axes[2].set_ylabel("Fotos latentes con el daño encontrado\n(al menos un acierto a conf 0.25)")
axes[2].legend(frameon=False, labelcolor=INK, fontsize=9)
for ax, title in zip(axes, ("AP por zona vs posición", "AP por zona vs luz", "Dentro de cada zona: posición")):
    ax.set_title(title, loc="left", color=INK, fontsize=12)
fig.text(0.01, 0.01, "AP: validación de los folds 0-3 (brazos terminados) y test de los folds 0 y 1, semilla 42 (plan 8.6). "
         "Panel 3: validación de los folds 0-2 y test de los folds 0 y 1, todos los brazos. Descriptivo, sin inferencia.",
         color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(ROOT / "zone_analysis.png", dpi=200, facecolor=SURFACE)
print(ROOT / "zone_analysis.png")
