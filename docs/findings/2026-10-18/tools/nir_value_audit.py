"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

Does registered NIR add information over RGB? Alignment, quality, damage separability, redundancy.

Writes nir_audit.json to --out-dir (per-image rows + pooled leave-one-lot-out pixel-classifier AUCs).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from _paths import resolve_roots  # noqa: E402

WT, M, OUT, _ = resolve_roots(__doc__)  # WT = code root (src/, notebooks/, configs/); M = data root (data/)
sys.path.insert(0, str(WT))
from src.training.nir_registration import load_homography, register_nir_to_rgb  # noqa: E402

H = load_homography(WT / "notebooks/matriz_homografia_aruco.npy")
LABELS = {p.stem: p for p in (M / "data/annotations/yolo/labels").glob("*/*.txt")}
lots = json.loads((WT / "configs/folds/lot_5fold.json").read_text())["lots"]
lot_of = {s: l for l, v in lots.items() for s in v["stems"]}
rng = np.random.default_rng(0)


def grad(img):
    g = img.astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


rows, pix = [], {"X": [], "y": [], "lot": []}
for s, lot in sorted(lot_of.items()):
    rgb = cv2.imread(str(M / "data/cache/mango/rgb" / f"{s}.jpg"))
    nir = cv2.imread(str(M / "data/cache/mango/nir" / f"{s.replace('mango_rgb_', 'mango_nir_')}.jpg"), cv2.IMREAD_GRAYSCALE)
    Hh, Ww = rgb.shape[:2]
    reg = register_nir_to_rgb(nir, H, (Ww, Hh), 0)
    raw = cv2.resize(nir, (Ww, Hh)) if nir.shape[:2] != (Hh, Ww) else nir
    boxes = [list(map(float, l.split())) for l in LABELS[s].read_text().splitlines() if l.strip()]
    _, cx, cy, w, h = [b for b in boxes if int(b[0]) == 0][0]
    dmg = [b for b in boxes if int(b[0]) == 1]
    # Region for alignment: mango box + 30% so its silhouette edge is included.
    x0, x1 = int(max((cx - w * 0.65) * Ww, 0)), int(min((cx + w * 0.65) * Ww, Ww))
    y0, y1 = int(max((cy - h * 0.65) * Hh, 0)), int(min((cy + h * 0.65) * Hh, Hh))
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gr, gn_reg, gn_raw = grad(gray[y0:y1, x0:x1]), grad(reg[y0:y1, x0:x1]), grad(raw[y0:y1, x0:x1])
    (dx, dy), _ = cv2.phaseCorrelate(gr, gn_reg)
    # Fruit mask (ellipse) and damage/healthy pixel masks inside it.
    fruit = np.zeros((Hh, Ww), np.uint8)
    cv2.ellipse(fruit, (int(cx * Ww), int(cy * Hh)), (int(w * Ww * 0.45), int(h * Hh * 0.45)), 0, 0, 360, 1, -1)
    dmg_core, dmg_any = np.zeros_like(fruit), np.zeros_like(fruit)
    for _, dx_, dy_, dw, dh in dmg:
        cv2.rectangle(dmg_any, (int((dx_ - dw * 0.8) * Ww), int((dy_ - dh * 0.8) * Hh)), (int((dx_ + dw * 0.8) * Ww), int((dy_ + dh * 0.8) * Hh)), 1, -1)
        cv2.rectangle(dmg_core, (int((dx_ - dw * 0.3) * Ww), int((dy_ - dh * 0.3) * Hh)), (int((dx_ + dw * 0.3) * Ww), int((dy_ + dh * 0.3) * Hh)), 1, -1)
    healthy = (fruit == 1) & (dmg_any == 0)
    damage = (fruit == 1) & (dmg_core == 1)
    fr = fruit == 1
    b, g, r = (rgb[..., i].astype(np.float32) for i in range(3))
    n = reg.astype(np.float32)
    # Redundancy: how much of NIR is a linear function of RGB inside the fruit.
    A = np.stack([r[fr], g[fr], b[fr], np.ones(fr.sum())], 1)
    coef, *_ = np.linalg.lstsq(A, n[fr], rcond=None)
    r2 = 1 - ((n[fr] - A @ coef) ** 2).sum() / (((n[fr] - n[fr].mean()) ** 2).sum() + 1e-9)
    row = {"stem": s, "lot": lot, "ncc_raw": ncc(gr, gn_raw), "ncc_reg": ncc(gr, gn_reg), "shift": float(np.hypot(dx, dy)),
           "nir_p50": float(np.percentile(n[fr], 50)), "nir_p99": float(np.percentile(n[fr], 99)),
           "nir_clip_hi": float((n[fr] >= 250).mean()), "nir_clip_lo": float((n[fr] <= 2).mean()),
           "r2_nir_from_rgb": float(r2), "corr_nir_r": float(np.corrcoef(n[fr], r[fr])[0, 1]), "n_dmg": len(dmg)}
    if damage.sum() > 30 and healthy.sum() > 30:
        y = np.r_[np.ones(damage.sum()), np.zeros(healthy.sum())]
        for name, ch in (("R", r), ("G", g), ("B", b), ("NIR", n)):
            sc = np.r_[ch[damage], ch[healthy]]
            a = roc_auc_score(y, sc)
            row[f"auc_{name}"] = float(max(a, 1 - a))
        # Pixel samples for the pooled RGB vs RGB+NIR classifier; per-image median normalization.
        k = min(400, int(damage.sum())), min(1200, int(healthy.sum()))
        for mask, lab, kk in ((damage, 1, k[0]), (healthy, 0, k[1])):
            idx = rng.choice(int(mask.sum()), kk, replace=False)
            feats = np.stack([c[mask][idx] / (np.median(c[fr]) + 1) for c in (r, g, b, n)], 1)
            pix["X"].append(feats); pix["y"].append(np.full(kk, lab)); pix["lot"].append(np.full(kk, lot))
    rows.append(row)

X, y, L = np.concatenate(pix["X"]), np.concatenate(pix["y"]), np.concatenate(pix["lot"])
pooled = {}
for cols, name in (([0, 1, 2], "RGB"), ([0, 1, 2, 3], "RGB+NIR"), ([3], "NIR")):
    aucs = []
    for test_lot in sorted(set(L)):  # leave-one-lot-out, same grouping as the folds
        tr, te = L != test_lot, L == test_lot
        clf = LogisticRegression(max_iter=1000).fit(X[tr][:, cols], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te][:, cols])[:, 1]))
    pooled[name] = [round(a, 3) for a in aucs]
(OUT / "nir_audit.json").write_text(json.dumps({"rows": rows, "pooled_lolo_auc": pooled}, indent=1))

print("per lot medians: ncc_raw ncc_reg shift_px nir_p50 nir_p99 clip_hi% R2(NIR|RGB) corr(NIR,R) | AUC R G B NIR")
for lot in sorted(lots):
    rr = [x for x in rows if x["lot"] == lot]
    med = lambda k: np.median([x[k] for x in rr if k in x])
    print(f"{lot}: {med('ncc_raw'):.3f} {med('ncc_reg'):.3f} {med('shift'):5.1f} {med('nir_p50'):5.1f} {med('nir_p99'):5.1f} "
          f"{100*med('nir_clip_hi'):5.2f} {med('r2_nir_from_rgb'):.3f} {med('corr_nir_r'):.3f} | "
          f"{med('auc_R'):.3f} {med('auc_G'):.3f} {med('auc_B'):.3f} {med('auc_NIR'):.3f}")
print("pooled pixel classifier, leave-one-lot-out AUC per test lot (lot1..lot5):", pooled)
print("mean AUC:", {k: round(float(np.mean(v)), 3) for k, v in pooled.items()})
