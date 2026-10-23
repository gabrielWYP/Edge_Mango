"""Quantify two augmentation mechanisms on the real labeled pairs (CPU only, no training).

1. Box inflation: area of the augmented axis-aligned damage box over the area of the
   rotated object, for A.Rotate(limit=20) with the default 'largest_box' and with 'ellipse'.
2. Bruise contrast: per image, mean healthy-fruit minus mean damage gray level per RGB channel
   (damage = central 60% of each damage box; healthy = mango box minus damage boxes enlarged
   to 160%), to compare against the GaussNoise std range (0.02-0.08 of 255).

Usage: python aug_audit.py <rgb_dir> <labels_root>
"""

import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

rgb_dir, labels_root = Path(sys.argv[1]), Path(sys.argv[2])
labels = {p.stem: p for p in labels_root.rglob("*.txt")}
images = {p.stem: p for p in rgb_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")}
stems = sorted(set(labels) & set(images))
print(f"pairs with labels: {len(stems)}")

side_px, ratios = [], {"largest_box": [], "ellipse": []}
contrast = []
for stem in stems:
    rows = [list(map(float, l.split())) for l in labels[stem].read_text().splitlines() if l.strip()]
    dmg = [r[1:] for r in rows if int(r[0]) == 1]
    mango = [r[1:] for r in rows if int(r[0]) == 0]
    if not dmg:
        continue
    img = cv2.imread(str(images[stem]))
    h, w = img.shape[:2]
    scale = 640 / max(h, w)                          # letterbox scale used in training
    for b in dmg:
        side_px.append(np.sqrt(b[2] * w * b[3] * h) * scale)

    # 1) inflation: rotate a blank canvas with the real boxes by a fixed angle
    for method in ratios:
        t = A.Compose([A.Rotate(limit=(20, 20), p=1.0, border_mode=0, rotate_method=method)],
                      bbox_params=A.BboxParams(format="yolo", label_fields=["c"], min_visibility=0.0))
        out = t(image=np.zeros((h, w, 3), np.uint8), bboxes=dmg, c=[1] * len(dmg))
        for b0, b1 in zip(dmg, out["bboxes"]):
            ratios[method].append((b1[2] * b1[3]) / (b0[2] * b0[3]))

    # 2) contrast in gray levels (BGR order from OpenCV)
    if not mango:
        continue
    mx, my, mw, mh = mango[0]
    yy, xx = np.mgrid[0:h, 0:w]                      # ellipse inscribed in the mango box, at 90%
    fruit = ((xx / w - mx) / (0.45 * mw)) ** 2 + ((yy / h - my) / (0.45 * mh)) ** 2 <= 1.0
    dmg_px = np.zeros((h, w), bool)
    excl = np.zeros((h, w), bool)
    ring = np.zeros((h, w), bool)                    # local ring (2.2x box) cancels the illumination gradient
    for x, y, bw, bh in dmg:
        for mask, f in ((dmg_px, 0.6), (excl, 1.4), (ring, 2.2)):
            x0, x1 = int((x - bw * f / 2) * w), int((x + bw * f / 2) * w)
            y0, y1 = int((y - bh * f / 2) * h), int((y + bh * f / 2) * h)
            mask[max(y0, 0):y1, max(x0, 0):x1] = True
    healthy = fruit & ring & ~excl
    if dmg_px.sum() < 30 or healthy.sum() < 30:
        continue
    d = img[dmg_px].mean(0)
    s = img[healthy].mean(0)
    contrast.append((s - d)[::-1])                   # RGB order: healthy minus damage

side_px = np.array(side_px)
print(f"damage boxes: {len(side_px)}; side at 640 px letterbox: median {np.median(side_px):.0f}, "
      f"p10 {np.percentile(side_px, 10):.0f}, p90 {np.percentile(side_px, 90):.0f}")
for m, r in ratios.items():
    r = np.array(r)
    print(f"Rotate 20 deg, {m:11s}: box area x{np.median(r):.2f} (median), IoU(box, object) <= {1 / np.median(r):.2f}")
c = np.array(contrast)
print(f"images with contrast: {len(c)}")
for i, ch in enumerate("RGB"):
    print(f"  {ch}: healthy - damage = median {np.median(c[:, i]):+.1f} gray levels, "
          f"p25 {np.percentile(c[:, i], 25):+.1f}, p75 {np.percentile(c[:, i], 75):+.1f}")
print("GaussNoise std range 0.02-0.08 of 255 = 5.1-20.4 gray levels per pixel")

# 3) hue units of the two hue transforms in the pipeline, on a uint8 RGB pixel
px = np.full((8, 8, 3), (200, 180, 40), np.uint8)
hue_deg = lambda im: int(cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)[0, 0, 0]) * 360 / 256
for name, t in (
        ("HueSaturationValue(hue_shift_limit=15)",
         A.HueSaturationValue(hue_shift_limit=(15, 15), sat_shift_limit=0, val_shift_limit=0, p=1)),
        ("ColorJitter(hue=0.05)", A.ColorJitter(brightness=0, contrast=0, saturation=0, hue=(0.05, 0.05), p=1))):
    print(f"{name}: max hue shift {hue_deg(t(image=px)['image']) - hue_deg(px):+.0f} deg")

# 4) per-image weights of build_weighted_sampler (src/training/dataset.py), reproduced on these labels
per_image = [[int(l.split()[0]) for l in labels[s].read_text().splitlines() if l.strip()] for s in stems]
counts = {}
for row in per_image:
    for k in row:
        counts[k] = counts.get(k, 0) + 1
total = sum(counts.values())
class_w = {k: total / (2 * v) for k, v in counts.items()}
sample_w = [max(class_w[k] for k in row) if row else 1.0 for row in per_image]
print(f"sampler: instances {counts}, class weights { {k: round(v, 2) for k, v in class_w.items()} }, "
      f"distinct per-image weights {sorted({round(w, 2) for w in sample_w})}")
