"""RGBA mango cutouts from the captura_v3 polygon dataset (class-0 contour = colour-mask outline)."""
import csv, random, sys
from pathlib import Path
import cv2, numpy as np
D = Path("/home/rsphobos/PROYECTOS_GABO/MCP_LangGraph_Project/data/cache/v4_captura_v3_poly")
OUT = Path(sys.argv[1]); N = int(sys.argv[2])  # usage: make_cutouts.py <out_dir> <n>
rows = list(csv.DictReader(open(D / "index.csv")))
random.seed(7)
by_key = {}
for r in rows:  # stratify: lot x hour, keep faces mixed
    by_key.setdefault((r["lot"], r["ronda_h"]), []).append(r)
per = max(1, N // len(by_key))
pick = [r for v in by_key.values() for r in random.sample(v, min(per, len(v)))]
meta = open(OUT / "cutouts.csv", "w"); meta.write("file,lot,mango_id,cara,ronda_h,w,h\n")
n = 0
for r in pick:
    img = cv2.imread(str(D / "rgb" / f"{r['stem']}.jpg"))
    if img is None: continue
    H, W = img.shape[:2]
    lines = [l.split() for l in open(D / "labels" / f"{r['stem']}.txt") if l.startswith("0 ")]
    if not lines: continue
    pts = (np.array(lines[0][1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)
    mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [pts], 255)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)  # soft edge for pasting
    x, y, w, h = cv2.boundingRect(pts)
    rgba = np.dstack([img[y:y+h, x:x+w], mask[y:y+h, x:x+w]])
    s = 512 / max(w, h); rgba = cv2.resize(rgba, (round(w*s), round(h*s)), interpolation=cv2.INTER_AREA)
    f = f"m{n:04d}.png"; cv2.imwrite(str(OUT / f), rgba)
    meta.write(f"{f},{r['lot']},{r['mango_id']},{r['cara']},{r['ronda_h']},{rgba.shape[1]},{rgba.shape[0]}\n"); n += 1
print("cutouts", n)
