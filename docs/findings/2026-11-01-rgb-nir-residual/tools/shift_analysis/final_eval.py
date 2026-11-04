"""Final check on the built dataset. Reads only baked files plus the raw photos, not the module.

Zones: fruit = grown mask; ring = around the outline; skin = eroded mask. Gradient correlation at zero displacement,
for the homography alone (warped here) and for the baked NIR. Writes one row per photo.
"""
import csv, math, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, numpy as np
WT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WT))
from src.annotation.mango_registration import rgb_mango_mask
DATA = Path("data/cache/captura_v3")
OUT = Path(sys.argv[1])
def gradient(image):
    smooth = cv2.GaussianBlur(image, (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
def corr(a, b, m):
    x, y = a[m > 0].astype(np.float64), b[m > 0].astype(np.float64)
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")
def job(r):
    ts = int(r["timestamp"])
    rgb_path = OUT / "rgb" / f"mango_rgb_{ts}.jpg"
    H = np.load(WT / r["homography"])
    rgb = cv2.resize(cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE), None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    baked = cv2.resize(cv2.imread(str(OUT / "nir" / f"mango_nir_{ts}.png"), cv2.IMREAD_UNCHANGED), None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    raw = cv2.imread(str(DATA / "news/nir" / f"mango_nir_{ts}.jpg"))
    gray = np.clip(np.rint((raw[..., 0].astype(np.float32) + raw[..., 2]) / 2), 0, 255).astype(np.uint8)
    honly = cv2.resize(cv2.warpPerspective(gray, np.linalg.inv(H), (3264, 2448)), None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    m4 = rgb_mango_mask(cv2.imread(str(rgb_path), cv2.IMREAD_REDUCED_COLOR_4))
    mask = cv2.resize(m4, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    el = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
    radius = math.sqrt(np.count_nonzero(mask) / math.pi)
    inner = cv2.erode(mask, el(round(.2 * radius))); fruit = cv2.dilate(mask, el(16)); ring = cv2.subtract(fruit, inner)
    gr, gb, gh = gradient(rgb), gradient(baked), gradient(honly)
    rec = dict(ts=ts, lot=r["lot"], ronda=r["ronda_h"], dx=r["residual_dx"], dy=r["residual_dy"], source=r["residual_source"], rcorr=r["residual_corr"])
    for name, zone in (("fruit", fruit), ("ring", ring), ("skin", inner)):
        rec[f"{name}_h"] = round(corr(gr, gh, zone), 4)
        rec[f"{name}_b"] = round(corr(gr, gb, zone), 4)
    return rec
def main():
    cv2.setNumThreads(1)
    rows = list(csv.DictReader((OUT / "index.csv").open()))
    with ProcessPoolExecutor(10) as pool:
        recs = list(pool.map(job, rows, chunksize=8))
    with open(sys.argv[2], "w", newline="") as h:
        w = csv.DictWriter(h, list(recs[0])); w.writeheader(); w.writerows(recs)
main()
