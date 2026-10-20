"""Analysis tool, not part of the training pipeline.

Review an unlabeled RGB/NIR capture: register the NIR to the RGB (stored homography + ECC refinement on
the fruit), find dark spots inside the fruit, and ROI-crop both images as the training pipeline does.

Usage:
  python process_new_capture.py RGB_IMAGE NIR_IMAGE OUT_DIR [--repo-root REPO]

--repo-root is the checkout that holds src/ and notebooks/matriz_homografia_aruco.npy
(default: the repository this file lives in). Writes numbered figures 01-08 to OUT_DIR and prints the
alignment NCC, the dark spots and the crop size. CPU only.
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# <repo>/docs/findings/2026-10-18/tools/process_new_capture.py -> parents[4] is <repo>.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("rgb", type=Path, help="RGB capture (mango_rgb_<ts>.jpg)")
_ap.add_argument("nir", type=Path, help="NIR capture (mango_nir_<ts>.jpg)")
_ap.add_argument("out", type=Path, help="output directory for the figures")
_ap.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT,
                 help="code checkout with src/ and notebooks/ (default: this repository)")
_args = _ap.parse_args()

WT = _args.repo_root.resolve()
sys.path.insert(0, str(WT))
from src.training.nir_registration import load_homography, register_nir_to_rgb  # noqa: E402
from src.training.roi_crop import crop_to_roi  # noqa: E402

RGB_P, NIR_P, OUT = _args.rgb, _args.nir, _args.out
OUT.mkdir(parents=True, exist_ok=True)
shutil.copy(RGB_P, OUT / "01_rgb_original.jpg")
shutil.copy(NIR_P, OUT / "02_nir_original.jpg")

rgb = cv2.imread(str(RGB_P))
nir = cv2.imread(str(NIR_P), cv2.IMREAD_GRAYSCALE)  # exactly how the training pipeline reads NIR
H_img, W_img = rgb.shape[:2]
H_stored = load_homography(WT / "notebooks/matriz_homografia_aruco.npy")
reg0 = register_nir_to_rgb(nir, H_stored, (W_img, H_img), 0)
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

# Fruit mask and box from RGB color (yellow/orange, saturated) on the white sheet.
hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
mask = ((hsv[..., 0] > 10) & (hsv[..., 0] < 40) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
fruit = (lab == k).astype(np.uint8)
fruit = cv2.morphologyEx(fruit, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
x, y, w, h = st[k, :4]
print(f"mango box (RGB px): x={x} y={y} w={w} h={h} -> {w * h / (W_img * H_img) * 100:.1f}% of the frame")


def grad(a):
    a = cv2.GaussianBlur(a.astype(np.float32), (5, 5), 0)
    return cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


# ECC refinement on gradient magnitude around the fruit (the fruit is 3D, so a tray-plane H leaves parallax).
pad = 60
sl = (slice(max(y - pad, 0), min(y + h + pad, H_img)), slice(max(x - pad, 0), min(x + w + pad, W_img)))
ga, gb = grad(gray)[sl], grad(reg0)[sl]
ga, gb = ga / (ga.max() + 1e-6), gb / (gb.max() + 1e-6)
warp = np.eye(3, dtype=np.float32)
try:
    _, warp = cv2.findTransformECC(ga, gb, warp, cv2.MOTION_HOMOGRAPHY,
                                   (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), None, 5)
    # warp maps template (RGB ROI) coords -> input (registered NIR ROI) coords
    T = np.array([[1, 0, sl[1].start], [0, 1, sl[0].start], [0, 0, 1]], np.float32)
    corr = T @ warp @ np.linalg.inv(T)
    reg = cv2.warpPerspective(reg0, corr, (W_img, H_img), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    refined = True
except cv2.error as e:
    print("ECC refinement failed, using stored H:", e)
    reg, refined = reg0, False
ncc0 = ncc(grad(gray)[sl], grad(reg0)[sl])
ncc1 = ncc(grad(gray)[sl], grad(reg)[sl])
(dx0, dy0), _ = cv2.phaseCorrelate(grad(gray)[sl], grad(reg0)[sl])
(dx1, dy1), _ = cv2.phaseCorrelate(grad(gray)[sl], grad(reg)[sl])
print(f"alignment NCC: stored H {ncc0:.3f} -> refined {ncc1:.3f} | residual shift {np.hypot(dx0, dy0):.1f} px -> {np.hypot(dx1, dy1):.1f} px")
cv2.imwrite(str(OUT / "03_nir_registered_to_rgb.png"), reg)

# Edge overlay for alignment check (red = RGB edges, green = NIR edges).
def edges(a):
    return cv2.Canny(cv2.GaussianBlur(a, (5, 5), 0), 30, 90) > 0
ov = cv2.cvtColor(reg, cv2.COLOR_GRAY2BGR) // 2
ov[edges(gray)] = (0, 0, 255)
ov[edges(reg)] = (0, 255, 0)
ov[edges(gray) & edges(reg)] = (0, 255, 255)
cv2.imwrite(str(OUT / "04_alignment_edges.png"), ov[sl])

# Dark spots: divide by a heavily smoothed version of itself inside the fruit (removes the illumination gradient).
inner = cv2.erode(fruit, np.ones((21, 21), np.uint8)).astype(bool)


def ratio_map(img):
    f = img.astype(np.float32)
    m = inner.astype(np.float32)
    bg = cv2.GaussianBlur(f * m, (0, 0), 25) / (cv2.GaussianBlur(m, (0, 0), 25) + 1e-6)
    r = f / (bg + 1e-6)
    r[~inner] = 1.0
    return r


r_nir, r_rgb = ratio_map(reg), ratio_map(gray)
dark = ((r_nir < 0.92) & inner).astype(np.uint8)
dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(dark)
spots = []
for i in range(1, n2):
    if st2[i, cv2.CC_STAT_AREA] < 40:
        continue
    m = lab2 == i
    spots.append((st2[i, cv2.CC_STAT_AREA], cen2[i], float(r_nir[m].mean()), float(r_rgb[m].mean())))
spots.sort(key=lambda s: s[2])
print("dark spots in NIR inside the fruit (darkness = 1 - local ratio; 0.10 = 10% darker than surroundings):")
for j, (area, (cx, cy), rn, rr) in enumerate(spots[:8], 1):
    print(f"  #{j}: center=({cx:.0f},{cy:.0f}) area={area:4d} px  NIR darkness={1 - rn:.3f}  RGB darkness at same spot={1 - rr:.3f}")

# Annotated side-by-side of the fruit region.
ann_r, ann_n = rgb[sl].copy(), cv2.cvtColor(reg[sl], cv2.COLOR_GRAY2BGR)
for j, (area, (cx, cy), rn, rr) in enumerate(spots[:8], 1):
    p = (int(cx - sl[1].start), int(cy - sl[0].start))
    rad = int(max(8, np.sqrt(area / np.pi) * 1.6))
    for im in (ann_r, ann_n):
        cv2.circle(im, p, rad, (0, 0, 255), 2)
        cv2.putText(im, str(j), (p[0] + rad, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.imwrite(str(OUT / "05_spots_rgb_vs_nir.png"), np.hstack([ann_r, np.full((ann_r.shape[0], 8, 3), 255, np.uint8), ann_n]))

# ROI crop exactly as the training pipeline does (GT-like mango box + 10% margin), RGB and registered NIR.
box = np.array([[(x + w / 2) / W_img, (y + h / 2) / H_img, w / W_img, h / H_img]], np.float32)
rgb_c, nir_c, _, _ = crop_to_roi(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), reg, box, np.array([0]), margin=0.10)
cv2.imwrite(str(OUT / "06_rgb_crop.png"), cv2.cvtColor(rgb_c, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(OUT / "07_nir_registered_crop.png"), nir_c)
side = np.hstack([cv2.cvtColor(rgb_c, cv2.COLOR_RGB2BGR), np.full((rgb_c.shape[0], 8, 3), 255, np.uint8),
                  cv2.cvtColor(nir_c, cv2.COLOR_GRAY2BGR)])
cv2.imwrite(str(OUT / "08_final_crop_rgb_nir.png"), side)
print(f"crop size: {rgb_c.shape[1]}x{rgb_c.shape[0]} px (native resolution, before the 640 letterbox); refined={refined}")
