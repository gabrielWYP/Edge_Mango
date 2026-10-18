"""Find exact and same-view duplicate images (retakes of the same mango side).

Analysis tool used for the data leakage audit (see docs/data/leakage-audit.md).
It is NOT part of the training pipeline: nothing in `src/` or `scripts/` imports it.

Exact: identical RGB file bytes (whole dataset).
Same view: SIFT keypoints restricted to the fruit, ratio-test matches, RANSAC
homography; a pair is a candidate when many inliers cover a large part of the
fruit in both images. Compared within each capture lot (different lots hold
different mangos) and, for safety, across lots too.

Usage (CPU only; the real data lives in the checkout that holds `data/`):

    CUDA_VISIBLE_DEVICES="" python dup_detect.py --repo-root <checkout-with-data> \\
        --out-dir <dir>                       # writes <dir>/dup_pairs.json   (MIN_N = 12)
    CUDA_VISIBLE_DEVICES="" python dup_detect.py --repo-root <checkout-with-data> \\
        --out-dir <dir> --min-n 8 --suffix _n8   # writes <dir>/dup_pairs_n8.json

`MIN_N` and `SUFFIX` environment variables are still honoured as defaults.
"""
import argparse
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

# docs/data/leakage-audit/tools/dup_detect.py -> repository root is four levels up.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT_DIR = Path(tempfile.gettempdir()) / "leakage_audit"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT,
                        help="Checkout holding configs/folds/lot_5fold.json and data/ "
                             "(default: the repository this script lives in).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Where dup_pairs<suffix>.json is written (default: {DEFAULT_OUT_DIR}).")
    parser.add_argument("--min-n", type=int, default=int(os.environ.get("MIN_N", 12)),
                        help="Minimum RANSAC inliers (and ratio-test matches) for a candidate pair "
                             "(default: $MIN_N or 12).")
    parser.add_argument("--suffix", default=os.environ.get("SUFFIX", ""),
                        help="Suffix of the output file name (default: $SUFFIX or empty).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    M = args.repo_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    OUT = args.out_dir / ("dup_pairs" + args.suffix + ".json")
    LABELS = {p.stem: p for p in (M / "data/annotations/yolo/labels").glob("*/*.txt")}
    lots = json.loads((M / "configs/folds/lot_5fold.json").read_text())["lots"]
    order = {l: sorted(v["stems"], key=lambda s: int(s.rsplit("_", 1)[1])) for l, v in lots.items()}
    lot_of = {s: l for l, ss in order.items() for s in ss}
    idx_of = {s: i + 1 for ss in order.values() for i, s in enumerate(ss)}

    sift = cv2.SIFT_create(nfeatures=2000)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    feats, md5 = {}, {}
    for s in lot_of:
        path = M / "data/cache/mango/rgb" / f"{s}.jpg"
        md5[s] = hashlib.md5(path.read_bytes()).hexdigest()
        im = cv2.imread(str(path))
        H, W = im.shape[:2]
        r = [l.split() for l in LABELS[s].read_text().splitlines() if l.startswith("0 ")][0]
        _, cx, cy, w, h = map(float, r)
        x0, y0 = int(max((cx - w / 2) * W, 0)), int(max((cy - h / 2) * H, 0))
        crop = im[y0:int((cy + h / 2) * H), x0:int((cx + w / 2) * W)]
        s_ = 400 / max(crop.shape[:2])
        gray = clahe.apply(cv2.cvtColor(cv2.resize(crop, None, fx=s_, fy=s_), cv2.COLOR_BGR2GRAY))
        mask = np.zeros(gray.shape, np.uint8)  # fruit only: avoid matching the tray/background
        cv2.ellipse(mask, (gray.shape[1] // 2, gray.shape[0] // 2), (int(gray.shape[1] * 0.42), int(gray.shape[0] * 0.42)), 0, 0, 360, 255, -1)
        kp, des = sift.detectAndCompute(gray, mask)
        feats[s] = (np.float32([k.pt for k in kp]), des, float(mask.sum() / 255))

    MIN_N = args.min_n
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = []
    for a, b in itertools.combinations(sorted(lot_of), 2):
        pa, da, area_a = feats[a]
        pb, db, area_b = feats[b]
        if da is None or db is None or len(da) < 10 or len(db) < 10:
            continue
        good = [m for m, n in (x for x in bf.knnMatch(da, db, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < MIN_N:
            continue
        src = pa[[m.queryIdx for m in good]]
        dst = pb[[m.trainIdx for m in good]]
        Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if Hm is None:
            continue
        inl = inl.ravel().astype(bool)
        n = int(inl.sum())
        if n < MIN_N:
            continue
        cov = min(cv2.contourArea(cv2.convexHull(src[inl])) / area_a, cv2.contourArea(cv2.convexHull(dst[inl])) / area_b)
        pairs.append({"a": a, "b": b, "lot_a": lot_of[a], "lot_b": lot_of[b], "i": idx_of[a], "j": idx_of[b],
                      "good": len(good), "inliers": n, "coverage": round(float(cov), 3)})

    exact = {}
    for s, h in md5.items():
        exact.setdefault(h, []).append(s)
    exact = [v for v in exact.values() if len(v) > 1]
    pairs.sort(key=lambda p: -p["inliers"])
    OUT.write_text(json.dumps({"exact": exact, "pairs": pairs}, indent=1))
    print("wrote", OUT)
    print("exact duplicate groups:", len(exact))
    print(f"candidate pairs (>={MIN_N} inliers):", len(pairs), "| cross-lot:", sum(p["lot_a"] != p["lot_b"] for p in pairs))
    for p in pairs[:40]:
        print(f'{p["lot_a"]}#{p["i"]:<3} {p["lot_b"]}#{p["j"]:<3} inliers={p["inliers"]:4d} good={p["good"]:4d} cov={p["coverage"]:.2f}')


if __name__ == "__main__":
    main()
