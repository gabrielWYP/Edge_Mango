"""Per-image appearance features to support manual mango re-identification (one lot).

Analysis tool used for the data leakage audit (see docs/data/leakage-audit.md).
It is NOT part of the training pipeline: nothing in `src/` or `scripts/` imports it.

    python mango_features.py [--repo-root R] [--out-dir D] <lot>

Prints one row per image of the lot (capture order): local time, mango box size
(w, h, area in thousands of px^2, long side), the share of red / green / dark
pixels inside an elliptical mask, the median hue and the number of damage boxes.
The same table is also written to <out-dir>/features_<lot>.txt.
"""
import argparse
import datetime as dt
import json
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

# docs/data/leakage-audit/tools/mango_features.py -> repository root is four levels up.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT_DIR = Path(tempfile.gettempdir()) / "leakage_audit"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("lot", help="lot id, e.g. lot2")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT,
                        help="Checkout holding configs/folds/lot_5fold.json and data/ "
                             "(default: the repository this script lives in).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Where features_<lot>.txt is written (default: {DEFAULT_OUT_DIR}).")
    args = parser.parse_args(argv)

    M = args.repo_root.resolve()
    LABELS = {p.stem: p for p in (M / "data/annotations/yolo/labels").glob("*/*.txt")}
    lot = args.lot
    stems = sorted(json.loads((M / "configs/folds/lot_5fold.json").read_text())["lots"][lot]["stems"],
                   key=lambda s: int(s.rsplit("_", 1)[1]))
    lines = [" #   time      w   h   area  long  red%  grn%  dark%  hue  d"]
    for i, s in enumerate(stems, 1):
        im = cv2.imread(str(M / "data/cache/mango/rgb" / f"{s}.jpg"))
        H, W = im.shape[:2]
        rows = [l.split() for l in LABELS[s].read_text().splitlines() if l.strip()]
        _, cx, cy, w, h = map(float, [r for r in rows if r[0] == "0"][0])
        x0, x1, y0, y1 = int((cx - w / 2) * W), int((cx + w / 2) * W), int((cy - h / 2) * H), int((cy + h / 2) * H)
        crop = im[max(y0, 0):y1, max(x0, 0):x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Elliptical mask approximates the fruit inside its box.
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.ellipse(mask, (crop.shape[1] // 2, crop.shape[0] // 2), (int(crop.shape[1] * 0.45), int(crop.shape[0] * 0.45)), 0, 0, 360, 255, -1)
        hh, ss, vv = (c[mask > 0].astype(float) for c in cv2.split(hsv))
        red = ((hh < 8) | (hh > 170)) & (ss > 90) & (vv > 60)
        grn = (hh > 32) & (hh < 85) & (ss > 60)
        dark = vv < 90
        t = dt.datetime.fromtimestamp(int(s.rsplit("_", 1)[1]), ZoneInfo("America/Lima"))
        pw, ph = w * W, h * H
        lines.append(f"{i:2d} {t:%d %H:%M:%S} {pw:4.0f} {ph:4.0f} {pw*ph/1000:5.1f}k {max(pw,ph):4.0f} "
                     f"{red.mean()*100:5.1f} {grn.mean()*100:5.1f} {dark.mean()*100:5.1f} {np.median(hh):4.0f} "
                     f"{sum(r[0]=='1' for r in rows)}")
    text = "\n".join(lines)
    print(text)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"features_{lot}.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
