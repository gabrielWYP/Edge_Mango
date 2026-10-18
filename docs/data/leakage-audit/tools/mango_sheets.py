"""Contact sheets for manual mango re-identification within a capture lot.

Analysis tool used for the data leakage audit (see docs/data/leakage-audit.md).
It is NOT part of the training pipeline: nothing in `src/` or `scripts/` imports it.

  python mango_sheets.py [--repo-root R] [--out-dir D] sheet  <lot> <start> <count> <out.png>
  python mango_sheets.py [--repo-root R] [--out-dir D] groups <lot> <groups.json> <out.png>

`sheet` draws time-ordered tiles (1-based capture-order index); `groups` draws one
row per proposed mango. A relative <out.png> is written under --out-dir. The
capture-order index -> stem map of the lot is written to <out-dir>/index_<lot>.json.
Tile size and columns come from the TILE (default 210) and COLS (default 6)
environment variables.

As a module, call `configure(repo_root, out_dir)` once and then use `tile()`.
"""
import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

# docs/data/leakage-audit/tools/mango_sheets.py -> repository root is four levels up.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT_DIR = Path(tempfile.gettempdir()) / "leakage_audit"
TILE = int(os.environ.get("TILE", 210))
COLS = int(os.environ.get("COLS", 6))

M = DEFAULT_REPO_ROOT
OUT = DEFAULT_OUT_DIR
LABELS = {}


def configure(repo_root=None, out_dir=None):
    """Point the module at a checkout that holds `data/` and `configs/folds/`."""
    global M, OUT, LABELS
    M = Path(repo_root).resolve() if repo_root is not None else DEFAULT_REPO_ROOT
    OUT = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    LABELS = {p.stem: p for p in (M / "data/annotations/yolo/labels").glob("*/*.txt")}


configure()


def lot_stems(lot):
    stems = json.loads((M / "configs/folds/lot_5fold.json").read_text())["lots"][lot]["stems"]
    stems = sorted(stems, key=lambda s: int(s.rsplit("_", 1)[1]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"index_{lot}.json").write_text(json.dumps({i + 1: s for i, s in enumerate(stems)}, indent=1))
    return stems


def tile(stem, idx, tag=""):
    im = cv2.imread(str(M / "data/cache/mango/rgb" / f"{stem}.jpg"))
    H, W = im.shape[:2]
    rows = [l.split() for l in LABELS[stem].read_text().splitlines() if l.strip()]
    _, cx, cy, w, h = map(float, [r for r in rows if r[0] == "0"][0])
    m = 1.12
    x0, x1 = int(max((cx - w / 2 * m) * W, 0)), int(min((cx + w / 2 * m) * W, W))
    y0, y1 = int(max((cy - h / 2 * m) * H, 0)), int(min((cy + h / 2 * m) * H, H))
    crop = im[y0:y1, x0:x1]
    s = TILE / max(crop.shape[:2])
    crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
    canvas = np.full((TILE + 22, TILE, 3), 30, np.uint8)
    oy, ox = (TILE - crop.shape[0]) // 2, (TILE - crop.shape[1]) // 2
    canvas[22 + oy:22 + oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    t = dt.datetime.fromtimestamp(int(stem.rsplit("_", 1)[1]), ZoneInfo("America/Lima"))
    nd = sum(r[0] == "1" for r in rows)
    cv2.putText(canvas, f"#{idx} {t:%d %H:%M:%S} d{nd} {tag}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
    return canvas


def grid(tiles, cols):
    tiles = tiles + [np.zeros_like(tiles[0])] * ((-len(tiles)) % cols)
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT,
                        help="Checkout holding configs/folds/lot_5fold.json and data/ "
                             "(default: the repository this script lives in).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR}).")
    sub = parser.add_subparsers(dest="mode", required=True)
    p_sheet = sub.add_parser("sheet", help="time-ordered tiles")
    p_sheet.add_argument("lot")
    p_sheet.add_argument("start", type=int)
    p_sheet.add_argument("count", type=int)
    p_sheet.add_argument("out")
    p_groups = sub.add_parser("groups", help="one row per proposed mango")
    p_groups.add_argument("lot")
    p_groups.add_argument("groups_json", type=Path)
    p_groups.add_argument("out")
    args = parser.parse_args(argv)

    configure(args.repo_root, args.out_dir)
    stems = lot_stems(args.lot)
    out = OUT / args.out  # an absolute <out.png> overrides --out-dir
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "sheet":
        sel = list(range(args.start, min(args.start + args.count, len(stems) + 1)))
        cv2.imwrite(str(out), grid([tile(stems[i - 1], i) for i in sel], COLS))
    else:
        groups = json.loads(args.groups_json.read_text())
        width = max(len(g) for g in groups.values())
        rows = []
        for gid, members in groups.items():
            ts = [tile(stems[i - 1], i, gid) for i in members]
            ts += [np.zeros_like(ts[0])] * (width - len(ts))
            rows.append(np.hstack(ts))
        cv2.imwrite(str(out), np.vstack(rows))
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
