"""Draw docs/data/leakage-audit/figures/original_split_leak.jpg.

Analysis tool used for the data leakage audit (see docs/data/leakage-audit.md).
It is NOT part of the training pipeline: nothing in `src/` or `scripts/` imports it.

One block per verified same-face duplicate group (configs/folds/duplicate_groups.json)
whose members sit in DIFFERENT splits of the original `splits.json`, plus the
hand-identified green-patch mango (lot2 #4 / #5, a different pose of the same fruit,
so it is not a same-face pair). Every tile shows `#<lot index> <day> <local time>
d<damage boxes> <original split>`, and its border is coloured by split.

    CUDA_VISIBLE_DEVICES="" python make_leak_figure.py --repo-root <checkout-with-data>
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TILE", "300")  # read by mango_sheets at import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mango_sheets  # noqa: E402

# docs/data/leakage-audit/tools/make_leak_figure.py -> repository root is four levels up.
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[4]
BORDER = {"train": (200, 120, 40), "val": (0, 165, 255), "test": (60, 60, 230)}  # BGR
MANUAL = [("lot2 #4 / #5: same mango, different pose (manual)", "lot2", [4, 5])]


def framed(tile, split):
    out = tile.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), BORDER[split], 4)
    return out


def block(title, tiles):
    header = np.full((26, sum(t.shape[1] for t in tiles), 3), 60, np.uint8)
    cv2.putText(header, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.vstack([header, np.hstack(tiles)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_REPO_ROOT,
                        help="Checkout holding configs/folds/ and data/ (default: this repository).")
    parser.add_argument("--groups", type=Path, default=None,
                        help="duplicate_groups.json (default: <repo-root>/configs/folds/duplicate_groups.json).")
    parser.add_argument("--out", type=Path, default=SCRIPT_REPO_ROOT / "docs/data/leakage-audit/figures/original_split_leak.jpg")
    parser.add_argument("--quality", type=int, default=85)
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    mango_sheets.configure(root, None)
    manifest = json.loads((root / "data/annotations/yolo/splits.json").read_text())
    split_of = {stem: split for split, stems in manifest.items() for stem in stems}
    groups = json.loads((args.groups or root / "configs/folds/duplicate_groups.json").read_text())["groups"]

    blocks = []
    for g in groups:
        if len({m["original_split"] for m in g["members"]}) < 2:
            continue
        tiles = [framed(mango_sheets.tile(m["stem"], m["lot_index"], m["original_split"]), m["original_split"])
                 for m in g["members"]]
        splits = "/".join(m["original_split"] for m in g["members"])
        blocks.append(block(f'{g["group_id"]} ({g["status"]}): {splits}', tiles))
    for title, lot, indices in MANUAL:
        stems = sorted(json.loads((root / "configs/folds/lot_5fold.json").read_text())["lots"][lot]["stems"],
                       key=lambda s: int(s.rsplit("_", 1)[1]))
        tiles = [framed(mango_sheets.tile(stems[i - 1], i, split_of[stems[i - 1]]), split_of[stems[i - 1]]) for i in indices]
        blocks.append(block(title, tiles))

    cols = 2
    width = max(b.shape[1] for b in blocks)
    rows = []
    for i in range(0, len(blocks), cols):
        row = []
        for b in blocks[i:i + cols]:
            pad = np.full((b.shape[0], width - b.shape[1], 3), 30, np.uint8)
            row.append(np.hstack([b, pad]))
        row += [np.full_like(row[0], 30)] * (cols - len(row))
        rows.append(np.hstack(row))
    image = np.vstack(rows)
    if image.shape[1] > 1600:
        scale = 1600 / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), image, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    print(f"wrote {args.out} ({image.shape[1]}x{image.shape[0]}, {len(blocks)} blocks)")


if __name__ == "__main__":
    main()
