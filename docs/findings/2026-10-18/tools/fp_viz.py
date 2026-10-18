"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

Stage 3: contact sheets of the top-12 class-(c) false positives per
checkpoint (RGB crop around the FP box + 50% context, FP red, GT damage green).
Reads fp_records.json + cached letterboxed-crop RGB; never touches the repos.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from _paths import resolve_roots  # noqa: E402

sys.dont_write_bytecode = True
_, _, OUT, _ = resolve_roots(__doc__, repo=False, data=False)
IMG = 640
TILE = 360
COLS = 4
N_TILES = 12
RED = (0, 0, 255)  # BGR
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)


def xyxy(cx, cy, w, h):
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def tile_for(img_rgb, rec, gt_dmg, mango, idx):
    """gt_dmg: (N,4) xyxy px; mango: (4,) xyxy px; rec box in normalized cxcywh."""
    bw, bh = rec["w"] * IMG, rec["h"] * IMG
    cx, cy = rec["cx"] * IMG, rec["cy"] * IMG
    side = max(2.0 * max(bw, bh), 80.0)  # box + 50% context each side, floor 80px
    side = min(side, IMG)
    x0 = min(max(cx - side / 2, 0), IMG - side)
    y0 = min(max(cy - side / 2, 0), IMG - side)
    x1, y1 = x0 + side, y0 + side
    crop = img_rgb[int(round(y0)) : int(round(y1)), int(round(x0)) : int(round(x1))]
    k = TILE / crop.shape[0]
    tile = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)

    def to_tile(b):
        return (
            int(round((b[0] - x0) * k)),
            int(round((b[1] - y0) * k)),
            int(round((b[2] - x0) * k)),
            int(round((b[3] - y0) * k)),
        )

    for g in gt_dmg:
        if g[2] > x0 and g[0] < x1 and g[3] > y0 and g[1] < y1:
            a = to_tile(g)
            cv2.rectangle(tile, (a[0], a[1]), (a[2], a[3]), GREEN, 2)
    fb = to_tile(xyxy(cx, cy, bw, bh))
    cv2.rectangle(tile, (fb[0], fb[1]), (fb[2], fb[3]), RED, 2)

    # whole-crop inset (top-right) so edge / stem / shadow can be judged
    ins = 96
    thumb = cv2.cvtColor(cv2.resize(img_rgb, (ins, ins), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    s = ins / IMG
    cv2.rectangle(thumb, (int(mango[0] * s), int(mango[1] * s)), (int(mango[2] * s), int(mango[3] * s)), YELLOW, 1)
    cv2.rectangle(
        thumb,
        (int((cx - bw / 2) * s) - 1, int((cy - bh / 2) * s) - 1),
        (int((cx + bw / 2) * s) + 1, int((cy + bh / 2) * s) + 1),
        RED,
        1,
    )
    tile[2 : 2 + ins, TILE - ins - 2 : TILE - 2] = thumb
    cv2.rectangle(tile, (TILE - ins - 2, 2), (TILE - 3, 2 + ins - 1), (255, 255, 255), 1)

    strip = np.full((22, TILE, 3), 30, np.uint8)
    label = f"#{idx} s={rec['score']:.2f} {bw:.0f}x{bh:.0f}px img{rec['img']} iou{rec['max_iou']:.2f}"
    cv2.putText(strip, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, tile])


def sheet(tag, recs_all, cat="c_background_on_fruit"):
    cache = torch.load(OUT / "cache" / f"{tag}.pt", weights_only=False)
    cands = sorted(
        (r for r in recs_all if r["tag"] == tag and r["cat"] == cat), key=lambda r: -r["score"]
    )
    top = cands[:N_TILES]
    tiles = []
    for i, r in enumerate(top, 1):
        gtb, gtl = cache["gts"][r["img"]]
        gt_dmg = torch.stack(
            [gtb[gtl == 1][:, 0] - gtb[gtl == 1][:, 2] / 2, gtb[gtl == 1][:, 1] - gtb[gtl == 1][:, 3] / 2,
             gtb[gtl == 1][:, 0] + gtb[gtl == 1][:, 2] / 2, gtb[gtl == 1][:, 1] + gtb[gtl == 1][:, 3] / 2],
            dim=-1,
        ).numpy() * IMG if (gtl == 1).any() else np.zeros((0, 4))
        m = gtb[gtl == 0][0]
        mango = np.array([m[0] - m[2] / 2, m[1] - m[3] / 2, m[0] + m[2] / 2, m[1] + m[3] / 2]) * IMG
        tiles.append(tile_for(cache["imgs"][r["img"]], r, gt_dmg, mango, i))
    rows = []
    th = tiles[0].shape[0] if tiles else TILE + 22
    while len(tiles) % COLS:
        tiles.append(np.full((th, TILE, 3), 30, np.uint8))
    for k in range(0, len(tiles), COLS):
        rows.append(np.hstack(tiles[k : k + COLS]))
    body = np.vstack(rows)
    title = np.full((30, body.shape[1], 3), 15, np.uint8)
    cv2.putText(
        title,
        f"{tag} (epoch {cache['epoch']}) test: top-{len(top)} of {len(cands)} class-(c) FPs, conf>=0.25, NMS 0.5 | red=FP green=GT damage yellow(inset)=mango",
        (6, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    img = np.vstack([title, body])
    return img, top, len(cands)


if __name__ == "__main__":
    recs = json.loads((OUT / "fp_records.json").read_text())
    meta = {}
    for tag in ["f0_best", "f1_best", "f2_best", "f3_best", "f4_best", "f4_e100"]:
        img, top, n = sheet(tag, recs)
        fold = tag.split("_")[0][1:]
        name = f"fp_top_fold{fold}.png" if tag.endswith("best") else f"fp_top_fold{fold}_epoch100.png"
        cv2.imwrite(str(OUT / name), img)
        meta[name] = {"shape": img.shape[:2], "n_class_c": n, "tiles": top}
        print(name, img.shape, "n_class_c =", n)
    (OUT / "fp_top_tiles_meta.json").write_text(json.dumps(meta, indent=2, default=str))
