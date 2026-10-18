"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

Where do the damage boxes sit inside the mango box, and how well does a fixed "prior box" score?

1. Position statistics. Every damage box (class 1) is expressed relative to the largest mango box
   (class 0) of its image: centre (x, y) in 0..1 inside the mango box (y from the top), width and
   height as a fraction of the mango width / height. Reported: mean and sd of the centres, the share
   of centres inside the central 50% of the mango box (a uniform layout would give 25%), the same
   share per lot, a 5x5 count heatmap (rows top to bottom) and the median relative width / height
   per lot.
2. Prior-box baseline. For each leave-one-lot-out fold the prior is the median (cx, cy, w, h) of the
   fold's TRAIN damage boxes in that relative frame. One copy of it is predicted in every TEST
   image with the same score, so the precision-recall curve is a single point and
   AP = precision x recall. A prediction is a TP when its IoU with some damage GT of that image is
   >= the threshold (IoU is invariant to the per-axis rescaling of the relative frame).
   Reported at IoU 0.5, 0.3 and 0.1.

Pure numpy, CPU only. GT boxes are taken from the original labels and are NOT clipped to the ROI-crop
window (mango box +10%); every damage-box centre lies inside the mango box, so the effect is small
but it is not quantified here.

Reads (relative to --repo-root / --data-root):
  configs/folds/lot_5fold.json                       lot -> stems       (--repo-root)
  data/annotations/yolo/labels/*/*.txt               YOLO labels        (--data-root)
  data/annotations/yolo/folds_lot/fold_K/splits.json train/val/test     (--data-root)
Writes position_prior.json to --out-dir.
"""
from __future__ import annotations

import json

import numpy as np
from _paths import resolve_roots

REPO, DATA, OUT, _ = resolve_roots(__doc__)
IOU_THRESHOLDS = (0.5, 0.3, 0.1)


def load_relative_boxes(label_file):
    """Return (damage boxes relative to the mango box, shape (N, 4) as cx, cy, w, h)."""
    rows = [list(map(float, ln.split())) for ln in label_file.read_text().splitlines() if ln.strip()]
    mango = max((r for r in rows if int(r[0]) == 0), key=lambda r: r[3] * r[4])
    _, mcx, mcy, mw, mh = mango
    rel = [[(cx - mcx) / mw + 0.5, (cy - mcy) / mh + 0.5, w / mw, h / mh]
           for c, cx, cy, w, h in rows if int(c) == 1]
    return np.asarray(rel, dtype=np.float64).reshape(-1, 4)


def iou_one_to_many(a, b):
    """IoU of one (cx, cy, w, h) box against an (N, 4) array of (cx, cy, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    bx2, by2 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    return inter / (a[2] * a[3] + b[:, 2] * b[:, 3] - inter)


def central_share(boxes, lo=0.25, hi=0.75):
    inside = (boxes[:, 0] >= lo) & (boxes[:, 0] <= hi) & (boxes[:, 1] >= lo) & (boxes[:, 1] <= hi)
    return float(inside.mean())


def main():
    lots = json.loads((REPO / "configs/folds/lot_5fold.json").read_text())["lots"]
    labels = {p.stem: p for p in (DATA / "data/annotations/yolo/labels").glob("*/*.txt")}
    rel = {stem: load_relative_boxes(labels[stem]) for lot in lots.values() for stem in lot["stems"]}

    all_boxes = np.concatenate(list(rel.values()))
    heat, _, _ = np.histogram2d(all_boxes[:, 1], all_boxes[:, 0], bins=5, range=[[0, 1], [0, 1]])
    per_lot = {}
    for lot, v in lots.items():
        b = np.concatenate([rel[s] for s in v["stems"]])
        per_lot[lot] = {
            "n_images": len(v["stems"]),
            "n_damage_boxes": int(len(b)),
            "central50_share": round(central_share(b), 3),
            "median_w_over_mango_w": round(float(np.median(b[:, 2])), 3),
            "median_h_over_mango_h": round(float(np.median(b[:, 3])), 3),
        }
    stats = {
        "n_images": len(rel),
        "n_damage_boxes": int(len(all_boxes)),
        "center_x": {"mean": round(float(all_boxes[:, 0].mean()), 3), "sd": round(float(all_boxes[:, 0].std()), 3)},
        "center_y_from_top": {"mean": round(float(all_boxes[:, 1].mean()), 3), "sd": round(float(all_boxes[:, 1].std()), 3)},
        "uniform_reference_sd": round(float(np.sqrt(1 / 12)), 3),
        "central50_share_all": round(central_share(all_boxes), 3),
        "central50_share_uniform_reference": 0.25,
        "centers_outside_mango_box": int(((all_boxes[:, :2] < 0) | (all_boxes[:, :2] > 1)).any(axis=1).sum()),
        "heatmap_5x5_rows_top_to_bottom": heat.astype(int).tolist(),
        "per_lot": per_lot,
    }

    folds = {}
    for k in range(5):
        split = json.loads((DATA / f"data/annotations/yolo/folds_lot/fold_{k}/splits.json").read_text())
        train = np.concatenate([rel[s] for s in split["train"]])
        prior = np.median(train, axis=0)
        n_img = len(split["test"])
        n_gt = int(sum(len(rel[s]) for s in split["test"]))
        ap = {}
        for thr in IOU_THRESHOLDS:
            tp = sum(1 for s in split["test"] if len(rel[s]) and iou_one_to_many(prior, rel[s]).max() >= thr)
            precision, recall = tp / n_img, tp / n_gt
            ap[str(thr)] = {"tp": tp, "precision": round(precision, 4), "recall": round(recall, 4),
                            "ap": round(precision * recall, 4)}
        folds[f"fold{k}"] = {
            "prior_cx_cy_w_h": [round(float(x), 2) for x in prior],
            "n_train_damage_boxes": int(len(train)),
            "n_test_images": n_img,
            "n_test_damage_gt": n_gt,
            "prior_ap_by_iou": ap,
        }

    result = {
        "method": "Damage boxes relative to the largest mango box; prior = median train box; one prior box per test "
                  "image, equal scores, AP = precision x recall; GT not clipped to the ROI-crop window.",
        "position_stats": stats,
        "prior_baseline": folds,
    }
    (OUT / "position_prior.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(stats, indent=1))
    for name, f in folds.items():
        print(name, f["prior_cx_cy_w_h"], {t: v["ap"] for t, v in f["prior_ap_by_iou"].items()})


if __name__ == "__main__":
    main()
