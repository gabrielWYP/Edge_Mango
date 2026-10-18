"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

Stage 2 (CPU only): NMS sensitivity + FP taxonomy from the cached forward
passes (fp_forward.py). Decode and metrics reuse the repo's own code:
Trainer._decode_predictions_static (-> decode.decode_detections) and
metrics.compute_map, exactly as Trainer.evaluate() calls them.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys

sys.dont_write_bytecode = True
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from _paths import resolve_roots  # noqa: E402

# Optional positional MODE (all | nms | tax) is whatever the root flags leave over.
REPO, M, OUT, _REST = resolve_roots(__doc__)  # M = data root; REPO = code root
os.chdir(M)
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from torchvision.ops import box_iou

CFG = REPO / "configs/experiment/twostream_roi_crop.yaml"
TAGS = ["f0_best", "f1_best", "f2_best", "f3_best", "f4_best", "f4_e100"]
NMS_VALUES = [0.3, 0.4, 0.5, 0.6]
IMG = 640


def _cfg(nms: float):
    from src.training.config import TrainingConfig

    cfg = TrainingConfig.from_yaml(str(CFG))
    cfg.nms_iou_threshold = nms
    cfg.device = "cpu"
    return cfg


def decode_all(cache: dict, nms: float):
    """Decode every image exactly as Trainer.evaluate() does (eval_conf 0.001)."""
    from src.training.loop import Trainer

    cfg = _cfg(nms)
    cfg.head_strides = cache["head_strides"]
    boxes, scores, labels = [], [], []
    for bt in cache["batches"]:
        for b in range(bt["n"]):
            bx, sc, lb = Trainer._decode_predictions_static(
                bt["preds"], bt["cls_preds"], b, cfg, conf_threshold=cfg.eval_conf_threshold
            )
            boxes.append(bx)
            scores.append(sc)
            labels.append(lb)
    return cfg, boxes, scores, labels


def nms_job(args):
    tag, nms = args
    torch.set_num_threads(1)
    from src.training.metrics import compute_map

    cache = torch.load(OUT / "cache" / f"{tag}.pt", weights_only=False)
    cfg, boxes, scores, labels = decode_all(cache, nms)
    gt_boxes = [g[0] for g in cache["gts"]]
    gt_labels = [g[1] for g in cache["gts"]]
    m = compute_map(
        pred_boxes=boxes,
        pred_scores=scores,
        pred_labels=labels,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        num_classes=cfg.num_classes,
        score_threshold=cfg.conf_threshold,
    )
    n_dmg_total = int(sum(int((l == 1).sum()) for l in labels))
    n_dmg_025 = int(sum(int(((l == 1) & (s >= cfg.conf_threshold)).sum()) for l, s in zip(labels, scores)))
    return {
        "tag": tag,
        "nms": nms,
        "ap50": float(m["per_class_ap_50"][1]),
        "ap50_95": float(m["per_class_ap_50_95"][1]),
        "tp": int(m["per_class_counts"][1]["tp"]),
        "fp": int(m["per_class_counts"][1]["fp"]),
        "fn": int(m["per_class_counts"][1]["fn"]),
        "n_dmg_preds_001": n_dmg_total,
        "n_dmg_preds_025": n_dmg_025,
        "n_images": len(boxes),
    }


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def taxonomy(tag: str, nms: float = 0.5, conf: float = 0.25):
    """Classify every damage prediction >= conf using the evaluator's own
    matching convention (_compute_operating_point): greedy by score, best-IoU
    GT only, a GT can be matched once."""
    cache = torch.load(OUT / "cache" / f"{tag}.pt", weights_only=False)
    cfg, boxes, scores, labels = decode_all(cache, nms)
    recs = []
    per_image = []
    for i, (bx, sc, lb) in enumerate(zip(boxes, scores, labels)):
        gtb, gtl = cache["gts"][i]
        dmg_gt = cxcywh_to_xyxy(gtb[gtl == 1]) if (gtl == 1).any() else torch.zeros((0, 4))
        mango = cxcywh_to_xyxy(gtb[gtl == 0])[0] if (gtl == 0).any() else None
        sel = (lb == 1) & (sc >= conf)
        pb, ps = bx[sel], sc[sel]
        order = torch.argsort(ps, descending=True)
        pb, ps = pb[order], ps[order]
        pxy = cxcywh_to_xyxy(pb) if len(pb) else pb
        matched = set()
        n_tp = 0
        for j in range(len(pb)):
            cx, cy = float(pb[j, 0]), float(pb[j, 1])
            in_mango = bool(
                mango is not None and mango[0] <= cx <= mango[2] and mango[1] <= cy <= mango[3]
            )
            if len(dmg_gt) == 0:
                best_iou, best_idx = 0.0, -1
            else:
                ious = box_iou(pxy[j].unsqueeze(0), dmg_gt).squeeze(0)
                bi, bidx = ious.max(dim=0)
                best_iou, best_idx = float(bi), int(bidx)
            if best_iou >= 0.5 and best_idx not in matched:
                matched.add(best_idx)
                cat, n_tp = "TP", n_tp + 1
            elif best_iou >= 0.5:
                cat = "a_duplicate"
            elif best_iou >= 0.1:
                cat = "b_localization"
            elif in_mango:
                cat = "c_background_on_fruit"
            else:
                cat = "d_off_fruit"
            recs.append(
                {
                    "tag": tag,
                    "img": i,
                    "name": cache["names"][i],
                    "score": float(ps[j]),
                    "cx": cx,
                    "cy": cy,
                    "w": float(pb[j, 2]),
                    "h": float(pb[j, 3]),
                    "max_iou": best_iou,
                    "center_in_mango": in_mango,
                    "cat": cat,
                }
            )
        per_image.append(
            {"img": i, "name": cache["names"][i], "n_pred": int(len(pb)), "n_gt": int(len(dmg_gt)), "tp": n_tp}
        )
    return recs, per_image


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def summarize_taxonomy(recs, per_image):
    cats = ["a_duplicate", "b_localization", "c_background_on_fruit", "d_off_fruit"]
    fps = [r for r in recs if r["cat"] != "TP"]
    tps = [r for r in recs if r["cat"] == "TP"]
    s = {
        "n_images": len(per_image),
        "n_pred_025": len(recs),
        "n_tp": len(tps),
        "n_fp": len(fps),
        "n_gt": int(sum(p["n_gt"] for p in per_image)),
        "preds_per_image_mean": float(np.mean([p["n_pred"] for p in per_image])),
        "preds_per_image_median": float(np.median([p["n_pred"] for p in per_image])),
        "gt_per_image_mean": float(np.mean([p["n_gt"] for p in per_image])),
        "counts": {c: sum(1 for r in fps if r["cat"] == c) for c in cats},
        "fp_score_median": q([r["score"] for r in fps], 50),
        "fp_score_p90": q([r["score"] for r in fps], 90),
        "fp_score_max": q([r["score"] for r in fps], 100),
        "tp_score_median": q([r["score"] for r in tps], 50),
        "tp_score_p90": q([r["score"] for r in tps], 90),
        "fp_ge_0.5": sum(1 for r in fps if r["score"] >= 0.5),
        "fp_ge_0.7": sum(1 for r in fps if r["score"] >= 0.7),
        "tp_ge_0.5": sum(1 for r in tps if r["score"] >= 0.5),
        "score_by_cat": {
            c: {
                "median": q([r["score"] for r in fps if r["cat"] == c], 50),
                "p90": q([r["score"] for r in fps if r["cat"] == c], 90),
                "n_ge_0.5": sum(1 for r in fps if r["cat"] == c and r["score"] >= 0.5),
            }
            for c in cats
        },
        "fp_size_px_median": [
            float(np.median([r["w"] * IMG for r in fps])) if fps else None,
            float(np.median([r["h"] * IMG for r in fps])) if fps else None,
        ],
        "tp_size_px_median": [
            float(np.median([r["w"] * IMG for r in tps])) if tps else None,
            float(np.median([r["h"] * IMG for r in tps])) if tps else None,
        ],
    }
    return s


if __name__ == "__main__":
    mode = _REST[0] if _REST else "all"
    if mode in ("all", "nms"):
        jobs = [(t, n) for t in TAGS for n in NMS_VALUES]
        with ProcessPoolExecutor(max_workers=6) as ex:
            res = list(ex.map(nms_job, jobs))
        (OUT / "nms_sensitivity.json").write_text(json.dumps(res, indent=2))
        print(f"{'ckpt':8s} {'nms':>4s} {'AP50':>7s} {'AP50-95':>8s} {'TP':>4s} {'FP':>5s} {'FN':>4s} {'dmgPred>=.25':>12s}")
        for r in res:
            print(
                f"{r['tag']:8s} {r['nms']:4.1f} {r['ap50']:7.4f} {r['ap50_95']:8.4f} {r['tp']:4d} {r['fp']:5d} {r['fn']:4d} {r['n_dmg_preds_025']:12d}"
            )
    if mode in ("all", "tax"):
        torch.set_num_threads(4)
        all_recs, summ = [], {}
        for tag in TAGS:
            recs, per_image = taxonomy(tag)
            all_recs.extend(recs)
            summ[tag] = summarize_taxonomy(recs, per_image)
            summ[tag]["per_image"] = per_image
        (OUT / "fp_records.json").write_text(json.dumps(all_recs))
        (OUT / "taxonomy_summary.json").write_text(json.dumps(summ, indent=2))
        print("taxonomy written")
