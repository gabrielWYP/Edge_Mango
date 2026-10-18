"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

Stage 1 (CPU only): run each checkpoint's forward pass once over the fold's
test split and cache the raw head outputs, GT, image names and the
letterboxed-crop RGB so decode / NMS / taxonomy can be redone without
re-running the model.

Reuses the repo's own loaders (scripts/evaluate_checkpoint.py helpers), so the
ROI-crop binding check, dataset and model construction are the real ones.
Writes nothing into either repo checkout.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys

sys.dont_write_bytecode = True
import time
from pathlib import Path

from _paths import resolve_roots  # noqa: E402

REPO, M, OUT, _ = resolve_roots(__doc__)  # M = data root (data/, checkpoints/); REPO = code root
os.chdir(M)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import torch

torch.set_num_threads(10)

import evaluate_checkpoint as ec  # noqa: E402
from src.training.config import TrainingConfig  # noqa: E402
from src.training.dataset import build_dataloader, collate_fn  # noqa: E402

CFG = REPO / "configs/experiment/twostream_roi_crop.yaml"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def run(tag: str, fold: int, ckpt_name: str) -> None:
    dst = OUT / "cache" / f"{tag}.pt"
    if dst.exists():
        print(f"[skip] {tag}")
        return
    t0 = time.time()
    fold_dir = M / f"data/annotations/yolo/folds_lot/fold_{fold}"
    ckpt = M / f"checkpoints/kfold_lot/twostream_roi_crop/fold{fold}/seed42/{ckpt_name}"
    cfg = TrainingConfig.from_yaml(str(CFG))
    ec._apply_overrides(
        cfg,
        [
            f"labels_dir={fold_dir}/labels",
            f"split_manifest={fold_dir}/splits.json",
            "device=cpu",
            "batch_size=2",
            "num_workers=2",
            f"output_dir={OUT}/trainer_out",
        ],
    )
    cfg.model_type = "master"
    dev = torch.device("cpu")
    model = ec._load_model("master", str(ckpt), cfg, dev)
    cfg.head_strides = model.head_strides
    ds = ec._build_eval_dataset(cfg, "test")
    loader = build_dataloader(
        ds, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=False
    )
    ck_meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    epoch = ck_meta.get("epoch") if isinstance(ck_meta, dict) else None
    del ck_meta

    batches = []  # per batch: preds, cls_preds (lists of tensors)
    gts = []  # per image (boxes cxcywh norm, labels)
    imgs = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["rgb"], batch["nir"])
            batches.append(
                {
                    "preds": [p.cpu() for p in out["preds"]],
                    "cls_preds": [c.cpu() for c in out["cls_preds"]],
                    "n": batch["rgb"].shape[0],
                }
            )
            for b in range(batch["rgb"].shape[0]):
                gts.append((batch["bboxes"][b].clone(), batch["labels"][b].clone()))
                rgb = batch["rgb"][b].numpy().transpose(1, 2, 0)
                rgb = (rgb * IMAGENET_STD + IMAGENET_MEAN) * 255.0
                imgs.append(np.clip(rgb.round(), 0, 255).astype(np.uint8))
    names = [p["rgb_path"].stem for p in ds.pairs]
    assert len(names) == len(gts) == len(imgs), (len(names), len(gts), len(imgs))
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tag": tag,
            "fold": fold,
            "ckpt": str(ckpt),
            "epoch": epoch,
            "head_strides": list(model.head_strides),
            "batches": batches,
            "gts": gts,
            "imgs": imgs,
            "names": names,
        },
        dst,
    )
    print(f"[done] {tag} epoch={epoch} n={len(names)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    jobs = [(f"f{k}_best", k, "best_model.pt") for k in range(5)]
    jobs.append(("f4_e100", 4, "checkpoint_epoch100.pt"))
    for tag, fold, ck in jobs:
        run(tag, fold, ck)
