#!/usr/bin/env python3
"""H-A (fusion-redesign): does mean-preserving stem inflation preserve
ImageNet feature statistics better than zero-init or naive copy?

Zero-cost rung — no training. Builds all three stem-init strategies for the
4-channel `EarlyFusionBackbone`, runs them on the same real-shaped random
batches, and compares each stage's per-channel activation mean/std against
an RGB-only pretrained ConvNeXt-Tiny reference run on the RGB channels of
the same batches.

CONFIRM: inflation's per-stage activation statistics fall within 2x the
reference's own across-image std at every stage, and closer to the
reference than naive (non-rescaled) copy.
REFUTE: inflation is outside that band -> fall back to zero-init, recorded
here with the measurement that justified it.

Usage:
    ./.venv/bin/python scripts/stem_init_audit.py --init inflation zero copy
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.master.backbone import EarlyFusionBackbone


def _build_backbone(init: str) -> EarlyFusionBackbone:
    """Build a 4-channel EarlyFusionBackbone under the named init strategy.

    - "inflation": the shipped `_load_pretrained_stem` (D-1/D-B) — this IS
      what `EarlyFusionBackbone(pretrained=True)` does.
    - "zero": zero-init the stem conv weight/bias entirely (the H-A fallback).
    - "copy": naive, non-rescaled copy — `W_new[:, :3] = W_imagenet` (no 3/4
      factor), `W_new[:, 3] = mean(W_imagenet, dim=1)` (no 3/4 factor either)
      — reproduces the "raises pre-activation by ~4/3" failure mode D-1
      argues against, for direct comparison.
    """
    backbone = EarlyFusionBackbone(pretrained=(init == "inflation"), variant="tiny", in_channels=4)
    if init == "zero":
        with torch.no_grad():
            backbone.stem[0].weight.zero_()
            if backbone.stem[0].bias is not None:
                backbone.stem[0].bias.zero_()
    elif init == "copy":
        pretrained = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        w = pretrained.features[0][0].weight
        with torch.no_grad():
            new_w = torch.empty_like(backbone.stem[0].weight)
            new_w[:, :3] = w
            new_w[:, 3] = w.mean(dim=1)
            backbone.stem[0].weight.copy_(new_w)
            backbone.stem[0].bias.copy_(pretrained.features[0][0].bias)
            backbone.stem[1].norm.weight.copy_(pretrained.features[0][1].weight)
            backbone.stem[1].norm.bias.copy_(pretrained.features[0][1].bias)
    elif init != "inflation":
        raise ValueError(f"Unknown init strategy '{init}'")
    return backbone


def _load_batches(
    num_images: int, image_size: int, device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Real RGB+NIR batches from the training dataset when available,
    falling back to synthetic Gaussian noise otherwise (e.g. CI without the
    dataset checked out). Real images are strongly preferred: per design.md
    H-A, "real batches" give a meaningful across-image std band — synthetic
    Gaussian noise images are nearly statistically identical to each other,
    making the 2x-std CONFIRM band artificially, uninformatively narrow.
    """
    try:
        from src.training.dataset import YOLODataset
        from src.training.nir_registration import DEFAULT_NIR_HOMOGRAPHY_PATH

        dataset = YOLODataset(
            rgb_dir="data/cache/mango/rgb",
            nir_dir="data/cache/mango/nir",
            labels_dir="data/annotations/yolo/labels",
            split="train",
            image_size=image_size,
            manifest_path="data/annotations/yolo/splits.json",
            nir_homography_path=DEFAULT_NIR_HOMOGRAPHY_PATH,
        )
        if len(dataset) >= num_images:
            rgb_batches, nir_batches = [], []
            for i in range(num_images):
                sample = dataset[i]
                rgb_batches.append(sample["rgb"].unsqueeze(0).to(device))
                nir_batches.append(sample["nir"].unsqueeze(0).to(device))
            print(f"Using {num_images} real images from data/cache/mango/rgb (split=train).\n")
            return rgb_batches, nir_batches
    except Exception as e:  # noqa: BLE001 - deliberately broad: any dataset-load failure falls back
        print(f"Could not load real dataset ({e!r}); falling back to synthetic Gaussian noise.\n")

    rgb_batches = [
        torch.randn(1, 3, image_size, image_size, device=device) for _ in range(num_images)
    ]
    nir_batches = [
        torch.randn(1, 1, image_size, image_size, device=device) for _ in range(num_images)
    ]
    return rgb_batches, nir_batches


def _per_stage_stats(features: list[torch.Tensor]) -> list[dict]:
    return [
        {"mean": f.mean().item(), "std": f.std().item()}
        for f in features
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="H-A: audit stem-init strategies' activation statistics.")
    parser.add_argument("--init", nargs="+", default=["inflation", "zero", "copy"])
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    rgb_batches, nir_batches = _load_batches(args.num_images, args.image_size, device)
    fused_batches = [torch.cat([r, n], dim=1) for r, n in zip(rgb_batches, nir_batches)]

    # Reference: RGB-only pretrained ConvNeXt-Tiny stem+stages, run on the
    # same RGB channels used in every 4-channel probe.
    reference = EarlyFusionBackbone(pretrained=True, variant="tiny", in_channels=3).to(device)
    reference.eval()
    ref_stats_per_image = []
    with torch.no_grad():
        for rgb in rgb_batches:
            ref_stats_per_image.append(_per_stage_stats(reference(rgb)))

    num_stages = len(ref_stats_per_image[0])
    ref_mean_per_stage = [
        sum(img[s]["mean"] for img in ref_stats_per_image) / len(ref_stats_per_image)
        for s in range(num_stages)
    ]
    # Reference's OWN across-image std at each stage — the CONFIRM band width.
    ref_across_image_std = [
        torch.tensor([img[s]["mean"] for img in ref_stats_per_image]).std().item()
        for s in range(num_stages)
    ]

    results = {}
    for init in args.init:
        backbone = _build_backbone(init).to(device)
        backbone.eval()
        per_image_stats = []
        with torch.no_grad():
            for fused in fused_batches:
                per_image_stats.append(_per_stage_stats(backbone(fused)))

        stage_means = [
            sum(img[s]["mean"] for img in per_image_stats) / len(per_image_stats)
            for s in range(num_stages)
        ]
        deviations = [
            abs(stage_means[s] - ref_mean_per_stage[s]) for s in range(num_stages)
        ]
        within_band = [
            deviations[s] <= 2 * max(ref_across_image_std[s], 1e-8) for s in range(num_stages)
        ]
        results[init] = {
            "stage_means": stage_means,
            "deviation_from_reference": deviations,
            "within_2x_reference_std_band": within_band,
        }
        print(f"[{init}] stage means: {[f'{m:.4f}' for m in stage_means]}")
        print(f"[{init}] |deviation| from reference: {[f'{d:.4f}' for d in deviations]}")
        print(f"[{init}] within 2x reference across-image std: {within_band}\n")

    print(f"Reference (RGB-only pretrained) stage means: {[f'{m:.4f}' for m in ref_mean_per_stage]}")
    print(f"Reference across-image std per stage: {[f'{s:.4f}' for s in ref_across_image_std]}\n")

    if "inflation" in results and "copy" in results:
        inflation_dev = sum(results["inflation"]["deviation_from_reference"])
        copy_dev = sum(results["copy"]["deviation_from_reference"])
        print(
            f"Sum |deviation| — inflation: {inflation_dev:.4f}, naive copy: {copy_dev:.4f} "
            f"({'inflation closer' if inflation_dev < copy_dev else 'naive copy closer'})"
        )
        if all(results["inflation"]["within_2x_reference_std_band"]) and inflation_dev < copy_dev:
            print("H-A: CONFIRM — inflation is within band and closer than naive copy.")
        else:
            print("H-A: REFUTE — falling back to zero-init is recommended; see design.md H-A.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
