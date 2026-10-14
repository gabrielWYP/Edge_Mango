#!/usr/bin/env python3
"""Visualize per-stage cross-modal attention maps for MasterModel (RGB<-NIR).

`fusion_mode="cross_attention"` only: `src/models/master/fusion.py`'s
`StageAttentionFusion` runs real `nn.MultiheadAttention` at each of the 4
backbone stages (RGB queries NIR) and already computes a spatial attention
map per stage (`return_attention=True`) — this script is the first consumer
of that plumbing. For each sample it shows where the RGB stream is pulling
NIR context from, at each stride, next to the ground-truth damage boxes.

Example usage:
    ./myLinuxVenv/bin/python scripts/visualize_attention.py \\
        --checkpoint checkpoints/twostream/seed42/best_model.pt \\
        --config configs/experiment/twostream.yaml \\
        --split val --num-samples 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visualize_damage_predictions import (
    CLASS_COLORS,
    discover_image_pairs,
    gt_to_letterbox_pixels,
    load_model,
    load_yolo_labels,
    preprocess_nir,
    preprocess_rgb,
)
from src.training.config import TrainingConfig
from src.training.dataset import letterbox
from src.training.nir_registration import load_homography, register_nir_to_rgb
from src.training.fusion_modes import FUSION_MODE_CROSS_ATTENTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STAGE_STRIDES = (4, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize per-stage RGB<-NIR cross-attention maps."
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--rgb-dir", type=str, default=None)
    parser.add_argument("--nir-dir", type=str, default=None)
    parser.add_argument("--labels-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--damage-only", action=argparse.BooleanOptionalAction, default=True,
        help="Only visualize samples that have >=1 damage (class 1) GT box (default: True).",
    )
    parser.add_argument("--output-dir", type=str, default="visualizations/attention")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def overlay_heatmap(base_bgr: np.ndarray, attn_map: np.ndarray) -> np.ndarray:
    """Alpha-blend a [0,1] attention map (H, W) onto a BGR image as a JET heatmap."""
    heat_u8 = np.clip(attn_map * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(base_bgr, 0.5, heat_color, 0.5, 0)


def draw_gt_boxes(image: np.ndarray, boxes_cxcywh: np.ndarray, labels: np.ndarray) -> np.ndarray:
    for box, cls_id in zip(boxes_cxcywh, labels):
        color = CLASS_COLORS.get(int(cls_id), (255, 255, 255))
        cx, cy, w, h = box
        x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = TrainingConfig.from_yaml(args.config)
    if config.fusion_mode != FUSION_MODE_CROSS_ATTENTION:
        raise ValueError(
            f"--config has fusion_mode='{config.fusion_mode}'; attention maps only "
            f"exist for fusion_mode='{FUSION_MODE_CROSS_ATTENTION}' (early fusion has "
            "no cross-modal attention module)."
        )

    rgb_dir = Path(args.rgb_dir or config.rgb_dir)
    nir_dir = Path(args.nir_dir or config.nir_dir)
    labels_dir = Path(args.labels_dir or config.labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _ = load_model("master", args.checkpoint, config, device)

    pairs = discover_image_pairs(rgb_dir, nir_dir, labels_dir, args.split, "master")
    logger.info("Found %d RGB/NIR pairs with labels in split=%s", len(pairs), args.split)

    selected = []
    for pair in pairs:
        gt_bboxes, gt_labels = load_yolo_labels(pair["label_path"])
        if args.damage_only and not (gt_labels == 1).any():
            continue
        pair["gt_bboxes"], pair["gt_labels"] = gt_bboxes, gt_labels
        selected.append(pair)
        if len(selected) >= args.num_samples:
            break

    if not selected:
        logger.warning("No samples matched (damage_only=%s) — nothing to visualize.", args.damage_only)
        return
    logger.info("Visualizing %d samples -> %s", len(selected), output_dir)

    # Same NIR registration the training dataset applies (None = disabled).
    H_rgb_to_nir = (
        load_homography(config.nir_homography_path) if config.nir_homography_path else None
    )

    for pair in selected:
        rgb_bgr = cv2.imread(str(pair["rgb_path"]))
        rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        nir_gray = cv2.imread(str(pair["nir_path"]), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = rgb_rgb.shape[:2]

        if H_rgb_to_nir is not None:
            nir_gray = register_nir_to_rgb(
                nir_gray, H_rgb_to_nir, (orig_w, orig_h), int(config.nir_mean * 255)
            )

        rgb_lb_display, scale, pad_x, pad_y = letterbox(rgb_bgr, config.image_size, config.letterbox_value)

        rgb_tensor, _, _, _ = preprocess_rgb(rgb_rgb, config.image_size, config.letterbox_value)
        nir_tensor = preprocess_nir(nir_gray, config.image_size, config.nir_mean, config.nir_std)

        with torch.no_grad():
            rgb_feats, nir_feats = model.backbone(rgb_tensor.to(device), nir_tensor.to(device))
            _, _, attn_maps = model.fusion(rgb_feats, nir_feats, return_attention=True)

        gt_lb = gt_to_letterbox_pixels(pair["gt_bboxes"], orig_w, orig_h, scale, pad_x, pad_y)

        panels = []
        for stride, attn_map in zip(STAGE_STRIDES, attn_maps):
            attn_np = attn_map[0].detach().cpu().float().numpy()  # (H_stage, W_stage)
            attn_full = cv2.resize(attn_np, (config.image_size, config.image_size), interpolation=cv2.INTER_LINEAR)
            panel = overlay_heatmap(rgb_lb_display.copy(), attn_full)
            panel = draw_gt_boxes(panel, gt_lb, pair["gt_labels"])
            cv2.putText(
                panel, f"stride {stride}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            panels.append(panel)

        top = np.hstack(panels[:2])
        bottom = np.hstack(panels[2:])
        grid = np.vstack([top, bottom])

        out_path = output_dir / f"{pair['stem']}_attention.png"
        cv2.imwrite(str(out_path), grid)
        logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
