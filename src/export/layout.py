"""Exported output layout: flat per-anchor tensors in a fixed anchor order.

The exported student returns two tensors instead of per-level feature maps:

    boxes_raw:  (1, N, 4)            [dx, dy, log_w, log_h] per anchor
    cls_logits: (1, N, num_classes)  raw class logits per anchor

with N = sum over levels of (image_size / stride)^2 (8400 at 640 with
strides 8/16/32). Anchor order is level-major (finest stride first), then
row-major within a level (y outer, x inner) — the same order as
`src/training/loss.py::_generate_anchors`. Anchor `i` in level `l` with
`offset_l <= i < offset_l + grid_h * grid_w` sits at
`gy = (i - offset_l) // grid_w`, `gx = (i - offset_l) % grid_w`.

Why two outputs rather than one `(1, N, 4 + nc)` tensor: box regressions are
pixel offsets / log-sizes while class channels are logits, so they have very
different ranges. Keeping them apart lets a later INT8 export give each its
own quantization scale; concatenating them would force one shared scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

BOXES_OUTPUT_NAME = "boxes_raw"
CLASS_OUTPUT_NAME = "cls_logits"
BOX_CHANNELS: tuple[str, ...] = ("dx", "dy", "log_w", "log_h")

# A segment student (YOLOv8-seg) adds two outputs, in this order after the two above:
#   mask_coefs: (1, N, nm)             mask coefficients per anchor, same anchor order as the boxes
#   protos:     (1, nm, H/4, W/4)      prototype masks of the image (NCHW, as the model produces them)
# A mask is `sigmoid(coefs[anchor] . protos)` cropped to its box; see `reference.decode_exported_segmentation`.
MASK_COEFS_OUTPUT_NAME = "mask_coefs"
PROTOS_OUTPUT_NAME = "protos"
DETECT_OUTPUT_NAMES: tuple[str, ...] = (BOXES_OUTPUT_NAME, CLASS_OUTPUT_NAME)
SEGMENT_OUTPUT_NAMES: tuple[str, ...] = (*DETECT_OUTPUT_NAMES, MASK_COEFS_OUTPUT_NAME, PROTOS_OUTPUT_NAME)
PROTO_STRIDE = 4  # the prototypes are at 1/4 of the input resolution


@dataclass(frozen=True)
class LevelLayout:
    """One pyramid level's slice of the flat anchor axis."""

    stride: int
    grid_h: int
    grid_w: int
    offset: int

    @property
    def count(self) -> int:
        return self.grid_h * self.grid_w


def anchor_layout(image_size: int, strides: Sequence[int]) -> list[LevelLayout]:
    """Return the per-level slices of the flat anchor axis, finest level first."""
    levels = []
    offset = 0
    for stride in strides:
        if image_size % stride != 0:
            raise ValueError(f"image_size={image_size} is not divisible by stride={stride}.")
        grid = image_size // stride
        levels.append(LevelLayout(stride=stride, grid_h=grid, grid_w=grid, offset=offset))
        offset += grid * grid
    return levels


def num_anchors(image_size: int, strides: Sequence[int]) -> int:
    return sum(level.count for level in anchor_layout(image_size, strides))


def flatten_levels(levels: Sequence[torch.Tensor]) -> torch.Tensor:
    """(B, C, H_i, W_i) per level -> (B, sum_i H_i*W_i, C) in anchor order."""
    return torch.cat([t.flatten(2).transpose(1, 2) for t in levels], dim=1)


def unflatten_levels(
    flat: torch.Tensor,
    image_size: int,
    strides: Sequence[int],
) -> list[torch.Tensor]:
    """Inverse of `flatten_levels`: (B, N, C) -> [(B, C, H_i, W_i), ...]."""
    layout = anchor_layout(image_size, strides)
    expected = sum(level.count for level in layout)
    if flat.shape[1] != expected:
        raise ValueError(
            f"Flat tensor has {flat.shape[1]} anchors but image_size={image_size} "
            f"with strides={list(strides)} implies {expected}."
        )
    batch, _, channels = flat.shape
    return [
        flat[:, level.offset: level.offset + level.count]
        .transpose(1, 2)
        .reshape(batch, channels, level.grid_h, level.grid_w)
        for level in layout
    ]
