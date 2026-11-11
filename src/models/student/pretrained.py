"""Copy the official YOLOv8n COCO weights into a student, from a plain state_dict.

The file holds the tensors of the official checkpoint under their own names, `model.<i>...`, so
loading it needs no ultralytics package. Its head predicts 80 classes and 64 box values, so only
what has the same shape is copied. The backbone must match in full: both variants have it.
A yolov8n-seg file also fills the `nano` student's Proto and mask-coefficient layers (`task="segment"`). A
student with a NIR input channel starts that channel from the mean of the three RGB filters.
"""

from __future__ import annotations

from pathlib import Path

import torch

_BACKBONE = {
    "stem.0": 0, "stem.1": 1, "stage1": 2, "stage2.0": 3, "stage2.1": 4,
    "stage3.0": 5, "stage3.1": 6, "stage4.0": 7, "stage4.1": 8, "sppf": 9,
}
_FIRST_FILTER = "backbone.stem.0.conv.weight"
_NANO_NECK = {
    "td_c2f_4": 12, "td_c2f_3": 15, "bu_conv_3": 16, "bu_c2f_4": 18, "bu_conv_4": 19, "bu_c2f_5": 21,
}


def official_key(key: str, variant: str) -> str | None:
    """Name of a student tensor in the official checkpoint, or None when it has no counterpart."""
    section, _, rest = key.partition(".")
    if section == "backbone":
        for prefix, index in _BACKBONE.items():
            if rest.startswith(prefix + "."):
                return f"model.{index}.{rest[len(prefix) + 1:]}"
    if variant != "nano":
        return None
    if section == "neck":
        block, _, tail = rest.partition(".")
        return f"model.{_NANO_NECK[block]}.{tail}"
    if section == "head" and rest.startswith("proto."):
        # yolov8n-seg: `Proto` keeps its layer names (cv1, upsample, cv2, cv3).
        return f"model.22.{rest}"
    parts = key.split(".")  # head.heads.<level>.<cls_stem|reg_stem|mask_stem>.<layer>.<tail>
    if section == "head" and parts[3] in ("cls_stem", "reg_stem"):
        branch = "cv3" if parts[3] == "cls_stem" else "cv2"
        return f"model.22.{branch}.{parts[2]}.{parts[4]}.{'.'.join(parts[5:])}"
    if section == "head" and parts[3] == "mask_stem":
        # The official `cv4` branch: two Conv blocks, then a plain Conv2d at index 2.
        return f"model.22.cv4.{parts[2]}.{parts[4]}.{'.'.join(parts[5:])}"
    if section == "head" and parts[3] == "mask_pred":
        return f"model.22.cv4.{parts[2]}.2.{'.'.join(parts[4:])}"
    return None


def load_yolov8n_weights(model: torch.nn.Module, path: str | Path) -> dict:
    """Copy every tensor with a counterpart of the same shape; a backbone mismatch is an error."""
    official = torch.load(path, map_location="cpu", weights_only=True)
    state = model.state_dict()
    loaded, skipped = {}, []
    for key, value in state.items():
        source = official.get(official_key(key, model.variant))
        if key == _FIRST_FILTER and source is not None and value.shape[1] == source.shape[1] + 1:
            source = torch.cat([source, source.mean(dim=1, keepdim=True)], dim=1)
        if source is not None and source.shape == value.shape:
            loaded[key] = source.to(value.dtype)
        else:
            skipped.append(key)
    wrong = [key for key in skipped if key.startswith("backbone.")]
    if wrong:
        raise ValueError(f"{path}: the backbone does not match, first at {wrong[0]}")
    model.load_state_dict(loaded, strict=False)
    return {"loaded": len(loaded), "skipped": skipped}
