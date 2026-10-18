"""Single source of truth for the neck normalization layer (`neck_norm`).

The only BatchNorm layers downstream of the ConvNeXt backbone live in the
`DualFPN.fusion_convs` of `fusion_mode="cross_attention"`. The backbone and
the cross-modal fusion use LayerNorm; the detection head and the early-fusion
`FPNNeck` have no normalization layer at all. So "the neck norm" is exactly
those fusion convs, and `neck_norm` is a no-op by construction for
`fusion_mode="early"` — which is why `group` is rejected there instead of
being silently ignored (same policy as `fusion_pos_encoding`).

- ``"batch"`` (default): `nn.BatchNorm2d`. Every existing checkpoint was
  trained with it and keeps loading.
- ``"group"``: `nn.GroupNorm`. No batch statistics, so the normalization used
  in train and eval mode is identical and one image's output does not depend
  on the other images in the batch. On the GTX 1660S the physical batch is 2
  (effective 8 via gradient accumulation, which does not fix BatchNorm: its
  statistics still come from 2 images), so the running statistics are noisy
  and drift from epoch to epoch.

GroupNorm group count: `resolve_gn_groups(channels, gn_groups)` is the
largest divisor of `channels` that is <= `gn_groups` (256 channels, 32 groups
-> 32; 48 channels, 32 groups -> 24; a prime above the cap falls back to one
group, i.e. LayerNorm over C, H, W).

The two choices have different state_dict schemas (BatchNorm carries
`running_mean`/`running_var`/`num_batches_tracked`, GroupNorm does not), so a
checkpoint of one cannot be loaded into the other. The Trainer records the
choice under the checkpoint's `"config"`; `neck_norm_from_checkpoint` reads it
back (absent -> BatchNorm, what every earlier checkpoint was trained with) and
`check_norm_compatible` turns a residual mismatch into one actionable
sentence.

Lives under `src/training/` next to `fusion_modes.py` and `strides.py`, which
the models already import from: `TrainingConfig` validates with it, and a
`src.models` import here would be circular.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from .fusion_modes import FUSION_MODE_CROSS_ATTENTION

NORM_BATCH = "batch"
NORM_GROUP = "group"
NECK_NORMS: tuple[str, ...] = (NORM_BATCH, NORM_GROUP)
DEFAULT_NECK_NORM = NORM_BATCH
DEFAULT_NECK_GN_GROUPS = 32

# State_dict prefixes whose norm layers `neck_norm` governs.
_NECK_PREFIXES: tuple[str, ...] = ("neck.", "head.")


def validate_neck_norm(neck_norm: str, gn_groups: int) -> None:
    """Raise `ValueError` on an unknown norm or a non-positive group count."""
    if neck_norm not in NECK_NORMS:
        raise ValueError(
            f"Invalid neck_norm {neck_norm!r}. Must be one of: {list(NECK_NORMS)}."
        )
    if isinstance(gn_groups, bool) or not isinstance(gn_groups, int) or gn_groups < 1:
        raise ValueError(f"neck_gn_groups must be a positive int, got {gn_groups!r}.")


def validate_neck_norm_for_fusion_mode(neck_norm: str, fusion_mode: str) -> None:
    """Raise unless `fusion_mode` has a neck norm layer for `neck_norm` to swap.

    Only the cross-attention neck (`DualFPN`) has one. Accepting
    ``neck_norm="group"`` on the early path would train a BatchNorm-free
    network that is identical to the default and let the user believe the
    option was applied.
    """
    if neck_norm != NORM_BATCH and fusion_mode != FUSION_MODE_CROSS_ATTENTION:
        raise ValueError(
            f"neck_norm={neck_norm!r} requires fusion_mode="
            f"'{FUSION_MODE_CROSS_ATTENTION}' (the only neck with a "
            f"normalization layer), got fusion_mode={fusion_mode!r}: the "
            "early-fusion FPNNeck has none, so the option would be silently "
            "ignored."
        )


def resolve_gn_groups(channels: int, gn_groups: int) -> int:
    """Largest divisor of `channels` that is <= `gn_groups` (at least 1)."""
    for groups in range(min(gn_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def make_norm(
    neck_norm: str, channels: int, gn_groups: int = DEFAULT_NECK_GN_GROUPS
) -> nn.Module:
    """Build the 2D norm layer selected by `neck_norm` over `channels` channels."""
    validate_neck_norm(neck_norm, gn_groups)
    if neck_norm == NORM_GROUP:
        return nn.GroupNorm(resolve_gn_groups(channels, gn_groups), channels)
    return nn.BatchNorm2d(channels)


def neck_norm_from_checkpoint(checkpoint: Mapping) -> tuple[str, int]:
    """Return `(neck_norm, neck_gn_groups)` a checkpoint was trained with.

    The norm is part of the state_dict schema but only discoverable from the
    weights by inference; the Trainer records its whole config under
    `"config"`, which is the trustworthy source. Checkpoints written before
    the field existed record nothing and were trained with BatchNorm, so
    absence resolves to the default.
    """
    config_dict = checkpoint.get("config") or {}
    return (
        config_dict.get("neck_norm", DEFAULT_NECK_NORM),
        config_dict.get("neck_gn_groups", DEFAULT_NECK_GN_GROUPS),
    )


def _has_batchnorm_stats(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return any(
        key.startswith(_NECK_PREFIXES) and key.endswith(".running_mean")
        for key in state_dict
    )


def check_norm_compatible(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    source: str = "checkpoint",
) -> None:
    """Raise unless `state_dict`'s neck norm schema matches `model`'s.

    BatchNorm is detectable by its running-statistics buffers; their presence
    in exactly one of the two means a BatchNorm checkpoint is being loaded
    into a GroupNorm model or vice versa. Models with no neck norm at all
    (the early-fusion path) always pass.
    """
    model_bn = _has_batchnorm_stats(model.state_dict())
    checkpoint_bn = _has_batchnorm_stats(state_dict)
    if model_bn == checkpoint_bn:
        return
    checkpoint_kind, model_kind = (
        (NORM_BATCH, NORM_GROUP) if checkpoint_bn else (NORM_GROUP, NORM_BATCH)
    )
    raise ValueError(
        f"{source} has neck_norm={checkpoint_kind!r} weights but the model was "
        f"built with neck_norm={model_kind!r}. BatchNorm and GroupNorm "
        "checkpoints are not interchangeable - rebuild the model with "
        f"neck_norm={checkpoint_kind!r} (the config the checkpoint was trained "
        "with, or --override neck_norm=...)."
    )
