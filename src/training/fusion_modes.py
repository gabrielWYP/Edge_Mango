"""Single source of truth for `fusion_mode` and its checkpoint schema tag.

`MasterModel` supports two mutually exclusive multimodal fusion
architectures, selected by `TrainingConfig.fusion_mode`:

- ``"early"`` (default, fusion-redesign): RGB and NIR are stacked into one
  4-channel tensor before a single ConvNeXt stem; one backbone stream, one
  FPN (`EarlyFusionBackbone` + `FPNNeck`).
- ``"cross_attention"``: the pre-redesign two-stream architecture — two
  modality-specific stems over shared ConvNeXt stages, per-stage
  cross-attention (RGB queries NIR), and two FPNs fused per level
  (`DualConvNeXtBackbone` + `CrossModalFusion` + `DualFPN`).

Everything upstream of the shared detection head is disjoint between the
two: `backbone.stem.*`/`backbone.stages.*`/`neck.fpn.*` on the early path vs
`backbone.rgb_stem.*`/`backbone.nir_stem.*`/`backbone.shared_stages.*`/
`fusion.*`/`neck.fpn_rgb.*`/`neck.fpn_nir.*`/`neck.fusion_convs.*` on the
cross-attention path. Loading one into the other is therefore always an
error, and `arch_version` is what turns it into a single actionable sentence
instead of a 200-line missing/unexpected-key dump:

    arch_version 1 -> cross_attention   (two-stream)
    arch_version 2 -> early             (single-stream, early fusion)

`None` is accepted only on the cross-attention path: checkpoints saved
before the `arch_version` tag existed (e.g. `checkpoints/mastermodel_mango`)
predate the redesign entirely, so an untagged checkpoint is a two-stream one
by construction. It is deliberately NOT accepted for `"early"` — an untagged
checkpoint can never be an early-fusion one.
"""

from __future__ import annotations

from collections.abc import Mapping

FUSION_MODE_EARLY = "early"
FUSION_MODE_CROSS_ATTENTION = "cross_attention"

FUSION_MODES: tuple[str, ...] = (FUSION_MODE_EARLY, FUSION_MODE_CROSS_ATTENTION)

DEFAULT_FUSION_MODE: str = FUSION_MODE_EARLY

# Checkpoint state_dict schema version per fusion mode (fusion-redesign D-C).
ARCH_VERSION_BY_FUSION_MODE: dict[str, int] = {
    FUSION_MODE_CROSS_ATTENTION: 1,
    FUSION_MODE_EARLY: 2,
}

# What `checkpoint.get("arch_version")` may be for a checkpoint of each mode.
# Only the cross-attention path accepts the untagged `None` (see module
# docstring) — a checkpoint written by the current Trainer always carries an
# explicit int.
ACCEPTED_ARCH_VERSIONS_BY_FUSION_MODE: dict[str, frozenset] = {
    FUSION_MODE_CROSS_ATTENTION: frozenset({1, None}),
    FUSION_MODE_EARLY: frozenset({2}),
}


def validate_fusion_mode(fusion_mode: str) -> None:
    """Raise `ValueError` unless `fusion_mode` is one of `FUSION_MODES`.

    Loud on anything else: a typo like `"cross-attention"` must not fall
    back to the default and silently train the wrong architecture.
    """
    if fusion_mode not in FUSION_MODES:
        raise ValueError(
            f"Invalid fusion_mode {fusion_mode!r}. Must be one of "
            f"{list(FUSION_MODES)}."
        )


def arch_version_for_mode(fusion_mode: str) -> int:
    """Return the `arch_version` a checkpoint of `fusion_mode` is tagged with."""
    validate_fusion_mode(fusion_mode)
    return ARCH_VERSION_BY_FUSION_MODE[fusion_mode]


def accepted_arch_versions(fusion_mode: str) -> frozenset:
    """Return the `arch_version` values loadable on `fusion_mode`'s path."""
    validate_fusion_mode(fusion_mode)
    return ACCEPTED_ARCH_VERSIONS_BY_FUSION_MODE[fusion_mode]


def fusion_mode_for_arch_version(arch_version: int | None) -> str | None:
    """Return the fusion mode an `arch_version` belongs to, or None.

    Used only to make a mismatch error message name the mode the checkpoint
    actually came from; returns `None` for an unrecognised tag rather than
    guessing.
    """
    if arch_version is None:
        return FUSION_MODE_CROSS_ATTENTION
    for mode, version in ARCH_VERSION_BY_FUSION_MODE.items():
        if version == arch_version:
            return mode
    return None


def validate_fusion_pos_encoding(fusion_pos_encoding: bool, fusion_mode: str) -> None:
    """Raise `ValueError` unless `fusion_pos_encoding` is a bool that
    `fusion_mode` can honour.

    Strictly a bool: a YAML string like `"false"` is truthy and would
    silently enable the encoding. True is rejected on the early path, which
    has no cross-attention to encode position into.
    """
    if not isinstance(fusion_pos_encoding, bool):
        raise ValueError(
            f"fusion_pos_encoding must be a bool, got {fusion_pos_encoding!r} "
            f"({type(fusion_pos_encoding).__name__})."
        )
    if fusion_pos_encoding and fusion_mode != FUSION_MODE_CROSS_ATTENTION:
        raise ValueError(
            f"fusion_pos_encoding=True requires fusion_mode="
            f"'{FUSION_MODE_CROSS_ATTENTION}' (the only mode with "
            f"cross-attention), got fusion_mode={fusion_mode!r}."
        )


def fusion_pos_encoding_from_checkpoint(checkpoint: dict) -> bool:
    """Return the `fusion_pos_encoding` a checkpoint was trained with.

    The encoding adds no state_dict key, so a checkpoint trained with it
    loads strict into a model built without it (and vice versa) and would
    silently run the wrong forward. The Trainer records its whole config
    under `"config"`; that recorded value is the only trustworthy source.
    Checkpoints written before the field existed (including the untagged
    `checkpoints/mastermodel_mango`) record none and were trained without
    it, so absence resolves to False.
    """
    config_dict = checkpoint.get("config") or {}
    return bool(config_dict.get("fusion_pos_encoding", False))


# --------------------------------------------------------------------------
# Attention ablation flags (cross_attention only)
# --------------------------------------------------------------------------
#
# Two opt-in switches carve the two-stream teacher into the three arms of the
# attention ablation, all `arch_version` 1:
#
#   A  fusion_attention=True,  use_nir=True   the default two-stream model
#   B  fusion_attention=False, use_nir=True   no `CrossModalFusion`: F_i = RGB_i,
#                                             NIR reaches the head through the
#                                             `DualFPN` concat only
#   C  fusion_attention=False, use_nir=False  RGB-only two-stream skeleton: no
#                                             `nir_stem`, no NIR pass, no
#                                             `fpn_nir`; `nir` is accepted and
#                                             ignored
#
# B and C each drop state_dict keys (`fusion.*`; plus `backbone.nir_stem.*` and
# `neck.fpn_nir.*`, and the fusion convs shrink from 512 to 256 inputs), so a
# checkpoint of one arm cannot be loaded into another. Loaders rebuild the arm
# from the checkpoint's recorded config (`*_from_checkpoint`, absent -> True,
# which every checkpoint predating the flags was trained with) and
# `check_fusion_ablation_compatible` turns a residual mismatch into one
# sentence.

# State_dict prefixes that exist only in arm A / only when NIR is streamed.
_FUSION_ATTENTION_PREFIX = "fusion."
_NIR_STREAM_PREFIX = "backbone.nir_stem."
# Present in every cross-attention state_dict, whatever the arm.
_CROSS_ATTENTION_PREFIX = "backbone.rgb_stem."


def validate_fusion_ablation(
    fusion_attention: bool,
    use_nir: bool,
    fusion_mode: str,
    fusion_pos_encoding: bool = False,
) -> None:
    """Raise `ValueError` unless the ablation flags form a buildable arm.

    Strictly bools, for the same reason as `validate_fusion_pos_encoding`.
    Both flags only exist on the cross-attention path: the early path has no
    attention module and fuses NIR at the input, so turning either off there
    would be silently ignored. `use_nir=False` removes the NIR stream the
    attention attends over, so it requires `fusion_attention=False`; and
    `fusion_pos_encoding=True` without the attention would encode position
    into a module that does not exist.
    """
    for name, value in (("fusion_attention", fusion_attention), ("use_nir", use_nir)):
        if not isinstance(value, bool):
            raise ValueError(
                f"{name} must be a bool, got {value!r} ({type(value).__name__})."
            )
    if fusion_mode != FUSION_MODE_CROSS_ATTENTION and not (fusion_attention and use_nir):
        raise ValueError(
            f"fusion_attention={fusion_attention} / use_nir={use_nir} are "
            f"ablations of fusion_mode='{FUSION_MODE_CROSS_ATTENTION}', got "
            f"fusion_mode={fusion_mode!r}: the early path has no attention "
            "module and fuses NIR at the input, so the flags would be silently "
            "ignored. Use in_channels=3 for its RGB-only arm."
        )
    if not use_nir and fusion_attention:
        raise ValueError(
            "use_nir=False requires fusion_attention=False: the cross-modal "
            "attention attends RGB queries over NIR keys, which no longer exist."
        )
    if fusion_pos_encoding and not fusion_attention:
        raise ValueError(
            "fusion_pos_encoding=True requires fusion_attention=True: with no "
            "cross-modal attention there is no query/key to encode position "
            "into, and the flag would be silently ignored."
        )


def fusion_attention_from_checkpoint(checkpoint: dict) -> bool:
    """Return the `fusion_attention` a checkpoint was trained with.

    Read from the Trainer's recorded `"config"`, like
    `fusion_pos_encoding_from_checkpoint`. Checkpoints written before the
    field existed record none and were trained with the attention, so absence
    resolves to True.
    """
    config_dict = checkpoint.get("config") or {}
    return bool(config_dict.get("fusion_attention", True))


def use_nir_from_checkpoint(checkpoint: dict) -> bool:
    """Return the `use_nir` a checkpoint was trained with (absent -> True)."""
    config_dict = checkpoint.get("config") or {}
    return bool(config_dict.get("use_nir", True))


def check_fusion_ablation_compatible(
    model,
    state_dict: Mapping[str, object],
    source: str = "checkpoint",
) -> None:
    """Raise unless `state_dict` and `model` (a `MasterModel`) are the same arm.

    Compares the presence of the `fusion.*` (attention) and
    `backbone.nir_stem.*` (NIR stream) keys, which tell arms A, B and C apart.
    Only cross-attention state_dicts on both sides are compared: anything else
    (an early-fusion model or checkpoint) is `arch_version`'s job, and is left
    to strict loading.
    """
    def has(sd: Mapping[str, object], prefix: str) -> bool:
        return any(key.startswith(prefix) for key in sd)

    model_state_dict = model.state_dict()
    if not (has(model_state_dict, _CROSS_ATTENTION_PREFIX) and has(state_dict, _CROSS_ATTENTION_PREFIX)):
        return
    mismatches = []
    for name, prefix in (
        ("fusion_attention", _FUSION_ATTENTION_PREFIX),
        ("use_nir", _NIR_STREAM_PREFIX),
    ):
        checkpoint_value, model_value = has(state_dict, prefix), has(model_state_dict, prefix)
        if checkpoint_value != model_value:
            mismatches.append((name, checkpoint_value, model_value))
    if not mismatches:
        return
    checkpoint_flags = ", ".join(f"{name}={ckpt}" for name, ckpt, _ in mismatches)
    model_flags = ", ".join(f"{name}={model}" for name, _, model in mismatches)
    raise ValueError(
        f"{source} has {checkpoint_flags} weights but the model was built with "
        f"{model_flags}. The attention-ablation arms have different state_dict "
        f"schemas - rebuild the model with {checkpoint_flags} (the config the "
        "checkpoint was trained with, or --override ...)."
    )
