"""
ConvNeXt backbones for the two `MasterModel` fusion modes.

Supports ConvNeXt-Tiny and ConvNeXt-Small variants (both have identical
stage channels [96, 192, 384, 768], differing only in block depth).

Two backbones live here, selected by `TrainingConfig.fusion_mode`
(`src/training/fusion_modes.py`):

- `EarlyFusionBackbone` — `fusion_mode="early"` (default). One stem over the
  stacked 4-channel RGB+NIR tensor, one forward pass through the stages.
- `DualConvNeXtBackbone` — `fusion_mode="cross_attention"`. Two
  modality-specific stems (3ch RGB, 1ch NIR) over *shared* ConvNeXt stages,
  two forward passes; returns one feature list per modality for
  `CrossModalFusion`.

They share no state_dict key (`backbone.stem.*`/`backbone.stages.*` vs
`backbone.rgb_stem.*`/`backbone.nir_stem.*`/`backbone.shared_stages.*`),
which is part of why the checkpoint `arch_version` tag exists.

fusion-redesign (design.md D-A/D-B) originally deleted the dual-stream
design in favour of `EarlyFusionBackbone`: RGB and NIR are stacked into
one 4-channel tensor before the stem, so there is exactly one stem and one
forward pass through the stages. The ImageNet-pretrained 3-channel stem is
loaded into the 4-channel stem via mean-preserving inflation (D-1/D-B):

    W_new[:, :3] = W_imagenet * 3/4
    W_new[:, 3]  = mean(W_imagenet, dim=1) * 3/4

Not zero-init: the measured RGB/NIR cue is textural, so the NIR channel
should start with ImageNet edge/texture priors rather than learn them from
scratch on ~150 images. Not a naive copy: adding a 4th contributing channel
without rescaling raises the stem pre-activation by ~4/3 and perturbs the
statistics the pretrained LayerNorm and stage 1 expect; the 3/4 factor keeps
the input scale mean-preserving.

`in_channels=3` is also supported (verbatim, unscaled ImageNet copy) as the
RGB-only control arm for hypothesis H-D.
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_small,
    convnext_tiny,
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
)

# Supported backbone variants — both have identical stage channels
SUPPORTED_VARIANTS = {"tiny", "small"}

# Channels an ImageNet-pretrained stem can be loaded into:
# 4 = the early-fusion RGB+NIR input (mean-preserving inflation, D-1/D-B);
# 3 = the RGB-only H-D control arm (verbatim ImageNet copy, no rescale).
_SUPPORTED_STEM_IN_CHANNELS = {3, 4}


def _build_convnext_tiny_body(pretrained: bool = True) -> nn.ModuleList:
    """
    Extract the 4 stages of ConvNeXt-Tiny (without the stem and classifier).
    Returns them as a ModuleList so we can iterate per-stage.
    """
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    stages = nn.ModuleList([
        model.features[1],   # stage 1  → 96ch,  H/4  x W/4
        nn.Sequential(model.features[2], model.features[3]),   # down + stage 2 → 192ch, H/8  x W/8
        nn.Sequential(model.features[4], model.features[5]),   # down + stage 3 → 384ch, H/16 x W/16
        nn.Sequential(model.features[6], model.features[7]),   # down + stage 4 → 768ch, H/32 x W/32
    ])
    return stages


def _build_convnext_small_body(pretrained: bool = True) -> nn.ModuleList:
    """
    Extract the 4 stages of ConvNeXt-Small (without the stem and classifier).
    Returns them as a ModuleList so we can iterate per-stage.
    """
    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_small(weights=weights)
    # model.features is a Sequential with:
    #   [0] stem (Conv2d 4x4 s4 + LayerNorm)
    #   [1] stage 1
    #   [2] downsampling
    #   [3] stage 2
    #   [4] downsampling
    #   [5] stage 3
    #   [6] downsampling
    #   [7] stage 4
    stages = nn.ModuleList([
        model.features[1],   # stage 1  → 96ch,  H/4  x W/4
        nn.Sequential(model.features[2], model.features[3]),   # down + stage 2 → 192ch, H/8  x W/8
        nn.Sequential(model.features[4], model.features[5]),   # down + stage 3 → 384ch, H/16 x W/16
        nn.Sequential(model.features[6], model.features[7]),   # down + stage 4 → 768ch, H/32 x W/32
    ])
    return stages


def _build_stem(in_channels: int) -> nn.Sequential:
    """
    ConvNeXt stem: Conv2d (4x4, stride 4) + LayerNorm.
    in_channels=4 for the early-fused RGB+NIR input (default); 3 is also
    supported as the H-D RGB-only control arm, and 3/1 are the RGB/NIR stems
    of the cross-attention path's `DualConvNeXtBackbone`.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, 96, kernel_size=4, stride=4),
        # ConvNeXt uses LayerNorm on (N, H, W, C) — we use a wrapper
        _LayerNorm2d(96),
    )


class _LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors (N, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W) → (N, H, W, C) → norm → (N, C, H, W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class EarlyFusionBackbone(nn.Module):
    """
    Single-stream ConvNeXt backbone (Tiny or Small) over early-fused input.

    One stem consumes the stacked (RGB, NIR) tensor directly; the same
    ConvNeXt stages run once. This replaces the two-stem, two-forward-pass
    `DualConvNeXtBackbone` (fusion-redesign W1/D-A): NIR is no longer a
    parallel stream with its own near-empty stem — it is a genuine input
    channel to the same 27.8M-parameter trunk RGB uses.

    Args:
        pretrained (bool): Load ImageNet-1K weights for the stages and
            inflate the stem from the pretrained 3-channel stem (D-1/D-B).
        variant (str): Backbone variant — "tiny" (28M params) or "small" (50M params).
            Both produce identical stage channels [96, 192, 384, 768].
        in_channels (int): Stem input channels. 4 (default) for the
            early-fused RGB+NIR input; 3 for the RGB-only H-D control arm.

    Returns (forward):
        features: list of 4 tensors [S1, S2, S3, S4]

        Channel dims: [96, 192, 384, 768]
        Spatial dims (for 640x640 input): [160x160, 80x80, 40x40, 20x20]
    """

    # Both Tiny and Small have identical stage channel dimensions
    STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(self, pretrained: bool = True, variant: str = "tiny", in_channels: int = 4):
        super().__init__()

        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant '{variant}'. Choose from {SUPPORTED_VARIANTS}"
            )

        self.variant = variant
        self.in_channels = in_channels

        # --- Single stem over the stacked input ---
        self.stem = _build_stem(in_channels=in_channels)

        # --- Stages (single pass, no sharing across streams — there is only one stream) ---
        if variant == "tiny":
            self.stages = _build_convnext_tiny_body(pretrained=pretrained)
        else:
            self.stages = _build_convnext_small_body(pretrained=pretrained)

        if pretrained:
            self._load_pretrained_stem()
        else:
            self._init_stem()

    # ------------------------------------------------------------------
    # Introspection (shared backbone contract)
    # ------------------------------------------------------------------

    @property
    def stem_modules(self) -> dict[str, nn.Module]:
        """The backbone's input stems, keyed by their attribute name.

        Both backbones expose this so `Trainer` never has to branch on
        `fusion_mode` to build its discriminative parameter groups or its
        per-module gradient-norm instrumentation. A single-stream backbone
        has exactly one entry; `DualConvNeXtBackbone` has two. A property,
        not a submodule — it registers nothing and adds no state_dict key.
        """
        return {"stem": self.stem}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (N, in_channels, H, W) — stacked RGB+NIR (or RGB-only for the
                H-D control), pre-normalized per-channel by the caller.

        Returns:
            features: [S1, S2, S3, S4]

        Precision scoping (fusion-redesign D-I): ConvNeXt stage 4 was
        reported numerically fragile under CUDA autocast in the old
        dual-stream setup; that justification no longer applies to a
        single stream, so the guard is now scoped to fp16 ambient autocast
        only — a no-op under fp32 (nothing to disable) and, critically,
        a no-op under bf16 as well (bf16 carries fp32's exponent range,
        so the fp16 overflow mechanism the original comment describes does
        not apply; H-BF16 specifically needs bf16 to reach these stages
        rather than being silently forced back to fp32).
        """
        device_type = x.device.type
        fp16_ambient = (
            torch.is_autocast_enabled(device_type)
            and torch.get_autocast_dtype(device_type) == torch.float16
        )
        ctx = (
            torch.amp.autocast(device_type=device_type, enabled=False)
            if fp16_ambient
            else nullcontext()
        )
        with ctx:
            x_in = x.float() if fp16_ambient else x
            f = self.stem(x_in)  # (N, 96, H/4, W/4)

            features = []
            for stage in self.stages:
                f = stage(f)
                features.append(f)

        return features

    # ------------------------------------------------------------------
    # Weight initialization helpers
    # ------------------------------------------------------------------

    def _init_stem(self):
        """Initialize the stem with Kaiming normal (standard for conv layers)."""
        nn.init.kaiming_normal_(
            self.stem[0].weight, mode="fan_out", nonlinearity="relu"
        )
        if self.stem[0].bias is not None:
            nn.init.zeros_(self.stem[0].bias)

    def _load_pretrained_stem(self):
        """Load the ImageNet-pretrained 3-channel stem into `self.stem`.

        `in_channels == 4` (default): mean-preserving inflation (D-1/D-B) —
        `W_new[:, :3] = W_imagenet * 3/4`, `W_new[:, 3] = mean(W_imagenet,
        dim=1) * 3/4`. Not zero-init (NIR gets ImageNet edge/texture priors);
        not a naive copy (a 4th contributing channel without rescaling would
        raise the stem pre-activation by ~4/3 and perturb the pretrained
        LayerNorm/stage-1 statistics).

        `in_channels == 3`: verbatim ImageNet copy, unscaled — the H-D
        RGB-only control arm. Each arm gets its own best-available init;
        this is a control for "does the extra modality help", not an
        init-parity control (design.md D-B states this rejection openly).

        Bias and stem LayerNorm are copied verbatim in both cases.
        """
        if self.in_channels not in _SUPPORTED_STEM_IN_CHANNELS:
            raise ValueError(
                f"Pretrained stem inflation only supports in_channels in "
                f"{sorted(_SUPPORTED_STEM_IN_CHANNELS)}, got {self.in_channels}."
            )

        if self.variant == "tiny":
            pretrained_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            pretrained_model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)

        w = pretrained_model.features[0][0].weight  # (96, 3, 4, 4)

        if self.in_channels == 4:
            new_weight = torch.empty(96, 4, 4, 4, dtype=w.dtype)
            new_weight[:, :3] = w * 0.75
            new_weight[:, 3] = w.mean(dim=1) * 0.75
        else:  # in_channels == 3
            new_weight = w.clone()

        pretrained_stem_state = {
            "0.weight": new_weight,
            "0.bias": pretrained_model.features[0][0].bias,
            "1.norm.weight": pretrained_model.features[0][1].weight,
            "1.norm.bias": pretrained_model.features[0][1].bias,
        }
        self.stem.load_state_dict(pretrained_stem_state, strict=False)


class DualConvNeXtBackbone(nn.Module):
    """
    Dual-stream ConvNeXt backbone (Tiny or Small) — `fusion_mode="cross_attention"`.

    The two streams share the 4 ConvNeXt stages (shared weights act as
    regularization for the small dataset). Only the input stems differ
    to handle 3-channel RGB and 1-channel NIR independently.

    Restored from the pre-redesign architecture (deleted by fusion-redesign
    W1/D-A) with its module names and weight-init behaviour unchanged, so
    the two-stream checkpoints trained against it still load key-for-key.

    Args:
        pretrained (bool): Load ImageNet-1K weights for the shared stages.
            Both stems are then seeded from the pretrained 3-channel stem
            (RGB verbatim, NIR channel-averaged).
        variant (str): Backbone variant — "tiny" (28M params) or "small" (50M params).
            Both produce identical stage channels [96, 192, 384, 768].
        use_nir (bool): Build the NIR stream (default True). False is the
            RGB-only arm of the attention ablation
            (`src/training/fusion_modes.py`): no `nir_stem` (`self.nir_stem`
            is None and adds no state_dict key), no NIR pass through the
            shared stages, and `nir` is accepted but ignored by `forward`.

    Returns (forward):
        rgb_features: list of 4 tensors [S1, S2, S3, S4]
        nir_features: list of 4 tensors [S1, S2, S3, S4], or None when
            `use_nir=False`

        Channel dims: [96, 192, 384, 768]
        Spatial dims (for 640x640 input): [160x160, 80x80, 40x40, 20x20]
    """

    # Both Tiny and Small have identical stage channel dimensions
    STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(self, pretrained: bool = True, variant: str = "tiny", use_nir: bool = True):
        super().__init__()

        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant '{variant}'. Choose from {SUPPORTED_VARIANTS}"
            )

        self.variant = variant
        self.use_nir = use_nir

        # --- Stems (different per modality) ---
        self.rgb_stem = _build_stem(in_channels=3)
        # NIR is grayscale. A plain None attribute when absent: registers no
        # submodule, so the RGB-only arm has no `backbone.nir_stem.*` key.
        self.nir_stem = _build_stem(in_channels=1) if use_nir else None

        # --- Shared stages ---
        if variant == "tiny":
            self.shared_stages = _build_convnext_tiny_body(pretrained=pretrained)
        else:
            self.shared_stages = _build_convnext_small_body(pretrained=pretrained)

        if pretrained:
            self._load_pretrained_rgb_stem()
            if use_nir:
                self._load_pretrained_nir_stem()
        elif use_nir:
            self._init_nir_stem()

    # ------------------------------------------------------------------
    # Introspection (shared backbone contract)
    # ------------------------------------------------------------------

    @property
    def stages(self) -> nn.ModuleList:
        """Alias for `shared_stages` — the pretrained ConvNeXt stage stack.

        `Trainer._discriminative_param_groups` and the D-F gradient-norm
        instrumentation address "the pretrained stages" by this one name on
        both backbones. A plain property, so the registered submodule (and
        therefore every `backbone.shared_stages.*` state_dict key) is
        untouched.
        """
        return self.shared_stages

    @property
    def stem_modules(self) -> dict[str, nn.Module]:
        """The modality-specific stems that exist, keyed by attribute name —
        two by default, only `rgb_stem` when `use_nir=False`."""
        stems = {"rgb_stem": self.rgb_stem}
        if self.nir_stem is not None:
            stems["nir_stem"] = self.nir_stem
        return stems

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, rgb: torch.Tensor, nir: torch.Tensor | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """
        Args:
            rgb: (N, 3, H, W)
            nir: (N, 1, H, W)  — grayscale NIR image. Required when
                `use_nir=True`; ignored (may be None) when `use_nir=False`.

        Returns:
            rgb_features: [S1, S2, S3, S4]
            nir_features: [S1, S2, S3, S4], or None when `use_nir=False`

        Precision scoping (fusion-redesign D-I): ConvNeXt stage 4 was
        reported numerically fragile under CUDA autocast in this dual-stream
        setup, so this forward used to force `autocast(enabled=False)`
        unconditionally. That guard is scoped to fp16 only: it is a no-op
        under fp32 (nothing to disable) and, critically, a no-op under bf16
        as well — bf16 carries fp32's exponent range, so the fp16 overflow
        mechanism the original comment describes does not apply, and H-BF16
        (see design.md) specifically needs bf16 to reach these stages rather
        than being silently forced back to fp32.
        """
        if self.use_nir and nir is None:
            raise ValueError(
                "DualConvNeXtBackbone(use_nir=True) needs the NIR tensor; got "
                "nir=None."
            )
        device_type = rgb.device.type
        fp16_ambient = (
            torch.is_autocast_enabled(device_type)
            and torch.get_autocast_dtype(device_type) == torch.float16
        )
        ctx = (
            torch.amp.autocast(device_type=device_type, enabled=False)
            if fp16_ambient
            else nullcontext()
        )
        with ctx:
            # Pass through modality-specific stems
            rgb_in = rgb.float() if fp16_ambient else rgb
            rgb_x = self.rgb_stem(rgb_in)   # (N, 96, H/4, W/4)

            if not self.use_nir:
                # RGB-only arm: one pass through the shared stages.
                rgb_features = []
                for stage in self.shared_stages:
                    rgb_x = stage(rgb_x)
                    rgb_features.append(rgb_x)
                return rgb_features, None

            nir_in = nir.float() if fp16_ambient else nir
            nir_x = self.nir_stem(nir_in)   # (N, 96, H/4, W/4)

            rgb_features, nir_features = [], []

            # Pass through shared stages: same weights, different activations
            for stage in self.shared_stages:
                rgb_x = stage(rgb_x)
                nir_x = stage(nir_x)
                rgb_features.append(rgb_x)
                nir_features.append(nir_x)

        return rgb_features, nir_features

    # ------------------------------------------------------------------
    # Weight initialization helpers
    # ------------------------------------------------------------------

    def _init_nir_stem(self):
        """Initialize NIR stem with Kaiming normal (standard for conv layers)."""
        nn.init.kaiming_normal_(
            self.nir_stem[0].weight, mode="fan_out", nonlinearity="relu"
        )
        if self.nir_stem[0].bias is not None:
            nn.init.zeros_(self.nir_stem[0].bias)

    def _load_pretrained_rgb_stem(self):
        """
        Copy the pretrained ConvNeXt stem weights into rgb_stem.
        The pretrained stem expects 3-channel input, which matches RGB.
        Works for both Tiny and Small variants (identical stem architecture).
        """
        if self.variant == "tiny":
            pretrained_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            pretrained_model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        pretrained_stem_state = {
            "0.weight": pretrained_model.features[0][0].weight,
            "0.bias":   pretrained_model.features[0][0].bias,
            "1.norm.weight": pretrained_model.features[0][1].weight,
            "1.norm.bias":   pretrained_model.features[0][1].bias,
        }
        self.rgb_stem.load_state_dict(pretrained_stem_state, strict=False)

    def _load_pretrained_nir_stem(self):
        """Initialize the 1-channel NIR stem from RGB ImageNet stem weights.

        Averaging pretrained RGB kernels gives the NIR stream low-level edge and
        texture priors while keeping the stem trainable for 850 nm reflectance.
        """
        if self.variant == "tiny":
            pretrained_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            pretrained_model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)

        rgb_weight = pretrained_model.features[0][0].weight
        pretrained_stem_state = {
            "0.weight": rgb_weight.mean(dim=1, keepdim=True),
            "0.bias": pretrained_model.features[0][0].bias,
            "1.norm.weight": pretrained_model.features[0][1].weight,
            "1.norm.bias": pretrained_model.features[0][1].bias,
        }
        self.nir_stem.load_state_dict(pretrained_stem_state, strict=False)
