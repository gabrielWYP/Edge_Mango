"""
Master Model — Full architecture for multimodal mango damage detection.

Input:  RGB image (N, 3, H, W) + NIR image (N, 1, H, W)
Output: YOLO-style detections + intermediate features for knowledge distillation

Two mutually exclusive fusion architectures, selected by `fusion_mode`
(`src/training/fusion_modes.py`). `forward(rgb, nir)` and the 7-key output
dict are identical in both, so `dataset.py`, `Trainer._train_epoch` and
`KDTrainer` need no call-site branch.

`fusion_mode="early"` (default) — simple early fusion. RGB and NIR are
stacked into one 4-channel tensor and fed through a single-stream backbone;
no cross-modal attention module, no second FPN:

    RGB + NIR
        ↓ cat(dim=1)
    EarlyFusionBackbone (single stem, single stage stack)
        ↓
    [S1..S4] — 4 stage features
        ↓
    FPNNeck ( SingleFPN )
        ↓
    pyramid — levels named by `head_strides`, default [P2, P3, P4, P5]
        ↓
    YOLODetectionHead (anchor-free, decoupled)

`fusion_mode="cross_attention"` — the pre-redesign two-stream architecture,
restored as a selectable mode (not a revert):

    RGB, NIR
        ↓
    DualConvNeXtBackbone (two stems, shared stages)
        ↓
    [rgb S1..S4], [nir S1..S4]
        ↓
    CrossModalFusion (per-stage cross-attention, RGB queries NIR)
        ↓
    [F1..F4], [nir S1..S4]
        ↓
    DualFPN (two FPNs + per-level fusion)
        ↓
    pyramid — levels named by `head_strides`, default [P3, P4, P5];
    [P2, P3, P4, P5] is also supported here (see
    `CROSS_ATTENTION_SUPPORTED_HEAD_STRIDES`)
        ↓
    YOLODetectionHead (anchor-free, decoupled)

Distillation outputs exposed (same 4 keys in both modes):
    - distill_backbone:  early → [S1..S4] from the single stream;
                         cross_attention → [F1..F4], the cross-attention
                         fused features. Both are the 4-stage,
                         [96, 192, 384, 768]-channel feature set that
                         carries BOTH modalities, which is what `KDTrainer`
                         projects (`t_out["distill_backbone"][2:]`).
    - distill_fpn:       the emitted pyramid levels (per `head_strides`)
    - distill_head_cls:  [cls_stem per level] head classification features
    - distill_head_reg:  [reg_stem per level] head regression features

Checkpoints of the two modes share no backbone, fusion or neck state_dict
key (only the detection head's names coincide); `arch_version` (2 = early,
1 = cross_attention) is what keeps one from being loaded as the other — see
`src/training/fusion_modes.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import DualConvNeXtBackbone, EarlyFusionBackbone
from .fusion import CrossModalFusion
from .neck import DualFPN, FPNNeck
from .head import YOLODetectionHead, NUM_CLASSES
from src.training.fusion_modes import (
    DEFAULT_FUSION_MODE,
    FUSION_MODE_CROSS_ATTENTION,
    FUSION_MODE_EARLY,
    validate_fusion_mode,
)
from src.training.neck_norm import (
    DEFAULT_NECK_GN_GROUPS,
    DEFAULT_NECK_NORM,
    validate_neck_norm,
    validate_neck_norm_for_fusion_mode,
)
from src.training.strides import (
    CROSS_ATTENTION_HEAD_STRIDES,
    CROSS_ATTENTION_SUPPORTED_HEAD_STRIDES,
    DEFAULT_HEAD_STRIDES,
    validate_strides,
)


class MasterModel(nn.Module):
    """
    Multimodal master model for cross-modal knowledge distillation.

    Args:
        num_classes (int): Detection classes (default 2: mango, danado).
        pretrained_backbone (bool): Load ImageNet weights for ConvNeXt stages
            and seed the stem(s) — mean-preserving inflation for the early
            path's `in_channels=4` stem, verbatim/channel-averaged copies for
            the cross-attention path's RGB/NIR stems.
        fpn_channels (int): FPN output channels (default 256).
        head_strides (list[int] | None): Pyramid strides the head is built
            over, finest-first. `fusion_mode="early"`: defaults to
            `[4, 8, 16, 32]` — includes the reconnected P2 level
            (fusion-redesign D-3); pass `[8, 16, 32]` for the pre-redesign
            3-level pyramid. `fusion_mode="cross_attention"`: defaults to
            `CROSS_ATTENTION_HEAD_STRIDES` (`[8, 16, 32]`, the pre-redesign
            two-stream schema) and also accepts `[4, 8, 16, 32]`, which
            emits the P2 level `DualFPN` computes either way — see
            `CROSS_ATTENTION_SUPPORTED_HEAD_STRIDES`. Any other value raises.
        backbone_variant (str): ConvNeXt variant — "tiny" or "small".
        in_channels (int): Backbone stem input channels, `fusion_mode="early"`
            only. 4 (default) for the early-fused RGB+NIR input; 3 for the
            RGB-only H-D control arm (`nir` is then accepted but ignored, so
            call sites need no separate code path). The cross-attention path
            has fixed 3ch/1ch modality stems and accepts only the default.
        fusion_mode (str): "early" (default) | "cross_attention". See the
            module docstring and `src/training/fusion_modes.py`.
        fusion_dropout (float): Cross-attention dropout —
            `fusion_mode="cross_attention"` only.
        fpn_dropout (float): `DualFPN` per-level fusion dropout —
            `fusion_mode="cross_attention"` only.
        fusion_pos_encoding (bool): Add a fixed 2D sinusoidal positional
            encoding to the cross-attention query/key (see
            `src/models/master/fusion.py`). Default False keeps the
            pre-existing behaviour; adds no state_dict key either way.
            `fusion_mode="cross_attention"` only — True raises on "early".
        neck_norm (str): Norm layer of the `DualFPN` fusion convs - "batch"
            (default, `nn.BatchNorm2d`) or "group" (`nn.GroupNorm`, no batch
            statistics). These are the only BatchNorm layers in the model:
            the backbone and the cross-modal fusion use LayerNorm and the
            detection head has no normalization layer. Changes the
            state_dict schema (GroupNorm has no running statistics), so a
            checkpoint of one cannot be loaded into the other.
            `fusion_mode="cross_attention"` only - "group" raises on "early"
            (its `FPNNeck` has no norm layer). See
            `src/training/neck_norm.py`.
        neck_gn_groups (int): GroupNorm group-count cap for `neck_norm="group"`.
    """

    # ConvNeXt stage channels (identical for Tiny and Small)
    STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained_backbone: bool = True,
        fpn_channels: int = 256,
        head_strides: list[int] | None = None,
        backbone_variant: str = "tiny",
        in_channels: int = 4,
        fusion_mode: str = DEFAULT_FUSION_MODE,
        fusion_dropout: float = 0.1,
        fpn_dropout: float = 0.1,
        fusion_pos_encoding: bool = False,
        neck_norm: str = DEFAULT_NECK_NORM,
        neck_gn_groups: int = DEFAULT_NECK_GN_GROUPS,
    ):
        super().__init__()

        validate_fusion_mode(fusion_mode)
        self.fusion_mode = fusion_mode

        validate_neck_norm(neck_norm, neck_gn_groups)
        validate_neck_norm_for_fusion_mode(neck_norm, fusion_mode)
        self.neck_norm = neck_norm

        strides = self._resolve_strides(fusion_mode, head_strides)
        validate_strides(strides)
        self.head_strides = strides
        self.in_channels = in_channels

        if fusion_mode == FUSION_MODE_EARLY:
            if fusion_pos_encoding:
                raise ValueError(
                    "fusion_pos_encoding=True encodes position into the "
                    f"cross-attention fusion, which only fusion_mode="
                    f"'{FUSION_MODE_CROSS_ATTENTION}' has; fusion_mode="
                    f"'{FUSION_MODE_EARLY}' would silently ignore it."
                )
            # --- 1. Single-stream early-fusion backbone ---
            self.backbone = EarlyFusionBackbone(
                pretrained=pretrained_backbone,
                variant=backbone_variant,
                in_channels=in_channels,
            )
            # --- 2. FPN neck (SingleFPN, wrapped to emit the configured levels) ---
            self.neck = FPNNeck(
                in_channels=self.STAGE_CHANNELS,
                out_channels=fpn_channels,
                strides=tuple(strides),
            )
        else:
            if in_channels != 4:
                raise ValueError(
                    f"fusion_mode='{FUSION_MODE_CROSS_ATTENTION}' has fixed "
                    f"3-channel RGB and 1-channel NIR stems, so in_channels="
                    f"{in_channels} cannot be honoured. Leave in_channels at "
                    "its default; the RGB-only control arm (in_channels=3) "
                    f"exists only for fusion_mode='{FUSION_MODE_EARLY}'."
                )
            # --- 1. Dual backbone (shared stages, separate stems) ---
            self.backbone = DualConvNeXtBackbone(
                pretrained=pretrained_backbone, variant=backbone_variant
            )
            # --- 2. Cross-modal attention fusion (per backbone stage) ---
            self.fusion = CrossModalFusion(
                stage_channels=self.STAGE_CHANNELS,
                dropout=fusion_dropout,
                pos_encoding=fusion_pos_encoding,
            )
            # --- 3. Dual FPN neck (two FPNs + per-level fusion) ---
            self.neck = DualFPN(
                in_channels=self.STAGE_CHANNELS,
                out_channels=fpn_channels,
                dropout=fpn_dropout,
                emit_strides=tuple(strides),
                norm=neck_norm,
                gn_groups=neck_gn_groups,
            )

        # --- YOLO-style detection head (compatible with YOLO Nano student) ---
        self.head = YOLODetectionHead(
            fpn_channels=fpn_channels,
            num_classes=num_classes,
            strides=strides,
        )

    @staticmethod
    def _resolve_strides(fusion_mode: str, head_strides: list[int] | None) -> list[int]:
        """Return the pyramid strides for `fusion_mode`, failing loudly on a
        combination the architecture cannot honour.

        `DualFPN` sizes its fusion convs from the pyramid it is asked to
        emit, so the cross-attention path accepts any member of
        `CROSS_ATTENTION_SUPPORTED_HEAD_STRIDES` and defaults to
        `CROSS_ATTENTION_HEAD_STRIDES`. Anything else still raises here: an
        unsupported `head_strides` would otherwise surface as a `zip` that
        silently truncates the pyramid — the failure class
        `src/training/strides.py` exists to eliminate.
        """
        if fusion_mode == FUSION_MODE_CROSS_ATTENTION:
            if head_strides is None:
                return list(CROSS_ATTENTION_HEAD_STRIDES)
            requested = tuple(head_strides)
            if requested not in CROSS_ATTENTION_SUPPORTED_HEAD_STRIDES:
                supported = [list(s) for s in CROSS_ATTENTION_SUPPORTED_HEAD_STRIDES]
                raise ValueError(
                    f"fusion_mode='{FUSION_MODE_CROSS_ATTENTION}' supports "
                    f"head_strides in {supported} (DualFPN builds one fusion "
                    f"conv per emitted level over the [P2, P3, P4, P5] "
                    f"pyramid), got {list(head_strides)}."
                )
            return list(requested)
        if head_strides is not None:
            return list(head_strides)
        return list(DEFAULT_HEAD_STRIDES)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> dict:
        """
        Args:
            rgb: (N, 3, H, W) — RGB image, normalized.
            nir: (N, 1, H, W) — NIR grayscale image, normalized. Ignored
                when `fusion_mode="early"` and `self.in_channels == 3` (the
                H-D RGB-only control arm), accepted anyway so call sites do
                not need a separate path.

        Returns:
            dict with keys (identical in both fusion modes):
                'preds'              : list of (B, nc+4, H_i, W_i) per level
                'cls_preds'          : list of (B, nc, H_i, W_i) per level
                'reg_preds'          : list of (B, 4, H_i, W_i) per level
                'distill_backbone'   : 4 stage features carrying both modalities
                'distill_fpn'        : emitted FPN pyramid levels
                'distill_head_cls'   : [cls_stem per level] head features
                'distill_head_reg'   : [reg_stem per level] head features
        """
        if self.fusion_mode == FUSION_MODE_EARLY:
            # --- Early fusion: stack RGB+NIR into one input tensor ---
            if self.in_channels == 4:
                x = torch.cat([rgb, nir], dim=1)  # (N, 4, H, W)
            else:
                x = rgb  # H-D RGB-only control arm

            # --- Backbone (single stream) ---
            # [S1..S4], channels [96, 192, 384, 768]
            backbone_features = self.backbone(x)

            # --- FPN neck ---
            pyramid = self.neck(backbone_features)  # levels named by self.head_strides
        else:
            # --- Backbone (two streams, shared stages) ---
            rgb_features, nir_features = self.backbone(rgb, nir)

            # --- Cross-modal fusion: RGB queries NIR, per stage ---
            # backbone_features: [F1..F4] — RGB enriched with NIR context.
            # Same 4-stage, [96, 192, 384, 768] shape the early path emits,
            # so `distill_backbone` means the same thing downstream.
            backbone_features, nir_features = self.fusion(rgb_features, nir_features)

            # --- Dual FPN ---
            # levels named by self.head_strides
            pyramid = self.neck(backbone_features, nir_features)

        # --- YOLO Detection Head ---
        head_output = self.head(pyramid)

        return {
            # Detection outputs
            "preds":      head_output["preds"],
            "cls_preds":  head_output["cls_preds"],
            "reg_preds":  head_output["reg_preds"],

            # Distillation features — exposed for student training
            "distill_backbone": backbone_features,          # 4 stage features
            "distill_fpn":       pyramid,                    # emitted levels
            "distill_head_cls":  head_output["distill_cls"], # [cls_stem per level]
            "distill_head_reg":  head_output["distill_reg"], # [reg_stem per level]
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def freeze_backbone(self, freeze_stages: int = 2):
        """
        Freeze the backbone stem(s) plus the first N stages.

        No stem/stage asymmetry in either mode: the stems are frozen or
        trainable exactly like the stages are (fusion-redesign D-4). The
        pre-redesign two-stream code froze `rgb_stem` unconditionally and
        kept `nir_stem` always trainable; that split is NOT restored — it is
        the documented E6/Q10 defect behind the maestro underperforming its
        own student, and it would also break `schedule="end_to_end"`, which
        calls this with `freeze_stages=0` expecting everything trainable.

        Args:
            freeze_stages: Number of stages to freeze (0-4).
                           0 = freeze nothing (stems stay trainable),
                           4 = freeze the stems and all stages.
        """
        for stem in self.backbone.stem_modules.values():
            for param in stem.parameters():
                param.requires_grad = freeze_stages == 0

        for i, stage in enumerate(self.backbone.stages):
            trainable = i >= freeze_stages
            for param in stage.parameters():
                param.requires_grad = trainable

        print(f"[MasterModel] Frozen: stem(s) (if freeze_stages>0) + first {freeze_stages} stages.")

    def unfreeze_backbone_stages(
        self,
        unfreeze_stages: list[int],
        unfreeze_stem: bool = False,
    ):
        """
        Unfreeze specific backbone stages by index (0-based).

        Args:
            unfreeze_stages: List of stage indices to unfreeze (e.g., [2, 3] for stages 3-4).
            unfreeze_stem: If True, also unfreeze every backbone stem.
                Renamed from the previous `unfreeze_rgb_stem` (fusion-redesign
                D-4). Defaults to False to preserve prior call-site behavior
                for any caller that does not explicitly opt in.
        """
        for i, stage in enumerate(self.backbone.stages):
            if i in unfreeze_stages:
                for param in stage.parameters():
                    param.requires_grad = True

        if unfreeze_stem:
            for stem in self.backbone.stem_modules.values():
                for param in stem.parameters():
                    param.requires_grad = True

        print(
            f"[MasterModel] Unfrozen stages: {unfreeze_stages}. "
            f"Stem(s) {'unfrozen' if unfreeze_stem else 'unchanged'}."
        )

    def count_parameters(self) -> dict[str, int]:
        """Returns parameter counts per module.

        `fusion` appears only under `fusion_mode="cross_attention"` — the
        early path has no such submodule, so reporting a 0 for it would
        misrepresent an absent module as an empty one.
        """
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {"backbone": count(self.backbone)}
        if self.fusion_mode == FUSION_MODE_CROSS_ATTENTION:
            counts["fusion"] = count(self.fusion)
        counts["neck"] = count(self.neck)
        counts["head"] = count(self.head)
        counts["total"] = count(self)
        return counts
