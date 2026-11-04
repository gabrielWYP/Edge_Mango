"""
PANet neck for YOLOv8-Nano student model.

Top-down FPN + bottom-up PAN path for bidirectional feature fusion.
Produces 3 pyramid levels at channels [128, 256, 256] matching
fpn_projections student_channels contract.

Input from backbone (last 3 stages):
    S2: 64ch,  80x80 (stride 8)
    S3: 128ch, 40x40 (stride 16)
    S4: 256ch, 20x20 (stride 32)

Output:
    P3: 128ch, 80x80 (stride 8)
    P4: 256ch, 40x40 (stride 16)
    P5: 256ch, 20x20 (stride 32)

Channel contract with ProjectionLayers:
    fpn_projections student_channels = [128, 256, 256] for (P3, P4, P5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.student.backbone import Conv_BN_SiLU, C2f


class PANet(nn.Module):
    """
    PANet neck: FPN top-down + PAN bottom-up.

    Takes backbone stages [S1, S2, S3, S4] and produces 3 FPN levels.
    Uses S2, S3, S4 (S1 is not used by the neck).

    Top-down path propagates semantic information from coarse to fine.
    Bottom-up path propagates localization information from fine to coarse.
    """

    OUTPUT_CHANNELS = [128, 256, 256]

    def __init__(self) -> None:
        super().__init__()

        # Lateral 1x1 convs — project backbone channels to FPN channels
        self.lateral_p3 = Conv_BN_SiLU(64, 128, 1, 1)    # S2: 64 -> 128
        self.lateral_p4 = Conv_BN_SiLU(128, 256, 1, 1)   # S3: 128 -> 256
        self.lateral_p5 = Conv_BN_SiLU(256, 256, 1, 1)   # S4: 256 -> 256

        # Top-down C2f blocks (after concat of upsampled + lateral)
        self.td_c2f_4 = C2f(512, 128, n=1)   # concat(p5_up, p4_lat): 256+256 -> 128
        self.td_c2f_3 = C2f(256, 128, n=1)   # concat(td4_up, p3_lat): 128+128 -> 128

        # Bottom-up downsample convs (stride 2)
        self.bu_conv_3 = Conv_BN_SiLU(128, 128, 3, 2)   # F3: 80x80 -> 40x40
        self.bu_conv_4 = Conv_BN_SiLU(256, 256, 3, 2)   # N4: 40x40 -> 20x20

        # Bottom-up C2f blocks (after concat of downsampled + top-down feature)
        self.bu_c2f_4 = C2f(256, 256, n=1)   # concat(f3_down, td4): 128+128 -> 256
        self.bu_c2f_5 = C2f(512, 256, n=1)   # concat(n4_down, p5_lat): 256+256 -> 256

    def forward(self, backbone_features: list[Tensor]) -> dict[str, Tensor]:
        """
        Forward pass through PANet neck.

        Args:
            backbone_features: [S1, S2, S3, S4] from CSPDarknetNano.
                S1: (N, 32, H/4, W/4)
                S2: (N, 64, H/8, W/8)
                S3: (N, 128, H/16, W/16)
                S4: (N, 256, H/32, W/32)

        Returns:
            dict with keys 'p3', 'p4', 'p5':
                p3: (N, 128, H/8, W/8)   — 80x80 for 640 input
                p4: (N, 256, H/16, W/16)  — 40x40 for 640 input
                p5: (N, 256, H/32, W/32)  — 20x20 for 640 input
        """
        _, s2, s3, s4 = backbone_features

        # ── Lateral projections ──────────────────────────────────────
        p3_lat = self.lateral_p3(s2)    # (N, 128, 80, 80)
        p4_lat = self.lateral_p4(s3)    # (N, 256, 40, 40)
        p5_lat = self.lateral_p5(s4)    # (N, 256, 20, 20)

        # ── Top-down path (FPN) ──────────────────────────────────────
        p5_up = F.interpolate(p5_lat, scale_factor=2, mode="nearest")
        # p5_up: (N, 256, 40, 40)

        td4 = self.td_c2f_4(torch.cat([p5_up, p4_lat], dim=1))
        # td4: (N, 128, 40, 40)

        td4_up = F.interpolate(td4, scale_factor=2, mode="nearest")
        # td4_up: (N, 128, 80, 80)

        f3 = self.td_c2f_3(torch.cat([td4_up, p3_lat], dim=1))
        # f3: (N, 128, 80, 80) — this is N3 (final P3 output)

        # ── Bottom-up path (PAN) ─────────────────────────────────────
        f3_down = self.bu_conv_3(f3)
        # f3_down: (N, 128, 40, 40)

        n4 = self.bu_c2f_4(torch.cat([f3_down, td4], dim=1))
        # n4: (N, 256, 40, 40) — final P4 output

        n4_down = self.bu_conv_4(n4)
        # n4_down: (N, 256, 20, 20)

        n5 = self.bu_c2f_5(torch.cat([n4_down, p5_lat], dim=1))
        # n5: (N, 256, 20, 20) — final P5 output

        return {"p3": f3, "p4": n4, "p5": n5}


class NanoPANet(nn.Module):
    """
    The neck of the official YOLOv8n: no lateral convs, and C2f blocks without shortcut.

    Produces 3 pyramid levels at channels [64, 128, 256], for the `nano` student.
    """

    OUTPUT_CHANNELS = [64, 128, 256]

    def __init__(self) -> None:
        super().__init__()
        self.td_c2f_4 = C2f(384, 128, n=1, shortcut=False)  # concat(s4_up, s3): 256+128
        self.td_c2f_3 = C2f(192, 64, n=1, shortcut=False)   # concat(td4_up, s2): 128+64
        self.bu_conv_3 = Conv_BN_SiLU(64, 64, 3, 2)
        self.bu_c2f_4 = C2f(192, 128, n=1, shortcut=False)  # concat(f3_down, td4): 64+128
        self.bu_conv_4 = Conv_BN_SiLU(128, 128, 3, 2)
        self.bu_c2f_5 = C2f(384, 256, n=1, shortcut=False)  # concat(n4_down, s4): 128+256

    def forward(self, backbone_features: list[Tensor]) -> dict[str, Tensor]:
        _, s2, s3, s4 = backbone_features
        td4 = self.td_c2f_4(torch.cat([F.interpolate(s4, scale_factor=2, mode="nearest"), s3], 1))
        f3 = self.td_c2f_3(torch.cat([F.interpolate(td4, scale_factor=2, mode="nearest"), s2], 1))
        n4 = self.bu_c2f_4(torch.cat([self.bu_conv_3(f3), td4], 1))
        n5 = self.bu_c2f_5(torch.cat([self.bu_conv_4(n4), s4], 1))
        return {"p3": f3, "p4": n4, "p5": n5}
