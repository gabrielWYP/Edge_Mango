"""
CSPDarknet-Nano backbone for YOLOv8-Nano student model.

Pure PyTorch implementation (no ultralytics dependency) with building blocks:
Conv_BN_SiLU, Bottleneck, C2f, SPPF.

Architecture (for 640x640 input):
    Stem: Conv(3->16, s=2) -> Conv(16->32, s=2)     stride 4,  160x160
    S1:   C2f(32, n=1)                               stride 4,  160x160
    S2:   Conv(32->64, s=2) -> C2f(64, n=2)          stride 8,   80x80
    S3:   Conv(64->128, s=2) -> C2f(128, n=2)        stride 16,  40x40
    S4:   Conv(128->256, s=2) -> C2f(256, n=1) -> SPPF(256)  stride 32, 20x20

Output:
    stages: [S1, S2, S3, S4_sppf] — all 4 stage feature maps
    distill: (S3, S4_sppf) — backbone distillation features at 128ch and 256ch

Channel contract with ProjectionLayers:
    backbone_projections student_channels = [128, 256] for (S3, S4).
"""

import torch
import torch.nn as nn
from torch import Tensor


class Conv_BN_SiLU(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU activation block."""

    def __init__(self, in_ch: int, out_ch: int, k: int, s: int) -> None:
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    Standard bottleneck block with optional residual connection.

    Two 3x3 convolutions with residual added when input and output channels match.
    """

    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True) -> None:
        super().__init__()
        self.cv1 = Conv_BN_SiLU(in_ch, out_ch, 3, 1)
        self.cv2 = Conv_BN_SiLU(out_ch, out_ch, 3, 1)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x: Tensor) -> Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C2f(nn.Module):
    """
    C2f block from YOLOv8: split-merge with n bottleneck branches.

    1. cv1 splits input into 2 halves (c_ channels each)
    2. Second half passes through n sequential Bottleneck blocks
    3. All branches concatenated and merged via cv2 (1x1 conv)
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 1, shortcut: bool = True) -> None:
        super().__init__()
        self.c_ = out_ch // 2  # hidden channels per branch
        self.cv1 = Conv_BN_SiLU(in_ch, 2 * self.c_, 1, 1)
        self.cv2 = Conv_BN_SiLU((2 + n) * self.c_, out_ch, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c_, self.c_, shortcut) for _ in range(n)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) block.

    Sequential max-pooling with kernel k applied 3 times, then concatenated
    with the reduced input and merged via 1x1 conv.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 5) -> None:
        super().__init__()
        c_ = in_ch // 2
        self.cv1 = Conv_BN_SiLU(in_ch, c_, 1, 1)
        self.cv2 = Conv_BN_SiLU(c_ * 4, out_ch, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class CSPDarknetNano(nn.Module):
    """
    CSPDarknet-Nano backbone for YOLOv8-Nano.

    4-stage backbone with C2f blocks and SPPF on the final stage.
    Designed for single-modality RGB input (N, 3, 640, 640).

    Stage outputs (for 640x640 input):
        S1: 32ch,  160x160 (stride 4)
        S2: 64ch,   80x80  (stride 8)
        S3: 128ch,  40x40  (stride 16)
        S4: 256ch,  20x20  (stride 32, after SPPF)

    Forward returns:
        stages: list of [S1, S2, S3, S4] feature maps
        distill: tuple (S3, S4) for backbone-level knowledge distillation
    """

    STAGE_CHANNELS = [32, 64, 128, 256]

    def __init__(self) -> None:
        super().__init__()

        # Stem: two stride-2 convolutions (stride 4 total)
        self.stem = nn.Sequential(
            Conv_BN_SiLU(3, 16, 3, 2),    # -> 16ch, 320x320
            Conv_BN_SiLU(16, 32, 3, 2),   # -> 32ch, 160x160
        )

        # Stage 1: C2f at stride 4
        self.stage1 = C2f(32, 32, n=1)     # -> 32ch, 160x160

        # Stage 2: downsample + C2f at stride 8
        self.stage2 = nn.Sequential(
            Conv_BN_SiLU(32, 64, 3, 2),    # -> 64ch, 80x80
            C2f(64, 64, n=2),              # -> 64ch, 80x80
        )

        # Stage 3: downsample + C2f at stride 16
        self.stage3 = nn.Sequential(
            Conv_BN_SiLU(64, 128, 3, 2),   # -> 128ch, 40x40
            C2f(128, 128, n=2),            # -> 128ch, 40x40
        )

        # Stage 4: downsample + C2f + SPPF at stride 32
        self.stage4 = nn.Sequential(
            Conv_BN_SiLU(128, 256, 3, 2),  # -> 256ch, 20x20
            C2f(256, 256, n=1),            # -> 256ch, 20x20
        )
        self.sppf = SPPF(256, 256, k=5)    # -> 256ch, 20x20

    def forward(self, x: Tensor) -> tuple[list[Tensor], tuple[Tensor, Tensor]]:
        """
        Forward pass through CSPDarknet-Nano backbone.

        Args:
            x: Input tensor of shape (N, 3, H, W). Expected (N, 3, 640, 640).

        Returns:
            stages: [S1, S2, S3, S4] — list of 4 feature maps for the neck.
                S1: (N, 32, H/4, W/4)
                S2: (N, 64, H/8, W/8)
                S3: (N, 128, H/16, W/16)
                S4: (N, 256, H/32, W/32) — after SPPF
            distill: (S3, S4) — backbone distillation features matching
                backbone_projections student_channels [128, 256].
        """
        x = self.stem(x)

        s1 = self.stage1(x)       # (N, 32, 160, 160)
        s2 = self.stage2(s1)      # (N, 64, 80, 80)
        s3 = self.stage3(s2)      # (N, 128, 40, 40)
        s4 = self.sppf(self.stage4(s3))  # (N, 256, 20, 20)

        return [s1, s2, s3, s4], (s3, s4)
