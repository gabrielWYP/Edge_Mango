"""
Exponential moving average (EMA) of model weights, YOLOv8-style.

Leave-one-lot-out training sets are small (117-162 pairs -> 15-21 optimizer
steps per epoch at `effective_batch=8`), so the live weights at the end of any
single epoch are a noisy sample and validation AP flips from one epoch to the
next (val damage TP 36 -> 0 -> 49 at epochs 50/60/70). An EMA of the weights
averages over the last ~1 / (1 - decay) optimizer steps; when
`TrainingConfig.ema` is on it is what gets validated, selected and saved.

Decay warm-up: the decay applied by update `n` is
`d_n = decay * (1 - exp(-n / tau))`. Early on `d` is close to 0, so the EMA
tracks the live model instead of staying anchored to the random/pretrained
initialization; after a few `tau` updates it approaches `decay`. `tau` must be
scaled to the run's step budget - YOLOv8's 2000 assumes tens of thousands of
steps; a 100-epoch fold run here takes ~1.5k-2.1k.

Only floating-point **parameters** are averaged. **Buffers** (BatchNorm
running statistics, `num_batches_tracked`) are copied verbatim from the live
model on every update: BatchNorm statistics are already running averages, and
averaging an average (and an integer counter) adds nothing. With
`neck_norm="group"` the master model has no buffers at all, so this only
matters for the default BatchNorm neck and for the student.

`update()` is called once per optimizer step, never per micro-batch, so
gradient accumulation does not change the effective averaging horizon.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

# Fold runs: ~15-21 optimizer steps/epoch, 100 epochs. decay 0.99 averages the
# last ~100 steps (5-6 epochs); tau 50 hands over from "track the live weights"
# to "average" within the first ~10 epochs. See `TrainingConfig.ema_decay`.
DEFAULT_EMA_DECAY = 0.99
DEFAULT_EMA_TAU = 50.0


def validate_ema_params(decay: float, tau: float) -> None:
    """Raise `ValueError` unless `decay` is in (0, 1) and `tau` is positive."""
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0.0 < decay < 1.0:
        raise ValueError(f"ema_decay must be in (0, 1), got {decay!r}.")
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not tau > 0.0:
        raise ValueError(f"ema_tau must be > 0, got {tau!r}.")


def validate_ema(ema: bool, decay: float, tau: float) -> None:
    """Validate the `ema`, `ema_decay`, `ema_tau` config trio.

    `ema` must be a real bool: a YAML string like `"false"` is truthy and
    would silently enable the EMA. The numeric parameters are validated even
    when the EMA is off, so a typo cannot hide until the day it is switched on.
    """
    if not isinstance(ema, bool):
        raise ValueError(f"ema must be a bool, got {ema!r} ({type(ema).__name__}).")
    validate_ema_params(decay, tau)


class ModelEMA:
    """EMA shadow copy of `model`.

    Args:
        model: The live model. Deep-copied once; the copy is kept in eval
            mode with `requires_grad=False`, on the live model's device.
        decay: Asymptotic EMA decay in (0, 1).
        tau: Warm-up time constant, in optimizer steps (> 0).
        updates: Number of updates already applied (for resuming).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = DEFAULT_EMA_DECAY,
        tau: float = DEFAULT_EMA_TAU,
        updates: int = 0,
    ):
        validate_ema_params(decay, tau)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = updates

    def current_decay(self) -> float:
        """Decay applied by the most recent update (0.0 before any update)."""
        return self.decay * (1.0 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend `model`'s parameters into the EMA and copy its buffers."""
        self.updates += 1
        d = self.current_decay()
        live_params = dict(model.named_parameters())
        for name, ema_p in self.module.named_parameters():
            live_p = live_params[name].detach()
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(d).add_(live_p.to(ema_p.dtype), alpha=1.0 - d)
            else:
                ema_p.copy_(live_p)
        live_buffers = dict(model.named_buffers())
        for name, ema_b in self.module.named_buffers():
            ema_b.copy_(live_buffers[name].detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()
