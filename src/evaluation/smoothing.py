"""Curve smoothing shared by the run-curve plots."""

from __future__ import annotations


def ema_smooth(values, w):
    """Exponential moving average with bias correction (TensorBoard style), so the first epochs are not pulled towards 0."""
    out, acc = [], 0.0
    for t, v in enumerate(values, 1):
        acc = w * acc + (1 - w) * v
        out.append(acc / (1 - w**t) if w else v)
    return out
