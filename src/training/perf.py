"""
Launch-overhead helpers for the trainers: cudnn autotuning and a CUDA graph for the frozen teacher.

A nano student at batch 8 launches many tiny kernels, so the main process is CPU-bound on
launches, not the GPU on maths. Two of the remedies live here (the loss is vectorized in
`loss.py`):

- `enable_cudnn_benchmark`: input shapes are fixed during training, so cudnn can autotune its
  convolution algorithms once. Called by the training entrypoints (`train.py`, `kd_train.py`)
  on CUDA only.
- `GraphedForward`: replays a frozen, eval-mode forward (the CrossKD / feature-KD teacher) as
  one CUDA graph launch instead of relaunching it op by op every step. It captures on the
  first call and replays while the input signature stays the same; any other call (a different
  shape such as a last partial batch, grad mode, CPU, a failed capture) runs the original
  callable eagerly. Outputs equal the eager ones.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

logger = logging.getLogger(__name__)

__all__ = ["GraphedForward", "enable_cudnn_benchmark", "tree_flatten"]


def enable_cudnn_benchmark(device: str | torch.device) -> bool:
    """Turn on `torch.backends.cudnn.benchmark` when `device` is CUDA; return whether it is on.

    `device` is the config value: "auto" resolves like the trainer does (CUDA when available).
    Safe for training because the registered protocol uses fixed input shapes. It is not
    bit-deterministic across runs (autotuning may pick another algorithm), which is already
    true of the cudnn convolutions it replaces.
    """
    name = str(device)
    on_cuda = (name == "auto" and torch.cuda.is_available()) or name.startswith("cuda")
    if on_cuda:
        torch.backends.cudnn.benchmark = True
    return on_cuda


class _CudaGraphBackend:
    """Captures a callable into a `torch.cuda.CUDAGraph` with static input buffers."""

    def __init__(self, warmup: int = 3):
        self.warmup = warmup

    @staticmethod
    def usable(tensors: list[torch.Tensor]) -> bool:
        return torch.cuda.is_available() and all(t.is_cuda for t in tensors)

    def capture(self, fn: Callable[..., Any], example_inputs: list[torch.Tensor]):
        static_in = [t.detach().clone() for t in example_inputs]
        # Warm up on a side stream: lazy init, cudnn autotuning and allocator growth must not
        # happen inside the capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                fn(*static_in)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = fn(*static_in)

        def replay(inputs: list[torch.Tensor]):
            for buffer, value in zip(static_in, inputs):
                buffer.copy_(value)
            graph.replay()
            return static_out

        return replay


class GraphedForward:
    """Call `fn(*tensors)` through a CUDA graph when that is safe, eagerly otherwise.

    Meant for a frozen module called under `torch.no_grad()` with fixed input shapes. The
    outputs are cloned out of the graph's static buffers, so they stay valid after the next
    call (CrossKD calls the teacher twice per step with the same shapes).

    Eager fallbacks, each leaving results unchanged: CPU inputs; grad mode or inputs that
    require grad; a call whose signature (arity, shapes, dtypes, autocast state) differs from
    the captured one; non-tensor arguments; and any failure while capturing, logged once.

    Args:
        fn: The callable to run (typically the teacher module or a closure over it).
        enabled: False forces the eager path.
        backend: Graph backend; defaults to the real CUDA one (tests pass a fake).
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        enabled: bool = True,
        backend: Any | None = None,
    ):
        self.fn = fn
        self.enabled = enabled
        self.backend = backend if backend is not None else _CudaGraphBackend()
        self._key: tuple | None = None
        self._replay: Callable[[list[torch.Tensor]], Any] | None = None
        self._failed = False

    @property
    def active(self) -> bool:
        """True once a graph is captured and in use."""
        return self._replay is not None and not self._failed

    @staticmethod
    def _autocast_state() -> tuple:
        try:
            return (
                torch.is_autocast_enabled("cuda"),
                str(torch.get_autocast_dtype("cuda")) if torch.is_autocast_enabled("cuda") else None,
            )
        except (TypeError, RuntimeError):  # older torch without the device argument
            return (torch.is_autocast_enabled(), None)

    def _signature(self, inputs: tuple) -> tuple:
        return (
            tuple((tuple(t.shape), t.dtype, str(t.device)) for t in inputs),
            self._autocast_state(),
        )

    def __call__(self, *inputs: torch.Tensor) -> Any:
        if (
            not self.enabled
            or self._failed
            or torch.is_grad_enabled()
            or not inputs
            or not all(isinstance(t, torch.Tensor) and not t.requires_grad for t in inputs)
            or not self.backend.usable(list(inputs))
        ):
            return self.fn(*inputs)

        key = self._signature(inputs)
        if self._replay is None:
            try:
                self._replay = self.backend.capture(self.fn, list(inputs))
                self._key = key
            except Exception as exc:  # capture is best effort: never let it stop a run
                self._failed = True
                logger.warning("CUDA graph capture failed, running eagerly from now on: %s", exc)
                return self.fn(*inputs)
        elif key != self._key:
            return self.fn(*inputs)
        return self._detach(self._replay(list(inputs)))

    @staticmethod
    def _detach(outputs: Any) -> Any:
        leaves, spec = tree_flatten(outputs)
        return tree_unflatten(
            [leaf.clone() if isinstance(leaf, torch.Tensor) else leaf for leaf in leaves], spec
        )
