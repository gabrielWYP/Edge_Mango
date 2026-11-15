"""CUDA-graph wrapper for the frozen teacher and the cudnn.benchmark entrypoint helper.

CPU only. The graph machinery sits behind a backend object, so the wrapper's decisions
(capture once, replay on a matching input, eager otherwise, fall back once on a failed capture)
are tested here with a fake backend; the real CUDA backend is covered by
`scripts/check_gpu_speedups.py` on a GPU.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from src.training import perf
from src.training.perf import GraphedForward, enable_cudnn_benchmark


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.calls = 0

    def forward(self, x, y=None):
        self.calls += 1
        out = {"feat": [self.lin(x), self.lin(x) * 2], "pair": (x.sum(), "tag")}
        if y is not None:
            out["y"] = y + 1
        return out


class _FakeBackend:
    """Pretends to be CUDA: 'captures' by remembering static buffers, replays by re-running fn."""

    def __init__(self, fail_capture: bool = False):
        self.fail_capture = fail_capture
        self.captures = 0
        self.replays = 0

    def usable(self, tensors) -> bool:
        return True

    def capture(self, fn, example_inputs):
        if self.fail_capture:
            raise RuntimeError("capture refused")
        self.captures += 1
        static_in = [t.clone() for t in example_inputs]
        static_out = fn(*static_in)

        def replay(inputs):
            self.replays += 1
            for dst, src in zip(static_in, inputs):
                dst.copy_(src)
            fresh = fn(*static_in)
            leaves_out, spec = perf.tree_flatten(static_out)
            leaves_new, _ = perf.tree_flatten(fresh)
            for dst, new in zip(leaves_out, leaves_new):
                if isinstance(dst, torch.Tensor):
                    dst.copy_(new)
            return static_out

        return replay


def _assert_same(a, b) -> None:
    la, sa = perf.tree_flatten(a)
    lb, sb = perf.tree_flatten(b)
    assert sa == sb
    for x, y in zip(la, lb):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y)
        else:
            assert x == y


# --- CPU: always eager ----------------------------------------------------------------------------


def test_on_cpu_the_wrapper_is_the_eager_call() -> None:
    teacher = _Teacher().eval()
    graphed = GraphedForward(teacher)
    x = torch.randn(2, 4)
    with torch.no_grad():
        _assert_same(graphed(x), teacher(x))
    assert not graphed.active


# --- decisions, with a fake CUDA backend -----------------------------------------------------------


def _graphed(backend: _FakeBackend, teacher: nn.Module) -> GraphedForward:
    return GraphedForward(teacher, backend=backend)


def test_captures_once_then_replays_and_matches_eager() -> None:
    teacher, backend = _Teacher().eval(), _FakeBackend()
    graphed = _graphed(backend, teacher)
    with torch.no_grad():
        for seed in range(3):
            x = torch.randn(2, 4, generator=torch.Generator().manual_seed(seed))
            _assert_same(graphed(x), teacher(x))
    assert backend.captures == 1 and backend.replays == 3 and graphed.active


def test_outputs_are_cloned_so_two_calls_do_not_alias() -> None:
    teacher, backend = _Teacher().eval(), _FakeBackend()
    graphed = _graphed(backend, teacher)
    with torch.no_grad():
        first = graphed(torch.ones(2, 4))
        second = graphed(torch.zeros(2, 4))
        third = graphed(torch.ones(2, 4))
    assert not torch.equal(first["feat"][0], second["feat"][0])
    assert torch.equal(first["feat"][0], third["feat"][0])
    assert first["feat"][0].data_ptr() != third["feat"][0].data_ptr()


def test_a_new_input_shape_runs_eagerly_without_recapturing() -> None:
    teacher, backend = _Teacher().eval(), _FakeBackend()
    graphed = _graphed(backend, teacher)
    with torch.no_grad():
        graphed(torch.randn(4, 4))
        partial = torch.randn(3, 4)  # the last, smaller batch
        _assert_same(graphed(partial), teacher(partial))
        graphed(torch.randn(4, 4))
    assert backend.captures == 1 and backend.replays == 2


def test_grad_mode_and_requires_grad_inputs_stay_eager() -> None:
    teacher, backend = _Teacher().eval(), _FakeBackend()
    graphed = _graphed(backend, teacher)
    x = torch.randn(2, 4, requires_grad=True)
    out = graphed(x)  # grad enabled
    assert out["feat"][0].requires_grad and backend.captures == 0
    with torch.no_grad():
        graphed(x)
    assert backend.captures == 0


def test_a_failed_capture_falls_back_to_eager_and_logs_once(caplog) -> None:
    teacher, backend = _Teacher().eval(), _FakeBackend(fail_capture=True)
    graphed = _graphed(backend, teacher)
    with caplog.at_level(logging.WARNING), torch.no_grad():
        for _ in range(3):
            x = torch.randn(2, 4)
            _assert_same(graphed(x), teacher(x))
    assert not graphed.active
    assert len([r for r in caplog.records if "CUDA graph" in r.getMessage()]) == 1


def test_inputs_of_different_dtype_or_arity_do_not_share_a_graph() -> None:
    teacher, backend = _Teacher().eval(), _FakeBackend()
    graphed = _graphed(backend, teacher)
    with torch.no_grad():
        graphed(torch.randn(2, 4))
        out = graphed(torch.randn(2, 4), torch.randn(2, 4))  # other signature: eager
        assert "y" in out
        graphed(torch.randn(2, 4))
    assert backend.captures == 1


def test_non_tensor_outputs_pass_through() -> None:
    teacher, backend = _Teacher().eval(), _FakeBackend()
    graphed = _graphed(backend, teacher)
    with torch.no_grad():
        graphed(torch.randn(2, 4))
        out = graphed(torch.randn(2, 4))
    assert out["pair"][1] == "tag"


# --- cudnn.benchmark -------------------------------------------------------------------------------


@pytest.fixture
def restore_benchmark():
    previous = torch.backends.cudnn.benchmark
    yield
    torch.backends.cudnn.benchmark = previous


@pytest.mark.parametrize("device, available, expected", [
    ("auto", True, True),
    ("cuda", True, True),
    ("cuda:0", True, True),
    ("auto", False, False),
    ("cpu", True, False),
])
def test_cudnn_benchmark_is_set_only_on_cuda(monkeypatch, restore_benchmark, device, available, expected) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    torch.backends.cudnn.benchmark = False
    assert enable_cudnn_benchmark(device) is expected
    assert torch.backends.cudnn.benchmark is expected


def test_both_entrypoints_call_the_helper() -> None:
    import inspect

    from src.training import kd_train, train

    for module in (train, kd_train):
        assert module.enable_cudnn_benchmark is enable_cudnn_benchmark
        assert "enable_cudnn_benchmark(config.device)" in inspect.getsource(module.main)
