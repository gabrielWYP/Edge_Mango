"""Converting an official Ultralytics checkpoint into the plain state_dict the student loader reads."""

import sys
import types
from pathlib import Path

import torch
from torch import nn

from scripts.convert_ultralytics_weights import convert


def _fake_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Pickle a checkpoint whose model classes live in a fake `ultralytics` package."""
    pkg = types.ModuleType("ultralytics")
    tasks = types.ModuleType("ultralytics.nn.tasks")

    class SegmentationModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))

    SegmentationModel.__module__ = "ultralytics.nn.tasks"
    SegmentationModel.__qualname__ = "SegmentationModel"
    tasks.SegmentationModel = SegmentationModel
    sys.modules["ultralytics"] = pkg
    sys.modules["ultralytics.nn"] = types.ModuleType("ultralytics.nn")
    sys.modules["ultralytics.nn.tasks"] = tasks
    try:
        model = SegmentationModel().half()
        torch.save({"model": model, "epoch": -1}, path)
        expected = {
            k: (v.float() if v.is_floating_point() else v).clone() for k, v in model.state_dict().items()
        }
    finally:
        for name in ("ultralytics.nn.tasks", "ultralytics.nn", "ultralytics"):
            sys.modules.pop(name, None)
    return expected


def test_converts_without_the_ultralytics_package(tmp_path: Path) -> None:
    src, dst = tmp_path / "seg.pt", tmp_path / "seg_state.pt"
    expected = _fake_checkpoint(src)
    assert "ultralytics" not in sys.modules

    keys = convert(src, dst)

    state = torch.load(dst, map_location="cpu", weights_only=True)
    assert set(state) == set(expected) == set(keys)
    for name, tensor in expected.items():
        assert torch.equal(state[name], tensor)
        if tensor.is_floating_point():
            assert state[name].dtype == torch.float32
    assert "ultralytics" not in sys.modules


def test_keys_follow_the_official_model_index_names(tmp_path: Path) -> None:
    src, dst = tmp_path / "seg.pt", tmp_path / "seg_state.pt"
    _fake_checkpoint(src)
    keys = convert(src, dst)
    assert keys and all(k.startswith("model.") for k in keys)
