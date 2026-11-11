"""yolov8n-seg official weights initialise the proto and mask-coefficient layers (key mapping only)."""

from __future__ import annotations

import pytest
import torch

from src.models.student.pretrained import load_yolov8n_weights, official_key
from src.models.student.student_model import StudentModel


@pytest.mark.parametrize(
    "key, expected",
    [
        ("head.proto.cv1.conv.weight", "model.22.proto.cv1.conv.weight"),
        ("head.proto.upsample.weight", "model.22.proto.upsample.weight"),
        ("head.proto.upsample.bias", "model.22.proto.upsample.bias"),
        ("head.proto.cv3.bn.running_var", "model.22.proto.cv3.bn.running_var"),
        ("head.heads.1.mask_stem.0.conv.weight", "model.22.cv4.1.0.conv.weight"),
        ("head.heads.2.mask_stem.1.bn.bias", "model.22.cv4.2.1.bn.bias"),
        ("head.heads.0.mask_pred.weight", "model.22.cv4.0.2.weight"),
        ("head.heads.0.mask_pred.bias", "model.22.cv4.0.2.bias"),
        # Unchanged detection mapping.
        ("head.heads.0.cls_stem.0.conv.weight", "model.22.cv3.0.0.conv.weight"),
    ],
)
def test_official_key_maps_the_seg_layers_of_the_nano_variant(key: str, expected: str) -> None:
    assert official_key(key, "nano") == expected


def test_the_wide_variant_has_no_seg_counterpart() -> None:
    assert official_key("head.proto.cv1.conv.weight", "wide") is None
    assert official_key("head.heads.0.mask_pred.weight", "wide") is None


def _fake_official_checkpoint(student: StudentModel, path) -> dict[str, torch.Tensor]:
    """A state dict named like yolov8n-seg whose mapped tensors have the student's shapes."""
    official = {}
    for index, key in enumerate(student.state_dict()):
        name = official_key(key, student.variant)
        if name is not None:
            official[name] = torch.full(student.state_dict()[key].shape, float(index % 7 + 1))
    torch.save(official, path)
    return official


def test_loading_a_seg_checkpoint_fills_proto_and_mask_layers(tmp_path) -> None:
    student = StudentModel(num_classes=2, variant="nano", task="segment")
    path = tmp_path / "yolov8n-seg.pt"
    official = _fake_official_checkpoint(student, path)
    report = load_yolov8n_weights(student, path)
    state = student.state_dict()
    for key in ("head.proto.cv1.conv.weight", "head.proto.upsample.bias", "head.heads.2.mask_pred.weight"):
        assert torch.equal(state[key], official[official_key(key, "nano")]), key
    skipped = set(report["skipped"])
    assert "head.proto.cv1.conv.weight" not in skipped
    assert "head.heads.0.mask_pred.weight" not in skipped


def test_a_detect_student_ignores_the_seg_tensors_of_the_official_file(tmp_path) -> None:
    seg = StudentModel(num_classes=2, variant="nano", task="segment")
    path = tmp_path / "yolov8n-seg.pt"
    _fake_official_checkpoint(seg, path)
    detect = StudentModel(num_classes=2, variant="nano")
    report = load_yolov8n_weights(detect, path)
    assert report["loaded"] > 0 and not any("proto" in k or "mask_" in k for k in report["skipped"])
