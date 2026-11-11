"""YOLOv8-seg additions to the student: the Proto module and the mask coefficient branch."""

from __future__ import annotations

import pytest
import torch

from src.models.student.student_model import StudentModel

SIZE = 64


def test_detect_is_the_default_and_adds_no_parameters() -> None:
    detect = StudentModel(num_classes=2)
    assert detect.task == "detect"
    assert not any("proto" in key or "mask" in key for key in detect.state_dict())
    assert "protos" not in detect.eval()(torch.zeros(1, 3, SIZE, SIZE))


def test_unknown_task_is_refused() -> None:
    with pytest.raises(ValueError, match="task"):
        StudentModel(num_classes=2, task="pose")


@pytest.mark.parametrize("variant", ["wide", "nano"])
def test_segment_output_shapes(variant: str) -> None:
    model = StudentModel(num_classes=2, variant=variant, task="segment").eval()
    out = model(torch.zeros(2, 3, SIZE, SIZE))
    # Prototypes at 1/4 of the input, one coefficient vector per anchor at every scale.
    assert out["protos"].shape == (2, 32, SIZE // 4, SIZE // 4)
    assert [c.shape for c in out["mask_coefs"]] == [
        (2, 32, SIZE // 8, SIZE // 8),
        (2, 32, SIZE // 16, SIZE // 16),
        (2, 32, SIZE // 32, SIZE // 32),
    ]
    # The detection outputs are exactly the ones a detect student has.
    assert [p.shape[1] for p in out["preds"]] == [6, 6, 6]


def test_segment_keeps_every_detect_key_unchanged() -> None:
    detect = StudentModel(num_classes=2)
    segment = StudentModel(num_classes=2, task="segment")
    extra = set(segment.state_dict()) - set(detect.state_dict())
    assert extra and all(".proto." in f".{k}" or "mask_" in k for k in extra)
    # A detect checkpoint loads into a segment student (the new layers stay at their init).
    result = segment.load_state_dict(detect.state_dict(), strict=False)
    # (`num_batches_tracked` buffers are never reported as missing by PyTorch.)
    assert not result.unexpected_keys
    assert set(result.missing_keys) == {k for k in extra if not k.endswith("num_batches_tracked")}
    # And a segment checkpoint minus its mask layers loads into a detect student, strictly.
    detect.load_state_dict({k: v for k, v in segment.state_dict().items() if k not in extra})


def test_segment_backward_reaches_proto_and_coefficients() -> None:
    model = StudentModel(num_classes=2, task="segment").train()
    out = model(torch.randn(2, 3, SIZE, SIZE))
    (out["protos"].mean() + sum(c.mean() for c in out["mask_coefs"])).backward()
    for name, param in model.named_parameters():
        if "proto" in name or "mask_" in name:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
