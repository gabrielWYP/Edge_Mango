"""The first stem conv of each head branch is exposed for cross-head distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.export.layout import flatten_levels
from src.export.student_export import StudentExportWrapper
from src.models.student.head import DecoupledHead
from src.models.student.student_model import StudentModel

MID_KEYS = ("distill_head_cls_mid", "distill_head_reg_mid")


def test_decoupled_head_mid_is_the_first_stem_conv_before_its_activation() -> None:
    head = DecoupledHead(c_in=8, c_stem=16).eval()
    x = torch.randn(2, 8, 6, 6)
    with torch.no_grad():
        out = head(x)
        assert torch.equal(F.silu(out["distill_cls_mid"]), head.cls_stem[0](x))
        assert torch.equal(F.silu(out["distill_reg_mid"]), head.reg_stem[0](x))
        assert not torch.equal(out["distill_cls_mid"], head.cls_stem[0](x))
        assert torch.equal(head.cls_stem[1](F.silu(out["distill_cls_mid"])), out["distill_cls"])
        assert torch.equal(head.reg_stem[1](F.silu(out["distill_reg_mid"])), out["distill_reg"])


def test_student_exposes_mid_features_per_level_finest_first() -> None:
    model = StudentModel().eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    for key, full in zip(MID_KEYS, ("distill_head_cls", "distill_head_reg")):
        assert len(out[key]) == 3
        assert [t.shape for t in out[key]] == [t.shape for t in out[full]]
        assert [t.shape[1] for t in out[key]] == [64, 128, 256]


def test_existing_outputs_are_unchanged_by_the_mid_outputs() -> None:
    model = StudentModel().eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
        for level, head in enumerate(model.head.heads):
            assert torch.equal(
                head.cls_stem[1](F.silu(out["distill_head_cls_mid"][level])), out["distill_head_cls"][level]
            )
            assert torch.equal(
                head.cls_pred(out["distill_head_cls"][level]), out["cls_preds"][level]
            )


def test_export_wrapper_returns_only_the_two_raw_tensors_from_the_unchanged_heads() -> None:
    model = StudentModel().eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        result = StudentExportWrapper(model)(x)
        out = model(x)
    assert isinstance(result, tuple) and len(result) == 2
    assert torch.equal(result[0], flatten_levels(out["reg_preds"]))
    assert torch.equal(result[1], flatten_levels(out["cls_preds"]))
