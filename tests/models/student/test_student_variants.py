"""The nano student variant and the YOLOv8n weight loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.student.pretrained import load_yolov8n_weights, official_key
from src.models.student.student_model import StudentModel
from src.training.config import TrainingConfig


def _official_state(model: StudentModel, cls_channels: int = 80) -> dict[str, torch.Tensor]:
    """A checkpoint with the official names and shapes, each tensor filled with a known value."""
    state = {}
    for key, value in model.state_dict().items():
        name = official_key(key, model.variant)
        if name is None:
            continue
        shape = list(value.shape)
        if ".cv3." in name:
            shape = [cls_channels if s == 64 else s for s in shape]
        state[name] = torch.full(shape, 0.5, dtype=value.dtype)
    return state


def test_wide_keeps_the_parameters_and_state_of_the_current_student() -> None:
    model = StudentModel(num_classes=2)
    assert model.variant == "wide"
    assert model.count_parameters()["total"] == 6_872_546


def test_nano_has_the_backbone_of_wide_and_the_size_of_yolov8n() -> None:
    nano, wide = StudentModel(num_classes=2, variant="nano"), StudentModel(num_classes=2)
    assert nano.count_parameters()["backbone"] == wide.count_parameters()["backbone"]
    assert nano.count_parameters()["neck"] == 986_880
    # Official nc=2 head without its 64-channel DFL outputs, which become 4 direct box values.
    assert nano.count_parameters()["total"] < 3_011_238


def test_nano_returns_the_shapes_every_consumer_reads() -> None:
    out = StudentModel(num_classes=2, variant="nano")(torch.zeros(1, 3, 640, 640))
    assert [p.shape[1:] for p in out["preds"]] == [(6, 80, 80), (6, 40, 40), (6, 20, 20)]
    assert [f.shape[1] for f in out["distill_head_cls"]] == [64, 64, 64]


def test_unknown_variant_is_refused() -> None:
    with pytest.raises(ValueError, match="variant"):
        StudentModel(num_classes=2, variant="small")


@pytest.mark.parametrize("variant, covered", [("wide", "backbone"), ("nano", "all but cls")])
def test_loader_copies_what_matches_and_reports_the_rest(tmp_path: Path, variant, covered) -> None:
    model = StudentModel(num_classes=2, variant=variant)
    path = tmp_path / "state.pt"
    torch.save(_official_state(model), path)

    report = load_yolov8n_weights(model, path)

    assert (model.backbone.stem[0].conv.weight == 0.5).all()
    assert report["loaded"] > 0 and report["skipped"]
    assert not any(key.startswith("backbone.") for key in report["skipped"])
    if covered == "backbone":
        assert all(not key.startswith("backbone.") for key in report["skipped"])
        assert (model.neck.lateral_p3.conv.weight != 0.5).any()
    else:
        assert (model.neck.td_c2f_3.cv1.conv.weight == 0.5).all()
        assert (model.head.heads[0].reg_stem[0].conv.weight == 0.5).all()
        assert (model.head.heads[0].cls_stem[0].conv.weight != 0.5).any()
    assert (model.head.heads[0].cls_pred.weight != 0.5).any()


def test_loader_refuses_a_backbone_that_does_not_match(tmp_path: Path) -> None:
    model = StudentModel(num_classes=2)
    state = _official_state(model)
    state["model.0.conv.weight"] = torch.zeros(8, 3, 3, 3)
    path = tmp_path / "state.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="backbone"):
        load_yolov8n_weights(model, path)


def test_config_defaults_keep_todays_student_and_refuse_a_typo() -> None:
    config = TrainingConfig()
    assert config.student_variant == "wide" and config.student_pretrained_path == ""
    with pytest.raises(ValueError, match="student_variant"):
        TrainingConfig(student_variant="small")
