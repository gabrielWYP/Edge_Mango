"""Segment student export: checkpoint loading, wrapper, sidecar, mask assembly and ONNX.

CPU only, a random-init `nano` student at image size 64 (no training, no GPU). The ONNX tests need the
`onnx` and `onnxruntime` packages and skip without them; everything else runs in the training venv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.export.layout import (
    BOXES_OUTPUT_NAME,
    CLASS_OUTPUT_NAME,
    MASK_COEFS_OUTPUT_NAME,
    PROTOS_OUTPUT_NAME,
    SEGMENT_OUTPUT_NAMES,
    flatten_levels,
)
from src.export.reference import decode_exported_outputs, decode_exported_segmentation
from src.export.student_export import (
    StudentExportWrapper,
    build_metadata,
    load_student_checkpoint,
    resolve_decode_settings,
)
from src.models.student.student_model import StudentModel
from src.training.decode import decode_detections_with_coefs, process_masks
from src.training.strides import STUDENT_STRIDES

SIZE = 64
N_ANCHORS = (SIZE // 8) ** 2 + (SIZE // 16) ** 2 + (SIZE // 32) ** 2  # 84
NM = 32


def _student(task: str = "segment", **kwargs) -> StudentModel:
    torch.manual_seed(0)
    return StudentModel(num_classes=2, variant="nano", task=task, **kwargs).eval()


def _save(path: Path, student: StudentModel, **config) -> Path:
    cfg = {"task": student.task, "student_variant": student.variant,
           "student_in_channels": student.in_channels, "num_classes": 2, **config}
    torch.save({"model_state_dict": student.state_dict(), "config": cfg}, path)
    return path


# --- checkpoint loading --------------------------------------------------------------------------


def test_a_segment_nano_checkpoint_loads_with_its_task_and_variant(tmp_path) -> None:
    loaded = load_student_checkpoint(_save(tmp_path / "seg.pt", _student("segment")))
    assert loaded.model.task == "segment" and loaded.model.variant == "nano"


def test_a_detect_nano_checkpoint_loads_too(tmp_path) -> None:
    loaded = load_student_checkpoint(_save(tmp_path / "det.pt", _student("detect")))
    assert loaded.model.task == "detect" and loaded.model.variant == "nano"


def test_a_checkpoint_without_the_new_fields_is_the_wide_detect_student(tmp_path) -> None:
    path = tmp_path / "old.pt"
    torch.save({"model_state_dict": StudentModel().state_dict(), "config": {"num_classes": 2}}, path)
    loaded = load_student_checkpoint(path)
    assert (loaded.model.task, loaded.model.variant, loaded.model.in_channels) == ("detect", "wide", 3)


def test_a_four_channel_student_cannot_be_exported(tmp_path) -> None:
    path = _save(tmp_path / "rgbnir.pt", _student("detect", in_channels=4))
    with pytest.raises(ValueError, match="in_channels"):
        load_student_checkpoint(path)


# --- wrapper -------------------------------------------------------------------------------------


def test_the_segment_wrapper_returns_boxes_logits_coefficients_and_prototypes() -> None:
    student = _student()
    x = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        out = StudentExportWrapper(student).eval()(x)
        raw = student(x)
    assert len(out) == 4
    boxes_raw, cls_logits, coefs, protos = out
    assert boxes_raw.shape == (1, N_ANCHORS, 4) and cls_logits.shape == (1, N_ANCHORS, 2)
    assert coefs.shape == (1, N_ANCHORS, NM) and protos.shape == (1, NM, SIZE // 4, SIZE // 4)
    assert torch.equal(coefs, flatten_levels(raw["mask_coefs"])) and torch.equal(protos, raw["protos"])


def test_the_detect_wrapper_still_returns_two_tensors() -> None:
    assert len(StudentExportWrapper(_student("detect")).eval()(torch.randn(1, 3, SIZE, SIZE))) == 2


# --- sidecar -------------------------------------------------------------------------------------


def _io(names) -> dict:
    return {"signature_key": "serving_default", "input": {"signature_name": "images", "tensor_name": "images"},
            "outputs": {n: {"signature_name": n, "tensor_name": n} for n in names}}


def _metadata(tmp_path: Path, student: StudentModel, names, fmt: str = "onnx") -> dict:
    model_file = tmp_path / f"student.{fmt}"
    model_file.write_bytes(b"x")
    return build_metadata(
        model_file=model_file, precision="fp32", image_size=SIZE, num_classes=2,
        decode=resolve_decode_settings({}), letterbox_value=114, io_names=_io(names), checkpoint=None,
        num_parameters=1, task=student.task, tools={"python": "3"},
    )


def test_the_segment_sidecar_lists_four_outputs_and_how_to_assemble_masks(tmp_path) -> None:
    meta = _metadata(tmp_path, _student(), SEGMENT_OUTPUT_NAMES)
    outputs = {o["logical_name"]: o for o in meta["outputs"]}
    assert list(outputs) == list(SEGMENT_OUTPUT_NAMES)
    assert outputs[MASK_COEFS_OUTPUT_NAME]["shape"] == [1, N_ANCHORS, NM]
    assert outputs[PROTOS_OUTPUT_NAME]["shape"] == [1, NM, SIZE // 4, SIZE // 4]
    seg = meta["segmentation"]
    assert seg["mask_ratio"] == 4 and seg["threshold"] == 0.5 and seg["num_prototypes"] == NM
    assert "sigmoid" in seg["assembly"] and "crop" in seg["assembly"]
    assert meta["task"] == "segment"


def test_the_detect_sidecar_is_unchanged_by_the_task_support(tmp_path) -> None:
    meta = _metadata(tmp_path, _student("detect"), (BOXES_OUTPUT_NAME, CLASS_OUTPUT_NAME))
    assert "segmentation" not in meta and "task" not in meta
    assert [o["logical_name"] for o in meta["outputs"]] == [BOXES_OUTPUT_NAME, CLASS_OUTPUT_NAME]


def test_an_onnx_sidecar_declares_an_nchw_input(tmp_path) -> None:
    meta = _metadata(tmp_path, _student(), SEGMENT_OUTPUT_NAMES, fmt="onnx")
    assert meta["input"]["layout"] == "NCHW" and meta["input"]["shape"] == [1, 3, SIZE, SIZE]
    assert any("NCHW" in step for step in meta["preprocessing"]["steps"])


# --- mask assembly from the raw outputs ----------------------------------------------------------


def test_assembled_masks_equal_the_training_decode_of_the_same_model() -> None:
    student = _student()
    x = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        raw = student(x)
        exported = StudentExportWrapper(student).eval()(x)
    kwargs = dict(image_size=SIZE, strides=STUDENT_STRIDES, conf_threshold=0.005, nms_iou_threshold=0.5,
                  max_detections=50, per_class_candidates=True)
    boxes, scores, labels, masks = decode_exported_segmentation(*exported, **kwargs)

    preds = [torch.cat([c, r], dim=1) for c, r in zip(raw["cls_preds"], raw["reg_preds"])]
    ref_boxes, ref_scores, ref_labels, coefs = decode_detections_with_coefs(
        preds, raw["cls_preds"], raw["mask_coefs"], batch_idx=0, num_classes=2, image_size=SIZE,
        strides=tuple(STUDENT_STRIDES), conf_threshold=0.005, nms_iou_threshold=0.5, nms_enabled=True,
        per_class_candidates=True, max_detections=50, normalize=False,
    )
    assert len(boxes) > 0
    assert torch.allclose(boxes, ref_boxes) and torch.equal(labels, ref_labels)
    expected = process_masks(raw["protos"][0], coefs, ref_boxes / SIZE)
    assert masks.dtype == torch.bool and masks.shape == (len(boxes), SIZE // 4, SIZE // 4)
    assert torch.equal(masks, expected)


def test_the_box_decode_of_a_segment_export_is_the_detect_decode() -> None:
    student = _student()
    with torch.no_grad():
        exported = StudentExportWrapper(student).eval()(torch.randn(1, 3, SIZE, SIZE))
    kwargs = dict(image_size=SIZE, strides=STUDENT_STRIDES, conf_threshold=0.005, nms_iou_threshold=0.5,
                  max_detections=50, per_class_candidates=True)
    boxes, scores, labels, _ = decode_exported_segmentation(*exported, **kwargs)
    plain = decode_exported_outputs(exported[0], exported[1], **kwargs)
    assert torch.equal(boxes, plain[0]) and torch.equal(scores, plain[1]) and torch.equal(labels, plain[2])


# --- ONNX ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def onnx_libs():
    pytest.importorskip("onnx")
    return pytest.importorskip("onnxruntime")


def test_the_onnx_outputs_match_pytorch_on_random_input(onnx_libs, tmp_path) -> None:
    from src.export.onnx_export import convert_to_onnx, make_onnx_runner, onnx_io_names

    student = _student()
    wrapper = StudentExportWrapper(student).eval()
    path = convert_to_onnx(wrapper, tmp_path / "seg.onnx", image_size=SIZE)
    assert onnx_io_names(wrapper)["outputs"].keys() == set(SEGMENT_OUTPUT_NAMES)

    run = make_onnx_runner(path)
    for seed in range(3):
        torch.manual_seed(seed)
        x = torch.randn(1, 3, SIZE, SIZE)
        with torch.no_grad():
            expected = wrapper(x)
        actual = run(x.numpy())
        assert len(actual) == 4
        for name, e, a in zip(SEGMENT_OUTPUT_NAMES, expected, actual):
            assert a.shape == tuple(e.shape), name
            np.testing.assert_allclose(a, e.numpy(), atol=1e-4, rtol=1e-4, err_msg=name)


def test_masks_assembled_from_the_onnx_outputs_equal_those_from_pytorch(onnx_libs, tmp_path) -> None:
    from src.export.onnx_export import convert_to_onnx, make_onnx_runner

    wrapper = StudentExportWrapper(_student()).eval()
    run = make_onnx_runner(convert_to_onnx(wrapper, tmp_path / "seg.onnx", image_size=SIZE))
    x = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        expected = wrapper(x)
    kwargs = dict(image_size=SIZE, strides=STUDENT_STRIDES, conf_threshold=0.005, nms_iou_threshold=0.5,
                  max_detections=50, per_class_candidates=True)
    ref = decode_exported_segmentation(*expected, **kwargs)
    got = decode_exported_segmentation(*run(x.numpy()), **kwargs)
    assert len(ref[0]) > 0 and torch.allclose(ref[0], got[0], atol=1e-3) and torch.equal(ref[2], got[2])
    assert torch.equal(ref[3], got[3])


def test_a_detect_student_exports_two_onnx_outputs(onnx_libs, tmp_path) -> None:
    from src.export.onnx_export import convert_to_onnx, make_onnx_runner

    wrapper = StudentExportWrapper(_student("detect")).eval()
    run = make_onnx_runner(convert_to_onnx(wrapper, tmp_path / "det.onnx", image_size=SIZE))
    x = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        expected = wrapper(x)
    actual = run(x.numpy())
    assert len(actual) == 2
    for e, a in zip(expected, actual):
        np.testing.assert_allclose(a, e.numpy(), atol=1e-4, rtol=1e-4)


def test_the_export_script_writes_an_onnx_model_and_its_sidecar_and_verifies_it(onnx_libs, tmp_path, capsys) -> None:
    from scripts import export_student

    checkpoint = _save(tmp_path / "seg.pt", _student(), image_size=SIZE)
    rc = export_student.main(["--checkpoint", str(checkpoint), "--format", "onnx", "--out-dir", str(tmp_path),
                              "--name", "seg", "--verify"])
    assert rc == 0
    meta = json.loads((tmp_path / "seg.json").read_text())
    assert (tmp_path / "seg.onnx").is_file()
    assert [o["logical_name"] for o in meta["outputs"]] == list(SEGMENT_OUTPUT_NAMES)
    assert meta["input"]["layout"] == "NCHW" and meta["task"] == "segment"
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] and set(report["raw"]) == set(SEGMENT_OUTPUT_NAMES)


def test_the_export_script_can_smoke_test_a_random_segment_student(onnx_libs, tmp_path) -> None:
    from scripts import export_student

    rc = export_student.main(["--random-init", "--task", "segment", "--variant", "nano", "--image-size", str(SIZE),
                              "--format", "onnx", "--out-dir", str(tmp_path), "--verify"])
    assert rc == 0 and (tmp_path / "student_fp32.onnx").is_file()
