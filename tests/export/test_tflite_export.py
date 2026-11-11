"""End-to-end LiteRT export tests for the student (export venv only).

Skipped unless litert-torch and ai-edge-litert are installed, so the main
training venv's suite stays green without the export toolchain. Runs one
conversion of a random-init student through scripts/export_student.py and
reuses it for every test in the module.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

pytest.importorskip("litert_torch")
pytest.importorskip("ai_edge_litert")

from scripts import check_export_parity, export_student  # noqa: E402
from src.export.layout import BOXES_OUTPUT_NAME, CLASS_OUTPUT_NAME  # noqa: E402
from src.export.student_export import (  # noqa: E402
    StudentExportWrapper,
    make_tflite_runner,
    make_torch_runner,
    tflite_io_names,
)
from src.models.student.student_model import StudentModel  # noqa: E402

SEED = 3
N_ANCHORS = 8400


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> tuple[Path, dict]:
    out_dir = tmp_path_factory.mktemp("export")
    rc = export_student.main(["--random-init", "--seed", str(SEED), "--out-dir", str(out_dir)])
    assert rc == 0
    model_path = out_dir / "student_fp32.tflite"
    metadata = json.loads((out_dir / "student_fp32.json").read_text())
    return model_path, metadata


@pytest.fixture(scope="module")
def data_root(tmp_path_factory) -> Path:
    """Tiny dataset in the training layout: two val images with labels."""
    root = tmp_path_factory.mktemp("data")
    rgb_dir = root / "cache" / "mango" / "rgb"
    labels_dir = root / "annotations" / "yolo" / "labels" / "val"
    rgb_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for idx, (h, w) in enumerate([(480, 640), (720, 540)]):
        stem = f"mango_rgb_{idx:05d}"
        cv2.imwrite(str(rgb_dir / f"{stem}.jpg"), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    return root


def test_tflite_io_contract(exported) -> None:
    model_path, metadata = exported
    assert model_path.stat().st_size > 1_000_000
    io_names = tflite_io_names(model_path)
    assert io_names["signature_key"] == metadata["signature_key"]
    assert io_names["input"]["signature_name"] == metadata["input"]["signature_name"]

    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path))
    runner = interpreter.get_signature_runner(io_names["signature_key"])
    input_detail = runner.get_input_details()[io_names["input"]["signature_name"]]
    assert list(input_detail["shape"]) == [1, 640, 640, 3]
    assert input_detail["dtype"] == np.float32
    outputs = runner.get_output_details()
    assert list(outputs[io_names["outputs"][BOXES_OUTPUT_NAME]["signature_name"]]["shape"]) == [1, N_ANCHORS, 4]
    assert list(outputs[io_names["outputs"][CLASS_OUTPUT_NAME]["signature_name"]]["shape"]) == [1, N_ANCHORS, 2]


def test_only_builtin_ops(exported) -> None:
    """No Flex/custom ops: the model must run on the stock LiteRT runtime."""
    from ai_edge_litert.interpreter import Interpreter

    ops = {op["op_name"] for op in Interpreter(model_path=str(exported[0]))._get_ops_details()}
    assert not any(name.startswith("Flex") or name == "CUSTOM" for name in ops), ops


def test_raw_outputs_match_pytorch_on_random_input(exported) -> None:
    model_path, metadata = exported
    torch.manual_seed(SEED)
    wrapper = StudentExportWrapper(StudentModel().eval())
    nchw = torch.randn(1, 3, 640, 640)
    ref = make_torch_runner(wrapper)(nchw)
    io_names = {
        "signature_key": metadata["signature_key"],
        "input": metadata["input"],
        "outputs": {o["logical_name"]: o for o in metadata["outputs"]},
    }
    got = make_tflite_runner(model_path, io_names)(np.ascontiguousarray(nchw.permute(0, 2, 3, 1).numpy()))
    for r, g in zip(ref, got):
        assert r.shape == g.shape
        assert np.abs(r - g).max() < 1e-3


def test_a_segment_student_exports_four_outputs_that_match_pytorch(tmp_path) -> None:
    """UNVERIFIED in the training venv (no litert-torch there): runs in the export venv."""
    rc = export_student.main(["--random-init", "--seed", str(SEED), "--task", "segment", "--variant", "nano",
                              "--image-size", "64", "--out-dir", str(tmp_path)])
    assert rc == 0
    metadata = json.loads((tmp_path / "student_fp32.json").read_text())
    names = [o["logical_name"] for o in metadata["outputs"]]
    assert names == ["boxes_raw", "cls_logits", "mask_coefs", "protos"]
    torch.manual_seed(SEED)
    wrapper = StudentExportWrapper(StudentModel(variant="nano", task="segment").eval())
    nchw = torch.randn(1, 3, 64, 64)
    io_names = {
        "signature_key": metadata["signature_key"],
        "input": metadata["input"],
        "outputs": {o["logical_name"]: o for o in metadata["outputs"]},
    }
    got = make_tflite_runner(tmp_path / "student_fp32.tflite", io_names, names)(
        np.ascontiguousarray(nchw.permute(0, 2, 3, 1).numpy()))
    for ref, out in zip(make_torch_runner(wrapper)(nchw), got):
        assert ref.shape == out.shape and np.abs(ref - out).max() < 1e-3


def test_parity_cli_passes_on_matching_weights(exported, data_root, tmp_path) -> None:
    model_path, _ = exported
    report_path = tmp_path / "report.json"
    rc = check_export_parity.main([
        "--model", str(model_path), "--data-root", str(data_root),
        "--num-images", "5", "--report-json", str(report_path),
    ])
    report = json.loads(report_path.read_text())
    assert rc == 0, report["failures"]
    assert report["images"] == 2 and report["passed"]


def test_parity_cli_fails_on_different_weights(exported, data_root, tmp_path) -> None:
    model_path, _ = exported
    # Same seed, then shift one prediction bias: random-init head outputs are
    # near-constant (std=0.01 weights), so a different seed alone would not
    # move the raw outputs past the tolerance.
    torch.manual_seed(SEED)
    state = StudentModel().state_dict()
    state["head.heads.0.reg_pred.bias"] += 0.5
    other = tmp_path / "other.pt"
    torch.save(state, other)
    rc = check_export_parity.main([
        "--model", str(model_path), "--data-root", str(data_root), "--checkpoint", str(other),
    ])
    assert rc == 1
