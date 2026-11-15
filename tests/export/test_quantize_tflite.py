"""FP16 and static INT8 export of the student (export venv only).

Skipped without litert-torch / ai-edge-quantizer, so the training venv's suite stays green. One FP32,
one FP16 and one INT8 export of a random-init nano student at 64 px are shared by the module, written
by `scripts/export_student.py --precision`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("litert_torch")
pytest.importorskip("ai_edge_litert")
pytest.importorskip("ai_edge_quantizer")

from ai_edge_litert.interpreter import Interpreter  # noqa: E402

from scripts import export_student  # noqa: E402
from src.export.student_export import make_tflite_runner  # noqa: E402
from tests.export.test_quantize import SEED, SIZE, _make_dataset  # noqa: E402


@pytest.fixture(scope="module")
def exports(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("quant")
    rgb_dir, labels_dir = _make_dataset(root)
    out = {}
    for precision in ("fp32", "fp16", "int8"):
        out_dir = root / precision
        argv = ["--random-init", "--seed", str(SEED), "--variant", "nano", "--image-size", str(SIZE),
                "--precision", precision, "--out-dir", str(out_dir),
                "--rgb-dir", str(rgb_dir), "--labels-dir", str(labels_dir), "--calib-samples", "6"]
        assert export_student.main(argv) == 0
        model_path = out_dir / f"student_{precision}.tflite"
        out[precision] = (model_path, json.loads(model_path.with_suffix(".json").read_text()))
    return out


def _run(path_and_meta, nhwc):
    path, meta = path_and_meta
    io_names = {"signature_key": meta["signature_key"], "input": meta["input"],
                "outputs": {o["logical_name"]: o for o in meta["outputs"]}}
    return make_tflite_runner(path, io_names, [o["logical_name"] for o in meta["outputs"]])(nhwc)


def _nhwc(seed=0):
    return np.random.default_rng(seed).normal(size=(1, SIZE, SIZE, 3)).astype(np.float32)


@pytest.mark.parametrize("precision", ["fp16", "int8"])
def test_quantized_models_keep_float32_io(exports, precision) -> None:
    path, meta = exports[precision]
    runner = Interpreter(model_path=str(path)).get_signature_runner(meta["signature_key"])
    for detail in [*runner.get_input_details().values(), *runner.get_output_details().values()]:
        assert detail["dtype"] == np.float32
    assert meta["input"]["dtype"] == "float32" and meta["model"]["precision"] == precision
    assert all(o["dtype"] == "float32" for o in meta["outputs"])
    outputs = _run(exports[precision], _nhwc())
    assert all(o.dtype == np.float32 and np.isfinite(o).all() for o in outputs)


def test_int8_is_static_integer_and_much_smaller(exports) -> None:
    path, meta = exports["int8"]
    assert path.stat().st_size < 0.4 * exports["fp32"][0].stat().st_size
    interpreter = Interpreter(model_path=str(path))
    dtypes = {np.dtype(d["dtype"]).name for d in interpreter.get_tensor_details()}
    assert "int8" in dtypes
    ops = {op["op_name"] for op in interpreter._get_ops_details()}
    assert {"QUANTIZE", "DEQUANTIZE", "CONV_2D"} <= ops
    assert not any(name.startswith("Flex") or name == "CUSTOM" for name in ops)
    assert meta["quantization"]["method"].startswith("static")
    assert meta["quantization"]["activation_bits"] == 8 and meta["quantization"]["weight_bits"] == 8
    assert meta["calibration"]["split"] == "train" and meta["calibration"]["num_samples"] == 6


def test_fp16_halves_the_weights_and_stays_close_to_fp32(exports) -> None:
    fp32, fp16 = exports["fp32"], exports["fp16"]
    assert fp16[0].stat().st_size < 0.6 * fp32[0].stat().st_size
    nhwc = _nhwc(1)
    for ref, got in zip(_run(fp32, nhwc), _run(fp16, nhwc)):
        assert np.abs(ref - got).max() < 5e-2 * max(1.0, np.abs(ref).max())


def test_int8_tracks_fp32_on_a_network_with_real_dynamic_range(tmp_path) -> None:
    """The random-init student's heads output near-constants (weights ~0.01), where INT8 error is not
    meaningful; a small convnet with ordinary weights shows the calibrate -> quantize -> float-IO chain is right."""
    import litert_torch
    import torch.nn as nn

    from src.export.quantize import quantize_int8
    from src.export.student_export import tflite_io_names

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.c = nn.Conv2d(3, 16, 3, 2, 1), nn.Conv2d(16, 8, 3, 1, 1), nn.Conv2d(8, 4, 1)

        def forward(self, x):
            hidden = torch.relu(self.b(torch.relu(self.a(x))))
            return self.c(hidden).flatten(1), hidden

    torch.manual_seed(0)
    module = litert_torch.to_channel_last_io(Tiny().eval(), args=[0]).eval()
    float_path = tmp_path / "tiny_fp32.tflite"
    litert_torch.convert(module, (torch.zeros(1, 32, 32, 3),)).export(str(float_path))
    names = tflite_io_names(float_path, ("o0", "o1"))
    rng = np.random.default_rng(0)
    calibration = [rng.normal(size=(1, 32, 32, 3)).astype(np.float32) for _ in range(20)]
    int8_path = tmp_path / "tiny_int8.tflite"
    info = quantize_int8(float_path, int8_path, calibration,
                         signature_key=names["signature_key"], input_name=names["input"]["signature_name"])
    assert info["internal_interface"]["input"][0]["dtype"] == "int8" and len(info["internal_interface"]["outputs"]) == 2
    x = rng.normal(size=(1, 32, 32, 3)).astype(np.float32)
    reference = make_tflite_runner(float_path, names, ("o0", "o1"))(x)
    got = make_tflite_runner(int8_path, tflite_io_names(int8_path, ("o0", "o1")), ("o0", "o1"))(x)
    for ref, out in zip(reference, got):
        assert out.dtype == np.float32
        assert np.linalg.norm(ref - out) < 0.05 * np.linalg.norm(ref)  # relative L2 error of an 8-bit network


def test_sidecar_records_complexity_and_roi(exports) -> None:
    for precision, (path, meta) in exports.items():
        assert meta["complexity"]["parameters"] == meta["model"]["num_parameters"]
        assert meta["complexity"]["flops"] > 0 and meta["complexity"]["flops_input"] == [1, 3, SIZE, SIZE]
        assert meta["model"]["file_size_bytes"] == path.stat().st_size
        roi = meta["preprocessing"]["roi_crop"]
        assert roi["enabled"] is True and roi["margin"] == pytest.approx(0.10)
