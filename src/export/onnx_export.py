"""ONNX export of the student (detect or segment) and its CPU parity check.

The graph is `StudentExportWrapper`'s: raw head outputs only, with the input NCHW float32 `images`
(ONNX keeps PyTorch's layout; only the LiteRT file is channel-last). Outputs, in order and named:

    detect:   boxes_raw (1, N, 4), cls_logits (1, N, nc)
    segment:  the two above, mask_coefs (1, N, nm), protos (1, nm, H/4, W/4)

As for the .tflite, box decode, NMS and mask assembly stay outside the graph (YOLOv8-seg style);
`src/export/reference.py` is the executable spec and docs/deployment/mobile-export.md the description.

Needs the `onnx` package to export (the TorchScript exporter serializes through it) and
`onnxruntime` to run; both are imported lazily so this module imports in the training venv.
The legacy exporter (`dynamo=False`) is used: it needs no `onnxscript` and has been exact (<1e-5)
on this network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.export.parity import DEFAULT_MAX_ABS_DIFF, DEFAULT_MIN_MATCH_RATE, ParityReport, match_detections
from src.export.reference import decode_exported_outputs
from src.export.student_export import StudentExportWrapper, resolve_decode_settings
from src.training.strides import STUDENT_STRIDES

ONNX_INPUT_NAME = "images"
ONNX_OPSET = 17


def convert_to_onnx(wrapper: StudentExportWrapper, out_path: str | Path, *, image_size: int) -> Path:
    """Export the wrapper to a static-shape ONNX file (batch 1, NCHW input) and return its path."""
    try:
        import onnx  # noqa: F401  (torch.onnx.export needs it)
    except ModuleNotFoundError as error:
        raise RuntimeError("ONNX export needs the `onnx` package (pip install onnx onnxruntime).") from error

    wrapper = wrapper.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (sample,),
            str(out_path),
            input_names=[ONNX_INPUT_NAME],
            output_names=list(wrapper.output_names),
            opset_version=ONNX_OPSET,
            dynamo=False,
        )
    return out_path


def onnx_io_names(wrapper: StudentExportWrapper) -> dict:
    """The `io_names` of `build_metadata` for an ONNX file: tensor names are the graph's own."""
    return {
        "signature_key": None,
        "input": {"signature_name": ONNX_INPUT_NAME, "tensor_name": ONNX_INPUT_NAME},
        "outputs": {name: {"signature_name": name, "tensor_name": name} for name in wrapper.output_names},
    }


def make_onnx_runner(model_path: str | Path) -> Callable[[np.ndarray], tuple[np.ndarray, ...]]:
    """`fn(nchw_float32) -> outputs in graph order`, one CPU onnxruntime session reused across calls."""
    import onnxruntime  # lazy

    session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    names = [output.name for output in session.get_outputs()]

    def run(nchw_input: np.ndarray) -> tuple[np.ndarray, ...]:
        return tuple(session.run(names, {ONNX_INPUT_NAME: np.ascontiguousarray(nchw_input, dtype=np.float32)}))

    return run


def check_onnx_parity(
    wrapper: StudentExportWrapper,
    model_path: str | Path,
    *,
    image_size: int,
    checkpoint_config: dict | None = None,
    num_inputs: int = 3,
    seed: int = 0,
    max_abs_diff: float = DEFAULT_MAX_ABS_DIFF,
    min_match_rate: float = DEFAULT_MIN_MATCH_RATE,
) -> ParityReport:
    """PyTorch wrapper vs the ONNX file on random normal inputs (no dataset needed).

    Every output is compared raw (max abs diff below `max_abs_diff`); the box outputs are also decoded
    with the reference decode and matched, as the .tflite parity does.
    """
    wrapper = wrapper.eval()
    run = make_onnx_runner(model_path)
    decode = resolve_decode_settings(checkpoint_config or {})
    kwargs = dict(
        image_size=image_size,
        strides=STUDENT_STRIDES,
        conf_threshold=decode.conf_threshold,
        nms_iou_threshold=decode.nms_iou_threshold,
        max_detections=decode.max_detections,
        per_class_candidates=decode.per_class_candidates,
    )
    report = ParityReport(max_abs_diff_threshold=max_abs_diff, min_match_rate=min_match_rate)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(num_inputs):
        x = torch.randn(1, 3, image_size, image_size, generator=generator)
        with torch.no_grad():
            expected = [t.numpy() for t in wrapper(x)]
        actual = run(x.numpy())
        for name, e, a in zip(wrapper.output_names, expected, actual):
            report.add_raw(name, e, a)
        ref_boxes, _, ref_labels = decode_exported_outputs(expected[0], expected[1], **kwargs)
        got_boxes, _, got_labels = decode_exported_outputs(actual[0], actual[1], **kwargs)
        report.add_detections(match_detections(ref_boxes, ref_labels, got_boxes, got_labels, report.match_iou))
    return report
