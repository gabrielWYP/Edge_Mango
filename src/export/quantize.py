"""FP16 and static INT8 post-training quantization of an exported FP32 student `.tflite`.

Both start from the FP32 file litert-torch wrote, so one conversion feeds all three precisions:

- `quantize_int8`: static PTQ with ai-edge-quantizer (int8 weights, per-channel symmetric; int8
  activations, per-tensor asymmetric; min/max calibration) on representative inputs. The tool
  quantizes the graph input and outputs too (they become int8 tensors), so `add_float_io` then puts a
  QUANTIZE op after a float32 input and a DEQUANTIZE op before each float32 output. The app contract
  of `docs/deployment/mobile-export.md` (float32 NHWC in, float32 raw tensors out) does not change.
- `weights_to_fp16`: stores every float32 weight constant as float16 plus a DEQUANTIZE op in front of
  its consumers (the TFLite "float16 quantization" scheme: half the file, the same float32 graph).
  ai-edge-quantizer 0.9.0 has no float16 recipe, hence the direct flatbuffer edit.

The calibration inputs are built exactly like the training inputs (`reference.preprocess_roi_bgr`:
ROI crop to the label's mango box + margin, letterbox, ImageNet normalization) from images of the
fold's TRAIN split. `model_complexity` counts parameters and FLOPs.

`ai_edge_quantizer` / `ai_edge_litert` are imported lazily, so the calibration helpers and
`model_complexity` work in the training venv.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np
import torch

from src.export.reference import preprocess_roi_bgr

# Static PTQ recipe: int8 weights (per output channel, symmetric), int8 activations (per tensor,
# asymmetric), integer compute. Same as ai-edge-quantizer's `default_a8w8_recipe.json`.
INT8_RECIPE: list[dict[str, Any]] = [
    {
        "regex": ".*",
        "operation": "*",
        "algorithm_key": "min_max_uniform_quantize",
        "op_config": {
            "activation_tensor_config": {
                "num_bits": 8, "symmetric": False, "granularity": "TENSORWISE", "dtype": "INT",
            },
            "weight_tensor_config": {
                "num_bits": 8, "symmetric": True, "granularity": "CHANNELWISE", "dtype": "INT",
            },
            "compute_precision": "INTEGER",
            "explicit_dequantize": False,
            "skip_checks": False,
            "min_weight_elements": 0,
        },
    }
]

# Float32 constants smaller than this stay float32 in the FP16 file (scalars, tiny shape-like tensors).
FP16_MIN_ELEMENTS = 16


# --- representative inputs -------------------------------------------------------------------------


def select_calibration_samples(
    rgb_dir: str | Path, labels_dir: str | Path, split: str = "train", n: int = 200, seed: int = 0
) -> list[tuple[Path, Path]]:
    """`(image, label)` pairs of `split`: a seeded sample of `n`, or the whole split if it is smaller.

    A split is the label files under `labels_dir/<split>/` (`<fold>/labels` of the v4 folds), the
    image is `rgb_dir/<stem>.jpg`, as in `YOLODataset._load_pairs`. The result is sorted by stem.
    """
    split_dir = Path(labels_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Label split directory not found: {split_dir}")
    pairs = [
        (Path(rgb_dir) / f"{label.stem}.jpg", label)
        for label in sorted(split_dir.glob("*.txt"))
        if (Path(rgb_dir) / f"{label.stem}.jpg").exists()
    ]
    if not pairs:
        raise FileNotFoundError(f"No '{split}' images with labels under {rgb_dir} / {labels_dir}.")
    if n < len(pairs):
        pairs = [pairs[i] for i in sorted(random.Random(seed).sample(range(len(pairs)), n))]
    return pairs


class CalibrationInputs:
    """Re-iterable stream of `(1, S, S, 3)` float32 NHWC arrays, one per sample, built when iterated.

    ai-edge-quantizer walks the calibration data twice (a size pass, then the calibration), so a
    one-shot generator would arrive empty the second time.
    """

    def __init__(self, samples: Sequence[tuple[Path, Path]], image_size: int, pad_value: int, roi_margin: float | None):
        self.samples = list(samples)
        self.image_size, self.pad_value, self.roi_margin = image_size, pad_value, roi_margin

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[np.ndarray]:
        for image_path, label_path in self.samples:
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            tensor, _ = preprocess_roi_bgr(image, label_path, self.image_size, self.pad_value, self.roi_margin)
            yield np.ascontiguousarray(tensor.permute(0, 2, 3, 1).numpy(), dtype=np.float32)


def calibration_inputs(
    samples: Sequence[tuple[Path, Path]],
    image_size: int = 640,
    pad_value: int = 114,
    roi_margin: float | None = 0.10,
) -> CalibrationInputs:
    """The `(1, S, S, 3)` float32 NHWC calibration arrays of `samples`, as the exported model takes them."""
    return CalibrationInputs(samples, image_size, pad_value, roi_margin)


class _SignatureInputs:
    """`inputs` as the `{input_name: array}` dicts the quantizer feeds to one signature (re-iterable)."""

    def __init__(self, inputs: Iterable[np.ndarray], input_name: str):
        self.inputs = inputs if iter(inputs) is not inputs else list(inputs)
        self.input_name = input_name

    def __iter__(self):
        return ({self.input_name: array} for array in self.inputs)


# --- complexity ------------------------------------------------------------------------------------


def model_complexity(wrapper: torch.nn.Module, image_size: int = 640) -> dict[str, Any]:
    """Parameters and FLOPs of one forward pass at `image_size` (batch 1).

    FLOPs are counted by `torch.utils.flop_counter.FlopCounterMode` over the export wrapper (the
    convolutions and matmuls the graph runs; a multiply-accumulate counts as 2 FLOPs). Elementwise ops
    are not counted, the usual convention for detector FLOPs.
    """
    from torch.utils.flop_counter import FlopCounterMode

    wrapper = wrapper.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        wrapper(torch.zeros(1, 3, image_size, image_size))
    return {
        "parameters": sum(p.numel() for p in wrapper.parameters()),
        "flops": int(counter.get_total_flops()),
        "flops_tool": "torch.utils.flop_counter.FlopCounterMode (conv + matmul, 1 MAC = 2 FLOPs)",
        "flops_input": [1, 3, image_size, image_size],
    }


# --- flatbuffer helpers ----------------------------------------------------------------------------


def _read(model: bytes | bytearray):
    from ai_edge_litert.tools import flatbuffer_utils  # lazy

    return flatbuffer_utils.read_model_from_bytearray(bytearray(model))


def _write(model) -> bytes:
    from ai_edge_litert.tools import flatbuffer_utils  # lazy

    return bytes(flatbuffer_utils.convert_object_to_bytearray(model))


def _op_code_index(model, builtin_code: int) -> int:
    from ai_edge_quantizer import qtyping  # lazy

    for index, code in enumerate(model.operatorCodes):
        if code.builtinCode == builtin_code:
            return index
    code = qtyping.OperatorCodeT()
    code.builtinCode = builtin_code
    code.deprecatedBuiltinCode = min(builtin_code, 127)
    code.version = 1
    model.operatorCodes.append(code)
    return len(model.operatorCodes) - 1


def _new_tensor(subgraph, name: bytes, shape, tensor_type: int) -> int:
    from ai_edge_quantizer import qtyping  # lazy

    tensor = qtyping.TensorT()
    tensor.name, tensor.shape, tensor.type, tensor.buffer = name, list(shape), tensor_type, 0
    subgraph.tensors.append(tensor)
    return len(subgraph.tensors) - 1


def _quant_params(tensor) -> dict[str, Any] | None:
    q = tensor.quantization
    if q is None or q.scale is None or len(q.scale) == 0:
        return None
    return {
        "dtype": "int8" if tensor.type == 9 else str(tensor.type),
        "scale": [float(s) for s in q.scale],
        "zero_point": [int(z) for z in q.zeroPoint],
    }


def add_float_io(model: bytes) -> tuple[bytes, dict[str, Any]]:
    """Give a quantized single-signature model float32 input and outputs.

    Every int8 graph input gets a float32 tensor in front of it (QUANTIZE op) and every int8 graph
    output one behind it (DEQUANTIZE op). The new tensors take the original tensor names and the
    signature entries point to them, so signature and tensor names do not change; the int8 tensors
    keep the quantization parameters the tool computed.

    Returns:
        The new model and `{"input": {...}, "outputs": [...]}` with the dtype, scale and zero point
        of each tensor that was int8 (the integer interface now internal to the graph).
    """
    from ai_edge_quantizer import qtyping  # lazy

    flat = _read(model)
    subgraph = flat.subgraphs[0]
    int8, float32 = qtyping.TensorType.INT8, qtyping.TensorType.FLOAT32
    quantize_code = _op_code_index(flat, qtyping.BuiltinOperator.QUANTIZE)
    dequantize_code = _op_code_index(flat, qtyping.BuiltinOperator.DEQUANTIZE)
    replaced: dict[int, int] = {}
    info: dict[str, Any] = {"input": [], "outputs": []}

    def rename_internal(tensor_id: int) -> bytes:
        tensor = subgraph.tensors[tensor_id]
        public_name = tensor.name
        tensor.name = public_name + b"_int8"
        return public_name

    for position, tensor_id in enumerate(list(subgraph.inputs)):
        tensor = subgraph.tensors[tensor_id]
        if tensor.type != int8:
            continue
        info["input"].append(_quant_params(tensor))
        float_id = _new_tensor(subgraph, rename_internal(tensor_id), tensor.shape, float32)
        op = qtyping.OperatorT()
        op.opcodeIndex, op.inputs, op.outputs = quantize_code, [float_id], [tensor_id]
        subgraph.operators.insert(0, op)
        subgraph.inputs[position] = float_id
        replaced[tensor_id] = float_id

    for position, tensor_id in enumerate(list(subgraph.outputs)):
        tensor = subgraph.tensors[tensor_id]
        if tensor.type != int8:
            continue
        info["outputs"].append(_quant_params(tensor))
        float_id = _new_tensor(subgraph, rename_internal(tensor_id), tensor.shape, float32)
        op = qtyping.OperatorT()
        op.opcodeIndex, op.inputs, op.outputs = dequantize_code, [tensor_id], [float_id]
        subgraph.operators.append(op)  # after every producer, so the order stays topological
        subgraph.outputs[position] = float_id
        replaced[tensor_id] = float_id

    for signature in flat.signatureDefs or []:
        for entry in [*(signature.inputs or []), *(signature.outputs or [])]:
            entry.tensorIndex = replaced.get(entry.tensorIndex, entry.tensorIndex)
    return _write(flat), info


# --- INT8 ------------------------------------------------------------------------------------------


def quantize_int8(
    float_model: str | Path,
    out_path: str | Path,
    inputs: Iterable[np.ndarray],
    *,
    signature_key: str,
    input_name: str,
) -> dict[str, Any]:
    """Static INT8 PTQ of the FP32 `.tflite` at `float_model`, calibrated on `inputs`, written to `out_path`.

    `inputs` yields `(1, S, S, 3)` float32 NHWC arrays (`calibration_inputs`). The written model has
    float32 input and outputs (`add_float_io`). Returns what the sidecar records about the
    quantization: recipe, tool version and the int8 scale / zero point of the graph interface.
    """
    from importlib import metadata as importlib_metadata

    from ai_edge_quantizer import quantizer  # lazy

    q = quantizer.Quantizer(Path(float_model), INT8_RECIPE)
    calibration = q.calibrate({signature_key: _SignatureInputs(inputs, input_name)})
    quantized = q.quantize(calibration, enable_progress_report=False).quantized_model
    final, interface = add_float_io(bytes(quantized))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(final)
    return {
        "method": "static post-training quantization (int8 weights, int8 activations)",
        "tool": f"ai-edge-quantizer {importlib_metadata.version('ai-edge-quantizer')}",
        "algorithm": "min_max_uniform_quantize",
        "weight_bits": 8,
        "weight_granularity": "per output channel, symmetric",
        "activation_bits": 8,
        "activation_granularity": "per tensor, asymmetric",
        "calibration_statistic": "min / max over the calibration inputs",
        "float_io": True,
        "float_io_note": (
            "A float32 QUANTIZE op follows the input and a DEQUANTIZE op precedes each output inside the "
            "graph; the app feeds and reads float32 exactly as for FP32. The int8 interface behind them "
            "is listed in `internal_interface`."
        ),
        "internal_interface": interface,
        "recipe": INT8_RECIPE,
    }


# --- FP16 ------------------------------------------------------------------------------------------


def weights_to_fp16(model: bytes) -> tuple[bytes, dict[str, Any]]:
    """Store every float32 constant of at least `FP16_MIN_ELEMENTS` values as float16.

    A DEQUANTIZE op (float16 -> float32) is inserted in front of the consumers, so the graph still
    computes in float32 and input and outputs stay float32. Returns the model and
    `{"converted_tensors": n, "kept_float32_tensors": m}`.
    """
    from ai_edge_quantizer import qtyping  # lazy

    flat = _read(model)
    subgraph = flat.subgraphs[0]
    dequantize_code = _op_code_index(flat, qtyping.BuiltinOperator.DEQUANTIZE)
    float32, float16 = qtyping.TensorType.FLOAT32, qtyping.TensorType.FLOAT16
    producers = {out for op in subgraph.operators for out in (op.outputs if op.outputs is not None else [])}
    rewired: dict[int, int] = {}
    kept = 0
    dequant_ops = []
    for tensor_id, tensor in enumerate(list(subgraph.tensors)):
        buffer = flat.buffers[tensor.buffer] if tensor.buffer else None
        constant = buffer is not None and buffer.data is not None and len(buffer.data) > 0
        if tensor.type != float32 or not constant or tensor_id in producers:
            continue
        values = np.frombuffer(bytes(buffer.data), dtype=np.float32)
        if values.size < FP16_MIN_ELEMENTS:
            kept += 1
            continue
        half = values.astype(np.float16)
        if not np.isfinite(half[np.isfinite(values)]).all():
            kept += 1  # a finite weight that overflows float16 stays float32
            continue
        buffer.data = np.frombuffer(half.tobytes(), dtype=np.uint8)
        tensor.type = float16
        float_id = _new_tensor(subgraph, tensor.name + b"_fp32", tensor.shape, float32)
        op = qtyping.OperatorT()
        op.opcodeIndex, op.inputs, op.outputs = dequantize_code, [tensor_id], [float_id]
        dequant_ops.append(op)
        rewired[tensor_id] = float_id

    for op in subgraph.operators:
        op.inputs = [rewired.get(i, i) for i in op.inputs]
    subgraph.operators[0:0] = dequant_ops  # constants only: no dependencies, so they can all come first
    return _write(flat), {"converted_tensors": len(rewired), "kept_float32_tensors": kept}
