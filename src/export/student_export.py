"""Export wrapper, checkpoint loading, metadata sidecar and LiteRT conversion
for the RGB-only StudentModel.

Design (see docs/deployment/mobile-export.md):
- The exported graph stops at the raw head outputs. Box decode (exp) and NMS
  run in app code: exp() and NMS both quantize badly and NMS has no portable
  fixed-shape TFLite op, so keeping them out keeps FP32 and a later INT8
  model on the same, simple contract.
- Outputs follow `src/export/layout.py`: `boxes_raw` (1, N, 4) and
  `cls_logits` (1, N, num_classes).
- The .tflite input is NHWC (1, 640, 640, 3) float32 — LiteRT's native
  layout, and the order an Android bitmap is read in. It is produced with
  `litert_torch.to_channel_last_io`, which only inserts a transpose at the
  graph input; the PyTorch wrapper itself stays NCHW.
- ImageNet normalization is NOT baked into the graph; the app applies it
  (spelled out in the metadata sidecar), so the exported graph is exactly
  the trained network.

`litert_torch` / `ai_edge_litert` are imported lazily inside the functions
that need them, so this module is importable in the training venv.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from src.export.layout import (
    BOX_CHANNELS,
    BOXES_OUTPUT_NAME,
    CLASS_OUTPUT_NAME,
    DETECT_OUTPUT_NAMES,
    MASK_COEFS_OUTPUT_NAME,
    PROTO_STRIDE,
    PROTOS_OUTPUT_NAME,
    SEGMENT_OUTPUT_NAMES,
    anchor_layout,
    flatten_levels,
    num_anchors,
)
from src.models.student.student_model import StudentModel, strip_legacy_kd_adapter_keys
from src.training.config import TrainingConfig
from src.training.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.training.strides import STUDENT_STRIDES

METADATA_SCHEMA_VERSION = 1
CLASS_NAMES: tuple[str, ...] = ("mango", "damage")
SUPPORTED_PRECISIONS: tuple[str, ...] = ("fp32",)
EXPORT_TOOL_PACKAGES: tuple[str, ...] = (
    "torch", "litert-torch", "ai-edge-litert", "litert-converter", "numpy",
)


class StudentExportWrapper(nn.Module):
    """StudentModel -> (boxes_raw, cls_logits), plain tensors only.

    No dict, no distillation features, no decode, no exp, no NMS. The
    distillation tensors the student also returns are dead code in the
    exported graph and are pruned by `torch.export`.

    A segment student (`task="segment"`) returns two more tensors,
    `(boxes_raw, cls_logits, mask_coefs, protos)`: YOLOv8-seg style, the
    mask assembly stays outside the graph like the box decode.
    """

    def __init__(self, student: StudentModel) -> None:
        super().__init__()
        self.student = student

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.student(x)
        boxes_raw = flatten_levels(out["reg_preds"])
        cls_logits = flatten_levels(out["cls_preds"])
        if self.student.task == "segment":
            return boxes_raw, cls_logits, flatten_levels(out["mask_coefs"]), out["protos"]
        return boxes_raw, cls_logits

    @property
    def output_names(self) -> tuple[str, ...]:
        return SEGMENT_OUTPUT_NAMES if self.student.task == "segment" else DETECT_OUTPUT_NAMES


@dataclass(frozen=True)
class DecodeSettings:
    """Eval-time decode parameters the app must reproduce."""

    conf_threshold: float
    nms_iou_threshold: float
    max_detections: int
    per_class_candidates: bool


@dataclass
class LoadedStudent:
    model: StudentModel
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_config: dict[str, Any]
    checkpoint_info: dict[str, Any]
    legacy_kd_keys_stripped: int


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_student_checkpoint(path: str | Path) -> LoadedStudent:
    """Load a plain-student or KD-student checkpoint into a bare StudentModel.

    Accepts a Trainer/KDTrainer checkpoint dict (`model_state_dict`) or a raw
    state_dict. Legacy `kd_proj_*` adapter keys are stripped; everything else
    loads with `strict=True`, so a teacher checkpoint or a mismatched
    architecture fails loudly.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config") or {}
        if not isinstance(config, dict):
            config = dict(vars(config))
        info = {
            key: checkpoint[key]
            for key in ("epoch", "phase", "arch_version", "best_map50", "experiment_sha256")
            if key in checkpoint
        }
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict, config, info = checkpoint, {}, {}
    else:
        raise ValueError(f"{path} is neither a checkpoint dict with 'model_state_dict' nor a raw state_dict.")

    model_type = config.get("model_type")
    if model_type not in (None, "student"):
        raise ValueError(f"{path} was trained with model_type={model_type!r}; only student checkpoints export.")

    in_channels = int(config.get("student_in_channels", 3))
    if in_channels != 3:
        raise ValueError(
            f"{path} has student_in_channels={in_channels}: the export graph takes one RGB input, "
            "so an RGB+NIR student (est_rgbnir) cannot be exported."
        )

    stripped = strip_legacy_kd_adapter_keys(state_dict)
    model = StudentModel(
        num_classes=int(config.get("num_classes", len(CLASS_NAMES))),
        variant=config.get("student_variant", "wide"),
        task=config.get("task", "detect"),
    )
    model.load_state_dict(stripped, strict=True)
    model.eval()
    return LoadedStudent(
        model=model,
        checkpoint_path=path,
        checkpoint_sha256=sha256_file(path),
        checkpoint_config=config,
        checkpoint_info=info,
        legacy_kd_keys_stripped=len(state_dict) - len(stripped),
    )


def _config_default(name: str) -> Any:
    for field in dataclasses.fields(TrainingConfig):
        if field.name == name:
            return field.default
    raise KeyError(name)


def resolve_decode_settings(checkpoint_config: dict[str, Any]) -> DecodeSettings:
    """Decode settings the checkpoint was evaluated with (TrainingConfig
    defaults when the checkpoint did not record them)."""
    def get(name: str) -> Any:
        return checkpoint_config.get(name, _config_default(name))

    return DecodeSettings(
        conf_threshold=float(get("conf_threshold")),
        nms_iou_threshold=float(get("nms_iou_threshold")),
        max_detections=int(get("max_detections")),
        per_class_candidates=bool(get("decode_per_class")),
    )


def tool_versions(packages: Sequence[str] = EXPORT_TOOL_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_metadata(
    *,
    model_file: str | Path,
    precision: str,
    image_size: int,
    num_classes: int,
    decode: DecodeSettings,
    letterbox_value: int,
    io_names: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    num_parameters: int,
    strides: Sequence[int] = STUDENT_STRIDES,
    class_names: Sequence[str] = CLASS_NAMES,
    tools: dict[str, str | None] | None = None,
    task: str = "detect",
    num_prototypes: int = 32,
) -> dict[str, Any]:
    """Build the metadata sidecar describing the exported model's contract.

    `io_names` is `tflite_io_names(...)`'s result: the signature key plus
    the signature-level and tensor-level names of the input and of each
    logical output (`boxes_raw`, `cls_logits`).
    """
    model_file = Path(model_file)
    n = num_anchors(image_size, strides)
    nchw = model_file.suffix == ".onnx"  # ONNX keeps PyTorch's layout; only the .tflite input is channel-last
    proto_side = image_size // PROTO_STRIDE
    metadata = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "model": {
            "name": "student",
            "architecture": "StudentModel (CSPDarknetNano + PANet + decoupled anchor-free head, no DFL)",
            "num_parameters": num_parameters,
            "format": model_file.suffix.lstrip("."),
            "precision": precision,
            "file": model_file.name,
            "file_sha256": sha256_file(model_file) if model_file.exists() else None,
            "file_size_bytes": model_file.stat().st_size if model_file.exists() else None,
        },
        "source_checkpoint": checkpoint,
        "signature_key": io_names["signature_key"],
        "input": {
            **io_names["input"],
            "shape": [1, 3, image_size, image_size] if nchw else [1, image_size, image_size, 3],
            "layout": "NCHW" if nchw else "NHWC",
            "dtype": "float32",
            "color_order": "RGB",
        },
        "preprocessing": {
            "reference": "src/export/reference.py::preprocess_bgr (== YOLODataset val/test path)",
            "steps": [
                "Decode the image to 8-bit RGB (the training loader reads BGR with cv2.imread and converts BGR->RGB).",
                "Letterbox: scale = image_size / max(w, h); new_w = floor(w * scale), new_h = floor(h * scale).",
                "Resize to (new_w, new_h) with bilinear interpolation (cv2.INTER_LINEAR).",
                "Paste centered on an image_size x image_size canvas filled with pad_value on every channel: "
                "pad_x = (image_size - new_w) // 2, pad_y = (image_size - new_h) // 2.",
                "Per channel: value = (pixel / 255 - mean[c]) / std[c], float32.",
                "Write as NCHW (planar: all R, then all G, then all B)."
                if nchw
                else "Write as NHWC (row-major, RGB interleaved).",
            ],
            "letterbox": {
                "target_size": image_size,
                "pad_value": letterbox_value,
                "interpolation": "bilinear (cv2.INTER_LINEAR)",
                "centered": True,
            },
            "normalization": {
                "scale": 1.0 / 255.0,
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "formula": "(pixel / 255 - mean[c]) / std[c]",
                "baked_into_model": False,
            },
        },
        "outputs": [
            {
                "logical_name": BOXES_OUTPUT_NAME,
                **io_names["outputs"][BOXES_OUTPUT_NAME],
                "shape": [1, n, 4],
                "dtype": "float32",
                "channels": list(BOX_CHANNELS),
            },
            {
                "logical_name": CLASS_OUTPUT_NAME,
                **io_names["outputs"][CLASS_OUTPUT_NAME],
                "shape": [1, n, num_classes],
                "dtype": "float32",
                "channels": list(class_names),
            },
        ],
        "anchors": {
            "num_anchors": n,
            "order": "level-major (finest stride first), then row-major within a level (y outer, x inner)",
            "levels": [
                {
                    "stride": level.stride,
                    "grid_h": level.grid_h,
                    "grid_w": level.grid_w,
                    "offset": level.offset,
                    "count": level.count,
                }
                for level in anchor_layout(image_size, strides)
            ],
        },
        "decode": {
            "reference": "src/export/reference.py::decode_exported_outputs (delegates to src/training/decode.py)",
            "cx": "(gx + 0.5) * stride + dx",
            "cy": "(gy + 0.5) * stride + dy",
            "w": "exp(log_w)",
            "h": "exp(log_h)",
            "box_units": "pixels of the letterboxed image_size x image_size input (dx/dy are NOT scaled by stride)",
            "score": "sigmoid(cls_logit)",
            "candidates": (
                "one candidate per (anchor, class) with score > conf_threshold"
                if decode.per_class_candidates
                else "argmax class per anchor, kept if its score > conf_threshold"
            ),
            "per_class_candidates": decode.per_class_candidates,
            "conf_threshold": decode.conf_threshold,
            "nms": {
                "type": "greedy, per class (class-aware / batched), on xyxy boxes, highest score first",
                "iou_threshold": decode.nms_iou_threshold,
            },
            "max_detections": decode.max_detections,
            "to_original_image": "x = (x_letterboxed - pad_x) / scale, y = (y_letterboxed - pad_y) / scale, clip to image",
        },
        "class_names": list(class_names),
        "num_classes": num_classes,
        "tools": tools if tools is not None else tool_versions(),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if task == "segment":
        metadata["task"] = "segment"
        metadata["outputs"] += [
            {
                "logical_name": MASK_COEFS_OUTPUT_NAME,
                **io_names["outputs"][MASK_COEFS_OUTPUT_NAME],
                "shape": [1, n, num_prototypes],
                "dtype": "float32",
                "channels": "mask coefficients of each anchor, same anchor order as boxes_raw",
            },
            {
                "logical_name": PROTOS_OUTPUT_NAME,
                **io_names["outputs"][PROTOS_OUTPUT_NAME],
                "shape": [1, num_prototypes, proto_side, proto_side],
                "dtype": "float32",
                "channels": "prototype masks, NCHW, at 1/4 of the input resolution",
            },
        ]
        metadata["segmentation"] = {
            "reference": "src/export/reference.py::decode_exported_segmentation (delegates to src/training/decode.py)",
            "num_prototypes": num_prototypes,
            "mask_ratio": PROTO_STRIDE,
            "prototype_size": [proto_side, proto_side],
            "threshold": 0.5,
            "assembly": (
                "For each detection kept by the box decode, take the coefficient vector of the anchor it came "
                "from (row i of mask_coefs, the same i as in boxes_raw). mask = sigmoid(coefs . protos), with "
                "protos flattened to (nm, h*w) and the result reshaped to (h, w). Crop it to the detection box in "
                "prototype coordinates (box in letterboxed pixels / mask_ratio): pixels outside the box are zeroed. "
                "Optionally upsample bilinearly to the input size. Threshold at `threshold`. Letterbox padding "
                "and the ROI crop are undone on the mask exactly as on the box."
            ),
        }
    return metadata


def write_metadata(metadata: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def convert_to_tflite(
    wrapper: StudentExportWrapper,
    out_path: str | Path,
    *,
    image_size: int,
    precision: str = "fp32",
) -> Path:
    """Convert the wrapper to a .tflite with NHWC input via litert-torch.

    `precision` is the extension point for later FP16/INT8 variants; only
    "fp32" is implemented.
    """
    if precision not in SUPPORTED_PRECISIONS:
        raise NotImplementedError(
            f"precision={precision!r} is not implemented yet; supported: {SUPPORTED_PRECISIONS}."
        )
    import litert_torch  # lazy: only present in the export venv

    wrapper = wrapper.eval()
    nhwc_module = litert_torch.to_channel_last_io(wrapper, args=[0]).eval()
    sample = (torch.zeros(1, image_size, image_size, 3, dtype=torch.float32),)
    with torch.no_grad():
        edge_model = litert_torch.convert(nhwc_module, sample)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(out_path))
    return out_path


def tflite_io_names(
    model_path: str | Path, output_names: Sequence[str] = DETECT_OUTPUT_NAMES
) -> dict[str, Any]:
    """Read the input/output names of a converted model's single signature.

    litert-torch names signature outputs `output_<i>` in the wrapper's
    return order, so `output_0` is `boxes_raw` and `output_1` is
    `cls_logits` (a segment model, `output_names=SEGMENT_OUTPUT_NAMES`,
    continues with `mask_coefs` and `protos`). Returns the signature key and, for the input and each
    logical output, both the signature name (for `get_signature_runner`)
    and the underlying tensor name (for index-based interpreter APIs).
    """
    from ai_edge_litert.interpreter import Interpreter  # lazy

    interpreter = Interpreter(model_path=str(model_path))
    signatures = interpreter.get_signature_list()
    if len(signatures) != 1:
        raise ValueError(f"Expected exactly one signature, got {list(signatures)}.")
    signature_key = next(iter(signatures))
    runner = interpreter.get_signature_runner(signature_key)
    input_details = runner.get_input_details()
    output_details = runner.get_output_details()
    if len(input_details) != 1 or len(output_details) != len(output_names):
        raise ValueError(
            f"Expected 1 input and {len(output_names)} outputs, got {list(input_details)} / {list(output_details)}."
        )
    input_name = next(iter(input_details))
    ordered_outputs = sorted(output_details, key=lambda name: int(name.rsplit("_", 1)[-1]))
    return {
        "signature_key": signature_key,
        "input": {
            "signature_name": input_name,
            "tensor_name": input_details[input_name]["name"],
        },
        "outputs": {
            logical: {
                "signature_name": name,
                "tensor_name": output_details[name]["name"],
            }
            for logical, name in zip(output_names, ordered_outputs)
        },
    }


def make_tflite_runner(
    model_path: str | Path,
    io_names: dict[str, Any] | None = None,
    output_names: Sequence[str] = DETECT_OUTPUT_NAMES,
):
    """Return `fn(nhwc_float32) -> (boxes_raw, cls_logits[, mask_coefs, protos])` backed by one
    LiteRT interpreter (created once, reused across calls)."""
    from ai_edge_litert.interpreter import Interpreter  # lazy

    io_names = io_names or tflite_io_names(model_path)
    interpreter = Interpreter(model_path=str(model_path))
    runner = interpreter.get_signature_runner(io_names["signature_key"])
    input_name = io_names["input"]["signature_name"]
    signature_outputs = [io_names["outputs"][logical]["signature_name"] for logical in output_names]

    def run(nhwc_input):
        result = runner(**{input_name: nhwc_input})
        return tuple(result[name] for name in signature_outputs)

    return run


def make_torch_runner(wrapper: StudentExportWrapper):
    """Return `fn(nchw_tensor) -> (boxes_raw, cls_logits[, mask_coefs, protos])` as numpy arrays."""
    wrapper = wrapper.eval()

    def run(nchw_input: torch.Tensor):
        with torch.no_grad():
            return tuple(t.numpy() for t in wrapper(nchw_input))

    return run
