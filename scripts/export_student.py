#!/usr/bin/env python3
"""Export the RGB-only StudentModel to an on-device (LiteRT / .tflite) or ONNX model.

Loads a plain-student or KD-student checkpoint (legacy `kd_proj_*` adapter
keys are stripped, everything else loads strictly; `task`, `student_variant`
come from the checkpoint's config), wraps it so the graph returns only raw
head outputs, converts it with litert-torch (formerly ai-edge-torch) to an
FP32 .tflite with an NHWC input, or with torch.onnx to an .onnx with an NCHW
input, and writes a metadata sidecar JSON next to it describing preprocessing,
output layout, anchors, decode/NMS settings, parameter count and FLOPs, source
checkpoint and tool versions.

`--precision fp16` stores the weights as float16; `--precision int8` is static
post-training quantization (int8 weights and activations, ai-edge-quantizer)
calibrated on `--calib-samples` train images preprocessed exactly like training
inputs (ROI crop, letterbox, ImageNet normalization). Both keep a float32 input
and float32 outputs, so the app contract does not change. Their `--verify` run
compares against PyTorch FP32 and is informational: it never fails the export.

A segment student (`task=segment`) exports four outputs instead of two:
`boxes_raw`, `cls_logits`, `mask_coefs` and `protos` (YOLOv8-seg style; the mask
assembly stays outside the graph, see docs/deployment/mobile-export.md).

Runs in the export venv only (see requirements-export.txt):

    CUDA_VISIBLE_DEVICES="" .venv-export/bin/python scripts/export_student.py \\
        --checkpoint checkpoints/final_runs/<run>/destilado/<best>/best_model.pt \\
        --out-dir exports --verify --data-root data

    # INT8 from the fold's train split (labels/<split>/*.txt define the splits)
    CUDA_VISIBLE_DEVICES="" .venv-export/bin/python scripts/export_student.py --precision int8 \\
        --checkpoint <run>/best_model.pt --rgb-dir data/cache/v4_captura_v3_poly/rgb \\
        --labels-dir data/cache/v4_captura_v3_poly/folds_lot/fold_0/labels --verify

    # ONNX (needs `onnx` and `onnxruntime`); --verify compares it with PyTorch on random inputs
    CUDA_VISIBLE_DEVICES="" python scripts/export_student.py --format onnx \\
        --checkpoint <run>/best_model.pt --out-dir exports --verify

See docs/deployment/mobile-export.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export.parity import (
    DEFAULT_MAX_ABS_DIFF,
    DEFAULT_MIN_MATCH_RATE,
    run_parity,
)
from src.export.onnx_export import check_onnx_parity, convert_to_onnx, onnx_io_names
from src.export.quantize import calibration_inputs, model_complexity, select_calibration_samples
from src.export.student_export import (
    SUPPORTED_PRECISIONS,
    StudentExportWrapper,
    build_metadata,
    export_tflite,
    load_student_checkpoint,
    resolve_decode_settings,
    tflite_io_names,
    write_metadata,
)
from src.models.student.head import NUM_MASKS
from src.models.student.student_model import STUDENT_VARIANTS, StudentModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the student detector for mobile (LiteRT).")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=str, help="Student or KD-student .pt checkpoint.")
    source.add_argument(
        "--random-init", action="store_true",
        help="Export a randomly initialised student (pipeline smoke test only).",
    )
    parser.add_argument("--out-dir", type=str, default="exports", help="Output directory (default: exports).")
    parser.add_argument("--name", type=str, default=None, help="Output basename (default: student_<precision>).")
    parser.add_argument("--format", type=str, choices=["tflite", "onnx"], default="tflite")
    parser.add_argument(
        "--precision", type=str, choices=list(SUPPORTED_PRECISIONS), default="fp32",
        help="fp32; fp16 (float16 weights); int8 (static PTQ, needs --rgb-dir/--labels-dir calibration data). "
             "The input and outputs stay float32 in every precision.",
    )
    parser.add_argument("--rgb-dir", type=str, default=None,
                        help="RGB images for --precision int8 calibration and --verify (default: <data-root>/cache/mango/rgb).")
    parser.add_argument("--labels-dir", type=str, default=None,
                        help="Label root holding <split>/*.txt, e.g. <dataset>/folds_lot/fold_0/labels "
                             "(default: <data-root>/annotations/yolo/labels).")
    parser.add_argument("--calib-split", type=str, default="train", help="Split the INT8 calibration images come from.")
    parser.add_argument("--calib-samples", type=int, default=200, help="INT8 calibration images (default: 200).")
    parser.add_argument("--calib-seed", type=int, default=0, help="Seed of the calibration sample.")
    parser.add_argument("--roi-margin", type=float, default=None,
                        help="ROI crop margin (default: the checkpoint's roi_crop_margin, else 0.10).")
    parser.add_argument("--no-roi-crop", action="store_true",
                        help="Whole-frame model: no ROI crop in calibration and verification. "
                             "Default: the checkpoint's roi_crop (--random-init: on).")
    parser.add_argument("--image-size", type=int, default=None, help="Default: checkpoint config, else 640.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --random-init.")
    parser.add_argument("--task", type=str, choices=["detect", "segment"], default="detect",
                        help="Task of the --random-init student (a checkpoint carries its own).")
    parser.add_argument("--variant", type=str, choices=list(STUDENT_VARIANTS), default="wide",
                        help="Variant of the --random-init student (a checkpoint carries its own).")
    parser.add_argument("--verify", action="store_true", help="Run the parity check after export.")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root for --verify.")
    parser.add_argument("--split", type=str, default="val", help="Split for --verify (default: val).")
    parser.add_argument("--num-images", type=int, default=20, help="Images for --verify (default: 20).")
    parser.add_argument("--max-abs-diff", type=float, default=DEFAULT_MAX_ABS_DIFF)
    parser.add_argument("--min-match-rate", type=float, default=DEFAULT_MIN_MATCH_RATE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"student_{args.precision}"
    model_path = out_dir / f"{name}.{args.format}"
    metadata_path = out_dir / f"{name}.json"

    if args.checkpoint:
        loaded = load_student_checkpoint(args.checkpoint)
        student = loaded.model
        config = loaded.checkpoint_config
        checkpoint_meta = {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": loaded.checkpoint_sha256,
            "legacy_kd_adapter_keys_stripped": loaded.legacy_kd_keys_stripped,
            "model_type": config.get("model_type"),
            **loaded.checkpoint_info,
        }
        logger.info(
            "Loaded %s (stripped %d legacy kd_proj_* keys, epoch=%s, best_map50=%s)",
            args.checkpoint, loaded.legacy_kd_keys_stripped,
            loaded.checkpoint_info.get("epoch"), loaded.checkpoint_info.get("best_map50"),
        )
    else:
        torch.manual_seed(args.seed)
        student = StudentModel(variant=args.variant, task=args.task).eval()
        config = {}
        checkpoint_meta = {"path": None, "random_init_seed": args.seed}
        if (args.variant, args.task) != ("wide", "detect"):  # so check_export_parity.py rebuilds the same student
            checkpoint_meta.update(variant=args.variant, task=args.task)
        logger.warning("Exporting a RANDOM-INIT student (no checkpoint).")

    roi_enabled = not args.no_roi_crop and bool(config.get("roi_crop", args.checkpoint is None))
    roi_margin = args.roi_margin if args.roi_margin is not None else float(config.get("roi_crop_margin", 0.10))
    roi_crop = {"enabled": roi_enabled, "margin": roi_margin}
    data_root = Path(args.data_root)
    rgb_dir = Path(args.rgb_dir) if args.rgb_dir else data_root / "cache" / "mango" / "rgb"
    labels_dir = Path(args.labels_dir) if args.labels_dir else data_root / "annotations" / "yolo" / "labels"

    image_size = args.image_size or int(config.get("image_size", 640))
    letterbox_value = int(config.get("letterbox_value", 114))
    decode = resolve_decode_settings(config)
    wrapper = StudentExportWrapper(student).eval()

    layout = "NCHW" if args.format == "onnx" else "NHWC"
    logger.info("Converting to %s %s at %dx%d (%s input, task=%s) ...",
                args.format, args.precision, image_size, image_size, layout, student.task)
    quantization = calibration = None
    if args.format == "onnx":
        if args.precision != "fp32":
            raise SystemExit("--format onnx exports FP32 only.")
        convert_to_onnx(wrapper, model_path, image_size=image_size)
        io_names = onnx_io_names(wrapper)
    else:
        calibration_stream = None
        if args.precision == "int8":
            samples = select_calibration_samples(
                rgb_dir, labels_dir, split=args.calib_split, n=args.calib_samples, seed=args.calib_seed
            )
            calibration_stream = calibration_inputs(
                samples, image_size, letterbox_value, roi_margin if roi_enabled else None
            )
            calibration = {
                "split": args.calib_split,
                "num_samples": len(samples),
                "seed": args.calib_seed,
                "rgb_dir": str(rgb_dir),
                "labels_dir": str(labels_dir),
                "preprocessing": (
                    f"ROI crop (margin {roi_margin:g}), letterbox {image_size} pad {letterbox_value}, ImageNet "
                    "normalization: the training eval path (src/export/reference.py::preprocess_roi_bgr)"
                    if roi_enabled
                    else f"letterbox {image_size} pad {letterbox_value}, ImageNet normalization"
                ),
                "stems": [image.stem for image, _ in samples],
            }
            logger.info("INT8 calibration on %d '%s' images", len(samples), args.calib_split)
        quantization = export_tflite(
            wrapper, model_path, image_size=image_size, precision=args.precision,
            calibration_inputs=calibration_stream,
        ) or None
        io_names = tflite_io_names(model_path, wrapper.output_names)
    size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.2f MiB)", model_path, size_mb)

    metadata = build_metadata(
        model_file=model_path,
        precision=args.precision,
        image_size=image_size,
        num_classes=student.num_classes,
        decode=decode,
        letterbox_value=letterbox_value,
        io_names=io_names,
        checkpoint=checkpoint_meta,
        num_parameters=sum(p.numel() for p in student.parameters()),
        task=student.task,
        num_prototypes=NUM_MASKS,
        roi_crop=roi_crop,
        complexity=model_complexity(wrapper, image_size),
        quantization=quantization,
        calibration=calibration,
    )
    write_metadata(metadata, metadata_path)
    logger.info("Wrote %s", metadata_path)

    if not args.verify:
        return 0

    if args.format == "onnx":  # random inputs: needs no dataset
        report = check_onnx_parity(
            wrapper, model_path, image_size=image_size, checkpoint_config=config,
            max_abs_diff=args.max_abs_diff, min_match_rate=args.min_match_rate,
        )
    else:
        report = run_parity(
            model_path=model_path,
            metadata=metadata,
            wrapper=wrapper,
            data_root=data_root,
            rgb_dir=rgb_dir,
            labels_dir=labels_dir,
            split=args.split,
            num_images=args.num_images,
            max_abs_diff=args.max_abs_diff,
            min_match_rate=args.min_match_rate,
            informational=args.precision != "fp32",
        )
    result = report.to_dict()
    print(json.dumps(result, indent=2))
    if result["informational"]:
        logger.info(
            "%s vs PyTorch FP32 on %d images (informational, not enforced): detection match rate %.4f, raw max abs diff %s",
            args.precision, result["images"], result["detections"]["match_rate"],
            {name: f"{v['max_abs_diff']:.3g}" for name, v in result["raw"].items()},
        )
        return 0 if not result["failures"] else 1
    if not result["passed"]:
        logger.error("Parity FAILED: %s", "; ".join(result["failures"]))
        return 1
    logger.info("Parity PASSED on %d images.", result["images"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
