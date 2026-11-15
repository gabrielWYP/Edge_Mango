#!/usr/bin/env python3
"""Run an exported .tflite (FP32, FP16 or INT8) over a fold split and write `predictions.jsonl`.

The inputs are built exactly as `scripts/evaluate_checkpoint.py` builds them for a ROI-crop
student (ground-truth mango box + margin, letterbox 640 pad 114, ImageNet normalization; the
margin comes from the model's sidecar), the raw outputs are decoded with `src/export/reference.py`
(the training decode, at the evaluation threshold 0.001 so AP sees the whole ranked list) and the
result is written in the SAME schema `evaluate_checkpoint.py --save-predictions` writes, boxes and,
for a segment model, mask polygons in the source RGB frame. `scripts/v4_metrics.py` then gives
AP50 / AP30 latent, face AUROC and false positives per healthy photo for the .tflite exactly as it
does for the PyTorch checkpoint, so the three can be compared.

    CUDA_VISIBLE_DEVICES="" .venv-export/bin/python scripts/eval_tflite.py \\
        --model exports/student_int8.tflite \\
        --rgb-dir data/cache/v4_captura_v3_poly/rgb \\
        --labels-dir data/cache/v4_captura_v3_poly/folds_lot/fold_0/labels \\
        --split val --output-dir reports/tflite_int8_fold0
    python scripts/v4_metrics.py --predictions reports/tflite_int8_fold0/predictions.jsonl \\
        --index data/cache/v4_captura_v3_poly/index.csv \\
        --labels-dir data/cache/v4_captura_v3_poly/folds_lot/fold_0/labels/val \\
        --output reports/tflite_int8_fold0/metrics_v4.json

Also writes `eval_tflite.json` (model, precision, sha256, images, decode settings, host-CPU latency of
the interpreter call). The latency is a desktop-CPU figure for orientation, not a phone benchmark.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_checkpoint import PREDICTIONS_FILENAME, _damage_mask_polygons, _prediction_line  # noqa: E402
from src.export.reference import (  # noqa: E402
    decode_exported_outputs,
    decode_exported_segmentation,
    preprocess_roi_bgr,
    roi_letterbox_to_source,
)
from src.export.student_export import make_tflite_runner, sha256_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# TrainingConfig.eval_conf_threshold: the evaluation decodes far below the operating point so AP sees the full ranking.
DEFAULT_EVAL_CONF_THRESHOLD = 0.001
SUMMARY_FILENAME = "eval_tflite.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a .tflite over a fold split, writing predictions.jsonl.")
    parser.add_argument("--model", required=True, help="Exported .tflite file.")
    parser.add_argument("--metadata", default=None, help="Sidecar JSON (default: <model> with a .json suffix).")
    parser.add_argument("--rgb-dir", required=True, help="RGB images, e.g. data/cache/v4_captura_v3_poly/rgb.")
    parser.add_argument("--labels-dir", required=True,
                        help="Label root holding <split>/*.txt, e.g. <dataset>/folds_lot/fold_0/labels.")
    parser.add_argument("--split", default="val", help="Split to evaluate (default: val).")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N images of the split.")
    parser.add_argument("--output-dir", required=True, help="Where predictions.jsonl and eval_tflite.json go.")
    parser.add_argument("--eval-conf-threshold", type=float, default=DEFAULT_EVAL_CONF_THRESHOLD,
                        help=f"Decode threshold (default {DEFAULT_EVAL_CONF_THRESHOLD}, the training evaluation's).")
    return parser.parse_args(argv)


def _io_names(metadata: dict) -> dict:
    return {
        "signature_key": metadata["signature_key"],
        "input": metadata["input"],
        "outputs": {output["logical_name"]: output for output in metadata["outputs"]},
    }


def predict_image(
    run: Callable[[np.ndarray], tuple[np.ndarray, ...]],
    image_path: str | Path,
    label_path: str | Path,
    metadata: dict,
    eval_conf_threshold: float = DEFAULT_EVAL_CONF_THRESHOLD,
) -> str:
    """One `predictions.jsonl` line for one image: the model output `run(nhwc)` decoded and mapped to the source frame.

    `run` takes the `(1, S, S, 3)` float32 NHWC input and returns the raw outputs in the sidecar's order
    (`boxes_raw`, `cls_logits` [, `mask_coefs`, `protos`]).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_size = metadata["input"]["shape"][1]
    roi = metadata["preprocessing"].get("roi_crop") or {}
    tensor, info = preprocess_roi_bgr(
        image, label_path, image_size, metadata["preprocessing"]["letterbox"]["pad_value"],
        float(roi["margin"]) if roi.get("enabled") else None,
    )
    outputs = run(np.ascontiguousarray(tensor.permute(0, 2, 3, 1).numpy()))

    decode = metadata["decode"]
    kwargs = dict(
        image_size=image_size,
        strides=[level["stride"] for level in metadata["anchors"]["levels"]],
        conf_threshold=eval_conf_threshold,
        nms_iou_threshold=decode["nms"]["iou_threshold"],
        max_detections=decode["max_detections"],
        per_class_candidates=decode["per_class_candidates"],
    )
    segment = metadata.get("task") == "segment"
    if segment:
        boxes, scores, labels, masks = decode_exported_segmentation(
            *outputs, mask_threshold=metadata["segmentation"]["threshold"], **kwargs
        )
        polygons = _damage_mask_polygons(masks, labels)
    else:
        boxes, scores, labels = decode_exported_outputs(outputs[0], outputs[1], **kwargs)
        polygons = None

    # cxcywh in letterboxed pixels -> xyxy normalized to the source frame (the same map as the polygon points).
    cx, cy, w, h = boxes.double().unbind(-1)
    corners = torch.stack([torch.stack([cx - w / 2, cy - h / 2], -1), torch.stack([cx + w / 2, cy + h / 2], -1)], 1)
    source_boxes = roi_letterbox_to_source(corners, info).reshape(-1, 4)
    mask_polygons = None
    if polygons is not None:
        mask_polygons = {
            index: [] if polygon is None else [
                round(v, 6)
                for v in roi_letterbox_to_source(torch.from_numpy(polygon) * image_size, info).reshape(-1).tolist()
            ]
            for index, polygon in polygons.items()
        }
    return _prediction_line(Path(image_path).stem, source_boxes, scores, labels, mask_polygons)


def list_split(rgb_dir: Path, labels_dir: Path, split: str, limit: int | None) -> list[tuple[Path, Path]]:
    """`(image, label)` of every label file of the split with its image, sorted by stem (the dataset's order)."""
    split_dir = labels_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Label split directory not found: {split_dir}")
    pairs = [(rgb_dir / f"{p.stem}.jpg", p) for p in sorted(split_dir.glob("*.txt")) if (rgb_dir / f"{p.stem}.jpg").exists()]
    if not pairs:
        raise FileNotFoundError(f"No '{split}' images with labels under {rgb_dir} / {labels_dir}.")
    return pairs[:limit]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = Path(args.model)
    metadata_path = Path(args.metadata) if args.metadata else model_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pairs = list_split(Path(args.rgb_dir), Path(args.labels_dir), args.split, args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_run = make_tflite_runner(model_path, _io_names(metadata), [o["logical_name"] for o in metadata["outputs"]])
    timings: list[float] = []

    def run(nhwc: np.ndarray) -> tuple[np.ndarray, ...]:
        start = time.perf_counter()
        result = raw_run(nhwc)
        timings.append(time.perf_counter() - start)
        return result

    roi = metadata["preprocessing"].get("roi_crop") or {}
    logger.info(
        "%s (%s): %d '%s' images, roi_crop=%s, eval_conf=%g",
        model_path.name, metadata["model"]["precision"], len(pairs), args.split, roi.get("enabled", False),
        args.eval_conf_threshold,
    )
    path = output_dir / PREDICTIONS_FILENAME
    partial = path.with_name(path.name + ".partial")
    with partial.open("w") as handle:
        for image_path, label_path in pairs:
            handle.write(predict_image(run, image_path, label_path, metadata, args.eval_conf_threshold) + "\n")
    partial.replace(path)  # whole file or none, as evaluate_checkpoint.py

    latency = np.array(timings) * 1000.0
    summary = {
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "precision": metadata["model"]["precision"],
        "task": metadata.get("task", "detect"),
        "split": args.split,
        "images": len(pairs),
        "eval_conf_threshold": args.eval_conf_threshold,
        "decode": {
            "nms_iou_threshold": metadata["decode"]["nms"]["iou_threshold"],
            "max_detections": metadata["decode"]["max_detections"],
            "per_class_candidates": metadata["decode"]["per_class_candidates"],
        },
        "roi_crop": roi,
        "host_cpu_interpreter_ms": {
            "note": "LiteRT Python interpreter on this machine, model call only; not a phone benchmark",
            "mean": float(latency.mean()), "median": float(np.median(latency)), "p95": float(np.percentile(latency, 95)),
        },
    }
    (output_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Predictions written to %s (median %.1f ms per image on this CPU)", path, summary["host_cpu_interpreter_ms"]["median"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
