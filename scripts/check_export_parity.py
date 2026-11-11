#!/usr/bin/env python3
"""Check an exported student .tflite against its PyTorch source checkpoint.

Runs the PyTorch export wrapper and the .tflite (LiteRT interpreter) on the
same preprocessed images from a dataset split, then reports:
- max / mean absolute difference of each RAW output (`boxes_raw`,
  `cls_logits`);
- detection agreement after the shared reference decode + NMS
  (`src/export/reference.py`): greedy same-class matches at IoU >= 0.9,
  unmatched counts and images whose detection count differs.

Exits 1 if any raw max abs diff >= --max-abs-diff (default 1e-3) or the
detection match rate < --min-match-rate (default 0.99).

The checkpoint path, preprocessing and decode settings come from the
metadata sidecar written by scripts/export_student.py; the checkpoint's
sha256 is re-checked so a sidecar cannot silently pair with other weights.

    CUDA_VISIBLE_DEVICES="" .venv-export/bin/python scripts/check_export_parity.py \\
        --model exports/student_fp32.tflite --data-root data --num-images 20
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

from src.export.parity import DEFAULT_MAX_ABS_DIFF, DEFAULT_MIN_MATCH_RATE, run_parity
from src.export.student_export import StudentExportWrapper, load_student_checkpoint
from src.models.student.student_model import StudentModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch vs .tflite parity check for the student.")
    parser.add_argument("--model", required=True, type=str, help="Exported .tflite file.")
    parser.add_argument(
        "--metadata", type=str, default=None,
        help="Metadata sidecar JSON (default: <model> with a .json suffix).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Override the checkpoint recorded in the metadata.",
    )
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root (default: data).")
    parser.add_argument("--rgb-dir", type=str, default=None, help="Default: <data-root>/cache/mango/rgb.")
    parser.add_argument("--labels-dir", type=str, default=None, help="Default: <data-root>/annotations/yolo/labels.")
    parser.add_argument("--split", type=str, default="val", help="Split to sample images from (default: val).")
    parser.add_argument("--num-images", type=int, default=20, help="Number of images (default: 20).")
    parser.add_argument("--max-abs-diff", type=float, default=DEFAULT_MAX_ABS_DIFF)
    parser.add_argument("--min-match-rate", type=float, default=DEFAULT_MIN_MATCH_RATE)
    parser.add_argument("--report-json", type=str, default=None, help="Optional path to write the report.")
    return parser.parse_args(argv)


def _load_wrapper(metadata: dict, checkpoint_override: str | None) -> StudentExportWrapper:
    source = metadata.get("source_checkpoint") or {}
    if checkpoint_override is None and source.get("path") is None:
        seed = source.get("random_init_seed")
        if seed is None:
            raise ValueError("Metadata records neither a checkpoint nor a random-init seed.")
        torch.manual_seed(seed)
        student = StudentModel(variant=source.get("variant", "wide"), task=source.get("task", "detect"))
        return StudentExportWrapper(student.eval())

    path = checkpoint_override or source["path"]
    loaded = load_student_checkpoint(path)
    expected = source.get("sha256")
    if checkpoint_override is None and expected and loaded.checkpoint_sha256 != expected:
        raise ValueError(
            f"Checkpoint {path} sha256 {loaded.checkpoint_sha256} does not match the "
            f"metadata's {expected}; the sidecar belongs to different weights."
        )
    return StudentExportWrapper(loaded.model)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = Path(args.model)
    metadata_path = Path(args.metadata) if args.metadata else model_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    wrapper = _load_wrapper(metadata, args.checkpoint)
    report = run_parity(
        model_path=model_path,
        metadata=metadata,
        wrapper=wrapper,
        data_root=Path(args.data_root),
        rgb_dir=Path(args.rgb_dir) if args.rgb_dir else None,
        labels_dir=Path(args.labels_dir) if args.labels_dir else None,
        split=args.split,
        num_images=args.num_images,
        max_abs_diff=args.max_abs_diff,
        min_match_rate=args.min_match_rate,
    )
    result = report.to_dict()
    print(json.dumps(result, indent=2))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if not result["passed"]:
        logger.error("Parity FAILED: %s", "; ".join(result["failures"]))
        return 1
    logger.info("Parity PASSED on %d images.", result["images"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
