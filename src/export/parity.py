"""PyTorch-vs-exported-model parity helpers.

Two levels of agreement are checked per image:
- raw outputs: max / mean absolute difference of `boxes_raw` and
  `cls_logits` (before any exp, sigmoid or NMS), and for a segment export
  of `mask_coefs` and `protos` too;
- detections: both raw outputs are decoded with the same reference decode
  (`src/export/reference.py`) and matched greedily by class and IoU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

from src.export.layout import BOXES_OUTPUT_NAME, CLASS_OUTPUT_NAME, DETECT_OUTPUT_NAMES
from src.export.reference import cxcywh_to_xyxy, decode_exported_outputs, preprocess_bgr

# Pass thresholds for an FP32 export. Raw outputs are logits / pixel
# offsets / log-sizes, so an absolute tolerance of 1e-3 is far below any
# visible effect but well above FP32 reordering noise (~1e-5).
DEFAULT_MAX_ABS_DIFF = 1e-3
DEFAULT_MATCH_IOU = 0.9
# Fraction of detections (over the larger of the two sets, summed over all
# images) that must be matched. A candidate whose score sits within float
# noise of `conf_threshold` can legitimately flip, so 100% is not required.
DEFAULT_MIN_MATCH_RATE = 0.99


@dataclass
class DetectionMatch:
    matched: int
    only_reference: int
    only_candidate: int
    min_matched_iou: float | None


def match_detections(
    ref_boxes: torch.Tensor,
    ref_labels: torch.Tensor,
    cand_boxes: torch.Tensor,
    cand_labels: torch.Tensor,
    iou_threshold: float = DEFAULT_MATCH_IOU,
) -> DetectionMatch:
    """Greedy one-to-one matching of cxcywh detections, same class required."""
    if ref_boxes.shape[0] == 0 or cand_boxes.shape[0] == 0:
        return DetectionMatch(0, int(ref_boxes.shape[0]), int(cand_boxes.shape[0]), None)

    iou = box_iou(cxcywh_to_xyxy(ref_boxes), cxcywh_to_xyxy(cand_boxes))
    iou[ref_labels.view(-1, 1) != cand_labels.view(1, -1)] = -1.0

    matched_ious: list[float] = []
    while True:
        best = torch.argmax(iou)
        r, c = divmod(int(best), iou.shape[1])
        value = float(iou[r, c])
        if value < iou_threshold:
            break
        matched_ious.append(value)
        iou[r, :] = -1.0
        iou[:, c] = -1.0

    matched = len(matched_ious)
    return DetectionMatch(
        matched=matched,
        only_reference=int(ref_boxes.shape[0]) - matched,
        only_candidate=int(cand_boxes.shape[0]) - matched,
        min_matched_iou=min(matched_ious) if matched_ious else None,
    )


@dataclass
class ParityReport:
    """Accumulates per-image comparisons and evaluates the pass criteria."""

    max_abs_diff_threshold: float = DEFAULT_MAX_ABS_DIFF
    match_iou: float = DEFAULT_MATCH_IOU
    min_match_rate: float = DEFAULT_MIN_MATCH_RATE
    raw_max: dict[str, float] = field(default_factory=dict)
    raw_sum: dict[str, float] = field(default_factory=dict)
    raw_count: dict[str, int] = field(default_factory=dict)
    images: int = 0
    matched: int = 0
    only_reference: int = 0
    only_candidate: int = 0
    reference_detections: int = 0
    candidate_detections: int = 0
    images_with_count_mismatch: int = 0
    min_matched_iou: float | None = None

    def add_raw(self, name: str, reference: np.ndarray, candidate: np.ndarray) -> None:
        if reference.shape != candidate.shape:
            raise ValueError(f"{name}: shape mismatch {reference.shape} vs {candidate.shape}.")
        diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        self.raw_max[name] = max(self.raw_max.get(name, 0.0), float(diff.max()))
        self.raw_sum[name] = self.raw_sum.get(name, 0.0) + float(diff.sum())
        self.raw_count[name] = self.raw_count.get(name, 0) + int(diff.size)

    def add_detections(self, match: DetectionMatch) -> None:
        self.images += 1
        self.matched += match.matched
        self.only_reference += match.only_reference
        self.only_candidate += match.only_candidate
        n_ref = match.matched + match.only_reference
        n_cand = match.matched + match.only_candidate
        self.reference_detections += n_ref
        self.candidate_detections += n_cand
        if n_ref != n_cand:
            self.images_with_count_mismatch += 1
        if match.min_matched_iou is not None:
            self.min_matched_iou = (
                match.min_matched_iou
                if self.min_matched_iou is None
                else min(self.min_matched_iou, match.min_matched_iou)
            )

    @property
    def match_rate(self) -> float:
        denominator = self.matched + max(self.only_reference, self.only_candidate)
        return 1.0 if denominator == 0 else self.matched / denominator

    def failures(self) -> list[str]:
        problems = [
            f"{name}: max abs diff {value:.3e} >= {self.max_abs_diff_threshold:.1e}"
            for name, value in self.raw_max.items()
            if value >= self.max_abs_diff_threshold
        ]
        if self.match_rate < self.min_match_rate:
            problems.append(f"detection match rate {self.match_rate:.4f} < {self.min_match_rate}")
        if self.images == 0:
            problems.append("no images were compared")
        return problems

    def to_dict(self) -> dict:
        return {
            "images": self.images,
            "raw": {
                name: {
                    "max_abs_diff": self.raw_max[name],
                    "mean_abs_diff": self.raw_sum[name] / max(self.raw_count[name], 1),
                }
                for name in self.raw_max
            },
            "detections": {
                "reference": self.reference_detections,
                "candidate": self.candidate_detections,
                "matched": self.matched,
                "only_reference": self.only_reference,
                "only_candidate": self.only_candidate,
                "images_with_count_mismatch": self.images_with_count_mismatch,
                "min_matched_iou": self.min_matched_iou,
                "match_rate": self.match_rate,
            },
            "thresholds": {
                "max_abs_diff": self.max_abs_diff_threshold,
                "match_iou": self.match_iou,
                "min_match_rate": self.min_match_rate,
            },
            "failures": self.failures(),
            "passed": not self.failures(),
        }


def list_split_images(rgb_dir: str | Path, labels_dir: str | Path, split: str, limit: int) -> list[Path]:
    """First `limit` images of a split, in sorted label-stem order.

    Mirrors `YOLODataset._load_pairs`: a split is defined by the label files
    under `labels_dir/<split>/`, the image is `rgb_dir/<stem>.jpg`.
    """
    split_dir = Path(labels_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Label split directory not found: {split_dir}")
    images = []
    for label_path in sorted(split_dir.glob("*.txt")):
        image_path = Path(rgb_dir) / f"{label_path.stem}.jpg"
        if image_path.exists():
            images.append(image_path)
        if len(images) >= limit:
            break
    return images


def compare_on_images(
    reference_fn: Callable[[torch.Tensor], tuple[np.ndarray, np.ndarray]],
    candidate_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    image_paths: Sequence[Path],
    metadata: dict[str, Any],
    report: ParityReport,
) -> ParityReport:
    """Feed identical preprocessed inputs to both models and accumulate diffs.

    Args:
        reference_fn: NCHW float32 tensor -> (boxes_raw, cls_logits) numpy.
        candidate_fn: NHWC float32 array -> (boxes_raw, cls_logits) numpy.
        metadata: the export sidecar (image size, strides, decode settings).
    """
    output_names = [output["logical_name"] for output in metadata["outputs"]]
    image_size = metadata["input"]["shape"][1]
    pad_value = metadata["preprocessing"]["letterbox"]["pad_value"]
    strides = [level["stride"] for level in metadata["anchors"]["levels"]]
    decode = metadata["decode"]
    decode_kwargs = dict(
        image_size=image_size,
        strides=strides,
        conf_threshold=decode["conf_threshold"],
        nms_iou_threshold=decode["nms"]["iou_threshold"],
        max_detections=decode["max_detections"],
        per_class_candidates=decode["per_class_candidates"],
    )
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        nchw, _ = preprocess_bgr(image, image_size, pad_value)
        nhwc = np.ascontiguousarray(nchw.permute(0, 2, 3, 1).numpy())

        ref_outputs, cand_outputs = reference_fn(nchw), candidate_fn(nhwc)
        (ref_boxes_raw, ref_cls), (cand_boxes_raw, cand_cls) = ref_outputs[:2], cand_outputs[:2]
        for name, reference, candidate in zip(output_names, ref_outputs, cand_outputs):
            report.add_raw(name, reference, candidate)

        ref_boxes, _, ref_labels = decode_exported_outputs(ref_boxes_raw, ref_cls, **decode_kwargs)
        cand_boxes, _, cand_labels = decode_exported_outputs(cand_boxes_raw, cand_cls, **decode_kwargs)
        report.add_detections(
            match_detections(ref_boxes, ref_labels, cand_boxes, cand_labels, report.match_iou)
        )
    return report


def run_parity(
    *,
    model_path: Path,
    metadata: dict[str, Any],
    wrapper,
    data_root: Path,
    split: str = "val",
    num_images: int = 20,
    rgb_dir: Path | None = None,
    labels_dir: Path | None = None,
    max_abs_diff: float = DEFAULT_MAX_ABS_DIFF,
    min_match_rate: float = DEFAULT_MIN_MATCH_RATE,
) -> ParityReport:
    """PyTorch `wrapper` vs the .tflite at `model_path` on `num_images`
    images of `split`. `rgb_dir` / `labels_dir` default to the training
    config's layout under `data_root` (`cache/mango/rgb`,
    `annotations/yolo/labels`)."""
    from src.export.student_export import make_tflite_runner, make_torch_runner

    rgb_dir = rgb_dir or data_root / "cache" / "mango" / "rgb"
    labels_dir = labels_dir or data_root / "annotations" / "yolo" / "labels"
    image_paths = list_split_images(rgb_dir, labels_dir, split, num_images)
    if not image_paths:
        raise FileNotFoundError(f"No '{split}' images found under {rgb_dir} / {labels_dir}.")

    io_names = {
        "signature_key": metadata["signature_key"],
        "input": metadata["input"],
        "outputs": {output["logical_name"]: output for output in metadata["outputs"]},
    }
    report = ParityReport(max_abs_diff_threshold=max_abs_diff, min_match_rate=min_match_rate)
    return compare_on_images(
        make_torch_runner(wrapper),
        make_tflite_runner(model_path, io_names, [output["logical_name"] for output in metadata["outputs"]]),
        image_paths,
        metadata,
        report,
    )
