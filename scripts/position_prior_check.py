#!/usr/bin/env python3
"""Where do the damage boxes land in the model input? (CPU only, no model.)

Damage labels come from INDUCED impacts, and 85% of the damage-box centers lie
in the central 50% of the mango box (uniform would be 25%). The symmetric mango
ROI crop centers the mango every time, so damage lands at nearly the same input
pixels in every training image, and a detector can learn "damage = a blob at the
crop center". This script measures that without a model: it iterates the TRAIN
split of one leave-one-lot-out fold through `YOLODataset` under two crop
configurations and reports the distribution of the damage-box centers in INPUT
(640 x 640) pixel coordinates.

    current: roi_crop_margin 0.10, roi_crop_jitter 0.5            (twostream_roi_crop*.yaml)
    shift:   roi_crop_train_margin_range [0.2, 1.0], no jitter    (twostream_roi_crop_stable_shift.yaml)

The default training spatial augmentations (flips, rotate90, rotate +-20 deg,
perspective, transpose) act around the image center and are included by default,
so the table is what the model sees; `--no-augment` shows the crop alone. The
RGB-only photometric transforms never move a box and are skipped for speed.

For each configuration it prints the number of boxes, the mean and the standard
deviation of the center per axis, and the fraction of centers within +-WINDOW px
of the input center on BOTH axes.

Usage (from the repo root, the data lives in the parent checkout):
    python scripts/position_prior_check.py \\
        --rgb-dir ../../../data/cache/mango/rgb \\
        --nir-dir ../../../data/cache/mango/nir \\
        --labels-dir ../../../data/annotations/yolo/folds_lot/fold_0/labels \\
        --manifest ../../../data/annotations/yolo/folds_lot/fold_0/splits.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.augmentations import get_val_transforms  # noqa: E402
from src.training.dataset import YOLODataset  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOMOGRAPHY = "notebooks/matriz_homografia_aruco.npy"
DAMAGE_CLASS_ID = 1

# The crop configurations compared (the YAML keys of the experiment configs).
CURRENT_CROP = {
    "roi_crop_margin": 0.10,
    "roi_crop_jitter": 0.5,
    "roi_crop_train_margin_range": None,
}
SHIFT_CROP = {
    "roi_crop_margin": 0.10,
    "roi_crop_jitter": 0.0,
    "roi_crop_train_margin_range": (0.2, 1.0),
}
CONFIGS = (
    ("current: margin 0.10, jitter 0.5", CURRENT_CROP),
    ("shift: total margin U(0.2, 1.0), asymmetric", SHIFT_CROP),
)


def build_dataset(
    *,
    rgb_dir: str | Path,
    nir_dir: str | Path,
    labels_dir: str | Path,
    manifest: str | Path | None,
    homography: str | Path | None,
    split: str,
    image_size: int,
    augment: bool,
    roi_crop_margin: float,
    roi_crop_jitter: float,
    roi_crop_train_margin_range: tuple[float, float] | None,
) -> YOLODataset:
    """The dataset the trainer would build for `split`, with the ROI crop on.

    `augment=False` swaps the default train augmentation for the no-op
    transform, which leaves the crop + letterbox alone.
    """
    dataset = YOLODataset(
        rgb_dir=rgb_dir,
        nir_dir=nir_dir,
        labels_dir=labels_dir,
        split=split,
        image_size=image_size,
        manifest_path=manifest,
        nir_homography_path=homography,
        transform=None if augment else get_val_transforms(image_size),
        roi_crop=True,
        roi_crop_margin=roi_crop_margin,
        roi_crop_jitter=roi_crop_jitter,
        roi_crop_train_margin_range=roi_crop_train_margin_range,
    )
    # Photometric transforms recolor pixels and never move a box.
    dataset.rgb_photometric_transform = None
    return dataset


def collect_damage_centers(dataset: YOLODataset, passes: int, seed: int) -> np.ndarray:
    """(N, 2) damage-box centers (x, y) in input pixels over `passes` passes of `dataset`.

    Seeds torch (the crop window), python/numpy and the albumentations pipeline,
    so the same `seed` gives the same centers.
    """
    import random

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if dataset.spatial_transform is not None:
        dataset.spatial_transform.set_random_seed(seed)

    centers: list[list[float]] = []
    for _ in range(passes):
        for index in range(len(dataset)):
            sample = dataset[index]
            damage = sample["bboxes"][sample["labels"] == DAMAGE_CLASS_ID]
            centers.extend((damage[:, :2] * dataset.image_size).tolist())
    return np.array(centers, dtype=np.float64).reshape(-1, 2)


def summarize_centers(centers: np.ndarray, image_size: int, window: float) -> dict:
    """Mean/sd per axis and the fraction within +-`window` px of the input center.

    The sd is the population standard deviation of the pooled centers. The
    fraction counts a center when BOTH axes are within `window` (borders
    included). Everything is NaN, and `n` 0, for no centers.
    """
    n = len(centers)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "mean_x": nan, "mean_y": nan, "sd_x": nan, "sd_y": nan, "frac_central": nan}
    middle = image_size / 2
    central = (np.abs(centers - middle) <= window).all(axis=1)
    return {
        "n": n,
        "mean_x": float(centers[:, 0].mean()),
        "mean_y": float(centers[:, 1].mean()),
        "sd_x": float(centers[:, 0].std()),
        "sd_y": float(centers[:, 1].std()),
        "frac_central": float(central.mean()),
    }


def format_table(rows: list[tuple[str, dict]], window: float) -> str:
    """One line per configuration."""
    label_width = max(len(label) for label, _ in rows)
    header = (
        f"{'config':<{label_width}} | {'n':>5} | {'mean x':>7} | {'mean y':>7} | "
        f"{'sd x':>6} | {'sd y':>6} | {'within ±' + format(window, 'g') + ' px':>13}"
    )
    lines = [header, "-" * len(header)]
    for label, stats in rows:
        lines.append(
            f"{label:<{label_width}} | {stats['n']:>5d} | {stats['mean_x']:>7.1f} | "
            f"{stats['mean_y']:>7.1f} | {stats['sd_x']:>6.1f} | {stats['sd_y']:>6.1f} | "
            f"{stats['frac_central']:>12.1%}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distribution of damage-box centers in the model input under two crop configs."
    )
    parser.add_argument("--rgb-dir", required=True, help="Directory of RGB images.")
    parser.add_argument("--nir-dir", required=True, help="Directory of NIR images.")
    parser.add_argument("--labels-dir", required=True,
                        help="Fold labels root (contains <split>/*.txt), e.g. folds_lot/fold_0/labels.")
    parser.add_argument("--manifest", default=None,
                        help="Fold splits.json, checked against the label directories (optional).")
    parser.add_argument("--homography", default=DEFAULT_HOMOGRAPHY,
                        help=f"RGB->NIR homography .npy (default: {DEFAULT_HOMOGRAPHY}); "
                             "'none' disables NIR registration (synthetic fixtures only).")
    parser.add_argument("--split", default="train", help="Split to iterate (default: train).")
    parser.add_argument("--passes", type=int, default=3, help="Passes over the split (default: 3).")
    parser.add_argument("--image-size", type=int, default=640, help="Model input size (default: 640).")
    parser.add_argument("--window", type=float, default=64.0,
                        help="Half-width in px of the central window (default: 64).")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the crop/augmentation draws.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Crop + letterbox only, without the spatial training augmentations.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    homography = None if args.homography.lower() == "none" else Path(args.homography)
    if homography is not None and not homography.is_absolute() and not homography.exists():
        homography = PROJECT_ROOT / homography

    rows: list[tuple[str, dict]] = []
    n_images = 0
    for label, crop in CONFIGS:
        dataset = build_dataset(
            rgb_dir=args.rgb_dir,
            nir_dir=args.nir_dir,
            labels_dir=args.labels_dir,
            manifest=args.manifest,
            homography=homography,
            split=args.split,
            image_size=args.image_size,
            augment=not args.no_augment,
            **crop,
        )
        n_images = len(dataset)
        centers = collect_damage_centers(dataset, args.passes, args.seed)
        if len(centers) == 0:
            print(f"ERROR: no damage boxes found in the '{args.split}' split of "
                  f"{args.labels_dir} ({n_images} images).", file=sys.stderr)
            return 1
        rows.append((label, summarize_centers(centers, args.image_size, args.window)))

    augmentation = "crop + letterbox only" if args.no_augment else "with the spatial training augmentations"
    print(f"Damage-box centers in the {args.image_size}x{args.image_size} input: "
          f"'{args.split}' split, {n_images} images, {args.passes} passes, seed {args.seed}, "
          f"{augmentation}.")
    print(format_table(rows, args.window))
    return 0


if __name__ == "__main__":
    sys.exit(main())
