"""
YOLO-format dataset for RGB+NIR multimodal detection.

Loads paired RGB and NIR images with YOLO-format bounding box labels.
Implements custom letterbox resizing and bbox-aware augmentation via albumentations.

Image pair matching: RGB filename `mango_rgb_XXXXX.jpg` → NIR `mango_nir_XXXXX.jpg`
Label format: `class_id cx cy w h` (normalized 0-1, YOLO convention)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .augmentations import (
    apply_transforms_with_bbox_tracking,
    get_rgb_photometric_transforms,
    get_train_spatial_transforms,
    get_train_transforms,
    get_val_transforms,
)
from .nir_registration import load_homography, register_nir_to_rgb

# ImageNet RGB normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def letterbox(
    image: np.ndarray,
    target_size: int = 640,
    pad_value: int = 114,
) -> tuple[np.ndarray, float, int, int]:
    """Resize image maintaining aspect ratio, pad with gray.

    Args:
        image: Input image (H, W, C) in BGR or grayscale.
        target_size: Target square size.
        pad_value: Padding pixel value (114 = YOLO gray).

    Returns:
        padded: Letterboxed image (target_size, target_size, C).
        scale: Scale factor applied to the image.
        pad_x: Horizontal padding offset.
        pad_y: Vertical padding offset.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded canvas
    if len(image.shape) == 3:
        padded = np.full((target_size, target_size, image.shape[2]), pad_value, dtype=np.uint8)
    else:
        padded = np.full((target_size, target_size), pad_value, dtype=np.uint8)

    # Center the resized image
    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2
    padded[pad_y: pad_y + new_h, pad_x: pad_x + new_w] = resized

    return padded, scale, pad_x, pad_y


def normalize_rgb(image: np.ndarray) -> torch.Tensor:
    """Normalize an RGB image with ImageNet stats and convert it to CHW.

    Module-level so inference/export code (src/export/) reuses the exact
    eval-time normalization instead of re-deriving it.

    Args:
        image: (H, W, 3) uint8 RGB image.

    Returns:
        Tensor[3, H, W] float32 normalized.
    """
    img = image.astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def _scale_bboxes_letterbox(
    bboxes: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    target_size: int,
) -> np.ndarray:
    """Rescale YOLO-normalized bboxes after letterbox transform.

    Input bboxes are in YOLO format: [cx, cy, w, h] normalized to [0, 1]
    relative to the original image dimensions.

    After letterbox, we need them normalized to [0, 1] relative to target_size.

    Args:
        bboxes: (N, 4) array of [cx, cy, w, h] in original normalized coords.
        scale: Scale factor from letterbox.
        pad_x: Horizontal padding offset in pixels.
        pad_y: Vertical padding offset in pixels.
        target_size: Target image size.

    Returns:
        (N, 4) array of [cx, cy, w, h] normalized to target_size.
    """
    if len(bboxes) == 0:
        return bboxes

    result = bboxes.copy()
    # Convert from normalized to pixel coords in original image
    # cx_pixel = cx * orig_w, etc. But we don't have orig_w/h here.
    # Instead, we work with the fact that normalized coords are fractions of image dims.
    # After letterbox: new_cx_pixel = cx_pixel * scale + pad_x
    # Normalized to target: new_cx = new_cx_pixel / target_size
    #
    # Since cx_norm = cx_pixel / orig_w, then cx_pixel = cx_norm * orig_w
    # And scale = target_size / max(orig_h, orig_w)
    # So: new_cx_pixel = cx_norm * orig_w * scale + pad_x
    #     new_cx_norm = new_cx_pixel / target_size
    #
    # For width: new_w_pixel = w_norm * orig_w * scale
    #            new_w_norm = new_w_pixel / target_size
    #
    # But we don't have orig_w/h. The bboxes are already normalized to [0,1]
    # relative to original image. We need to convert them to pixel space first.
    #
    # Actually, the simpler approach: the dataset passes bboxes already in
    # YOLO normalized format. We need to convert to pixel coords in the
    # letterboxed image, then re-normalize to [0,1].
    #
    # The trick: we know the original image dimensions from the caller.
    # But this function doesn't receive them. Let's adjust the approach:
    # We'll pass orig_h, orig_w from the dataset.
    #
    # REVISED: This function is called from the dataset with known orig dims.
    # For now, assume bboxes are already in pixel-space of the letterboxed image.
    # The dataset will handle the conversion.
    return result


class YOLODataset(Dataset):
    """Dataset for paired RGB+NIR images with YOLO-format labels.

    Args:
        rgb_dir: Path to RGB images directory.
        nir_dir: Path to NIR images directory.
        labels_dir: Path to YOLO label files (contains {split}/ subdirectories).
        split: Dataset split ("train", "val", "test").
        transform: Albumentations transform (bbox-aware).
        image_size: Target image size for letterbox.
        nir_mean: NIR normalization mean.
        nir_std: NIR normalization std.
        letterbox_value: Padding pixel value.
        manifest_path: Optional split manifest checked against the label dirs.
        nir_homography_path: Path to the RGB -> NIR homography (.npy). When
            set, every NIR image is warped into the RGB frame (with
            `inv(H)`) right after loading, before letterbox/augmentation, and
            the file MUST exist. None disables registration (NIR is used as
            captured, i.e. misaligned with the RGB-frame labels) and is only
            meant for synthetic fixtures and explicit ablations. Required
            keyword-only (no default): every caller must choose explicitly,
            so forgetting it cannot silently train on unregistered NIR.
    """

    def __init__(
        self,
        rgb_dir: str | Path,
        nir_dir: str | Path,
        labels_dir: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        image_size: int = 640,
        nir_mean: float = 0.0569,
        nir_std: float = 0.0546,
        letterbox_value: int = 114,
        manifest_path: str | Path | None = None,
        *,
        nir_homography_path: str | Path | None,
    ):
        self.rgb_dir = Path(rgb_dir)
        self.nir_dir = Path(nir_dir)
        self.labels_dir = Path(labels_dir) / split
        self.split = split
        self.image_size = image_size
        self.nir_mean = nir_mean
        self.nir_std = nir_std
        self.letterbox_value = letterbox_value
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.nir_homography_path = (
            Path(nir_homography_path) if nir_homography_path is not None else None
        )
        # Loaded once (fails loudly if missing); never per item.
        self.H_rgb_to_nir = (
            load_homography(self.nir_homography_path)
            if self.nir_homography_path is not None
            else None
        )

        self._uses_default_multimodal_transforms = transform is None
        if self._uses_default_multimodal_transforms:
            if split == "train":
                self.spatial_transform = get_train_spatial_transforms(image_size)
                self.rgb_photometric_transform = get_rgb_photometric_transforms(image_size)
            else:
                self.spatial_transform = None
                self.rgb_photometric_transform = None
            self.transform = get_val_transforms(image_size)
        else:
            self.spatial_transform = None
            self.rgb_photometric_transform = None
            self.transform = transform

        self.pairs = self._load_pairs()

    def _load_pairs(self) -> list[dict]:
        """Discover RGB+NIR image pairs and their label files.

        Matches by replacing '_rgb' with '_nir' in the filename stem.
        Only includes pairs that have a label file in the current split.

        Returns:
            List of dicts with keys: rgb_path, nir_path, label_path, image_id.
        """
        if self.manifest_path is not None:
            self._verify_manifest_alignment()

        pairs = []
        rgb_files = sorted(self.rgb_dir.glob("*.jpg"))

        for idx, rgb_path in enumerate(rgb_files):
            # Match NIR by stem substitution
            nir_name = rgb_path.name.replace("_rgb", "_nir")
            nir_path = self.nir_dir / nir_name

            if not nir_path.exists():
                print(f"[YOLODataset] Warning: missing NIR pair for {rgb_path.name}, skipping.")
                continue

            # Label file uses the RGB stem in the split subdirectory
            label_path = self.labels_dir / f"{rgb_path.stem}.txt"

            # Only include pairs that have labels in this split
            if not label_path.exists():
                continue

            pairs.append({
                "rgb_path": rgb_path,
                "nir_path": nir_path,
                "label_path": label_path,
                "image_id": idx,
            })

        return pairs

    def _verify_manifest_alignment(self) -> None:
        """Fail closed if this split's directory contents diverge from the manifest.

        `prepare_yolo_splits.py` previously left stray label files behind from
        earlier shuffles, and this dataset historically keyed off directory
        contents rather than the manifest — silently training/evaluating on a
        leaked split (see design.md Prerequisite P1 / D1). A loud error here
        forces one explicit `scripts/prepare_yolo_splits.py --reconcile` fix
        rather than silently auto-correcting, which would hide that a run
        used a contaminated split.
        """
        if not self.manifest_path.exists():
            return  # No manifest yet (e.g., dataset not split); nothing to check.

        manifest = json.loads(self.manifest_path.read_text())
        expected = set(manifest.get(self.split, []))
        actual = {path.stem for path in self.labels_dir.glob("*.txt")}

        if actual != expected:
            extra = sorted(actual - expected)
            missing = sorted(expected - actual)
            raise ValueError(
                f"YOLODataset split '{self.split}' at {self.labels_dir} diverges from "
                f"manifest {self.manifest_path}. "
                f"On disk but not in manifest for this split: {extra}. "
                f"In manifest but missing on disk: {missing}. "
                "Run `./.venv/bin/python scripts/prepare_yolo_splits.py --reconcile "
                "--no-dry-run` to fix, or pass manifest_path=None to skip this check."
            )

    def gt_count_report(self) -> dict:
        """Report per-class GT instance counts with a direct on-disk recount.

        `loaded` reflects the counts `get_class_counts()` derives from the
        pairs this dataset actually discovered; `on_disk` recounts straight
        from every label file under this split's directory, independent of
        pairing/loading logic. Equal values confirm the dataset is not
        silently dropping or duplicating GT instances relative to the label
        files on disk (design.md Prerequisite P1 self-reporting, W0-c).
        """
        loaded = self.get_class_counts()

        on_disk: dict[int, int] = {}
        for label_path in self.labels_dir.glob("*.txt"):
            _, labels = self._load_labels(label_path)
            for lbl in labels:
                on_disk[int(lbl)] = on_disk.get(int(lbl), 0) + 1

        return {
            "loaded": loaded,
            "on_disk": on_disk,
            "image_count": len(self.pairs),
            "label_file_count": sum(1 for _ in self.labels_dir.glob("*.txt")),
        }

    def _load_labels(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Parse YOLO-format label file.

        Format: class_id cx cy w h (normalized 0-1, one per line).

        Args:
            path: Path to .txt label file.

        Returns:
            bboxes: (N, 4) array of [cx, cy, w, h] normalized.
            labels: (N,) array of class IDs.
        """
        if not path.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        bboxes, labels = [], []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # Skip degenerate boxes
                if w <= 0 or h <= 0:
                    continue
                bboxes.append([cx, cy, w, h])
                labels.append(cls_id)

        if not bboxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        return np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """Load and process a single RGB+NIR pair with labels.

        Returns:
            dict with keys:
                rgb: Tensor[3, image_size, image_size] — ImageNet normalized.
                nir: Tensor[1, image_size, image_size] — dataset normalized.
                bboxes: Tensor[N, 4] — YOLO format [cx, cy, w, h] in [0, 1].
                labels: Tensor[N] — int64 class IDs.
                image_id: int.
        """
        pair = self.pairs[idx]

        # Load images (BGR for RGB, grayscale for NIR)
        rgb_img = cv2.imread(str(pair["rgb_path"]))  # (H, W, 3) BGR
        nir_img = cv2.imread(str(pair["nir_path"]), cv2.IMREAD_GRAYSCALE)  # (H, W)

        if rgb_img is None:
            raise FileNotFoundError(f"Cannot read RGB image: {pair['rgb_path']}")
        if nir_img is None:
            raise FileNotFoundError(f"Cannot read NIR image: {pair['nir_path']}")

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # Load labels
        bboxes, labels = self._load_labels(pair["label_path"])

        # Register NIR to the RGB frame (labels live in the RGB frame) before
        # any letterbox/augmentation. Border fill = the NIR letterbox pad value.
        if self.H_rgb_to_nir is not None:
            orig_h, orig_w = rgb_img.shape[:2]
            nir_img = register_nir_to_rgb(
                nir_img, self.H_rgb_to_nir, (orig_w, orig_h), int(self.nir_mean * 255)
            )

        # Letterbox both images (RGB uses gray=114, NIR uses its own mean ~14)
        rgb_lb, scale, pad_x, pad_y = letterbox(rgb_img, self.image_size, self.letterbox_value)
        nir_lb, _, _, _ = letterbox(nir_img, self.image_size, int(self.nir_mean * 255))

        # Rescale bboxes from original normalized coords to letterbox normalized coords
        if len(bboxes) > 0:
            orig_h, orig_w = rgb_img.shape[:2]
            bboxes = self._rescale_bboxes(bboxes, orig_w, orig_h, scale, pad_x, pad_y)

        if self._uses_default_multimodal_transforms:
            rgb_aug, nir_aug, bboxes_aug, labels_aug = self._apply_default_transforms(
                rgb_lb=rgb_lb,
                nir_lb=nir_lb,
                bboxes=bboxes,
                labels=labels,
            )
        else:
            # Backward-compatible custom transform path. Custom callers are
            # responsible for preserving RGB/NIR geometric alignment.
            if len(bboxes) > 0:
                augmented = apply_transforms_with_bbox_tracking(
                    transform=self.transform,
                    image=rgb_lb,
                    bboxes=bboxes.tolist(),
                    class_labels=labels.tolist(),
                )
                rgb_aug = augmented["image"]
                bboxes_aug = (
                    np.array(augmented["bboxes"], dtype=np.float32).reshape(-1, 4)
                    if augmented["bboxes"]
                    else np.zeros((0, 4), dtype=np.float32)
                )
                labels_aug = (
                    np.array(augmented["class_labels"], dtype=np.int64)
                    if augmented["class_labels"]
                    else np.zeros((0,), dtype=np.int64)
                )
            else:
                augmented = apply_transforms_with_bbox_tracking(
                    transform=self.transform,
                    image=rgb_lb,
                    bboxes=[],
                    class_labels=[],
                )
                rgb_aug = augmented["image"]
                bboxes_aug = np.zeros((0, 4), dtype=np.float32)
                labels_aug = np.zeros((0,), dtype=np.int64)
            nir_aug = nir_lb

        # Normalize RGB (ImageNet stats)
        rgb_tensor = self._normalize_rgb(rgb_aug)

        # Normalize NIR (dataset stats)
        nir_tensor = self._normalize_nir(nir_aug)

        # Convert bboxes to tensor
        bboxes_tensor = torch.from_numpy(bboxes_aug).float()
        labels_tensor = torch.from_numpy(labels_aug).long()

        return {
            "rgb": rgb_tensor,
            "nir": nir_tensor,
            "bboxes": bboxes_tensor,
            "labels": labels_tensor,
            "image_id": pair["image_id"],
        }

    def _rescale_bboxes(
        self,
        bboxes: np.ndarray,
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> np.ndarray:
        """Rescale YOLO-normalized bboxes to letterbox-normalized coords.

        Args:
            bboxes: (N, 4) [cx, cy, w, h] normalized to original image.
            orig_w, orig_h: Original image dimensions.
            scale: Letterbox scale factor.
            pad_x, pad_y: Letterbox padding offsets.

        Returns:
            (N, 4) [cx, cy, w, h] normalized to target_size.
        """
        result = np.zeros_like(bboxes)
        # Convert to pixel coords in original image
        cx_px = bboxes[:, 0] * orig_w
        cy_px = bboxes[:, 1] * orig_h
        w_px = bboxes[:, 2] * orig_w
        h_px = bboxes[:, 3] * orig_h

        # Apply letterbox transform
        cx_lb = cx_px * scale + pad_x
        cy_lb = cy_px * scale + pad_y
        w_lb = w_px * scale
        h_lb = h_px * scale

        # Re-normalize to [0, 1] relative to target_size
        result[:, 0] = cx_lb / self.image_size
        result[:, 1] = cy_lb / self.image_size
        result[:, 2] = w_lb / self.image_size
        result[:, 3] = h_lb / self.image_size

        # Clip to valid range
        result = np.clip(result, 0.0, 1.0)
        return result

    def _apply_default_transforms(
        self,
        rgb_lb: np.ndarray,
        nir_lb: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply default transforms while preserving RGB/NIR alignment.

        Training uses one sampled spatial transform for RGB, NIR, and bounding
        boxes. RGB-only photometric transforms are applied afterwards because
        they do not change geometry. Validation uses the no-op transform.
        """
        bboxes_list = bboxes.tolist() if len(bboxes) > 0 else []
        labels_list = labels.tolist() if len(labels) > 0 else []

        if self.split == "train" and self.spatial_transform is not None:
            augmented = apply_transforms_with_bbox_tracking(
                transform=self.spatial_transform,
                image=rgb_lb,
                bboxes=bboxes_list,
                class_labels=labels_list,
                nir=nir_lb,
            )
            rgb_aug = augmented["image"]
            nir_aug = augmented["nir"]
        else:
            augmented = apply_transforms_with_bbox_tracking(
                transform=self.transform,
                image=rgb_lb,
                bboxes=bboxes_list,
                class_labels=labels_list,
            )
            rgb_aug = augmented["image"]
            nir_aug = nir_lb

        bboxes_aug = (
            np.array(augmented["bboxes"], dtype=np.float32).reshape(-1, 4)
            if augmented["bboxes"]
            else np.zeros((0, 4), dtype=np.float32)
        )
        labels_aug = (
            np.array(augmented["class_labels"], dtype=np.int64)
            if augmented["class_labels"]
            else np.zeros((0,), dtype=np.int64)
        )

        if self.split == "train" and self.rgb_photometric_transform is not None:
            rgb_aug = self.rgb_photometric_transform(image=rgb_aug)["image"]

        return rgb_aug, nir_aug, bboxes_aug, labels_aug

    def _normalize_rgb(self, image: np.ndarray) -> torch.Tensor:
        """Normalize RGB image with ImageNet stats and convert to tensor.

        Args:
            image: (H, W, 3) uint8 RGB image.

        Returns:
            Tensor[3, H, W] float32 normalized.
        """
        return normalize_rgb(image)

    def _normalize_nir(self, image: np.ndarray) -> torch.Tensor:
        """Normalize NIR grayscale image with dataset stats.

        Args:
            image: (H, W) uint8 grayscale image.

        Returns:
            Tensor[1, H, W] float32 normalized.
        """
        img = image.astype(np.float32) / 255.0
        img = (img - self.nir_mean) / self.nir_std
        return torch.from_numpy(img[np.newaxis, :, :]).float()

    def get_class_counts(self) -> dict[int, int]:
        """Count instances per class across all label files.

        Returns:
            Dict mapping class_id → count.
        """
        counts: dict[int, int] = {}
        for pair in self.pairs:
            _, labels = self._load_labels(pair["label_path"])
            for lbl in labels:
                counts[int(lbl)] = counts.get(int(lbl), 0) + 1
        return counts


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for variable-length bbox lists.

    Stacks RGB and NIR tensors; keeps bboxes and labels as lists.

    Args:
        batch: List of dataset items.

    Returns:
        dict with stacked rgb/nir tensors and list bboxes/labels.
    """
    rgb = torch.stack([item["rgb"] for item in batch])
    nir = torch.stack([item["nir"] for item in batch])
    bboxes = [item["bboxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    image_ids = [item["image_id"] for item in batch]

    return {
        "rgb": rgb,
        "nir": nir,
        "bboxes": bboxes,
        "labels": labels,
        "image_ids": image_ids,
    }


def build_weighted_sampler(dataset: YOLODataset) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler based on class frequency.

    Computes inverse-frequency weights per sample based on the dominant
    class in each image. Oversamples images with rare classes (danado).

    Args:
        dataset: YOLODataset instance (must have get_class_counts).

    Returns:
        WeightedRandomSampler with replacement.
    """
    class_counts = dataset.get_class_counts()
    total_instances = sum(class_counts.values())

    if total_instances == 0:
        # Fallback: uniform sampling
        return WeightedRandomSampler(
            weights=[1.0] * len(dataset),
            num_samples=len(dataset),
            replacement=True,
        )

    # Compute per-class weights (inverse frequency)
    num_classes = max(class_counts.keys()) + 1 if class_counts else 2
    class_weights = np.zeros(num_classes, dtype=np.float64)
    for cls_id, count in class_counts.items():
        class_weights[cls_id] = total_instances / (num_classes * count)

    # Assign per-sample weight based on the rarest class in the image
    sample_weights = []
    for pair in dataset.pairs:
        _, labels = dataset._load_labels(pair["label_path"])
        if len(labels) > 0:
            # Weight = max class weight among present classes (prioritize rare classes)
            weight = max(class_weights[lbl] for lbl in labels)
        else:
            weight = 1.0
        sample_weights.append(weight)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    *,
    num_workers: int = 0,
    shuffle: bool = False,
    sampler=None,
    collate_fn=None,
    pin_memory: bool = True,
    drop_last: bool = False,
    **kwargs,
) -> DataLoader:
    """Build a DataLoader with graceful fallback to single-process loading.

    Some environments (WSL, containers with network-mounted /tmp, certain
    FUSE filesystems) cannot create the Unix sockets that PyTorch's
    multiprocessing uses to share file descriptors between workers. This
    manifests as ``OSError: [Errno 95] Operation not supported`` when the
    DataLoader starts iterating.

    To keep the pipeline robust, we probe the requested ``num_workers`` with
    a single-batch iterator. If the probe fails, we fall back to
    ``num_workers=0`` and emit a warning instead of crashing.

    Args:
        dataset: PyTorch Dataset.
        batch_size: Batch size.
        num_workers: Number of worker processes to request.
        shuffle: Whether to shuffle the dataset.
        sampler: Optional sampler.
        collate_fn: Optional collate function.
        pin_memory: Whether to pin memory.
        drop_last: Whether to drop the last incomplete batch.
        **kwargs: Additional DataLoader arguments.

    Returns:
        A DataLoader configured with ``num_workers`` if supported, otherwise
        with ``num_workers=0``.
    """
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        **kwargs,
    }
    loader = DataLoader(dataset, **loader_kwargs)

    if num_workers == 0:
        return loader

    # Probe that multiprocessing IPC actually works in this environment.
    # We use a tiny batch to avoid wasting work; if it fails we recreate the
    # loader with num_workers=0 so training can still proceed (slower).
    probe_kwargs = {**loader_kwargs, "batch_size": min(batch_size, 1), "drop_last": False}
    probe = DataLoader(dataset, **probe_kwargs)
    try:
        next(iter(probe))
    except (OSError, RuntimeError) as exc:
        warnings.warn(
            f"DataLoader multiprocessing failed ({exc!r}); falling back to "
            f"num_workers=0. Training will continue in single-process mode, "
            f"which is slower. If you need workers, run from a native Linux "
            f"filesystem or set TMPDIR to a directory that supports Unix sockets.",
            stacklevel=2,
        )
        loader_kwargs["num_workers"] = 0
        loader = DataLoader(dataset, **loader_kwargs)
    return loader
