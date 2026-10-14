# Data Preprocessing Specification

## Purpose

Transform raw downloaded RGB+NIR image pairs into spatially-aligned, normalized, MasterModel-ready tensors via resize, homography warp, and dual-stream format conversion.

## Requirements

### Requirement: Image Loading and Validation

The system SHALL load cached RGB and NIR images from local disk and validate basic integrity.

- RGB images MUST be loaded as (H, W, 3) uint8 arrays via OpenCV (BGR→RGB conversion).
- NIR images MUST be loaded as (H, W) uint8 grayscale arrays.
- Corrupt or unreadable images MUST be logged and skipped (not crash the pipeline).

#### Scenario: Valid paired images loaded

- GIVEN cached rgb_path and nir_path pointing to valid JPEG files
- WHEN preprocessor loads the pair
- THEN rgb_array shape is (H, W, 3) uint8 and nir_array shape is (H, W) uint8

#### Scenario: Corrupt image skipped gracefully

- GIVEN nir_path points to a corrupt or zero-byte file
- WHEN preprocessor attempts to load the NIR image
- THEN the pair is logged as invalid with file path and error detail
- AND the pair is excluded from the output dataset without crashing

### Requirement: NIR→RGB Spatial Alignment via Homography

The system SHALL align NIR images to RGB spatial coordinates using a homography matrix.

- The homography `H` (`notebooks/matriz_homografia_aruco.npy`) maps RGB → NIR (`cv2.findHomography(pts_rgb, pts_nir)`), so NIR MUST be warped into the RGB frame via `cv2.warpPerspective(nir, np.linalg.inv(H), (w_rgb, h_rgb))`.
- Registration MUST happen at load time, before letterbox and augmentation, with border fill equal to the NIR letterbox pad value. Labels are expressed in the RGB frame, so unregistered NIR is misaligned with them.
- The homography is configured by `nir_homography_path` (default `notebooks/matriz_homografia_aruco.npy`). If the path is set but the file is missing or is not a finite, invertible 3x3 matrix, loading MUST fail loudly; there is NO silent fallback to resizing.
- `nir_homography_path: null` explicitly disables registration (ablations and synthetic fixtures only). Checkpoints record `nir_registration` (path and SHA-256) so registered and unregistered runs are distinguishable.
- Warped NIR output MUST have the same (H, W) spatial dimensions as the paired RGB image.

#### Scenario: Homography warp produces aligned NIR

- GIVEN a valid RGB → NIR homography `H` and a NIR image
- WHEN alignment is applied for an RGB image of size (w, h)
- THEN warped NIR shape is (h, w) and the NIR pixel `H(p)` appears at RGB pixel `p`

#### Scenario: Missing homography fails loudly

- GIVEN `nir_homography_path` points to a file that does not exist
- WHEN the dataset is constructed
- THEN a FileNotFoundError is raised
- AND NIR is NOT silently resized or used unregistered

### Requirement: Resize and Normalization

The system SHALL resize images to configurable target dimensions and normalize per-channel.

- Both RGB and resized/warped NIR MUST be resized to (target_h, target_w), default (640, 640).
- RGB normalization MUST use ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
- NIR normalization MUST use dataset-computed stats (or placeholder mean=0.45, std=0.22 as fallback).
- The normalization step MUST be applied after resize, consistent with albumentations order.

#### Scenario: Default preprocessing produces MasterModel-compatible shapes

- GIVEN a paired 800×600 RGB+NIR input with default config (640×640)
- WHEN preprocessing pipeline completes
- THEN rgb_tensor shape is (3, 640, 640) float32
- AND nir_tensor shape is (1, 640, 640) float32
- AND both are normalized per their respective mean/std

#### Scenario: Configurable target size

- GIVEN target_size=(416, 416) in config
- WHEN preprocessing completes
- THEN rgb_tensor shape is (3, 416, 416) and nir_tensor shape is (1, 416, 416)

### Requirement: Dual-Stream Output Format

The preprocessor SHALL output separate RGB and NIR tensors matching MasterModel's `forward(rgb, nir)` signature.

- Output MUST be two separate tensors: `rgb_tensor` of shape `(3, H, W)` and `nir_tensor` of shape `(1, H, W)`.
- The system MUST NOT stack them into a single 4-channel tensor (that is the ConvNeXtTeacher format, not MasterModel).
- Each preprocessed sample MUST include metadata: source class, filename stem, and whether homography was applied.

#### Scenario: Output tensors match MasterModel forward signature

- GIVEN preprocessed sample from input pair (rgb, nir)
- WHEN rgb_tensor and nir_tensor are passed to MasterModel.forward()
- THEN the model accepts them without shape mismatch error