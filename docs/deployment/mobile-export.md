# Mobile export of the student detector

This guide covers exporting the RGB-only `StudentModel` (plain or KD) to a LiteRT `.tflite` file (FP32, FP16 or static INT8; or an ONNX file), checking it against PyTorch, measuring its accuracy on a fold, and what the Android app has to implement around it. Section 7 covers segment students, section 8 the FP16/INT8 files, section 9 the accuracy-parity procedure.

## 1. Set up the export venv (one time)

The export toolchain gets its own venv. `litert-torch` (the renamed `ai-edge-torch`) pins its own torch range, and the training venv must stay untouched.

```bash
uv venv .venv-export --python 3.12
uv pip install --python .venv-export/bin/python \
    --index-strategy unsafe-best-match -r requirements-export.txt
```

Everything runs on CPU. Prefix the commands below with `CUDA_VISIBLE_DEVICES=""`.

## 2. Export

```bash
.venv-export/bin/python scripts/export_student.py \
    --checkpoint <path>/best_model.pt --out-dir exports \
    --verify --data-root data
```

- **Checkpoints:** the script accepts plain-student or KD checkpoints (`model_state_dict`) and raw state_dicts. It strips legacy `kd_proj_*` keys, then loads everything else with `strict=True`. It rejects teacher checkpoints.
- **Outputs:** `exports/student_fp32.tflite` and the sidecar `exports/student_fp32.json` (the whole `exports/` directory is git-ignored; ship both files with the app). The sidecar is the app's contract: input, preprocessing, outputs, anchors, decode and NMS settings, the source checkpoint's sha256, and tool versions.
- **Smoke test:** `--random-init` exports an untrained student to test the pipeline.
- **Precision:** `--precision fp32|fp16|int8`, see section 8. The input and the outputs are float32 in all three.
- **Data for INT8 and `--verify`:** `--rgb-dir` and `--labels-dir` (the fold's `labels/`, which holds `train/`, `val/`, `test/`). Without them the old `<data-root>/cache/mango/rgb` and `<data-root>/annotations/yolo/labels` layout is used.
- **ROI crop:** the models are trained on a crop of the mango, so calibration, `--verify` and `eval_tflite.py` crop the same way. A checkpoint carries `roi_crop` and `roi_crop_margin` in its config; for `--random-init` the crop is on with margin 0.10 (`--roi-margin`, `--no-roi-crop` override). The sidecar records it in `preprocessing.roi_crop`.
- **Sidecar extras (schema 2):** `complexity` (parameters; FLOPs of one 640 forward pass, counted by `torch.utils.flop_counter.FlopCounterMode`, conv + matmul, 1 MAC = 2 FLOPs), `model.file_size_bytes`, and for INT8 `quantization` and `calibration`.

## 3. Verify

`--verify` runs the parity check after the export. To run it on its own:

```bash
.venv-export/bin/python scripts/check_export_parity.py \
    --model exports/student_fp32.tflite --data-root data --split val --num-images 20
```

The check sends the same preprocessed images through the PyTorch wrapper and the LiteRT interpreter, then compares the results in two ways:

- **Raw outputs:** the max abs diff of each output must be below `1e-3`.
- **Detections:** both sets of outputs are decoded with the reference decode and NMS. Boxes are matched when they have the same class and IoU ≥ 0.9. At least 99% must match.

The script exits with status 1 if either check fails. This is the FP32 contract. For `fp16` and `int8` files the same comparison against PyTorch FP32 is **informational**: `export_student.py --verify` prints the raw max/mean abs diffs and the decoded-detection match rate, never fails on them, and reports `"passed": null`. With random-init weights the INT8 numbers carry no information (the heads output near-constants, so the quantization ranges degenerate); judge INT8 on a real checkpoint with section 9.

`--rgb-dir`/`--labels-dir`/`--split` choose the images; the sidecar's `preprocessing.roi_crop` makes the check crop each image to its mango box like training.

## 4. Model contract

| | |
|---|---|
| Input | `(1, 640, 640, 3)` float32, **NHWC**, RGB |
| Output `boxes_raw` | `(1, 8400, 4)` float32: `dx, dy, log_w, log_h` |
| Output `cls_logits` | `(1, 8400, 2)` float32: logits for `mango`, `damage` |
| Anchors | stride 8 (80×80, offset 0), 16 (40×40, offset 6400), 32 (20×20, offset 8000). Row-major within a level. |

- **Signature names:** `args_0` is the input, `output_0` is `boxes_raw` and `output_1` is `cls_logits`. The exact names are recorded in the sidecar.
- **NHWC input:** LiteRT's native layout and the order Android reads bitmaps in. `litert_torch.to_channel_last_io` only adds one transpose at the input.
- **Two outputs:** boxes and class logits have very different value ranges. Keeping them apart lets INT8 give each output its own quantization scale.
- **Nothing after the head:** no exp, sigmoid, decode or NMS in the graph. exp and NMS quantize badly, so they run in the app.
- **Ops:** the converted graph uses only builtin ops (CONV_2D, LOGISTIC, MUL, ADD, CONCATENATION, SLICE, TRANSPOSE, RESHAPE, PAD, MAX_POOL_2D, RESIZE_NEAREST_NEIGHBOR). It has no Flex or custom ops.

## 5. What the app must implement

The Python reference for each step is in `src/export/reference.py`. The decode step delegates to `src/training/decode.py`, the same decode used for training evaluation.

1. **Preprocess.** This is the same as `YOLODataset`'s val/test path.
   - `scale = 640 / max(w, h)`.
   - Resize bilinearly to `(floor(w·scale), floor(h·scale))`.
   - Paste the result centered on a 640×640 canvas filled with **114**. `pad_x = (640 − new_w) // 2` and `pad_y = (640 − new_h) // 2`.
   - Per channel, compute `(v / 255 − mean) / std`, with ImageNet `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`.
   - The model does not normalize internally.
2. **Decode.** For anchor `i` in level `(stride, grid_w, offset)`:
   - `gy = (i − offset) / grid_w` and `gx = (i − offset) % grid_w`.
   - `cx = (gx + 0.5)·stride + dx` and `cy = (gy + 0.5)·stride + dy`. `dx` and `dy` are in pixels and are **not** multiplied by the stride.
   - `w = exp(log_w)` and `h = exp(log_h)`, also in pixels.
   - `score_c = sigmoid(logit_c)`.
   - Emit one candidate per (anchor, class) with `score > 0.25`. This is the per-class rule, not argmax.
3. **NMS.** Run greedy NMS per class on xyxy boxes with IoU threshold **0.5**, highest score first. Keep at most **300** detections.
4. **Map back to the original image.** `x = (x − pad_x) / scale` and `y = (y − pad_y) / scale`, then clip to the image.

The thresholds come from the checkpoint's training config and are recorded in the sidecar's `decode` section. The app should read them from the sidecar rather than hard-coding them.

## 6. On-device benchmark

Run LiteRT's `benchmark_model` over `adb` on the target phone, for each of `student_fp32.tflite`, `student_fp16.tflite` and `student_int8.tflite`:

```bash
adb push exports/student_int8.tflite /data/local/tmp/
adb shell /data/local/tmp/benchmark_model \
    --graph=/data/local/tmp/student_int8.tflite --num_threads=4 \
    [--use_gpu=true | --use_nnapi=true]
```

Report latency (ms), FPS and peak memory per delegate. Measure end-to-end latency, including preprocessing, decode and NMS, separately in the app. The thesis reports the size of the INT8 file on disk (`model.file_size_bytes` in its sidecar), the parameter count and the FLOPs (`complexity`). `host_cpu_interpreter_ms` of `eval_tflite.py` is a desktop-CPU figure, not a phone number.

## 7. Segment students and ONNX

A checkpoint trained with `task=segment` (YOLOv8-seg: a `Proto` head on P3 and 32 mask coefficients per anchor) exports with the same script, which reads `task` and `student_variant` from the checkpoint config. An RGB+NIR student (`student_in_channels=4`) cannot be exported: the graph has one RGB input.

```bash
# ONNX (needs `onnx` and `onnxruntime`); --verify compares it with PyTorch on random inputs, no dataset needed
CUDA_VISIBLE_DEVICES="" python scripts/export_student.py --format onnx \
    --checkpoint <run>/best_model.pt --out-dir exports --name master_rgb_seg --verify
# the same checkpoint as LiteRT, in the export venv (section 1)
CUDA_VISIBLE_DEVICES="" .venv-export/bin/python scripts/export_student.py --checkpoint <run>/best_model.pt --out-dir exports
```

`--random-init --task segment --variant nano` exports an untrained student to smoke-test the pipeline.

**Outputs.** Four, in this order, named in the sidecar (`outputs`, `segmentation`). Detect models keep the two first ones and are unchanged.

| Output | Shape (640 input, 32 prototypes) | |
|---|---|---|
| `boxes_raw` | `(1, 8400, 4)` | as in section 4 |
| `cls_logits` | `(1, 8400, 2)` | as in section 4 |
| `mask_coefs` | `(1, 8400, 32)` | mask coefficients of each anchor, the anchor order of the boxes |
| `protos` | `(1, 32, 160, 160)` | prototype masks, NCHW, at 1/4 of the input |

The ONNX input is `images`, `(1, 3, 640, 640)` float32, **NCHW** (the `.tflite` input stays NHWC). Nothing after the head is in the graph: no sigmoid, box decode, NMS or mask assembly, as in YOLOv8-seg exports.

**Assembling the masks** (reference: `decode_exported_segmentation` in `src/export/reference.py`, over `process_masks` in `src/training/decode.py`):

1. Decode the boxes and run NMS as in section 5, keeping for each surviving detection the index `i` of the anchor it came from.
2. `mask = sigmoid(mask_coefs[i] @ protos.reshape(32, 160 * 160))`, reshaped to `160 x 160`.
3. Crop it to the detection box in prototype coordinates (`box in letterboxed pixels / 4`): zero every pixel outside the box.
4. Optionally upsample bilinearly to the input size, then threshold at `0.5`. Without the upsampling the mask is at prototype resolution, the one mask AP is measured at.
5. Undo the letterbox as for the boxes (`(x - pad_x) / scale`). Training with `roi_crop` also needs the crop window added back; `scripts/evaluate_checkpoint.py` does both and stores the mask contour in the source frame.

**Parity.** `--verify` with `--format onnx` runs three random inputs through the PyTorch wrapper and the ONNX file and requires every output (`protos` and `mask_coefs` included) within `1e-3` of PyTorch (measured on an untrained `nano`: about `1e-7`) and the decoded boxes to match. `tests/export/test_segment_export.py` runs the same check, and the mask assembly from the ONNX outputs against the one from PyTorch, when `onnx` and `onnxruntime` are installed.

**LiteRT.** The wrapper, the sidecar and `make_tflite_runner` carry the four outputs. `tests/export/test_tflite_export.py::test_a_segment_student_exports_four_outputs_that_match_pytorch` needs `litert-torch` (export venv) and passes there; a random-init `nano` segment student also exports and passes the FP32 parity at the real 640 input (13.0 MiB). The prototypes leave the graph as NCHW, whatever the NHWC input.

## 8. FP16 and INT8

All three precisions start from the FP32 file litert-torch writes, so one conversion feeds them. The file always has a float32 NHWC input and float32 outputs, so the app contract of section 4 does not change.

```bash
CUDA_VISIBLE_DEVICES="" .venv-export/bin/python scripts/export_student.py \
    --checkpoint <run>/best_model.pt --out-dir exports --precision fp16 \
    --rgb-dir data/cache/v4_captura_v3_poly/rgb \
    --labels-dir data/cache/v4_captura_v3_poly/folds_lot/fold_0/labels --verify
# same line with --precision int8 (calibrates on --calib-samples 200 train images by default)
```

Outputs `exports/student_fp16.tflite` / `.json` and `exports/student_int8.tflite` / `.json` (`--name` changes the base name). `--random-init --task segment --variant nano` smoke-tests the pipeline without a checkpoint.

**FP16.** Every float32 weight constant with at least 16 values is stored as float16 and a `DEQUANTIZE` op feeds it to its consumers, so the graph still computes in float32 (about half the size). ai-edge-quantizer 0.9.0 has no float16 recipe, hence `src/export/quantize.py::weights_to_fp16` edits the flatbuffer directly.

**INT8.** Static post-training quantization with ai-edge-quantizer (`min_max_uniform_quantize`, the `default_a8w8` recipe): int8 weights per output channel (symmetric), int8 activations per tensor (asymmetric), integer compute, min/max calibration. The tool also quantizes the graph input and outputs; a float32 `QUANTIZE` op after the input and a float32 `DEQUANTIZE` op before each output restore the float interface (`src/export/quantize.py::add_float_io`), so the file reads and writes float32. The int8 scales and zero points of that internal interface are in the sidecar's `quantization.internal_interface`. The path is litert-torch (FP32 conversion) then ai-edge-quantizer, installed as pinned in `requirements-export.txt` (litert-torch 0.9.4, ai-edge-quantizer 0.9.0); the PT2E route of litert-torch (`litert_torch.quantize`) was not used.

**Measured (random-init `nano` segment student, 640 input, 3.35 M parameters, 11.4 GFLOPs):** FP32 13.02 MiB, FP16 6.72 MiB, INT8 3.63 MiB.

**Calibration.** `--calib-samples` (default 200) images of the fold's `train` split (`--calib-split`), a seeded sample (`--calib-seed`), each built exactly like a training input by `src/export/reference.py::preprocess_roi_bgr`: ROI crop to the label's mango box plus the margin, letterbox 640 with pad 114, ImageNet normalization. The sidecar's `calibration` block records the split, count, seed, directories, preprocessing and the stems used.

**Sidecar fields.** `model` (format, precision, file, sha256, `file_size_bytes`, `num_parameters`), `complexity` (parameters, FLOPs, tool, input shape), `input` / `outputs` (shape, dtype, layout, names), `preprocessing` (steps, `roi_crop`, `letterbox` size and pad value, `normalization` mean/std), `anchors` (strides, offsets), `decode` (conf 0.25, NMS 0.5, max_det 300, per-class), `segmentation` (mask threshold 0.5, `mask_ratio` 4, prototype size), `source_checkpoint` (path, sha256), `quantization`, `calibration`, `tools`.

**ROI caveat.** The crop window comes from the ground-truth mango box in evaluation, so the AP of every precision is an oracle-ROI figure. On the phone the app must supply a window (mango detector or framing guide); that stage is outside these files.

On the phone the window comes from a stage-1 mango detector (an Ultralytics YOLOv8n TFLite export with a `"format": "ultralytics"`
sidecar) in the app's "Detector (2-stage)" ROI mode; see `mobile/android/README.md`, "Two-stage detection".

## 9. Accuracy parity on a fold (real checkpoint)

Compare PyTorch (`metrics_v4.json` from `evaluate_checkpoint.py`) with the TFLite FP32, FP16 and INT8 files on the fold-0 validation split. `scripts/eval_tflite.py` builds its inputs like `evaluate_checkpoint.py` (ROI crop, letterbox, normalize), decodes with `src/export/reference.py` at the evaluation threshold 0.001 and writes the same `predictions.jsonl` (boxes, and mask polygons for a segment model in the source frame), so `scripts/v4_metrics.py` scores all of them identically.

```bash
D=data/cache/v4_captura_v3_poly; F=$D/folds_lot/fold_0; CK=<run>/best_model.pt
# 1. PyTorch reference (the experiment's config yaml; GPU or CPU)
python scripts/evaluate_checkpoint.py --config configs/experiment/<v4 config>.yaml --checkpoint $CK \
    --split val --output-dir reports/parity/pytorch --save-predictions
python scripts/v4_metrics.py --predictions reports/parity/pytorch/predictions.jsonl --index $D/index.csv \
    --labels-dir $F/labels/val --output reports/parity/pytorch/metrics_v4.json
# 2. Export the three precisions (CPU, export venv)
for p in fp32 fp16 int8; do
  CUDA_VISIBLE_DEVICES="" .venv-export/bin/python scripts/export_student.py --checkpoint $CK \
      --out-dir exports --precision $p --rgb-dir $D/rgb --labels-dir $F/labels --verify
done
# 3. TFLite predictions and metrics
for p in fp32 fp16 int8; do
  CUDA_VISIBLE_DEVICES="" .venv-export/bin/python scripts/eval_tflite.py --model exports/student_$p.tflite \
      --rgb-dir $D/rgb --labels-dir $F/labels --split val --output-dir reports/parity/$p
  .venv-export/bin/python scripts/v4_metrics.py --predictions reports/parity/$p/predictions.jsonl \
      --index $D/index.csv --labels-dir $F/labels/val --output reports/parity/$p/metrics_v4.json
done
```

What to read: latent AP50 and AP30 (and the mask twins for a segment model), face AUROC and false positives per `sano` photo, side by side. Expected: FP32 TFLite equals PyTorch up to float noise (the FP32 parity of section 3 already enforces `1e-3` on the raw outputs); FP16 is practically identical; INT8 is the one to judge. A trained model whose heads output near-constants through a collapsed activation range can saturate in INT8 (seen with random-init weights, where `cls_logits` collapse to 0); if a real checkpoint shows it, keep the last head convs in float. A drop of a point or two of AP is the usual cost of INT8; a large drop points at the calibration (try more samples, or check that the calibration split and the ROI crop match training) or at an output whose range is dominated by a few anchors (`boxes_raw` shares one scale across `dx, dy, log_w, log_h`). Use `eval_tflite.py --limit 50` for a quick check. Only after this does the INT8 file go to the phone.
