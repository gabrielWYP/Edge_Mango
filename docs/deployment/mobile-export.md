# Mobile export of the student detector

This guide covers exporting the RGB-only `StudentModel` (plain or KD) to an FP32 LiteRT `.tflite` file (or an ONNX file), checking it against PyTorch, and what the Android app has to implement around it. Section 7 covers segment students.

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
- **Precision:** `--precision` only accepts `fp32`. It is where FP16 and INT8 will be added.

## 3. Verify

`--verify` runs the parity check after the export. To run it on its own:

```bash
.venv-export/bin/python scripts/check_export_parity.py \
    --model exports/student_fp32.tflite --data-root data --split val --num-images 20
```

The check sends the same preprocessed images through the PyTorch wrapper and the LiteRT interpreter, then compares the results in two ways:

- **Raw outputs:** the max abs diff of each output must be below `1e-3`.
- **Detections:** both sets of outputs are decoded with the reference decode and NMS. Boxes are matched when they have the same class and IoU ≥ 0.9. At least 99% must match.

The script exits with status 1 if either check fails.

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

## 6. Next steps

- **INT8 PTQ.** Use `ai-edge-quantizer`, which is already installed with litert-torch. Calibrate on about 100–200 letterboxed **train** images. Re-run the parity check with a looser, INT8-specific tolerance and compare mAP on val/test against FP32 with `scripts/evaluate_checkpoint.py`. Keep the float input and outputs at first, or quantize the I/O and record the scale and zero-point in the sidecar.
- **FP16.** An FP16-weight variant is about half the size. It is the natural format for GPU delegates.
- **On-device benchmark.** Run LiteRT's `benchmark_model` over `adb` on the target phone:

  ```bash
  adb push exports/student_fp32.tflite /data/local/tmp/
  adb shell /data/local/tmp/benchmark_model \
      --graph=/data/local/tmp/student_fp32.tflite --num_threads=4 \
      [--use_gpu=true | --use_nnapi=true]
  ```

  Report latency (ms), FPS and peak memory per delegate. Measure end-to-end latency, including preprocessing, decode and NMS, separately in the app.

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

**LiteRT.** The wrapper, the sidecar and `make_tflite_runner` carry the four outputs, but `tests/export/test_tflite_export.py::test_a_segment_student_exports_four_outputs_that_match_pytorch` has not been run: it needs `litert-torch`, which the training venv does not have. Run it in the export venv before shipping a segment `.tflite`. The prototypes leave the graph as NCHW, whatever the NHWC input.
