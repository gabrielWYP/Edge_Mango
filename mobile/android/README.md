# Edge Mango - Android live detector and benchmark

Kotlin + Jetpack Compose + CameraX + LiteRT (`CompiledModel` API, no NDK code). It opens the camera, crops the mango with the
same colour segmentation and margin that training used, runs the exported student (detect or segment), and draws boxes
and/or damage masks on the preview. A second screen measures the thesis metrics on a fixed input and saves them locally.

Target device: Xiaomi 14T (Dimensity 8300-Ultra, Android 14/15). minSdk 26, compileSdk 36, targetSdk 35, arm64-v8a only.

## Build

Requirements: JDK 17 and the Android SDK (platform 36, build-tools 35.0.0, platform-tools). `local.properties` (git-ignored)
points at the SDK:

```bash
echo "sdk.dir=$HOME/Android/Sdk" > local.properties
cd mobile/android
./gradlew testDebugUnitTest assembleDebug      # unit tests + APK
# APK: app/build/outputs/apk/debug/app-debug.apk
```

Versions (see `gradle/libs.versions.toml`): Gradle 8.14.3, AGP 8.13.2, Kotlin 2.3.21, Compose BOM 2025.12.01, CameraX 1.6.2,
LiteRT `com.google.ai.edge.litert:litert`, `litert-gpu` and `litert-npu-runtime-mediatek` 2.3.2. The newest AndroidX releases need
AGP 9 and compileSdk 37, so the AndroidX libraries are pinned to the last ones that build with AGP 8.13 / SDK 36.

## Install on the phone

1. On the phone: Settings > About phone > tap the MIUI/HyperOS version 7 times; then Additional settings > Developer options >
   enable USB debugging (and "Install via USB" on Xiaomi).
2. Connect it by USB and accept the debugging prompt, then:

```bash
adb devices                                   # the phone must be listed as "device"
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

3. Open "Edge Mango" and grant the camera permission.

## Screens

- **Live**: camera preview with the overlay. Chips switch **Boxes / Mask / Both** and the **ROI mode** (Color mask, Full frame,
  Center square, Detector (2-stage), see below); a slider sets the confidence threshold; the **session** switch records a
  field test (see "Field-test sessions"); the **record** switch appends one CSV row per frame to
  `files/results/live_<timestamp>.csv`. The yellow dashed rectangle is the crop fed to the model; the thin blue rectangle is the raw
  mango box found by the colour mask. The HUD shows live FPS, inference ms, ROI / preprocessing / post-processing ms, the accelerator
  that is really running, the model name and the highest damage score.
- **Benchmark**: model, accelerator, W warm-up (default 10) and N runs (default 100), input = bundled sample crop or the last camera
  frame. Shows mean / std / p50 / p95 / min / max latency, FPS, peak RAM, model size, parameters and FLOPs, device and Android
  version. Each run is saved as `files/results/bench_<timestamp>.json` and appended to `files/results/benchmarks.csv`; "Share" opens the
  Android share sheet with both.
- **Settings**: model, accelerator (CPU XNNPACK x1/x2/x4, GPU, NPU), default ROI/overlay, the stage-1 detector, mango threshold,
  max mangos, NMS IoU (0.5 / 0.6 / 0.7, applied to both stages instead of the sidecar value), class-agnostic NMS for the damage
  model, camera analysis width (640/1280/1920/2560, 4:3).

Pull the logs without sharing: `adb shell run-as com.edgemango.app ls files/results` and
`adb exec-out run-as com.edgemango.app cat files/results/benchmarks.csv > benchmarks.csv`.

## Loading a new model

The app needs the `.tflite` and, next to it, the sidecar `.json` that `scripts/export_student.py` writes (same base name). Only
`task` (`detect` | `segment`) and the input size are required in the sidecar; everything else defaults to the contract in
`docs/deployment/mobile-export.md` (ImageNet normalization, pad 114, conf 0.25, NMS IoU 0.5, max 300, strides 8/16/32). Without a sidecar
the app guesses the task from the file name (`seg` -> segment) and says so. Output tensors are identified by element count, so the
order inside the file does not matter.

Two ways, no rebuild:

```bash
# A) adb: the app scans this folder every time a screen opens (no storage permission needed)
adb push student_seg_nano_fp32.tflite student_seg_nano_fp32.json \
    /sdcard/Android/data/com.edgemango.app/files/models/

# B) in the app: Benchmark or Settings > "Pick model file(s)", select the .tflite and the .json together
#    (they are copied to the app's private storage)
```

Then choose it in Settings (live) or Benchmark. To make a new model the bundled default instead, put both files in
`app/src/main/assets/models/` and change `ModelCatalog.DEFAULT_ID`.

The bundled `student_seg_nano_random_fp32.tflite` is a **random-init** segment nano (untrained, FP32) exported with

```bash
CUDA_VISIBLE_DEVICES="" .venv-export/bin/python scripts/export_student.py --random-init --task segment --variant nano \
    --format tflite --out-dir <dir> --name student_seg_nano_random_fp32
```

It runs end to end (so timings, memory and size are real for that architecture) but detects nothing: its class bias gives every
score about 0.01. Replace it with a trained checkpoint's export as soon as one exists.

## Two-stage detection (stage-1 mango detector)

The damage model was only ever trained on mango crops, so its own "mango" class scores about 1.00 even on background: it cannot
tell a frame with a mango from one without. The ROI mode **Detector (2-stage)** adds a separate stage-1 mango detector:

1. Stage 1 runs on the full frame. Boxes with score >= *mango threshold* (default 0.5) are kept, best first, at most *max mangos*
   (default 10). **No mango box means "no mango" and stage 2 does not run.**
2. For each mango the crop is the box plus 10 % margin (`RoiRule.window`), letterboxed to the damage model's input; only class-1
   (damage) detections whose box centre lies inside the mango box are kept, with their masks.
3. A greedy IoU tracker (IoU >= 0.3, dropped after 15 unseen frames) gives each mango a stable `trackId` shown on the overlay.

The stage-1 model is a stock Ultralytics YOLOv8n, one class `mango`, exported with `yolo export format=tflite imgsz=640`. Put the
`.tflite` and a sidecar with the same base name next to the other models (`adb push` or "Pick model file(s)"), then choose it in
Settings > "Stage-1 mango detector". Sidecar template (`format: ultralytics` selects the decoder; everything else is optional):

```json
{
  "format": "ultralytics",
  "task": "detect",
  "input": {"shape": [1, 640, 640, 3], "layout": "NHWC", "dtype": "float32"},
  "class_names": ["mango"],
  "preprocessing": {
    "normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]},
    "letterbox": {"target_size": 640, "pad_value": 114}
  },
  "decode": {"conf_threshold": 0.25, "nms": {"iou_threshold": 0.5}, "max_detections": 300}
}
```

Assumed output contract of that export: input `(1, 640, 640, 3)` float32 RGB in [0,1] (letterboxed, pad 114); one output
`(1, 4+nc, N)` (`N = 8400`) or `(1, N, 4+nc)` with `cx, cy, w, h, score...`, scores already sigmoid, boxes normalized to [0,1] or in
input pixels (decided by magnitude: max <= 2 means normalized), no NMS inside. The layout is guessed from the data; add
`"output_layout": "channels_first"` (or `"channels_last"`) to the sidecar if a model gets mis-read. Verify the first real export by
looking at the overlay on the phone before trusting a session.

**COCO gate (recommended stage 1).** The single-class detectors trained on our data failed on real photos (6/22 web mango photos),
while a stock COCO YOLO11n counting its fruit classes as mango found 17/17 mango photos, 40/40 capture-box frames (median IoU
0.95) and 0/40 false positives on frame strips without fruit. Use the stock 80-class model and add `keep_classes` to the sidecar:
the listed COCO ids (46 banana, 47 apple, 49 orange) are the only columns scored, and detections are relabelled as class 0
(mango); the overlay always draws stage-1 boxes as "mango", whatever `class_names[0]` says.

```json
{
  "format": "ultralytics",
  "task": "detect",
  "input": {"shape": [1, 640, 640, 3], "layout": "NHWC", "dtype": "float32"},
  "output_layout": "channels_first",
  "keep_classes": [46, 47, 49],
  "class_names": ["<the 80 COCO names in order>"],
  "preprocessing": {
    "normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]},
    "letterbox": {"target_size": 640, "pad_value": 114}
  },
  "decode": {"conf_threshold": 0.25, "nms": {"iou_threshold": 0.5}, "max_detections": 300}
}
```

## Field-test sessions

A session records what the app saw and predicted, to evaluate it later with ground truth from Label Studio.

1. Live > ROI "Detector (2-stage)" > switch **session** on. The app creates `files/results/session_<yyyyMMdd_HHmmss>/` with
   `frames/f_000000.jpg ...` (the upright analysed frame, JPEG q=90, every processed frame), `predictions.jsonl` (one line per frame: index,
   wall-clock ms, camera timestamp, frame size, ROI mode, thresholds, stage-1 mangos with track id and score, stage-2 damage per mango with
   box and mask polygon in frame pixels, max damage per mango, timings, accelerator, model names) and `session.json` (device, models and
   their sidecars, settings, app version, start/end, counters). JPEG writing runs on one background thread with a bounded queue; if
   it cannot keep up a frame is dropped from disk (its prediction line stays, `frame: null`) and counted in the HUD and `session.json`.
2. Switch it off (or leave the Live tab): the folder is zipped to `session_<stamp>.zip`; **Share session** opens the share sheet.
3. Or pull it with adb (the zip, or the folder):

```bash
adb exec-out run-as com.edgemango.app cat files/results/session_20261121_101500.zip > session.zip
```

4. Evaluate on the PC (`scripts/field_eval.py`, repo `.venv`; needs `torch` only for `eval`):

```bash
python scripts/field_eval.py report    session.zip --out reports/field/market1            # rates, tracks, confidence, latency
python scripts/field_eval.py export-ls session.zip --every 10 --out reports/field/market1/ls   # or --per-track 3
#   import ls/tasks.json in Label Studio with ls/labeling_config.xml, serve ls/frames as local files, fix the pre-annotations
python scripts/field_eval.py eval      session.zip --gt ls_export.json --out reports/field/market1
```

`eval` reports damage box AP50/AP30, mask AP50 (polygon ground truth), mango AP50, per-mango (track) damage AUROC, recall and specificity
at the session's confidence, and false-positive damage boxes per healthy mango. The assumptions behind each number are in the docstring
of `scripts/field_eval.py`.

## What each metric means and how it is measured

| Metric | Definition |
|---|---|
| Latency per inference | Wall time (`System.nanoTime`) of one `infer()`: copy the preprocessed input tensor into the LiteRT buffer, run the model, copy every output tensor out as float arrays. W warm-up runs are discarded, then N runs are timed one by one on the same input. Reported: mean, sample std, p50, p95 (linear interpolation, as numpy), min, max. |
| End-to-end latency | Same W/N, per frame: ROI (colour mask), crop + letterbox + normalize, inference, decode + NMS + mask assembly. |
| FPS (benchmark) | `1000 / mean latency`: sequential single-image throughput. |
| Live FPS (HUD) | Frames per second the live pipeline completes (exponential average). It includes camera delivery, RGBA conversion, ROI, preprocessing, inference and post-processing, and `KEEP_ONLY_LATEST` drops frames when it is slower than the camera. |
| Peak RAM | `VmHWM` of `/proc/self/status`: high-water mark of the resident memory of the whole app process during model load + all runs. The app first writes `5` to `/proc/self/clear_refs` to reset it (if the kernel refuses, the record says the peak is since process start). It includes the app's own baseline, so the record also stores `RSS before model load`, plus `Debug.getNativeHeapAllocatedSize()` (where LiteRT keeps weights and activations). |
| Model size | Size of the `.tflite` file on disk. |
| Parameters / FLOPs | Read from the sidecar (`model.num_parameters`; `flops`, `gflops`, `macs` or `gmacs` in the root, `model` or `complexity`; 1 MAC = 2 FLOPs). Shown as "not in sidecar" when absent. |

The ROI step mirrors training: the mango box comes from `rgb_mango_mask` (ported to Kotlin, run on the frame averaged down to ~320 px wide),
the window is that box plus 10 % of its width/height on each side, snapped outward to integers and clipped (`src/training/roi_crop.py`; the window
is not squared), then letterboxed to 640 with pad 114. If no mango is found the whole frame is used, as in training.

## Layout

```
app/src/main/java/com/edgemango/app/
  core/        pure Kotlin (JVM-testable): colour mask, ROI rule, letterbox, decode/NMS/masks (edge_mango and Ultralytics),
               sidecar, stats, single-crop and two-stage pipelines, tracker, mask polygons, session JSON format
  inference/   LiteRT CompiledModel session + accelerator fallback
  camera/      CameraX preview + RGBA analysis, frame rotation
  data/        settings, model catalog, result log, field-test session recorder (+ zip)
  device/      device info, memory probe
  ui/          live | benchmark | settings; each screen = container (view model) + presentational *Content
app/src/test/  JVM unit tests and fixtures generated from the Python reference
tools/make_fixtures.py   regenerates the fixtures (see below)
```

## Tests and fixtures

`./gradlew testDebugUnitTest` runs 91 JVM tests. The Kotlin ports are checked against the Python reference:

- colour mask: bbox IoU with `rgb_mango_mask` on 3 dataset frames downscaled to 320 px (required >= 0.95);
- ROI window, letterbox geometry and pixels (within one level of `cv2.resize`), normalization;
- decode, NMS, max-detections cap, letterbox inverse and mask assembly against `decode_exported_segmentation` on random raw outputs
  (tolerance 1e-4 on scores/boxes, masks pixel for pixel);
- sidecar parsing, output-role resolution, latency statistics, CSV, frame rotation, pipeline with a fake model;
- not checked against Python (no reference export yet): the Ultralytics decoder (synthetic tensors), the two-stage pipeline and the
  tracker (fake models), NMS IoU / agnostic NMS, mask polygons and the session JSON format.

Regenerate the fixtures (CPU only) from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python mobile/android/tools/make_fixtures.py \
    --rgb-dir data/cache/v4_captura_v3_poly/rgb
```

## Known limitations

- The Ultralytics stage-1 contract (tensor layout, normalized boxes) is an assumption until a real export has run on the phone.
- Leaving the Live tab ends a running session (the model is released), and a mid-session settings change that reloads a model
  pauses recording until it is ready again.
- The colour ROI relies on the capture backdrop (cyan/white sheet, yellow fruit). On another background, switch to Full frame or Center
  square, which are not what the model was trained on (it saw ROI crops).
- Live FPS includes camera delivery, RGBA conversion, the colour mask and the overlay, not just the network.
- Training frames were 3264x2448 and the mango crop was resized to 640. With a 1280x960 camera stream the mango has about a third of the
  pixels, so a higher analysis width (Settings) can matter for small damage. Check it against the real checkpoint.
- GPU and NPU: LiteRT `CompiledModel` may refuse the model (unsupported ops) or fall back partly to CPU; the app retries on CPU x4 and
  reports the accelerator that is actually running. The MediaTek NPU path is experimental: it depends on the device's vendor runtime.
  FP32 models are the safe choice for CPU/GPU; INT8/FP16 models depend on what the exporter and the delegate support.
- Nothing here has been run on a phone yet (no device was available while writing it). Treat the first on-device run as the real test of the
  camera, accelerator and memory parts; the pre/post-processing is covered by the JVM tests.
- The candidate cap before NMS (3000 by default) only matters for degenerate models; a trained one stays far below it.
