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
  Center square); a slider sets the confidence threshold; the **record** switch appends one CSV row per frame to
  `files/results/live_<timestamp>.csv`. The yellow dashed rectangle is the crop fed to the model; the thin blue rectangle is the raw
  mango box found by the colour mask. The HUD shows live FPS, inference ms, ROI / preprocessing / post-processing ms, the accelerator
  that is really running, the model name and the highest damage score.
- **Benchmark**: model, accelerator, W warm-up (default 10) and N runs (default 100), input = bundled sample crop or the last camera
  frame. Shows mean / std / p50 / p95 / min / max latency, FPS, peak RAM, model size, parameters and FLOPs, device and Android
  version. Each run is saved as `files/results/bench_<timestamp>.json` and appended to `files/results/benchmarks.csv`; "Share" opens the
  Android share sheet with both.
- **Settings**: model, accelerator (CPU XNNPACK x1/x2/x4, GPU, NPU), default ROI/overlay, camera analysis width (640/1280/1920/2560, 4:3).

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
  core/        pure Kotlin (JVM-testable): colour mask, ROI rule, letterbox, decode/NMS/masks, sidecar, stats, pipeline
  inference/   LiteRT CompiledModel session + accelerator fallback
  camera/      CameraX preview + RGBA analysis, frame rotation
  data/        settings, model catalog, result log
  device/      device info, memory probe
  ui/          live | benchmark | settings; each screen = container (view model) + presentational *Content
app/src/test/  JVM unit tests and fixtures generated from the Python reference
tools/make_fixtures.py   regenerates the fixtures (see below)
```

## Tests and fixtures

`./gradlew testDebugUnitTest` runs 48 JVM tests. The Kotlin ports are checked against the Python reference:

- colour mask: bbox IoU with `rgb_mango_mask` on 3 dataset frames downscaled to 320 px (required >= 0.95);
- ROI window, letterbox geometry and pixels (within one level of `cv2.resize`), normalization;
- decode, NMS, max-detections cap, letterbox inverse and mask assembly against `decode_exported_segmentation` on random raw outputs
  (tolerance 1e-4 on scores/boxes, masks pixel for pixel);
- sidecar parsing, output-role resolution, latency statistics, CSV, frame rotation, pipeline with a fake model.

Regenerate the fixtures (CPU only) from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python mobile/android/tools/make_fixtures.py \
    --rgb-dir data/cache/v4_captura_v3_poly/rgb
```

## Known limitations

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
