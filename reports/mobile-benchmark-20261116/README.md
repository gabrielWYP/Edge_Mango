# On-device benchmark: crosskd_rgbnir_seg fold 0 on the Xiaomi 14T (2026-11-16)

Deployed model: `crosskd_rgbnir_seg`, fold 0, seed 42 (release `thesis-runs-20261114`). Fold 0 was chosen by its
best **val** latent AP50 (0.428), not by test. Exported with litert-torch as FP32, FP16 and INT8 (ai-edge-quantizer
a8w8, default recipe); see `PARITY_fold0_val.md` for the desktop accuracy of each export.

- Device: Xiaomi 14T (2406APNFAG), MediaTek Dimensity 8300-Ultra (MT6897), Android 16 (API 36).
- App: `mobile/android` (Kotlin, CameraX, LiteRT 2.3.2 `CompiledModel`), version 0.1.0, Benchmark screen.
- Input: the last camera frame (1440x1920), color-mask ROI crop, resized to 640x640.
- Protocol: 10 warm-up runs, then 100 timed runs per row; one model and accelerator per session.
- Raw files: `raw/benchmarks.csv` and one `raw/bench_*.json` per row.

## Results

| Precision | Accelerator | Size (MB) | Load (ms) | Infer mean ± sd (ms) | Infer p95 (ms) | Infer FPS | End-to-end mean (ms) | End-to-end FPS | Peak RAM (MB) | Where it ran |
|---|---|---|---|---|---|---|---|---|---|---|
| FP32 | NPU | 13.06 | 2125 | **17.6 ± 1.2** | 19.8 | **56.8** | 236.7 | 4.2 | 475 | 100% NPU |
| INT8 | NPU | 3.65 | 1956 | **10.5 ± 1.0** | 12.8 | **95.6** | 233.3 | 4.3 | 412 | 100% NPU |
| FP16 | GPU | 6.78 | 1564 | 43.4 ± 2.9 | 47.1 | 23.1 | 246.2 | 4.1 | 529 | 100% GPU |
| FP32 | GPU | 13.06 | 1437 | 42.1 ± 3.0 | 46.1 | 23.8 | 245.2 | 4.1 | 530 | 100% GPU |
| INT8 | GPU | 3.65 | 1503 | 45.4 ± 2.3 | 48.5 | 22.0 | 249.0 | 4.0 | 504 | 100% GPU |
| FP16 | CPU (XNNPACK, 4 threads) | 6.78 | 43 | 134.3 ± 35.4 | 220.2 | 7.4 | 354.0 | 2.8 | 476 | CPU |
| FP32 | CPU (XNNPACK, 4 threads) | 13.06 | 32 | 169.1 ± 58.5 | 291.6 | 5.9 | 364.2 | 2.7 | 456 | CPU |
| INT8 | CPU (XNNPACK, 4 threads) | 3.65 | 24 | 39.8 ± 10.1 | 59.6 | 25.1 | 256.4 | 3.9 | 442 | CPU |

The model has 3,354,818 parameters and 11.4 GFLOPs at 640x640 (from the sidecar JSON), the same for every precision.
End-to-end = ROI crop + preprocessing + inference + postprocessing. Peak RAM is the process VmHWM since just
before the model load (camera and UI included). NPU load time includes the on-device JIT compilation.

Desktop val accuracy of the same exports (from `PARITY_fold0_val.md`, latent AP50): PyTorch 0.4280, TFLite FP32
0.4279, FP16 0.4277, INT8 0.2486.

## How the accelerator was verified

The app's accelerator label is the requested one, so every row was checked in logcat (tags `litert`, `tflite`,
`neuron`):

- **NPU, FP32 and INT8:** the MediaTek compiler plugin was applied and the whole graph became one
  `DispatchDelegate` node ("Replacing 1 out of 1 node(s)"), with no CPU partition.
- **NPU, FP16:** rejected by Neuron (`bias should be constant`, `dynamic weight is unsupported`, then
  `NeuronCompilation_finish failed with error 6`), so LiteRT fell back to XNNPACK. The FP16 file stores the
  weights as fp16 constants followed by DEQUANTIZE ops, so the convolutions see non-constant weights. No FP16 NPU
  row is reported.
- **GPU:** every op went to the `LITERT_CL` delegate in one partition (432/432 for FP16, 286/286 for FP32, 324/324
  for INT8).
- **CPU:** XNNPACK took all ops but 2.

The NPU only worked after the APK extracted its native libraries (`jniLibs.useLegacyPackaging = true`). Without
that, LiteRT logged "No compiler plugin found" and every NPU request silently ran on the CPU.

## Findings

1. The NPU is the fastest backend: FP32 on the NPU is 9.6x faster than on the CPU (17.6 vs 169.1 ms) and 2.4x
   faster than on the GPU, with a much smaller spread (sd 1.2 ms vs 58.5 ms on the CPU).
2. INT8 is the fastest and smallest variant (10.5 ms on the NPU, 3.65 MB), but its val latent AP50 drops from
   0.428 to 0.249 with the default a8w8 recipe. It is not usable as exported.
3. End-to-end latency is about 235 ms (about 4 FPS) on every accelerator. The bottleneck is the Kotlin color-mask
   ROI and preprocessing of the 2.7 MP frame (the ROI step alone measured about 254 ms in the live view), not the
   network.

## Limitations

- Accuracy was not measured on the device. In particular, Neuron may run FP32 graphs with fp16 arithmetic, so the
  NPU FP32 accuracy is assumed equal to the desktop TFLite FP32 one, not shown.
- One device, one session, one fold; each row is 100 runs on the same frame, so the spread is run-to-run timing
  noise, not variation across images.
- Mixed INT8 (head kept in float) was not exported; the INT8 row is the collapsed default recipe.
- The ROI crop uses a color mask on the capture backdrop, so it assumes the capture setup of the dataset.
