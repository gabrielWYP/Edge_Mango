# Fold-0 val parity: PyTorch vs TFLite (seed 42, oracle-ROI crop, val only)

Checkpoints: thesis-runs-20261114 fold0_seed42. All three students are RGB-only (student_in_channels=3). Code: origin/main 44ec75b (PR #104). INT8: 200 train calibration images, default recipe (no float head).

### master_rgb_seg  (sizes MiB: fp32 13.06, fp16 6.78, int8 3.65)

| model | latent_ap50 | latent_ap30 | latent_mask_ap50 | latent_mask_ap30 | face_auroc | sano_fp/img |
|---|---|---|---|---|---|---|
| PyTorch | 0.4671 | 0.6201 | 0.4715 | 0.6100 | 0.6501 | 4.6000 |
| TFLite FP32 | 0.4671 | 0.6201 | 0.4715 | 0.6099 | 0.6502 | 4.6000 |
| FP16 | 0.4670 | 0.6200 | 0.4715 | 0.6099 | 0.6502 | 4.6000 |
| INT8 | 0.3640 | 0.5149 | 0.3671 | 0.5024 | 0.6539 | 5.4900 |

### crosskd_rgb_seg  (sizes MiB: fp32 13.06, fp16 6.78, int8 3.65)

| model | latent_ap50 | latent_ap30 | latent_mask_ap50 | latent_mask_ap30 | face_auroc | sano_fp/img |
|---|---|---|---|---|---|---|
| PyTorch | 0.4542 | 0.6298 | 0.4619 | 0.6157 | 0.6439 | 3.8700 |
| TFLite FP32 | 0.4543 | 0.6299 | 0.4620 | 0.6158 | 0.6441 | 3.8700 |
| FP16 | 0.4544 | 0.6298 | 0.4621 | 0.6158 | 0.6441 | 3.8900 |
| INT8 | 0.3018 | 0.4618 | 0.3295 | 0.4525 | 0.6289 | 4.7400 |

### crosskd_rgbnir_seg  (sizes MiB: fp32 13.06, fp16 6.78, int8 3.65)

| model | latent_ap50 | latent_ap30 | latent_mask_ap50 | latent_mask_ap30 | face_auroc | sano_fp/img |
|---|---|---|---|---|---|---|
| PyTorch | 0.4280 | 0.5991 | 0.4402 | 0.5960 | 0.6570 | 3.6500 |
| TFLite FP32 | 0.4279 | 0.5991 | 0.4404 | 0.5959 | 0.6570 | 3.6500 |
| FP16 | 0.4277 | 0.5991 | 0.4402 | 0.5960 | 0.6570 | 3.6300 |
| INT8 | 0.2486 | 0.4905 | 0.3230 | 0.4877 | 0.6690 | 5.9400 |

INT8 collapses vs FP32 (latent AP50 -0.10/-0.15/-0.18); the exporter has no flag to keep head convs in float (recipe regex is hardcoded to .*), so no retry was run. boxes_raw int8 output scale is ~0.61 per step.
