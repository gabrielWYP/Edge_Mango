# Augmentation and sampler audit (2026-10-22)

**Question.** Thesis section 5.4.4 says the image-weighted sampler compensates the damage-class imbalance, and that the online augmentation is "flips, rotations and perspective, plus CLAHE and ColorJitter on RGB". Is that what the code does, and does each transform help?

**Short answer.** The text does not match the code. **No run in the repository has measured the effect of any augmentation on AP.** The training config has no switch to turn a family of transforms off. So the "does it help" part can only be answered by an ablation. This document records what can be verified without training: what the code applies, three mechanisms measured on the 252 labeled pairs, and the ablation that would settle it.

Reproduce (CPU only):

```bash
python -I docs/findings/2026-10-22-augmentation/tools/aug_audit.py data/cache/mango/rgb data/annotations/yolo/labels
```

Output: [evidence/aug_audit.txt](evidence/aug_audit.txt).

## 1. What the code applies

Source: `src/training/augmentations.py` (albumentations 2.0.8), used by `YOLODataset` for train splits.

| Family | Transform (probability) | Applied to |
| :-- | :-- | :-- |
| Dihedral (exact) | HorizontalFlip (0.5), VerticalFlip (0.1), RandomRotate90 (0.3), Transpose (0.1) | RGB, NIR and boxes, same parameters |
| Continuous geometric | Rotate ±20° (0.4, black border), Perspective scale 0.02-0.05 (0.2) | RGB, NIR and boxes, same parameters |
| Color | RandomBrightnessContrast ±0.25 (0.5), HueSaturationValue hue 15 / sat 30 / val 20 (0.4), ColorJitter 0.15 / hue 0.05 (0.3) | RGB only |
| Local contrast, blur, noise | CLAHE clip 2.0, 8x8 tiles (0.2), GaussianBlur 3-5 px (0.2), GaussNoise std 0.02-0.08 (0.2), ISONoise (0.15) | RGB only |
| Occlusion | CoarseDropout 1-4 holes of 8-32 px, fill 114 (0.2) | RGB only, boxes unchanged |

The thesis text names 5 of these 14 transforms.

## 2. Measured mechanisms

| # | Observation | Evidence |
| :-- | :-- | :-- |
| 1 | **The weighted sampler is a no-op.** `build_weighted_sampler` weights each image by the largest inverse-frequency weight among its classes. Every image has a mango box, and in this corpus mango is the rarer instance class (252 mango vs 390 damage boxes), so every image gets the same weight, 1.27. Sampling is uniform with replacement. The damage class is not compensated by the sampler. The per-class loss weights (`class_weights: [0.5, 1.5]`) are what compensates it. | `distinct per-image weights [1.27]` |
| 2 | **Rotate ±20° inflates the damage boxes.** With the default `rotate_method="largest_box"`, a damage box rotated 20° becomes an axis-aligned box 1.66x larger (median), so the label overlaps the object with IoU ≤ 0.60. At 10° the same geometry gives about 1.34x. With `rotate_method="ellipse"` the inflation is 1.01x. Damage boxes are small (median side 29 px at the 640 px letterbox, p10 20 px), so this is regression-target noise on the class that matters. | `box area x1.66` vs `x1.01` |
| 3 | **The hue shifts are large.** `HueSaturationValue(hue_shift_limit=15)` shifts hue up to ±30° on uint8 images (OpenCV hue units). `ColorJitter(hue=0.05)` adds up to ±18°. Both can apply to the same image (up to ±48°), and three transforms overlap on brightness, contrast and saturation. | `max hue shift +30 deg`, `+18 deg` |

**Unexpected, not interpreted further.** Against a local ring around each box, the damage region is about 25-29 gray levels **brighter** in R and G (median), not darker. One possible explanation is that the impacts land on the most illuminated zone (central, specular). This needs a visual check before it is used anywhere. Because of it, this audit does not estimate a bruise contrast to compare against GaussNoise (std 5-20 gray levels).

## 3. Assessment per family (reasoned; not measured on AP)

- **Dihedral: keep.** It is exact: no interpolation, no box inflation. The fruit is placed by hand at any orientation, so the invariance is real.
- **None of the spatial transforms breaks the central positional prior** ([2026-10-18 finding 2](../2026-10-18/README.md)). Flips, 90° rotations, transposition and a rotation about the image center all keep a central box central. Breaking the prior needs translation and scale (Affine shift/scale or a jittered ROI crop). The protocol v3 random impact zone also breaks it at the data level.
- **Rotate ±20°: switch to `rotate_method="ellipse"`** (or reduce the limit). The dihedral group already covers coarse orientation.
- **Perspective: low value.** The rig is fixed top-down at a fixed distance, so test images have no perspective variation.
- **Color: reduce to one mild transform.** The capture protocol fixes exposure and calibrates white balance per session, so test images have low photometric variance; ±48° of hue simulates variation the protocol removed. Contrast jitter (factor 0.75-1.25) scales any weak local damage signal down by up to 25%.
- **Photometric transforms on RGB only: a confound for the modality ablations.** The NIR stays clean while the RGB is perturbed. The RGB+NIR teacher can fall back on the NIR; an RGB-only teacher cannot. A teacher modality ablation run with this pipeline measures the value of NIR when RGB is degraded, which overstates it relative to clean test images. Run the modality ablations with the same photometric regime for both arms, or with mild photometric augmentation.
- **CLAHE: candidate to drop.** It enhances local contrast, the cue the detector needs, and it amplifies lenticels, scars and noise. Test images never get it.
- **Blur and noise: keep mild.** Use one noise transform, with a lower upper bound.
- **CoarseDropout: rare and conceptually label-noisy.** The holes are as large as a damage box (8-32 px vs median 29 px), and the boxes are not updated. The chance of hitting a damage box is small, below 1% of samples.

## 4. The ablation that answers "does it help"

Use the same folds and seeds and pair by fold, with the experiment-statistics tooling (corrected t plus a pre-registered ROPE).

1. A1: no augmentation (validation transforms in training).
2. A2: dihedral only.
3. A3: A2 plus geometric fixes (ellipse rotation, plus translation/scale).
4. A4: A3 plus one mild color transform.
5. A5: the current full pipeline.

This needs a config switch per family. None exists today.

## Resumen

El texto de la sección 5.4.4 no coincide con el código. El pipeline aplica 14 transformaciones y la tesis nombra 5. El muestreo ponderado por imagen no compensa el desbalance: todas las imágenes reciben el mismo peso, porque todas contienen una caja de mango; la compensación real proviene de los pesos por clase en la pérdida. La rotación de ±20 grados agranda las cajas de daño hasta 1.66 veces (IoU de 0.60 o menos); con el método `ellipse` el agrandamiento es 1.01 veces. Los corrimientos de tono llegan a ±30 y ±18 grados. Ninguna corrida midió el efecto de las aumentaciones sobre el AP, de modo que su utilidad solo puede establecerse con una ablación por familias.
