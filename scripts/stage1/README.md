# Stage-1 mango detector

The damage student cannot gate "mango vs background" on a full frame: it was trained only on mango crops and scores
its mango class at 1.00 on pure background. This folder trains a separate single-class YOLOv8n detector that the
Android app runs first (ROI mode `DETECTOR`) before the damage model.

1. `make_cutouts.py <out_dir> 800`: RGBA mango cutouts from `data/cache/v4_captura_v3_poly`, using the class-0
   contour and stratified by lot and hour.
2. `gen_synth.py`: pastes 0-6 cutouts on COCO val2017 backgrounds with scale, rotation, ripeness hue, shadow, noise
   and JPEG jitter. 30 % of the backgrounds contain bananas, apples or oranges (hard negatives). Validation uses lot-5
   fruit and held-out backgrounds. `--real` adds full capture frames.
3. `run_stage1.sh`: run on a rented GPU. It installs Ultralytics, downloads COCO val2017, generates 6000 + 800
   images and trains `yolov8n.pt` (patience 30). Export TFLite on a CPU machine with `yolo export format=tflite`
   (the GPU box lacked TensorFlow).

Run of 2026-11-21 (Thunder A6000, stopped by hand at epoch 96 of 150, about 1.7 h):

- Synthetic validation (900 images: lot-5 fruit on held-out backgrounds plus 100 real frames): P 0.994, R 0.978,
  mAP50 0.994, mAP50-95 0.988.
- TFLite FP32 on 30 real lot-5 frames, full frame: best-box IoU against the mango box had a median of 0.984
  (minimum 0.957).
- Frame strips without a mango: maximum score had a median of 0.000.

No market images exist yet. The synthetic validation does not show that the detector generalises to a market;
the field test does. Artifacts are outside git, in `data/cache/mobile_models/mango_stage1/`, as `.pt`, `.tflite`
FP32/FP16 and sidecars.
