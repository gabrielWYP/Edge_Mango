#!/usr/bin/env bash
# Stage-1 mango detector on a rented GPU. Run from the unpacked folder: bash run_stage1.sh
set -euo pipefail
cd "$(dirname "$0")"
LOG=run_stage1.log; exec > >(tee -a "$LOG") 2>&1
echo "== $(date -u) start"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m pip install -q --upgrade pip
python3 -m pip install -q "ultralytics==8.3.*" opencv-python-headless
mkdir -p coco && cd coco
[ -d val2017 ] || { curl -sSL -o val2017.zip http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip && rm val2017.zip; }
[ -f annotations/instances_val2017.json ] || { curl -sSL -o ann.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q ann.zip annotations/instances_val2017.json && rm ann.zip; }
cd ..
echo "== $(date -u) generate"
[ -f synth/data.yaml ] || python3 gen_synth.py --cutouts cutouts --coco coco --real real --out synth --n-train 6000 --n-val 800
echo "== $(date -u) train"
yolo detect train model=yolov8n.pt data=synth/data.yaml imgsz=640 epochs=150 patience=30 batch=64 \
     workers=8 cos_lr=True close_mosaic=10 degrees=15 hsv_h=0.02 seed=0 project=runs name=mango_stage1 exist_ok=True
BEST=runs/mango_stage1/weights/best.pt
echo "== $(date -u) val"
yolo detect val model=$BEST data=synth/data.yaml imgsz=640 split=val project=runs name=val exist_ok=True | tee val.txt
echo "== $(date -u) export"
yolo export model=$BEST format=tflite imgsz=640
mkdir -p out && cp $BEST out/mango_stage1.pt && cp runs/mango_stage1/results.csv out/ && cp val.txt out/
cp runs/mango_stage1/weights/best_saved_model/best_float32.tflite out/mango_stage1_fp32.tflite
cp runs/mango_stage1/weights/best_saved_model/best_float16.tflite out/mango_stage1_fp16.tflite 2>/dev/null || true
cp runs/mango_stage1/*.png out/ 2>/dev/null || true
tar -czf stage1_out.tar.gz out
echo "== $(date -u) done"; ls -la out
