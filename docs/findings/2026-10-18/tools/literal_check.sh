#!/usr/bin/env bash
# Analysis tool for the 2026-10-18 findings, not part of the training pipeline.
#
# Cross-check: run the repo's own scripts/evaluate_checkpoint.py (CPU only) on three checkpoints with a given NMS IoU and
# compare its damage AP50 / TP / FP / FN with nms_sensitivity.json (the re-implemented decode in fp_analyze.py).
#
# Usage: literal_check.sh [--repo-root DIR] [--data-root DIR] [--out-dir DIR]
#   --repo-root  code checkout (default: the repository this script lives in)
#   --data-root  checkout holding data/ and checkpoints/ (default: --repo-root)
#   --out-dir    where eval/<tag>/{metrics.json,stdout.txt} are written (default: <repo-root>/reports/findings-2026-10-18-work)
# Environment: PYTHON selects the interpreter (default: python).
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
M=""
OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --repo-root) REPO="$2"; shift 2 ;;
    --data-root) M="$2"; shift 2 ;;
    --out-dir) OUT="$2"; shift 2 ;;
    -h|--help) echo "Usage: literal_check.sh [--repo-root DIR] [--data-root DIR] [--out-dir DIR]  (env PYTHON selects the interpreter)"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
M="${M:-$REPO}"
OUT="${OUT:-$REPO/reports/findings-2026-10-18-work}"
PY="${PYTHON:-python}"
mkdir -p "$OUT"
cd "$M"
run() { # fold ckpt nms tag
  mkdir -p $OUT/eval/$4
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $PY -B $REPO/scripts/evaluate_checkpoint.py --config $REPO/configs/experiment/twostream_roi_crop.yaml --checkpoint checkpoints/kfold_lot/twostream_roi_crop/fold$1/seed42/$2 --split test --model master --device cpu --output-dir $OUT/eval/$4 --override labels_dir=data/annotations/yolo/folds_lot/fold_$1/labels split_manifest=data/annotations/yolo/folds_lot/fold_$1/splits.json device=cpu batch_size=2 num_workers=2 output_dir=$OUT/trainer_out nms_iou_threshold=$3 > $OUT/eval/$4/stdout.txt 2>&1
}
run 0 best_model.pt 0.5 f0_nms0.5
run 1 best_model.pt 0.4 f1_nms0.4
run 4 checkpoint_epoch100.pt 0.3 f4e100_nms0.3
echo ALLDONE > $OUT/eval/literal_done.txt
