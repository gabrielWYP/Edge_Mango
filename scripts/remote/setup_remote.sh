#!/usr/bin/env bash
# Prepare a rented Ubuntu GPU machine (NVIDIA driver installed, Python 3.12 preferred) for the thesis runs.
# Idempotent: every step checks what is already done, so re-running after a failure or a reboot is safe.
#
#   scripts/remote/setup_remote.sh --data ARCHIVE --workdir DIR [options]
#
#     --data FILE         the .tar.zst/.tar.gz of scripts/remote/package_v4.sh (required)
#     --workdir DIR       everything lives here: repo, .venv, data/, runs/ (required)
#     --repo URL          default https://github.com/gabrielWYP/Edge_Mango.git
#     --branch NAME       default main (the branch with the thesis preset)
#     --dataset-name N    folder of the dataset under <workdir>/data/cache (default v4_captura_v3_poly)
#     --weights FILE      YOLOv8n state_dict the arms start from; recorded in env.sh (optional)
#     --machine CFG       configs/machines/<x>.yaml; default rtx_a5000 (>= 20 GB of GPU memory) or rtx3080
#     --python BIN        default python3.12, else python3
#     --torch-index URL   wheel index of torch, e.g. cu126, cu128, cpu or a full URL (overrides the detection);
#                         the environment variable TORCH_INDEX does the same
#     --skip-cache        do not build the pixel cache
#     --skip-tests        do not run the quick pytest subset
#     --smoke             after the setup, train 2 epochs of master_rgb_seg on fold 0 into <workdir>/runs/smoke and
#                         print seconds per epoch (the number the cost estimate needs)
#     --smoke-arm ARM     default master_rgb_seg
#     --image-size N      input side of the smoke run, recorded in env.sh (default 640, the thesis value)
#
# GitHub: the script never stores a secret. If the repository is private, log in once by hand with the device flow:
#   gh auth login -h github.com -p https -w     (then re-run this script)
#
# Afterwards: source <workdir>/env.sh   (defines PY, REPO, DATASET, PIXEL, RUNS, WEIGHTS, MACHINE, IMAGE_SIZE)
set -euo pipefail

REPO_URL="https://github.com/gabrielWYP/Edge_Mango.git"
BRANCH="main"
DATA=""
WORKDIR=""
DATASET_NAME="v4_captura_v3_poly"
WEIGHTS=""
MACHINE=""
PYTHON_BIN=""
TORCH_INDEX="${TORCH_INDEX:-}"
SKIP_CACHE=0
SKIP_TESTS=0
SMOKE=0
SMOKE_ARM="master_rgb_seg"
IMAGE_SIZE=640

usage() {
    sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}
step() { printf '\n==> %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO_URL="${2:?}"; shift 2 ;;
        --branch) BRANCH="${2:?}"; shift 2 ;;
        --data) DATA="${2:?}"; shift 2 ;;
        --workdir) WORKDIR="${2:?}"; shift 2 ;;
        --dataset-name) DATASET_NAME="${2:?}"; shift 2 ;;
        --weights) WEIGHTS="${2:?}"; shift 2 ;;
        --machine) MACHINE="${2:?}"; shift 2 ;;
        --python) PYTHON_BIN="${2:?}"; shift 2 ;;
        --torch-index) TORCH_INDEX="${2:?}"; shift 2 ;;
        --skip-cache) SKIP_CACHE=1; shift ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        --smoke-arm) SMOKE_ARM="${2:?}"; shift 2 ;;
        --image-size) IMAGE_SIZE="${2:?}"; shift 2 ;;
        -h | --help) usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 2 ;;
    esac
done
[ -n "$DATA" ] && [ -n "$WORKDIR" ] || usage 2
[ -f "$DATA" ] || die "data archive not found: $DATA"
mkdir -p "$WORKDIR"
WORKDIR="$(cd "$WORKDIR" && pwd)"
DATA="$(cd "$(dirname "$DATA")" && pwd)/$(basename "$DATA")"
REPO="$WORKDIR/edge_mango"
VENV="$WORKDIR/.venv"
DS="$WORKDIR/data/cache/$DATASET_NAME"
PIXEL="$WORKDIR/data/cache/pixels"
RUNS="$WORKDIR/runs/thesis"
NIR_AS_RGB="$WORKDIR/data/cache/v4_nir_as_rgb_clahe"  # run_small_arms.py's default: <dataset parent>/v4_nir_as_rgb_clahe

# 1. Hardware and interpreter ---------------------------------------------------------------------------------------
step "GPU and Python"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    die "nvidia-smi not found: this machine has no NVIDIA driver, or the image is CPU-only"
fi
if [ -z "$PYTHON_BIN" ]; then
    if command -v python3.12 >/dev/null 2>&1; then PYTHON_BIN=python3.12; else PYTHON_BIN=python3; fi
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "$PYTHON_BIN not found (Ubuntu: sudo apt-get install -y python3.12 python3.12-venv)"
"$PYTHON_BIN" --version
case "$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" in
    3.12) ;;
    *) echo "WARNING: the thesis was developed on Python 3.12; continuing with $PYTHON_BIN" >&2 ;;
esac
echo "CPU threads: $(nproc); RAM: $(awk '/MemTotal/ {printf "%.0f GB", $2/1048576}' /proc/meminfo); disk free in $WORKDIR: $(df -h --output=avail "$WORKDIR" | tail -1 | tr -d ' ')"

# 2. GitHub access and the repository -------------------------------------------------------------------------------
step "GitHub access"
if command -v gh >/dev/null 2>&1; then
    if gh auth status >/dev/null 2>&1; then
        echo "gh is logged in."
        gh auth setup-git >/dev/null 2>&1 || true
    else
        echo "gh is NOT logged in. For a private repository and for pushing progress, run once, by hand:"
        echo "    gh auth login -h github.com -p https -w     # device flow: open the printed URL, type the code"
    fi
else
    echo "gh is not installed. To install it on Ubuntu: https://github.com/cli/cli/blob/trunk/docs/install_linux.md"
    echo "Then: gh auth login -h github.com -p https -w"
fi

step "Repository ($REPO_URL, $BRANCH)"
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" fetch origin "$BRANCH"
    git -C "$REPO" checkout "$BRANCH"
    git -C "$REPO" merge --ff-only "origin/$BRANCH" || echo "WARNING: $BRANCH did not fast-forward; the checkout is left as it is" >&2
else
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO" || die "clone failed; if the repository is private run: gh auth login -h github.com -p https -w"
fi
git -C "$REPO" log -1 --format='HEAD %h %s'

# 3. Environment ----------------------------------------------------------------------------------------------------
step "Python environment ($VENV)"
[ -x "$VENV/bin/python" ] || "$PYTHON_BIN" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q -U pip wheel

if [ -z "$TORCH_INDEX" ]; then
    cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | head -1)"
    [ -n "$cuda" ] || die "cannot read the CUDA version from nvidia-smi; pass --torch-index cu126 (or cu128, cu124, cpu)"
    code=$(($(echo "$cuda" | cut -d' ' -f1) * 100 + $(echo "$cuda" | cut -d' ' -f2)))
    # The driver's CUDA version is the highest runtime it supports. cu128 is the newest index used here; a newer driver
    # runs it too. For another one pass --torch-index.
    if [ "$code" -ge 1208 ]; then TORCH_INDEX=cu128
    elif [ "$code" -ge 1206 ]; then TORCH_INDEX=cu126
    elif [ "$code" -ge 1204 ]; then TORCH_INDEX=cu124
    elif [ "$code" -ge 1201 ]; then TORCH_INDEX=cu121
    elif [ "$code" -ge 1108 ]; then TORCH_INDEX=cu118
    else die "driver CUDA $cuda is too old for the supported wheels; update the driver or pass --torch-index"; fi
    echo "driver CUDA $cuda -> torch wheels $TORCH_INDEX (override: --torch-index or TORCH_INDEX)"
fi
case "$TORCH_INDEX" in http*) INDEX_URL="$TORCH_INDEX" ;; *) INDEX_URL="https://download.pytorch.org/whl/$TORCH_INDEX" ;; esac

if "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() or "'"$TORCH_INDEX"'" == "cpu" else 1)' 2>/dev/null; then
    echo "torch already installed and working."
else
    "$PY" -m pip install -q torch torchvision --index-url "$INDEX_URL"
fi
"$PY" -m pip install -q -r "$REPO/requirements-remote.txt"
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "| memory %.0f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
else:
    raise SystemExit("torch does not see the GPU: check the driver, or pass --torch-index for the right CUDA wheel")
EOF

if [ -z "$MACHINE" ]; then
    mem_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
    if [ "${mem_mb:-0}" -ge 20000 ]; then MACHINE=configs/machines/rtx_a5000.yaml; else MACHINE=configs/machines/rtx3080.yaml; fi
fi
echo "machine profile: $MACHINE"

# 4. Data -----------------------------------------------------------------------------------------------------------
step "Data ($DATA -> $DS)"
mkdir -p "$DS"
if [ -f "$DS/MANIFEST.sha256" ] && (cd "$DS" && sha256sum -c --quiet MANIFEST.sha256 2>/dev/null); then
    echo "already unpacked and verified."
else
    case "$DATA" in
        *.zst) command -v zstd >/dev/null 2>&1 || { sudo apt-get install -y zstd >/dev/null 2>&1 || die "zstd is needed to read $DATA: sudo apt-get install -y zstd"; } ;;
    esac
    [ ! -f "$DATA.sha256" ] || (cd "$(dirname "$DATA")" && sha256sum -c --quiet "$(basename "$DATA").sha256") || die "archive checksum mismatch: upload it again"
    tar -xf "$DATA" --strip-components=1 -C "$DS"
    (cd "$DS" && sha256sum -c --quiet MANIFEST.sha256) || die "manifest verification failed: some files are missing or damaged"
    echo "unpacked and verified."
fi
for need in rgb nir_clahe labels homographies index.csv; do [ -e "$DS/$need" ] || die "$DS/$need missing"; done

# 5. Folds, NIR links and pixel cache (the commands of docs/tesis/runbook-v4.md) -------------------------------------
cd "$REPO"
step "Folds"
if [ -f "$DS/folds_lot/folds_meta.json" ]; then
    echo "folds_lot/ present."
else
    [ -d "$DS/nir" ] || die "no folds_lot/ and no raw nir/ in $DS: build the folds where the dataset was built and package again"
    "$PY" scripts/make_v4_folds.py --dataset "$DS" --fold-spec configs/folds/v4_lot_5fold_poly.json
fi
if [ ! -f configs/folds/v4_lot_5fold_poly.json ]; then
    # aggregate_kfold.py --fold-spec reads this; it is derived from the folds, so it is rebuilt from folds_meta.json.
    "$PY" - "$DS/folds_lot/folds_meta.json" <<'EOF' || echo "WARNING: could not write configs/folds/v4_lot_5fold_poly.json; copy it from where the folds were built" >&2
import json, sys
from pathlib import Path
from scripts.make_v4_folds import write_fold_spec
write_fold_spec(json.loads(Path(sys.argv[1]).read_text()), Path("configs/folds/v4_lot_5fold_poly.json"))
print("wrote configs/folds/v4_lot_5fold_poly.json (untracked: commit it if it is the final spec)")
EOF
fi

step "NIR-as-RGB links"
"$PY" - "$DS" "$NIR_AS_RGB" <<'EOF'
import sys
from pathlib import Path
from scripts.run_small_arms import make_nir_as_rgb
print(make_nir_as_rgb(Path(sys.argv[1]), Path(sys.argv[2])), "links in", sys.argv[2])
EOF

if [ "$SKIP_CACHE" -eq 0 ]; then
    step "Pixel cache ($PIXEL)"
    workers="$(nproc)"
    "$PY" scripts/build_pixel_cache.py --dataset "$DS" --rgb-dir "$DS/rgb" --output "$PIXEL/rgb" --workers "$workers"
    "$PY" scripts/build_pixel_cache.py --dataset "$DS" --rgb-dir "$NIR_AS_RGB" --output "$PIXEL/nir_as_rgb" --workers "$workers"
fi

# 6. Tests ----------------------------------------------------------------------------------------------------------
if [ "$SKIP_TESTS" -eq 0 ]; then
    step "Quick tests"
    "$PY" -m pytest -q -p no:cacheprovider tests/test_run_small_arms.py tests/test_seg_launchers.py tests/test_thesis_arms.py \
        tests/test_export_run_results.py tests/evaluation/test_arms.py tests/test_runbook_v4.py
fi

# 7. env.sh ---------------------------------------------------------------------------------------------------------
step "Environment file"
mkdir -p "$RUNS"
{
    echo "# generated by setup_remote.sh; source it before every command of docs/tesis/runbook-rental.md"
    printf 'export WORK=%q\nexport REPO=%q\nexport PY=%q\nexport DATASET=%q\nexport PIXEL=%q\nexport RUNS=%q\n' \
        "$WORKDIR" "$REPO" "$PY" "$DS" "$PIXEL" "$RUNS"
    printf 'export WEIGHTS=%q\nexport MACHINE=%q\nexport IMAGE_SIZE=%q\n' "$WEIGHTS" "$MACHINE" "$IMAGE_SIZE"
    # shellcheck disable=SC2016  # written literally: expanded when env.sh is sourced
    echo 'export PATH="$(dirname "$PY"):$PATH"'
} > "$WORKDIR/env.sh"
echo "wrote $WORKDIR/env.sh"

# 8. Smoke test -----------------------------------------------------------------------------------------------------
if [ "$SMOKE" -eq 1 ]; then
    step "Smoke test: 2 epochs of $SMOKE_ARM, fold 0, seed 42 (scratch runs root, not a result)"
    SMOKE_ROOT="$WORKDIR/runs/smoke"
    rm -rf "$SMOKE_ROOT"
    LOG="$WORKDIR/smoke.log"
    smoke_args=(--dataset "$DS" --variant nano --epochs 2 --folds 0 --seeds 42 --val-only --arms "$SMOKE_ARM"
        --checkpoints-root "$SMOKE_ROOT" --reports-root "$SMOKE_ROOT/reports" --machine "$MACHINE"
        --image-size "$IMAGE_SIZE")
    [ -n "$WEIGHTS" ] && smoke_args+=(--weights "$WEIGHTS")
    [ "$SKIP_CACHE" -eq 0 ] && smoke_args+=(--pixel-cache-root "$PIXEL")
    "$PY" scripts/run_small_arms.py "${smoke_args[@]}" 2>&1 | tee "$LOG"
    # The trainer prints "... | <seconds>s" at the end of each epoch line (training plus validation).
    awk -v logfile="$LOG" '/\| [0-9]+(\.[0-9]+)?s$/ {t = $NF; sub(/s$/, "", t); s += t; n++}
         END {if (n) printf "\nSeconds per epoch (mean of %d, training + validation): %.1f\n", n, s / n;
              else print "\nNo epoch time found in the log; read it from " logfile}' "$LOG"
    echo "(The first epoch includes warm-up; with --jobs N the epochs of N concurrent runs are slower: re-measure.)"
fi

step "Ready. Next: source $WORKDIR/env.sh   (then see docs/tesis/runbook-rental.md)"
