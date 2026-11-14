#!/usr/bin/env bash
# Publish each finished fold of the thesis runs as one GitHub release asset, from the training machine.
#
#   scripts/remote/publish_folds.sh --runs-root DIR --tag TAG [options]
#     --runs-root DIR     <arm>/fold<k>/seed<s>/ and reports/ of run_small_arms.py (default: $RUNS)
#     --tag TAG           release tag, created as a prerelease if missing (e.g. thesis-runs-20261114)
#     --seed N            default 42
#     --arms "A B .."     default: the `thesis` preset of src/evaluation/arms.py
#     --folds "0 1 .."    default "0 1 2 3 4"
#     --val-only          a fold is finished once every arm has its val metrics (default: val and test)
#     --dataset DIR       forwarded to push_progress.sh (default: $DATASET)
#     --no-push-progress  do not run push_progress.sh after a fold is published
#     --every-min N       loop every N minutes until every fold is published (default: one pass)
#
# A fold is finished when every arm has reports/<arm>/fold<k>_seed<s>/<split>/metrics_v4.json. Its asset
# fold<k>_seed<s>.tar.gz holds, per arm: the run folder without eval_scratch/ and without any .pt but
# best_model.pt (history, config, completion marker), the val/test reports with predictions.jsonl and the
# training log. Parts of 1900 MB are uploaded if the archive is larger (GitHub's limit is 2 GB per asset).
# A marker in <runs-root>/.published/ keeps each fold from being uploaded twice. Then push_progress.sh pushes
# the small report files. Nothing is deleted from the machine.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SELF_DIR/../.." && pwd)"
RUNS_ROOT="${RUNS:-}"
DATASET="${DATASET:-}"
TAG=""
SEED=42
ARMS=""
FOLDS="0 1 2 3 4"
SPLITS="val test"
PUSH_PROGRESS=1
EVERY_MIN=0
PART_BYTES=$((1900 * 1024 * 1024))

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --runs-root) RUNS_ROOT="${2:?}"; shift 2 ;;
        --tag) TAG="${2:?}"; shift 2 ;;
        --seed) SEED="${2:?}"; shift 2 ;;
        --arms) ARMS="${2:?}"; shift 2 ;;
        --folds) FOLDS="${2:?}"; shift 2 ;;
        --val-only) SPLITS="val"; shift ;;
        --dataset) DATASET="${2:?}"; shift 2 ;;
        --no-push-progress) PUSH_PROGRESS=0; shift ;;
        --every-min) EVERY_MIN="${2:?}"; shift 2 ;;
        -h | --help) usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 2 ;;
    esac
done

[ -n "$RUNS_ROOT" ] || { echo "--runs-root (or \$RUNS) is required" >&2; exit 2; }
[ -n "$TAG" ] || { echo "--tag is required" >&2; exit 2; }
RUNS_ROOT="$(cd "$RUNS_ROOT" && pwd)"
if [ -z "$ARMS" ]; then
    ARMS="$(cd "$REPO" && "${PY:-python}" -c \
        'from src.evaluation.arms import SEGMENT_PRESETS; print(*SEGMENT_PRESETS["thesis"])')"
fi
MARKERS="$RUNS_ROOT/.published"
mkdir -p "$MARKERS"

fold_finished() {
    local fold="$1" arm split
    for arm in $ARMS; do
        for split in $SPLITS; do
            [ -e "$RUNS_ROOT/reports/$arm/fold${fold}_seed${SEED}/$split/metrics_v4.json" ] || return 1
        done
    done
}

ensure_release() {
    gh release view "$TAG" > /dev/null 2>&1 && return 0
    gh release create "$TAG" --prerelease --title "$TAG" \
        --notes "Thesis runs: one asset per finished fold (checkpoints, predictions, metrics, logs)."
}

publish_fold() {
    local fold="$1" name="fold${1}_seed${SEED}" arm list tmp archive log
    list="$(mktemp)"
    tmp="$(mktemp -d)"
    archive="$tmp/$name.tar.gz"
    (
        cd "$RUNS_ROOT"
        for arm in $ARMS; do
            find "$arm/fold$fold/seed$SEED" -path '*/eval_scratch' -prune -o -type f \
                \( ! -name '*.pt' -o -name 'best_model.pt' \) -print
            find "reports/$arm/fold${fold}_seed${SEED}" -type f -print
            log="reports/logs/${arm}_fold${fold}_seed${SEED}.log"
            if [ -e "$log" ]; then echo "$log"; fi
        done
    ) | sort > "$list"
    tar -czf "$archive" -C "$RUNS_ROOT" -T "$list"
    rm -f "$list"
    ensure_release
    if [ "$(stat -c %s "$archive")" -gt "$PART_BYTES" ]; then
        split -b "$PART_BYTES" -d "$archive" "$archive.part"
        gh release upload "$TAG" "$archive".part* --clobber
    else
        gh release upload "$TAG" "$archive" --clobber
    fi
    rm -rf "$tmp"
    touch "$MARKERS/$name"
    echo "published $name -> release $TAG"
}

one_pass() {
    local fold published=0 pending=0
    for fold in $FOLDS; do
        if [ -e "$MARKERS/fold${fold}_seed${SEED}" ]; then continue; fi
        if fold_finished "$fold"; then
            publish_fold "$fold"
            published=1
        else
            pending=1
        fi
    done
    if [ "$published" -eq 1 ] && [ "$PUSH_PROGRESS" -eq 1 ]; then
        "$SELF_DIR/push_progress.sh" --runs-root "$RUNS_ROOT" ${DATASET:+--dataset "$DATASET"}
    fi
    return "$pending"
}

while true; do
    if one_pass; then
        echo "every fold is published"
        exit 0
    fi
    [ "$EVERY_MIN" -gt 0 ] || exit 0
    sleep "$((EVERY_MIN * 60))"
done
