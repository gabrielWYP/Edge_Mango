#!/usr/bin/env bash
# Export the results of the runs so far and push ONLY the small report files to a branch, from the training machine.
#
#   scripts/remote/push_progress.sh [options]
#     --repo DIR          the repository checkout (default: the one this script is in)
#     --runs-root DIR     <arm>/fold<k>/seed<s>/ and reports/ (default: $RUNS, else <work>/runs/thesis)
#     --dataset DIR       the v4 dataset of the runs (default: $DATASET)
#     --out DIR           export folder, inside reports/ of the repo (default: reports/thesis-export)
#     --branch NAME       default runs/thesis-<YYYYMMDD>
#     --remote NAME       default origin
#     --test-folds N..    forwarded to export_run_results.py; test results are exported ONLY when you pass these,
#     --test-seeds N..    --test-seeds and (optionally) --test-arms, with the semantics of that script (a test
#     --test-arms A..     look the plan registers for some folds/seeds/arms only). Folds and seeds come as a pair.
#     --every-min N       loop: export and push every N minutes until interrupted (a cron line works too)
#     --no-export         skip the export; push what is already in --out
#     --worktree-dir DIR  where the progress branch is checked out (default <repo>/../.progress_wt)
#     --dry-run           print the export command and what would be pushed
#
# The progress branch lives in its OWN git worktree: the working tree under the running jobs is never switched or
# modified. Only files under reports/ that are .csv/.json/.md are copied there, so checkpoints, logs, data and the
# per-image predictions.jsonl (ignored by .gitignore) cannot reach the branch. Plots (*.png) are ignored by the repo
# too, so they stay on the machine. With nothing new, the script prints "nothing new" and exits 0.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SELF_DIR/../.." && pwd)"
RUNS_ROOT="${RUNS:-}"
DATASET="${DATASET:-}"
OUT="reports/thesis-export"
BRANCH="runs/thesis-$(date +%Y%m%d)"
REMOTE="origin"
TEST_FOLDS=()
TEST_SEEDS=()
TEST_ARMS=()
EVERY_MIN=0
NO_EXPORT=0
DRY_RUN=0
WT_DIR=""
PYTHON="${PYTHON:-}"

usage() {
    sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}
while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO="$(cd "${2:?}" && pwd)"; shift 2 ;;
        --runs-root) RUNS_ROOT="${2:?}"; shift 2 ;;
        --dataset) DATASET="${2:?}"; shift 2 ;;
        --out) OUT="${2:?}"; shift 2 ;;
        --branch) BRANCH="${2:?}"; shift 2 ;;
        --remote) REMOTE="${2:?}"; shift 2 ;;
        --every-min) EVERY_MIN="${2:?}"; shift 2 ;;
        --no-export) NO_EXPORT=1; shift ;;
        --worktree-dir) WT_DIR="${2:?}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --test-folds) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do TEST_FOLDS+=("$1"); shift; done ;;
        --test-seeds) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do TEST_SEEDS+=("$1"); shift; done ;;
        --test-arms) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do TEST_ARMS+=("$1"); shift; done ;;
        -h | --help) usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 2 ;;
    esac
done

case "$OUT" in
    reports/* | reports) ;;
    *) echo "--out must be inside reports/ of the repository (got: $OUT)" >&2; exit 2 ;;
esac
case "$OUT" in *..*) echo "--out must not contain '..'" >&2; exit 2 ;; esac
if [ "${#TEST_FOLDS[@]}" -gt 0 ] && [ "${#TEST_SEEDS[@]}" -eq 0 ]; then
    echo "--test-folds needs --test-seeds: test results are read only for explicit folds AND seeds" >&2; exit 2
fi
if [ "${#TEST_SEEDS[@]}" -gt 0 ] && [ "${#TEST_FOLDS[@]}" -eq 0 ]; then
    echo "--test-seeds needs --test-folds: test results are read only for explicit folds AND seeds" >&2; exit 2
fi
if [ "${#TEST_ARMS[@]}" -gt 0 ] && [ "${#TEST_FOLDS[@]}" -eq 0 ]; then
    echo "--test-arms only narrows a test look: it needs --test-folds and --test-seeds" >&2; exit 2
fi
[ -n "$WT_DIR" ] || WT_DIR="$(dirname "$REPO")/.progress_wt"
if [ -z "$PYTHON" ]; then
    if [ -x "$REPO/../.venv/bin/python" ]; then PYTHON="$REPO/../.venv/bin/python"
    elif [ -x "$REPO/.venv/bin/python" ]; then PYTHON="$REPO/.venv/bin/python"
    else PYTHON="python3"; fi
fi

export_command() {
    local cmd=("$PYTHON" scripts/export_run_results.py --runs-root "$RUNS_ROOT" --dataset "$DATASET" --output "$OUT")
    [ "${#TEST_FOLDS[@]}" -gt 0 ] && cmd+=(--test-folds "${TEST_FOLDS[@]}" --test-seeds "${TEST_SEEDS[@]}")
    [ "${#TEST_ARMS[@]}" -gt 0 ] && cmd+=(--test-arms "${TEST_ARMS[@]}")
    printf '%q ' "${cmd[@]}"
    echo
}

once() {
    cd "$REPO"
    if [ "$NO_EXPORT" -eq 0 ]; then
        if [ "$DRY_RUN" -eq 0 ] && { [ -z "$RUNS_ROOT" ] || [ -z "$DATASET" ]; }; then
            echo "--runs-root and --dataset are needed (or set RUNS and DATASET)" >&2
            return 2
        fi
        echo "Export: $(export_command)"
        if [ "$DRY_RUN" -eq 0 ]; then
            local cmd=("$PYTHON" scripts/export_run_results.py --runs-root "$RUNS_ROOT" --dataset "$DATASET" --output "$OUT")
            [ "${#TEST_FOLDS[@]}" -gt 0 ] && cmd+=(--test-folds "${TEST_FOLDS[@]}" --test-seeds "${TEST_SEEDS[@]}")
            [ "${#TEST_ARMS[@]}" -gt 0 ] && cmd+=(--test-arms "${TEST_ARMS[@]}")
            "${cmd[@]}" || { echo "export failed; nothing pushed" >&2; return 1; }
        fi
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "Would copy *.csv, *.json and *.md of $OUT into branch $BRANCH (worktree $WT_DIR) and push to $REMOTE."
        return 0
    fi
    [ -d "$REPO/$OUT" ] || { echo "nothing new: $OUT does not exist yet"; return 0; }

    # The progress branch in its own worktree: continue the remote one, else start from the current HEAD.
    git fetch -q "$REMOTE" "$BRANCH" 2>/dev/null || true
    if [ ! -d "$WT_DIR/.git" ] && [ ! -f "$WT_DIR/.git" ]; then
        if git show-ref -q --verify "refs/remotes/$REMOTE/$BRANCH"; then
            git worktree add -q -B "$BRANCH" "$WT_DIR" "$REMOTE/$BRANCH"
        elif git show-ref -q --verify "refs/heads/$BRANCH"; then
            git worktree add -q "$WT_DIR" "$BRANCH"
        else
            git worktree add -q -b "$BRANCH" "$WT_DIR" HEAD
        fi
    fi
    (
        cd "$WT_DIR"
        [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || { echo "$WT_DIR is not on $BRANCH" >&2; exit 1; }
        mkdir -p "$OUT"
        # Mirror the allowed files, so a file the export no longer writes disappears from the branch too.
        find "$OUT" -type f \( -name '*.csv' -o -name '*.json' -o -name '*.md' \) -delete
        (cd "$REPO" && find "$OUT" -type f \( -name '*.csv' -o -name '*.json' -o -name '*.md' \) -print0) \
            | while IFS= read -r -d '' file; do
                mkdir -p "$(dirname "$file")"
                cp "$REPO/$file" "$file"
            done
        git add -A -- "$OUT"
        if git diff --cached --quiet; then
            echo "nothing new in $OUT; $BRANCH is up to date"
            exit 0
        fi
        git commit -q -m "chore(reports): progress export $(date -u +%Y-%m-%dT%H:%MZ)"
        git push -q -u "$REMOTE" "$BRANCH"
        echo "pushed $(git rev-parse --short HEAD) to $REMOTE/$BRANCH"
    )
}

if [ "$EVERY_MIN" -gt 0 ]; then
    trap 'echo "stopped"; exit 0' INT TERM
    while true; do
        echo "[$(date -u +%H:%M:%S)] progress push"
        once || echo "this round failed; trying again in $EVERY_MIN min" >&2
        sleep "$((EVERY_MIN * 60))" &
        wait $!
    done
else
    once
fi
