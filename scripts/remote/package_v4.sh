#!/usr/bin/env bash
# Pack a built v4 dataset into ONE archive for another machine.
#
#   scripts/remote/package_v4.sh <dataset-dir> [--out FILE] [--with-raw-nir] [--dry-run]
#
# The archive has a single top folder named like the dataset and holds what training reads:
#   rgb/ (symlinks DEREFERENCED, so it works anywhere), nir_clahe/ (every arm reads this variant), labels/,
#   homographies/, index.csv, report.json, exclusions.csv, nir_lines.csv, folds_lot/ (if built) and MANIFEST.sha256.
#
# Left out: nir_clahe_view/ (display images), and the raw nir/ unless --with-raw-nir. Raw NIR is read ONLY by
# scripts/make_v4_folds.py (it computes the mean/std of both NIR variants); no arm trains on it. So when folds_lot/ is
# in the archive the raw NIR is dead weight (about 7 GB), and when folds_lot/ is missing the archive needs --with-raw-nir
# so that the remote machine can build the folds.
#
# Compression: zstd (.tar.zst) when available, else pigz/gzip (.tar.gz). PNG and JPEG do not compress, so the archive is
# about as large as the sum printed by --dry-run. PACKAGE_V4_COMPRESSOR=zstd|pigz|gzip forces one.
set -euo pipefail

usage() {
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

DATASET=""
OUT=""
WITH_RAW_NIR=0
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="${2:?--out needs a file}"; shift 2 ;;
        --with-raw-nir) WITH_RAW_NIR=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h | --help) usage 0 ;;
        -*) echo "unknown option: $1" >&2; usage 2 ;;
        *) [ -z "$DATASET" ] || { echo "only one dataset directory" >&2; exit 2; }; DATASET="$1"; shift ;;
    esac
done
[ -n "$DATASET" ] || usage 2
[ -d "$DATASET" ] || { echo "not a directory: $DATASET" >&2; exit 2; }
DATASET="$(cd "$DATASET" && pwd)"
NAME="$(basename "$DATASET")"
PARENT="$(dirname "$DATASET")"

# Compressor
COMPRESSOR="${PACKAGE_V4_COMPRESSOR:-}"
if [ -z "$COMPRESSOR" ]; then
    if command -v zstd >/dev/null 2>&1; then COMPRESSOR=zstd
    elif command -v pigz >/dev/null 2>&1; then COMPRESSOR=pigz
    else COMPRESSOR=gzip; fi
fi
case "$COMPRESSOR" in
    zstd) EXT="tar.zst"; COMPRESS=(zstd -T0 -3 -q) ;;
    pigz) EXT="tar.gz"; COMPRESS=(pigz -1) ;;
    gzip) EXT="tar.gz"; COMPRESS=(gzip -1) ;;
    *) echo "unknown PACKAGE_V4_COMPRESSOR: $COMPRESSOR" >&2; exit 2 ;;
esac
[ -n "$OUT" ] || OUT="$PWD/$NAME.$EXT"

# What goes in. Directories are walked with symlinks followed.
DIRS=(rgb nir_clahe labels homographies)
FILES=(index.csv report.json)
OPTIONAL_FILES=(exclusions.csv nir_lines.csv)
[ "$WITH_RAW_NIR" -eq 1 ] && DIRS+=(nir)
[ -d "$DATASET/folds_lot" ] && DIRS+=(folds_lot)

for item in "${DIRS[@]}"; do
    [ -d "$DATASET/$item" ] || { [ "$item" = folds_lot ] || { echo "missing $DATASET/$item" >&2; exit 1; }; }
done
for item in "${FILES[@]}"; do
    [ -f "$DATASET/$item" ] || { echo "missing $DATASET/$item" >&2; exit 1; }
done
for item in "${OPTIONAL_FILES[@]}"; do
    [ -f "$DATASET/$item" ] && FILES+=("$item")
done
if [ ! -d "$DATASET/folds_lot" ] && [ "$WITH_RAW_NIR" -eq 0 ]; then
    cat >&2 <<EOF
$DATASET has no folds_lot/, and the remote machine can only build it from the raw nir/ (make_v4_folds.py reads both NIR
variants). Either build the folds here first (scripts/make_v4_folds.py --dataset $DATASET ...) or pass --with-raw-nir.
EOF
    exit 2
fi

echo "Dataset: $DATASET"
echo "Archive: $OUT ($COMPRESSOR)"
echo "Contents (top folder $NAME/):"
TOTAL=0
for item in "${DIRS[@]}"; do
    count=$(find -L "$DATASET/$item" -type f | wc -l)
    bytes=$(find -L "$DATASET/$item" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
    TOTAL=$((TOTAL + bytes))
    printf '  %-16s %8d files %10.2f GB\n' "$item/" "$count" "$(awk -v b="$bytes" 'BEGIN {print b/1e9}')"
done
for item in "${FILES[@]}"; do
    bytes=$(stat -L -c %s "$DATASET/$item")
    TOTAL=$((TOTAL + bytes))
    printf '  %-16s %8s       %10.2f GB\n' "$item" "1 file" "$(awk -v b="$bytes" 'BEGIN {print b/1e9}')"
done
echo "  MANIFEST.sha256  (sha256 of every file above, relative to $NAME/)"
echo "Excluded:"
[ "$WITH_RAW_NIR" -eq 0 ] && [ -d "$DATASET/nir" ] && echo "  nir/             raw NIR, read only by make_v4_folds.py (use --with-raw-nir)"
[ -d "$DATASET/nir_clahe_view" ] && echo "  nir_clahe_view/  display images, not read by training"
for other in "$DATASET"/*; do
    base="$(basename "$other")"
    case " ${DIRS[*]} ${FILES[*]} nir nir_clahe_view " in *" $base "*) ;; *) echo "  $base" ;; esac
done
printf 'Estimated size: %.2f GB (uncompressed %d bytes; images do not compress, expect about the same)\n' \
    "$(awk -v b="$TOTAL" 'BEGIN {print b/1e9}')" "$TOTAL"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run: nothing written."
    exit 0
fi

command -v "${COMPRESS[0]}" >/dev/null 2>&1 || { echo "${COMPRESS[0]} not found" >&2; exit 1; }
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$NAME"

echo "Hashing files (sha256, 8 at a time) ..."
(
    cd "$DATASET"
    { for item in "${DIRS[@]}"; do find -L "$item" -type f; done; printf '%s\n' "${FILES[@]}"; } \
        | LC_ALL=C sort | tr '\n' '\0' | xargs -0 -P 8 -n 64 sha256sum
) | LC_ALL=C sort -k2 > "$STAGE/$NAME/MANIFEST.sha256"
echo "  $(wc -l < "$STAGE/$NAME/MANIFEST.sha256") files in the manifest"

MEMBERS=()
for item in "${DIRS[@]}" "${FILES[@]}"; do MEMBERS+=("$NAME/$item"); done
mkdir -p "$(dirname "$OUT")"
PARTIAL="$OUT.partial"
echo "Writing $OUT ..."
tar --dereference -cf - -C "$PARENT" "${MEMBERS[@]}" -C "$STAGE" "$NAME/MANIFEST.sha256" | "${COMPRESS[@]}" > "$PARTIAL"
mv "$PARTIAL" "$OUT"
(cd "$(dirname "$OUT")" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256")
echo "Done: $OUT ($(du -h "$OUT" | cut -f1)); archive checksum in $OUT.sha256"
