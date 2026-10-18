"""Write configs/folds/duplicate_groups.json from the visually verified duplicate groups.

Analysis tool used for the data leakage audit (see docs/data/leakage-audit.md).
It is NOT part of the training pipeline: nothing in `src/` or `scripts/` imports it.

The groups below were found with `dup_detect.py` (SIFT, MIN_N 12 and 8) and then
verified by eye on contact sheets (`mango_sheets.py`). Members are listed as 1-based
capture-order indices inside each lot (the same numbering as `index_<lot>.json`); the
stems, timestamps, original splits and damage-box counts are read from the data, so
no stem is typed by hand.

    CUDA_VISIBLE_DEVICES="" python build_duplicate_groups.py --repo-root <checkout-with-data>
"""
import argparse
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

# docs/data/leakage-audit/tools/build_duplicate_groups.py -> repository root is four levels up.
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[4]
DAMAGE_CLASS = "1"
TZ = ZoneInfo("America/Lima")

# lot -> list of (members as 1-based capture-order indices, status)
GROUPS = {
    "lot1": [([7, 12], "certain"), ([6, 15], "certain"), ([14, 21], "certain")],
    "lot2": [([9, 25], "certain"), ([20, 49], "certain"), ([4, 47], "certain"),
             ([19, 36], "certain"), ([22, 41], "certain")],
    "lot3": [([18, 41], "certain"), ([3, 14, 38], "certain"), ([9, 21, 24], "certain"), ([13, 30], "certain"),
             ([33, 36], "certain")],
    "lot4": [([27, 43], "certain"), ([11, 45], "probable")],
    "lot5": [([27, 30], "certain")],
}

METHOD = (
    "SIFT keypoints (nfeatures=2000, CLAHE clip 2.0 tile 8x8) on the mango box crop (long side 400 px) "
    "inside an ellipse mask (0.42 of the box); Lowe ratio test 0.75; RANSAC homography with a 5.0 px "
    "reprojection threshold; candidates at >=12 inliers (docs/data/leakage-audit/tools/dup_detect.py, "
    "MIN_N 12, repeated at MIN_N 8) were confirmed by eye on contact sheets as the same face of the "
    "same mango. Exact byte duplicates: none."
)
KEEP_RULE = "Keep the member with the most damage boxes (class 1); on a tie, keep the earliest timestamp."


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_REPO_ROOT,
                        help="Checkout holding configs/folds/lot_5fold.json and data/ "
                             "(default: the repository this script lives in).")
    parser.add_argument("--out", type=Path, default=SCRIPT_REPO_ROOT / "configs/folds/duplicate_groups.json",
                        help="Output file (default: <this repository>/configs/folds/duplicate_groups.json).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.repo_root.resolve()
    lots = json.loads((root / "configs/folds/lot_5fold.json").read_text())["lots"]
    manifest = json.loads((root / "data/annotations/yolo/splits.json").read_text())
    split_of = {stem: split for split, stems in manifest.items() for stem in stems}
    labels = root / "data/annotations/yolo/labels"

    def damage_boxes(stem):
        rows = (labels / split_of[stem] / f"{stem}.txt").read_text().splitlines()
        return sum(1 for r in rows if r.split()[:1] == [DAMAGE_CLASS])

    groups = []
    for lot, lot_groups in GROUPS.items():
        stems = sorted(lots[lot]["stems"], key=lambda s: int(s.rsplit("_", 1)[1]))
        for n, (indices, status) in enumerate(lot_groups, 1):
            members = []
            for index in indices:
                stem = stems[index - 1]
                ts = int(stem.rsplit("_", 1)[1])
                members.append({
                    "stem": stem,
                    "lot_index": index,
                    "timestamp_local": dt.datetime.fromtimestamp(ts, TZ).isoformat(),
                    "original_split": split_of[stem],
                    "damage_boxes": damage_boxes(stem),
                })
            keep = min(members, key=lambda m: (-m["damage_boxes"], int(m["stem"].rsplit("_", 1)[1])))
            groups.append({"group_id": f"{lot}_g{n}", "lot": lot, "status": status, "members": members,
                           "keep": keep["stem"]})

    out = {"schema_version": 1, "method": METHOD, "keep_rule": KEEP_RULE, "groups": groups}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}: {len(groups)} groups, {sum(len(g['members']) for g in groups)} member images")


if __name__ == "__main__":
    main()
