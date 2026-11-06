#!/usr/bin/env python3
"""Damage AP per impact zone next to the light, contrast and geometry of the damage in each zone.

Question (exploratory, raised after the zone strata were seen): the damage AP is low in `derecha`, `superior` and
`inferior` and high in `centro` and `izquierda`. Is that illumination, or something else such as where the damage
falls on the fruit as the camera sees it?

Part 1, AP per zone, from the saved evaluations: the `ap50_by_zone` of `metrics_v4.json` (all post-impact hours) and
the same AP restricted to the latent window (ronda_h 0-6), computed here from `predictions.jsonl` with
`src.evaluation.v4_metrics.damage_ap`. `sano` photos have no zone and enter neither.

Part 2, conditions per zone, from the images and labels only (no model output), for every photo with a damage box:

- position: centre of the damage box relative to the mango box, in semi-axes (x: -1 image left, +1 image right;
  y: -1 image top, +1 image bottom), and its distance to the centre of the fruit;
- size: damage box area over mango box area;
- light: mean RGB luma and mean NIR inside the damage box, and each over the mean of the whole fruit (1 = as lit as
  the fruit on average);
- contrast: |mean in the box - mean of the rest of the fruit| / std of the rest, for RGB luma and NIR, as in
  `scripts/lot_descriptors.py`.

Test results are read only for the folds and seeds given (`--test-folds`, `--test-seeds`; the plan, section 8.6).

    python scripts/zone_analysis.py --reports-root ~/edge_mango/runs/final/reports \\
        --dataset ~/edge_mango/data/v4_captura_v3_final --test-folds 0 1 --test-seeds 42 \\
        --photos-export reports/final-runs-export/photos.csv --output reports/zone-analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lot_descriptors import REDUCE, box_mask, damage_contrast, fruit_mask  # noqa: E402
from src.data_pipeline.v4_selection import LATENT_HOURS  # noqa: E402
from src.evaluation.v4_metrics import (  # noqa: E402
    AP50_IOU,
    damage_ap,
    join_photos,
    read_damage_labels,
    read_index,
    read_predictions,
)

ZONES = ("centro", "izquierda", "derecha", "superior", "inferior")
ARMS = ("est_rgb", "est_rgbnir", "kd_rgb", "kd_rgbnir", "est_nir")
RUN_DIR = re.compile(r"fold(\d+)_seed(\d+)")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--test-folds", type=int, nargs="*", default=[])
    parser.add_argument("--test-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--photos-export", type=Path, default=None,
                        help="photos.csv of scripts/export_run_results.py, for zone_within.csv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def zone_ap_rows(args, index: dict) -> list[dict]:
    rows = []
    for arm in ARMS:
        for run in sorted((args.reports_root / arm).glob("fold*_seed*")):
            match = RUN_DIR.fullmatch(run.name)
            if not match:
                continue
            fold, seed = int(match[1]), int(match[2])
            for split in ("val", "test"):
                if split == "test" and (fold not in args.test_folds or seed not in args.test_seeds):
                    continue
                directory = run / split
                if not (directory / "metrics_v4.json").exists():
                    continue
                v4 = json.loads((directory / "metrics_v4.json").read_text())
                labels = args.dataset / "folds_lot" / f"fold_{fold}" / "labels" / split
                photos = join_photos(read_predictions(directory / "predictions.jsonl"), index,
                                     read_damage_labels(labels))
                for zone in ZONES:
                    latent = damage_ap([p for p in photos if p.zona == zone and p.hour in LATENT_HOURS], AP50_IOU)
                    rows.append({
                        "arm": arm, "fold": fold, "seed": seed, "split": split, "zone": zone,
                        "ap50_all_hours": v4["ap50_by_zone"][zone]["ap"],
                        "n_gt_all_hours": v4["ap50_by_zone"][zone]["n_gt"],
                        "ap50_latent": latent["ap"], "n_gt_latent": latent["n_gt"],
                    })
    return rows


def measure(row: dict, dataset: str) -> dict | None:
    if not row["zona"] or not row["damage_x0"]:
        return None
    dataset = Path(dataset)
    stem = row["stem"]
    ts = row["timestamp"]
    rgb = cv2.imread(str(dataset / "rgb" / f"{stem}.jpg"), cv2.IMREAD_REDUCED_COLOR_4)
    nir = cv2.imread(str(dataset / "nir" / f"mango_nir_{ts}.png"), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if rgb is None or nir is None:
        return None
    mango = tuple(float(row[f"mango_{k}"]) for k in ("x0", "y0", "x1", "y1"))
    damage = tuple(float(row[f"damage_{k}"]) for k in ("x0", "y0", "x1", "y1"))
    fruit = fruit_mask(rgb.shape[:2], mango)
    box = box_mask(rgb.shape[:2], damage) & fruit
    b, g, r = (rgb[..., i].astype(np.float32) for i in range(3))
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    nirf = nir.astype(np.float32)
    mx, my = (mango[0] + mango[2]) / 2, (mango[1] + mango[3]) / 2
    sx, sy = (mango[2] - mango[0]) / 2, (mango[3] - mango[1]) / 2
    dx = ((damage[0] + damage[2]) / 2 - mx) / sx
    dy = ((damage[1] + damage[3]) / 2 - my) / sy
    out = {
        "stem": stem, "lot": row["lot"], "zone": row["zona"], "hour": row["ronda_h"],
        "visible_rgb": row["visible_rgb"], "provenance": row["provenance"],
        "pos_x": round(dx, 4), "pos_y": round(dy, 4), "dist_center": round(float(np.hypot(dx, dy)), 4),
        "area_rel": round((damage[2] - damage[0]) * (damage[3] - damage[1]) / (4 * sx * sy), 5),
    }
    if box.sum() >= 20:
        out.update(
            luma_box=float(luma[box].mean()), luma_fruit=float(luma[fruit].mean()),
            nir_box=float(nirf[box].mean()), nir_fruit=float(nirf[fruit].mean()),
            contrast_rgb=damage_contrast(luma, fruit, box), contrast_nir=damage_contrast(nirf, fruit, box),
        )
        out["luma_ratio"] = out["luma_box"] / out["luma_fruit"]
        out["nir_ratio"] = out["nir_box"] / out["nir_fruit"]
    return out


def summarize_conditions(photos: list[dict]) -> list[dict]:
    keys = ("pos_x", "pos_y", "dist_center", "area_rel", "luma_box", "luma_ratio", "nir_box", "nir_ratio",
            "contrast_rgb", "contrast_nir")
    rows = []
    for subset, select in (("all_hours", lambda p: True), ("latent", lambda p: p["hour"] in {"0", "2", "4", "6"})):
        for zone in ZONES:
            ps = [p for p in photos if p["zone"] == zone and select(p)]
            row = {"subset": subset, "zone": zone, "n_photos": len(ps)}
            for k in keys:
                values = [p[k] for p in ps if p.get(k) is not None]
                row[k] = round(statistics.median(values), 4) if values else None
            rows.append(row)
    return rows


def within_zone(photos: list[dict], export: Path) -> list[dict]:
    """Share of latent photo evaluations with a correct damage box at 0.25, per zone, split at the zone median of
    the light and of the distance to the centre. Evaluations of the selected checkpoint only."""
    cond = {p["stem"]: p for p in photos if p.get("luma_ratio") is not None}
    evals = [r for r in csv.DictReader(export.open())
             if r["checkpoint"] == "best" and r["hour"] in {"0", "2", "4", "6"} and r["stem"] in cond]
    rate = lambda rows: sum(int(r["tp_at_0.25"]) > 0 for r in rows) / len(rows) if rows else None  # noqa: E731
    out = []
    for zone in ZONES:
        rows = [r for r in evals if r["zona"] == zone]
        row = {"zone": zone, "n": len(rows), "hit_rate": rate(rows)}
        for key, low, high in (("luma_ratio", "hit_low_light", "hit_high_light"),
                               ("dist_center", "hit_near_center", "hit_far_center")):
            median = statistics.median(cond[r["stem"]][key] for r in rows)
            row[low] = rate([r for r in rows if cond[r["stem"]][key] < median])
            row[high] = rate([r for r in rows if cond[r["stem"]][key] >= median])
            row[f"{key}_median"] = median
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    index = read_index(args.dataset / "index.csv")
    write_csv(args.output / "zone_ap.csv", zone_ap_rows(args, index))
    rows = list(csv.DictReader((args.dataset / "index.csv").open()))
    with ProcessPoolExecutor(args.workers) as pool:
        photos = [p for p in pool.map(measure, rows, [str(args.dataset)] * len(rows), chunksize=16) if p]
    write_csv(args.output / "zone_photos.csv", photos)
    write_csv(args.output / "zone_conditions.csv", summarize_conditions(photos))
    if args.photos_export:
        write_csv(args.output / "zone_within.csv", within_zone(photos, args.photos_export))
    print(f"{len(photos)} photos measured -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
