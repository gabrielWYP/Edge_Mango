#!/usr/bin/env python3
"""Leave-one-lot-out folds of the v4 dataset (plan-experimental 3.3).

`make_grouped_folds.py` finds lots from the capture day and requires more than a day between a
train capture and a val/test one. captura_v3 lots 1 and 2 were captured on consecutive days, so
here the lots come from the `lot` column of the builder's `index.csv`, and the leakage check is
the one of the plan: no `mango_id` is in two splits of any fold. The fold assignment and the
files on disk are those of `make_grouped_folds.py`:

    fold k:  test = lot k,  val = lot (k + 1) mod K,  train = every other lot

    <output-dir>/fold_{k}/splits.json
    <output-dir>/fold_{k}/labels/{train,val,test}/<stem>.txt   copies of <dataset>/labels
    <output-dir>/folds_meta.json

`folds_meta.json` also holds, per fold and per NIR variant (`nir`, `nir_clahe`), the mean and
std of the NIR on [0, 1] as `YOLODataset` normalizes it, over the TRAIN split only, on the full
frame and on the mango ROI crop (`crop_to_roi`, margin 0.10). A fold trains with

    --override labels_dir=<output-dir>/fold_{k}/labels \\
               split_manifest=<output-dir>/fold_{k}/splits.json \\
               nir_mean=<...> nir_std=<...>

Usage:
    ./.venv/bin/python scripts/make_v4_folds.py --dataset data/cache/v4_captura_v3_lines
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import sys
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_grouped_folds import (  # noqa: E402
    SPLITS,
    LeakageError,
    Lot,
    assign_folds,
    count_damage_boxes,
    fold_manifests,
    local_date,
    stem_timestamp,
    write_folds,
)
from src.training.roi_crop import crop_to_roi  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
NIR_VARIANTS = ("nir", "nir_clahe")
NIR_VIEWS = ("frame", "roi")
ROI_MARGIN = 0.10  # TrainingConfig.roi_crop_margin default
DEFAULT_OUTPUT_SUBDIR = "folds_lot"


def read_index(index_path: Path) -> list[dict[str, str]]:
    with Path(index_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def lots_from_index(rows: list[dict[str, str]]) -> list[Lot]:
    """One lot per value of the `lot` column, ordered by lot number (`lot<n>`)."""
    stems: dict[int, list[str]] = {}
    for row in rows:
        stems.setdefault(int(row["lot"]), []).append(row["stem"])
    return [
        Lot(
            f"lot{number}",
            sorted({local_date(stem_timestamp(s)).isoformat() for s in group}),
            sorted(group),
        )
        for number, group in sorted(stems.items())
    ]


def check_mango_ids(rows: list[dict[str, str]], manifests: list[dict[str, list[str]]]) -> None:
    """Raise `LeakageError` unless every fold keeps each mango, and each stem, in one split.

    Also requires every stem to be a test stem in exactly one fold.
    """
    mango_of: dict[str, int] = {}
    for row in rows:
        if row["stem"] in mango_of:
            raise LeakageError(f"Stem {row['stem']} is listed twice in the index.")
        mango_of[row["stem"]] = int(row["mango_id"])
    tested: Counter[str] = Counter()
    for index, manifest in enumerate(manifests):
        mangos = {split: {mango_of[s] for s in manifest[split]} for split in SPLITS}
        stems = {split: set(manifest[split]) for split in SPLITS}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = sorted(mangos[a] & mangos[b])
            if shared or stems[a] & stems[b]:
                raise LeakageError(
                    f"fold {index}: {a} and {b} share mango_id {shared[:5]} or stems."
                )
        tested.update(manifest["test"])
    if set(tested) != set(mango_of) or any(count != 1 for count in tested.values()):
        raise LeakageError(
            f"Not every stem is a test stem in exactly one fold ({len(tested)} tested of "
            f"{len(mango_of)})."
        )


def nir_path(dataset_dir: Path, variant: str, stem: str) -> Path:
    return dataset_dir / variant / f"{stem.replace('mango_rgb_', 'mango_nir_')}.png"


def _moments(image: np.ndarray) -> list[float]:
    values = image.astype(np.float64) / 255.0
    return [float(values.size), float(values.sum()), float(np.square(values).sum())]


def image_moments(job: tuple[str, Path, Path]) -> tuple[str, list[float]]:
    """(stem, [n, sum, sum of squares] of the frame, then of the ROI crop), pixels on [0, 1]."""
    stem, path, label_path = job
    # Same read as YOLODataset; the baked NIR is already in the RGB frame.
    nir = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if nir is None:
        raise ValueError(f"{stem}: cannot decode {path}")
    labels = np.loadtxt(label_path, ndmin=2)
    _, roi, _, _ = crop_to_roi(nir, nir, labels[:, 1:5], labels[:, 0], margin=ROI_MARGIN)
    return stem, _moments(nir) + _moments(roi)


def nir_stats(moments: dict[str, list[float]], stems: list[str]) -> dict[str, dict[str, float]]:
    """Pixel mean and std over `stems`, per view (`frame`, `roi`)."""
    total = np.sum([moments[s] for s in stems], axis=0).reshape(len(NIR_VIEWS), 3)
    stats = {}
    for view, (n, s, ss) in zip(NIR_VIEWS, total):
        mean = s / n
        stats[view] = {"mean": round(mean, 6), "std": round(math.sqrt(max(ss / n - mean**2, 0)), 6)}
    return stats


def _split_counts(stems: list[str], mango_of: dict[str, str], damage: dict[str, int]) -> dict:
    return {
        "damage_boxes": sum(damage[s] for s in stems),
        "images": len(stems),
        "mangos": len({mango_of[s] for s in stems}),
        "negatives": sum(damage[s] == 0 for s in stems),
    }


def make_v4_folds(
    dataset_dir: Path,
    output_dir: Path | None = None,
    workers: int = 8,
) -> dict:
    """Write the folds of `dataset_dir` and return their meta (`folds_meta.json`)."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) if output_dir is not None else dataset_dir / DEFAULT_OUTPUT_SUBDIR
    index_path = dataset_dir / "index.csv"
    rows = read_index(index_path)
    lots = lots_from_index(rows)
    folds = assign_folds(lots)
    manifests = fold_manifests(lots, folds)
    check_mango_ids(rows, manifests)

    label_sources = {r["stem"]: dataset_dir / "labels" / f"{r['stem']}.txt" for r in rows}
    damage = {stem: count_damage_boxes(path) for stem, path in label_sources.items()}
    mango_of = {r["stem"]: r["mango_id"] for r in rows}
    moments = {}
    for variant in NIR_VARIANTS:
        jobs = [(s, nir_path(dataset_dir, variant, s), p) for s, p in label_sources.items()]
        if workers > 1:
            with ProcessPoolExecutor(workers) as pool:
                moments[variant] = dict(pool.map(image_moments, jobs, chunksize=8))
        else:
            moments[variant] = dict(map(image_moments, jobs))

    labels_digest = hashlib.sha256()
    for stem, path in sorted(label_sources.items()):
        labels_digest.update(f"{stem}\t{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode())
    meta = {
        "folds": {
            str(fold.index): {
                "nir_stats": {v: nir_stats(moments[v], manifest["train"]) for v in NIR_VARIANTS},
                "splits": {s: _split_counts(manifest[s], mango_of, damage) for s in SPLITS},
                "test": fold.test,
                "train": fold.train,
                "val": fold.val,
            }
            for fold, manifest in zip(folds, manifests)
        },
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "labels_sha256": labels_digest.hexdigest(),
        "leakage": {"mango_ids_disjoint": True, "every_stem_tested_exactly_once": True},
        "lots": {
            lot.lot_id: {"days": lot.days, **_split_counts(lot.stems, mango_of, damage)}
            for lot in lots
        },
        "nir_stats_params": {
            "pixels": "uint8 / 255 of the NIR PNG read as grayscale, train split only",
            "roi_margin": ROI_MARGIN,
            "views": {"frame": "whole RGB frame", "roi": "crop_to_roi of the class 0 box"},
        },
        "num_images": len(rows),
        "schema_version": SCHEMA_VERSION,
    }
    write_folds(output_dir, manifests, label_sources, meta)
    return meta


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="output of build_v4_dataset")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help=f"default <dataset>/{DEFAULT_OUTPUT_SUBDIR}"
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    meta = make_v4_folds(args.dataset, args.output_dir, args.workers)
    for lot_id, lot in meta["lots"].items():
        logger.info(
            "%s %s: %d images, %d mangos, %d damage boxes, %d negatives",
            lot_id,
            "+".join(lot["days"]),
            lot["images"],
            lot["mangos"],
            lot["damage_boxes"],
            lot["negatives"],
        )
    for index, fold in meta["folds"].items():
        sizes = {s: fold["splits"][s]["images"] for s in SPLITS}
        logger.info(
            "fold_%s test=%s val=%s train=%s %s",
            index,
            fold["test"],
            fold["val"],
            fold["train"],
            sizes,
        )
        for variant, views in fold["nir_stats"].items():
            logger.info(
                "  %-9s mean/std  frame %.4f/%.4f  roi %.4f/%.4f",
                variant,
                views["frame"]["mean"],
                views["frame"]["std"],
                views["roi"]["mean"],
                views["roi"]["std"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
