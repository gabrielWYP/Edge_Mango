#!/usr/bin/env python3
"""Capture conditions of each lot of the v4 dataset, for the exploratory per-lot analysis (plan, section 8.5).

Reads only images, capture metadata and labels: no model output and no test result. Per photo it measures, on the
images the models see (the dataset `rgb/` and the raw registered `nir/`, both in the RGB frame):

- fruit: mean luma and R/B ratio of the RGB, mean and std of the NIR, inside the ellipse inscribed in the mango box;
- damage contrast, in the latent window only (ronda_h 0-6): (mean in the first damage box - mean of the rest of the
  fruit) / std of the rest of the fruit, for RGB luma and for NIR, as an absolute value;

and from the capture metadata: the RGB white balance and exposure set, the light level (`Lux`) and colour temperature
that the NIR camera estimated, and the brightness of the background sheet at the border (`calidad.borde_media`).
From `index.csv` and `exclusions.csv`: the residual RGB-NIR shift, the NIR line artefacts, the white-balance
correction, `visible_rgb = yes` in the latent window and the photos left out per reason.

Writes `lot_descriptors.csv` (one row per lot) and `per_photo.csv` under `--output`.

    python scripts/lot_descriptors.py --dataset data/cache/v4_captura_v3_final \\
        --captura data/captura_v3 --output reports/lot-descriptors --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

LATENT_HOURS = {"0", "2", "4", "6"}
# Images are read at 1/4 of their side: the measures are means over regions of thousands of pixels.
REDUCE = 4


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="built v4 dataset, with index.csv")
    parser.add_argument("--captura", type=Path, required=True, help="raw capture, with news/meta/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="first N photos only, for a quick check")
    return parser.parse_args(argv)


def _float(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


def fruit_mask(shape: tuple[int, int], box: tuple[float, float, float, float]) -> np.ndarray:
    """The ellipse inscribed in the mango box, at the reduced scale."""
    x0, y0, x1, y1 = (v / REDUCE for v in box)
    mask = np.zeros(shape, np.uint8)
    center = (int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2)))
    axes = (max(1, int(round((x1 - x0) / 2))), max(1, int(round((y1 - y0) / 2))))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    return mask.astype(bool)


def box_mask(shape: tuple[int, int], box: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = (int(round(v / REDUCE)) for v in box)
    mask = np.zeros(shape, bool)
    mask[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = True
    return mask


def damage_contrast(channel: np.ndarray, fruit: np.ndarray, damage: np.ndarray) -> float | None:
    inside = channel[fruit & damage]
    rest = channel[fruit & ~damage]
    if inside.size < 20 or rest.size < 20 or rest.std() == 0:
        return None
    return float(abs(inside.mean() - rest.mean()) / rest.std())


def measure_photo(row: dict, dataset: Path, captura: Path) -> dict:
    ts = row["timestamp"]
    out = {"lot": row["lot"], "timestamp": ts, "ronda_h": row["ronda_h"]}
    rgb = cv2.imread(str(dataset / "rgb" / f"mango_rgb_{ts}.jpg"), cv2.IMREAD_REDUCED_COLOR_4)
    nir = cv2.imread(str(dataset / "nir" / f"mango_nir_{ts}.png"), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    mango = tuple(_float(row[f"mango_{k}"]) for k in ("x0", "y0", "x1", "y1"))
    if rgb is not None and nir is not None and None not in mango:
        fruit = fruit_mask(rgb.shape[:2], mango)
        b, g, r = (rgb[..., i].astype(np.float32) for i in range(3))
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        nirf = nir.astype(np.float32)
        out.update(
            rgb_luma=float(luma[fruit].mean()),
            rgb_r_over_b=float(r[fruit].mean() / max(b[fruit].mean(), 1e-6)),
            nir_mean=float(nirf[fruit].mean()),
            nir_std=float(nirf[fruit].std()),
        )
        damage = tuple(_float(row[f"damage_{k}"]) for k in ("x0", "y0", "x1", "y1"))
        if row["ronda_h"] in LATENT_HOURS and None not in damage:
            dmask = box_mask(rgb.shape[:2], damage)
            out.update(
                latent_contrast_rgb=damage_contrast(luma, fruit, dmask),
                latent_contrast_nir=damage_contrast(nirf, fruit, dmask),
            )
    meta_path = captura / "news" / "meta" / f"mango_meta_{ts}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        controls = meta.get("rgb", {}).get("controles", {})
        rpicam = meta.get("nir", {}).get("rpicam_metadata", {})
        quality = meta.get("calidad", {})
        out.update(
            rgb_wb_k=controls.get("white_balance_temperature"),
            rgb_exposure=controls.get("exposure_time_absolute"),
            nir_lux=rpicam.get("Lux"),
            nir_colour_temp_k=rpicam.get("ColourTemperature"),
            rgb_border_mean=(quality.get("rgb") or {}).get("borde_media"),
            nir_border_mean=(quality.get("nir") or {}).get("borde_media"),
            room_temp_c=(meta.get("jornada") or {}).get("temperatura_c"),
        )
    return out


def _median(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return float(statistics.median(values)) if values else None


def _spread(values: list) -> float | None:
    """p90 - p10: how much the measure drifts inside the lot."""
    values = [v for v in values if v is not None]
    return float(np.percentile(values, 90) - np.percentile(values, 10)) if len(values) > 1 else None


def summarize(rows: list[dict], photos: list[dict], exclusions: list[dict]) -> list[dict]:
    by_lot = defaultdict(list)
    for p in photos:
        by_lot[p["lot"]].append(p)
    index_by_lot = defaultdict(list)
    for r in rows:
        index_by_lot[r["lot"]].append(r)
    excluded = Counter((e["lot"], e["reason"]) for e in exclusions)
    reasons = sorted({e["reason"] for e in exclusions})
    lots = []
    for lot in sorted(by_lot, key=int):
        ps, rs = by_lot[lot], index_by_lot[lot]
        col = lambda k: [p.get(k) for p in ps]  # noqa: E731
        latent = [r for r in rs if r["ronda_h"] in LATENT_HOURS]
        shift = [np.hypot(_float(r["residual_dx"]) or 0, _float(r["residual_dy"]) or 0) for r in rs if r["residual_dx"]]
        summary = {
            "lot": lot,
            "n_photos": len(rs),
            "rgb_wb_k": _median(col("rgb_wb_k")),
            "rgb_exposure": _median(col("rgb_exposure")),
            "room_temp_c": _median(col("room_temp_c")),
            "nir_lux": _median(col("nir_lux")),
            "nir_lux_spread": _spread(col("nir_lux")),
            "nir_colour_temp_k": _median(col("nir_colour_temp_k")),
            "nir_colour_temp_spread": _spread(col("nir_colour_temp_k")),
            "rgb_border_mean": _median(col("rgb_border_mean")),
            "rgb_border_spread": _spread(col("rgb_border_mean")),
            "nir_border_mean": _median(col("nir_border_mean")),
            "nir_border_spread": _spread(col("nir_border_mean")),
            "rgb_fruit_luma": _median(col("rgb_luma")),
            "rgb_fruit_r_over_b": _median(col("rgb_r_over_b")),
            "nir_fruit_mean": _median(col("nir_mean")),
            "nir_fruit_std": _median(col("nir_std")),
            "latent_contrast_rgb": _median(col("latent_contrast_rgb")),
            "latent_contrast_nir": _median(col("latent_contrast_nir")),
            "residual_shift_px": _median(shift),
            "residual_corr": _median([_float(r["residual_corr"]) for r in rs]),
            "wb_corrected_share": sum(r["wb_corrected"] == "1" for r in rs) / len(rs),
            "nir_lines_share": sum(r["nir_lines"] not in ("", "0") for r in rs) / len(rs),
            "latent_visible_rgb_yes_share": (
                sum(r["visible_rgb"] == "yes" for r in latent) / len(latent) if latent else None
            ),
        }
        summary.update({f"excluded_{reason}": excluded[(lot, reason)] for reason in reasons})
        lots.append(summary)
    return lots


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = list(csv.DictReader((args.dataset / "index.csv").open()))
    if args.limit:
        rows = rows[: args.limit]
    exclusions_path = args.dataset / "exclusions.csv"
    exclusions = list(csv.DictReader(exclusions_path.open())) if exclusions_path.exists() else []
    with ProcessPoolExecutor(args.workers) as pool:
        photos = list(
            pool.map(measure_photo, rows, [args.dataset] * len(rows), [args.captura] * len(rows), chunksize=16)
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_photo.csv", photos)
    lots = summarize(rows, photos, exclusions)
    write_csv(args.output / "lot_descriptors.csv", lots)
    for lot in lots:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in lot.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
