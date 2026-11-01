"""Measure, photo by photo, how far the fruit still is from aligned after the RGB -> NIR homography.

The NIR is warped into the RGB frame with the matrix of its lot, then the shift that best matches the gradients of
both images over the central area is searched at half resolution (precision 2 px, range +-160 px).

Usage: registration_residuals.py --data-root data/cache/captura_v3 --calib-root calibracion/captura_v3 --out-dir DIR
"""
import argparse
import collections
import csv
import json
import multiprocessing
import random
import statistics
from pathlib import Path

import cv2
import numpy as np

SCALE = 0.5
MARGIN = int(160 * SCALE)
LOT3_MORNING_LAST_TS = 1793131668
MOMENTS = ["sano", "0", "2", "4", "6", "8", "10", "12"]
OVERLAYS = [("lote1", "6"), ("lote2", "6"), ("lote3", "2"), ("lote3", "12"), ("lote4", "sano"), ("lote4", "12")]
FIELDS = ["jornada", "mango_id", "cara", "momento", "timestamp", "matrix", "corr_h_only", "corr_best", "dx_px", "dy_px", "at_search_edge"]


def matrix_paths(calib_root: Path) -> dict[str, Path]:
    return {
        "lote1": calib_root / "lote1-20261024/matriz_homografia_lote1_20261024_centros.npy",
        "lote2": calib_root / "lote2-20261025/matriz_homografia_aruco_3264x2448_lote2_20261025_centros.npy",
        "lote3_morning": calib_root / "lote3-20261027/matriz_homografia_aruco_3264x2448_lote3_20261027_centros.npy",
        "lote3_closing": calib_root / "lote3-20261027/matriz_homografia_aruco_3264x2448_lote3_20261027_final_centros.npy",
        "lote4": calib_root / "lote4-20261031/matriz_homografia_aruco_3264x2448_lote4_20261031_centros.npy",
    }


def matrix_key(jornada: str, timestamp: int) -> str:
    lot = jornada.split("-")[0]
    if lot == "lote3":
        return "lote3_morning" if timestamp <= LOT3_MORNING_LAST_TS else "lote3_closing"
    return lot


def gradient(image: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(image, (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))


def warped_pair(data_root: Path, matrix_path: Path, timestamp: int) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.imread(str(data_root / f"news/rgb/mango_rgb_{timestamp}.jpg"), cv2.IMREAD_GRAYSCALE)
    nir = cv2.imread(str(data_root / f"news/nir/mango_nir_{timestamp}.jpg"), cv2.IMREAD_GRAYSCALE)
    return rgb, cv2.warpPerspective(nir, np.linalg.inv(np.load(matrix_path)), (rgb.shape[1], rgb.shape[0]))


def measure(job: tuple) -> dict:
    meta, data_root, matrices = job
    cv2.setNumThreads(1)
    timestamp, jornada = meta["timestamp"], meta["jornada"]["id"]
    key = matrix_key(jornada, timestamp)
    rgb, warped = warped_pair(data_root, matrices[key], timestamp)
    rgb_grad = gradient(cv2.resize(rgb, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA))
    nir_grad = gradient(cv2.resize(warped, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA))
    height, width = rgb_grad.shape
    top, bottom, left, right = int(0.2 * height), int(0.85 * height), int(0.3 * width), int(0.8 * width)
    scores = cv2.matchTemplate(
        rgb_grad[top - MARGIN : bottom + MARGIN, left - MARGIN : right + MARGIN],
        nir_grad[top:bottom, left:right],
        cv2.TM_CCOEFF_NORMED,
    )
    _, best, _, (x, y) = cv2.minMaxLoc(scores)
    return {
        "jornada": jornada,
        "mango_id": meta["mango_id"],
        "cara": meta["cara"],
        "momento": "sano" if meta["fase"] == "sano" else meta["ronda_h"],
        "timestamp": timestamp,
        "matrix": key,
        "corr_h_only": round(float(scores[MARGIN, MARGIN]), 3),
        "corr_best": round(float(best), 3),
        "dx_px": (x - MARGIN) / SCALE,
        "dy_px": (y - MARGIN) / SCALE,
        "at_search_edge": int(x in (0, 2 * MARGIN) or y in (0, 2 * MARGIN)),
    }


def summarize(rows: list[dict]) -> None:
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["jornada"], str(row["momento"]))].append(row)
    print("| Lot | Moment | n | corr H only | corr best | dx | dy | at search edge |")
    print("|---|---|---|---|---|---|---|---|")
    for jornada in sorted({row["jornada"] for row in rows}):
        for moment in MOMENTS:
            group = groups[(jornada, moment)]
            median = lambda field: statistics.median(float(row[field]) for row in group)
            print(
                f"| {jornada} | {moment} | {len(group)} | {median('corr_h_only'):.2f} | {median('corr_best'):.2f} "
                f"| {median('dx_px'):+.0f} | {median('dy_px'):+.0f} | {sum(row['at_search_edge'] for row in group)} |"
            )


def write_overlays(rows: list[dict], data_root: Path, matrices: dict[str, Path], out: Path) -> None:
    """Green = RGB, magenta = warped NIR. Left: homography only. Right: plus the measured shift."""
    random.seed(1)
    clahe = cv2.createCLAHE(3.0, (8, 8))
    tiles = []
    for lot, moment in OVERLAYS:
        row = random.choice([r for r in rows if r["jornada"].startswith(lot) and str(r["momento"]) == moment])
        rgb, warped = warped_pair(data_root, matrices[row["matrix"]], row["timestamp"])
        rgb, warped = clahe.apply(rgb), clahe.apply(warped)
        panels = []
        for shift, text in (((0.0, 0.0), f"{lot} {row['mango_id']}-{row['cara']} {moment} h: homography only"),
                            ((row["dx_px"], row["dy_px"]), f"plus shift {row['dx_px']:+.0f}, {row['dy_px']:+.0f} px")):
            moved = cv2.warpAffine(warped, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (rgb.shape[1], rgb.shape[0]))
            panel = cv2.resize(np.dstack([moved, rgb, moved]), (816, 612), interpolation=cv2.INTER_AREA)
            cv2.putText(panel, text, (8, 24), 0, 0.6, (255, 255, 255), 2)
            panels.append(panel)
        tiles.append(np.hstack(panels))
    cv2.imwrite(str(out), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calib-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    matrices = matrix_paths(args.calib_root)
    metas = [json.loads(path.read_text()) for path in sorted((args.data_root / "news/meta").glob("*.json"))]
    # Photos the capture program set aside are not part of the capture set.
    jobs = [(meta, args.data_root, matrices) for meta in metas if not meta.get("apartada")]
    with multiprocessing.Pool(args.workers) as pool:
        rows = sorted(pool.imap_unordered(measure, jobs, chunksize=8), key=lambda row: row["timestamp"])

    with (args.out_dir / "registration_residuals.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows)
    write_overlays(rows, args.data_root, matrices, args.out_dir / "overlays.jpg")


if __name__ == "__main__":
    main()
