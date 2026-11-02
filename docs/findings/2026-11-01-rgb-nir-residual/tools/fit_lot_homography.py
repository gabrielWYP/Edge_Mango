"""Fit the lot 4 RGB -> NIR homography and check the method against lots 1-3.

The script that produced the lots 1-3 matrices (homografia_nueva.py) is not in the repository, so the same
estimator (marker centres, RANSAC 5 px) is rebuilt here and compared with the published matrices on ChArUco
corners, which the fit does not use.

Usage: fit_lot_homography.py --calib-root calibracion/captura_v3
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# Legacy pattern, 11 x 8 squares: the only layout that finds all 70 inner corners on these boards.
BOARD = cv2.aruco.CharucoBoard((11, 8), 0.012, 0.009, DICTIONARY)
BOARD.setLegacyPattern(True)
CLAHE = cv2.createCLAHE(3.0, (8, 8))
RGB_SIZE = (3264, 2448)

# (label, folder, pair prefix, timestamp, published matrix or None)
PAIRS = [
    ("lot 1", "lote1-20261024", "calib-aruco", 1792857744, "matriz_homografia_lote1_20261024_centros.npy"),
    ("lot 2", "lote2-20261025", "calib-aruco", 1792941738, "matriz_homografia_aruco_3264x2448_lote2_20261025_centros.npy"),
    ("lot 3 morning", "lote3-20261027", "calib-aruco", 1793100324, "matriz_homografia_aruco_3264x2448_lote3_20261027_centros.npy"),
    ("lot 3 closing", "lote3-20261027", "calib-aruco-final", 1793154347, "matriz_homografia_aruco_3264x2448_lote3_20261027_final_centros.npy"),
    ("lot 4", "lote4-20261031", "calib-aruco", 1793454960, None),
]
LOT4_MATRIX = "matriz_homografia_aruco_3264x2448_lote4_20261031_centros.npy"


def detector_params() -> cv2.aruco.DetectorParameters:
    """Wider adaptive threshold than the defaults: they miss up to 40% of the markers on the NIR image."""
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 63
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.errorCorrectionRate = 0.8
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


def marker_centres(gray: np.ndarray) -> dict[int, np.ndarray]:
    corners, ids, _ = cv2.aruco.ArucoDetector(DICTIONARY, detector_params()).detectMarkers(CLAHE.apply(gray))
    return {} if ids is None else {int(i): c.reshape(4, 2).mean(0) for i, c in zip(ids.ravel(), corners)}


def charuco_corners(gray: np.ndarray) -> dict[int, np.ndarray]:
    detector = cv2.aruco.CharucoDetector(BOARD, cv2.aruco.CharucoParameters(), detector_params())
    corners, ids, _, _ = detector.detectBoard(CLAHE.apply(gray))
    return {} if ids is None else {int(i): c.ravel() for i, c in zip(ids.ravel(), corners)}


def corner_error(homography: np.ndarray, rgb: dict, nir: dict) -> dict:
    """Error on the ChArUco corners common to both images, in the RGB frame (the README convention)."""
    common = sorted(set(rgb) & set(nir))
    rgb_pts = np.float32([rgb[i] for i in common])
    nir_pts = np.float32([nir[i] for i in common])
    errors = np.linalg.norm(cv2.perspectiveTransform(nir_pts[None], np.linalg.inv(homography))[0] - rgb_pts, axis=1)
    return {
        "mediana": round(float(np.median(errors)), 2),
        "p95": round(float(np.percentile(errors, 95)), 2),
        "max": round(float(errors.max()), 2),
        "n": len(common),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib-root", type=Path, required=True)
    args = parser.parse_args()

    cv2.setRNGSeed(0)
    previous_published = None
    for label, folder, prefix, timestamp, published_name in PAIRS:
        directory = args.calib_root / folder
        rgb = cv2.imread(str(directory / f"{prefix}-rgb-{timestamp}.jpg"), cv2.IMREAD_GRAYSCALE)
        nir = cv2.imread(str(directory / f"{prefix}-nir-{timestamp}.jpg"), cv2.IMREAD_GRAYSCALE)
        rgb_centres, nir_centres = marker_centres(rgb), marker_centres(nir)
        common = sorted(set(rgb_centres) & set(nir_centres))
        homography, inliers = cv2.findHomography(
            np.float32([rgb_centres[i] for i in common]),
            np.float32([nir_centres[i] for i in common]),
            cv2.RANSAC,
            5.0,
        )
        rgb_corners, nir_corners = charuco_corners(rgb), charuco_corners(nir)
        fitted = corner_error(homography, rgb_corners, nir_corners)
        if published_name is not None:
            published = np.load(directory / published_name)
            print(f"{label}: rebuilt {fitted} | published {corner_error(published, rgb_corners, nir_corners)}")
            previous_published = published
            continue

        print(f"{label}: fitted {fitted}")
        np.save(directory / LOT4_MATRIX, homography)
        report = {
            "marcadores": {
                "rgb": len(rgb_centres),
                "nir": len(nir_centres),
                "comunes": len(common),
                "inliers_centros": int(inliers.sum()),
            },
            "esquinas_charuco_comunes": len(set(rgb_corners) & set(nir_corners)),
            "error_en_esquinas_charuco_px": {
                "lote3_final": corner_error(previous_published, rgb_corners, nir_corners),
                "nueva_centros": fitted,
            },
            "zona_rgb_cubierta_por_esquinas_pct": {
                "x": [round(float(min(c[0] for c in rgb_corners.values())) / RGB_SIZE[0] * 100, 1),
                      round(float(max(c[0] for c in rgb_corners.values())) / RGB_SIZE[0] * 100, 1)],
                "y": [round(float(min(c[1] for c in rgb_corners.values())) / RGB_SIZE[1] * 100, 1),
                      round(float(max(c[1] for c in rgb_corners.values())) / RGB_SIZE[1] * 100, 1)],
            },
            "fuente": f"calib/{prefix}-{{rgb,nir}}-{timestamp}.jpg de la jornada {folder}",
            "metodo": "docs/findings/2026-11-01-rgb-nir-residual/tools/fit_lot_homography.py",
        }
        (directory / "informe_homografia.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
