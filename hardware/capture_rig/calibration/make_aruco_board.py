"""Generate the printable ArUco board used as the calibration-jig plate texture.

The board is a 5 x 3 GridBoard of DICT_4X4_50 markers (ids 0-14) on a white 140 x 90 mm sheet
rendered at 300 dpi. Print it at 100 % scale (no "fit to page") and glue it on the jig plate.

Run from the repository virtual environment (it provides opencv with the aruco module)::

    .venv/bin/python -I hardware/capture_rig/calibration/make_aruco_board.py

Marker geometry (mm) is written to ``aruco_board_140x90mm.json`` so the calibration script can
build the same ``cv2.aruco.GridBoard`` without guessing.
"""

import json
import sys
from pathlib import Path

import cv2

DPI = 300
SHEET_MM = (140.0, 90.0)
GRID = (5, 3)              # columns, rows
MARKER_MM = 22.0
SEPARATION_MM = 5.0
DICT_NAME = "DICT_4X4_50"
OUT_NAME = "aruco_board_140x90mm"


def main(out_dir: Path) -> int:
    if not hasattr(cv2, "aruco"):
        print("cv2.aruco is not available in this interpreter", file=sys.stderr)
        return 2
    px_per_mm = DPI / 25.4
    size_px = (round(SHEET_MM[0] * px_per_mm), round(SHEET_MM[1] * px_per_mm))
    board_mm = (GRID[0] * MARKER_MM + (GRID[0] - 1) * SEPARATION_MM,
                GRID[1] * MARKER_MM + (GRID[1] - 1) * SEPARATION_MM)
    margin_px = round((SHEET_MM[0] - board_mm[0]) / 2 * px_per_mm)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    board = cv2.aruco.GridBoard(GRID, MARKER_MM, SEPARATION_MM, dictionary)
    image = board.generateImage(size_px, marginSize=margin_px, borderBits=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    jpg = out_dir / f"{OUT_NAME}.jpg"
    cv2.imwrite(str(jpg), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    meta = {
        "dictionary": DICT_NAME,
        "grid_columns_rows": list(GRID),
        "marker_length_mm": MARKER_MM,
        "marker_separation_mm": SEPARATION_MM,
        "sheet_mm": list(SHEET_MM),
        "dpi": DPI,
        "image_px": list(size_px),
        "ids": list(range(GRID[0] * GRID[1])),
        "origin_note": "board frame origin = sheet centre; plate top surface sits at Z = 150 mm in the rig",
    }
    (out_dir / f"{OUT_NAME}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {jpg} ({size_px[0]}x{size_px[1]} px)")
    return 0


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else here))
