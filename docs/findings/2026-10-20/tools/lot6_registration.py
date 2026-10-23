"""Lot 6 (capture day 2026-10-18): capture selection and lot-level NIR -> RGB registration.

Analysis tooling for docs/findings/2026-10-20, not part of the training pipeline. Run from the
repository root under isolated mode:

    <venv>/bin/python -I docs/findings/2026-10-20/tools/lot6_registration.py

What it does:
  1. Applies the lot-6 selection (16 kept captures, 20 excluded) to the QC table and writes
     lot6_selection.csv.
  2. Warps the NIR of each kept pair onto the RGB grid with the project's own code path
     (src.training.nir_registration.register_nir_to_rgb and the base homography).
  3. Searches the integer shift (dx, dy) of the warped NIR that maximises the edge-map normalized
     cross-correlation (NCC) inside the fruit ROI, summed over the 16 captures: the lot shift.
  4. Composes the lot-6 homography in the project convention (RGB -> NIR):
         H_lot6 = H_base @ T(-dx, -dy)
     and verifies it by re-registering through register_nir_to_rgb (SELF-CHECK). The tool exits
     non-zero, and writes no evidence file, when the check fails.
  5. Writes matriz_homografia_lot6_20261018.npy, lot6_registration.json and lot6_selection.csv, plus an
     edge-overlay figure (base H against H_lot6) for visual confirmation.

Shift convention: the RGB pixel (x, y) is compared with the warped-NIR pixel (x - dx, y - dy), so the
corrected image is W'(p) = W(p - d) with W(p) = NIR(H p).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Isolated mode (-I) does not add the script directory to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import build_parser, resolve_roots  # noqa: E402

BASE_HOMOGRAPHY_RELATIVE = "notebooks/matriz_homografia_aruco.npy"
INCOMING_RELATIVE = "data/cache/incoming/2026-10-18"
LOT6_NPY = "matriz_homografia_lot6_20261018.npy"
LOT6_JSON = "lot6_registration.json"
LOT6_CSV = "lot6_selection.csv"
LOT6_FIGURE = "lot6_registration_overlay.jpg"
FIGURE_RELATIVE = "docs/findings/2026-10-20/figures"

# --- Lot-6 selection (decided by the dataset owner; see lot6-intake.md) ---
KEEP_TIMESTAMPS = (
    1792348876, 1792349334, 1792351957, 1792353173,
    1792353731, 1792353836, 1792353937, 1792354158,
    1792355197, 1792355284, 1792356266, 1792356345,
    1792357237, 1792357582, 1792357752, 1792357845,
)
PRE_METADATA_TIMESTAMPS = (1792336581, 1792336785, 1792336910, 1792337046)
BLOWN_OUT_MANGOS = (5, 6, 7, 8)
LATE_MINUTES = 10.0
N_KEEP, N_EXCLUDE = 16, 20

SELECTION_COLUMNS = (
    "stem", "timestamp", "local_time", "mango_id", "face", "decision", "reason",
    "qc_flags", "impacted_faces", "impact_zone", "minutes_since_impact",
)

# --- Registration parameters ---
BLUR_KSIZE = 7          # Gaussian blur before the Sobel magnitude
ROI_MARGIN = 30         # px added around the QC fruit bounding box
SEARCH = 18             # shifts searched: [-SEARCH, SEARCH] in x and y
SELF_CHECK_TOLERANCE = 1    # px: the lot shift after correction must be within this of (0, 0)
SELF_CHECK_MIN_NCC = 0.15   # mean NCC at (0, 0) after correction


# ----------------------------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------------------------
def _bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _number(value: str) -> float | None:
    value = str(value).strip()
    return float(value) if value else None


def _int_text(value: str) -> str:
    """'3.000' -> '3'; '' -> ''."""
    number = _number(value)
    return "" if number is None else str(int(round(number)))


def _faces_text(value: str) -> str:
    """Impacted faces as '1' or '1;2' (the QC table stores one number per capture)."""
    return ";".join(_int_text(part) for part in str(value).replace(";", ",").split(",") if part.strip())


def _minutes_text(value: str) -> str:
    number = _number(value)
    return "" if number is None else f"{number:g}"


def build_selection(qc_rows: list[dict]) -> list[dict]:
    """Apply the lot-6 decision to every QC row that has an RGB or a NIR image.

    The two test captures only have a metadata JSON (no image), so they are not rows of the table; their
    (mango_id, face) is used to flag the kept capture that repeats it.
    """
    test_keys: dict[tuple[str, str], list[str]] = {}
    for row in qc_rows:
        if not (_bool(row["has_rgb"]) or _bool(row["has_nir"])):
            test_keys.setdefault((_int_text(row["mango_id"]), _int_text(row["cara"])), []).append(row["timestamp"])

    selection = []
    for row in qc_rows:
        if not (_bool(row["has_rgb"]) or _bool(row["has_nir"])):
            continue
        ts = int(row["timestamp"])
        mango, face = _int_text(row["mango_id"]), _int_text(row["cara"])
        flags = [f for f in row["flags"].split("|") if f]
        notes = row["notes"].strip()
        minutes = _number(row["minutos_desde_impacto"])
        qc_flags = "|".join(flags + ([notes] if notes and " " not in notes else []))

        if ts in KEEP_TIMESTAMPS:
            decision = "keep"
            parts = ["kept"]
            if "NIR_OVER" in flags:
                parts.append(f"NIR_OVER: {float(row['nir_region_sat_pct']):.1f} % of the fruit region saturated in NIR")
            if minutes is not None and minutes > LATE_MINUTES:
                parts.append(f"{minutes:g} min after impact (> {LATE_MINUTES:g} min)")
            if (mango, face) in test_keys:
                tests = test_keys[(mango, face)]
                parts.append(f"same mango_id and face as the {len(tests)} test captures ({', '.join(tests)})")
            if "MISALIGN" in flags:
                parts.append("MISALIGN is not a reason to exclude (the whole lot shares one offset)")
            reason = "; ".join(parts)
        elif ts in PRE_METADATA_TIMESTAMPS:
            decision = "exclude"
            reason = "pre-metadata capture: no JSON sidecar, auto exposure, RGB overexposed"
        elif mango and int(mango) in BLOWN_OUT_MANGOS:
            decision = "exclude"
            sheet = ",".join(f"{float(row[c]):.0f}" for c in ("sheet_R", "sheet_G", "sheet_B"))
            reason = f"RGB blown out (white reference R,G,B = {sheet})"
            if "DUP_FACE" in flags:
                reason += f"; duplicate of mango {mango} face {face}"
        else:
            raise ValueError(f"capture {ts} (mango {mango!r}, face {face!r}) has no selection rule")

        selection.append({
            "stem": f"mango_rgb_{ts}", "timestamp": ts, "local_time": row["local_time"],
            "mango_id": mango, "face": face, "decision": decision, "reason": reason,
            "qc_flags": qc_flags, "impacted_faces": _faces_text(row["caras_impactadas"]),
            "impact_zone": row["zona"], "minutes_since_impact": _minutes_text(row["minutos_desde_impacto"]),
        })

    kept = [s for s in selection if s["decision"] == "keep"]
    if (len(kept), len(selection) - len(kept)) != (N_KEEP, N_EXCLUDE):
        raise ValueError(f"expected {N_KEEP} keep / {N_EXCLUDE} exclude, got {len(kept)} / {len(selection) - len(kept)}")
    if {s["timestamp"] for s in kept} != set(KEEP_TIMESTAMPS):
        raise ValueError("the kept captures do not match KEEP_TIMESTAMPS")
    return selection


# ----------------------------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------------------------
def edge_map(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude: Gaussian blur (BLUR_KSIZE) followed by the Sobel magnitude."""
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (BLUR_KSIZE, BLUR_KSIZE), 0)
    return cv2.magnitude(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))


def fruit_roi(bbox: tuple[float, float, float, float], image_size: tuple[int, int], margin: int, search: int) -> tuple[int, int, int, int]:
    """QC bounding box (x, y, w, h) expanded by `margin`, clipped so every searched shift stays inside the image."""
    width, height = image_size
    x, y, w, h = bbox
    x0, y0 = max(int(round(x)) - margin, search), max(int(round(y)) - margin, search)
    x1, y1 = min(int(round(x + w)) + margin, width - search), min(int(round(y + h)) + margin, height - search)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty ROI for bbox {bbox}")
    return x0, y0, x1, y1


def ncc_surface(edges_rgb: np.ndarray, edges_nir: np.ndarray, roi: tuple[int, int, int, int], search: int) -> np.ndarray:
    """NCC of RGB(p) against warped-NIR(p - d) for every integer d = (dx, dy) in [-search, search]^2.

    Returns an array `s` of shape (2*search+1, 2*search+1) with s[dy + search, dx + search] = NCC (the Pearson
    correlation over the ROI pixels, which is cv2.TM_CCOEFF_NORMED). The ROI must lie at least `search` px
    inside the image (see fruit_roi) so that all shifts use the same pixels.
    """
    x0, y0, x1, y1 = roi
    template = np.ascontiguousarray(edges_rgb[y0:y1, x0:x1])
    window = np.ascontiguousarray(edges_nir[y0 - search:y1 + search, x0 - search:x1 + search])
    # result[j, i] correlates the template with window[j:j+h, i:i+w]; that window starts at NIR column
    # x0 - search + i, which is RGB column x0 minus dx for dx = search - i (likewise dy = search - j).
    # Reversing both axes therefore indexes the surface by (dy + search, dx + search).
    result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    return np.ascontiguousarray(result[::-1, ::-1]).astype(np.float64)


def surface_at(surface: np.ndarray, dx: int, dy: int, search: int) -> float:
    return float(surface[dy + search, dx + search])


def argmax_shift(surface: np.ndarray, search: int) -> tuple[int, int]:
    """(dx, dy) of the maximum; ties resolve to the first in row-major order."""
    j, i = np.unravel_index(int(np.argmax(surface)), surface.shape)
    return int(i) - search, int(j) - search


def compose_lot_homography(h_base: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """H_lot = H_base @ T(-dx, -dy), in the same RGB -> NIR convention as H_base.

    With W(p) = NIR(H p), the corrected image W'(p) = W(p - d) = NIR(H (p - d)) = NIR(H T(-d) p).
    """
    dx, dy = shift
    translation = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy], [0.0, 0.0, 1.0]])
    return np.asarray(h_base, dtype=np.float64) @ translation


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def surfaces_for(pairs: list[dict], h_rgb_to_nir: np.ndarray, register, search: int) -> list[np.ndarray]:
    """One NCC surface per capture after registering its NIR with `h_rgb_to_nir`."""
    surfaces = []
    for pair in pairs:
        height, width = pair["gray"].shape
        warped = register(pair["nir"], h_rgb_to_nir, (width, height), 0)
        surfaces.append(ncc_surface(pair["edges_rgb"], edge_map(warped), pair["roi"], search))
    return surfaces


def load_pairs(incoming: Path, selection: list[dict], qc_rows: list[dict], margin: int, search: int) -> list[dict]:
    qc_by_ts = {int(r["timestamp"]): r for r in qc_rows}
    pairs = []
    for item in (s for s in selection if s["decision"] == "keep"):
        ts = item["timestamp"]
        rgb = cv2.imread(str(incoming / "rgb" / f"mango_rgb_{ts}.jpg"), cv2.IMREAD_COLOR)
        nir = cv2.imread(str(incoming / "nir" / f"mango_nir_{ts}.jpg"), cv2.IMREAD_GRAYSCALE)
        if rgb is None or nir is None:
            raise FileNotFoundError(f"missing RGB or NIR image for {ts} under {incoming}")
        if nir.shape != rgb.shape[:2]:
            raise ValueError(f"{ts}: RGB {rgb.shape[:2]} and NIR {nir.shape} differ in size")
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        qc = qc_by_ts[ts]
        bbox = tuple(float(qc[c]) for c in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
        pairs.append({
            "stem": item["stem"], "mango_id": int(item["mango_id"]), "face": int(item["face"]),
            "gray": gray, "nir": nir, "edges_rgb": edge_map(gray),
            "roi": fruit_roi(bbox, (gray.shape[1], gray.shape[0]), margin, search),
        })
    return pairs


def overlay_tile(pair: dict, h_rgb_to_nir: np.ndarray, register, label: str, size: tuple[int, int] = (420, 320)) -> np.ndarray:
    """Edge overlay around the fruit: red = RGB edges, green = registered-NIR edges, yellow = they coincide."""
    height, width = pair["gray"].shape
    edges_nir = edge_map(register(pair["nir"], h_rgb_to_nir, (width, height), 0))
    x0r, y0r, x1r, y1r = pair["roi"]
    tile_w, tile_h = size
    x0 = int(np.clip((x0r + x1r) // 2 - tile_w // 2, 0, width - tile_w))
    y0 = int(np.clip((y0r + y1r) // 2 - tile_h // 2, 0, height - tile_h))

    def scaled(edges: np.ndarray) -> np.ndarray:
        crop = edges[y0:y0 + tile_h, x0:x0 + tile_w]
        return np.clip(255.0 * crop / max(float(np.percentile(crop, 99)), 1e-6), 0, 255).astype(np.uint8)

    tile = np.zeros((tile_h, tile_w, 3), np.uint8)  # BGR
    tile[..., 2], tile[..., 1] = scaled(pair["edges_rgb"]), scaled(edges_nir)
    cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def write_overlay(path: Path, pairs: list[dict], per_capture: list[dict], h_base: np.ndarray, h_lot6: np.ndarray, register) -> list[str]:
    """Contact sheet, base H on the left and H_lot6 on the right, for the best, the median and the worst-residual capture."""
    n = len(pairs)
    by_ncc = sorted(range(n), key=lambda i: per_capture[i]["ncc_at_lot_shift"])
    rank = {i: r for r, i in enumerate(by_ncc)}
    best = by_ncc[-1]
    worst = max((i for i in range(n) if i != best),
                key=lambda i: abs(per_capture[i]["residual_dx_lot6"]) + abs(per_capture[i]["residual_dy_lot6"]))
    middle = min((i for i in range(n) if i not in (best, worst)), key=lambda i: abs(rank[i] - n // 2))
    picks = [best, middle, worst]
    rows = []
    for i in picks:
        ts = pairs[i]["stem"].split("_")[-1]
        rows.append(np.hstack([overlay_tile(pairs[i], h_base, register, f"{ts}  base H"),
                               overlay_tile(pairs[i], h_lot6, register, f"{ts}  H_lot6")]))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return [pairs[i]["stem"] for i in picks]


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def _round(value: float) -> float:
    return round(float(value), 6)


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--incoming-dir", type=Path, default=None,
                        help=f"incoming capture day (rgb/, nir/, qc/qc.csv); default <data-root>/{INCOMING_RELATIVE}")
    parser.add_argument("--base-homography", type=Path, default=None,
                        help=f"RGB -> NIR homography to correct; default <repo-root>/{BASE_HOMOGRAPHY_RELATIVE} (tracked in git)")
    parser.add_argument("--search", type=int, default=SEARCH, help="shifts searched in [-N, N] (default %(default)s)")
    parser.add_argument("--roi-margin", type=int, default=ROI_MARGIN,
                        help="px added around the QC fruit bounding box (default %(default)s)")
    parser.add_argument("--figure-dir", type=Path, default=None,
                        help=f"directory of the overlay figure; default <repo-root>/{FIGURE_RELATIVE}")
    ns = parser.parse_args()
    repo_root, data_root, out_dir = resolve_roots(ns)
    figure_dir = (ns.figure_dir or repo_root / FIGURE_RELATIVE).resolve()
    incoming = (ns.incoming_dir or data_root / INCOMING_RELATIVE).resolve()
    base_path = (ns.base_homography or repo_root / BASE_HOMOGRAPHY_RELATIVE).resolve()
    search, margin = ns.search, ns.roi_margin

    sys.path.insert(0, str(repo_root))
    from src.training.nir_registration import homography_sha256, load_homography, register_nir_to_rgb  # noqa: E402

    qc_path = incoming / "qc" / "qc.csv"
    for required in (qc_path, base_path):
        if not required.is_file():
            print(f"missing input: {required}", file=sys.stderr)
            return 2
    with qc_path.open(newline="") as handle:
        qc_rows = list(csv.DictReader(handle))
    selection = build_selection(qc_rows)
    pairs = load_pairs(incoming, selection, qc_rows, margin, search)
    h_base = load_homography(base_path)

    # 1. Lot shift with the base homography.
    surfaces_base = surfaces_for(pairs, h_base, register_nir_to_rgb, search)
    lot_surface = np.mean(surfaces_base, axis=0)
    shift = argmax_shift(lot_surface, search)
    dx, dy = shift
    per_capture = []
    for pair, surf in zip(pairs, surfaces_base):
        best_dx, best_dy = argmax_shift(surf, search)
        per_capture.append({
            "stem": pair["stem"], "mango": pair["mango_id"], "face": pair["face"],
            "best_dx": best_dx, "best_dy": best_dy,
            "ncc_at_zero_base": surface_at(surf, 0, 0, search),
            "ncc_at_lot_shift": surface_at(surf, dx, dy, search),
        })
    mean_zero_base = float(np.mean([c["ncc_at_zero_base"] for c in per_capture]))
    mean_at_shift = float(np.mean([c["ncc_at_lot_shift"] for c in per_capture]))
    order = np.argsort(lot_surface, axis=None)[::-1][:5]
    top_shifts = [{"dx": int(i) - search, "dy": int(j) - search, "mean_ncc": _round(lot_surface[j, i])}
                  for j, i in (np.unravel_index(k, lot_surface.shape) for k in order)]

    # Direction check: registering with H instead of inv(H) must not produce a coherent peak.
    surfaces_alt = surfaces_for(pairs, np.linalg.inv(h_base), register_nir_to_rgb, search)
    alt_surface = np.mean(surfaces_alt, axis=0)
    alt_shift = argmax_shift(alt_surface, search)

    # 2. Lot-6 homography and SELF-CHECK through the project's registration code.
    h_lot6 = compose_lot_homography(h_base, shift)
    surfaces_lot6 = surfaces_for(pairs, h_lot6, register_nir_to_rgb, search)
    lot6_surface = np.mean(surfaces_lot6, axis=0)
    shift_after = argmax_shift(lot6_surface, search)
    mean_zero_lot6 = float(np.mean([surface_at(s, 0, 0, search) for s in surfaces_lot6]))
    passed = (max(abs(shift_after[0]), abs(shift_after[1])) <= SELF_CHECK_TOLERANCE and mean_zero_lot6 >= SELF_CHECK_MIN_NCC)
    for entry, surf in zip(per_capture, surfaces_lot6):
        entry["ncc_at_zero_lot6"] = surface_at(surf, 0, 0, search)
        entry["residual_dx_lot6"], entry["residual_dy_lot6"] = argmax_shift(surf, search)

    print(f"lot shift (dx, dy) = ({dx}, {dy}) px over {len(pairs)} captures; ROI = fruit bbox + {margin} px; search +-{search} px")
    print(f"mean edge NCC: {mean_zero_base:.3f} at (0, 0) with the base H -> {mean_at_shift:.3f} at ({dx}, {dy})")
    print(f"opposite direction (inv(H) passed to register_nir_to_rgb): best mean NCC {alt_surface.max():.3f} at {alt_shift}")
    header = f"{'stem':<24}{'mango':>6}{'face':>5}{'best':>10}{'ncc@0 base':>12}{'ncc@lot':>9}{'ncc@0 lot6':>12}{'resid':>9}"
    print(header)
    for c in per_capture:
        best = "({},{})".format(c["best_dx"], c["best_dy"])
        residual = "({},{})".format(c["residual_dx_lot6"], c["residual_dy_lot6"])
        print(f"{c['stem']:<24}{c['mango']:>6}{c['face']:>5}{best:>10}"
              f"{c['ncc_at_zero_base']:>12.3f}{c['ncc_at_lot_shift']:>9.3f}{c['ncc_at_zero_lot6']:>12.3f}{residual:>9}")
    print(f"after correction: lot shift {shift_after}, mean NCC at (0, 0) = {mean_zero_lot6:.3f}")
    if not passed:
        print(f"SELF-CHECK FAILED: need |lot shift| <= {SELF_CHECK_TOLERANCE} px and mean NCC at (0, 0) >= {SELF_CHECK_MIN_NCC}; "
              "no evidence file written", file=sys.stderr)
        return 1
    print("SELF-CHECK PASSED")

    # 3. Evidence files.
    npy_path = out_dir / LOT6_NPY
    np.save(npy_path, h_lot6.astype(np.float64))
    if not np.array_equal(load_homography(npy_path), h_lot6):
        print("round trip of the saved homography failed", file=sys.stderr)
        return 1

    def relative(path: Path) -> str:
        for root in (repo_root, data_root):
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                continue
        return path.name

    report = {
        "convention": "H maps RGB -> NIR (src.training.nir_registration); the NIR is resampled with inv(H). "
                      "H_lot6 = H_base @ T(-dx, -dy) with T a translation; the RGB pixel (x, y) is compared with the "
                      "warped-NIR pixel (x - dx, y - dy).",
        "parameters": {"gaussian_blur_ksize": BLUR_KSIZE, "sobel_ksize": 3, "roi_margin_px": margin,
                       "search_px": search, "similarity": "NCC of the edge maps (cv2.TM_CCOEFF_NORMED)"},
        "inputs": {"qc_csv": relative(qc_path), "qc_csv_sha256": sha256_file(qc_path), "n_captures": len(pairs)},
        "base_homography": {"path": relative(base_path), "sha256": homography_sha256(base_path), "matrix": h_base.tolist()},
        "lot6_homography": {"path": relative(npy_path), "sha256": homography_sha256(npy_path), "matrix": h_lot6.tolist()},
        "lot_shift": {"dx": int(dx), "dy": int(dy)},
        "mean_ncc_at_zero_base": _round(mean_zero_base),
        "mean_ncc_at_lot_shift_base": _round(mean_at_shift),
        "mean_ncc_at_zero_lot6": _round(mean_zero_lot6),
        "lot_surface_top5": top_shifts,
        "opposite_direction_check": {"best_mean_ncc": _round(alt_surface.max()),
                                     "best_shift": {"dx": alt_shift[0], "dy": alt_shift[1]}},
        "captures": [
            {"stem": c["stem"], "mango": int(c["mango"]), "face": int(c["face"]),
             "best_dx": int(c["best_dx"]), "best_dy": int(c["best_dy"]),
             "ncc_at_zero_base": _round(c["ncc_at_zero_base"]), "ncc_at_lot_shift": _round(c["ncc_at_lot_shift"]),
             "ncc_at_zero_lot6": _round(c["ncc_at_zero_lot6"]),
             "residual_dx_lot6": int(c["residual_dx_lot6"]), "residual_dy_lot6": int(c["residual_dy_lot6"])}
            for c in per_capture
        ],
        "self_check": {"passed": bool(passed),
                       "criteria": {"max_abs_lot_shift_px": SELF_CHECK_TOLERANCE, "min_mean_ncc_at_zero": SELF_CHECK_MIN_NCC},
                       "lot_shift_after": {"dx": int(shift_after[0]), "dy": int(shift_after[1])},
                       "mean_ncc_at_zero": _round(mean_zero_lot6)},
    }
    (out_dir / LOT6_JSON).write_text(json.dumps(report, indent=2) + "\n")
    with (out_dir / LOT6_CSV).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selection)
    shown = write_overlay(figure_dir / LOT6_FIGURE, pairs, per_capture, h_base, h_lot6, register_nir_to_rgb)
    print(f"wrote {npy_path.name}, {LOT6_JSON}, {LOT6_CSV} to {out_dir}")
    print(f"wrote {LOT6_FIGURE} to {figure_dir} (captures: {', '.join(s.split('_')[-1] for s in shown)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
