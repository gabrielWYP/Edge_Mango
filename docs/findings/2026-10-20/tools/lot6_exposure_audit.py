"""Lot 6 (capture day 2026-10-18): evidence for the RGB overexposure from 16:12 on.

Analysis tooling for docs/findings/2026-10-20, not part of the training pipeline. Run from the repository
root under isolated mode:

    <venv>/bin/python -I docs/findings/2026-10-20/tools/lot6_exposure_audit.py

For every capture with RGB and NIR images it measures, from the incoming files and the QC table:
  * the white reference sheet colour and the share of saturated pixels in the fruit box (QC columns),
  * the RGB JPEG size,
  * two controls: the mean of the NIR frame and of an unlit top-left corner patch (NIR and RGB green channel),
    and the corner noise (standard deviation of the Laplacian of the patch),
and summarises the camera controls read back in the metadata JSON files. Writes lot6_exposure_audit.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Isolated mode (-I) does not add the script directory to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import build_parser, resolve_roots  # noqa: E402

INCOMING_RELATIVE = "data/cache/incoming/2026-10-18"
OUT_NAME = "lot6_exposure_audit.json"
PATCH = 120  # px: top-left corner, outside the lit area of both cameras

GROUPS = {"mangos 1-4": range(1, 5), "mangos 5-8": range(5, 9)}


def laplacian_std(channel: np.ndarray) -> float:
    return float(cv2.Laplacian(channel.astype(np.float32), cv2.CV_32F).std())


def capture_row(incoming: Path, qc: dict) -> dict:
    ts = int(qc["timestamp"])
    rgb_path = incoming / "rgb" / f"mango_rgb_{ts}.jpg"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    nir = cv2.imread(str(incoming / "nir" / f"mango_nir_{ts}.jpg"), cv2.IMREAD_GRAYSCALE)
    green = rgb[..., 1]
    return {
        "timestamp": ts, "local_time": qc["local_time"],
        "mango_id": int(float(qc["mango_id"])) if qc["mango_id"] else None,
        "rgb_jpeg_kb": rgb_path.stat().st_size / 1024,
        "sheet_rgb": [float(qc[c]) for c in ("sheet_R", "sheet_G", "sheet_B")],
        "rgb_sat_pct_bbox": float(qc["rgb_sat_pct_bbox"]),
        "nir_frame_mean": float(nir.mean()),
        "nir_corner_mean": float(nir[:PATCH, :PATCH].mean()),
        "rgb_corner_green_mean": float(green[:PATCH, :PATCH].mean()),
        "rgb_corner_laplacian_std": laplacian_std(green[:PATCH, :PATCH]),
        "nir_corner_laplacian_std": laplacian_std(nir[:PATCH, :PATCH]),
    }


def group_summary(rows: list[dict]) -> dict:
    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    def span(key):
        return [float(min(r[key] for r in rows)), float(max(r[key] for r in rows))]

    keys = ("rgb_jpeg_kb", "rgb_sat_pct_bbox", "nir_frame_mean", "nir_corner_mean",
            "rgb_corner_green_mean", "rgb_corner_laplacian_std", "nir_corner_laplacian_std")
    summary = {"n": len(rows), "sheet_rgb_mean": [float(np.mean([r["sheet_rgb"][i] for r in rows])) for i in range(3)]}
    summary.update({f"{k}_mean": mean(k) for k in keys})
    summary.update({f"{k}_range": span(k) for k in keys})
    return summary


def metadata_readback(incoming: Path) -> dict:
    """Distinct RGB control read-backs and requested gains over all metadata JSON files."""
    readbacks: dict[str, int] = {}
    gains: dict[str, int] = {}
    paths = sorted((incoming / "meta").glob("mango_meta_*.json"))
    for path in paths:
        rgb = json.loads(path.read_text()).get("rgb", {})
        key = json.dumps(rgb.get("controles_leidos"), sort_keys=True)
        readbacks[key] = readbacks.get(key, 0) + 1
        gain = json.dumps(rgb.get("pedido", {}).get("gain"))
        gains[gain] = gains.get(gain, 0) + 1
    return {"n_json": len(paths),
            "rgb_controls_read_back": [{"controls": json.loads(k), "n": n} for k, n in readbacks.items()],
            "rgb_gain_requested": [{"gain": json.loads(k), "n": n} for k, n in gains.items()]}


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--incoming-dir", type=Path, default=None,
                        help=f"incoming capture day (rgb/, nir/, meta/, qc/qc.csv); default <data-root>/{INCOMING_RELATIVE}")
    ns = parser.parse_args()
    _, data_root, out_dir = resolve_roots(ns)
    incoming = (ns.incoming_dir or data_root / INCOMING_RELATIVE).resolve()
    qc_path = incoming / "qc" / "qc.csv"
    if not qc_path.is_file():
        print(f"missing input: {qc_path}", file=sys.stderr)
        return 2
    with qc_path.open(newline="") as handle:
        qc_rows = [r for r in csv.DictReader(handle) if r["has_rgb"] == "True" and r["has_nir"] == "True"]
    rows = sorted((capture_row(incoming, r) for r in qc_rows), key=lambda r: r["timestamp"])

    groups = {name: group_summary([r for r in rows if r["mango_id"] in ids]) for name, ids in GROUPS.items()}
    groups["pre-metadata"] = group_summary([r for r in rows if r["mango_id"] is None])
    report = {"patch_px": PATCH, "n_captures": len(rows), "groups": groups,
              "metadata": metadata_readback(incoming), "captures": rows}
    (out_dir / OUT_NAME).write_text(json.dumps(report, indent=2) + "\n")

    print(f"{len(rows)} captures; corner patch {PATCH}x{PATCH} px")
    print(f"{'group':<14}{'n':>3}{'sheet R,G,B':>16}{'sat bbox %':>12}{'JPEG KB':>9}{'NIR mean':>10}{'NIR corner':>12}"
          f"{'RGB corner G':>14}{'RGB lap std':>13}")
    for name, g in groups.items():
        sheet = ",".join(f"{v:.0f}" for v in g["sheet_rgb_mean"])
        print(f"{name:<14}{g['n']:>3}{sheet:>16}{g['rgb_sat_pct_bbox_mean']:>12.1f}{g['rgb_jpeg_kb_mean']:>9.1f}"
              f"{g['nir_frame_mean_mean']:>10.1f}{g['nir_corner_mean_mean']:>12.1f}"
              f"{g['rgb_corner_green_mean_mean']:>14.1f}{g['rgb_corner_laplacian_std_mean']:>13.2f}")
    print("around the onset (16:09 - 16:20):")
    for r in rows:
        if "16:09" <= r["local_time"] <= "16:20":
            sheet = ",".join(f"{v:.0f}" for v in r["sheet_rgb"])
            print(f"  {r['local_time']} mango {r['mango_id']}  sheet {sheet:<12} sat {r['rgb_sat_pct_bbox']:5.1f} %  "
                  f"JPEG {r['rgb_jpeg_kb']:5.1f} KB  NIR corner {r['nir_corner_mean']:4.1f}  "
                  f"RGB corner G {r['rgb_corner_green_mean']:5.1f}  lap std {r['rgb_corner_laplacian_std']:.2f}")
    print(f"metadata: {report['metadata']['n_json']} JSON files; wrote {OUT_NAME} to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
