"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

Where do the false positives sit? For every checkpoint, the dispersion of the FP box centres (in pixels of the 640 px
letterboxed crop, whose centre is (320, 320)): median centre, sd, and the share of FPs within 10 / 20 / 40 px of the median
centre, plus the median FP size. Reads fp_records.json (written by fp_analyze.py) from --out-dir and writes
fp_positions.json there. Pure numpy, CPU only.
"""
from __future__ import annotations

import json

import numpy as np
from _paths import resolve_roots

_, _, OUT, _ = resolve_roots(__doc__, repo=False, data=False)
IMG = 640


def main():
    records = json.loads((OUT / "fp_records.json").read_text())
    result = {}
    for tag in sorted({r["tag"] for r in records}):
        fps = [r for r in records if r["tag"] == tag and r["cat"] != "TP"]
        cx = np.array([r["cx"] * IMG for r in fps])
        cy = np.array([r["cy"] * IMG for r in fps])
        mx, my = float(np.median(cx)), float(np.median(cy))
        dist = np.hypot(cx - mx, cy - my)
        result[tag] = {
            "n_fp": len(fps),
            "median_center_px": [round(mx), round(my)],
            "sd_center_px": [round(float(cx.std()), 1), round(float(cy.std()), 1)],
            "share_within_px_of_median_center": {str(d): round(float((dist <= d).mean()), 3) for d in (10, 20, 40)},
            "median_size_px": [round(float(np.median([r["w"] * IMG for r in fps]))),
                               round(float(np.median([r["h"] * IMG for r in fps])))],
            "score_p5_p50_p95": [round(float(np.percentile([r["score"] for r in fps], p)), 3) for p in (5, 50, 95)],
        }
        print(tag, result[tag])
    (OUT / "fp_positions.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
