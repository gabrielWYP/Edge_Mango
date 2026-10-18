"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

Trims nir_audit.json (written by nir_value_audit.py, per-image rows) to per-lot medians plus the pooled
leave-one-lot-out pixel-classifier AUCs. Reads nir_audit.json from --out-dir and writes nir_audit_summary.json there.
Pure standard library, CPU only.
"""
from __future__ import annotations

import json
import statistics as st

from _paths import resolve_roots

_, _, OUT, _ = resolve_roots(__doc__, repo=False, data=False)
MEDIAN_KEYS = ["ncc_raw", "ncc_reg", "shift", "nir_p50", "nir_p99", "nir_clip_hi", "nir_clip_lo",
               "r2_nir_from_rgb", "corr_nir_r", "auc_R", "auc_G", "auc_B", "auc_NIR"]


def main():
    audit = json.loads((OUT / "nir_audit.json").read_text())
    rows = audit["rows"]
    per_lot = {}
    for lot in sorted({r["lot"] for r in rows}):
        rr = [r for r in rows if r["lot"] == lot]
        entry = {"n_images": len(rr), "n_images_with_pixel_auc": sum(1 for r in rr if "auc_R" in r)}
        for key in MEDIAN_KEYS:
            vals = [r[key] for r in rr if key in r]
            entry[f"median_{key}"] = round(st.median(vals), 4)
        per_lot[lot] = entry
    pooled = audit["pooled_lolo_auc"]
    summary = {
        "note": "Per-lot medians over images; AUCs are max(a, 1-a), i.e. sign-agnostic. pooled_lolo_auc lists the AUC "
                "per held-out lot (lot1..lot5) of a logistic classifier on damage vs healthy fruit pixels.",
        "per_lot_medians": per_lot,
        "pooled_lolo_auc_per_test_lot": pooled,
        "pooled_lolo_auc_mean": {k: round(st.mean(v), 3) for k, v in pooled.items()},
    }
    (OUT / "nir_audit_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
