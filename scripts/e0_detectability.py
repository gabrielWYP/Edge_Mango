#!/usr/bin/env python3
"""E0 of docs/tesis/plan-experimental.md: impact detectability per hour, per channel, without deep learning.

For every face of every captura_v3 jornada given:

1. Crop the fruit from the RGB (largest green-dominant blob, square side 1.1x the box, resized to 256)
   and the NIR registered onto the RGB frame with the lot's RGB->NIR homography.
2. Align each moment's crop onto the face's `sano` crop (ECC, Euclidean, on the blurred RGB luminance).
3. Measure the disk-vs-ring contrast at the crop center for R, G, B, luminance, NIR and NIR/R, and its
   change against `sano` (`src/evaluation/detectability.py` explains why).

Faces hit in the `centro` zone are the positives; the others are controls. Per hour it reports the AUROC of
the change (positives vs controls) with a mango-cluster bootstrap interval, the mean change of both groups,
logistic combinations (RGB, NIR, RGB+NIR) scored out of fold and calibrated by permuting the zone labels,
a pre-impact placebo and the onset hour.

This is the centro-only version of E0: until the damage boxes exist (plan section 3.2) only the centro
zone has a known position in the crop. Moments outside the jornada tolerance or whose alignment did not
converge are excluded and counted.

    python scripts/e0_detectability.py \
        --lote data/cache/incoming/20261024/lote1-20261024 \
               data/cache/incoming/20261024/homografia/lote1_oficial/matriz_homografia_lote1_20261024_centros.npy \
        --out reports/e0-detectability

Needs numpy and opencv only. Repeat `--lote` once per lot.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.evaluation import detectability as det  # noqa: E402

SIDE = 256
CHANNELS = ("R", "G", "B", "lum", "NIR", "NIR/R")
COMBOS = {"RGB": ("R", "G", "B"), "NIR (logistic)": ("NIR",), "RGB+NIR": ("R", "G", "B", "NIR")}
POSITIVE_ZONE = "centro"


def fruit_box(rgb_small: np.ndarray) -> tuple[int, int, int, int] | None:
    """Box of the largest green-dominant blob (the mango) on a downscaled BGR image."""
    b, g, _ = [c.astype(np.int16) for c in cv2.split(rgb_small)]
    mask = cv2.morphologyEx(((g - b > 0.4 * g) & (g > 35)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[k]
    return x, y, x + w, y + h


def crop_pair(root: Path, ts: int, H: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    rgb = cv2.imread(str(root / "news" / "rgb" / f"mango_rgb_{ts}.jpg"))
    nir = cv2.imread(str(root / "news" / "nir" / f"mango_nir_{ts}.jpg"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or nir is None:
        return None
    h, w = rgb.shape[:2]
    box = fruit_box(cv2.resize(rgb, (w // 4, h // 4), interpolation=cv2.INTER_AREA))
    if box is None:
        return None
    x0, y0, x1, y1 = [v * 4 for v in box]
    side = max(x1 - x0, y1 - y0) * 1.1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = SIDE / side
    A = np.array([[s, 0, -(cx - side / 2) * s], [0, s, -(cy - side / 2) * s], [0, 0, 1]])  # RGB frame -> crop
    rgb_c = cv2.warpPerspective(rgb, A, (SIDE, SIDE), flags=cv2.INTER_AREA)
    nir_c = cv2.warpPerspective(nir, A @ np.linalg.inv(H), (SIDE, SIDE), flags=cv2.INTER_LINEAR)
    return rgb_c, nir_c


def align(ref_gray: np.ndarray, mov_gray: np.ndarray) -> np.ndarray | None:
    """2x3 Euclidean warp taking `mov` onto `ref` (ECC on blurred luminance); None if it does not converge."""
    W = np.eye(2, 3, dtype=np.float32)
    a = cv2.GaussianBlur(ref_gray.astype(np.float32), (0, 0), 2)
    b = cv2.GaussianBlur(mov_gray.astype(np.float32), (0, 0), 2)
    try:
        _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_EUCLIDEAN,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), None, 5)
    except cv2.error:
        return None
    return W


def channel_contrasts(rgb_c: np.ndarray, nir_c: np.ndarray, disk: np.ndarray, ring: np.ndarray) -> dict:
    b, g, r = [c.astype(np.float32) for c in cv2.split(rgb_c)]
    nir = nir_c.astype(np.float32)
    planes = {"R": r, "G": g, "B": b, "lum": cv2.cvtColor(rgb_c, cv2.COLOR_BGR2GRAY), "NIR": nir,
              "NIR/R": 100 * nir / (r + 1)}
    return {k: det.contrast(planes[k], disk, ring) for k in CHANNELS}


def measure_face(task: dict) -> list[dict]:
    """One row per moment of one face; the sano row carries the reference contrasts."""
    root, H = Path(task["root"]), np.load(task["H"])
    disk, ring = det.disk_and_ring(SIDE)
    base = {k: task[k] for k in ("lot", "mango", "face", "zone")}
    sano = crop_pair(root, task["moments"]["sano"]["ts"], H)
    if sano is None:
        return [{**base, "moment": "sano", "status": "no_fruit_or_file"}]
    ref_gray = cv2.cvtColor(sano[0], cv2.COLOR_BGR2GRAY)
    rows = [{**base, "moment": "sano", "hours": None, "status": "ok", **channel_contrasts(*sano, disk, ring)}]
    for moment, info in task["moments"].items():
        if moment == "sano":
            continue
        row = {**base, "moment": moment, "hours": info["hours"]}
        if not info["in_tolerance"]:
            rows.append({**row, "status": "out_of_tolerance"})
            continue
        pair = crop_pair(root, info["ts"], H)
        if pair is None:
            rows.append({**row, "status": "no_fruit_or_file"})
            continue
        W = align(ref_gray, cv2.cvtColor(pair[0], cv2.COLOR_BGR2GRAY))
        if W is None:
            rows.append({**row, "status": "not_aligned"})
            continue
        flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        rgb_a = cv2.warpAffine(pair[0], W, (SIDE, SIDE), flags=flags)
        nir_a = cv2.warpAffine(pair[1], W, (SIDE, SIDE), flags=flags)
        rows.append({**row, "status": "ok", **channel_contrasts(rgb_a, nir_a, disk, ring)})
    return rows


def tasks_for_lot(root: Path, H_path: Path) -> list[dict]:
    j = json.loads((root / "jornada.json").read_text())
    tolerance = j["tolerancia_min"]
    tasks = []
    for m in j["mangos"]:
        for c in j["caras"]:
            sano = j["sanas"].get(str(m), {}).get(str(c))
            if not sano:
                continue
            moments = {"sano": {"ts": sano["timestamp"]}}
            for h in j["intervalos_h"]:
                reg = j["rondas"].get(str(m), {}).get(str(h), {}).get(str(c))
                if reg:
                    moments[str(h)] = {"ts": reg["timestamp"], "hours": h,
                                       "in_tolerance": abs(reg["desviacion_min"]) <= tolerance}
            tasks.append({"root": str(root), "H": str(H_path), "lot": j["id"], "mango": m, "face": c,
                          "zone": j["zonas"][str(m)][str(c)], "moments": moments, "hours": j["intervalos_h"]})
    return tasks


def summarize(rows: list[dict], n_boot: int, n_perm: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ok = [r for r in rows if r["status"] == "ok"]
    sano = {(r["lot"], r["mango"], r["face"]): r for r in ok if r["moment"] == "sano"}
    lots = sorted({r["lot"] for r in rows})
    hours = sorted({r["hours"] for r in ok if r["hours"] is not None})
    group_kind = "lot" if len(lots) >= 2 else "mango"
    out = {"lots": lots, "hours": hours, "n_boot": n_boot, "n_perm": n_perm, "seed": seed, "oof_grouping": group_kind,
           "exclusions": {}, "per_hour": {}, "placebo_sano": {}, "onset_hour": {}}
    for r in rows:
        if r["status"] != "ok":
            out["exclusions"][r["status"]] = out["exclusions"].get(r["status"], 0) + 1

    # Placebo: before the impact, centro faces and control faces must not differ.
    faces = list(sano.values())
    lab = np.array([f["zone"] == POSITIVE_ZONE for f in faces])
    mango_ids = np.array([f"{f['lot']}:{f['mango']}" for f in faces])
    strata = np.array([f["lot"] for f in faces])
    for ch in CHANNELS:
        v = np.array([f[ch] for f in faces])
        lo, hi = det.cluster_bootstrap(det.auroc, v, lab, mango_ids, strata, n_boot, rng)
        out["placebo_sano"][ch] = {"auroc": det.auroc(v[lab], v[~lab]), "ci": [lo, hi]}

    for h in hours:
        cur = [r for r in ok if r["hours"] == h and (r["lot"], r["mango"], r["face"]) in sano]
        lab = np.array([r["zone"] == POSITIVE_ZONE for r in cur])
        mango_ids = np.array([f"{r['lot']}:{r['mango']}" for r in cur])
        strata = np.array([r["lot"] for r in cur])
        delta = {ch: np.array([r[ch] - sano[(r["lot"], r["mango"], r["face"])][ch] for r in cur]) for ch in CHANNELS}
        res = {"n_pos": int(lab.sum()), "n_neg": int((~lab).sum()), "channels": {}, "combos": {}}
        for ch in CHANNELS:
            v = delta[ch]
            lo, hi = det.cluster_bootstrap(det.auroc, v, lab, mango_ids, strata, n_boot, rng)
            res["channels"][ch] = {
                "auroc": det.auroc(v[lab], v[~lab]), "ci": [lo, hi],
                "mean_delta_pos": float(v[lab].mean()) if lab.any() else None,
                "mean_delta_neg": float(v[~lab].mean()) if (~lab).any() else None,
            }
            if lab.any() and (~lab).any():
                res["channels"][ch]["did"] = res["channels"][ch]["mean_delta_pos"] - res["channels"][ch]["mean_delta_neg"]
        groups = strata if group_kind == "lot" else mango_ids
        for name, chans in COMBOS.items():
            X = np.column_stack([delta[c] for c in chans])
            scores = det.grouped_oof_scores(X, lab.astype(float), groups)
            observed = det.auroc(scores[lab], scores[~lab])

            def stat(perm, X=X):
                s = det.grouped_oof_scores(X, perm.astype(float), groups)
                return det.auroc(s[perm], s[~perm])

            null_mean, p = det.permutation_test(stat, lab, strata, observed, n_perm, rng)
            res["combos"][name] = {"auroc": observed, "null_mean": null_mean, "p_perm": p}
        out["per_hour"][str(h)] = res

    per_hour = [out["per_hour"][str(h)] for h in hours]
    for ch in CHANNELS:
        out["onset_hour"][ch] = det.onset_hour(hours, [det.excludes_chance(*r["channels"][ch]["ci"]) for r in per_hour])
    for name in COMBOS:
        out["onset_hour"][name] = det.onset_hour(hours, [r["combos"][name]["p_perm"] < 0.05 for r in per_hour])
    return out


def _fmt(a, ci) -> str:
    if a is None:
        return "—"
    lo, hi = ci
    return f"{a:.2f} [{lo:.2f}, {hi:.2f}]" if lo is not None else f"{a:.2f}"


def write_report(summary: dict, rows: list[dict], path: Path) -> None:
    hours = summary["hours"]
    head = "| | " + " | ".join(f"+{h} h" for h in hours) + " |\n|:--|" + "--:|" * len(hours) + "\n"
    n_line = "| n centro / control | " + " | ".join(
        f"{summary['per_hour'][str(h)]['n_pos']} / {summary['per_hour'][str(h)]['n_neg']}" for h in hours) + " |\n"
    auc = head + n_line
    for ch in CHANNELS:
        auc += f"| {ch} | " + " | ".join(_fmt(summary["per_hour"][str(h)]["channels"][ch]["auroc"],
                                              summary["per_hour"][str(h)]["channels"][ch]["ci"]) for h in hours) + " |\n"
    combos = head
    for name in COMBOS:
        cells = []
        for h in hours:
            c = summary["per_hour"][str(h)]["combos"][name]
            cells.append("—" if c["auroc"] is None else f"{c['auroc']:.2f} ({c['null_mean']:.2f}, p {c['p_perm']:.3f})")
        combos += f"| {name} | " + " | ".join(cells) + " |\n"
    did = head
    for ch in CHANNELS:
        cells = []
        for h in hours:
            c = summary["per_hour"][str(h)]["channels"][ch]
            cells.append("—" if "did" not in c else f"{100 * c['did']:+.1f} ({100 * c['mean_delta_neg']:+.1f})")
        did += f"| {ch} | " + " | ".join(cells) + " |\n"
    placebo = "| channel | AUROC at sano [95% CI] |\n|:--|--:|\n" + "".join(
        f"| {ch} | {_fmt(v['auroc'], v['ci'])} |\n" for ch, v in summary["placebo_sano"].items())
    onset = "| channel | onset hour |\n|:--|--:|\n" + "".join(
        f"| {k} | {'never' if v is None else f'+{v:g} h'} |\n" for k, v in summary["onset_hour"].items())
    lots = {}
    for r in rows:
        lots.setdefault(r["lot"], set()).add((r["mango"], r["face"]))
    excl = ", ".join(f"{k}: {v}" for k, v in sorted(summary["exclusions"].items())) or "none"
    text = f"""# E0: impact detectability per hour (centro-only version)

Generated by `scripts/e0_detectability.py`; plan in [`docs/tesis/plan-experimental.md`](../../docs/tesis/plan-experimental.md), E0.

| | |
|:--|:--|
| Lots | {", ".join(f"{k} ({len(v)} faces)" for k, v in lots.items())} |
| Positives | faces hit in the `centro` zone; controls = faces hit in another zone (their center was not hit) |
| Measure | disk (12 % of the crop side) vs ring (18-32 %) contrast, change against the face's own `sano` photo |
| Excluded moments | {excl} |
| Bootstrap | {summary['n_boot']} replicates, mangos resampled within lot, seed {summary['seed']} |
| Logistic combinations | scored out of fold, one fold per {summary['oof_grouping']} |

## AUROC of the change against `sano`, centro vs control [95% CI]

Not sign-agnostic: 0.5 is chance, below 0.5 means the center of hit faces got darker than the controls.

{auc}
## Logistic combinations: AUROC (null mean, permutation p)

Out-of-fold scores pooled into one AUROC sit below 0.5 even without signal (leave-one-out bias), so each value
is compared with its own null: {summary['n_perm']} permutations of the zone labels within lot, which is the
randomization of the capture protocol. Read the AUROC against the null mean, not against 0.5.

{combos}
## Difference of differences, percentage points of contrast

`mean change of centro faces - mean change of controls`; in parentheses the controls' own mean change
(capture drift, should stay near 0).

{did}
## Placebo before the impact

At `sano` the centro and control faces were not hit yet, so the AUROC must cover 0.5.

{placebo}
## Onset

First hour detected and still detected at every later hour: the bootstrap interval excludes 0.5 for a
channel, permutation p < 0.05 for a combination (uncorrected).

{onset}
## Limits

- Centro-only: about one face in five is a positive. With one lot the intervals are wide.
- A control face hit near the center (for example `superior` on a small fruit) can put its bruise inside the
  ring; that biases its contrast and pulls the AUROC toward 0.5 (conservative).
- ECC is a rigid 2D alignment of a fruit placed by hand; residual 3D pose differences add noise.
- Intervals and p-values are descriptive (plan section 8.4) and not corrected for multiplicity (6 channels and
  3 combinations per hour).
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lote", nargs=2, action="append", required=True, metavar=("JORNADA_DIR", "HOMOGRAPHY_NPY"),
                    help="captura_v3 jornada directory and its RGB->NIR homography; repeat per lot")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "e0-detectability")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    tasks = [t for root, H in args.lote for t in tasks_for_lot(Path(root), Path(H))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for face_rows in pool.map(measure_face, tasks) for r in face_rows]

    args.out.mkdir(parents=True, exist_ok=True)
    fields = ["lot", "mango", "face", "zone", "moment", "hours", "status", *CHANNELS]
    with open(args.out / "faces.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, args.n_boot, args.n_perm, args.seed)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write_report(summary, rows, args.out / "README.md")
    print(f"{len(tasks)} faces, {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
