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

This is the centro-only version of E0 (`--mode centro`, the default): without the damage boxes (plan
section 3.2) only the centro zone has a known position in the crop. Moments outside the jornada tolerance
or whose alignment did not converge are excluded and counted.

    python scripts/e0_detectability.py \
        --lote data/cache/incoming/20261024/lote1-20261024 \
               data/cache/incoming/20261024/homografia/lote1_oficial/matriz_homografia_lote1_20261024_centros.npy \
        --out reports/e0-detectability

Needs numpy and opencv only. Repeat `--lote` once per lot.

`--mode boxes` is E0 with the boxes, on a built v4 dataset (`scripts/build_v4_dataset.py`), whose NIR is
already in the RGB frame. Faces hit in every zone count: the impact region is the damage box and the
control is the control zone of plan section 2. For every face and moment:

1. Crop the `sano` photo and photo t around their indexed mango boxes, both at the scale of the `sano` box,
   because the camera does not move, and align t onto `sano` with the same ECC as the centro mode.
2. Bring the box to the `sano` photo, which has none of its own: the box of photo t goes through the crop
   of t and then the inverse of that alignment, which lands it in the `sano` crop. Photo t is warped there
   too, so both photos are measured at the same pixels. Each moment carries its box this way.
3. Mean of R, G, B, NIR and NIR/R in the box and in the control zone, and the difference of differences
   normalized by the sd of the fruit in `sano` (`src/evaluation/detectability.py`).
4. Patches of the box and the control zone: AUROC of a logistic regression trained leave-one-lot-out, per
   hour and per modality (RGB, NIR, RGB+NIR). The negative control is the same AUROC between the control
   zone and a second region that was not hit.

Moments are excluded and counted, in the `status` column, when they have no `sano` photo for the face
(`no_sano`), no box (`no_damage_box`), no zone (`no_zone`), an unreadable photo (`no_fruit_or_file`), no
alignment (`not_aligned`), a box off the fruit (`box_outside_fruit`) or no room for a control
(`no_control_zone`). A dataset with no box at all is an error.

    python scripts/e0_detectability.py --mode boxes --dataset <v4_dataset> --out reports/e0-detectability/boxes

It writes faces.csv, patches.csv, table.csv (median per mango, range between lots), summary.json, figure.png
and README.md, which states the quantities of the plan's reading table and chooses no sentence of it.
Also needs pyyaml (for `LATENT_HOURS`) and matplotlib. `--max-mangos-per-lot` limits a trial run.
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
DEFAULT_OUT = PROJECT_ROOT / "reports" / "e0-detectability"
DEFAULT_N_PERM = 500
SANO = "sano"
BOX_COLUMNS = ("x0", "y0", "x1", "y1")
INDEX_COLUMNS = ("stem", "timestamp", "lot", "mango_id", "cara", "ronda_h", "zona",
                 *(f"mango_{k}" for k in BOX_COLUMNS), *(f"damage_{k}" for k in BOX_COLUMNS))
MODALITY_COLOR = {"RGB": "#2a78d6", "NIR": "#eb6834", "RGB+NIR": "#1baf7a"}
CHANNEL_STYLE = {"R": ("RGB", "-"), "G": ("RGB", "--"), "B": ("RGB", ":"), "NIR": ("NIR", "-"), "NIR/R": ("RGB+NIR", "-")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


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


def box_centre(box) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2, (y0 + y1) / 2


def crop_side(box) -> float:
    """Side of the square crop of a fruit box, in photo pixels."""
    x0, y0, x1, y1 = box
    return max(x1 - x0, y1 - y0) * 1.1


def crop_matrix(cx: float, cy: float, side: float) -> np.ndarray:
    """RGB frame -> `SIDE` x `SIDE` crop of `side` photo pixels centered on (cx, cy)."""
    s = SIDE / side
    return np.array([[s, 0, -(cx - side / 2) * s], [0, s, -(cy - side / 2) * s], [0, 0, 1]])


def crop_pair(root: Path, ts: int, H: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    rgb = cv2.imread(str(root / "news" / "rgb" / f"mango_rgb_{ts}.jpg"))
    nir = cv2.imread(str(root / "news" / "nir" / f"mango_nir_{ts}.jpg"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or nir is None:
        return None
    h, w = rgb.shape[:2]
    box = fruit_box(cv2.resize(rgb, (w // 4, h // 4), interpolation=cv2.INTER_AREA))
    if box is None:
        return None
    box = [v * 4 for v in box]
    A = crop_matrix(*box_centre(box), crop_side(box))
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


def channel_planes(rgb_c: np.ndarray, nir_c: np.ndarray) -> dict[str, np.ndarray]:
    b, g, r = [c.astype(np.float32) for c in cv2.split(rgb_c)]
    nir = nir_c.astype(np.float32)
    return {"R": r, "G": g, "B": b, "lum": cv2.cvtColor(rgb_c, cv2.COLOR_BGR2GRAY), "NIR": nir,
            "NIR/R": 100 * nir / (r + 1)}


def channel_contrasts(rgb_c: np.ndarray, nir_c: np.ndarray, disk: np.ndarray, ring: np.ndarray) -> dict:
    planes = channel_planes(rgb_c, nir_c)
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


def _indexed_box(photo: dict, prefix: str) -> list[float] | None:
    values = [photo[f"{prefix}_{k}"] for k in BOX_COLUMNS]
    return None if "" in values else [float(v) for v in values]


def _photo_info(photo: dict) -> dict:
    return {"stem": photo["stem"], "timestamp": photo["timestamp"], "mango": _indexed_box(photo, "mango"),
            "damage": _indexed_box(photo, "damage")}


def box_tasks(dataset: Path, max_mangos_per_lot: int | None = None) -> list[dict]:
    """One task per face of the dataset's index.csv, with its `sano` photo and its moments."""
    index = Path(dataset) / "index.csv"
    if not index.is_file():
        raise ValueError(f"{index} not found: --dataset must be a built v4 dataset")
    with open(index, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in INDEX_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{index} lacks the columns {missing}")
        photos = list(reader)
    if not any(p["ronda_h"] != SANO and _indexed_box(p, "damage") for p in photos):
        raise ValueError(f"{index}: no photo has a damage box, and E0 with boxes needs a labeled dataset")

    faces: dict[tuple[int, int, int], dict[str, dict]] = {}
    for p in photos:
        key = (int(p["lot"]), int(p["mango_id"]), int(p["cara"]))
        if p["ronda_h"] in faces.setdefault(key, {}):
            raise ValueError(f"{index}: two photos for lot, mango, face {key} at {p['ronda_h']}")
        faces[key][p["ronda_h"]] = p

    allowed = {}
    for lot in {lot for lot, _, _ in faces}:
        mangos = sorted({mango for other, mango, _ in faces if other == lot})
        allowed[lot] = set(mangos if max_mangos_per_lot is None else mangos[:max_mangos_per_lot])

    tasks = []
    for (lot, mango, cara), moments in sorted(faces.items()):
        if mango not in allowed[lot]:
            continue
        zones = sorted({p["zona"] for p in moments.values()} - {""})
        if len(zones) > 1:
            raise ValueError(f"{index}: lot, mango, face {(lot, mango, cara)} has more than one zone: {zones}")
        zone = zones[0] if zones else ""
        if zone and zone != det.CENTRO and zone not in det.ZONE_AXIS:
            raise ValueError(f"{index}: lot, mango, face {(lot, mango, cara)} has the unknown zone {zone!r}")
        tasks.append({
            "root": str(dataset), "lot": lot, "mango": mango, "face": cara, "zone": zone,
            "sano": _photo_info(moments[SANO]) if SANO in moments else None,
            "moments": {h: _photo_info(moments[str(h)]) for h in sorted(int(k) for k in moments if k != SANO)},
        })
    return tasks


def box_photo(root: Path, photo: dict, side: float) -> tuple[np.ndarray, np.ndarray] | None:
    """(rgb, nir) crops of a dataset photo around its mango box, `side` photo pixels wide.

    The blur is the anti-aliasing a reduction by this factor needs (`warpPerspective` does not do it), so
    the local standard deviation of a patch measures the fruit and not aliased sensor noise.
    """
    rgb = cv2.imread(str(root / "rgb" / f"{photo['stem']}.jpg"))
    nir = cv2.imread(str(root / "nir" / f"mango_nir_{photo['timestamp']}.png"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or nir is None or photo["mango"] is None:
        return None
    A = crop_matrix(*box_centre(photo["mango"]), side)
    sigma = 0.5 * side / SIDE
    return tuple(cv2.warpPerspective(cv2.GaussianBlur(image, (0, 0), sigma), A, (SIDE, SIDE), flags=cv2.INTER_LINEAR)
                 for image in (rgb, nir))


def measure_face_boxes(task: dict) -> tuple[list[dict], list[dict]]:
    """Rows (one per moment of one face) and patches, measured in the frame of the `sano` crop."""
    from src.annotation.mango_registration import rgb_mango_mask

    root = Path(task["root"])
    base = {k: task[k] for k in ("lot", "mango", "face", "zone")}
    moments = task["moments"]

    def excluded(hours: int, status: str) -> dict:
        return {**base, "moment": str(hours), "hours": hours, "status": status}

    sano = task["sano"]
    if sano is None:
        return [excluded(h, "no_sano") for h in moments], []
    side = None if sano["mango"] is None else crop_side(sano["mango"])
    pair = None if side is None else box_photo(root, sano, side)
    # A zero NIR pixel had no source in the warp that registered it.
    fruit = None if pair is None else (rgb_mango_mask(pair[0]) > 0) & (pair[1] > 0)
    if fruit is None or not fruit.any():
        return [{**base, "moment": SANO, "hours": None, "status": "no_fruit_or_file"},
                *(excluded(h, "no_sano") for h in moments)], []
    rgb_sano, nir_sano = pair
    planes_sano = channel_planes(rgb_sano, nir_sano)
    sd = {ch: float(planes_sano[ch][fruit].std()) for ch in det.BOX_CHANNELS}
    gray_sano = cv2.cvtColor(rgb_sano, cv2.COLOR_BGR2GRAY)
    rows = [{**base, "moment": SANO, "hours": None, "status": "ok", **{f"{ch}_sd": sd[ch] for ch in det.BOX_CHANNELS}}]
    patches = []
    flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    everywhere = np.full((SIDE, SIDE), 255, np.uint8)

    for hours, photo in moments.items():
        if not task["zone"]:
            rows.append(excluded(hours, "no_zone"))
            continue
        if photo["damage"] is None:
            rows.append(excluded(hours, "no_damage_box"))
            continue
        pair = box_photo(root, photo, side)
        if pair is None:
            rows.append(excluded(hours, "no_fruit_or_file"))
            continue
        W = align(gray_sano, cv2.cvtColor(pair[0], cv2.COLOR_BGR2GRAY))
        if W is None:
            rows.append(excluded(hours, "not_aligned"))
            continue
        rgb_t = cv2.warpAffine(pair[0], W, (SIDE, SIDE), flags=flags)
        nir_t = cv2.warpAffine(pair[1], W, (SIDE, SIDE), flags=flags)
        # `align` takes sano onto t, so the box of t reaches the sano crop through the inverse warp.
        box = det.map_box(det.map_box(photo["damage"], crop_matrix(*box_centre(photo["mango"]), side)), det.invert_warp(W))
        covered = cv2.warpAffine(everywhere, W, (SIDE, SIDE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST) > 0
        valid = fruit & covered & (nir_t > 0)
        if det.fruit_share(valid, box) == 0:
            rows.append(excluded(hours, "box_outside_fruit"))
            continue
        regions = det.control_regions(task["zone"], box, valid)
        if not regions:
            rows.append(excluded(hours, "no_control_zone"))
            continue
        control_side, control = regions[0]
        second_side, second = regions[1] if len(regions) > 1 else ("", None)
        planes_t = channel_planes(rgb_t, nir_t)
        gray_t = cv2.cvtColor(rgb_t, cv2.COLOR_BGR2GRAY)
        row = {
            **base, "moment": str(hours), "hours": hours, "status": "ok",
            "reg_corr": float(np.corrcoef(gray_sano[valid], gray_t[valid])[0, 1]),
            "control_side": control_side, "control2_side": second_side,
            **{f"box_{k}": float(v) for k, v in zip(BOX_COLUMNS, box)},
            **{f"control_{k}": float(v) for k, v in zip(BOX_COLUMNS, control)},
        }
        if second is not None:
            row |= {f"control2_{k}": float(v) for k, v in zip(BOX_COLUMNS, second)}
        for ch in det.BOX_CHANNELS:
            impact_t = det.region_mean(planes_t[ch], box, valid)
            impact_sano = det.region_mean(planes_sano[ch], box, valid)
            control_t = det.region_mean(planes_t[ch], control, valid)
            control_sano = det.region_mean(planes_sano[ch], control, valid)
            row |= {f"{ch}_impact_t": impact_t, f"{ch}_impact_sano": impact_sano, f"{ch}_control_t": control_t,
                    f"{ch}_control_sano": control_sano, f"{ch}_sd": sd[ch],
                    f"{ch}_delta": det.difference_in_differences(impact_t, impact_sano, control_t, control_sano, sd[ch])}
        rows.append(row)
        identity = {"lot": task["lot"], "mango": task["mango"], "face": task["face"], "hours": hours}
        for region, region_box in (("impact", box), ("control", control), ("control2", second)):
            if region_box is None:
                continue
            features = det.patch_features(planes_t, planes_sano, region_box, valid, sd)
            patches += [{**identity, "region": region, **p} for p in features]
    return rows, patches


def _fmt_signed(a, ci) -> str:
    if a is None:
        return "—"
    lo, hi = ci
    return f"{a:+.2f} [{lo:+.2f}, {hi:+.2f}]" if lo is not None else f"{a:+.2f}"


def _median_cell(entry: dict, signed: bool = False) -> str:
    if entry["median_per_mango"] is None:
        return "—"
    spec = "+.2f" if signed else ".2f"
    low, high = entry["lot_range"]
    return f"{entry['median_per_mango']:{spec}} [{low:{spec}}, {high:{spec}}]"


def write_boxes_report(summary: dict, path: Path, figure_name: str) -> None:
    hours, per_hour = summary["hours"], summary["per_hour"]
    latent = set(summary["latent_hours"])
    head = ("| | " + " | ".join(f"+{h} h{' (latent)' if h in latent else ''}" for h in hours)
            + " |\n|:--|" + "--:|" * len(hours) + "\n")

    def table(rows: list[tuple[str, list[str]]]) -> str:
        return head + "".join(f"| {name} | " + " | ".join(cells) + " |\n" for name, cells in rows)

    def curves(metric: str) -> str:
        rows = [("n patches (one class / the other)", [
            f"{per_hour[str(h)][metric]['RGB']['n_pos']} / {per_hour[str(h)][metric]['RGB']['n_neg']}" for h in hours])]
        rows += [(m, [_fmt(per_hour[str(h)][metric][m]["auroc"], per_hour[str(h)][metric][m]["ci"]) for h in hours])
                 for m in det.MODALITIES]
        return table(rows)

    def gap_cell(h: int, name: str) -> str:
        gap = per_hour[str(h)]["gaps"][name]
        return _fmt_signed(gap["gap"], gap["ci"])

    def lots_cell(h: int, name: str) -> str:
        gap = per_hour[str(h)]["gaps"][name]
        return f"{gap['lots_positive']} of {gap['n_lots']}"

    gaps = table([
        ("NIR - RGB", [gap_cell(h, "NIR-RGB") for h in hours]),
        ("lots with NIR > RGB", [lots_cell(h, "NIR-RGB") for h in hours]),
        ("RGB - negative control", [gap_cell(h, "RGB-negative_control") for h in hours]),
        ("lots with RGB > negative control", [lots_cell(h, "RGB-negative_control") for h in hours]),
    ])
    deltas = table([(ch, [_fmt_signed(per_hour[str(h)]["delta"][ch]["mean"], per_hour[str(h)]["delta"][ch]["ci"]) for h in hours])
                    for ch in det.BOX_CHANNELS])
    medians = table(
        [(f"delta {ch}", [_median_cell(per_hour[str(h)]["delta"][ch], signed=True) for h in hours]) for ch in det.BOX_CHANNELS]
        + [(f"AUROC {m}", [_median_cell(per_hour[str(h)]["auroc"][m]) for h in hours]) for m in det.MODALITIES]
        + [(f"negative control {m}", [_median_cell(per_hour[str(h)]["negative_control"][m]) for h in hours]) for m in det.MODALITIES]
    )
    lots = summary["lots"]
    lot_head = "| quantity | hour | " + " | ".join(f"lot {lot}" for lot in lots) + " |\n|:--|:--|" + "--:|" * len(lots) + "\n"
    lot_rows = ""
    for name, shown in (("NIR-RGB", "NIR - RGB"), ("RGB-negative_control", "RGB - negative control")):
        for h in hours:
            if h in latent:
                by_lot = per_hour[str(h)]["gaps"][name]["per_lot"]
                lot_rows += f"| {shown} | +{h} h | " + " | ".join(
                    f"{by_lot[lot]:+.2f}" if lot in by_lot else "—" for lot in lots) + " |\n"
    excl = ", ".join(f"{k}: {v}" for k, v in summary["exclusions"].items()) or "none"
    onset = "| modality | onset hour |\n|:--|--:|\n" + "".join(
        f"| {k} | {'never' if v is None else f'+{v:g} h'} |\n" for k, v in summary["onset_hour"].items())
    grouping = sorted({per_hour[str(h)]["auroc"]["RGB"]["oof_grouping"] for h in hours})
    away_from_chance = "; ".join(
        f"{m}: " + (", ".join(f"+{h} h" for h in hours if per_hour[str(h)]["negative_control"][m]["covers_chance"] is False) or "none")
        for m in det.MODALITIES)
    reg = summary["registration"]
    reg_text = "—" if reg is None else (f"correlation inside the fruit after alignment: median {reg['corr_median']:.2f}, "
                                        f"5th percentile {reg['corr_p05']:.2f}, minimum {reg['corr_min']:.2f}")
    text = f"""# E0: impact detectability per hour (boxes version)

Generated by `scripts/e0_detectability.py --mode boxes`; plan in [`docs/tesis/plan-experimental.md`](../../../docs/tesis/plan-experimental.md), E0.
This report states what the plan's reading table compares and chooses none of its sentences.

| | |
|:--|:--|
| Lots | {", ".join(str(lot) for lot in lots)} |
| Faces, moments measured | {summary['n_faces']}, {summary['n_moments']} |
| Excluded moments | {excl} |
| Latent hours (`LATENT_HOURS`) | {", ".join(str(h) for h in summary['latent_hours'])} |
| Registration | ECC against the face's `sano` crop; {reg_text} |
| Logistic models | one fold per {" / ".join(grouping)}; a patch has the mean and local sd of each channel, as a change against `sano` |
| Bootstrap | {summary['n_boot']} replicates, mangos resampled within lot, seed {summary['seed']} |

![Delta_t and AUROC against hours, per modality, faceted by lot]({figure_name})

## Detectability per patch: AUROC, impact vs control [95% CI]

{curves("auroc")}
## Negative control: AUROC, control zone vs another region not hit [95% CI]

The plan expects about 0.5 at every hour. Moments without room for a second region: {summary['no_second_control']}.

{curves("negative_control")}
Hours whose interval excludes 0.5: {away_from_chance}.

## What the reading table compares

Differences of AUROC at the same hour [95% CI], with the same mangos resampled on both sides, and how many
lots the sign holds in. Hours marked latent come from `LATENT_HOURS`.

{gaps}
Per lot, in the latent hours:

{lot_head}{lot_rows}
## Delta_t, in sd of the fruit in `sano` [95% CI]

`[impact_t - impact_sano] - [control_t - control_sano]`. Negative means the box got darker than the control.

{deltas}
## Median per mango [range between lots]

Faces are averaged per mango first. The range goes from the lowest to the highest median of a lot. The AUROC of
a mango is that of its own patches, with the scores of the fold that held its lot out. Also in `table.csv`.

{medians}
## Onset

First hour whose AUROC interval stays above 0.5 from there on (uncorrected).

{onset}
## Limits

- The box of a moment reaches `sano` through a rigid ECC alignment of a fruit placed by hand; the plan describes
  SIFT + RANSAC, and residual 3D pose differences add noise to both regions.
- Most boxes of the latent hours were drawn by hand on the NIR view (plan 3.2), so the region measured here
  was chosen looking at that modality, which can favor the NIR. They can also be larger than the real bruise,
  which dilutes the mean of the box in early hours.
- The NIR comes as the dataset holds it, so any residual of its RGB-NIR registration moves the bruise inside the
  NIR crop while the box stays put.
- Intervals are descriptive (plan section 8.4): they resample mangos inside each lot, not lots, and are not
  corrected for multiplicity. The unit of inference is the lot, hence the per-lot values.
"""
    path.write_text(text, encoding="utf-8")


def write_boxes_figure(summary: dict, path: Path) -> None:
    """Delta_t and AUROC against hours, one panel per lot; one color per modality, on one axis per row."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hours, lots, per_hour = summary["hours"], summary["lots"], summary["per_hour"]
    fig, axes = plt.subplots(2, len(lots), figsize=(max(2.9 * len(lots), 6.5), 5.6), sharex=True, sharey="row", squeeze=False)

    def series(get) -> list[float]:
        values = [get(per_hour[str(h)]) for h in hours]
        return [np.nan if v is None else v for v in values]

    for column, lot in enumerate(lots):
        top, bottom = axes[0, column], axes[1, column]
        top.axhline(0, color=MUTED, lw=0.8)
        bottom.axhline(0.5, color=MUTED, lw=0.8)
        for ch, (modality, style) in CHANNEL_STYLE.items():
            top.plot(hours, series(lambda hour, ch=ch: hour["delta"][ch]["per_lot"].get(lot)), linestyle=style, marker="o", ms=4,
                     lw=1.2, color=MODALITY_COLOR[modality], label=ch)
        for modality, color in MODALITY_COLOR.items():
            bottom.plot(hours, series(lambda hour, m=modality: hour["auroc"][m]["per_lot"].get(lot)), marker="o", ms=4, lw=1.2,
                        color=color, label=modality)
        for modality, color in MODALITY_COLOR.items():
            bottom.plot(hours, series(lambda hour, m=modality: hour["negative_control"][m]["per_lot"].get(lot)), linestyle="--",
                        marker="s", ms=3, lw=0.8, color=color, label=f"{modality}, control vs control")
        top.set_title(f"Lot {lot}", color=INK, fontsize=10)
        bottom.set_xlabel("Hours after impact", color=MUTED, fontsize=9)
        bottom.set_xticks(hours)
        for ax in (top, bottom):
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            for name in ("top", "right", "left"):
                ax.spines[name].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
    axes[0, 0].set_ylabel("Delta_t (sd of the fruit in sano)", color=MUTED, fontsize=9)
    axes[1, 0].set_ylabel("AUROC (lot held out)", color=MUTED, fontsize=9)
    for row in (0, 1):
        axes[row, -1].legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_boxes(args: argparse.Namespace) -> int:
    # Imported here so the centro mode keeps needing only numpy and opencv.
    from src.data_pipeline.v4_selection import LATENT_HOURS

    try:
        tasks = box_tasks(args.dataset, args.max_mangos_per_lot)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            measured = list(pool.map(measure_face_boxes, tasks))
        rows = [r for face_rows, _ in measured for r in face_rows]
        patches = [p for _, face_patches in measured for p in face_patches]
        summary = det.summarize_boxes(rows, patches, args.n_boot, args.seed, LATENT_HOURS)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    per_channel = [f"{ch}_{k}" for ch in det.BOX_CHANNELS
                   for k in ("impact_t", "impact_sano", "control_t", "control_sano", "sd", "delta")]
    geometry = [f"{name}_{k}" for name in ("box", "control", "control2") for k in BOX_COLUMNS]
    _write_rows(args.out / "faces.csv",
                ["lot", "mango", "face", "zone", "moment", "hours", "status", "reg_corr", "control_side", "control2_side",
                 *geometry, *per_channel], rows)
    features = [f"{ch}_{feature}" for ch in det.PATCH_CHANNELS for feature in det.PATCH_FEATURES]
    _write_rows(args.out / "patches.csv", ["lot", "mango", "face", "hours", "region", *features], patches)
    _write_rows(args.out / "table.csv",
                ["metric", "name", "hours", "median_per_mango", "lot_min", "lot_max", "n_mangos", "n_lots"],
                det.summary_table(summary))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write_boxes_figure(summary, args.out / "figure.png")
    write_boxes_report(summary, args.out / "README.md", "figure.png")
    print(f"{len(tasks)} faces, {len(rows)} rows, {len(patches)} patches -> {args.out}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("centro", "boxes"), default="centro",
                    help="centro: disk-vs-ring contrast on captura_v3 jornadas (default); boxes: damage box vs control "
                         "zone on a built v4 dataset")
    ap.add_argument("--lote", nargs=2, action="append", metavar=("JORNADA_DIR", "HOMOGRAPHY_NPY"),
                    help="centro mode, required: captura_v3 jornada directory and its RGB->NIR homography; repeat per lot")
    ap.add_argument("--dataset", type=Path, help="boxes mode, required: built v4 dataset, with index.csv, rgb/ and nir/")
    ap.add_argument("--max-mangos-per-lot", type=int, help="boxes mode: use only the first N mangos of each lot, for a trial run")
    ap.add_argument("--out", type=Path, help=f"output folder (default {DEFAULT_OUT.relative_to(PROJECT_ROOT)}, "
                                             f"and its boxes/ subfolder in boxes mode)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, help=f"centro mode: label permutations (default {DEFAULT_N_PERM})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)

    if args.mode == "centro":
        if not args.lote:
            ap.error("the following arguments are required: --lote")
        for option, value in (("--dataset", args.dataset), ("--max-mangos-per-lot", args.max_mangos_per_lot)):
            if value is not None:
                ap.error(f"{option} applies only to --mode boxes")
        args.n_perm = DEFAULT_N_PERM if args.n_perm is None else args.n_perm
        args.out = DEFAULT_OUT if args.out is None else args.out
    else:
        if args.dataset is None:
            ap.error("--mode boxes requires --dataset")
        for option, value in (("--lote", args.lote), ("--n-perm", args.n_perm)):
            if value is not None:
                ap.error(f"{option} applies only to --mode centro")
        if args.max_mangos_per_lot is not None and args.max_mangos_per_lot < 1:
            ap.error("--max-mangos-per-lot must be at least 1")
        args.out = DEFAULT_OUT / "boxes" if args.out is None else args.out
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "boxes":
        return run_boxes(args)

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
