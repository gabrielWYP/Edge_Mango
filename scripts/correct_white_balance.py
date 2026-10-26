#!/usr/bin/env python3
"""Correct captura_v3 RGB photos that were shot with the camera's automatic white balance.

During lot 1's +4 h round the Arducam USB camera reset (EMI) and fell back to automatic white
balance while the metadata still said manual 5300 K, so the metadata cannot find the bad photos.
The affected set is therefore selected explicitly (--faces) and then validated from pixels:

1. Measure the linear-light border chroma (R/G, B/G) of every RGB photo of the jornada, using the
   same border frame as the capture QC.
2. Split the R/G distribution at its largest gap; the photos above it must be exactly the
   selected ones, otherwise the run aborts with the diff.
3. Reference = per-channel median of the border means of the non-selected photos of the same round.
4. Per selected photo, von Kries gains = reference / own border means, applied in linear light at
   full resolution. Originals are never touched; outputs go to --out-subdir with the same file name.
5. Non-circular check (the border matches the reference by construction): the CIELAB colour of the
   mango is compared with the same mango+face one round earlier, before and after the correction.

    python scripts/correct_white_balance.py --data-root data/cache/captura_v3 \
        --jornada lote1-20261024 --ronda 4 --faces 11:4,12-25 --expected-count 57

Needs numpy, opencv and Pillow only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from capture.calidad import cargar_reducido  # noqa: E402
from src.data_pipeline import white_balance as wb  # noqa: E402

FINDINGS_DIR = PROJECT_ROOT / "docs" / "findings" / "2026-10-26-lot1-auto-wb"
N_FACES = 4
PROTECTED_DIRS = ("rgb", "nir", "meta")
MIN_GAP_RATIO = 2.0
OFF_ROUND_TOL = 0.25
REFERENCE_DEVIANT_TOL = 0.10
FIGURE_MAX_BYTES = 300_000
FIGURE_ROWS = 6
TILE_W, TILE_H = 320, 240
METHOD = (
    "Diagonal von Kries gains in linear sRGB: gain_c = reference_c / observed_c, where the means are "
    "taken over capture.calidad.region_borde(frac=0.10) in linear light. Gains are measured on 1/4-scale "
    "decodes and applied at full resolution through per-channel LUTs, then re-encoded as 4:4:4 sRGB JPEG "
    "(4:2:0 bleeds chroma across the dark tray edge inside the border frame)."
)
VALIDATION_FORMULA = (
    "CIE76 (Euclidean distance in CIELAB, D65) between the median Lab colour of the mango mask in the "
    "+4 h photo and in the same mango+face at the comparison round"
)


WhiteBalanceError = wb.WhiteBalanceError


def _expand(token: str, lo: int, hi: int, what: str) -> range:
    parts = token.split("-")
    if not 1 <= len(parts) <= 2 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"bad {what} '{token}'")
    a, b = int(parts[0]), int(parts[-1])
    if a > b or a < lo or b > hi:
        raise ValueError(f"{what} '{token}' must be an ascending range within {lo}..{hi}")
    return range(a, b + 1)


def parse_face_spec(spec: str) -> set[tuple[int, int]]:
    """'11:4,12-25' -> {(mango, cara)}: mango 11 face 4 plus mangos 12..25 on all faces."""
    slots: set[tuple[int, int]] = set()
    for item in spec.split(","):
        mangos, _, faces = item.strip().partition(":")
        if ":" in item and not faces.strip():
            raise ValueError(f"empty face list in '{item}'")
        face_range = _expand(faces.strip(), 1, N_FACES, "face") if faces else range(1, N_FACES + 1)
        for mango in _expand(mangos.strip(), 1, 10_000, "mango"):
            slots.update((mango, cara) for cara in face_range)
    return slots


def largest_gap_threshold(values, min_ratio: float = MIN_GAP_RATIO):
    """Geometric midpoint of the largest gap in log-space; refuses data without a clear gap."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    if len(v) < 2 or v[0] <= 0:
        raise WhiteBalanceError("need at least two positive R/G values to find a gap")
    gaps = np.diff(np.log(v))
    i = int(np.argmax(gaps))
    if math.exp(gaps[i]) < min_ratio:
        raise WhiteBalanceError(
            f"R/G is not bimodal: largest gap {v[i]:.3f} -> {v[i + 1]:.3f} is under x{min_ratio:g}")
    return float(math.sqrt(v[i] * v[i + 1])), float(v[i]), float(v[i + 1])


def refuse_protected(path: Path, data_root: Path, flag: str, shown: str | None = None) -> Path:
    """Resolve `path` (following `..` and symlinks) and refuse anything at or below news/{rgb,nir,meta}."""
    resolved = path.resolve()
    root = data_root.resolve()
    for name in PROTECTED_DIRS:
        protected = (root / "news" / name).resolve()
        if resolved == protected or protected in resolved.parents:
            raise WhiteBalanceError(
                f"{flag} '{shown or path}' resolves to/inside the original news/{name}; refusing to write there")
    return resolved


def resolve_out_dir(data_root: Path, out_subdir: str) -> Path:
    root = data_root.resolve()
    out = (root / out_subdir).resolve()
    if not out.is_relative_to(root) or out == root:
        raise WhiteBalanceError(f"--out-subdir '{out_subdir}' must resolve to a folder inside the data root")
    return refuse_protected(out, data_root, "--out-subdir", out_subdir)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jornada(data_root: Path, jornada: str) -> list[dict]:
    photos, missing = [], []
    for meta_path in sorted((data_root / "news" / "meta").glob("mango_meta_*.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("jornada", {}).get("id") != jornada:
            continue
        ts = int(meta["timestamp"])
        rgb = data_root / "news" / "rgb" / f"mango_rgb_{ts}.jpg"
        if not rgb.is_file():
            missing.append(str(rgb.name))
            continue
        photos.append({"ts": ts, "mango": int(meta["mango_id"]), "cara": int(meta["cara"]),
                       "ronda": meta.get("ronda_h"), "path": rgb})
    if missing:
        raise WhiteBalanceError(f"{len(missing)} RGB files listed in the metadata are missing, e.g. {missing[:3]}")
    if not photos:
        raise WhiteBalanceError(f"no photos found for jornada '{jornada}' under {data_root / 'news' / 'meta'}")
    return photos


def tag(p: dict) -> str:
    return f"{p['mango']}-{p['cara']}"


def median_lab(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.median(wb.srgb_to_lab(img[mask]), axis=0)


def mango_color_diff(img, mask, ref_img, ref_mask) -> tuple[float, list[float]]:
    """CIE76 distance and signed (dL, da, db) between the median Lab colours of two mango masks."""
    lab, ref_lab = median_lab(img, mask), median_lab(ref_img, ref_mask)
    return wb.delta_e76(lab, ref_lab), [float(v) for v in lab - ref_lab]


def print_distribution(photos: list[dict], chroma: dict[int, tuple[float, float]], threshold: float) -> None:
    rounds = defaultdict(list)
    for p in photos:
        rounds[p["ronda"]].append(chroma[p["ts"]])
    print("\nBorder chroma (linear light) per round")
    print("  round     n   R/G min  med   max  |  B/G min  med   max")
    for r in sorted(rounds, key=lambda x: (x is None, x)):
        a = np.array(rounds[r])
        label = "sano" if r is None else f"+{r} h"
        print(f"  {label:<7} {len(a):>4}   {a[:, 0].min():.3f} {np.median(a[:, 0]):.3f} {a[:, 0].max():.3f}"
              f"  |  {a[:, 1].min():.3f} {np.median(a[:, 1]):.3f} {a[:, 1].max():.3f}")
    rg = np.array([chroma[p["ts"]][0] for p in photos])
    edges = np.exp(np.linspace(np.log(rg.min() * 0.99), np.log(rg.max() * 1.01), 13))
    counts, _ = np.histogram(rg, bins=edges)
    print(f"\nR/G histogram, all {len(rg)} photos (log bins); threshold {threshold:.3f}")
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        mark = "  <- threshold" if lo <= threshold < hi else ""
        print(f"  {lo:6.3f} - {hi:6.3f} {c:>5} {'#' * int(math.ceil(60 * c / counts.max())) if c else ''}{mark}")


def off_colour_in_other_rounds(photos, chroma, selected_ts) -> list[tuple[dict, str]]:
    by_round = defaultdict(list)
    for p in photos:
        by_round[p["ronda"]].append(p)
    flagged = []
    for group in by_round.values():
        members = [p for p in group if p["ts"] not in selected_ts]
        if not members:
            continue
        med = np.median([chroma[p["ts"]] for p in members], axis=0)
        for p in members:
            dev = np.abs(np.log(np.array(chroma[p["ts"]]) / med))
            if dev.max() > math.log(1 + OFF_ROUND_TOL):
                flagged.append((p, f"R/G {chroma[p['ts']][0]:.3f} B/G {chroma[p['ts']][1]:.3f} "
                                   f"vs round median {med[0]:.3f} / {med[1]:.3f}"))
    return flagged


def label_tile(img: np.ndarray, text: str) -> np.ndarray:
    tile = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA).copy()
    cv2.rectangle(tile, (0, 0), (TILE_W, 20), (0, 0, 0), -1)
    cv2.putText(tile, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def save_jpeg_under(path: Path, img: np.ndarray, max_bytes: int = FIGURE_MAX_BYTES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for quality in range(88, 40, -6):
        Image.fromarray(img).save(path, "JPEG", quality=quality)
        if path.stat().st_size < max_bytes:
            return
    raise WhiteBalanceError(f"could not keep {path.name} under {max_bytes} bytes")


def make_figure(figure_dir: Path, rows: list[dict], good_row: dict | None) -> Path:
    blank = np.full((TILE_H, TILE_W, 3), 40, dtype=np.uint8)
    tiles = []
    for r in rows:
        tiles.append([label_tile(r["before"], f"{r['tag']} +4h as shot  R/G {r['rg_before']:.2f}"),
                      label_tile(r["after"], f"{r['tag']} corrected  R/G {r['rg_after']:.2f}"),
                      label_tile(r["ref"], f"{r['tag']} +2h (good WB)")])
    if good_row is not None:
        note = label_tile(blank, "no correction needed")
        tiles.append([label_tile(good_row["before"], f"{good_row['tag']} +4h as shot (good)"), note,
                      label_tile(good_row["ref"], f"{good_row['tag']} +2h")])
    grid = np.concatenate([np.concatenate(t, axis=1) for t in tiles], axis=0)
    out = figure_dir / "before_after.jpg"
    save_jpeg_under(out, grid)
    return out


def outline(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    drawn = img.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(drawn, contours, -1, (255, 0, 255), 3)
    return drawn


def make_mask_figure(figure_dir: Path, rows: list[dict], border_frac: float) -> Path:
    """The mango mask outlined on the corrected photo and on its +2 h counterpart (visual check)."""
    tiles = []
    for r in rows:
        tiles.append([
            label_tile(outline(r["after"], wb.mango_mask(r["after"], border_frac)), f"{r['tag']} corrected + mask"),
            label_tile(outline(r["ref"], wb.mango_mask(r["ref"], border_frac)), f"{r['tag']} +2h + mask")])
    out = figure_dir / "mango_masks.jpg"
    save_jpeg_under(out, np.concatenate([np.concatenate(t, axis=1) for t in tiles], axis=0))
    return out


def pick_rows(entries: list[dict], n: int) -> list[dict]:
    ordered = sorted(entries, key=lambda e: e["rg_before"])
    idx = sorted({int(round(i)) for i in np.linspace(0, len(ordered) - 1, min(n, len(ordered)))})
    return [ordered[i] for i in idx]


def describe(values, digits: int = 4) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"min": round(float(a.min()), digits), "median": round(float(np.median(a)), digits),
            "max": round(float(a.max()), digits)}


def select_photos(photos: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """The photos of the round and the selected subset; aborts unless the subset has the expected size."""
    try:
        slots = parse_face_spec(args.faces)
    except ValueError as exc:
        raise WhiteBalanceError(f"--faces: {exc}") from exc
    in_round = [p for p in photos if p["ronda"] == args.ronda]
    keyed = defaultdict(list)
    for p in in_round:
        keyed[(p["mango"], p["cara"])].append(p)
    dupes = {k: [p["ts"] for p in v] for k, v in keyed.items() if k in slots and len(v) > 1}
    if dupes:
        raise WhiteBalanceError(f"several photos for the same selected face (ambiguous): {dupes}")
    selected = [keyed[k][0] for k in sorted(slots) if k in keyed]
    if len(selected) != args.expected_count:
        raise WhiteBalanceError(
            f"selected {len(selected)} photos from --faces {args.faces} at +{args.ronda} h, expected "
            f"{args.expected_count}; faces without a photo: {sorted(slots - set(keyed))}")
    return in_round, selected


def measure_photos(photos, keep_small_for, border_frac):
    """Linear border means of every photo (1/4-scale decode); keeps the decodes of the selected ones."""
    means, smalls = {}, {}
    for p in photos:
        small = cargar_reducido(p["path"])
        means[p["ts"]] = wb.border_means_linear(small, border_frac)
        if not np.all(means[p["ts"]] > 0):
            raise WhiteBalanceError(
                f"{p['path'].name}: a border channel mean is zero (black frame or dead channel), "
                f"cannot measure chroma: {means[p['ts']].tolist()}")
        if p["ts"] in keep_small_for:
            smalls[p["ts"]] = small
    return means, smalls


def require_outliers_equal_selection(photos, selected, chroma, threshold) -> None:
    selected_ts = {p["ts"] for p in selected}
    outliers = [p for p in photos if chroma[p["ts"]][0] >= threshold]

    def describe_photo(p):
        return f"{tag(p)} (+{p['ronda']} h) ts {p['ts']} R/G {chroma[p['ts']][0]:.3f}"

    stray = [p for p in outliers if p["ts"] not in selected_ts]
    wrong = [p for p in selected if chroma[p["ts"]][0] < threshold]
    if stray or wrong:
        lines = [f"outlier set (R/G >= {threshold:.3f}) differs from the selected set:"]
        lines += [f"  outliers not selected: {describe_photo(p)}" for p in stray]
        lines += [f"  selected but not outliers: {describe_photo(p)}" for p in wrong]
        raise WhiteBalanceError("\n".join(lines))


def build_reference(reference_photos, means, chroma):
    """Per-channel median border means of the reference photos, plus the photos that sit far from it."""
    if not reference_photos:
        raise WhiteBalanceError("no non-selected photos in the round to build the reference from")
    ref_means = np.median([means[p["ts"]] for p in reference_photos], axis=0)
    ref_rg, ref_bg = wb.chroma(ref_means)
    deviants = [p for p in reference_photos
                if max(abs(chroma[p["ts"]][0] / ref_rg - 1), abs(chroma[p["ts"]][1] / ref_bg - 1))
                > REFERENCE_DEVIANT_TOL]
    return ref_means, deviants


def correct_photo(p, small, gains, out_path, args):
    """Write the corrected photo at full resolution; returns (1/4-scale corrected image, clipped fraction)."""
    if args.dry_run:
        return wb.apply_gains(small, gains)[0], None
    with Image.open(p["path"]) as im:
        full = np.asarray(im.convert("RGB"))
    corrected, clipped = wb.apply_gains(full, gains)
    Image.fromarray(corrected).save(out_path, "JPEG", quality=args.quality, subsampling=0)
    return cargar_reducido(out_path), clipped


def validate_photo(before_small, after_small, ref_photo, border_frac):
    """Mango colour against the same face one round earlier; (validation dict, ref decode) or (None, None)."""
    if ref_photo is None:
        return None, None
    mask = wb.mango_mask(after_small, border_frac)
    ref_small = cargar_reducido(ref_photo["path"])
    ref_mask = wb.mango_mask(ref_small, border_frac)
    if not mask.any() or not ref_mask.any():
        return None, None
    de_before, lab_before = mango_color_diff(before_small, mask, ref_small, ref_mask)
    de_after, lab_after = mango_color_diff(after_small, mask, ref_small, ref_mask)
    return {
        "compared_with_timestamp": ref_photo["ts"],
        "delta_e_before": round(de_before, 3),
        "delta_e_after": round(de_after, 3),
        "delta_lab_before": [round(v, 2) for v in lab_before],
        "delta_lab_after": [round(v, 2) for v in lab_after],
        "mango_source_saturated_fraction": round(float((before_small[mask].max(axis=-1) >= 250).mean()), 4),
    }, ref_small


def control_diffs(reference_photos, counterpart, border_frac) -> list[tuple[float, list[float]]]:
    """Same mango-colour measurement on photos that never had the problem: the natural drift floor."""
    diffs = []
    for p in reference_photos:
        ref_photo = counterpart(p)
        if ref_photo is None:
            continue
        small, ref_small = cargar_reducido(p["path"]), cargar_reducido(ref_photo["path"])
        mask, ref_mask = wb.mango_mask(small, border_frac), wb.mango_mask(ref_small, border_frac)
        if mask.any() and ref_mask.any():
            diffs.append(mango_color_diff(small, mask, ref_small, ref_mask))
    return diffs


def build_validation(entries, controls, compare_ronda) -> dict:
    valid = [e["validation"] for e in entries if e["validation"]]

    def median(key):
        return round(float(np.median([v[key] for v in valid])), 3) if valid else None

    def median_triplet(rows):
        return [round(float(v), 2) for v in np.median(rows, axis=0)] if rows else None

    return {
        "formula": VALIDATION_FORMULA,
        "mask": "pixels with b* more than 12 above the sheet (median b* of the bright border pixels), "
                "largest connected component, holes filled, eroded 1% of the short side; 1/4-scale decode",
        "compare_ronda_h": compare_ronda,
        "n": len(valid),
        "median_delta_e_before": median("delta_e_before"),
        "median_delta_e_after": median("delta_e_after"),
        "median_delta_lab_before": median_triplet([v["delta_lab_before"] for v in valid]),
        "median_delta_lab_after": median_triplet([v["delta_lab_after"] for v in valid]),
        "control_unaffected_photos": {
            "n": len(controls),
            "median_delta_e": round(float(np.median([de for de, _ in controls])), 3) if controls else None,
            "median_delta_lab": median_triplet([lab for _, lab in controls]),
            "note": f"same measurement for the non-selected photos of the round vs +{compare_ronda} h: "
                    "the floor expected from real ripening and light drift",
        },
    }


def round_pair(rg, bg) -> dict:
    return {"rg": round(float(rg), 5), "bg": round(float(bg), 5)}


def print_result(summary, validation) -> None:
    print("\nResult")
    print(f"  border chroma after: R/G {summary['chroma_after']['rg']}  B/G {summary['chroma_after']['bg']}")
    gain_text = "  ".join(f"{c}={g['min']}-{g['median']}-{g['max']}" for c, g in summary["gains"].items())
    print(f"  gains (min-median-max): {gain_text}")
    if summary["clipped_fraction"]:
        print(f"  clipped fraction per photo: {summary['clipped_fraction']}")
    control = validation["control_unaffected_photos"]
    print(f"  mango dE76 vs +{validation['compare_ronda_h']} h ({validation['n']} photos): median before "
          f"{validation['median_delta_e_before']}, after {validation['median_delta_e_after']}; "
          f"control ({control['n']} unaffected photos) {control['median_delta_e']}")
    print(f"  median signed (dL, da, db) vs +{validation['compare_ronda_h']} h: after "
          f"{validation['median_delta_lab_after']}, control {control['median_delta_lab']}")


def run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    out_dir = resolve_out_dir(data_root, args.out_subdir)
    figure_dir = refuse_protected(Path(args.figure_dir), data_root, "--figure-dir")
    summary_path = refuse_protected(Path(args.summary_json), data_root, "--summary-json") if args.summary_json else None
    photos = load_jornada(data_root, args.jornada)
    in_round, selected = select_photos(photos, args)
    selected_ts = {p["ts"] for p in selected}

    print(f"Measuring border chroma of {len(photos)} RGB photos of {args.jornada} ...")
    means, smalls = measure_photos(photos, selected_ts, args.border_frac)
    chroma = {ts: wb.chroma(m) for ts, m in means.items()}

    if args.rg_threshold is not None:
        threshold, gap_lo, gap_hi = args.rg_threshold, None, None
    else:
        threshold, gap_lo, gap_hi = largest_gap_threshold([c[0] for c in chroma.values()])
    print_distribution(photos, chroma, threshold)
    require_outliers_equal_selection(photos, selected, chroma, threshold)
    print(f"\nOutlier set == selected set: {len(selected)} photos, R/G >= {threshold:.3f}")

    flagged = off_colour_in_other_rounds(photos, chroma, selected_ts)
    print(f"\nNon-selected photos more than {OFF_ROUND_TOL:.0%} off their own round's median chroma: {len(flagged)}")
    for p, why in flagged:
        print(f"  {tag(p)} (+{p['ronda']} h) ts {p['ts']}: {why}")

    reference_photos = [p for p in in_round if p["ts"] not in selected_ts]
    ref_means, deviants = build_reference(reference_photos, means, chroma)
    ref_rg, ref_bg = wb.chroma(ref_means)
    print(f"\nReference: median of {len(reference_photos)} photos, linear means "
          f"{np.round(ref_means, 4).tolist()}, R/G {ref_rg:.4f}, B/G {ref_bg:.4f}")
    for p in deviants:
        print(f"  reference photo >{REFERENCE_DEVIANT_TOL:.0%} off the reference chroma: {tag(p)} ts {p['ts']} "
              f"R/G {chroma[p['ts']][0]:.3f} B/G {chroma[p['ts']][1]:.3f}")
    clean = [means[p["ts"]] for p in reference_photos if p not in deviants]
    clean_chroma = round_pair(*wb.chroma(np.median(clean, axis=0))) if clean else None

    compare = defaultdict(list)
    for p in photos:
        if p["ronda"] == args.compare_ronda:
            compare[(p["mango"], p["cara"])].append(p)

    def counterpart(p):
        found = sorted(compare.get((p["mango"], p["cara"]), []), key=lambda q: q["ts"])
        return found[-1] if found else None

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    entries, figure_rows = [], []
    for p in selected:
        small = smalls[p["ts"]]
        gains = wb.von_kries_gains(ref_means, means[p["ts"]])
        out_path = out_dir / p["path"].name
        after_small, clipped = correct_photo(p, small, gains, out_path, args)
        validation, ref_small = validate_photo(small, after_small, counterpart(p), args.border_frac)
        after = wb.chroma(wb.border_means_linear(after_small, args.border_frac))
        entries.append({
            "timestamp": p["ts"], "mango_id": p["mango"], "cara": p["cara"],
            "gains": [round(float(g), 5) for g in gains],
            "before": round_pair(*chroma[p["ts"]]),
            "after": round_pair(*after),
            "clipped_fraction": None if clipped is None else round(clipped, 6),
            "source_sha256": sha256_file(p["path"]),
            "output_sha256": None if args.dry_run else sha256_file(out_path),
            "validation": validation,
        })
        if validation:
            figure_rows.append({"tag": tag(p), "rg_before": chroma[p["ts"]][0], "rg_after": after[0],
                                "before": small, "after": after_small, "ref": ref_small})

    controls = control_diffs(reference_photos, counterpart, args.border_frac)
    validation = build_validation(entries, controls, args.compare_ronda)
    valid = [e["validation"] for e in entries if e["validation"]]
    clipped = [e["clipped_fraction"] for e in entries if e["clipped_fraction"] is not None]
    manifest = {
        "method": METHOD,
        "jornada": args.jornada,
        "ronda_h": args.ronda,
        "faces_spec": args.faces,
        "border_frac": args.border_frac,
        "jpeg_quality": args.quality,
        "threshold": {
            "rg_linear": round(threshold, 5),
            "rule": "geometric midpoint of the largest gap in log(R/G) over every RGB photo of the jornada"
                    if gap_lo is not None else "given with --rg-threshold",
            "gap": None if gap_lo is None else [round(gap_lo, 5), round(gap_hi, 5)],
        },
        "reference": {
            "n": len(reference_photos),
            "timestamps": sorted(p["ts"] for p in reference_photos),
            "means_linear": [round(float(v), 6) for v in ref_means],
            "chroma": round_pair(ref_rg, ref_bg),
            "deviants_timestamps": sorted(p["ts"] for p in deviants),
            "chroma_without_deviants": clean_chroma,
        },
        "validation": validation,
        "files": entries,
    }
    summary = {
        "jornada": args.jornada, "ronda_h": args.ronda, "n_corrected": len(entries),
        "n_photos_measured": len(photos), "threshold_rg_linear": round(threshold, 4),
        "outlier_gap": None if gap_lo is None else [round(gap_lo, 4), round(gap_hi, 4)],
        "reference": {"n": len(reference_photos), **round_pair(ref_rg, ref_bg),
                      "n_deviants": len(deviants), "without_deviants": clean_chroma},
        "chroma_before": {k: describe([e["before"][k] for e in entries]) for k in ("rg", "bg")},
        "chroma_after": {k: describe([e["after"][k] for e in entries]) for k in ("rg", "bg")},
        "gains": {c: describe([e["gains"][i] for e in entries]) for i, c in enumerate("rgb")},
        "clipped_fraction": describe(clipped, digits=6) if clipped else None,
        "validation": {k: v for k, v in validation.items() if k not in ("formula", "mask")},
        "mango_source_saturated_fraction":
            describe([v["mango_source_saturated_fraction"] for v in valid]) if valid else None,
        "non_selected_off_colour": [{"timestamp": p["ts"], "tag": tag(p), "ronda_h": p["ronda"], "why": why}
                                    for p, why in flagged],
    }
    print_result(summary, validation)

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\nWrote {len(entries)} photos + manifest.json to {out_dir}")
    if not args.no_figure and figure_rows:
        good = next((p for p in reference_photos if counterpart(p)), None)
        good_row = None if good is None else {
            "tag": tag(good), "before": cargar_reducido(good["path"]),
            "ref": cargar_reducido(counterpart(good)["path"])}
        shown = pick_rows(figure_rows, FIGURE_ROWS)
        print(f"Figures: {make_figure(figure_dir, shown, good_row)}, "
              f"{make_mask_figure(figure_dir, shown, args.border_frac)}")
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=1) + "\n")
        print(f"Summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", required=True, help="captura_v3 root holding news/{rgb,nir,meta}")
    ap.add_argument("--jornada", required=True, help="jornada id, e.g. lote1-20261024")
    ap.add_argument("--ronda", type=int, required=True, help="ronda_h of the affected photos (e.g. 4)")
    ap.add_argument("--faces", required=True,
                    help="affected photos as mango[:faces],...; '11:4,12-25' = mango 11 face 4 + mangos 12-25")
    ap.add_argument("--expected-count", type=int, required=True,
                    help="abort unless exactly this many photos are selected")
    ap.add_argument("--out-subdir", default="news/rgb_wb", help="output folder under the data root (never news/rgb)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality of the corrected photos")
    ap.add_argument("--dry-run", action="store_true", help="measure and validate, write nothing")
    ap.add_argument("--compare-ronda", type=int, default=2, help="round used as colour truth for the mango check")
    ap.add_argument("--border-frac", type=float, default=wb.BORDER_FRAC,
                    help="border frame thickness (capture QC uses 0.10)")
    ap.add_argument("--rg-threshold", type=float, default=None, help="override the data-driven R/G outlier threshold")
    ap.add_argument("--figure-dir", default=str(FINDINGS_DIR / "figures"))
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--summary-json", default=str(FINDINGS_DIR / "evidence" / "summary.json"),
                    help="small evidence file with the summary numbers ('' to skip)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except WhiteBalanceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
