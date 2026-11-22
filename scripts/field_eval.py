#!/usr/bin/env python3
"""Evaluate a field-test session recorded by the Android app (`mobile/android`, Live > "session").

A session folder (or its `.zip`, pulled with `adb`) holds `frames/f_<index>.jpg`, `predictions.jsonl` (one line per processed
frame, see `mobile/android/.../core/SessionLog.kt`) and `session.json`. Three subcommands:

    report     no ground truth: detection rates, tracks, confidence histogram, latency -> report.json + report.md
    export-ls  sample frames and write a Label Studio import file with the predictions as pre-annotations
    eval       ground truth from a Label Studio export of those frames -> AP / AUROC / recall / specificity

    python scripts/field_eval.py report    session_20261121_101500.zip --out reports/field/market1
    python scripts/field_eval.py export-ls session_20261121_101500 --every 10 --out reports/field/market1/ls
    python scripts/field_eval.py eval      session_20261121_101500 --gt ls_export.json --out reports/field/market1

Assumptions (read before quoting a number):

* A "mango" is a stage-1 track: one face of one fruit seen from above, identified by `trackId`. A `trackId` is only
  meaningful inside one session (the tracker restarts at every session), so two sessions never share mangos and a
  physical mango seen from two faces is two tracks.
* `--nms-iou` (report, export-ls, eval; a float or a comma list such as `0.7,0.6,0.5`) re-runs greedy per-class NMS on the
  damage detections of every frame before anything is computed: score-sorted, a box is suppressed when its IoU with a kept
  box is > the threshold (the rule of the app's `Decoder.nms`). It can only remove boxes, so a value larger than the NMS IoU
  recorded in the session is an error. With a list there is one block per threshold. Default: the recorded value (no change).
* `--conf` (report only, default 0.25) is the operational damage threshold: detections below it are ignored even when the
  session was recorded at a lower one (e.g. 0.05). `eval` keeps using the session's own confidence.

* Boxes and polygons are in upright frame pixels; Label Studio percentages are converted with the frame size of the
  prediction line. Frames are matched to Label Studio tasks by the `f_<6 digits>.jpg` name in `data.image` (a
  `<hash>-` prefix added by an upload is fine). Frames that were not annotated are ignored by `eval`.
* Only the DETECTOR ROI mode records tracked mangos. For the single-crop modes `report` and `eval` treat the crop's damage
  detections as belonging to one pseudo mango (track 0, the whole frame); mango AP and the per-mango metrics are then
  meaningless.
* Held mangos: a mango record with `"held": true` is the tracker's memory of an earlier frame (stage 1 missed it in this
  frame; the damage model still ran on its last box). `report` counts held and detected mango-frames separately. `eval`
  computes mango AP50 from the detected (non-held) mangos only, because a held box is not a stage-1 detection of that
  frame, but damage AP and the per-mango metrics still use held mangos and their damage detections. Sessions recorded
  before the hold feature have no `held` field and count as all detected.
* Ground truth classes are `mango` and `damage` (label text, case-insensitive), as rectangles or polygons; a polygon also
  gives a box (its bounding box). The latest non-cancelled annotation of a task is used.
* AP is `src.training.metrics._compute_ap_at_iou` (all-point interpolation) via `src.evaluation.v4_metrics`; mask AP50
  rasterizes polygons on a 512 px normalized canvas like the training metrics, and is reported only when the ground truth
  has damage polygons. Predictions of a session recorded without a segment model have no polygons (zero-area masks).
* Predicted damage boxes are those of the session (confidence >= the session's `confThreshold`, centre inside a kept
  mango), so AP is computed over that truncated ranking: it is a lower bound of the AP at confidence 0.
* Per-mango (track) metrics: in every annotated frame a predicted mango is matched to a ground-truth mango (greedy by IoU,
  >= 0.5). A ground-truth mango is "damaged" when a ground-truth damage box has its centre inside it. A track is positive
  when any frame it matched has a damaged ground-truth mango, its score is the maximum damage confidence over the annotated
  frames it matched (0 without a detection), and it is predicted positive when that score is >= the session threshold.
  Tracks that never matched a ground-truth mango are excluded and counted. Recall and specificity are over those tracks.
* False-positive damage boxes per healthy mango are the predicted damage boxes of matched predicted mangos whose ground-truth
  mango is healthy, divided by the number of healthy tracks (`per_track`) and by the number of healthy matched mango-frames
  (`per_mango_frame`).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FRAME_NAME = re.compile(r"f_(\d{6})\.jpg", re.IGNORECASE)
MATCH_IOU = 0.5
LS_FROM_BOXES = "boxes"
LS_FROM_POLYGONS = "polygons"
LS_TO = "image"
NMS_HELP = "re-run per-class NMS on damage detections at this IoU (float or comma list); default: the recorded value"

LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="boxes" toName="image" strokeWidth="2">
    <Label value="mango" background="#00e676"/>
    <Label value="damage" background="#ff1744"/>
  </RectangleLabels>
  <PolygonLabels name="polygons" toName="image" strokeWidth="2" opacity="0.35">
    <Label value="damage" background="#ff1744"/>
  </PolygonLabels>
</View>
"""


# --------------------------------------------------------------------------
# Session loading
# --------------------------------------------------------------------------


@dataclass
class Session:
    root: Path
    info: dict
    frames: list[dict]
    by_index: dict[int, dict] = field(default_factory=dict)
    conf_override: float | None = None

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def nms_iou(self) -> float | None:
        """The NMS IoU the app used (most common value of the frame records), or None for an empty session."""
        values = Counter(f["nmsIou"] for f in self.frames if f.get("nmsIou") is not None)
        return float(values.most_common(1)[0][0]) if values else None

    @property
    def conf(self) -> float:
        if self.conf_override is not None:
            return self.conf_override
        values = Counter(f["confThreshold"] for f in self.frames)
        return float(values.most_common(1)[0][0]) if values else 0.0


def load_session(path: str | Path, workdir: Path | None = None) -> Session:
    """Read a session folder or `.zip`. A zip is extracted to `workdir` (a temporary folder by default)."""
    path = Path(path)
    if path.is_file() and path.suffix == ".zip":
        target = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="field_eval_"))
        with zipfile.ZipFile(path) as z:
            z.extractall(target)
        path = target
    if not (path / "predictions.jsonl").exists():
        inner = [p for p in sorted(path.glob("session_*")) if (p / "predictions.jsonl").exists()]
        if len(inner) != 1:
            raise FileNotFoundError(f"{path} has no predictions.jsonl (and no single session_* folder inside)")
        path = inner[0]
    frames = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines() if line.strip()]
    info_path = path / "session.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    session = Session(path, info, frames)
    session.by_index = {f["index"]: f for f in frames}
    return session


def frame_mangos(frame: dict) -> list[dict]:
    """The mangos of a frame record; single-crop sessions get one pseudo mango (track 0) holding every damage detection."""
    if frame.get("noMango") is not None or frame.get("detectorModel"):
        return frame.get("mangos", [])
    damages = [
        {"box": d["box"], "score": d["score"], "polygon": None} for d in frame.get("detections", []) if d["cls"] == 1
    ]
    if not damages and not frame.get("detections"):
        return []
    return [
        {
            "trackId": 0,
            "box": [0.0, 0.0, float(frame["frameW"]), float(frame["frameH"])],
            "score": 1.0,
            "damages": damages,
            "maxDamage": max((d["score"] for d in damages), default=0.0),
        }
    ]


def parse_thresholds(text: str) -> list[float]:
    """`"0.7"` or `"0.7,0.6,0.5"` -> floats in (0, 1]; raises `argparse.ArgumentTypeError` otherwise."""
    try:
        values = [float(t) for t in text.split(",") if t.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a float or comma list of floats: {text!r}") from None
    if not values or any(not 0.0 < v <= 1.0 for v in values):
        raise argparse.ArgumentTypeError(f"NMS IoU thresholds must be in (0, 1]: {text!r}")
    return values


def threshold_key(t: float) -> str:
    return f"{t:g}"


def nms_boxes(items: list[dict], iou: float) -> list[dict]:
    """Greedy NMS over `{"box", "score"}` dicts: score-sorted, suppress when IoU > `iou`. Returns the kept items by score."""
    kept: list[dict] = []
    for d in sorted(items, key=lambda d: -d["score"]):
        if all(box_iou(k["box"], d["box"]) <= iou for k in kept):
            kept.append(d)
    return kept


def apply_nms(session: Session, iou: float) -> Session:
    """A copy of `session` whose damage detections went through NMS at `iou` (damage class only, per class)."""
    recorded = session.nms_iou
    if recorded is not None and iou > recorded + 1e-9:
        raise ValueError(
            f"--nms-iou {iou:g} is larger than the NMS IoU recorded in {session.name} ({recorded:g}): "
            "NMS can only remove boxes, so a looser threshold cannot be reproduced"
        )
    frames = []
    for f in session.frames:
        g = dict(f)
        if f.get("mangos"):
            g["mangos"] = []
            for m in f["mangos"]:
                damages = nms_boxes(m["damages"], iou)
                g["mangos"].append({**m, "damages": damages, "maxDamage": max((d["score"] for d in damages), default=0.0)})
        if f.get("detections"):
            damage = nms_boxes([d for d in f["detections"] if d["cls"] == 1], iou)
            g["detections"] = [d for d in f["detections"] if d["cls"] != 1] + damage
        frames.append(g)
    out = Session(session.root, session.info, frames, conf_override=session.conf_override)
    out.by_index = {f["index"]: f for f in frames}
    return out


def apply_conf(session: Session, conf: float) -> Session:
    """A copy of `session` keeping only damage detections with score >= `conf`; `session.conf` becomes `conf`."""
    frames = []
    for f in session.frames:
        g = dict(f)
        if f.get("mangos"):
            g["mangos"] = []
            for m in f["mangos"]:
                damages = [d for d in m["damages"] if d["score"] >= conf]
                g["mangos"].append({**m, "damages": damages, "maxDamage": max((d["score"] for d in damages), default=0.0)})
        if f.get("detections"):
            g["detections"] = [d for d in f["detections"] if d["cls"] != 1 or d["score"] >= conf]
        g["confThreshold"] = conf
        frames.append(g)
    out = Session(session.root, session.info, frames, conf_override=conf)
    out.by_index = {f["index"]: f for f in frames}
    return out


def _stats(values) -> dict:
    arr = np.asarray(list(values), float)
    if arr.size == 0:
        return {"mean": None, "median": None, "p90": None, "p95": None, "max": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def memory_stats(session: Session) -> dict | None:
    """RAM summary in kB from the per-frame `mem` (falling back to `session.json`'s `memory`); None for old sessions."""
    info = session.info.get("memory") or {}
    mems = [f["mem"] for f in session.frames if f.get("mem")]
    if not mems and not info:
        return None
    rss = [m["rssKb"] for m in mems]
    pss = [m["pssKb"] for m in mems if m.get("pssKb") is not None]
    pss_max = max(pss) if pss else info.get("pssMaxKb")
    return {
        "rss_max_kb": max(rss) if rss else info.get("rssMaxKb"),
        "rss_median_kb": float(np.median(rss)) if rss else info.get("rssMedianKb"),
        "hwm_end_kb": mems[-1]["hwmKb"] if mems else info.get("hwmAtEndKb"),
        "pss_max_kb": pss_max,
        "ram_before_load_kb": info.get("ramBeforeLoadKb"),
        "n_samples": len(mems) if mems else info.get("samples"),
    }


def build_report(session: Session) -> dict:
    frames = session.frames
    n = len(frames)
    mangos = [frame_mangos(f) for f in frames]
    per_frame = [len(m) for m in mangos]
    n_held = sum(1 for ms in mangos for m in ms if m.get("held"))
    n_detected = sum(len(ms) for ms in mangos) - n_held
    tracks: dict[int, list[tuple[int, list[dict]]]] = defaultdict(list)
    scores: list[float] = []
    for f, ms in zip(frames, mangos):
        for m in ms:
            tracks[m["trackId"]].append((f["index"], m))
            scores.extend(d["score"] for d in m["damages"])
    conf = session.conf
    edges = np.linspace(conf, 1.0, 11)
    counts, _ = np.histogram(scores, bins=edges) if scores else (np.zeros(10, int), None)
    track_rows = []
    for tid in sorted(tracks):
        seen = tracks[tid]
        with_damage = sum(1 for _, m in seen if m["damages"])
        peak = [m["maxDamage"] for _, m in seen]
        track_rows.append(
            {
                "track_id": tid,
                "frames_seen": len(seen),
                "first_frame": seen[0][0],
                "last_frame": seen[-1][0],
                "pct_frames_with_damage": 100.0 * with_damage / len(seen),
                "max_damage": {"mean": _stats(peak)["mean"], "median": _stats(peak)["median"], "p90": _stats(peak)["p90"]},
            }
        )
    walls = [f["wallMs"] for f in frames]
    span_s = (max(walls) - min(walls)) / 1000.0 if n > 1 else 0.0
    timings = {k: _stats(f["timings"][k] for f in frames if f["timings"].get(k) is not None)
               for k in ("roiMs", "preMs", "inferMs", "postMs", "totalMs", "detectorMs")}
    return {
        "session": session.name,
        "roi_mode": frames[0]["roiMode"] if frames else None,
        "device": session.info.get("device"),
        "detector_model": frames[0].get("detectorModel") if frames else None,
        "damage_model": frames[0].get("damageModel") if frames else None,
        "accelerator": frames[0].get("accelerator") if frames else None,
        "conf_threshold": conf,
        "mango_threshold": frames[0].get("mangoThreshold") if frames else None,
        "keep_threshold": frames[0].get("keepThreshold") if frames else None,
        "hold_frames": frames[0].get("holdFrames") if frames else None,
        "nms_iou": frames[0].get("nmsIou") if frames else None,
        "n_frames": n,
        "n_frames_saved": sum(1 for f in frames if f.get("frame")),
        "dropped_writes": session.info.get("droppedWrites"),
        "duration_s": span_s,
        "fps_wall": (n - 1) / span_s if span_s > 0 else None,
        "pct_frames_with_mango": 100.0 * sum(1 for c in per_frame if c) / n if n else None,
        "mangos_per_frame": {"mean": float(np.mean(per_frame)) if per_frame else None,
                             "median": float(np.median(per_frame)) if per_frame else None,
                             "max": max(per_frame, default=None)},
        "n_mango_frames_detected": n_detected,
        "n_mango_frames_held": n_held,
        "n_tracks": len(tracks),
        "tracks": track_rows,
        "damage_confidence_histogram": {"bin_edges": [float(e) for e in edges], "counts": [int(c) for c in counts],
                                        "n_detections": len(scores)},
        "latency_ms": timings,
        "memory": memory_stats(session),
    }


def _fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def report_markdown(r: dict) -> str:
    lines = [
        f"# Field session {r['session']}",
        "",
        f"- Device: {r['device']}; accelerator: {r['accelerator']}",
        f"- ROI mode: {r['roi_mode']}; detector: {r['detector_model']}; damage model: {r['damage_model']}",
        f"- Damage conf {_fmt(r['conf_threshold'])}, mango threshold {_fmt(r['mango_threshold'])} "
        f"(keep {_fmt(r['keep_threshold'])}, hold {_fmt(r['hold_frames'], 0)} frames), NMS IoU {_fmt(r['nms_iou'])}",
        f"- Frames: {r['n_frames']} processed, {r['n_frames_saved']} saved, {_fmt(r['dropped_writes'])} dropped writes; "
        f"{_fmt(r['duration_s'], 1)} s, {_fmt(r['fps_wall'], 1)} FPS (wall)",
        "",
        "## Detection",
        "",
        f"- Frames with at least one mango: {_fmt(r['pct_frames_with_mango'], 1)} %",
        f"- Mangos per frame: mean {_fmt(r['mangos_per_frame']['mean'])}, median {_fmt(r['mangos_per_frame']['median'])}, "
        f"max {_fmt(r['mangos_per_frame']['max'])}",
        f"- Mango-frames: {r['n_mango_frames_detected']} detected, {r['n_mango_frames_held']} held (from tracker memory)",
        f"- Distinct tracks: {r['n_tracks']}",
        "",
        "| track | frames | % frames with damage | max damage mean | median | p90 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for t in r["tracks"]:
        md = t["max_damage"]
        lines.append(
            f"| {t['track_id']} | {t['frames_seen']} | {t['pct_frames_with_damage']:.1f} | "
            f"{_fmt(md['mean'])} | {_fmt(md['median'])} | {_fmt(md['p90'])} |"
        )
    mem = r.get("memory")
    if mem:
        mb = lambda kb: "-" if kb is None else f"{kb / 1024:.0f}"  # noqa: E731
        lines += [
            "", "## Memory (MB)", "",
            f"- RSS max {mb(mem['rss_max_kb'])}, median {mb(mem['rss_median_kb'])}; VmHWM at end {mb(mem['hwm_end_kb'])} "
            f"(process lifetime); PSS max {mb(mem['pss_max_kb'])}; RSS before model load {mb(mem['ram_before_load_kb'])} "
            f"({_fmt(mem['n_samples'], 0)} samples)",
        ]
    h = r["damage_confidence_histogram"]
    lines += ["", f"## Damage confidence histogram ({h['n_detections']} detections)", "", "| bin | count |", "|---|---:|"]
    for lo, hi, c in zip(h["bin_edges"][:-1], h["bin_edges"][1:], h["counts"]):
        lines.append(f"| {lo:.2f}-{hi:.2f} | {c} |")
    lines += ["", "## Latency (ms)", "", "| stage | mean | median | p90 | p95 | max |", "|---|---:|---:|---:|---:|---:|"]
    for k, s in r["latency_ms"].items():
        if s["n"]:
            lines.append(f"| {k} | {_fmt(s['mean'])} | {_fmt(s['median'])} | {_fmt(s['p90'])} | {_fmt(s['p95'])} | {_fmt(s['max'])} |")
    return "\n".join(lines) + "\n"


def _prepare(session: Session, nms_iou: float | None) -> Session:
    """Applies `--nms-iou` for one threshold; None keeps the recorded detections."""
    return session if nms_iou is None else apply_nms(session, nms_iou)


def _thresholds(args) -> list[float | None]:
    return list(args.nms_iou) if args.nms_iou else [None]


def multi_report_markdown(reports: dict[str, dict]) -> str:
    """One row per NMS IoU threshold, then the full report of each."""
    first = next(iter(reports.values()))
    lines = [
        f"# Field session {first['session']}: NMS IoU sweep", "",
        "| NMS IoU | damage detections | frames with mango (%) | tracks | mean max damage |", "|---:|---:|---:|---:|---:|",
    ]
    for key, r in reports.items():
        peaks = [t["max_damage"]["mean"] for t in r["tracks"] if t["max_damage"]["mean"] is not None]
        lines.append(
            f"| {key} | {r['damage_confidence_histogram']['n_detections']} | {_fmt(r['pct_frames_with_mango'], 1)} | "
            f"{r['n_tracks']} | {_fmt(float(np.mean(peaks)) if peaks else None, 3)} |"
        )
    for key, r in reports.items():
        lines += ["", f"## NMS IoU {key}", "", report_markdown(r).replace("# Field session", "### Field session", 1)]
    return "\n".join(lines) + "\n"


def cmd_report(args) -> int:
    session = load_session(args.session)
    thresholds = _thresholds(args)
    try:
        reports = {}
        for t in thresholds:
            reports["recorded" if t is None else threshold_key(t)] = build_report(apply_conf(_prepare(session, t), args.conf))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(thresholds) == 1:
        report = next(iter(reports.values()))
        text = report_markdown(report)
        (out / "report.json").write_text(json.dumps(report, indent=2))
    else:
        text = multi_report_markdown(reports)
        (out / "report.json").write_text(json.dumps({"nms_iou": reports}, indent=2))
    (out / "report.md").write_text(text)
    print(text)
    return 0


# --------------------------------------------------------------------------
# export-ls
# --------------------------------------------------------------------------


def sample_frames(session: Session, every: int | None, per_track: int | None) -> list[int]:
    """Frame indices to annotate: every Nth saved frame, or K evenly spaced frames per track (union)."""
    saved = [f for f in session.frames if f.get("frame")]
    if per_track:
        chosen: set[int] = set()
        seen: dict[int, list[int]] = defaultdict(list)
        for f in saved:
            for m in frame_mangos(f):
                seen[m["trackId"]].append(f["index"])
        for idx in seen.values():
            k = min(per_track, len(idx))
            chosen.update(idx[int(round(i * (len(idx) - 1) / max(k - 1, 1)))] for i in range(k))
        return sorted(chosen)
    step = every or 10
    return [f["index"] for f in saved if f["index"] % step == 0]


def _pct(v: float, total: float) -> float:
    return round(100.0 * v / total, 4)


def ls_result(frame: dict) -> list[dict]:
    """Pre-annotation results of a frame: mango rectangles, damage polygons (a rectangle when there is no polygon)."""
    w, h = frame["frameW"], frame["frameH"]
    base = {"to_name": LS_TO, "original_width": w, "original_height": h, "image_rotation": 0}
    out = []
    for m in frame_mangos(frame):
        x1, y1, x2, y2 = m["box"]
        if m["trackId"] != 0 or frame.get("detectorModel"):
            out.append({**base, "id": f"m{m['trackId']}", "from_name": LS_FROM_BOXES, "type": "rectanglelabels", "score": m["score"],
                        "value": {"x": _pct(x1, w), "y": _pct(y1, h), "width": _pct(x2 - x1, w), "height": _pct(y2 - y1, h),
                                  "rotation": 0, "rectanglelabels": ["mango"]}})
        for j, d in enumerate(m["damages"]):
            rid = f"d{m['trackId']}_{j}"
            poly = d.get("polygon")
            if poly and len(poly) >= 6:
                pts = [[_pct(poly[i], w), _pct(poly[i + 1], h)] for i in range(0, len(poly) - 1, 2)]
                out.append({**base, "id": rid, "from_name": LS_FROM_POLYGONS, "type": "polygonlabels", "score": d["score"],
                            "value": {"points": pts, "polygonlabels": ["damage"]}})
            else:
                bx1, by1, bx2, by2 = d["box"]
                out.append({**base, "id": rid, "from_name": LS_FROM_BOXES, "type": "rectanglelabels", "score": d["score"],
                            "value": {"x": _pct(bx1, w), "y": _pct(by1, h), "width": _pct(bx2 - bx1, w), "height": _pct(by2 - by1, h),
                                      "rotation": 0, "rectanglelabels": ["damage"]}})
    return out


def build_ls_tasks(session: Session, indices: list[int], url_prefix: str) -> list[dict]:
    tasks = []
    for idx in indices:
        f = session.by_index[idx]
        name = Path(f["frame"]).name
        results = ls_result(f)
        task = {"data": {"image": f"{url_prefix}{name}", "frame_index": idx, "session": session.name}}
        if results:
            mean = float(np.mean([r["score"] for r in results]))
            task["predictions"] = [{"model_version": f"{f.get('detectorModel')}+{f['damageModel']}", "score": mean, "result": results}]
        tasks.append(task)
    return tasks


def cmd_export_ls(args) -> int:
    session = load_session(args.session)
    thresholds = _thresholds(args)
    try:
        variants = {t: _prepare(session, t) for t in thresholds}
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    indices = sample_frames(session, args.every, args.per_track)
    if not indices:
        print("no frame to export (no saved frames, or nothing matched the sampling)", file=sys.stderr)
        return 1
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for idx in indices:
        rel = session.by_index[idx]["frame"]
        shutil.copy2(session.root / rel, out / "frames" / Path(rel).name)
    (out / "labeling_config.xml").write_text(LABEL_CONFIG)
    for t, variant in variants.items():
        tasks = build_ls_tasks(variant, indices, args.image_url_prefix)
        name = "tasks.json" if len(variants) == 1 else f"tasks_nms{threshold_key(t)}.json"
        (out / name).write_text(json.dumps(tasks, indent=2))
        print(f"{len(tasks)} tasks -> {out / name}; frames copied to {out / 'frames'}")
    print(f"image URLs start with {args.image_url_prefix!r}: serve {out / 'frames'} accordingly (Label Studio local files).")
    return 0


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------


@dataclass
class GtFrame:
    mangos: list[list[float]] = field(default_factory=list)
    damages: list[list[float]] = field(default_factory=list)  # xyxy
    damage_polygons: list[np.ndarray | None] = field(default_factory=list)  # pixel (n, 2) or None, aligned with damages


def _polygon_box(poly: np.ndarray) -> list[float]:
    return [float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())]


def load_ground_truth(path: str | Path, session: Session) -> tuple[dict[int, GtFrame], dict]:
    """Label Studio export -> `{frame index: GtFrame}` in frame pixels, plus counts of skipped tasks."""
    data = json.loads(Path(path).read_text())
    tasks = data["tasks"] if isinstance(data, dict) and "tasks" in data else data
    gt: dict[int, GtFrame] = {}
    skipped = Counter()
    for task in tasks:
        match = FRAME_NAME.search(str(task.get("data", {}).get("image", "")))
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not match or not anns:
            skipped["no_annotation_or_unmatched_name"] += 1
            continue
        idx = int(match.group(1))
        frame = session.by_index.get(idx)
        if frame is None:
            skipped["frame_not_in_session"] += 1
            continue
        w, h = frame["frameW"], frame["frameH"]
        g = GtFrame()
        for r in anns[-1].get("result", []):
            v = r.get("value", {})
            labels = v.get("rectanglelabels") or v.get("polygonlabels") or []
            if not labels:
                continue
            cls = str(labels[0]).strip().lower()
            if cls not in ("mango", "damage"):
                skipped[f"unknown_class_{cls}"] += 1
                continue
            poly = None
            if r.get("type") == "polygonlabels":
                poly = np.array([[p[0] / 100.0 * w, p[1] / 100.0 * h] for p in v["points"]], float)
                box = _polygon_box(poly)
            elif r.get("type") == "rectanglelabels":
                x, y = v["x"] / 100.0 * w, v["y"] / 100.0 * h
                box = [x, y, x + v["width"] / 100.0 * w, y + v["height"] / 100.0 * h]
            else:
                continue
            if cls == "mango":
                g.mangos.append(box)
            else:
                g.damages.append(box)
                g.damage_polygons.append(poly)
        gt[idx] = g
    return gt, dict(skipped)


def box_iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _inside(box, outer) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def match_mangos(pred_mangos: list[dict], gt_mangos: list[list[float]], min_iou: float = MATCH_IOU) -> list[tuple[int, int]]:
    """Greedy one-to-one `(pred index, gt index)` pairs, highest IoU first, IoU >= `min_iou`."""
    pairs = sorted(
        ((box_iou(p["box"], g), i, j) for i, p in enumerate(pred_mangos) for j, g in enumerate(gt_mangos)),
        reverse=True,
    )
    used_p, used_g, out = set(), set(), []
    for iou, i, j in pairs:
        if iou < min_iou:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j))
    return out


def track_metrics(session: Session, gt: dict[int, GtFrame], iou_module) -> dict:
    """Per-mango (track) damage classification and false positives on healthy mangos (module docstring)."""
    conf = session.conf
    score: dict[int, float] = {}
    positive: dict[int, bool] = defaultdict(bool)
    healthy_fp_boxes: dict[int, int] = defaultdict(int)
    healthy_frames = 0
    healthy_fp_total = 0
    n_unmatched_pred = 0
    n_missed_gt = 0
    for idx, g in sorted(gt.items()):
        preds = frame_mangos(session.by_index[idx])
        pairs = match_mangos(preds, g.mangos)
        n_unmatched_pred += len(preds) - len(pairs)
        n_missed_gt += len(g.mangos) - len(pairs)
        for i, j in pairs:
            m = preds[i]
            tid = m["trackId"]
            damaged = any(_inside(d, g.mangos[j]) for d in g.damages)
            positive[tid] = positive[tid] or damaged
            score[tid] = max(score.get(tid, 0.0), float(m["maxDamage"]))
            if not damaged:
                n_fp = sum(1 for d in m["damages"] if d["score"] >= conf)
                healthy_fp_boxes[tid] += n_fp
                healthy_fp_total += n_fp
                healthy_frames += 1
    pos = [score[t] for t in score if positive[t]]
    neg = [score[t] for t in score if not positive[t]]
    tp = sum(1 for s in pos if s >= conf)
    tn = sum(1 for s in neg if s < conf)
    healthy_tracks = [t for t in score if not positive[t]]
    return {
        "n_tracks_evaluated": len(score),
        "n_positive_tracks": len(pos),
        "n_negative_tracks": len(neg),
        "auroc": iou_module.auroc(pos, neg),
        "conf_threshold": conf,
        "recall": tp / len(pos) if pos else None,
        "specificity": tn / len(neg) if neg else None,
        "fp_damage_boxes_per_healthy_mango": {
            "per_track": healthy_fp_total / len(healthy_tracks) if healthy_tracks else None,
            "per_mango_frame": healthy_fp_total / healthy_frames if healthy_frames else None,
            "n_fp_boxes": healthy_fp_total,
            "n_healthy_tracks": len(healthy_tracks),
            "n_healthy_mango_frames": healthy_frames,
        },
        "n_unmatched_predicted_mangos": n_unmatched_pred,
        "n_missed_gt_mangos": n_missed_gt,
    }


def evaluate(session: Session, gt: dict[int, GtFrame]) -> dict:
    """The content of `eval.json`; imports torch-based metrics lazily."""
    import torch

    from src.evaluation import detectability
    from src.evaluation import v4_metrics as v4
    from src.training.metrics import _compute_ap_at_iou

    def norm(box, w, h):
        return [box[0] / w, box[1] / h, box[2] / w, box[3] / h]

    def tens(rows):
        return torch.tensor(rows, dtype=torch.float32).reshape(-1, 4)

    photos, mango_pb, mango_ps, mango_gb = [], [], [], []
    for idx, g in sorted(gt.items()):
        f = session.by_index[idx]
        w, h = f["frameW"], f["frameH"]
        preds = frame_mangos(f)
        dmg = [d for m in preds for d in m["damages"]]
        pred_polys = tuple(
            (np.array(d["polygon"], float).reshape(-1, 2) / np.array([w, h]) if d.get("polygon") else np.zeros((0, 2)))
            for d in dmg
        )
        gt_polys = tuple(p / np.array([w, h]) for p in g.damage_polygons if p is not None)
        photos.append(
            v4.EvaluatedPhoto(
                stem=f"f_{idx:06d}", hour=None, zona="", provenance="", face=("session", str(idx), "1"),
                gt_boxes=tens([norm(b, w, h) for b in g.damages]),
                pred_boxes=tens([norm(d["box"], w, h) for d in dmg]),
                pred_scores=torch.tensor([d["score"] for d in dmg], dtype=torch.float32),
                pred_polygons=pred_polys, gt_polygons=gt_polys,
            )
        )
        detected = [m for m in preds if not m.get("held")]  # a held box is not a stage-1 detection of this frame
        mango_pb.append(tens([norm(m["box"], w, h) for m in detected]))
        mango_ps.append(torch.tensor([m["score"] for m in detected], dtype=torch.float32))
        mango_gb.append(tens([norm(b, w, h) for b in g.mangos]))

    result: dict = {
        "session": session.name,
        "n_annotated_frames": len(gt),
        "conf_threshold": session.conf,
        "roi_mode": session.frames[0]["roiMode"] if session.frames else None,
        "damage_box_ap50": v4.damage_ap(photos, 0.50),
        "damage_box_ap30": v4.damage_ap(photos, 0.30),
    }
    n_gt_poly = sum(len(p.gt_polygons) for p in photos)
    result["damage_mask_ap50"] = (
        v4.damage_mask_ap(photos, 0.50, v4._MaskOverlap(photos)) if n_gt_poly else {"ap": None, "n_gt": 0, "n_images": len(photos)}
    )
    n_gt_mango = sum(len(b) for b in mango_gb)
    mango_ap = None
    if n_gt_mango:
        _, per_class = _compute_ap_at_iou(
            mango_pb, mango_ps, [torch.zeros(len(s), dtype=torch.long) for s in mango_ps],
            mango_gb, [torch.zeros(len(b), dtype=torch.long) for b in mango_gb], 0.50, 1,
        )
        mango_ap = per_class[0]
    result["mango_ap50"] = {"ap": mango_ap, "n_gt": n_gt_mango, "n_images": len(photos)}
    result["per_mango"] = track_metrics(session, gt, detectability)
    return result


def eval_markdown(r: dict) -> str:
    pm = r["per_mango"]
    fp = pm["fp_damage_boxes_per_healthy_mango"]
    rows = [
        ("Damage box AP50", r["damage_box_ap50"]["ap"], f"{r['damage_box_ap50']['n_gt']} GT boxes"),
        ("Damage box AP30", r["damage_box_ap30"]["ap"], ""),
        ("Damage mask AP50", r["damage_mask_ap50"]["ap"], f"{r['damage_mask_ap50']['n_gt']} GT polygons"),
        ("Mango AP50", r["mango_ap50"]["ap"], f"{r['mango_ap50']['n_gt']} GT mangos"),
        ("Per-mango AUROC", pm["auroc"], f"{pm['n_positive_tracks']} damaged / {pm['n_negative_tracks']} healthy tracks"),
        (f"Recall @ conf {r['conf_threshold']:.2f}", pm["recall"], ""),
        (f"Specificity @ conf {r['conf_threshold']:.2f}", pm["specificity"], ""),
        ("FP damage boxes / healthy mango (track)", fp["per_track"], f"{fp['n_fp_boxes']} boxes"),
        ("FP damage boxes / healthy mango-frame", fp["per_mango_frame"], f"{fp['n_healthy_mango_frames']} mango-frames"),
    ]
    lines = [f"# Field evaluation {r['session']}", "", f"{r['n_annotated_frames']} annotated frames, ROI mode {r['roi_mode']}.", "",
             "| metric | value | note |", "|---|---:|---|"]
    lines += [f"| {name} | {_fmt(v, 3)} | {note} |" for name, v, note in rows]
    lines += ["", f"Unmatched predicted mangos: {pm['n_unmatched_predicted_mangos']}; missed GT mangos: {pm['n_missed_gt_mangos']}."]
    return "\n".join(lines) + "\n"


def multi_eval_markdown(results: dict[str, dict]) -> str:
    """One row per NMS IoU threshold with the headline metrics."""
    first = next(iter(results.values()))
    lines = [
        f"# Field evaluation {first['session']}: NMS IoU sweep", "",
        f"{first['n_annotated_frames']} annotated frames, ROI mode {first['roi_mode']}.", "",
        "| NMS IoU | box AP50 | box AP30 | mask AP50 | mango AP50 | AUROC | recall | specificity | FP / healthy track |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, r in results.items():
        pm = r["per_mango"]
        cells = [
            r["damage_box_ap50"]["ap"], r["damage_box_ap30"]["ap"], r["damage_mask_ap50"]["ap"], r["mango_ap50"]["ap"],
            pm["auroc"], pm["recall"], pm["specificity"], pm["fp_damage_boxes_per_healthy_mango"]["per_track"],
        ]
        lines.append(f"| {key} | " + " | ".join(_fmt(c, 3) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def cmd_eval(args) -> int:
    session = load_session(args.session)
    gt, skipped = load_ground_truth(args.gt, session)
    if not gt:
        print(f"no annotated frame matched the session (skipped: {skipped})", file=sys.stderr)
        return 1
    thresholds = _thresholds(args)
    try:
        results = {}
        for t in thresholds:
            r = evaluate(_prepare(session, t), gt)
            r["skipped_tasks"] = skipped
            results["recorded" if t is None else threshold_key(t)] = r
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(thresholds) == 1:
        result = next(iter(results.values()))
        text = eval_markdown(result)
        (out / "eval.json").write_text(json.dumps(result, indent=2))
    else:
        text = multi_eval_markdown(results)
        (out / "eval.json").write_text(json.dumps({"nms_iou": results}, indent=2))
    (out / "eval.md").write_text(text)
    print(text)
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="summary without ground truth")
    r.add_argument("session", help="session folder or .zip")
    r.add_argument("--out", required=True, help="folder for report.json and report.md")
    r.add_argument("--conf", type=float, default=0.25, help="operational damage confidence (default 0.25), even if recorded lower")
    r.add_argument("--nms-iou", type=parse_thresholds, metavar="IOU[,IOU...]", help=NMS_HELP)
    r.set_defaults(func=cmd_report)

    e = sub.add_parser("export-ls", help="Label Studio import file with predictions as pre-annotations")
    e.add_argument("session")
    e.add_argument("--out", required=True, help="folder for tasks.json, labeling_config.xml and frames/")
    g = e.add_mutually_exclusive_group()
    g.add_argument("--every", type=int, help="every Nth saved frame (default 10)")
    g.add_argument("--per-track", type=int, help="K evenly spaced frames per mango track")
    e.add_argument("--nms-iou", type=parse_thresholds, metavar="IOU[,IOU...]", help=NMS_HELP)
    e.add_argument("--image-url-prefix", default="/data/local-files/?d=frames/",
                   help="prefix of data.image in each task (default for Label Studio local files)")
    e.set_defaults(func=cmd_export_ls)

    v = sub.add_parser("eval", help="metrics against a Label Studio export")
    v.add_argument("session")
    v.add_argument("--gt", required=True, help="Label Studio JSON export of the annotated frames")
    v.add_argument("--out", required=True, help="folder for eval.json and eval.md")
    v.add_argument("--nms-iou", type=parse_thresholds, metavar="IOU[,IOU...]", help=NMS_HELP)
    v.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
