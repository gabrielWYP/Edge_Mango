"""Calibration on the real selection with the module itself: distribution, null distribution of wrong pairs."""
import csv, json, sys, random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, numpy as np, yaml
WT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WT))
from src.annotation.captura_v3_tasks import load_wb_timestamps
from src.data_pipeline.v4_dataset import raw_nir, registered_nir, read_nir, fruit_mask, _rgb_source
from src.data_pipeline.v4_residual import measure_fruit_match
from src.data_pipeline.v4_selection import *
from src.training.nir_registration import load_homography
DATA = Path("data/cache/captura_v3")
def job(p):
    photo, other = p
    rgb_src = _rgb_source(photo, DATA)
    mask = fruit_mask(rgb_src)
    c = photo.capture
    rec = dict(ts=c.timestamp, lot=lot_number(c.jornada), ronda=hour_label(c.ronda_h), mango_id=c.mango_id, cara=c.cara, view=photo.label.view if photo.label else "")
    if mask is None:
        return rec | dict(fail=1)
    rgb = cv2.imread(str(rgb_src), cv2.IMREAD_GRAYSCALE)
    nir_ts = other.capture.timestamp if other else c.timestamp
    H = load_homography(WT / (other.homography if other else photo.homography))
    warped = registered_nir(raw_nir(read_nir(DATA / "news/nir" / f"mango_nir_{nir_ts}.jpg")), H)
    m = measure_fruit_match(rgb, warped, mask)
    rec |= dict(fail=0, dx=m.dx, dy=m.dy, corr=round(m.corr, 4), zero=round(m.corr_zero, 4), lim=int(m.at_limit), sx=m.surface[0] if m.surface else "", sy=m.surface[1] if m.surface else "", other=nir_ts if other else "")
    return rec
def main():
    cv2.setNumThreads(1)
    metas = [json.loads(p.read_text()) for p in sorted((DATA / "news/meta").glob("mango_meta_*.json"))]
    labels, _ = parse_tasks(json.loads((WT / "data/labels/captura_v3_lots1-5_12h_0h.json").read_text()))
    entries = yaml.safe_load((WT / "configs/data/v4_identity_exclusions.yaml").read_text())["exclusions"] or []
    identity = {int(e["timestamp"]): str(e["reason"]) for e in entries}
    rules = load_homography_rules(WT / "configs/data/v4_homographies.yaml")
    tol = yaml.safe_load((WT / "configs/data/v4_tolerance_exceptions.yaml").read_text())["exceptions"] or []
    photos, _ = select_photos(metas, labels, identity, load_wb_timestamps(DATA), rules, tol)
    print(len(photos), flush=True)
    rng = random.Random(11)
    if sys.argv[2] == "real":
        jobs = [(p, None) for p in photos]
    else:  # wrong pairs: same lot, another mango
        byl = {}
        for p in photos:
            byl.setdefault(lot_number(p.capture.jornada), []).append(p)
        jobs = []
        for p in rng.sample(photos, 600):
            cand = [q for q in byl[lot_number(p.capture.jornada)] if q.capture.mango_id != p.capture.mango_id]
            jobs.append((p, rng.choice(cand)))
    with ProcessPoolExecutor(10) as pool:
        recs = list(pool.map(job, jobs, chunksize=4))
    fields = sorted({k for r in recs for k in r})
    with open(sys.argv[1], "w", newline="") as h:
        w = csv.DictWriter(h, fields); w.writeheader(); w.writerows(recs)
main()
