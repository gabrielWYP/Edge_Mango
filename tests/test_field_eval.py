"""Tests for scripts/field_eval.py on synthetic sessions (the format written by the Android app)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts import field_eval as fe

W, H = 1000, 500
A = [100.0, 100.0, 300.0, 300.0]  # mango A
B = [600.0, 100.0, 800.0, 300.0]  # mango B
DMG_A = [150.0, 150.0, 200.0, 200.0]
SQUARE_A = [150.0, 150.0, 200.0, 150.0, 200.0, 200.0, 150.0, 200.0]  # flat x, y polygon of DMG_A


def mango(track, box, score=0.9, damages=(), held=None):
    ds = [{"box": b, "score": s, **({"polygon": p} if p else {})} for b, s, p in damages]
    m = {"trackId": track, "box": box, "score": score, "damages": ds, "maxDamage": max((d["score"] for d in ds), default=0.0)}
    if held is not None:  # sessions recorded before the hold feature have no "held" key
        m["held"] = held
    return m


def record(index, mangos, *, saved=True, no_mango=None, t=100.0):
    return {
        "index": index, "wallMs": 1_000_000 + index * 100, "cameraTimestampNs": index * 100_000_000,
        "frameW": W, "frameH": H, "roiMode": "DETECTOR", "confThreshold": 0.25, "mangoThreshold": 0.5, "nmsIou": 0.5,
        "agnosticNms": False, "maxMangos": 10, "frame": f"frames/f_{index:06d}.jpg" if saved else None,
        "accelerator": "CPU x4", "detectorModel": "det.tflite", "damageModel": "seg.tflite",
        "noMango": (not mangos) if no_mango is None else no_mango, "mangos": mangos,
        "timings": {"roiMs": 0.0, "preMs": 5.0, "inferMs": t - 10, "postMs": 5.0, "totalMs": t, "detectorMs": 20.0},
    }


def write_session(root: Path, records: list[dict], name="session_20261121_101500", with_frames=True) -> Path:
    d = root / name
    (d / "frames").mkdir(parents=True)
    (d / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (d / "session.json").write_text(json.dumps({"device": "Test Phone", "droppedWrites": 1}))
    if with_frames:
        for r in records:
            if r["frame"]:
                (d / r["frame"]).write_bytes(b"\xff\xd8jpeg")
    return d


@pytest.fixture
def basic(tmp_path):
    records = [
        record(0, [mango(1, A, 0.95, [(DMG_A, 0.9, None)]), mango(2, B, 0.8)], t=100.0),
        record(1, [mango(1, A, 0.95, [(DMG_A, 0.7, None)]), mango(2, B, 0.8)], t=120.0),
        record(2, [], t=80.0),
        record(3, [mango(1, A, 0.9)], t=100.0),
    ]
    return write_session(tmp_path, records)


def gt_task(idx, results):
    return {"data": {"image": f"/data/local-files/?d=frames/abc123-f_{idx:06d}.jpg"}, "annotations": [{"result": results}]}


def gt_rect(box, label):
    x1, y1, x2, y2 = box
    return {"type": "rectanglelabels", "value": {"x": x1 / W * 100, "y": y1 / H * 100, "width": (x2 - x1) / W * 100,
                                                  "height": (y2 - y1) / H * 100, "rotation": 0, "rectanglelabels": [label]}}


def gt_poly(flat, label):
    pts = [[flat[i] / W * 100, flat[i + 1] / H * 100] for i in range(0, len(flat), 2)]
    return {"type": "polygonlabels", "value": {"points": pts, "polygonlabels": [label]}}


def write_gt(path: Path, frames: dict) -> Path:
    path.write_text(json.dumps([gt_task(i, r) for i, r in frames.items()]))
    return path


# ---- loading ------------------------------------------------------------------------------------------------------


def test_load_session_folder_and_zip(basic, tmp_path):
    s = fe.load_session(basic)
    assert len(s.frames) == 4 and s.info["device"] == "Test Phone" and s.conf == 0.25

    z = tmp_path / "s.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in basic.rglob("*"):
            if f.is_file():
                zf.write(f, f"{basic.name}/{f.relative_to(basic)}")
    s2 = fe.load_session(z, workdir=tmp_path / "unzipped")
    assert [f["index"] for f in s2.frames] == [0, 1, 2, 3]


def test_load_session_without_predictions_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        fe.load_session(tmp_path)


# ---- report -------------------------------------------------------------------------------------------------------


def test_report_numbers(basic):
    r = fe.build_report(fe.load_session(basic))
    assert r["n_frames"] == 4
    assert r["pct_frames_with_mango"] == pytest.approx(75.0)
    assert r["mangos_per_frame"]["mean"] == pytest.approx((2 + 2 + 0 + 1) / 4)
    assert r["mangos_per_frame"]["max"] == 2
    assert r["n_tracks"] == 2
    t1, t2 = r["tracks"]
    assert (t1["track_id"], t1["frames_seen"]) == (1, 3)
    assert t1["pct_frames_with_damage"] == pytest.approx(200 / 3)  # damage in 2 of its 3 frames
    assert t1["max_damage"]["mean"] == pytest.approx((0.9 + 0.7 + 0.0) / 3)
    assert t1["max_damage"]["median"] == pytest.approx(0.7)
    assert t2["frames_seen"] == 2 and t2["pct_frames_with_damage"] == 0.0
    h = r["damage_confidence_histogram"]
    assert h["n_detections"] == 2 and sum(h["counts"]) == 2
    assert r["latency_ms"]["totalMs"]["mean"] == pytest.approx(100.0)
    assert r["latency_ms"]["totalMs"]["max"] == 120.0
    assert r["fps_wall"] == pytest.approx(10.0)  # 100 ms between frames
    assert r["dropped_writes"] == 1


def test_report_counts_held_and_detected_mango_frames(tmp_path):
    recs = [
        record(0, [mango(1, A, 0.9, held=False), mango(2, B, 0.8)]),  # no "held" key counts as detected
        record(1, [mango(1, A, 0.9, held=False), mango(2, B, 0.8, held=True)]),
        record(2, [mango(1, A, 0.4, held=True)]),
    ]
    for r_ in recs:
        r_["keepThreshold"], r_["holdFrames"] = 0.15, 5
    r = fe.build_report(fe.load_session(write_session(tmp_path, recs)))
    assert (r["n_mango_frames_detected"], r["n_mango_frames_held"]) == (3, 2)
    assert (r["keep_threshold"], r["hold_frames"]) == (0.15, 5)
    md = fe.report_markdown(r)
    assert "3 detected, 2 held" in md and "keep 0.15, hold 5 frames" in md


def test_report_cli_writes_json_and_markdown(basic, tmp_path):
    out = tmp_path / "out"
    assert fe.main(["report", str(basic), "--out", str(out)]) == 0
    assert json.loads((out / "report.json").read_text())["n_tracks"] == 2
    md = (out / "report.md").read_text()
    assert "Frames with at least one mango: 75.0 %" in md and "| 1 | 3 |" in md


def test_single_crop_session_uses_one_pseudo_mango(tmp_path):
    rec = record(0, [])
    rec.pop("noMango"), rec.pop("mangos")
    rec["detectorModel"] = None
    rec["roiMode"] = "FULL_FRAME"
    rec["detections"] = [{"cls": 0, "box": A, "score": 0.9}, {"cls": 1, "box": DMG_A, "score": 0.8}]
    d = write_session(tmp_path, [rec])
    r = fe.build_report(fe.load_session(d))
    assert r["n_tracks"] == 1 and r["tracks"][0]["pct_frames_with_damage"] == 100.0


# ---- export-ls ----------------------------------------------------------------------------------------------------


def test_sampling_every_and_per_track(basic):
    s = fe.load_session(basic)
    assert fe.sample_frames(s, every=2, per_track=None) == [0, 2]
    # track 1 is seen in frames 0, 1, 3 and track 2 in 0, 1: two evenly spaced frames each (first and last).
    assert fe.sample_frames(s, every=None, per_track=2) == [0, 1, 3]
    assert fe.sample_frames(s, every=None, per_track=1) == [0]


def test_unsaved_frames_are_not_sampled(tmp_path):
    d = write_session(tmp_path, [record(0, [mango(1, A)], saved=False), record(1, [mango(1, A)])])
    assert fe.sample_frames(fe.load_session(d), every=1, per_track=None) == [1]


def test_export_ls_tasks_carry_preannotations(basic, tmp_path):
    out = tmp_path / "ls"
    assert fe.main(["export-ls", str(basic), "--every", "1", "--out", str(out)]) == 0
    tasks = json.loads((out / "tasks.json").read_text())
    assert len(tasks) == 4
    assert (out / "frames" / "f_000000.jpg").exists() and (out / "labeling_config.xml").exists()
    t0 = tasks[0]
    assert t0["data"]["image"] == "/data/local-files/?d=frames/f_000000.jpg"
    res = t0["predictions"][0]["result"]
    kinds = [(r["type"], (r["value"].get("rectanglelabels") or r["value"].get("polygonlabels"))[0]) for r in res]
    assert kinds == [("rectanglelabels", "mango"), ("rectanglelabels", "damage"), ("rectanglelabels", "mango")]
    m = res[0]["value"]
    assert (m["x"], m["y"], m["width"], m["height"]) == pytest.approx((10.0, 20.0, 20.0, 40.0))
    assert "predictions" not in tasks[2]  # the frame without mango has nothing to pre-annotate


def test_export_ls_uses_polygons_when_the_session_has_them(tmp_path):
    d = write_session(tmp_path, [record(0, [mango(1, A, 0.9, [(DMG_A, 0.8, SQUARE_A)])])])
    tasks = fe.build_ls_tasks(fe.load_session(d), [0], "u/")
    res = tasks[0]["predictions"][0]["result"]
    poly = [r for r in res if r["type"] == "polygonlabels"]
    assert len(poly) == 1 and poly[0]["value"]["points"][0] == pytest.approx([15.0, 30.0])
    assert not [r for r in res if r["type"] == "rectanglelabels" and r["value"]["rectanglelabels"] == ["damage"]]


def test_export_ls_without_frames_fails(tmp_path):
    d = write_session(tmp_path, [record(0, [mango(1, A)], saved=False)])
    assert fe.main(["export-ls", str(d), "--out", str(tmp_path / "ls")]) == 1


# ---- ground truth -------------------------------------------------------------------------------------------------


def test_ground_truth_parsing(basic, tmp_path):
    s = fe.load_session(basic)
    gt_file = write_gt(tmp_path / "gt.json", {0: [gt_rect(A, "mango"), gt_rect(DMG_A, "Damage"), gt_poly(SQUARE_A, "damage")]})
    # An unannotated task, a cancelled one and a frame the session does not have.
    tasks = json.loads(gt_file.read_text())
    tasks.append({"data": {"image": "f_000001.jpg"}, "annotations": []})
    tasks.append({"data": {"image": "f_000002.jpg"}, "annotations": [{"was_cancelled": True, "result": []}]})
    tasks.append(gt_task(99, [gt_rect(A, "mango")]))
    gt_file.write_text(json.dumps(tasks))
    gt, skipped = fe.load_ground_truth(gt_file, s)
    assert list(gt) == [0]
    g = gt[0]
    assert g.mangos == [pytest.approx(A)]
    assert g.damages[0] == pytest.approx(DMG_A) and g.damages[1] == pytest.approx(DMG_A)
    assert g.damage_polygons[0] is None and g.damage_polygons[1].shape == (4, 2)
    assert skipped == {"no_annotation_or_unmatched_name": 2, "frame_not_in_session": 1}


def test_mango_matching_is_greedy_and_one_to_one():
    preds = [{"box": A}, {"box": [110.0, 110.0, 310.0, 310.0]}]
    assert fe.match_mangos(preds, [A]) == [(0, 0)]
    assert fe.match_mangos(preds, [[500.0, 0.0, 600.0, 50.0]]) == []


# ---- eval ---------------------------------------------------------------------------------------------------------


def perfect_gt(tmp_path):
    frame = [gt_rect(A, "mango"), gt_rect(B, "mango"), gt_rect(DMG_A, "damage")]
    return write_gt(tmp_path / "gt.json", {0: frame, 1: frame})


def test_perfect_predictions(basic, tmp_path):
    r = fe.evaluate(fe.load_session(basic), fe.load_ground_truth(perfect_gt(tmp_path), fe.load_session(basic))[0])
    assert r["n_annotated_frames"] == 2
    assert r["damage_box_ap50"]["ap"] == pytest.approx(1.0) and r["damage_box_ap50"]["n_gt"] == 2
    assert r["damage_box_ap30"]["ap"] == pytest.approx(1.0)
    assert r["mango_ap50"]["ap"] == pytest.approx(1.0) and r["mango_ap50"]["n_gt"] == 4
    assert r["damage_mask_ap50"]["ap"] is None  # no GT polygons
    pm = r["per_mango"]
    assert pm["n_positive_tracks"] == 1 and pm["n_negative_tracks"] == 1
    assert pm["auroc"] == pytest.approx(1.0)
    assert pm["recall"] == 1.0 and pm["specificity"] == 1.0
    assert pm["fp_damage_boxes_per_healthy_mango"]["per_track"] == 0.0
    assert pm["n_unmatched_predicted_mangos"] == 0 and pm["n_missed_gt_mangos"] == 0


def test_false_positive_on_a_healthy_mango(tmp_path):
    fp =([B[0], B[1], B[0] + 40, B[1] + 40], 0.95, None)
    recs = [record(i, [mango(1, A, 0.95, [(DMG_A, 0.9, None)]), mango(2, B, 0.8, [fp])]) for i in (0, 1)]
    d = write_session(tmp_path, recs)
    s = fe.load_session(d)
    gt, _ = fe.load_ground_truth(perfect_gt(tmp_path), s)
    r = fe.evaluate(s, gt)
    # Ranking: FP .95, FP .95, TP .9, TP .9 -> precision envelope 0.5 at both recall steps -> AP 0.5.
    assert r["damage_box_ap50"]["ap"] == pytest.approx(0.5)
    pm = r["per_mango"]
    assert pm["auroc"] == pytest.approx(0.0)  # the healthy mango scores above the damaged one
    assert pm["recall"] == 1.0 and pm["specificity"] == 0.0
    f = pm["fp_damage_boxes_per_healthy_mango"]
    assert f["per_track"] == 2.0 and f["per_mango_frame"] == 1.0 and f["n_fp_boxes"] == 2


def test_mango_ap_ignores_held_mangos_but_damage_uses_their_crops(tmp_path):
    # Frame 0: mango A detected. Frame 1: mango A held (its crop still finds the damage); a detected FP mango scores 0.99.
    far = [500.0, 0.0, 560.0, 50.0]
    recs = [
        record(0, [mango(1, A, 0.9, [(DMG_A, 0.9, None)], held=False)]),
        record(1, [mango(1, A, 0.9, [(DMG_A, 0.8, None)], held=True), mango(3, far, 0.99, held=False)]),
    ]
    d = write_session(tmp_path, recs)
    s = fe.load_session(d)
    gt, _ = fe.load_ground_truth(
        write_gt(tmp_path / "gt.json", {i: [gt_rect(A, "mango"), gt_rect(DMG_A, "damage")] for i in (0, 1)}), s)
    r = fe.evaluate(s, gt)
    assert r["damage_box_ap50"]["ap"] == pytest.approx(1.0)  # the held crop's damage counts
    # Mango ranking uses detected boxes only: FP .99, TP .9 -> precision 0.5 at recall 0.5 (the held GT mango is unmatched) -> AP 0.25.
    assert r["mango_ap50"]["n_gt"] == 2
    assert r["mango_ap50"]["ap"] == pytest.approx(0.25)


def test_missed_damage_and_missed_mango(tmp_path):
    recs = [record(0, [mango(1, A, 0.95)])]  # damage missed, mango B missed
    d = write_session(tmp_path, recs)
    s = fe.load_session(d)
    gt, _ = fe.load_ground_truth(write_gt(tmp_path / "gt.json", {0: [gt_rect(A, "mango"), gt_rect(B, "mango"), gt_rect(DMG_A, "damage")]}), s)
    r = fe.evaluate(s, gt)
    assert r["damage_box_ap50"]["ap"] == 0.0
    assert r["mango_ap50"]["ap"] == pytest.approx(0.5)  # one of two GT mangos found
    assert r["per_mango"]["recall"] == 0.0 and r["per_mango"]["n_missed_gt_mangos"] == 1
    assert r["per_mango"]["auroc"] is None  # no healthy track to rank against


def test_mask_ap_with_polygon_ground_truth(tmp_path):
    good = [record(0, [mango(1, A, 0.95, [(DMG_A, 0.9, SQUARE_A)])])]
    far = [150.0 + 500, 150.0, 200.0 + 500, 150.0, 200.0 + 500, 200.0, 150.0 + 500, 200.0]
    bad = [record(0, [mango(1, A, 0.95, [(DMG_A, 0.9, far)])])]
    for recs, expected in ((good, 1.0), (bad, 0.0)):
        d = write_session(tmp_path / f"s{expected}", recs)
        s = fe.load_session(d)
        gt, _ = fe.load_ground_truth(write_gt(tmp_path / f"gt{expected}.json", {0: [gt_rect(A, "mango"), gt_poly(SQUARE_A, "damage")]}), s)
        r = fe.evaluate(s, gt)
        assert r["damage_mask_ap50"]["ap"] == pytest.approx(expected)
        assert r["damage_mask_ap50"]["n_gt"] == 1
        assert r["damage_box_ap50"]["ap"] == pytest.approx(1.0)  # the polygon's bounding box equals the predicted box


def test_eval_cli(basic, tmp_path):
    out = tmp_path / "ev"
    assert fe.main(["eval", str(basic), "--gt", str(perfect_gt(tmp_path)), "--out", str(out)]) == 0
    assert json.loads((out / "eval.json").read_text())["damage_box_ap50"]["ap"] == pytest.approx(1.0)
    assert "Damage box AP50" in (out / "eval.md").read_text()

    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    assert fe.main(["eval", str(basic), "--gt", str(empty), "--out", str(out)]) == 1
