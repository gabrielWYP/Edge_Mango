"""scripts/eval_tflite.py: a .tflite over a fold split, writing evaluate_checkpoint.py's predictions.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

pytest.importorskip("litert_torch")
pytest.importorskip("ai_edge_litert")

from scripts import eval_tflite, export_student  # noqa: E402
from src.export.student_export import StudentExportWrapper, make_tflite_runner, make_torch_runner  # noqa: E402
from src.models.student.student_model import StudentModel  # noqa: E402

SIZE = 64
SEED = 3


def _dataset(root: Path) -> tuple[Path, Path]:
    rgb_dir, labels_dir = root / "rgb", root / "labels"
    (labels_dir / "val").mkdir(parents=True)
    rgb_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        stem = f"mango_rgb_{i:03d}"
        cv2.imwrite(str(rgb_dir / f"{stem}.jpg"), rng.integers(0, 255, (72, 128, 3), dtype=np.uint8))
        (labels_dir / "val" / f"{stem}.txt").write_text("0 0.5 0.5 0.4 0.6\n1 0.52 0.5 0.1 0.1\n")
    return rgb_dir, labels_dir


@pytest.fixture(scope="module")
def segment_export(tmp_path_factory) -> tuple[Path, Path, Path, Path]:
    root = tmp_path_factory.mktemp("eval")
    rgb_dir, labels_dir = _dataset(root)
    rc = export_student.main(["--random-init", "--seed", str(SEED), "--task", "segment", "--variant", "nano",
                              "--image-size", str(SIZE), "--out-dir", str(root / "model")])
    assert rc == 0
    return root / "model" / "student_fp32.tflite", rgb_dir, labels_dir, root


def test_predictions_follow_the_evaluate_checkpoint_schema(segment_export) -> None:
    model, rgb_dir, labels_dir, root = segment_export
    out = root / "out"
    rc = eval_tflite.main(["--model", str(model), "--rgb-dir", str(rgb_dir), "--labels-dir", str(labels_dir),
                           "--split", "val", "--output-dir", str(out)])
    assert rc == 0
    lines = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert [line["stem"] for line in lines] == [f"mango_rgb_{i:03d}" for i in range(3)]
    detections = [box for line in lines for box in line["boxes"]]
    assert detections, "a random-init model at conf 0.001 still emits candidates"
    for box in detections:
        assert set(box) <= {"class", "confidence", "xyxy", "mask"}
        assert box["class"] in (0, 1) and 0.0 < box["confidence"] <= 1.0 and len(box["xyxy"]) == 4
        assert ("mask" in box) == (box["class"] == 1)  # segment: damage detections carry a polygon (maybe [])
        if box.get("mask"):
            assert len(box["mask"]) % 2 == 0
    assert (out / "eval_tflite.json").exists()


def test_limit_and_missing_split_are_handled(segment_export) -> None:
    model, rgb_dir, labels_dir, root = segment_export
    out = root / "limited"
    assert eval_tflite.main(["--model", str(model), "--rgb-dir", str(rgb_dir), "--labels-dir", str(labels_dir),
                             "--limit", "2", "--output-dir", str(out)]) == 0
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 2
    with pytest.raises(FileNotFoundError):
        eval_tflite.main(["--model", str(model), "--rgb-dir", str(rgb_dir), "--labels-dir", str(labels_dir),
                          "--split", "test", "--output-dir", str(root / "none")])


def test_tflite_and_pytorch_predictions_agree(segment_export) -> None:
    """The same decode + mapping over the .tflite and over the PyTorch wrapper gives the same detections."""
    model, rgb_dir, labels_dir, root = segment_export
    metadata = json.loads(model.with_suffix(".json").read_text())
    torch.manual_seed(SEED)
    wrapper = StudentExportWrapper(StudentModel(variant="nano", task="segment").eval())
    io_names = {"signature_key": metadata["signature_key"], "input": metadata["input"],
                "outputs": {o["logical_name"]: o for o in metadata["outputs"]}}
    tflite = make_tflite_runner(model, io_names, [o["logical_name"] for o in metadata["outputs"]])
    torch_run = make_torch_runner(wrapper)
    image_path = next(rgb_dir.glob("*.jpg"))
    label_path = labels_dir / "val" / f"{image_path.stem}.txt"
    from_tflite = eval_tflite.predict_image(tflite, image_path, label_path, metadata)
    from_torch = eval_tflite.predict_image(
        lambda nhwc: torch_run(torch.from_numpy(nhwc).permute(0, 3, 1, 2).contiguous()), image_path, label_path, metadata
    )
    a, b = json.loads(from_tflite)["boxes"], json.loads(from_torch)["boxes"]
    assert len(a) == len(b)
    for x, y in zip(a[:10], b[:10]):
        assert x["class"] == y["class"] and x["confidence"] == pytest.approx(y["confidence"], abs=1e-4)
