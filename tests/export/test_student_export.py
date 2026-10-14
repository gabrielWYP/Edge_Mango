"""Tests for the student mobile-export package (src/export/).

Toolchain-free: runs in the training venv. Everything that needs
litert-torch / ai-edge-litert lives in test_tflite_export.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.export.layout import (
    BOXES_OUTPUT_NAME,
    CLASS_OUTPUT_NAME,
    anchor_layout,
    flatten_levels,
    num_anchors,
    unflatten_levels,
)
from src.export.parity import ParityReport, match_detections
from src.export.reference import (
    LetterboxInfo,
    decode_exported_outputs,
    preprocess_bgr,
    unletterbox_boxes,
)
from src.export.student_export import (
    METADATA_SCHEMA_VERSION,
    StudentExportWrapper,
    build_metadata,
    load_student_checkpoint,
    resolve_decode_settings,
    write_metadata,
)
from src.models.student.student_model import StudentModel
from src.training.dataset import YOLODataset
from src.training.decode import decode_detections
from src.training.strides import STUDENT_STRIDES

IMAGE_SIZE = 640
N_ANCHORS = 80 * 80 + 40 * 40 + 20 * 20


@pytest.fixture(scope="module")
def student() -> StudentModel:
    torch.manual_seed(0)
    return StudentModel().eval()


@pytest.fixture(scope="module")
def wrapper(student) -> StudentExportWrapper:
    return StudentExportWrapper(student).eval()


def _fake_io_names() -> dict:
    return {
        "signature_key": "serving_default",
        "input": {"signature_name": "args_0", "tensor_name": "serving_default_args_0"},
        "outputs": {
            BOXES_OUTPUT_NAME: {"signature_name": "output_0", "tensor_name": "t0"},
            CLASS_OUTPUT_NAME: {"signature_name": "output_1", "tensor_name": "t1"},
        },
    }


class TestLayout:
    def test_anchor_layout_offsets_are_contiguous_finest_first(self) -> None:
        levels = anchor_layout(IMAGE_SIZE, STUDENT_STRIDES)
        assert [level.stride for level in levels] == list(STUDENT_STRIDES)
        assert [(level.grid_h, level.offset) for level in levels] == [(80, 0), (40, 6400), (20, 8000)]
        assert num_anchors(IMAGE_SIZE, STUDENT_STRIDES) == N_ANCHORS

    def test_non_divisible_image_size_raises(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            anchor_layout(100, STUDENT_STRIDES)

    def test_unflatten_inverts_flatten(self) -> None:
        levels = [torch.randn(2, 3, g, g) for g in (80, 40, 20)]
        flat = flatten_levels(levels)
        assert flat.shape == (2, N_ANCHORS, 3)
        for original, restored in zip(levels, unflatten_levels(flat, IMAGE_SIZE, STUDENT_STRIDES)):
            assert torch.equal(original, restored)

    def test_unflatten_rejects_wrong_anchor_count(self) -> None:
        with pytest.raises(ValueError, match="anchors"):
            unflatten_levels(torch.zeros(1, 100, 4), IMAGE_SIZE, STUDENT_STRIDES)


class TestWrapper:
    def test_returns_two_plain_tensors_with_documented_shapes(self, wrapper) -> None:
        with torch.no_grad():
            out = wrapper(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert isinstance(out, tuple) and len(out) == 2
        boxes_raw, cls_logits = out
        assert isinstance(boxes_raw, torch.Tensor) and isinstance(cls_logits, torch.Tensor)
        assert boxes_raw.shape == (1, N_ANCHORS, 4)
        assert cls_logits.shape == (1, N_ANCHORS, 2)
        assert boxes_raw.dtype == cls_logits.dtype == torch.float32

    def test_anchor_index_maps_to_level_grid_cell(self, student, wrapper) -> None:
        """Anchor i = offset_l + gy * grid_w + gx holds head output (l, gy, gx)."""
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            raw = student(x)
            boxes_raw, cls_logits = wrapper(x)
        for level_idx, (gy, gx) in [(0, (7, 11)), (1, (3, 5)), (2, (19, 0))]:
            level = anchor_layout(IMAGE_SIZE, STUDENT_STRIDES)[level_idx]
            i = level.offset + gy * level.grid_w + gx
            assert torch.equal(boxes_raw[0, i], raw["reg_preds"][level_idx][0, :, gy, gx])
            assert torch.equal(cls_logits[0, i], raw["cls_preds"][level_idx][0, :, gy, gx])

    def test_outputs_are_raw_head_values_not_decoded(self, student, wrapper) -> None:
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            raw = student(x)
            boxes_raw, cls_logits = wrapper(x)
        assert torch.equal(boxes_raw, flatten_levels(raw["reg_preds"]))
        assert torch.equal(cls_logits, flatten_levels(raw["cls_preds"]))

    def test_has_no_adapter_or_distillation_parameters(self, wrapper) -> None:
        names = [name for name, _ in wrapper.named_parameters()]
        assert names and all(name.startswith("student.") for name in names)
        assert not any("kd_proj" in name or "distill" in name or "proj" in name for name in names)
        n_wrapper = sum(p.numel() for p in wrapper.parameters())
        assert n_wrapper == sum(p.numel() for p in StudentModel().parameters())


class TestCheckpointLoading:
    def _kd_style_checkpoint(self, student: StudentModel) -> dict:
        state = dict(student.state_dict())
        state["kd_proj_backbone.projections.0.0.weight"] = torch.zeros(1)
        state["kd_proj_head_reg.projections.2.0.bias"] = torch.zeros(1)
        return {
            "epoch": 3,
            "best_map50": 0.5,
            "model_state_dict": state,
            "config": {"model_type": "student", "conf_threshold": 0.3, "num_classes": 2},
        }

    def test_kd_checkpoint_strips_legacy_adapter_keys(self, student, tmp_path: Path) -> None:
        path = tmp_path / "kd.pt"
        torch.save(self._kd_style_checkpoint(student), path)
        loaded = load_student_checkpoint(path)
        assert loaded.legacy_kd_keys_stripped == 2
        assert loaded.checkpoint_info == {"epoch": 3, "best_map50": 0.5}
        assert len(loaded.checkpoint_sha256) == 64
        for key, value in student.state_dict().items():
            assert torch.equal(loaded.model.state_dict()[key], value)

    def test_raw_state_dict_loads(self, student, tmp_path: Path) -> None:
        path = tmp_path / "raw.pt"
        torch.save(student.state_dict(), path)
        assert load_student_checkpoint(path).legacy_kd_keys_stripped == 0

    def test_unknown_extra_key_still_fails_strict_load(self, student, tmp_path: Path) -> None:
        checkpoint = self._kd_style_checkpoint(student)
        checkpoint["model_state_dict"]["head.extra.weight"] = torch.zeros(1)
        path = tmp_path / "bad.pt"
        torch.save(checkpoint, path)
        with pytest.raises(RuntimeError, match="Unexpected key"):
            load_student_checkpoint(path)

    def test_teacher_checkpoint_is_rejected(self, student, tmp_path: Path) -> None:
        checkpoint = self._kd_style_checkpoint(student)
        checkpoint["config"]["model_type"] = "master"
        path = tmp_path / "teacher.pt"
        torch.save(checkpoint, path)
        with pytest.raises(ValueError, match="only student"):
            load_student_checkpoint(path)

    def test_decode_settings_come_from_checkpoint_then_config_defaults(self) -> None:
        settings = resolve_decode_settings({"conf_threshold": 0.3})
        assert settings.conf_threshold == 0.3
        assert settings.nms_iou_threshold == 0.5
        assert settings.max_detections == 300
        assert settings.per_class_candidates is True


class TestReferenceDecode:
    def _synthetic_levels(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        torch.manual_seed(1)
        cls_levels, reg_levels = [], []
        for grid in (80, 40, 20):
            cls = torch.full((1, 2, grid, grid), -8.0)
            reg = torch.randn(1, 4, grid, grid) * 3.0
            reg[:, 2:] = torch.log(torch.full((1, 2, grid, grid), 40.0)) + torch.randn(1, 2, grid, grid) * 0.3
            idx = torch.randint(0, grid * grid, (12,))
            cls.view(1, 2, -1)[0, 0, idx[:6]] = torch.rand(6) * 6 - 1
            cls.view(1, 2, -1)[0, 1, idx[6:]] = torch.rand(6) * 6 - 1
            cls_levels.append(cls)
            reg_levels.append(reg)
        return cls_levels, reg_levels

    def test_matches_training_decode_on_same_tensors(self) -> None:
        cls_levels, reg_levels = self._synthetic_levels()
        preds = [torch.cat([c, r], dim=1) for c, r in zip(cls_levels, reg_levels)]
        kwargs = dict(conf_threshold=0.25, nms_iou_threshold=0.5, max_detections=300)
        expected = decode_detections(
            preds, cls_levels, batch_idx=0, num_classes=2, image_size=IMAGE_SIZE,
            strides=STUDENT_STRIDES, nms_enabled=True, per_class_candidates=True,
            normalize=False, **kwargs,
        )
        got = decode_exported_outputs(
            flatten_levels(reg_levels).numpy(), flatten_levels(cls_levels).numpy(),
            image_size=IMAGE_SIZE, strides=STUDENT_STRIDES, **kwargs,
        )
        assert expected[0].shape[0] > 0
        for e, g in zip(expected, got):
            assert torch.equal(e, g)

    def test_hand_computed_single_anchor(self) -> None:
        """cx = (gx + 0.5) * stride + dx, w = exp(log_w), score = sigmoid."""
        boxes_raw = torch.zeros(1, N_ANCHORS, 4)
        cls_logits = torch.full((1, N_ANCHORS, 2), -10.0)
        level = anchor_layout(IMAGE_SIZE, STUDENT_STRIDES)[1]  # stride 16
        gy, gx = 4, 9
        i = level.offset + gy * level.grid_w + gx
        boxes_raw[0, i] = torch.tensor([2.0, -3.0, np.log(50.0), np.log(20.0)])
        cls_logits[0, i, 1] = 2.0
        boxes, scores, labels = decode_exported_outputs(
            boxes_raw, cls_logits, image_size=IMAGE_SIZE, strides=STUDENT_STRIDES,
            conf_threshold=0.25, nms_iou_threshold=0.5, max_detections=300,
        )
        assert labels.tolist() == [1]
        assert scores.item() == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())
        assert boxes[0].tolist() == pytest.approx([(gx + 0.5) * 16 + 2.0, (gy + 0.5) * 16 - 3.0, 50.0, 20.0], rel=1e-5)

    def test_unletterbox_maps_back_to_original_pixels(self) -> None:
        info = LetterboxInfo(orig_w=1280, orig_h=960, scale=0.5, pad_x=0, pad_y=80)
        xyxy = unletterbox_boxes(torch.tensor([[320.0, 320.0, 100.0, 50.0]]), info)
        assert xyxy[0].tolist() == pytest.approx([540.0, 430.0, 740.0, 530.0])


class TestPreprocessing:
    def test_matches_yolo_dataset_eval_path(self, tmp_path: Path) -> None:
        """preprocess_bgr must produce the exact tensor YOLODataset(val) feeds the model."""
        rng = np.random.default_rng(0)
        for sub in ("rgb", "nir", "labels/val"):
            (tmp_path / sub).mkdir(parents=True)
        image = rng.integers(0, 255, size=(300, 500, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "rgb" / "mango_rgb_00001.jpg"), image)
        cv2.imwrite(str(tmp_path / "nir" / "mango_nir_00001.jpg"), image[:, :, 0])
        (tmp_path / "labels" / "val" / "mango_rgb_00001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        dataset = YOLODataset(
            rgb_dir=tmp_path / "rgb", nir_dir=tmp_path / "nir", labels_dir=tmp_path / "labels",
            split="val", image_size=IMAGE_SIZE, letterbox_value=114,
            nir_homography_path=None,
        )
        expected = dataset[0]["rgb"]
        got, info = preprocess_bgr(cv2.imread(str(tmp_path / "rgb" / "mango_rgb_00001.jpg")), IMAGE_SIZE, 114)
        assert got.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE) and got.dtype == torch.float32
        assert torch.equal(got[0], expected)
        assert (info.scale, info.pad_x, info.pad_y) == (IMAGE_SIZE / 500, 0, (IMAGE_SIZE - 384) // 2)


class TestMetadata:
    def test_schema(self, tmp_path: Path) -> None:
        model_file = tmp_path / "student_fp32.tflite"
        model_file.write_bytes(b"fake")
        metadata = build_metadata(
            model_file=model_file, precision="fp32", image_size=IMAGE_SIZE, num_classes=2,
            decode=resolve_decode_settings({}), letterbox_value=114, io_names=_fake_io_names(),
            checkpoint={"path": "/x/best_model.pt", "sha256": "0" * 64}, num_parameters=123,
            tools={"python": "3.12"},
        )
        path = write_metadata(metadata, tmp_path / "student_fp32.json")
        loaded = json.loads(path.read_text())

        assert loaded["schema_version"] == METADATA_SCHEMA_VERSION
        assert set(loaded) >= {
            "model", "source_checkpoint", "signature_key", "input", "preprocessing",
            "outputs", "anchors", "decode", "class_names", "num_classes", "tools", "exported_at",
        }
        assert loaded["model"]["file_size_bytes"] == 4 and len(loaded["model"]["file_sha256"]) == 64
        assert loaded["input"]["shape"] == [1, IMAGE_SIZE, IMAGE_SIZE, 3]
        assert loaded["input"]["layout"] == "NHWC" and loaded["input"]["signature_name"] == "args_0"
        norm = loaded["preprocessing"]["normalization"]
        assert norm["mean"] == [0.485, 0.456, 0.406] and norm["std"] == [0.229, 0.224, 0.225]
        assert loaded["preprocessing"]["letterbox"]["pad_value"] == 114
        outputs = {o["logical_name"]: o for o in loaded["outputs"]}
        assert outputs[BOXES_OUTPUT_NAME]["shape"] == [1, N_ANCHORS, 4]
        assert outputs[BOXES_OUTPUT_NAME]["signature_name"] == "output_0"
        assert outputs[CLASS_OUTPUT_NAME]["channels"] == ["mango", "damage"]
        levels = loaded["anchors"]["levels"]
        assert [lv["stride"] for lv in levels] == list(STUDENT_STRIDES)
        assert sum(lv["count"] for lv in levels) == loaded["anchors"]["num_anchors"] == N_ANCHORS
        assert loaded["decode"]["conf_threshold"] == 0.25
        assert loaded["decode"]["nms"]["iou_threshold"] == 0.5
        assert loaded["decode"]["max_detections"] == 300


class TestParityHelpers:
    def test_match_detections_requires_same_class_and_iou(self) -> None:
        ref = torch.tensor([[100.0, 100.0, 40.0, 40.0], [300.0, 300.0, 20.0, 20.0]])
        cand = torch.tensor([[100.5, 100.0, 40.0, 40.0], [300.0, 300.0, 20.0, 20.0]])
        match = match_detections(ref, torch.tensor([0, 1]), cand, torch.tensor([0, 0]))
        assert (match.matched, match.only_reference, match.only_candidate) == (1, 1, 1)

    def test_report_fails_on_large_raw_diff(self) -> None:
        report = ParityReport(max_abs_diff_threshold=1e-3)
        report.add_raw(BOXES_OUTPUT_NAME, np.zeros((1, 4, 4)), np.full((1, 4, 4), 2e-3))
        report.add_detections(match_detections(torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, 4), torch.zeros(0)))
        assert report.failures() and not report.to_dict()["passed"]

    def test_report_passes_on_identical_outputs(self) -> None:
        report = ParityReport()
        report.add_raw(CLASS_OUTPUT_NAME, np.ones((1, 4, 2)), np.ones((1, 4, 2)))
        boxes = torch.tensor([[10.0, 10.0, 5.0, 5.0]])
        report.add_detections(match_detections(boxes, torch.tensor([0]), boxes, torch.tensor([0])))
        assert report.to_dict()["passed"] and report.match_rate == 1.0
