"""Intermediate (P3/P4/P5) fusion of two separate backbones, `student_fusion="mid"`."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from src.models.student.pretrained import load_yolov8n_weights, official_key
from src.models.student.student_model import StudentModel

SIZE = 64


def _inputs(seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, 3, SIZE, SIZE, generator=g), torch.rand(2, 1, SIZE, SIZE, generator=g)


def _mid(task: str = "segment", variant: str = "nano") -> StudentModel:
    torch.manual_seed(0)
    return StudentModel(num_classes=2, variant=variant, in_channels=4, task=task, fusion="mid").eval()


def _flat(out: dict) -> list[torch.Tensor]:
    tensors = []
    for value in out.values():
        tensors.extend(value if isinstance(value, list) else [value])
    return tensors


@pytest.mark.parametrize("variant", ["nano", "wide"])
def test_output_shapes_equal_those_of_early_fusion(variant: str) -> None:
    x, nir = _inputs()
    early = StudentModel(num_classes=2, variant=variant, in_channels=4, task="segment").eval()
    mid = _mid(variant=variant)
    with torch.no_grad():
        a, b = early(x, nir), mid(x, nir)
    assert a.keys() == b.keys()
    for key in a:
        sa = [t.shape for t in (a[key] if isinstance(a[key], list) else [a[key]])]
        sb = [t.shape for t in (b[key] if isinstance(b[key], list) else [b[key]])]
        assert sa == sb, key


def test_each_modality_changes_the_outputs() -> None:
    # Train mode: batch statistics keep the signal at O(1); with fresh running stats an untrained net shrinks it to ~1e-9.
    model = _mid().train()
    x, nir = _inputs()
    with torch.no_grad():
        base = model(x, nir)["preds"]
        nir_only = model(x, torch.rand_like(nir))["preds"]
        rgb_only = model(torch.rand_like(x), nir)["preds"]
    assert any(not torch.allclose(a, b) for a, b in zip(base, nir_only))
    assert any(not torch.allclose(a, b) for a, b in zip(base, rgb_only))


def test_mid_needs_the_nir_tensor_and_four_channels() -> None:
    with pytest.raises(ValueError, match="NIR"):
        _mid()(torch.zeros(1, 3, SIZE, SIZE))
    with pytest.raises(ValueError, match="fusion"):
        StudentModel(num_classes=2, in_channels=3, fusion="mid")
    with pytest.raises(ValueError, match="fusion"):
        StudentModel(num_classes=2, in_channels=4, fusion="late")


@pytest.mark.parametrize("variant", ["nano", "wide"])
def test_every_parameter_of_both_backbones_and_the_fusion_gets_a_gradient(variant: str) -> None:
    model = _mid(variant=variant).train()
    x, nir = _inputs()
    out = model(x, nir)
    sum(t.pow(2).sum() for t in _flat(out)).backward()
    dead = [
        name for name, p in model.named_parameters()
        if name.startswith("backbone.") and (p.grad is None or not p.grad.abs().sum() > 0)
    ]
    assert dead == []
    groups = {name.split(".")[1] for name, _ in model.named_parameters() if name.startswith("backbone.")}
    assert groups == {"rgb", "nir", "fuse"}


def test_backbones_are_separate_with_their_own_stem_channels() -> None:
    model = _mid()
    assert model.backbone.rgb.stem[0].conv.weight.shape[1] == 3
    assert model.backbone.nir.stem[0].conv.weight.shape[1] == 1
    assert model.backbone.rgb.stem[0].conv.weight is not model.backbone.nir.stem[0].conv.weight


def test_fusion_convs_restore_the_channels_of_each_level() -> None:
    fuse = _mid().backbone.fuse
    assert [(f.conv.in_channels, f.conv.out_channels, f.conv.kernel_size) for f in fuse] == [
        (128, 64, (1, 1)), (256, 128, (1, 1)), (512, 256, (1, 1)),
    ]


# Recorded on origin/main before `fusion` existed: nano, in_channels=4, seed 0.
EARLY_GOLDEN = {
    "detect": (354, "3aae29aed01491f2", 2_999_666, [-588.1754, -147.0438, -36.761]),
    "segment": (416, "6a3be7f3fe6b24c1", 3_354_962, [-588.1754, -147.0438, -36.761]),
}


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_early_fusion_is_unchanged(task: str) -> None:
    torch.manual_seed(0)
    model = StudentModel(num_classes=2, variant="nano", in_channels=4, task=task).eval()
    n_keys, key_hash, n_params, pred_sums = EARLY_GOLDEN[task]
    keys = list(model.state_dict())
    assert len(keys) == n_keys
    assert hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16] == key_hash
    assert sum(p.numel() for p in model.parameters()) == n_params
    x, nir = _inputs()
    g = torch.Generator().manual_seed(1)
    x, nir = torch.rand(1, 3, SIZE, SIZE, generator=g), torch.rand(1, 1, SIZE, SIZE, generator=g)
    with torch.no_grad():
        got = [round(float(p.sum()), 4) for p in model(x, nir)["preds"]]
    assert got == pytest.approx(pred_sums, abs=1e-3)


def _official_state(model: StudentModel) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for key, value in StudentModel(num_classes=2, variant="nano", task="segment").state_dict().items():
        name = official_key(key, "nano")
        if name is not None:
            state[name] = torch.randn(value.shape) if value.dim() else value.clone()
    return state


def test_pretrained_init_gives_rgb_the_rgb_weights_and_nir_their_channel_sum(tmp_path: Path) -> None:
    model = _mid()
    path = tmp_path / "state.pt"
    official = _official_state(model)
    torch.save(official, path)
    report = load_yolov8n_weights(model, path)
    stem = official["model.0.conv.weight"]
    assert torch.equal(model.backbone.rgb.stem[0].conv.weight, stem)
    assert torch.allclose(model.backbone.nir.stem[0].conv.weight, stem.sum(dim=1, keepdim=True))
    assert torch.equal(model.backbone.nir.stage3[0].conv.weight, official["model.5.conv.weight"])
    assert torch.equal(model.backbone.rgb.stage3[0].conv.weight, official["model.5.conv.weight"])
    # Only the fusion convs are left at their default init.
    assert [k for k in report["skipped"] if k.startswith("backbone.")] == [
        k for k in report["skipped"] if k.startswith("backbone.fuse.")
    ]


def test_inflated_nir_stem_equals_a_gray_image_replicated_to_three_channels(tmp_path: Path) -> None:
    model = _mid()
    path = tmp_path / "state.pt"
    torch.save(_official_state(model), path)
    load_yolov8n_weights(model, path)
    gray = torch.rand(1, 1, SIZE, SIZE)
    nir_stem = model.backbone.nir.stem[0].conv
    rgb_stem = model.backbone.rgb.stem[0].conv
    assert torch.allclose(nir_stem(gray), rgb_stem(gray.repeat(1, 3, 1, 1)), atol=1e-5)


def test_early_fusion_loader_is_unchanged(tmp_path: Path) -> None:
    model = StudentModel(num_classes=2, variant="nano", in_channels=4, task="segment")
    path = tmp_path / "state.pt"
    official = _official_state(model)
    torch.save(official, path)
    load_yolov8n_weights(model, path)
    stem = official["model.0.conv.weight"]
    assert torch.equal(model.backbone.stem[0].conv.weight[:, :3], stem)
    assert torch.allclose(model.backbone.stem[0].conv.weight[:, 3:], stem.mean(dim=1, keepdim=True))


def test_checkpoint_round_trip_rebuilds_mid(tmp_path: Path) -> None:
    from scripts.evaluate_checkpoint import _load_model
    from src.training.config import TrainingConfig

    model = _mid(task="segment")
    config = TrainingConfig(
        model_type="student", student_variant="nano", student_in_channels=4, student_fusion="mid", task="segment",
    )
    ckpt = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__.copy()}, ckpt)

    # The evaluator is given no fusion flag: a fresh default config must not decide it.
    fresh = TrainingConfig(model_type="student", student_variant="nano", student_in_channels=4)
    loaded = _load_model("student", str(ckpt), fresh, torch.device("cpu"))
    assert loaded.fusion == "mid"
    x, nir = _inputs()
    with torch.no_grad():
        for a, b in zip(model(x, nir)["preds"], loaded.eval()(x, nir)["preds"]):
            assert torch.allclose(a, b)


def test_config_validates_the_fusion_field() -> None:
    from src.training.config import TrainingConfig

    assert TrainingConfig().student_fusion == "early"
    with pytest.raises(ValueError, match="student_fusion"):
        TrainingConfig(model_type="student", student_in_channels=4, student_fusion="late")
    with pytest.raises(ValueError, match="student_fusion"):
        TrainingConfig(model_type="student", student_in_channels=3, student_fusion="mid")
