"""Modality ablation of plan E1 (`docs/tesis/plan-experimental.md`).

`maestro_rgb` is "RGB + NIR reemplazado por una constante (0 tras la
normalización)" and `maestro_nir` is "RGB reemplazado por una constante + NIR",
with the same two-stream architecture and identical parameters, so the arms
differ only in input information. The constant replaces the modality in
`YOLODataset` output, so a KD teacher sees it too.

Everything goes through the public entry points (`YOLODataset`,
`TrainingConfig`, `KDTrainer` and the `main()` of both CLIs). The CLIs only
replace `Trainer.fit`: datasets, loaders, the override loop and the model
construction are the real ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts import evaluate_checkpoint as ec
from src.models.master.master_model import MasterModel
from src.models.student.student_model import StudentModel
from src.training import kd_train as kd_train_module
from src.training import train as train_module
from src.training.config import TrainingConfig
from src.training.dataset import YOLODataset, letterbox, normalize_rgb
from src.training.kd_config import KDConfig
from src.training.kd_trainer import KDTrainer

_IMAGE_SIZE = 64
_NUM_CLASSES = 2
_LABELS = "0 0.5 0.5 0.5 0.5\n1 0.5 0.5 0.1 0.1\n"
_STEMS_BY_SPLIT = {"train": ["00000", "00001", "00002"], "val": ["00003", "00004"]}


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory) -> Path:
    """Non-square noise images so letterbox pads, in the on-disk YOLO layout."""
    root = tmp_path_factory.mktemp("modality_ablation_dataset")
    rng = np.random.default_rng(0)
    (root / "rgb").mkdir()
    (root / "nir").mkdir()
    for split, stems in _STEMS_BY_SPLIT.items():
        (root / "labels" / split).mkdir(parents=True)
        for stem in stems:
            rgb = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
            nir = rng.integers(0, 256, (48, 64), dtype=np.uint8)
            assert cv2.imwrite(str(root / "rgb" / f"mango_rgb_{stem}.jpg"), rgb)
            assert cv2.imwrite(str(root / "nir" / f"mango_nir_{stem}.jpg"), nir)
            (root / "labels" / split / f"mango_rgb_{stem}.txt").write_text(_LABELS)
    return root


@pytest.fixture(scope="module")
def teacher_checkpoint(tmp_path_factory) -> Path:
    """An untrained two-stream MasterModel saved as a teacher checkpoint."""
    teacher = MasterModel(
        num_classes=_NUM_CLASSES,
        pretrained_backbone=False,
        backbone_variant="tiny",
        fusion_mode="cross_attention",
        # KDTrainer rebuilds the teacher with the config's head_strides.
        head_strides=TrainingConfig().head_strides,
    )
    path = tmp_path_factory.mktemp("modality_ablation_teacher") / "teacher.pt"
    torch.save({"model_state_dict": teacher.state_dict()}, path)
    return path


def _dataset(root: Path, split: str = "val", **kwargs) -> YOLODataset:
    return YOLODataset(
        rgb_dir=root / "rgb",
        nir_dir=root / "nir",
        labels_dir=root / "labels",
        split=split,
        image_size=_IMAGE_SIZE,
        nir_homography_path=None,
        **kwargs,
    )


def _train_sample(root: Path, **kwargs) -> dict:
    """First train sample with the augmentations seeded, so two datasets draw alike."""
    dataset = _dataset(root, split="train", **kwargs)
    dataset.spatial_transform.set_random_seed(0)
    dataset.rgb_photometric_transform.set_random_seed(0)
    return dataset[0]


def _write_config(root: Path, tmp_path: Path, **extra) -> Path:
    data = {
        "rgb_dir": str(root / "rgb"),
        "nir_dir": str(root / "nir"),
        "labels_dir": str(root / "labels"),
        "split_manifest": None,
        "nir_homography_path": None,
        "output_dir": str(tmp_path / "out"),
        "image_size": _IMAGE_SIZE,
        "batch_size": 2,
        "effective_batch": 2,
        "num_workers": 0,
        "pin_memory": False,
        "device": "cpu",
        "fusion_mode": "cross_attention",
        **extra,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _run_train(monkeypatch, config_path: Path, *overrides: str) -> "_FakeTrainer":
    """`src.training.train.main()` with a real setup and no training."""
    monkeypatch.setattr(
        sys, "argv", ["train", "--config", str(config_path), "--override", *overrides]
    )
    # main() builds the model with pretrained_backbone=True, which downloads.
    monkeypatch.setattr(
        train_module,
        "MasterModel",
        lambda **kwargs: MasterModel(**{**kwargs, "pretrained_backbone": False}),
    )
    monkeypatch.setattr(train_module, "Trainer", _FakeTrainer)
    train_module.main()
    return _FakeTrainer.last


class _FakeTrainer:
    """Records what `train.main()` hands to the Trainer and trains nothing."""

    last = None

    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        _FakeTrainer.last = self

    def fit(self) -> dict:
        return {"checkpoint": "unused.pt", "best_map50": 0.0}


def _run_kd_train(monkeypatch, config_path: Path, *overrides: str) -> dict:
    """`src.training.kd_train.main()` with one real KD epoch instead of `fit()`.

    Returns the tensors the teacher and the student received, the epoch metrics
    and the config the CLI resolved.
    """
    seen: dict = {}

    def one_epoch(trainer: KDTrainer) -> dict:
        teacher_forward = trainer.teacher.forward
        student_forward = trainer.model.forward

        def spy_teacher(rgb, nir):
            seen["teacher_rgb"], seen["teacher_nir"] = rgb.clone(), nir.clone()
            return teacher_forward(rgb, nir)

        def spy_student(rgb):
            seen["student_rgb"] = rgb.clone()
            return student_forward(rgb)

        trainer.teacher.forward = spy_teacher
        trainer.model.forward = spy_student
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
        seen["metrics"] = trainer._train_epoch(optimizer, epoch=1, phase=1)
        seen["val_batch"] = next(iter(trainer.val_loader))
        seen["config"] = trainer.config
        return {"checkpoint": "unused.pt", "best_map50": 0.0}

    monkeypatch.setattr(
        sys, "argv", ["kd_train", "--config", str(config_path), "--override", *overrides]
    )
    monkeypatch.setattr(KDTrainer, "fit", one_epoch)
    kd_train_module.main()
    return seen


class TestDatasetBlanksTheModality:
    """The constant replaces one modality after normalization, on every split."""

    @pytest.mark.parametrize("split", ["train", "val"])
    def test_nir_const_zeroes_nir_and_keeps_everything_else(self, dataset_root, split):
        none = _train_sample(dataset_root) if split == "train" else _dataset(dataset_root)[0]
        blanked = (
            _train_sample(dataset_root, modality_ablation="nir_const")
            if split == "train"
            else _dataset(dataset_root, modality_ablation="nir_const")[0]
        )

        assert blanked["nir"].shape == none["nir"].shape == (1, _IMAGE_SIZE, _IMAGE_SIZE)
        assert blanked["nir"].dtype == none["nir"].dtype
        assert torch.count_nonzero(blanked["nir"]) == 0
        assert torch.count_nonzero(none["nir"]) > 0

        assert torch.equal(blanked["rgb"], none["rgb"])
        assert torch.equal(blanked["bboxes"], none["bboxes"])
        assert torch.equal(blanked["labels"], none["labels"])
        assert blanked["image_id"] == none["image_id"]

    @pytest.mark.parametrize("split", ["train", "val"])
    def test_rgb_const_zeroes_rgb_and_keeps_everything_else(self, dataset_root, split):
        none = _train_sample(dataset_root) if split == "train" else _dataset(dataset_root)[0]
        blanked = (
            _train_sample(dataset_root, modality_ablation="rgb_const")
            if split == "train"
            else _dataset(dataset_root, modality_ablation="rgb_const")[0]
        )

        assert blanked["rgb"].shape == none["rgb"].shape == (3, _IMAGE_SIZE, _IMAGE_SIZE)
        assert blanked["rgb"].dtype == none["rgb"].dtype
        assert torch.count_nonzero(blanked["rgb"]) == 0
        assert torch.count_nonzero(none["rgb"]) > 0

        assert torch.equal(blanked["nir"], none["nir"])
        assert torch.equal(blanked["bboxes"], none["bboxes"])
        assert torch.equal(blanked["labels"], none["labels"])
        assert blanked["image_id"] == none["image_id"]


class TestInvalidValue:
    """A typo raises ValueError naming the accepted values, on every entry path."""

    _ACCEPTED = ("none", "nir_const", "rgb_const")

    def _assert_names_accepted_values(self, excinfo) -> None:
        message = str(excinfo.value)
        assert "modality_ablation" in message
        for accepted in self._ACCEPTED:
            assert accepted in message

    @pytest.mark.parametrize("config_class", [TrainingConfig, KDConfig])
    def test_config_refuses_it(self, config_class):
        with pytest.raises(ValueError) as excinfo:
            config_class(modality_ablation="nir_zero")
        self._assert_names_accepted_values(excinfo)

    def test_dataset_refuses_it(self, dataset_root):
        with pytest.raises(ValueError) as excinfo:
            _dataset(dataset_root, modality_ablation="nir_zero")
        self._assert_names_accepted_values(excinfo)

    def test_train_override_refuses_it(self, dataset_root, tmp_path, monkeypatch):
        # Overrides are applied with setattr, so __post_init__ never sees them.
        config_path = _write_config(dataset_root, tmp_path)
        with pytest.raises(ValueError) as excinfo:
            _run_train(monkeypatch, config_path, "modality_ablation=nir_zero")
        self._assert_names_accepted_values(excinfo)

    def test_kd_train_override_refuses_it(
        self, dataset_root, teacher_checkpoint, tmp_path, monkeypatch
    ):
        config_path = _write_config(
            dataset_root, tmp_path, teacher_checkpoint=str(teacher_checkpoint)
        )
        with pytest.raises(ValueError) as excinfo:
            _run_kd_train(monkeypatch, config_path, "modality_ablation=nir_zero")
        self._assert_names_accepted_values(excinfo)


class TestOverrideReachesTheDataset:
    """`--override modality_ablation=...` blanks the batches of each entry point."""

    @pytest.mark.parametrize(
        ("arm", "blanked", "kept"),
        [("nir_const", "nir", "rgb"), ("rgb_const", "rgb", "nir")],
    )
    def test_train(self, dataset_root, tmp_path, monkeypatch, capsys, arm, blanked, kept):
        config_path = _write_config(dataset_root, tmp_path)

        trainer = _run_train(monkeypatch, config_path, f"modality_ablation={arm}")

        assert trainer.config.modality_ablation == arm
        for loader in (trainer.train_loader, trainer.val_loader):
            batch = next(iter(loader))
            assert torch.count_nonzero(batch[blanked]) == 0
            assert torch.count_nonzero(batch[kept]) > 0
        assert f"modality_ablation: {arm}" in capsys.readouterr().out

    def test_evaluate_checkpoint_override_sets_the_config(self, dataset_root, tmp_path):
        config = TrainingConfig.from_yaml(_write_config(dataset_root, tmp_path))

        ec._apply_overrides(config, ["modality_ablation=nir_const"])

        assert config.modality_ablation == "nir_const"

    def test_evaluate_checkpoint_eval_dataset_blanks_the_modality(self, dataset_root, tmp_path):
        config = TrainingConfig.from_yaml(_write_config(dataset_root, tmp_path))
        ec._apply_overrides(config, ["modality_ablation=nir_const"])

        sample = ec._build_eval_dataset(config, "val")[0]

        assert torch.count_nonzero(sample["nir"]) == 0, (
            "the override was silently ignored: _build_eval_dataset in "
            "scripts/evaluate_checkpoint.py must pass "
            "modality_ablation=config.modality_ablation to YOLODataset"
        )
        assert torch.count_nonzero(sample["rgb"]) > 0


class TestParametersAreIdentical:
    """The arms differ only in input information: the model is the same."""

    def test_every_arm_builds_the_same_two_stream_model(
        self, dataset_root, tmp_path, monkeypatch
    ):
        config_path = _write_config(dataset_root, tmp_path)
        models = {}
        for arm in ("none", "nir_const", "rgb_const"):
            trainer = _run_train(monkeypatch, config_path, f"modality_ablation={arm}")
            assert trainer.config.modality_ablation == arm
            models[arm] = trainer.model

        reference = models["none"]
        reference_shapes = {k: v.shape for k, v in reference.state_dict().items()}
        assert any(key.startswith("backbone.nir_stem.") for key in reference_shapes)
        for arm in ("nir_const", "rgb_const"):
            assert models[arm].count_parameters() == reference.count_parameters()
            assert {k: v.shape for k, v in models[arm].state_dict().items()} == reference_shapes


class TestKnowledgeDistillation:
    """The teacher of `est_kd_rgb` gets the blanked NIR; its student keeps RGB."""

    def test_nir_const_blanks_the_teachers_nir_only(
        self, dataset_root, teacher_checkpoint, tmp_path, monkeypatch, capsys
    ):
        checkpoint = torch.load(teacher_checkpoint, weights_only=False)
        blanked_teacher = tmp_path / "maestro_rgb.pt"
        torch.save({**checkpoint, "config": {"modality_ablation": "nir_const"}}, blanked_teacher)
        config_path = _write_config(
            dataset_root, tmp_path, teacher_checkpoint=str(blanked_teacher)
        )

        seen = _run_kd_train(monkeypatch, config_path, "modality_ablation=nir_const")

        assert seen["config"].modality_ablation == "nir_const"
        assert torch.count_nonzero(seen["teacher_nir"]) == 0
        assert torch.count_nonzero(seen["teacher_rgb"]) > 0
        assert torch.equal(seen["student_rgb"], seen["teacher_rgb"])
        assert seen["metrics"]["steps_taken"] > 0
        assert torch.count_nonzero(seen["val_batch"]["nir"]) == 0
        assert torch.count_nonzero(seen["val_batch"]["rgb"]) > 0
        assert "modality_ablation: nir_const" in capsys.readouterr().out

    @pytest.mark.parametrize("route", ["override", "yaml"])
    def test_kd_train_refuses_rgb_const(
        self, dataset_root, teacher_checkpoint, tmp_path, monkeypatch, route
    ):
        extra = {"modality_ablation": "rgb_const"} if route == "yaml" else {}
        config_path = _write_config(
            dataset_root, tmp_path, teacher_checkpoint=str(teacher_checkpoint), **extra
        )
        overrides = ["modality_ablation=rgb_const"] if route == "override" else ["seed=1"]

        with pytest.raises(ValueError, match=r"rgb_const.*student"):
            _run_kd_train(monkeypatch, config_path, *overrides)

        assert not (tmp_path / "out").exists()

    def test_trainer_refuses_rgb_const_before_touching_anything(self, tmp_path):
        output_dir = tmp_path / "kd_out"
        config = KDConfig(
            teacher_checkpoint=str(tmp_path / "never_read.pt"),
            num_classes=_NUM_CLASSES,
            batch_size=2,
            effective_batch=2,
            device="cpu",
            output_dir=str(output_dir),
            modality_ablation="rgb_const",
        )

        with pytest.raises(ValueError, match=r"rgb_const.*student"):
            KDTrainer(
                model=StudentModel(num_classes=_NUM_CLASSES),
                config=config,
                train_loader=[],
                val_loader=[],
            )

        assert not output_dir.exists()


class TestPreviousBehaviourHolds:
    """Without the flag every entry point behaves as it did before it existed."""

    def test_default_dataset_output_is_the_normalized_pair(self, dataset_root):
        stem = _STEMS_BY_SPLIT["val"][0]
        sample = _dataset(dataset_root)[0]

        bgr = cv2.imread(str(dataset_root / "rgb" / f"mango_rgb_{stem}.jpg"))
        rgb, *_ = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), _IMAGE_SIZE, 114)
        nir_mean, nir_std = 0.0569, 0.0546
        nir_img = cv2.imread(
            str(dataset_root / "nir" / f"mango_nir_{stem}.jpg"), cv2.IMREAD_GRAYSCALE
        )
        nir, *_ = letterbox(nir_img, _IMAGE_SIZE, int(nir_mean * 255))
        expected_nir = (nir.astype(np.float32) / 255.0 - nir_mean) / nir_std

        assert torch.equal(sample["rgb"], normalize_rgb(rgb))
        assert torch.equal(sample["nir"], torch.from_numpy(expected_nir[np.newaxis]).float())
        assert sample["labels"].tolist() == [0, 1]
        assert sample["bboxes"].shape == (2, 4)

    def test_train_override_coercion_and_output_are_unchanged(
        self, dataset_root, tmp_path, monkeypatch, capsys
    ):
        config_path = _write_config(dataset_root, tmp_path)

        trainer = _run_train(monkeypatch, config_path, "seed=7", "ema=true", "weight_decay=0.1")

        assert trainer.config.seed == 7
        assert trainer.config.ema is True
        assert trainer.config.weight_decay == pytest.approx(0.1)
        batch = next(iter(trainer.val_loader))
        assert torch.count_nonzero(batch["rgb"]) > 0
        assert torch.count_nonzero(batch["nir"]) > 0
        assert "modality_ablation:" not in capsys.readouterr().out

    def test_kd_teacher_keeps_the_real_nir_by_default(
        self, dataset_root, teacher_checkpoint, tmp_path, monkeypatch, capsys
    ):
        config_path = _write_config(
            dataset_root, tmp_path, teacher_checkpoint=str(teacher_checkpoint)
        )

        seen = _run_kd_train(monkeypatch, config_path, "kd_weight=0.5")

        assert seen["config"].kd_weight == pytest.approx(0.5)
        assert torch.count_nonzero(seen["teacher_nir"]) > 0
        assert torch.equal(seen["student_rgb"], seen["teacher_rgb"])
        assert seen["metrics"]["steps_taken"] > 0
        assert "modality_ablation:" not in capsys.readouterr().out

    def test_a_yaml_without_the_field_resolves_to_none_and_round_trips(
        self, dataset_root, tmp_path
    ):
        config = TrainingConfig.from_yaml(_write_config(dataset_root, tmp_path))
        assert config.modality_ablation == "none"

        config.modality_ablation = "nir_const"
        round_trip = tmp_path / "round_trip.yaml"
        config.to_yaml(round_trip)
        assert TrainingConfig.from_yaml(round_trip).modality_ablation == "nir_const"


class TestArmBinding:
    """A checkpoint is only used with the arm it was trained on: nothing else would catch it."""

    @pytest.mark.parametrize("trained, current", [("nir_const", "none"), ("none", "rgb_const")])
    def test_evaluation_refuses_a_checkpoint_of_another_arm(self, dataset_root, tmp_path, trained, current):
        config = TrainingConfig.from_yaml(_write_config(dataset_root, tmp_path))
        config.modality_ablation = current
        with pytest.raises(ValueError, match="modality_ablation"):
            ec._check_modality_ablation_binding("x.pt", {"modality_ablation": trained}, config)

    def test_evaluation_accepts_its_own_arm_and_older_checkpoints(self, dataset_root, tmp_path):
        config = TrainingConfig.from_yaml(_write_config(dataset_root, tmp_path))
        ec._check_modality_ablation_binding("x.pt", {}, config)  # recorded before the flag: none
        ec._check_modality_ablation_binding("x.pt", None, config)  # no recorded config
        config.modality_ablation = "nir_const"
        ec._check_modality_ablation_binding("x.pt", {"modality_ablation": "nir_const"}, config)

    @pytest.mark.parametrize("recorded, current", [("nir_const", "none"), ("none", "nir_const")])
    def test_distillation_refuses_a_teacher_of_another_arm(self, tmp_path, recorded, current):
        path = tmp_path / "teacher.pt"
        torch.save({"config": {"modality_ablation": recorded}}, path)
        config = KDConfig(teacher_checkpoint=str(path), modality_ablation=current)
        with pytest.raises(ValueError, match="modality_ablation"):
            KDTrainer._load_teacher(config)
