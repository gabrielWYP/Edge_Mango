"""The attention-ablation flags of the two-stream teacher.

`fusion_attention` and `use_nir` (both default True, `fusion_mode=
"cross_attention"` only) carve `MasterModel` into the three arms of the
attention ablation (`src/training/fusion_modes.py`):

    A  defaults                                  configs/experiment/twostream.yaml
    B  fusion_attention=False                    twostream_no_attention.yaml
    C  fusion_attention=False, use_nir=False     twostream_rgb_only.yaml

What these tests pin down:
- the defaults are the existing model: same state_dict schema and a
  bit-identical forward as explicitly passing both flags True;
- exact parameter totals per arm, and which state_dict keys each arm drops;
- B passes the RGB stage features through unchanged and still lets NIR reach
  the head through the neck; C ignores the NIR tensor entirely;
- every arm keeps the 7-key output contract and the optimizer/grad-norm
  instrumentation the Trainer relies on;
- invalid combinations fail loudly, in the model and in `TrainingConfig`;
- loaders rebuild the arm the checkpoint was trained as (absent -> True) and
  turn a residual mismatch into one sentence instead of a missing-key dump;
- the two experiment files differ from `twostream.yaml` only where intended.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts import evaluate_checkpoint as ec
from scripts.visualize_damage_predictions import load_model as visualize_load_model
from src.models.master.master_model import MasterModel
from src.models.master.neck import DualFPN
from src.training.config import TrainingConfig
from src.training.fusion_modes import (
    fusion_attention_from_checkpoint,
    use_nir_from_checkpoint,
    validate_fusion_ablation,
)
from src.training.kd_config import KDConfig
from src.training.kd_trainer import KDTrainer
from src.training.loop import Trainer

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_EXPERIMENTS = _REPO_ROOT / "configs" / "experiment"
_TWOSTREAM = _EXPERIMENTS / "twostream.yaml"
_NO_ATTENTION = _EXPERIMENTS / "twostream_no_attention.yaml"
_RGB_ONLY = _EXPERIMENTS / "twostream_rgb_only.yaml"

IMAGE_SIZE = 128
CPU = torch.device("cpu")

ARMS = {
    "A": {},
    "B": {"fusion_attention": False},
    "C": {"fusion_attention": False, "use_nir": False},
}
# head_strides [8, 16, 32]; the count does not depend on the input size.
EXPECTED_TOTALS = {
    "A": 46_642_930,
    "B": 46_642_930 - 9_421_920,                                  # CrossModalFusion
    "C": 46_642_930 - 9_421_920 - 1_824 - 2_729_984 - 3 * 256 * 256,  # nir_stem, fpn_nir, fusion-conv inputs
}
OUTPUT_KEYS = {
    "preds", "cls_preds", "reg_preds",
    "distill_backbone", "distill_fpn", "distill_head_cls", "distill_head_reg",
}


def _build(seed: int = 0, **flags) -> MasterModel:
    torch.manual_seed(seed)
    return MasterModel(
        num_classes=2, pretrained_backbone=False, fusion_mode="cross_attention", **flags
    )


def _inputs(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=gen),
        torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, generator=gen),
    )


def _shapes(output: dict) -> dict:
    return {key: [tuple(t.shape) for t in value] for key, value in output.items()}


def _has_prefix(model: MasterModel, prefix: str) -> bool:
    return any(key.startswith(prefix) for key in model.state_dict())


@pytest.fixture(scope="module")
def arms() -> dict[str, MasterModel]:
    return {name: _build(**flags).eval() for name, flags in ARMS.items()}


# --------------------------------------------------------------------------- #
# Defaults are the existing model
# --------------------------------------------------------------------------- #


class TestDefaultsAreUnchanged:
    def test_state_dict_schema_matches_explicit_true_flags(self, arms):
        explicit = _build(fusion_attention=True, use_nir=True)
        assert {k: v.shape for k, v in explicit.state_dict().items()} == {
            k: v.shape for k, v in arms["A"].state_dict().items()
        }

    def test_forward_is_bit_identical_to_explicit_true_flags(self, arms):
        explicit = _build(fusion_attention=True, use_nir=True).eval()
        rgb, nir = _inputs()
        with torch.no_grad():
            expected = arms["A"](rgb, nir)
            actual = explicit(rgb, nir)
        for key in OUTPUT_KEYS:
            for a, b in zip(actual[key], expected[key]):
                assert torch.equal(a, b), key

    def test_default_has_every_two_stream_module(self, arms):
        model = arms["A"]
        assert model.fusion is not None
        assert model.backbone.nir_stem is not None
        assert model.neck.fpn_nir is not None
        assert set(model.backbone.stem_modules) == {"rgb_stem", "nir_stem"}


# --------------------------------------------------------------------------- #
# Schema and parameter totals per arm
# --------------------------------------------------------------------------- #


class TestArmSchemas:
    @pytest.mark.parametrize("arm", list(ARMS))
    def test_exact_parameter_totals(self, arms, arm):
        model = arms[arm]
        assert sum(p.numel() for p in model.parameters()) == EXPECTED_TOTALS[arm]
        counts = model.count_parameters()
        assert counts["total"] == EXPECTED_TOTALS[arm]
        assert ("fusion" in counts) == (arm == "A")

    def test_arm_b_drops_only_the_fusion_keys(self, arms):
        a_keys = set(arms["A"].state_dict())
        b_keys = set(arms["B"].state_dict())
        assert b_keys < a_keys
        assert {k for k in a_keys - b_keys if not k.startswith("fusion.")} == set()
        assert arms["B"].fusion is None

    def test_arm_c_drops_the_nir_stream(self, arms):
        model = arms["C"]
        assert not _has_prefix(model, "fusion.")
        assert not _has_prefix(model, "backbone.nir_stem.")
        assert not _has_prefix(model, "neck.fpn_nir.")
        assert model.backbone.nir_stem is None
        assert model.neck.fpn_nir is None
        assert set(model.backbone.stem_modules) == {"rgb_stem"}
        # Same Conv1x1 -> norm -> ReLU -> Dropout2d block, over 256 inputs.
        for block in model.neck.fusion_convs:
            assert block[0].in_channels == block[0].out_channels == 256
            assert [type(m).__name__ for m in block] == [
                "Conv2d", "BatchNorm2d", "ReLU", "Dropout2d",
            ]

    def test_arms_b_and_c_differ_only_by_the_nir_stream(self, arms):
        b = {k: v.shape for k, v in arms["B"].state_dict().items()}
        c = {k: v.shape for k, v in arms["C"].state_dict().items()}
        nir_only = {
            k for k in b
            if k.startswith(("backbone.nir_stem.", "neck.fpn_nir."))
        }
        assert set(c) == set(b) - nir_only
        changed = {k for k in c if c[k] != b[k]}
        assert changed == {f"neck.fusion_convs.{i}.0.weight" for i in range(3)}


# --------------------------------------------------------------------------- #
# Forward behaviour
# --------------------------------------------------------------------------- #


class TestForward:
    @pytest.mark.parametrize("arm", list(ARMS))
    def test_output_contract_is_the_same_in_every_arm(self, arms, arm):
        rgb, nir = _inputs()
        with torch.no_grad():
            reference = _shapes(arms["A"](rgb, nir))
            output = arms[arm](rgb, nir)
        assert set(output) == OUTPUT_KEYS
        assert _shapes(output) == reference

    def test_arm_b_distill_backbone_is_the_rgb_backbone_features(self, arms):
        model = arms["B"]
        rgb, nir = _inputs()
        with torch.no_grad():
            rgb_features, _ = model.backbone(rgb, nir)
            output = model(rgb, nir)
        assert len(output["distill_backbone"]) == 4
        for got, expected in zip(output["distill_backbone"], rgb_features):
            assert torch.equal(got, expected)

    def test_arm_b_still_reaches_the_head_through_the_nir_fpn(self, arms):
        """Control for the arm-C invariance test: in B the NIR tensor still
        changes the predictions (through `fpn_nir` and the concat)."""
        model = arms["B"]
        rgb, nir = _inputs()
        _, other_nir = _inputs(seed=1)
        with torch.no_grad():
            first = model(rgb, nir)["preds"]
            second = model(rgb, other_nir)["preds"]
        assert any(not torch.equal(a, b) for a, b in zip(first, second))

    def test_arm_c_ignores_the_nir_tensor(self, arms):
        model = arms["C"]
        rgb, nir = _inputs()
        _, other_nir = _inputs(seed=1)
        with torch.no_grad():
            first = model(rgb, nir)
            second = model(rgb, other_nir)
            third = model(rgb, torch.zeros_like(nir))
        for key in OUTPUT_KEYS:
            for a, b, c in zip(first[key], second[key], third[key]):
                assert torch.equal(a, b) and torch.equal(a, c), key

    def test_arm_c_backbone_returns_no_nir_features(self, arms):
        rgb, nir = _inputs()
        with torch.no_grad():
            rgb_features, nir_features = arms["C"].backbone(rgb, nir)
        assert nir_features is None
        assert len(rgb_features) == 4

    def test_dual_fpn_rejects_a_stream_mismatch(self):
        features = [
            torch.randn(1, ch, IMAGE_SIZE // (4 * 2**i), IMAGE_SIZE // (4 * 2**i))
            for i, ch in enumerate(MasterModel.STAGE_CHANNELS)
        ]
        with pytest.raises(ValueError, match="use_nir=False"):
            DualFPN(use_nir=False)(features, features)
        with pytest.raises(ValueError, match="use_nir=True"):
            DualFPN()(features, None)

    def test_two_stream_backbone_still_requires_nir(self, arms):
        rgb, _ = _inputs()
        with pytest.raises(ValueError, match="nir=None"):
            arms["A"].backbone(rgb, None)


# --------------------------------------------------------------------------- #
# Trainer instrumentation and freezing
# --------------------------------------------------------------------------- #


def _trainer_config(**flags) -> TrainingConfig:
    return TrainingConfig(
        model_type="master",
        fusion_mode="cross_attention",
        head_strides=[8, 16, 32],
        assigner_level_ranges=[64.0, 128.0],
        nir_homography_path=None,
        split_manifest=None,
        **flags,
    )


class TestTrainerIntegration:
    @pytest.mark.parametrize("arm", list(ARMS))
    def test_param_groups_cover_every_parameter(self, arms, arm):
        model = arms[arm]
        groups = Trainer._discriminative_param_groups(
            SimpleNamespace(model=model, config=_trainer_config(**ARMS[arm])), lr=1e-3
        )
        assert len(groups) == 2
        stage_ids = {id(p) for p in model.backbone.stages.parameters()}
        assert {id(p) for p in groups[0]["params"]} == stage_ids
        assert groups[0]["lr"] == pytest.approx(1e-4)
        assert groups[1]["lr"] == pytest.approx(1e-3)
        grouped = {id(p) for group in groups for p in group["params"]}
        assert grouped == {id(p) for p in model.parameters() if p.requires_grad}

    @pytest.mark.parametrize("arm", list(ARMS))
    def test_grad_norms_report_only_existing_modules(self, arms, arm):
        norms = Trainer._module_grad_norms(SimpleNamespace(model=arms[arm]))
        assert ("fusion" in norms) == (arm == "A")
        assert ("backbone_nir_stem" in norms) == (arm != "C")
        assert "backbone_rgb_stem" in norms

    def test_freeze_and_unfreeze_touch_only_existing_stems(self):
        model = _build(**ARMS["C"])
        model.freeze_backbone(freeze_stages=2)
        assert not any(p.requires_grad for p in model.backbone.rgb_stem.parameters())
        model.unfreeze_backbone_stages([0, 1], unfreeze_stem=True)
        assert all(p.requires_grad for p in model.backbone.rgb_stem.parameters())


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

_INVALID = [
    pytest.param({"fusion_mode": "early", "fusion_attention": False}, "fusion_attention", id="early-no-attention"),
    pytest.param({"fusion_mode": "early", "use_nir": False}, "use_nir", id="early-no-nir"),
    pytest.param(
        {"fusion_mode": "early", "fusion_attention": False, "use_nir": False}, "early", id="early-rgb-only"
    ),
    pytest.param(
        {"fusion_mode": "cross_attention", "use_nir": False}, "requires fusion_attention=False",
        id="no-nir-with-attention",
    ),
    pytest.param(
        {"fusion_mode": "cross_attention", "fusion_attention": False, "fusion_pos_encoding": True},
        "fusion_pos_encoding=True requires fusion_attention=True", id="posenc-without-attention",
    ),
    pytest.param(
        {"fusion_mode": "cross_attention", "fusion_attention": "false"}, "must be a bool", id="string-flag",
    ),
    pytest.param(
        {"fusion_mode": "cross_attention", "fusion_attention": False, "use_nir": 0}, "must be a bool",
        id="int-flag",
    ),
]


class TestValidation:
    @pytest.mark.parametrize("kwargs,match", _INVALID)
    def test_master_model_rejects_it(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MasterModel(num_classes=2, pretrained_backbone=False, **kwargs)

    @pytest.mark.parametrize("kwargs,match", _INVALID)
    def test_training_config_rejects_it(self, kwargs, match):
        head = (
            {"head_strides": [8, 16, 32], "assigner_level_ranges": [64.0, 128.0]}
            if kwargs["fusion_mode"] == "cross_attention"
            else {}
        )
        with pytest.raises(ValueError, match=match):
            TrainingConfig(model_type="master", **head, **kwargs)

    @pytest.mark.parametrize("arm", list(ARMS))
    def test_every_arm_is_a_valid_training_config(self, arm):
        config = _trainer_config(**ARMS[arm])
        assert config.fusion_attention is ARMS[arm].get("fusion_attention", True)
        assert config.use_nir is ARMS[arm].get("use_nir", True)

    def test_validator_accepts_the_three_arms_and_the_early_default(self):
        validate_fusion_ablation(True, True, "early")
        for flags in ARMS.values():
            validate_fusion_ablation(
                flags.get("fusion_attention", True), flags.get("use_nir", True), "cross_attention"
            )

    def test_defaults_are_on(self):
        config = TrainingConfig()
        assert config.fusion_attention is True
        assert config.use_nir is True


# --------------------------------------------------------------------------- #
# Checkpoint readers and loaders
# --------------------------------------------------------------------------- #


class TestCheckpointReaders:
    @pytest.mark.parametrize(
        "checkpoint,expected",
        [
            ({}, True),
            ({"config": None}, True),
            ({"config": {"head_strides": [8, 16, 32]}}, True),
            ({"config": {"fusion_attention": True, "use_nir": True}}, True),
            ({"config": {"fusion_attention": False, "use_nir": False}}, False),
        ],
    )
    def test_absent_means_true(self, checkpoint, expected):
        assert fusion_attention_from_checkpoint(checkpoint) is expected
        assert use_nir_from_checkpoint(checkpoint) is expected

    def test_evaluate_checkpoint_prefers_the_recorded_flags(self, caplog):
        recorded = {"config": {"fusion_attention": False, "use_nir": False}}
        with caplog.at_level(logging.INFO, logger=ec.logger.name):
            assert ec._resolve_fusion_ablation(recorded, _trainer_config()) == (False, False)
        assert any("fusion_attention" in r.getMessage() for r in caplog.records)

    def test_a_recorded_config_without_the_fields_is_arm_a(self):
        config = _trainer_config(**ARMS["B"])
        assert ec._resolve_fusion_ablation({"config": {"head_strides": [8, 16, 32]}}, config) == (
            True, True,
        )

    def test_no_recorded_config_falls_back_to_the_eval_config(self):
        config = _trainer_config(**ARMS["C"])
        assert ec._resolve_fusion_ablation({"model_state_dict": {}}, config) == (False, False)
        assert ec._resolve_fusion_ablation(None, _trainer_config()) == (True, True)


def _payload(model: MasterModel, recorded: dict | None) -> dict:
    """What `Trainer._save_checkpoint` writes, minus the bulky extras."""
    payload = {
        "epoch": 3,
        "phase": 1,
        "arch_version": 1,
        "model_state_dict": model.state_dict(),
        "metrics": {},
        "best_map50": 0.0,
    }
    if recorded is not None:
        payload["config"] = {"head_strides": [8, 16, 32], **recorded}
    return payload


@pytest.fixture
def save(tmp_path):
    """`torch.save` into `tmp_path`, deleted again on teardown: each full-size
    checkpoint is ~150-190 MB, and pytest keeps the last three sessions'
    `tmp_path`s (often on a RAM-backed /tmp)."""
    saved: list[Path] = []

    def _save(name: str, obj) -> str:
        path = tmp_path / name
        torch.save(obj, path)
        saved.append(path)
        return str(path)

    yield _save
    for path in saved:
        path.unlink(missing_ok=True)


class TestLoaders:
    def test_recorded_arm_b_is_rebuilt_under_an_arm_a_config(self, save, arms):
        path = save("b.pt", _payload(arms["B"], ARMS["B"]))
        model = ec._load_model("master", path, _trainer_config(), CPU)
        assert model.fusion is None and model.use_nir
        for key, value in arms["B"].state_dict().items():
            assert torch.equal(model.state_dict()[key], value), key

    def test_recorded_arm_c_is_rebuilt_under_an_arm_b_config(self, save, arms):
        path = save("c.pt", _payload(arms["C"], ARMS["C"]))
        model = ec._load_model("master", path, _trainer_config(**ARMS["B"]), CPU)
        assert model.backbone.nir_stem is None and model.neck.fpn_nir is None

    def test_legacy_checkpoint_loads_as_arm_a(self, save, arms):
        path = save("a.pt", _payload(arms["A"], {}))
        model = ec._load_model("master", path, _trainer_config(), CPU)
        assert model.fusion is not None

    def test_raw_arm_b_state_dict_under_an_arm_a_config_fails_loudly(self, save, arms):
        path = save("raw_b.pt", arms["B"].state_dict())
        with pytest.raises(ValueError, match="fusion_attention") as excinfo:
            ec._load_model("master", path, _trainer_config(), CPU)
        message = str(excinfo.value)
        assert path in message
        assert "fusion_attention=False" in message and "fusion_attention=True" in message
        assert "use_nir" not in message

    def test_raw_arm_c_state_dict_under_an_arm_b_config_fails_loudly(self, save, arms):
        path = save("raw_c.pt", arms["C"].state_dict())
        with pytest.raises(ValueError, match="use_nir=False"):
            ec._load_model("master", path, _trainer_config(**ARMS["B"]), CPU)

    def test_raw_arm_c_state_dict_under_an_arm_c_config_loads(self, save, arms):
        path = save("raw_c.pt", arms["C"].state_dict())
        model = ec._load_model("master", path, _trainer_config(**ARMS["C"]), CPU)
        assert model.neck.fpn_nir is None

    def test_a_recorded_config_that_contradicts_the_weights_fails_loudly(self, save, arms):
        path = save("liar.pt", _payload(arms["A"], ARMS["B"]))
        with pytest.raises(ValueError, match="fusion_attention"):
            ec._load_model("master", path, _trainer_config(), CPU)

    def test_visualize_loader_rebuilds_the_recorded_arm(self, save, arms):
        path = save("c.pt", _payload(arms["C"], ARMS["C"]))
        model, strides = visualize_load_model("master", path, _trainer_config(), CPU)
        assert model.backbone.nir_stem is None
        assert strides == [8, 16, 32]

    def test_visualize_loader_fails_loudly_on_a_mismatch(self, save, arms):
        path = save("raw_b.pt", arms["B"].state_dict())
        with pytest.raises(ValueError, match="fusion_attention"):
            visualize_load_model("master", path, _trainer_config(), CPU)

    @staticmethod
    def _kd_config(path: str) -> KDConfig:
        return KDConfig(
            teacher_checkpoint=path,
            num_classes=2,
            backbone_variant="tiny",
            fusion_mode="cross_attention",
            head_strides=[8, 16, 32],
            device="cpu",
        )

    @pytest.mark.parametrize("arm", ["B", "C"])
    def test_kd_teacher_is_rebuilt_as_the_recorded_arm(self, save, arms, arm):
        path = save(f"t_{arm}.pt", _payload(arms[arm], ARMS[arm]))
        teacher = KDTrainer._load_teacher(self._kd_config(path))
        assert teacher.fusion is None
        assert (teacher.backbone.nir_stem is None) == (arm == "C")

    def test_kd_teacher_fails_loudly_when_the_record_contradicts_the_weights(self, save, arms):
        path = save("t_liar.pt", _payload(arms["C"], ARMS["B"]))
        with pytest.raises(ValueError, match="use_nir=False"):
            KDTrainer._load_teacher(self._kd_config(path))


# --------------------------------------------------------------------------- #
# Experiment files
# --------------------------------------------------------------------------- #


class TestExperimentFiles:
    @pytest.mark.parametrize(
        "path,arm", [(_TWOSTREAM, "A"), (_NO_ATTENTION, "B"), (_RGB_ONLY, "C")]
    )
    def test_each_file_loads_as_its_arm(self, path, arm):
        config = TrainingConfig.from_yaml(path)
        assert config.fusion_mode == "cross_attention"
        assert config.fusion_attention is ARMS[arm].get("fusion_attention", True)
        assert config.use_nir is ARMS[arm].get("use_nir", True)
        assert config.head_strides == [8, 16, 32]

    @pytest.mark.parametrize(
        "path,flags,output_dir",
        [
            (_NO_ATTENTION, {"fusion_attention"}, "checkpoints/twostream_no_attention"),
            (_RGB_ONLY, {"fusion_attention", "use_nir"}, "checkpoints/twostream_rgb_only"),
        ],
    )
    def test_file_differs_from_twostream_only_where_intended(self, path, flags, output_dir):
        """experiment_sha256 hashes file bytes, hence separate files; their
        parsed content must still differ from arm A in nothing else."""
        with open(_TWOSTREAM) as f:
            base = yaml.safe_load(f)
        with open(path) as f:
            arm = yaml.safe_load(f)
        changed = {k for k in base.keys() | arm.keys() if base.get(k) != arm.get(k)}
        assert changed == flags | {"output_dir"}
        assert arm["output_dir"] == output_dir
        assert not flags & base.keys()

    def test_yaml_round_trip(self, tmp_path):
        config = _trainer_config(**ARMS["C"])
        path = tmp_path / "config.yaml"
        config.to_yaml(path)
        loaded = TrainingConfig.from_yaml(path)
        assert (loaded.fusion_attention, loaded.use_nir) == (False, False)
