"""`neck_norm` / `neck_gn_groups` / `ema*`: option plumbing and config validation.

Why these options exist: leave-one-lot-out runs of
`configs/experiment/twostream_roi_crop.yaml` swing from epoch to epoch (val
damage TP 36 -> 0 -> 49 at epochs 50/60/70), so checkpoint selection on the
curve is close to random. The only BatchNorm layers in the master model are
the `DualFPN.fusion_convs` of the cross-attention path, and with a physical
batch of 2 their batch statistics are noisy (gradient accumulation does not
fix that). `neck_norm="group"` swaps them for GroupNorm, which has no batch
statistics at all; `ema=True` validates and saves an exponential moving
average of the weights instead of the last-step weights.

Every default keeps today's behaviour: `neck_norm="batch"`, `ema=False`.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.training.config import TrainingConfig
from src.training.neck_norm import (
    DEFAULT_NECK_GN_GROUPS,
    DEFAULT_NECK_NORM,
    NECK_NORMS,
    check_norm_compatible,
    make_norm,
    neck_norm_from_checkpoint,
    resolve_gn_groups,
    validate_neck_norm,
    validate_neck_norm_for_fusion_mode,
)

EXPERIMENT_DIR = REPO_ROOT / "configs" / "experiment"
STABLE = EXPERIMENT_DIR / "twostream_roi_crop_stable.yaml"
CROP = EXPERIMENT_DIR / "twostream_roi_crop.yaml"


class TestConstants:
    def test_the_two_norms_and_the_default(self):
        assert NECK_NORMS == ("batch", "group")
        assert DEFAULT_NECK_NORM == "batch"
        assert DEFAULT_NECK_GN_GROUPS == 32


class TestValidateNeckNorm:
    @pytest.mark.parametrize("norm", ["batch", "group"])
    def test_accepts_the_known_norms(self, norm):
        validate_neck_norm(norm, 32)

    @pytest.mark.parametrize("norm", ["gn", "bn", "Group", "", None, "layer"])
    def test_rejects_anything_else_naming_the_key(self, norm):
        with pytest.raises(ValueError, match="neck_norm"):
            validate_neck_norm(norm, 32)

    @pytest.mark.parametrize("groups", [0, -4, True, 1.5, "32", None])
    def test_rejects_a_bad_group_count(self, groups):
        with pytest.raises(ValueError, match="neck_gn_groups"):
            validate_neck_norm("group", groups)


class TestFusionModeCompatibility:
    def test_group_is_rejected_on_the_early_path(self):
        """`FPNNeck` has no normalization layer, so `group` would be a silent
        no-op there — the same failure class `fusion_pos_encoding=True` on
        `early` is rejected for."""
        with pytest.raises(ValueError, match="early"):
            validate_neck_norm_for_fusion_mode("group", "early")

    def test_group_is_accepted_on_cross_attention(self):
        validate_neck_norm_for_fusion_mode("group", "cross_attention")

    @pytest.mark.parametrize("fusion_mode", ["early", "cross_attention"])
    def test_batch_is_accepted_everywhere(self, fusion_mode):
        validate_neck_norm_for_fusion_mode("batch", fusion_mode)


class TestResolveGnGroups:
    @pytest.mark.parametrize(
        "channels,groups,expected",
        [
            (256, 32, 32),   # the neck: 8 channels per group
            (48, 32, 24),    # largest divisor of 48 that is <= 32
            (7, 32, 7),      # fewer channels than groups
            (37, 32, 1),     # prime above the cap: one group
            (256, 1, 1),     # explicit single group (LayerNorm over C,H,W)
            (256, 300, 256), # cap above the channel count
        ],
    )
    def test_largest_divisor_not_above_the_cap(self, channels, groups, expected):
        assert resolve_gn_groups(channels, groups) == expected


class TestMakeNorm:
    def test_batch(self):
        layer = make_norm("batch", 256, 32)
        assert isinstance(layer, nn.BatchNorm2d)
        assert layer.num_features == 256

    def test_group(self):
        layer = make_norm("group", 256, 32)
        assert isinstance(layer, nn.GroupNorm)
        assert (layer.num_groups, layer.num_channels) == (32, 256)

    def test_group_count_is_resolved_against_the_channels(self):
        assert make_norm("group", 48, 32).num_groups == 24

    def test_invalid_norm_raises(self):
        with pytest.raises(ValueError, match="neck_norm"):
            make_norm("instance", 8, 4)


class TestNeckNormFromCheckpoint:
    """Checkpoints written before the field existed are BatchNorm by
    construction, so absence resolves to the default — not to whatever the
    caller's config says."""

    @pytest.mark.parametrize("checkpoint", [{}, {"config": None}, {"config": {}}])
    def test_absent_is_the_legacy_batch_norm(self, checkpoint):
        assert neck_norm_from_checkpoint(checkpoint) == ("batch", 32)

    def test_recorded_values_win(self):
        checkpoint = {"config": {"neck_norm": "group", "neck_gn_groups": 16}}
        assert neck_norm_from_checkpoint(checkpoint) == ("group", 16)


class _Net(nn.Module):
    """Minimal stand-in whose neck norm lives under the `neck.` prefix."""

    def __init__(self, norm: str | None):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(4, 8, 1)]
        if norm is not None:
            layers.append(make_norm(norm, 8, 4))
        self.neck = nn.Sequential(*layers)


class TestCheckNormCompatible:
    @pytest.mark.parametrize("norm", ["batch", "group", None])
    def test_matching_schemas_pass(self, norm):
        check_norm_compatible(_Net(norm), _Net(norm).state_dict())

    def test_a_batch_checkpoint_into_a_group_model_fails_loudly(self):
        with pytest.raises(ValueError, match="neck_norm") as excinfo:
            check_norm_compatible(
                _Net("group"), _Net("batch").state_dict(), source="Checkpoint at a.pt"
            )
        message = str(excinfo.value)
        assert "Checkpoint at a.pt" in message
        assert "'batch'" in message and "'group'" in message

    def test_a_group_checkpoint_into_a_batch_model_fails_loudly(self):
        with pytest.raises(ValueError, match="neck_norm='group'"):
            check_norm_compatible(_Net("batch"), _Net("group").state_dict())

    def test_a_model_with_no_neck_norm_never_conflicts(self):
        """The early-fusion neck has no norm layer: nothing to disagree on."""
        check_norm_compatible(_Net(None), _Net(None).state_dict())


class TestTrainingConfigNeckNormAndEma:
    def test_defaults_keep_todays_behaviour(self):
        config = TrainingConfig()
        assert config.neck_norm == "batch"
        assert config.neck_gn_groups == 32
        assert config.ema is False
        assert 0.0 < config.ema_decay < 1.0
        assert config.ema_tau > 0.0

    def test_cross_attention_accepts_group(self):
        config = TrainingConfig(
            fusion_mode="cross_attention",
            head_strides=[8, 16, 32],
            assigner_level_ranges=[64.0, 128.0],
            neck_norm="group",
        )
        assert config.neck_norm == "group"

    def test_early_master_rejects_group(self):
        with pytest.raises(ValueError, match="early"):
            TrainingConfig(fusion_mode="early", neck_norm="group")

    def test_student_ignores_the_norm_like_the_other_master_only_keys(self):
        TrainingConfig(model_type="student", neck_norm="group")

    @pytest.mark.parametrize("norm", ["gn", "Group", ""])
    def test_unknown_norm_is_rejected(self, norm):
        with pytest.raises(ValueError, match="neck_norm"):
            TrainingConfig(neck_norm=norm)

    @pytest.mark.parametrize("groups", [0, -1, True, 2.5])
    def test_bad_group_count_is_rejected(self, groups):
        with pytest.raises(ValueError, match="neck_gn_groups"):
            TrainingConfig(neck_gn_groups=groups)

    @pytest.mark.parametrize("decay", [0.0, 1.0, 1.5, -0.1])
    def test_ema_decay_must_be_in_the_open_unit_interval(self, decay):
        with pytest.raises(ValueError, match="ema_decay"):
            TrainingConfig(ema_decay=decay)

    @pytest.mark.parametrize("tau", [0.0, -5.0])
    def test_ema_tau_must_be_positive(self, tau):
        with pytest.raises(ValueError, match="ema_tau"):
            TrainingConfig(ema_tau=tau)

    @pytest.mark.parametrize("value", ["false", "true", 1, None])
    def test_ema_must_be_a_real_bool(self, value):
        """A YAML string like "false" is truthy and would silently enable it."""
        with pytest.raises(ValueError, match="ema"):
            TrainingConfig(ema=value)

    def test_yaml_round_trip_keeps_the_keys(self, tmp_path):
        config = TrainingConfig(
            fusion_mode="cross_attention",
            head_strides=[8, 16, 32],
            assigner_level_ranges=[64.0, 128.0],
            neck_norm="group",
            neck_gn_groups=16,
            ema=True,
            ema_decay=0.98,
            ema_tau=30.0,
        )
        config.to_yaml(tmp_path / "c.yaml")
        loaded = TrainingConfig.from_yaml(tmp_path / "c.yaml")
        assert (loaded.neck_norm, loaded.neck_gn_groups) == ("group", 16)
        assert (loaded.ema, loaded.ema_decay, loaded.ema_tau) == (True, 0.98, 30.0)

    def test_a_config_file_that_predates_the_keys_loads_with_the_defaults(self):
        loaded = TrainingConfig.from_yaml(CROP)
        assert loaded.neck_norm == "batch"
        assert loaded.ema is False


class TestStableExperimentConfig:
    def test_differs_from_the_crop_config_only_by_norm_ema_and_output_dir(self):
        crop = yaml.safe_load(CROP.read_text())
        stable = yaml.safe_load(STABLE.read_text())

        changed = {k for k in crop.keys() | stable.keys() if crop.get(k) != stable.get(k)}
        assert changed == {
            "output_dir",
            "neck_norm",
            "neck_gn_groups",
            "ema",
            "ema_decay",
            "ema_tau",
        }

    def test_loads_with_group_norm_and_ema_on(self):
        config = TrainingConfig.from_yaml(STABLE)
        assert config.output_dir == "checkpoints/twostream_roi_crop_stable"
        assert config.neck_norm == "group"
        assert config.neck_gn_groups == 32
        assert config.ema is True
        assert config.fusion_mode == "cross_attention"
        # Everything else is the crop experiment.
        assert config.roi_crop is True
        assert config.selection_metric == "damage_ap50"

    def test_output_dir_does_not_collide_with_the_crop_experiment(self):
        assert (
            TrainingConfig.from_yaml(STABLE).output_dir
            != TrainingConfig.from_yaml(CROP).output_dir
        )

    def test_every_other_experiment_config_keeps_batch_norm_and_no_ema(self):
        paths = sorted(EXPERIMENT_DIR.glob("*.yaml"))
        assert STABLE in paths
        for path in paths:
            if path == STABLE:
                continue
            config = TrainingConfig.from_yaml(path)
            assert config.neck_norm == "batch", path.name
            assert config.ema is False, path.name


class TestEntryPointsForwardTheNormKeys:
    """Every MasterModel built from a TrainingConfig or a checkpoint must be
    told the norm, or it silently builds BatchNorm and cannot load a
    GroupNorm checkpoint (or trains the wrong architecture)."""

    @pytest.mark.parametrize(
        "relative",
        [
            "src/training/train.py",
            "src/training/kd_trainer.py",
            "scripts/evaluate_checkpoint.py",
            "scripts/visualize_damage_predictions.py",
        ],
    )
    def test_master_model_calls_pass_the_norm_keywords(self, relative):
        tree = ast.parse((REPO_ROOT / relative).read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "MasterModel"
        ]
        assert calls, f"{relative} no longer builds a MasterModel"
        for call in calls:
            passed = {kw.arg for kw in call.keywords}
            assert {"neck_norm", "neck_gn_groups"} <= passed, (
                f"{relative}:{call.lineno} drops the neck norm keys"
            )


def test_neck_norm_module_imports_without_the_model_package():
    """`TrainingConfig` imports the validators; they must not drag in (or
    circularly depend on) the model code."""
    source = (REPO_ROOT / "src" / "training" / "neck_norm.py").read_text()
    imported = {
        node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(name.startswith("src.models") for name in imported)
