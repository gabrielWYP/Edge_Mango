"""D-H: a machine profile must actually reach every stage of a real run.

`tests/training/test_machine.py` covers the whitelist and the
`grad_accum_steps` derivation. This file covers the plumbing instead: the KD
entry point has to accept `--machine`, and the final-pipeline runner has to
forward it to all three stages. Without both, the profile is unreachable from
the command that actually trains the pipeline, and the batch size has to be
edited into the experiment YAML — exactly what D-H exists to prevent.

See openspec/changes/fusion-redesign/design.md D-H.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import scripts.run_final_training_pipeline as pipeline
from src.training.kd_config import KDConfig
from src.training.kd_train import parse_args as kd_parse_args
from src.training.machine import (
    MACHINE_PROFILE_WHITELIST,
    apply_machine_profile,
    derive_grad_accum_steps,
    load_machine_profile,
)

RTX3080_PROFILE = Path("configs/machines/rtx3080.yaml")


class TestKDTrainMachineFlag:
    """The KD entry point accepts and applies a machine profile."""

    def test_machine_flag_defaults_to_none(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["kd_train"])
        assert kd_parse_args().machine is None

    def test_machine_flag_is_parsed(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["kd_train", "--machine", str(RTX3080_PROFILE)]
        )
        assert kd_parse_args().machine == str(RTX3080_PROFILE)

    def test_profile_applies_to_kd_config(self, tmp_path):
        profile_path = tmp_path / "probe.yaml"
        profile_path.write_text("batch_size: 4\nnum_workers: 4\ndevice: cpu\n")

        config = KDConfig()
        original_effective_batch = config.effective_batch
        apply_machine_profile(config, load_machine_profile(profile_path))

        assert config.batch_size == 4
        assert config.num_workers == 4
        assert config.device == "cpu"
        # The experiment constant is untouched, so the smaller physical batch
        # is compensated by accumulation rather than silently changing what
        # was measured.
        assert config.effective_batch == original_effective_batch


class TestShippedRTX3080Profile:
    """The profile this machine runs with must stay loadable and divisible."""

    def test_profile_only_sets_whitelisted_keys(self):
        profile = load_machine_profile(RTX3080_PROFILE)
        assert set(profile) <= MACHINE_PROFILE_WHITELIST

    def test_profile_batch_size_divides_the_default_effective_batch(self):
        profile = load_machine_profile(RTX3080_PROFILE)
        config = KDConfig()
        apply_machine_profile(config, profile)

        assert derive_grad_accum_steps(config.effective_batch, config.batch_size) == 2


class TestPipelineForwardsMachine:
    """The runner forwards one profile to master, student, and KD alike."""

    def test_stage_command_omits_machine_when_absent(self, tmp_path):
        command = pipeline.run_training_stage(
            python="python",
            module="src.training.train",
            config_path="configs/training_mango.yaml",
            output_dir=tmp_path,
            dry_run=True,
            model="master",
        )

        assert "--machine" not in command

    def test_stage_command_places_machine_before_override(self, tmp_path):
        command = pipeline.run_training_stage(
            python="python",
            module="src.training.train",
            config_path="configs/training_mango.yaml",
            output_dir=tmp_path,
            dry_run=True,
            model="master",
            machine=str(RTX3080_PROFILE),
        )

        machine_index = command.index("--machine")
        assert command[machine_index + 1] == str(RTX3080_PROFILE)
        # Both entry points apply the profile first and the overrides second;
        # the recorded command has to reproduce that order.
        assert machine_index < command.index("--override")

    def test_all_three_stages_receive_the_profile(self, tmp_path, monkeypatch):
        commands: list[list[str]] = []
        monkeypatch.setattr(
            pipeline, "run_command", lambda command, dry_run: commands.append(command)
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_final_training_pipeline",
                "--dry-run",
                "--machine",
                str(RTX3080_PROFILE),
                "--output-root",
                str(tmp_path),
            ],
        )

        assert pipeline.main() == 0

        assert len(commands) == 3
        modules = [command[2] for command in commands]
        assert modules == [
            "src.training.train",
            "src.training.train",
            "src.training.kd_train",
        ]
        for command in commands:
            assert command[command.index("--machine") + 1] == str(RTX3080_PROFILE)
