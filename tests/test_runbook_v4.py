"""The commands of docs/tesis/runbook-v4.md: a stale flag here would silently change a result."""

from __future__ import annotations

import re
from pathlib import Path

RUNBOOK = Path(__file__).resolve().parents[1] / "docs" / "tesis" / "runbook-v4.md"


def test_the_build_command_names_the_labels_it_reads() -> None:
    section = RUNBOOK.read_text(encoding="utf-8").split("## 2.")[0]
    assert re.search(r"build_v4_dataset\.py[^`]*--labels data/labels/", section)


def test_the_seed_plan_reads_the_primary_metric() -> None:
    assert re.search(r"plan_seeds\.py[^`]*--metric latent_ap50", RUNBOOK.read_text(encoding="utf-8"))


def test_a_batch_override_is_not_suggested_for_the_students_alone() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "same `--extra-override` in E1" in text


def test_the_commands_use_the_pre_registered_nir_variant() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    variants = re.findall(r"--nir-variant (\S+) \\", text)
    assert variants and set(variants) == {"nir_clahe"}
    assert set(re.findall(r"--reports-root (\S+) \\", text)) == {"reports/v4/nir_clahe"}
