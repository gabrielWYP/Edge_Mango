"""Design checks of the automated capture rig, executed by Blender in background mode.

The Blender script (hardware/capture_rig/build_capture_rig.py) builds the whole rig and runs
checks on frustums, tailstock travel, rotation clearance, print volume, manifold meshes, direct
line of sight from the LEDs, frame exclusion and the enclosure envelope. These tests only drive
that script; they are skipped when ``blender`` is not installed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "hardware/capture_rig/build_capture_rig.py"


@pytest.fixture(scope="module")
def blender():
    exe = shutil.which("blender")
    if exe is None:
        pytest.skip("blender is not on PATH")
    return exe


def _run(blender, *script_args):
    cmd = [blender, "-b", "--factory-startup", "--python", SCRIPT, "--", *script_args]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)


def test_capture_rig_design_checks_pass(blender):
    res = _run(blender, "--check-only")
    assert res.returncode == 0, f"design checks failed:\n{res.stdout[-4000:]}\n{res.stderr[-1000:]}"
    assert "ALL CHECKS PASSED" in res.stdout


def test_design_checks_fail_on_an_empty_scene(blender):
    """The checks must have teeth: with nothing built they have to report failures and exit 1."""
    res = _run(blender, "--check-only", "--skip-build")
    assert res.returncode == 1
    assert "ALL CHECKS PASSED" not in res.stdout
    for tag in ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(k)"):
        assert f"FAIL {tag}" in res.stdout, f"check {tag} did not fail on an empty scene"
