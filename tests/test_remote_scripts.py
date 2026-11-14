"""The rental-machine scripts in `scripts/remote/`: packaging of the v4 dataset and the progress push."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

REMOTE = Path(__file__).resolve().parent.parent / "scripts" / "remote"
PACKAGE = REMOTE / "package_v4.sh"
PUSH = REMOTE / "push_progress.sh"
SETUP = REMOTE / "setup_remote.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run(*args, cwd=None, env=None, check=True) -> subprocess.CompletedProcess:
    full_env = {**os.environ, **(env or {})}
    result = subprocess.run([str(a) for a in args], cwd=cwd, env=full_env, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise AssertionError(f"{args} -> {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result


@pytest.mark.parametrize("script", [PACKAGE, PUSH, SETUP])
def test_the_scripts_parse_and_are_executable(script) -> None:
    _run("bash", "-n", script)
    assert os.access(script, os.X_OK)


def _dataset(tmp_path: Path, folds: bool = True) -> Path:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    ds = tmp_path / "v4_poly"
    for sub in ("rgb", "nir", "nir_clahe", "nir_clahe_view", "labels", "homographies"):
        (ds / sub).mkdir(parents=True)
    for i in range(3):
        (outside / f"mango_rgb_{i}.jpg").write_bytes(b"jpg%d" % i * 50)
        (ds / "rgb" / f"mango_rgb_{i}.jpg").symlink_to(outside / f"mango_rgb_{i}.jpg")
        (ds / "nir" / f"mango_nir_{i}.png").write_bytes(b"raw%d" % i * 80)
        (ds / "nir_clahe" / f"mango_nir_{i}.png").write_bytes(b"cla%d" % i * 80)
        (ds / "nir_clahe_view" / f"mango_nir_{i}.png").write_bytes(b"view%d" % i * 80)
        (ds / "labels" / f"mango_rgb_{i}.txt").write_text("1 0.1 0.1 0.2 0.1 0.2 0.2\n")
    (ds / "homographies" / "identity.npy").write_bytes(b"npy")
    (ds / "index.csv").write_text("stem,lot\nmango_rgb_0,1\n")
    (ds / "report.json").write_text("{}")
    if folds:
        (ds / "folds_lot" / "fold_0" / "labels").mkdir(parents=True)
        (ds / "folds_lot" / "folds_meta.json").write_text("{}")
    return ds


def _extract(archive: Path, target: Path) -> Path:
    target.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    (root,) = list(target.iterdir())
    return root


def test_the_archive_dereferences_rgb_and_leaves_out_raw_nir_and_the_view_folder(tmp_path) -> None:
    ds = _dataset(tmp_path)
    archive = tmp_path / "out.tar.gz"
    _run("bash", PACKAGE, ds, "--out", archive, env={"PACKAGE_V4_COMPRESSOR": "gzip"})
    root = _extract(archive, tmp_path / "x")
    assert root.name == "v4_poly"
    rgb = sorted((root / "rgb").iterdir())
    assert len(rgb) == 3 and not any(p.is_symlink() for p in rgb) and rgb[0].read_bytes().startswith(b"jpg0")
    for kept in ("nir_clahe", "labels", "homographies", "folds_lot/folds_meta.json", "index.csv", "report.json"):
        assert (root / kept).exists(), kept
    assert not (root / "nir").exists() and not (root / "nir_clahe_view").exists()
    check = _run("sha256sum", "-c", "--quiet", "MANIFEST.sha256", cwd=root)
    assert check.returncode == 0
    assert "rgb/mango_rgb_0.jpg" in (root / "MANIFEST.sha256").read_text()


def test_raw_nir_is_included_on_request(tmp_path) -> None:
    ds = _dataset(tmp_path)
    archive = tmp_path / "out.tar.gz"
    _run("bash", PACKAGE, ds, "--out", archive, "--with-raw-nir", env={"PACKAGE_V4_COMPRESSOR": "gzip"})
    assert (_extract(archive, tmp_path / "x") / "nir" / "mango_nir_0.png").exists()


def test_a_dataset_without_folds_needs_the_raw_nir_to_build_them_remotely(tmp_path) -> None:
    ds = _dataset(tmp_path, folds=False)
    result = _run("bash", PACKAGE, ds, "--out", tmp_path / "o.tar.gz", "--dry-run", check=False)
    assert result.returncode != 0 and "folds_lot" in result.stderr
    ok = _run("bash", PACKAGE, ds, "--out", tmp_path / "o.tar.gz", "--dry-run", "--with-raw-nir", check=False)
    assert ok.returncode == 0


def test_dry_run_lists_the_contents_and_the_size_and_writes_nothing(tmp_path) -> None:
    ds = _dataset(tmp_path)
    out = tmp_path / "never.tar.zst"
    result = _run("bash", PACKAGE, ds, "--out", out, "--dry-run")
    assert not out.exists() and not list(tmp_path.glob("never*"))
    text = result.stdout
    assert "rgb/" in text and "nir_clahe/" in text and "folds_lot/" in text and "Estimated size" in text
    assert "nir_clahe_view" in text and "excluded" in text.lower()


def _git(*args, cwd) -> str:
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return _run("git", *args, cwd=cwd, env=env).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    remote = tmp_path / "remote.git"
    _run("git", "init", "--bare", "-b", "main", remote)
    work = tmp_path / "work"
    _run("git", "clone", remote, work)
    (work / "README.md").write_text("x")
    _git("add", ".", cwd=work)
    _git("commit", "-m", "init", cwd=work)
    _git("push", "-u", "origin", "HEAD:main", cwd=work)
    return work, remote


def test_push_progress_commits_only_the_report_outputs_to_the_dated_branch(repo, tmp_path) -> None:
    work, remote = repo
    out = work / "reports" / "thesis-export"
    out.mkdir(parents=True)
    (out / "runs.csv").write_text("a,b\n1,2\n")
    (out / "predictions.jsonl").write_text("{}\n")  # heavy artefact: never committed
    (work / "checkpoints").mkdir()
    (work / "checkpoints" / "best_model.pt").write_bytes(b"w")
    (work / "logs").mkdir()
    (work / "logs" / "run.log").write_text("log")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    args = ("bash", PUSH, "--repo", work, "--no-export", "--branch", "runs/thesis-test", "--worktree-dir", tmp_path / "wt")
    first = _run(*args, env=env)
    files = _git("ls-tree", "-r", "--name-only", "runs/thesis-test", cwd=remote).splitlines()
    assert "reports/thesis-export/runs.csv" in files
    assert not any(f.startswith(("checkpoints", "logs")) or f.endswith(".pt") or "predictions" in f for f in files)
    assert "pushed" in first.stdout.lower()
    before = _git("rev-parse", "runs/thesis-test", cwd=remote)
    second = _run(*args, env=env)
    assert "nothing new" in second.stdout.lower()
    assert _git("rev-parse", "runs/thesis-test", cwd=remote) == before
    (out / "runs.csv").write_text("a,b\n1,3\n")
    _run(*args, env=env)
    assert _git("rev-parse", "runs/thesis-test", cwd=remote) != before
    assert _git("log", "-1", "--format=%s%n%b", "runs/thesis-test", cwd=remote).lower().count("co-authored") == 0


def test_push_progress_refuses_an_output_outside_reports_and_test_flags_need_their_pair(repo, tmp_path) -> None:
    work, _ = repo
    bad = _run("bash", PUSH, "--repo", work, "--no-export", "--out", "data/x", check=False)
    assert bad.returncode != 0 and "reports/" in bad.stderr
    lone = _run("bash", PUSH, "--repo", work, "--test-folds", "0", "--dry-run", check=False)
    assert lone.returncode != 0 and "--test-seeds" in lone.stderr


def test_push_progress_does_not_pass_test_flags_unless_given(repo) -> None:
    work, _ = repo
    plain = _run("bash", PUSH, "--repo", work, "--dry-run").stdout
    assert "export_run_results.py" in plain and "--test-" not in plain
    given = _run("bash", PUSH, "--repo", work, "--dry-run", "--test-folds", "0", "1", "--test-seeds", "42").stdout
    assert "--test-folds 0 1" in given and "--test-seeds 42" in given


def test_setup_passes_the_image_size_to_the_smoke_and_to_env_sh() -> None:
    text = SETUP.read_text()
    assert 'IMAGE_SIZE=800' in text
    assert '--image-size) IMAGE_SIZE=' in text
    assert '--image-size "$IMAGE_SIZE"' in text
    assert "export IMAGE_SIZE=" in text
