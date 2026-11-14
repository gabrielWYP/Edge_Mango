"""`scripts/remote/publish_folds.sh`: one GitHub release asset per finished fold, uploaded once."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "remote" / "publish_folds.sh"
ARMS = ("master_a", "crosskd_b")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _fake_gh(bin_dir: Path, log: Path, store: Path) -> None:
    """A `gh` that records its calls and keeps uploaded files; `release view` fails until a create."""
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> "{log}"\n'
        f'if [ "$1 $2" = "release view" ]; then [ -e "{store}/.created" ]; exit $?; fi\n'
        f'if [ "$1 $2" = "release create" ]; then mkdir -p "{store}"; touch "{store}/.created"; exit 0; fi\n'
        'if [ "$1 $2" = "release upload" ]; then shift 3; for f in "$@"; do case "$f" in --*) ;; '
        f'*) cp "$f" "{store}/";; esac; done; exit 0; fi\n'
        "exit 0\n"
    )
    gh.chmod(0o755)


def _run_files(runs: Path, arm: str, fold: int, splits=("val", "test")) -> None:
    run = runs / arm / f"fold{fold}" / "seed42"
    (run / "eval_scratch").mkdir(parents=True)
    (run / "best_model.pt").write_bytes(b"best")
    (run / "last.pt").write_bytes(b"last" * 100)
    (run / "run_complete.json").write_text("{}")
    (run / "history.csv").write_text("epoch,loss\n1,0.5\n")
    (run / "eval_scratch" / "tmp.bin").write_bytes(b"x")
    report = runs / "reports" / arm / f"fold{fold}_seed42"
    for split in splits:
        (report / split).mkdir(parents=True)
        (report / split / "predictions.jsonl").write_text('{"image": 1}\n')
        (report / split / "metrics.json").write_text("{}")
        (report / split / "metrics_v4.json").write_text("{}")
    (runs / "reports" / "logs").mkdir(parents=True, exist_ok=True)
    (runs / "reports" / "logs" / f"{arm}_fold{fold}_seed42.log").write_text("epoch 1 | 10s\n")


def _publish(tmp_path: Path, runs: Path, *extra: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}
    args = ["bash", str(SCRIPT), "--runs-root", str(runs), "--tag", "thesis-test", "--arms", " ".join(ARMS),
            "--folds", "0 1", "--no-push-progress", *extra]
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture
def setup(tmp_path: Path):
    log, store = tmp_path / "gh.log", tmp_path / "store"
    _fake_gh(tmp_path / "bin", log, store)
    runs = tmp_path / "runs"
    runs.mkdir()
    return runs, log, store


def test_the_script_parses_and_is_executable() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert os.access(SCRIPT, os.X_OK)


def test_an_unfinished_fold_is_not_published(tmp_path, setup) -> None:
    runs, log, store = setup
    _run_files(runs, ARMS[0], 0)
    _run_files(runs, ARMS[1], 0, splits=("val",))  # test evaluation still missing
    _publish(tmp_path, runs)
    assert not log.exists() or "upload" not in log.read_text()


def test_a_finished_fold_is_uploaded_once_with_everything_needed_for_the_analysis(tmp_path, setup) -> None:
    runs, log, store = setup
    for arm in ARMS:
        _run_files(runs, arm, 0)
    _publish(tmp_path, runs)
    _publish(tmp_path, runs)
    assert log.read_text().count("release upload") == 1
    assert log.read_text().count("release create") == 1
    with tarfile.open(store / "fold0_seed42.tar.gz") as tar:
        names = set(tar.getnames())
    for arm in ARMS:
        assert f"{arm}/fold0/seed42/best_model.pt" in names
        assert f"{arm}/fold0/seed42/run_complete.json" in names
        assert f"{arm}/fold0/seed42/history.csv" in names
        assert f"reports/{arm}/fold0_seed42/test/predictions.jsonl" in names
        assert f"reports/{arm}/fold0_seed42/val/metrics_v4.json" in names
        assert f"reports/logs/{arm}_fold0_seed42.log" in names
    assert not any(n.endswith("last.pt") or "eval_scratch" in n for n in names)
    assert not (runs / "fold0_seed42.tar.gz").exists()


def test_val_only_folds_are_finished_without_test(tmp_path, setup) -> None:
    runs, log, store = setup
    for arm in ARMS:
        _run_files(runs, arm, 1, splits=("val",))
    _publish(tmp_path, runs, "--val-only")
    assert (store / "fold1_seed42.tar.gz").exists()
