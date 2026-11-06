#!/usr/bin/env python3
"""Publish the trained v4 checkpoints as assets of a GitHub Release, with a manifest kept in git.

A checkpoint is ~12 MB (weights in fp32, no optimizer state), and 75+ of them would bloat the git history forever,
so the files go to a Release: the repository only keeps `models/MANIFEST.csv` and `models/SHA256SUMS`, and a clone
downloads no weights. Assets are uploaded unchanged (the exact files that were evaluated), one per checkpoint, with a
flat name `<group>__<arm>__fold<k>__seed<s>__<checkpoint>.pt`:

- `final`: the definitive runs (`runs/final`), the checkpoint selected on validation (`best`) of every finished run,
  and the epoch-80 checkpoint of the runs evaluated for the selection sensitivity (`reports_epoch80`);
- `pilot-<name>`: the `best` checkpoint of the pilots given with `--pilot NAME=DIR`.

Re-running uploads only the assets the Release does not have yet, so it can follow the runs as they finish.

    python scripts/publish_models.py --tag models-v4 --target b34668f --runs-root ~/edge_mango/runs/final \\
        --pilot con_mango=checkpoints/v4_pilots/con_mango --pilot sin_mango=checkpoints/v4_pilots/sin_mango \\
        --pilot rgbnir=checkpoints/v4_pilots_rgbnir/rgbnir_con_mango --output models --upload

Download later with `gh release download models-v4 --repo gabrielWYP/Edge_Mango -p 'final__est_rgbnir__*'` and
check with `sha256sum -c models/SHA256SUMS --ignore-missing`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = "gabrielWYP/Edge_Mango"
ARMS = ("est_rgb", "est_rgbnir", "kd_rgb", "kd_rgbnir", "est_nir")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target", required=True, help="commit the Release tag points to (the training code)")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--pilot", action="append", default=[], help="NAME=DIR with <arm>/fold<k>/seed<s>/")
    parser.add_argument("--output", type=Path, required=True, help="directory of MANIFEST.csv and SHA256SUMS")
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runs(root: Path):
    """(arm, fold, seed, run directory) under root."""
    for arm in ARMS:
        for run in sorted(root.glob(f"{arm}/fold*/seed*")):
            yield arm, int(run.parent.name.removeprefix("fold")), int(run.name.removeprefix("seed")), run


def candidates(args) -> list[dict]:
    out = []
    for arm, fold, seed, run in runs(args.runs_root):
        if not (run / "run_complete.json").exists():
            continue  # still training
        out.append({"group": "final", "arm": arm, "fold": fold, "seed": seed, "checkpoint": "best",
                    "path": run / "best_model.pt"})
        if (args.runs_root / "reports_epoch80" / arm / f"fold{fold}_seed{seed}" / "test" / "metrics_v4.json").exists():
            out.append({"group": "final", "arm": arm, "fold": fold, "seed": seed, "checkpoint": "epoch80",
                        "path": run / "checkpoint_epoch80.pt"})
    for spec in args.pilot:
        name, directory = spec.split("=", 1)
        for arm, fold, seed, run in runs(Path(directory)):
            if (run / "best_model.pt").exists():
                out.append({"group": f"pilot-{name}", "arm": arm, "fold": fold, "seed": seed, "checkpoint": "best",
                            "path": run / "best_model.pt"})
    return out


def val_ap_at(run: Path, epoch) -> float | None:
    """Validation damage AP50 (all hours) of the run at `epoch`, from its metrics_history.csv."""
    history = run / "metrics_history.csv"
    if epoch is None or not history.exists():
        return None
    for row in csv.DictReader(history.open()):
        if row["epoch"] == str(epoch) and row.get("ap50_class_1"):
            return float(row["ap50_class_1"])
    return None


def describe(item: dict) -> dict:
    checkpoint = torch.load(item["path"], map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or {}
    return {
        "asset": f"{item['group']}__{item['arm']}__fold{item['fold']}__seed{item['seed']}__{item['checkpoint']}.pt",
        "group": item["group"], "arm": item["arm"], "fold": item["fold"], "seed": item["seed"],
        "checkpoint": item["checkpoint"], "epoch": checkpoint.get("epoch"),
        "val_damage_ap50_at_epoch": val_ap_at(item["path"].parent, checkpoint.get("epoch")),
        "student_variant": config.get("student_variant"),
        "student_in_channels": config.get("student_in_channels", 3),
        "size_bytes": item["path"].stat().st_size, "sha256": sha256(item["path"]),
    }


def gh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], check=check, capture_output=True, text=True)


def upload(tag: str, target: str, files: list[Path], notes: str) -> int:
    if gh("api", f"repos/{REPO}/releases/tags/{tag}", check=False).returncode != 0:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
            fh.write(notes)
        gh("release", "create", tag, "--repo", REPO, "--target", target, "--title", f"Modelos {tag}",
           "--notes-file", fh.name)
    present = set(json.loads(gh("api", f"repos/{REPO}/releases/tags/{tag}", "--jq", "[.assets[].name]").stdout))
    missing = [f for f in files if f.name not in present]
    for i in range(0, len(missing), 10):
        gh("release", "upload", tag, "--repo", REPO, *map(str, missing[i : i + 10]))
        print(f"uploaded {min(i + 10, len(missing))}/{len(missing)}", flush=True)
    return len(missing)


def main(argv=None) -> int:
    args = parse_args(argv)
    items = candidates(args)
    rows = [describe(item) for item in items]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "MANIFEST.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "SHA256SUMS").write_text("".join(f"{r['sha256']}  {r['asset']}\n" for r in rows))
    print(f"{len(rows)} checkpoints, {sum(r['size_bytes'] for r in rows) / 1e6:.0f} MB")
    if args.upload:
        with tempfile.TemporaryDirectory() as staging:
            files = []
            for item, row in zip(items, rows):
                link = Path(staging) / row["asset"]
                link.symlink_to(Path(item["path"]).resolve())
                files.append(link)
            notes = (f"Checkpoints de los modelos v4 de la tesis. El tag apunta al código de entrenamiento "
                     f"(`{args.target}`). Lista, épocas, puntajes de validación y SHA-256 en `models/MANIFEST.csv` "
                     "del repositorio. Los pilotos se entrenaron con commits anteriores (ver "
                     "`docs/tesis/resultados-causas-y-soluciones.md`).")
            print(f"{upload(args.tag, args.target, files, notes)} new assets on {args.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
