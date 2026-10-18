"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

Finding 1 tables (training instability): recomputed from the per-epoch metrics of the baseline LOLO run
(checkpoints/kfold_lot/twostream_roi_crop/fold*/seed42/metrics_history.csv, read from --data-root) and from the
fold-0 stability pair kept in evidence/stability/ (baseline vs EMA, 60 epochs, unregistered NIR).

Prints the tables and writes finding1_tables.json to --out-dir. Pure standard library, CPU only.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

from _paths import resolve_roots

_, DATA, OUT, _ = resolve_roots(__doc__, repo=False)
STABILITY_DIR = Path(__file__).resolve().parents[1] / "evidence" / "stability"
EPOCHS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RUN = "checkpoints/kfold_lot/twostream_roi_crop/fold{k}/seed42/metrics_history.csv"


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    result = {"epoch_tables": {}, "best_epoch_per_fold": {}}
    for k in (1, 3):
        rows = read(DATA / RUN.format(k=k))
        table = []
        for r in rows:
            if int(r["epoch"]) in EPOCHS:
                table.append({
                    "epoch": int(r["epoch"]),
                    "train_cls_loss": round(float(r["cls_loss"]), 3),
                    "train_box_loss": round(float(r["box_loss"]), 3),
                    "val_ap50_class_1": round(float(r["ap50_class_1"]), 4),
                    "val_tp_class_1": int(float(r["tp_class_1"])),
                    "val_fp_class_1": int(float(r["fp_class_1"])),
                    "val_fn_class_1": int(float(r["fn_class_1"])),
                    "val_ap50_class_0": round(float(r["ap50_class_0"]), 4),
                    "optimizer_steps": int(float(r["steps_taken"])),
                })
        result["epoch_tables"][f"fold{k}"] = table
        print(f"fold {k}")
        print("epoch cls_loss box_loss | val ap50_dmg tp fp fn | ap50_mango")
        for t in table:
            print(f"{t['epoch']:5d} {t['train_cls_loss']:8.3f} {t['train_box_loss']:8.3f} | {t['val_ap50_class_1']:.4f} "
                  f"{t['val_tp_class_1']:3d} {t['val_fp_class_1']:5d} {t['val_fn_class_1']:3d} | {t['val_ap50_class_0']:.4f}")

    best_ap = []
    print("\nfold best_epoch val_dmg_ap50 mango_ap50_at_best tp_at_best  optimizer_steps/epoch  train_cls_loss@100")
    for k in range(5):
        rows = read(DATA / RUN.format(k=k))
        ap1 = [float(r["ap50_class_1"]) for r in rows]
        b = max(range(len(ap1)), key=lambda i: ap1[i])
        steps = sorted({int(float(r["steps_taken"])) for r in rows})
        entry = {
            "best_epoch": int(rows[b]["epoch"]),
            "val_ap50_class_1": round(ap1[b], 4),
            "val_ap50_class_0_at_best": round(float(rows[b]["ap50_class_0"]), 4),
            "val_tp_class_1_at_best": int(float(rows[b]["tp_class_1"])),
            "optimizer_steps_per_epoch_values": steps,
            "train_cls_loss_epoch100": round(float(rows[-1]["cls_loss"]), 3),
            "epochs_after_30_with_mango_ap50_below_0.1": sum(
                1 for r in rows if int(r["epoch"]) > 30 and float(r["ap50_class_0"]) < 0.1),
        }
        result["best_epoch_per_fold"][f"fold{k}"] = entry
        best_ap.append(ap1[b])
        print(f"{k}     {entry['best_epoch']:3d}      {entry['val_ap50_class_1']:.4f}       {entry['val_ap50_class_0_at_best']:.4f}"
              f"             {entry['val_tp_class_1_at_best']:3d}        {steps}        {entry['train_cls_loss_epoch100']}")
    result["val_damage_ap50_mean_sd"] = [round(st.mean(best_ap), 3), round(st.stdev(best_ap), 3)]
    print("val damage AP50 at the selected epoch, mean / sample sd:", result["val_damage_ap50_mean_sd"])

    result["stability_fold0"] = {}
    for name in ("base", "ema"):
        rows = read(STABILITY_DIR / f"kfold_stability_fold0_{name}.csv")
        ap = [float(r["ap50_class_1"]) for r in rows]
        jumps = [abs(ap[i] - ap[i - 1]) for i in range(1, len(ap))]
        entry = {
            "epochs": len(ap),
            "best_val_ap50_class_1": round(max(ap), 4),
            "best_epoch": int(rows[ap.index(max(ap))]["epoch"]),
            "last10_mean": round(st.mean(ap[-10:]), 4),
            "max_epoch_to_epoch_jump": round(max(jumps), 4),
            "mean_epoch_to_epoch_jump": round(st.mean(jumps), 4),
        }
        result["stability_fold0"][name] = entry
        print(f"\nstability fold0 {name}: {entry}")

    (OUT / "finding1_tables.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
