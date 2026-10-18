"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

Per-epoch val damage FP/TP/AP50 of the five LOLO folds, binned by epoch (reads metrics_history.csv under --data-root).
"""
import csv, json, statistics as st
from _paths import resolve_roots
_,_M,_OUT,_=resolve_roots(__doc__,repo=False)
M=str(_M); OUT=str(_OUT)
res={}
for k in range(5):
    rows=list(csv.DictReader(open(f"{M}/checkpoints/kfold_lot/twostream_roi_crop/fold{k}/seed42/metrics_history.csv")))
    ep=[int(r["epoch"]) for r in rows]
    fp=[float(r["fp_class_1"]) for r in rows]; tp=[float(r["tp_class_1"]) for r in rows]; fn=[float(r["fn_class_1"]) for r in rows]
    ap=[float(r["ap50_class_1"]) for r in rows]
    best=max(range(len(ap)),key=lambda i:ap[i])
    pick=lambda xs,lo,hi:[x for e,x in zip(ep,xs) if lo<=e<=hi]
    res[k]={"epochs":ep,"fp":fp,"tp":tp,"fn":fn,"ap":ap,"best_epoch":ep[best]}
    print(f"fold{k}: best-val-AP50 epoch {ep[best]} (ap={ap[best]:.3f}, fp={fp[best]:.0f}, tp={tp[best]:.0f})")
    for lo,hi in [(1,5),(6,10),(11,20),(21,30),(31,50),(51,70),(71,90),(91,100)]:
        f=pick(fp,lo,hi); t=pick(tp,lo,hi); a=pick(ap,lo,hi)
        print(f"   ep {lo:>3}-{hi:<3}  fp mean {st.mean(f):7.0f}  tp mean {st.mean(t):5.1f}  ap50 mean {st.mean(a):.3f}")
json.dump({k:{kk:vv for kk,vv in v.items()} for k,v in res.items()},open(f"{OUT}/trend_fp_over_epochs.json","w"))
