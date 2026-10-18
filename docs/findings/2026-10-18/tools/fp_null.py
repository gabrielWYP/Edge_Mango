"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

Null baseline: keep every damage prediction's size and score, but re-place its center uniformly at
random inside the GT mango box. If real AP50 is not clearly above this, the detector's damage
localisation is not distinguishable from chance placement on the fruit.

Needs fp_forward.py (cache) and fp_analyze.py (fp_records.json) outputs in --out-dir.
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.dont_write_bytecode=True
from pathlib import Path
from _paths import resolve_roots
REPO,M,OUT,_=resolve_roots(__doc__)  # M = data root; REPO = code root
os.chdir(M); sys.path.insert(0,str(REPO))
import numpy as np, torch
torch.set_num_threads(8)
import fp_analyze as fa
from src.training.metrics import _compute_ap_at_iou, _cxcywh_to_xyxy
rng=np.random.default_rng(0)
recs=json.load(open(OUT/"fp_records.json"))
out={}
for tag in fa.TAGS:
    cache=torch.load(OUT/"cache"/f"{tag}.pt",weights_only=False)
    cfg,boxes,scores,labels=fa.decode_all(cache,0.5)
    gtb=[g[0] for g in cache["gts"]]; gtl=[g[1] for g in cache["gts"]]
    gxy=[_cxcywh_to_xyxy(b) if len(b) else b for b in gtb]
    mango=[ _cxcywh_to_xyxy(gtb[i][gtl[i]==0])[0] for i in range(len(gtb))]
    real_xy=[_cxcywh_to_xyxy(b) if len(b) else b for b in boxes]
    _,pc=_compute_ap_at_iou(real_xy,scores,labels,gxy,gtl,0.5,2); real=pc[1]
    aps=[]
    for rep in range(10):
        shifted=[]
        for i,b in enumerate(boxes):
            if len(b)==0: shifted.append(b); continue
            m=mango[i]; nb=b.clone()
            nb[:,0]=torch.from_numpy(rng.uniform(float(m[0]),float(m[2]),len(b))).float()
            nb[:,1]=torch.from_numpy(rng.uniform(float(m[1]),float(m[3]),len(b))).float()
            shifted.append(_cxcywh_to_xyxy(nb))
        _,pc=_compute_ap_at_iou(shifted,scores,labels,gxy,gtl,0.5,2); aps.append(pc[1])
    # AUC of score (TP vs FP) at >=0.25
    tp=[r["score"] for r in recs if r["tag"]==tag and r["cat"]=="TP"]
    fp=[r["score"] for r in recs if r["tag"]==tag and r["cat"]!="TP"]
    auc=None
    if tp and fp:
        a=np.array(tp); b=np.array(fp)
        auc=float(((a[:,None]>b[None,:]).sum()+0.5*(a[:,None]==b[None,:]).sum())/(len(a)*len(b)))
    out[tag]={"real_ap50":real,"null_ap50_mean":float(np.mean(aps)),"null_ap50_sd":float(np.std(aps)),"null_ap50_max":float(np.max(aps)),"auc_tp_vs_fp_score_at_0.25":auc,"n_tp":len(tp)}
    print(tag,out[tag],flush=True)
json.dump(out,open(OUT/"null_baseline.json","w"),indent=2)
