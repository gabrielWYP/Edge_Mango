"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

Extra diagnostics (CPU only): threshold sweep, loc-vs-cls AP, FP size plateau, FPs that enclose GT.

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
torch.set_num_threads(6)
import fp_analyze as fa
from src.training.metrics import _compute_ap_at_iou, _compute_operating_point, _cxcywh_to_xyxy
from torchvision.ops import box_iou

recs_all=json.load(open(OUT/"fp_records.json"))
res={}
for tag in fa.TAGS:
    cache=torch.load(OUT/"cache"/f"{tag}.pt",weights_only=False)
    cfg,boxes,scores,labels=fa.decode_all(cache,0.5)
    gtb=[g[0] for g in cache["gts"]]; gtl=[g[1] for g in cache["gts"]]
    pxy=[_cxcywh_to_xyxy(b) if len(b) else b for b in boxes]
    gxy=[_cxcywh_to_xyxy(b) if len(b) else b for b in gtb]
    r={}
    # threshold sweep (damage class)
    sweep={}
    for thr in [0.25,0.30,0.35,0.40,0.50]:
        _,_,_,c=_compute_operating_point(pxy,scores,labels,gxy,gtl,iou_threshold=0.5,score_threshold=thr,num_classes=2)
        sweep[str(thr)]=c[1]
    r["threshold_sweep_nms0.5"]=sweep
    # AP at looser IoU (localization vs classification)
    ap={}
    for it in [0.1,0.3,0.5]:
        _,pc=_compute_ap_at_iou(pxy,scores,labels,gxy,gtl,it,2)
        ap[str(it)]=pc[1]
    r["damage_ap_by_iou"]=ap
    # GT-centric: best IoU among preds>=0.25 per damage GT
    best=[]
    for i in range(len(pxy)):
        g=gxy[i][gtl[i]==1]
        if len(g)==0: continue
        m=(labels[i]==1)&(scores[i]>=0.25)
        if m.sum()==0:
            best+= [0.0]*len(g); continue
        ious=box_iou(g,pxy[i][m]); best+=ious.max(dim=1).values.tolist()
    best=np.array(best)
    r["gt_n"]=int(len(best)); r["gt_best_iou_ge_0.5"]=int((best>=0.5).sum()); r["gt_best_iou_ge_0.3"]=int((best>=0.3).sum()); r["gt_best_iou_ge_0.1"]=int((best>=0.1).sum())
    # FP-level: encloses GT, size plateau, IoU hist of (b)
    rr=[x for x in recs_all if x["tag"]==tag]
    enc=0; nfp=0; enc_c=0; nc=0
    for x in rr:
        if x["cat"]=="TP": continue
        nfp+=1
        i=x["img"]; g=gtb[i][gtl[i]==1]
        fb=torch.tensor([[x["cx"]-x["w"]/2,x["cy"]-x["h"]/2,x["cx"]+x["w"]/2,x["cy"]+x["h"]/2]])
        encl=False
        if len(g):
            gx=_cxcywh_to_xyxy(g)
            inter=(torch.min(fb[:,None,2:],gx[None,:,2:])-torch.max(fb[:,None,:2],gx[None,:,:2])).clamp(min=0).prod(-1)[0]
            area=(gx[:,2]-gx[:,0])*(gx[:,3]-gx[:,1])
            encl=bool(((inter/area)>=0.8).any())  # FP box covers >=80% of some GT box
        enc+=encl
        if x["cat"]=="c_background_on_fruit":
            nc+=1; enc_c+=encl
    r["fp_total"]=nfp; r["fp_encloses_gt_ge80pct"]=enc; r["fp_c_total"]=nc; r["fp_c_encloses_gt"]=enc_c
    fps=[x for x in rr if x["cat"]!="TP"]
    w=np.array([x["w"]*640 for x in fps]); h=np.array([x["h"]*640 for x in fps]); s=np.array([x["score"] for x in fps])
    if len(w):
        mw,mh=np.median(w),np.median(h)
        r["fp_size_within10pct_of_median"]=float(np.mean((np.abs(w-mw)/mw<0.10)&(np.abs(h-mh)/mh<0.10)))
        r["fp_size_px_p10_p50_p90_w"]=[float(np.percentile(w,p)) for p in (10,50,90)]
        r["fp_size_px_p10_p50_p90_h"]=[float(np.percentile(h,p)) for p in (10,50,90)]
        r["fp_score_range_p1_p99"]=[float(np.percentile(s,1)),float(np.percentile(s,99))]
    b_iou=[x["max_iou"] for x in fps if x["cat"]=="b_localization"]
    r["b_iou_hist_0.1-0.2,0.2-0.3,0.3-0.4,0.4-0.5"]=np.histogram(b_iou,bins=[0.1,0.2,0.3,0.4,0.5])[0].tolist() if b_iou else []
    # GT size
    gw=[];gh=[]
    for i in range(len(gtb)):
        g=gtb[i][gtl[i]==1]
        gw+= (g[:,2]*640).tolist(); gh+=(g[:,3]*640).tolist()
    r["gt_size_px_median_wh"]=[float(np.median(gw)),float(np.median(gh))]
    r["max_damage_score_any"]=float(max((s_[l==1].max().item() if (l==1).any() else 0.0) for s_,l in zip(scores,labels)))
    res[tag]=r
    print(tag, json.dumps(r))
json.dump(res,open(OUT/"extra_diagnostics.json","w"),indent=2)
