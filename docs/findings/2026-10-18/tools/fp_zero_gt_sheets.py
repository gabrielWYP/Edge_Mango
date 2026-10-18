"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

One contact sheet per fold (0-3) with the test images that have no damage GT, annotated with their
prediction count at score >= 0.25. Needs fp_forward.py (cache) and fp_analyze.py (taxonomy_summary.json) outputs in --out-dir.
"""
import sys, json
sys.dont_write_bytecode=True
import cv2, numpy as np, torch
from _paths import resolve_roots
_,_,_OUT,_=resolve_roots(__doc__,repo=False,data=False)
OUT=str(_OUT)
S=json.load(open(OUT+"/taxonomy_summary.json"))
T=290; COLS=5
for k in range(4):
    tag=f"f{k}_best"
    c=torch.load(f"{OUT}/cache/{tag}.pt",weights_only=False)
    z=[p for p in S[tag]["per_image"] if p["n_gt"]==0]
    tiles=[]
    for p in z:
        i=p["img"]
        im=cv2.cvtColor(c["imgs"][i],cv2.COLOR_RGB2BGR)
        # crop the mango GT box (+pad) so the fruit fills the tile
        gb,gl=c["gts"][i]; m=gb[gl==0][0]*640
        cx,cy,w,h=[float(v) for v in m]; s=max(w,h)*1.05
        x0=int(max(0,min(640-s,cx-s/2))); y0=int(max(0,min(640-s,cy-s/2))); s=int(min(s,640))
        im=cv2.resize(im[y0:y0+s,x0:x0+s],(T,T),interpolation=cv2.INTER_AREA)
        cv2.putText(im,f"img{i} preds>=.25: {p['n_pred']}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(im,f"img{i} preds>=.25: {p['n_pred']}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA) if False else None
        tiles.append(im)
    while len(tiles)%COLS: tiles.append(np.full((T,T,3),30,np.uint8))
    sheet=np.vstack([np.hstack(tiles[r:r+COLS]) for r in range(0,len(tiles),COLS)])
    cv2.imwrite(f"{OUT}/zero_gt_images_fold{k}.png",sheet); print(k,len(z),sheet.shape)
