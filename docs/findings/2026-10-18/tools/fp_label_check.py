"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

Contact sheet of four fold-2 test images (indices 23, 4, 30, 2) with their GT damage boxes drawn in green.
Needs the fold-2 cache written by fp_forward.py in --out-dir.
"""
import sys, os, json
sys.dont_write_bytecode=True
import cv2, numpy as np, torch
from _paths import resolve_roots
_,_,_OUT,_=resolve_roots(__doc__,repo=False,data=False)
OUT=str(_OUT)
c=torch.load(f"{OUT}/cache/f2_best.pt",weights_only=False)
tiles=[]
for i in [23,4,30,2]:
    img=cv2.cvtColor(c["imgs"][i].copy(),cv2.COLOR_RGB2BGR)
    gb,gl=c["gts"][i]
    print("img",i,c["names"][i],"n_gt_damage",int((gl==1).sum()), [ [round(float(v)*640) for v in b] for b in gb[gl==1]])
    for b in gb[gl==1]:
        cx,cy,w,h=[float(v)*640 for v in b]
        cv2.rectangle(img,(int(cx-w/2),int(cy-h/2)),(int(cx+w/2),int(cy+h/2)),(0,255,0),2)
    cv2.putText(img,f"f2 img{i} GT damage={int((gl==1).sum())}",(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    tiles.append(cv2.resize(img,(560,560)))
cv2.imwrite(f"{OUT}/fold2_label_check.png",np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
