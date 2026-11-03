"""Exploration only: bruise-blob shift. NCC of band-passed intensity (bruise-sized blobs, shading removed), NIR box patch vs RGB."""
import csv, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd
WT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WT))
from src.data_pipeline.v4_dataset import raw_nir, registered_nir, read_nir
from src.data_pipeline.v4_selection import load_homography_rules, homography_for
DATA = Path("data/cache/captura_v3")
D = pd.read_pickle("/home/gabo/v4-residual-scratch-gnCJnS/explore4.pkl")
rules = load_homography_rules(WT/"configs/data/v4_homographies.yaml")
SCALE=0.25; LO,HI,RY=-100,200,32
def dog(g,s1,s2):
    return cv2.GaussianBlur(g,(0,0),s1)-cv2.GaussianBlur(g,(0,0),s2)
def job(r):
    ts=int(r["ts"])
    H=np.load(WT/homography_for(rules,r["jornada"],ts))
    src=DATA/"news/rgb_wb"/f"mango_rgb_{ts}.jpg"
    if not src.exists():
        src=DATA/"news/rgb"/f"mango_rgb_{ts}.jpg"
    bgr=cv2.imread(str(src),cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    nir=registered_nir(raw_nir(read_nir(DATA/f"news/nir/mango_nir_{ts}.jpg")),H)
    rg=cv2.resize(gray,None,fx=SCALE,fy=SCALE,interpolation=cv2.INTER_AREA).astype(np.float32)
    ng=cv2.resize(nir,None,fx=SCALE,fy=SCALE,interpolation=cv2.INTER_AREA).astype(np.float32)
    rec=dict(ts=ts)
    for name,(s1,s2) in dict(a=(3,24),b=(6,36)).items():
        ra,na=dog(rg,s1,s2),dog(ng,s1,s2)
        x0,y0,x1,y1=(np.array([r["bx0"],r["by0"],r["bx1"],r["by1"]])*SCALE).astype(int)
        tmpl=na[y0:y1,x0:x1]
        # search in the RGB: NIR shift in out + [LO,HI], out_dy +- RY
        dx0,dy0=r["out_dx"]*SCALE,r["out_dy"]*SCALE
        tx0=int(x0+dx0+LO*SCALE); tx1=int(x1+dx0+HI*SCALE); ty0=int(y0+dy0-RY*SCALE); ty1=int(y1+dy0+RY*SCALE)
        if tx0<0 or ty0<0 or tx1>ra.shape[1] or ty1>ra.shape[0] or tmpl.size==0:
            continue
        s=cv2.matchTemplate(ra[ty0:ty1,tx0:tx1],tmpl,cv2.TM_CCOEFF_NORMED)
        _,best,_,(ix,iy)=cv2.minMaxLoc(s)
        rec|={f"{name}_dx":(tx0+ix-x0)/SCALE,f"{name}_dy":(ty0+iy-y0)/SCALE,f"{name}_c":round(float(best),3),f"{name}_edge":int(ix in (0,s.shape[1]-1) or iy in (0,s.shape[0]-1))}
    return rec
def main():
    cv2.setNumThreads(1)
    m=D[(D.ronda.isin(["12","10"]))&(D.view=="nir")]
    rows=m[["ts","jornada","bx0","by0","bx1","by1","out_dx","out_dy"]].to_dict("records")
    with ProcessPoolExecutor(10) as pool:
        recs=list(pool.map(job,rows,chunksize=8))
    pd.DataFrame(recs).to_pickle(sys.argv[1])
main()
