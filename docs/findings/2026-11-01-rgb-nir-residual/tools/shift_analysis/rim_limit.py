"""The rigid-shift limit: error by box position on the fruit, and how much of each box lands on the RGB fruit."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd
WT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WT))
from src.annotation.mango_registration import rgb_mango_mask
pd.set_option("display.width",220); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",100)
S="/home/gabo/v4-residual-scratch-gnCJnS/"
OUT=Path(S+"build/fruit")
idx=pd.read_csv(OUT/"index.csv"); rep=json.loads((OUT/"report.json").read_text())
def on_fruit(r):
    m=(rgb_mango_mask(cv2.imread(str(OUT/"rgb"/f"mango_rgb_{r['timestamp']}.jpg"),cv2.IMREAD_REDUCED_COLOR_4))>0).astype(np.float64)
    ii=cv2.integral(m)
    def frac(b):
        x0,y0,x1,y1=[int(np.clip(v/4,0,lim)) for v,lim in zip(b,(816,612,816,612))]
        a=(x1-x0)*(y1-y0)
        return float((ii[y1,x1]-ii[y0,x1]-ii[y1,x0]+ii[y0,x0])/a) if a>0 else 0.0
    b=np.array([r["damage_x0"],r["damage_y0"],r["damage_x1"],r["damage_y1"]])
    d=np.array([r["residual_dx"],r["residual_dy"]]*2)
    par=rep["residual"][f"lot{r['lot']}"][str(r["ronda_h"])]
    p=np.array([par["parallax_dx"],par["parallax_dy"]]*2)
    return dict(timestamp=r["timestamp"],final=frac(b),honly=frac(b-d),outline=frac(b-p))
if __name__=="__main__":
    cv2.setNumThreads(1)
    d=idx[(idx.n_damage>0)&(idx.view=="nir")].copy()
    with ProcessPoolExecutor(10) as pool:
        f=pd.DataFrame(list(pool.map(on_fruit,d.to_dict("records"),chunksize=16)))
    d=d.merge(f,on="timestamp")
    d["cx"]=(d.damage_x0+d.damage_x1)/2; d["cy"]=(d.damage_y0+d.damage_y1)/2
    d["r"]=np.hypot((d.cx-(d.mango_x0+d.mango_x1)/2)/((d.mango_x1-d.mango_x0)/2),(d.cy-(d.mango_y0+d.mango_y1)/2)/((d.mango_y1-d.mango_y0)/2))
    print("share of the NIR-drawn box on the RGB fruit: median and share under 0.9 / 0.75, per lot")
    rows=[]
    for lot,g in d.groupby("lot"):
        r=dict(lot=lot,n=len(g))
        for k in("honly","outline","final"):
            r|={k+"_med":round(g[k].median(),3),k+"<.9":round((g[k]<.9).mean(),3),k+"<.75":round((g[k]<.75).mean(),3)}
        rows.append(r)
    print(pd.DataFrame(rows).T.to_string())
    print("all lots",{k:(round(d[k].median(),3),round((d[k]<.9).mean(),3),round((d[k]<.75).mean(),3)) for k in("honly","outline","final")})
    b=pd.read_pickle(S+"blob_m.pkl")[["ts","a_dx","a_dy","a_c","a_edge"]].merge(d,left_on="ts",right_on="timestamp")
    b=b[(b.a_c>=.8)&(b.a_edge==0)]
    b["x"]=b.a_dx-b.residual_dx; b["y"]=b.a_dy-b.residual_dy
    print("blob reference by box position (r: centre of the box, 0 centre of the mango box, 1 its edge)")
    g=b.groupby(pd.cut(b.r,[0,.3,.5,.7,2]),observed=True)
    print(pd.DataFrame(dict(n=g.size(),bias_x=g.x.median(),err=g.apply(lambda s:np.hypot(s.x,s.y).median()),p90=g.apply(lambda s:np.hypot(s.x,s.y).quantile(.9)))).round(1))
    print("share of boxes by position",pd.cut(d.r,[0,.3,.5,.7,2]).value_counts(normalize=True).sort_index().round(3).to_dict())
    print("by hour, on the RGB fruit (final):",d.groupby("ronda_h").final.median().round(3).to_dict(),"share<.75",d.groupby("ronda_h").final.apply(lambda s:(s<.75).mean()).round(3).to_dict())
