import json, sys
from pathlib import Path
import numpy as np, pandas as pd
pd.set_option("display.width",240); pd.set_option("display.max_columns",60); pd.set_option("display.max_rows",300)
S="/home/gabo/v4-residual-scratch-gnCJnS/"
OUT=Path(S+"build/fruit"); WIP=Path("/home/gabo/MCP_Agent/data/cache/v4_captura_v3_wip")
idx=pd.read_csv(OUT/"index.csv"); rep=json.loads((OUT/"report.json").read_text())
ev=pd.read_csv(S+"final_eval.csv")
print("included",rep["num_included"],"excluded",rep["num_excluded"],"params",rep["params"]["residual_shift"]["mode"],{k:v for k,v in rep["params"]["residual_shift"].items() if k not in("mode","method")})
print("sources",idx.residual_source.value_counts().to_dict())
print(idx[idx.residual_source!="fruit"][["stem","lot","ronda_h","residual_dx","residual_dy","residual_corr","residual_source"]].to_string())
rows=[]
for lot,g in ev.groupby("lot"):
    i=idx[idx.lot==lot]
    moments=rep["residual"][f"lot{lot}"]
    rows.append(dict(lot=lot,n=len(g),dx=g.dx.median(),dy=g.dy.median(),par=np.median([m["parallax_dx"] for m in moments.values()]),fb_pct=round(100*(i.residual_source!="fruit").mean(),2),fruit_h=g.fruit_h.median(),fruit_b=g.fruit_b.median(),ring_h=g.ring_h.median(),ring_b=g.ring_b.median(),skin_h=g.skin_h.median(),skin_b=g.skin_b.median(),rcorr=g.rcorr.median()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("per lot and moment (report):")
for lot,ms in rep["residual"].items():
    print(lot," ".join(f"{k}:{v['median_dx']:+.0f},{v['median_dy']:+.0f}(p{v['parallax_dx']:+.0f} a{v['accepted']} f{v['fallback']})" for k,v in ms.items()))
# blob reference
b=pd.read_pickle(S+"blob_m.pkl").merge(idx[["timestamp","residual_dx","residual_dy"]],left_on="ts",right_on="timestamp")
s=(b.a_c>=.8)&(b.a_edge==0)
def iou(ex,ey,bw,bh):
    ix=(bw-ex.abs()).clip(lower=0); iy=(bh-ey.abs()).clip(lower=0); i=ix*iy; return i/(2*bw*bh-i)
rows=[]
for lot,g in b[s].groupby("lot"):
    r=dict(lot=lot,n=len(g))
    for k,(ex,ey) in dict(H=(0*g.a_dx,0*g.a_dx),final=(g.residual_dx,g.residual_dy)).items():
        x,y=g.a_dx-ex,g.a_dy-ey; e=np.hypot(x,y); o=iou(x,y,g.bw,g.bh)
        r|={k+"_bias_x":x.median(),k+"_err":e.median(),k+"_p90":e.quantile(.9),k+"_iou":o.median(),k+"_iou<.5":(o<.5).mean()}
    # outline only: final minus parallax of its group
    par=[rep["residual"][f"lot{lot}"][str(m)]["parallax_dx"] for m in g.ronda]; pary=[rep["residual"][f"lot{lot}"][str(m)]["parallax_dy"] for m in g.ronda]
    x,y=g.a_dx-(g.residual_dx-par),g.a_dy-(g.residual_dy-pary); e=np.hypot(x,y); o=iou(x,y,g.bw,g.bh)
    r|={"out_bias_x":x.median(),"out_err":e.median(),"out_p90":e.quantile(.9),"out_iou":o.median(),"out_iou<.5":(o<.5).mean()}
    rows.append(r)
print("box against the bruise blob reference (c>=0.8, +10/+12 h NIR boxes):")
print(pd.DataFrame(rows).round(2).T.to_string())
# damage in mango
w=pd.read_csv(WIP/"index.csv") if (WIP/"index.csv").exists() else None
d=idx[idx.n_damage>0]
print("damage_in_mango (fruit): ",d.groupby("lot").damage_in_mango.median().round(3).to_dict(),"mostly outside",len(rep["damage_mostly_outside_mango"]))
if w is not None:
    d2=w[w.n_damage>0]
    wr=json.loads((WIP/"report.json").read_text())
    print("damage_in_mango (wip, H only):",d2.groupby("lot").damage_in_mango.median().round(3).to_dict(),"mostly outside",len(wr["damage_mostly_outside_mango"]),"same labels sha:",wr["inputs"]["labels"]["sha256"]==rep["inputs"]["labels"]["sha256"],"rows",len(w))
print("boxes touching the frame border:",int(((d.damage_x0<=0)|(d.damage_y0<=0)|(d.damage_x1>=3264)|(d.damage_y1>=2448)).sum()))
print("nir lines:",rep["nir_lines"])
if w is not None: print("nir lines wip:",wr["nir_lines"])
