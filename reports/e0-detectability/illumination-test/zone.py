import json,numpy as np,pandas as pd
pd.set_option('display.width',170)
W,H=3264,2448
d=pd.read_csv('/home/gabo/MCP_Agent/data/cache/v4_captura_v3_final/index.csv')
p=d[d.ronda_h!='sano'].copy()
mw=p.mango_x1-p.mango_x0; mh=p.mango_y1-p.mango_y0
p['cx']=((p.damage_x0+p.damage_x1)/2-p.mango_x0)/mw; p['cy']=((p.damage_y0+p.damage_y1)/2-p.mango_y0)/mh
p['rw']=(p.damage_x1-p.damage_x0)/mw; p['rh']=(p.damage_y1-p.damage_y0)/mh
print('A. all lots, by zone')
A=p.groupby('zona').agg(n=('cx','size'),cx=('cx','mean'),cx_sd=('cx','std'),cy=('cy','mean'),cy_sd=('cy','std'),rw=('rw','median'),rh=('rh','median'),nir_yes=('visible_nir',lambda s:(s=='yes').mean()),rgb_yes=('visible_rgb',lambda s:(s=='yes').mean()),rgb_no=('visible_rgb',lambda s:(s=='no').mean()))
print(A.round(3).to_string())
print('zone share per lot');print(pd.crosstab(p.lot,p.zona).to_string())
print('B. E0 faces, latent hours, by zone')
f=pd.read_csv('/home/gabo/MCP_Agent-worktrees/v4-final/reports/e0-detectability/boxes-final/faces.csv')
g=f[(f.status=='ok')] if 'ok' in set(f.status) else f
print(sorted(set(f.status)))
g=f[f.status==f.status.value_counts().index[0]].copy(); g['lat']=g.hours<=6.9
for c in ('R','NIR'):
    g[c+'_rel']=g[f'{c}_impact_sano']/g[f'{c}_control_sano']
print(g.groupby('zone').agg(R_sano=('R_impact_sano','mean'),NIR_sano=('NIR_impact_sano','mean'),R_vs_ctrl=('R_rel','median'),NIR_vs_ctrl=('NIR_rel','median'),n=('lot','size')).round(3).to_string())
print(g[g.lat].groupby('zone')[['R_delta','G_delta','NIR_delta']].mean().round(3).to_string())
print(g[~g.lat].groupby('zone')[['R_delta','G_delta','NIR_delta']].mean().round(3).to_string())
print('C. val lot 2, predictions')
preds={}
for l in open('/home/gabo/MCP_Agent-worktrees/v4-final/reports/v4/nir_clahe/maestro_rgbnir/fold0_seed42/val/predictions.jsonl'):
    x=json.loads(l);preds[x['stem']]=x['boxes']
v=p[p.lot==2].copy()
def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0]));iy=max(0,min(a[3],b[3])-max(a[1],b[1]));i=ix*iy
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-12)
tr=p[p.lot.isin([3,4,5])]
fx=[tr.cx.mean(),tr.cy.mean(),tr.rw.median(),tr.rh.median()]
print('fixed train box (rel mango)',np.round(fx,3))
rows=[]
for _,r in v.iterrows():
    if r.stem not in preds: continue
    gt=[r.damage_x0/W,r.damage_y0/H,r.damage_x1/W,r.damage_y1/H]
    bs=[b for b in preds[r.stem] if b['class']==1]
    top=max(bs,key=lambda b:b['confidence']) if bs else None
    m=[r.mango_x0/W,r.mango_y0/H,(r.mango_x1-r.mango_x0)/W,(r.mango_y1-r.mango_y0)/H]
    fb=[m[0]+m[2]*(fx[0]-fx[2]/2),m[1]+m[3]*(fx[1]-fx[3]/2),m[0]+m[2]*(fx[0]+fx[2]/2),m[1]+m[3]*(fx[1]+fx[3]/2)]
    bests=max([iou(b['xyxy'],gt) for b in bs if b['confidence']>=.25] or [0])
    t=top['xyxy'] if top else None
    rows.append(dict(zona=r.zona,top_iou=iou(t,gt) if t else 0,hit=bests>=.5,fixed_iou=iou(fb,gt),n=sum(b['confidence']>=.25 for b in bs),
      px=((t[0]+t[2])/2-m[0])/m[2] if t else np.nan,py=((t[1]+t[3])/2-m[1])/m[3] if t else np.nan,gx=r.cx,gy=r.cy,conf=top['confidence'] if top else 0))
c=pd.DataFrame(rows)
print(c.groupby('zona').agg(n=('hit','size'),hit25=('hit','mean'),top_iou=('top_iou','mean'),top50=('top_iou',lambda s:(s>=.5).mean()),fixed_iou=('fixed_iou','mean'),fixed50=('fixed_iou',lambda s:(s>=.5).mean()),preds=('n','mean'),conf=('conf','mean'),pred_x=('px','mean'),pred_xsd=('px','std'),pred_y=('py','mean'),pred_ysd=('py','std'),gt_x=('gx','mean'),gt_y=('gy','mean')).round(3).to_string())
print('corr pred vs gt position: x %.3f y %.3f'%(c.px.corr(c.gx),c.py.corr(c.gy)))
print('all: top_iou>=.5 %.3f fixed>=.5 %.3f'%((c.top_iou>=.5).mean(),(c.fixed_iou>=.5).mean()))
s=[(k,preds[k]) for k in d[(d.lot==2)&(d.ronda_h=='sano')].stem if k in preds]
print('sano confident damage preds, mean',np.mean([sum(b['class']==1 and b['confidence']>=.25 for b in x) for _,x in s]))
