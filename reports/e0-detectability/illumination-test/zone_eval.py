import sys,numpy as np,pandas as pd
sys.path.insert(0,'/home/gabo/MCP_Agent-worktrees/v4-final')
from src.evaluation import detectability as det
base='/home/gabo/MCP_Agent-worktrees/v4-final/reports/e0-detectability/boxes-final'
V={'orig':base,'poly2':'/var/tmp/v4_flat/e0_poly2','gauss25':'/var/tmp/v4_flat/e0_gauss25'}
MOD={'RGB':[f'{c}_{k}' for c in 'RGB' for k in('mean','tex')],'NIR':['NIR_mean','NIR_tex'],'RGB+NIR':[f'{c}_{k}' for c in('R','G','B','NIR') for k in('mean','tex')]}
def scores(variant,neg=False):
    p=pd.read_csv(f'{V[variant]}/patches.csv');f=pd.read_csv(f'{V[variant]}/faces.csv')
    z=f.drop_duplicates(['lot','mango','face'])[['lot','mango','face','zone']]
    p=p.merge(z,on=['lot','mango','face'])
    regs=sorted(p.region.unique())
    a,b=('control',[r for r in regs if r not in('impact','control')][0]) if neg else ('impact','control')
    p=p[p.region.isin([a,b])].copy();p['y']=(p.region==a).astype(int)
    out=[]
    for h,g in p.groupby('hours'):
        for m,cols in MOD.items():
            q=g.dropna(subset=cols)
            s=det.grouped_oof_scores(q[cols].to_numpy(float),q.y.to_numpy(float),q.lot.to_numpy())
            out.append(q[['lot','mango','face','zone','hours','y']].assign(mod=m,s=s))
    return pd.concat(out)
def au(d): return det.auroc(d.s[d.y==1].to_numpy(),d.s[d.y==0].to_numpy())
ZS=['centro','izquierda','derecha','inferior','superior']
res={v:scores(v) for v in V};neg={v:scores(v,True) for v in V}
for win,name in (((0,6),'latent 0-6'),((8,12),'late 8-12')):
    print('==',name)
    rows=[]
    for m in MOD:
        for v in V:
            d=res[v];d=d[(d['mod']==m)&d.hours.between(*win)]
            n=neg[v];n=n[(n['mod']==m)&n.hours.between(*win)]
            r={'mod':m,'var':v,'all':au(d),'neg':au(n)}
            for z in ZS:r[z]=au(d[d.zone==z])
            rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
rng=np.random.default_rng(0)
print('== bootstrap gain vs orig, latent, mangos within lot (2000 reps)')
for v in ('poly2','gauss25'):
  for m in ('RGB','NIR','RGB+NIR'):
    a=res['orig'];b=res[v]
    a=a[(a['mod']==m)&a.hours.between(0,6)];b=b[(b['mod']==m)&b.hours.between(0,6)]
    for z in ZS+['all']:
        A=a if z=='all' else a[a.zone==z];B=b if z=='all' else b[b.zone==z]
        ids=A[['lot','mango']].drop_duplicates()
        A_={k:g for k,g in A.groupby(['lot','mango'])};B_={k:g for k,g in B.groupby(['lot','mango'])}
        keys=[k for k in A_ if k in B_];lots={}
        for k in keys:lots.setdefault(k[0],[]).append(k)
        gs=[]
        for _ in range(300):
            kk=[lots[l][i] for l in lots for i in rng.integers(0,len(lots[l]),len(lots[l]))]
            xa=pd.concat([A_[k] for k in kk]);xb=pd.concat([B_[k] for k in kk]);gs.append(au(xb)-au(xa))
        print(v,m,z,'gain %+.3f [%+.3f,%+.3f]'%(au(B)-au(A),*np.percentile(gs,[2.5,97.5])))
