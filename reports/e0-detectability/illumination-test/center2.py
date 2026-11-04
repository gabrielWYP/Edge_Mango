import sys,numpy as np,pandas as pd,json,runpy
sys.argv=['x'];src=open(sys.path[0]+'/center_check.py').read().split("rows=list(csv")[0];exec(src)
all_=list(csv.DictReader(open(f'{SRC}/index.csv')))
if __name__=='__main__':
    with Pool(12) as p:out=[x for x in p.map(w,all_,chunksize=8) if x]
    d=pd.DataFrame(out);m=pd.read_csv(f'{SRC}/index.csv').set_index('stem')
    d['mango']=d.stem.map(m.mango_id);d['cara']=d.stem.map(m.cara)
    b=d[d.h=='sano'].set_index(['lot','mango','cara'])[['rgb','nir']]
    x=d[d.h!='sano'].join(b,on=['lot','mango','cara'],rsuffix='_s').dropna();x['h']=x.h.astype(int)
    x['drgb']=x.rgb-x.rgb_s;x['dnir']=x.nir-x.nir_s;x['g']=np.where(x.zona=='centro','centro','otras')
    rng=np.random.default_rng(0)
    for win,n in(((0,6),'0-6 h'),((8,12),'8-12 h')):
        y=x[x.h.between(*win)]
        print(n,y.groupby('zona')[['drgb','dnir']].mean().round(4).to_string())
        fa=y.groupby(['lot','mango','cara','g'])[['drgb','dnir']].mean().reset_index()
        for c in('drgb','dnir'):
            A=fa[fa.g=='centro'];B=fa[fa.g=='otras'];g=[]
            for _ in range(2000):
                a=[];bb=[]
                for l in fa.lot.unique():
                    ms=fa[fa.lot==l].mango.unique();pk=rng.choice(ms,len(ms))
                    t=pd.concat([fa[(fa.lot==l)&(fa.mango==k)] for k in pk]) if False else None
                g.append(0);break
            ms=fa[['lot','mango']].drop_duplicates().to_numpy();by={tuple(k):v for k,v in fa.groupby(['lot','mango'])}
            for _ in range(1000):
                pk=ms[rng.integers(0,len(ms),len(ms))];t=pd.concat([by[tuple(k)] for k in pk]);g.append(t[t.g=='centro'][c].mean()-t[t.g=='otras'][c].mean())
            print(c,'centro - otras %+.4f [%+.4f,%+.4f]; otras mean change %+.4f'%(A[c].mean()-B[c].mean(),*np.percentile(g[1:],[2.5,97.5]),B[c].mean()))
    # fixed box at the teacher's mean top prediction
    W,H=3264,2448;p=m[m.ronda_h!='sano'].copy();v=p[p.lot==2].copy();tr=p[p.lot.isin([3,4,5])]
    rw=((tr.damage_x1-tr.damage_x0)/(tr.mango_x1-tr.mango_x0)).median();rh=((tr.damage_y1-tr.damage_y0)/(tr.mango_y1-tr.mango_y0)).median()
    mx,my=.54,.43
    def iou(a,b):
        ix=max(0,min(a[2],b[2])-max(a[0],b[0]));iy=max(0,min(a[3],b[3])-max(a[1],b[1]));i=ix*iy
        return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-12)
    rr=[]
    for _,r in v.iterrows():
        w_=r.mango_x1-r.mango_x0;h_=r.mango_y1-r.mango_y0
        f=[r.mango_x0+w_*(mx-rw/2),r.mango_y0+h_*(my-rh/2),r.mango_x0+w_*(mx+rw/2),r.mango_y0+h_*(my+rh/2)]
        g=[r.damage_x0,r.damage_y0,r.damage_x1,r.damage_y1];rr.append((r.zona,iou(f,g)))
    q=pd.DataFrame(rr,columns=['zona','iou'])
    print('fixed central box, val lot 2, IoU>=.5 and mean IoU');print(q.groupby('zona').iou.agg([lambda s:(s>=.5).mean(),'mean','size']).round(3).to_string());print('all',round((q.iou>=.5).mean(),3))
