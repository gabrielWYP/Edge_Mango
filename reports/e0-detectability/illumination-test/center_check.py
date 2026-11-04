import csv,sys,cv2,numpy as np,pandas as pd,json
from multiprocessing import Pool
sys.path.insert(0,'/home/gabo/MCP_Agent-worktrees/v4-final')
from src.annotation.mango_registration import rgb_mango_mask
cv2.setNumThreads(1)
SRC='/home/gabo/MCP_Agent/data/cache/v4_captura_v3_final'
def w(r):
    x0,y0,x1,y1=[float(r[k]) for k in('mango_x0','mango_y0','mango_x1','mango_y1')]
    rgb=cv2.imread(f"{SRC}/rgb/{r['stem']}.jpg");nir=cv2.imread(f"{SRC}/nir/mango_nir_{r['timestamp']}.png",0)
    if rgb is None or nir is None:return None
    A=rgb[int(y0):int(y1),int(x0):int(x1)];N=nir[int(y0):int(y1),int(x0):int(x1)]
    A=cv2.resize(A,(128,128),interpolation=cv2.INTER_AREA);N=cv2.resize(N,(128,128),interpolation=cv2.INTER_AREA)
    m=rgb_mango_mask(A)>0
    yy,xx=np.mgrid[0:128,0:128];rr=np.hypot(xx-63.5,yy-63.5)/64
    d=m&(rr<.25);ring=m&(rr>=.35)&(rr<.6)
    if d.sum()<50 or ring.sum()<50:return None
    L=cv2.cvtColor(A,cv2.COLOR_BGR2GRAY).astype(float)
    return dict(stem=r['stem'],lot=int(r['lot']),zona=r['zona'],h=r['ronda_h'],rgb=L[d].mean()/L[ring].mean(),nir=N[d].mean()/max(N[ring].mean(),1))
rows=list(csv.DictReader(open(f'{SRC}/index.csv')));rows=[r for r in rows if r['ronda_h'] in('sano','12')]
with Pool(8) as p:out=[x for x in p.map(w,rows,chunksize=8) if x]
d=pd.DataFrame(out);d['t']=np.where(d.h=='sano','sano','+12h')
print('center disk / ring brightness, <1 means darker center')
print(d.groupby(['t','zona'])[['rgb','nir']].median().round(3).to_string())
print('frac darker center',d.groupby('t')[['rgb','nir']].agg(lambda s:(s<1).mean()).round(3).to_string())
idx=pd.read_csv(f'{SRC}/index.csv').set_index('stem')
xs=[];ys=[]
for l in open('/home/gabo/MCP_Agent-worktrees/v4-final/reports/v4/nir_clahe/maestro_rgbnir/fold0_seed42/val/predictions.jsonl'):
    j=json.loads(l);r=idx.loc[j['stem']]
    if r.ronda_h!='sano':continue
    for b in j['boxes']:
        if b['class']==1 and b['confidence']>=.25:
            q=b['xyxy'];xs.append((((q[0]+q[2])/2*3264)-r.mango_x0)/(r.mango_x1-r.mango_x0));ys.append((((q[1]+q[3])/2*2448)-r.mango_y0)/(r.mango_y1-r.mango_y0))
print('val sano damage FP centers: n %d x %.3f sd %.3f y %.3f sd %.3f; within 0.2 of center %.3f'%(len(xs),np.mean(xs),np.std(xs),np.mean(ys),np.std(ys),np.mean(np.hypot(np.array(xs)-.5,np.array(ys)-.5)<.2)))
