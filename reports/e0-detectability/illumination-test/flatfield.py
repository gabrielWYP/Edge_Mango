"""Exploratory: illumination-corrected copies of the v4 dataset, for E0 only. Not for training."""
import argparse,csv,os,sys
from multiprocessing import Pool
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,'/home/gabo/MCP_Agent-worktrees/v4-final')
from src.annotation.mango_registration import rgb_mango_mask
cv2.setNumThreads(1)
SRC=Path('/home/gabo/MCP_Agent/data/cache/v4_captura_v3_final')
K=4
def field_poly(log,m,deg=2):
    h,w=log.shape
    yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
    ys,xs=np.nonzero(m)
    cx,cy=xs.mean(),ys.mean();s=max(xs.std(),ys.std())+1e-6
    x,y=(xx-cx)/s,(yy-cy)/s
    A=np.stack([x**i*y**j for i in range(deg+1) for j in range(deg+1-i)],-1)
    a=A[m][::5];b=log[m][::5]
    c,*_=np.linalg.lstsq(a,b,rcond=None)
    # robust second pass: the bruise and specular spots must not bend the surface
    r=b-a@c;keep=np.abs(r)<2.5*(np.median(np.abs(r))*1.4826+1e-6)
    c,*_=np.linalg.lstsq(a[keep],b[keep],rcond=None)
    return A@c
def field_gauss(log,m,frac=.25):
    ys,xs=np.nonzero(m);w=np.sqrt(m.sum()*4/np.pi)
    sig=frac*w
    num=cv2.GaussianBlur(log*m,(0,0),sig);den=cv2.GaussianBlur(m.astype(np.float32),(0,0),sig)
    return num/np.maximum(den,1e-3)
def gain(gray,m,variant):
    log=np.log(np.maximum(gray,1.0)).astype(np.float32)
    f=field_poly(log,m) if variant=='poly2' else field_gauss(log,m)
    g=np.exp(np.mean(f[m])-f).astype(np.float32)
    return np.clip(g,.25,4)
def correct(rgb,nir,variant):
    small=cv2.resize(rgb,None,fx=1/K,fy=1/K,interpolation=cv2.INTER_AREA)
    mask=rgb_mango_mask(small)>0
    if mask.sum()<100: return rgb,nir,None
    size=(rgb.shape[1],rgb.shape[0])
    lum=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY).astype(np.float32)
    ns=cv2.resize(nir,(small.shape[1],small.shape[0]),interpolation=cv2.INTER_AREA).astype(np.float32)
    mn=mask&(ns>0)
    grgb=gain(lum,mask,variant);gnir=gain(ns,mn if mn.sum()>100 else mask,variant)
    soft=cv2.GaussianBlur(mask.astype(np.float32),(0,0),6)
    grgb=1+(grgb-1)*soft;gnir=1+(gnir-1)*soft
    grgb=cv2.resize(grgb,size,interpolation=cv2.INTER_LINEAR);gnir=cv2.resize(gnir,size,interpolation=cv2.INTER_LINEAR)
    out=np.clip(rgb.astype(np.float32)*grgb[...,None],0,255).astype(np.uint8)
    outn=np.clip(nir.astype(np.float32)*gnir,0,255).astype(np.uint8)
    outn[nir==0]=0
    return out,outn,mask
def work(a):
    row,variant,out=a
    rgb=cv2.imread(str(SRC/'rgb'/f"{row['stem']}.jpg"));nir=cv2.imread(str(SRC/'nir'/f"mango_nir_{row['timestamp']}.png"),0)
    if rgb is None or nir is None: return row['stem'],'missing'
    r,n,m=correct(rgb,nir,variant)
    cv2.imwrite(str(out/'rgb'/f"{row['stem']}.jpg"),r,[cv2.IMWRITE_JPEG_QUALITY,95])
    cv2.imwrite(str(out/'nir'/f"mango_nir_{row['timestamp']}.png"),n,[cv2.IMWRITE_PNG_COMPRESSION,1])
    return row['stem'],'ok' if m is not None else 'nomask'
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--variant');ap.add_argument('--out',type=Path);ap.add_argument('--workers',type=int,default=12);a=ap.parse_args()
    (a.out/'rgb').mkdir(parents=True,exist_ok=True);(a.out/'nir').mkdir(exist_ok=True)
    for name in ('index.csv','exclusions.csv'):
        if (SRC/name).exists() and not (a.out/name).exists(): os.symlink(SRC/name,a.out/name)
    rows=list(csv.DictReader(open(SRC/'index.csv')))
    from collections import Counter
    with Pool(a.workers) as p: res=p.map(work,[(r,a.variant,a.out) for r in rows],chunksize=8)
    print(a.variant,Counter(s for _,s in res))
