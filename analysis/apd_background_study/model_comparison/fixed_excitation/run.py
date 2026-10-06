import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent))
from models import *
K=16

def fixed_predict(p,model,n,tm,di=-1):
 d=cone(p[:3],mechanical_phase(p,n,K,di));q=q_channels(d);s2=np.sum(d[:,:2]**2,axis=1);a0=np.exp(p[-1]);a=a0*np.sqrt(s2)
 f,pol,ang=p[19:22];bt=f*tm;bg=stokes_background(bt,pol,ang);cross=np.zeros_like(q)
 if model=='richard':cross=2*np.sqrt(bg[:,None]*q)*p[22:26,None]*a[None,:]
 if model=='field':
  share,eta,rel,common=p[22:26];b=np.sqrt(bt*share/2)*np.exp(1j*common)*np.array([np.cos(eta),np.sin(eta)*np.exp(1j*rel)])
  bj=ANALYZERS@b;bg=stokes_background(bt*(1-share),pol,ang)+np.abs(bj)**2
  cross=2*a0*GAMMA*np.real((ANALYZERS@d[:,:2].T)*(d[:,0]+1j*d[:,1])[None,:]*np.conj(bj[:,None]))
 signal=q*a[None,:]**2;pred=signal+cross+bg[:,None]
 return pred,dict(signal=signal,cross=cross,background=bg,a0=a0,theta=np.rad2deg(np.arccos(np.abs(d[:,2]))))

def loss(p,m,data,tm):return ((fixed_predict(p,m,data.shape[1],tm)[0]-data)/data.sum(0)).ravel()
def score(p,m,data,tm):
 pred,d=fixed_predict(p,m,data.shape[1],tm)
 return dict(xy_rms=float(np.sqrt(np.mean(np.sum((xy(pred)-xy(data))**2,axis=1)))),channel_rms=float(np.sqrt(np.mean(((pred-data)/data.sum(0))**2))),total_relative_rms=float(np.sqrt(np.mean(((pred.sum(0)-data.sum(0))/data.sum(0))**2))),bg_fraction=float(d['background'].sum()/tm),theta_range=[float(d['theta'].min()),float(d['theta'].max())],a0=float(d['a0']))
def fit(m,train,test,starts,cap=.95):
 lo,hi=bounds(m,K,cap);lo=np.r_[lo,-8];hi=np.r_[hi,4];tm=train.sum(0).min();runs=[]
 for p in starts:
  f=least_squares(loss,np.clip(p,lo+1e-8,hi-1e-8),args=(m,train,tm),bounds=(lo,hi),max_nfev=900,ftol=2e-9,xtol=2e-9,gtol=2e-9,x_scale='jac')
  runs.append(dict(p=f.x.tolist(),cost=float(np.mean(f.fun**2)),success=bool(f.success),nfev=int(f.nfev),train=score(f.x,m,train,tm),test=score(f.x,m,test,tm)))
 return dict(best=min(runs,key=lambda x:x['cost']),runs=runs,cap=cap)

if __name__=='__main__':
 rng=np.random.default_rng(5006);res={};checks=json.loads((R.parent/'interval10/checks.json').read_text())
 for key,path,fp,label in [('30s','fitted_predictions.npz','fits.json','30–31 s'),('10s','interval10/averages.npz','interval10/fits.json','10–11 s')]:
  z=np.load(R.parent/path);train=z['train'];test=z['test'];tm=train.sum(0).min();old=json.loads((R.parent/fp).read_text());res[key]={}
  for m in ['bg','richard','field']:
   bases=[np.array(old[m]['best']['p'])]
   if m=='field':bases += [np.array(e['p']) for e in checks[label]['field_checks']]
   starts=[]
   for b in bases:
    _,d=predict(b,m,train,tm,16,-1);dirs=cone(b[:3],mechanical_phase(b,128,16,-1));s2=np.sum(dirs[:,:2]**2,axis=1)
    starts.append(np.r_[b,.5*np.log(np.median(d['brightness']/s2))])
   for j in range(16):
    p=starts[j%len(bases)].copy();p[:3]+=rng.normal(0,.3,3);p[3]+=rng.normal(0,.3);p[4:19]+=rng.normal(0,.25,15);p[19]=rng.uniform(.05,.9);p[20]=rng.uniform(.1,.99);p[21]=rng.uniform(-np.pi,np.pi);p[-1]+=rng.normal(0,.25)
    if m=='richard':p[22:26]=rng.uniform(-.8,.8,4)
    if m=='field':p[22:26]=[rng.uniform(.05,.95),rng.uniform(.1,1.45),rng.uniform(-np.pi,np.pi),rng.uniform(-np.pi,np.pi)]
    starts.append(p)
   if m!='bg':
    b=np.array(res[key]['bg']['best']['p']);starts.append(np.r_[b[:-1],([0,0,0,0] if m=='richard' else [.001,.7,.2,.2]),b[-1]])
   res[key][m]=fit(m,train,test,starts);print(key,m,res[key][m]['best'],flush=True)
   (R/'fits.json').write_text(json.dumps(res,indent=2))
  # A wider background domain tests whether retaining the previous cap prejudices the new constraint.
  res[key]['field_cap_checks']={}
  for cap in [.8,2.,5.]:
   starts=[np.array(res[key]['field']['best']['p'])]
   for b in bases:
    starts.append(np.r_[b,starts[0][-1]])
   v=fit('field',train,test,starts,cap);res[key]['field_cap_checks'][str(cap)]=v;print(key,'cap',cap,v['best']['train'],v['best']['test'],flush=True)
  (R/'fits.json').write_text(json.dumps(res,indent=2))
