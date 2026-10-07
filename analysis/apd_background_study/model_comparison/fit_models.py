from pathlib import Path
import json,time
import numpy as np
from scipy.optimize import least_squares
from models import *
R=Path(__file__).resolve().parent
z=np.load(R.parent/'interval_study/averaged_channels.npz')
allc=np.einsum('ij,kjl->kil',z['calibration'],z['channels_per_cycle'])
train=allc[::2].mean(axis=0);test=allc[1::2].mean(axis=0)
TMIN=float(train.sum(axis=0).min());K=16
legacy=json.loads((R/'legacy_control.json').read_text())


def initial_phase(geometry,bg):
 data=xy(train-bg[:,None]);n=len(data)
 ds=np.r_[0,np.cumsum(np.linalg.norm(np.diff(np.vstack([data,data[0]]),axis=0),axis=1))];ds/=ds[-1]
 chi=np.arange(1440)*2*np.pi/1440;mm=xy(q_channels(cone(geometry,chi)))
 starts=np.argsort(np.sum((mm-data[0])**2,axis=1))[:12]
 best=None
 for direction in [1,-1]:
  for start in starts:
   ids=(start+direction*np.arange(len(mm)+1))%len(mm);pp=mm[ids]
   ss=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pp,axis=0),axis=1))];ss/=ss[-1]
   frac=np.interp(ds[:-1],ss,np.linspace(0,1,len(mm)+1))
   qq=np.column_stack([np.interp(ds[:-1],ss,pp[:,j]) for j in range(2)])
   err=np.mean((qq-data)**2)
   if best is None or err<best[0]:best=(err,chi[start],frac,direction)
 err,start,frac,direction=best
 knotfrac=np.interp(np.linspace(0,1,K+1),np.arange(n+1)/n,np.r_[frac,1.])
 inc=np.maximum(np.diff(knotfrac),1e-5)
 logs=np.clip(np.log(inc[:-1]/inc[-1]),-3.9,3.9)
 return start,logs,direction


def seed_from_legacy():
 l=np.array(legacy['averaged_physical']['best']['p']);geom=np.deg2rad(l[:3])
 dc,fa,fb=l[3:];pol=min(.999,np.hypot(fa,fb));angle=np.arctan2(fb,fa)
 bg=stokes_background(2*dc,pol,angle);start,logs,di=initial_phase(geom,bg)
 return np.r_[geom,start,logs,2*dc/TMIN,pol,angle],di

def stats(p,model,di,knots=K,data=train):
 pred,d=predict(p,model,data,TMIN,knots,di)
 return dict(xy_rms=float(np.sqrt(np.mean(np.sum((xy(pred)-xy(data))**2,axis=1)))),
  normalized_channel_rms=float(np.sqrt(np.mean(((pred-data)/data.sum(axis=0))**2))),
  pair_imbalance_pred_rms=float(np.sqrt(np.mean(((pred[0]+pred[1]-pred[2]-pred[3])/pred.sum(axis=0))**2))),
  background_total_fraction_of_minimum=float(d['total_bg']/TMIN),
  folded_theta_range_deg=[float(d['theta'].min()),float(d['theta'].max())])

def run(model,starts,cap=.95,knots=K,max_nfev=650):
 lo,hi=bounds(model,knots,cap);runs=[]
 for num,(start,di) in enumerate(starts):
  start=np.clip(start,lo+1e-7,hi-1e-7);beg=time.time()
  f=least_squares(residual,start,args=(model,train,TMIN,knots,di),bounds=(lo,hi),
        max_nfev=max_nfev,ftol=2e-9,xtol=2e-9,gtol=2e-9,x_scale='jac')
  entry=dict(p=f.x.tolist(),direction=di,cost=float(np.mean(f.fun**2)),success=bool(f.success),
   message=str(f.message),nfev=f.nfev,seconds=time.time()-beg,train=stats(f.x,model,di,knots),
   test=stats(f.x,model,di,knots,test),jac_singular_values=np.linalg.svd(f.jac,compute_uv=False).tolist())
  runs.append(entry)
  print(model,num,entry['cost'],entry['train']['xy_rms'],entry['test']['xy_rms'],entry['nfev'],entry['seconds'],flush=True)
 return dict(best=min(runs,key=lambda a:a['cost']),runs=runs,cap=cap,knots=knots)

if __name__=='__main__':
 rng=np.random.default_rng(23);p0,di=seed_from_legacy();starts=[(p0,di)]
 for i in range(7):
  p=p0.copy();p[:3]+=rng.normal(0,.20,3);p[3]+=rng.normal(0,.15)
  p[4:K+3]+=rng.normal(0,.20,K-1);p[K+3]=rng.uniform(.05,.7)
  p[K+4]=rng.uniform(.3,.95);p[K+5]=rng.uniform(-np.pi,np.pi)
  starts.append((p,di))
 results={'bg':run('bg',starts)}
 (R/'fits.json').write_text(json.dumps(results,indent=2)+'\n')
 pb=np.array(results['bg']['best']['p']);di=results['bg']['best']['direction']
 for model in ['richard','field']:
  starts=[]
  for i in range(10):
   pp=pb.copy()
   if i:pp[:3]+=rng.normal(0,.12,3);pp[4:K+3]+=rng.normal(0,.1,K-1)
   if model=='richard':extra=np.zeros(4) if i==0 else rng.uniform(-.7,.7,4)
   else:extra=[.005 if i==0 else rng.uniform(.1,.8),rng.uniform(.15,1.4),rng.uniform(-np.pi,np.pi),rng.uniform(-np.pi,np.pi)]
   starts.append((np.r_[pp,extra],di))
  results[model]=run(model,starts)
  (R/'fits.json').write_text(json.dumps(results,indent=2)+'\n')
 np.savez_compressed(R/'fitted_predictions.npz',train=train,test=test,all_cycles=allc,
  **{model:predict(np.array(v['best']['p']),model,train,TMIN,K,v['best']['direction'])[0] for model,v in results.items()})
