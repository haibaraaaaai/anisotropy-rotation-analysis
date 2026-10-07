"""Correct final-reference arc correspondence, preserving original four-channel bins.
No flexible phase warp. Held-out data use training-derived fractions without realignment.
"""
from pathlib import Path
import importlib.util,sys,json
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent))
from models import cone,q_channels,xy
spec=importlib.util.spec_from_file_location('arc_previous',R.parent/'arc_matching/run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
Z=np.load(R.parent/'arc_matching/inputs.npz');PREV=json.load(open(R.parent/'arc_matching/fits.json'));COMPLEX=json.load(open(R.parent/'inverse_circle/fits.json'));KEYS=['30s','10s'];MODELS=['none','square','richard4','complex']
def fractions(dat,subdivision=32):
 """Arc along closed ordered piecewise-linear CHANNEL interpolant, then ratios.
 Returns exact positions of original bins on that interpolated anisotropy track.
 """
 n=dat.shape[1];tt=np.arange(n*subdivision+1)/subdivision;xx=np.arange(n+1)
 dense=np.array([np.interp(tt,xx,np.r_[row,row[0]]) for row in dat]);pp=xy(dense)
 s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pp,axis=0),axis=1))]
 assert s[-1]>0
 return (s/s[-1])[::subdivision][:-1]

def forward(p,model,tm,di,u,density=4096,parts=False):
 chi=np.linspace(0,2*np.pi,density+1);d=cone(p[:3],chi);base=np.sum(d[:,:2]**2,1)[None,:]*q_channels(d);s=(p[3] if model=='richard_product' else np.exp(2*p[3]))*base
 bg=np.zeros(4);c=np.zeros(4);h=p[5:]
 if model in ['square','complex']:
  bg=h[0]*tm/4*np.array([1+h[1],1-h[1],1+h[2],1-h[2]])
  if model=='complex':c=2*np.sqrt(bg)*h[3:]
 if model=='richard4':c=h
 cross=c[:,None]*np.sqrt(s)
 if model=='richard_product':
  base=np.sum(d[:,:2]**2,1)[None,:]*q_channels(d);s=p[3]*base;cross=h[:,None]*np.sqrt(base)
 y=s+bg[:,None]+cross;mn=y.min()
 zz=np.column_stack([(y[0]-y[1])/np.maximum(y[0]+y[1],1e-9),(y[2]-y[3])/np.maximum(y[2]+y[3],1e-9)])
 ar=np.r_[0,np.cumsum(np.linalg.norm(np.diff(zz,axis=0),axis=1))];ar/=max(ar[-1],1e-20)
 target=(p[4]+di*np.asarray(u))%1
 pred=np.array([np.interp(target,ar,row) for row in y]);ds=cone(p[:3],np.interp(target,ar,chi))
 if not parts:return pred
 sig=np.array([np.interp(target,ar,row) for row in s]);inter=np.array([np.interp(target,ar,row) for row in cross])
 return pred,sig,inter,bg,ds,mn

def residual(p,model,tm,di,u,dat,density=2048):
 y,s,c,b,d,mn=forward(p,model,tm,di,u,density,True)
 return np.r_[((y-dat)/dat.sum(0)).ravel(),100*max(0,1e-8-mn)]
def bounds(model):
 if model!='complex':return old.limits(model,'arc')
 lo,hi=old.limits('square','arc');return np.r_[lo,[-1]*4],np.r_[hi,[1]*4]
def metrics(p,model,tm,di,u,dat,density=8192):
 y,s,c,b,d,mn=forward(p,model,tm,di,u,density,True);th=np.rad2deg(np.arccos(abs(d[:,2])))
 return dict(channel_rms=float(np.sqrt(np.mean(((y-dat)/dat.sum(0))**2))),xy_rms=float(np.sqrt(np.mean(np.sum((xy(y)-xy(dat))**2,1)))),total_rms=float(np.sqrt(np.mean(((y.sum(0)-dat.sum(0))/dat.sum(0))**2))),theta_range=[float(th.min()),float(th.max())],min_pred=float(mn),bg_total=float(b.sum()))

def main():
 results=json.load(open(R/'fits.json')) if (R/'fits.json').exists() else {};rng=np.random.default_rng(61740);coords={}
 for key in KEYS:
  tr=Z[key+'_train']/Z['scale'];te=Z[key+'_test']/Z['scale'];tm=tr.sum(0).min();u=fractions(tr);coords[key]=u.tolist();results.setdefault(key,{})
  for model in MODELS:
   if model in results[key]:continue
   starts=[]
   if model=='complex':source=np.array(COMPLEX[key]['arc_complex']['best']['p'])
   else:source=np.array(PREV[key]['arc_'+model]['best']['p'])
   for di in [-1,1]:
    for off in [0,.25,.5,.75]:p=source.copy();p[4]=off;starts.append((p,di))
   # Geometry and brightness from previous flexible best, with same correction parameters.
   for oldmodel in ([model] if model!='complex' else ['square','richard4']):
    pp=np.array(PREV[key]['flex_'+oldmodel]['best']['p']);h=source[5:] if model=='complex' else pp[20:]
    for off in [0,.25,.5,.75]:starts.append((np.r_[pp[:3],pp[19],off,h],-1))
   if model!='none':
    b=results[key]['none']['best'];hh=[0,0,0] if model=='square' else ([0]*4 if model=='richard4' else [0]*7);starts.append((np.r_[b['p'],hh],b['direction']))
   if model=='complex':
    b=results[key]['square']['best'];starts.append((np.r_[b['p'],[0]*4],b['direction']))
   for j in range(6):
    p,di=starts[j];p=p.copy();p[:3]+=rng.normal(0,.25,3);p[3]+=rng.normal(0,.2);starts.append((p,di))
   lo,hi=bounds(model);runs=[]
   for j,(p,di) in enumerate(starts):
    p=np.clip(p,lo+1e-9,hi-1e-9)
    # Retain feasible seeds: local optimization must not invalidate nested-model bounds.
    init=residual(p,model,tm,di,u,tr);seedcost=float(np.mean(init[:512]**2))
    f=least_squares(residual,p,args=(model,tm,di,u,tr),bounds=(lo,hi),max_nfev=700,ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale='jac')
    runs.append(dict(p=f.x.tolist(),direction=di,cost=float(np.mean(f.fun[:512]**2)),success=bool(f.success),nfev=f.nfev,train=metrics(f.x,model,tm,di,u,tr),test=metrics(f.x,model,tm,di,u,te)))
    if init[-1]==0 and seedcost<runs[-1]['cost']:runs.append(dict(p=p.tolist(),direction=di,cost=seedcost,success=False,nfev=0,train=metrics(p,model,tm,di,u,tr),test=metrics(p,model,tm,di,u,te),note='feasible seed retained'))
   best=min([b for b in runs if b['train']['min_pred']>0],key=lambda a:a['cost'])
   # Fine-grid local refinement resolves residual discretization dependence.
   pp=np.array(best['p']);di=best['direction'];f=least_squares(residual,pp,args=(model,tm,di,u,tr,8192),bounds=(lo,hi),max_nfev=400,ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale='jac')
   refined=dict(p=f.x.tolist(),direction=di,cost=float(np.mean(f.fun[:512]**2)),success=bool(f.success),nfev=f.nfev,train=metrics(f.x,model,tm,di,u,tr),test=metrics(f.x,model,tm,di,u,te))
   y0=forward(f.x,model,tm,di,u,8192);y1=forward(f.x,model,tm,di,u,16384)
   refined['resolution_max_delta_over_Tmax']=float(np.max(abs(y1-y0))/tr.sum(0).max())
   results[key][model]=dict(best=refined,coarse_best=best,runs=runs)
   (R/'fits.json').write_text(json.dumps(results,indent=2));print(key,model,refined['success'],refined['test'],flush=True)
 (R/'reference_fractions.json').write_text(json.dumps(coords,indent=2))
 # The reference-coordinate interpolation itself must be converged.
 checks={}
 for key in KEYS:
  tr=Z[key+'_train']/Z['scale'];u32=fractions(tr,32);u128=fractions(tr,128);checks[key]=dict(reference_fraction_max_change=float(np.max(abs(u128-u32))))
 # Known synthetic curve: its nonuniform samples must match by their actual arc coordinates.
 model='complex';tm=.3;p=np.array([.65,.3,.32,-.5,.27,.2,.2,-.3,.2,.1,-.1,.3]);uf=(np.arange(128)/128)**1.4;dat=forward(p,model,tm,-1,uf,32768)
 fref=fractions(dat,128);correct=forward(p,model,tm,-1,fref,32768);wrong=forward(p,model,tm,-1,np.arange(128)/128,32768)
 checks['synthetic_nonuniform_reference']=dict(corrected_channel_rms=float(np.sqrt(np.mean(((correct-dat)/dat.sum(0))**2))),uniform_index_channel_rms=float(np.sqrt(np.mean(((wrong-dat)/dat.sum(0))**2))),note='Residual corrected error is finite polygon/interpolant approximation, not exact continuous-curve identity.')
 (R/'checks.json').write_text(json.dumps(checks,indent=2));print('checks',checks,flush=True)
if __name__=='__main__':main()
