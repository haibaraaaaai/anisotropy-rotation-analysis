"""Matched intensity fits on weighted, uniform-arc and provisional-phase references.
Phase reference is diagnostic only if the guide monotonicity check fails.
"""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('corr_sampling',R.parent/'corrected_arc/run.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
Z=np.load(R/'references.npz');BASE=json.load(open(R.parent/'corrected_arc/fits.json'));RICH=json.load(open(R.parent/'corrected_arc/richard_product_fits.json'))
MODELS=['none','square','richard_product','complex'];METHODS=['weighted','uniform','phase'];KEYS=r.KEYS

def objective(p,model,tm,di,u,dat,weights,density=2048):
 res=r.residual(p,model,tm,di,u,dat,density);return np.r_[res[:-1].reshape(4,-1)*np.sqrt(weights)[None,:],].ravel().tolist()+[res[-1]]
def score(p,model,tm,di,u,dat,weights):
 mm=r.metrics(p,model,tm,di,u,dat,8192);y=r.forward(p,model,tm,di,u,8192);e=((y-dat)/dat.sum(0))**2
 du=np.diff(np.r_[u,1]);wa=(du+np.roll(du,1))/2
 mm.update(objective_channel_rms=float(np.sqrt(np.sum(e*weights[None,:])/(4*weights.sum()))),arc_weighted_channel_rms=float(np.sqrt(np.sum(e*wa[None,:])/4)),limiting=bool(model=='richard_product' and p[3]<1e-8))
 return mm

def main():
 out=json.load(open(R/'fits.json')) if (R/'fits.json').exists() else {};rng=np.random.default_rng(61807)
 for key in KEYS:
  out.setdefault(key,{});tm=r.Z[key+'_train'].sum(0).min()/Z['scale']
  for method in METHODS:
   out[key].setdefault(method,{});tr=Z[key+'_'+method+'_train']/Z['scale'];te=Z[key+'_'+method+'_test']/Z['scale'];u=r.fractions(tr,64);weights=Z[key+'_weighted_weights'] if method=='weighted' else np.ones(128)
   for model in MODELS:
    if model in out[key][method]:continue
    ref=RICH[key]['best'] if model=='richard_product' else BASE[key][model]['best'];p=np.array(ref['p']);di=ref['direction'];starts=[(p,di)]
    # Four offsets, both directions; +nested starts and guide-specific nearby geometries.
    for direction in [-1,1]:
     for offset in [0,.25,.5,.75]:q=p.copy();q[4]=offset;starts.append((q,direction))
    if model!='none':
     b=out[key][method]['none']['best'];q=np.array(b['p']);q[3]=np.exp(2*q[3]) if model=='richard_product' else q[3]
     starts.append((np.r_[q,[0]*({'square':3,'richard_product':4,'complex':7}[model])],b['direction']))
    if model=='complex':
     b=out[key][method]['square']['best'];starts.append((np.r_[b['p'],[0]*4],b['direction']))
    if model=='richard_product':lo,hi=r.bounds('richard4');lo[3]=0;hi[3]=np.exp(8)
    else:lo,hi=r.bounds(model)
    runs=[]
    for pp,dd in starts:
     pp=np.clip(pp,lo+1e-12,hi-1e-12)
     f=least_squares(objective,pp,args=(model,tm,dd,u,tr,weights),bounds=(lo,hi),max_nfev=500,ftol=1e-9,xtol=1e-9,gtol=1e-9,x_scale='jac')
     runs.append(dict(p=f.x.tolist(),direction=dd,cost=float(np.mean(f.fun[:-1]**2)),success=bool(f.success),nfev=f.nfev))
    best=min(runs,key=lambda x:x['cost']);pp=np.array(best['p']);dd=best['direction']
    f=least_squares(objective,pp,args=(model,tm,dd,u,tr,weights,8192),bounds=(lo,hi),max_nfev=700,ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale='jac')
    best=dict(p=f.x.tolist(),direction=dd,cost=float(np.mean(f.fun[:-1]**2)),success=bool(f.success),nfev=f.nfev,train=score(f.x,model,tm,dd,u,tr,weights),test=score(f.x,model,tm,dd,u,te,weights))
    out[key][method][model]=dict(best=best,runs=runs);(R/'fits.json').write_text(json.dumps(out,indent=2));print(key,method,model,best['success'],best['test']['objective_channel_rms'],best['test']['theta_range'],best['test']['limiting'],flush=True)
 (R/'fits.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
