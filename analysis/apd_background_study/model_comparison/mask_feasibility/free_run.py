import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent));from models import *
def signal(p,n=128):
 d=cone(p[:3],mechanical_phase(p,n,16,-1));return np.exp(2*p[19])*np.sum(d[:,:2]**2,axis=1)[None,:]*q_channels(d)
def envelopes(p,data):
 s=signal(p);return (np.sqrt(data)-np.sqrt(s))**2,(np.sqrt(data)+np.sqrt(s))**2
# This relaxation permits independent channel overlap/phase at EVERY point.
# A feasible solution is an intensity-budget witness, not a stationary common field.
if __name__=='__main__':
 old=json.loads((R.parent/'bg_cap_matrix/results.json').read_text());rng=np.random.default_rng(482);res={}
 for key,path in [('30s','fitted_predictions.npz'),('10s','interval10/averages.npz')]:
  z=np.load(R.parent/path);scale=z['train'].sum(0).max();data=z['train']/scale;test=z['test']/scale;runs=[]
  for j in range(8):
   base=old[key]['plus_cap10' if j%2==0 else 'plus_wide']['best']['p'];p=np.r_[base[:19],base[-1]-.5*np.log(scale)]
   if j>1:p[:3]+=rng.normal(0,.15,3);p[4:19]+=rng.normal(0,.15,15)
   lo=np.r_[[.001,-np.pi,.001,-4*np.pi],[-4.]*15,-8,[1e-9]*4];hi=np.r_[[np.pi/2-.001,np.pi,np.pi/2-.001,4*np.pi],[4.]*15,4,[5.]*4]
   p=np.clip(p,lo[:20]+1e-7,hi[:20]-1e-7);l,u=envelopes(p,data);p=np.r_[p,l.max(1)+1e-5]
   def con(p):
    l,u=envelopes(p,data);return np.r_[(p[20:,None]-l).ravel(),(u-p[20:,None]).ravel()]
   f=minimize(lambda p:p[20:].sum(),p,method='SLSQP',bounds=list(zip(lo,hi)),constraints=[{'type':'ineq','fun':con}],options={'maxiter':500,'ftol':1e-10})
   pp=f.x;s=signal(pp);b=pp[20:];cross=data-s-b[:,None];tau=cross/(2*np.sqrt(s*b[:,None]));lt,ut=envelopes(pp,test)
   runs.append(dict(p=pp.tolist(),success=bool(f.success),message=str(f.message),bg_fraction_max=float(b.sum()),min_constraint=float(con(pp).min()),max_abs_tau=float(np.max(np.abs(tau))),test_minimum_budget_same_cone=float(lt.max(1).sum()),theta_range=np.rad2deg(np.arccos(np.abs(cone(pp[:3],mechanical_phase(pp,128,16,-1))[:,2]))).tolist()))
   print(key,j,runs[-1]['bg_fraction_max'],runs[-1]['success'],runs[-1]['min_constraint'],flush=True)
  valid=[v for v in runs if v['min_constraint']>=-1e-7];best=min(valid,key=lambda a:a['bg_fraction_max']);res[key]=dict(best=best,runs=runs,scale=float(scale))
  (R/'free_fits.json').write_text(json.dumps(res,indent=2))
