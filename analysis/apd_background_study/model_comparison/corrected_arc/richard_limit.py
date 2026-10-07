"""Exact Richard reparameterization alpha=a0^2, K=a0*C; alpha=0 is closure only."""
import run as r
import numpy as np,json
from scipy.optimize import least_squares
v=json.load(open(r.R/'fits.json'));out={}
for key in r.KEYS:
 tr=r.Z[key+'_train']/r.Z['scale'];te=r.Z[key+'_test']/r.Z['scale'];tm=tr.sum(0).min();u=r.fractions(tr);p=np.array(v[key]['richard4']['best']['p']);p[5:]*=np.exp(p[3]);p[3]=np.exp(2*p[3]);starts=[]
 for di in [-1,1]:
  for off in [0,.25,.5,.75]:q=p.copy();q[4]=off;starts.append((q,di))
 starts.append((p,v[key]['richard4']['best']['direction']))
 lo,hi=r.bounds('richard4');lo[3]=0;hi[3]=np.exp(8);runs=[]
 for p,di in starts:
  f=least_squares(r.residual,p,args=('richard_product',tm,di,u,tr,4096),bounds=(lo,hi),max_nfev=700,ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale='jac')
  runs.append(dict(p=f.x.tolist(),direction=di,success=bool(f.success),cost=float(np.mean(f.fun[:512]**2)),train=r.metrics(f.x,'richard_product',tm,di,u,tr),test=r.metrics(f.x,'richard_product',tm,di,u,te)))
 best=min(runs,key=lambda x:x['cost']);out[key]=dict(best=best,runs=runs)
 # Fixed-alpha approach to limit, preserving geometry and K, shows finite-model approach.
 p=np.array(best['p']);approach=[]
 for alpha in [1e-2,1e-4,1e-6,1e-8]:
  pp=p.copy();pp[3]=alpha;approach.append(dict(alpha=alpha,C=(pp[5:]/np.sqrt(alpha)).tolist(),test=r.metrics(pp,'richard_product',tm,best['direction'],u,te)))
 out[key]['fixed_geometry_approach']=approach
 (r.R/'richard_product_fits.json').write_text(json.dumps(out,indent=2));print(key,best,flush=True)
