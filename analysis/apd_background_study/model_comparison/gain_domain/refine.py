import run as f
import numpy as np,json
from scipy.optimize import least_squares
v=json.load(open(f.R/'fits.json'));lo,hi=f.f.limits();low=v['0']['runs'][1]
for w in [.01,.1,1,10]:
 candidates=[]
 source=v[str(w)]['best'] if str(w) in v else v['1']['best']
 for b in [source,low]:
  di=b['directions'];q=least_squares(f.residual,np.clip(b['p'],lo+1e-9,hi-1e-9),args=(di,w,8192),bounds=(lo,hi),max_nfev=100,ftol=2e-8,xtol=2e-8,gtol=2e-8,x_scale='jac');rec=dict(p=q.x.tolist(),directions=di,cost=float(np.sum(q.fun**2)),success=bool(q.success),nfev=q.nfev,scores=f.score(q.x,di));candidates.append(rec)
  print(w,rec['success'],rec['cost'],[(k,rec['scores'][k]['test']['invalid']) for k in f.f.KEYS],flush=True)
 v[str(w)]=dict(best=min(candidates,key=lambda a:a['cost']),runs=v.get(str(w),{}).get('runs',[])+candidates);(f.R/'fits.json').write_text(json.dumps(v,indent=2))
