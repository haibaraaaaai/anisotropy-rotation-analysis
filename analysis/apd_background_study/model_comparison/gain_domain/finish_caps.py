import caps as c
import json,numpy as np
from scipy.optimize import least_squares
v=json.load(open(c.R/'fits.json'))
for key,a in v.items():
 b=a['best']
 if b['success']:continue
 c.setup(a['m'],a['cap']);lo,hi=c.g.limits();di=b['directions'];q=least_squares(c.st.residual,b['p'],args=(di,8192),bounds=(lo,hi),max_nfev=500,ftol=1e-8,xtol=1e-8,gtol=1e-8,x_scale='jac');a['prior_refined']=b;a['best']=dict(p=q.x.tolist(),directions=di,cost=float(np.sum(q.fun**2)),success=bool(q.success),nfev=q.nfev,scores=c.f.score(q.x,di));(c.R/'fits.json').write_text(json.dumps(v,indent=2));print(key,a['best']['success'],a['best']['cost'],flush=True)
