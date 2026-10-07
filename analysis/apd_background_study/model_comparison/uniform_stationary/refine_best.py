"""Check the three stronger historical (-1,-1) field starts and refine selected minimum."""
import run as r
import numpy as np,json
from scipy.optimize import least_squares
v=json.load(open(r.R/'fits.json'));lo,hi=r.limits()
for source in ['general_cap0.2','general_cap0.3','general_verified_cap2.0']:
 p=r.seed(source);di=[-1,-1];fit=least_squares(r.residual,np.clip(p,lo+1e-9,hi-1e-9),args=(di,),bounds=(lo,hi),max_nfev=300,ftol=2e-9,xtol=2e-9,gtol=2e-9,x_scale='jac');rec=dict(p=fit.x.tolist(),directions=di,cost=float(np.mean(fit.fun**2)),success=bool(fit.success),nfev=fit.nfev,source=source);v['runs'].append(rec);print(source,rec['cost'],rec['success'],flush=True)
v['best']=min(v['runs'],key=lambda a:a['cost']);b=v['best'];di=b['directions'];fit=least_squares(r.residual,b['p'],args=(di,8192),bounds=(lo,hi),max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale='jac');v['refined']=dict(p=fit.x.tolist(),directions=di,cost=float(np.mean(fit.fun**2)),success=bool(fit.success),nfev=fit.nfev,scores=r.scores(fit.x,di))
for i,k in enumerate(r.KEYS):
 y0=r.predict(fit.x[5*i:5*i+5],fit.x[10:],di[i],r.U[i],8192);y1=r.predict(fit.x[5*i:5*i+5],fit.x[10:],di[i],r.U[i],16384);v['checks'][k]=float(np.max(abs(y1-y0))/r.TR[i].sum(0).max())
(r.R/'fits.json').write_text(json.dumps(v,indent=2));print('refined',v['refined']['cost'],v['refined']['success'],flush=True)
