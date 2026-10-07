import run as f
import numpy as np,json
from scipy.optimize import least_squares
R=f.R;old=json.load(open(R/'fits.json'));lo,hi=f.f.limits()
def residual(p,di,density=8192):
 out=[]
 for i in range(2):
  y,s,c,bg,d,mn=f.f.predict(p[5*i:5*i+5],p[10:],di[i],f.U[i],density,True);dat=f.TR[i];t=dat.sum(0);cor=dat-c-bg[:,None];out.extend(((y-dat)/t).ravel())
  floor=1e-4*t;x=(cor[0]-cor[1])/np.maximum(cor[0]+cor[1],floor);z=(cor[2]-cor[3])/np.maximum(cor[2]+cor[3],floor)
  q=np.maximum(0,(np.hypot(x,z)-f.rmax)/(1-f.rmax))
  out.extend(10*q**3) # squared penalty =100*q^6; zero inside rmax, 100 at r=1
  out.extend((np.sqrt(10)*np.minimum(cor,0)/t).ravel())
 return np.array(out)
if __name__=='__main__':
 out=dict(formula='100 * max(0,(r-rmax)/(1-rmax))**6 per training point',negative_weight=10,runs=[])
 for label,b in [('previous_strong',old['10']['best']),('lower_background',old['0']['runs'][1])]:
  di=b['directions'];p=np.clip(b['p'],lo+1e-9,hi-1e-9);fit=least_squares(residual,p,args=(di,),bounds=(lo,hi),max_nfev=160,ftol=2e-8,xtol=2e-8,gtol=2e-8,x_scale='jac');rec=dict(start=label,p=fit.x.tolist(),directions=di,success=bool(fit.success),nfev=fit.nfev,cost=float(np.sum(fit.fun**2)),scores=f.score(fit.x,di));out['runs'].append(rec);out['best']=min(out['runs'],key=lambda a:a['cost']);(R/'steep_fits.json').write_text(json.dumps(out,indent=2));print(label,rec['success'],rec['cost'],[(k,rec['scores'][k]['train'],rec['scores'][k]['test']) for k in f.f.KEYS],flush=True)
