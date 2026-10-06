import run as r
import numpy as np,json
from scipy.optimize import least_squares
R=r.R;rng=np.random.default_rng(88310)
def fitone(start,i,m,cap,field=None):
 lo,hi=r.bounds(m,cap);lo=np.r_[lo[:20],lo[40:]];hi=np.r_[hi[:20],hi[40:]]
 if field is not None:lo=lo[:20];hi=hi[:20];start=start[:20]
 def res(p):
  y=r.predict(p[:20],field if field is not None else p[20:],m)
  return ((y-r.TR[i])/r.TR[i].sum(0)).ravel()
 f=least_squares(res,np.clip(start,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=700,ftol=1e-8,xtol=1e-8,gtol=1e-8,x_scale='jac')
 p=np.r_[f.x,field] if field is not None else f.x
 y=r.predict(p[:20],p[20:],m)
 return dict(p=p.tolist(),cost=float(np.mean(f.fun**2)),success=bool(f.success),nfev=f.nfev,test_xy_rms=float(np.sqrt(np.mean(np.sum((r.xy(y)-r.xy(r.TE[i]))**2,axis=1)))),test_channel_rms=float(np.sqrt(np.mean(((y-r.TE[i])/r.TE[i].sum(0))**2))))
if __name__=='__main__':
 shared=json.loads((R/'fits.json').read_text());results={}
 for cap in [2.,.1]:
  m=5;key=f'm5_cap{cap}';p=np.array(shared[key]['best']['p']);results[key]={}
  for i,k in enumerate(r.KEYS):
   starts=[np.r_[p[20*i:20*i+20],p[40:]]]
   for j in range(4):
    pp=starts[0].copy();pp[:20]+=rng.normal(0,.06,20);pp[24:]+=rng.normal(0,.3,len(pp)-24);starts.append(pp)
   runs=[fitone(pp,i,m,cap) for pp in starts];results[key][k]=min(runs,key=lambda a:a['cost']);print(key,k,'independent',results[key][k]['test_xy_rms'],flush=True)
  for i,k in enumerate(r.KEYS):
   source=r.KEYS[1-i];h=np.array(results[key][source]['p'][20:]);start=np.array(results[key][k]['p']);runs=[fitone(start,i,m,cap,h),fitone(np.r_[p[20*i:20*i+20],h],i,m,cap,h)]
   results[key][f'{source}_to_{k}']=min(runs,key=lambda a:a['cost']);print(key,source,'to',k,results[key][f'{source}_to_{k}']['test_xy_rms'],flush=True)
  (R/'controls.json').write_text(json.dumps(results,indent=2))
