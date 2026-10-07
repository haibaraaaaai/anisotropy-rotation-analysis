"""Feasible hemisphere-lift sensitivity: strict optical domain, closed lift.
This is a limited branch test, not a global solver over equator crossings.
"""
import run as r
import numpy as np,json
from scipy.optimize import minimize
R=r.R;v=json.load(open(R/'fits.json'));out=json.load(open(R/'strict_fits.json'))
for key in r.KEYS:
 tr=r.Z[key+'_train']/r.Z['scale'];te=r.Z[key+'_test']/r.Z['scale'];tm=tr.sum(0).min();out.setdefault(key,{})
 for model in ['square','complex','richard4']:
  valid=list(np.load(R/(key+'_'+model+'_validstarts.npy')))
  if model=='richard4':valid.append(np.array(v[key]['circle_richard4']['best']['h']))
  for method in ['circle','reconstruct']:
   if model!='complex' and method+'_'+model in out[key]:continue
   def cost(p):
    hh=p[:-1] if method=='reconstruct' else p;s,d,rr,b,c,invalid,w=r.sphere(hh,model,tm,tr)
    if method=='circle':main=r.plane(d)[2]
    else:main=r.inv_res(p,model,tm,tr,method)[:512]
    return np.mean(main**2)+(1 if w%2 else 0)
   def constraint(p):
    hh=p[:-1] if method=='reconstruct' else p;s,d,rr,b,c,invalid,w=r.sphere(hh,model,tm,tr)
    return np.r_[(tr-b[:,None]).ravel()-1e-8,r.RMAX-1e-7-rr,rr-1e-7]
   starts=[np.r_[p,-.8] if method=='reconstruct' else p for p in valid];starts=sorted(starts,key=cost)[:8]
   if model=='complex':
    sb=out[key][method+'_square']['best']['p'];starts.append(np.r_[sb[:3],np.zeros(4),sb[3:]])
   low,high=r.bounds(model);low+=([-8] if method=='reconstruct' else []);high+=([4] if method=='reconstruct' else []);runs=[]
   for p in starts:
    if constraint(p).min()>=-1e-8:
     runs.append(dict(p=p.tolist(),success=False,message='Feasible starting point retained',cost=float(cost(p)),constraint_min=float(constraint(p).min()),result=r.evaluate(p,model,tm,tr,te,method)))
    f=minimize(cost,p,method='SLSQP',bounds=list(zip(low,high)),constraints={'type':'ineq','fun':constraint},options={'maxiter':500,'ftol':1e-11})
    ev=r.evaluate(f.x,model,tm,tr,te,method);runs.append(dict(p=f.x.tolist(),success=bool(f.success),message=f.message,cost=float(f.fun),constraint_min=float(constraint(f.x).min()),result=ev))
   feasible=[a for a in runs if a['constraint_min']>=-1e-8 and a['result']['train']['hemisphere_lift_closed']]
   best=min(feasible,key=lambda a:a['cost']) if feasible else None;out[key][method+'_'+model]=dict(best=best,runs=runs)
   if best:
    a=best['result'];print(key,method,model,'success',best['success'],'ang',a['test']['angular_circle_rms_deg'],'chan',a['test']['channel_rms'],'BG',a['bg_percent_Tmax'],'testinvalid',a['test']['max_invalid'],'closed',a['test']['hemisphere_lift_closed'],flush=True)
   else:print(key,method,model,'no feasible returned',flush=True)
   (R/'strict_fits.json').write_text(json.dumps(out,indent=2))
# Counterexample: increase negative C scale without refitting; circle shrinks arbitrarily.
trap={}
for key in r.KEYS:
 tr=r.Z[key+'_train']/r.Z['scale'];te=r.Z[key+'_test']/r.Z['scale'];tm=tr.sum(0).min();h=np.array(v[key]['circle_richard4']['best']['h']);trap[key]=[]
 for mult in [1,2,4,8]:
  a=r.evaluate(h*mult,'richard4',tm,tr,te,'circle');trap[key].append(dict(multiplier=mult,half_angle=a['cone_half_angle_deg'],theta=a['test']['theta_range'],circle_deg=a['test']['angular_circle_rms_deg'],channel=a['test']['channel_rms'],pair=a['test']['pair_imbalance_rms']))
(R/'collapse_check.json').write_text(json.dumps(trap,indent=2))
