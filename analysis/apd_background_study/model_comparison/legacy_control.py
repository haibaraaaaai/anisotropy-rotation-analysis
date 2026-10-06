from pathlib import Path
import json,numpy as np
from scipy.optimize import differential_evolution,minimize
from models import cone,q_channels,xy
R=Path(__file__).resolve().parent
z=np.load(R.parent/'interval_study/averaged_channels.npz')
c=np.einsum('ij,kjl->kil',z['calibration'],z['channels_per_cycle'])
new=c[::2].mean(axis=0)
old=np.load(R.parent/'prefit-pipeline/analysis/pupil_comparison/report/fit_channels.npz')['channels']

def arclen(p,n=256):
 p=np.roll(p,-np.argmin(p[:,0]),axis=0);p=np.vstack([p,p[0]])
 s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
 return np.column_stack([np.interp(np.arange(n)/n,s/s[-1],p[:,j]) for j in range(2)])
def cost(p,data,physical=False):
 dc,fa,fb=p[3:]
 if physical and fa*fa+fb*fb>1:return 1e3+(fa*fa+fb*fb-1)*1e3
 bg=dc/2*np.array([1+fa,1-fa,1+fb,1-fb]);s=data-bg[:,None]
 if np.any(s<=0):return 1e3
 dd=arclen(xy(s));mm=xy(q_channels(cone(np.deg2rad(p[:3]),np.arange(720)*2*np.pi/720)))
 return min(np.mean(np.sum((dd-arclen(mm))**2,axis=1)),np.mean(np.sum((dd-arclen(mm[::-1]))**2,axis=1)))
res={}
for label,data,physical in [('old_selected',old,False),('averaged',new,False),('averaged_physical',new,True)]:
 hi=min(data[0].min()+data[1].min(),data[2].min()+data[3].min())
 b=[(0,90),(-180,180),(0,90),(0,hi),(-1,1),(-1,1)];runs=[]
 for seed in [42,91]:
  r=differential_evolution(cost,b,args=(data,physical),seed=seed,maxiter=400,popsize=16,tol=1e-7,polish=False)
  pol=minimize(cost,r.x,args=(data,physical),method='Powell',bounds=b,options={'maxiter':250,'ftol':1e-9})
  best=pol if pol.fun<r.fun else r
  runs.append(dict(p=best.x.tolist(),cost=float(best.fun),success=bool(best.success)))
  print(label,seed,runs[-1],flush=True)
 res[label]=dict(best=min(runs,key=lambda x:x['cost']),runs=runs)
 (R/'legacy_control.json').write_text(json.dumps(res,indent=2)+'\n')
