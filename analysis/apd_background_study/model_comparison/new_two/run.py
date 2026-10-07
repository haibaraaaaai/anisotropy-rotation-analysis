"""Matched fixed-excitation tests: square-constrained additive BG and Richard four-C."""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent))
from models import cone,mechanical_phase,q_channels,xy,stokes_background
KEYS=['30s','10s'];DATA=[dict(np.load(R.parent/p)) for p in ['fitted_predictions.npz','interval10/averages.npz']]
SCALE=max(d['train'].sum(0).max() for d in DATA)
TR=[d['train']/SCALE for d in DATA];TE=[d['test']/SCALE for d in DATA]
def predict(p,model,tm,parts=False):
 d=cone(p[:3],mechanical_phase(p,128,16,-1));s=np.exp(2*p[19])*np.sum(d[:,:2]**2,axis=1)[None,:]*q_channels(d)
 if model=='square':
  bt,fa,fb=p[20:];b=bt*tm/4*np.array([1+fa,1-fa,1+fb,1-fb]);cross=np.zeros_like(s)
 else:b=np.zeros(4);cross=p[20:,None]*np.sqrt(s)
 y=s+b[:,None]+cross
 return (y,s,cross,b,d) if parts else y

def residual(p,model,dat,tm):return ((predict(p,model,tm)-dat)/dat.sum(0)).ravel()
def stats(p,model,dat,tm):
 y,s,c,b,d=predict(p,model,tm,True);th=np.rad2deg(np.arccos(abs(d[:,2])))
 return dict(xy_rms=float(np.sqrt(np.mean(np.sum((xy(y)-xy(dat))**2,axis=1)))),channel_rms=float(np.sqrt(np.mean(((y-dat)/dat.sum(0))**2))),total_rms=float(np.sqrt(np.mean(((y.sum(0)-dat.sum(0))/dat.sum(0))**2))),theta_range=[float(th.min()),float(th.max())],min_pred=float(y.min()),bg_total=float(b.sum()),cross_range=[float(c.sum(0).min()),float(c.sum(0).max())])
def run():
 old=json.load(open(R.parent/'fixed_excitation/fits.json'));shared=json.load(open(R.parent/'shared_field/fits.json'));rng=np.random.default_rng(606151)
 results=json.load(open(R/'fits.json')) if (R/'fits.json').exists() else {}
 for i,k in enumerate(KEYS):
  results.setdefault(k,{});tm=TR[i].sum(0).min()
  for model in ['square','richard4']:
   if results[k].get(model,{}).get('complete'):continue
   starts=[]
   for family in ['bg','richard','field']:
    p=np.array(old[k][family]['best']['p']);local=np.r_[p[:19],p[-1]-.5*np.log(SCALE)]
    bg=stokes_background(p[19]*tm,p[20],p[21])
    h=np.r_[p[19],p[20]*np.cos(p[21]),p[20]*np.sin(p[21])] if model=='square' else (bg[:,None]/np.sqrt(predict(np.r_[local,np.zeros(4)],'richard4',tm))).mean(1)
    if model=='richard4' and family=='richard':h+=2*np.sqrt(bg)*p[22:26]
    starts.append(np.r_[local,h])
   pp=np.array(shared['m5_cap0.1']['best']['p']);loc=pp[i*20:i*20+20];starts.append(np.r_[loc,[.15,0,0] if model=='square' else [0]*4])
   for j in range(6):
    p=starts[j%4].copy();p[:3]+=rng.normal(0,.20,3);p[3]+=rng.normal(0,.2);p[4:19]+=rng.normal(0,.2,15);p[19]+=rng.normal(0,.2)
    if model=='square':p[20:]=[rng.uniform(.05,.9),rng.uniform(-1,1),rng.uniform(-1,1)]
    else:p[20:]+=rng.normal(0,.15,4)
    starts.append(p)
   lo=np.r_[[.001,-np.pi,.001,-4*np.pi],[-4]*15,-8,[0,-1,-1] if model=='square' else [-np.inf]*4]
   hi=np.r_[[np.pi/2-.001,np.pi,np.pi/2-.001,4*np.pi],[4]*15,4,[.95,1,1] if model=='square' else [np.inf]*4]
   runs=[]
   for j,p in enumerate(starts):
    f=least_squares(residual,np.clip(p,lo+1e-9,hi-1e-9),args=(model,TR[i],tm),bounds=(lo,hi),max_nfev=700,ftol=2e-9,xtol=2e-9,gtol=2e-9,x_scale='jac')
    e=dict(p=f.x.tolist(),cost=float(np.mean(f.fun**2)),success=bool(f.success),nfev=int(f.nfev),optimality=float(f.optimality),train=stats(f.x,model,TR[i],tm),test=stats(f.x,model,TE[i],tm));runs.append(e)
    results[k][model]=dict(best=min(runs,key=lambda e:e['cost']),runs=runs,complete=j==len(starts)-1,scale=SCALE)
    (R/'fits.json').write_text(json.dumps(results,indent=2));print(k,model,j,e['cost'],e['test']['xy_rms'],e['success'],flush=True)
if __name__=='__main__':run()
