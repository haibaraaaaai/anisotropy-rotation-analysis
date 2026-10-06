import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent))
from models import *
# Added mode: normalized collected field pattern of a z-directed dipole.
# Radially polarized in the pupil, orthogonal to both uniform transverse modes.
# Each analyzer transmits half its power; overlap with signal is sqrt(A/2)*d_z.
KAPPA=np.sqrt(ABC[0]/2)
def pred(p,n,tm,k=16,radial=True):
 d=cone(p[:3],mechanical_phase(p,n,k,-1));q=q_channels(d);a0=np.exp(p[-1]);o=k+3
 f,pol,ang,share,eta,rel,common=p[o:o+7];bt=f*tm
 rf,rp=p[o+7:o+9] if radial else (0.,0.)
 b=np.sqrt(bt*share*(1-rf)/2)*np.exp(1j*common)*np.array([np.cos(eta),np.sin(eta)*np.exp(1j*rel)])
 br=np.sqrt(bt*share*rf/2)*np.exp(1j*rp);bj=ANALYZERS@b
 bg=stokes_background(bt*(1-share),pol,ang)+np.abs(bj)**2+abs(br)**2/2
 ex=d[:,0]+1j*d[:,1];s2=np.abs(ex)**2
 cross=2*a0*np.real((GAMMA*(ANALYZERS@d[:,:2].T)*np.conj(bj[:,None])+KAPPA*d[:,2][None,:]*np.conj(br))*ex[None,:])
 return q*(a0*a0*s2)[None,:]+cross+bg[:,None]
def residual_fixed(p,data,tm,k,radial):return ((pred(p,data.shape[1],tm,k,radial)-data)/data.sum(0)).ravel()
def remap(p,oldk,newk,radial):
 inc=softmax(np.r_[p[4:oldk+3],0]);cum=np.r_[0,np.cumsum(inc)]
 inc2=np.diff(np.interp(np.linspace(0,1,newk+1),np.linspace(0,1,oldk+1),cum))
 return np.r_[p[:4],np.log(inc2[:-1]/inc2[-1]),p[oldk+3:]]
def fit(starts,train,test,k=16,radial=True,cap=2.):
 lo,hi=bounds('field',k,cap)
 if radial:lo=np.r_[lo,0,-np.pi];hi=np.r_[hi,1,np.pi]
 lo=np.r_[lo,-8];hi=np.r_[hi,4];tm=train.sum(0).min();runs=[]
 for p in starts:
  f=least_squares(residual_fixed,np.clip(p,lo+1e-9,hi-1e-9),args=(train,tm,k,radial),bounds=(lo,hi),max_nfev=700,ftol=3e-9,xtol=3e-9,gtol=3e-9,x_scale='jac')
  entry=dict(p=f.x.tolist(),cost=float(np.mean(f.fun**2)),success=bool(f.success),nfev=int(f.nfev))
  for name,data in [('train',train),('test',test)]:
   y=pred(f.x,128,tm,k,radial)
   entry[name]=dict(xy_rms=float(np.sqrt(np.mean(np.sum((xy(y)-xy(data))**2,axis=1)))),channel_rms=float(np.sqrt(np.mean(((y-data)/data.sum(0))**2))),total_relative_rms=float(np.sqrt(np.mean(((y.sum(0)-data.sum(0))/data.sum(0))**2))))
  runs.append(entry)
 return dict(best=min(runs,key=lambda a:a['cost']),runs=runs,k=k,radial=radial,cap=cap)
if __name__=='__main__':
 prev=json.loads((R.parent/'fixed_excitation/fits.json').read_text());rng=np.random.default_rng(786);res={}
 for key,path in [('30s','fitted_predictions.npz'),('10s','interval10/averages.npz')]:
  z=np.load(R.parent/path);train=z['train'];test=z['test'];pb=np.array(prev[key]['field_cap_checks']['2.0']['best']['p']);starts=[]
  for j in range(18):
   p=np.r_[pb[:-1],.001 if j==0 else rng.uniform(.01,.65),rng.uniform(-np.pi,np.pi),pb[-1]]
   if j:p[:3]+=rng.normal(0,.15,3);p[4:19]+=rng.normal(0,.15,15)
   starts.append(p)
  res[key]={'radial16':fit(starts,train,test)};best=res[key]['radial16']['best'];print(key,'radial16',best,flush=True)
  for k in [8,32]:
   for radial,p in [(False,pb),(True,np.array(best['p']))]:
    name=('radial' if radial else 'uniform')+str(k);res[key][name]=fit([remap(p,16,k,radial)],train,test,k,radial);print(key,name,res[key][name]['best']['train'],res[key][name]['best']['test'],flush=True)
  res[key]['radial16_cap5']=fit([np.array(best['p'])],train,test,cap=5.)
  print(key,'cap5',res[key]['radial16_cap5']['best'],flush=True)
  (R/'fits.json').write_text(json.dumps(res,indent=2))
