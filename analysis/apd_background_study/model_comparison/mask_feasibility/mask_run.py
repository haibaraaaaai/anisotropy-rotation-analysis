import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent;ROOT=R.parent.parent
sys.path.insert(0,str(R.parent));from models import cone,mechanical_phase,bounds,stokes_background,xy,ABC
sys.path.insert(0,str(ROOT/'prefit-pipeline/analysis/pupil_comparison'));import pupil_model as pm
sys.path.insert(0,str(ROOT))
from study_paths import mask_path

def response(maskrot=None,radius=320):
 x,y,fields,area=pm.basis(radius=radius);keep=np.hypot(x,y)>=.38 if maskrot is None else ~pm.mask_blocked(x,y,mask_path(),maskrot)
 f=fields()[:,keep,:];norm=(np.sum(f[0,:,0]**2)+np.sum(f[0,:,1]**2))*area/2;f=f/np.sqrt(norm)
 # Full-vector orthonormal modes, then project onto each analyzer.
 g=np.zeros((3,2,keep.sum()));g[0,0]=1;g[1,1]=1;g[2]=np.stack([f[0,:,2],f[1,:,2]])
 for j in range(3):
  for l in range(j):g[j]-=np.sum(g[j]*g[l])*area*g[l]
  g[j]/=np.sqrt(np.sum(g[j]**2)*area)
 a=np.array([[1,0],[0,1],[1,1],[1,-1]],float);a[2:]/=np.sqrt(2);gm=np.einsum('ij,mjp->ipm',a,g)
 q=np.einsum('ipj,ipk->ijk',f,f)*area
 h=np.einsum('ipj,ipm->ijm',f,gm)*area
 gram=np.einsum('ipm,ipn->imn',gm,gm)*area
 return q,h,gram

def predict(p,n,tm,opt):
 q,h,g=opt;d=cone(p[:3],mechanical_phase(p,n,16,-1));ex=d[:,0]+1j*d[:,1];a0=np.exp(p[-1]);f,pol,ang,share,eta,rel,common,rf,rp=p[19:28];bt=f*tm
 b=np.r_[np.sqrt(bt*share*(1-rf)/2)*np.exp(1j*common)*np.array([np.cos(eta),np.sin(eta)*np.exp(1j*rel)]),np.sqrt(bt*share*rf/2)*np.exp(1j*rp)]
 bg=stokes_background(bt*(1-share),pol,ang)+np.einsum('m,imn,n->i',b.conj(),g,b).real
 sig=a0*a0*np.abs(ex)[None,:]**2*np.einsum('tj,ijk,tk->it',d,q,d)
 cross=2*a0*np.real(np.einsum('tj,ijm,m->it',d,h,b.conj())*ex[None,:])
 return sig+cross+bg[:,None]
def stats(p,data,tm,opt):
 y=predict(p,128,tm,opt)
 return dict(xy_rms=float(np.sqrt(np.mean(np.sum((xy(y)-xy(data))**2,axis=1)))),channel_rms=float(np.sqrt(np.mean(((y-data)/data.sum(0))**2))),total_rms=float(np.sqrt(np.mean(((y.sum(0)-data.sum(0))/data.sum(0))**2))))
if __name__=='__main__':
 old=json.loads((R.parent/'radial_field/fits.json').read_text());res={};rng=np.random.default_rng(883)
 optics={str(a):response(a) for a in [None,0,45,90,135]};np.savez_compressed(R/'mask_optics.npz',**{f'{k}_{j}':v[j] for k,v in optics.items() for j in range(3)})
 # Numerical circular pupil must reproduce previous analytic field implementation.
 sys.path.insert(0,str(R.parent/'radial_field'));from run import pred
 p=np.array(old['30s']['radial16']['best']['p']);print('annulus validation max intensity diff',np.max(np.abs(predict(p,128,1,optics['None'])-pred(p,128,1))),flush=True)
 for key,path in [('30s','fitted_predictions.npz'),('10s','interval10/averages.npz')]:
  z=np.load(R.parent/path);tr=z['train'];te=z['test'];tm=tr.sum(0).min();pb=np.array(old[key]['radial16']['best']['p']);res[key]={}
  for label,opt in optics.items():
   lo,hi=bounds('field',16,2);lo=np.r_[lo,0,-np.pi,-8];hi=np.r_[hi,1,np.pi,4];runs=[]
   for j in range(6):
    p=pb.copy()
    if j:p[:3]+=rng.normal(0,.15,3);p[22:28]+=rng.normal(0,.15,6)
    f=least_squares(lambda p:((predict(p,128,tm,opt)-tr)/tr.sum(0)).ravel(),np.clip(p,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=600,ftol=3e-9,xtol=3e-9,gtol=3e-9,x_scale='jac')
    runs.append(dict(p=f.x.tolist(),cost=float(np.mean(f.fun**2)),success=bool(f.success),train=stats(f.x,tr,tm,opt),test=stats(f.x,te,tm,opt)))
   res[key][label]=min(runs,key=lambda a:a['cost']);print(key,label,res[key][label]['test'],'bg/max',res[key][label]['p'][19]*tm/tr.sum(0).max(),flush=True)
  (R/'mask_fits.json').write_text(json.dumps(res,indent=2))
