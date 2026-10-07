"""Shared general stationary common-pupil field, uniform final-arc references.
One cyclic offset per interval; no fitted flexible phase mapping.
"""
from pathlib import Path
import importlib.util,sys,json
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent))
from models import cone,xy,stokes_background,q_channels
sp=importlib.util.spec_from_file_location('corrected',R.parent/'corrected_arc/run.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
Z=np.load(R.parent/'sampling_tests/references.npz');KEYS=['30s','10s'];TR=[Z[k+'_uniform_train']/Z['scale'] for k in KEYS];TE=[Z[k+'_uniform_test']/Z['scale'] for k in KEYS];U=[r.fractions(x,64) for x in TR]
Q=np.load(R.parent/'three_tests/general_optics.npz');q,H,g=[Q[k] for k in ['q','h','g']];M=H.shape[-1]
BREF=min((r.Z[k+'_train']/Z['scale']).sum(0).max() for k in KEYS);BCAP=.95*min((r.Z[k+'_train']/Z['scale']).sum(0).min() for k in KEYS)/BREF
OLD=json.load(open(R.parent/'three_tests/fits.json'));BASE=json.load(open(R.parent/'sampling_tests/fits.json'))
olddata=[np.load(R.parent/p)['train'] for p in ['fitted_predictions.npz','interval10/averages.npz']];oldscale=max(x.sum(0).max() for x in olddata);oldbref=min(x.sum(0).max() for x in olddata)/oldscale

def spherical(a):
 prod=1.;out=[]
 for x in a:out.append(prod*np.cos(x));prod*=np.sin(x)
 return np.r_[out,prod]
def field(h):
 bt=h[0]*BREF;share=h[1];u=spherical(h[4:]);b=np.sqrt(bt*share/2)*(u[:M]+1j*u[M:]);inc=stokes_background(bt*(1-share),h[2],h[3]);bg=inc+np.einsum('m,imn,n->i',b.conj(),g,b).real;hb=np.einsum('ijm,m->ij',H,b.conj());return b,inc,bg,hb

def dense(local,h,density=2048):
 chi=np.linspace(0,2*np.pi,density+1);d=cone(local[:3],chi);a0=np.exp(local[3]);exc=d[:,0]+1j*d[:,1];b,inc,bg,hb=field(h)
 signal=a0*a0*abs(exc)[None,:]**2*np.einsum('tj,ijk,tk->it',d,q,d)
 cross=2*a0*np.real((hb@d.T)*exc[None,:]);y=signal+cross+bg[:,None];pp=xy(y);ss=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pp,axis=0),axis=1))];ss/=ss[-1]
 return chi,ss,y,signal,cross,bg,d

def predict(local,h,di,u,density=2048,parts=False):
 chi,ss,y,s,c,bg,d=dense(local,h,density);target=(local[4]+di*u)%1
 def interp(z):return np.array([np.interp(target,ss,row) for row in z])
 yp=interp(y)
 if not parts:return yp
 dd=cone(local[:3],np.interp(target,ss,chi));return yp,interp(s),interp(c),bg,dd,float(y.min())
def residual(p,directions,density=2048):
 return np.concatenate([((predict(p[5*i:5*i+5],p[10:],directions[i],U[i],density)-TR[i])/TR[i].sum(0)).ravel() for i in range(2)])
def limits():
 lo=[.001,-np.pi,.001,-8,-2];hi=[np.pi/2-.001,np.pi,np.pi/2-.001,4,2]
 return np.r_[lo*2,[0,0,0,-np.pi],[0]*(2*M-2),-np.pi],np.r_[hi*2,[BCAP,1,1,np.pi],[np.pi]*(2*M-2),np.pi]
def scores(p,di):
 out={};b,inc,bg,hb=field(p[10:]);out['background']=bg.tolist();out['incoherent']=inc.tolist();out['coherent']=(bg-inc).tolist();out['coherent_share']=float(p[11]);out['budget_fraction']=float(p[10]);out['cap']=float(BCAP)
 for i,key in enumerate(KEYS):
  y,s,c,bg,d,mn=predict(p[5*i:5*i+5],p[10:],di[i],U[i],8192,True);th=np.rad2deg(np.arccos(abs(d[:,2])))
  out[key]=dict(pred=y.tolist(),signal=s.tolist(),cross=c.tolist(),theta=th.tolist(),bg_percent_original_Tmax=float(100*bg.sum()/(r.Z[key+'_train']/Z['scale']).sum(0).max()),theta_range=[float(th.min()),float(th.max())],min_pred=mn)
  for label,dat in [('train',TR[i]),('test',TE[i])]:out[key][label]=dict(channel_rms=float(np.sqrt(np.mean(((y-dat)/dat.sum(0))**2))),xy_rms=float(np.sqrt(np.mean(np.sum((xy(y)-xy(dat))**2,1)))),total_rms=float(np.sqrt(np.mean(((y.sum(0)-dat.sum(0))/dat.sum(0))**2))))
 return out

def seed(source):
 p=np.array(OLD[source]['best']['p']);h=p[40:].copy();h[0]=min(BCAP,h[0]*oldbref*oldscale/Z['scale']/BREF)
 local=[]
 for i in range(2):
  pp=p[20*i:20*i+20];l=np.r_[pp[:3],pp[19]+.5*np.log(oldscale/Z['scale']),0.];chi,ss,*_=dense(l,h);l[4]=np.interp(pp[3]%(2*np.pi),chi,ss);local.extend(l)
 return np.r_[local,h]

def main():
 rng=np.random.default_rng(61834);starts=[]
 for name in ['general_verified_cap0.1','general_cap0.2','general_cap0.3','general_verified_cap2.0']:
  p=seed(name);starts.append((p,[-1,-1]))
  if name=='general_verified_cap0.1':
   for di in [[1,1],[-1,1],[1,-1]]:starts.append((p.copy(),di))
 # Uniform-reference bounded-overlap geometry, with stationary-field seeds.
 for source in ['general_cap0.2','general_verified_cap2.0']:
  p=seed(source)
  for i,key in enumerate(KEYS):p[5*i:5*i+5]=BASE[key]['uniform']['complex']['best']['p'][:5]
  starts.append((p,[-1,-1]))
 for j in range(2):
  p,di=starts[4+j];p=p.copy();p[:10]+=rng.normal(0,.15,10);p[14:]+=rng.normal(0,.2,len(p)-14);starts.append((p,di))
 lo,hi=limits();out=json.load(open(R/'fits.json')) if (R/'fits.json').exists() else dict(runs=[])
 for j,(p,di) in enumerate(starts):
  if j<len(out['runs']):continue
  f=least_squares(residual,np.clip(p,lo+1e-9,hi-1e-9),args=(di,),bounds=(lo,hi),max_nfev=300,ftol=2e-9,xtol=2e-9,gtol=2e-9,x_scale='jac')
  rec=dict(p=f.x.tolist(),directions=di,cost=float(np.mean(f.fun**2)),success=bool(f.success),nfev=f.nfev);out['runs'].append(rec);out['best']=min(out['runs'],key=lambda a:a['cost']);(R/'fits.json').write_text(json.dumps(out,indent=2));print(j,rec['cost'],rec['success'],rec['nfev'],flush=True)
 b=out['best'];p=np.array(b['p']);di=b['directions'];f=least_squares(residual,p,args=(di,8192),bounds=(lo,hi),max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale='jac');out['refined']=dict(p=f.x.tolist(),directions=di,cost=float(np.mean(f.fun**2)),success=bool(f.success),nfev=f.nfev,scores=scores(f.x,di))
 checks={}
 for i,k in enumerate(KEYS):
  y0=predict(f.x[5*i:5*i+5],f.x[10:],di[i],U[i],8192);y1=predict(f.x[5*i:5*i+5],f.x[10:],di[i],U[i],16384);checks[k]=float(np.max(abs(y1-y0))/TR[i].sum(0).max())
 dd=cone([.7,.3,.4],np.linspace(0,2*np.pi,200));checks['max_unit_signal_q_difference']=float(np.max(abs(np.einsum('tj,ijk,tk->it',dd,q,dd)-q_channels(dd))))
 out['checks']=checks;(R/'fits.json').write_text(json.dumps(out,indent=2));print('final',out['refined']['success'],out['refined']['cost'],checks,flush=True)
if __name__=='__main__':main()
