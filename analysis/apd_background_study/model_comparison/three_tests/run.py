import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent;ROOT=R.parent.parent
import importlib.util
spec=importlib.util.spec_from_file_location('previous_shared',R.parent/'shared_field/run.py');prev=importlib.util.module_from_spec(spec);spec.loader.exec_module(prev)
from models import cone,mechanical_phase,stokes_background,xy,ANALYZERS
TR,TE=prev.TR,prev.TE;KEYS=prev.KEYS;BREF=prev.BREF

def make_basis(kind,width=.06,radius=180):
 x,y,get,da=prev.pm.basis(radius=radius);keep=np.hypot(x,y)>=.38;x=x[keep];y=y[keep];f=get()[:,keep,:]
 norm=(np.sum(f[0,:,0]**2)+np.sum(f[0,:,1]**2))*da/2;f=f/np.sqrt(norm)
 if kind=='general':
  # Scalar span of all Cartesian components of each of three dipole fields.
  raw=f[:2].transpose(0,2,1).reshape(6,-1);G=raw@raw.T*da;ev,U=np.linalg.eigh(G);sel=ev>ev.max()*1e-10
  scalar=(U[:,sel].T@raw)/np.sqrt(ev[sel,None]);ns=len(scalar)
  modes=np.zeros((2*ns,2,len(x)));modes[:ns,0]=scalar;modes[ns:,1]=scalar
 else:
  rho=np.hypot(x,y);env=np.exp(-.5*((rho-.38)/width)**2)
  scalar=np.array([env,env*x/rho,env*y/rho]);G=scalar@scalar.T*da;ev,U=np.linalg.eigh(G);scalar=(U.T@scalar)/np.sqrt(ev[:,None])
  modes=np.zeros((6,2,len(x)));modes[:3,0]=scalar;modes[3:,1]=scalar
 gm=np.einsum('ij,mjp->ipm',ANALYZERS,modes)
 q=np.einsum('ipj,ipk->ijk',f,f)*da;h=np.einsum('ipj,ipm->ijm',f,gm)*da;g=np.einsum('ipm,ipn->imn',gm,gm)*da
 return dict(q=q,h=h,g=g,m=len(modes),kind=kind,width=width,x=x,y=y,f=f,gm=gm,modes=modes,da=da,scalar=scalar)

def field(h,opt,pure):
 m=opt['m'];bt=h[0]*BREF
 if pure:share=1.;inc=np.zeros(4);angles=h[1:]
 else:share,pol,ang=h[1:4];inc=stokes_background(bt*(1-share),pol,ang);angles=h[4:]
 u=prev.spherical(angles);b=np.sqrt(bt*share/2)*(u[:m]+1j*u[m:]);return b,inc

def predict(local,h,opt,pure=False,height=None):
 chi=mechanical_phase(local,128,16,-1);d=cone(local[:3],chi);ex=d[:,0]+1j*d[:,1];a0=np.exp(local[-1]);b,inc=field(h,opt,pure)
 q,H,g=opt['q'],opt['h'],opt['g']
 if height is None:
  cross=2*a0*np.real(np.einsum('tj,ijm,m->it',d,H,b.conj())*ex[None,:]);exc=np.ones(128)
 else:
  radius,phase=height;z=radius*np.sin(local[0])*np.cos(chi+phase)
  # Linear interpolation of complex overlap tensors; z in micrometres.
  iz=np.clip((z-opt['zgrid'][0])/opt['dz'],0,len(opt['zgrid'])-1.000001);ix=np.floor(iz).astype(int);w=iz-ix
  Ht=opt['hinterp'](z) if 'hinterp' in opt else opt['hz'][ix]*(1-w[:,None,None,None])+opt['hz'][ix+1]*w[:,None,None,None]
  exc=1/np.sqrt(1+(z/2.)**2) # conservative Gaussian Rayleigh range 2um; mean plane at waist
  cross=2*a0*np.real(np.einsum('tj,tijm,m->it',d,Ht,b.conj())*ex[None,:]*np.exp(-1j*np.arctan(z/2.))[None,:])*exc[None,:]
 sig=a0*a0*np.abs(ex)[None,:]**2*exc[None,:]**2*np.einsum('tj,ijk,tk->it',d,q,d)
 bg=inc+np.einsum('m,imn,n->i',b.conj(),g,b).real
 return sig+cross+bg[:,None]

def residual(p,opt,pure=False,motion=False):
 off=44 if motion else 40
 return np.concatenate([((predict(p[20*i:20*i+20],p[off:],opt,pure,p[40+2*i:42+2*i] if motion else None)-dat)/dat.sum(0)).ravel() for i,dat in enumerate(TR)])

def bounds(opt,cap,pure=False,motion=False,rmax=.3):
 lo,hi=prev.bounds(opt['m'],cap)
 if pure:lo=np.r_[lo[:41],lo[44:]];hi=np.r_[hi[:41],hi[44:]]
 if motion:lo=np.r_[lo[:40],0,-np.pi,0,-np.pi,lo[40:]];hi=np.r_[hi[:40],rmax,np.pi,rmax,np.pi,hi[40:]]
 return lo,hi

def stats(p,opt,pure=False,motion=False):
 off=44 if motion else 40;out={}
 for i,k in enumerate(KEYS):
  y=predict(p[i*20:i*20+20],p[off:],opt,pure,p[40+2*i:42+2*i] if motion else None);d=cone(p[i*20:i*20+3],mechanical_phase(p[i*20:i*20+20],128,16,-1))
  out[k]={'bg_fraction_max':float(p[off]*BREF/TR[i].sum(0).max()),'theta_range':np.rad2deg(np.arccos(np.abs(d[:,2]))).tolist()}
  if motion:out[k]['height_amplitude_um']=float(p[40+2*i]*np.sin(p[i*20]));out[k]['orbit_radius_um']=float(p[40+2*i])
  for name,dat in [('train',TR[i]),('test',TE[i])]:out[k][name]={'xy_rms':float(np.sqrt(np.mean(np.sum((xy(y)-xy(dat))**2,axis=1)))),'channel_rms':float(np.sqrt(np.mean(((y-dat)/dat.sum(0))**2))),'total_rms':float(np.sqrt(np.mean(((y.sum(0)-dat.sum(0))/dat.sum(0))**2)))}
 return out

def seed(opt,pure,cap,source=None):
 old=json.load(open(R.parent/'shared_field/fits.json'));p=np.array(old[f'm5_cap{.1 if cap==.1 else 2.0}']['best']['p']) if source is None else source
 h=np.zeros((1 if pure else 4)+2*opt['m']-1);h[0]=min(cap,p[40]);u=np.random.default_rng(30).normal(size=2*opt['m']);h[(1 if pure else 4):]=prev.invsphere(u)
 if not pure:h[1:4]=[.9,.8,0.]
 return np.r_[p[:40],h]

def fit(key,opt,cap,pure=False,motion=False,rmax=.3,starts=None,nrandom=4):
 results=json.loads((R/'fits.json').read_text()) if (R/'fits.json').exists() else {}
 if key in results and results[key].get('complete'):return results[key]
 rng=np.random.default_rng(332)
 if starts is None:
  starts=[seed(opt,pure,cap)]
  # alternative low-angle geometry from unconstrained-overlap witness
  p=starts[0].copy();free=json.load(open(R.parent/'mask_feasibility/free_fits.json'))
  for i,k in enumerate(KEYS):
   fp=np.array(free[k]['best']['p']);p[20*i:20*i+19]=fp[:19];p[20*i+19]=fp[19]+.5*np.log(prev.DATA[i]['train'].sum(0).max()/prev.SCALE)
  starts.append(p)
 base=list(starts)
 for j in range(nrandom):
  p=base[j%len(base)].copy();p[:40]+=rng.normal(0,.12,40);off=44 if motion else 40;p[off+(1 if pure else 4):]+=rng.normal(0,.45,len(p)-off-(1 if pure else 4));starts.append(p)
 lo,hi=bounds(opt,cap,pure,motion,rmax);runs=[]
 for j,p in enumerate(starts):
  f=least_squares(residual,np.clip(p,lo+1e-8,hi-1e-8),args=(opt,pure,motion),bounds=(lo,hi),max_nfev=600,ftol=2e-8,xtol=2e-8,gtol=2e-8,x_scale='jac')
  e={'p':f.x.tolist(),'cost':float(np.mean(f.fun**2)),'success':bool(f.success),'nfev':int(f.nfev),'stats':stats(f.x,opt,pure,motion)};runs.append(e)
  results[key]={'best':min(runs,key=lambda a:a['cost']),'runs':runs,'complete':j==len(starts)-1,'modes':opt['m'],'cap':cap,'pure':pure,'motion':motion,'rmax_um':rmax};(R/'fits.json').write_text(json.dumps(results,indent=2))
  print(key,j,e['cost'],[e['stats'][k]['test']['xy_rms'] for k in KEYS],flush=True)
 return results[key]

if __name__=='__main__':
 for w in [.03,.1,.25]:
  opt=make_basis('edge',w)
  for cap in [.1,2.]:fit(f'edge_w{w}_cap{cap}',opt,cap,True,nrandom=2)
 opt=make_basis('general');print('general modes',opt['m'],flush=True)
 np.savez_compressed(R/'general_optics.npz',q=opt['q'],h=opt['h'],g=opt['g'])
 for cap in [.1,2.]:fit(f'general_cap{cap}',opt,cap,nrandom=8)
