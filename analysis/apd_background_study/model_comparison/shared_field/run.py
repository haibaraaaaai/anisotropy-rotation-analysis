import sys,json,os
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent;ROOT=R.parent.parent
sys.path.insert(0,str(R.parent));from models import cone,mechanical_phase,stokes_background,xy
sys.path.insert(0,str(ROOT/'prefit-pipeline/analysis/pupil_comparison'));import pupil_model as pm
KEYS=['30s','10s'];DATA=[dict(np.load(R.parent/p)) for p in ['fitted_predictions.npz','interval10/averages.npz']]
SCALE=max(z['train'].sum(0).max() for z in DATA)
TR=[z['train']/SCALE for z in DATA];TE=[z['test']/SCALE for z in DATA]
BREF=min(z.sum(0).max() for z in TR)
def optics(m,radius=240):
 x,y,fields,area=pm.basis(radius=radius);keep=np.hypot(x,y)>=.38;f=fields()[:,keep,:]
 norm=(np.sum(f[0,:,0]**2)+np.sum(f[0,:,1]**2))*area/2;f=f/np.sqrt(norm)
 raw=np.zeros((5,2,keep.sum()));raw[0,0]=1;raw[1,1]=1
 for j,l in enumerate([2,0,1]):raw[j+2]=f[:2,:,l]
 g=raw.copy()
 for j in range(5):
  for l in range(j):g[j]-=np.sum(g[j]*g[l])*area*g[l]
  g[j]/=np.sqrt(np.sum(g[j]**2)*area)
 g=g[:m];a=np.array([[1,0],[0,1],[1,1],[1,-1]],float);a[2:]/=np.sqrt(2);gm=np.einsum('ij,mjp->ipm',a,g)
 return (np.einsum('ipj,ipk->ijk',f,f)*area,np.einsum('ipj,ipm->ijm',f,gm)*area,np.einsum('ipm,ipn->imn',gm,gm)*area)
OPT={m:optics(m) for m in [3,5]}
def spherical(angles):
 out=[];prod=1.
 for t in angles:out.append(prod*np.cos(t));prod*=np.sin(t)
 return np.r_[out,prod]
def invsphere(u):
 u=np.asarray(u);a=[np.arctan2(np.linalg.norm(u[i+1:]),u[i]) for i in range(len(u)-2)]
 return np.r_[a,np.arctan2(u[-1],u[-2])]
def bgparts(h,m):
 bt,share,pol,ang=h[:4];u=spherical(h[4:]);b=np.sqrt(bt*BREF*share/2)*(u[:m]+1j*u[m:])
 return b,stokes_background(bt*BREF*(1-share),pol,ang)
def predict(local,h,m,parts=False):
 q,H,g=OPT[m];d=cone(local[:3],mechanical_phase(local,128,16,-1));ex=d[:,0]+1j*d[:,1];a0=np.exp(local[-1]);b,inc=bgparts(h,m)
 sig=a0*a0*np.abs(ex)[None,:]**2*np.einsum('tj,ijk,tk->it',d,q,d)
 cross=2*a0*np.real(np.einsum('tj,ijm,m->it',d,H,b.conj())*ex[None,:])
 bg=inc+np.einsum('m,imn,n->i',b.conj(),g,b).real
 return (sig,cross,bg,d) if parts else sig+cross+bg[:,None]
def residual(p,m,dat=TR):return np.concatenate([((predict(p[20*i:20*i+20],p[40:],m)-d)/d.sum(0)).ravel() for i,d in enumerate(dat)])
def bounds(m,cap):
 low=[.001,-np.pi,.001,-4*np.pi]+[-4]*15+[-8];high=[np.pi/2-.001,np.pi,np.pi/2-.001,4*np.pi]+[4]*15+[4]
 return np.r_[low*2,0,0,0,-np.pi,[0]*(2*m-2),-np.pi],np.r_[high*2,cap,1,1,np.pi,[np.pi]*(2*m-2),np.pi]
def oldstart(m,which):
 old=json.loads((R.parent/'radial_field/fits.json').read_text());ps=[np.array(old[k]['radial16']['best']['p']) for k in KEYS];locals=[np.r_[p[:19],p[-1]-.5*np.log(SCALE)] for p in ps]
 p=ps[which];bt=p[19]*TR[which].sum(0).min();eta,rel,common,rf,rp=p[23:28]
 b=np.r_[np.sqrt(1-rf)*np.exp(1j*common)*np.array([np.cos(eta),np.sin(eta)*np.exp(1j*rel)]),np.sqrt(rf)*np.exp(1j*rp)]
 b=np.r_[b,np.zeros(m-3)];u=np.r_[b.real,b.imag]
 return np.r_[*locals,bt/BREF,p[22],p[20],p[21],invsphere(u)]
def stats(p,m):
 out={};bg=p[40]*BREF
 for i,k in enumerate(KEYS):
  y=predict(p[i*20:i*20+20],p[40:],m);s,c,b,d=predict(p[i*20:i*20+20],p[40:],m,True)
  out[k]={'background_fraction_max':float(bg/TR[i].sum(0).max()),'theta_range':np.rad2deg(np.arccos(np.abs(d[:,2]))).tolist(),'a0':float(np.exp(p[i*20+19]))}
  for label,dat in [('train',TR[i]),('test',TE[i])]:out[k][label]={'xy_rms':float(np.sqrt(np.mean(np.sum((xy(y)-xy(dat))**2,axis=1)))),'channel_rms':float(np.sqrt(np.mean(((y-dat)/dat.sum(0))**2))),'total_rms':float(np.sqrt(np.mean(((y.sum(0)-dat.sum(0))/dat.sum(0))**2)))}
 return out
if __name__=='__main__':
 rng=np.random.default_rng(2405);results={};path=R/'fits.json'
 if path.exists():results=json.loads(path.read_text())
 for m,cap in [(3,2.),(3,.1),(5,2.),(5,.1)]:
  key=f'm{m}_cap{cap}'
  if key in results:continue
  starts=[oldstart(m,i) for i in range(2)]
  if m==5:
   p=np.array(results[f'm3_cap{cap}']['best']['p']);b,_=bgparts(p[40:],3);b=np.r_[b,0j,0j];u=np.r_[b.real,b.imag];starts.append(np.r_[p[:44],invsphere(u)])
  if cap==.1:
   free=json.loads((R.parent/'mask_feasibility/free_fits.json').read_text())
   # Prior flexible-overlap cones provide deliberately different initial geometries.
   for j in range(2):
    p=oldstart(m,j)
    for i,k in enumerate(KEYS):
     fp=np.array(free[k]['best']['p']);p[20*i:20*i+19]=fp[:19];p[20*i+19]=fp[19]+.5*np.log(DATA[i]['train'].sum(0).max()/SCALE)
    starts.append(p)
  base=list(starts)
  for j in range(6):
   p=base[j%len(base)].copy();p[:40]+=rng.normal(0,.08,40);p[44:]+=rng.normal(0,.4,len(p)-44);p[40]=rng.uniform(.02,min(cap,1.));p[41]=rng.uniform(.2,.95);starts.append(p)
  lo,hi=bounds(m,cap);runs=[]
  for j,p in enumerate(starts):
   f=least_squares(residual,np.clip(p,lo+1e-8,hi-1e-8),args=(m,),bounds=(lo,hi),max_nfev=650,ftol=1e-8,xtol=1e-8,gtol=1e-8,x_scale='jac')
   e=dict(p=f.x.tolist(),cost=float(np.mean(f.fun**2)),success=bool(f.success),nfev=f.nfev,optimality=float(f.optimality),stats=stats(f.x,m));runs.append(e)
   print(key,j,'cost',e['cost'],'xy',[e['stats'][k]['test']['xy_rms'] for k in KEYS],flush=True)
   results[key]={'best':min(runs,key=lambda a:a['cost']),'runs':runs,'complete':j==len(starts)-1};path.write_text(json.dumps(results,indent=2))
