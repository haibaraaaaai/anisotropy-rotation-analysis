"""Complex equal-arc fit and profiled inverse-circle experiments.
No pointwise fitted phase parameters. See README for inversion branch limitations.
"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent));sys.path.insert(0,str(R.parent/'arc_matching'))
from models import cone,q_channels,xy,ABC
import run as arc
# Import by distinct module name when executed via other modules.
import importlib.util
spec=importlib.util.spec_from_file_location('arc_reference',R.parent/'arc_matching/run.py');arc=importlib.util.module_from_spec(spec);spec.loader.exec_module(arc)
Z=np.load(R.parent/'arc_matching/inputs.npz');OLD=json.load(open(R.parent/'arc_matching/fits.json'));KEYS=['30s','10s'];A,B,C=ABC;RMAX=C/(A+B)
def bg(h,model,tm):
 if model=='none':return np.zeros(4),np.zeros(4)
 if model=='richard4':return np.zeros(4),np.asarray(h)
 bt,fa,fb=h[:3];b=bt*tm/4*np.array([1+fa,1-fa,1+fb,1-fb]);c=2*np.sqrt(b)*h[3:] if model=='complex' else np.zeros(4)
 return b,c

def bounds(model):
 return {'none':([],[]),'square':([0,-1,-1],[.95,1,1]),'richard4':([-5]*4,[5]*4),'complex':([0,-1,-1]+[-1]*4,[.95,1,1]+[1]*4)}[model]
def complex_forward(p,tm,di,density=2048):
 chi=np.linspace(0,2*np.pi,density+1);d=cone(p[:3],chi);s=np.exp(2*p[3])*np.sum(d[:,:2]**2,1)[None,:]*q_channels(d);b,c=bg(p[5:],'complex',tm);y=s+b[:,None]+c[:,None]*np.sqrt(s)
 pp=xy(y);ar=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pp,axis=0),axis=1))];ar/=max(ar[-1],1e-20);u=(p[4]+di*(np.arange(128)+.5)/128)%1
 pred=np.array([np.interp(u,ar,row) for row in y]);ds=cone(p[:3],np.interp(u,ar,chi))
 return pred,ds,b,c

def sphere(h,model,tm,dat):
 b,c=bg(h,model,tm);remaining=dat-b[:,None];disc=c[:,None]**2+4*remaining
 amp=(-c[:,None]+np.sqrt(np.maximum(disc,1e-16)))/2;s=np.maximum(amp,1e-12)**2
 zz=xy(s);rr=np.linalg.norm(zz,axis=1);sin2=A*rr/np.maximum(C-B*rr,1e-12);th=np.arcsin(np.sqrt(np.clip(sin2,1e-12,1-1e-12)))
 psi=np.unwrap(np.arctan2(zz[:,1],zz[:,0]));phi=psi/2;d=np.column_stack([np.sin(th)*np.cos(phi),np.sin(th)*np.sin(phi),np.cos(th)])
 winding=int(np.rint((np.unwrap(np.r_[np.arctan2(zz[:,1],zz[:,0]),np.arctan2(zz[0,1],zz[0,0])])[-1]-psi[0])/(2*np.pi)))
 # Penalties operate only for trial evaluation; invalid final inversions are explicitly flagged.
 invalid=np.r_[np.maximum(-remaining.ravel(),0)/dat.sum(0).max(),np.maximum(rr-RMAX,0)]
 return s,d,rr,b,c,invalid,winding

def plane(d):
 centre=d.mean(0);cov=(d-centre).T@(d-centre)/len(d);val,vec=np.linalg.eigh(cov);axis=vec[:,0];offset=float(centre@axis)
 if offset<0:axis=-axis;offset=-offset
 residual=d@axis-offset;return axis,offset,residual

def projected(d,axis,offset):
 tangent=d-(d@axis)[:,None]*axis;length=np.linalg.norm(tangent,axis=1);return offset*axis+np.sqrt(max(1-offset**2,1e-12))*tangent/np.maximum(length[:,None],1e-12)

def inv_res(h,model,tm,dat,method):
 hh=h[:-1] if method=='reconstruct' else h
 s,d,rr,b,c,invalid,w=sphere(hh,model,tm,dat);axis,offset,res=plane(d)
 if method=='circle':main=res
 else:
  dp=projected(d,axis,offset);sp=np.exp(2*h[-1])*np.sum(dp[:,:2]**2,1)[None,:]*q_channels(dp);pred=sp+b[:,None]+c[:,None]*np.sqrt(sp);main=((pred-dat)/dat.sum(0)).ravel()
 return np.r_[main,100*invalid]

def evaluate(h,model,tm,tr,te,method):
 hh=h[:-1] if method=='reconstruct' else h
 s,d,rr,b,c,invalid,w=sphere(hh,model,tm,tr);axis,offset,res=plane(d);dp=projected(d,axis,offset)
 if method=='reconstruct':a0=float(np.exp(h[-1]))
 else:
  base=np.sum(dp[:,:2]**2,1)[None,:]*q_channels(dp)
  q=base/tr.sum(0);ell=c[:,None]*np.sqrt(base)/tr.sum(0);zz=(b[:,None]-tr)/tr.sum(0)
  roots=np.roots([4*np.sum(q*q),6*np.sum(q*ell),2*np.sum(ell*ell+2*q*zz),2*np.sum(ell*zz)])
  candidates=[0.]+[float(x.real) for x in roots if abs(x.imag)<1e-7 and x.real>=0]
  a0=min(candidates,key=lambda a:np.sum((a*a*q+a*ell+zz)**2))
 out=dict(h=hh.tolist(),a0=a0,axis=axis.tolist(),offset=offset,cone_half_angle_deg=float(np.rad2deg(np.arccos(np.clip(offset,-1,1)))),bg_percent_Tmax=float(100*b.sum()/tr.sum(0).max()) if model!='richard4' else None,background=b.tolist(),C=c.tolist())
 # For held-out data use TRAIN plane and brightness, but inversions/phase positions use held-out channels.
 for label,dat in [('train',tr),('test',te)]:
  s,d,rr,b,c,invalid,w=sphere(hh,model,tm,dat)
  # Choose global xy sign to agree with training lift; sign is unobservable.
  if label=='test':
   alt=d.copy();alt[:,:2]*=-1
   if np.mean((alt@axis-offset)**2)<np.mean((d@axis-offset)**2):d=alt
  dist=d@axis-offset;dp=projected(d,axis,offset);sp=a0*a0*np.sum(dp[:,:2]**2,1)[None,:]*q_channels(dp);pred=sp+b[:,None]+c[:,None]*np.sqrt(sp)
  theta=np.rad2deg(np.arccos(np.clip(d[:,2],-1,1)));alpha=np.arccos(np.clip(d@axis,-1,1));planeangle=np.arccos(np.clip(offset,-1,1))
  brightness=s.sum(0)/(4*np.maximum(np.sin(np.deg2rad(theta))**2,1e-12)*(A+B*np.sin(np.deg2rad(theta))**2))
  # phase around profiled circle: chronological samples retained, no sorting or independent matching.
  e1=dp[0]-offset*axis;e1/=np.linalg.norm(e1);e2=np.cross(axis,e1);phase=np.unwrap(np.arctan2(dp@e2,dp@e1));diff=np.diff(phase);dominant=np.sign(phase[-1]-phase[0]);backtrack=float(np.sum(abs(diff[diff*dominant<0]))/(2*np.pi))
  out[label]=dict(plane_rms=float(np.sqrt(np.mean(dist**2))),angular_circle_rms_deg=float(np.rad2deg(np.sqrt(np.mean((alpha-planeangle)**2)))),channel_rms=float(np.sqrt(np.mean(((pred-dat)/dat.sum(0))**2))),xy_rms=(float(np.sqrt(np.mean(np.sum((xy(pred)-xy(dat))**2,1)))) if np.all(pred.sum(0)>0) and np.all(pred[0]+pred[1]>0) and np.all(pred[2]+pred[3]>0) else None),theta_range=[float(theta.min()),float(theta.max())],theta=theta.tolist(),pred=pred.tolist(),signal=s.tolist(),directions=d.tolist(),pair_imbalance_rms=float(np.sqrt(np.mean(((s[0]+s[1]-s[2]-s[3])/s.sum(0))**2))),brightness_cv=float(np.std(brightness)/np.mean(brightness)),invalid_points=int(np.sum(rr>RMAX+1e-8)),invalid_channel_roots=int(np.sum(dat-b[:,None]<-1e-8)),max_invalid=float(np.max(invalid)),anisotropy_winding=w,hemisphere_lift_closed=bool(w%2==0),backtracking_turns=backtrack,phase_span_turns=float(abs(phase[-1]-phase[0])/(2*np.pi)))
 return out

def main():
 out=json.load(open(R/'fits.json')) if (R/'fits.json').exists() else {};rng=np.random.default_rng(61713)
 for key in KEYS:
  tr=Z[key+'_train']/Z['scale'];te=Z[key+'_test']/Z['scale'];tm=tr.sum(0).min();out.setdefault(key,{})
  if 'arc_complex' not in out[key]:
   starts=[]
   for model in ['square','none','richard4']:
    p=np.array(OLD[key]['arc_'+model]['best']['p']);h=np.r_[p[5:],np.zeros(4)] if model=='square' else np.r_[.3,0,0,np.zeros(4)]
    for di in [-1,1]:
     for off in [0,.25,.5,.75]:starts.append((np.r_[p[:4],off,h],di))
   for j in range(8):
    p,di=starts[j];p=p.copy();p[:3]+=rng.normal(0,.2,3);p[5]=rng.uniform(.1,.9);p[8:]=rng.uniform(-.8,.8,4);starts.append((p,di))
   l,h=bounds('complex');lo=np.r_[[.001,-np.pi,.001,-8,-2],l];hi=np.r_[[np.pi/2-.001,np.pi,np.pi/2-.001,4,2],h];runs=[]
   for p,di in starts:
    f=least_squares(lambda p:((complex_forward(p,tm,di)[0]-tr)/tr.sum(0)).ravel(),np.clip(p,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=700,ftol=1e-9,xtol=1e-9,gtol=1e-9,x_scale='jac');yp,ds,b,c=complex_forward(f.x,tm,di,8192)
    runs.append(dict(p=f.x.tolist(),direction=di,cost=float(np.mean(f.fun**2)),success=bool(f.success),test_channel_rms=float(np.sqrt(np.mean(((yp-te)/te.sum(0))**2))),test_xy_rms=float(np.sqrt(np.mean(np.sum((xy(yp)-xy(te))**2,1)))),bg_percent_Tmax=float(100*b.sum()/tr.sum(0).max()),theta_range=np.rad2deg(np.arccos(abs(ds[:,2]))).tolist(),pred=yp.tolist(),background=b.tolist(),C=c.tolist()))
   out[key]['arc_complex']=dict(best=min(runs,key=lambda a:a['cost']),runs=runs);(R/'fits.json').write_text(json.dumps(out,indent=2));print(key,'arc complex',out[key]['arc_complex']['best']['test_channel_rms'],flush=True)
  for model in ['none','square','richard4','complex']:
   for method in ['circle','reconstruct']:
    name=method+'_'+model
    if name in out[key]:continue
    if model=='none' and method=='circle':
     out[key][name]=dict(best=evaluate(np.array([]),model,tm,tr,te,method),runs=[]);continue
    nh={'none':0,'square':3,'richard4':4,'complex':7}[model];starts=[]
    if model=='none':starts=[np.array([-.8])]
    else:
     for match in ['arc','flex']:
      if model=='complex':hh=np.array(out[key]['arc_complex']['best']['p'][5:])
      else:hh=np.array(OLD[key][match+'_'+model]['best']['p'][5 if match=='arc' else 20:])
      starts.append(hh)
     starts.append(np.zeros(nh))
     if model=='complex':starts.append(np.r_[out[key]['circle_square']['best']['h'],np.zeros(4)])
     for j in range(6):
      if model=='richard4':hh=rng.uniform(-.3,1,nh)
      else:hh=np.r_[rng.uniform(.05,.7),rng.uniform(-.6,.6,2),rng.uniform(-.7,.7,4) if model=='complex' else []]
      starts.append(hh)
     if method=='reconstruct':starts=[np.r_[p,-.8] for p in starts]+[np.r_[out[key]['circle_'+model]['best']['h'],np.log(out[key]['circle_'+model]['best']['a0'])]]
    low,high=bounds(model);lo=np.array(low+([-8] if method=='reconstruct' else []));hi=np.array(high+([4] if method=='reconstruct' else []));runs=[]
    for p in starts:
     f=least_squares(inv_res,np.clip(p,lo+1e-9,hi-1e-9),args=(model,tm,tr,method),bounds=(lo,hi),max_nfev=700,ftol=1e-9,xtol=1e-9,gtol=1e-9,x_scale='jac')
     ev=evaluate(f.x,model,tm,tr,te,method);runs.append(dict(p=f.x.tolist(),cost=float(np.mean(f.fun**2)),success=bool(f.success),result=ev))
    best=min(runs,key=lambda a:a['cost']);out[key][name]=dict(best=best['result'],selected_success=best['success'],selected_cost=best['cost'],runs=runs);(R/'fits.json').write_text(json.dumps(out,indent=2));s=best['result']['test'];print(key,name,s['angular_circle_rms_deg'],s['channel_rms'],s['brightness_cv'],s['invalid_points'],s['hemisphere_lift_closed'],flush=True)
 (R/'fits.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
