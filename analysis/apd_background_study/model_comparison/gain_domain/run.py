from pathlib import Path
import sys,json,importlib.util
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('stationary',R.parent/'uniform_stationary/run.py');f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
cal=np.load(R.parent.parent/'interval_study/averaged_channels.npz')['calibration'];pair=np.array([1,1,-1,-1]);Z=f.r.Z
# Multiplicative adjustments to historical inverse gains, before unchanged matrix.
def fitgain(keys):
 dat=np.concatenate([Z[k+'_train']/f.Z['scale'] for k in keys],axis=1);raw=np.linalg.solve(cal,dat);design=(raw*(pair@cal)[:,None]/dat.sum(0)).T
 def fun(q):
  g=np.exp(np.r_[q,-sum(q)]);return design@g
 opt=least_squares(fun,np.zeros(3),max_nfev=1000,gtol=1e-13,ftol=1e-13,xtol=1e-13)
 return np.exp(np.r_[opt.x,-sum(opt.x)]),np.linalg.svd(design,compute_uv=False),opt
G,sv,op=fitgain(f.KEYS);transform=cal@np.diag(G)@np.linalg.inv(cal)
calreport=dict(inverse_gain_multipliers=G.tolist(),singular_values=sv.tolist(),success=bool(op.success),separate={})
for k in f.KEYS:calreport['separate'][k]=fitgain([k])[0].tolist()
TR=[];TE=[];U=[];original=[]
for k in f.KEYS:
 tr=transform@(Z[k+'_train']/f.Z['scale']);te=transform@(Z[k+'_test']/f.Z['scale']);original.append(tr)
 tt=np.arange(16385)/128;dense=np.array([np.interp(tt,np.arange(129),np.r_[a,a[0]]) for a in tr]);a=f.xy(dense);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(a,axis=0),axis=1))];idx=np.interp(np.arange(128)/128,arc/arc[-1],tt)
 res=lambda dat:np.array([np.interp(idx,np.arange(129),np.r_[a,a[0]]) for a in dat])
 TR.append(res(tr));TE.append(res(te));U.append(f.r.fractions(TR[-1],64))
 calreport[k]={}
 for label,dat0,dat in [('train',Z[k+'_train']/f.Z['scale'],tr),('heldout',Z[k+'_test']/f.Z['scale'],te)]:calreport[k][label]=dict(before=float(np.sqrt(np.mean(((pair@dat0)/dat0.sum(0))**2))),after=float(np.sqrt(np.mean(((pair@dat)/dat.sum(0))**2))))
f.TR=TR;f.TE=TE;f.U=U;f.BREF=min(x.sum(0).max() for x in original);f.BCAP=.95*min(x.sum(0).min() for x in original)/f.BREF
(R/'calibration.json').write_text(json.dumps(calreport,indent=2));np.savez(R/'references.npz',**{k+'_'+label:dat for k,i in zip(f.KEYS,range(2)) for label,dat in [('train',TR[i]),('test',TE[i])]},transform=transform)
print('calibration',calreport,flush=True)
rmax=.9277805723143716

def residual(p,di,weight,density=1024):
 out=[]
 for i in range(2):
  y,s,c,bg,d,mn=f.predict(p[5*i:5*i+5],p[10:],di[i],U[i],density,True);t=TR[i].sum(0);cor=TR[i]-c-bg[:,None];out.extend(((y-TR[i])/t).ravel())
  if weight:
   # Smooth soft domain excess expressed without singular anisotropy division:
   # enforce pair magnitudes <= radial bound by stable positive denominators.
   p1=cor[0]+cor[1];p2=cor[2]+cor[3];floor=1e-4*t
   x=(cor[0]-cor[1])/np.maximum(p1,floor);z=(cor[2]-cor[3])/np.maximum(p2,floor)
   excess=np.maximum(0,np.hypot(x,z)/rmax-1)
   out.extend(np.sqrt(weight)*excess);out.extend((np.sqrt(weight)*np.minimum(cor,0)/t).ravel())
 return np.array(out)
def score(p,di):
 out={}
 for i,k in enumerate(f.KEYS):
  y,s,c,bg,d,mn=f.predict(p[5*i:5*i+5],p[10:],di[i],U[i],8192,True);rec={}
  for label,dat in [('train',TR[i]),('test',TE[i])]:
   cor=dat-c-bg[:,None];rr=np.linalg.norm(f.xy(cor),axis=1);invalid=(cor<0).any(0)|(rr>rmax)|~np.isfinite(rr)
   rec[label]=dict(channel_rms=float(np.sqrt(np.mean(((y-dat)/dat.sum(0))**2))),invalid=int(invalid.sum()),negative=int((cor<0).any(0).sum()),max_radial_excess=float(np.max(np.maximum(0,rr-rmax))))
  rec['bg_percent']=float(bg.sum()/original[i].sum(0).max()*100);rec['pred']=y.tolist();rec['signal']=s.tolist();rec['cross']=c.tolist();rec['background']=bg.tolist();out[k]=rec
 return out
if __name__=='__main__':
 old=json.load(open(R.parent/'uniform_stationary/fits.json'));seeds=[old['refined'],old['runs'][10]];out={};lo,hi=f.limits()
 for w in [0,.01,.1,1]:
  runs=[]
  starts=seeds if w==0 else [out['0']['best'],out[str(prev)]['best']]
  for b in starts:
   p=np.clip(b['p'],lo+1e-9,hi-1e-9);di=b['directions'];opt=least_squares(residual,p,args=(di,w),bounds=(lo,hi),max_nfev=120,ftol=2e-8,xtol=2e-8,gtol=2e-8,x_scale='jac');rec=dict(p=opt.x.tolist(),directions=di,cost=float(np.sum(opt.fun**2)),success=bool(opt.success),nfev=opt.nfev,scores=score(opt.x,di));runs.append(rec);print('fit',w,rec['success'],rec['cost'],[(k,rec['scores'][k]['test']) for k in f.KEYS],flush=True)
  out[str(w)]=dict(best=min(runs,key=lambda b:b['cost']),runs=runs);(R/'fits.json').write_text(json.dumps(out,indent=2));prev=w
