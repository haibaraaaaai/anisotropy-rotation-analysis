"""Ordered arc correspondence versus flexible phase, fixed channel-intensity objective.
Run with numpy/scipy/nptdms; no network or publication side effects.
"""
from pathlib import Path
import sys,json,os
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent))
from models import cone,q_channels,xy,mechanical_phase
N=128; KEYS=['30s','10s']; MODELS=['none','square','richard4']
def prepare():
 from nptdms import TdmsFile
 cal=np.load(R.parent.parent/'interval_study/averaged_channels.npz')['calibration'];out={}
 for key,start in [('30s',30),('10s',10)]:
  with TdmsFile.open(Path(os.environ.get('APD_TDMS_PATH',str(R.parents[3]/'files3_first40s.tdms')))) as f:raw=np.array([c[start*250000:(start+1)*250000] for c in f.groups()[0].channels()],float)[[3,0,1,2]]
  oldguide=xy(savgol_filter(raw,41,3,axis=1));x,y=oldguide.T;bounds=[];armed=False
  for j in range(1,len(x)):
   if x[j]>.3:armed=True
   if armed and x[j]<-.5<=x[j-1] and y[j]<-.3:bounds.append(j);armed=False
  channels=cal@raw;guide=xy(savgol_filter(channels,41,3,axis=1));cc=[];sample_u=[]
  for a,b in zip(bounds[:-1],bounds[1:]):
   pp=guide[a:b];s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pp,axis=0),axis=1))];u=s/s[-1]
   edges=np.r_[0,np.searchsorted(u,np.arange(1,N)/N),len(pp)]
   for j in range(1,N):edges[j]=np.clip(edges[j],edges[j-1]+1,len(pp)-(N-j))
   cc.append(np.column_stack([channels[:,a+i:a+j].mean(1) for i,j in zip(edges[:-1],edges[1:])]))
   sample_u.append([u[i:j] for i,j in zip(edges[:-1],edges[1:])])
  cc=np.array(cc);out[key+'_train']=cc[::2].mean(0);out[key+'_test']=cc[1::2].mean(0);out[key+'_cycles']=cc;out[key+'_bounds']=bounds
  # Stratified inverse-CDF quadrature for within-bin dwell distribution; equal cycle weights.
  for label,ids in [('train',range(0,len(cc),2)),('test',range(1,len(cc),2))]:
   qu=[]
   for k in range(N):
    vals=np.concatenate([sample_u[j][k] for j in ids]);ww=np.concatenate([np.full(len(sample_u[j][k]),1/len(sample_u[j][k])) for j in ids]);order=np.argsort(vals);vals=vals[order];ww=ww[order];cdf=(np.cumsum(ww)-.5*ww)/ww.sum()
    qu.append(np.interp((np.arange(32)+.5)/32,cdf,vals))
   out[key+'_'+label+'_u']=np.array(qu)
 out['scale']=max(out[k+'_train'].sum(0).max() for k in KEYS)
 np.savez_compressed(R/'inputs.npz',**out)
 print('prepared',[(k,len(out[k+'_cycles'])) for k in KEYS],flush=True)

def forward(p,model,tm,kind='arc',direction=-1,density=2048,u=None,parts=False):
 if u is None:u=(np.arange(N)+.5)/N
 if kind=='flex':
  d=cone(p[:3],mechanical_phase(p,N,16,direction));amp=p[19];h=p[20:]
 else:
  chi=np.linspace(0,2*np.pi,density+1);d=cone(p[:3],chi);amp=p[3];h=p[5:]
 s=np.exp(2*amp)*np.sum(d[:,:2]**2,1)[None,:]*q_channels(d)
 b=np.zeros(4);cross=np.zeros_like(s)
 if model=='square':b=h[0]*tm/4*np.array([1+h[1],1-h[1],1+h[2],1-h[2]])
 if model=='richard4':cross=h[:,None]*np.sqrt(s)
 y=s+b[:,None]+cross;minimum=y.min()
 if kind=='flex':return (y,s,cross,b,d,minimum) if parts else y
 # Only invalid trial candidates use a denominator floor; all selected curves must be positive.
 z=np.column_stack([(y[0]-y[1])/np.maximum(y[0]+y[1],1e-9),(y[2]-y[3])/np.maximum(y[2]+y[3],1e-9)])
 arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(z,axis=0),axis=1))];arc/=max(arc[-1],1e-20)
 target=(p[4]+direction*np.asarray(u))%1;f=target.ravel()
 def resample(a):
  z=np.array([np.interp(f,arc,row) for row in a]);z=z.reshape((len(a),)+target.shape)
  return z.mean(-1) if target.ndim==2 else z
 yp=resample(y);sp=resample(s);cp=resample(cross)
 # Directions at bin centre for angle display, not averaged unit vectors.
 centre=(p[4]+direction*(np.arange(N)+.5)/N)%1;dp=cone(p[:3],np.interp(centre,arc,chi))
 return (yp,sp,cp,b,dp,minimum) if parts else yp

def residual(p,model,tm,dat,kind,direction,u=None,density=2048):
 y,s,c,b,d,mn=forward(p,model,tm,kind,direction,density,u,True)
 return np.r_[((y-dat)/dat.sum(0)).ravel(),100*max(0,1e-8-mn)]
def limits(model,kind):
 lo=[.001,-np.pi,.001];hi=[np.pi/2-.001,np.pi,np.pi/2-.001]
 if kind=='arc':lo += [-8,-2];hi += [4,2]
 else:lo += [-4*np.pi]+[-4]*15+[-8];hi += [4*np.pi]+[4]*15+[4]
 lo+= {'none':[],'square':[0,-1,-1],'richard4':[-np.inf]*4}[model];hi+= {'none':[],'square':[.95,1,1],'richard4':[np.inf]*4}[model]
 return np.array(lo),np.array(hi)
def metrics(p,model,tm,dat,kind,di,u=None,density=2048):
 y,s,c,b,d,mn=forward(p,model,tm,kind,di,density,u,True);th=np.rad2deg(np.arccos(abs(d[:,2])))
 return dict(channel_rms=float(np.sqrt(np.mean(((y-dat)/dat.sum(0))**2))),xy_rms=float(np.sqrt(np.mean(np.sum((xy(y)-xy(dat))**2,1)))),total_rms=float(np.sqrt(np.mean(((y.sum(0)-dat.sum(0))/dat.sum(0))**2))),theta=[float(th.min()),float(th.max())],min_dense_intensity=float(mn),background=float(b.sum()))

def main():
 if not (R/'inputs.npz').exists():prepare()
 z=np.load(R/'inputs.npz');old=json.load(open(R.parent/'new_two/fits.json'));results=json.load(open(R/'fits.json')) if (R/'fits.json').exists() else {};rng=np.random.default_rng(60649)
 for key in KEYS:
  tr=z[key+'_train']/z['scale'];te=z[key+'_test']/z['scale'];tm=tr.sum(0).min();results.setdefault(key,{})
  for kind in ['arc','flex']:
   for model in MODELS:
    name=kind+'_'+model
    if name in results[key]:continue
    starts=[]
    for source in ['square','richard4']:
     p=np.array(old[key][source]['best']['p']);p[19]+=.5*np.log(old[key][source]['scale']/z['scale'])
     h=[] if model=='none' else ([.3,0,0] if model=='square' else [0]*4)
     if model==source:h=p[20:]
     if kind=='arc':
      for di in [-1,1]:
       for offset in [0,.25,.5,.75]:starts.append((np.r_[p[:3],p[19],offset,h],di))
     else:
      starts.append((np.r_[p[:20],h],-1));starts.append((np.r_[p[:3],-p[3],p[4:20],h],1))
    if model!='none':
     base=results[key][kind+'_none']['best'];p=np.array(base['p']);h=[0,0,0] if model=='square' else [0]*4;starts.append((np.r_[p,h],base['direction']))
    if kind=='flex':
     # Convert fitted arc correspondence into an approximate 16-knot phase start.
     ar=results[key]['arc_'+model]['best'];p=np.array(ar['p']);di=ar['direction'];dg=4096
     chi=np.linspace(0,2*np.pi,dg+1);d=cone(p[:3],chi);s=np.exp(2*p[3])*np.sum(d[:,:2]**2,1)[None,:]*q_channels(d);h=p[5:]
     yy=s.copy()
     if model=='square':yy+=h[0]*tm/4*np.array([1+h[1],1-h[1],1+h[2],1-h[2]])[:,None]
     if model=='richard4':yy+=h[:,None]*np.sqrt(s)
     arcl=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy(yy),axis=0),axis=1))];arcl/=arcl[-1]
     xx=p[4]+di*(np.linspace(0,1,17)+.5/N);cc=np.interp(xx%1,arcl,chi)+2*np.pi*np.floor(xx);inc=abs(np.diff(cc))/(2*np.pi);logs=np.clip(np.log(inc[:-1]/inc[-1]),-3.9,3.9)
     starts.append((np.r_[p[:3],cc[0],logs,p[3],h],di))
    # Extra varied cone starts; same budgets for three models.
    for j in range(4):
     p,di=starts[j%len(starts)];p=p.copy();p[:3]+=rng.normal(0,.25,3);starts.append((p,di))
    lo,hi=limits(model,kind);runs=[]
    for j,(p,di) in enumerate(starts):
     fit=least_squares(residual,np.clip(p,lo+1e-9,hi-1e-9),args=(model,tm,tr,kind,di),bounds=(lo,hi),max_nfev=500,ftol=2e-9,xtol=2e-9,gtol=2e-9,x_scale='jac')
     rec=dict(p=fit.x.tolist(),direction=di,cost=float(np.mean(fit.fun[:512]**2)),success=bool(fit.success),nfev=fit.nfev,train=metrics(fit.x,model,tm,tr,kind,di),test=metrics(fit.x,model,tm,te,kind,di));runs.append(rec)
    valid=[r for r in runs if r['train']['min_dense_intensity']>0];best=min(valid,key=lambda a:a['cost']);results[key][name]=dict(best=best,runs=runs)
    (R/'fits.json').write_text(json.dumps(results,indent=2));print(key,name,best['cost'],best['test'],best['success'],flush=True)
  # Numerical density and empirical dwell-weighted bin-average sensitivity.
  for model in MODELS:
   name='checks_'+model
   if name in results[key]:continue
   base=results[key]['arc_'+model]['best'];p=np.array(base['p']);di=base['direction'];checks={}
   for density in [4096,8192]:
    y0=forward(p,model,tm,direction=di);y1=forward(p,model,tm,direction=di,density=density)
    checks[str(density)]=dict(max_intensity_change_over_Tmax=float(np.max(abs(y1-y0))/tr.sum(0).max()),test=metrics(p,model,tm,te,'arc',di,density=density))
   u=z[key+'_train_u'];testu=z[key+'_test_u'];lo,hi=limits(model,'arc')
   checks['averaged_at_midpoint_fit']=metrics(p,model,tm,te,'arc',di,u=testu,density=4096)
   fit=least_squares(residual,p,args=(model,tm,tr,'arc',di,u,4096),bounds=(lo,hi),max_nfev=500,ftol=2e-9,xtol=2e-9,gtol=2e-9,x_scale='jac')
   checks['averaged_refit']=dict(p=fit.x.tolist(),success=bool(fit.success),cost=float(np.mean(fit.fun[:512]**2)),test=metrics(fit.x,model,tm,te,'arc',di,u=testu,density=8192))
   checks['theta_change']=float(np.max(abs(np.rad2deg(np.arccos(abs(forward(p,model,tm,direction=di,parts=True)[4][:,2])))-np.rad2deg(np.arccos(abs(forward(fit.x,model,tm,direction=di,parts=True)[4][:,2]))))))
   results[key][name]=checks;(R/'fits.json').write_text(json.dumps(results,indent=2));print(key,name,checks,flush=True)
if __name__=='__main__':main()
