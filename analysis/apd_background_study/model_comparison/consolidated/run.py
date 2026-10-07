from pathlib import Path
import sys,json,importlib.util
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'gain_domain'))
import caps as c
f=c.f;g=c.g
LABEL={'none':'No background','square':'Additive background','richard_product':'Richard four-C','complex':'Additive + bounded interference','uniform':'Uniform pupil field','three':'Three-mode field','five':'Five-mode field','general':'General stationary field','edge':'Hole-edge field (width 0.10 NA)','axial':'General field + bounded axial motion'}
MODES={'uniform':2,'three':3,'five':5,'general':10,'edge':6,'axial':10}
# Use the same 180-grid pupil integration as the most recent general/five comparison.
sp=importlib.util.spec_from_file_location('edge_basis',R.parent/'three_tests/run.py');basis=importlib.util.module_from_spec(sp);sp.loader.exec_module(basis)
edge=basis.make_basis('edge',width=.1,radius=180)
OPTS={'uniform':c.old.optics(2,radius=180),'three':c.old.optics(3,radius=180),'five':c.opts[5],'general':c.opts[10],'edge':tuple(edge[k] for k in ['q','h','g'])}
OPTS['axial']=OPTS['general']
for key,opt in OPTS.items():np.savez(R/(key+'_optics.npz'),q=opt[0],h=opt[1],g=opt[2])
oldfield=json.load(open(f.R/'caps/fits.json'));simpleold=g.BASE

def penalty(y,s,cross,bg,dat):
 t=dat.sum(0);cor=dat-cross-bg[:,None];floor=1e-4*t
 rr=np.hypot((cor[0]-cor[1])/np.maximum(cor[0]+cor[1],floor),(cor[2]-cor[3])/np.maximum(cor[2]+cor[3],floor));ex=np.maximum(0,(rr-f.rmax)/(1-f.rmax))
 return np.r_[((y-dat)/t).ravel(),10*ex**3,(np.sqrt(10)*np.minimum(cor,0)/t).ravel()]
def metrics(parts,dat,mx):
 y,s,cr,bg,d,mn=parts;cor=dat-cr-bg[:,None];rr=np.linalg.norm(g.xy(cor),axis=1);bad=~np.isfinite(rr)|(rr>f.rmax+1e-8)|(cor<0).any(0)
 return dict(channel_rms=float(np.sqrt(np.mean(((y-dat)/dat.sum(0))**2))),signal_rms=float(np.sqrt(np.mean(((y-dat)/np.maximum(s.sum(0),1e-12))**2))),invalid=int(bad.sum()),radial_exceed=int((rr>f.rmax+1e-8).sum()),negative=int((cor<0).any(0).sum()),max_r=float(rr.max()),bg_percent=float(100*bg.sum()/mx),theta_range=[float(np.degrees(np.arccos(abs(d[:,2]))).min()),float(np.degrees(np.arccos(abs(d[:,2]))).max())])

def setup(model,cap):
 g.q,g.H,g.g=OPTS[model];g.M=MODES[model];g.BCAP=np.inf if cap is None else cap

def fieldseed(model,cap):
 if model=='axial':
  return [dict(p=b['p']+[.10,0.,.10,0.],directions=b['directions']) for b in fieldseed('general',cap)]
 if model in ['five','general']:
  mm=MODES[model];tag='broad' if cap is None or cap>.1 else ('5pct' if cap==.05 else '10pct');return [oldfield[f'm{mm}_{tag}']['best'],oldfield[f'm{mm}_broad']['best']]
 if model in ['uniform','three']:
  pp=np.array(c.arch['m3_cap2.0']['best']['p']);m=MODES[model];b,inc=c.old.bgparts(pp[40:],3);b=b[:m];h=pp[40:44].copy();h[0]*=c.old.BREF*c.old.SCALE/g.Z['scale']/g.BREF;h[4:]=[] if False else h[4:]
  # Retain coherent budget while setting a valid reduced basis direction.
  u=np.r_[b.real,b.imag];u/=np.linalg.norm(u);hh=np.r_[h,c.old.invsphere(u)]
  loc=[]
  for i in range(2):
   a=pp[20*i:20*i+20];l=np.r_[a[:3],a[19]+.5*np.log(c.old.SCALE/g.Z['scale']),0.];chi,arc,*_=g.dense(l,hh);l[4]=np.interp(a[3]%(2*np.pi),chi,arc);loc.extend(l)
  seed=dict(p=np.r_[loc,hh].tolist(),directions=[-1,-1]);q=np.array(seed['p']);q[:10]=np.array(oldfield['m5_broad']['best']['p'])[:10];return [seed,dict(p=q.tolist(),directions=[-1,-1])]
 # Existing pure edge fit: insert noninterfering share/polarization slots.
 saved=json.load(open(R.parent/'three_tests/fits.json'));pp=np.array(saved['edge_w0.1_cap2.0']['best']['p']);hh=np.r_[pp[40],1.,0.,0.,pp[41:]];hh[0]*=basis.BREF*c.old.SCALE/g.Z['scale']/g.BREF;loc=[]
 for i in range(2):
  a=pp[20*i:20*i+20];l=np.r_[a[:3],a[19]+.5*np.log(c.old.SCALE/g.Z['scale']),0.];chi,arc,*_=g.dense(l,hh);l[4]=np.interp(a[3]%(2*np.pi),chi,arc);loc.extend(l)
 q=np.r_[loc,hh];q2=q.copy();q2[:10]=np.array(oldfield['m5_broad']['best']['p'])[:10];return [dict(p=q.tolist(),directions=[-1,-1]),dict(p=q2.tolist(),directions=[-1,-1])]

HZ=None
BASEPRED=g.predict

def fieldpredict(model,p,di,i,density):
 global HZ
 if model!='axial':return BASEPRED(p[5*i:5*i+5],p[10:],di[i],f.U[i],density,True)
 from scipy.interpolate import CubicSpline
 if HZ is None:
  cache=R/'height_optics.npz'
  if cache.exists():zz=np.load(cache);HZ=CubicSpline(zz['z'],zz['hz'],axis=0)
  else:
   opt=basis.make_basis('general',radius=180);x,y=opt['x'],opt['y'];zz=np.linspace(-.31,.31,125);kz=2*np.pi/.633*(1.33+np.sqrt(1.33**2-x*x-y*y));product=np.einsum('ipj,ipm->pijm',opt['f'],opt['gm']).reshape(len(x),-1)*opt['da'];hz=np.concatenate([np.exp(1j*z[:,None]*kz)@product for z in np.array_split(zz,25)]).reshape(len(zz),4,3,10);np.savez(cache,z=zz,hz=hz);HZ=CubicSpline(zz,hz,axis=0)
 local=p[5*i:5*i+5];h=p[10:-4];radius,phase=p[-4+2*i:None if i==1 else -2];chi=np.linspace(0,2*np.pi,density+1);d=g.cone(local[:3],chi);height=radius*np.sin(local[0])*np.cos(chi+phase);amp=np.exp(local[3]);exc=d[:,0]+1j*d[:,1];focus=1/np.sqrt(1+(height/2)**2);b,inc,bg,hb=g.field(h);Hb=CubicSpline(HZ.x,np.einsum('tijm,m->tij',HZ(HZ.x),b.conj()),axis=0);cross=2*amp*np.real(np.einsum('tj,tij->it',d,Hb(height))*exc[None,:]*np.exp(-1j*np.arctan(height/2))[None,:])*focus[None,:];sig=amp**2*abs(exc)[None,:]**2*focus[None,:]**2*np.einsum('tj,ijk,tk->it',d,g.q,d);y=sig+cross+bg[:,None];xy=g.xy(y);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];arc/=arc[-1];target=(local[4]+di[i]*f.U[i])%1
 interp=lambda a:np.array([np.interp(target,arc,row) for row in a]);dd=g.cone(local[:3],np.interp(target,arc,chi));return interp(y),interp(sig),interp(cross),bg,dd,float(y.min())

def fitfield(model,cap,prior=None):
 setup(model,cap);lo,hi=g.limits();seeds=fieldseed(model,cap)
 if model=='axial':lo=np.r_[lo,0,-np.pi,0,-np.pi];hi=np.r_[hi,.3,np.pi,.3,np.pi]
 if prior is not None:seeds=[prior]+seeds[:1]
 def fun(p,di,density):
  return np.concatenate([penalty(*fieldpredict(model,p,di,i,density)[:4],f.TR[i]) for i in range(2)])
 runs=[]
 for b in seeds:
  pp=np.clip(b['p'],lo+1e-9,hi-1e-9);di=b['directions'];res=least_squares(fun,pp,args=(di,2048),bounds=(lo,hi),max_nfev=120,ftol=2e-7,xtol=2e-7,gtol=2e-7,x_scale='jac');runs.append(dict(p=res.x.tolist(),directions=di,cost=float(np.sum(res.fun**2)),success=bool(res.success)))
 b=min(runs,key=lambda a:a['cost']);di=b['directions'];res=least_squares(fun,b['p'],args=(di,8192),bounds=(lo,hi),max_nfev=160,ftol=1e-7,xtol=1e-7,gtol=1e-7,x_scale='jac');p=res.x;scores={}
 for i,k in enumerate(g.KEYS):
  parts=fieldpredict(model,p,di,i,8192);scores[k]={label:metrics(parts,dat,f.original[i].sum(0).max()) for label,dat in [('train',f.TR[i]),('test',f.TE[i])]};y,s,cr,bg,d,mn=parts;scores[k].update(pred=y.tolist(),signal=s.tolist(),cross=cr.tolist(),background=bg.tolist(),directions=d.tolist());_,inc,_,_=g.field(p[10:-4] if model=='axial' else p[10:]);scores[k]['incoherent']=inc.tolist()
 return dict(p=p.tolist(),directions=di,cost=float(np.sum(res.fun**2)),success=bool(res.success),nfev=res.nfev,scores=scores,runs=runs)

def fitsimple(model,cap,prior=None):
 scores={};solutions=[];runsall=[]
 for i,k in enumerate(g.KEYS):
  tm=f.original[i].sum(0).min();mx=f.original[i].sum(0).max();b=simpleold[k]['uniform'][model]['best'];seeds=[dict(p=b['p'],direction=b['direction'])]
  if prior:seeds.insert(0,prior['solutions'][i])
  else:q=np.array(b['p']);q[4]=(q[4]+.5)%1;seeds.append(dict(p=q.tolist(),direction=b['direction']))
  if model=='richard_product':
   baseline=json.load(open(R/'simple.json'))['none']['cases']['uncapped']['solutions'][i];q=np.array(baseline['p']);q[3]=np.exp(2*q[3]);seeds.append(dict(p=np.r_[q,[0.]*4].tolist(),direction=baseline['direction']))
   lo,hi=g.r.bounds('richard4');lo[3]=0;hi[3]=np.exp(8)
  else:lo,hi=g.r.bounds(model)
  if model in ['square','complex']:hi[5]=np.inf if cap is None else cap*mx/tm
  def fun(p,di,density):
   y,s,cr,bg,d,mn=g.r.forward(p,model,tm,di,f.U[i],density,True);return np.r_[penalty(y,s,cr,bg,f.TR[i]),100*max(0,-mn)]
  runs=[]
  for b in seeds:
   pp=np.clip(b['p'],lo+1e-10,hi-1e-10);di=b['direction'];res=least_squares(fun,pp,args=(di,2048),bounds=(lo,hi),max_nfev=160,ftol=2e-7,xtol=2e-7,gtol=2e-7,x_scale='jac');runs.append(dict(p=res.x.tolist(),direction=di,cost=float(np.sum(res.fun**2))))
  b=min(runs,key=lambda a:a['cost']);di=b['direction'];res=least_squares(fun,b['p'],args=(di,8192),bounds=(lo,hi),max_nfev=240,ftol=1e-7,xtol=1e-7,gtol=1e-7,x_scale='jac');p=res.x;solutions.append(dict(p=p.tolist(),direction=di,cost=float(np.sum(res.fun**2)),success=bool(res.success),nfev=res.nfev));parts=g.r.forward(p,model,tm,di,f.U[i],8192,True);scores[k]={label:metrics(parts,dat,mx) for label,dat in [('train',f.TR[i]),('test',f.TE[i])]};y,s,cr,bg,d,mn=parts;scores[k].update(pred=y.tolist(),signal=s.tolist(),cross=cr.tolist(),background=bg.tolist(),directions=d.tolist());runsall.append(runs)
 return dict(solutions=solutions,cost=sum(a['cost'] for a in solutions),success=all(a['success'] for a in solutions),scores=scores,runs=runsall)

def fails(b):return any(b['scores'][k][split]['invalid'] for k in g.KEYS for split in ['train','test'])
def main(group):
 path=R/(group+'.json');out=json.load(open(path)) if path.exists() else {}
 models=['none','square','richard_product','complex'] if group=='simple' else (['axial'] if group=='axial' else [m for m in MODES if m!='axial'])
 for model in models:
  fn=fitsimple if model not in MODES else fitfield;record=out.setdefault(model,dict(cases={},stop=None));cases=record['cases']
  for tag,cap in [('uncapped',None),('5pct',.05),('10pct',.1)]:
   if model in ['none','richard_product'] and cap is not None:continue
   if tag in cases:continue
   prior=cases.get('5pct') if tag=='10pct' else None;cases[tag]=fn(model,cap,prior);path.write_text(json.dumps(out,indent=2));print(group,model,tag,cases[tag]['success'],cases[tag]['cost'],[(k,cases[tag]['scores'][k]['test']['invalid']) for k in g.KEYS],flush=True)
  if model in ['none','richard_product']:record['stop']='No BG-power cap applicable';path.write_text(json.dumps(out,indent=2));continue
  if fails(cases['5pct']) or fails(cases['10pct']):record['stop']='Clipping/negative correction already at 5% or 10%';path.write_text(json.dumps(out,indent=2));continue
  prev=cases['10pct']
  for percent in range(20,101,10):
   tag=f'{percent}pct'
   if tag not in cases:cases[tag]=fn(model,percent/100,prev);path.write_text(json.dumps(out,indent=2));print(group,model,tag,cases[tag]['success'],cases[tag]['cost'],[(k,cases[tag]['scores'][k]['test']['invalid']) for k in g.KEYS],flush=True)
   prev=cases[tag]
   if fails(prev):record['stop']=f'First tested failure at {percent}%';break
   used=max(prev['scores'][k]['train']['bg_percent'] for k in g.KEYS)
   if used<percent-.1 and not fails(cases['uncapped']) and abs(prev['cost']-cases['uncapped']['cost'])<1e-4:record['stop']='Cap inactive; uncapped solution also admissible';break
  if record['stop'] is None:record['stop']='No failure through100%; uncapped case reported separately'
  path.write_text(json.dumps(out,indent=2))
if __name__=='__main__':main(sys.argv[1])
