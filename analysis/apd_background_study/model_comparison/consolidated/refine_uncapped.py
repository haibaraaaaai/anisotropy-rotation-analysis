from pathlib import Path
import importlib.util,json,numpy as np
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('consolidated_runner',Path(__file__).with_name('run.py'));a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
pth=a.R/'axial.json';v=json.load(open(pth));cases=v['axial']['cases'];seed=min(cases.values(),key=lambda b:b['cost']);a.setup('axial',None);lo,hi=a.g.limits();lo=np.r_[lo,0,-np.pi,0,-np.pi];hi=np.r_[hi,.3,np.pi,.3,np.pi];di=seed['directions']
def fun(p):return np.concatenate([a.penalty(*a.fieldpredict('axial',p,di,i,8192)[:4],a.f.TR[i]) for i in range(2)])
q=least_squares(fun,seed['p'],bounds=(lo,hi),max_nfev=200,ftol=1e-7,xtol=1e-7,gtol=1e-7,x_scale='jac');p=q.x;sc={}
for i,k in enumerate(a.g.KEYS):
 parts=a.fieldpredict('axial',p,di,i,8192);sc[k]={name:a.metrics(parts,dat,a.f.original[i].sum(0).max()) for name,dat in [('train',a.f.TR[i]),('test',a.f.TE[i])]};y,s,c,bg,d,mn=parts;sc[k].update(pred=y.tolist(),signal=s.tolist(),cross=c.tolist(),background=bg.tolist(),directions=d.tolist(),incoherent=a.g.field(p[10:-4])[1].tolist())
cases['uncapped']=dict(p=p.tolist(),directions=di,cost=float(np.sum(q.fun**2)),success=bool(q.success),nfev=q.nfev,scores=sc,prior_uncapped_cost=cases['uncapped']['cost']);pth.write_text(json.dumps(v,indent=2));print('uncapped',q.success,float(np.sum(q.fun**2)),flush=True)
