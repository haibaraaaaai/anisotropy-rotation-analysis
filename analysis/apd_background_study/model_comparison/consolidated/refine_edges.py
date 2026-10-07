from pathlib import Path
import importlib.util,json,numpy as np
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('consolidated_runner',Path(__file__).with_name('run.py'));a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
v=json.load(open(a.R/'fields.json'))
for tag,b in v['edge']['cases'].items():
 if b['success']:continue
 cap=float(tag[:-3])/100;a.setup('edge',cap);lo,hi=a.g.limits();di=b['directions']
 def fun(p):return np.concatenate([a.penalty(*a.fieldpredict('edge',p,di,i,8192)[:4],a.f.TR[i]) for i in range(2)])
 q=least_squares(fun,b['p'],bounds=(lo,hi),max_nfev=350,ftol=2e-7,xtol=2e-7,gtol=2e-7,x_scale='jac');p=q.x;sc={}
 for i,k in enumerate(a.g.KEYS):
  parts=a.fieldpredict('edge',p,di,i,8192);sc[k]={name:a.metrics(parts,dat,a.f.original[i].sum(0).max()) for name,dat in [('train',a.f.TR[i]),('test',a.f.TE[i])]};y,s,c,bg,d,mn=parts;sc[k].update(pred=y.tolist(),signal=s.tolist(),cross=c.tolist(),background=bg.tolist(),directions=d.tolist(),incoherent=a.g.field(p[10:])[1].tolist())
 v['edge']['cases'][tag]=dict(p=p.tolist(),directions=di,cost=float(np.sum(q.fun**2)),success=bool(q.success),nfev=q.nfev,scores=sc,prior_cost=b['cost']);(a.R/'fields.json').write_text(json.dumps(v,indent=2));print(tag,q.success,float(np.sum(q.fun**2)),flush=True)
