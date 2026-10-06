import sys
from pathlib import Path
R=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(R))
from models import *
from scipy.optimize import least_squares
import json
s=(R.parent/'interval_study/study.py').read_text();ns={'__file__':str(R.parent/'interval_study/study.py')};exec(s[:s.index('baseguide=')],ns)
r10=json.loads((R/'interval10/fits.json').read_text());r30=json.loads((R/'fits.json').read_text());z10=np.load(R/'interval10/averages.npz');z30=np.load(R/'fitted_predictions.npz');out={}
for label,z,r in [('10–11 s',z10,r10),('30–31 s',z30,r30)]:
 train=z['train'];test=z['test'];tm=train.sum(0).min();out[label]={}
 for m in r:
  b=r[m]['best'];p=np.array(b['p']);di=b['direction'];pred,_=predict(p,m,test,tm,16,di)
  out[label][m]={'geometric_test_rms':ns['geom'](xy(pred),xy(test))}
 # Check alternate field minimum and caps; these are diagnostic refits, no held-out optimization.
 starts=[np.array(r10['field']['best']['p']),np.array(r30['field']['best']['p'])]
 checks=[]
 for cap in [.8,.95,.99]:
  runs=[]
  for p in starts:
   lo,hi=bounds('field',16,cap)
   f=least_squares(residual,np.clip(p,lo+1e-7,hi-1e-7),args=('field',train,tm,16,-1),bounds=(lo,hi),max_nfev=1000,ftol=2e-10,xtol=2e-10,gtol=2e-10,x_scale='jac')
   pred,_=predict(f.x,'field',test,tm,16,-1)
   runs.append(dict(cap=cap,p=f.x.tolist(),train_cost=float(np.mean(f.fun**2)),test_xy_rms=float(np.sqrt(np.mean(np.sum((xy(pred)-xy(test))**2,axis=1)))),test_geometric_rms=ns['geom'](xy(pred),xy(test))))
  checks.append(min(runs,key=lambda a:a['train_cost']))
 out[label]['field_checks']=checks
 print(label,json.dumps(out[label]),flush=True)
(R/'interval10/checks.json').write_text(json.dumps(out,indent=2))
