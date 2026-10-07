from pathlib import Path
import json,time
import numpy as np
from scipy.optimize import least_squares
from nptdms import TdmsFile
from models import *
R=Path(__file__).resolve().parent;out=R/'interval10';out.mkdir(exist_ok=True)
s=(R.parent/'interval_study/study.py').read_text();ns={'__file__':str(R.parent/'interval_study/study.py')};exec(s[:s.index('baseguide=')],ns)
import sys
sys.path.insert(0,str(R.parent))
from study_paths import tdms_path
with TdmsFile.open(tdms_path()) as f:
 ch=f.groups()[0].channels();raw=np.array([c[2500000:2750000] for c in ch],float)[[3,0,1,2]]
guide=ns['guide'](raw,41);edges=ns['anchors'](guide);channels,_,counts=ns['bin_cycles'](raw,guide,edges)
cal=np.load(R.parent/'interval_study/averaged_channels.npz')['calibration'];allc=np.einsum('ij,kjl->kil',cal,channels)
train=allc[::2].mean(0);test=allc[1::2].mean(0);TMIN=float(train.sum(0).min());K=16
prev=json.loads((R/'fits.json').read_text());rng=np.random.default_rng(20261005);fits={}
def stats(p,m,data,di):
 pred,d=predict(p,m,data,TMIN,K,di)
 return dict(xy_rms=float(np.sqrt(np.mean(np.sum((xy(pred)-xy(data))**2,axis=1)))),channel_rms=float(np.sqrt(np.mean(((pred-data)/data.sum(0))**2))),bg_fraction=d['total_bg']/TMIN,min_remaining_total=float(np.min(data.sum(0)-d['total_bg'])))
for m in prev:
 bestold=prev[m]['best'];pold=np.array(bestold['p']);di=bestold['direction'];lo,hi=bounds(m,K);runs=[]
 starts=[pold]
 for j in range(15):
  p=pold.copy();p[:3]+=rng.normal(0,.2,3);p[3]+=rng.normal(0,.15);p[4:19]+=rng.normal(0,.2,15);p[19]=rng.uniform(.15,.75);p[20]=rng.uniform(.1,.95);p[21]=rng.uniform(-np.pi,np.pi)
  if m=='richard':p[22:]=rng.uniform(-.8,.8,4)
  if m=='field':p[22:]=[rng.uniform(.1,.9),rng.uniform(.1,1.45),rng.uniform(-np.pi,np.pi),rng.uniform(-np.pi,np.pi)]
  starts.append(p)
 if m!='bg':
  pb=np.array(fits['bg']['best']['p']);starts.append(np.r_[pb,([0,0,0,0] if m=='richard' else [.001,.7,.2,.2])])
 for p in starts:
  f=least_squares(residual,np.clip(p,lo+1e-7,hi-1e-7),args=(m,train,TMIN,K,di),bounds=(lo,hi),max_nfev=650,ftol=2e-9,xtol=2e-9,gtol=2e-9,x_scale='jac')
  runs.append(dict(p=f.x.tolist(),direction=di,cost=float(np.mean(f.fun**2)),success=bool(f.success),train=stats(f.x,m,train,di),test=stats(f.x,m,test,di)))
 fits[m]=dict(best=min(runs,key=lambda a:a['cost']),runs=runs)
 print(m,fits[m]['best'],flush=True)
(out/'fits.json').write_text(json.dumps(fits,indent=2))
np.savez_compressed(out/'averages.npz',train=train,test=test,channels_per_cycle=channels,corrected_cycles=allc,bounds=edges,counts=counts,calibration=cal)
metrics=dict(n_cycles=len(channels),n_train=len(allc[::2]),n_test=len(allc[1::2]),total_min=TMIN,
 ordered_split_xy_rms=ns['rms'](xy(train),xy(test)),geometric_split_xy_rms=ns['geom'](xy(train),xy(test)),
 quarter_geometric_rms=[ns['geom'](xy(allc[ids].mean(0)),xy(allc.mean(0))) for ids in np.array_split(np.arange(len(allc)),4)],
 transfer_from30={m:stats(np.r_[np.array(v['best']['p'])[:19],np.array(v['best']['p'])[19]*(np.load(R/'fitted_predictions.npz')['train'].sum(0).min()/TMIN),np.array(v['best']['p'])[20:]],m,test,v['best']['direction']) for m,v in prev.items()})
(out/'metrics.json').write_text(json.dumps(metrics,indent=2));print(metrics)
