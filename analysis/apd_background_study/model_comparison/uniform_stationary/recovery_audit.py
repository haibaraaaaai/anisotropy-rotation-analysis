import sys,json,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import run as f
from models import ABC,xy
v=json.load(open(R/'fits.json'));out=[]
for i,k in enumerate(f.KEYS):
 te=f.TE[i];orig=f.r.Z[k+'_train']/f.Z['scale'];tm=orig.sum(0).min()
 cases=[]
 for m in ['square','richard_product','complex']:
  b=f.BASE[k]['uniform'][m]['best'];p=np.array(b['p']);y,s,c,bg,d,mn=f.r.forward(p,m,tm,b['direction'],f.U[i],8192,True);cases.append((m,y,s,c,bg))
 for name,b in [('stationary_best',v['refined'])]+[(f'stationary_run{j}',b) for j,b in enumerate(v['runs']) if b['success']]:
  p=np.array(b['p']);y,s,c,bg,d,mn=f.predict(p[5*i:5*i+5],p[10:],b['directions'][i],f.U[i],8192,True);cases.append((name,y,s,c,bg))
 for name,y,s,c,bg in cases:
  cor=s+te-y;rr=np.linalg.norm(xy(cor),axis=1);a=ABC[0]*rr/(ABC[2]-ABC[1]*rr);valid=(cor>=0).all(0)&(a>=0)&(a<=1)&np.isfinite(a)
  out.append(dict(window=k,model=name,heldout_measured_normalized_rms=float(np.sqrt(np.mean(((y-te)/te.sum(0))**2))),signal_normalized_rms=float(np.sqrt(np.mean(((y-te)/s.sum(0))**2))),invalid=int((~valid).sum()),bg_percent=float(bg.sum()/orig.sum(0).max()*100),min_signal_percent=float(s.sum(0).min()/orig.sum(0).max()*100)))
(R/'recovery_audit.json').write_text(json.dumps(out,indent=2))
for k in f.KEYS:
 a=[x for x in out if x['window']==k];print(k)
 for x in a[:4]:print(x)
 print('lowest signal error stationary',min(a[3:],key=lambda x:x['signal_normalized_rms']))
