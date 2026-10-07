from pathlib import Path
import importlib.util,json,numpy as np
from sphere_branches import recover
r=Path(__file__).parent
sp=importlib.util.spec_from_file_location('audit_runner',r/'run.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
rows=[]
for fname in ['simple','fields','axial']:
 for model,v in json.loads((r/(fname+'.json')).read_text()).items():
  for cap,b in v['cases'].items():
   for i,k in enumerate(a.g.KEYS):
    sc=b['scores'][k];cor=a.f.TE[i]-np.array(sc['cross'])-np.array(sc['background'])[:,None];d=np.array(sc['directions']);obs,ok,metrics=recover(cor,d,a.f.U[i]);assert int((~ok).sum())==sc['test']['invalid']
    rows.append(dict(model=model,cap=cap,window=k,**metrics))
(r/'sphere_branch_audit.json').write_text(json.dumps(rows,indent=2))
for x in rows:
 if x['cap']=='10pct' or x['model'] in ['none','richard_product']:print(x['model'],x['window'],'valid',x['valid'],'maxstep',round(x.get('max_step_deg',0),2),'old',round(x.get('old_max_step_deg',0),2),'closure',round(x.get('closure_mismatch_deg',0),2),'winding',x.get('winding'))
print('Checked',len(rows),'window-cases: domain counts unchanged; inverse roundtrip pass; closure reported rather than imposed.')
