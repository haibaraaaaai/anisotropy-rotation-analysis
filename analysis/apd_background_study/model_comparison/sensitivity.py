from fit_models import *
r=json.loads((R/'fits.json').read_text());out={}
for cap,k in [(.8,16),(.99,16),(.95,24),(.95,32)]:
 key=f'cap{cap}_k{k}';out[key]={}
 for m,v in r.items():
  b=v['best'];p=np.array(b['p']);di=b['direction']
  if k!=16:
   inc=softmax(np.r_[p[4:19],0]);cum=np.r_[0,np.cumsum(inc)]
   newinc=np.diff(np.interp(np.linspace(0,1,k+1),np.linspace(0,1,17),cum));logs=np.log(newinc[:-1]/newinc[-1])
   p=np.r_[p[:4],logs,p[19:]]
  rr=run(m,[(p,di)],cap,k);out[key][m]=rr['best']
(R/'sensitivity.json').write_text(json.dumps(out,indent=2))
