import steep as st
import numpy as np,json,importlib.util
from scipy.optimize import least_squares
f=st.f;g=f.f;R=f.R/'caps';R.mkdir(exist_ok=True)
sp=importlib.util.spec_from_file_location('sharedold',f.R.parent/'shared_field/run.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
opt5=old.optics(5,radius=180);opt10=np.load(f.R.parent/'three_tests/general_optics.npz');opts={5:opt5,10:tuple(opt10[k] for k in ['q','h','g'])};broad=g.BCAP
arch=json.load(open(f.R.parent/'shared_field/fits.json'));recent=json.load(open(f.R/'steep_fits.json'));unpen=json.load(open(f.R/'fits.json'))
def setup(m,cap):g.q,g.H,g.g=opts[m];g.M=m;g.BCAP=cap

def seed5(name):
 p=np.array(arch[name]['best']['p']);h=p[40:].copy();h[0]*=old.BREF*old.SCALE/g.Z['scale']/g.BREF
 loc=[]
 for i in range(2):
  a=p[i*20:i*20+20];l=np.r_[a[:3],a[19]+.5*np.log(old.SCALE/g.Z['scale']),0.];chi,arc,*_=g.dense(l,h);l[4]=np.interp(a[3]%(2*np.pi),chi,arc);loc.extend(l)
 return dict(p=np.r_[loc,h].tolist(),directions=[-1,-1])
if __name__=='__main__':
 out=json.load(open(R/'fits.json')) if (R/'fits.json').exists() else {}
 for m in [5,10]:
  for tag,cap in [('broad',broad),('10pct',.1),('5pct',.05)]:
   key=f'm{m}_{tag}'
   if key in out:continue
   setup(m,cap);lo,hi=g.limits()
   seeds=([seed5('m5_cap2.0'),seed5('m5_cap0.1')] if m==5 else [recent['best'],unpen['0']['runs'][1]])
   if tag!='broad':seeds[0]=out[f'm{m}_'+('broad' if tag=='10pct' else '10pct')]['best']
   runs=[]
   for b in seeds:
    di=b['directions'];p=np.clip(b['p'],lo+1e-8,hi-1e-8);fit=least_squares(st.residual,p,args=(di,2048),bounds=(lo,hi),max_nfev=180,ftol=3e-8,xtol=3e-8,gtol=3e-8,x_scale='jac');rec=dict(p=fit.x.tolist(),directions=di,cost=float(np.sum(fit.fun**2)),success=bool(fit.success),nfev=fit.nfev);runs.append(rec);print(key,'start',rec['cost'],rec['success'],flush=True)
   b=min(runs,key=lambda a:a['cost']);di=b['directions'];fit=least_squares(st.residual,b['p'],args=(di,8192),bounds=(lo,hi),max_nfev=180,ftol=1e-9,xtol=1e-9,gtol=1e-9,x_scale='jac');best=dict(p=fit.x.tolist(),directions=di,cost=float(np.sum(fit.fun**2)),success=bool(fit.success),nfev=fit.nfev,scores=f.score(fit.x,di));out[key]=dict(best=best,runs=runs,cap=cap,m=m);(R/'fits.json').write_text(json.dumps(out,indent=2));print(key,'FINAL',best['success'],[(k,best['scores'][k]['test']) for k in g.KEYS],flush=True)
 np.savez(R/'optics5.npz',q=opt5[0],h=opt5[1],g=opt5[2])
