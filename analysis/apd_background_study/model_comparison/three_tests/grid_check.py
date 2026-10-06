import run as r
import numpy as np,json
saved=json.load(open(r.R/'fits.json'));checks={}
for kind,width,keys in [('general',.06,['general_verified_cap0.1','general_verified_cap2.0']),('edge',.03,['edge_w0.03_cap0.1']),('edge',.25,['edge_w0.25_cap2.0'])]:
 old=r.make_basis(kind,width,180);new=r.make_basis(kind,width,300)
 if kind=='general':
  rawold=old['f'][:2].transpose(0,2,1).reshape(6,-1);rawnew=new['f'][:2].transpose(0,2,1).reshape(6,-1)
 else:
  def raw(o):
   rho=np.hypot(o['x'],o['y']);env=np.exp(-.5*((rho-.38)/width)**2);return np.array([env,env*o['x']/rho,env*o['y']/rho])
  rawold=raw(old);rawnew=raw(new)
 A=(old['scalar']@rawold.T)@np.linalg.pinv(rawold@rawold.T);scalar=A@rawnew;ns=len(scalar);modes=np.zeros((2*ns,2,scalar.shape[1]));modes[:ns,0]=scalar;modes[ns:,1]=scalar
 gm=np.einsum('ij,mjp->ipm',r.ANALYZERS,modes);new['h']=np.einsum('ipj,ipm->ijm',new['f'],gm)*new['da'];new['g']=np.einsum('ipm,ipn->imn',gm,gm)*new['da']
 for key in keys:
  p=np.array(saved[key]['best']['p']);pure=kind=='edge';diff=max(np.max(abs(r.predict(p[i*20:i*20+20],p[40:],old,pure)-r.predict(p[i*20:i*20+20],p[40:],new,pure))) for i in range(2));checks[key]=float(diff)
print(checks);(r.R/'grid_checks.json').write_text(json.dumps(checks,indent=2))
