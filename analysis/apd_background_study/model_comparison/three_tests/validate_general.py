import run as r
import numpy as np,json
from scipy.optimize import least_squares
opt=r.make_basis('general');f,da=opt['f'],opt['da'];g=np.zeros((5,2,f.shape[1]));g[0,0]=1;g[1,1]=1
for j,l in enumerate([2,0,1]):g[j+2]=f[:2,:,l]
for j in range(5):
 for l in range(j):g[j]-=np.sum(g[j]*g[l])*da*g[l]
 g[j]/=np.sqrt(np.sum(g[j]**2)*da)
old=json.load(open(r.R.parent/'shared_field/fits.json'));out={}
for cap in [.1,2.]:
 p=np.array(old[f'm5_cap{cap}']['best']['p']);b,inc=r.prev.bgparts(p[40:],5);E=np.einsum('m,mip->ip',b,g)
 bn=np.einsum('mip,ip->m',opt['modes'],E)*da;perp=E-np.einsum('m,mip->ip',bn,opt['modes'])
 ip=r.ANALYZERS@perp;bi=inc+np.sum(abs(ip)**2,axis=1)*da;it=bi.sum();bt=p[40]*r.BREF
 Q=bi[0]-bi[1];U=bi[2]-bi[3];pol=2*np.hypot(Q,U)/it;ang=np.arctan2(U,Q);share=2*np.sum(abs(bn)**2)/bt
 h=np.r_[p[40],share,pol,ang,r.prev.invsphere(np.r_[bn.real,bn.imag])];pn=np.r_[p[:40],h]
 diff=max(np.max(abs(r.predict(pn[i*20:i*20+20],h,opt)-r.prev.predict(p[i*20:i*20+20],p[40:],5))) for i in range(2))
 print('embedding cap',cap,'modes',opt['m'],'maxdiff',diff,'residualbackground',it,flush=True)
 out[str(cap)]={'p':pn.tolist(),'embedding_difference':float(diff)}
(r.R/'embedded_starts.json').write_text(json.dumps(out,indent=2))
print('gram error',np.max(abs(opt['g'].sum(0)-2*np.eye(opt['m']))),flush=True)
# Completeness: arbitrary deterministic random spatial field and its projection
rng=np.random.default_rng(123);E=rng.normal(size=f[:2,:,0].shape)+1j*rng.normal(size=f[:2,:,0].shape);bn=np.einsum('mip,ip->m',opt['modes'],E)*da
ip=r.ANALYZERS@E;htrue=np.einsum('ipj,ip->ij',f,ip.conj())*da;hpred=np.einsum('ijm,m->ij',opt['h'],bn.conj());print('random field overlap reconstruction error',np.max(abs(htrue-hpred)),flush=True)
