import run as r
import numpy as np,json
from scipy.interpolate import CubicSpline
R=r.R

def add_height(opt):
 z=np.linspace(-.31,.31,125);x,y,f,gm,da=opt['x'],opt['y'],opt['f'],opt['gm'],opt['da'];kz=2*np.pi/.633*(1.33+np.sqrt(1.33**2-x*x-y*y))
 product=np.einsum('ipj,ipm->pijm',f,gm).reshape(len(x),-1)*da
 hz=[]
 for block in np.array_split(z,25):hz.append(np.exp(1j*block[:,None]*kz[None,:])@product)
 hz=np.concatenate(hz).reshape(len(z),4,3,opt['m']);opt.update(zgrid=z,dz=z[1]-z[0],hz=hz,hinterp=CubicSpline(z,hz,axis=0))
 errors=[]
 for zz in [-.2037,.0723,.1491]:
  direct=(np.exp(1j*zz*kz)@product).reshape(4,3,opt['m']);errors.append(float(np.max(abs(direct-opt['hinterp'](zz)))))
 (R/'height_interpolation_check.json').write_text(json.dumps({'max_abs_overlap_error':max(errors),'zero_height_error':float(np.max(abs(opt['hinterp'](0)-opt['h'])))}))
 return opt
if __name__=='__main__':
 opt=add_height(r.make_basis('general'));saved=json.load(open(R/'fits.json'));p=np.array(saved['general_cap0.1']['best']['p'])
 for bound in [.05,.15,.30]:
  starts=[np.r_[p[:40],bound*.5,phase,bound*.5,phase,p[40:]] for phase in [-2.,0.,2.]]
  if bound>.05:
   out=json.load(open(R/'fits.json'));prior=np.array(out[f'height_R{.05 if bound==.15 else .15}_cap0.1']['best']['p']);starts.append(prior)
  r.fit(f'height_R{bound}_cap0.1',opt,.1,motion=True,rmax=bound,starts=starts,nrandom=2)
