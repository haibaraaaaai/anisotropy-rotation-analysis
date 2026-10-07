"""Forward, collected-channel models for a controlled comparison.
Fixed .38..1.3 annulus, n_water=1.33, n_oil=1.51. No matrix calibration fit.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import softmax

ABC=np.array([.8908059777413628,.10919402225863714,.9277805723143716])
ANALYZERS=np.array([[1.,0.],[0.,1.],[1.,1.],[1.,-1.]])
ANALYZERS[2:]/=np.sqrt(2)

def uniform_pupil_overlap(order=128):
 x,w=leggauss(order);rho=.38+(x+1)*(1.3-.38)/2;dr=w*(1.3-.38)/2
 sw=rho/1.33;cw=np.sqrt(1-sw*sw);co=np.sqrt(1-(rho/1.51)**2)
 tp=2*1.33*cw/(1.33*co+1.51*cw);ts=2*1.33*cw/(1.33*cw+1.51*co)
 measure=2*np.pi*rho*dr
 qxx=np.sum(measure*co/cw**2*(3*(tp*cw)**2+2*tp*cw*ts+3*ts**2)/8)
 qyy=np.sum(measure*co/cw**2*(tp*cw-ts)**2/8)
 # Same normalization A+B=1 as the signal quadratic form.
 norm=(qxx+qyy)/2;area=np.pi*(1.3**2-.38**2)
 return np.sum(measure*np.sqrt(co)/cw*(tp*cw+ts)/2)/np.sqrt(area*norm)
GAMMA=uniform_pupil_overlap()

def cone(geometry,chi):
 th,ph,lam=geometry
 k=np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
 e1=np.array([-np.cos(th)*np.cos(ph),-np.cos(th)*np.sin(ph),np.sin(th)])
 e2=np.cross(k,e1)
 return np.cos(lam)*k+np.sin(lam)*(np.cos(chi)[:,None]*e1+np.sin(chi)[:,None]*e2)

def q_channels(d):
 A,B,C=ABC;x,y,z=d.T;s2=x*x+y*y
 return np.array([A+B*s2+C*(x*x-y*y),A+B*s2-C*(x*x-y*y),A+B*s2+2*C*x*y,A+B*s2-2*C*x*y])

def xy(c):
 return np.column_stack([(c[0]-c[1])/(c[0]+c[1]),(c[2]-c[3])/(c[2]+c[3])])

def stokes_background(total,polar,angle):
 # total is sum of four ideal-analyser intensities; pair sums are total/2.
 return total/4*np.array([1+polar*np.cos(angle),1-polar*np.cos(angle),1+polar*np.sin(angle),1-polar*np.sin(angle)])

def mechanical_phase(p,n,knots,direction):
 inc=softmax(np.r_[p[4:4+knots-1],0.])
 f=np.r_[0,np.cumsum(inc)]
 return p[3]+direction*2*np.pi*np.interp(np.arange(n)/n,np.linspace(0,1,knots+1),f)

def bounds(model,knots,cap=.95):
 lo=[.001,-np.pi,.001,-4*np.pi]+[-4.]*(knots-1)+[0.,0.,-np.pi]
 hi=[np.pi/2-.001,np.pi,np.pi/2-.001,4*np.pi]+[4.]*(knots-1)+[cap,1.,np.pi]
 if model=='richard':lo += [-1.]*4;hi += [1.]*4
 if model=='field':lo += [0.,0.,-np.pi,-np.pi];hi += [1.,np.pi/2,np.pi,np.pi]
 return np.array(lo),np.array(hi)

def predict(p,model,data,total_min,knots=16,direction=1):
 phase=mechanical_phase(p,data.shape[1],knots,direction)
 d=cone(p[:3],phase);q=q_channels(d);t=data.sum(axis=0)
 offset=knots+3
 f,pol,ang=p[offset:offset+3];btot=f*total_min
 bg=stokes_background(btot,pol,ang);linear=np.zeros_like(q)
 details={}
 if model=='richard':
  c=2*np.sqrt(bg)*p[offset+3:offset+7]
  linear=c[:,None]*np.sqrt(q);details['c']=c.tolist()
 elif model=='field':
  share,eta,rel,common=p[offset+3:offset+7]
  b=np.sqrt(btot*share/2)*np.exp(1j*common)*np.array([np.cos(eta),np.sin(eta)*np.exp(1j*rel)])
  bj=ANALYZERS@b
  bg=stokes_background(btot*(1-share),pol,ang)+np.abs(bj)**2
  # Circular excitation: orientation-dependent excitation phase is retained.
  # Its amplitude is absorbed in the common per-point brightness scale.
  exc=np.exp(1j*np.arctan2(d[:,1],d[:,0]))
  overlap=GAMMA*(ANALYZERS@d[:,:2].T)*exc[None,:]
  linear=2*np.real(overlap*np.conj(bj[:,None]))
  details.update(b_complex=[[float(v.real),float(v.imag)] for v in b],coherent_share=float(share))
 remaining=t-bg.sum();qsum=q.sum(axis=0);lsum=linear.sum(axis=0)
 # Within chosen conservative domain B_total < min observed total there is one
 # nonnegative root; no discarded data or branch-dependent positivity mask.
 disc=lsum**2+4*qsum*remaining
 z=(-lsum+np.sqrt(np.maximum(disc,0)))/(2*qsum)
 pred=q*z[None,:]**2+linear*z[None,:]+bg[:,None]
 details.update(background=bg.tolist(),total_bg=float(bg.sum()),brightness=z*z,
                theta=np.rad2deg(np.arccos(np.abs(d[:,2]))),phase=phase,
                signal=q*z[None,:]**2,cross=linear*z[None,:])
 return pred,details

def residual(p,model,data,total_min,knots=16,direction=1):
 pred,_=predict(p,model,data,total_min,knots,direction)
 return ((pred-data)/data.sum(axis=0)).ravel()
